// SPDX-License-Identifier: BSD-3-Clause
//
// GPU side of the MILP solver (host code; kernels in milp_kernels.cuh,
// compiled at run time by NVRTC like the QP kernels, see qp_backend_cuda.h).
//
//   DeviceProblem      the constraint matrix as CSR and CSC, row sides and
//                      integrality on the device
//   GpuPropagator      batched domain propagation: one domain, or B probes /
//                      children at once; double probing at the root
//                      (probe_binaries: every binary at 0 and at 1, fixings
//                      from infeasible probes, bounds both probes imply)
//   GpuFeasibilityJump feasibility jump with W walkers (thread blocks) from
//                      different start points and seeds
//   GpuBatchLp         Lagrangian bounds of up to 32 LP relaxations that
//                      differ in column bounds, by batched PDHG from a warm
//                      start (strong branching on the GPU)
//
// Everything runs on the legacy default stream: a download waits for the
// kernels before it.
#pragma once

#if !defined(AXOS_ENABLE_CUDA)
#error "milp_cuda.h needs -DAXOS_ENABLE_CUDA"
#endif

#include "solver/milp/milp_model.h"
#include "solver/qp/qp_backend_cuda.h" // cuda_rt: check, Nvrtc, source_dir
#include <algorithm>
#include <atomic>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <random>
#include <vector>

namespace AXOS {
namespace Solver {
namespace milp {
namespace gpu {

using qp::cuda_rt::check;

// The compiled MILP kernels, once per process.
class Module {
  public:
    static Module &
    get()
    {
        static std::once_flag once;
        static Module *m = nullptr;
        std::call_once(once, [] { m = new Module(); });
        return *m;
    }

    CUfunction
    fn(const char *name)
    {
        std::lock_guard<std::mutex> g(mu_);
        auto it = fns_.find(name);
        if (it != fns_.end()) return it->second;
        CUfunction f;
        check(cuModuleGetFunction(&f, mod_, name), name);
        fns_.emplace(name, f);
        return f;
    }

    double
    compile_seconds() const
    { return compile_s_; }
    const std::string &
    device() const
    { return device_; }

  private:
    Module()
    {
        check(cudaFree(nullptr), "cudaFree(0) (runtime initialization)");
        int dev = 0, major = 0, minor = 0;
        check(cudaGetDevice(&dev), "cudaGetDevice");
        check(cudaDeviceGetAttribute(
                  &major, cudaDevAttrComputeCapabilityMajor, dev),
            "attribute");
        check(cudaDeviceGetAttribute(
                  &minor, cudaDevAttrComputeCapabilityMinor, dev),
            "attribute");
        cudaDeviceProp prop;
        check(cudaGetDeviceProperties(&prop, dev), "cudaGetDeviceProperties");
        device_ = prop.name;
        const auto t0 = std::chrono::steady_clock::now();
        const qp::cuda_rt::Nvrtc &nv = qp::cuda_rt::Nvrtc::get();
        const std::string src = "#include \"solver/milp/milp_kernels.cuh\"\n";
        const std::string opt_arch =
            "--gpu-architecture=sm_" + std::to_string(major * 10 + minor);
        const std::string opt_inc = "-I" + qp::cuda_rt::source_dir();
        const char *opts[] = {
            opt_arch.c_str(), "-std=c++17", opt_inc.c_str(), "--fmad=true"};
        void *prog = nullptr;
        if (nv.create(
                &prog, src.c_str(), "axos_milp.cu", 0, nullptr, nullptr) != 0)
            throw std::runtime_error("nvrtcCreateProgram failed");
        const int rc = nv.compile(prog, 4, opts);
        size_t ls = 0;
        nv.log_size(prog, &ls);
        std::string log(ls, '\0');
        if (ls) nv.log(prog, &log[0]);
        if (rc != 0) {
            nv.destroy(&prog);
            throw std::runtime_error(
                "NVRTC compilation of the MILP kernels failed (" + opt_inc +
                "):\n" + log);
        }
        size_t cs = 0;
        nv.cubin_size(prog, &cs);
        std::vector<char> bin(cs);
        nv.cubin(prog, bin.data());
        nv.destroy(&prog);
        check(cuModuleLoadData(&mod_, bin.data()), "cuModuleLoadData");
        compile_s_ =
            std::chrono::duration<double>(std::chrono::steady_clock::now() - t0)
                .count();
    }

    CUmodule mod_ = nullptr;
    double compile_s_ = 0;
    std::string device_;
    std::mutex mu_;
    std::unordered_map<std::string, CUfunction> fns_;
};

constexpr unsigned kBlock = 256;

inline unsigned
blocks_for(long long threads)
{
    const long long g = (threads + kBlock - 1) / kBlock;
    return static_cast<unsigned>(std::max<long long>(1, g));
}

inline void
launch(const char *name, unsigned grid, unsigned block, void **args)
{
    check(cuLaunchKernel(Module::get().fn(name), grid, 1, 1, block, 1, 1, 0,
              nullptr, args, nullptr),
        name);
}

// A device array (grows, never shrinks).
template <typename T> class DBuf {
  public:
    DBuf() = default;
    explicit DBuf(size_t n) { resize(n); }
    ~DBuf()
    {
        if (p_) cudaFree(p_);
    }
    DBuf(const DBuf &) = delete;
    DBuf &operator=(const DBuf &) = delete;

    void
    resize(size_t n)
    {
        if (n > cap_ || !p_) {
            if (p_) cudaFree(p_);
            p_ = nullptr;
            check(cudaMalloc(reinterpret_cast<void **>(&p_),
                      std::max<size_t>(n, 1) * sizeof(T)),
                "cudaMalloc");
            cap_ = std::max<size_t>(n, 1);
        }
        n_ = n;
    }
    void
    upload(const T *h, size_t n)
    {
        resize(n);
        if (n)
            check(cudaMemcpy(p_, h, n * sizeof(T), cudaMemcpyHostToDevice),
                "upload");
    }
    void
    upload(const std::vector<T> &h)
    { upload(h.data(), h.size()); }
    void
    download(T *h, size_t n, size_t offset = 0) const
    {
        if (n)
            check(cudaMemcpy(
                      h, p_ + offset, n * sizeof(T), cudaMemcpyDeviceToHost),
                "download");
    }
    void
    download(std::vector<T> &h) const
    {
        h.resize(n_);
        download(h.data(), n_);
    }
    void
    zero()
    {
        if (n_) check(cudaMemset(p_, 0, n_ * sizeof(T)), "cudaMemset");
    }
    T *
    get() const
    { return p_; }
    size_t
    size() const
    { return n_; }

  private:
    T *p_ = nullptr;
    size_t n_ = 0, cap_ = 0;
};

// The constraint matrix (CSR and CSC), row sides and integrality.
struct DeviceProblem {
    int n = 0, m = 0;
    DBuf<int32_t> rp, ci, cp, ri;
    DBuf<double> va, cv, rlb, rub;
    DBuf<unsigned char> isint;

    explicit DeviceProblem(const LpProblem &p)
    {
        n = static_cast<int>(p.cols());
        m = static_cast<int>(p.rows());
        const size_t nnz = p.A.nnz();
        rp.upload(p.A.row_ptr(), m + 1);
        ci.upload(p.A.col_ind(), nnz);
        va.upload(p.A.values(), nnz);
        const HostMatrix At = p.A.transpose();
        cp.upload(At.row_ptr(), n + 1);
        ri.upload(At.col_ind(), nnz);
        cv.upload(At.values(), nnz);
        rlb.upload(p.row_lb);
        rub.upload(p.row_ub);
        std::vector<unsigned char> ii(n);
        for (int j = 0; j < n; ++j)
            ii[j] = is_int(p, j) ? 1 : 0;
        isint.upload(ii);
    }
};

// ---- propagation and probing
// ---------------------------------------------------

class GpuPropagator {
  public:
    explicit GpuPropagator(const DeviceProblem &d) : d_(d) {}

    // Propagates B = fix_col.size() domains: the base bounds with column
    // fix_col[b] restricted to [fix_lo[b], fix_hi[b]] (fix_col[b] < 0: the base
    // domain). infeasible[b] reports an empty domain; the propagated bounds (B
    // x n, batch-major) are downloaded when lb_out is given. Returns the rounds
    // run.
    int
    run(const std::vector<double> &lb0, const std::vector<double> &ub0,
        const std::vector<int> &fix_col, const std::vector<double> &fix_lo,
        const std::vector<double> &fix_hi, int max_rounds,
        std::vector<int> &infeasible, std::vector<double> *lb_out = nullptr,
        std::vector<double> *ub_out = nullptr)
    {
        int n = d_.n, m = d_.m, B = static_cast<int>(fix_col.size());
        lb0_.upload(lb0);
        ub0_.upload(ub0);
        fcol_.upload(fix_col);
        flo_.upload(fix_lo);
        fhi_.upload(fix_hi);
        lb_.resize(size_t(B) * n);
        ub_.resize(size_t(B) * n);
        minact_.resize(size_t(B) * m);
        maxact_.resize(size_t(B) * m);
        ninfmin_.resize(size_t(B) * m);
        ninfmax_.resize(size_t(B) * m);
        active_.resize(B);
        changed_.resize(B);
        infeas_.resize(B);
        {
            int32_t *fc = fcol_.get();
            double *a0 = lb0_.get(), *a1 = ub0_.get(), *a2 = flo_.get(),
                   *a3 = fhi_.get(), *l = lb_.get(), *u = ub_.get();
            int *ac = active_.get(), *ch = changed_.get(), *inf = infeas_.get();
            void *args[] = {
                &n, &B, &a0, &a1, &fc, &a2, &a3, &l, &u, &ac, &ch, &inf};
            launch("prop_init", blocks_for((long long)n * B), kBlock, args);
        }
        std::vector<int> act(B);
        int rounds = 0;
        for (; rounds < max_rounds; ++rounds) {
            const int32_t *rp = d_.rp.get(), *ci = d_.ci.get(),
                          *cp = d_.cp.get(), *ri = d_.ri.get();
            const double *va = d_.va.get(), *cv = d_.cv.get(),
                         *rl = d_.rlb.get(), *ru = d_.rub.get();
            const unsigned char *ii = d_.isint.get();
            double *l = lb_.get(), *u = ub_.get(), *mn = minact_.get(),
                   *mx = maxact_.get();
            int *c0 = ninfmin_.get(), *c1 = ninfmax_.get(), *ac = active_.get(),
                *ch = changed_.get(), *inf = infeas_.get();
            {
                void *args[] = {&m, &n, &B, &rp, &ci, &va, &rl, &ru, &l, &u,
                    &mn, &mx, &c0, &c1, &ac, &inf};
                launch("prop_activity", blocks_for((long long)m * B * 32),
                    kBlock, args);
            }
            {
                void *args[] = {&n, &m, &B, &cp, &ri, &cv, &rl, &ru, &mn, &mx,
                    &c0, &c1, &l, &u, &ii, &ac, &ch, &inf};
                launch(
                    "prop_tighten", blocks_for((long long)n * B), kBlock, args);
            }
            {
                void *args[] = {&B, &ac, &ch, &inf};
                launch("prop_next", blocks_for(B), kBlock, args);
            }
            active_.download(act.data(), B);
            bool any = false;
            for (int v : act)
                any = any || v;
            if (!any) {
                ++rounds;
                break;
            }
        }
        infeasible.resize(B);
        infeas_.download(infeasible.data(), B);
        if (lb_out) lb_.download(*lb_out);
        if (ub_out) ub_.download(*ub_out);
        return rounds;
    }

    // One domain: tightens lb/ub in place; false if proven empty.
    bool
    propagate(
        std::vector<double> &lb, std::vector<double> &ub, int max_rounds = 1000)
    {
        std::vector<int> inf;
        std::vector<double> l, u;
        run(lb, ub, {-1}, {0.0}, {0.0}, max_rounds, inf, &l, &u);
        if (inf[0]) return false;
        lb = l;
        ub = u;
        return true;
    }

    // After run() on 2K probes (pairs: binary at 0, at 1): the bounds that
    // hold whatever each binary takes, merged over the pairs (probe_merge).
    void
    merge_pairs(int K, std::vector<double> &out_lb, std::vector<double> &out_ub)
    {
        int n = d_.n;
        mlb_.resize(n);
        mub_.resize(n);
        double *a0 = lb0_.get(), *a1 = ub0_.get(), *l = lb_.get(),
               *u = ub_.get(), *ol = mlb_.get(), *ou = mub_.get();
        int *inf = infeas_.get();
        void *args[] = {&n, &K, &a0, &a1, &l, &u, &inf, &ol, &ou};
        launch("probe_merge", blocks_for(n), kBlock, args);
        mlb_.download(out_lb);
        mub_.download(out_ub);
    }

  private:
    const DeviceProblem &d_;
    DBuf<double> lb0_, ub0_, flo_, fhi_, lb_, ub_, minact_, maxact_, mlb_, mub_;
    DBuf<int32_t> fcol_;
    DBuf<int> ninfmin_, ninfmax_, active_, changed_, infeas_;
};

struct ProbeResult {
    int probes = 0, fixed = 0, tightened = 0, rounds = 0;
    bool infeasible = false;
    double seconds = 0;
};

// Same layout as the kernels' ProbeData.
struct ProbeData {
    int n, m, B;
    const int32_t *rp, *ci;
    const double *va;
    const int32_t *cp, *ri;
    const double *cv;
    const double *rlb, *rub;
    const unsigned char *isint;
    const double *blb, *bub;
    const double *bmin, *bmax;
    const int *bcmin, *bcmax;
    double *lb, *ub;
    double *mn, *mx;
    int *cmn, *cmx;
    unsigned *rbits, *cbits;
    unsigned *trbits, *tcbits;
    unsigned *cf, *tc, *tr;
    unsigned *cnt;
    int *infeas;
    double *out_lb, *out_ub;
};

// Double probing of the binary columns on the GPU: `slots` probes at once
// (each binary at 0 and at 1 in neighbouring slots), propagated with row /
// column frontiers (milp_kernels.cuh, pb_*), so a probe costs what its
// propagation touches. A slot that empties its domain fixes the binary the
// other way; otherwise each column gets the weaker of the two probes' bounds.
// All probes of a pass start from the same base domain; the merged bounds come
// back once at the end.
class GpuProber {
  public:
    GpuProber(const DeviceProblem &d, int slots = 256) : d_(d)
    {
        const long long words_n = (long long)d.n * slots,
                        words_m = (long long)d.m * slots;
        // keep the slot arrays within about 1.5 GB of device memory
        const double per_slot = 16.0 * d.n + 24.0 * d.m +
                                8.0 * (d.n + 2.0 * d.m) + 4.0 * (d.n + d.m);
        B_ = std::max(2,
            std::min(slots, static_cast<int>(1.5e9 / std::max(per_slot, 1.0))) &
                ~1);
        (void)words_n;
        (void)words_m;
    }

    // Probes the binaries of lb/ub (base domain) until the deadline; on
    // return lb/ub hold the merged bounds. Infeasible: some binary empties
    // the domain both ways.
    ProbeResult
    run(const LpProblem &p, std::vector<double> &lb, std::vector<double> &ub,
        double seconds)
    {
        const auto t0 = std::chrono::steady_clock::now();
        auto elapsed = [&] {
            return std::chrono::duration<double>(
                std::chrono::steady_clock::now() - t0)
                .count();
        };
        ProbeResult res;
        int n = d_.n, m = d_.m, B = B_;
        const size_t N = size_t(B) * n, M = size_t(B) * m;
        std::vector<int> bins;
        for (int j = 0; j < n; ++j)
            if (is_int(p, j) && lb[j] == 0 && ub[j] == 1) bins.push_back(j);
        if (bins.empty()) return res;
        // base domain and its activities
        blb_.upload(lb);
        bub_.upload(ub);
        bmin_.resize(m);
        bmax_.resize(m);
        bcmin_.resize(m);
        bcmax_.resize(m);
        {
            int one = 1;
            DBuf<int> act(1), inf(1);
            const int on = 1, zero = 0;
            act.upload(&on, 1);
            inf.upload(&zero, 1);
            const int32_t *rp = d_.rp.get(), *ci = d_.ci.get();
            const double *va = d_.va.get(), *rl = d_.rlb.get(),
                         *ru = d_.rub.get();
            double *l = blb_.get(), *u = bub_.get(), *mn = bmin_.get(),
                   *mx = bmax_.get();
            int *c0 = bcmin_.get(), *c1 = bcmax_.get(), *ac = act.get(),
                *in = inf.get();
            void *args[] = {&m, &n, &one, &rp, &ci, &va, &rl, &ru, &l, &u, &mn,
                &mx, &c0, &c1, &ac, &in};
            launch(
                "prop_activity", blocks_for((long long)m * 32), kBlock, args);
            int bad = 0;
            inf.download(&bad, 1);
            if (bad) {
                res.infeasible = true;
                return res;
            }
        }
        lb_.resize(N);
        ub_.resize(N);
        mn_.resize(M);
        mx_.resize(M);
        cmn_.resize(M);
        cmx_.resize(M);
        rbits_.resize(M / 32 + 1);
        rbits_.zero();
        trbits_.resize(M / 32 + 1);
        trbits_.zero();
        cbits_.resize(N / 32 + 1);
        cbits_.zero();
        tcbits_.resize(N / 32 + 1);
        tcbits_.zero();
        rfa_.resize(M);
        rfb_.resize(M);
        cf_.resize(N);
        tc_.resize(N);
        tr_.resize(M);
        cnt_.resize(8);
        cnt_.zero();
        infeas_.resize(B);
        infeas_.zero();
        olb_.resize(n);
        oub_.resize(n);
        check(cudaMemcpy(olb_.get(), blb_.get(), n * sizeof(double),
                  cudaMemcpyDeviceToDevice),
            "copy");
        check(cudaMemcpy(oub_.get(), bub_.get(), n * sizeof(double),
                  cudaMemcpyDeviceToDevice),
            "copy");
        ProbeData pd{n, m, B, d_.rp.get(), d_.ci.get(), d_.va.get(),
            d_.cp.get(), d_.ri.get(), d_.cv.get(), d_.rlb.get(), d_.rub.get(),
            d_.isint.get(), blb_.get(), bub_.get(), bmin_.get(), bmax_.get(),
            bcmin_.get(), bcmax_.get(), lb_.get(), ub_.get(), mn_.get(),
            mx_.get(), cmn_.get(), cmx_.get(), rbits_.get(), cbits_.get(),
            trbits_.get(), tcbits_.get(), cf_.get(), tc_.get(), tr_.get(),
            cnt_.get(), infeas_.get(), olb_.get(), oub_.get()};
        {
            void *args[] = {&pd};
            launch(
                "pb_fill", blocks_for((long long)std::max(N, M)), kBlock, args);
        }
        std::vector<int> pcol(B);
        std::vector<double> plo(B), phi(B);
        std::vector<int> inf(B);
        unsigned cnt[8];
        const int pairs = B / 2;
        for (size_t s = 0; s < bins.size() && elapsed() < seconds; s += pairs) {
            const int K =
                static_cast<int>(std::min<size_t>(pairs, bins.size() - s));
            int Bk = 2 * K;
            for (int k = 0; k < K; ++k) {
                pcol[2 * k] = pcol[2 * k + 1] = bins[s + k];
                plo[2 * k] = 0;
                phi[2 * k] = 0;
                plo[2 * k + 1] = 1;
                phi[2 * k + 1] = 1;
            }
            pcol_.upload(pcol.data(), Bk);
            plo_.upload(plo.data(), Bk);
            phi_.upload(phi.data(), Bk);
            pd.B = Bk; // slots beyond 2K stay at the base domain
            unsigned *rf = rfa_.get(), *rf_next = rfb_.get();
            {
                int32_t *pc = pcol_.get();
                double *lo = plo_.get(), *hi = phi_.get();
                void *args[] = {&pd, &pc, &lo, &hi, &rf};
                launch("pb_start", blocks_for(Bk), kBlock, args);
            }
            bool capped = false;
            for (int round = 0;; ++round) {
                cnt_.download(cnt, 5);
                unsigned nrf = cnt[1];
                if (nrf == 0) break;
                if (round >= 40) {
                    capped = true;
                    break;
                }
                check(cudaMemset(cnt_.get() + 1, 0, 2 * sizeof(unsigned)),
                    "cudaMemset");
                {
                    void *args[] = {&pd, &rf, &nrf};
                    launch("pb_rows", blocks_for((long long)nrf * 32), kBlock,
                        args);
                }
                cnt_.download(cnt, 5);
                unsigned ncf = cnt[2];
                if (ncf == 0) break;
                {
                    void *args[] = {&pd, &ncf, &rf_next};
                    launch("pb_cols", blocks_for(ncf), kBlock, args);
                }
                std::swap(rf, rf_next);
            }
            res.rounds += 0;
            cnt_.download(cnt, 5);
            unsigned ntc = cnt[3], ntr = cnt[4];
            infeas_.download(inf.data(), Bk);
            res.probes += Bk;
            bool both = false;
            for (int k = 0; k < K; ++k) {
                if (inf[2 * k] && inf[2 * k + 1])
                    both = true;
                else if (inf[2 * k] || inf[2 * k + 1])
                    ++res.fixed;
            }
            if (both) {
                res.infeasible = true;
                break;
            }
            if (ntc) {
                void *args[] = {&pd, &ntc};
                launch("pb_merge", blocks_for(ntc), kBlock, args);
            }
            {
                const unsigned nr = std::max(ntc, ntr);
                void *args[] = {&pd, &ntc, &ntr};
                if (nr) launch("pb_reset", blocks_for(nr), kBlock, args);
            }
            cnt_.zero();
            infeas_.zero();
            if (capped) {
                rbits_.zero();
                cbits_.zero();
            }
        }
        if (!res.infeasible) {
            std::vector<double> ol, ou;
            olb_.download(ol);
            oub_.download(ou);
            for (int j = 0; j < n; ++j) {
                if (ol[j] > lb[j] + 1e-9) {
                    lb[j] = ol[j];
                    ++res.tightened;
                }
                if (ou[j] < ub[j] - 1e-9) {
                    ub[j] = ou[j];
                    ++res.tightened;
                }
                if (lb[j] > ub[j] + 1e-9) res.infeasible = true;
            }
        }
        res.seconds = elapsed();
        return res;
    }

    int
    slots() const
    { return B_; }

  private:
    const DeviceProblem &d_;
    int B_ = 256;
    DBuf<double> blb_, bub_, bmin_, bmax_, lb_, ub_, mn_, mx_, olb_, oub_, plo_,
        phi_;
    DBuf<int> bcmin_, bcmax_, cmn_, cmx_, infeas_;
    DBuf<int32_t> pcol_;
    DBuf<unsigned> rbits_, cbits_, trbits_, tcbits_, rfa_, rfb_, cf_, tc_, tr_,
        cnt_;
};

// Double probing of the binaries of lb/ub on the GPU within `seconds`
// (GpuProber); lb/ub receive the fixings and tightened bounds.
inline ProbeResult
probe_binaries(const DeviceProblem &d, const LpProblem &p,
    std::vector<double> &lb, std::vector<double> &ub, double seconds,
    int slots = 256)
{
    GpuProber pr(d, slots);
    return pr.run(p, lb, ub, seconds);
}

// ---- feasibility jump
// --------------------------------------------------------------

// Same layout as the kernels' FjData.
struct FjData {
    int n, m;
    const int32_t *rp, *ci;
    const double *va;
    const int32_t *cp, *ri;
    const double *cv;
    const double *rlb, *rub, *lb, *ub;
    const unsigned char *isint;
    double *x, *act, *wt;
    int *vl, *vpos, *vcnt, *status;
    unsigned long long *rng;
    long long *moves;
};

class GpuFeasibilityJump {
  public:
    GpuFeasibilityJump(const DeviceProblem &d, int walkers = 64)
        : d_(d), W_(walkers)
    {
    }

    // W walkers from the start points (cycled, with random perturbation beyond
    // the first round) within lb/ub. True with x when one walker reached a
    // point satisfying every row. Gives up at the time limit or when *stop is
    // set (another thread).
    bool
    run(const std::vector<double> &lb, const std::vector<double> &ub,
        const std::vector<std::vector<double>> &starts,
        const std::vector<uint8_t> &isint, double time_limit, uint64_t seed,
        std::vector<double> &x, long long *moves_out = nullptr,
        const std::atomic<bool> *stop = nullptr)
    {
        const auto t0 = std::chrono::steady_clock::now();
        auto elapsed = [&] {
            return std::chrono::duration<double>(
                std::chrono::steady_clock::now() - t0)
                .count();
        };
        const int n = d_.n, m = d_.m, W = W_;
        std::mt19937_64 rng(seed);
        std::vector<double> xs(size_t(W) * n);
        for (int w = 0; w < W; ++w) {
            const std::vector<double> *s0 =
                starts.empty() ? nullptr : &starts[w % starts.size()];
            const bool perturb =
                !starts.empty() && w >= static_cast<int>(starts.size());
            for (int j = 0; j < n; ++j) {
                double v = s0 ? (*s0)[j] : 0.0;
                if (perturb && isint[j]) {
                    const double f = v - std::floor(v);
                    v = (std::uniform_real_distribution<double>(0, 1)(rng) < f)
                            ? std::ceil(v)
                            : std::floor(v);
                }
                v = std::min(std::max(v, lb[j]), ub[j]);
                if (!std::isfinite(v) || std::abs(v) > 1e15)
                    v = std::abs(lb[j]) < 1e15
                            ? lb[j]
                            : (std::abs(ub[j]) < 1e15 ? ub[j] : 0.0);
                if (isint[j])
                    v = std::min(
                        std::max(std::round(v), std::ceil(lb[j] - 1e-9)),
                        std::floor(ub[j] + 1e-9));
                xs[size_t(w) * n + j] = v;
            }
        }
        lb_.upload(lb);
        ub_.upload(ub);
        x_.upload(xs);
        act_.resize(size_t(W) * m);
        wt_.resize(size_t(W) * m);
        vl_.resize(size_t(W) * m);
        vpos_.resize(size_t(W) * m);
        vcnt_.resize(W);
        status_.resize(W);
        status_.zero();
        moves_.resize(W);
        moves_.zero();
        std::vector<unsigned long long> rs(W);
        for (int w = 0; w < W; ++w)
            rs[w] = (rng() | 1ull);
        rng_.upload(rs);
        FjData fd{n, m, d_.rp.get(), d_.ci.get(), d_.va.get(), d_.cp.get(),
            d_.ri.get(), d_.cv.get(), d_.rlb.get(), d_.rub.get(), lb_.get(),
            ub_.get(), d_.isint.get(), x_.get(), act_.get(), wt_.get(),
            vl_.get(), vpos_.get(), vcnt_.get(), status_.get(), rng_.get(),
            moves_.get()};
        {
            void *args[] = {&fd};
            launch("fj_init", W, 128, args);
        }
        std::vector<int> st(W);
        int steps = 64;
        bool found = false;
        int wfound = -1;
        while (elapsed() < time_limit && !(stop && stop->load())) {
            void *args[] = {&fd, &steps};
            launch("fj_run", W, 128, args);
            status_.download(st.data(), W);
            for (int w = 0; w < W && !found; ++w)
                if (st[w] == 1) {
                    found = true;
                    wfound = w;
                }
            if (found) break;
            steps = std::min(steps * 2, 4096);
        }
        if (moves_out) {
            std::vector<long long> mv(W);
            moves_.download(mv.data(), W);
            *moves_out = 0;
            for (auto v : mv)
                *moves_out += v;
        }
        if (!found) return false;
        x.resize(n);
        x_.download(x.data(), n, size_t(wfound) * n);
        return true;
    }

  private:
    const DeviceProblem &d_;
    int W_;
    DBuf<double> lb_, ub_, x_, act_, wt_;
    DBuf<int> vl_, vpos_, vcnt_, status_;
    DBuf<unsigned long long> rng_;
    DBuf<long long> moves_;
};

// ---- batched LP bounds
// ------------------------------------------------------------

// Lagrangian bounds of up to 32 LPs min c^T x, rl <= A x <= ru, l^b <= x <= u^b
// by PDHG on the Ruiz-scaled problem from a common warm start.
class GpuBatchLp {
  public:
    static constexpr int kMaxBatch = 32;

    explicit GpuBatchLp(const LpProblem &lp)
        : n_(static_cast<int>(lp.cols())), m_(static_cast<int>(lp.rows()))
    {
        offset_ = lp.offset;
        // Ruiz equilibration: rows and columns to unit infinity norm
        dr_.assign(m_, 1.0);
        dc_.assign(n_, 1.0);
        const int32_t *rp = lp.A.row_ptr(), *ci = lp.A.col_ind();
        const double *va = lp.A.values();
        const size_t nnz = lp.A.nnz();
        std::vector<double> v(va, va + nnz);
        for (int it = 0; it < 10; ++it) {
            std::vector<double> rmax(m_, 0.0), cmax(n_, 0.0);
            for (int i = 0; i < m_; ++i)
                for (int32_t k = rp[i]; k < rp[i + 1]; ++k) {
                    const double a = std::abs(v[k]);
                    rmax[i] = std::max(rmax[i], a);
                    cmax[ci[k]] = std::max(cmax[ci[k]], a);
                }
            for (int i = 0; i < m_; ++i)
                if (rmax[i] > 0) dr_[i] /= std::sqrt(rmax[i]);
            for (int j = 0; j < n_; ++j)
                if (cmax[j] > 0) dc_[j] /= std::sqrt(cmax[j]);
            for (int i = 0; i < m_; ++i)
                for (int32_t k = rp[i]; k < rp[i + 1]; ++k)
                    v[k] = va[k] * dr_[i] * dc_[ci[k]];
        }
        HostMatrix As(m_, n_, std::vector<int32_t>(rp, rp + m_ + 1),
            std::vector<int32_t>(ci, ci + nnz), v);
        const HostMatrix At = As.transpose();
        rp_.upload(rp, m_ + 1);
        ci_.upload(ci, nnz);
        va_.upload(v);
        cp_.upload(At.row_ptr(), n_ + 1);
        ri_.upload(At.col_ind(), nnz);
        cv_.upload(At.values(), nnz);
        std::vector<double> c(n_), rl(m_), ru(m_);
        for (int j = 0; j < n_; ++j)
            c[j] = lp.c[j] * dc_[j];
        for (int i = 0; i < m_; ++i) {
            rl[i] =
                std::abs(lp.row_lb[i]) < 1e20 ? lp.row_lb[i] * dr_[i] : -kInf;
            ru[i] =
                std::abs(lp.row_ub[i]) < 1e20 ? lp.row_ub[i] * dr_[i] : kInf;
        }
        c_.upload(c);
        rl_.upload(rl);
        ru_.upload(ru);
        // ||A_s||_2 by power iteration on A^T A
        std::vector<double> x(n_, 1.0), y(m_);
        double norm = 1;
        for (int it = 0; it < 40; ++it) {
            double s = 0;
            for (double t : x)
                s += t * t;
            s = std::sqrt(s);
            if (s == 0) break;
            for (double &t : x)
                t /= s;
            for (int i = 0; i < m_; ++i) {
                double a = 0;
                for (int32_t k = rp[i]; k < rp[i + 1]; ++k)
                    a += v[k] * x[ci[k]];
                y[i] = a;
            }
            std::fill(x.begin(), x.end(), 0.0);
            for (int i = 0; i < m_; ++i)
                for (int32_t k = rp[i]; k < rp[i + 1]; ++k)
                    x[ci[k]] += v[k] * y[i];
            double q = 0;
            for (double t : x)
                q += t * t;
            norm = std::sqrt(std::sqrt(q)); // ||A^T A x|| -> sigma_max^2
        }
        // primal weight ||c|| / ||finite row sides||
        double cn = 0, bn = 0;
        for (double t : c)
            cn += t * t;
        for (int i = 0; i < m_; ++i) {
            const double t = std::isfinite(rl[i])
                                 ? rl[i]
                                 : (std::isfinite(ru[i]) ? ru[i] : 0.0);
            bn += t * t;
        }
        omega_ = (cn > 0 && bn > 0) ? std::sqrt(cn) / std::sqrt(bn) : 1.0;
        eta_ = 0.95 / std::max(norm, 1e-12);
    }

    // Bounds of the LPs whose column bounds are lb/ub with column col[b]
    // restricted to [lo[b], hi[b]], after `iters` PDHG iterations from (x0, y0)
    // (original scale), evaluated every `every` iterations; the best Lagrangian
    // bound of each LP (with the objective offset), -inf when every evaluation
    // needed an infinite bound.
    std::vector<double>
    child_bounds(const std::vector<double> &x0, const std::vector<double> &y0,
        const std::vector<double> &lb, const std::vector<double> &ub,
        const std::vector<int> &col, const std::vector<double> &lo,
        const std::vector<double> &hi, int iters, int every = 10)
    {
        int B = static_cast<int>(col.size()), n = n_, m = m_;
        if (B > kMaxBatch)
            throw std::invalid_argument("GpuBatchLp: at most 32 LPs per batch");
        std::vector<double> L(size_t(n) * B), U(size_t(n) * B),
            X(size_t(n) * B), Y(size_t(m) * B);
        for (int j = 0; j < n; ++j)
            for (int b = 0; b < B; ++b) {
                double l = lb[j], u = ub[j];
                if (col[b] == j) {
                    l = std::max(l, lo[b]);
                    u = std::min(u, hi[b]);
                }
                const size_t o = size_t(j) * B + b;
                L[o] = std::abs(l) < 1e20 ? l / dc_[j] : -kInf;
                U[o] = std::abs(u) < 1e20 ? u / dc_[j] : kInf;
                X[o] = std::min(std::max(x0[j] / dc_[j], L[o]), U[o]);
            }
        for (int i = 0; i < m; ++i)
            for (int b = 0; b < B; ++b)
                Y[size_t(i) * B + b] = y0[i] / dr_[i];
        L_.upload(L);
        U_.upload(U);
        X_.upload(X);
        Xb_.resize(X.size());
        Y_.upload(Y);
        acc_.resize(B);
        acc_.zero();
        bad_.resize(B);
        bad_.zero();
        std::vector<double> best(B, -kInf);
        best_.upload(best);
        double tau = eta_ / omega_, sigma = eta_ * omega_;
        const int32_t *rp = rp_.get(), *ci = ci_.get(), *cp = cp_.get(),
                      *ri = ri_.get();
        const double *va = va_.get(), *cv = cv_.get(), *c = c_.get(),
                     *rl = rl_.get(), *ru = ru_.get();
        double *Lp = L_.get(), *Up = U_.get(), *x = X_.get(), *xb = Xb_.get(),
               *y = Y_.get(), *acc = acc_.get(), *bst = best_.get();
        int *bad = bad_.get();
        for (int k = 0; k <= iters; ++k) {
            int eval = (k % every == 0 || k == iters) ? 1 : 0;
            {
                void *args[] = {&n, &B, &cp, &ri, &cv, &c, &Lp, &Up, &y, &x,
                    &xb, &tau, &eval, &acc, &bad};
                launch(
                    "pd_primal", blocks_for((long long)n * 32), kBlock, args);
            }
            {
                void *args[] = {&m, &B, &rp, &ci, &va, &rl, &ru, &xb, &y,
                    &sigma, &eval, &acc};
                launch("pd_dual", blocks_for((long long)m * 32), kBlock, args);
            }
            if (eval) {
                void *args[] = {&B, &acc, &bad, &bst};
                launch("pd_take", 1, 32, args);
            }
        }
        best_.download(best);
        for (double &v : best)
            if (v > -kInf) v += offset_;
        return best;
    }

  private:
    int n_, m_;
    double offset_ = 0, omega_ = 1, eta_ = 1;
    std::vector<double> dr_, dc_;
    DBuf<int32_t> rp_, ci_, cp_, ri_;
    DBuf<double> va_, cv_, c_, rl_, ru_, L_, U_, X_, Xb_, Y_, acc_, best_;
    DBuf<int> bad_;
};

} // namespace gpu
} // namespace milp
} // namespace Solver
} // namespace AXOS
