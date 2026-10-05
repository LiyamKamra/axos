// SPDX-License-Identifier: BSD-3-Clause
//
// CPU backend of the first-order QP solvers: the per-element ops of
// qp_ops.h run in OpenMP loops (tensor/parallel.h), sparse products use the
// CSR kernels of sparse_cpu.h. Reductions add per-chunk partial results in
// chunk order, so results do not depend on thread timing.
//
// Backend interface (shared with qp_backend_cuda.h):
//   Vec vec(n, v), upload(host), download(vec, host), ptr(vec)
//   Mat upload(HostMatrix), spmv(Mat, x, y)            y = A x
//   spmv_epi(name, Mat, x, args, f)   f(i, (A x)_i, args) for every row i
//   map(name, n, args, f)       f(i, args) for i < n
//   reduce(name, n, args, f)    array of Args::K results (sum or max)
//   reduce_to(name, n, args, f, slot), copy_results(n), sync(), results()
//                               the same, deferred: results()[slot + k]
//   results_dev()               the slots where device ops can read them
//   run_block(key, fn)          fn() (the GPU backend replays it as a graph),
//   forget(key)                 drops that graph; use_graphs(false): no graphs
//   epi(name, n, s, args, f)    f(i, s[i], args): an epilogue on given row sums
//   through_host(v, n, fn)      fn(h) on a host copy h of v[0..n), copied back
#pragma once

#include "solver/model.h"
#include "solver/qp/qp_ops.h"
#include "sparse/sparse_cpu.h"
#include "tensor/parallel.h"
#include "tensorET.h"
#include <array>
#include <chrono>
#include <cstring>
#include <string>
#include <vector>

namespace AXOS {
namespace Solver {
namespace qp {

class CpuBackend {
  public:
    static constexpr bool is_gpu = false;
    using Vec = tensorET<1, double>;
    struct Mat {
        HostMatrix A;
        size_t
        rows() const
        { return A.rows(); }
        size_t
        nnz() const
        { return A.nnz(); }
    };

    std::string
    name() const
    { return "cpu"; }

    Vec
    vec(size_t n, double v = 0.0) const
    { return Vec({n}, v); }

    Vec
    upload(const std::vector<double> &h) const
    {
        Vec v({h.size()});
        if (!h.empty())
            std::memcpy(v.data, h.data(), h.size() * sizeof(double));
        return v;
    }

    void
    download(const Vec &v, std::vector<double> &h) const
    { h.assign(v.data, v.data + v.size()); }

    static double *
    ptr(Vec &v)
    { return v.data; }
    static const double *
    ptr(const Vec &v)
    { return v.data; }

    Mat
    upload(const HostMatrix &A) const
    { return Mat{A}; }

    void
    spmv(const Mat &M, const double *x, double *y) const
    {
        if (M.rows() == 0) return;
        Sparse::Kernels<Cpu::Backend>::spmv(M.A, x, y, 1.0, 0.0);
    }

    template <class Args, class F>
    void
    spmv_epi(
        const char *, const Mat &M, const double *x, const Args &a, F f) const
    {
        const auto *rp = M.A.row_ptr();
        const auto *ci = M.A.col_ind();
        const double *v = M.A.values();
        detail::parallel_for(M.rows(), kGrain / 8, [&](size_t b, size_t e) {
            for (size_t i = b; i < e; ++i) {
                double s = 0.0;
                for (auto k = rp[i]; k < rp[i + 1]; ++k)
                    s += v[k] * x[ci[k]];
                f(static_cast<axos_qp::idx>(i), s, a);
            }
        });
    }

    template <class Args, class F>
    void
    epi(const char *, size_t n, const double *s, const Args &a, F f) const
    {
        detail::parallel_for(n, kGrain, [&](size_t b, size_t e) {
            for (size_t i = b; i < e; ++i)
                f(static_cast<axos_qp::idx>(i), s[i], a);
        });
    }

    template <class Args, class F>
    void
    map(const char *, size_t n, const Args &a, F f) const
    {
        detail::parallel_for(n, kGrain, [&](size_t b, size_t e) {
            for (size_t i = b; i < e; ++i)
                f(static_cast<axos_qp::idx>(i), a);
        });
    }

    template <class Args, class F>
    std::array<double, 8>
    reduce(const char *, size_t n, const Args &a, F f) const
    {
        constexpr int K = Args::K;
        std::array<double, 8> out{};
        const int nt = detail::max_threads();
        const size_t chunks = detail::chunk_count(n, kGrain, nt);
        std::vector<std::array<double, K>> part(chunks);
        detail::parallel_for(
            chunks, 1,
            [&](size_t c0, size_t c1) {
                for (size_t c = c0; c < c1; ++c) {
                    double acc[Args::K] =
                        {}; // (MSVC: no constexpr locals in lambdas)
                    const size_t b = detail::chunk_begin(n, chunks, c, 1);
                    const size_t e = detail::chunk_begin(n, chunks, c + 1, 1);
                    for (size_t i = b; i < e; ++i)
                        f(static_cast<axos_qp::idx>(i), a, acc);
                    for (int k = 0; k < Args::K; ++k)
                        part[c][k] = acc[k];
                }
            },
            1);
        for (size_t c = 0; c < chunks; ++c)
            for (int k = 0; k < K; ++k)
                out[k] = ((Args::kMax >> k) & 1u) ? std::max(out[k], part[c][k])
                                                  : out[k] + part[c][k];
        return out;
    }

    template <class Args, class F>
    void
    reduce_to(const char *name, size_t n, const Args &a, F f, int slot)
    {
        const auto r = reduce(name, n, a, f);
        for (int k = 0; k < Args::K; ++k)
            res_[slot + k] = r[k];
    }
    void
    copy_results(int) const
    {
    }
    const double *
    results() const
    { return res_.data(); }
    const double *
    results_dev() const
    { return res_.data(); }

    template <class F>
    void
    run_block(long, F &&fn) const
    { fn(); }
    void
    forget(long) const
    {
    }
    void
    use_graphs(bool) const
    {
    }

    template <class F>
    void
    through_host(double *v, size_t n, F &&fn) const
    {
        if (n) fn(v);
    }

    void
    sync() const
    {
    }

    template <class F>
    double
    time_us(int reps, F &&fn) const
    {
        fn();
        const auto t0 = std::chrono::steady_clock::now();
        for (int r = 0; r < reps; ++r)
            fn();
        return 1e6 *
               std::chrono::duration<double>(
                   std::chrono::steady_clock::now() - t0)
                   .count() /
               reps;
    }

  private:
    static constexpr size_t kGrain = 16384;
    std::array<double, 64> res_{};
};

} // namespace qp
} // namespace Solver
} // namespace AXOS
