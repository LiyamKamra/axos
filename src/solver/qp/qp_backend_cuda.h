// SPDX-License-Identifier: BSD-3-Clause
//
// GPU backend of the first-order QP solvers. Host code only: it needs the
// CUDA runtime and driver headers and libraries (cudart, cuda) but no nvcc.
// The kernels (qp_kernels.cuh, built on qp_ops.h) are compiled once per
// process at run time with NVRTC for the device's architecture and launched
// through the driver API; vectors live in the GPU memory pool. No cuBLAS or
// cuSPARSE: the CSR products are this project's own kernels.
//
// NVRTC is loaded dynamically (nvrtc64_120_0.dll / libnvrtc.so.12, found
// on the library path), so it is not a link-time dependency. The kernel
// sources are read from the source tree: AXOS_SRC_DIR (environment
// variable, else the compile-time macro, else derived from this file's
// path) must name the directory that contains solver/qp/.
//
// Same interface as qp_backend_cpu.h. All work is queued on one stream of
// the backend, so a block of iterations can be captured as a CUDA graph
// (run_block) and replayed with a single launch.
#pragma once

#if !defined(AXOS_ENABLE_CUDA)
#error "qp_backend_cuda.h needs -DAXOS_ENABLE_CUDA (CUDA runtime headers)"
#endif

#include "solver/qp/qp_ops.h"
#include "solver/model.h"
#include "tensorCuda.h"
#include <cuda.h>
#include <cuda_runtime.h>
#include <array>
#include <chrono>
#include <cstdlib>
#include <cstring>
#include <map>
#include <mutex>
#include <stdexcept>
#include <string>
#include <unordered_map>
#include <vector>

#ifdef _WIN32
#ifndef WIN32_LEAN_AND_MEAN
#define WIN32_LEAN_AND_MEAN
#endif
#ifndef NOMINMAX
#define NOMINMAX
#endif
#include <windows.h>
#else
#include <dlfcn.h>
#endif

namespace AXOS {
namespace Solver {
namespace qp {

namespace cuda_rt {

inline void
check(CUresult r, const char *what)
{
    if (r != CUDA_SUCCESS) {
        const char *s = nullptr;
        cuGetErrorString(r, &s);
        throw std::runtime_error(std::string("CUDA driver error in ") + what + ": " +
                                 (s ? s : "?"));
    }
}

inline void
check(cudaError_t e, const char *what)
{
    if (e != cudaSuccess)
        throw std::runtime_error(std::string("CUDA error in ") + what + ": " +
                                 cudaGetErrorString(e));
}

// The subset of the NVRTC API used here, resolved at run time.
struct Nvrtc {
    typedef int (*Create)(void **, const char *, const char *, int, const char *const *, const char *const *);
    typedef int (*Compile)(void *, int, const char *const *);
    typedef int (*Size)(void *, size_t *);
    typedef int (*Get)(void *, char *);
    typedef int (*Destroy)(void **);
    typedef const char *(*ErrStr)(int);
    Create create = nullptr;
    Compile compile = nullptr;
    Size cubin_size = nullptr, log_size = nullptr;
    Get cubin = nullptr, log = nullptr;
    Destroy destroy = nullptr;
    ErrStr error_string = nullptr;

    static const Nvrtc &
    get()
    {
        static Nvrtc n = load();
        return n;
    }

  private:
    static Nvrtc
    load()
    {
        Nvrtc n;
#ifdef _WIN32
        const char *names[] = {"nvrtc64_120_0.dll", "nvrtc64_130_0.dll", "nvrtc64_112_0.dll"};
        HMODULE h = nullptr;
        for (const char *nm : names)
            if ((h = LoadLibraryA(nm)) != nullptr) break;
        auto sym = [&](const char *s) { return reinterpret_cast<void *>(GetProcAddress(h, s)); };
#else
        const char *names[] = {"libnvrtc.so.12", "libnvrtc.so.13", "libnvrtc.so"};
        void *h = nullptr;
        for (const char *nm : names)
            if ((h = dlopen(nm, RTLD_NOW | RTLD_LOCAL)) != nullptr) break;
        auto sym = [&](const char *s) { return dlsym(h, s); };
#endif
        if (!h)
            throw std::runtime_error("NVRTC not found: put the nvrtc library "
                                     "(pip: nvidia-cuda-nvrtc-cu12) on the library path");
        n.create = reinterpret_cast<Create>(sym("nvrtcCreateProgram"));
        n.compile = reinterpret_cast<Compile>(sym("nvrtcCompileProgram"));
        n.cubin_size = reinterpret_cast<Size>(sym("nvrtcGetCUBINSize"));
        n.cubin = reinterpret_cast<Get>(sym("nvrtcGetCUBIN"));
        n.log_size = reinterpret_cast<Size>(sym("nvrtcGetProgramLogSize"));
        n.log = reinterpret_cast<Get>(sym("nvrtcGetProgramLog"));
        n.destroy = reinterpret_cast<Destroy>(sym("nvrtcDestroyProgram"));
        n.error_string = reinterpret_cast<ErrStr>(sym("nvrtcGetErrorString"));
        if (!n.create || !n.compile || !n.cubin_size || !n.cubin || !n.log_size ||
            !n.log || !n.destroy)
            throw std::runtime_error("NVRTC library lacks a required function");
        return n;
    }
};

inline std::string
source_dir()
{
#ifdef _MSC_VER
    char *e = nullptr;
    size_t len = 0;
    if (_dupenv_s(&e, &len, "AXOS_SRC_DIR") == 0 && e) {
        std::string s(e);
        std::free(e);
        return s;
    }
#else
    if (const char *e = std::getenv("AXOS_SRC_DIR")) return e;
#endif
#ifdef AXOS_SRC_DIR
    return AXOS_SRC_DIR;
#else
    std::string f = __FILE__; // .../src/solver/qp/qp_backend_cuda.h
    for (int k = 0; k < 3; ++k) {
        const size_t s = f.find_last_of("/\\");
        if (s == std::string::npos) return ".";
        f = f.substr(0, s);
    }
    return f;
#endif
}

// The compiled kernels, once per process (current device).
class Module {
  public:
    static Module &
    get()
    {
        static std::once_flag once;
        static Module *m = nullptr;
        std::call_once(once, [] { m = new Module(); }); // kept until exit
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

    std::string arch() const { return arch_; }
    double compile_seconds() const { return compile_s_; }

  private:
    Module()
    {
        check(cudaFree(nullptr), "cudaFree(0) (runtime initialization)");
        int dev = 0, major = 0, minor = 0;
        check(cudaGetDevice(&dev), "cudaGetDevice");
        check(cudaDeviceGetAttribute(&major, cudaDevAttrComputeCapabilityMajor, dev), "attribute");
        check(cudaDeviceGetAttribute(&minor, cudaDevAttrComputeCapabilityMinor, dev), "attribute");
        arch_ = "sm_" + std::to_string(major * 10 + minor);
        const auto t0 = std::chrono::steady_clock::now();
        const Nvrtc &nv = Nvrtc::get();
        const std::string src = "#include \"solver/qp/qp_kernels.cuh\"\n";
        const std::string opt_arch = "--gpu-architecture=" + arch_;
        const std::string opt_inc = "-I" + source_dir();
        const char *opts[] = {opt_arch.c_str(), "-std=c++17", opt_inc.c_str(),
            "--fmad=true", "-lineinfo"};
        void *prog = nullptr;
        if (nv.create(&prog, src.c_str(), "axos_qp.cu", 0, nullptr, nullptr) != 0)
            throw std::runtime_error("nvrtcCreateProgram failed");
        const int rc = nv.compile(prog, 5, opts);
        size_t ls = 0;
        nv.log_size(prog, &ls);
        std::string log(ls, '\0');
        if (ls) nv.log(prog, &log[0]);
        if (rc != 0) {
            nv.destroy(&prog);
            throw std::runtime_error("NVRTC compilation of the QP kernels failed (" +
                                     opt_inc + "):\n" + log);
        }
        size_t cs = 0;
        nv.cubin_size(prog, &cs);
        std::vector<char> bin(cs);
        nv.cubin(prog, bin.data());
        nv.destroy(&prog);
        check(cuModuleLoadData(&mod_, bin.data()), "cuModuleLoadData");
        compile_s_ = std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count();
    }

    CUmodule mod_ = nullptr;
    std::string arch_;
    double compile_s_ = 0;
    std::mutex mu_;
    std::unordered_map<std::string, CUfunction> fns_;
};

} // namespace cuda_rt

class CudaBackend {
  public:
    static constexpr bool is_gpu = true;
    using Vec = tensorET<1, double, Cuda::CudaStorage<double>>;
    using IVec = tensorET<1, int32_t, Cuda::CudaStorage<int32_t>>;
    // CSR on the device, T = tpr threads per row (from the mean row length).
    // A row is summed in len / T sequential steps, so a matrix whose longest
    // row would take much longer than the target (ch = T * steps nonzeros,
    // steps chosen for about kThreads threads in flight) is split: the
    // product then runs over chunks of at most ch nonzeros (cp, ctpr threads
    // each) into part, and a second kernel adds each row's chunks (rcp)
    // before the epilogue, so a few dense rows no longer serialize the whole
    // product.
    struct Mat {
        IVec rp, ci;
        Vec v;
        size_t nrows = 0, nnzs = 0;
        int tpr = 1; // threads per row
        bool split = false;
        IVec cp, rcp;
        Vec part;
        size_t nchunks = 0;
        int ctpr = 1; // threads per chunk
        size_t rows() const { return nrows; }
        size_t nnz() const { return nnzs; }
    };

    CudaBackend() : mod_(cuda_rt::Module::get())
    {
        cuda_rt::check(cuStreamCreate(&stream_, CU_STREAM_DEFAULT), "cuStreamCreate");
        part_ = Vec({size_t(kMaxBlocks) * 8}, 0.0);
        out_ = Vec({size_t(kSlots)}, 0.0);
        cuda_rt::check(cudaMallocHost(reinterpret_cast<void **>(&host_), kSlots * sizeof(double)),
            "cudaMallocHost");
        cudaDeviceProp prop;
        cuda_rt::check(cudaGetDeviceProperties(&prop, 0), "cudaGetDeviceProperties");
        name_ = std::string("gpu ") + prop.name;
    }

    ~CudaBackend()
    {
        if (stream_) cuStreamSynchronize(stream_);
        for (auto &g : graphs_) cuGraphExecDestroy(g.second);
        if (stream_) cuStreamDestroy(stream_);
        if (host_) cudaFreeHost(host_);
    }
    CudaBackend(const CudaBackend &) = delete;
    CudaBackend &operator=(const CudaBackend &) = delete;

    std::string name() const { return name_; }
    static double compile_seconds() { return cuda_rt::Module::get().compile_seconds(); }

    // Vectors are created, uploaded and downloaded on the legacy default
    // stream, which is ordered with the backend stream (a blocking stream).
    Vec vec(size_t n, double v = 0.0) const { return Vec({n}, v); }

    Vec
    upload(const std::vector<double> &h) const
    {
        Vec v({h.size()});
        if (!h.empty())
            cuda_rt::check(cudaMemcpy(v.data, h.data(), h.size() * sizeof(double),
                               cudaMemcpyHostToDevice), "upload");
        return v;
    }

    void
    download(const Vec &v, std::vector<double> &h) const
    {
        h.resize(v.size());
        if (!h.empty())
            cuda_rt::check(cudaMemcpy(h.data(), v.data, h.size() * sizeof(double),
                               cudaMemcpyDeviceToHost), "download");
    }

    static double *ptr(Vec &v) { return v.data; }
    static const double *ptr(const Vec &v) { return v.data; }

    Mat
    upload(const HostMatrix &A) const
    {
        Mat M;
        M.nrows = A.rows();
        M.nnzs = A.nnz();
        const int32_t *hrp = A.row_ptr();
        M.rp = ivec(hrp, A.rows() + 1);
        M.ci = ivec(A.nnz() ? A.col_ind() : nullptr, A.nnz());
        tensorET<1, double> v({std::max<size_t>(A.nnz(), 1)}, 0.0);
        if (A.nnz()) std::memcpy(v.data, A.values(), A.nnz() * sizeof(double));
        M.v = Vec(v);
        M.tpr = threads_for(A.rows() ? double(A.nnz()) / double(A.rows()) : 0.0);
        int32_t longest = 0;
        for (size_t i = 0; i < A.rows(); ++i) longest = std::max(longest, hrp[i + 1] - hrp[i]);
        const size_t steps = std::min<size_t>(32, std::max<size_t>(4, A.nnz() / kThreads));
        const int32_t ch = static_cast<int32_t>(steps) * M.tpr;
        if (longest > 4 * ch) {
            std::vector<int32_t> cp{0}, rcp{0};
            for (size_t i = 0; i < A.rows(); ++i) {
                for (int32_t b = hrp[i]; b < hrp[i + 1]; b += ch)
                    cp.push_back(std::min(b + ch, hrp[i + 1]));
                rcp.push_back(static_cast<int32_t>(cp.size() - 1));
            }
            M.split = true;
            M.nchunks = cp.size() - 1;
            M.cp = ivec(cp.data(), cp.size());
            M.rcp = ivec(rcp.data(), rcp.size());
            M.part = Vec({std::max<size_t>(M.nchunks, 1)}, 0.0);
            M.ctpr = threads_for(double(A.nnz()) / double(M.nchunks));
        }
        return M;
    }

    // s = (M x)_i, then f(i, s, a) for every row i (kernels "<name>_<T>").
    template <class Args, class F>
    void
    spmv_epi(const char *name, const Mat &M, const double *x, const Args &a, F)
    {
        if (M.nrows == 0) return;
        axos_qp::idx rows = static_cast<axos_qp::idx>(M.nrows);
        Args copy = a;
        const int *ci = M.ci.data;
        const double *v = M.v.data;
        if (!M.split) {
            const int *rp = M.rp.data;
            void *args[] = {&rows, &rp, &ci, &v, &x, &copy};
            launch(fn(name, M.tpr), blocks(M.nrows * size_t(M.tpr)), args);
            return;
        }
        axos_qp::idx nch = static_cast<axos_qp::idx>(M.nchunks);
        const int *cp = M.cp.data, *rcp = M.rcp.data;
        double *part = const_cast<double *>(M.part.data);
        void *cargs[] = {&nch, &cp, &ci, &v, &x, &part};
        launch(fn("k_chunks", M.ctpr), blocks(M.nchunks * size_t(M.ctpr)), cargs);
        const double *cpart = part;
        void *fargs[] = {&rows, &rcp, &cpart, &copy};
        launch(fn(name, 0), grid(M.nrows, 16384), fargs);
    }

    void
    spmv(const Mat &M, const double *x, double *y)
    {
        spmv_epi("k_store", M, x, axos_qp::Store{y}, 0);
    }

    template <class Args, class F>
    void
    map(const char *name, size_t n, const Args &a, F)
    {
        if (n == 0) return;
        axos_qp::idx nn = static_cast<axos_qp::idx>(n);
        Args copy = a;
        void *args[] = {&nn, &copy};
        launch(mod_.fn(name), grid(n, 16384), args);
    }

    // Reduction into the result slots [slot, slot + Args::K) on the device
    // (zeros when n == 0); nothing is copied back before copy_results().
    template <class Args, class F>
    void
    reduce_to(const char *name, size_t n, const Args &a, F, int slot)
    {
        double *part = part_.data;
        int nb = 0;
        if (n) {
            axos_qp::idx nn = static_cast<axos_qp::idx>(n);
            Args copy = a;
            void *args[] = {&nn, &copy, &part};
            const unsigned b = grid(n, kMaxBlocks);
            launch(mod_.fn(name), b, args);
            nb = static_cast<int>(b);
        }
        int kk = Args::K;
        unsigned mx = Args::kMax;
        double *o = out_.data + slot;
        void *fargs[] = {&nb, &kk, &mx, &part, &o};
        launch(mod_.fn("k_finish"), 1, fargs);
    }

    // Queues the copy of result slots [0, n) to results() (pinned memory, so
    // it can be part of a graph); valid after sync().
    void
    copy_results(int n)
    {
        cuda_rt::check(cuMemcpyDtoHAsync(host_, reinterpret_cast<CUdeviceptr>(out_.data),
                           size_t(n) * sizeof(double), stream_), "copy results");
    }
    const double *results() const { return host_; }
    // the result slots in device memory (for ops that read a reduction)
    const double *results_dev() const { return out_.data; }

    template <class Args, class F>
    std::array<double, 8>
    reduce(const char *name, size_t n, const Args &a, F f)
    {
        reduce_to(name, n, a, f, 0);
        copy_results(Args::K);
        sync();
        std::array<double, 8> out{};
        for (int k = 0; k < Args::K; ++k) out[k] = host_[k];
        return out;
    }

    // Runs fn(), which must only queue work whose arguments are the same on
    // every call with this key: captured into a CUDA graph on the first call
    // and replayed with one launch afterwards.
    template <class F>
    void
    run_block(long key, F &&fn)
    {
        auto it = graphs_.find(key);
        if (it == graphs_.end()) {
            cuda_rt::check(cuStreamBeginCapture(stream_, CU_STREAM_CAPTURE_MODE_THREAD_LOCAL),
                "cuStreamBeginCapture");
            CUgraph g = nullptr;
            try {
                fn();
            } catch (...) {
                cuStreamEndCapture(stream_, &g);
                if (g) cuGraphDestroy(g);
                throw;
            }
            cuda_rt::check(cuStreamEndCapture(stream_, &g), "cuStreamEndCapture");
            CUgraphExec exec = nullptr;
            const CUresult r = cuGraphInstantiate(&exec, g, 0);
            cuGraphDestroy(g);
            cuda_rt::check(r, "cuGraphInstantiate");
            it = graphs_.emplace(key, exec).first;
        }
        cuda_rt::check(cuGraphLaunch(it->second, stream_), "cuGraphLaunch");
    }

    // drops the graph of a key whose buffers are about to be freed
    void
    forget(long key)
    {
        auto it = graphs_.find(key);
        if (it == graphs_.end()) return;
        cuGraphExecDestroy(it->second);
        graphs_.erase(it);
    }

    void sync() const { cuda_rt::check(cuStreamSynchronize(stream_), "sync"); }

    // Microseconds per call of fn() (which queues work), over reps calls
    // timed with events on the backend stream.
    template <class F>
    double
    time_us(int reps, F &&fn)
    {
        CUevent a = nullptr, b = nullptr;
        cuda_rt::check(cuEventCreate(&a, CU_EVENT_DEFAULT), "cuEventCreate");
        cuda_rt::check(cuEventCreate(&b, CU_EVENT_DEFAULT), "cuEventCreate");
        fn(); // warm (graph capture, first launch)
        cuda_rt::check(cuEventRecord(a, stream_), "cuEventRecord");
        for (int r = 0; r < reps; ++r) fn();
        cuda_rt::check(cuEventRecord(b, stream_), "cuEventRecord");
        cuda_rt::check(cuEventSynchronize(b), "cuEventSynchronize");
        float ms = 0;
        cuEventElapsedTime(&ms, a, b);
        cuEventDestroy(a);
        cuEventDestroy(b);
        return 1e3 * double(ms) / reps;
    }

  private:
    static constexpr unsigned kBlock = 256, kMaxBlocks = 1024;
    static constexpr int kSlots = 64;
    static constexpr size_t kThreads = 65536;

    static int
    threads_for(double mean)
    {
        int t = 1;
        while (t < 32 && 2.0 * t <= mean) t *= 2;
        return t;
    }

    static IVec
    ivec(const int32_t *h, size_t n)
    {
        tensorET<1, int32_t> t({std::max<size_t>(n, 1)}, 0);
        if (n && h) std::memcpy(t.data, h, n * sizeof(int32_t));
        return IVec(t);
    }

    static unsigned
    grid(size_t n, unsigned cap)
    {
        const size_t g = (n + kBlock - 1) / kBlock;
        return static_cast<unsigned>(g < cap ? (g ? g : 1) : cap);
    }
    static unsigned blocks(size_t threads) { return grid(threads, 0x7fffffffu); }

    // "<name>_<t>" (t threads per row), or "<name>_fin" when t == 0
    CUfunction
    fn(const char *name, int t)
    {
        std::string s(name);
        s += t ? "_" + std::to_string(t) : std::string("_fin");
        return mod_.fn(s.c_str());
    }

    void
    launch(CUfunction f, unsigned nblocks, void **args)
    {
        cuda_rt::check(cuLaunchKernel(f, nblocks, 1, 1, kBlock, 1, 1, 0, stream_, args, nullptr),
            "cuLaunchKernel");
    }

    cuda_rt::Module &mod_;
    CUstream stream_ = nullptr;
    Vec part_, out_;
    double *host_ = nullptr;
    std::string name_;
    std::map<long, CUgraphExec> graphs_;
};

} // namespace qp
} // namespace Solver
} // namespace AXOS
