// SPDX-License-Identifier: BSD-3-Clause
//
// Cuda::CudaStorage<T>: a flat array of T in device memory, owned or
// borrowed, and the Cuda::Backend tag with its transfer functions
// (docs/TENSOR_SPEC.md §3-4). Same interface and copy semantics as
// Cpu::HostStorage (see storage/host_storage.h):
//
//   - owning memory comes from GPUMemoryPool::get() and goes back to it; there
//     is no cudaMalloc/cudaFree per object in steady state
//   - (n, x) fills on the device: one cudaMemset when every byte of x is the
//     same (0, 0.0, -1), otherwise a small seed block doubled by
//     device-to-device copies (no kernel, so plain g++ can compile this)
//   - deep copies are device-to-device; constructors and assignments are
//     ordered on the default stream, so later default-stream work sees the
//     data in place
//
// Without AXOS_ENABLE_CUDA (and outside nvcc) the type still exists with the
// same interface, so host-only builds can instantiate templates that mention
// it, but it can never allocate: doing so throws std::runtime_error.
#pragma once

#include "storage/host_storage.h"
#include <algorithm>
#include <array>
#include <cstddef>
#include <limits>
#include <new>
#include <stdexcept>
#include <string>
#include <type_traits>
#include <utility>
#include <vector>

#if defined(AXOS_ENABLE_CUDA) || defined(__CUDACC__)
#define AXOS_CUDA_STORAGE 1
#include "tensorcuda/gpu_pool.h"
#include <cuda_runtime.h>
#endif

namespace AXOS {
namespace Cuda {

namespace cuda_detail {

[[noreturn]] inline void
no_cuda()
{
    throw std::runtime_error("Cuda::CudaStorage: this build has no CUDA "
                             "support (compile with -DAXOS_ENABLE_CUDA)");
}

#if defined(AXOS_CUDA_STORAGE)
inline void
check(cudaError_t e, const char *what)
{
    if (e != cudaSuccess)
        throw std::runtime_error(std::string("CUDA error in ") + what + ": " +
                                 cudaGetErrorString(e));
}

inline void
copy(void *dst, const void *src, size_t bytes, cudaMemcpyKind kind)
{
    if (bytes) check(cudaMemcpy(dst, src, bytes, kind), "cudaMemcpy");
}

inline void
copy2d(void *dst, size_t dpitch, const void *src, size_t spitch, size_t width,
    size_t height, cudaMemcpyKind kind)
{
    if (width == 0 || height == 0) return;
    if (height == 1 || (dpitch == width && spitch == width))
        copy(dst, src, width * height, kind); // contiguous: no pitch limits
    else
        check(cudaMemcpy2D(dst, dpitch, src, spitch, width, height, kind),
            "cudaMemcpy2D");
}

template <class T>
void
fill(T *p, size_t n, const T &x)
{
    if (n == 0) return;
    unsigned char byte;
    if (AXOS::detail::uniform_bytes(x, byte)) {
        check(cudaMemset(p, byte, n * sizeof(T)), "cudaMemset");
        return;
    }
    const size_t seed = std::min<size_t>(n, 4096);
    std::vector<T> h(seed, x);
    copy(p, h.data(), seed * sizeof(T), cudaMemcpyHostToDevice);
    for (size_t done = seed; done < n;) {
        const size_t c = std::min(done, n - done);
        copy(p + done, p, c * sizeof(T), cudaMemcpyDeviceToDevice);
        done += c;
    }
}
#endif

// Visits the 2-D planes (last two dimensions) of t in row-major logical order:
// f(offset, flat, height, width, pitch), with the plane starting at
// t.data + offset, holding logical elements [flat, flat + height * width), rows
// `pitch` elements apart. A contiguous tensor is a single plane.
template <class Tensor, class F>
void
for_each_plane(const Tensor &t, F &&f)
{
    const size_t n = t.size();
    if (n == 0) return;
    constexpr int D = Tensor::rank;
    if constexpr (D == 1) {
        f(size_t(0), size_t(0), size_t(1), n, n);
    } else {
        if (AXOS::detail::tensor_is_contiguous(t)) {
            f(size_t(0), size_t(0), size_t(1), n, n);
            return;
        }
        const size_t h = t.size(D - 2), w = t.size(D - 1);
        const size_t pitch = t.stride(D - 2), plane = h * w;
        std::array<size_t, D> idx{};
        for (size_t flat = 0; flat < n; flat += plane) {
            size_t off = 0;
            for (int d = 0; d < D - 2; ++d)
                off += idx[d] * t.stride(d);
            f(off, flat, h, w, pitch);
            for (int d = D - 3; d >= 0; --d) {
                if (++idx[d] < t.size(d)) break;
                idx[d] = 0;
            }
        }
    }
}

} // namespace cuda_detail

// Backend tag for device memory. All transfers are blocking and move exactly
// t.size() elements (views copied in row-major logical order). read_element is
// a one-element device-to-host copy (about 10 us): do not use it in loops.
struct Backend {
    static constexpr bool is_host = false;

    template <class Tensor>
    static auto
    read_element(const Tensor &t, size_t i)
    {
        using T =
            std::remove_cv_t<std::remove_reference_t<decltype(t.data[0])>>;
        T v{};
#if defined(AXOS_CUDA_STORAGE)
        cuda_detail::copy(&v, t.data + i, sizeof(T), cudaMemcpyDeviceToHost);
#else
        (void)t, (void)i;
        cuda_detail::no_cuda();
#endif
        return v;
    }

    // dst and src have the same shape.
    template <class Tensor>
    static void
    copy_data(Tensor &dst, const Tensor &src)
    {
#if defined(AXOS_CUDA_STORAGE)
        using T = std::remove_reference_t<decltype(*dst.data)>;
        if (AXOS::detail::tensor_is_contiguous(dst) &&
            AXOS::detail::tensor_is_contiguous(src)) {
            cuda_detail::copy(dst.data, src.data, src.size() * sizeof(T),
                cudaMemcpyDeviceToDevice);
            return;
        }
        cuda_detail::for_each_plane(src,
            [&](size_t off, size_t flat, size_t h, size_t w, size_t pitch) {
                size_t doff = flat, dpitch = w;
                if (!AXOS::detail::tensor_is_contiguous(dst)) {
                    doff = AXOS::detail::logical_offset(dst, flat);
                    dpitch = dst.stride(Tensor::rank - 2);
                }
                cuda_detail::copy2d(dst.data + doff, dpitch * sizeof(T),
                    src.data + off, pitch * sizeof(T), w * sizeof(T), h,
                    cudaMemcpyDeviceToDevice);
            });
#else
        (void)dst, (void)src;
        cuda_detail::no_cuda();
#endif
    }

    template <class Tensor, class U>
    static void
    copy_to_host(const Tensor &t, U *host_dst)
    {
#if defined(AXOS_CUDA_STORAGE)
        using T =
            std::remove_cv_t<std::remove_reference_t<decltype(t.data[0])>>;
        if constexpr (!std::is_same_v<T, U>) {
            std::vector<T> tmp(t.size());
            copy_to_host(t, tmp.data());
            AXOS::detail::convert_n(host_dst, tmp.data(), tmp.size());
        } else {
            cuda_detail::for_each_plane(t,
                [&](size_t off, size_t flat, size_t h, size_t w, size_t pitch) {
                    cuda_detail::copy2d(host_dst + flat, w * sizeof(T),
                        t.data + off, pitch * sizeof(T), w * sizeof(T), h,
                        cudaMemcpyDeviceToHost);
                });
        }
#else
        (void)t, (void)host_dst;
        cuda_detail::no_cuda();
#endif
    }

    template <class Tensor, class U>
    static void
    copy_from_host(Tensor &t, const U *host_src)
    {
#if defined(AXOS_CUDA_STORAGE)
        using T =
            std::remove_cv_t<std::remove_reference_t<decltype(t.data[0])>>;
        if constexpr (!std::is_same_v<T, U>) {
            std::vector<T> tmp(t.size());
            AXOS::detail::convert_n(tmp.data(), host_src, tmp.size());
            copy_from_host(t, static_cast<const T *>(tmp.data()));
        } else {
            cuda_detail::for_each_plane(t,
                [&](size_t off, size_t flat, size_t h, size_t w, size_t pitch) {
                    cuda_detail::copy2d(t.data + off, pitch * sizeof(T),
                        host_src + flat, w * sizeof(T), w * sizeof(T), h,
                        cudaMemcpyHostToDevice);
                });
        }
#else
        (void)t, (void)host_src;
        cuda_detail::no_cuda();
#endif
    }

    // Sets every element of t (also of a view) to x.
    template <class Tensor, class U>
    static void
    fill(Tensor &t, const U &x)
    {
#if defined(AXOS_CUDA_STORAGE)
        using T = std::remove_reference_t<decltype(*t.data)>;
        const T v = static_cast<T>(x);
        if (AXOS::detail::tensor_is_contiguous(t)) {
            cuda_detail::fill(t.data, t.size(), v);
            return;
        }
        std::vector<T> plane;
        cuda_detail::for_each_plane(t, [&](size_t off, size_t, size_t h,
                                           size_t w, size_t pitch) {
            plane.assign(h * w, v);
            cuda_detail::copy2d(t.data + off, pitch * sizeof(T), plane.data(),
                w * sizeof(T), w * sizeof(T), h, cudaMemcpyHostToDevice);
        });
#else
        (void)t, (void)x;
        cuda_detail::no_cuda();
#endif
    }
};

template <typename T> class CudaStorage {
    static_assert(std::is_trivially_destructible_v<T>,
        "CudaStorage<T>: T must be a trivially copyable element type");

  public:
    using backend_type = Backend;
    using value_type = T;

    CudaStorage() noexcept = default;

    explicit CudaStorage(size_t n) : ptr_(allocate_raw(n)), n_(n), own_(n > 0)
    {
    }

    CudaStorage(size_t n, T x) : CudaStorage(n)
    {
#if defined(AXOS_CUDA_STORAGE)
        cuda_detail::fill(ptr_, n_, x);
#else
        (void)x;
#endif
    }

    CudaStorage(T *raw, size_t n, bool own = false) noexcept
        : ptr_(raw), n_(raw ? n : 0), own_(own && raw != nullptr)
    {
    }

    CudaStorage(const CudaStorage &o)
    {
        if (o.own_) {
            ptr_ = allocate_raw(o.n_);
            n_ = o.n_;
            own_ = n_ > 0;
            copy_device(ptr_, o.ptr_, n_);
        } else {
            ptr_ = o.ptr_;
            n_ = o.n_;
        }
    }

    CudaStorage &
    operator=(const CudaStorage &o)
    {
        if (this == &o) return *this;
        if (o.own_ && own_ && n_ == o.n_) { // same-size deep copy: reuse
            copy_device(ptr_, o.ptr_, n_);
            return *this;
        }
        CudaStorage tmp(o);
        swap(tmp);
        return *this;
    }

    CudaStorage(CudaStorage &&o) noexcept
        : ptr_(std::exchange(o.ptr_, nullptr)), n_(std::exchange(o.n_, 0)),
          own_(std::exchange(o.own_, false))
    {
    }

    CudaStorage &
    operator=(CudaStorage &&o) noexcept
    {
        if (this != &o) {
            release();
            ptr_ = std::exchange(o.ptr_, nullptr);
            n_ = std::exchange(o.n_, 0);
            own_ = std::exchange(o.own_, false);
        }
        return *this;
    }

    ~CudaStorage() { release(); }

    void
    allocate(size_t n)
    {
        T *p = allocate_raw(n);
        release();
        ptr_ = p;
        n_ = n;
        own_ = n > 0;
    }

    T *
    data() noexcept
    { return ptr_; }
    const T *
    data() const noexcept
    { return ptr_; }
    size_t
    size() const noexcept
    { return n_; }
    bool
    owns() const noexcept
    { return own_; }

    void
    swap(CudaStorage &o) noexcept
    {
        std::swap(ptr_, o.ptr_);
        std::swap(n_, o.n_);
        std::swap(own_, o.own_);
    }

    // The allocator behind owning storages (the GPU pool).
    static T *
    allocate_raw(size_t n)
    {
        if (n == 0) return nullptr;
#if defined(AXOS_CUDA_STORAGE)
        if (n > std::numeric_limits<size_t>::max() / sizeof(T))
            throw std::bad_alloc();
        return static_cast<T *>(GPUMemoryPool::get().allocate(n * sizeof(T)));
#else
        cuda_detail::no_cuda();
#endif
    }

    static void
    free_raw(T *p, size_t n) noexcept
    {
#if defined(AXOS_CUDA_STORAGE)
        if (p) GPUMemoryPool::get().deallocate(p, n * sizeof(T));
#else
        (void)p, (void)n;
#endif
    }

  private:
    static void
    copy_device(T *dst, const T *src, size_t n)
    {
#if defined(AXOS_CUDA_STORAGE)
        cuda_detail::copy(dst, src, n * sizeof(T), cudaMemcpyDeviceToDevice);
#else
        (void)dst, (void)src, (void)n;
        if (n) cuda_detail::no_cuda();
#endif
    }

    void
    release() noexcept
    {
        if (own_) free_raw(ptr_, n_);
        ptr_ = nullptr;
        n_ = 0;
        own_ = false;
    }

    T *ptr_ = nullptr;
    size_t n_ = 0;
    bool own_ = false;
};

} // namespace Cuda
} // namespace AXOS
