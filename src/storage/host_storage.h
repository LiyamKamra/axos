// SPDX-License-Identifier: BSD-3-Clause
//
// Cpu::HostStorage<T>: a flat, contiguous array of T in host memory, owned or
// borrowed, and the Cpu::Backend tag with its transfer functions
// (docs/TENSOR_SPEC.md §3-4). Csr uses the policy directly as a
// `template <typename> class Store` parameter, so this header is standalone.
//
//   HostStorage<T>()             empty: data() == nullptr, size() == 0
//   HostStorage<T>(n)            owns n elements, left uninitialized; no page
//                                is touched (first touch belongs to the caller)
//   HostStorage<T>(n, x)         owns n copies of x
//   HostStorage<T>(p, n, own)    wraps p; frees it only when own is true, in
//                                which case p must come from allocate_raw()
//
// Copying an owning storage deep-copies (reusing the target's buffer when it
// owns one of the same size); copying a borrowing storage aliases the same
// pointer. Moves transfer the pointer and ownership and leave the source
// empty. Buffers are 64-byte aligned. Allocation failure throws
// std::bad_alloc (the interior-point method catches it for oversized
// factorizations). Copies and fills above a few MiB run on all threads.
#pragma once

#include "tensor/parallel.h"
#include <array>
#include <cstddef>
#include <cstring>
#include <limits>
#include <new>
#include <type_traits>
#include <utility>

namespace AXOS {

namespace Cpu {
struct Backend;
template <typename T> class HostStorage;
} // namespace Cpu

namespace detail {

inline constexpr size_t kHostAlign = 64;

template <class, class = void> struct has_is_contiguous : std::false_type {};
template <class X>
struct has_is_contiguous<X,
    std::void_t<decltype(std::declval<const X &>().is_contiguous())>>
    : std::true_type {};

template <class Tensor>
bool
tensor_is_contiguous(const Tensor &t)
{
    if constexpr (has_is_contiguous<Tensor>::value)
        return t.is_contiguous();
    else
        return true;
}

// Element offset (from t.data) of the element with row-major logical index
// `flat`. Only needed for non-contiguous views.
template <class Tensor>
size_t
logical_offset(const Tensor &t, size_t flat)
{
    constexpr int D = Tensor::rank;
    size_t off = 0;
    for (int d = D - 1; d >= 0; --d) {
        const size_t e = t.size(d);
        off += (flat % e) * t.stride(d);
        flat /= e;
    }
    return off;
}

// Calls f(offset, flat, len) for each run of `len` consecutive elements of t,
// in row-major logical order: the run starts at t.data + offset and holds
// logical elements [flat, flat + len). A contiguous tensor is one run; a
// view has one run per innermost row (the innermost stride is always 1).
template <class Tensor, class F>
void
for_each_run(const Tensor &t, F &&f)
{
    const size_t n = t.size();
    if (n == 0) return;
    if (tensor_is_contiguous(t)) {
        f(size_t(0), size_t(0), n);
        return;
    }
    constexpr int D = Tensor::rank;
    const size_t len = t.size(D - 1);
    std::array<size_t, D> idx{};
    for (size_t flat = 0; flat < n; flat += len) {
        size_t off = 0;
        for (int d = 0; d < D - 1; ++d)
            off += idx[d] * t.stride(d);
        f(off, flat, len);
        for (int d = D - 2; d >= 0; --d) {
            if (++idx[d] < t.size(d)) break;
            idx[d] = 0;
        }
    }
}

template <class T, class U>
void
convert_n(T *dst, const U *src, size_t n)
{
    if constexpr (std::is_same_v<T, U>) {
        copy_bytes(dst, src, n * sizeof(T));
    } else {
        for (size_t i = 0; i < n; ++i)
            dst[i] = static_cast<T>(src[i]);
    }
}

} // namespace detail

namespace Cpu {

// Backend tag for host memory. The transfer functions work on anything with
// a `data` pointer and size() (tensorET); non-contiguous views are copied in
// row-major logical order.
struct Backend {
    static constexpr bool is_host = true;

    template <class Tensor>
    static auto
    read_element(const Tensor &t, size_t i)
    {
        using T = std::remove_cv_t<std::remove_reference_t<decltype(t.data[0])>>;
        return T(t.data[i]);
    }

    // dst and src have the same shape.
    template <class Tensor>
    static void
    copy_data(Tensor &dst, const Tensor &src)
    {
        using T = std::remove_reference_t<decltype(*dst.data)>;
        const bool dc = detail::tensor_is_contiguous(dst);
        const bool sc = detail::tensor_is_contiguous(src);
        if (dc && sc) {
            detail::copy_bytes(dst.data, src.data, src.size() * sizeof(T));
        } else if (sc) {
            detail::for_each_run(dst, [&](size_t off, size_t flat, size_t len) {
                std::memcpy(dst.data + off, src.data + flat, len * sizeof(T));
            });
        } else {
            detail::for_each_run(src, [&](size_t off, size_t flat, size_t len) {
                T *d = dc ? dst.data + flat
                          : dst.data + detail::logical_offset(dst, flat);
                std::memcpy(d, src.data + off, len * sizeof(T));
            });
        }
    }

    // Copies t.size() elements to host_dst (converting when U != T).
    template <class Tensor, class U>
    static void
    copy_to_host(const Tensor &t, U *host_dst)
    {
        detail::for_each_run(t, [&](size_t off, size_t flat, size_t len) {
            detail::convert_n(host_dst + flat, t.data + off, len);
        });
    }

    template <class Tensor, class U>
    static void
    copy_from_host(Tensor &t, const U *host_src)
    {
        detail::for_each_run(t, [&](size_t off, size_t flat, size_t len) {
            detail::convert_n(t.data + off, host_src + flat, len);
        });
    }

    // Sets every element of t (also of a view) to x.
    template <class Tensor, class U>
    static void
    fill(Tensor &t, const U &x)
    {
        using T = std::remove_reference_t<decltype(*t.data)>;
        const T v = static_cast<T>(x);
        detail::for_each_run(t, [&](size_t off, size_t, size_t len) {
            detail::fill_elems(t.data + off, len, v);
        });
    }
};

template <typename T> class HostStorage {
    static_assert(std::is_trivially_destructible_v<T>,
        "HostStorage<T>: T must be a trivially destructible element type");

  public:
    using backend_type = Backend;
    using value_type = T;

    HostStorage() noexcept = default;

    explicit HostStorage(size_t n) : ptr_(allocate_raw(n)), n_(n), own_(n > 0) {}

    HostStorage(size_t n, T x) : HostStorage(n) { detail::fill_elems(ptr_, n_, x); }

    HostStorage(T *raw, size_t n, bool own = false) noexcept
        : ptr_(raw), n_(raw ? n : 0), own_(own && raw != nullptr)
    {
    }

    HostStorage(const HostStorage &o)
    {
        if (o.own_) {
            ptr_ = allocate_raw(o.n_);
            n_ = o.n_;
            own_ = n_ > 0;
            detail::copy_bytes(ptr_, o.ptr_, n_ * sizeof(T));
        } else {
            ptr_ = o.ptr_;
            n_ = o.n_;
        }
    }

    HostStorage &
    operator=(const HostStorage &o)
    {
        if (this == &o) return *this;
        if (o.own_ && own_ && n_ == o.n_) { // same-size deep copy: reuse
            detail::copy_bytes(ptr_, o.ptr_, n_ * sizeof(T));
            return *this;
        }
        HostStorage tmp(o);
        swap(tmp);
        return *this;
    }

    HostStorage(HostStorage &&o) noexcept
        : ptr_(std::exchange(o.ptr_, nullptr)), n_(std::exchange(o.n_, 0)),
          own_(std::exchange(o.own_, false))
    {
    }

    HostStorage &
    operator=(HostStorage &&o) noexcept
    {
        if (this != &o) {
            release();
            ptr_ = std::exchange(o.ptr_, nullptr);
            n_ = std::exchange(o.n_, 0);
            own_ = std::exchange(o.own_, false);
        }
        return *this;
    }

    ~HostStorage() { release(); }

    // Releases the current contents, then owns n uninitialized elements.
    void
    allocate(size_t n)
    {
        T *p = allocate_raw(n); // may throw: leave *this unchanged
        release();
        ptr_ = p;
        n_ = n;
        own_ = n > 0;
    }

    T *data() noexcept { return ptr_; }
    const T *data() const noexcept { return ptr_; }
    size_t size() const noexcept { return n_; }
    bool owns() const noexcept { return own_; }

    void
    swap(HostStorage &o) noexcept
    {
        std::swap(ptr_, o.ptr_);
        std::swap(n_, o.n_);
        std::swap(own_, o.own_);
    }

    // The allocator behind owning storages (64-byte aligned, uninitialized).
    static T *
    allocate_raw(size_t n)
    {
        if (n == 0) return nullptr;
        if (n > std::numeric_limits<size_t>::max() / sizeof(T))
            throw std::bad_alloc();
        return static_cast<T *>(::operator new(
            n * sizeof(T), std::align_val_t(detail::kHostAlign)));
    }

    static void
    free_raw(T *p) noexcept
    {
        if (p) ::operator delete(p, std::align_val_t(detail::kHostAlign));
    }

  private:
    void
    release() noexcept
    {
        if (own_) free_raw(ptr_);
        ptr_ = nullptr;
        n_ = 0;
        own_ = false;
    }

    T *ptr_ = nullptr;
    size_t n_ = 0;
    bool own_ = false;
};

} // namespace Cpu

// True exactly when S::backend_type is Cpu::Backend.
template <class S, class = void> struct is_host_storage : std::false_type {};
template <class S>
struct is_host_storage<S, std::void_t<typename S::backend_type>>
    : std::is_same<typename S::backend_type, Cpu::Backend> {};
template <class S>
inline constexpr bool is_host_storage_v = is_host_storage<S>::value;

// backend_of_t<X>: X::backend_type when X has one, otherwise Cpu::Backend.
template <class X, class = void> struct backend_of { using type = Cpu::Backend; };
template <class X>
struct backend_of<X, std::void_t<typename X::backend_type>> {
    using type = typename X::backend_type;
};
template <class X> using backend_of_t = typename backend_of<X>::type;

} // namespace AXOS
