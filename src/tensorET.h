// SPDX-License-Identifier: BSD-3-Clause
//
// tensorET<DIM, T, Storage>: a dense, row-major, rank-DIM array whose
// elements live in the memory of Storage (docs/TENSOR_SPEC.md §5).
//
//   tensorET<1, double> v({n}, 0.0);          // host vector, zero-filled
//   tensorET<2, float> A({m, k});              // uninitialized (no page touched)
//   tensorET<1, double, Cuda::CudaStorage<double>> d(v);   // one upload
//   tensorET<1, double> h(d);                              // one download
//
// Layout and ownership:
//   - `data` (a public field) points at element 0 in the storage's memory
//     space; it always equals storage_.data(). Owning tensors hold their own
//     buffer; views (slice, reshape, the raw-pointer constructor, Csr's
//     values_view) hold a borrowing storage and are valid while the memory
//     they came from lives.
//   - Copies follow the storage: an owning tensor deep-copies, a view stays a
//     view of the same memory. Moves steal the buffer and leave the source
//     empty (size() == 0, data == nullptr).
//   - Copying between storages (host <-> device) allocates in the target's
//     storage and transfers once, blocking.
//   - The innermost stride is always 1; views keep their parent's strides.
//
// Element access on the host: operator[] / value(i) (flat buffer index, no
// bounds check) and operator()(i0, ..., i_{DIM-1}). On the device only the
// const by-value forms exist, each a device-to-host transfer; writable
// element access does not compile (per-element round trips in loops are a
// performance bug).
//
// Host tensors are expressions (tensor/expr.h): `C = A + B * 2.0` is one
// fused pass; see tensorMath.h for the operators and functions.
#pragma once

#include "storage/host_storage.h"
#include "tensor/element.h"
#include "tensor/expr.h"
#include "tensor/parallel.h"
#include <algorithm>
#include <array>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <initializer_list>
#include <optional>
#include <random>
#include <stdexcept>
#include <string>
#include <type_traits>
#include <utility>

namespace AXOS {

// Per-dimension extent and [start, end] index range (inclusive), as returned
// by tensorET::getIndex(). By default start = 0 and end = extent - 1.
template <int DIM> class Domain {
  public:
    struct Dim {
        size_t extent;
        long long start, end;
    };
    size_t extent(int d) const { return ext_.at(d); }
    long long start(int d) const { return lo_.at(d); }
    long long end(int d) const { return hi_.at(d); }
    Dim operator[](int d) const { return {extent(d), start(d), end(d)}; }
    static constexpr int rank = DIM;

  private:
    template <int, typename, typename> friend class tensorET;
    std::array<size_t, DIM> ext_{};
    std::array<long long, DIM> lo_{}, hi_{};
};

namespace detail {

template <class P>
inline constexpr bool is_dims_ptr_v =
    std::is_same_v<P, size_t *> || std::is_same_v<P, const size_t *>;

struct no_expr_base {};
template <bool Host, class T, class Self>
using tensor_base_t = std::conditional_t<Host, Expr<T, Self>, no_expr_base>;

[[noreturn]] inline void
tensor_error(const std::string &what)
{
    throw std::runtime_error("tensorET: " + what);
}

inline std::uint64_t
mix_seed(std::uint64_t x)
{ // splitmix64 finalizer
    x += 0x9e3779b97f4a7c15ULL;
    x = (x ^ (x >> 30)) * 0xbf58476d1ce4e5b9ULL;
    x = (x ^ (x >> 27)) * 0x94d049bb133111ebULL;
    return x ^ (x >> 31);
}

} // namespace detail

template <int DIM, typename T, typename Storage = Cpu::HostStorage<T>>
class tensorET
    : public detail::tensor_base_t<is_host_storage_v<Storage>, T,
          tensorET<DIM, T, Storage>> {
    static_assert(DIM >= 1, "tensorET: DIM must be at least 1");
    static_assert(is_tensor_element_v<T>,
        "tensorET: unsupported element type; allowed are float, double, "
        "int32_t, int64_t, std::complex<float|double> and AXOS::half "
        "(specialize AXOS::is_tensor_element to add more)");

  public:
    using value_type = T;
    using data_type = T;
    using storage_type = Storage;
    using backend_type = typename Storage::backend_type;
    using shape_type = std::array<size_t, DIM>;
    static constexpr int rank = DIM;
    static constexpr bool on_host = is_host_storage_v<Storage>;

    // Element 0 in the storage's memory space (a device pointer on CUDA).
    T *data = nullptr;
    // The owning or borrowing storage behind `data`.
    Storage storage_;

    // ---- construction ---------------------------------------------------

    tensorET() noexcept = default;

    // Shape `dims`, contents unspecified. Throws if dims.size() != DIM.
    tensorET(std::initializer_list<size_t> dims) { init_owned_(checked_(dims)); }
    tensorET(std::initializer_list<size_t> dims, T x)
    {
        init_owned_(checked_(dims), &x);
    }
    explicit tensorET(const shape_type &dims) { init_owned_(dims); }
    tensorET(const shape_type &dims, T x) { init_owned_(dims, &x); }

    // n elements: shape {n} for DIM == 1, {1, ..., 1, n} otherwise.
    explicit tensorET(size_t n) { init_owned_(vector_shape_(n)); }
    tensorET(size_t n, T x) { init_owned_(vector_shape_(n), &x); }

    // Array forms: size_t dims[DIM].
    template <class P, std::enable_if_t<detail::is_dims_ptr_v<P>, int> = 0>
    explicit tensorET(P dims)
    {
        init_owned_(from_ptr_(dims));
    }
    template <class P, std::enable_if_t<detail::is_dims_ptr_v<P>, int> = 0>
    tensorET(P dims, T x)
    {
        init_owned_(from_ptr_(dims), &x);
    }

    // View of an existing buffer in Storage's memory space. With own ==
    // true the tensor frees the buffer through Storage (it must come from
    // Storage's allocator, e.g. Cpu::HostStorage<T>::allocate_raw).
    template <class P, std::enable_if_t<detail::is_dims_ptr_v<P>, int> = 0>
    tensorET(T *raw, P dims, bool own = false)
        : storage_(raw, product_(from_ptr_(dims)), own)
    {
        set_shape_(from_ptr_(dims));
        data = storage_.data();
    }

    // Adopts an existing storage object holding the product of dims elements.
    template <class P, std::enable_if_t<detail::is_dims_ptr_v<P>, int> = 0>
    tensorET(Storage &&s, P dims) : storage_(std::move(s))
    {
        set_shape_(from_ptr_(dims));
        if (storage_.size() < size_)
            detail::tensor_error("adopted storage is smaller than the shape");
        data = storage_.data();
    }

    // Shape dims with a recorded sub-domain [start, end] (inclusive) per
    // dimension (read back through getIndex()). Contents unspecified.
    template <class P, std::enable_if_t<detail::is_dims_ptr_v<P>, int> = 0>
    tensorET(P dims, P start, P end)
    {
        init_owned_(from_ptr_(dims));
        custom_domain_ = true;
        for (int d = 0; d < DIM; ++d) {
            dlo_[d] = static_cast<long long>(start[d]);
            dhi_[d] = static_cast<long long>(end[d]);
        }
    }

    tensorET(const tensorET &o)
        : storage_(o.storage_), size_(o.size_), dims_(o.dims_),
          strides_(o.strides_), dlo_(o.dlo_), dhi_(o.dhi_),
          custom_domain_(o.custom_domain_)
    {
        data = storage_.data();
    }

    tensorET(tensorET &&o) noexcept
        : storage_(std::move(o.storage_)), size_(o.size_), dims_(o.dims_),
          strides_(o.strides_), dlo_(o.dlo_), dhi_(o.dhi_),
          custom_domain_(o.custom_domain_)
    {
        data = storage_.data();
        o.reset_empty_();
    }

    tensorET &
    operator=(const tensorET &o)
    {
        if (this != &o) {
            storage_ = o.storage_; // deep copy (reusing a same-size buffer) or alias
            copy_meta_(o);
            data = storage_.data();
        }
        return *this;
    }

    tensorET &
    operator=(tensorET &&o) noexcept
    {
        if (this != &o) {
            storage_ = std::move(o.storage_);
            copy_meta_(o);
            data = storage_.data();
            o.reset_empty_();
        }
        return *this;
    }

    // Cross-storage conversion: a new owning tensor of o's shape in this
    // storage, filled by one blocking transfer.
    template <class S2, std::enable_if_t<!std::is_same_v<S2, Storage>, int> = 0>
    tensorET(const tensorET<DIM, T, S2> &o)
    {
        init_owned_(o.shape());
        transfer_from_(o);
    }

    template <class S2, std::enable_if_t<!std::is_same_v<S2, Storage>, int> = 0>
    tensorET &
    operator=(const tensorET<DIM, T, S2> &o)
    {
        if (storage_.owns() && size_ == o.size() && size_ > 0) {
            set_shape_(o.shape()); // same element count: reuse the buffer
            transfer_from_(o);
        } else {
            tensorET t(o);
            *this = std::move(t);
        }
        return *this;
    }

    // Host only: evaluates an element-wise expression into a new tensor.
    template <class E, bool H = on_host,
        std::enable_if_t<H && !detail::is_tensor_v<E>, int> = 0>
    tensorET(const Expr<T, E> &e)
    {
        const E &x = e.self();
        init_owned_(expr_shape_(x));
        detail::assign_expr(*this, x);
    }

    // Host only: evaluates into the existing buffer (shapes must match; an
    // empty default-constructed tensor is allocated first).
    template <class E, bool H = on_host,
        std::enable_if_t<H && !detail::is_tensor_v<E>, int> = 0>
    tensorET &
    operator=(const Expr<T, E> &e)
    {
        const E &x = e.self();
        if (data == nullptr && size_ == 0) {
            const shape_type s = expr_shape_(x);
            if (product_(s) != 0) *this = tensorET(s);
        }
        detail::assign_expr(*this, x);
        return *this;
    }

    // Deep copy into a new owning, contiguous tensor (also of views).
    tensorET
    clone() const
    {
        tensorET t(dims_);
        if (size_) backend_type::copy_data(t, *this);
        t.dlo_ = dlo_;
        t.dhi_ = dhi_;
        t.custom_domain_ = custom_domain_;
        return t;
    }

    // Sets every element (also of views) to x.
    void
    fill(T x)
    {
        if (size_) backend_type::fill(*this, x);
    }

    // ---- size queries ---------------------------------------------------

    size_t size() const noexcept { return size_; }

    size_t
    size(int d) const
    {
        check_dim_(d);
        return dims_[d];
    }

    size_t
    stride(int d) const
    {
        check_dim_(d);
        return strides_[d];
    }

    const shape_type &shape() const noexcept { return dims_; }
    const shape_type &strides() const noexcept { return strides_; }
    bool empty() const noexcept { return size_ == 0; }
    bool owns_memory() const noexcept { return storage_.owns(); }

    bool
    is_contiguous() const noexcept
    {
        size_t s = 1;
        for (int d = DIM - 1; d >= 0; --d) {
            if (dims_[d] != 1 && strides_[d] != s) return false;
            s *= dims_[d];
        }
        return true;
    }

    // ---- element access -------------------------------------------------

  private:
    // By-value result that cannot be assigned to (const for class types).
    using ro_t = std::conditional_t<std::is_class_v<T>, const T, T>;

  public:
    template <bool H = on_host, std::enable_if_t<H, int> = 0>
    T &
    operator[](size_t i) noexcept
    {
        return data[i];
    }
    template <bool H = on_host, std::enable_if_t<H, int> = 0>
    const T &
    operator[](size_t i) const noexcept
    {
        return data[i];
    }
    template <bool H = on_host, std::enable_if_t<!H, int> = 0>
    ro_t
    operator[](size_t i) const
    {
        return backend_type::read_element(*this, i);
    }

    template <bool H = on_host, std::enable_if_t<H, int> = 0>
    T &
    value(size_t i) noexcept
    {
        return data[i];
    }
    ro_t
    value(size_t i) const
    {
        if constexpr (on_host)
            return data[i];
        else
            return backend_type::read_element(*this, i);
    }

    template <class... I,
        std::enable_if_t<sizeof...(I) == DIM &&
                             std::conjunction_v<std::is_integral<I>...>,
            int> = 0,
        bool H = on_host, std::enable_if_t<H, int> = 0>
    T &
    operator()(I... idx) noexcept
    {
        return data[offset_(idx...)];
    }
    template <class... I,
        std::enable_if_t<sizeof...(I) == DIM &&
                             std::conjunction_v<std::is_integral<I>...>,
            int> = 0,
        bool H = on_host, std::enable_if_t<H, int> = 0>
    const T &
    operator()(I... idx) const noexcept
    {
        return data[offset_(idx...)];
    }
    template <class... I,
        std::enable_if_t<sizeof...(I) == DIM &&
                             std::conjunction_v<std::is_integral<I>...>,
            int> = 0,
        bool H = on_host, std::enable_if_t<!H, int> = 0>
    ro_t
    operator()(I... idx) const
    {
        return backend_type::read_element(*this, offset_(idx...));
    }

    // ---- expression protocol (tensor/expr.h) ----------------------------

    static constexpr int cost = 0;
    static constexpr bool packet_ok = true;
    bool contiguous() const noexcept { return is_contiguous(); }
    T lvalue(size_t i) const { return data[detail::logical_offset(*this, i)]; }
    template <class P>
    AXOS_INLINE typename P::reg
    packet(size_t i) const
    {
        return P::loadu(data + i);
    }

    // ---- views (valid while the source memory lives) --------------------

    // The box [start, start + length); strides are this tensor's.
    tensorET
    slice(const shape_type &start, const shape_type &length) const
    {
        tensorET v;
        size_t off = 0;
        for (int d = 0; d < DIM; ++d)
            off += start[d] * strides_[d];
        v.dims_ = length;
        v.strides_ = strides_;
        v.size_ = product_(length);
        size_t span = 0;
        if (v.size_) {
            span = 1;
            for (int d = 0; d < DIM; ++d)
                span += (length[d] - 1) * strides_[d];
        }
        v.storage_ = Storage(data ? data + off : nullptr, span, false);
        v.data = v.storage_.data();
        return v;
    }

    // Rank-N view of the same (contiguous) buffer.
    template <int N>
    tensorET<N, T, Storage>
    reshape(const std::array<size_t, N> &dims) const
    {
        size_t n = 1;
        for (size_t e : dims)
            n *= e;
        if (n != size_)
            detail::tensor_error("reshape: element count " + std::to_string(n) +
                                 " != " + std::to_string(size_));
        if (!is_contiguous())
            detail::tensor_error("reshape: the tensor is not contiguous");
        size_t d[N];
        for (int k = 0; k < N; ++k)
            d[k] = dims[k];
        return tensorET<N, T, Storage>(data, d, false);
    }

    tensorET<1, T, Storage>
    flatten() const
    {
        return reshape<1>(std::array<size_t, 1>{size_});
    }

    // ---- domain ---------------------------------------------------------

    Domain<DIM>
    getIndex() const
    {
        Domain<DIM> r;
        for (int d = 0; d < DIM; ++d) {
            r.ext_[d] = dims_[d];
            r.lo_[d] = custom_domain_ ? dlo_[d] : 0;
            r.hi_[d] = custom_domain_ ? dhi_[d]
                                      : static_cast<long long>(dims_[d]) - 1;
        }
        return r;
    }

    // ---- 2-D tiling helpers (host) --------------------------------------
    // Tile (i, j) of size m covers rows [i*m, i*m+m) and columns
    // [j*m, j*m+m), clipped to the matrix.

    tensorET
    blockView(size_t i, size_t j, size_t m) const
    {
        static_assert(DIM == 2, "blockView needs a matrix");
        return slice({i * m, j * m}, {m, m});
    }

    // Writes tile (i, j), or with `transpose` the transpose of tile (j, i),
    // into the caller's m*m row-major buffer, zero outside the matrix, and
    // returns a non-owning m x m view of buf.
    tensorET<2, T, Storage>
    blockCopy(size_t i, size_t j, size_t m, T *buf, bool transpose = false) const
    {
        static_assert(DIM == 2 && on_host, "blockCopy needs a host matrix");
        std::fill(buf, buf + m * m, T(0));
        const size_t r0 = (transpose ? j : i) * m, c0 = (transpose ? i : j) * m;
        const size_t nr = r0 < dims_[0] ? std::min(m, dims_[0] - r0) : 0;
        const size_t nc = c0 < dims_[1] ? std::min(m, dims_[1] - c0) : 0;
        for (size_t r = 0; r < nr; ++r) {
            const T *src = data + (r0 + r) * strides_[0] + c0;
            if (!transpose)
                std::copy(src, src + nc, buf + r * m);
            else
                for (size_t c = 0; c < nc; ++c)
                    buf[c * m + r] = src[c];
        }
        size_t d[2] = {m, m};
        return tensorET<2, T, Storage>(buf, d, false);
    }

    // Owning, zero-padded copy of tile (i, j).
    tensorET
    blockCopy(size_t i, size_t j, size_t m) const
    {
        static_assert(DIM == 2 && on_host, "blockCopy needs a host matrix");
        tensorET t(shape_type{m, m});
        blockCopy(i, j, m, t.data);
        return t;
    }

    // Copies the in-range part of an m x m tile into tile (i, j).
    template <class S2>
    void
    blockWriteBack(size_t i, size_t j, size_t m, const tensorET<2, T, S2> &tile)
    {
        write_back_(i, j, m, tile, false);
    }
    template <class S2>
    void
    blockAddWriteBack(size_t i, size_t j, size_t m, const tensorET<2, T, S2> &tile)
    {
        write_back_(i, j, m, tile, true);
    }

    // ---- random fill ----------------------------------------------------
    // i.i.d. samples; without a seed the values are non-deterministic. With
    // a seed they depend only on the seed (not on the thread count).
    // Complex types draw real and imaginary parts independently from the
    // real parts of the bounds / mean.

    static tensorET
    uniform(const shape_type &dims, T lo = T(0), T hi = T(1),
        std::optional<std::uint64_t> seed = std::nullopt)
    {
        return random_(dims, seed, [lo, hi](std::mt19937_64 &g, T *p, size_t n) {
            if constexpr (std::is_integral_v<T>) {
                std::uniform_int_distribution<T> d(lo, hi);
                for (size_t i = 0; i < n; ++i) p[i] = d(g);
            } else if constexpr (is_complex_v<T>) {
                using R = real_type_t<T>;
                std::uniform_real_distribution<R> d(std::real(lo), std::real(hi));
                for (size_t i = 0; i < n; ++i) {
                    const R re = d(g);
                    p[i] = T(re, d(g));
                }
            } else {
                std::uniform_real_distribution<double> d(static_cast<double>(lo),
                    static_cast<double>(hi));
                for (size_t i = 0; i < n; ++i) p[i] = static_cast<T>(d(g));
            }
        });
    }

    static tensorET
    gaussian(const shape_type &dims, T mean = T(0), T stddev = T(1),
        std::optional<std::uint64_t> seed = std::nullopt)
    {
        static_assert(!std::is_integral_v<T>,
            "gaussian needs a floating-point or complex element type");
        return random_(dims, seed,
            [mean, stddev](std::mt19937_64 &g, T *p, size_t n) {
                if constexpr (is_complex_v<T>) {
                    using R = real_type_t<T>;
                    std::normal_distribution<R> d(std::real(mean), std::real(stddev));
                    for (size_t i = 0; i < n; ++i) {
                        const R re = d(g);
                        p[i] = T(re, d(g));
                    }
                } else {
                    std::normal_distribution<double> d(static_cast<double>(mean),
                        static_cast<double>(stddev));
                    for (size_t i = 0; i < n; ++i) p[i] = static_cast<T>(d(g));
                }
            });
    }

  private:
    template <int, typename, typename> friend class tensorET;

    static size_t
    product_(const shape_type &s) noexcept
    {
        size_t n = 1;
        for (size_t e : s)
            n *= e;
        return n;
    }

    static shape_type
    checked_(std::initializer_list<size_t> dims)
    {
        if (dims.size() != size_t(DIM))
            detail::tensor_error("expected " + std::to_string(DIM) +
                                 " dimensions, got " +
                                 std::to_string(dims.size()));
        shape_type s{};
        std::copy(dims.begin(), dims.end(), s.begin());
        return s;
    }

    template <class P>
    static shape_type
    from_ptr_(P dims)
    {
        shape_type s{};
        for (int d = 0; d < DIM; ++d)
            s[d] = dims[d];
        return s;
    }

    static shape_type
    vector_shape_(size_t n)
    {
        shape_type s;
        s.fill(1);
        s[DIM - 1] = n;
        return s;
    }

    template <class E>
    static shape_type
    expr_shape_(const E &x)
    {
        if constexpr (detail::has_rank<E>::value)
            static_assert(E::rank == DIM, "tensorET(expression): rank mismatch");
        shape_type s;
        for (int d = 0; d < DIM; ++d)
            s[d] = x.size(d);
        return s;
    }

    void
    set_shape_(const shape_type &s) noexcept
    {
        dims_ = s;
        size_t st = 1;
        for (int d = DIM - 1; d >= 0; --d) {
            strides_[d] = st;
            st *= s[d];
        }
        size_ = st;
        custom_domain_ = false;
    }

    void
    init_owned_(const shape_type &s, const T *x = nullptr)
    {
        set_shape_(s);
        storage_ = x ? Storage(size_, *x) : Storage(size_);
        data = storage_.data();
    }

    void
    copy_meta_(const tensorET &o) noexcept
    {
        size_ = o.size_;
        dims_ = o.dims_;
        strides_ = o.strides_;
        dlo_ = o.dlo_;
        dhi_ = o.dhi_;
        custom_domain_ = o.custom_domain_;
    }

    void
    reset_empty_() noexcept
    {
        data = nullptr;
        size_ = 0;
        dims_ = {};
        strides_ = {};
        custom_domain_ = false;
    }

    void
    check_dim_(int d) const
    {
        if (d < 0 || d >= DIM)
            detail::tensor_error("dimension " + std::to_string(d) +
                                 " out of range for rank " + std::to_string(DIM));
    }

    template <class... I>
    AXOS_INLINE size_t
    offset_(I... idx) const noexcept
    {
        const size_t ix[DIM] = {static_cast<size_t>(idx)...};
        size_t off = ix[DIM - 1]; // innermost stride is 1
        for (int d = 0; d < DIM - 1; ++d)
            off += ix[d] * strides_[d];
        return off;
    }

    // *this is owning, contiguous and has o's shape.
    template <class S2>
    void
    transfer_from_(const tensorET<DIM, T, S2> &o)
    {
        using SB = typename S2::backend_type;
        if (size_ == 0) return;
        if constexpr (on_host) {
            SB::copy_to_host(o, data);
        } else if constexpr (is_host_storage_v<S2>) {
            if (o.is_contiguous()) {
                backend_type::copy_from_host(*this, o.data);
            } else {
                const tensorET<DIM, T, S2> h = o.clone();
                backend_type::copy_from_host(*this, h.data);
            }
        } else { // between two device storages: through the host
            const tensorET<DIM, T> h(o);
            backend_type::copy_from_host(*this, h.data);
        }
    }

    template <class S2>
    void
    write_back_(size_t i, size_t j, size_t m, const tensorET<2, T, S2> &tile,
        bool add)
    {
        static_assert(DIM == 2 && on_host && is_host_storage_v<S2>,
            "blockWriteBack needs host matrices");
        const size_t r0 = i * m, c0 = j * m;
        const size_t nr = r0 < dims_[0] ? std::min(m, dims_[0] - r0) : 0;
        const size_t nc = c0 < dims_[1] ? std::min(m, dims_[1] - c0) : 0;
        nr_nc_check_(tile, m);
        for (size_t r = 0; r < nr; ++r) {
            const T *src = tile.data + r * tile.stride(0);
            T *dst = data + (r0 + r) * strides_[0] + c0;
            if (add)
                for (size_t c = 0; c < nc; ++c) dst[c] += src[c];
            else
                std::copy(src, src + nc, dst);
        }
    }

    template <class S2>
    static void
    nr_nc_check_(const tensorET<2, T, S2> &tile, size_t m)
    {
        if (tile.size(0) < m || tile.size(1) < m)
            detail::tensor_error("block write-back: tile smaller than m x m");
    }

    template <class F>
    static tensorET
    random_(const shape_type &dims, std::optional<std::uint64_t> seed, F fill)
    {
        tensorET<DIM, T> h(dims);
        std::uint64_t base;
        if (seed) {
            base = *seed;
        } else {
            std::random_device rd;
            base = (std::uint64_t(rd()) << 32) ^ rd();
        }
        constexpr size_t kBlock = size_t(1) << 16; // values per engine
        const size_t n = h.size(), nb = (n + kBlock - 1) / kBlock;
        T *p = h.data;
        detail::parallel_for(nb, 1, [&](size_t b0, size_t b1) {
            for (size_t b = b0; b < b1; ++b) {
                std::mt19937_64 g(detail::mix_seed(base ^ detail::mix_seed(b)));
                const size_t s = b * kBlock;
                fill(g, p + s, std::min(n, s + kBlock) - s);
            }
        }, 1);
        if constexpr (on_host)
            return h;
        else
            return tensorET(h);
    }

    size_t size_ = 0;
    shape_type dims_{};
    shape_type strides_{};
    std::array<long long, DIM> dlo_{}, dhi_{};
    bool custom_domain_ = false;
};

} // namespace AXOS
