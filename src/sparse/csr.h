// SPDX-License-Identifier: BSD-3-Clause
//
// Csr<T, Idx, Store>: compressed sparse row matrix.
//
// Invariants (checked by validate()):
//   - 0-based indices; row_ptr has rows+1 entries, non-decreasing,
//     row_ptr[0] == 0 and row_ptr[rows] == nnz
//   - column indices are sorted (strictly increasing) within each row, so
//     there are no duplicate entries
//   - the sparsity pattern is fixed after construction; values may change
//
// The three arrays live in Store<Idx> / Store<T> (Cpu::HostStorage or
// Cuda::CudaStorage), so the matrix reuses the GPU pool and the storage
// semantics of tensorET. Kernels live in sparse_cpu.h / sparse_cuda.h and are
// selected by backend (Sparse::Kernels<Backend>).
#pragma once

#include "storage/cuda_storage.h"
#include "storage/host_storage.h"
#include "tensorET.h"
#include <algorithm>
#include <complex>
#include <cstdint>
#include <cstring>
#include <memory>
#include <numeric>
#include <sstream>
#include <stdexcept>
#include <string>
#include <type_traits>
#include <utility>
#include <vector>

#if defined(__CUDACC__) || defined(AXOS_ENABLE_CUDA)
#include <cuda_runtime.h>
#endif

namespace AXOS {
namespace Sparse {

template <typename T> struct is_complex_value : std::false_type {};
template <typename R>
struct is_complex_value<std::complex<R>> : std::true_type {};

template <typename T>
inline constexpr bool is_supported_value_v =
    std::is_same_v<T, float> || std::is_same_v<T, double> ||
    std::is_same_v<T, std::complex<float>> ||
    std::is_same_v<T, std::complex<double>>;

template <typename T> struct real_of { using type = T; };
template <typename R> struct real_of<std::complex<R>> { using type = R; };
template <typename T> using real_of_t = typename real_of<T>::type;

enum class Norm { L1, L2, Linf };

// Per-backend kernels (specialized in sparse_cpu.h / sparse_cuda.h).
template <typename Backend> struct Kernels;

namespace detail {

// Base class for backend-specific per-matrix state (cuSPARSE descriptors,
// workspaces). Owned by the matrix, dropped on copy.
struct DeviceCache {
    virtual ~DeviceCache() = default;
};

template <typename Store>
inline constexpr bool is_host_store_v =
    std::is_same_v<typename Store::backend_type, Cpu::Backend>;

template <typename U, template <typename> class S>
void
host_to_store(S<U> &dst, const U *src, size_t n)
{
    if (n == 0) return;
    if constexpr (is_host_store_v<S<U>>) {
        std::memcpy(dst.data(), src, n * sizeof(U));
    } else {
#if defined(__CUDACC__) || defined(AXOS_ENABLE_CUDA)
        cudaError_t e = cudaMemcpy(
            dst.data(), src, n * sizeof(U), cudaMemcpyHostToDevice);
        if (e != cudaSuccess)
            throw std::runtime_error(
                std::string("Csr upload failed: ") + cudaGetErrorString(e));
#else
        throw std::runtime_error("Csr: CUDA storage requires a CUDA build");
#endif
    }
}

template <typename U, template <typename> class S>
void
store_to_host(U *dst, const S<U> &src, size_t n)
{
    if (n == 0) return;
    if constexpr (is_host_store_v<S<U>>) {
        std::memcpy(dst, src.data(), n * sizeof(U));
    } else {
#if defined(__CUDACC__) || defined(AXOS_ENABLE_CUDA)
        cudaError_t e = cudaMemcpy(
            dst, src.data(), n * sizeof(U), cudaMemcpyDeviceToHost);
        if (e != cudaSuccess)
            throw std::runtime_error(
                std::string("Csr download failed: ") + cudaGetErrorString(e));
#else
        throw std::runtime_error("Csr: CUDA storage requires a CUDA build");
#endif
    }
}

template <typename U, template <typename> class D, template <typename> class S>
void
store_to_store(D<U> &dst, const S<U> &src, size_t n)
{
    if (n == 0) return;
    constexpr bool dh = is_host_store_v<D<U>>, sh = is_host_store_v<S<U>>;
    if constexpr (dh && sh) {
        std::memcpy(dst.data(), src.data(), n * sizeof(U));
    } else {
#if defined(__CUDACC__) || defined(AXOS_ENABLE_CUDA)
        cudaMemcpyKind kind = dh   ? cudaMemcpyDeviceToHost
                              : sh ? cudaMemcpyHostToDevice
                                   : cudaMemcpyDeviceToDevice;
        cudaError_t e = cudaMemcpy(dst.data(), src.data(), n * sizeof(U), kind);
        if (e != cudaSuccess)
            throw std::runtime_error(
                std::string("Csr transfer failed: ") + cudaGetErrorString(e));
#else
        throw std::runtime_error("Csr: CUDA storage requires a CUDA build");
#endif
    }
}

} // namespace detail

template <typename T, typename Idx = int32_t,
    template <typename> class Store = Cpu::HostStorage>
class Csr {
    static_assert(std::is_same_v<Idx, int32_t> || std::is_same_v<Idx, int64_t>,
        "Csr index type must be int32_t or int64_t");
    static_assert(is_supported_value_v<T>,
        "Csr value type must be float, double, or std::complex of those");

  public:
    using value_type = T;
    using index_type = Idx;
    using backend_type = typename Store<T>::backend_type;
    using vector_type = tensorET<1, T, Store<T>>;
    static constexpr bool on_host = detail::is_host_store_v<Store<T>>;

    Csr() = default;

    // Allocates arrays (uninitialized) for a rows x cols matrix with nnz
    // stored entries. row_ptr is zero-filled only when nnz == 0.
    Csr(size_t rows, size_t cols, size_t nnz) : rows_(rows), cols_(cols), nnz_(nnz)
    {
        row_ptr_ = Store<Idx>(rows + 1);
        col_ind_ = Store<Idx>(nnz);
        vals_ = Store<T>(nnz);
        if (nnz == 0) {
            std::vector<Idx> z(rows + 1, 0);
            detail::host_to_store(row_ptr_, z.data(), rows + 1);
        }
    }

    // Upload from host arrays (row_ptr: rows+1, col_ind/vals: nnz).
    Csr(size_t rows, size_t cols, const std::vector<Idx> &row_ptr,
        const std::vector<Idx> &col_ind, const std::vector<T> &vals)
        : Csr(rows, cols, vals.size())
    {
        if (row_ptr.size() != rows + 1 || col_ind.size() != vals.size())
            throw std::invalid_argument("Csr: inconsistent array sizes");
        detail::host_to_store(row_ptr_, row_ptr.data(), rows + 1);
        detail::host_to_store(col_ind_, col_ind.data(), nnz_);
        detail::host_to_store(vals_, vals.data(), nnz_);
    }

    // Copy between storages (host <-> device, device <-> device).
    template <template <typename> class S2>
    Csr(const Csr<T, Idx, S2> &o) : Csr(o.rows_, o.cols_, o.nnz_)
    {
        detail::store_to_store(row_ptr_, o.row_ptr_, rows_ + 1);
        detail::store_to_store(col_ind_, o.col_ind_, nnz_);
        detail::store_to_store(vals_, o.vals_, nnz_);
    }

    Csr(const Csr &o) : Csr(o.rows_, o.cols_, o.nnz_)
    {
        detail::store_to_store(row_ptr_, o.row_ptr_, rows_ + 1);
        detail::store_to_store(col_ind_, o.col_ind_, nnz_);
        detail::store_to_store(vals_, o.vals_, nnz_);
    }

    Csr &
    operator=(const Csr &o)
    {
        if (this != &o) {
            Csr tmp(o);
            *this = std::move(tmp);
        }
        return *this;
    }

    Csr(Csr &&o) noexcept
        : rows_(std::exchange(o.rows_, 0)), cols_(std::exchange(o.cols_, 0)),
          nnz_(std::exchange(o.nnz_, 0)), row_ptr_(std::move(o.row_ptr_)),
          col_ind_(std::move(o.col_ind_)), vals_(std::move(o.vals_)),
          t_cache_(std::move(o.t_cache_)), dev_(std::move(o.dev_))
    {
    }

    Csr &
    operator=(Csr &&o) noexcept
    {
        if (this != &o) {
            rows_ = std::exchange(o.rows_, 0);
            cols_ = std::exchange(o.cols_, 0);
            nnz_ = std::exchange(o.nnz_, 0);
            row_ptr_ = std::move(o.row_ptr_);
            col_ind_ = std::move(o.col_ind_);
            vals_ = std::move(o.vals_);
            t_cache_ = std::move(o.t_cache_);
            dev_ = std::move(o.dev_);
        }
        return *this;
    }

    size_t rows() const { return rows_; }
    size_t cols() const { return cols_; }
    size_t nnz() const { return nnz_; }

    const Idx *row_ptr() const { return row_ptr_.data(); }
    const Idx *col_ind() const { return col_ind_.data(); }
    const T *values() const { return vals_.data(); }

    // Pattern arrays for kernels that build a matrix in place. The caller
    // must keep the Csr invariants.
    Idx *row_ptr_mut() { values_changed(); return row_ptr_.data(); }
    Idx *col_ind_mut() { values_changed(); return col_ind_.data(); }

    // Mutable access to the stored values. Any call drops the cached
    // transpose, since it would be stale once the values are written.
    T *
    values_mut()
    {
        values_changed();
        return vals_.data();
    }

    // Non-owning dense 1-D view of the stored values (nnz elements), so
    // existing elementwise tensor code can scale/clamp/reduce them.
    vector_type
    values_view()
    {
        values_changed();
        size_t n[1] = {nnz_};
        return vector_type(vals_.data(), n, false);
    }

    // Call after writing through a pointer obtained earlier.
    void values_changed() const { t_cache_.reset(); }

    // Cached transpose (CSR of A^T). Built on first use, dropped whenever
    // the values are accessed mutably.
    const Csr &
    transposed() const
    {
        if (!t_cache_)
            t_cache_ = std::make_shared<Csr>(
                Kernels<backend_type>::transpose(*this));
        return *t_cache_;
    }

    // Uncached transpose.
    Csr
    transpose() const
    {
        return Kernels<backend_type>::transpose(*this);
    }

    // Deep copy of the matrix on the host.
    Csr<T, Idx, Cpu::HostStorage>
    to_host() const
    {
        return Csr<T, Idx, Cpu::HostStorage>(*this);
    }

    // Structural / ordering invariants. Returns an empty string when valid.
    std::string
    validate() const
    {
        if (row_ptr_.data() == nullptr)
            return (rows_ == 0 && nnz_ == 0) ? "" : "row_ptr is null";
        std::vector<Idx> rp(rows_ + 1), ci(nnz_);
        detail::store_to_host(rp.data(), row_ptr_, rows_ + 1);
        detail::store_to_host(ci.data(), col_ind_, nnz_);
        std::ostringstream err;
        if (rp[0] != 0) err << "row_ptr[0] != 0; ";
        if (static_cast<size_t>(rp[rows_]) != nnz_)
            err << "row_ptr[rows] != nnz; ";
        for (size_t i = 0; i < rows_; ++i) {
            if (rp[i + 1] < rp[i]) {
                err << "row_ptr decreases at row " << i << "; ";
                break;
            }
        }
        if (!err.str().empty()) return err.str();
        for (size_t i = 0; i < rows_; ++i) {
            for (Idx k = rp[i]; k < rp[i + 1]; ++k) {
                if (ci[k] < 0 || static_cast<size_t>(ci[k]) >= cols_) {
                    err << "column index out of range in row " << i;
                    return err.str();
                }
                if (k > rp[i] && ci[k] <= ci[k - 1]) {
                    err << "columns not strictly increasing in row " << i;
                    return err.str();
                }
            }
        }
        return "";
    }

    // Dense copy, in the requested tensorET storage (host by default).
    template <typename S = Cpu::HostStorage<T>>
    tensorET<2, T, S>
    to_dense() const
    {
        Csr<T, Idx, Cpu::HostStorage> h(*this);
        tensorET<2, T> D({rows_, cols_}, T(0));
        for (size_t i = 0; i < rows_; ++i)
            for (Idx k = h.row_ptr()[i]; k < h.row_ptr()[i + 1]; ++k)
                D.data[i * cols_ + h.col_ind()[k]] = h.values()[k];
        if constexpr (std::is_same_v<S, Cpu::HostStorage<T>>) {
            return D;
        } else {
            return tensorET<2, T, S>(D);
        }
    }

    // Dense -> CSR (through the host). Entries with |v| <= drop_tol are
    // dropped. Result is on the host; convert with Csr<..., CudaStorage>(h).
    template <typename S>
    static Csr<T, Idx, Cpu::HostStorage>
    from_dense(const tensorET<2, T, S> &A, real_of_t<T> drop_tol = 0);

    // Stack B's rows below this matrix's rows (same column count). Used to
    // add cuts / constraints. Returns a new matrix (rebuilds the pattern).
    Csr
    append_rows(const Csr &B) const
    {
        if (B.cols_ != cols_)
            throw std::invalid_argument("append_rows: column count mismatch");
        Csr<T, Idx, Cpu::HostStorage> a(*this), b(B);
        std::vector<Idx> rp(a.rows() + b.rows() + 1);
        std::vector<Idx> ci;
        std::vector<T> v;
        ci.reserve(a.nnz() + b.nnz());
        v.reserve(a.nnz() + b.nnz());
        std::copy(a.row_ptr(), a.row_ptr() + a.rows() + 1, rp.begin());
        ci.insert(ci.end(), a.col_ind(), a.col_ind() + a.nnz());
        v.insert(v.end(), a.values(), a.values() + a.nnz());
        for (size_t i = 0; i < b.rows(); ++i)
            rp[a.rows() + 1 + i] = static_cast<Idx>(a.nnz()) + b.row_ptr()[i + 1];
        ci.insert(ci.end(), b.col_ind(), b.col_ind() + b.nnz());
        v.insert(v.end(), b.values(), b.values() + b.nnz());
        return Csr(a.rows() + b.rows(), cols_, rp, ci, v);
    }

    // Backend-specific per-matrix cache (descriptors, workspaces).
    detail::DeviceCache *device_cache() const { return dev_.get(); }
    void
    set_device_cache(std::shared_ptr<detail::DeviceCache> c) const
    { dev_ = std::move(c); }

  private:
    template <typename, typename, template <typename> class> friend class Csr;

    size_t rows_ = 0, cols_ = 0, nnz_ = 0;
    Store<Idx> row_ptr_;
    Store<Idx> col_ind_;
    Store<T> vals_;
    mutable std::shared_ptr<Csr> t_cache_;
    mutable std::shared_ptr<detail::DeviceCache> dev_;
};

template <typename T, typename Idx = int32_t>
using CsrCuda = Csr<T, Idx, Cuda::CudaStorage>;

// Triplet (COO) builder. Sorts entries, sums duplicates, optionally drops
// zeros, and produces a host CSR satisfying the Csr invariants.
template <typename T, typename Idx = int32_t> class CooBuilder {
  public:
    CooBuilder(size_t rows, size_t cols) : rows_(rows), cols_(cols) {}

    void
    reserve(size_t n)
    {
        i_.reserve(n);
        j_.reserve(n);
        v_.reserve(n);
    }

    void
    add(size_t i, size_t j, T v)
    {
        if (i >= rows_ || j >= cols_)
            throw std::out_of_range("CooBuilder::add: index out of range");
        i_.push_back(static_cast<Idx>(i));
        j_.push_back(static_cast<Idx>(j));
        v_.push_back(v);
    }

    size_t size() const { return v_.size(); }

    Csr<T, Idx, Cpu::HostStorage>
    build(bool drop_zeros = false) const
    {
        const size_t n = v_.size();
        // Counting sort by row, then sort each row by column.
        std::vector<size_t> start(rows_ + 1, 0);
        for (size_t k = 0; k < n; ++k)
            start[i_[k] + 1]++;
        for (size_t r = 0; r < rows_; ++r)
            start[r + 1] += start[r];
        std::vector<size_t> perm(n), fill(start.begin(), start.end() - 1);
        for (size_t k = 0; k < n; ++k)
            perm[fill[i_[k]]++] = k;

        std::vector<Idx> rp(rows_ + 1, 0), ci;
        std::vector<T> vals;
        ci.reserve(n);
        vals.reserve(n);
        for (size_t r = 0; r < rows_; ++r) {
            auto first = perm.begin() + start[r];
            auto last = perm.begin() + start[r + 1];
            std::stable_sort(first, last,
                [&](size_t a, size_t b) { return j_[a] < j_[b]; });
            size_t row_begin = ci.size();
            for (auto it = first; it != last; ++it) {
                if (ci.size() > row_begin && ci.back() == j_[*it]) {
                    vals.back() += v_[*it];
                } else {
                    ci.push_back(j_[*it]);
                    vals.push_back(v_[*it]);
                }
            }
            if (drop_zeros) {
                size_t w = row_begin;
                for (size_t k = row_begin; k < ci.size(); ++k) {
                    if (vals[k] != T(0)) {
                        ci[w] = ci[k];
                        vals[w] = vals[k];
                        ++w;
                    }
                }
                ci.resize(w);
                vals.resize(w);
            }
            rp[r + 1] = static_cast<Idx>(ci.size());
        }
        return Csr<T, Idx, Cpu::HostStorage>(rows_, cols_, rp, ci, vals);
    }

  private:
    size_t rows_, cols_;
    std::vector<Idx> i_, j_;
    std::vector<T> v_;
};

template <typename T, typename Idx, template <typename> class Store>
template <typename S>
Csr<T, Idx, Cpu::HostStorage>
Csr<T, Idx, Store>::from_dense(const tensorET<2, T, S> &A, real_of_t<T> drop_tol)
{
    tensorET<2, T> H(A); // host copy (no-op copy for host storage)
    const size_t m = H.size(0), n = H.size(1);
    CooBuilder<T, Idx> b(m, n);
    for (size_t i = 0; i < m; ++i)
        for (size_t j = 0; j < n; ++j) {
            T v = H.data[i * n + j];
            if (std::abs(v) > drop_tol) b.add(i, j, v);
        }
    return b.build();
}

} // namespace Sparse
} // namespace AXOS
