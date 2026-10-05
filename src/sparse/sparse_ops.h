// SPDX-License-Identifier: BSD-3-Clause
//
// User-facing sparse operations. They check shapes and forward to the
// backend's Kernels<Backend>. Dense vectors/matrices are tensorET in the same
// storage as the Csr.
#pragma once

#include "sparse/csr.h"
#include "sparse/sparse_cpu.h"

namespace AXOS {
namespace Sparse {

namespace detail {
template <typename M, typename V>
inline void
check_same_backend()
{
    static_assert(std::is_same_v<typename M::backend_type,
                      typename V::backend_type>,
        "sparse matrix and dense tensor must use the same backend");
}
} // namespace detail

// y = alpha * A * x + beta * y
template <typename T, typename Idx, template <typename> class S, typename VS>
void
spmv(const Csr<T, Idx, S> &A, const tensorET<1, T, VS> &x,
    tensorET<1, T, VS> &y, T alpha = T(1), T beta = T(0))
{
    using M = Csr<T, Idx, S>;
    static_assert(std::is_same_v<typename M::backend_type,
                      typename VS::backend_type>,
        "sparse matrix and dense tensor must use the same backend");
    if (x.size() != A.cols() || y.size() != A.rows())
        throw std::invalid_argument("spmv: dimension mismatch");
    if (A.rows() == 0) return;
    Kernels<typename M::backend_type>::spmv(A, x.data, y.data, alpha, beta);
}

// y = alpha * A^T * x + beta * y (uses the cached explicit transpose).
template <typename T, typename Idx, template <typename> class S, typename VS>
void
spmv_t(const Csr<T, Idx, S> &A, const tensorET<1, T, VS> &x,
    tensorET<1, T, VS> &y, T alpha = T(1), T beta = T(0))
{
    if (x.size() != A.rows() || y.size() != A.cols())
        throw std::invalid_argument("spmv_t: dimension mismatch");
    if (A.cols() == 0) return;
    spmv(A.transposed(), x, y, alpha, beta);
}

// Returns A * x.
template <typename T, typename Idx, template <typename> class S, typename VS>
tensorET<1, T, VS>
spmv(const Csr<T, Idx, S> &A, const tensorET<1, T, VS> &x)
{
    tensorET<1, T, VS> y({A.rows()}, T(0));
    spmv(A, x, y);
    return y;
}

// Y = alpha * A * X + beta * Y for row-major dense X (cols x k), Y (rows x k).
template <typename T, typename Idx, template <typename> class S, typename VS>
void
spmm(const Csr<T, Idx, S> &A, const tensorET<2, T, VS> &X,
    tensorET<2, T, VS> &Y, T alpha = T(1), T beta = T(0))
{
    using M = Csr<T, Idx, S>;
    static_assert(std::is_same_v<typename M::backend_type,
                      typename VS::backend_type>,
        "sparse matrix and dense tensor must use the same backend");
    if (X.size(0) != A.cols() || Y.size(0) != A.rows() ||
        X.size(1) != Y.size(1))
        throw std::invalid_argument("spmm: dimension mismatch");
    if (A.rows() == 0 || X.size(1) == 0) return;
    Kernels<typename M::backend_type>::spmm(
        A, X.data, X.size(1), Y.data, alpha, beta);
}

// A_ij *= r[i] * c[j] (r or c may be null). r has rows() entries and c has
// cols(); both are real-valued and live on the matrix's device.
template <typename T, typename Idx, template <typename> class S>
void
scale_rows_cols(Csr<T, Idx, S> &A, const real_of_t<T> *r, const real_of_t<T> *c)
{
    using M = Csr<T, Idx, S>;
    Kernels<typename M::backend_type>::scale(A, r, c);
}

// Row-wise norms: out has rows() entries.
template <typename T, typename Idx, template <typename> class S>
void
row_norms(const Csr<T, Idx, S> &A, real_of_t<T> *out, Norm p)
{
    using M = Csr<T, Idx, S>;
    if (A.rows() == 0) return;
    Kernels<typename M::backend_type>::row_norms(A, out, p);
}

// Column-wise norms: out has cols() entries (through the cached transpose).
template <typename T, typename Idx, template <typename> class S>
void
col_norms(const Csr<T, Idx, S> &A, real_of_t<T> *out, Norm p)
{
    row_norms(A.transposed(), out, p);
}

// C = A * B
template <typename T, typename Idx, template <typename> class S>
Csr<T, Idx, S>
spgemm(const Csr<T, Idx, S> &A, const Csr<T, Idx, S> &B)
{
    using M = Csr<T, Idx, S>;
    if (A.cols() != B.rows())
        throw std::invalid_argument("spgemm: dimension mismatch");
    return Kernels<typename M::backend_type>::spgemm(A, B);
}

// A * diag(d) * A^T, the normal-equations matrix of interior-point methods.
// d has cols() entries, on the matrix's device.
template <typename T, typename Idx, template <typename> class S>
Csr<T, Idx, S>
AdAt(const Csr<T, Idx, S> &A, const real_of_t<T> *d)
{
    Csr<T, Idx, S> Ad(A);
    scale_rows_cols(Ad, nullptr, d);
    return spgemm(Ad, A.transposed());
}

} // namespace Sparse
} // namespace AXOS
