// SPDX-License-Identifier: BSD-3-Clause
//
// Dense linear algebra on host tensors (docs/TENSOR_SPEC.md §7, §15.4-15.5).
// Matrices are tensorET<2, T> and vectors tensorET<1, T>, T = float or
// double. Errors (singular pivot, shape mismatch) throw std::runtime_error.
//
//   norm(A, B, p) / norm(A, B, "fro")      norms of A - B over all elements
//   conjugateGradient(A, b, x0, maxIter, tol)
//   gaussJordanElimination(A)              reduced row-echelon form
//   augmentMatrix(A, b)                    [A | b]
//   gramSchmidtOrthogonalization(V)        orthonormal basis of V's columns
//   luDcmp(A, tol)                         {L, U}, A = L U (no pivoting)
//   luDcmpPivoted(A, tol)                  {{L, U}, P}, P A = L U
//   luSolve(L, U, P, B)                    solves A X = B (B vector or matrix)
//   inverse(A)                             general inverse (LU + solves)
//   qrDecompositionTile(A)                 A <- Q (m x n), returns R (n x n)
//   lanczos(A, m, q0)                      {alpha, beta, Q}
//   expm(A)                                matrix exponential
//   lobpcg(A, nev, X0, eigvals, X, ...)    smallest eigenpairs (symmetric A)
//   symmetricEigen(A, w, V)                all eigenpairs (Jacobi, small A)
//   inverse_backs(U, m)                    inverse of an upper-triangular U
//   revEl / elimStep / matMul(history, ..) tournament elimination with a
//                                          recorded, replayable history
//
// How the speed is obtained (§15): LU is blocked and right-looking (panel on
// a column-major copy, then a GEMM trailing update that holds > 90% of the
// flops); triangular solves with several right-hand sides are blocked with
// GEMM updates, single right-hand sides with GEMV updates; QR is blocked
// Householder in compact-WY form (GEMM updates); CG preallocates its vectors
// and fuses A p with p^T A p and the three vector updates with r^T r; the
// triangular inverse inverts diagonal blocks in parallel and joins them
// pairwise with GEMMs; history replay is one parallel pass over independent
// columns (row operations) or rows (column operations).
#pragma once

#include "tensorMath.h"
#include <algorithm>
#include <cmath>
#include <cstdio>
#include <limits>
#include <numeric>
#include <random>
#include <string>
#include <tuple>
#include <utility>
#include <variant>
#include <vector>

#ifdef _OPENMP
#include <omp.h>
#endif

namespace AXOS {

namespace la_detail {

template <class T>
inline constexpr bool is_real_v =
    std::is_same_v<T, float> || std::is_same_v<T, double>;

template <class T>
using Mat = tensorET<2, T>;
template <class T>
using Vec = tensorET<1, T>;

[[noreturn]] inline void
fail(const std::string &what)
{
    throw std::runtime_error(what);
}

template <class Tn>
void
require_square(const Tn &A, const char *who)
{
    if (A.size(0) != A.size(1))
        fail(std::string(who) + ": matrix must be square (" +
             std::to_string(A.size(0)) + " x " + std::to_string(A.size(1)) + ")");
}

// Contiguous host copy (a no-op-cost view check: views are copied).
template <int D, class T, class S>
tensorET<D, T>
dense(const tensorET<D, T, S> &a)
{
    if constexpr (std::is_same_v<S, Cpu::HostStorage<T>>)
        return a.clone();
    else
        return tensorET<D, T>(a);
}

template <class T>
Mat<T>
mat(size_t m, size_t n)
{
    return Mat<T>(std::array<size_t, 2>{m, n});
}

template <class T>
Mat<T>
mat(size_t m, size_t n, T x)
{
    return Mat<T>(std::array<size_t, 2>{m, n}, x);
}

template <class T>
Mat<T>
identity(size_t n)
{
    Mat<T> I = mat<T>(n, n, T(0));
    for (size_t i = 0; i < n; ++i) I.data[i * n + i] = T(1);
    return I;
}

// y += a x, serial (for use inside parallel code).
template <class T>
AXOS_INLINE void
axpy_s(size_t n, T a, const T *AXOS_RESTRICT x, T *AXOS_RESTRICT y)
{
    using P = simd::Pack<T>;
    constexpr size_t W = P::W;
    const auto va = P::set1(a);
    size_t i = 0;
    for (; i + W <= n; i += W)
        P::storeu(y + i, P::fmadd(va, P::loadu(x + i), P::loadu(y + i)));
    for (; i < n; ++i) y[i] += a * x[i];
}

// sum x[i] y[i], serial.
template <class T>
AXOS_INLINE T
dot_s(size_t n, const T *x, const T *y)
{
    using P = simd::Pack<T>;
    constexpr size_t W = P::W;
    auto c0 = P::zero(), c1 = P::zero();
    size_t i = 0;
    for (; i + 2 * W <= n; i += 2 * W) {
        c0 = P::fmadd(P::loadu(x + i), P::loadu(y + i), c0);
        c1 = P::fmadd(P::loadu(x + i + W), P::loadu(y + i + W), c1);
    }
    T s = P::hsum(P::add(c0, c1));
    for (; i < n; ++i) s += x[i] * y[i];
    return s;
}

// ---- triangular solves --------------------------------------------------

// Solves M X = B in place (B is n x k, row-major, ldb), M lower or upper
// triangular (n x n, ldm), with unit or general diagonal. Blocked: each
// diagonal block is solved directly, the rest of B is updated with one GEMM
// (GEMV for k == 1).
template <class T>
void
trsm_left(bool lower, bool unit, size_t n, size_t k, const T *M, size_t ldm,
    T *B, size_t ldb)
{
    if (n == 0 || k == 0) return;
    constexpr size_t nb = 64;
    auto solve_block = [&](size_t i0, size_t i1) {
        if (k == 1) {
            if (lower) {
                for (size_t i = i0; i < i1; ++i) {
                    const T *mi = M + i * ldm;
                    T s = B[i * ldb] - dot_s(i - i0, mi + i0, B + i0 * ldb);
                    B[i * ldb] = unit ? s : s / mi[i];
                }
            } else {
                for (size_t i = i1; i-- > i0;) {
                    const T *mi = M + i * ldm;
                    T s = B[i * ldb] - dot_s(i1 - i - 1, mi + i + 1, B + (i + 1) * ldb);
                    B[i * ldb] = unit ? s : s / mi[i];
                }
            }
            return;
        }
        // about 64K multiply-adds per chunk: (i1 - i0)^2 / 2 per column
        const size_t per_col = (i1 - i0) * (i1 - i0) / 2 + 1;
        detail::parallel_for(k, std::max<size_t>(16, 65536 / per_col),
            [&](size_t c0, size_t c1) {
                const size_t w = c1 - c0;
                if (lower) {
                    for (size_t i = i0; i < i1; ++i) {
                        T *bi = B + i * ldb + c0;
                        const T *mi = M + i * ldm;
                        for (size_t j = i0; j < i; ++j)
                            if (mi[j] != T(0)) axpy_s(w, -mi[j], B + j * ldb + c0, bi);
                        if (!unit) {
                            const T inv = T(1) / mi[i];
                            for (size_t c = 0; c < w; ++c) bi[c] *= inv;
                        }
                    }
                } else {
                    for (size_t i = i1; i-- > i0;) {
                        T *bi = B + i * ldb + c0;
                        const T *mi = M + i * ldm;
                        for (size_t j = i + 1; j < i1; ++j)
                            if (mi[j] != T(0)) axpy_s(w, -mi[j], B + j * ldb + c0, bi);
                        if (!unit) {
                            const T inv = T(1) / mi[i];
                            for (size_t c = 0; c < w; ++c) bi[c] *= inv;
                        }
                    }
                }
            }, 16);
    };
    auto update = [&](size_t r0, size_t r1, size_t i0, size_t i1) {
        // B[r0:r1] -= M[r0:r1, i0:i1] B[i0:i1]
        if (r1 <= r0) return;
        if (k == 1)
            kernels::gemv<T>(false, r1 - r0, i1 - i0, T(-1), M + r0 * ldm + i0,
                ldm, B + i0 * ldb, T(1), B + r0 * ldb);
        else
            kernels::gemm<T>(false, false, r1 - r0, k, i1 - i0, T(-1),
                M + r0 * ldm + i0, ldm, B + i0 * ldb, ldb, T(1), B + r0 * ldb,
                ldb);
    };
    if (k == 1 && ldb != 1) { // gather to a contiguous vector for GEMV
        std::vector<T> v(n);
        for (size_t i = 0; i < n; ++i) v[i] = B[i * ldb];
        trsm_left(lower, unit, n, 1, M, ldm, v.data(), 1);
        for (size_t i = 0; i < n; ++i) B[i * ldb] = v[i];
        return;
    }
    if (k == 1) {
        // One right-hand side: left-looking, so the GEMV for each block of
        // rows reads long contiguous row segments (x[i0:i1] -= M[i0:i1, 0:i0]
        // x[0:i0] for lower, M[i0:i1, i1:n] x[i1:n] for upper), which streams
        // at memory bandwidth; the right-looking form reads narrow strips.
        constexpr size_t rb = 64;
        const size_t nblk = (n + rb - 1) / rb;
        for (size_t s = 0; s < nblk; ++s) {
            const size_t bk = lower ? s : nblk - 1 - s;
            const size_t i0 = bk * rb, i1 = std::min(n, i0 + rb);
            if (lower && i0 > 0)
                kernels::gemv<T>(false, i1 - i0, i0, T(-1), M + i0 * ldm, ldm, B,
                    T(1), B + i0);
            if (!lower && i1 < n)
                kernels::gemv<T>(false, i1 - i0, n - i1, T(-1), M + i0 * ldm + i1,
                    ldm, B + i1, T(1), B + i0);
            solve_block(i0, i1);
        }
        return;
    }
    if (lower) {
        for (size_t i0 = 0; i0 < n; i0 += nb) {
            const size_t i1 = std::min(n, i0 + nb);
            solve_block(i0, i1);
            update(i1, n, i0, i1);
        }
    } else {
        const size_t nblk = (n + nb - 1) / nb;
        for (size_t bk = nblk; bk-- > 0;) {
            const size_t i0 = bk * nb, i1 = std::min(n, i0 + nb);
            solve_block(i0, i1);
            update(0, i0, i0, i1);
        }
    }
}

// ---- LU -----------------------------------------------------------------

// Applies the row interchanges piv[j0..j1) (row j <-> row piv[j], in order)
// to columns [c0, c1) of the column-major panel P (leading dimension rows),
// column by column so every access stays inside one contiguous column.
template <class T>
void
panel_swap(T *P, size_t rows, size_t c0, size_t c1, const size_t *piv,
    size_t j0, size_t j1)
{
    for (size_t c = c0; c < c1; ++c) {
        T *col = P + c * rows;
        for (size_t j = j0; j < j1; ++j)
            if (piv[j] != j) std::swap(col[j], col[piv[j]]);
    }
}

// Unblocked LU (partial pivoting when `pivot`) of columns [c0, c0 + w) of a
// column-major panel P (leading dimension rows), using rows [c0, rows).
// Row swaps touch only these columns; piv[j] is the panel row swapped with
// row j. `col0` is the global column of the panel's first column.
template <class T>
void
panel_lu_cols(T *P, size_t rows, size_t c0, size_t w, size_t *piv, bool pivot,
    double tol, size_t col0, const char *who)
{
    for (size_t j = c0; j < c0 + w && j < rows; ++j) {
        T *col = P + j * rows;
        size_t p = j;
        if (pivot) {
            T best = std::abs(col[j]);
            for (size_t r = j + 1; r < rows; ++r) {
                const T v = std::abs(col[r]);
                if (v > best) {
                    best = v;
                    p = r;
                }
            }
        }
        piv[j] = p;
        if (p != j)
            for (size_t c = c0; c < c0 + w; ++c)
                std::swap(P[c * rows + j], P[c * rows + p]);
        const T d = col[j];
        if (!(std::abs(double(d)) > tol))
            fail(std::string(who) + ": matrix is singular to working precision "
                 "(pivot " + std::to_string(double(d)) + " in column " +
                 std::to_string(col0 + j) + ")");
        const T inv = T(1) / d;
        for (size_t r = j + 1; r < rows; ++r) col[r] *= inv;
        for (size_t c = j + 1; c < c0 + w; ++c) {
            T *cc = P + c * rows;
            const T f = cc[j];
            if (f != T(0)) axpy_s(rows - j - 1, -f, col + j + 1, cc + j + 1);
        }
    }
}

// Recursive LU of columns [c0, c0 + w) of the column-major panel (LAPACK
// dgetrf2): factor the left half, apply its interchanges to the right half,
// solve for the top of the right half, update the rest of it with one GEMM,
// factor it, and apply its interchanges back to the left half. Most of the
// panel's work is GEMM, and every swap runs down a contiguous column.
template <class T>
void
panel_lu_rec(T *P, size_t rows, size_t c0, size_t w, size_t *piv, bool pivot,
    double tol, size_t col0, const char *who)
{
    if (w <= 8 || c0 + w > rows) {
        panel_lu_cols(P, rows, c0, w, piv, pivot, tol, col0, who);
        return;
    }
    const size_t w1 = w / 2, w2 = w - w1, cm = c0 + w1, c1 = c0 + w;
    panel_lu_rec(P, rows, c0, w1, piv, pivot, tol, col0, who);
    if (pivot) panel_swap(P, rows, cm, c1, piv, c0, cm);
    // A12 = L11^{-1} A12 (column-major, unit lower): rows [c0, cm), cols [cm, c1)
    for (size_t c = cm; c < c1; ++c) {
        T *x = P + c * rows;
        for (size_t j = c0; j < cm; ++j)
            if (x[j] != T(0)) axpy_s(cm - j - 1, -x[j], P + j * rows + j + 1, x + j + 1);
    }
    // A22 -= A21 A12, rows [cm, rows): column-major C = C - A B is the
    // row-major C^T = C^T - B^T A^T on the same memory.
    kernels::gemm<T>(false, false, w2, rows - cm, w1, T(-1), P + cm * rows + c0,
        rows, P + c0 * rows + cm, rows, T(1), P + cm * rows + cm, rows);
    panel_lu_rec(P, rows, cm, w2, piv, pivot, tol, col0, who);
    if (pivot) panel_swap(P, rows, c0, cm, piv, cm, c1);
}

// Blocked right-looking LU of the n x n row-major matrix A (in place): unit
// lower L below the diagonal, U on and above it. With perm != nullptr rows
// are pivoted and perm[i] is the original row that ends up in row i. Each
// panel is transposed into a column-major buffer (tiled SIMD transpose),
// factored recursively and transposed back; its row interchanges are applied
// to the rest of the matrix in parallel over column blocks; U12 is a blocked
// triangular solve and the trailing update one GEMM.
template <class T>
void
lu_inplace(T *A, size_t n, size_t lda, std::vector<size_t> *perm, double tol,
    const char *who)
{
    if (perm) {
        perm->resize(n);
        std::iota(perm->begin(), perm->end(), size_t(0));
    }
    if (n == 0) return;
    const size_t nb = n >= 1024 ? 128 : (n >= 256 ? 64 : 32);
    std::vector<T> panel;
    std::vector<size_t> piv(nb);
    for (size_t k0 = 0; k0 < n; k0 += nb) {
        const size_t b = std::min(nb, n - k0), rows = n - k0;
        panel.resize(rows * b);
        T *akk = A + k0 * lda + k0;
        std::iota(piv.begin(), piv.begin() + b, size_t(0));
        kernels::transpose<T>(rows, b, akk, lda, panel.data(), rows);
        {
            // The panel's GEMMs are small and tall; on one thread they run
            // faster than with fork/join per call (measured).
            detail::SerialScope serial;
            panel_lu_rec(panel.data(), rows, 0, b, piv.data(),
                perm != nullptr, tol, k0, who);
        }
        kernels::transpose<T>(b, rows, panel.data(), rows, akk, lda);
        if (perm) {
            for (size_t j = 0; j < b; ++j)
                if (piv[j] != j) std::swap((*perm)[k0 + j], (*perm)[k0 + piv[j]]);
            // The same interchanges on the columns outside the panel,
            // [0, k0) and [k0 + b, n), split into column blocks.
            detail::parallel_for(n - b, 512, [&](size_t q0, size_t q1) {
                auto swap_cols = [&](size_t c0, size_t c1) {
                    for (size_t j = 0; j < b; ++j) {
                        if (piv[j] == j) continue;
                        T *ra = A + (k0 + j) * lda, *rb = A + (k0 + piv[j]) * lda;
                        std::swap_ranges(ra + c0, ra + c1, rb + c0);
                    }
                };
                if (q0 < k0) swap_cols(q0, std::min(q1, k0));
                if (q1 > k0) swap_cols(std::max(q0, k0) + b, q1 + b);
            }, 64);
        }
        if (k0 + b >= n) break;
        const size_t nr = n - k0 - b;
        trsm_left(true, true, b, nr, akk, lda, akk + b, lda); // U12 = L11^{-1} A12
        kernels::gemm<T>(false, false, nr, nr, b, T(-1), akk + b * lda, lda,
            akk + b, lda, T(1), akk + b * lda + b, lda); // A22 -= A21 U12
    }
}

// Solves A X = B given the in-place LU of A (lu_inplace) and its row
// permutation; B is n x k (row-major), overwritten with X.
template <class T>
void
lu_solve_inplace(const T *LU, size_t n, const std::vector<size_t> &perm, T *B,
    size_t k)
{
    if (!perm.empty()) {
        std::vector<T> tmp(n * k);
        for (size_t i = 0; i < n; ++i)
            std::copy(B + perm[i] * k, B + perm[i] * k + k, tmp.data() + i * k);
        std::copy(tmp.begin(), tmp.end(), B);
    }
    trsm_left(true, true, n, k, LU, n, B, k);
    trsm_left(false, false, n, k, LU, n, B, k);
}

// The permutations of this thread's recent luDcmpPivoted calls, keyed by the
// buffer of the returned P. luSolve takes P as a dense n x n matrix; reading
// all of it would cost more than the triangular solves. A remembered
// permutation is used only after checking P(i, perm[i]) == 1 for every row,
// which pins down a permutation matrix completely (O(n) reads).
template <class T> struct PermCache {
    struct Entry {
        const T *key = nullptr;
        std::vector<size_t> perm;
    };
    Entry e[4];
    size_t next = 0;
};

template <class T>
PermCache<T> &
perm_cache()
{
    thread_local PermCache<T> c;
    return c;
}

template <class T>
void
remember_perm(const T *key, const std::vector<size_t> &perm)
{
    PermCache<T> &c = perm_cache<T>();
    c.e[c.next].key = key;
    c.e[c.next].perm = perm;
    c.next = (c.next + 1) % 4;
}

// Row permutation encoded by a permutation matrix: perm[i] = j with
// P(i, j) == 1.
template <class T, class S>
std::vector<size_t>
perm_of(const tensorET<2, T, S> &P)
{
    const size_t n = P.size(0), ld = P.stride(0);
    if (P.size(1) != n) fail("luSolve: P must be square");
    for (const auto &en : perm_cache<T>().e) {
        if (en.key != P.data || en.perm.size() != n) continue;
        bool ok = true;
        for (size_t i = 0; i < n && ok; ++i) ok = P.data[i * ld + en.perm[i]] == T(1);
        if (ok) return en.perm;
    }
    // Otherwise: the non-zero of every row, in parallel over rows.
    std::vector<size_t> perm(n);
    detail::parallel_for(n, 64, [&](size_t r0, size_t r1) {
        for (size_t i = r0; i < r1; ++i) {
            const T *row = P.data + i * ld;
            size_t j = 0;
            while (j + 4 <= n && row[j] == T(0) && row[j + 1] == T(0) &&
                   row[j + 2] == T(0) && row[j + 3] == T(0))
                j += 4;
            while (j < n && row[j] == T(0)) ++j;
            perm[i] = j;
        }
    }, 1);
    for (size_t i = 0; i < n; ++i)
        if (perm[i] >= n)
            fail("luSolve: P is not a permutation matrix (row " + std::to_string(i) +
                 " is zero)");
    return perm;
}

// Splits the in-place LU (n x n, contiguous) into a new unit-lower L and
// leaves U in LU (strictly lower part zeroed).
template <class T>
Mat<T>
split_lu(Mat<T> &LU)
{
    const size_t n = LU.size(0);
    Mat<T> L = mat<T>(n, n);
    detail::parallel_for(n, 64, [&](size_t r0, size_t r1) {
        for (size_t i = r0; i < r1; ++i) {
            T *l = L.data + i * n, *u = LU.data + i * n;
            std::copy(u, u + i, l);
            l[i] = T(1);
            std::fill(l + i + 1, l + n, T(0));
            std::fill(u, u + i, T(0));
        }
    }, 1);
    return L;
}

// ---- QR -----------------------------------------------------------------

// Householder reflector (LAPACK dlarfg): given x = [alpha; tail] (length L,
// contiguous), overwrite alpha with beta and tail with v(1:), return tau so
// that (I - tau v v^T) x = [beta; 0], v = [1; tail]. Columns with norm below
// 1e-12 are left as they are (tau = 0).
template <class T>
T
householder(T *x, size_t L)
{
    if (L <= 1) return T(0);
    const T alpha = x[0];
    T xnorm = std::sqrt(dot_s(L - 1, x + 1, x + 1));
    const T full = std::hypot(alpha, xnorm);
    if (xnorm == T(0) || full < T(1e-12)) return T(0);
    const T beta = alpha >= T(0) ? -full : full;
    const T tau = (beta - alpha) / beta;
    const T scal = T(1) / (alpha - beta);
    for (size_t i = 1; i < L; ++i) x[i] *= scal;
    x[0] = beta;
    return tau;
}

template <class T> struct QrPanel {
    size_t k0 = 0, b = 0, rows = 0;
    std::vector<T> Vt; // b x rows, row-major: reflector i is row i (unit, zeros before)
    std::vector<T> Tm; // b x b upper triangular (compact WY)
};

// Blocked Householder QR of the m x n row-major matrix A (m >= n), in place:
// R on and above the diagonal; the reflectors are returned per panel.
template <class T>
std::vector<QrPanel<T>>
qr_inplace(T *A, size_t m, size_t n, size_t lda)
{
    const size_t nb = n >= 256 ? 48 : 32;
    std::vector<QrPanel<T>> panels;
    std::vector<T> pn, W, W2;
    for (size_t k0 = 0; k0 < n; k0 += nb) {
        const size_t b = std::min(nb, n - k0), rows = m - k0;
        pn.resize(rows * b);
        for (size_t r = 0; r < rows; ++r)
            for (size_t c = 0; c < b; ++c)
                pn[c * rows + r] = A[(k0 + r) * lda + k0 + c];
        std::vector<T> tau(b);
        for (size_t j = 0; j < b; ++j) {
            T *x = pn.data() + j * rows + j;
            const size_t L = rows - j;
            tau[j] = householder(x, L);
            if (tau[j] == T(0)) continue;
            for (size_t c = j + 1; c < b; ++c) {
                T *y = pn.data() + c * rows + j;
                const T w = tau[j] * (y[0] + dot_s(L - 1, x + 1, y + 1));
                y[0] -= w;
                axpy_s(L - 1, -w, x + 1, y + 1);
            }
        }
        for (size_t r = 0; r < rows; ++r)
            for (size_t c = 0; c < b; ++c)
                A[(k0 + r) * lda + k0 + c] = pn[c * rows + r];
        QrPanel<T> P;
        P.k0 = k0;
        P.b = b;
        P.rows = rows;
        P.Vt.assign(b * rows, T(0));
        for (size_t i = 0; i < b; ++i) {
            T *v = P.Vt.data() + i * rows;
            v[i] = T(1);
            for (size_t r = i + 1; r < rows; ++r) v[r] = pn[i * rows + r];
        }
        // T (dlarft, forward columnwise): T[0:j, j] = -tau_j T[0:j,0:j] V[:,0:j]^T v_j
        P.Tm.assign(b * b, T(0));
        std::vector<T> z(b);
        for (size_t j = 0; j < b; ++j) {
            P.Tm[j * b + j] = tau[j];
            if (j == 0 || tau[j] == T(0)) continue;
            const T *vj = P.Vt.data() + j * rows;
            for (size_t i = 0; i < j; ++i)
                z[i] = -tau[j] * dot_s(rows - j, P.Vt.data() + i * rows + j, vj + j);
            for (size_t i = 0; i < j; ++i) {
                T s(0);
                for (size_t l = i; l < j; ++l) s += P.Tm[i * b + l] * z[l];
                P.Tm[i * b + j] = s;
            }
        }
        // Trailing update: C = (I - V T^T V^T) C, C = A[k0:m, k0+b:n]
        const size_t nt = n - k0 - b;
        if (nt) {
            T *C = A + k0 * lda + k0 + b;
            W.resize(b * nt);
            W2.resize(b * nt);
            kernels::gemm<T>(false, false, b, nt, rows, T(1), P.Vt.data(), rows,
                C, lda, T(0), W.data(), nt);
            kernels::gemm<T>(true, false, b, nt, b, T(1), P.Tm.data(), b,
                W.data(), nt, T(0), W2.data(), nt);
            kernels::gemm<T>(true, false, rows, nt, b, T(-1), P.Vt.data(), rows,
                W2.data(), nt, T(1), C, lda);
        }
        panels.push_back(std::move(P));
    }
    return panels;
}

// Thin Q (m x n) from the panels: Q = H_1 ... H_n [I; 0], accumulated
// backwards with the block reflectors.
template <class T>
Mat<T>
qr_form_q(const std::vector<QrPanel<T>> &panels, size_t m, size_t n)
{
    Mat<T> Q = mat<T>(m, n, T(0));
    for (size_t i = 0; i < n; ++i) Q.data[i * n + i] = T(1);
    std::vector<T> W, W2;
    for (size_t p = panels.size(); p-- > 0;) {
        const QrPanel<T> &P = panels[p];
        const size_t nc = n - P.k0;
        T *C = Q.data + P.k0 * n + P.k0;
        W.resize(P.b * nc);
        W2.resize(P.b * nc);
        kernels::gemm<T>(false, false, P.b, nc, P.rows, T(1), P.Vt.data(),
            P.rows, C, n, T(0), W.data(), nc);
        kernels::gemm<T>(false, false, P.b, nc, P.b, T(1), P.Tm.data(), P.b,
            W.data(), nc, T(0), W2.data(), nc);
        kernels::gemm<T>(true, false, P.rows, nc, P.b, T(-1), P.Vt.data(),
            P.rows, W2.data(), nc, T(1), C, n);
    }
    return Q;
}

// Thin QR of a contiguous m x n matrix: returns {Q, R} with R's diagonal
// non-negative (the Gram-Schmidt convention).
template <class T>
std::pair<Mat<T>, Mat<T>>
thin_qr(Mat<T> A)
{
    const size_t m = A.size(0), n = A.size(1);
    if (m < n) fail("qrDecompositionTile: needs rows >= columns");
    auto panels = qr_inplace(A.data, m, n, n);
    Mat<T> R = mat<T>(n, n, T(0));
    for (size_t i = 0; i < n; ++i)
        std::copy(A.data + i * n + i, A.data + i * n + n, R.data + i * n + i);
    Mat<T> Q = qr_form_q(panels, m, n);
    for (size_t j = 0; j < n; ++j) {
        if (R.data[j * n + j] >= T(0)) continue;
        for (size_t c = j; c < n; ++c) R.data[j * n + c] = -R.data[j * n + c];
        for (size_t r = 0; r < m; ++r) Q.data[r * n + j] = -Q.data[r * n + j];
    }
    return {std::move(Q), std::move(R)};
}

// ---- symmetric eigenproblem (cyclic Jacobi) -----------------------------

// All eigenpairs of the symmetric n x n row-major matrix a (destroyed):
// w ascending, V's columns the eigenvectors. For small matrices.
template <class T>
void
jacobi_eigen(std::vector<T> &a, size_t n, std::vector<T> &w, std::vector<T> &V)
{
    V.assign(n * n, T(0));
    for (size_t i = 0; i < n; ++i) V[i * n + i] = T(1);
    T fro = 0;
    for (T v : a) fro += v * v;
    const T eps = std::numeric_limits<T>::epsilon();
    for (int sweep = 0; sweep < 64; ++sweep) {
        T off = 0;
        for (size_t p = 0; p < n; ++p)
            for (size_t q = p + 1; q < n; ++q) off += a[p * n + q] * a[p * n + q];
        if (off <= eps * eps * fro * T(0.25) || off == T(0)) break;
        for (size_t p = 0; p < n; ++p)
            for (size_t q = p + 1; q < n; ++q) {
                const T apq = a[p * n + q];
                if (std::abs(apq) <= std::numeric_limits<T>::min()) continue;
                const T theta = (a[q * n + q] - a[p * n + p]) / (T(2) * apq);
                const T t = (theta >= T(0) ? T(1) : T(-1)) /
                            (std::abs(theta) + std::sqrt(theta * theta + T(1)));
                const T c = T(1) / std::sqrt(t * t + T(1)), s = t * c;
                for (size_t k = 0; k < n; ++k) { // columns p, q
                    const T akp = a[k * n + p], akq = a[k * n + q];
                    a[k * n + p] = c * akp - s * akq;
                    a[k * n + q] = s * akp + c * akq;
                }
                for (size_t k = 0; k < n; ++k) { // rows p, q
                    const T apk = a[p * n + k], aqk = a[q * n + k];
                    a[p * n + k] = c * apk - s * aqk;
                    a[q * n + k] = s * apk + c * aqk;
                }
                for (size_t k = 0; k < n; ++k) {
                    const T vkp = V[k * n + p], vkq = V[k * n + q];
                    V[k * n + p] = c * vkp - s * vkq;
                    V[k * n + q] = s * vkp + c * vkq;
                }
            }
    }
    std::vector<size_t> idx(n);
    std::iota(idx.begin(), idx.end(), size_t(0));
    std::sort(idx.begin(), idx.end(),
        [&](size_t x, size_t y) { return a[x * n + x] < a[y * n + y]; });
    w.resize(n);
    std::vector<T> Vs(n * n);
    for (size_t j = 0; j < n; ++j) {
        w[j] = a[idx[j] * n + idx[j]];
        for (size_t k = 0; k < n; ++k) Vs[k * n + j] = V[k * n + idx[j]];
    }
    V.swap(Vs);
}

// ---- triangular inverse -------------------------------------------------

// X (n x n, ldx) = inverse of the upper-triangular s x s block of U starting
// at (i0, i0); only the upper triangle is written.
template <class T>
void
tri_inv_block(const T *U, size_t ldu, size_t i0, size_t s, T *X, size_t ldx)
{
    for (size_t j = 0; j < s; ++j) {
        const T d = U[(i0 + j) * ldu + i0 + j];
        if (d == T(0))
            fail("inverse_backs: zero on the diagonal at " + std::to_string(i0 + j));
        X[(i0 + j) * ldx + i0 + j] = T(1) / d;
        for (size_t i = j; i-- > 0;) {
            T sum(0);
            for (size_t k = i + 1; k <= j; ++k)
                sum += U[(i0 + i) * ldu + i0 + k] * X[(i0 + k) * ldx + i0 + j];
            X[(i0 + i) * ldx + i0 + j] = -sum / U[(i0 + i) * ldu + i0 + i];
        }
    }
}

} // namespace la_detail

// ==== norms ================================================================

// p-norm (p >= 1) of A - B over all elements.
template <int D, class T, class SA, class SB>
real_type_t<T>
norm(const tensorET<D, T, SA> &A, const tensorET<D, T, SB> &B, int p)
{
    using R = real_type_t<T>;
    if (p < 1) la_detail::fail("norm: p must be >= 1");
    for (int d = 0; d < D; ++d)
        if (A.size(d) != B.size(d)) la_detail::fail("norm: shape mismatch");
    const auto a = la_detail::dense(A), b = la_detail::dense(B);
    const T *x = a.data, *y = b.data;
    const double s = detail::parallel_sum<double>(a.size(), 1 << 14,
        [=](size_t i0, size_t i1) {
            double acc = 0;
            for (size_t i = i0; i < i1; ++i) {
                const double v = double(std::abs(x[i] - y[i]));
                acc += p == 1 ? v : (p == 2 ? v * v : std::pow(v, double(p)));
            }
            return acc;
        });
    return R(p == 1 ? s : (p == 2 ? std::sqrt(s) : std::pow(s, 1.0 / p)));
}

// Named norms of A - B: "fro" / "l2" / "2" (Frobenius, i.e. the element-wise
// 2-norm), "l1" / "1" (sum of absolute values), "inf" / "max" (largest
// absolute value).
template <int D, class T, class SA, class SB>
real_type_t<T>
norm(const tensorET<D, T, SA> &A, const tensorET<D, T, SB> &B,
    const std::string &name)
{
    using R = real_type_t<T>;
    if (name == "fro" || name == "l2" || name == "2") return norm(A, B, 2);
    if (name == "l1" || name == "1") return norm(A, B, 1);
    if (name == "inf" || name == "max") {
        for (int d = 0; d < D; ++d)
            if (A.size(d) != B.size(d)) la_detail::fail("norm: shape mismatch");
        const auto a = la_detail::dense(A), b = la_detail::dense(B);
        R m = 0;
        for (size_t i = 0; i < a.size(); ++i)
            m = std::max<R>(m, R(std::abs(a.data[i] - b.data[i])));
        return m;
    }
    la_detail::fail("norm: unknown norm \"" + name + "\"");
}

template <int D, class T, class SA, class SB>
real_type_t<T>
norm(const tensorET<D, T, SA> &A, const tensorET<D, T, SB> &B, const char *name)
{
    return norm(A, B, std::string(name));
}

// ==== conjugate gradient ====================================================

// Solves A x = b for SPD A from x0. maxIter < 0 means n. Stops when the
// residual norm drops below tol, or (with a note on stderr) when p^T A p
// falls below 1e-12 in absolute value.
template <class T, class SA, class SB, class SX>
tensorET<1, T>
conjugateGradient(const tensorET<2, T, SA> &A, const tensorET<1, T, SB> &b,
    const tensorET<1, T, SX> &x0, int maxIter = -1, double tol = 1e-6)
{
    static_assert(la_detail::is_real_v<T>, "conjugateGradient: T must be float or double");
    la_detail::require_square(A, "conjugateGradient");
    const size_t n = A.size(0), lda = A.stride(0);
    if (b.size() != n || x0.size() != n)
        la_detail::fail("conjugateGradient: dimension mismatch");
    tensorET<1, T> x = la_detail::dense(x0), r = la_detail::dense(b);
    tensorET<1, T> p({n}), Ap({n});
    const T *Ad = A.data;
    kernels::gemv<T>(false, n, n, T(-1), Ad, lda, x.data, T(1), r.data);
    T *xd = x.data, *rd = r.data, *pd = p.data, *apd = Ap.data;
    std::copy(rd, rd + n, pd);
    double rr = kernels::dot<T>(n, rd, rd);
    if (std::sqrt(rr) < tol) return x;
    const long iters = maxIter < 0 ? long(n) : long(maxIter);
    const size_t rgrain = std::max<size_t>(4, (size_t(1) << 16) / std::max<size_t>(n, 1));
    for (long it = 0; it < iters; ++it) {
        // Ap = A p and p^T A p in one pass over A
        const double pAp = detail::parallel_sum<double>(n, rgrain,
            [=](size_t i0, size_t i1) {
                kernels::blas_detail::gemv_rows<T>(i0, i1, n, T(1), Ad, lda, pd,
                    T(0), apd);
                return double(la_detail::dot_s(i1 - i0, pd + i0, apd + i0));
            }, 4);
        if (std::abs(pAp) < 1e-12) {
            std::fprintf(stderr,
                "conjugateGradient: p^T A p = %.3e below 1e-12 after %ld "
                "iterations; stopping\n", pAp, it);
            break;
        }
        const T alpha = T(rr / pAp);
        // x += alpha p, r -= alpha A p, and r^T r, in one pass
        const double rr_new = detail::parallel_sum<double>(n, 1 << 14,
            [=](size_t i0, size_t i1) {
                double s = 0;
                for (size_t i = i0; i < i1; ++i) {
                    xd[i] += alpha * pd[i];
                    const T ri = rd[i] - alpha * apd[i];
                    rd[i] = ri;
                    s += double(ri) * double(ri);
                }
                return s;
            });
        if (std::sqrt(rr_new) < tol) break;
        const T beta = T(rr_new / rr);
        rr = rr_new;
        detail::parallel_for(n, 1 << 14, [=](size_t i0, size_t i1) {
            for (size_t i = i0; i < i1; ++i) pd[i] = rd[i] + beta * pd[i];
        });
    }
    return x;
}

// ==== Gauss-Jordan ==========================================================

// Reduced row-echelon form of A (partial pivoting). A pivot candidate
// below max(m, n) * eps * max|A| counts as zero (the column is skipped).
template <class T, class S>
tensorET<2, T>
gaussJordanElimination(const tensorET<2, T, S> &A)
{
    static_assert(la_detail::is_real_v<T>, "gaussJordanElimination: T must be float or double");
    tensorET<2, T> R = la_detail::dense(A);
    const size_t m = R.size(0), n = R.size(1);
    if (m == 0 || n == 0) return R;
    T *a = R.data;
    T amax = 0;
    for (size_t i = 0; i < m * n; ++i) amax = std::max(amax, std::abs(a[i]));
    const T eps = T(std::max(m, n)) * std::numeric_limits<T>::epsilon() * amax;
    const bool par = double(m) * n * std::min(m, n) > 4e6 && detail::max_threads() > 1;
    size_t prow = 0;
    bool skip = false;
    auto step_serial = [&](size_t col) {
        size_t p = prow;
        T best = std::abs(a[prow * n + col]);
        for (size_t i = prow + 1; i < m; ++i) {
            const T v = std::abs(a[i * n + col]);
            if (v > best) {
                best = v;
                p = i;
            }
        }
        skip = !(best > eps);
        if (skip) return;
        if (p != prow)
            std::swap_ranges(a + p * n + col, a + p * n + n, a + prow * n + col);
        const T inv = T(1) / a[prow * n + col];
        for (size_t c = col; c < n; ++c) a[prow * n + c] *= inv;
        a[prow * n + col] = T(1);
    };
    auto eliminate = [&](size_t i, size_t col) {
        if (i == prow) return;
        T *ri = a + i * n;
        const T f = ri[col];
        if (f != T(0)) {
            la_detail::axpy_s(n - col, -f, a + prow * n + col, ri + col);
            ri[col] = T(0);
        }
    };
#ifdef _OPENMP
    if (par) {
#pragma omp parallel
        {
            for (size_t col = 0; col < n; ++col) {
#pragma omp barrier
                if (prow >= m) break;
#pragma omp single
                step_serial(col);
                if (skip) continue;
#pragma omp for schedule(static)
                for (long long i = 0; i < static_cast<long long>(m); ++i)
                    eliminate(size_t(i), col);
#pragma omp single
                ++prow;
            }
        }
        return R;
    }
#endif
    (void)par;
    for (size_t col = 0; col < n && prow < m; ++col) {
        step_serial(col);
        if (skip) continue;
        for (size_t i = 0; i < m; ++i) eliminate(i, col);
        ++prow;
    }
    return R;
}

// [A | b] for a vector b, or [A | B] for a matrix B.
template <class T, class SA, class SB>
tensorET<2, T>
augmentMatrix(const tensorET<2, T, SA> &A, const tensorET<1, T, SB> &b)
{
    const size_t m = A.size(0), n = A.size(1);
    if (b.size() != m) la_detail::fail("augmentMatrix: row count mismatch");
    tensorET<2, T> R = la_detail::mat<T>(m, n + 1);
    for (size_t i = 0; i < m; ++i) {
        std::copy(A.data + i * A.stride(0), A.data + i * A.stride(0) + n,
            R.data + i * (n + 1));
        R.data[i * (n + 1) + n] = b[i];
    }
    return R;
}

template <class T, class SA, class SB>
tensorET<2, T>
augmentMatrix(const tensorET<2, T, SA> &A, const tensorET<2, T, SB> &B)
{
    const size_t m = A.size(0), n = A.size(1), k = B.size(1);
    if (B.size(0) != m) la_detail::fail("augmentMatrix: row count mismatch");
    tensorET<2, T> R = la_detail::mat<T>(m, n + k);
    for (size_t i = 0; i < m; ++i) {
        std::copy(A.data + i * A.stride(0), A.data + i * A.stride(0) + n,
            R.data + i * (n + k));
        std::copy(B.data + i * B.stride(0), B.data + i * B.stride(0) + k,
            R.data + i * (n + k) + n);
    }
    return R;
}

// ==== QR / Gram-Schmidt ====================================================

// Thin QR of the m x n matrix A (m >= n), blocked Householder: A is
// overwritten with Q (orthonormal columns) and R (n x n, upper triangular,
// non-negative diagonal) is returned. Rank-deficient input is not handled.
template <class T, class S>
tensorET<2, T>
qrDecompositionTile(tensorET<2, T, S> &A)
{
    static_assert(la_detail::is_real_v<T> && std::is_same_v<S, Cpu::HostStorage<T>>,
        "qrDecompositionTile: host float or double matrix");
    auto qr = la_detail::thin_qr(la_detail::dense(A));
    Cpu::Backend::copy_data(A, qr.first); // A may be a view
    return std::move(qr.second);
}

// Orthonormal basis of V's columns, in order (the columns Gram-Schmidt
// would produce), computed through Householder QR.
template <class T, class S>
tensorET<2, T>
gramSchmidtOrthogonalization(const tensorET<2, T, S> &V)
{
    static_assert(la_detail::is_real_v<T>, "gramSchmidtOrthogonalization: T must be float or double");
    return la_detail::thin_qr(la_detail::dense(V)).first;
}

// ==== LU ====================================================================

// {L, U} with A = L U, unit-diagonal L, no pivoting. Throws when a pivot has
// absolute value <= tol.
template <class T, class S>
std::pair<tensorET<2, T>, tensorET<2, T>>
luDcmp(const tensorET<2, T, S> &A, double tol = 1e-12)
{
    static_assert(la_detail::is_real_v<T>, "luDcmp: T must be float or double");
    la_detail::require_square(A, "luDcmp");
    tensorET<2, T> LU = la_detail::dense(A);
    const size_t n = LU.size(0);
    la_detail::lu_inplace(LU.data, n, n, nullptr, tol, "luDcmp");
    tensorET<2, T> L = la_detail::split_lu(LU); // LU now holds U
    return {std::move(L), std::move(LU)};
}

// {{L, U}, P} with P A = L U (partial pivoting), P a permutation matrix.
template <class T, class S>
std::pair<std::pair<tensorET<2, T>, tensorET<2, T>>, tensorET<2, T>>
luDcmpPivoted(const tensorET<2, T, S> &A, double tol = 1e-12)
{
    static_assert(la_detail::is_real_v<T>, "luDcmpPivoted: T must be float or double");
    la_detail::require_square(A, "luDcmpPivoted");
    tensorET<2, T> LU = la_detail::dense(A);
    const size_t n = LU.size(0);
    std::vector<size_t> perm;
    la_detail::lu_inplace(LU.data, n, n, &perm, tol, "luDcmpPivoted");
    tensorET<2, T> L = la_detail::split_lu(LU); // LU now holds U
    tensorET<2, T> P = la_detail::mat<T>(n, n, T(0));
    for (size_t i = 0; i < n; ++i) P.data[i * n + perm[i]] = T(1);
    la_detail::remember_perm<T>(P.data, perm); // lets luSolve skip reading P
    // Explicit types: a nested braced list would select pair's const&
    // constructor and deep-copy all three matrices.
    using Factors = std::pair<tensorET<2, T>, tensorET<2, T>>;
    return std::pair<Factors, tensorET<2, T>>(
        Factors(std::move(L), std::move(LU)), std::move(P));
}

// Same result as luDcmpPivoted (kept for the old interface).
template <class T, class S>
std::pair<std::pair<tensorET<2, T>, tensorET<2, T>>, tensorET<2, T>>
luDcmpPivotedTile(const tensorET<2, T, S> &A, double tol = 1e-12)
{
    return luDcmpPivoted(A, tol);
}

// Solves A X = B for a matrix B, given P A = L U.
template <class T, class SL, class SU, class SP, class SB>
tensorET<2, T>
luSolve(const tensorET<2, T, SL> &L, const tensorET<2, T, SU> &U,
    const tensorET<2, T, SP> &P, const tensorET<2, T, SB> &B)
{
    static_assert(la_detail::is_real_v<T>, "luSolve: T must be float or double");
    const size_t n = L.size(0), k = B.size(1);
    if (U.size(0) != n || P.size(0) != n || B.size(0) != n)
        la_detail::fail("luSolve: dimension mismatch");
    static_assert(is_host_storage_v<SL> && is_host_storage_v<SU> && is_host_storage_v<SB>,
        "luSolve works on host tensors");
    const auto perm = la_detail::perm_of(P);
    tensorET<2, T> X = la_detail::mat<T>(n, k);
    for (size_t i = 0; i < n; ++i)
        std::copy(B.data + perm[i] * B.stride(0), B.data + perm[i] * B.stride(0) + k,
            X.data + i * k);
    la_detail::trsm_left(true, false, n, k, L.data, L.stride(0), X.data, k);
    la_detail::trsm_left(false, false, n, k, U.data, U.stride(0), X.data, k);
    return X;
}

// Solves A x = b for a vector b, given P A = L U.
template <class T, class SL, class SU, class SP, class SB>
tensorET<1, T>
luSolve(const tensorET<2, T, SL> &L, const tensorET<2, T, SU> &U,
    const tensorET<2, T, SP> &P, const tensorET<1, T, SB> &b)
{
    static_assert(la_detail::is_real_v<T>, "luSolve: T must be float or double");
    const size_t n = L.size(0);
    if (U.size(0) != n || P.size(0) != n || b.size() != n)
        la_detail::fail("luSolve: dimension mismatch");
    const auto perm = la_detail::perm_of(P);
    tensorET<1, T> x({n});
    for (size_t i = 0; i < n; ++i) x.data[i] = b[perm[i]];
    la_detail::trsm_left(true, false, n, 1, L.data, L.stride(0), x.data, 1);
    la_detail::trsm_left(false, false, n, 1, U.data, U.stride(0), x.data, 1);
    return x;
}

// General inverse: P A = L U, then A X = I by blocked triangular solves.
template <class T, class S>
tensorET<2, T>
inverse(const tensorET<2, T, S> &A, double tol = 1e-12)
{
    static_assert(la_detail::is_real_v<T>, "inverse: T must be float or double");
    la_detail::require_square(A, "inverse");
    tensorET<2, T> LU = la_detail::dense(A);
    const size_t n = LU.size(0);
    std::vector<size_t> perm;
    la_detail::lu_inplace(LU.data, n, n, &perm, tol, "inverse");
    tensorET<2, T> X = la_detail::identity<T>(n);
    la_detail::lu_solve_inplace(LU.data, n, perm, X.data, n);
    return X;
}

// ==== triangular inverse ====================================================

// Inverse of the upper-triangular U (only its upper triangle is read). The
// diagonal blocks are inverted in parallel, then adjacent inverted blocks
// are joined pairwise, X12 = -X11 (U12 X22), with GEMMs (about n^3/3 useful
// flops, all in GEMM). `m` is the old block-size hint; any n works.
template <class T, class S>
tensorET<2, T>
inverse_backs(const tensorET<2, T, S> &U, size_t m = 4)
{
    static_assert(la_detail::is_real_v<T>, "inverse_backs: T must be float or double");
    (void)m;
    la_detail::require_square(U, "inverse_backs");
    const size_t n = U.size(0), ldu = U.stride(0);
    tensorET<2, T> X = la_detail::mat<T>(n, n, T(0));
    if (n == 0) return X;
    const T *u = U.data;
    T *x = X.data;
    constexpr size_t nb = 64;
    const size_t nblk = (n + nb - 1) / nb;
    detail::parallel_for(nblk, 1, [&](size_t b0, size_t b1) {
        for (size_t bk = b0; bk < b1; ++bk)
            la_detail::tri_inv_block(u, ldu, bk * nb, std::min(nb, n - bk * nb), x, n);
    }, 1);
    std::vector<T> tmp;
    for (size_t s = nb; s < n; s *= 2) {
        const size_t pairs = (n + 2 * s - 1) / (2 * s);
        auto join = [&](size_t p, T *t) {
            const size_t i0 = p * 2 * s, i1 = i0 + s;
            if (i1 >= n) return;
            const size_t w = std::min(s, n - i1); // size of the second block
            // t = U12 X22 (s x w), then X12 = -X11 t
            kernels::gemm<T>(false, false, s, w, w, T(1), u + i0 * ldu + i1, ldu,
                x + i1 * n + i1, n, T(0), t, w);
            kernels::gemm<T>(false, false, s, w, s, T(-1), x + i0 * n + i0, n, t,
                w, T(0), x + i0 * n + i1, n);
        };
        const int nt = detail::max_threads();
        if (pairs >= size_t(nt) && nt > 1) {
            detail::parallel_for(pairs, 1, [&](size_t p0, size_t p1) {
                std::vector<T> t(s * s);
                for (size_t p = p0; p < p1; ++p) join(p, t.data());
            }, 1);
        } else {
            tmp.resize(s * s);
            for (size_t p = 0; p < pairs; ++p) join(p, tmp.data());
        }
    }
    return X;
}

// ==== eigenproblems ==========================================================

// All eigenpairs of the symmetric matrix A (cyclic Jacobi; meant for small
// matrices): w ascending, V's columns the eigenvectors.
template <class T, class S>
void
symmetricEigen(const tensorET<2, T, S> &A, tensorET<1, T> &w, tensorET<2, T> &V)
{
    static_assert(la_detail::is_real_v<T>, "symmetricEigen: T must be float or double");
    la_detail::require_square(A, "symmetricEigen");
    const size_t n = A.size(0);
    std::vector<T> a(n * n), wv, Vv;
    for (size_t i = 0; i < n; ++i)
        for (size_t j = 0; j < n; ++j)
            a[i * n + j] = (A(i, j) + A(j, i)) / T(2);
    la_detail::jacobi_eigen(a, n, wv, Vv);
    w = tensorET<1, T>({n});
    V = la_detail::mat<T>(n, n);
    std::copy(wv.begin(), wv.end(), w.data);
    std::copy(Vv.begin(), Vv.end(), V.data);
}

// Lanczos with full reorthogonalization (two Gram-Schmidt passes against
// all previous vectors): returns {alpha (m), beta (m - 1), Q (m x n)}, the
// tridiagonal coefficients and the Lanczos vectors (rows of Q), started at
// q0 (normalized). On breakdown the next vector is a random vector
// orthogonal to the previous ones and its beta is 0.
template <class T, class S, class S0>
std::tuple<tensorET<1, T>, tensorET<1, T>, tensorET<2, T>>
lanczos(const tensorET<2, T, S> &A, size_t m, const tensorET<1, T, S0> &q0)
{
    static_assert(la_detail::is_real_v<T>, "lanczos: T must be float or double");
    la_detail::require_square(A, "lanczos");
    const size_t n = A.size(0), lda = A.stride(0);
    if (m == 0 || m > n) la_detail::fail("lanczos: need 1 <= m <= n");
    if (q0.size() != n) la_detail::fail("lanczos: q0 has the wrong size");
    tensorET<1, T> alpha({m}, T(0)), beta({m - 1}, T(0));
    tensorET<2, T> Q = la_detail::mat<T>(m, n, T(0));
    std::vector<T> w(n), c(m);
    const tensorET<1, T> q = la_detail::dense(q0);
    const T qn = std::sqrt(kernels::dot<T>(n, q.data, q.data));
    if (!(qn > T(0))) la_detail::fail("lanczos: q0 is zero");
    for (size_t i = 0; i < n; ++i) Q.data[i] = q.data[i] / qn;
    T anorm = 0;
    std::mt19937_64 rng(12345);
    std::normal_distribution<double> nd;
    auto reorth = [&](size_t upto) { // w -= Q[0:upto]^T (Q[0:upto] w), twice
        for (int pass = 0; pass < 2; ++pass) {
            kernels::gemv<T>(false, upto, n, T(1), Q.data, n, w.data(), T(0), c.data());
            kernels::gemv<T>(true, upto, n, T(-1), Q.data, n, c.data(), T(1), w.data());
        }
    };
    for (size_t j = 0; j < m; ++j) {
        const T *qj = Q.data + j * n;
        kernels::gemv<T>(false, n, n, T(1), A.data, lda, qj, T(0), w.data());
        const T a = kernels::dot<T>(n, qj, w.data());
        alpha.data[j] = a;
        if (j + 1 == m) break;
        kernels::axpy<T>(n, -a, qj, w.data());
        if (j > 0) kernels::axpy<T>(n, -beta.data[j - 1], Q.data + (j - 1) * n, w.data());
        reorth(j + 1);
        T bnorm = std::sqrt(kernels::dot<T>(n, w.data(), w.data()));
        anorm = std::max({anorm, std::abs(a), bnorm});
        if (bnorm <= T(64) * std::numeric_limits<T>::epsilon() * std::max(anorm, T(1))) {
            for (auto &v : w) v = T(nd(rng)); // breakdown: restart orthogonally
            reorth(j + 1);
            const T rn = std::sqrt(kernels::dot<T>(n, w.data(), w.data()));
            for (auto &v : w) v /= rn;
            beta.data[j] = T(0);
        } else {
            beta.data[j] = bnorm;
            for (auto &v : w) v /= bnorm;
        }
        std::copy(w.begin(), w.end(), Q.data + (j + 1) * n);
    }
    return {std::move(alpha), std::move(beta), std::move(Q)};
}

// ==== matrix exponential =====================================================

namespace la_detail {

template <class T>
T
norm1(const Mat<T> &A)
{
    const size_t m = A.size(0), n = A.size(1);
    std::vector<T> s(n, T(0));
    for (size_t i = 0; i < m; ++i)
        for (size_t j = 0; j < n; ++j) s[j] += std::abs(A.data[i * n + j]);
    return n ? *std::max_element(s.begin(), s.end()) : T(0);
}

template <class T>
void
gemm_into(const Mat<T> &A, const Mat<T> &B, Mat<T> &C, T alpha = T(1), T beta = T(0))
{
    kernels::gemm<T>(false, false, A.size(0), B.size(1), A.size(1), alpha,
        A.data, A.size(1), B.data, B.size(1), beta, C.data, C.size(1));
}

// C = sum_k c[k] M_k (+ d I), element-wise over contiguous n x n matrices.
template <class T>
void
lincomb(Mat<T> &C, std::initializer_list<std::pair<T, const Mat<T> *>> terms, T d)
{
    const size_t n = C.size(0), nn = n * n;
    T *c = C.data;
    detail::parallel_for(nn, 1 << 14, [&](size_t b, size_t e) {
        for (size_t i = b; i < e; ++i) {
            T s(0);
            for (const auto &t : terms) s += t.first * t.second->data[i];
            c[i] = s;
        }
    });
    for (size_t i = 0; i < n; ++i) c[i * n + i] += d;
}

} // namespace la_detail

// Matrix exponential: scaling and squaring with Pade approximants of degree
// 3, 5, 7, 9 or 13 chosen from the 1-norm (Higham 2005). Every product is a
// GEMM, the Pade solve a blocked LU solve.
template <class T, class S>
tensorET<2, T>
expm(const tensorET<2, T, S> &A0)
{
    static_assert(la_detail::is_real_v<T>, "expm: T must be float or double");
    using namespace la_detail;
    require_square(A0, "expm");
    const size_t n = A0.size(0);
    Mat<T> A = dense(A0);
    if (n == 0) return A;
    const double theta[5] = {1.495585217958292e-2, 2.539398330063230e-1,
        9.504178996162932e-1, 2.097847961257068e0, 5.371920351148152e0};
    const int degs[5] = {3, 5, 7, 9, 13};
    const double b3[] = {120, 60, 12, 1};
    const double b5[] = {30240, 15120, 3360, 420, 30, 1};
    const double b7[] = {17297280, 8648640, 1995840, 277200, 25200, 756, 56, 1};
    const double b9[] = {17643225600., 8821612800., 2075673600., 302702400.,
        30270240., 2162160., 110880., 3960., 90., 1.};
    const double b13[] = {64764752532480000., 32382376266240000.,
        7771770303897600., 1187353796428800., 129060195264000.,
        10559470521600., 670442572800., 33522128640., 1323241920., 40840800.,
        960960., 16380., 182., 1.};
    const double a1 = double(norm1(A));
    int deg = 13, s = 0;
    for (int i = 0; i < 4; ++i)
        if (a1 <= theta[i]) {
            deg = degs[i];
            break;
        }
    if (deg == 13 && a1 > theta[4]) {
        s = std::max(0, int(std::ceil(std::log2(a1 / theta[4]))));
        const T scale = T(std::ldexp(1.0, -s));
        for (size_t i = 0; i < n * n; ++i) A.data[i] *= scale;
    }
    Mat<T> A2 = mat<T>(n, n), U = mat<T>(n, n), V = mat<T>(n, n), tmp = mat<T>(n, n);
    gemm_into(A, A, A2);
    if (deg == 13) {
        Mat<T> A4 = mat<T>(n, n), A6 = mat<T>(n, n);
        gemm_into(A2, A2, A4);
        gemm_into(A4, A2, A6);
        const double *b = b13;
        lincomb<T>(tmp, {{T(b[13]), &A6}, {T(b[11]), &A4}, {T(b[9]), &A2}}, T(0));
        Mat<T> inner = mat<T>(n, n);
        gemm_into(A6, tmp, inner);
        lincomb<T>(tmp, {{T(1), &inner}, {T(b[7]), &A6}, {T(b[5]), &A4},
                            {T(b[3]), &A2}}, T(b[1]));
        gemm_into(A, tmp, U);
        lincomb<T>(tmp, {{T(b[12]), &A6}, {T(b[10]), &A4}, {T(b[8]), &A2}}, T(0));
        gemm_into(A6, tmp, inner);
        lincomb<T>(V, {{T(1), &inner}, {T(b[6]), &A6}, {T(b[4]), &A4},
                          {T(b[2]), &A2}}, T(b[0]));
    } else {
        const double *b = deg == 3 ? b3 : deg == 5 ? b5 : deg == 7 ? b7 : b9;
        // powers A^2, A^4, ... up to A^(deg-1)
        std::vector<Mat<T>> pw;
        pw.push_back(A2);
        for (int p = 4; p < deg; p += 2) {
            Mat<T> next = mat<T>(n, n);
            gemm_into(pw.back(), A2, next);
            pw.push_back(std::move(next));
        }
        // tmp = sum_{odd k} b[k] A^(k-1), V = sum_{even k} b[k] A^k
        auto accumulate = [&](Mat<T> &dst, int first) {
            T *d = dst.data;
            std::fill(d, d + n * n, T(0));
            for (size_t i = 0; i < n; ++i) d[i * n + i] = T(b[first]);
            for (int k = first + 2; k <= deg; k += 2) {
                const Mat<T> &P = pw[size_t((k - first) / 2 - 1)];
                const T c = T(b[k]);
                for (size_t i = 0; i < n * n; ++i) d[i] += c * P.data[i];
            }
        };
        accumulate(tmp, 1);
        gemm_into(A, tmp, U);
        accumulate(V, 0);
    }
    // Solve (V - U) X = (V + U)
    Mat<T> Q = mat<T>(n, n), X = mat<T>(n, n);
    for (size_t i = 0; i < n * n; ++i) {
        Q.data[i] = V.data[i] - U.data[i];
        X.data[i] = V.data[i] + U.data[i];
    }
    std::vector<size_t> perm;
    lu_inplace(Q.data, n, n, &perm, 0.0, "expm");
    lu_solve_inplace(Q.data, n, perm, X.data, n);
    for (int k = 0; k < s; ++k) { // undo the scaling by squaring
        gemm_into(X, X, tmp);
        std::swap(X, tmp);
    }
    return X;
}

// ==== LOBPCG =================================================================

// The nev smallest eigenpairs of the symmetric matrix A, by LOBPCG from the
// initial block X0 (n x k, k >= nev columns). Writes eigvals (ascending)
// and X (n x nev, columns = eigenvectors); returns the iterations used.
// Converged when every residual norm ||A x - lambda x|| is below
// tol * max(1, |lambda|). Each Rayleigh-Ritz step orthonormalizes the basis
// [X, W, P] (Householder QR, dependent directions dropped) and solves the
// small eigenproblem with Jacobi.
template <class T, class S, class S0>
int
lobpcg(const tensorET<2, T, S> &A, int nev, const tensorET<2, T, S0> &X0,
    tensorET<1, T> &eigvals, tensorET<2, T> &X, int maxIter = 100,
    double tol = 1e-8)
{
    static_assert(la_detail::is_real_v<T>, "lobpcg: T must be float or double");
    using namespace la_detail;
    require_square(A, "lobpcg");
    const size_t n = A.size(0), k = X0.size(1);
    if (nev < 1 || size_t(nev) > k || X0.size(0) != n)
        fail("lobpcg: need X0 of size n x k with k >= nev >= 1");
    const Mat<T> Ad = dense(A);
    auto matmul = [](const Mat<T> &L, const Mat<T> &R, bool tl = false) {
        const size_t m = tl ? L.size(1) : L.size(0), kk = tl ? L.size(0) : L.size(1);
        Mat<T> C = mat<T>(m, R.size(1));
        kernels::gemm<T>(tl, false, m, R.size(1), kk, T(1), L.data, L.size(1),
            R.data, R.size(1), T(0), C.data, R.size(1));
        return C;
    };
    auto finish = [&](const std::vector<T> &lam, const Mat<T> &Xk) {
        eigvals = tensorET<1, T>({size_t(nev)});
        X = mat<T>(n, size_t(nev));
        for (int i = 0; i < nev; ++i) eigvals.data[i] = lam[size_t(i)];
        for (size_t r = 0; r < n; ++r)
            std::copy(Xk.data + r * Xk.size(1), Xk.data + r * Xk.size(1) + nev,
                X.data + r * size_t(nev));
    };
    // Small problems: dense Jacobi is exact and cheaper.
    if (n <= 3 * k || n <= 32) {
        std::vector<T> a(Ad.data, Ad.data + n * n), w, V;
        for (size_t i = 0; i < n; ++i)
            for (size_t j = i + 1; j < n; ++j)
                a[i * n + j] = a[j * n + i] = (a[i * n + j] + a[j * n + i]) / T(2);
        jacobi_eigen(a, n, w, V);
        Mat<T> Vm = mat<T>(n, n);
        std::copy(V.begin(), V.end(), Vm.data);
        finish(w, Vm);
        return 0;
    }
    // Rayleigh-Ritz on the orthonormal basis Q (n x s): the k smallest Ritz
    // pairs, coefficients C (s x k).
    auto rayleigh_ritz = [&](const Mat<T> &Q, const Mat<T> &AQ, std::vector<T> &lam,
                             Mat<T> &C) {
        const size_t s = Q.size(1);
        Mat<T> G = matmul(Q, AQ, true);
        std::vector<T> g(s * s), w, V;
        for (size_t i = 0; i < s; ++i)
            for (size_t j = 0; j < s; ++j)
                g[i * s + j] = (G.data[i * s + j] + G.data[j * s + i]) / T(2);
        jacobi_eigen(g, s, w, V);
        lam.assign(w.begin(), w.begin() + k);
        C = mat<T>(s, k);
        for (size_t i = 0; i < s; ++i)
            std::copy(V.data() + i * s, V.data() + i * s + k, C.data + i * k);
    };
    // Orthonormal basis of [blocks...] (first block kept first), dropping
    // directions whose QR diagonal is below 1e-10 of the largest.
    auto orth = [&](std::initializer_list<const Mat<T> *> blocks) {
        size_t s = 0;
        for (auto *b : blocks) s += b->size(1);
        Mat<T> S = mat<T>(n, s);
        size_t off = 0;
        for (auto *b : blocks) {
            const size_t c = b->size(1);
            for (size_t r = 0; r < n; ++r)
                std::copy(b->data + r * c, b->data + r * c + c, S.data + r * s + off);
            off += c;
        }
        auto qr = thin_qr(std::move(S));
        T rmax = 0;
        for (size_t j = 0; j < s; ++j) rmax = std::max(rmax, std::abs(qr.second.data[j * s + j]));
        std::vector<size_t> keep;
        for (size_t j = 0; j < s; ++j)
            if (std::abs(qr.second.data[j * s + j]) > T(1e-10) * rmax) keep.push_back(j);
        Mat<T> Q = mat<T>(n, keep.size());
        for (size_t r = 0; r < n; ++r)
            for (size_t j = 0; j < keep.size(); ++j)
                Q.data[r * keep.size() + j] = qr.first.data[r * s + keep[j]];
        return Q;
    };

    const Mat<T> X0d = dense(X0);
    Mat<T> Xk = orth({&X0d});
    if (Xk.size(1) < k) fail("lobpcg: X0 has linearly dependent columns");
    Mat<T> AX = matmul(Ad, Xk);
    std::vector<T> lam;
    Mat<T> C;
    rayleigh_ritz(Xk, AX, lam, C);
    Xk = matmul(Xk, C);
    AX = matmul(AX, C);
    Mat<T> P, AP;
    int it = 0;
    for (; it < maxIter; ++it) {
        // residuals R = AX - X diag(lam)
        Mat<T> R = mat<T>(n, k);
        std::vector<T> rn(k, T(0));
        for (size_t r = 0; r < n; ++r)
            for (size_t j = 0; j < k; ++j) {
                const T v = AX.data[r * k + j] - lam[j] * Xk.data[r * k + j];
                R.data[r * k + j] = v;
                rn[j] += v * v;
            }
        bool done = true;
        std::vector<size_t> active;
        for (size_t j = 0; j < k; ++j) {
            const bool conv = std::sqrt(rn[j]) <= T(tol) * std::max(T(1), std::abs(lam[j]));
            if (!conv) active.push_back(j);
            if (j < size_t(nev) && !conv) done = false;
        }
        if (done) break;
        Mat<T> W = mat<T>(n, active.size());
        for (size_t r = 0; r < n; ++r)
            for (size_t j = 0; j < active.size(); ++j)
                W.data[r * active.size() + j] = R.data[r * k + active[j]];
        Mat<T> Q = P.size() ? orth({&Xk, &W, &P}) : orth({&Xk, &W});
        if (Q.size(1) <= k) break; // no new directions
        Mat<T> AQ = matmul(Ad, Q);
        rayleigh_ritz(Q, AQ, lam, C);
        Mat<T> Xn = matmul(Q, C), AXn = matmul(AQ, C);
        // P = the part of the new X outside the old X: Q[:, k:] C[k:, :]
        const size_t s = Q.size(1), extra = s - k;
        Mat<T> Qe = mat<T>(n, extra), AQe = mat<T>(n, extra), Ce = mat<T>(extra, k);
        for (size_t r = 0; r < n; ++r) {
            std::copy(Q.data + r * s + k, Q.data + r * s + s, Qe.data + r * extra);
            std::copy(AQ.data + r * s + k, AQ.data + r * s + s, AQe.data + r * extra);
        }
        std::copy(C.data + k * k, C.data + s * k, Ce.data);
        P = matmul(Qe, Ce);
        AP = matmul(AQe, Ce);
        Xk = std::move(Xn);
        AX = std::move(AXn);
    }
    finish(lam, Xk);
    return it;
}

// ==== tournament elimination with a recorded history ========================

// Row operation "row target -= alpha * row source".
template <class T> struct ElimOp {
    size_t target_row, source_row;
    T alpha;
};
// Row swap.
struct PermOp {
    size_t target_row, source_row;
};
template <class T> using Op = std::variant<ElimOp<T>, PermOp>;
template <class T> using History = std::vector<Op<T>>;

// One level of pairwise elimination of column `pivot`: rows pivot + 2kd and
// pivot + 2kd + d (d = factor) are paired; the row with the larger entry in
// the pivot column is swapped to the top and the lower row is eliminated
// against it (entries below 1e-10 |pivot| are flushed to zero). Returns the
// operations in the order applied. `m` is a legacy tile-size hint.
template <class T, class S>
History<T>
elimStep(tensorET<2, T, S> &A, size_t factor, size_t pivot, size_t m = 0)
{
    static_assert(la_detail::is_real_v<T> && is_host_storage_v<S>,
        "elimStep: host float or double matrix");
    (void)m;
    const size_t rows = A.size(0), cols = A.size(1), lda = A.stride(0);
    if (factor == 0 || pivot >= cols) return {};
    T *a = A.data;
    const size_t npairs = pivot + factor < rows
                              ? (rows - pivot - factor + 2 * factor - 1) / (2 * factor)
                              : 0;
    std::vector<ElimOp<T>> el(npairs);
    std::vector<char> swapped(npairs, 0), elim(npairs, 0);
    detail::parallel_for(npairs, std::max<size_t>(1, 16384 / std::max<size_t>(cols, 1)),
        [&](size_t p0, size_t p1) {
            for (size_t p = p0; p < p1; ++p) {
                const size_t top = pivot + 2 * factor * p, bot = top + factor;
                T *rt = a + top * lda, *rb = a + bot * lda;
                if (std::abs(rb[pivot]) > std::abs(rt[pivot])) {
                    std::swap_ranges(rt, rt + cols, rb);
                    swapped[p] = 1;
                }
                const T pv = rt[pivot];
                if (pv == T(0) || rb[pivot] == T(0)) continue;
                const T alpha = rb[pivot] / pv;
                la_detail::axpy_s(cols - pivot - 1, -alpha, rt + pivot + 1, rb + pivot + 1);
                rb[pivot] = T(0);
                const T thr = T(1e-10) * std::abs(pv);
                for (size_t c = pivot + 1; c < cols; ++c)
                    if (std::abs(rb[c]) < thr) rb[c] = T(0);
                el[p] = ElimOp<T>{bot, top, alpha};
                elim[p] = 1;
            }
        }, 1);
    History<T> h;
    for (size_t p = 0; p < npairs; ++p) {
        const size_t top = pivot + 2 * factor * p;
        if (swapped[p]) h.push_back(PermOp{top, top + factor});
        if (elim[p]) h.push_back(el[p]);
    }
    return h;
}

// Reduces A to upper-triangular form in place by pairwise (tournament)
// elimination, column by column at distances 1, 2, 4, ..., and returns the
// history H of row operations, so that H A_original = U (H = E_k ... E_1).
template <class T, class S>
History<T>
revEl(tensorET<2, T, S> &A, size_t m = 0)
{
    History<T> h;
    const size_t rows = A.size(0), cols = A.size(1);
    for (size_t pivot = 0; pivot < std::min(rows, cols); ++pivot)
        for (size_t d = 1; pivot + d < rows; d *= 2) {
            History<T> s = elimStep(A, d, pivot, m);
            h.insert(h.end(), s.begin(), s.end());
        }
    return h;
}

// H A: replays the history as row operations on a copy of A, oldest
// first. One parallel pass: columns are independent.
template <class T, class S>
tensorET<2, T>
matMul(const History<T> &H, const tensorET<2, T, S> &A, size_t m = 0)
{
    (void)m;
    tensorET<2, T> R = la_detail::dense(A);
    const size_t cols = R.size(1);
    T *a = R.data;
    detail::parallel_for(cols, 256, [&](size_t c0, size_t c1) {
        const size_t w = c1 - c0;
        for (const Op<T> &op : H) {
            if (const auto *e = std::get_if<ElimOp<T>>(&op))
                la_detail::axpy_s(w, -e->alpha, a + e->source_row * cols + c0,
                    a + e->target_row * cols + c0);
            else {
                const PermOp &p = std::get<PermOp>(op);
                std::swap_ranges(a + p.target_row * cols + c0,
                    a + p.target_row * cols + c1, a + p.source_row * cols + c0);
            }
        }
    }, 16);
    return R;
}

// A H: replays the history as column operations on a copy of A, most
// recent first (an elimination "row t -= a row s" becomes "column s -= a
// column t", a swap swaps the columns). Rows are independent: blocks of
// rows are transposed into a small column-major buffer so every operation
// is one contiguous vector update. With revEl and inverse_backs:
//   H = revEl(Acopy), inverse = matMul(inverse_backs(Acopy), H).
template <class T, class S>
tensorET<2, T>
matMul(const tensorET<2, T, S> &A, const History<T> &H, size_t m = 0)
{
    (void)m;
    tensorET<2, T> R = la_detail::dense(A);
    const size_t rows = R.size(0), cols = R.size(1);
    constexpr size_t RB = 8;
    T *a = R.data;
    detail::parallel_for((rows + RB - 1) / RB, 1, [&](size_t b0, size_t b1) {
        std::vector<T> buf(cols * RB);
        for (size_t bk = b0; bk < b1; ++bk) {
            const size_t r0 = bk * RB, nr = std::min(RB, rows - r0);
            for (size_t c = 0; c < cols; ++c)
                for (size_t r = 0; r < RB; ++r)
                    buf[c * RB + r] = r < nr ? a[(r0 + r) * cols + c] : T(0);
            for (size_t k = H.size(); k-- > 0;) {
                const Op<T> &op = H[k];
                if (const auto *e = std::get_if<ElimOp<T>>(&op)) {
                    T *s = buf.data() + e->source_row * RB;
                    const T *t = buf.data() + e->target_row * RB;
                    const T al = e->alpha;
                    for (size_t r = 0; r < RB; ++r) s[r] -= al * t[r];
                } else {
                    const PermOp &p = std::get<PermOp>(op);
                    std::swap_ranges(buf.data() + p.target_row * RB,
                        buf.data() + p.target_row * RB + RB,
                        buf.data() + p.source_row * RB);
                }
            }
            for (size_t c = 0; c < cols; ++c)
                for (size_t r = 0; r < nr; ++r)
                    a[(r0 + r) * cols + c] = buf[c * RB + r];
        }
    }, 1);
    return R;
}

} // namespace AXOS
