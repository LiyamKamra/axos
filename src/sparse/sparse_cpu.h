// SPDX-License-Identifier: BSD-3-Clause
//
// CPU sparse kernels (OpenMP over rows). Operate on host Csr matrices.
#pragma once

#include "sparse/csr.h"
#include <cmath>

namespace AXOS {
namespace Sparse {

template <> struct Kernels<Cpu::Backend> {
    // y = alpha * A * x + beta * y.  x has cols() entries, y has rows().
    // When beta == 0, y is not read (so it may hold NaN / garbage).
    template <typename M>
    static void
    spmv(const M &A, const typename M::value_type *x,
        typename M::value_type *y, typename M::value_type alpha,
        typename M::value_type beta)
    {
        using T = typename M::value_type;
        const auto *rp = A.row_ptr();
        const auto *ci = A.col_ind();
        const T *v = A.values();
        const long m = static_cast<long>(A.rows());
#pragma omp parallel for schedule(static) if (A.nnz() > 20000)
        for (long i = 0; i < m; ++i) {
            T s = T(0);
            for (auto k = rp[i]; k < rp[i + 1]; ++k)
                s += v[k] * x[ci[k]];
            y[i] = (beta == T(0)) ? alpha * s : alpha * s + beta * y[i];
        }
    }

    // Y = alpha * A * X + beta * Y with row-major dense X (cols x k) and
    // Y (rows x k).
    template <typename M>
    static void
    spmm(const M &A, const typename M::value_type *X, size_t k,
        typename M::value_type *Y, typename M::value_type alpha,
        typename M::value_type beta)
    {
        using T = typename M::value_type;
        const auto *rp = A.row_ptr();
        const auto *ci = A.col_ind();
        const T *v = A.values();
        const long m = static_cast<long>(A.rows());
#pragma omp parallel for schedule(static) if (A.nnz() > 20000)
        for (long i = 0; i < m; ++i) {
            T *yi = Y + static_cast<size_t>(i) * k;
            for (size_t c = 0; c < k; ++c)
                yi[c] = (beta == T(0)) ? T(0) : beta * yi[c];
            for (auto p = rp[i]; p < rp[i + 1]; ++p) {
                const T a = alpha * v[p];
                const T *xr = X + static_cast<size_t>(ci[p]) * k;
                for (size_t c = 0; c < k; ++c)
                    yi[c] += a * xr[c];
            }
        }
    }

    // A_ij *= r[i] * c[j]; either factor may be null.
    template <typename M>
    static void
    scale(M &A, const real_of_t<typename M::value_type> *r,
        const real_of_t<typename M::value_type> *c)
    {
        const auto *rp = A.row_ptr();
        const auto *ci = A.col_ind();
        auto *v = A.values_mut();
        const long m = static_cast<long>(A.rows());
#pragma omp parallel for schedule(static) if (A.nnz() > 20000)
        for (long i = 0; i < m; ++i) {
            const auto ri = r ? r[i] : real_of_t<typename M::value_type>(1);
            for (auto k = rp[i]; k < rp[i + 1]; ++k)
                v[k] *= c ? ri * c[ci[k]] : ri;
        }
    }

    // Row-wise norm of A into out[rows()].
    template <typename M>
    static void
    row_norms(const M &A, real_of_t<typename M::value_type> *out, Norm p)
    {
        using R = real_of_t<typename M::value_type>;
        const auto *rp = A.row_ptr();
        const auto *v = A.values();
        const long m = static_cast<long>(A.rows());
#pragma omp parallel for schedule(static) if (A.nnz() > 20000)
        for (long i = 0; i < m; ++i) {
            R acc = 0;
            for (auto k = rp[i]; k < rp[i + 1]; ++k) {
                R a = std::abs(v[k]);
                if (p == Norm::L1)
                    acc += a;
                else if (p == Norm::L2)
                    acc += a * a;
                else
                    acc = std::max(acc, a);
            }
            out[i] = (p == Norm::L2) ? std::sqrt(acc) : acc;
        }
    }

    template <typename M>
    static M
    transpose(const M &A)
    {
        using T = typename M::value_type;
        using Idx = typename M::index_type;
        const size_t m = A.rows(), n = A.cols(), nz = A.nnz();
        M B(n, m, nz);
        std::vector<Idx> cnt(n + 1, 0);
        const Idx *rp = A.row_ptr();
        const Idx *ci = A.col_ind();
        const T *v = A.values();
        for (size_t k = 0; k < nz; ++k)
            cnt[ci[k] + 1]++;
        for (size_t j = 0; j < n; ++j)
            cnt[j + 1] += cnt[j];
        std::memcpy(B.row_ptr_mut(), cnt.data(),
            (n + 1) * sizeof(Idx));
        Idx *bci = B.col_ind_mut();
        T *bv = B.values_mut();
        std::vector<Idx> next(cnt.begin(), cnt.end() - 1);
        for (size_t i = 0; i < m; ++i) {
            for (Idx k = rp[i]; k < rp[i + 1]; ++k) {
                Idx dst = next[ci[k]]++;
                bci[dst] = static_cast<Idx>(i);
                bv[dst] = v[k];
            }
        }
        return B;
    }

    // C = A * B (Gustavson, sorted output).
    template <typename M>
    static M
    spgemm(const M &A, const M &B)
    {
        using T = typename M::value_type;
        using Idx = typename M::index_type;
        const size_t m = A.rows(), n = B.cols();
        const Idx *arp = A.row_ptr(), *aci = A.col_ind();
        const Idx *brp = B.row_ptr(), *bci = B.col_ind();
        const T *av = A.values(), *bv = B.values();

        std::vector<Idx> rp(m + 1, 0);
        // Symbolic pass: number of distinct columns per output row.
#pragma omp parallel
        {
            std::vector<Idx> mark(n, -1);
#pragma omp for schedule(dynamic, 64)
            for (long i = 0; i < static_cast<long>(m); ++i) {
                Idx cnt = 0;
                for (Idx p = arp[i]; p < arp[i + 1]; ++p) {
                    const Idx r = aci[p];
                    for (Idx q = brp[r]; q < brp[r + 1]; ++q) {
                        if (mark[bci[q]] != static_cast<Idx>(i)) {
                            mark[bci[q]] = static_cast<Idx>(i);
                            ++cnt;
                        }
                    }
                }
                rp[i + 1] = cnt;
            }
        }
        for (size_t i = 0; i < m; ++i)
            rp[i + 1] += rp[i];

        M C(m, n, static_cast<size_t>(rp[m]));
        std::memcpy(C.row_ptr_mut(), rp.data(),
            (m + 1) * sizeof(Idx));
        Idx *cci = C.col_ind_mut();
        T *cv = C.values_mut();

#pragma omp parallel
        {
            std::vector<Idx> mark(n, -1);
            std::vector<T> acc(n, T(0));
            std::vector<Idx> cols;
#pragma omp for schedule(dynamic, 64)
            for (long i = 0; i < static_cast<long>(m); ++i) {
                cols.clear();
                for (Idx p = arp[i]; p < arp[i + 1]; ++p) {
                    const Idx r = aci[p];
                    const T a = av[p];
                    for (Idx q = brp[r]; q < brp[r + 1]; ++q) {
                        const Idx j = bci[q];
                        if (mark[j] != static_cast<Idx>(i)) {
                            mark[j] = static_cast<Idx>(i);
                            acc[j] = a * bv[q];
                            cols.push_back(j);
                        } else {
                            acc[j] += a * bv[q];
                        }
                    }
                }
                std::sort(cols.begin(), cols.end());
                Idx w = rp[i];
                for (Idx j : cols) {
                    cci[w] = j;
                    cv[w] = acc[j];
                    ++w;
                }
            }
        }
        return C;
    }
};

} // namespace Sparse
} // namespace AXOS
