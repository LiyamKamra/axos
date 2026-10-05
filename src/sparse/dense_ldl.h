// SPDX-License-Identifier: BSD-3-Clause
//
// Small dense kernels for the multifrontal LDL^T (column-major, lower
// triangle). No BLAS dependency: the rank-k update is a register-blocked
// 8x4 micro-kernel written so the compiler vectorizes it (-O3 -march=native).
//
//   syrk_d           C -= L diag(d) L^T   (lower part of C)
//   partial_ldlt     factor the leading ns columns of an fs x fs front and
//                    form the Schur complement in its trailing block
//   trsv kernels     used by the triangular solves
#pragma once

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstring>
#include <cstdlib>
#include <type_traits>
#include <vector>

#ifdef _OPENMP
#include <omp.h>
#endif
#if defined(_MSC_VER) && !defined(__clang__)
#include <immintrin.h>
#include <malloc.h>
// MSVC's OpenMP has no `omp simd`; it vectorizes these loops on its own
#define AXOS_OMP_SIMD
#else
#define AXOS_OMP_SIMD _Pragma("omp simd")
#endif

namespace AXOS {
namespace Sparse {
namespace dense {

inline constexpr int kMR = 8;   // rows per micro-tile
inline constexpr int kNR = 4;   // columns per micro-tile
inline constexpr int kKC = 256; // inner-dimension block
inline constexpr int kPanel = 128; // maximum panel width

namespace packed {
#if defined(__AVX512F__)
inline constexpr int VW = 8;
inline constexpr int MRV = 3, NR = 8;
#else
inline constexpr int VW = 4;
inline constexpr int MRV = 3, NR = 4;
#endif
inline constexpr int MR = MRV * VW;
inline constexpr int KC = 256;
inline constexpr int MC = MR * 8;      // rows per A block
inline constexpr int NCB = NR * 16;    // columns per task
#if defined(__GNUC__) || defined(__clang__)
typedef double vd __attribute__((vector_size(VW * 8), aligned(8)));
#else
// MSVC has no vector extensions: the same 4-wide vector on AVX2 intrinsics
// (loads through vd* happen on the 32-byte aligned packed panels only).
static_assert(VW == 4, "the MSVC vector type is 4 doubles wide");
struct vd {
    __m256d x;
    vd() : x(_mm256_setzero_pd()) {}
    vd(__m256d y) : x(y) {}
    double operator[](int i) const
    {
        alignas(32) double t[4];
        _mm256_store_pd(t, x);
        return t[i];
    }
    vd &operator+=(vd o) { x = _mm256_add_pd(x, o.x); return *this; }
    vd &operator-=(vd o) { x = _mm256_sub_pd(x, o.x); return *this; }
    friend vd operator*(vd a, double b) { return vd(_mm256_mul_pd(a.x, _mm256_set1_pd(b))); }
};
#endif

struct Aligned {
    double *p = nullptr;
    size_t cap = 0;
    static void release(double *q)
    {
#if defined(_MSC_VER)
        _aligned_free(q);
#else
        std::free(q);
#endif
    }
    void reserve(size_t n) {
        if (n > cap) {
            release(p);
            const size_t bytes = ((n * 8 + 63) / 64) * 64;
#if defined(_MSC_VER)
            p = static_cast<double *>(_aligned_malloc(bytes, 64));
#else
            p = static_cast<double *>(std::aligned_alloc(64, bytes));
#endif
            cap = n;
        }
    }
    ~Aligned() { release(p); }
};

// pack rows [i0, i0+mr) x cols p in [0,kc) of L (col-major, ld) into MR-wide panel layout
inline void pack_a(const double *L, int ld, int i0, int mr, int p0, int kc, int m, double *dst) {
    for (int ir = 0; ir < mr; ir += MR) {
        const int rows = std::min(MR, m - (i0 + ir));
        double *d = dst + static_cast<size_t>(ir) * kc;
        for (int p = 0; p < kc; ++p) {
            const double *src = L + static_cast<size_t>(p0 + p) * ld + i0 + ir;
            int r = 0;
            for (; r < rows; ++r) d[p * MR + r] = src[r];
            for (; r < MR; ++r) d[p * MR + r] = 0.0;
        }
    }
}
// pack scaled columns: B[c][p] = L(j0+c, p0+p) * d[p0+p]; NR-wide panels
inline void pack_b(const double *L, int ld, const double *d, int j0, int nc, int p0, int kc, int m, double *dst) {
    for (int jr = 0; jr < nc; jr += NR) {
        const int cols = std::min(NR, m - (j0 + jr));
        double *o = dst + static_cast<size_t>(jr) * kc;
        for (int p = 0; p < kc; ++p) {
            const double dp = d[p0 + p];
            const double *src = L + static_cast<size_t>(p0 + p) * ld + j0 + jr;
            int c = 0;
            for (; c < cols; ++c) o[p * NR + c] = src[c] * dp;
            for (; c < NR; ++c) o[p * NR + c] = 0.0;
        }
    }
}

inline void micro(int kc, const double *Ap, const double *Bp, double *C, int ldc, int i0, int j0, int m) {
    vd acc[NR][MRV];
    for (int c = 0; c < NR; ++c)
        for (int r = 0; r < MRV; ++r) acc[c][r] = vd{} ;
    for (int p = 0; p < kc; ++p) {
        vd a[MRV];
        for (int r = 0; r < MRV; ++r) a[r] = *reinterpret_cast<const vd *>(Ap + r * VW);
        for (int c = 0; c < NR; ++c) {
            const double b = Bp[c];
            for (int r = 0; r < MRV; ++r) acc[c][r] += a[r] * b;
        }
        Ap += MR;
        Bp += NR;
    }
    const int mr = std::min(MR, m - i0), nr = std::min(NR, m - j0);
    if (mr == MR && i0 >= j0 + NR - 1) { // entire tile in the lower triangle
        for (int c = 0; c < nr; ++c) {
            double *cc = C + static_cast<size_t>(j0 + c) * ldc + i0;
            for (int r = 0; r < MRV; ++r) {
                vd v;
                std::memcpy(&v, cc + r * VW, sizeof v);
                v -= acc[c][r];
                std::memcpy(cc + r * VW, &v, sizeof v);
            }
        }
    } else {
        for (int c = 0; c < nr; ++c) {
            double *cc = C + static_cast<size_t>(j0 + c) * ldc + i0;
            for (int r = 0; r < mr; ++r)
                if (i0 + r >= j0 + c) cc[r] -= acc[c][r / VW][r % VW];
        }
    }
}

// C(lower) -= L diag(d) L^T ; L is m x k, leading dimension ldl.
// Column blocks of NCB columns are independent tasks (private packing buffers).
inline void syrk_d_packed(int m, int k, const double *L, int ldl, const double *d, double *C, int ldc, bool allow_parallel) {
    if (m <= 0 || k <= 0) return;
    const double flops = static_cast<double>(m) * m * k;
    bool par = allow_parallel && flops > 2e6;
#ifdef _OPENMP
    if (par && omp_in_parallel()) par = false;
#endif
    int ncb = NCB;
#ifdef _OPENMP
    if (par) { // finer tasks for load balance when there are few column blocks
        const int T = omp_get_max_threads();
        ncb = std::max(NR * 2, std::min(NCB, ((m / (4 * T)) / NR) * NR));
    }
#endif
    const int njc = (m + ncb - 1) / ncb;
    auto task = [&](int jb) {
        static thread_local Aligned At, Bt;
        Bt.reserve(static_cast<size_t>(NCB) * KC);
        At.reserve(static_cast<size_t>(MC) * KC);
        const int jc = jb * ncb;
        const int nc = std::min(ncb, m - jc);
        const int ic_begin = (jc / MC) * MC;
        for (int pc = 0; pc < k; pc += KC) {
            const int kc = std::min(KC, k - pc);
            pack_b(L, ldl, d, jc, nc, pc, kc, m, Bt.p);
            for (int ic = ic_begin; ic < m; ic += MC) {
                const int mc = std::min(MC, m - ic);
                pack_a(L, ldl, ic, mc, pc, kc, m, At.p);
                for (int jr = 0; jr < nc; jr += NR) {
                    const double *Bp = Bt.p + static_cast<size_t>(jr) * kc;
                    for (int ir = 0; ir < mc; ir += MR) {
                        if (ic + ir + MR - 1 < jc + jr) continue; // above the diagonal
                        micro(kc, At.p + static_cast<size_t>(ir) * kc, Bp, C, ldc, ic + ir, jc + jr, m);
                    }
                }
            }
        }
    };
#ifdef _OPENMP
    if (par && njc > 1) {
#pragma omp parallel for schedule(dynamic, 1)
        for (int t = 0; t < njc; ++t) task(t); // early blocks have the most work and run first
        return;
    }
#endif
    for (int jb = 0; jb < njc; ++jb) task(jb);
}
} // namespace packed

// C(i, j) -= sum_p L(i, p) * d(p) * L(j, p) for i >= j, all m x m, where L is
// m x k (leading dimension ldl) and C has leading dimension ldc.
// `work` must hold at least m * k elements (L scaled by d).
template <typename T>
void
syrk_d(int m, int k, const T *L, int ldl, const T *d, T *C, int ldc, T *work,
    bool allow_parallel)
{
    if (m <= 0 || k <= 0) return;
    if constexpr (std::is_same_v<T, double>) {
        packed::syrk_d_packed(m, k, L, ldl, d, C, ldc, allow_parallel);
        (void)work;
        return;
    }
    // W = L * diag(d), tight (leading dimension m)
    for (int p = 0; p < k; ++p) {
        const T dp = d[p];
        const T *lp = L + static_cast<size_t>(p) * ldl;
        T *wp = work + static_cast<size_t>(p) * m;
AXOS_OMP_SIMD
        for (int i = 0; i < m; ++i)
            wp[i] = lp[i] * dp;
    }
    const int nblocks = (m + kNR - 1) / kNR;
    auto column_block = [&](int jb) {
        const int j0 = jb * kNR;
        const int nr = std::min(kNR, m - j0);
        // rows i >= j0 only (lower triangle), in tiles of kMR
        for (int p0 = 0; p0 < k; p0 += kKC) {
            const int kc = std::min(kKC, k - p0);
            for (int i0 = j0; i0 < m; i0 += kMR) {
                const int mr = std::min(kMR, m - i0);
                T acc[kNR][kMR] = {};
                if (mr == kMR && nr == kNR) {
                    for (int p = 0; p < kc; ++p) {
                        const T *lp = L + static_cast<size_t>(p0 + p) * ldl + i0;
                        const T *wp = work + static_cast<size_t>(p0 + p) * m + j0;
                        for (int c = 0; c < kNR; ++c) {
                            const T w = wp[c];
AXOS_OMP_SIMD
                            for (int r = 0; r < kMR; ++r)
                                acc[c][r] += lp[r] * w;
                        }
                    }
                } else {
                    for (int p = 0; p < kc; ++p) {
                        const T *lp = L + static_cast<size_t>(p0 + p) * ldl + i0;
                        const T *wp = work + static_cast<size_t>(p0 + p) * m + j0;
                        for (int c = 0; c < nr; ++c)
                            for (int r = 0; r < mr; ++r)
                                acc[c][r] += lp[r] * wp[c];
                    }
                }
                for (int c = 0; c < nr; ++c) {
                    T *cc = C + static_cast<size_t>(j0 + c) * ldc + i0;
                    for (int r = 0; r < mr; ++r)
                        if (i0 + r >= j0 + c) cc[r] -= acc[c][r];
                }
            }
        }
    };
    const double flops = static_cast<double>(m) * m * k;
    bool par = allow_parallel && flops > 2e6;
#ifdef _OPENMP
    if (par && !omp_in_parallel()) {
#pragma omp parallel for schedule(dynamic, 1)
        for (int jb = 0; jb < nblocks; ++jb)
            column_block(jb);
        return;
    }
#endif
    (void)par;
    for (int jb = 0; jb < nblocks; ++jb)
        column_block(jb);
}

// Factors the leading ns columns of the fs x fs column-major lower front F
// (leading dimension ldf): on return F(0:ns, 0:ns) holds the strictly lower
// part of the unit L11, F(ns:fs, 0:ns) holds L21, d[0:ns) the pivots, and
// F(ns:fs, ns:fs) (lower) the Schur complement F22 - L21 D L21^T.
//
// sign[j] (may be null) is the expected pivot sign of column j (+1/-1/0);
// with eps > 0 a pivot whose sign*value is below eps is replaced by
// sign*eps. `check(j, value)` must return false for an unacceptable pivot;
// factorization then stops and the failing column index is returned in
// *fail (else -1). Returns the number of regularized pivots.
template <typename T, typename CheckFn>
size_t
partial_ldlt(int fs, int ns, T *F, int ldf, T *d, const signed char *sign,
    T eps, CheckFn &&check, int *fail, T *work, bool allow_parallel)
{
    *fail = -1;
    size_t nreg = 0;
    // wider panels for big fronts: fewer passes over the trailing matrix
    const int pw = fs >= 3000 ? 128 : 64;
    for (int j0 = 0; j0 < ns; j0 += pw) {
        const int jb = std::min(pw, ns - j0);
        // ---- diagonal block: columns j0 .. j0+jb, rows j0 .. j0+jb --------
        const int jend = j0 + jb;
        for (int c = j0; c < jend; ++c) {
            T *col = F + static_cast<size_t>(c) * ldf;
            T piv = col[c];
            if (sign && sign[c] != 0) {
                const T sg = static_cast<T>(sign[c]);
                if (!(sg * piv >= eps)) { piv = sg * eps; ++nreg; }
            }
            if (!check(c, piv)) { *fail = c; return nreg; }
            d[c] = piv;
            for (int cj = c + 1; cj < jend; ++cj) {
                T *colj = F + static_cast<size_t>(cj) * ldf;
                const T lj = col[cj] / piv; // L(cj, c)
                for (int i = cj; i < jend; ++i)
                    colj[i] -= col[i] * lj;
            }
            for (int i = c + 1; i < jend; ++i)
                col[i] /= piv;
        }
        // ---- rows below the block: L21 = A21 (L11 D)^-T, independent row chunks
        if (jend < fs) {
            auto chunk = [&](int ra, int rb) {
                for (int c = j0; c < jend; ++c) {
                    T *col = F + static_cast<size_t>(c) * ldf;
                    for (int p = j0; p < c; ++p) {
                        const T *cp = F + static_cast<size_t>(p) * ldf;
                        const T t = d[p] * cp[c]; // d_p L(c, p)
AXOS_OMP_SIMD
                        for (int i = ra; i < rb; ++i)
                            col[i] -= t * cp[i];
                    }
                    const T inv = T(1) / d[c];
AXOS_OMP_SIMD
                    for (int i = ra; i < rb; ++i)
                        col[i] *= inv;
                }
            };
            constexpr int kChunk = 256;
            const int nrows = fs - jend;
            bool parc = allow_parallel && nrows >= 2 * kChunk;
#ifdef _OPENMP
            if (parc && !omp_in_parallel()) {
                const int nch = (nrows + kChunk - 1) / kChunk;
#pragma omp parallel for schedule(static)
                for (int ch = 0; ch < nch; ++ch)
                    chunk(jend + ch * kChunk, std::min(fs, jend + (ch + 1) * kChunk));
            } else
#endif
            {
                (void)parc;
                for (int ra = jend; ra < fs; ra += kChunk)
                    chunk(ra, std::min(fs, ra + kChunk));
            }
        }
        // ---- trailing update: columns j0+jb .. fs -------------------------
        const int m = fs - (j0 + jb);
        if (m > 0) {
            syrk_d(m, jb, F + static_cast<size_t>(j0) * ldf + (j0 + jb), ldf,
                d + j0, F + static_cast<size_t>(j0 + jb) * ldf + (j0 + jb), ldf,
                work, allow_parallel);
        }
    }
    return nreg;
}

} // namespace dense
} // namespace Sparse
} // namespace AXOS
