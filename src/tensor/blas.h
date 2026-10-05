// SPDX-License-Identifier: BSD-3-Clause
//
// Dense BLAS-style kernels on raw row-major arrays (docs/TENSOR_SPEC.md §15).
// The tensor-level functions in tensorMath.h / tensorLinearAlgebra.h call
// these; they can also be used directly.
//
//   gemm(tA, tB, m, n, k, alpha, A, lda, B, ldb, beta, C, ldc)
//        C = alpha op(A) op(B) + beta C       (beta == 0: C is not read)
//   gemv(trans, m, n, alpha, A, lda, x, beta, y)
//        y = alpha op(A) x + beta y           (A is m x n)
//   transpose(m, n, A, lda, B, ldb)           B (n x m) = A^T
//   dot(n, x, y), axpy(n, a, x, y)
//
// GEMM follows the GotoBLAS/BLIS structure: a KC x NC panel of op(B) is
// packed once per (jc, pc) step into a buffer shared by all threads (L3),
// and C is cut into MC x NCt regions, one task each, handed out
// dynamically. A task packs its MC x KC block of op(A) (L2) and runs the
// register-blocked micro-kernel over NR-wide slivers of the B panel (L1).
// No two tasks write the same part of C, and all scratch buffers are
// thread-local and reused across calls (no allocation in the hot path).
// Packing zero-pads edges and applies transposes, so any shape works.
//
// Micro-kernel (MR x NR block of C held in registers):
//   AVX-512  double 8 x 24, float 8 x 48  (24 accumulators)
//   AVX2     double 6 x 8,  float 6 x 16  (12 accumulators)
//   SSE2     double 4 x 4,  float 4 x 8
//   other T  4 x 4 scalar (complex, integer, half)
#pragma once

#include "tensor/parallel.h"
#include "tensor/simd.h"
#include <algorithm>
#include <cstddef>
#include <cstring>
#include <new>
#include <type_traits>
#include <vector>

#ifdef _OPENMP
#include <omp.h>
#endif

namespace AXOS {
namespace kernels {

namespace blas_detail {

// Grow-only, 64-byte aligned scratch buffer.
template <class T> class Scratch {
  public:
    Scratch() = default;
    Scratch(const Scratch &) = delete;
    Scratch &operator=(const Scratch &) = delete;
    ~Scratch() { release(); }
    T *
    get(size_t n)
    {
        if (n > cap_) {
            release();
            p_ = static_cast<T *>(
                ::operator new(n * sizeof(T), std::align_val_t(64)));
            cap_ = n;
        }
        return p_;
    }

  private:
    void
    release()
    {
        if (p_) ::operator delete(p_, std::align_val_t(64));
        p_ = nullptr;
        cap_ = 0;
    }
    T *p_ = nullptr;
    size_t cap_ = 0;
};

// Distinct thread-local buffers for distinct purposes (Tag).
template <class T, int Tag>
T *
scratch(size_t n)
{
    thread_local Scratch<T> s;
    return s.get(n);
}

inline constexpr int kScratchA = 0, kScratchB = 1, kScratchGemv = 2;

// Work (multiply-adds) above which a kernel uses all threads.
inline constexpr double kParallelFma = 1.0e6;

template <class T> struct Blocking {
    using P = simd::Pack<T>;
    static constexpr int W = P::W;
#if defined(AXOS_SIMD_AVX512)
    static constexpr int MR = W > 1 ? 8 : 4, NV = W > 1 ? 3 : 4;
#elif defined(AXOS_SIMD_AVX2)
    static constexpr int MR = W > 1 ? 6 : 4, NV = W > 1 ? 2 : 4;
#elif defined(AXOS_SIMD_SSE2)
    static constexpr int MR = 4, NV = W > 1 ? 2 : 4;
#else
    static constexpr int MR = 4, NV = 4;
#endif
    static constexpr int NR = NV * W;
#ifdef AXOS_GEMM_KC
    static constexpr size_t KC = AXOS_GEMM_KC;
#else
    static constexpr size_t KC = 256; // B sliver (KC x NR) stays in L1
#endif
    static constexpr size_t MC = size_t(MR) * 16; // A block (MC x KC) in L2
    // B panel (KC x NC) of about 4 MiB, shared by all threads in L3.
    static constexpr size_t NC =
        std::max<size_t>(NR, (size_t(4) << 20) / (KC * sizeof(T)) / NR * NR);
};

// acc = Ap(MR x kc) * Bp(kc x NR). The generic version relies on the
// compiler to keep acc in registers; the specializations for the SIMD tile
// shapes name every accumulator, so any compiler (MSVC included) keeps them
// in registers.
template <class T, int MR, int NV> struct Accumulate {
    using P = simd::Pack<T>;
    using R = typename P::reg;
    static AXOS_INLINE void
    run(size_t kc, const T *AXOS_RESTRICT Ap, const T *AXOS_RESTRICT Bp,
        R (&acc)[MR][NV])
    {
        constexpr int W = P::W, NR = NV * W;
        for (int r = 0; r < MR; ++r)
            for (int v = 0; v < NV; ++v)
                acc[r][v] = P::zero();
        for (size_t p = 0; p < kc; ++p) {
            R b[NV];
            for (int v = 0; v < NV; ++v)
                b[v] = P::load(Bp + v * W);
            for (int r = 0; r < MR; ++r) {
                const R a = P::set1(Ap[r]);
                for (int v = 0; v < NV; ++v)
                    acc[r][v] = P::fmadd(a, b[v], acc[r][v]);
            }
            Ap += MR;
            Bp += NR;
        }
    }
};

#define AXOS_ROW2(r) \
    a = P::set1(Ap[r]); \
    c##r##0 = P::fmadd(a, b0, c##r##0); \
    c##r##1 = P::fmadd(a, b1, c##r##1);
#define AXOS_ROW3(r) \
    a = P::set1(Ap[r]); \
    c##r##0 = P::fmadd(a, b0, c##r##0); \
    c##r##1 = P::fmadd(a, b1, c##r##1); \
    c##r##2 = P::fmadd(a, b2, c##r##2);
#define AXOS_DECL2(r) R c##r##0 = P::zero(), c##r##1 = P::zero();
#define AXOS_DECL3(r) \
    R c##r##0 = P::zero(), c##r##1 = P::zero(), c##r##2 = P::zero();
#define AXOS_OUT2(r) \
    acc[r][0] = c##r##0; \
    acc[r][1] = c##r##1;
#define AXOS_OUT3(r) \
    acc[r][0] = c##r##0; \
    acc[r][1] = c##r##1; \
    acc[r][2] = c##r##2;

// 6 x 2 vectors (AVX2): 12 accumulators.
template <class T> struct Accumulate<T, 6, 2> {
    using P = simd::Pack<T>;
    using R = typename P::reg;
    static AXOS_INLINE void
    run(size_t kc, const T *AXOS_RESTRICT Ap, const T *AXOS_RESTRICT Bp,
        R (&acc)[6][2])
    {
        constexpr int W = P::W;
        AXOS_DECL2(0)
        AXOS_DECL2(1) AXOS_DECL2(2) AXOS_DECL2(3) AXOS_DECL2(4)
            AXOS_DECL2(5) for (size_t p = 0; p < kc; ++p)
        {
            const R b0 = P::load(Bp), b1 = P::load(Bp + W);
            R a;
            AXOS_ROW2(0)
            AXOS_ROW2(1) AXOS_ROW2(2) AXOS_ROW2(3) AXOS_ROW2(4) AXOS_ROW2(5)
                Ap += 6;
            Bp += 2 * W;
        }
        AXOS_OUT2(0)
        AXOS_OUT2(1) AXOS_OUT2(2) AXOS_OUT2(3) AXOS_OUT2(4) AXOS_OUT2(5)
    }
};

// 4 x 2 vectors (SSE2): 8 accumulators.
template <class T> struct Accumulate<T, 4, 2> {
    using P = simd::Pack<T>;
    using R = typename P::reg;
    static AXOS_INLINE void
    run(size_t kc, const T *AXOS_RESTRICT Ap, const T *AXOS_RESTRICT Bp,
        R (&acc)[4][2])
    {
        constexpr int W = P::W;
        AXOS_DECL2(0)
        AXOS_DECL2(1) AXOS_DECL2(2)
            AXOS_DECL2(3) for (size_t p = 0; p < kc; ++p)
        {
            const R b0 = P::load(Bp), b1 = P::load(Bp + W);
            R a;
            AXOS_ROW2(0) AXOS_ROW2(1) AXOS_ROW2(2) AXOS_ROW2(3) Ap += 4;
            Bp += 2 * W;
        }
        AXOS_OUT2(0) AXOS_OUT2(1) AXOS_OUT2(2) AXOS_OUT2(3)
    }
};

// 8 x 3 vectors (AVX-512): 24 accumulators.
template <class T> struct Accumulate<T, 8, 3> {
    using P = simd::Pack<T>;
    using R = typename P::reg;
    static AXOS_INLINE void
    run(size_t kc, const T *AXOS_RESTRICT Ap, const T *AXOS_RESTRICT Bp,
        R (&acc)[8][3])
    {
        constexpr int W = P::W;
        AXOS_DECL3(0)
        AXOS_DECL3(1) AXOS_DECL3(2) AXOS_DECL3(3) AXOS_DECL3(4) AXOS_DECL3(5)
            AXOS_DECL3(6) AXOS_DECL3(7) for (size_t p = 0; p < kc; ++p)
        {
            const R b0 = P::load(Bp), b1 = P::load(Bp + W),
                    b2 = P::load(Bp + 2 * W);
            R a;
            AXOS_ROW3(0)
            AXOS_ROW3(1) AXOS_ROW3(2) AXOS_ROW3(3) AXOS_ROW3(4) AXOS_ROW3(5)
                AXOS_ROW3(6) AXOS_ROW3(7) Ap += 8;
            Bp += 3 * W;
        }
        AXOS_OUT3(0)
        AXOS_OUT3(1) AXOS_OUT3(2) AXOS_OUT3(3) AXOS_OUT3(4) AXOS_OUT3(5)
            AXOS_OUT3(6) AXOS_OUT3(7)
    }
};

#undef AXOS_ROW2
#undef AXOS_ROW3
#undef AXOS_DECL2
#undef AXOS_DECL3
#undef AXOS_OUT2
#undef AXOS_OUT3

// C[mr x nr] = alpha * Ap(MR x kc) * Bp(kc x NR) + beta * C.
template <class T, int MR, int NV>
AXOS_INLINE void
micro_kernel(size_t kc, const T *AXOS_RESTRICT Ap, const T *AXOS_RESTRICT Bp,
    T *C, size_t ldc, T alpha, T beta, int mr, int nr)
{
    using P = simd::Pack<T>;
    using R = typename P::reg;
    constexpr int W = P::W, NR = NV * W;
    for (int r = 0; r < MR; ++r) { // C is touched only at the end
        AXOS_PREFETCH(C + r * ldc);
        AXOS_PREFETCH(C + r * ldc + NR - 1);
    }
    R acc[MR][NV];
    Accumulate<T, MR, NV>::run(kc, Ap, Bp, acc);
    const R va = P::set1(alpha);
    if (mr == MR && nr == NR) {
        if (beta == T(0)) {
            for (int r = 0; r < MR; ++r)
                for (int v = 0; v < NV; ++v)
                    P::storeu(C + r * ldc + v * W, P::mul(acc[r][v], va));
        } else if (beta == T(1)) {
            for (int r = 0; r < MR; ++r)
                for (int v = 0; v < NV; ++v) {
                    T *c = C + r * ldc + v * W;
                    P::storeu(c, P::fmadd(acc[r][v], va, P::loadu(c)));
                }
        } else {
            const R vb = P::set1(beta);
            for (int r = 0; r < MR; ++r)
                for (int v = 0; v < NV; ++v) {
                    T *c = C + r * ldc + v * W;
                    P::storeu(
                        c, P::fmadd(acc[r][v], va, P::mul(P::loadu(c), vb)));
                }
        }
        return;
    }
    alignas(64) T tmp[MR * NR];
    for (int r = 0; r < MR; ++r)
        for (int v = 0; v < NV; ++v)
            P::store(tmp + r * NR + v * W, P::mul(acc[r][v], va));
    for (int r = 0; r < mr; ++r) {
        T *c = C + r * ldc;
        const T *t = tmp + r * NR;
        if (beta == T(0))
            for (int j = 0; j < nr; ++j)
                c[j] = t[j];
        else
            for (int j = 0; j < nr; ++j)
                c[j] = t[j] + beta * c[j];
    }
}

// Packs rows [i0, i0+mc) x columns [p0, p0+kc) of op(A) into MR-row slivers
// (sliver s at dst + s*MR*kc, element (r, p) at [p*MR + r]), zero-padded.
template <class T, int MR>
void
pack_a(bool trans, const T *A, size_t lda, size_t i0, size_t mc, size_t p0,
    size_t kc, T *AXOS_RESTRICT dst)
{
    for (size_t ir = 0; ir < mc; ir += MR) {
        const size_t mr = std::min<size_t>(MR, mc - ir);
        T *d = dst + ir * kc;
        if (!trans) {
            const T *a[MR];
            for (size_t r = 0; r < size_t(MR); ++r)
                a[r] = A + (i0 + ir + std::min(r, mr - 1)) * lda + p0;
            if (mr == size_t(MR)) {
                for (size_t p = 0; p < kc; ++p)
                    for (int r = 0; r < MR; ++r)
                        d[p * MR + r] = a[r][p];
            } else {
                for (size_t p = 0; p < kc; ++p)
                    for (size_t r = 0; r < size_t(MR); ++r)
                        d[p * MR + r] = r < mr ? a[r][p] : T(0);
            }
        } else {
            for (size_t p = 0; p < kc; ++p) {
                const T *a = A + (p0 + p) * lda + i0 + ir;
                size_t r = 0;
                for (; r < mr; ++r)
                    d[p * MR + r] = a[r];
                for (; r < size_t(MR); ++r)
                    d[p * MR + r] = T(0);
            }
        }
    }
}

// Packs rows [p0, p0+kc) x columns [j, j+nr) of op(B) into one NR-wide
// sliver (element (p, c) at [p*NR + c]), zero-padded.
template <class T, int NR>
void
pack_b(bool trans, const T *B, size_t ldb, size_t p0, size_t kc, size_t j,
    size_t nr, T *AXOS_RESTRICT d)
{
    if (!trans) {
        for (size_t p = 0; p < kc; ++p) {
            const T *b = B + (p0 + p) * ldb + j;
            size_t c = 0;
            for (; c < nr; ++c)
                d[p * NR + c] = b[c];
            for (; c < size_t(NR); ++c)
                d[p * NR + c] = T(0);
        }
    } else {
        for (size_t c = 0; c < size_t(NR); ++c) {
            if (c < nr) {
                const T *b = B + (j + c) * ldb + p0;
                for (size_t p = 0; p < kc; ++p)
                    d[p * NR + c] = b[p];
            } else {
                for (size_t p = 0; p < kc; ++p)
                    d[p * NR + c] = T(0);
            }
        }
    }
}

template <class T>
void
scale_matrix(size_t m, size_t n, T beta, T *C, size_t ldc)
{
    if (beta == T(1)) return;
    for (size_t i = 0; i < m; ++i) {
        T *c = C + i * ldc;
        if (beta == T(0))
            std::fill(c, c + n, T(0));
        else
            for (size_t j = 0; j < n; ++j)
                c[j] *= beta;
    }
}

} // namespace blas_detail

template <class T>
void
gemm(bool transA, bool transB, size_t m, size_t n, size_t k, T alpha,
    const T *A, size_t lda, const T *B, size_t ldb, T beta, T *C, size_t ldc)
{
    using namespace blas_detail;
    // Blocking constants are spelled Cfg::X inside the lambdas below (MSVC
    // rejects constexpr locals used as template arguments in lambdas).
    using Cfg = Blocking<T>;
    constexpr int MR = Cfg::MR, NR = Cfg::NR;
    constexpr size_t KC = Cfg::KC, MC = Cfg::MC, NC = Cfg::NC;
    if (m == 0 || n == 0) return;
    if (k == 0 || alpha == T(0)) {
        scale_matrix(m, n, beta, C, ldc);
        return;
    }
    const double fma = double(m) * double(n) * double(k);

    // Tiny products: direct loops (packing would dominate).
    if (fma <= 4096.0 && !transA && !transB) {
        for (size_t i = 0; i < m; ++i) {
            T *c = C + i * ldc;
            if (beta == T(0))
                std::fill(c, c + n, T(0));
            else if (beta != T(1))
                for (size_t j = 0; j < n; ++j)
                    c[j] *= beta;
            for (size_t p = 0; p < k; ++p) {
                const T a = alpha * A[i * lda + p];
                const T *b = B + p * ldb;
                for (size_t j = 0; j < n; ++j)
                    c[j] += a * b[j];
            }
        }
        return;
    }

    const int nt = fma >= kParallelFma ? detail::max_threads() : 1;
    const size_t ncmax = std::min(n, NC);
    T *Bp = scratch<T, kScratchB>(KC * ((ncmax + NR - 1) / NR * NR));

    // Task grid over the m x nc region of one (jc, pc) step. A task spans the
    // whole B panel by default (its A block is packed once, the B slivers
    // stream from L3). With too few tasks for the threads, split the columns
    // first (each chunk re-packs the A block, which is cheap), then the rows
    // (smaller A blocks raise the B traffic per flop).
    auto round_up = [](size_t x, size_t q) { return (x + q - 1) / q * q; };
    size_t mcT = std::min(MC, round_up(m, MR));
    size_t ncT = round_up(ncmax, NR);
    auto ntasks = [&](size_t a, size_t b) {
        return ((m + a - 1) / a) * ((ncmax + b - 1) / b);
    };
    auto halve = [&](size_t x, size_t q) { return round_up(x / 2, q); };
    while (nt > 1 && ntasks(mcT, ncT) < size_t(3 * nt)) {
        if (ncT > size_t(4 * NR))
            ncT = halve(ncT, NR);
        else if (mcT > size_t(2 * MR))
            mcT = halve(mcT, MR);
        else if (ncT > size_t(NR))
            ncT = halve(ncT, NR);
        else if (mcT > size_t(MR))
            mcT = halve(mcT, MR);
        else
            break;
    }

    auto pack_panel = [&](size_t jc, size_t nc, size_t pc, size_t kc,
                          size_t s) {
        const size_t j = s * Cfg::NR;
        pack_b<T, Cfg::NR>(transB, B, ldb, pc, kc, jc + j,
            std::min<size_t>(Cfg::NR, nc - j), Bp + s * kc * Cfg::NR);
    };
    auto run_task = [&](size_t jc, size_t nc, size_t pc, size_t kc, T b_eff,
                        size_t t) {
        const size_t nit = (m + mcT - 1) / mcT;
        const size_t it = t % nit, jt = t / nit;
        const size_t i0 = it * mcT, mc = std::min(mcT, m - i0);
        const size_t j0 = jt * ncT;
        if (j0 >= nc) return;
        const size_t ncc = std::min(ncT, nc - j0);
        T *Ap = scratch<T, kScratchA>(Cfg::MC * Cfg::KC);
        pack_a<T, Cfg::MR>(transA, A, lda, i0, mc, pc, kc, Ap);
        // The next B sliver is prefetched into L2 a slice per micro-kernel
        // call, so it is not fetched from L3 on the critical path.
        const size_t sliver = kc * Cfg::NR, nir = (mc + Cfg::MR - 1) / Cfg::MR;
        const size_t step = 64 / sizeof(T) > 0 ? 64 / sizeof(T) : 1;
        const size_t per = ((sliver + step - 1) / step + nir - 1) / nir;
        for (size_t jr = 0; jr < ncc; jr += Cfg::NR) {
            const int nr =
                static_cast<int>(std::min<size_t>(Cfg::NR, ncc - jr));
            const T *bs = Bp + ((j0 + jr) / Cfg::NR) * sliver;
            const T *bnext = jr + Cfg::NR < ncc ? bs + sliver : nullptr;
            T *c = C + i0 * ldc + jc + j0 + jr;
            for (size_t ir = 0, q = 0; ir < mc; ir += Cfg::MR, ++q) {
                if (bnext)
                    for (size_t l = q * per;
                        l < (q + 1) * per && l * step < sliver; ++l)
                        AXOS_PREFETCH_L2(bnext + l * step);
                const int mr =
                    static_cast<int>(std::min<size_t>(Cfg::MR, mc - ir));
                micro_kernel<T, Cfg::MR, Cfg::NV>(kc, Ap + ir * kc, bs,
                    c + ir * ldc, ldc, alpha, b_eff, mr, nr);
            }
        }
    };

    auto body = [&](bool par) {
        (void)par;
        for (size_t jc = 0; jc < n; jc += Cfg::NC) {
            const size_t nc = std::min(Cfg::NC, n - jc);
            const size_t nsl = (nc + Cfg::NR - 1) / Cfg::NR;
            const size_t nit = (m + mcT - 1) / mcT, njt = (nc + ncT - 1) / ncT;
            const long long tasks = static_cast<long long>(nit * njt);
            for (size_t pc = 0; pc < k; pc += Cfg::KC) {
                const size_t kc = std::min(Cfg::KC, k - pc);
                const T b_eff = pc == 0 ? beta : T(1);
#ifdef _OPENMP
                if (par) {
#pragma omp for schedule(static)
                    for (long long s = 0; s < static_cast<long long>(nsl); ++s)
                        pack_panel(jc, nc, pc, kc, size_t(s));
#pragma omp for schedule(dynamic, 1)
                    for (long long t = 0; t < tasks; ++t)
                        run_task(jc, nc, pc, kc, b_eff, size_t(t));
                    continue;
                }
#endif
                for (size_t s = 0; s < nsl; ++s)
                    pack_panel(jc, nc, pc, kc, s);
                for (long long t = 0; t < tasks; ++t)
                    run_task(jc, nc, pc, kc, b_eff, size_t(t));
            }
        }
    };

#ifdef _OPENMP
    if (nt > 1) {
#pragma omp parallel num_threads(nt)
        body(true);
        return;
    }
#endif
    body(false);
}

// ---- GEMV ---------------------------------------------------------------

namespace blas_detail {

// y[i] = alpha * dot(A[i, :], x) + beta * y[i] for rows [i0, i1).
template <class T>
void
gemv_rows(size_t i0, size_t i1, size_t n, T alpha, const T *A, size_t lda,
    const T *x, T beta, T *y)
{
    using P = simd::Pack<T>;
    using R = typename P::reg;
    constexpr size_t W = P::W;
    auto finish = [&](size_t i, T s) {
        y[i] = beta == T(0) ? alpha * s : alpha * s + beta * y[i];
    };
    size_t i = i0;
    for (; i + 4 <= i1; i += 4) {
        const T *a0 = A + i * lda, *a1 = a0 + lda, *a2 = a1 + lda,
                *a3 = a2 + lda;
        R c00 = P::zero(), c01 = P::zero(), c10 = P::zero(), c11 = P::zero();
        R c20 = P::zero(), c21 = P::zero(), c30 = P::zero(), c31 = P::zero();
        size_t j = 0;
        for (; j + 2 * W <= n; j += 2 * W) {
            const R x0 = P::loadu(x + j), x1 = P::loadu(x + j + W);
            c00 = P::fmadd(P::loadu(a0 + j), x0, c00);
            c01 = P::fmadd(P::loadu(a0 + j + W), x1, c01);
            c10 = P::fmadd(P::loadu(a1 + j), x0, c10);
            c11 = P::fmadd(P::loadu(a1 + j + W), x1, c11);
            c20 = P::fmadd(P::loadu(a2 + j), x0, c20);
            c21 = P::fmadd(P::loadu(a2 + j + W), x1, c21);
            c30 = P::fmadd(P::loadu(a3 + j), x0, c30);
            c31 = P::fmadd(P::loadu(a3 + j + W), x1, c31);
        }
        T s0 = P::hsum(P::add(c00, c01)), s1 = P::hsum(P::add(c10, c11));
        T s2 = P::hsum(P::add(c20, c21)), s3 = P::hsum(P::add(c30, c31));
        for (; j < n; ++j) {
            const T xj = x[j];
            s0 += a0[j] * xj;
            s1 += a1[j] * xj;
            s2 += a2[j] * xj;
            s3 += a3[j] * xj;
        }
        finish(i, s0);
        finish(i + 1, s1);
        finish(i + 2, s2);
        finish(i + 3, s3);
    }
    for (; i < i1; ++i) {
        const T *a = A + i * lda;
        R c0 = P::zero(), c1 = P::zero();
        size_t j = 0;
        for (; j + 2 * W <= n; j += 2 * W) {
            c0 = P::fmadd(P::loadu(a + j), P::loadu(x + j), c0);
            c1 = P::fmadd(P::loadu(a + j + W), P::loadu(x + j + W), c1);
        }
        T s = P::hsum(P::add(c0, c1));
        for (; j < n; ++j)
            s += a[j] * x[j];
        finish(i, s);
    }
}

// y[j0:j1] += sum over rows i in [r0, r1) of (alpha x[i]) A[i, j0:j1].
template <class T>
void
gemv_t_cols(size_t r0, size_t r1, size_t j0, size_t j1, T alpha, const T *A,
    size_t lda, const T *x, T *y)
{
    using P = simd::Pack<T>;
    constexpr size_t W = P::W;
    size_t i = r0;
    for (; i + 4 <= r1; i += 4) {
        const T *a0 = A + i * lda, *a1 = a0 + lda, *a2 = a1 + lda,
                *a3 = a2 + lda;
        const T s0 = alpha * x[i], s1 = alpha * x[i + 1];
        const T s2 = alpha * x[i + 2], s3 = alpha * x[i + 3];
        const auto v0 = P::set1(s0), v1 = P::set1(s1), v2 = P::set1(s2),
                   v3 = P::set1(s3);
        size_t j = j0;
        for (; j + W <= j1; j += W) {
            auto acc = P::loadu(y + j);
            acc = P::fmadd(P::loadu(a0 + j), v0, acc);
            acc = P::fmadd(P::loadu(a1 + j), v1, acc);
            acc = P::fmadd(P::loadu(a2 + j), v2, acc);
            acc = P::fmadd(P::loadu(a3 + j), v3, acc);
            P::storeu(y + j, acc);
        }
        for (; j < j1; ++j)
            y[j] += s0 * a0[j] + s1 * a1[j] + s2 * a2[j] + s3 * a3[j];
    }
    for (; i < r1; ++i) {
        const T *a = A + i * lda;
        const T s = alpha * x[i];
        const auto v = P::set1(s);
        size_t j = j0;
        for (; j + W <= j1; j += W)
            P::storeu(y + j, P::fmadd(P::loadu(a + j), v, P::loadu(y + j)));
        for (; j < j1; ++j)
            y[j] += s * a[j];
    }
}

} // namespace blas_detail

// y = alpha op(A) x + beta y, A is m x n (row-major, leading dimension lda).
// trans == false: x has n entries, y has m. trans == true: x has m, y has n.
// Memory-bound: rows (or column blocks) are split across threads and each
// pass keeps several independent accumulators.
template <class T>
void
gemv(bool trans, size_t m, size_t n, T alpha, const T *A, size_t lda,
    const T *x, T beta, T *y)
{
    using namespace blas_detail;
    const size_t ylen = trans ? n : m;
    if (ylen == 0) return;
    if (m == 0 || n == 0 || alpha == T(0)) {
        for (size_t i = 0; i < ylen; ++i)
            y[i] = beta == T(0) ? T(0) : beta * y[i];
        return;
    }
    const size_t work = m * n;
    const int nt = double(work) >= kParallelFma / 4 ? detail::max_threads() : 1;
    if (!trans) {
        // About 64K matrix elements per chunk, in groups of 4 rows.
        const size_t grain = std::max<size_t>(4, (size_t(1) << 16) / n);
        if (nt <= 1) {
            gemv_rows(0, m, n, alpha, A, lda, x, beta, y);
            return;
        }
        detail::parallel_for(
            m, grain,
            [&](size_t b, size_t e) {
                gemv_rows(b, e, n, alpha, A, lda, x, beta, y);
            },
            4);
        return;
    }
    auto init_y = [&](size_t j0, size_t j1) {
        for (size_t j = j0; j < j1; ++j)
            y[j] = beta == T(0) ? T(0) : beta * y[j];
    };
    constexpr size_t kColBlock = 512;
    if (nt <= 1) {
        for (size_t j0 = 0; j0 < n; j0 += kColBlock) {
            const size_t j1 = std::min(n, j0 + kColBlock);
            init_y(j0, j1);
            gemv_t_cols(0, m, j0, j1, alpha, A, lda, x, y);
        }
        return;
    }
    if (n >= size_t(nt) * 256) { // enough columns: split them
        detail::parallel_for(
            n, 256,
            [&](size_t b, size_t e) {
                for (size_t j0 = b; j0 < e; j0 += kColBlock) {
                    const size_t j1 = std::min(e, j0 + kColBlock);
                    init_y(j0, j1);
                    gemv_t_cols(0, m, j0, j1, alpha, A, lda, x, y);
                }
            },
            16);
        return;
    }
    // Few columns, many rows: private partial sums per row chunk.
    const size_t chunks =
        std::min<size_t>(size_t(4) * nt, std::max<size_t>(1, m / 64));
    std::vector<T> part(chunks * n, T(0));
    detail::parallel_for(
        chunks, 1,
        [&](size_t c0, size_t c1) {
            for (size_t c = c0; c < c1; ++c) {
                const size_t r0 = detail::chunk_begin(m, chunks, c, 1);
                const size_t r1 = detail::chunk_begin(m, chunks, c + 1, 1);
                gemv_t_cols(
                    r0, r1, 0, n, alpha, A, lda, x, part.data() + c * n);
            }
        },
        1);
    init_y(0, n);
    for (size_t c = 0; c < chunks; ++c)
        for (size_t j = 0; j < n; ++j)
            y[j] += part[c * n + j];
}

// ---- level 1 --------------------------------------------------------------

template <class T>
T
dot(size_t n, const T *x, const T *y)
{
    using P = simd::Pack<T>;
    auto kernel = [x, y](size_t b, size_t e) {
        constexpr size_t W = P::W;
        auto c0 = P::zero(), c1 = P::zero(), c2 = P::zero(), c3 = P::zero();
        size_t i = b;
        for (; i + 4 * W <= e; i += 4 * W) {
            c0 = P::fmadd(P::loadu(x + i), P::loadu(y + i), c0);
            c1 = P::fmadd(P::loadu(x + i + W), P::loadu(y + i + W), c1);
            c2 = P::fmadd(P::loadu(x + i + 2 * W), P::loadu(y + i + 2 * W), c2);
            c3 = P::fmadd(P::loadu(x + i + 3 * W), P::loadu(y + i + 3 * W), c3);
        }
        T s = P::hsum(P::add(P::add(c0, c1), P::add(c2, c3)));
        for (; i < e; ++i)
            s += x[i] * y[i];
        return s;
    };
    return detail::parallel_sum<T>(n, size_t(1) << 15, kernel);
}

// y += a x
template <class T>
void
axpy(size_t n, T a, const T *x, T *y)
{
    using P = simd::Pack<T>;
    detail::parallel_for(n, size_t(1) << 15, [=](size_t b, size_t e) {
        constexpr size_t W = P::W;
        const auto va = P::set1(a);
        size_t i = b;
        for (; i + W <= e; i += W)
            P::storeu(y + i, P::fmadd(va, P::loadu(x + i), P::loadu(y + i)));
        for (; i < e; ++i)
            y[i] += a * x[i];
    });
}

// ---- transpose ------------------------------------------------------------

namespace blas_detail {

template <class T>
void
transpose_tile(size_t i0, size_t i1, size_t j0, size_t j1, const T *A,
    size_t lda, T *B, size_t ldb)
{
    size_t i = i0;
#if defined(AXOS_SIMD_AVX)
    if constexpr (std::is_same_v<T, double>) {
        for (; i + 4 <= i1; i += 4) {
            size_t j = j0;
            for (; j + 4 <= j1; j += 4) {
                const double *a = A + i * lda + j;
                const __m256d r0 = _mm256_loadu_pd(a);
                const __m256d r1 = _mm256_loadu_pd(a + lda);
                const __m256d r2 = _mm256_loadu_pd(a + 2 * lda);
                const __m256d r3 = _mm256_loadu_pd(a + 3 * lda);
                const __m256d t0 = _mm256_unpacklo_pd(r0, r1);
                const __m256d t1 = _mm256_unpackhi_pd(r0, r1);
                const __m256d t2 = _mm256_unpacklo_pd(r2, r3);
                const __m256d t3 = _mm256_unpackhi_pd(r2, r3);
                double *b = B + j * ldb + i;
                _mm256_storeu_pd(b, _mm256_permute2f128_pd(t0, t2, 0x20));
                _mm256_storeu_pd(b + ldb, _mm256_permute2f128_pd(t1, t3, 0x20));
                _mm256_storeu_pd(
                    b + 2 * ldb, _mm256_permute2f128_pd(t0, t2, 0x31));
                _mm256_storeu_pd(
                    b + 3 * ldb, _mm256_permute2f128_pd(t1, t3, 0x31));
            }
            for (; j < j1; ++j)
                for (size_t r = 0; r < 4; ++r)
                    B[j * ldb + i + r] = A[(i + r) * lda + j];
        }
    }
#endif
#if defined(AXOS_SIMD_AVX) || defined(AXOS_SIMD_SSE2) || \
    defined(AXOS_SIMD_AVX2) || defined(AXOS_SIMD_AVX512)
    if constexpr (std::is_same_v<T, float>) {
        for (; i + 4 <= i1; i += 4) {
            size_t j = j0;
            for (; j + 4 <= j1; j += 4) {
                const float *a = A + i * lda + j;
                __m128 r0 = _mm_loadu_ps(a), r1 = _mm_loadu_ps(a + lda);
                __m128 r2 = _mm_loadu_ps(a + 2 * lda),
                       r3 = _mm_loadu_ps(a + 3 * lda);
                _MM_TRANSPOSE4_PS(r0, r1, r2, r3);
                float *b = B + j * ldb + i;
                _mm_storeu_ps(b, r0);
                _mm_storeu_ps(b + ldb, r1);
                _mm_storeu_ps(b + 2 * ldb, r2);
                _mm_storeu_ps(b + 3 * ldb, r3);
            }
            for (; j < j1; ++j)
                for (size_t r = 0; r < 4; ++r)
                    B[j * ldb + i + r] = A[(i + r) * lda + j];
        }
    }
#endif
    for (; i < i1; ++i)
        for (size_t j = j0; j < j1; ++j)
            B[j * ldb + i] = A[i * lda + j];
}

} // namespace blas_detail

// B (n x m, leading dimension ldb) = A^T, A is m x n. Cache-blocked 32 x 32
// tiles, in parallel over tile rows; 4 x 4 register transposes inside.
template <class T>
void
transpose(size_t m, size_t n, const T *A, size_t lda, T *B, size_t ldb)
{
    constexpr size_t TB = 32;
    if (m == 0 || n == 0) return;
    const size_t tiles_i = (m + TB - 1) / TB;
    const size_t grain = std::max<size_t>(1, (size_t(1) << 15) / (TB * n));
    detail::parallel_for(
        tiles_i, grain,
        [&](size_t b, size_t e) {
            for (size_t ti = b; ti < e; ++ti) {
                const size_t i0 = ti * TB, i1 = std::min(m, i0 + TB);
                for (size_t j0 = 0; j0 < n; j0 += TB)
                    blas_detail::transpose_tile(
                        i0, i1, j0, std::min(n, j0 + TB), A, lda, B, ldb);
            }
        },
        1);
}

} // namespace kernels
} // namespace AXOS
