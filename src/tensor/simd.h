// SPDX-License-Identifier: BSD-3-Clause
//
// Minimal SIMD layer for the dense kernels. simd::Pack<T> holds W lanes of T
// in one register and provides the handful of operations the kernels need.
// The instruction set is chosen at compile time (no runtime dispatch):
//
//   AVX-512F      Pack<double>: 8 lanes, Pack<float>: 16 lanes
//   AVX2 + FMA    4 / 8 lanes   (MSVC: /arch:AVX2 implies FMA)
//   SSE2          2 / 4 lanes   (fmadd is mul + add)
//   otherwise     W == 1, plain scalar code
//
// Other element types always get the scalar Pack. Under nvcc the vector
// paths are disabled, so headers that include this compile in .cu files.
#pragma once

#include <cmath>
#include <cstddef>

#if !defined(__CUDACC__) && !defined(AXOS_NO_SIMD)
#if defined(__AVX512F__)
#define AXOS_SIMD_AVX512 1
#elif defined(__AVX2__) && (defined(__FMA__) || defined(_MSC_VER))
#define AXOS_SIMD_AVX2 1
#elif defined(__SSE2__) || defined(_M_X64) ||                                  \
    (defined(_M_IX86_FP) && _M_IX86_FP >= 2)
#define AXOS_SIMD_SSE2 1
#endif
#if defined(__AVX__)
#define AXOS_SIMD_AVX 1 // 256-bit float ops (used by the transpose kernel)
#endif
#if defined(AXOS_SIMD_AVX512) || defined(AXOS_SIMD_AVX2) ||                    \
    defined(AXOS_SIMD_SSE2) || defined(AXOS_SIMD_AVX)
#include <immintrin.h>
#endif
#endif

#if defined(_MSC_VER)
#define AXOS_INLINE __forceinline
#define AXOS_RESTRICT __restrict
#else
#define AXOS_INLINE inline __attribute__((always_inline))
#define AXOS_RESTRICT __restrict__
#endif

// Prefetch the cache line holding p into L1 (no-op where unavailable).
#if defined(AXOS_SIMD_AVX512) || defined(AXOS_SIMD_AVX2) ||                    \
    defined(AXOS_SIMD_SSE2) || defined(AXOS_SIMD_AVX)
#define AXOS_PREFETCH(p)                                                       \
    _mm_prefetch(reinterpret_cast<const char *>(p), _MM_HINT_T0)
#define AXOS_PREFETCH_L2(p)                                                    \
    _mm_prefetch(reinterpret_cast<const char *>(p), _MM_HINT_T1)
#elif defined(__GNUC__) && !defined(__CUDACC__)
#define AXOS_PREFETCH(p) __builtin_prefetch(p, 0, 3)
#define AXOS_PREFETCH_L2(p) __builtin_prefetch(p, 0, 2)
#else
#define AXOS_PREFETCH(p) ((void)(p))
#define AXOS_PREFETCH_L2(p) ((void)(p))
#endif

namespace AXOS {
namespace simd {

// Scalar fallback (W == 1). Also used for every non-float/double T.
template <class T> struct Pack {
    using reg = T;
    static constexpr int W = 1;
    static AXOS_INLINE reg load(const T *p) { return *p; }
    static AXOS_INLINE reg loadu(const T *p) { return *p; }
    static AXOS_INLINE void store(T *p, reg v) { *p = v; }
    static AXOS_INLINE void storeu(T *p, reg v) { *p = v; }
    static AXOS_INLINE void stream(T *p, reg v) { *p = v; }
    static AXOS_INLINE reg set1(T v) { return v; }
    static AXOS_INLINE reg zero() { return T(0); }
    static AXOS_INLINE reg add(reg a, reg b) { return a + b; }
    static AXOS_INLINE reg sub(reg a, reg b) { return a - b; }
    static AXOS_INLINE reg mul(reg a, reg b) { return a * b; }
    static AXOS_INLINE reg div(reg a, reg b) { return a / b; }
    static AXOS_INLINE reg fmadd(reg a, reg b, reg c) { return a * b + c; }
    static AXOS_INLINE reg sqrt(reg a) { return std::sqrt(a); }
    static AXOS_INLINE reg abs(reg a) { return std::abs(a); }
    static AXOS_INLINE reg neg(reg a) { return -a; }
    static AXOS_INLINE T hsum(reg a) { return a; }
};

#if defined(AXOS_SIMD_AVX512)
template <> struct Pack<double> {
    using reg = __m512d;
    static constexpr int W = 8;
    static AXOS_INLINE reg load(const double *p) { return _mm512_load_pd(p); }
    static AXOS_INLINE reg loadu(const double *p) { return _mm512_loadu_pd(p); }
    static AXOS_INLINE void store(double *p, reg v) { _mm512_store_pd(p, v); }
    static AXOS_INLINE void storeu(double *p, reg v) { _mm512_storeu_pd(p, v); }
    static AXOS_INLINE void stream(double *p, reg v) { _mm512_stream_pd(p, v); }
    static AXOS_INLINE reg set1(double v) { return _mm512_set1_pd(v); }
    static AXOS_INLINE reg zero() { return _mm512_setzero_pd(); }
    static AXOS_INLINE reg add(reg a, reg b) { return _mm512_add_pd(a, b); }
    static AXOS_INLINE reg sub(reg a, reg b) { return _mm512_sub_pd(a, b); }
    static AXOS_INLINE reg mul(reg a, reg b) { return _mm512_mul_pd(a, b); }
    static AXOS_INLINE reg div(reg a, reg b) { return _mm512_div_pd(a, b); }
    static AXOS_INLINE reg fmadd(reg a, reg b, reg c) { return _mm512_fmadd_pd(a, b, c); }
    static AXOS_INLINE reg sqrt(reg a) { return _mm512_sqrt_pd(a); }
    static AXOS_INLINE reg abs(reg a)
    {
        return _mm512_castsi512_pd(_mm512_and_si512(_mm512_castpd_si512(a),
            _mm512_set1_epi64(0x7fffffffffffffffLL)));
    }
    static AXOS_INLINE reg neg(reg a)
    {
        return _mm512_castsi512_pd(_mm512_xor_si512(_mm512_castpd_si512(a),
            _mm512_set1_epi64(static_cast<long long>(0x8000000000000000ULL))));
    }
    static AXOS_INLINE double hsum(reg a) { return _mm512_reduce_add_pd(a); }
};
template <> struct Pack<float> {
    using reg = __m512;
    static constexpr int W = 16;
    static AXOS_INLINE reg load(const float *p) { return _mm512_load_ps(p); }
    static AXOS_INLINE reg loadu(const float *p) { return _mm512_loadu_ps(p); }
    static AXOS_INLINE void store(float *p, reg v) { _mm512_store_ps(p, v); }
    static AXOS_INLINE void storeu(float *p, reg v) { _mm512_storeu_ps(p, v); }
    static AXOS_INLINE void stream(float *p, reg v) { _mm512_stream_ps(p, v); }
    static AXOS_INLINE reg set1(float v) { return _mm512_set1_ps(v); }
    static AXOS_INLINE reg zero() { return _mm512_setzero_ps(); }
    static AXOS_INLINE reg add(reg a, reg b) { return _mm512_add_ps(a, b); }
    static AXOS_INLINE reg sub(reg a, reg b) { return _mm512_sub_ps(a, b); }
    static AXOS_INLINE reg mul(reg a, reg b) { return _mm512_mul_ps(a, b); }
    static AXOS_INLINE reg div(reg a, reg b) { return _mm512_div_ps(a, b); }
    static AXOS_INLINE reg fmadd(reg a, reg b, reg c) { return _mm512_fmadd_ps(a, b, c); }
    static AXOS_INLINE reg sqrt(reg a) { return _mm512_sqrt_ps(a); }
    static AXOS_INLINE reg abs(reg a)
    {
        return _mm512_castsi512_ps(_mm512_and_si512(_mm512_castps_si512(a),
            _mm512_set1_epi32(0x7fffffff)));
    }
    static AXOS_INLINE reg neg(reg a)
    {
        return _mm512_castsi512_ps(_mm512_xor_si512(_mm512_castps_si512(a),
            _mm512_set1_epi32(static_cast<int>(0x80000000u))));
    }
    static AXOS_INLINE float hsum(reg a) { return _mm512_reduce_add_ps(a); }
};
#elif defined(AXOS_SIMD_AVX2)
template <> struct Pack<double> {
    using reg = __m256d;
    static constexpr int W = 4;
    static AXOS_INLINE reg load(const double *p) { return _mm256_load_pd(p); }
    static AXOS_INLINE reg loadu(const double *p) { return _mm256_loadu_pd(p); }
    static AXOS_INLINE void store(double *p, reg v) { _mm256_store_pd(p, v); }
    static AXOS_INLINE void storeu(double *p, reg v) { _mm256_storeu_pd(p, v); }
    static AXOS_INLINE void stream(double *p, reg v) { _mm256_stream_pd(p, v); }
    static AXOS_INLINE reg set1(double v) { return _mm256_set1_pd(v); }
    static AXOS_INLINE reg zero() { return _mm256_setzero_pd(); }
    static AXOS_INLINE reg add(reg a, reg b) { return _mm256_add_pd(a, b); }
    static AXOS_INLINE reg sub(reg a, reg b) { return _mm256_sub_pd(a, b); }
    static AXOS_INLINE reg mul(reg a, reg b) { return _mm256_mul_pd(a, b); }
    static AXOS_INLINE reg div(reg a, reg b) { return _mm256_div_pd(a, b); }
    static AXOS_INLINE reg fmadd(reg a, reg b, reg c) { return _mm256_fmadd_pd(a, b, c); }
    static AXOS_INLINE reg sqrt(reg a) { return _mm256_sqrt_pd(a); }
    static AXOS_INLINE reg abs(reg a) { return _mm256_andnot_pd(_mm256_set1_pd(-0.0), a); }
    static AXOS_INLINE reg neg(reg a) { return _mm256_xor_pd(a, _mm256_set1_pd(-0.0)); }
    static AXOS_INLINE double
    hsum(reg a)
    {
        __m128d s = _mm_add_pd(_mm256_castpd256_pd128(a), _mm256_extractf128_pd(a, 1));
        return _mm_cvtsd_f64(_mm_add_sd(s, _mm_unpackhi_pd(s, s)));
    }
};
template <> struct Pack<float> {
    using reg = __m256;
    static constexpr int W = 8;
    static AXOS_INLINE reg load(const float *p) { return _mm256_load_ps(p); }
    static AXOS_INLINE reg loadu(const float *p) { return _mm256_loadu_ps(p); }
    static AXOS_INLINE void store(float *p, reg v) { _mm256_store_ps(p, v); }
    static AXOS_INLINE void storeu(float *p, reg v) { _mm256_storeu_ps(p, v); }
    static AXOS_INLINE void stream(float *p, reg v) { _mm256_stream_ps(p, v); }
    static AXOS_INLINE reg set1(float v) { return _mm256_set1_ps(v); }
    static AXOS_INLINE reg zero() { return _mm256_setzero_ps(); }
    static AXOS_INLINE reg add(reg a, reg b) { return _mm256_add_ps(a, b); }
    static AXOS_INLINE reg sub(reg a, reg b) { return _mm256_sub_ps(a, b); }
    static AXOS_INLINE reg mul(reg a, reg b) { return _mm256_mul_ps(a, b); }
    static AXOS_INLINE reg div(reg a, reg b) { return _mm256_div_ps(a, b); }
    static AXOS_INLINE reg fmadd(reg a, reg b, reg c) { return _mm256_fmadd_ps(a, b, c); }
    static AXOS_INLINE reg sqrt(reg a) { return _mm256_sqrt_ps(a); }
    static AXOS_INLINE reg abs(reg a) { return _mm256_andnot_ps(_mm256_set1_ps(-0.0f), a); }
    static AXOS_INLINE reg neg(reg a) { return _mm256_xor_ps(a, _mm256_set1_ps(-0.0f)); }
    static AXOS_INLINE float
    hsum(reg a)
    {
        __m128 s = _mm_add_ps(_mm256_castps256_ps128(a), _mm256_extractf128_ps(a, 1));
        s = _mm_add_ps(s, _mm_movehl_ps(s, s));
        return _mm_cvtss_f32(_mm_add_ss(s, _mm_shuffle_ps(s, s, 1)));
    }
};
#elif defined(AXOS_SIMD_SSE2)
template <> struct Pack<double> {
    using reg = __m128d;
    static constexpr int W = 2;
    static AXOS_INLINE reg load(const double *p) { return _mm_load_pd(p); }
    static AXOS_INLINE reg loadu(const double *p) { return _mm_loadu_pd(p); }
    static AXOS_INLINE void store(double *p, reg v) { _mm_store_pd(p, v); }
    static AXOS_INLINE void storeu(double *p, reg v) { _mm_storeu_pd(p, v); }
    static AXOS_INLINE void stream(double *p, reg v) { _mm_stream_pd(p, v); }
    static AXOS_INLINE reg set1(double v) { return _mm_set1_pd(v); }
    static AXOS_INLINE reg zero() { return _mm_setzero_pd(); }
    static AXOS_INLINE reg add(reg a, reg b) { return _mm_add_pd(a, b); }
    static AXOS_INLINE reg sub(reg a, reg b) { return _mm_sub_pd(a, b); }
    static AXOS_INLINE reg mul(reg a, reg b) { return _mm_mul_pd(a, b); }
    static AXOS_INLINE reg div(reg a, reg b) { return _mm_div_pd(a, b); }
    static AXOS_INLINE reg fmadd(reg a, reg b, reg c) { return _mm_add_pd(_mm_mul_pd(a, b), c); }
    static AXOS_INLINE reg sqrt(reg a) { return _mm_sqrt_pd(a); }
    static AXOS_INLINE reg abs(reg a) { return _mm_andnot_pd(_mm_set1_pd(-0.0), a); }
    static AXOS_INLINE reg neg(reg a) { return _mm_xor_pd(a, _mm_set1_pd(-0.0)); }
    static AXOS_INLINE double hsum(reg a) { return _mm_cvtsd_f64(_mm_add_sd(a, _mm_unpackhi_pd(a, a))); }
};
template <> struct Pack<float> {
    using reg = __m128;
    static constexpr int W = 4;
    static AXOS_INLINE reg load(const float *p) { return _mm_load_ps(p); }
    static AXOS_INLINE reg loadu(const float *p) { return _mm_loadu_ps(p); }
    static AXOS_INLINE void store(float *p, reg v) { _mm_store_ps(p, v); }
    static AXOS_INLINE void storeu(float *p, reg v) { _mm_storeu_ps(p, v); }
    static AXOS_INLINE void stream(float *p, reg v) { _mm_stream_ps(p, v); }
    static AXOS_INLINE reg set1(float v) { return _mm_set1_ps(v); }
    static AXOS_INLINE reg zero() { return _mm_setzero_ps(); }
    static AXOS_INLINE reg add(reg a, reg b) { return _mm_add_ps(a, b); }
    static AXOS_INLINE reg sub(reg a, reg b) { return _mm_sub_ps(a, b); }
    static AXOS_INLINE reg mul(reg a, reg b) { return _mm_mul_ps(a, b); }
    static AXOS_INLINE reg div(reg a, reg b) { return _mm_div_ps(a, b); }
    static AXOS_INLINE reg fmadd(reg a, reg b, reg c) { return _mm_add_ps(_mm_mul_ps(a, b), c); }
    static AXOS_INLINE reg sqrt(reg a) { return _mm_sqrt_ps(a); }
    static AXOS_INLINE reg abs(reg a) { return _mm_andnot_ps(_mm_set1_ps(-0.0f), a); }
    static AXOS_INLINE reg neg(reg a) { return _mm_xor_ps(a, _mm_set1_ps(-0.0f)); }
    static AXOS_INLINE float
    hsum(reg a)
    {
        __m128 s = _mm_add_ps(a, _mm_movehl_ps(a, a));
        return _mm_cvtss_f32(_mm_add_ss(s, _mm_shuffle_ps(s, s, 1)));
    }
};
#endif

// Orders earlier non-temporal (stream) stores before later stores.
AXOS_INLINE void
stream_fence()
{
#if defined(AXOS_SIMD_AVX512) || defined(AXOS_SIMD_AVX2) || defined(AXOS_SIMD_SSE2)
    _mm_sfence();
#endif
}

// True when T has real vector lanes on this build.
template <class T>
inline constexpr bool has_simd_v = Pack<T>::W > 1;

} // namespace simd
} // namespace AXOS
