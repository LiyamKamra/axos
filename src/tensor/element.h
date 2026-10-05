// SPDX-License-Identifier: BSD-3-Clause
//
// Element types accepted by tensorET (docs/TENSOR_SPEC.md §10): float,
// double, int32_t, int64_t, std::complex<float|double> and a half-precision
// type. AXOS::half is the compiler's _Float16 where it exists, CUDA's __half
// under nvcc, and otherwise half_fallback (IEEE binary16 stored as bits, with
// arithmetic done in float). Other types are rejected at compile time;
// specialize is_tensor_element to allow more.
#pragma once

#include <cmath>
#include <complex>
#include <cstdint>
#include <cstring>
#include <type_traits>

#if defined(__CUDACC__)
#include <cuda_fp16.h>
#endif

namespace AXOS {

struct half_fallback {
    std::uint16_t bits = 0;

    half_fallback() = default;
    half_fallback(float f) : bits(from_float(f)) {}
    template <class S,
        std::enable_if_t<std::is_arithmetic_v<S> && !std::is_same_v<S, float>,
            int> = 0>
    half_fallback(S v) : bits(from_float(static_cast<float>(v)))
    {
    }
    operator float() const { return to_float(bits); }

    static half_fallback
    from_bits(std::uint16_t b)
    {
        half_fallback h;
        h.bits = b;
        return h;
    }

    // Round to nearest, ties to even; overflow gives inf, NaN stays NaN.
    static std::uint16_t
    from_float(float f)
    {
        std::uint32_t x;
        std::memcpy(&x, &f, 4);
        const std::uint32_t sign = (x >> 16) & 0x8000u;
        x &= 0x7fffffffu;
        if (x >= 0x7f800000u) // inf or NaN
            return static_cast<std::uint16_t>(
                sign | 0x7c00u | (x > 0x7f800000u ? 0x200u : 0u));
        if (x >= 0x477ff000u) // rounds to >= 65520: overflow
            return static_cast<std::uint16_t>(sign | 0x7c00u);
        if (x < 0x38800000u) { // below the smallest normal half: subnormal
            if (x < 0x33000000u) return static_cast<std::uint16_t>(sign);
            const std::uint32_t e = x >> 23;
            const std::uint32_t m = (x & 0x7fffffu) | 0x800000u;
            const std::uint32_t shift = 126 - e; // 14..24
            std::uint32_t r = m >> shift;
            const std::uint32_t rem = m & ((1u << shift) - 1);
            const std::uint32_t halfway = 1u << (shift - 1);
            if (rem > halfway || (rem == halfway && (r & 1u))) ++r;
            return static_cast<std::uint16_t>(sign | r);
        }
        std::uint32_t r = ((x >> 13) - (112u << 10));
        const std::uint32_t rem = x & 0x1fffu;
        if (rem > 0x1000u || (rem == 0x1000u && (r & 1u))) ++r;
        return static_cast<std::uint16_t>(sign | r);
    }

    static float
    to_float(std::uint16_t h)
    {
        const std::uint32_t sign = static_cast<std::uint32_t>(h & 0x8000u)
                                   << 16;
        std::uint32_t e = (h >> 10) & 0x1fu, m = h & 0x3ffu, x;
        if (e == 0x1f) {
            x = sign | 0x7f800000u | (m << 13);
        } else if (e != 0) {
            x = sign | ((e + 112u) << 23) | (m << 13);
        } else if (m == 0) {
            x = sign;
        } else { // subnormal: normalize
            e = 113;
            while (!(m & 0x400u)) {
                m <<= 1;
                --e;
            }
            x = sign | (e << 23) | ((m & 0x3ffu) << 13);
        }
        float f;
        std::memcpy(&f, &x, 4);
        return f;
    }

#define AXOS_HALF_OP(op) \
    friend half_fallback operator op(half_fallback a, half_fallback b) \
    { return half_fallback(float(a) op float(b)); } \
    template <class S, std::enable_if_t<std::is_arithmetic_v<S>, int> = 0> \
    friend half_fallback operator op(half_fallback a, S b) \
    { return half_fallback(float(a) op static_cast<float>(b)); } \
    template <class S, std::enable_if_t<std::is_arithmetic_v<S>, int> = 0> \
    friend half_fallback operator op(S a, half_fallback b) \
    { return half_fallback(static_cast<float>(a) op float(b)); } \
    half_fallback &operator op## = (half_fallback b) \
    { return *this = *this op b; }
    AXOS_HALF_OP(+)
    AXOS_HALF_OP(-)
    AXOS_HALF_OP(*)
    AXOS_HALF_OP(/)
#undef AXOS_HALF_OP

    half_fallback
    operator-() const
    { return from_bits(bits ^ 0x8000u); }

#define AXOS_HALF_CMP(op) \
    friend bool operator op(half_fallback a, half_fallback b) \
    { return float(a) op float(b); }
    AXOS_HALF_CMP(==)
    AXOS_HALF_CMP(!=)
    AXOS_HALF_CMP(<)
    AXOS_HALF_CMP(<=)
    AXOS_HALF_CMP(>)
    AXOS_HALF_CMP(>=)
#undef AXOS_HALF_CMP
};

#if defined(__CUDACC__)
using half = __half;
#elif defined(__FLT16_MAX__)
#define AXOS_HAS_FLOAT16 1
using half = _Float16;
#else
using half = half_fallback;
#endif

template <class T> struct is_half : std::false_type {};
template <> struct is_half<half_fallback> : std::true_type {};
#if defined(AXOS_HAS_FLOAT16)
template <> struct is_half<_Float16> : std::true_type {};
#endif
#if defined(__CUDACC__)
template <> struct is_half<__half> : std::true_type {};
#endif
template <class T> inline constexpr bool is_half_v = is_half<T>::value;

template <class T> struct is_complex : std::false_type {};
template <class R> struct is_complex<std::complex<R>> : std::true_type {};
template <class T> inline constexpr bool is_complex_v = is_complex<T>::value;

template <class T> struct is_tensor_element : std::false_type {};
template <> struct is_tensor_element<float> : std::true_type {};
template <> struct is_tensor_element<double> : std::true_type {};
template <> struct is_tensor_element<std::int32_t> : std::true_type {};
template <> struct is_tensor_element<std::int64_t> : std::true_type {};
template <> struct is_tensor_element<std::complex<float>> : std::true_type {};
template <> struct is_tensor_element<std::complex<double>> : std::true_type {};
template <> struct is_tensor_element<half_fallback> : std::true_type {};
#if defined(AXOS_HAS_FLOAT16)
template <> struct is_tensor_element<_Float16> : std::true_type {};
#endif
#if defined(__CUDACC__)
template <> struct is_tensor_element<__half> : std::true_type {};
#endif
template <class T>
inline constexpr bool is_tensor_element_v = is_tensor_element<T>::value;

// Real type underlying T (complex<R> -> R, otherwise T).
template <class T> struct real_type {
    using type = T;
};
template <class R> struct real_type<std::complex<R>> {
    using type = R;
};
template <class T> using real_type_t = typename real_type<T>::type;

// Floating-point types the dense linear algebra works on.
template <class T>
inline constexpr bool is_la_scalar_v =
    std::is_same_v<T, float> || std::is_same_v<T, double> ||
    std::is_same_v<T, std::complex<float>> ||
    std::is_same_v<T, std::complex<double>>;

} // namespace AXOS
