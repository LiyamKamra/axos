// SPDX-License-Identifier: BSD-3-Clause
//
// Expression-template core (docs/TENSOR_SPEC.md §6, §15.1).
//
// Expr<T, E> is the CRTP base of every element-wise expression. The library's
// nodes (tensorMath.h) and host tensors derive from it; user expressions may
// too. The requirements on an expression type E are:
//
//   T value(size_t i) const      element with row-major flat index i
//   size_t size(int d) const     extent of dimension d
//
// and, optionally (detected at compile time):
//
//   static constexpr int rank        checked against the target's rank
//   bool contiguous() const          false when value(i) is not the logical
//                                    element i (strided views); lvalue(i) is
//                                    then used instead
//   T lvalue(size_t i) const         element with logical index i
//   void prepare() const             called once before evaluation
//   bool assign_into(T *dst) const   evaluate straight into a contiguous
//                                    destination; return false to decline
//   static constexpr int cost        relative cost per element (default 1),
//                                    sets the threshold for threading
//   static constexpr bool packet_ok  with `template <class P> typename P::reg
//                                    packet(size_t i) const`: SIMD evaluation
//                                    of W = P::W elements starting at i
//
// Nodes hold tensors by reference and other nodes by value, so building an
// expression allocates nothing, and assigning it is one pass over the data
// (split across threads above a size threshold) with no temporaries.
#pragma once

#include "storage/host_storage.h"
#include "tensor/parallel.h"
#include "tensor/simd.h"
#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <stdexcept>
#include <string>
#include <type_traits>
#include <utility>

namespace AXOS {

template <typename T, typename E> struct Expr {
    using value_type = T;
    const E &
    self() const noexcept
    { return static_cast<const E &>(*this); }
};

template <int DIM, typename T, typename Storage> class tensorET;

namespace detail {

template <class E> struct is_tensor : std::false_type {};
template <int D, class T, class S>
struct is_tensor<tensorET<D, T, S>> : std::true_type {};
template <class E> inline constexpr bool is_tensor_v = is_tensor<E>::value;

// How a node stores a child: tensors by reference, expressions by value.
template <class E>
using nested_t = std::conditional_t<is_tensor_v<E>, const E &, const E>;

template <class, class = void> struct has_rank : std::false_type {};
template <class E>
struct has_rank<E, std::void_t<decltype(E::rank)>> : std::true_type {};

template <class, class = void> struct has_contiguous : std::false_type {};
template <class E>
struct has_contiguous<E,
    std::void_t<decltype(std::declval<const E &>().contiguous())>>
    : std::true_type {};

template <class, class = void> struct has_lvalue : std::false_type {};
template <class E>
struct has_lvalue<E,
    std::void_t<decltype(std::declval<const E &>().lvalue(size_t(0)))>>
    : std::true_type {};

template <class, class = void> struct has_prepare : std::false_type {};
template <class E>
struct has_prepare<E,
    std::void_t<decltype(std::declval<const E &>().prepare())>>
    : std::true_type {};

template <class, class, class = void>
struct has_assign_into : std::false_type {};
template <class E, class T>
struct has_assign_into<E, T,
    std::void_t<decltype(std::declval<const E &>().assign_into(
        std::declval<T *>()))>> : std::true_type {};

template <class, class = void> struct has_cost : std::false_type {};
template <class E>
struct has_cost<E, std::void_t<decltype(E::cost)>> : std::true_type {};

template <class, class = void> struct has_packet_ok : std::false_type {};
template <class E>
struct has_packet_ok<E, std::void_t<decltype(E::packet_ok)>> : std::true_type {
};

template <class E>
constexpr int
expr_cost()
{
    if constexpr (has_cost<E>::value)
        return E::cost > 0 ? E::cost : 1;
    else
        return 1;
}

// SIMD evaluation is possible for E with element type T.
template <class E, class T>
constexpr bool
expr_packet_ok()
{
    if constexpr (has_packet_ok<E>::value)
        return E::packet_ok && simd::has_simd_v<T>;
    else
        return false;
}

template <class E>
bool
expr_contiguous(const E &e)
{
    if constexpr (has_contiguous<E>::value)
        return e.contiguous();
    else
        return true;
}

template <class E>
void
expr_prepare(const E &e)
{
    if constexpr (has_prepare<E>::value) e.prepare();
}

template <class E>
auto
expr_lvalue(const E &e, size_t i)
{
    if constexpr (has_lvalue<E>::value)
        return e.lvalue(i);
    else
        return e.value(i);
}

// Elements per thread chunk: ~64K cheap operations, fewer for costly ones.
template <class E>
constexpr size_t
expr_grain()
{ return std::max<size_t>(1024, size_t(65536) / size_t(expr_cost<E>())); }

// Outputs at least this large are written with non-temporal stores: they
// would not stay in cache anyway, and streaming skips the read-for-ownership
// of every destination line (a quarter of the traffic of C = A + B).
inline constexpr size_t kStreamBytes = size_t(8) << 20;

template <class T, class E>
AXOS_INLINE void
eval_range(T *d, const E &e, size_t b, size_t en, bool stream = false)
{
    if constexpr (expr_packet_ok<E, T>()) {
        using P = simd::Pack<T>;
        constexpr size_t W = P::W;
        size_t i = b;
        if (stream) {
            constexpr size_t A = W * sizeof(T); // stream stores need alignment
            for (; i < en && reinterpret_cast<std::uintptr_t>(d + i) % A; ++i)
                d[i] = e.value(i);
            for (; i + W <= en; i += W)
                P::stream(d + i, e.template packet<P>(i));
            simd::stream_fence();
            for (; i < en; ++i)
                d[i] = e.value(i);
            return;
        }
        for (; i + 2 * W <= en; i += 2 * W) {
            const auto v0 = e.template packet<P>(i);
            const auto v1 = e.template packet<P>(i + W);
            P::storeu(d + i, v0);
            P::storeu(d + i + W, v1);
        }
        for (; i + W <= en; i += W)
            P::storeu(d + i, e.template packet<P>(i));
        for (; i < en; ++i)
            d[i] = e.value(i);
    } else {
        for (size_t i = b; i < en; ++i)
            d[i] = e.value(i);
    }
}

template <class Dst, class E>
void
check_expr_shape(const Dst &dst, const E &e)
{
    for (int d = 0; d < Dst::rank; ++d)
        if (e.size(d) != dst.size(d))
            throw std::runtime_error("tensorET = expression: shape mismatch "
                                     "in dimension " +
                                     std::to_string(d) + " (" +
                                     std::to_string(dst.size(d)) + " vs " +
                                     std::to_string(e.size(d)) + ")");
}

// dst = e, element-wise, into dst's existing buffer.
template <class Dst, class E>
void
assign_expr(Dst &dst, const E &e)
{
    using T = typename Dst::value_type;
    static_assert(Dst::on_host, "expressions are evaluated on the host only");
    if constexpr (has_rank<E>::value)
        static_assert(
            E::rank == Dst::rank, "tensorET = expression: rank mismatch");
    check_expr_shape(dst, e);
    const size_t n = dst.size();
    if (n == 0) return;
    const bool dc = dst.is_contiguous();
    if constexpr (has_assign_into<E, T>::value)
        if (dc && e.assign_into(dst.data)) return;
    expr_prepare(e);
    T *d = dst.data;
    if (dc && expr_contiguous(e)) {
        const bool stream = n * sizeof(T) >= kStreamBytes;
        parallel_for(n, expr_grain<E>(), [d, &e, stream](size_t b, size_t en) {
            eval_range(d, e, b, en, stream);
        });
    } else {
        parallel_for(n, expr_grain<E>(), [&](size_t b, size_t en) {
            for (size_t i = b; i < en; ++i)
                d[dc ? i : logical_offset(dst, i)] = expr_lvalue(e, i);
        });
    }
}

} // namespace detail
} // namespace AXOS
