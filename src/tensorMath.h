// SPDX-License-Identifier: BSD-3-Clause
//
// Element-wise math and matrix products on host tensors
// (docs/TENSOR_SPEC.md §6).
//
// Element-wise operations are lazy (tensor/expr.h): the operators and
// functions below build small expression objects, and assigning one to a
// tensor evaluates the whole chain in a single pass with no temporaries:
//
//   C = exp(sin(A) + cos(A)) * 2.0 + B;
//
//   x + y, x - y, x * y, x / y     element-wise (* is not a matrix product)
//   x op s, s op x                 with a scalar s, for + - * /
//   -x
//   sqrt exp Log sin cos tan sinh cosh tanh fabs
//
// Operands must have the same shape (checked when the node is built).
// Element i of the result depends only on element i of the operands, so
// `A = A + B` is safe. The evaluation runs on all threads above a size
// threshold that shrinks with the cost of the chain; +, -, *, /, sqrt and
// fabs use explicit SIMD, the transcendental functions call the scalar
// std:: versions (so results are bit-identical to the scalar expression).
//
// Matrix products (all accept views, whose row stride is used directly):
//   matMul(A, B)          lazy A (m x k) * B (k x n); value(i) is entry
//                         (i / n, i % n). Assigned directly it runs the fast
//                         GEMM into the target; inside a larger expression the
//                         product is computed once into a cache first. Do not
//                         assign it to one of its operands' views.
//   matMulTile(A, B)      eager product, new m x n tensor (GEMV for a vector)
//   matMulBlocked(A, B, C, m, alpha, bias, beta, transA, transB, accumulate)
//                         C = alpha op(A) op(B) + beta bias, or C += ... with
//                         accumulate; m is a legacy tile-size hint (the GEMM
//                         picks its own blocking)
//   matMulND(A, B)        batched over the leading dimensions (rank >= 3)
//   gemv, matVec          y = alpha op(A) x + beta y
//   transpose(A)          new n x m tensor
//   extractSlice / insertSlice, getValue / setValue
#pragma once

#include "tensor/blas.h"
#include "tensor/expr.h"
#include "tensorET.h"
#include <algorithm>
#include <cmath>
#include <complex>
#include <cstddef>
#include <stdexcept>
#include <string>
#include <type_traits>

namespace AXOS {

namespace detail {

// Rank of a node: that of the first child that declares one.
template <class L, class R = void, class = void> struct rank_from {};
template <class L, class R>
struct rank_from<L, R, std::enable_if_t<has_rank<L>::value>> {
    static constexpr int rank = L::rank;
};
template <class L, class R>
struct rank_from<L, R,
    std::enable_if_t<!has_rank<L>::value && !std::is_void_v<R> &&
                     has_rank<R>::value>> {
    static constexpr int rank = R::rank;
};

// Per-element cost without the >= 1 clamp (tensors cost 0).
template <class E>
constexpr int
raw_cost()
{
    if constexpr (has_cost<E>::value)
        return E::cost;
    else
        return 1;
}

template <class E>
constexpr bool
packet_ok_of()
{
    if constexpr (has_packet_ok<E>::value)
        return E::packet_ok;
    else
        return false;
}

template <class L, class R>
void
check_same_shape(const L &l, const R &r)
{
    if constexpr (has_rank<L>::value && has_rank<R>::value) {
        static_assert(
            L::rank == R::rank, "element-wise operation: rank mismatch");
        for (int d = 0; d < L::rank; ++d)
            if (l.size(d) != r.size(d))
                throw std::runtime_error(
                    "element-wise operation: shape mismatch in dimension " +
                    std::to_string(d) + " (" + std::to_string(l.size(d)) +
                    " vs " + std::to_string(r.size(d)) + ")");
    }
}

template <class T> struct type_identity {
    using type = T;
};
// Parameter types that do not take part in template argument deduction, so
// `x * 2`, `alpha = 0.5` with float tensors, or `bias = nullptr` just work.
template <class T> using scalar_arg_t = typename type_identity<T>::type;

namespace op {

struct Add {
    static constexpr int cost = 1;
    static constexpr bool has_packet = true;
    template <class T>
    static T
    apply(const T &a, const T &b)
    { return a + b; }
    template <class P, class R>
    static AXOS_INLINE R
    packet(R a, R b)
    { return P::add(a, b); }
};
struct Sub {
    static constexpr int cost = 1;
    static constexpr bool has_packet = true;
    template <class T>
    static T
    apply(const T &a, const T &b)
    { return a - b; }
    template <class P, class R>
    static AXOS_INLINE R
    packet(R a, R b)
    { return P::sub(a, b); }
};
struct Mul {
    static constexpr int cost = 1;
    static constexpr bool has_packet = true;
    template <class T>
    static T
    apply(const T &a, const T &b)
    { return a * b; }
    template <class P, class R>
    static AXOS_INLINE R
    packet(R a, R b)
    { return P::mul(a, b); }
};
struct Div {
    static constexpr int cost = 4;
    static constexpr bool has_packet = true;
    template <class T>
    static T
    apply(const T &a, const T &b)
    { return a / b; }
    template <class P, class R>
    static AXOS_INLINE R
    packet(R a, R b)
    { return P::div(a, b); }
};

// Scalar math that also works for half types (through float) and
// integers (through double).
template <class T, class F>
T
via(T x, F f)
{
    if constexpr (is_half_v<T>)
        return T(f(static_cast<float>(x)));
    else if constexpr (std::is_integral_v<T>)
        return static_cast<T>(f(static_cast<double>(x)));
    else
        return f(x);
}

#define AXOS_UNARY_FN(Name, expr, Cost, Packet) \
    struct Name { \
        static constexpr int cost = Cost; \
        static constexpr bool has_packet = Packet; \
        template <class T> \
        static T \
        apply(const T &x) \
        { \
            return via(x, [](auto v) { return expr; }); \
        } \
    };
AXOS_UNARY_FN(Exp, std::exp(v), 20, false)
AXOS_UNARY_FN(Log, std::log(v), 20, false)
AXOS_UNARY_FN(Sin, std::sin(v), 20, false)
AXOS_UNARY_FN(Cos, std::cos(v), 20, false)
AXOS_UNARY_FN(Tan, std::tan(v), 24, false)
AXOS_UNARY_FN(Sinh, std::sinh(v), 24, false)
AXOS_UNARY_FN(Cosh, std::cosh(v), 24, false)
AXOS_UNARY_FN(Tanh, std::tanh(v), 24, false)
#undef AXOS_UNARY_FN

struct Sqrt {
    static constexpr int cost = 4;
    static constexpr bool has_packet = true;
    template <class T>
    static T
    apply(const T &x)
    {
        return via(x, [](auto v) { return std::sqrt(v); });
    }
    template <class P, class R>
    static AXOS_INLINE R
    packet(R a)
    { return P::sqrt(a); }
};
struct Fabs {
    static constexpr int cost = 1;
    static constexpr bool has_packet = true;
    template <class T>
    static T
    apply(const T &x)
    {
        if constexpr (is_complex_v<T>)
            return T(std::abs(x));
        else
            return via(x, [](auto v) { return std::abs(v); });
    }
    template <class P, class R>
    static AXOS_INLINE R
    packet(R a)
    { return P::abs(a); }
};
struct Neg {
    static constexpr int cost = 1;
    static constexpr bool has_packet = true;
    template <class T>
    static T
    apply(const T &x)
    { return -x; }
    template <class P, class R>
    static AXOS_INLINE R
    packet(R a)
    { return P::neg(a); }
};

} // namespace op
} // namespace detail

// ---- expression nodes -----------------------------------------------------

template <class T, class L, class R, class Op>
class BinaryExpr : public Expr<T, BinaryExpr<T, L, R, Op>>,
                   public detail::rank_from<L, R> {
  public:
    static constexpr int cost =
        detail::raw_cost<L>() + detail::raw_cost<R>() + Op::cost;
    static constexpr bool packet_ok = Op::has_packet &&
                                      detail::packet_ok_of<L>() &&
                                      detail::packet_ok_of<R>();

    BinaryExpr(const L &l, const R &r) : l_(l), r_(r)
    { detail::check_same_shape(l, r); }

    T
    value(size_t i) const
    { return Op::apply(T(l_.value(i)), T(r_.value(i))); }
    size_t
    size(int d) const
    { return l_.size(d); }
    bool
    contiguous() const
    { return detail::expr_contiguous(l_) && detail::expr_contiguous(r_); }
    T
    lvalue(size_t i) const
    {
        return Op::apply(
            T(detail::expr_lvalue(l_, i)), T(detail::expr_lvalue(r_, i)));
    }
    void
    prepare() const
    {
        detail::expr_prepare(l_);
        detail::expr_prepare(r_);
    }
    template <class P>
    AXOS_INLINE typename P::reg
    packet(size_t i) const
    {
        return Op::template packet<P>(
            l_.template packet<P>(i), r_.template packet<P>(i));
    }

  private:
    detail::nested_t<L> l_;
    detail::nested_t<R> r_;
};

// Left ? (s op x) : (x op s)
template <class T, class E, class Op, bool Left>
class ScalarExpr : public Expr<T, ScalarExpr<T, E, Op, Left>>,
                   public detail::rank_from<E> {
  public:
    static constexpr int cost = detail::raw_cost<E>() + Op::cost;
    static constexpr bool packet_ok =
        Op::has_packet && detail::packet_ok_of<E>();

    ScalarExpr(const E &e, T s) : e_(e), s_(s) {}

    T
    value(size_t i) const
    { return apply(T(e_.value(i))); }
    size_t
    size(int d) const
    { return e_.size(d); }
    bool
    contiguous() const
    { return detail::expr_contiguous(e_); }
    T
    lvalue(size_t i) const
    { return apply(T(detail::expr_lvalue(e_, i))); }
    void
    prepare() const
    { detail::expr_prepare(e_); }
    template <class P>
    AXOS_INLINE typename P::reg
    packet(size_t i) const
    {
        const auto s = P::set1(s_);
        const auto x = e_.template packet<P>(i);
        if constexpr (Left)
            return Op::template packet<P>(s, x);
        else
            return Op::template packet<P>(x, s);
    }

  private:
    T
    apply(const T &x) const
    {
        if constexpr (Left)
            return Op::apply(s_, x);
        else
            return Op::apply(x, s_);
    }
    detail::nested_t<E> e_;
    T s_;
};

template <class T, class E, class F>
class UnaryExpr : public Expr<T, UnaryExpr<T, E, F>>,
                  public detail::rank_from<E> {
  public:
    static constexpr int cost = detail::raw_cost<E>() + F::cost;
    static constexpr bool packet_ok =
        F::has_packet && detail::packet_ok_of<E>();

    explicit UnaryExpr(const E &e) : e_(e) {}

    T
    value(size_t i) const
    { return F::apply(T(e_.value(i))); }
    size_t
    size(int d) const
    { return e_.size(d); }
    bool
    contiguous() const
    { return detail::expr_contiguous(e_); }
    T
    lvalue(size_t i) const
    { return F::apply(T(detail::expr_lvalue(e_, i))); }
    void
    prepare() const
    { detail::expr_prepare(e_); }
    template <class P>
    AXOS_INLINE typename P::reg
    packet(size_t i) const
    { return F::template packet<P>(e_.template packet<P>(i)); }

  private:
    detail::nested_t<E> e_;
};

// ---- operators ------------------------------------------------------------

#define AXOS_BINARY_OPERATOR(sym, Op) \
    template <class T, class L, class R> \
    BinaryExpr<T, L, R, detail::op::Op> operator sym( \
        const Expr<T, L> &l, const Expr<T, R> &r) \
    { return BinaryExpr<T, L, R, detail::op::Op>(l.self(), r.self()); } \
    template <class T, class L> \
    ScalarExpr<T, L, detail::op::Op, false> operator sym( \
        const Expr<T, L> &l, detail::scalar_arg_t<T> s) \
    { return ScalarExpr<T, L, detail::op::Op, false>(l.self(), s); } \
    template <class T, class R> \
    ScalarExpr<T, R, detail::op::Op, true> operator sym( \
        detail::scalar_arg_t<T> s, const Expr<T, R> &r) \
    { return ScalarExpr<T, R, detail::op::Op, true>(r.self(), s); }
AXOS_BINARY_OPERATOR(+, Add)
AXOS_BINARY_OPERATOR(-, Sub)
AXOS_BINARY_OPERATOR(*, Mul)
AXOS_BINARY_OPERATOR(/, Div)
#undef AXOS_BINARY_OPERATOR

template <class T, class E>
UnaryExpr<T, E, detail::op::Neg>
operator-(const Expr<T, E> &e)
{ return UnaryExpr<T, E, detail::op::Neg>(e.self()); }

#define AXOS_UNARY_FUNCTION(name, F) \
    template <class T, class E> \
    UnaryExpr<T, E, detail::op::F> name(const Expr<T, E> &e) \
    { return UnaryExpr<T, E, detail::op::F>(e.self()); }
AXOS_UNARY_FUNCTION(sqrt, Sqrt)
AXOS_UNARY_FUNCTION(exp, Exp)
AXOS_UNARY_FUNCTION(Log, Log)
AXOS_UNARY_FUNCTION(sin, Sin)
AXOS_UNARY_FUNCTION(cos, Cos)
AXOS_UNARY_FUNCTION(tan, Tan)
AXOS_UNARY_FUNCTION(sinh, Sinh)
AXOS_UNARY_FUNCTION(cosh, Cosh)
AXOS_UNARY_FUNCTION(tanh, Tanh)
AXOS_UNARY_FUNCTION(fabs, Fabs)
#undef AXOS_UNARY_FUNCTION

// ---- matrix products --------------------------------------------------------

namespace detail {

// [first, last) byte range covered by a tensor (views included).
template <class Tn>
std::pair<const char *, const char *>
byte_span(const Tn &t)
{
    using T = typename Tn::value_type;
    if (t.size() == 0 || !t.data) return {nullptr, nullptr};
    size_t last = 0;
    for (int d = 0; d < Tn::rank; ++d)
        last += (t.size(d) - 1) * t.stride(d);
    const char *b = reinterpret_cast<const char *>(t.data);
    return {b, b + (last + 1) * sizeof(T)};
}

template <class A, class B>
bool
overlaps(const A &a, const B &b)
{
    const auto x = byte_span(a), y = byte_span(b);
    return x.first && y.first && x.first < y.second && y.first < x.second;
}

template <class T>
bool
overlaps_ptr(const T *p, size_t n, std::pair<const char *, const char *> s)
{
    if (!p || !s.first || n == 0) return false;
    const char *b = reinterpret_cast<const char *>(p);
    return b < s.second && s.first < b + n * sizeof(T);
}

} // namespace detail

// Lazy matrix product. See the header comment.
template <class T, class SA, class SB>
class MatMulExpr : public Expr<T, MatMulExpr<T, SA, SB>> {
  public:
    using MA = tensorET<2, T, SA>;
    using MB = tensorET<2, T, SB>;
    static constexpr int rank = 2;
    static constexpr int cost = 1;
    static constexpr bool packet_ok = true; // reads the cache (after prepare)

    MatMulExpr(const MA &A, const MB &B) : A_(A), B_(B)
    {
        if (A.size(1) != B.size(0))
            throw std::runtime_error("matMul: inner dimensions differ (" +
                                     std::to_string(A.size(1)) + " vs " +
                                     std::to_string(B.size(0)) + ")");
    }

    size_t
    size(int d) const
    { return d == 0 ? A_.size(0) : B_.size(1); }

    T
    value(size_t i) const
    {
        if (cache_.data) return cache_.data[i];
        const size_t n = B_.size(1), r = i / n, c = i % n;
        T s(0);
        for (size_t p = 0; p < A_.size(1); ++p)
            s += A_(r, p) * B_(p, c);
        return s;
    }

    // Computes the product once into an internal cache.
    void
    prepare() const
    {
        if (cache_.data || A_.size(0) * B_.size(1) == 0) return;
        cache_ = tensorET<2, T>(std::array<size_t, 2>{A_.size(0), B_.size(1)});
        run(cache_.data);
    }

    template <class P>
    AXOS_INLINE typename P::reg
    packet(size_t i) const
    { return P::loadu(cache_.data + i); }

    // Direct evaluation into a contiguous m x n destination, unless it
    // overlaps an operand.
    bool
    assign_into(T *dst) const
    {
        const size_t n = A_.size(0) * B_.size(1);
        if (detail::overlaps_ptr(dst, n, detail::byte_span(A_)) ||
            detail::overlaps_ptr(dst, n, detail::byte_span(B_)))
            return false;
        run(dst);
        return true;
    }

  private:
    void
    run(T *dst) const
    {
        kernels::gemm<T>(false, false, A_.size(0), B_.size(1), A_.size(1), T(1),
            A_.data, A_.stride(0), B_.data, B_.stride(0), T(0), dst,
            B_.size(1));
    }
    const MA &A_;
    const MB &B_;
    mutable tensorET<2, T> cache_;
};

template <class T, class SA, class SB>
MatMulExpr<T, SA, SB>
matMul(const tensorET<2, T, SA> &A, const tensorET<2, T, SB> &B)
{
    static_assert(is_host_storage_v<SA> && is_host_storage_v<SB>,
        "matMul works on host tensors");
    return MatMulExpr<T, SA, SB>(A, B);
}

// y = alpha op(A) x + beta y.
template <class T, class SA, class SX, class SY>
void
gemv(const tensorET<2, T, SA> &A, const tensorET<1, T, SX> &x,
    tensorET<1, T, SY> &y, detail::scalar_arg_t<T> alpha = T(1),
    detail::scalar_arg_t<T> beta = T(0), bool trans = false)
{
    const size_t m = A.size(0), n = A.size(1);
    if (x.size() != (trans ? m : n) || y.size() != (trans ? n : m))
        throw std::runtime_error("gemv: dimension mismatch");
    kernels::gemv<T>(
        trans, m, n, alpha, A.data, A.stride(0), x.data, beta, y.data);
}

template <class T, class SA, class SX>
tensorET<1, T>
matVec(const tensorET<2, T, SA> &A, const tensorET<1, T, SX> &x,
    bool trans = false)
{
    tensorET<1, T> y(trans ? A.size(1) : A.size(0));
    gemv(A, x, y, T(1), T(0), trans);
    return y;
}

// Eager product: m x n matrix (GEMM), or a vector for a vector argument.
template <class T, class SA, class SB>
tensorET<2, T>
matMulTile(const tensorET<2, T, SA> &A, const tensorET<2, T, SB> &B)
{
    const size_t m = A.size(0), k = A.size(1), n = B.size(1);
    if (B.size(0) != k)
        throw std::runtime_error("matMulTile: inner dimensions differ");
    tensorET<2, T> C(std::array<size_t, 2>{m, n});
    if (n == 1 && B.is_contiguous())
        kernels::gemv<T>(
            false, m, k, T(1), A.data, A.stride(0), B.data, T(0), C.data);
    else
        kernels::gemm<T>(false, false, m, n, k, T(1), A.data, A.stride(0),
            B.data, B.stride(0), T(0), C.data, n);
    return C;
}

template <class T, class SA, class SX>
tensorET<1, T>
matMulTile(const tensorET<2, T, SA> &A, const tensorET<1, T, SX> &x)
{ return matVec(A, x); }

// C = alpha op(A) op(B) + beta bias   (accumulate: C += the same).
// C is allocated when empty; otherwise its shape must match. `m` is a
// legacy tile-size hint and is not used.
template <class T, class SA, class SB, class SC>
void
matMulBlocked(const tensorET<2, T, SA> &A, const tensorET<2, T, SB> &B,
    tensorET<2, T, SC> &C, size_t m, detail::scalar_arg_t<T> alpha = T(1),
    const detail::scalar_arg_t<tensorET<2, T>> *bias = nullptr,
    detail::scalar_arg_t<T> beta = T(1), bool transA = false,
    bool transB = false, bool accumulate = false)
{
    static_assert(std::is_same_v<SC, Cpu::HostStorage<T>>,
        "matMulBlocked writes a host matrix");
    (void)m;
    const size_t M = transA ? A.size(1) : A.size(0);
    const size_t K = transA ? A.size(0) : A.size(1);
    const size_t N = transB ? B.size(0) : B.size(1);
    if ((transB ? B.size(1) : B.size(0)) != K)
        throw std::runtime_error("matMulBlocked: inner dimensions differ");
    if (C.size() == 0 && C.data == nullptr) {
        C = tensorET<2, T, SC>(std::array<size_t, 2>{M, N});
        if (accumulate) C.fill(T(0));
    }
    if (C.size(0) != M || C.size(1) != N)
        throw std::runtime_error("matMulBlocked: C has the wrong shape");
    if (bias && (bias->size(0) != M || bias->size(1) != N))
        throw std::runtime_error("matMulBlocked: bias has the wrong shape");
    if (detail::overlaps(C, A) ||
        detail::overlaps(C, B)) { // C aliases an operand
        tensorET<2, T> tmp = C.clone();
        matMulBlocked(
            A, B, tmp, m, alpha, bias, beta, transA, transB, accumulate);
        Cpu::Backend::copy_data(C, tmp);
        return;
    }
    T cbeta = accumulate ? T(1) : T(0);
    if (bias) {
        const size_t ldc = C.stride(0), ldb = bias->stride(0);
        detail::parallel_for(
            M, std::max<size_t>(1, 32768 / std::max<size_t>(N, 1)),
            [&](size_t r0, size_t r1) {
                for (size_t i = r0; i < r1; ++i) {
                    T *c = C.data + i * ldc;
                    const T *b = bias->data + i * ldb;
                    for (size_t j = 0; j < N; ++j)
                        c[j] = accumulate ? c[j] + beta * b[j] : beta * b[j];
                }
            },
            1);
        cbeta = T(1);
    }
    kernels::gemm<T>(transA, transB, M, N, K, alpha, A.data, A.stride(0),
        B.data, B.stride(0), cbeta, C.data, C.stride(0));
}

// Batched product over the leading dimensions (rank >= 3): the last two
// dimensions multiply, the leading ones must match.
template <int R, class T, class SA, class SB>
tensorET<R, T>
matMulND(const tensorET<R, T, SA> &A, const tensorET<R, T, SB> &B)
{
    static_assert(
        R >= 3, "matMulND needs rank >= 3 (use matMulTile for matrices)");
    std::array<size_t, R> shape{};
    size_t batch = 1;
    for (int d = 0; d < R - 2; ++d) {
        if (A.size(d) != B.size(d))
            throw std::runtime_error("matMulND: leading dimensions differ");
        shape[d] = A.size(d);
        batch *= A.size(d);
    }
    const size_t m = A.size(R - 2), k = A.size(R - 1), n = B.size(R - 1);
    if (B.size(R - 2) != k)
        throw std::runtime_error("matMulND: inner dimensions differ");
    if (!A.is_contiguous() || !B.is_contiguous())
        throw std::runtime_error("matMulND: operands must be contiguous");
    shape[R - 2] = m;
    shape[R - 1] = n;
    tensorET<R, T> C(shape);
    auto one = [&](size_t b) {
        kernels::gemm<T>(false, false, m, n, k, T(1), A.data + b * m * k, k,
            B.data + b * k * n, n, T(0), C.data + b * m * n, n);
    };
    if (double(m) * n * k < kernels::blas_detail::kParallelFma && batch > 1)
        detail::parallel_for(
            batch, 1,
            [&](size_t b0, size_t b1) {
                for (size_t b = b0; b < b1; ++b)
                    one(b);
            },
            1);
    else
        for (size_t b = 0; b < batch; ++b)
            one(b);
    return C;
}

// The b-th matrix (last two dimensions) of a rank >= 3 tensor, with the
// leading dimensions flattened in row-major order.
template <int R, class T, class S>
tensorET<2, T>
extractSlice(const tensorET<R, T, S> &t, size_t b)
{
    static_assert(R >= 3, "extractSlice needs rank >= 3");
    const size_t m = t.size(R - 2), n = t.size(R - 1);
    const tensorET<R, T, S> c = t.is_contiguous() ? t : t.clone();
    if (b >= c.size() / std::max<size_t>(1, m * n))
        throw std::runtime_error("extractSlice: index out of range");
    tensorET<2, T> M(std::array<size_t, 2>{m, n});
    std::copy(c.data + b * m * n, c.data + (b + 1) * m * n, M.data);
    return M;
}

template <int R, class T, class S, class SM>
void
insertSlice(tensorET<R, T, S> &t, size_t b, const tensorET<2, T, SM> &M)
{
    static_assert(R >= 3, "insertSlice needs rank >= 3");
    const size_t m = t.size(R - 2), n = t.size(R - 1);
    if (M.size(0) != m || M.size(1) != n)
        throw std::runtime_error("insertSlice: matrix shape does not match");
    if (!t.is_contiguous())
        throw std::runtime_error("insertSlice: target must be contiguous");
    if (b >= t.size() / std::max<size_t>(1, m * n))
        throw std::runtime_error("insertSlice: index out of range");
    for (size_t i = 0; i < m; ++i)
        std::copy(M.data + i * M.stride(0), M.data + i * M.stride(0) + n,
            t.data + b * m * n + i * n);
}

// Element access by an array of DIM indices.
template <int D, class T, class S>
T
getValue(const tensorET<D, T, S> &t, const size_t *idx)
{
    size_t off = 0;
    for (int d = 0; d < D; ++d)
        off += idx[d] * t.stride(d);
    return t[off];
}

template <int D, class T, class S>
void
setValue(tensorET<D, T, S> &t, const size_t *idx, T v)
{
    static_assert(tensorET<D, T, S>::on_host, "setValue works on host tensors");
    size_t off = 0;
    for (int d = 0; d < D; ++d)
        off += idx[d] * t.stride(d);
    t[off] = v;
}

// New n x m tensor holding A^T (cache-blocked, parallel).
template <class T, class S>
tensorET<2, T>
transpose(const tensorET<2, T, S> &A)
{
    static_assert(is_host_storage_v<S>, "transpose works on host tensors");
    tensorET<2, T> B(std::array<size_t, 2>{A.size(1), A.size(0)});
    kernels::transpose<T>(
        A.size(0), A.size(1), A.data, A.stride(0), B.data, A.size(0));
    return B;
}

// Sum of x[i] * y[i] (no conjugation).
template <class T, class SX, class SY>
T
dot(const tensorET<1, T, SX> &x, const tensorET<1, T, SY> &y)
{
    if (x.size() != y.size()) throw std::runtime_error("dot: size mismatch");
    return kernels::dot<T>(x.size(), x.data, y.data);
}

} // namespace AXOS
