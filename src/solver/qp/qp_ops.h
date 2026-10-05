// SPDX-License-Identifier: BSD-3-Clause
//
// Per-element kernels of the first-order QP solvers. One definition serves
// both backends: the CPU backend calls them inside OpenMP loops, the GPU
// backend compiles this header at run time with NVRTC and wraps each
// function in a grid-stride kernel (qp_kernels.cuh). Plain C++: no standard
// headers (on the host, <cmath> for sqrt), no other library calls.
//
//   map ops       void f(idx i, const Args &a)             writes outputs
//   reduce ops    void f(idx i, const Args &a, double *acc) adds into acc[0..K)
//                 (Args::K accumulators; bit k of Args::kMax set = max, else
//                 sum; max-accumulators only see non-negative values, start at
//                 0)
//   epilogues     void f(idx i, double s, const Args &a)   s = (M v)_i, run
//                 for every row i of a sparse product (backend spmv_epi)
//
// Notation (HPR-QP, Chen, Sun, Yuan, Zhang, Zhao 2025): state u = (y, w, x),
// T(u) = (ybar, wbar, xbar); qw = Q w and aty = A^T y are carried along.
#pragma once

#if defined(__CUDACC_RTC__) || defined(__CUDACC__)
#define QP_HD __host__ __device__
#else
#define QP_HD
#endif

#if !defined(__CUDACC_RTC__)
#include <cmath>
#endif

namespace axos_qp {

typedef long long idx;
#if !defined(__CUDACC_RTC__)
using std::sqrt;
#endif

QP_HD inline double
dmin(double a, double b)
{ return a < b ? a : b; }
QP_HD inline double
dmax(double a, double b)
{ return a > b ? a : b; }
QP_HD inline double
dabs(double a)
{ return a < 0 ? -a : a; }
QP_HD inline double
clampd(double v, double lo, double hi)
{ return v < lo ? lo : (v > hi ? hi : v); }

// ---- generic --------------------------------------------------------------

struct Fill {
    double *dst;
    double v;
};
QP_HD inline void
fill(idx i, const Fill &a)
{ a.dst[i] = a.v; }

struct Copy {
    double *dst;
    const double *src;
};
QP_HD inline void
copy(idx i, const Copy &a)
{ a.dst[i] = a.src[i]; }

// dst = s * src
struct Scale {
    double *dst;
    const double *src;
    double s;
};
QP_HD inline void
scale(idx i, const Scale &a)
{ a.dst[i] = a.s * a.src[i]; }

// dst = src / sqrt(*nrm2), with nrm2 on the device (power method)
struct ScaleNorm {
    double *dst;
    const double *src, *nrm2;
};
QP_HD inline void
scale_norm(idx i, const ScaleNorm &a)
{ a.dst[i] = a.src[i] / sqrt(*a.nrm2); }

// out = a - b
struct Diff {
    double *out;
    const double *a, *b;
};
QP_HD inline void
diff(idx i, const Diff &p)
{ p.out[i] = p.a[i] - p.b[i]; }

// sum a*b, sum a*a
struct Dot2 {
    const double *a, *b;
    static constexpr int K = 2;
    static constexpr unsigned kMax = 0;
};
QP_HD inline void
dot2(idx i, const Dot2 &p, double *acc)
{
    acc[0] += p.a[i] * p.b[i];
    acc[1] += p.a[i] * p.a[i];
}

// ---- HPR-QP iteration
// --------------------------------------------------------- The scalars of a
// block of iterations live in device memory (written by hpr_scalars once per
// block), so a block's kernels have fixed arguments and can be replayed as a
// CUDA graph: sc[kSigma .. kInvRho] and, for the block's iteration j (0 <= j <
// kMaxBlock), the Halpern weights sc[kW + 2j] = (t+1)/(t+2) and sc[kW + 2j + 1]
// = 1/(t+2) with t = t0 + j. No division is left in the per-element steps (FP64
// division is slow on consumer GPUs, whose FP64 rate is 1/32 to 1/64 of FP32).
//
// Most steps are epilogues of a sparse product, f(i, s, a) with s = (M v)_i, so
// an iteration is four kernels (Q xhat, A v, A^T ybar, Q d) plus, at the start
// of a block, one map (hpr_primal).
enum HprScalar {
    kSigma = 0,
    kA = 1,
    kBq = 2,
    kRho = 3,
    kKappa = 4,
    kInvRho = 5,
    kW = 8,
    kMaxBlock = 64,
    kScalars = kW + 2 * kMaxBlock
};

// Factors of the two sGS half steps of w: whalf = a w + b xhat, and
// wbar = whalf + kappa A^T(ybar - y). With the semi-proximal term s (lq Q -
// Q^2) they are scalars (a = s lq / (1 + s lq), b = 1 / (1 + s lq), kappa = s /
// (1 + s lq), in sc[]). A diagonal Q needs no proximal term: the w step is
// exact, with lq replaced by q_i per element; then qb holds b_i = 1 / (1 + s
// q_i) (hpr_qb, whenever s changes), a_i = 1 - b_i and kappa_i = s b_i.
QP_HD inline void
hpr_factors(const double *sc, const double *qb, idx i, double &a, double &b,
    double &kappa)
{
    if (qb) {
        b = qb[i];
        a = 1.0 - b;
        kappa = sc[kSigma] * b;
    } else {
        a = sc[kA];
        b = sc[kBq];
        kappa = sc[kKappa];
    }
}

QP_HD inline void
halpern_weights(const double *sc, int j, double &w1, double &w0)
{
    w1 = sc[kW + 2 * j];
    w0 = sc[kW + 2 * j + 1];
}

// The scalars of the block that starts at iteration t0 (map over kMaxBlock).
struct HprScalars {
    double *sc;
    double sigma, a, b, rho, kappa, t0;
};
QP_HD inline void
hpr_scalars(idx i, const HprScalars &p)
{
    const double t = p.t0 + static_cast<double>(i);
    p.sc[kW + 2 * i] = (t + 1.0) / (t + 2.0);
    p.sc[kW + 2 * i + 1] = 1.0 / (t + 2.0);
    if (i == 0) {
        p.sc[kSigma] = p.sigma;
        p.sc[kA] = p.a;
        p.sc[kBq] = p.b;
        p.sc[kRho] = p.rho;
        p.sc[kKappa] = p.kappa;
        p.sc[kInvRho] = 1.0 / p.rho;
    }
}

// qb = 1 / (1 + s q) for a diagonal Q = diag(q)
struct HprQb {
    double *qb;
    const double *q;
    double sigma;
};
QP_HD inline void
hpr_qb(idx i, const HprQb &a)
{ a.qb[i] = 1.0 / (1.0 + a.sigma * a.q[i]); }

// plain product: y_i = s
struct Store {
    double *y;
};
QP_HD inline void
store(idx i, double s, const Store &a)
{ a.y[i] = s; }

// z and x: r = x + s (A^T y - Q w - c), xbar = P_C(r), xhat = 2 xbar - x,
// zbar = (xbar - r) / s (only when zbar is set: the KKT check needs it).
QP_HD inline void
hpr_primal_at(idx i, double x, double qw, double aty, const double *c,
    const double *lb, const double *ub, double sigma, double *xbar,
    double *xhat, double *zbar)
{
    const double r = x + sigma * (aty - qw - c[i]);
    const double xb = clampd(r, lb[i], ub[i]);
    xbar[i] = xb;
    xhat[i] = 2.0 * xb - x;
    if (zbar) zbar[i] = (xb - r) / sigma;
}
struct HprPrimal {
    const double *x, *qw, *aty, *c, *lb, *ub;
    double *xbar, *xhat, *zbar;
    const double *sc;
};
QP_HD inline void
hpr_primal(idx i, const HprPrimal &a)
{
    hpr_primal_at(i, a.x[i], a.qw[i], a.aty[i], a.c, a.lb, a.ub, a.sc[kSigma],
        a.xbar, a.xhat, a.zbar);
}

// Epilogue of s = (Q xhat)_i, first sGS half of w (restricted to range(Q)
// through Q w only): whalf = a w + b xhat, Q whalf = a Q w + b s with a, b from
// hpr_factors; v = xhat + s (Q w - Q whalf) is the point the y step multiplies
// by A.
struct HprWHalf {
    const double *w, *qw, *xhat;
    double *whalf, *qwhalf, *v;
    const double *sc, *qb;
};
QP_HD inline void
hpr_whalf(idx i, double s, const HprWHalf &p)
{
    double a, b, kappa;
    hpr_factors(p.sc, p.qb, i, a, b, kappa);
    const double qh = a * p.qw[i] + b * s;
    p.whalf[i] = a * p.w[i] + b * p.xhat[i];
    p.qwhalf[i] = qh;
    p.v[i] = p.xhat[i] + p.sc[kSigma] * (p.qw[i] - qh);
}

// Epilogue of s = (A v)_i, the y step (linearized with the semi-proximal
// term s (la I - A A^T)): r = s - rho y (rho = s la), ybar = (P_K(r) - r) /
// rho; with halpern set, also the Halpern step y <- w1 (2 ybar - y) + w0 y0.
struct HprDual {
    double *y, *ybar;
    const double *l, *u, *y0, *sc;
    int j, halpern;
};
QP_HD inline void
hpr_dual(idx i, double s, const HprDual &a)
{
    const double y = a.y[i];
    const double r = s - a.sc[kRho] * y;
    const double yb = (clampd(r, a.l[i], a.u[i]) - r) * a.sc[kInvRho];
    a.ybar[i] = yb;
    if (a.halpern) {
        double w1, w0;
        halpern_weights(a.sc, a.j, w1, w0);
        a.y[i] = w1 * (2.0 * yb - y) + w0 * a.y0[i];
    }
}

// Epilogue of s = (A^T ybar)_i: atybar = s, d = A^T ybar - A^T y.
struct HprAty {
    const double *aty;
    double *atybar, *d;
};
QP_HD inline void
hpr_aty(idx i, double s, const HprAty &a)
{
    a.atybar[i] = s;
    a.d[i] = s - a.aty[i];
}

// Epilogue of s = (Q d)_i: second sGS half of w (wbar = whalf + kappa d),
// the Halpern step v <- w1 (2 vbar - v) + w0 v0 for v in (x, w, Q w, A^T y),
// and, with next set, the z/x step of the next iteration (same sigma).
struct HprHalpern {
    double *x, *w, *qw, *aty, *xbar, *xhat;
    const double *whalf, *qwhalf, *d, *atybar, *x0, *w0, *qw0, *aty0, *c, *lb,
        *ub, *sc, *qb;
    int j, next;
};
QP_HD inline void
hpr_halpern(idx i, double s, const HprHalpern &p)
{
    double w1, w0, a, b, kappa;
    halpern_weights(p.sc, p.j, w1, w0);
    hpr_factors(p.sc, p.qb, i, a, b, kappa);
    const double wb = p.whalf[i] + kappa * p.d[i];
    const double qwb = p.qwhalf[i] + kappa * s;
    const double x = w1 * (2.0 * p.xbar[i] - p.x[i]) + w0 * p.x0[i];
    const double qw = w1 * (2.0 * qwb - p.qw[i]) + w0 * p.qw0[i];
    const double aty = w1 * (2.0 * p.atybar[i] - p.aty[i]) + w0 * p.aty0[i];
    p.x[i] = x;
    p.w[i] = w1 * (2.0 * wb - p.w[i]) + w0 * p.w0[i];
    p.qw[i] = qw;
    p.aty[i] = aty;
    if (p.next)
        hpr_primal_at(i, x, qw, aty, p.c, p.lb, p.ub, p.sc[kSigma], p.xbar,
            p.xhat, nullptr);
}

// The same step at a convergence check, as a reduction over the n-sized parts
// (qd = Q d stored); also stores wbar, Q wbar (needed at restarts) and
// accumulates the terms of the merit ||T(u) - u||_M, with ex = xbar - x,
// ew = wbar - w, eq = Q ew, d = A^T(ybar - y):
//   acc0 = |ex|^2, acc1 = <ew, eq>, acc2 = <eq, d>, acc3 = <d, Q d>,
//   acc4 = |eq|^2, acc5 = <eq, ex>, acc6 = <d, ex>,
//   acc7 = sum d_i (Q d)_i / (1 + s q_i)   (diagonal Q only)
// (second_half = 0: the free-variable variant below, whose wbar and Q wbar are
// whalf and qwhalf as they come.)
struct HprHalpernRed {
    double *x, *w, *qw, *aty, *wbar, *qwbar;
    const double *xbar, *whalf, *qwhalf, *d, *qd, *atybar, *x0, *w0, *qw0,
        *aty0, *sc, *qb;
    int j, second_half;
    static constexpr int K = 8;
    static constexpr unsigned kMax = 0;
};
QP_HD inline void
hpr_halpern_red(idx i, const HprHalpernRed &p, double *acc)
{
    double w1, w0, a, b, kappa;
    halpern_weights(p.sc, p.j, w1, w0);
    hpr_factors(p.sc, p.qb, i, a, b, kappa);
    if (!p.second_half) kappa = 0.0;
    const double di = p.d[i], qdi = p.qd[i];
    const double wb = p.whalf[i] + kappa * di;
    const double qwb = p.qwhalf[i] + kappa * qdi;
    const double ex = p.xbar[i] - p.x[i], ew = wb - p.w[i], eq = qwb - p.qw[i];
    acc[0] += ex * ex;
    acc[1] += ew * eq;
    acc[2] += eq * di;
    acc[3] += di * qdi;
    acc[4] += eq * eq;
    acc[5] += eq * ex;
    acc[6] += di * ex;
    if (p.qb) acc[7] += di * qdi * b;
    p.wbar[i] = wb;
    p.qwbar[i] = qwb;
    p.x[i] = w1 * (2.0 * p.xbar[i] - p.x[i]) + w0 * p.x0[i];
    p.w[i] = w1 * (2.0 * wb - p.w[i]) + w0 * p.w0[i];
    p.qw[i] = w1 * (2.0 * qwb - p.qw[i]) + w0 * p.qw0[i];
    p.aty[i] = w1 * (2.0 * p.atybar[i] - p.aty[i]) + w0 * p.aty0[i];
}

// ---- HPR-QP, variant for mostly free x --------------------------------------
// When (nearly) no x has bounds, the authors' HPR-QP.jl sweeps w -> x -> y
// without the second sGS half of w: three products per iteration instead of
// four, A^T y (the x and w steps in its epilogue), Q wbar (the Q w update and
// the point the y step multiplies by A) and A v (the y step, hpr_dual).

// Epilogue of s = (A^T y)_i, with the fresh A^T y: z/x step, wbar = a w +
// b xhat, and with halpern set the Halpern steps on x and w.
struct HprFreeXW {
    double *x, *w, *xbar, *xhat, *wbar, *aty;
    const double *qw, *c, *lb, *ub, *x0, *w0, *sc, *qb;
    int j, halpern;
};
QP_HD inline void
hpr_free_xw(idx i, double s, const HprFreeXW &p)
{
    const double x = p.x[i];
    const double r = x + p.sc[kSigma] * (s - p.qw[i] - p.c[i]);
    const double xb = clampd(r, p.lb[i], p.ub[i]);
    const double xh = 2.0 * xb - x;
    double a, b, kappa;
    hpr_factors(p.sc, p.qb, i, a, b, kappa);
    const double wb = a * p.w[i] + b * xh;
    p.aty[i] = s;
    p.xbar[i] = xb;
    p.xhat[i] = xh;
    p.wbar[i] = wb;
    if (p.halpern) {
        double w1, w0;
        halpern_weights(p.sc, p.j, w1, w0);
        p.x[i] = w1 * xh + w0 * p.x0[i];
        p.w[i] = w1 * (2.0 * wb - p.w[i]) + w0 * p.w0[i];
    }
}

// Epilogue of s = (Q wbar)_i: v = xhat + s (Q w - Q wbar), the point of the
// y step; Q wbar stored when qwbar is set; with halpern set the Halpern step
// Q w <- w1 (2 Q wbar - Q w) + w0 Q w0.
struct HprFreeQ {
    double *qw, *v, *qwbar;
    const double *xhat, *qw0, *sc;
    int j, halpern;
};
QP_HD inline void
hpr_free_q(idx i, double s, const HprFreeQ &p)
{
    const double qw = p.qw[i];
    p.v[i] = p.xhat[i] + p.sc[kSigma] * (qw - s);
    if (p.qwbar) p.qwbar[i] = s;
    if (p.halpern) {
        double w1, w0;
        halpern_weights(p.sc, p.j, w1, w0);
        p.qw[i] = w1 * (2.0 * s - qw) + w0 * p.qw0[i];
    }
}

// Halpern step on y at a check; acc0 = |y - ybar|^2.
struct HprHalpernYRed {
    double *y;
    const double *ybar, *y0, *sc;
    int j;
    static constexpr int K = 1;
    static constexpr unsigned kMax = 0;
};
QP_HD inline void
hpr_halpern_y_red(idx i, const HprHalpernYRed &p, double *acc)
{
    double w1, w0;
    halpern_weights(p.sc, p.j, w1, w0);
    const double dy = p.y[i] - p.ybar[i];
    acc[0] += dy * dy;
    p.y[i] = w1 * (2.0 * p.ybar[i] - p.y[i]) + w0 * p.y0[i];
}

// Penalty update terms (differences between T(u) and the epoch's anchor):
//   acc0 = |xbar - x0|^2, acc1 = <wbar - w0, Q(wbar - w0)>,
//   acc2 = <Q(wbar - w0), A^T(ybar - y0)>, acc3 = |Q(wbar - w0)|^2;
//   writes dq = A^T(ybar - y0).
struct HprThetaN {
    const double *xbar, *x0, *wbar, *w0, *qwbar, *qw0, *atybar, *aty0;
    double *dq;
    static constexpr int K = 4;
    static constexpr unsigned kMax = 0;
};
QP_HD inline void
hpr_theta_n(idx i, const HprThetaN &p, double *acc)
{
    const double dx = p.xbar[i] - p.x0[i], dw = p.wbar[i] - p.w0[i];
    const double dqw = p.qwbar[i] - p.qw0[i], da = p.atybar[i] - p.aty0[i];
    acc[0] += dx * dx;
    acc[1] += dw * dqw;
    acc[2] += dqw * da;
    acc[3] += dqw * dqw;
    p.dq[i] = da;
}

// acc0 = |a - b|^2
struct DiffSq {
    const double *a, *b;
    static constexpr int K = 1;
    static constexpr unsigned kMax = 0;
};
QP_HD inline void
diff_sq(idx i, const DiffSq &p, double *acc)
{
    const double d = p.a[i] - p.b[i];
    acc[0] += d * d;
}

// ---- KKT residuals at (xbar, ybar, zbar), in the original scale -----------
// rows: acc0 = max |Ax - P_K(Ax)| / e, acc1 = max |Ax| / e,
//       acc2 = sum of ybar_i * (l_i if ybar_i > 0 else u_i)
struct KktRows {
    const double *ax, *y, *l, *u, *rinv;
    static constexpr int K = 3;
    static constexpr unsigned kMax = 3;
};
QP_HD inline void
kkt_rows(idx i, const KktRows &a, double *acc)
{
    const double ax = a.ax[i];
    acc[0] = dmax(acc[0], dabs(ax - clampd(ax, a.l[i], a.u[i])) * a.rinv[i]);
    acc[1] = dmax(acc[1], dabs(ax) * a.rinv[i]);
    const double y = a.y[i];
    acc[2] += y > 0 ? a.l[i] * y : (y < 0 ? a.u[i] * y : 0.0);
}

// columns: acc0 = max |Q x + c - A^T y - z| / d, acc1 = max |Q x| / d,
//          acc2 = max |A^T y| / d, acc3 = x^T Q x, acc4 = c^T x,
//          acc5 = sum of z_j * (lb_j if z_j > 0 else ub_j)
struct KktCols {
    const double *x, *qx, *aty, *z, *c, *lb, *ub, *cinv;
    static constexpr int K = 6;
    static constexpr unsigned kMax = 7;
};
QP_HD inline void
kkt_cols(idx i, const KktCols &a, double *acc)
{
    const double ci = a.cinv[i], qx = a.qx[i], aty = a.aty[i], z = a.z[i];
    acc[0] = dmax(acc[0], dabs(qx + a.c[i] - aty - z) * ci);
    acc[1] = dmax(acc[1], dabs(qx) * ci);
    acc[2] = dmax(acc[2], dabs(aty) * ci);
    acc[3] += a.x[i] * qx;
    acc[4] += a.c[i] * a.x[i];
    acc[5] += z > 0 ? a.lb[i] * z : (z < 0 ? a.ub[i] * z : 0.0);
}

// ---- PDHCG iteration
// ------------------------------------------------------------ Primal
// subproblem  min_{x in C} 1/2 x^T Q x + (c - A^T y)^T x + |x - xk|^2 / (2
// tau), gradient (Q + I/tau) x + c - A^T y - xk/tau. The inner iterate xt
// starts at xk; qxt = Q xt is carried along (Q xk + sum of Q s).

// Start of the inner solve: xt = xk, qxt = Q xk, g = Q xk + c - A^T y
// (the gradient at xk), and for CG r = p = -g.
struct PdStart {
    const double *x, *qx, *c, *aty;
    double *xt, *qxt, *g, *r, *pdir;
};
QP_HD inline void
pd_start(idx i, const PdStart &a)
{
    const double g = a.qx[i] + a.c[i] - a.aty[i];
    a.xt[i] = a.x[i];
    a.qxt[i] = a.qx[i];
    a.g[i] = g;
    if (a.r) {
        a.r[i] = -g;
        a.pdir[i] = -g;
    }
}

// Projected (BB) gradient step: xt+ = P_C(xt - alpha g), s = xt+ - xt.
struct PdPgStep {
    double *xt, *s;
    const double *g, *lb, *ub;
    double alpha;
};
QP_HD inline void
pd_pg_step(idx i, const PdPgStep &a)
{
    const double xn = clampd(a.xt[i] - a.alpha * a.g[i], a.lb[i], a.ub[i]);
    a.s[i] = xn - a.xt[i];
    a.xt[i] = xn;
}

// After qs = Q s: g += qs + s / tau, qxt += qs;
// acc0 = s.s, acc1 = s.(Q + I/tau)s, acc2 = max|s|, acc3 = max|xt|
struct PdPgUpdate {
    double *g, *qxt;
    const double *s, *qs, *xt;
    double inv_tau;
    static constexpr int K = 4;
    static constexpr unsigned kMax = 12;
};
QP_HD inline void
pd_pg_update(idx i, const PdPgUpdate &a, double *acc)
{
    const double s = a.s[i], hs = a.qs[i] + s * a.inv_tau;
    a.g[i] += hs;
    a.qxt[i] += a.qs[i];
    acc[0] += s * s;
    acc[1] += s * hs;
    acc[2] = dmax(acc[2], dabs(s));
    acc[3] = dmax(acc[3], dabs(a.xt[i]));
}

// CG (no bounds): after qp = Q p: hp = qp + p / tau stored in hp;
// acc0 = p.hp
struct PdCgCurv {
    const double *p, *qp;
    double *hp;
    double inv_tau;
    static constexpr int K = 1;
    static constexpr unsigned kMax = 0;
};
QP_HD inline void
pd_cg_curv(idx i, const PdCgCurv &a, double *acc)
{
    const double h = a.qp[i] + a.p[i] * a.inv_tau;
    a.hp[i] = h;
    acc[0] += a.p[i] * h;
}

// xt += alpha p, qxt += alpha qp, r -= alpha hp;
// acc0 = r.r, acc1 = max|alpha p|, acc2 = max|xt|
struct PdCgStep {
    double *xt, *qxt, *r;
    const double *p, *qp, *hp;
    double alpha;
    static constexpr int K = 3;
    static constexpr unsigned kMax = 6;
};
QP_HD inline void
pd_cg_step(idx i, const PdCgStep &a, double *acc)
{
    const double dx = a.alpha * a.p[i];
    a.xt[i] += dx;
    a.qxt[i] += a.alpha * a.qp[i];
    const double r = a.r[i] - a.alpha * a.hp[i];
    a.r[i] = r;
    acc[0] += r * r;
    acc[1] = dmax(acc[1], dabs(dx));
    acc[2] = dmax(acc[2], dabs(a.xt[i]));
}

// p = r + beta p
struct PdCgDir {
    double *p;
    const double *r;
    double beta;
};
QP_HD inline void
pd_cg_dir(idx i, const PdCgDir &a)
{ a.p[i] = a.r[i] + a.beta * a.p[i]; }

// Dual step with extrapolation t = 2 A xt - A x (PDLP form):
//   yn = max(0, y + s(l - t)) + min(0, y + s(u - t))
struct PdDual {
    const double *y, *ax, *axn, *l, *u;
    double *yn;
    double sigma;
};
QP_HD inline void
pd_dual(idx i, const PdDual &a)
{
    const double t = 2.0 * a.axn[i] - a.ax[i];
    const double lo = a.y[i] + a.sigma * (a.l[i] - t);
    const double hi = a.y[i] + a.sigma * (a.u[i] - t);
    a.yn[i] = dmax(lo, 0.0) + dmin(hi, 0.0);
}

// Reflected Halpern step v <- w1 (2 vn - v) + w0 v0 on one vector.
struct Halpern {
    double *v;
    const double *vn, *v0;
    double w1, w0c;
};
QP_HD inline void
halpern(idx i, const Halpern &a)
{ a.v[i] = a.w1 * (2.0 * a.vn[i] - a.v[i]) + a.w0c * a.v0[i]; }

// Halpern step on the n-sized PDHCG state (x, Q x, A^T y).
struct PdHalpernN {
    double *x, *qx, *aty;
    const double *xn, *qxn, *atyn, *x0, *qx0, *aty0;
    double w1, w0c;
};
QP_HD inline void
pd_halpern_n(idx i, const PdHalpernN &a)
{
    a.x[i] = a.w1 * (2.0 * a.xn[i] - a.x[i]) + a.w0c * a.x0[i];
    a.qx[i] = a.w1 * (2.0 * a.qxn[i] - a.qx[i]) + a.w0c * a.qx0[i];
    a.aty[i] = a.w1 * (2.0 * a.atyn[i] - a.aty[i]) + a.w0c * a.aty0[i];
}

// KKT columns for a PDHG point: the bound duals z are the projection of
// g = Q x + c - A^T y onto their sign cone (as in evaluate_qp); same
// accumulators as KktCols.
struct PdKktCols {
    const double *x, *qx, *aty, *c, *lb, *ub, *cinv;
    static constexpr int K = 6;
    static constexpr unsigned kMax = 7;
};
QP_HD inline void
pd_kkt_cols(idx i, const PdKktCols &a, double *acc)
{
    const double ci = a.cinv[i], qx = a.qx[i], aty = a.aty[i];
    const double g = qx + a.c[i] - aty;
    const bool lo = a.lb[i] > -1e300, hi = a.ub[i] < 1e300;
    const double z =
        (lo && hi) ? g : (lo ? dmax(g, 0.0) : (hi ? dmin(g, 0.0) : 0.0));
    acc[0] = dmax(acc[0], dabs(g - z) * ci);
    acc[1] = dmax(acc[1], dabs(qx) * ci);
    acc[2] = dmax(acc[2], dabs(aty) * ci);
    acc[3] += a.x[i] * qx;
    acc[4] += a.c[i] * a.x[i];
    acc[5] += z > 0 ? a.lb[i] * z : (z < 0 ? a.ub[i] * z : 0.0);
}

} // namespace axos_qp
