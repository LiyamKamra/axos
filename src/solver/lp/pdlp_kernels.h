// SPDX-License-Identifier: BSD-3-Clause
//
// Elementwise / reduction kernels for the PDLP solver, written once as
// host+device functors and executed by Parallel<Backend>:
//
//   for_each(n, f)             f(i)
//   reduce<K>(slot, n, f)      f(i, double acc[K]) accumulates K sums
//   fetch()                    returns the accumulated slots on the host
//
// Parallel<Cpu::Backend> is OpenMP; Parallel<Cuda::Backend> lives in
// pdlp_cuda.h.
#pragma once

#include "storage/host_storage.h"
#include <cmath>
#include <cstddef>
#include <cstring>

#if defined(__CUDACC__)
#define AXOS_HD __host__ __device__
#else
#define AXOS_HD
#endif

namespace AXOS {
namespace Solver {
namespace pdlp {

inline constexpr int kSlots = 24;
// Below this many elements OpenMP thread launch costs more than it saves.
inline constexpr size_t kParallelMin = 16384;

AXOS_HD inline double dmin(double a, double b) { return a < b ? a : b; }
AXOS_HD inline double dmax(double a, double b) { return a > b ? a : b; }
AXOS_HD inline bool fin(double v) { return v > -1e300 && v < 1e300; }
AXOS_HD inline double
clampd(double v, double lo, double hi)
{
    return v < lo ? lo : (v > hi ? hi : v);
}

// x+ = proj_[lb,ub](x - tau (c - A^T y))
struct XUpdate {
    const double *c, *aty, *lb, *ub, *x;
    double *xn;
    double tau;
    AXOS_HD void
    operator()(size_t i) const
    {
        xn[i] = clampd(x[i] - tau * (c[i] - aty[i]), lb[i], ub[i]);
    }
};

// y+ = max(0, y + s (l - t)) + min(0, y + s (u - t)),  t = 2 A x+ - A x
struct YUpdate {
    const double *l, *u, *y, *ax, *axn;
    double *yn;
    double sigma;
    AXOS_HD void
    operator()(size_t i) const
    {
        const double t = 2.0 * axn[i] - ax[i];
        const double a = y[i] + sigma * (l[i] - t);
        const double b = y[i] + sigma * (u[i] - t);
        yn[i] = dmax(a, 0.0) + dmin(b, 0.0);
    }
};

// Fused x update and ||dx||^2 (one pass over the columns).
struct XUpdateStep {
    const double *c, *aty, *lb, *ub, *x;
    double *xn;
    double tau;
    static constexpr int K = 1;
    AXOS_HD void
    operator()(size_t i, double *acc) const
    {
        const double v = clampd(x[i] - tau * (c[i] - aty[i]), lb[i], ub[i]);
        xn[i] = v;
        const double d = v - x[i];
        acc[0] += d * d;
    }
};

// Fused y update, ||dy||^2 and dy^T (A dx) (one pass over the rows).
struct YUpdateStep {
    const double *l, *u, *y, *ax, *axn;
    double *yn;
    double sigma;
    static constexpr int K = 2;
    AXOS_HD void
    operator()(size_t i, double *acc) const
    {
        const double t = 2.0 * axn[i] - ax[i];
        const double a = y[i] + sigma * (l[i] - t);
        const double b = y[i] + sigma * (u[i] - t);
        const double v = dmax(a, 0.0) + dmin(b, 0.0);
        yn[i] = v;
        const double dy = v - y[i];
        acc[0] += dy * dy;
        acc[1] += dy * (axn[i] - ax[i]);
    }
};

// Reflected Halpern step, in place on the iterate z:
//   z = w1 (2 T(z) - z) + w0 anchor,
// where `image` holds T(z) and `anchor` is the last restart point.
struct HalpernMix {
    double *z;
    const double *image, *anchor;
    double w1, w0;
    AXOS_HD void
    operator()(size_t i) const
    {
        z[i] = w1 * (2.0 * image[i] - z[i]) + w0 * anchor[i];
    }
};

// ||dx||^2
struct StepDiffN {
    const double *x, *xn;
    static constexpr int K = 1;
    AXOS_HD void
    operator()(size_t i, double *acc) const
    {
        const double d = xn[i] - x[i];
        acc[0] += d * d;
    }
};

// ||dy||^2 and dy^T (A dx)
struct StepDiffM {
    const double *y, *yn, *ax, *axn;
    static constexpr int K = 2;
    AXOS_HD void
    operator()(size_t i, double *acc) const
    {
        const double dy = yn[i] - y[i];
        acc[0] += dy * dy;
        acc[1] += dy * (axn[i] - ax[i]);
    }
};

// running weighted sums of the iterates
struct AvgN {
    double eta;
    const double *xn, *atyn;
    double *sx, *saty;
    AXOS_HD void
    operator()(size_t i) const
    {
        sx[i] += eta * xn[i];
        saty[i] += eta * atyn[i];
    }
};
struct AvgM {
    double eta;
    const double *yn, *axn;
    double *sy, *sax;
    AXOS_HD void
    operator()(size_t i) const
    {
        sy[i] += eta * yn[i];
        sax[i] += eta * axn[i];
    }
};

// KKT pieces over rows: primal residual (scaled and original scale) and the
// row part of the dual objective.
struct KktM {
    const double *ax, *y, *l, *u, *rinv;
    static constexpr int K = 3;
    AXOS_HD void
    operator()(size_t i, double *acc) const
    {
        const double d = ax[i] - clampd(ax[i], l[i], u[i]);
        const double dorig = d * rinv[i];
        acc[0] += d * d;
        acc[1] += dorig * dorig;
        const double yi = y[i];
        acc[2] += yi > 0 ? l[i] * yi : (yi < 0 ? u[i] * yi : 0.0);
    }
};

// KKT pieces over columns: dual residual (scaled / original), column part of
// the dual objective, and c^T x.
struct KktN {
    const double *x, *aty, *c, *lb, *ub, *cinv;
    static constexpr int K = 4;
    AXOS_HD void
    operator()(size_t i, double *acc) const
    {
        const double z = c[i] - aty[i];
        const double zlo = fin(ub[i]) ? -INFINITY : 0.0;
        const double zhi = fin(lb[i]) ? INFINITY : 0.0;
        const double lam = clampd(z, zlo, zhi);
        const double dr = z - lam;
        const double drorig = dr * cinv[i];
        acc[0] += dr * dr;
        acc[1] += drorig * drorig;
        acc[2] += lam > 0 ? lb[i] * lam : (lam < 0 ? ub[i] * lam : 0.0);
        acc[3] += c[i] * x[i];
    }
};

// Candidate primal ray dx = x - xr (columns): |dx|^2, c^T dx and the
// violation of the recession cone of the variable bounds.
struct PrimalRayN {
    const double *x, *xr, *c, *lb, *ub;
    static constexpr int K = 3;
    AXOS_HD void
    operator()(size_t i, double *acc) const
    {
        const double dx = x[i] - xr[i];
        acc[0] += dx * dx;
        acc[1] += c[i] * dx;
        const bool lf = fin(lb[i]), uf = fin(ub[i]);
        const double v = (lf && !uf) ? dmin(dx, 0.0)
                         : (!lf && uf) ? dmax(dx, 0.0)
                         : (lf && uf) ? dx
                                      : 0.0;
        acc[2] += v * v;
    }
};

// ... and of the row activity: violation of the recession cone of [l, u].
struct PrimalRayM {
    const double *ax, *axr, *l, *u;
    static constexpr int K = 1;
    AXOS_HD void
    operator()(size_t i, double *acc) const
    {
        const double d = ax[i] - axr[i];
        const bool lf = fin(l[i]), uf = fin(u[i]);
        const double v = (lf && !uf) ? dmin(d, 0.0)
                         : (!lf && uf) ? dmax(d, 0.0)
                         : (lf && uf) ? d
                                      : 0.0;
        acc[0] += v * v;
    }
};

// Candidate dual ray dy = y - yr (rows): |dy|^2, violation of the sign cone
// and the row part of the homogeneous dual objective.
struct DualRayM {
    const double *y, *yr, *l, *u;
    static constexpr int K = 3;
    AXOS_HD void
    operator()(size_t i, double *acc) const
    {
        const double dy = y[i] - yr[i];
        acc[0] += dy * dy;
        const bool lf = fin(l[i]), uf = fin(u[i]);
        const double v = (lf && !uf) ? dmin(dy, 0.0)
                         : (!lf && uf) ? dmax(dy, 0.0)
                         : (!lf && !uf) ? dy
                                        : 0.0;
        acc[1] += v * v;
        const double p = (lf && !uf) ? dmax(dy, 0.0)
                         : (!lf && uf) ? dmin(dy, 0.0)
                         : (lf && uf) ? dy
                                      : 0.0;
        acc[2] += p > 0 ? l[i] * p : (p < 0 ? u[i] * p : 0.0);
    }
};

// ... and over columns: reduced-cost violation for c = 0, lambda = -A^T dy.
struct DualRayN {
    const double *aty, *atyr, *lb, *ub;
    static constexpr int K = 2;
    AXOS_HD void
    operator()(size_t i, double *acc) const
    {
        const double lin = -(aty[i] - atyr[i]);
        const double zlo = fin(ub[i]) ? -INFINITY : 0.0;
        const double zhi = fin(lb[i]) ? INFINITY : 0.0;
        const double lam = clampd(lin, zlo, zhi);
        const double dr = lin - lam;
        acc[0] += dr * dr;
        acc[1] += lam > 0 ? lb[i] * lam : (lam < 0 ? ub[i] * lam : 0.0);
    }
};

struct DiffSq {
    const double *a, *b;
    static constexpr int K = 1;
    AXOS_HD void
    operator()(size_t i, double *acc) const
    {
        const double d = a[i] - b[i];
        acc[0] += d * d;
    }
};

// sum over i of (a[i] * w[i])^2
struct WSqNorm {
    const double *a, *w;
    static constexpr int K = 1;
    AXOS_HD void
    operator()(size_t i, double *acc) const
    {
        const double v = a[i] * w[i];
        acc[0] += v * v;
    }
};

struct SqNormN {
    const double *a;
    static constexpr int K = 1;
    AXOS_HD void
    operator()(size_t i, double *acc) const
    {
        acc[0] += a[i] * a[i];
    }
};

struct Copy {
    double *dst;
    const double *src;
    AXOS_HD void operator()(size_t i) const { dst[i] = src[i]; }
};

struct ScaleCopy {
    double *dst;
    const double *src;
    double alpha;
    AXOS_HD void operator()(size_t i) const { dst[i] = alpha * src[i]; }
};

struct Fill {
    double *dst;
    double v;
    AXOS_HD void operator()(size_t i) const { dst[i] = v; }
};

} // namespace pdlp

template <typename Backend> struct Parallel;

template <> struct Parallel<Cpu::Backend> {
    static constexpr size_t kParallelMin = pdlp::kParallelMin;
    double slots[pdlp::kSlots] = {};

    template <typename F>
    void
    for_each(size_t n, F f)
    {
#pragma omp parallel for schedule(static) if (n > kParallelMin)
        for (long i = 0; i < static_cast<long>(n); ++i)
            f(static_cast<size_t>(i));
    }

    void zero() { std::memset(slots, 0, sizeof(slots)); }

    // Adds the K sums of f over [0, n) into slots[slot .. slot+K).
    template <typename F>
    void
    reduce(int slot, size_t n, F f)
    {
        constexpr int K = F::K;
#pragma omp parallel if (n > kParallelMin)
        {
            double loc[K] = {};
#pragma omp for schedule(static) nowait
            for (long i = 0; i < static_cast<long>(n); ++i)
                f(static_cast<size_t>(i), loc);
#pragma omp critical(axos_pdlp_reduce)
            for (int k = 0; k < K; ++k)
                slots[slot + k] += loc[k];
        }
    }

    void set_slot(int slot, double v) { slots[slot] = v; }

    // slots[slot] = min(slots[slot], min over i of f(i)).
    template <typename F>
    void
    reduce_min(int slot, size_t n, F f)
    {
        double m = slots[slot];
        // per-thread minima (MSVC's OpenMP has no min reduction)
#pragma omp parallel if (n > kParallelMin)
        {
            double loc = m;
#pragma omp for schedule(static) nowait
            for (long i = 0; i < static_cast<long>(n); ++i) {
                const double v = f(static_cast<size_t>(i));
                if (v < loc) loc = v;
            }
#pragma omp critical(axos_pdlp_reduce_min)
            if (loc < m) m = loc;
        }
        slots[slot] = m;
    }

    const double *fetch() { return slots; }

    void
    copy(double *dst, const double *src, size_t n)
    {
        std::memcpy(dst, src, n * sizeof(double));
    }
};

} // namespace Solver
} // namespace AXOS
