// SPDX-License-Identifier: BSD-3-Clause
//
// Elementwise / reduction functors for the interior-point solver (see ipm.h),
// executed by Parallel<Backend> (pdlp_kernels.h / pdlp_cuda.h).
//
// The IPM works on z = (x, w) of length N = n + m, where w = A x are row
// slacks. Bounds are lbz/ubz (concatenated column and row bounds). Per entry:
//   barrier on the lower bound   bl = finite(lb) && lb != ub
//   barrier on the upper bound   bu = finite(ub) && lb != ub
//   fixed                        lb == ub (an equality row's slack)
// tl = z - lb and tu = ub - z are the slacks to the bounds, sl/su their duals.
#pragma once

#include "solver/lp/pdlp_kernels.h"

namespace AXOS {
namespace Solver {
namespace ipm {

using namespace pdlp;

AXOS_HD inline bool bl(double lb, double ub) { return fin(lb) && lb != ub; }
AXOS_HD inline bool bu(double lb, double ub) { return fin(ub) && lb != ub; }
AXOS_HD inline bool fixedv(double lb, double ub) { return lb == ub; }

constexpr double kBig = 1e300;

// rd = c - A^T lam - sl + su (columns);  lam - sl + su (row slacks, 0 if fixed)
struct ResD {
    double *rd;
    const double *c, *aty, *lam, *sl, *su, *lb, *ub;
    size_t n;
    AXOS_HD void
    operator()(size_t j) const
    {
        if (j < n) {
            rd[j] = c[j] - aty[j] - sl[j] + su[j];
        } else {
            rd[j] = fixedv(lb[j], ub[j]) ? 0.0 : lam[j - n] - sl[j] + su[j];
        }
    }
};

// rp = w - A x
struct ResP {
    double *rp;
    const double *w, *ax;
    AXOS_HD void operator()(size_t i) const { rp[i] = w[i] - ax[i]; }
};

// |rd|^2, c^T x, dual objective, sum of complementarity products, count.
struct Stats {
    const double *rd, *z, *c, *lam, *tl, *tu, *sl, *su, *lb, *ub;
    size_t n;
    static constexpr int K = 5;
    AXOS_HD void
    operator()(size_t j, double *acc) const
    {
        acc[0] += rd[j] * rd[j];
        if (j < n) acc[1] += c[j] * z[j];
        if (bl(lb[j], ub[j])) {
            acc[2] += lb[j] * sl[j];
            acc[3] += tl[j] * sl[j];
            acc[4] += 1.0;
        }
        if (bu(lb[j], ub[j])) {
            acc[2] -= ub[j] * su[j];
            acc[3] += tu[j] * su[j];
            acc[4] += 1.0;
        }
        if (j >= n && fixedv(lb[j], ub[j])) acc[2] += lam[j - n] * lb[j];
    }
};

struct SqNorm {
    const double *a;
    static constexpr int K = 1;
    AXOS_HD void
    operator()(size_t i, double *acc) const { acc[0] += a[i] * a[i]; }
};

// Theta^{-1} = sl/tl + su/tu  (huge for fixed entries)
struct BuildTheta {
    double *hz;
    const double *tl, *tu, *sl, *su, *lb, *ub;
    AXOS_HD void
    operator()(size_t j) const
    {
        if (fixedv(lb[j], ub[j])) { hz[j] = kBig; return; }
        double h = 0.0;
        if (bl(lb[j], ub[j])) h += sl[j] / tl[j];
        if (bu(lb[j], ub[j])) h += su[j] / tu[j];
        hz[j] = h;
    }
};

// KKT diagonal: -(Theta_x^{-1} + rho) for columns, Theta_w + delta for rows.
struct ScatterKkt {
    double *vals;
    double *thw;
    const int *dxpos, *dwpos;
    const double *hz;
    double rho, delta;
    size_t n;
    AXOS_HD void
    operator()(size_t j) const
    {
        if (j < n) {
            vals[dxpos[j]] = -(hz[j] + rho);
        } else {
            const size_t i = j - n;
            const double th = hz[j] > 0.0 ? 1.0 / hz[j] : 1e12;
            thw[i] = th;
            vals[dwpos[i]] = th + delta;
        }
    }
};

// predictor complementarity right-hand sides (mu = 0)
struct CompAff {
    double *rcl, *rcu;
    const double *tl, *tu, *sl, *su, *lb, *ub;
    AXOS_HD void
    operator()(size_t j) const
    {
        rcl[j] = bl(lb[j], ub[j]) ? -tl[j] * sl[j] : 0.0;
        rcu[j] = bu(lb[j], ub[j]) ? -tu[j] * su[j] : 0.0;
    }
};

// corrector: sigma*mu - t s - (second-order term of the predictor step)
struct CompCorr {
    double *rcl, *rcu;
    const double *tl, *tu, *sl, *su, *lb, *ub, *dz, *dsl, *dsu;
    double sigma_mu;
    AXOS_HD void
    operator()(size_t j) const
    {
        rcl[j] = bl(lb[j], ub[j])
                     ? sigma_mu - tl[j] * sl[j] - dz[j] * dsl[j] : 0.0;
        rcu[j] = bu(lb[j], ub[j])
                     ? sigma_mu - tu[j] * su[j] + dz[j] * dsu[j] : 0.0;
    }
};

// g = -rd + rcl/tl - rcu/tu
struct BuildG {
    double *g;
    const double *rd, *rcl, *rcu, *tl, *tu, *lb, *ub;
    AXOS_HD void
    operator()(size_t j) const
    {
        double v = -rd[j];
        if (bl(lb[j], ub[j])) v += rcl[j] / tl[j];
        if (bu(lb[j], ub[j])) v -= rcu[j] / tu[j];
        g[j] = v;
    }
};

// reduced KKT right-hand side over (n + m): [-g_x ; rp + Theta_w g_w]
struct PackRhs {
    double *rhs;
    const double *g, *rp, *thw;
    size_t n;
    AXOS_HD void
    operator()(size_t j) const
    {
        if (j < n) rhs[j] = -g[j];
        else rhs[j] = rp[j - n] + thw[j - n] * g[j];
    }
};

// directions from the KKT solution
struct Unpack {
    double *dz, *dsl, *dsu, *dlam;
    const double *sol, *g, *thw, *rcl, *rcu, *tl, *tu, *sl, *su, *lb, *ub;
    size_t n;
    AXOS_HD void
    operator()(size_t j) const
    {
        double d;
        if (j < n) {
            d = sol[j];
        } else {
            const size_t i = j - n;
            dlam[i] = sol[n + i];
            d = fixedv(lb[j], ub[j]) ? 0.0 : thw[i] * (g[j] - sol[n + i]);
        }
        dz[j] = d;
        dsl[j] = bl(lb[j], ub[j]) ? (rcl[j] - sl[j] * d) / tl[j] : 0.0;
        dsu[j] = bu(lb[j], ub[j]) ? (rcu[j] + su[j] * d) / tu[j] : 0.0;
    }
};

// largest primal step keeping tl, tu > 0
struct StepP {
    const double *tl, *tu, *dz, *lb, *ub;
    AXOS_HD double
    operator()(size_t j) const
    {
        double v = 1e300;
        if (bl(lb[j], ub[j]) && dz[j] < 0) v = dmin(v, tl[j] / -dz[j]);
        if (bu(lb[j], ub[j]) && dz[j] > 0) v = dmin(v, tu[j] / dz[j]);
        return v;
    }
};

// largest dual step keeping sl, su > 0
struct StepD {
    const double *sl, *su, *dsl, *dsu, *lb, *ub;
    AXOS_HD double
    operator()(size_t j) const
    {
        double v = 1e300;
        if (bl(lb[j], ub[j]) && dsl[j] < 0) v = dmin(v, sl[j] / -dsl[j]);
        if (bu(lb[j], ub[j]) && dsu[j] < 0) v = dmin(v, su[j] / -dsu[j]);
        return v;
    }
};

// sum of complementarity products after a trial step
struct MuAff {
    const double *tl, *tu, *sl, *su, *dz, *dsl, *dsu, *lb, *ub;
    double ap, ad;
    static constexpr int K = 1;
    AXOS_HD void
    operator()(size_t j, double *acc) const
    {
        if (bl(lb[j], ub[j]))
            acc[0] += (tl[j] + ap * dz[j]) * (sl[j] + ad * dsl[j]);
        if (bu(lb[j], ub[j]))
            acc[0] += (tu[j] - ap * dz[j]) * (su[j] + ad * dsu[j]);
    }
};

struct UpdatePrimalDual {
    double *z, *tl, *tu, *sl, *su;
    const double *dz, *dsl, *dsu, *lb, *ub;
    double ap, ad;
    AXOS_HD void
    operator()(size_t j) const
    {
        const double zj = z[j] + ap * dz[j];
        z[j] = zj;
        if (bl(lb[j], ub[j])) {
            tl[j] = dmax(zj - lb[j], 1e-300);
            sl[j] += ad * dsl[j];
        }
        if (bu(lb[j], ub[j])) {
            tu[j] = dmax(ub[j] - zj, 1e-300);
            su[j] += ad * dsu[j];
        }
    }
};

// Keeps every complementarity pair centered: mu/(kappa t) <= s <= kappa mu/t
// (the safeguard used by Ipopt), so products cannot collapse or explode
// relative to mu.
struct Recenter {
    double *sl, *su;
    const double *tl, *tu, *lb, *ub;
    double mu, kappa;
    AXOS_HD void
    operator()(size_t j) const
    {
        if (bl(lb[j], ub[j]))
            sl[j] = clampd(sl[j], mu / (kappa * tl[j]), kappa * mu / tl[j]);
        if (bu(lb[j], ub[j]))
            su[j] = clampd(su[j], mu / (kappa * tu[j]), kappa * mu / tu[j]);
    }
};

struct Axpy {
    double *y;
    const double *x;
    double a;
    AXOS_HD void operator()(size_t i) const { y[i] += a * x[i]; }
};

// residual of the unregularized reduced KKT system
//   [-Theta_x^{-1}  A^T; A  Theta_w] v = rhs
struct RefineRes {
    double *res;
    const double *rhs, *v, *tmpn, *tmpm, *hz, *thw;
    size_t n;
    AXOS_HD void
    operator()(size_t j) const
    {
        if (j < n) res[j] = rhs[j] - (-hz[j] * v[j] + tmpn[j]);
        else res[j] = rhs[j] - (tmpm[j - n] + thw[j - n] * v[j]);
    }
};

} // namespace ipm
} // namespace Solver
} // namespace AXOS
