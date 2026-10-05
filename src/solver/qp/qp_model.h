// SPDX-License-Identifier: BSD-3-Clause
//
// Convex quadratic programs:
//
//   minimize    1/2 x^T Q x + c^T x + offset
//   subject to  row_lb <= A x <= row_ub
//               col_lb <=  x  <= col_ub
//
// Q is symmetric positive semidefinite, stored as a full (both triangles)
// CSR matrix; the linear part and all bounds are an LpProblem, so every LP
// is a QP with an empty Q. Duals follow the LP convention of model.h:
// y_i >= 0 only at a lower row bound, z_j >= 0 only at a lower column bound,
// and at an optimum  Q x + c - A^T y - z = 0.
//
// evaluate_qp() is the one yardstick used for every solver in the
// benchmarks: the relative KKT residuals of HPR-QP / PDHCG (infinity norms),
//   eta_p   = max(|Ax - P_K(Ax)|, |x - P_C(x)|) / (1 + max(|b|, |Ax|))
//   eta_d   = |Q x + c - A^T y - z| / (1 + max(|c|, |A^T y|, |Q x|))
//   eta_gap = |P - D| / (1 + max(|P|, |D|))
// with b = max(|row_lb|, |row_ub|) over finite entries, P the primal and D
// the dual objective, and z the bound duals that minimize eta_d for the
// given (x, y) (z_j free for a boxed column, of one sign for a one-sided
// column, zero for a free one).
#pragma once

#include "solver/model.h"
#include <algorithm>
#include <cmath>
#include <string>
#include <vector>

namespace AXOS {
namespace Solver {

struct QpProblem {
    LpProblem lp;  // A, c, bounds, offset, names; lp.maximize is reported only
    HostMatrix Q;  // n x n, symmetric, both triangles stored; empty = LP

    size_t rows() const { return lp.rows(); }
    size_t cols() const { return lp.cols(); }
    bool has_quadratic() const { return Q.nnz() > 0; }

    std::string
    validate() const
    {
        std::string e = lp.validate();
        if (!e.empty()) return e;
        if (Q.nnz() > 0 && (Q.rows() != cols() || Q.cols() != cols()))
            return "Q must be cols() x cols()";
        return Q.nnz() > 0 ? Q.validate() : std::string();
    }

    // y = Q x (host)
    void
    q_times(const std::vector<double> &x, std::vector<double> &y) const
    {
        y.assign(cols(), 0.0);
        if (Q.nnz() == 0) return;
        const auto *rp = Q.row_ptr();
        const auto *ci = Q.col_ind();
        const double *v = Q.values();
        for (size_t i = 0; i < cols(); ++i) {
            double s = 0;
            for (int k = rp[i]; k < rp[i + 1]; ++k) s += v[k] * x[ci[k]];
            y[i] = s;
        }
    }

    double
    objective(const std::vector<double> &x) const
    {
        std::vector<double> qx;
        q_times(x, qx);
        double s = lp.offset;
        for (size_t j = 0; j < cols(); ++j) s += x[j] * (0.5 * qx[j] + lp.c[j]);
        return s;
    }
};

struct QpSolution {
    Status status = Status::NotSolved;
    std::vector<double> x, y, z;
    double primal_objective = 0, dual_objective = 0;
    double rel_primal = 0, rel_dual = 0, rel_gap = 0; // evaluate_qp metrics
    long iterations = 0;
    double seconds = 0;       // solve time (after reading the file)
    double setup_seconds = 0; // part of `seconds`: scaling, upload, spectral estimates
    std::string method, device;

    double kkt() const { return std::max({rel_primal, rel_dual, rel_gap}); }
};

// HprQp and Pdhcg: first-order, CPU or GPU; Ipm: interior point, CPU;
// Auto: the interior point when its factorization is cheap enough
// (ipm_auto_flops), else or when it fails HPR-QP (GPU for large problems).
enum class QpMethod { HprQp, Pdhcg, Ipm, Auto };

class QpComm; // MPI ranks of a distributed solve (qp_dist.h)

struct QpOptions {
    QpMethod method = QpMethod::HprQp;
    bool use_gpu = false;
    // With auto_device, solve_qp ignores use_gpu and takes the GPU (when there
    // is one) for problems with at least gpu_min_work = nnz(A) + nnz(Q) + n + m;
    // below that, kernel launch latency outweighs the GPU's bandwidth.
    bool auto_device = false;
    size_t gpu_min_work = 100000;
    // QpMethod::Auto's first-order fallback (after the interior point failed:
    // a hard problem, many iterations) takes the GPU from this size up, where
    // a GPU iteration is cheaper than a CPU one (about even at 3e3 - 1e4,
    // 3x cheaper at 1e4 - 3e4, 9x at 3e4 - 1e5 on the RTX 3050 laptop GPU)
    size_t gpu_min_work_hard = 5000;
    double tol = 1e-6;           // on max(eta_p, eta_d, eta_gap)
    long max_iterations = 1000000000L;
    double time_limit = 3600;    // seconds
    int check_every = 100;       // iterations between restart checks (HPR-QP: also
                                 // termination checks every 10 below 1000 iterations)
    bool scaling = true;
    int ruiz_iterations = 10;
    double pock_chambolle_alpha = 1.0; // < 0 disables
    int power_iterations = 5000;       // spectral estimates of A A^T and Q (converges earlier)
    double ipm_max_flops = 1e12;       // IPM: give up when the KKT factor needs more
    double ipm_auto_flops = 1e10;      // Auto: IPM only below this many flops per factor
    bool ipm_project = false;          // IPM: stop after the first factorization when 25
                                       // factorizations would not fit in time_limit
                                       // (5 without it)
    int verbose = 0;                   // 0 silent, 1 summary, 2 every check
    // HPR-QP sweep order: 0 the sGS order (x, w, y, w); 2 the order for free
    // variables (w, x, y: three products per iteration instead of four);
    // 1 the second when more than 80% of the columns have no bounds.
    int hpr_free_variant = 1;
    bool profile = false; // HPR-QP: time each step of an iteration after the setup,
                          // print the table and stop (the iterates are not a solve)
    // HPR-QP over MPI ranks (qp_dist.h): every rank calls solve_qp with the
    // same problem and options, holds a block of the constraint rows and
    // gets the whole solution. nullptr or one rank: a single process.
    const QpComm *comm = nullptr;
};

// Relative KKT residuals of (x, y) for p (see the header comment); fills a
// QpSolution with x, y, the chosen z and the objectives.
inline QpSolution
evaluate_qp(const QpProblem &p, const std::vector<double> &x,
    const std::vector<double> &y)
{
    const LpProblem &q = p.lp;
    const size_t m = p.rows(), n = p.cols();
    QpSolution s;
    s.x = x;
    s.y = y;
    s.z.assign(n, 0.0);
    // A dual on an infinite row bound is infeasible: drop it (the same
    // clean-up for every solver), so its effect shows in eta_d.
    for (size_t i = 0; i < m; ++i) {
        if (s.y[i] > 0 && !std::isfinite(q.row_lb[i])) s.y[i] = 0;
        if (s.y[i] < 0 && !std::isfinite(q.row_ub[i])) s.y[i] = 0;
    }
    const std::vector<double> &yv = s.y;
    std::vector<double> ax(m, 0.0), aty(n, 0.0), qx;
    const auto *rp = q.A.row_ptr();
    const auto *ci = q.A.col_ind();
    const double *v = q.A.values();
    for (size_t i = 0; i < m; ++i)
        for (int k = rp[i]; k < rp[i + 1]; ++k) {
            ax[i] += v[k] * x[ci[k]];
            aty[ci[k]] += v[k] * yv[i];
        }
    p.q_times(x, qx);

    double bmax = 0, axmax = 0, pviol = 0, dual = q.offset;
    for (size_t i = 0; i < m; ++i) {
        if (std::isfinite(q.row_lb[i])) bmax = std::max(bmax, std::abs(q.row_lb[i]));
        if (std::isfinite(q.row_ub[i])) bmax = std::max(bmax, std::abs(q.row_ub[i]));
        axmax = std::max(axmax, std::abs(ax[i]));
        pviol = std::max({pviol, q.row_lb[i] - ax[i], ax[i] - q.row_ub[i]});
        if (yv[i] > 0) dual += yv[i] * q.row_lb[i];
        else if (yv[i] < 0) dual += yv[i] * q.row_ub[i];
    }
    double cmax = 0, atymax = 0, qxmax = 0, dres = 0, xqx = 0, cx = 0;
    for (size_t j = 0; j < n; ++j) {
        pviol = std::max({pviol, q.col_lb[j] - x[j], x[j] - q.col_ub[j]});
        cmax = std::max(cmax, std::abs(q.c[j]));
        atymax = std::max(atymax, std::abs(aty[j]));
        qxmax = std::max(qxmax, std::abs(qx[j]));
        const double g = qx[j] + q.c[j] - aty[j]; // = z_j at an optimum
        const bool lo = std::isfinite(q.col_lb[j]), hi = std::isfinite(q.col_ub[j]);
        double zj = 0;
        if (lo && hi) zj = g;
        else if (lo) zj = std::max(g, 0.0);
        else if (hi) zj = std::min(g, 0.0);
        s.z[j] = zj;
        dres = std::max(dres, std::abs(g - zj));
        if (zj > 0) dual += zj * q.col_lb[j];
        else if (zj < 0) dual += zj * q.col_ub[j];
        xqx += x[j] * qx[j];
        cx += q.c[j] * x[j];
    }
    s.primal_objective = 0.5 * xqx + cx + q.offset;
    s.dual_objective = dual - 0.5 * xqx;
    s.rel_primal = std::max(pviol, 0.0) / (1 + std::max(bmax, axmax));
    s.rel_dual = dres / (1 + std::max({cmax, atymax, qxmax}));
    s.rel_gap = std::abs(s.primal_objective - s.dual_objective) /
                (1 + std::max(std::abs(s.primal_objective), std::abs(s.dual_objective)));
    return s;
}

} // namespace Solver
} // namespace AXOS
