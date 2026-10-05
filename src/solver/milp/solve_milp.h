// SPDX-License-Identifier: BSD-3-Clause
//
// solve_milp(problem, options): the AXOS MILP solver.
//
//   1. presolve: the LP presolve (presolve/presolve.h: empty, fixed and
//      singleton rows / columns, duplicate rows), integrality carried to the
//      reduced columns through its column map, integer bounds rounded
//   2. branch and cut on the reduced problem (bnb.h)
//   3. postsolve of the best solution and a feasibility check on the ORIGINAL
//      problem (check_milp): a solution failing it is not reported
//
// Objective values and bounds are in minimization form (io/mps.h negates the
// costs of a maximization problem and sets maximize): negate them for output
// when orig.maximize is set.
#pragma once

#include "solver/milp/bnb.h"
#include "solver/milp/milp_model.h"
#include "solver/presolve/presolve.h"
#include <chrono>
#include <memory>

namespace AXOS {
namespace Solver {

// MIP coefficient tightening (M. Savelsbergh, ORSA J. Computing 6, 1994) on
// one-sided rows a x <= b: when a binary x_k leaves the row redundant at one of
// its values (the largest activity then stays below b by d > 0), its
// coefficient moves toward zero by d (and b by d when a_k > 0). The integer
// points are unchanged; the LP relaxation gets tighter. Returns the number of
// changes.
inline int
tighten_coefficients(LpProblem &p)
{
    const size_t m = p.rows(), n = p.cols();
    std::vector<int32_t> rp(p.A.row_ptr(), p.A.row_ptr() + m + 1);
    std::vector<int32_t> ci(p.A.col_ind(), p.A.col_ind() + p.A.nnz());
    std::vector<double> va(p.A.values(), p.A.values() + p.A.nnz());
    auto fin = [](double v) { return std::abs(v) < 1e20; };
    int changed = 0;
    for (size_t i = 0; i < m; ++i) {
        const bool lf = fin(p.row_lb[i]), uf = fin(p.row_ub[i]);
        if (lf == uf) continue;             // ranged, equality or free
        const double sgn = uf ? 1.0 : -1.0; // sgn a x <= b
        double b = uf ? p.row_ub[i] : -p.row_lb[i];
        double maxact = 0;
        bool finite = true;
        for (int32_t k = rp[i]; k < rp[i + 1] && finite; ++k) {
            const double a = sgn * va[k];
            const int j = ci[k];
            const double bd = a > 0 ? p.col_ub[j] : p.col_lb[j];
            if (!fin(bd))
                finite = false;
            else
                maxact += a * bd;
        }
        if (!finite) continue;
        for (int32_t k = rp[i]; k < rp[i + 1]; ++k) {
            const int j = ci[k];
            if (!is_int(p, j) || p.col_lb[j] != 0 || p.col_ub[j] != 1) continue;
            const double a = sgn * va[k], tol = 1e-9 * (1 + std::abs(b));
            if (a > 0) {
                const double d = b - (maxact - a); // slack at x_k = 0
                if (d > tol && d < a - tol) {
                    va[k] = sgn * (a - d);
                    b -= d;
                    maxact -= d;
                    ++changed;
                }
            } else if (a < 0) {
                const double d = b - (maxact + a); // slack at x_k = 1
                if (d > tol && d < -a - tol) {
                    va[k] = sgn * (a + d);
                    ++changed;
                }
            }
        }
        if (uf)
            p.row_ub[i] = b;
        else
            p.row_lb[i] = -b;
    }
    if (changed) p.A = HostMatrix(m, n, rp, ci, va);
    return changed;
}

inline MilpSolution
solve_milp(const LpProblem &orig, const MilpOptions &opt)
{
    const auto t0 = std::chrono::steady_clock::now();
    auto elapsed = [&] {
        return std::chrono::duration<double>(
            std::chrono::steady_clock::now() - t0)
            .count();
    };
    MilpSolution out;
    const std::string bad = orig.validate();
    if (!bad.empty()) throw std::invalid_argument("solve_milp: " + bad);

    LpProblem work;
    std::unique_ptr<Presolve> pre;
    if (opt.presolve) {
        pre.reset(new Presolve(orig));
        if (pre->status() == Status::Infeasible) {
            out.status = Status::Infeasible;
            out.seconds = elapsed();
            return out;
        }
        work = pre->reduced();
        work.is_integer.assign(work.cols(), 0);
        if (!orig.is_integer.empty())
            for (size_t k = 0; k < work.cols(); ++k)
                work.is_integer[k] = orig.is_integer[pre->col_map()[k]];
    } else {
        work = orig;
        if (work.is_integer.empty()) work.is_integer.assign(work.cols(), 0);
    }
    for (size_t j = 0; j < work.cols(); ++j) {
        if (!work.is_integer[j]) continue;
        work.col_lb[j] = std::ceil(work.col_lb[j] - 1e-9);
        work.col_ub[j] = std::floor(work.col_ub[j] + 1e-9);
        if (work.col_lb[j] > work.col_ub[j]) {
            out.status = Status::Infeasible;
            out.seconds = elapsed();
            return out;
        }
    }

    const int tightened = opt.presolve ? tighten_coefficients(work) : 0;
    if (opt.verbose && tightened)
        std::printf("[milp] coefficient tightening changed %d coefficients\n",
            tightened);

    MilpSolution s;
    if (work.cols() == 0 || work.rows() == 0) {
        // nothing left to search: every column at the bound its cost prefers
        std::vector<double> x(work.cols());
        bool unbounded = false;
        for (size_t j = 0; j < work.cols(); ++j) {
            const double c = work.c[j];
            double v = c > 0 ? work.col_lb[j] : (c < 0 ? work.col_ub[j] : 0.0);
            if (c == 0)
                v = std::min(std::max(0.0, work.col_lb[j]), work.col_ub[j]);
            if (!std::isfinite(v)) unbounded = true;
            x[j] = v;
        }
        s.status = unbounded ? Status::Unbounded : Status::Optimal;
        if (!unbounded) {
            s.x = x;
            s.objective = check_milp(work, x).objective;
            s.bound = s.objective;
        }
    } else {
        milp::BranchAndBound bnb(work, opt, t0);
        s = bnb.solve();
    }

    out = s;
    out.x.clear();
    out.objective = kInf;
    if (s.has_solution()) {
        std::vector<double> x;
        if (pre) {
            LpSolution r;
            r.x = s.x;
            r.y.assign(work.rows(), 0.0);
            r.z.assign(work.cols(), 0.0);
            x = pre->postsolve(r).x;
        } else {
            x = s.x;
        }
        if (!orig.is_integer.empty())
            for (size_t j = 0; j < x.size(); ++j)
                if (orig.is_integer[j]) x[j] = std::round(x[j]);
        const MilpCheck c = check_milp(orig, x);
        if (c.ok(10 * opt.feas_tol, opt.int_tol)) {
            out.x = x;
            out.objective = c.objective;
        } else if (opt.verbose) {
            std::printf("[milp] postsolved solution fails the check: bounds "
                        "%.2e rows %.2e integrality %.2e\n",
                c.bound_viol, c.row_viol, c.int_viol);
        }
        if (out.x.empty() && out.status == Status::Optimal)
            out.status = Status::NumericalError;
    }
    out.seconds = elapsed();
    return out;
}

} // namespace Solver
} // namespace AXOS
