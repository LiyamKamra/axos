// SPDX-License-Identifier: BSD-3-Clause
//
// Optimization problem model shared by all AXOS solvers.
//
//   minimize    c^T x + offset
//   subject to  row_lb <= A x <= row_ub
//               col_lb <=  x  <= col_ub
//
// Any bound may be +-infinity (Solver::kInf). Equalities have row_lb ==
// row_ub. A maximization problem is stored negated (c -> -c, offset -> -offset)
// with `maximize` set so results can be reported in the original sense.
//
// Duals follow the convention used throughout the solvers:
//   z = c - A^T y  (reduced costs)
//   y_i >= 0 only if the row is at its lower bound, y_i <= 0 only if at its
//   upper bound; z_j >= 0 only if x_j is at its lower bound, z_j <= 0 only if
//   x_j is at its upper bound.
//   dual objective = offset + sum_i (y_i > 0 ? row_lb_i : row_ub_i) * y_i
//                           + sum_j (z_j > 0 ? col_lb_j : col_ub_j) * z_j
#pragma once

#include "sparse/csr.h"
#include <cmath>
#include <limits>
#include <sstream>
#include <string>
#include <vector>

namespace AXOS {
namespace Solver {

inline constexpr double kInf = std::numeric_limits<double>::infinity();

enum class Status {
    NotSolved,
    Optimal,
    Infeasible,      // primal infeasible (a dual ray was found)
    Unbounded,       // dual infeasible: a primal improving ray exists
    IterationLimit,
    TimeLimit,
    NumericalError,
};

inline const char *
to_string(Status s)
{
    switch (s) {
    case Status::NotSolved: return "not solved";
    case Status::Optimal: return "optimal";
    case Status::Infeasible: return "infeasible";
    case Status::Unbounded: return "unbounded";
    case Status::IterationLimit: return "iteration limit";
    case Status::TimeLimit: return "time limit";
    case Status::NumericalError: return "numerical error";
    }
    return "?";
}

using HostMatrix = Sparse::Csr<double, int32_t, Cpu::HostStorage>;

struct LpProblem {
    std::string name;
    HostMatrix A; // rows x cols
    std::vector<double> c, col_lb, col_ub, row_lb, row_ub;
    double offset = 0;
    bool maximize = false;
    std::vector<uint8_t> is_integer; // empty for a pure LP
    std::vector<std::string> row_names, col_names;

    size_t rows() const { return A.rows(); }
    size_t cols() const { return A.cols(); }
    bool has_integers() const
    {
        for (auto v : is_integer)
            if (v) return true;
        return false;
    }

    // Empty string when the problem is well formed.
    std::string
    validate() const
    {
        std::ostringstream e;
        const size_t m = rows(), n = cols();
        if (c.size() != n || col_lb.size() != n || col_ub.size() != n)
            e << "column vector sizes must equal cols(); ";
        if (row_lb.size() != m || row_ub.size() != m)
            e << "row vector sizes must equal rows(); ";
        if (!is_integer.empty() && is_integer.size() != n)
            e << "is_integer must be empty or have cols() entries; ";
        if (!e.str().empty()) return e.str();
        std::string v = A.validate();
        if (!v.empty()) return "matrix: " + v;
        for (size_t j = 0; j < n; ++j) {
            if (std::isnan(c[j]) || std::isnan(col_lb[j]) || std::isnan(col_ub[j]))
                return "NaN in column data at " + std::to_string(j);
            if (col_lb[j] > col_ub[j])
                return "column " + std::to_string(j) + " has lb > ub";
        }
        for (size_t i = 0; i < m; ++i) {
            if (std::isnan(row_lb[i]) || std::isnan(row_ub[i]))
                return "NaN in row bounds at " + std::to_string(i);
            if (row_lb[i] > row_ub[i])
                return "row " + std::to_string(i) + " has lb > ub";
        }
        return "";
    }

    // Objective value c^T x + offset for a given x (in the stored, minimized
    // sense).
    double
    objective(const std::vector<double> &x) const
    {
        double s = offset;
        for (size_t j = 0; j < cols(); ++j)
            s += c[j] * x[j];
        return s;
    }
};

// Solution in the problem's stored (minimization) form.
struct LpSolution {
    Status status = Status::NotSolved;
    std::vector<double> x;     // primal, cols()
    std::vector<double> y;     // row duals, rows()
    std::vector<double> z;     // reduced costs, cols()
    double primal_objective = 0;
    double dual_objective = 0;
    double primal_residual = 0; // ||violation of Ax in [row_lb,row_ub]||_2
    double dual_residual = 0;   // ||violation of reduced-cost sign||_2
    double gap = 0;             // |primal - dual objective|
    // Bound on the objective error of (x, y): the gap plus each residual
    // times the size of the variables it multiplies,
    //   gap + ||dual residual||_2 ||x||_2 + ||primal residual||_2 ||y||_2.
    // A small relative gap with a large error_bound means the point is not
    // trustworthy (badly scaled problem).
    double error_bound = 0;
    long iterations = 0;
    double seconds = 0;
    double factor_flops = 0; // Ipm (CPU): estimated flops of one KKT factorization, when known
};

// Which algorithm solve_lp() uses. Pdlp is cheap per iteration (best on the GPU
// and on huge problems), reaches moderate accuracy and detects infeasibility;
// Ipm is accurate in few iterations but has no infeasibility certificates;
// Simplex is the dual simplex (exact vertex solutions, certificates, warm
// starts); Auto runs Ipm, then Simplex, then Pdlp until one reports a result.
enum class LpMethod { Pdlp, Ipm, Simplex, Auto };

// Termination tolerances follow the PDLP convention:
//   ||primal residual||_2 <= eps_primal * (1 + ||finite row bounds||_2)
//   ||dual residual||_2   <= eps_dual   * (1 + ||c||_2)
//   |primal obj - dual obj| <= eps_gap * (1 + |primal obj| + |dual obj|)
struct SolverOptions {
    LpMethod method = LpMethod::Pdlp;
    int ipm_max_iterations = 200;
    int ipm_ordering = 0; // 0 default (AMD), 1 nested dissection (GPU only)
    double eps_primal = 1e-6;
    double eps_dual = 1e-6;
    double eps_gap = 1e-6;
    long max_iterations = 200000;
    double time_limit = 1e100; // seconds
    bool verbose = false;
    bool pdlp_polish = false;  // Pdlp: reach tight tolerances by polishing a 1e-4 solution (primal / dual feasibility solves)
    bool pdlp_halpern = true;  // Pdlp: reflected Halpern PDHG with fixed steps instead of adaptive-step PDHG with averaging
    int ipm_normal = 0;       // Ipm (CPU): 0 choose between normal equations and the augmented system, 1 force normal, -1 never
    double ipm_max_flops = 0; // Ipm (CPU): give up (NotSolved) if one factorization exceeds this many flops; 0 = no limit
    bool crossover = false; // Ipm: finish with a simplex crossover to a basic (vertex) solution
    bool presolve = true;
    bool scaling = true;
    int ruiz_iterations = 10;
    double pock_chambolle_alpha = 1.0; // <0 disables
    int check_frequency = 64;          // iterations between termination checks

    void
    set_tolerance(double eps)
    {
        eps_primal = eps_dual = eps_gap = eps;
    }
};

// Residuals of a candidate (x, y) for `p`, computed on the host in double
// precision. z is recomputed as c - A^T y. Used by tests and by the solvers'
// final verification against the original problem.
inline LpSolution
evaluate_solution(const LpProblem &p, const std::vector<double> &x,
    const std::vector<double> &y)
{
    const size_t m = p.rows(), n = p.cols();
    LpSolution s;
    s.x = x;
    s.y = y;
    s.z.assign(n, 0.0);
    std::vector<double> ax(m, 0.0);
    const auto *rp = p.A.row_ptr();
    const auto *ci = p.A.col_ind();
    const double *v = p.A.values();
    for (size_t i = 0; i < m; ++i)
        for (int k = rp[i]; k < rp[i + 1]; ++k) {
            ax[i] += v[k] * x[ci[k]];
            s.z[ci[k]] -= v[k] * y[i];
        }
    for (size_t j = 0; j < n; ++j)
        s.z[j] += p.c[j];

    double pr = 0, dr = 0, dual = p.offset;
    for (size_t i = 0; i < m; ++i) {
        double viol = std::max({p.row_lb[i] - ax[i], ax[i] - p.row_ub[i], 0.0});
        pr += viol * viol;
        double yi = y[i];
        // A dual on an infinite bound is a dual infeasibility.
        if (yi > 0 && std::isinf(p.row_lb[i])) dr += yi * yi;
        else if (yi < 0 && std::isinf(p.row_ub[i])) dr += yi * yi;
        else if (yi > 0) dual += yi * p.row_lb[i];
        else if (yi < 0) dual += yi * p.row_ub[i];
    }
    for (size_t j = 0; j < n; ++j) {
        double viol = std::max({p.col_lb[j] - x[j], x[j] - p.col_ub[j], 0.0});
        pr += viol * viol;
        double zj = s.z[j];
        if (zj > 0 && std::isinf(p.col_lb[j])) dr += zj * zj;
        else if (zj < 0 && std::isinf(p.col_ub[j])) dr += zj * zj;
        else if (zj > 0) dual += zj * p.col_lb[j];
        else if (zj < 0) dual += zj * p.col_ub[j];
    }
    s.primal_objective = p.objective(x);
    s.dual_objective = dual;
    s.primal_residual = std::sqrt(pr);
    s.dual_residual = std::sqrt(dr);
    s.gap = std::abs(s.primal_objective - s.dual_objective);
    double xn = 0, yn = 0;
    for (double v : x) xn += v * v;
    for (double v : y) yn += v * v;
    s.error_bound = s.gap + s.dual_residual * std::sqrt(xn) +
                    s.primal_residual * std::sqrt(yn);
    return s;
}

} // namespace Solver
} // namespace AXOS
