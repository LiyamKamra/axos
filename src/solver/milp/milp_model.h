// SPDX-License-Identifier: BSD-3-Clause
//
// Mixed-integer linear programs for the MILP solver (solve_milp.h):
//
//   min c^T x + offset   s.t.  row_lb <= A x <= row_ub,  col_lb <= x <= col_ub,
//                              x_j integer for is_integer[j]
//
// An LpProblem with is_integer filled (io/mps.h reads the MARKER INTORG /
// INTEND sections). check_milp() is the one feasibility test: every
// incumbent the solver reports is checked on the ORIGINAL problem with it.
#pragma once

#include "solver/model.h"
#include <algorithm>
#include <cmath>
#include <limits>
#include <string>
#include <vector>

namespace AXOS {
namespace Solver {

struct MilpOptions {
    double time_limit = 3600;  // seconds
    double gap_rel = 1e-4;     // stop when (primal - dual) / max(|primal|, 1e-9) <= gap_rel
    double gap_abs = 1e-6;     //   or primal - dual <= gap_abs (MIPLIB conventions)
    double int_tol = 1e-6;     // |x - round(x)| for integrality
    double feas_tol = 1e-6;    // constraint and bound violations (relative to 1 + |rhs|)
    long node_limit = std::numeric_limits<long>::max();
    int verbose = 0;           // 0 silent, 1 progress lines, 2 more
    bool presolve = true;
    bool propagation = true;   // domain propagation at the nodes
    bool cuts = true;          // root cuts (Gomory mixed-integer)
    int cut_rounds = 20;
    double cut_time_frac = 0.1;    // share of the time limit for the root cut loop
    bool heuristics = true;    // rounding, diving, feasibility jump, fix-and-propagate, RINS
    bool rins = true;          // RINS sub-MIPs (off inside them)
    double fj_effort = 0.05;   // share of the time limit for feasibility jump at the root
    int strong_branch_iters = 30;  // simplex iterations per strong-branching child
    int reliability = 4;           // strong-branch a variable until it has this many pseudocost samples
    bool probing = true;           // double probing on the binaries at the root
    double probing_time_frac = 0.05; // share of the time limit for probing (at most 10 s)
    // GPU (builds with AXOS_ENABLE_CUDA): root probing on the GPU, feasibility
    // jump walkers on the GPU next to the CPU root, and batched PDHG strong
    // branching for LPs with at least gpu_min_nnz nonzeros
    bool gpu = false;
    long gpu_min_nnz = 10000;
    long gpu_lp_min_nnz = 50000;   // root LP first by HPR on the GPU from this size (heuristic seed)
    bool gpu_lp_crossover = false; // also start the dual simplex from that point (measured slower)
    int gpu_fj_walkers = 64;
    int gpu_sb_iters = 400;        // PDHG iterations per strong-branching batch
};

struct MilpSolution {
    Status status = Status::NotSolved; // Optimal (gap closed), Infeasible, TimeLimit, ...
    std::vector<double> x;             // best solution (original columns), empty if none
    double objective = std::numeric_limits<double>::infinity(); // its objective (with offset)
    double bound = -std::numeric_limits<double>::infinity();    // proven lower bound
    long nodes = 0;
    long lp_iterations = 0;
    double seconds = 0;
    double first_solution_seconds = -1; // time of the first incumbent
    double root_bound = -std::numeric_limits<double>::infinity(); // LP bound after root cuts
    int cuts = 0;
    std::string incumbent_source;       // heuristic that found the final incumbent
    std::string notes;                  // root statistics ("probing_gpu=probes/fixed/tightened;...")

    bool has_solution() const { return !x.empty(); }
    double
    gap() const
    {
        if (!has_solution()) return std::numeric_limits<double>::infinity();
        const double d = objective - bound;
        return d <= 0 ? 0.0 : d / std::max({std::abs(objective), std::abs(bound), 1e-9});
    }
};

// Feasibility of x for p: the largest relative violation of a bound, a row
// (relative to 1 + |rhs|) and integrality. Returns the objective in obj.
struct MilpCheck {
    double bound_viol = 0, row_viol = 0, int_viol = 0, objective = 0;
    bool ok(double feas_tol, double int_tol) const
    {
        return bound_viol <= feas_tol && row_viol <= feas_tol && int_viol <= int_tol;
    }
};

inline MilpCheck
check_milp(const LpProblem &p, const std::vector<double> &x)
{
    MilpCheck c;
    const size_t n = p.cols(), m = p.rows();
    if (x.size() != n) {
        c.bound_viol = c.row_viol = c.int_viol = std::numeric_limits<double>::infinity();
        return c;
    }
    double obj = p.offset;
    for (size_t j = 0; j < n; ++j) {
        obj += p.c[j] * x[j];
        const double v = std::max(p.col_lb[j] - x[j], x[j] - p.col_ub[j]);
        if (v > 0) c.bound_viol = std::max(c.bound_viol, v / (1 + std::abs(x[j])));
        if (!p.is_integer.empty() && p.is_integer[j])
            c.int_viol = std::max(c.int_viol, std::abs(x[j] - std::round(x[j])));
    }
    const auto *rp = p.A.row_ptr();
    const auto *ci = p.A.col_ind();
    const double *va = p.A.values();
    for (size_t i = 0; i < m; ++i) {
        double ax = 0;
        for (auto k = rp[i]; k < rp[i + 1]; ++k) ax += va[k] * x[ci[k]];
        if (ax < p.row_lb[i]) c.row_viol = std::max(c.row_viol, (p.row_lb[i] - ax) / (1 + std::abs(p.row_lb[i])));
        if (ax > p.row_ub[i]) c.row_viol = std::max(c.row_viol, (ax - p.row_ub[i]) / (1 + std::abs(p.row_ub[i])));
    }
    c.objective = obj;
    return c;
}

inline bool
is_int(const LpProblem &p, size_t j)
{
    return !p.is_integer.empty() && p.is_integer[j];
}

} // namespace Solver
} // namespace AXOS
