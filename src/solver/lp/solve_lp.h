// SPDX-License-Identifier: BSD-3-Clause
//
// solve_lp(): presolve -> LP algorithm (PDLP or interior point, see
// SolverOptions::method) -> postsolve.
#pragma once

#include "solver/lp/ipm.h"
#include "solver/lp/pdlp.h"
#include "solver/lp/simplex.h"

namespace AXOS {
namespace Solver {

// Presolve + solve + postsolve. The returned solution is for `p` as given.
template <template <typename> class Store = Cpu::HostStorage>
LpSolution
solve_lp(const LpProblem &p, const SolverOptions &opt = SolverOptions())
{
    auto t0 = std::chrono::steady_clock::now();
    auto secs = [&] {
        return std::chrono::duration<double>(std::chrono::steady_clock::now() - t0)
            .count();
    };
    // The interior-point method needs presolve (free rows, fixed columns).
    const bool need_presolve = opt.presolve || opt.method != LpMethod::Pdlp ||
                               p.rows() == 0 || p.cols() == 0;
    auto usable = [](const LpProblem &q, const LpSolution &s) {
        if (s.x.size() != q.cols() || s.y.size() != q.rows()) return false;
        for (double v : s.x) if (!std::isfinite(v)) return false;
        for (double v : s.y) if (!std::isfinite(v)) return false;
        return true;
    };
    // `start` (an interior / approximate solution of q) seeds the simplex by crossover
    auto run = [&](const LpProblem &q, LpMethod m, const SolverOptions &o,
                   const LpSolution *start = nullptr) {
        switch (m) {
        case LpMethod::Ipm: {
            LpSolution s = Ipm<Store>().solve(q, o);
            if (o.crossover && s.status == Status::Optimal && usable(q, s)) {
                SolverOptions oc = o;
                oc.time_limit = std::max(1.0, o.time_limit - s.seconds);
                LpSolution c = DualSimplex().solve(q, oc, nullptr, nullptr, &s);
                if (c.status == Status::Optimal) return c;
            }
            return s;
        }
        case LpMethod::Simplex: return DualSimplex().solve(q, o, nullptr, nullptr, start);
        default: return Pdlp<Store>().solve(q, o);
        }
    };
    // Auto. A cheap interior-point factorization (predicted from the symbolic
    // analysis) means the interior-point method goes first, then simplex, then
    // PDLP. Otherwise the dual simplex gets a slice of the predicted interior-
    // point time (its typical win on network / sparse problems), then the
    // interior-point method, then PDLP. The time limit is shared.
    auto run_auto = [&](const LpProblem &q) {
        constexpr double kCheapFlops = 4e9;    // Sum of squared column counts
        constexpr double kFlopsPerSec = 4e10;  // measured multifrontal rate on the CPU
        auto remaining = [&] { return std::max(0.0, opt.time_limit - secs()); };
        auto accept = [&](LpMethod m, const LpSolution &s) {
            // Optimal, or a certificate from a method that can give one
            return s.status == Status::Optimal ||
                   (m != LpMethod::Ipm && (s.status == Status::Infeasible ||
                                           s.status == Status::Unbounded));
        };
        auto attempt = [&](LpMethod m, double budget, LpSolution &best, double flops_limit = 0) {
            SolverOptions o = opt;
            o.time_limit = std::min(budget, remaining());
            o.ipm_max_flops = flops_limit;
            if (o.time_limit <= 0) return false;
            LpSolution s = run(q, m, o);
            if (accept(m, s)) { best = s; return true; }
            if (s.status != Status::NotSolved) best = s;
            else if (best.status == Status::NotSolved) best.factor_flops = s.factor_flops;
            return false;
        };
        LpSolution best;
        if (attempt(LpMethod::Ipm, remaining(), best, kCheapFlops)) return best;
        if (best.status == Status::NotSolved) { // IPM predicted to be expensive
            const double t_ipm = 20.0 * best.factor_flops / kFlopsPerSec;
            if (attempt(LpMethod::Simplex, std::max(2.0, 0.3 * t_ipm), best)) return best;
            if (attempt(LpMethod::Ipm, 0.6 * remaining(), best)) return best;
            attempt(LpMethod::Pdlp, remaining(), best);
            return best;
        }
        if (attempt(LpMethod::Simplex, 0.5 * remaining(), best)) return best;
        attempt(LpMethod::Pdlp, remaining(), best);
        return best;
    };
    if (!need_presolve) return opt.method == LpMethod::Auto ? run_auto(p)
                                                            : run(p, opt.method, opt);

    Presolve pre(p);
    if (pre.status() != Status::NotSolved) {
        LpSolution s;
        s.status = pre.status();
        s.x.assign(p.cols(), 0.0);
        s.y.assign(p.rows(), 0.0);
        s.seconds = secs();
        return s;
    }
    const LpProblem &r = pre.reduced();
    LpSolution rs;
    if (r.rows() > 0 && r.cols() > 0) {
        rs = opt.method == LpMethod::Auto ? run_auto(r) : run(r, opt.method, opt);
    } else {
        rs.status = Status::Optimal; // nothing left to optimize
    }
    LpSolution s = pre.postsolve(rs);
    s.seconds = secs();
    return s;
}

} // namespace Solver
} // namespace AXOS
