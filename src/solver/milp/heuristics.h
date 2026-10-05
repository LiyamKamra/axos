// SPDX-License-Identifier: BSD-3-Clause
//
// LP-free primal heuristics for MILP.
//
// FeasibilityJump (B. Luteberget, G. Sartor, Math. Prog. Comp. 2023): the point
// x stays within the bounds; a row's violation is its distance to [l, u],
// weighted by w_i. One variable moves at a time, to its "jump value": the value
// (integer for integer columns) minimizing the weighted violation of the rows
// of its column, the others fixed. That one-variable function is convex and
// piecewise linear, so its minimum is at a breakpoint (a row becoming
// satisfied) or a bound. Moves come from a violated row; when no variable of it
// improves, the row's weight grows (the Lagrangian part), eventually making
// some move pay off. Every step needs one column and its rows, no LP.
//
// FixAndPropagate: integer columns are fixed one by one in a given order to a
// preferred value (rounded LP value or a guess), each fix followed by domain
// propagation (propagate.h); a fix that empties a domain is retried with the
// other rounding once, then the heuristic gives up. The continuous columns are
// left to an LP over the fixed integers (the caller's).
#pragma once

#include "solver/milp/milp_model.h"
#include "solver/milp/propagate.h"
#include <algorithm>
#include <chrono>
#include <cmath>
#include <random>
#include <vector>

namespace AXOS {
namespace Solver {
namespace milp {

class FeasibilityJump {
  public:
    explicit FeasibilityJump(const LpProblem &p) : p_(p), At_(p.A.transpose())
    {
        isint_.assign(p.cols(), 0);
        for (size_t j = 0; j < p.cols(); ++j)
            isint_[j] = is_int(p, j) ? 1 : 0;
    }

    // Looks for a point satisfying all rows within [lb, ub], starting from x0
    // when given (else the bound nearest zero), for at most time_limit
    // seconds. On success x holds the point.
    bool
    run(const std::vector<double> &lb, const std::vector<double> &ub,
        const std::vector<double> *x0, double time_limit,
        std::vector<double> &x, uint64_t seed = 1)
    {
        const auto t0 = std::chrono::steady_clock::now();
        auto elapsed = [&] {
            return std::chrono::duration<double>(
                std::chrono::steady_clock::now() - t0)
                .count();
        };
        const size_t n = p_.cols(), m = p_.rows();
        rng_.seed(seed);
        x.assign(n, 0.0);
        for (size_t j = 0; j < n; ++j) {
            double v = x0 ? (*x0)[j] : 0.0;
            v = std::min(std::max(v, lb[j]), ub[j]);
            if (!std::isfinite(v) || std::abs(v) > 1e15)
                v = std::abs(lb[j]) < 1e15
                        ? lb[j]
                        : (std::abs(ub[j]) < 1e15 ? ub[j] : 0.0);
            if (isint_[j])
                v = std::min(std::max(std::round(v), std::ceil(lb[j] - 1e-9)),
                    std::floor(ub[j] + 1e-9));
            x[j] = v;
        }
        act_.assign(m, 0.0);
        w_.assign(m, 1.0);
        const auto *rp = p_.A.row_ptr();
        const auto *ci = p_.A.col_ind();
        const double *va = p_.A.values();
        for (size_t i = 0; i < m; ++i)
            for (auto k = rp[i]; k < rp[i + 1]; ++k)
                act_[i] += va[k] * x[ci[k]];
        // violated rows as an indexed set
        pos_.assign(m, -1);
        viol_.clear();
        for (size_t i = 0; i < m; ++i)
            update_violated(static_cast<int>(i));
        long steps = 0;
        std::vector<int> cand;
        while (!viol_.empty()) {
            if ((++steps & 255) == 0 && elapsed() > time_limit) return false;
            const int r = viol_[rng_() % viol_.size()];
            // candidate columns of row r (a sample of long rows)
            cand.clear();
            const auto b = rp[r], e = rp[r + 1];
            const long len = static_cast<long>(e - b);
            if (len <= 48) {
                for (auto k = b; k < e; ++k)
                    cand.push_back(ci[k]);
            } else {
                for (int s = 0; s < 48; ++s)
                    cand.push_back(ci[b + static_cast<long>(rng_() % len)]);
            }
            int best_j = -1;
            double best_v = 0, best_score = 1e-9;
            for (int j : cand) {
                if (lb[j] == ub[j]) continue;
                double v, score;
                jump(j, lb, ub, x, v, score);
                if (score > best_score ||
                    (score == best_score && best_j >= 0 && (rng_() & 1))) {
                    best_score = score;
                    best_j = j;
                    best_v = v;
                }
            }
            if (best_j < 0) {
                w_[r] += 1.0; // local minimum for this row: weigh it up
                continue;
            }
            const double delta = best_v - x[best_j];
            x[best_j] = best_v;
            for (auto k = At_.row_ptr()[best_j]; k < At_.row_ptr()[best_j + 1];
                ++k) {
                const int i = At_.col_ind()[k];
                act_[i] += At_.values()[k] * delta;
                update_violated(i);
            }
        }
        // exact activities (drift) and a final check
        for (size_t i = 0; i < m; ++i) {
            double a = 0;
            for (auto k = rp[i]; k < rp[i + 1]; ++k)
                a += va[k] * x[ci[k]];
            if (a < p_.row_lb[i] - 1e-6 * (1 + std::abs(p_.row_lb[i])) ||
                a > p_.row_ub[i] + 1e-6 * (1 + std::abs(p_.row_ub[i])))
                return false;
        }
        return true;
    }

  private:
    const LpProblem &p_;
    HostMatrix At_;
    std::vector<uint8_t> isint_;
    std::vector<double> act_, w_;
    std::vector<int> pos_, viol_;
    std::mt19937_64 rng_;
    std::vector<double> cands_;

    double
    row_viol(int i, double a) const
    {
        const double l = p_.row_lb[i], u = p_.row_ub[i];
        if (a > u) return a - u;
        if (a < l) return l - a;
        return 0.0;
    }
    void
    update_violated(int i)
    {
        const double a = act_[i];
        const bool v = a > p_.row_ub[i] + 1e-6 * (1 + std::abs(p_.row_ub[i])) ||
                       a < p_.row_lb[i] - 1e-6 * (1 + std::abs(p_.row_lb[i]));
        if (v && pos_[i] < 0) {
            pos_[i] = static_cast<int>(viol_.size());
            viol_.push_back(i);
        } else if (!v && pos_[i] >= 0) {
            const int last = viol_.back();
            viol_[pos_[i]] = last;
            pos_[last] = pos_[i];
            viol_.pop_back();
            pos_[i] = -1;
        }
    }

    // Weighted violation of the rows of column j with x_j = v.
    double
    column_cost(int j, double v, double xj) const
    {
        double f = 0;
        for (auto k = At_.row_ptr()[j]; k < At_.row_ptr()[j + 1]; ++k) {
            const int i = At_.col_ind()[k];
            const double a = At_.values()[k];
            f += w_[i] * row_viol(i, act_[i] + a * (v - xj));
        }
        return f;
    }

    // Best value of column j (jump value) and its score (decrease of the
    // weighted violation); the minimum of a convex piecewise-linear
    // function lies at a breakpoint or a bound.
    void
    jump(int j, const std::vector<double> &lb, const std::vector<double> &ub,
        const std::vector<double> &x, double &best_v, double &score)
    {
        const double xj = x[j];
        cands_.clear();
        auto add = [&](double v) {
            if (!std::isfinite(v)) return;
            v = std::min(std::max(v, lb[j]), ub[j]);
            if (std::abs(v) > 1e15) return;
            if (isint_[j]) {
                cands_.push_back(std::floor(v + 1e-9));
                cands_.push_back(std::ceil(v - 1e-9));
            } else {
                cands_.push_back(v);
            }
        };
        for (auto k = At_.row_ptr()[j]; k < At_.row_ptr()[j + 1]; ++k) {
            const int i = At_.col_ind()[k];
            const double a = At_.values()[k];
            if (a == 0) continue;
            const double rest = act_[i] - a * xj;
            if (std::abs(p_.row_ub[i]) < 1e20) add((p_.row_ub[i] - rest) / a);
            if (std::abs(p_.row_lb[i]) < 1e20) add((p_.row_lb[i] - rest) / a);
        }
        add(lb[j]);
        add(ub[j]);
        const double f0 = column_cost(j, xj, xj);
        double best_f = f0;
        best_v = xj;
        for (double v : cands_) {
            if (v == xj) continue;
            if (isint_[j] && (v < lb[j] - 1e-9 || v > ub[j] + 1e-9)) continue;
            const double f = column_cost(j, v, xj);
            if (f < best_f - 1e-12) {
                best_f = f;
                best_v = v;
            }
        }
        score = f0 - best_f;
    }
};

// Fixes the integer columns in `order` to `want` (rounded into the current
// domain), propagating after each fix; a fix that empties a domain is undone
// (through the trail) and retried with the other rounding once, then the
// heuristic gives up. lb/ub hold the final domains: the integers fixed, the
// continuous columns propagated.
inline bool
fix_and_propagate(Propagator &prop, const LpProblem &p,
    const std::vector<int> &order, const std::vector<double> &want,
    std::vector<double> &lb, std::vector<double> &ub, double deadline_s,
    const std::chrono::steady_clock::time_point &t0)
{
    std::vector<int> changed(1);
    Trail trail;
    long count = 0;
    for (int j : order) {
        if (!is_int(p, j) || lb[j] == ub[j]) continue;
        if ((++count & 63) == 0 &&
            std::chrono::duration<double>(std::chrono::steady_clock::now() - t0)
                    .count() > deadline_s)
            return false;
        const double v = std::min(std::max(std::round(want[j]), lb[j]), ub[j]);
        const double alt =
            want[j] >= v ? std::min(v + 1, ub[j]) : std::max(v - 1, lb[j]);
        bool ok = false;
        for (int attempt = 0; attempt < 2 && !ok; ++attempt) {
            const double val = attempt == 0 ? v : alt;
            if (attempt == 1 && val == v) break;
            const size_t mark = trail.mark();
            trail.record(j, lb[j], ub[j]);
            lb[j] = ub[j] = val;
            changed[0] = j;
            ok = prop.propagate(lb, ub, &changed, &trail, 2.0);
            if (!ok) trail.undo(mark, lb, ub);
        }
        if (!ok) return false;
        trail.col.clear(); // keep the fixes: the trail only serves the retry
        trail.old_lb.clear();
        trail.old_ub.clear();
    }
    return true;
}

} // namespace milp
} // namespace Solver
} // namespace AXOS
