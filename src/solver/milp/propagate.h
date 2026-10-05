// SPDX-License-Identifier: BSD-3-Clause
//
// Domain propagation (activity-based bound tightening) for MILP: for a row
// l <= sum_j a_j x_j <= u, the minimum and maximum activities over the
// current bounds give, for every x_j in the row,
//     a_j > 0:  x_j <= (u - minact_{-j}) / a_j,   x_j >= (l - maxact_{-j}) / a_j
//     a_j < 0:  the same with the inequalities reversed,
// where minact_{-j} is the minimum activity of the other terms. Integer
// bounds are rounded. A row whose minimum activity exceeds u (or maximum is
// below l) proves the domain empty.
//
// Rows are processed from a queue: initially all rows (or the rows of the
// columns whose bounds changed, the node / probing case), then the rows of
// every column a row tightened, until nothing changes or the work budget is
// spent. Activities are recomputed per visit (O(row length)); infinite
// contributions are counted so a row with one infinite term can still
// tighten that term's variable.
#pragma once

#include "solver/milp/milp_model.h"
#include <cmath>
#include <vector>

namespace AXOS {
namespace Solver {
namespace milp {

// Bound changes in order, with the values they replaced: undo() restores
// the domains to any earlier mark (fix-and-propagate, probing, diving).
struct Trail {
    std::vector<int> col;
    std::vector<double> old_lb, old_ub;
    size_t mark() const { return col.size(); }
    void
    record(int j, double l, double u)
    {
        col.push_back(j);
        old_lb.push_back(l);
        old_ub.push_back(u);
    }
    void
    undo(size_t to, std::vector<double> &lb, std::vector<double> &ub)
    {
        while (col.size() > to) {
            lb[col.back()] = old_lb.back();
            ub[col.back()] = old_ub.back();
            col.pop_back();
            old_lb.pop_back();
            old_ub.pop_back();
        }
    }
};

class Propagator {
  public:
    explicit Propagator(const LpProblem &p) : p_(p), At_(p.A.transpose())
    {
        const size_t n = p.cols();
        isint_.assign(n, 0);
        for (size_t j = 0; j < n; ++j) isint_[j] = is_int(p, j) ? 1 : 0;
        inq_.assign(p.rows(), 0);
    }

    // Tightens lb/ub (column bounds, size n) in place. changed: the columns
    // whose bounds changed since the last fixpoint (nullptr: start from all
    // rows). trail: receives every bound change with the value it replaced.
    // Returns false when the domain is proven empty (the bounds may then be
    // partly tightened: undo through the trail).
    bool
    propagate(std::vector<double> &lb, std::vector<double> &ub, const std::vector<int> *changed = nullptr,
        Trail *trail = nullptr, double work_factor = 10.0)
    {
        const size_t m = p_.rows();
        queue_.clear();
        if (changed) {
            for (int j : *changed) push_rows_of(j);
        } else {
            for (size_t i = 0; i < m; ++i) push_row(static_cast<int>(i));
        }
        const double budget = work_factor * static_cast<double>(p_.A.nnz() + m) + 1000;
        double work = 0;
        size_t head = 0;
        bool feasible = true;
        while (head < queue_.size()) {
            const int i = queue_[head++];
            inq_[i] = 0;
            const auto b = p_.A.row_ptr()[i], e = p_.A.row_ptr()[i + 1];
            work += static_cast<double>(e - b);
            if (!row(i, lb, ub, trail)) {
                feasible = false;
                break;
            }
            if (work > budget) break;
        }
        for (size_t k = head; k < queue_.size(); ++k) inq_[queue_[k]] = 0;
        queue_.clear();
        return feasible;
    }

    const HostMatrix &transposed() const { return At_; }

  private:
    const LpProblem &p_;
    HostMatrix At_;
    std::vector<uint8_t> isint_, inq_;
    std::vector<int> queue_;

    static bool fin(double v) { return std::abs(v) < 1e20; }

    void
    push_row(int i)
    {
        if (!inq_[i]) {
            inq_[i] = 1;
            queue_.push_back(i);
        }
    }
    void
    push_rows_of(int j)
    {
        for (auto k = At_.row_ptr()[j]; k < At_.row_ptr()[j + 1]; ++k) push_row(At_.col_ind()[k]);
    }

    // Tightens the bounds of the columns of row i; false if infeasible.
    bool
    row(int i, std::vector<double> &lb, std::vector<double> &ub, Trail *trail)
    {
        const auto *ci = p_.A.col_ind();
        const double *va = p_.A.values();
        const auto b = p_.A.row_ptr()[i], e = p_.A.row_ptr()[i + 1];
        const double l = p_.row_lb[i], u = p_.row_ub[i];
        double minact = 0, maxact = 0;
        int ninf_min = 0, ninf_max = 0;
        for (auto k = b; k < e; ++k) {
            const double a = va[k];
            const int j = ci[k];
            if (a > 0) {
                if (fin(lb[j])) minact += a * lb[j]; else ++ninf_min;
                if (fin(ub[j])) maxact += a * ub[j]; else ++ninf_max;
            } else if (a < 0) {
                if (fin(ub[j])) minact += a * ub[j]; else ++ninf_min;
                if (fin(lb[j])) maxact += a * lb[j]; else ++ninf_max;
            }
        }
        const double tol_u = 1e-6 * (1 + std::abs(u)), tol_l = 1e-6 * (1 + std::abs(l));
        if (fin(u) && ninf_min == 0 && minact > u + tol_u) return false;
        if (fin(l) && ninf_max == 0 && maxact < l - tol_l) return false;
        const bool use_u = fin(u) && ninf_min <= 1, use_l = fin(l) && ninf_max <= 1;
        if (!use_u && !use_l) return true;
        for (auto k = b; k < e; ++k) {
            const double a = va[k];
            const int j = ci[k];
            if (std::abs(a) < 1e-9) continue;
            double nlb = lb[j], nub = ub[j];
            // residual activities of the other terms
            if (use_u) {
                double rest;
                bool ok = true;
                const double own = a > 0 ? lb[j] : ub[j];
                if (ninf_min == 0) rest = minact - a * own;
                else if (!fin(own)) rest = minact; // j is the one infinite term
                else ok = false;
                if (ok) {
                    const double v = (u - rest) / a;
                    if (a > 0) nub = std::min(nub, v); else nlb = std::max(nlb, v);
                }
            }
            if (use_l) {
                double rest;
                bool ok = true;
                const double own = a > 0 ? ub[j] : lb[j];
                if (ninf_max == 0) rest = maxact - a * own;
                else if (!fin(own)) rest = maxact;
                else ok = false;
                if (ok) {
                    const double v = (l - rest) / a;
                    if (a > 0) nlb = std::max(nlb, v); else nub = std::min(nub, v);
                }
            }
            if (isint_[j]) {
                nlb = std::ceil(nlb - 1e-6);
                nub = std::floor(nub + 1e-6);
            }
            const double range = fin(lb[j]) && fin(ub[j]) ? ub[j] - lb[j] : 1e300;
            // continuous bounds move only by a meaningful amount (no creeping)
            const double step = isint_[j] ? 0.5 : 1e-3 * std::max(1.0, std::min(range, std::abs(nlb) + std::abs(nub)));
            const bool up_lb = nlb > lb[j] + step && std::abs(nlb) < 1e15;
            const bool up_ub = nub < ub[j] - step && std::abs(nub) < 1e15;
            if (!up_lb && !up_ub) continue;
            if (trail) trail->record(j, lb[j], ub[j]);
            if (up_lb) lb[j] = nlb;
            if (up_ub) ub[j] = nub;
            if (lb[j] > ub[j] + 1e-6 * (1 + std::abs(ub[j]))) return false;
            if (lb[j] > ub[j]) lb[j] = ub[j]; // within tolerance
            push_rows_of(j);
        }
        return true;
    }
};

} // namespace milp
} // namespace Solver
} // namespace AXOS
