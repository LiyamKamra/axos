// SPDX-License-Identifier: BSD-3-Clause
//
// LP presolve with a postsolve stack that restores primal AND dual solutions.
//
// Reductions:
//   - empty rows            (feasibility check, then removed); free rows
//                            (both bounds infinite) are removed too
//   - empty columns         (fixed at the bound the cost prefers; may prove the
//                            problem dual infeasible / unbounded)
//   - fixed columns         (lb == ub: substituted out of the rows and cost)
//   - singleton rows        (turned into column bounds)
//   - duplicate / parallel rows (merged; their bounds intersected)
// applied until nothing changes.
//
// postsolve() maps a solution of the reduced problem back to the original
// problem: x from the stack, y for removed rows from the rules below, and
// z = c - A^T y recomputed on the original matrix.
//   * singleton row i on column j: if the reduced problem wants x_j at a bound
//     that this row tightened (z_j > 0 at the lower bound, z_j < 0 at the upper
//     bound), the row takes the dual: y_i = z_j / a_ij, which zeroes z_j.
//   * duplicate rows: the merged row's dual goes to whichever original row
//     supplies the binding bound.
#pragma once

#include "solver/model.h"
#include <algorithm>
#include <cmath>
#include <numeric>
#include <unordered_map>

namespace AXOS {
namespace Solver {

struct PresolveOptions {
    bool empty_rows = true;
    bool empty_cols = true;
    bool fixed_cols = true;
    bool singleton_rows = true;
    bool duplicate_rows = true;
    double feas_tol = 1e-9;
};

class Presolve {
  public:
    Presolve(const LpProblem &p, const PresolveOptions &opt = {})
        : orig_(p), opt_(opt)
    {
        run();
    }

    // Outcome of presolve itself: NotSolved (normal), Infeasible or
    // Unbounded (proven during presolve; no solve needed).
    Status status() const { return status_; }
    const LpProblem &reduced() const { return reduced_; }
    size_t removed_rows() const { return orig_.rows() - reduced_.rows(); }
    size_t removed_cols() const { return orig_.cols() - reduced_.cols(); }
    // reduced column -> original column (MILP: integrality follows the columns)
    const std::vector<int> &col_map() const { return col_map_; }

    // Build the original-space solution from a reduced-space one. Also works
    // when the reduced problem is empty (pass an empty LpSolution with
    // status Optimal).
    LpSolution
    postsolve(const LpSolution &r) const
    {
        const size_t m = orig_.rows(), n = orig_.cols();
        LpSolution s;
        s.status = r.status;
        s.iterations = r.iterations;
        s.seconds = r.seconds;
        s.x.assign(n, 0.0);
        s.y.assign(m, 0.0);
        for (size_t j = 0; j < col_map_.size(); ++j)
            if (j < r.x.size()) s.x[col_map_[j]] = r.x[j];
        for (size_t i = 0; i < row_map_.size(); ++i)
            if (i < r.y.size()) s.y[row_map_[i]] = r.y[i];
        for (auto &op : stack_) // primal values of removed columns
            if (op.kind == Op::EmptyCol || op.kind == Op::FixedCol)
                s.x[op.j] = op.value;

        for (auto it = stack_.rbegin(); it != stack_.rend(); ++it) {
            const Op &op = *it;
            if (op.kind == Op::SingletonRow) {
                double zj = orig_.c[op.j];
                for (int k = At_.row_ptr()[op.j]; k < At_.row_ptr()[op.j + 1]; ++k)
                    zj -= At_.values()[k] * s.y[At_.col_ind()[k]];
                if ((zj > 0 && op.tight_lb) || (zj < 0 && op.tight_ub))
                    s.y[op.i] = zj / op.a;
            } else if (op.kind == Op::DuplicateRow) {
                const double y = s.y[op.i];
                if (y > 0) {
                    if (op.lb_i >= op.impl_lb_k) { /* row i binds */ }
                    else { s.y[op.k] = y / op.alpha; s.y[op.i] = 0; }
                } else if (y < 0) {
                    if (op.ub_i <= op.impl_ub_k) { /* row i binds */ }
                    else { s.y[op.k] = y / op.alpha; s.y[op.i] = 0; }
                }
            }
        }

        // z = c - A^T y on the original matrix.
        s.z = orig_.c;
        for (size_t i = 0; i < m; ++i)
            for (int k = orig_.A.row_ptr()[i]; k < orig_.A.row_ptr()[i + 1]; ++k)
                s.z[orig_.A.col_ind()[k]] -= orig_.A.values()[k] * s.y[i];
        LpSolution e = evaluate_solution(orig_, s.x, s.y);
        e.status = s.status;
        e.iterations = s.iterations;
        e.seconds = s.seconds;
        return e;
    }

  private:
    struct Op {
        enum Kind { EmptyRow, EmptyCol, FixedCol, SingletonRow, DuplicateRow } kind;
        int i = -1, j = -1, k = -1;
        double value = 0, a = 0, alpha = 0;
        bool tight_lb = false, tight_ub = false;
        double lb_i = 0, ub_i = 0, impl_lb_k = 0, impl_ub_k = 0;
    };

    const LpProblem &orig_;
    PresolveOptions opt_;
    Status status_ = Status::NotSolved;
    LpProblem reduced_;
    HostMatrix At_;
    std::vector<Op> stack_;
    std::vector<int> row_map_, col_map_; // reduced index -> original index

    // working state
    std::vector<double> rlb_, rub_, clb_, cub_, cost_;
    std::vector<char> row_on_, col_on_;
    std::vector<int> row_cnt_, col_cnt_;
    double offset_ = 0;
    std::vector<int> rq_, cq_;

    static double
    tol_scale(double v, double tol)
    { return tol * (1.0 + std::abs(v)); }

    void
    remove_row(int i)
    {
        row_on_[i] = 0;
        for (int k = orig_.A.row_ptr()[i]; k < orig_.A.row_ptr()[i + 1]; ++k) {
            int j = orig_.A.col_ind()[k];
            if (col_on_[j] && --col_cnt_[j] <= 1) cq_.push_back(j);
        }
    }

    void
    remove_col(int j)
    {
        col_on_[j] = 0;
        for (int k = At_.row_ptr()[j]; k < At_.row_ptr()[j + 1]; ++k) {
            int i = At_.col_ind()[k];
            if (row_on_[i] && --row_cnt_[i] <= 1) rq_.push_back(i);
        }
    }

    void
    do_row(int i)
    {
        if (!row_on_[i]) return;
        if (opt_.empty_rows && rlb_[i] == -kInf && rub_[i] == kInf) {
            // a free row constrains nothing
            Op op; op.kind = Op::EmptyRow; op.i = i;
            stack_.push_back(op);
            remove_row(i);
            return;
        }
        if (row_cnt_[i] == 0 && opt_.empty_rows) {
            if (rlb_[i] > tol_scale(rlb_[i], opt_.feas_tol) ||
                rub_[i] < -tol_scale(rub_[i], opt_.feas_tol)) {
                status_ = Status::Infeasible;
                return;
            }
            Op op; op.kind = Op::EmptyRow; op.i = i;
            stack_.push_back(op);
            row_on_[i] = 0;
        } else if (row_cnt_[i] == 1 && opt_.singleton_rows) {
            int j = -1;
            double a = 0;
            for (int k = orig_.A.row_ptr()[i]; k < orig_.A.row_ptr()[i + 1]; ++k)
                if (col_on_[orig_.A.col_ind()[k]]) {
                    j = orig_.A.col_ind()[k];
                    a = orig_.A.values()[k];
                    break;
                }
            if (j < 0 || a == 0) return;
            double lo = a > 0 ? rlb_[i] / a : rub_[i] / a;
            double hi = a > 0 ? rub_[i] / a : rlb_[i] / a;
            Op op; op.kind = Op::SingletonRow; op.i = i; op.j = j; op.a = a;
            if (lo > clb_[j]) { clb_[j] = lo; op.tight_lb = true; }
            if (hi < cub_[j]) { cub_[j] = hi; op.tight_ub = true; }
            if (clb_[j] > cub_[j]) {
                if (clb_[j] - cub_[j] > tol_scale(cub_[j], opt_.feas_tol)) {
                    status_ = Status::Infeasible;
                    return;
                }
                clb_[j] = cub_[j] = 0.5 * (clb_[j] + cub_[j]);
            }
            stack_.push_back(op);
            remove_row(i);
            cq_.push_back(j);
        }
    }

    void
    do_col(int j)
    {
        if (!col_on_[j]) return;
        if (col_cnt_[j] == 0 && opt_.empty_cols) {
            double v;
            if (cost_[j] > 0) v = clb_[j];
            else if (cost_[j] < 0) v = cub_[j];
            else v = std::min(std::max(0.0, clb_[j]), cub_[j]);
            if (std::isinf(v)) {
                if (cost_[j] != 0) { status_ = Status::Unbounded; return; }
                v = std::isinf(clb_[j]) ? cub_[j] : clb_[j];
                if (std::isinf(v)) v = 0;
            }
            Op op; op.kind = Op::EmptyCol; op.j = j; op.value = v;
            stack_.push_back(op);
            offset_ += cost_[j] * v;
            col_on_[j] = 0;
        } else if (clb_[j] == cub_[j] && opt_.fixed_cols) {
            double v = clb_[j];
            Op op; op.kind = Op::FixedCol; op.j = j; op.value = v;
            stack_.push_back(op);
            offset_ += cost_[j] * v;
            for (int k = At_.row_ptr()[j]; k < At_.row_ptr()[j + 1]; ++k) {
                int i = At_.col_ind()[k];
                if (!row_on_[i]) continue;
                double d = At_.values()[k] * v;
                rlb_[i] -= d;
                rub_[i] -= d;
            }
            remove_col(j);
        }
    }

    bool
    merge_duplicates()
    {
        // Group active rows (>= 2 entries) by a hash of their normalized
        // pattern, then verify candidates exactly.
        std::unordered_map<size_t, std::vector<int>> groups;
        auto norm_of = [&](int i, int &first_col, double &first_val) {
            first_col = -1;
            for (int k = orig_.A.row_ptr()[i]; k < orig_.A.row_ptr()[i + 1]; ++k)
                if (col_on_[orig_.A.col_ind()[k]]) {
                    first_col = orig_.A.col_ind()[k];
                    first_val = orig_.A.values()[k];
                    return;
                }
        };
        for (size_t i = 0; i < orig_.rows(); ++i) {
            if (!row_on_[i] || row_cnt_[i] < 2) continue;
            int fc; double fv = 1;
            norm_of(static_cast<int>(i), fc, fv);
            size_t h = 1469598103934665603ULL;
            for (int k = orig_.A.row_ptr()[i]; k < orig_.A.row_ptr()[i + 1]; ++k) {
                int j = orig_.A.col_ind()[k];
                if (!col_on_[j]) continue;
                double v = orig_.A.values()[k] / fv;
                long long q = std::llround(v * 1e9);
                h = (h ^ static_cast<size_t>(j)) * 1099511628211ULL;
                h = (h ^ static_cast<size_t>(q)) * 1099511628211ULL;
            }
            groups[h].push_back(static_cast<int>(i));
        }
        bool changed = false;
        for (auto &g : groups) {
            auto &rows = g.second;
            for (size_t a = 0; a + 1 < rows.size(); ++a) {
                int i = rows[a];
                if (!row_on_[i]) continue;
                for (size_t b = a + 1; b < rows.size(); ++b) {
                    int k = rows[b];
                    if (!row_on_[k]) continue;
                    double alpha;
                    if (!parallel(i, k, alpha)) continue;
                    // a_k = alpha * a_i  =>  a_i x in [lb_k/alpha, ub_k/alpha]
                    double ilo = alpha > 0 ? rlb_[k] / alpha : rub_[k] / alpha;
                    double ihi = alpha > 0 ? rub_[k] / alpha : rlb_[k] / alpha;
                    Op op; op.kind = Op::DuplicateRow; op.i = i; op.k = k;
                    op.alpha = alpha; op.lb_i = rlb_[i]; op.ub_i = rub_[i];
                    op.impl_lb_k = ilo; op.impl_ub_k = ihi;
                    double nlb = std::max(rlb_[i], ilo), nub = std::min(rub_[i], ihi);
                    if (nlb > nub) {
                        if (nlb - nub > tol_scale(nub, opt_.feas_tol)) {
                            status_ = Status::Infeasible;
                            return true;
                        }
                        nlb = nub = 0.5 * (nlb + nub);
                    }
                    rlb_[i] = nlb;
                    rub_[i] = nub;
                    stack_.push_back(op);
                    remove_row(k);
                    changed = true;
                }
            }
        }
        return changed;
    }

    bool
    parallel(int i, int k, double &alpha)
    {
        const auto *rp = orig_.A.row_ptr();
        const auto *ci = orig_.A.col_ind();
        const double *v = orig_.A.values();
        int p = rp[i], q = rp[k];
        bool have = false;
        while (true) {
            while (p < rp[i + 1] && !col_on_[ci[p]]) ++p;
            while (q < rp[k + 1] && !col_on_[ci[q]]) ++q;
            if (p >= rp[i + 1] || q >= rp[k + 1]) break;
            if (ci[p] != ci[q]) return false;
            double r = v[q] / v[p];
            if (!have) { alpha = r; have = true; }
            else if (std::abs(r - alpha) > 1e-12 * std::abs(alpha)) return false;
            ++p; ++q;
        }
        while (p < rp[i + 1] && !col_on_[ci[p]]) ++p;
        while (q < rp[k + 1] && !col_on_[ci[q]]) ++q;
        return have && p >= rp[i + 1] && q >= rp[k + 1];
    }

    void
    run()
    {
        const size_t m = orig_.rows(), n = orig_.cols();
        At_ = orig_.A.transpose();
        rlb_ = orig_.row_lb; rub_ = orig_.row_ub;
        clb_ = orig_.col_lb; cub_ = orig_.col_ub;
        cost_ = orig_.c;
        offset_ = orig_.offset;
        row_on_.assign(m, 1);
        col_on_.assign(n, 1);
        row_cnt_.resize(m);
        col_cnt_.resize(n);
        for (size_t i = 0; i < m; ++i)
            row_cnt_[i] = orig_.A.row_ptr()[i + 1] - orig_.A.row_ptr()[i];
        for (size_t j = 0; j < n; ++j)
            col_cnt_[j] = At_.row_ptr()[j + 1] - At_.row_ptr()[j];
        for (size_t i = 0; i < m; ++i) rq_.push_back(static_cast<int>(i));
        for (size_t j = 0; j < n; ++j) cq_.push_back(static_cast<int>(j));

        while (status_ == Status::NotSolved) {
            while (status_ == Status::NotSolved && (!rq_.empty() || !cq_.empty())) {
                if (!rq_.empty()) {
                    int i = rq_.back(); rq_.pop_back();
                    do_row(i);
                } else {
                    int j = cq_.back(); cq_.pop_back();
                    do_col(j);
                }
            }
            if (status_ != Status::NotSolved) break;
            if (!(opt_.duplicate_rows && merge_duplicates())) break;
            // merged rows / new bounds may enable more reductions
            for (size_t i = 0; i < m; ++i)
                if (row_on_[i] && row_cnt_[i] <= 1) rq_.push_back(static_cast<int>(i));
        }
        build_reduced();
    }

    void
    build_reduced()
    {
        const size_t m = orig_.rows(), n = orig_.cols();
        std::vector<int> new_row(m, -1), new_col(n, -1);
        for (size_t i = 0; i < m; ++i)
            if (row_on_[i]) { new_row[i] = (int)row_map_.size(); row_map_.push_back((int)i); }
        for (size_t j = 0; j < n; ++j)
            if (col_on_[j]) { new_col[j] = (int)col_map_.size(); col_map_.push_back((int)j); }
        Sparse::CooBuilder<double> b(row_map_.size(), col_map_.size());
        for (size_t i = 0; i < m; ++i) {
            if (!row_on_[i]) continue;
            for (int k = orig_.A.row_ptr()[i]; k < orig_.A.row_ptr()[i + 1]; ++k)
                if (col_on_[orig_.A.col_ind()[k]])
                    b.add(new_row[i], new_col[orig_.A.col_ind()[k]], orig_.A.values()[k]);
        }
        reduced_.name = orig_.name + "_presolved";
        reduced_.A = b.build();
        for (int i : row_map_) { reduced_.row_lb.push_back(rlb_[i]); reduced_.row_ub.push_back(rub_[i]); }
        for (int j : col_map_) {
            reduced_.c.push_back(cost_[j]);
            reduced_.col_lb.push_back(clb_[j]);
            reduced_.col_ub.push_back(cub_[j]);
        }
        reduced_.offset = offset_;
        reduced_.maximize = orig_.maximize;
    }
};

} // namespace Solver
} // namespace AXOS
