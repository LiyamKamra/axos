// SPDX-License-Identifier: BSD-3-Clause
//
// Gomory mixed-integer (GMI) cuts from the optimal simplex tableau (MILP root).
//
// A tableau row of a basic integer column x_k with fractional value reads
//     x_k + sum_{j nonbasic} a_j v_j = 0
// (v_j a column, or the activity w_i = (A x)_i of row i, see
// DualSimplex::tableau_row). Writing each nonbasic through its distance to the
// bound it sits at, s_j = v_j - L_j (at lower) or U_j - v_j (at upper), s_j >=
// 0, gives x_k + sum a'_j s_j = b with b = the current x_k and a'_j = +-a_j.
// With f0 = frac(b) the GMI inequality is
//     sum g_j s_j >= 1,
//     g_j = f_j / f0 or (1 - f_j) / (1 - f0)   (s_j integer, f_j = frac(a'_j))
//     g_j = a'_j / f0 or -a'_j / (1 - f0)       (s_j continuous, by sign)
// s_j is integer when v_j is an integer column with integral bounds; row
// activities count as continuous. Substituting back and expanding the row
// activities gives a cut sum pi_t x_t >= pi_0 valid for every integer point
// within the bounds it was derived from (the global root bounds).
//
// A cut is kept only if numerically safe: no free nonbasic in the row,
// coefficients within a 1e6 range (tiny ones moved to the right-hand side
// through the column bounds, keeping validity), and a violation by the LP point
// of at least 1e-4 relative to its norm (efficacy).
#pragma once

#include "solver/lp/simplex.h"
#include "solver/milp/milp_model.h"
#include <algorithm>
#include <cmath>
#include <unordered_map>
#include <vector>

namespace AXOS {
namespace Solver {
namespace milp {

struct Cut {
    std::vector<int> idx;
    std::vector<double> coef;
    double rhs = 0; // sum coef_t x_t >= rhs
    double efficacy = 0;
};

// GMI cut from basis position r of the last solve of `lp` by `spx`. lb/ub:
// the column bounds the cut must be valid for. x: the LP point (columns).
inline bool
gmi_cut(const DualSimplex &spx, const LpProblem &lp, int r,
    const std::vector<double> &lb, const std::vector<double> &ub,
    const std::vector<double> &x, Cut &cut, std::vector<int> &tidx,
    std::vector<double> &tcoef, std::vector<double> &dense,
    std::vector<int> &touched)
{
    const int n = static_cast<int>(lp.cols());
    const int k = spx.basic_at(r);
    if (k >= n || !is_int(lp, k)) return false;
    const double xk = spx.value_of(k);
    const double f0 = xk - std::floor(xk);
    if (f0 < 0.01 || f0 > 0.99) return false;
    spx.tableau_row(r, tidx, tcoef);
    // cut in the nonbasic variables: sum g_j s_j >= 1
    // pi accumulates the coefficients over the columns, rhs the constant
    double rhs = 1.0;
    auto add = [&](int t, double v) {
        if (dense[t] == 0.0) touched.push_back(t);
        dense[t] += v;
        if (dense[t] == 0.0) dense[t] = 1e-300; // keep it marked as touched
    };
    touched.clear();
    const auto *rp = lp.A.row_ptr();
    const auto *ci = lp.A.col_ind();
    const double *va = lp.A.values();
    bool ok = true;
    for (size_t t = 0; t < tidx.size() && ok; ++t) {
        const int j = tidx[t];
        const double a = tcoef[t];
        double L, U;
        if (j < n) {
            L = lb[j];
            U = ub[j];
        } else {
            L = lp.row_lb[j - n];
            U = lp.row_ub[j - n];
        }
        if (L == U) continue; // fixed: s_j = 0
        const VarStatus st = spx.status_of(j);
        if (st != VarStatus::AtLower && st != VarStatus::AtUpper) {
            ok = false; // a free nonbasic: no valid GMI from this row
            break;
        }
        const bool at_lower = st == VarStatus::AtLower;
        const double bound = at_lower ? L : U;
        if (std::abs(bound) > 1e9) {
            ok = false;
            break;
        }
        const double ap = at_lower ? a : -a;
        const bool integer_s =
            j < n && is_int(lp, j) && bound == std::round(bound);
        double g;
        if (integer_s) {
            const double fj = ap - std::floor(ap);
            g = fj <= f0 ? fj / f0 : (1 - fj) / (1 - f0);
        } else {
            g = ap >= 0 ? ap / f0 : -ap / (1 - f0);
        }
        if (g == 0) continue;
        // g s_j = g (v_j - L) at lower, g (U - v_j) at upper
        const double sign = at_lower ? 1.0 : -1.0;
        rhs += sign * g * bound;
        if (j < n) {
            add(j, sign * g);
        } else {
            const int i = j - n;
            for (auto q = rp[i]; q < rp[i + 1]; ++q)
                add(ci[q], sign * g * va[q]);
        }
    }
    if (!ok) {
        for (int t : touched)
            dense[t] = 0.0;
        return false;
    }
    // collect, drop tiny coefficients through the bounds, check the range
    double mx = 0;
    for (int t : touched)
        mx = std::max(mx, std::abs(dense[t]));
    cut.idx.clear();
    cut.coef.clear();
    if (mx == 0) {
        for (int t : touched)
            dense[t] = 0.0;
        return false;
    }
    double mn = 1e300;
    for (int t : touched) {
        const double v = dense[t];
        dense[t] = 0.0;
        if (std::abs(v) < 1e-9 * mx) {
            // sum pi x >= rhs  ->  drop pi_t x_t: rhs -= max over the bounds of
            // pi_t x_t
            const double worst = v > 0 ? v * ub[t] : v * lb[t];
            if (!std::isfinite(worst) || std::abs(worst) > 1e12) {
                ok = false;
                continue;
            }
            rhs -= worst;
            continue;
        }
        cut.idx.push_back(t);
        cut.coef.push_back(v);
        mn = std::min(mn, std::abs(v));
    }
    if (!ok || cut.idx.empty() || mx / mn > 1e6) return false;
    cut.rhs = rhs;
    // violation and efficacy at the LP point
    double act = 0, norm2 = 0;
    for (size_t t = 0; t < cut.idx.size(); ++t) {
        act += cut.coef[t] * x[cut.idx[t]];
        norm2 += cut.coef[t] * cut.coef[t];
    }
    const double viol = rhs - act;
    cut.efficacy = viol / std::sqrt(norm2);
    if (viol <= 1e-6 * (1 + std::abs(rhs)) || cut.efficacy < 1e-4) return false;
    // scale to a unit largest coefficient
    const double s = 1.0 / mx;
    for (double &c : cut.coef)
        c *= s;
    cut.rhs *= s;
    return true;
}

// Appends cuts as rows (rhs <= row <= +inf) to lp.
inline void
append_cuts(LpProblem &lp, const std::vector<Cut> &cuts)
{
    if (cuts.empty()) return;
    const size_t m = lp.rows(), n = lp.cols();
    std::vector<int32_t> rp(lp.A.row_ptr(), lp.A.row_ptr() + m + 1);
    std::vector<int32_t> ci(lp.A.col_ind(), lp.A.col_ind() + lp.A.nnz());
    std::vector<double> va(lp.A.values(), lp.A.values() + lp.A.nnz());
    for (const Cut &c : cuts) {
        std::vector<std::pair<int, double>> e;
        for (size_t t = 0; t < c.idx.size(); ++t)
            e.emplace_back(c.idx[t], c.coef[t]);
        std::sort(e.begin(), e.end());
        for (auto &pr : e) {
            ci.push_back(pr.first);
            va.push_back(pr.second);
        }
        rp.push_back(static_cast<int32_t>(ci.size()));
        lp.row_lb.push_back(c.rhs);
        lp.row_ub.push_back(kInf);
    }
    lp.A = HostMatrix(m + cuts.size(), n, rp, ci, va);
}

// Keeps the rows i with keep[i] set (cut removal).
inline void
remove_rows(LpProblem &lp, const std::vector<char> &keep)
{
    const size_t m = lp.rows(), n = lp.cols();
    const int32_t *rp0 = lp.A.row_ptr(), *ci0 = lp.A.col_ind();
    const double *va0 = lp.A.values();
    std::vector<int32_t> rp(1, 0), ci;
    std::vector<double> va, rl, ru;
    for (size_t i = 0; i < m; ++i) {
        if (!keep[i]) continue;
        for (int32_t t = rp0[i]; t < rp0[i + 1]; ++t) {
            ci.push_back(ci0[t]);
            va.push_back(va0[t]);
        }
        rp.push_back(static_cast<int32_t>(ci.size()));
        rl.push_back(lp.row_lb[i]);
        ru.push_back(lp.row_ub[i]);
    }
    lp.A = HostMatrix(rl.size(), n, rp, ci, va);
    lp.row_lb.swap(rl);
    lp.row_ub.swap(ru);
    if (lp.row_names.size() == m) {
        std::vector<std::string> nm;
        for (size_t i = 0; i < m; ++i)
            if (keep[i]) nm.push_back(lp.row_names[i]);
        lp.row_names.swap(nm);
    }
}

} // namespace milp
} // namespace Solver
} // namespace AXOS
