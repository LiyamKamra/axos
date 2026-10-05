// SPDX-License-Identifier: BSD-3-Clause
//
// Complemented mixed-integer rounding (c-MIR) cuts from the model rows,
// after H. Marchand, L. Wolsey, "Aggregation and mixed integer rounding to
// solve MIPs", Operations Research 49 (2001).
//
// A base inequality sum_j a_j x_j <= b (a model row, or an aggregation of
// rows that eliminates continuous columns sitting between their bounds) is
// turned into a mixed-integer knapsack set:
//   - every continuous x_j is replaced through its closest bound, a simple
//     one or a variable bound x_j <= c y_k + d / x_j >= c y_k + d (y_k
//     integer; found in two-column rows): x_j = L + x'_j or U - x'_j with
//     x'_j >= 0; terms with positive coefficient on x' are dropped (a
//     relaxation), the others form s = sum |c'_j| x'_j >= 0;
//   - every integer x_j is complemented to its nearer bound: z_j = x_j - l_j
//     or u_j - x_j >= 0, integer;
// giving sum_j g_j z_j - s <= beta. For a divisor delta > 0, with
// f0 = frac(beta / delta), the MIR inequality
//   sum_j (floor(g_j/delta) + max(0, frac(g_j/delta) - f0) / (1 - f0)) z_j
//       - s / (delta (1 - f0)) <= floor(beta / delta)
// is valid; the divisors tried are the |g_j| of the integer columns strictly
// inside their bounds, then the best one halved up to three times. The cut is
// mapped back to x and kept when its efficacy (violation / norm) is at least
// 1e-4 and its coefficients are numerically sane.
//
// Aggregation: when a row yields no cut, the continuous column of the base
// inequality farthest from its bounds is eliminated with another row that
// contains it (an equality with any multiplier, an inequality on its finite
// side), up to max_aggr times. This is what turns flow-conservation rows
// with variable upper bounds into flow-cover-like cuts.
#pragma once

#include "solver/milp/cuts.h"
#include "solver/milp/milp_model.h"
#include <algorithm>
#include <chrono>
#include <cmath>
#include <vector>

namespace AXOS {
namespace Solver {
namespace milp {

class CmirSeparator {
  public:
    explicit CmirSeparator(const LpProblem &p) : p_(p), At_(p.A.transpose())
    {
        const size_t n = p.cols(), m = p.rows();
        vlb_.assign(n, VB());
        vub_.assign(n, VB());
        isint_.assign(n, 0);
        for (size_t j = 0; j < n; ++j)
            isint_[j] = is_int(p, j) ? 1 : 0;
        // variable bounds from rows  a x_j + e y_k  {<=, >=, =}  r
        const int32_t *rp = p.A.row_ptr(), *ci = p.A.col_ind();
        const double *va = p.A.values();
        for (size_t i = 0; i < m; ++i) {
            if (rp[i + 1] - rp[i] != 2) continue;
            int t0 = rp[i], t1 = rp[i] + 1;
            int j = ci[t0], k = ci[t1];
            double a = va[t0], e = va[t1];
            if (isint_[j] && !isint_[k]) {
                std::swap(j, k);
                std::swap(a, e);
            }
            if (isint_[j] || !isint_[k] || a == 0) continue;
            // a x_j + e y_k <= u  ->  x_j <= (-e/a) y_k + u/a  (a > 0), >= when
            // a < 0
            const double c = -e / a;
            if (std::abs(p.row_ub[i]) < 1e20) {
                const double d = p.row_ub[i] / a;
                if (a > 0)
                    set_vb(vub_[j], k, c, d);
                else
                    set_vb(vlb_[j], k, c, d);
            }
            if (std::abs(p.row_lb[i]) < 1e20) {
                const double d = p.row_lb[i] / a;
                if (a > 0)
                    set_vb(vlb_[j], k, c, d);
                else
                    set_vb(vub_[j], k, c, d);
            }
        }
        dense_.assign(n, 0.0);
        g_.assign(n, 0.0);
        cutv_.assign(n, 0.0);
        inrow_.assign(n, 0);
        used_.assign(m, 0);
    }

    // Separates c-MIR cuts at x (the LP point over the columns) for the
    // domain lb/ub, from the model rows; appends at most max_cuts cuts with
    // efficacy >= min_eff to out. Stops at the deadline (steady clock).
    int
    separate(const std::vector<double> &x, const std::vector<double> &lb,
        const std::vector<double> &ub, std::vector<Cut> &out, int max_cuts,
        std::chrono::steady_clock::time_point deadline, int max_aggr = 5,
        double min_eff = 1e-4)
    {
        const size_t m = p_.rows();
        const int32_t *rp = p_.A.row_ptr(), *ci = p_.A.col_ind();
        const double *va = p_.A.values();
        // start rows: rows with an integer column, nearly tight at x first
        std::vector<std::pair<double, int>> start;
        for (size_t i = 0; i < m; ++i) {
            bool has_int = false;
            double act = 0;
            for (int32_t t = rp[i]; t < rp[i + 1]; ++t) {
                act += va[t] * x[ci[t]];
                has_int = has_int || isint_[ci[t]];
            }
            if (!has_int && !(rp[i + 1] - rp[i] > 2)) continue;
            double slack = 1e300;
            if (std::abs(p_.row_ub[i]) < 1e20)
                slack = std::min(slack, p_.row_ub[i] - act);
            if (std::abs(p_.row_lb[i]) < 1e20)
                slack = std::min(slack, act - p_.row_lb[i]);
            const double rel = slack / (1.0 + std::abs(act));
            if (rel > 0.1) continue;
            start.emplace_back(
                rel + 1e-3 * (rp[i + 1] - rp[i]) / (1.0 + n_cols()),
                static_cast<int>(i));
        }
        std::sort(start.begin(), start.end());
        int found = 0;
        Cut cut;
        for (const auto &sr : start) {
            if (found >= max_cuts ||
                std::chrono::steady_clock::now() > deadline)
                break;
            const int r = sr.second;
            for (int side = 0; side < 2 && found < max_cuts; ++side) {
                // side 0: a x <= u; side 1: -a x <= -l
                const double bound = side == 0 ? p_.row_ub[r] : p_.row_lb[r];
                if (!(std::abs(bound) < 1e20)) continue;
                const double sgn = side == 0 ? 1.0 : -1.0;
                clear_row();
                add_row(r, sgn);
                rhs_ = sgn * bound;
                used_list_.clear();
                used_[r] = 1;
                used_list_.push_back(r);
                for (int aggr = 0; aggr <= max_aggr; ++aggr) {
                    if (mir(x, lb, ub, cut) && cut.efficacy >= min_eff) {
                        if (!duplicate(cut, out)) {
                            out.push_back(cut);
                            ++found;
                        }
                        break;
                    }
                    if (aggr == max_aggr || !aggregate(x, lb, ub)) break;
                }
                for (int u : used_list_)
                    used_[u] = 0;
            }
        }
        clear_row();
        return found;
    }

  private:
    struct VB {
        int k = -1;          // integer column y_k
        double c = 0, d = 0; // x_j <= c y_k + d (vub) or >= (vlb)
    };
    // one continuous term of the base inequality after substitution
    struct Sub {
        int j;
        int type;    // 0 lb, 1 ub, 2 vlb, 3 vub
        double coef; // coefficient of x' (x' >= 0)
    };

    const LpProblem &p_;
    HostMatrix At_;
    std::vector<VB> vlb_, vub_;
    std::vector<uint8_t> isint_, inrow_, used_;
    std::vector<double> dense_, g_, cutv_;
    std::vector<int> nz_, gnz_, cnz_, used_list_;
    double rhs_ = 0;

    size_t
    n_cols() const
    { return p_.cols(); }

    static void
    set_vb(VB &v, int k, double c, double d)
    {
        if (v.k < 0) {
            v.k = k;
            v.c = c;
            v.d = d;
        }
    }

    void
    clear_row()
    {
        for (int j : nz_) {
            dense_[j] = 0;
            inrow_[j] = 0;
        }
        nz_.clear();
        rhs_ = 0;
    }

    void
    add_row(int r, double mult)
    {
        const int32_t *rp = p_.A.row_ptr(), *ci = p_.A.col_ind();
        const double *va = p_.A.values();
        for (int32_t t = rp[r]; t < rp[r + 1]; ++t) {
            const int j = ci[t];
            if (!inrow_[j]) {
                inrow_[j] = 1;
                nz_.push_back(j);
            }
            dense_[j] += mult * va[t];
        }
    }

    // Eliminates the continuous column of the base inequality farthest from
    // its (variable) bounds with an unused row; false if none applies.
    bool
    aggregate(const std::vector<double> &x, const std::vector<double> &lb,
        const std::vector<double> &ub)
    {
        int best_j = -1;
        double best_d = 1e-6;
        for (int j : nz_) {
            if (isint_[j] || std::abs(dense_[j]) < 1e-12) continue;
            double lo = lb[j], hi = ub[j];
            if (vlb_[j].k >= 0)
                lo = std::max(lo, vlb_[j].c * x[vlb_[j].k] + vlb_[j].d);
            if (vub_[j].k >= 0)
                hi = std::min(hi, vub_[j].c * x[vub_[j].k] + vub_[j].d);
            const double d = std::min(std::abs(lo) < 1e20 ? x[j] - lo : 1e20,
                std::abs(hi) < 1e20 ? hi - x[j] : 1e20);
            if (d > best_d) {
                best_d = d;
                best_j = j;
            }
        }
        if (best_j < 0) return false;
        const double cj = dense_[best_j];
        int best_r = -1;
        double best_score = 1e300, best_mult = 0;
        const int32_t *crp = At_.row_ptr(), *cri = At_.col_ind();
        const double *cva = At_.values();
        const int32_t *rp = p_.A.row_ptr();
        for (int32_t t = crp[best_j]; t < crp[best_j + 1]; ++t) {
            const int r = cri[t];
            if (used_[r]) continue;
            const double a = cva[t];
            if (std::abs(a) < 1e-9) continue;
            const double lam = -cj / a;
            const bool eq = p_.row_lb[r] == p_.row_ub[r];
            double side;
            if (eq)
                side = p_.row_ub[r];
            else if (lam > 0 && std::abs(p_.row_ub[r]) < 1e20)
                side = p_.row_ub[r];
            else if (lam < 0 && std::abs(p_.row_lb[r]) < 1e20)
                side = p_.row_lb[r];
            else
                continue;
            const double score = (eq ? 0 : 1) * 1e6 + (rp[r + 1] - rp[r]);
            if (score < best_score) {
                best_score = score;
                best_r = r;
                best_mult = lam;
            }
            (void)side;
        }
        if (best_r < 0) return false;
        const double side =
            (p_.row_lb[best_r] == p_.row_ub[best_r] || best_mult > 0)
                ? p_.row_ub[best_r]
                : p_.row_lb[best_r];
        add_row(best_r, best_mult);
        rhs_ += best_mult * side;
        dense_[best_j] = 0; // eliminated exactly
        used_[best_r] = 1;
        used_list_.push_back(best_r);
        return true;
    }

    // c-MIR cut from the base inequality sum dense_ x <= rhs_.
    bool
    mir(const std::vector<double> &x, const std::vector<double> &lb,
        const std::vector<double> &ub, Cut &cut)
    {
        // ---- substitute the continuous columns, collect integer coefficients
        double beta = rhs_;
        for (int j : gnz_)
            g_[j] = 0;
        gnz_.clear();
        auto addg = [&](int k, double v) {
            if (g_[k] == 0) gnz_.push_back(k);
            g_[k] += v;
            if (g_[k] == 0) g_[k] = 1e-300; // keep it listed
        };
        std::vector<Sub> subs;
        double sstar = 0; // value of s at x
        bool has_frac_int = false;
        for (int j : nz_) {
            const double a = dense_[j];
            if (std::abs(a) < 1e-12) continue;
            if (isint_[j]) {
                addg(j, a);
                continue;
            }
            // candidate bounds and their distances at x
            double dist[4] = {1e300, 1e300, 1e300, 1e300};
            if (std::abs(lb[j]) < 1e20) dist[0] = x[j] - lb[j];
            if (std::abs(ub[j]) < 1e20) dist[1] = ub[j] - x[j];
            if (vlb_[j].k >= 0)
                dist[2] = x[j] - (vlb_[j].c * x[vlb_[j].k] + vlb_[j].d);
            if (vub_[j].k >= 0)
                dist[3] = vub_[j].c * x[vub_[j].k] + vub_[j].d - x[j];
            // prefer the bound that keeps the term (negative x' coefficient)
            // when it is about as close; variable bounds win ties
            int type = -1;
            double best = 1e299;
            for (int t : {2, 3, 0, 1}) {
                if (dist[t] >= 1e299) continue;
                const double dd = std::max(dist[t], 0.0);
                if (dd < best - 1e-9) {
                    best = dd;
                    type = t;
                }
            }
            if (type < 0) return false; // free continuous column
            double coef;
            switch (type) {
            case 0:
                beta -= a * lb[j];
                coef = a;
                break;
            case 1:
                beta -= a * ub[j];
                coef = -a;
                break;
            case 2:
                beta -= a * vlb_[j].d;
                addg(vlb_[j].k, a * vlb_[j].c);
                coef = a;
                break;
            default:
                beta -= a * vub_[j].d;
                addg(vub_[j].k, a * vub_[j].c);
                coef = -a;
                break;
            }
            if (coef < 0) {
                subs.push_back({j, type, coef});
                sstar += -coef * std::max(0.0, dist[type]);
            }
        }
        // ---- complement the integer columns
        struct Z {
            int j;
            bool comp; // z = u - x (else x - l)
            double g, zs, range;
        };
        std::vector<Z> zs;
        zs.reserve(gnz_.size());
        for (int j : gnz_) {
            const double gj = std::abs(g_[j]) < 1e-12 ? 0.0 : g_[j];
            if (gj == 0) continue;
            const double l = lb[j], u = ub[j];
            const bool lf = std::abs(l) < 1e20, uf = std::abs(u) < 1e20;
            if (!lf && !uf) return false;
            const bool comp = uf && (!lf || x[j] > 0.5 * (l + u));
            Z z;
            z.j = j;
            z.comp = comp;
            if (comp) {
                beta -= gj * u;
                z.g = -gj;
                z.zs = u - x[j];
            } else {
                beta -= gj * l;
                z.g = gj;
                z.zs = x[j] - l;
            }
            z.range = (lf && uf) ? u - l : 1e300;
            if (z.zs > 1e-6 && z.zs < z.range - 1e-6) has_frac_int = true;
            zs.push_back(z);
        }
        if (zs.empty() || !has_frac_int) return false;
        // ---- divisors
        auto violation = [&](double delta, double &f0out) {
            const double bd = beta / delta;
            const double f0 = bd - std::floor(bd);
            f0out = f0;
            if (f0 < 0.05 || f0 > 0.95)
                return -1e300; // 0.999 (as SCIP) gave stronger bounds but
                               // slower trees here
            double lhs = 0, nrm = 0;
            for (const Z &z : zs) {
                const double gd = z.g / delta, fl = std::floor(gd),
                             fj = gd - fl;
                const double pi = fl + std::max(0.0, fj - f0) / (1 - f0);
                lhs += pi * z.zs;
                nrm += pi * pi;
            }
            const double sc = 1.0 / (delta * (1 - f0));
            lhs -= sc * sstar;
            for (const Sub &sb : subs)
                nrm += (sb.coef * sc) * (sb.coef * sc);
            return (lhs - std::floor(bd)) / std::sqrt(std::max(nrm, 1e-300));
        };
        std::vector<double> deltas;
        for (const Z &z : zs)
            if (z.zs > 1e-6 && z.zs < z.range - 1e-6 && std::abs(z.g) > 1e-6)
                deltas.push_back(std::abs(z.g));
        std::sort(deltas.begin(), deltas.end());
        deltas.erase(std::unique(deltas.begin(), deltas.end(),
                         [](double a, double b) {
                             return std::abs(a - b) <=
                                    1e-9 * std::max(1.0, std::abs(a));
                         }),
            deltas.end());
        if (deltas.size() > 8) deltas.erase(deltas.begin(), deltas.end() - 8);
        deltas.push_back(1.0);
        double best_v = 1e-6, best_delta = 0, f0;
        for (double dl : deltas) {
            const double v = violation(dl, f0);
            if (v > best_v) {
                best_v = v;
                best_delta = dl;
            }
        }
        if (best_delta == 0) return false;
        for (int h = 0; h < 3; ++h) {
            const double dl = best_delta / 2;
            const double v = violation(dl, f0);
            if (v > best_v + 1e-9) {
                best_v = v;
                best_delta = dl;
            } else {
                break;
            }
        }
        // ---- the cut in x:  sum pi_z z - sc s <= floor(beta/delta)
        const double delta = best_delta;
        const double bd = beta / delta;
        f0 = bd - std::floor(bd);
        const double sc = 1.0 / (delta * (1 - f0));
        double rhs = std::floor(bd);
        for (int j : cnz_)
            cutv_[j] = 0;
        cnz_.clear();
        auto addc = [&](int j, double v) {
            if (cutv_[j] == 0) cnz_.push_back(j);
            cutv_[j] += v;
            if (cutv_[j] == 0) cutv_[j] = 1e-300;
        };
        for (const Z &z : zs) {
            const double gd = z.g / delta, fl = std::floor(gd), fj = gd - fl;
            const double pi = fl + std::max(0.0, fj - f0) / (1 - f0);
            if (pi == 0) continue;
            if (z.comp) { // z = u - x
                addc(z.j, -pi);
                rhs -= pi * ub[z.j];
            } else { // z = x - l
                addc(z.j, pi);
                rhs += pi * lb[z.j];
            }
        }
        for (const Sub &sb : subs) {
            // term  -sc * |coef| * x'  with x' >= 0
            const double k =
                -sc * (-sb.coef); // coefficient of x' in the cut (negative)
            const int j = sb.j;
            switch (sb.type) {
            case 0:
                addc(j, k);
                rhs += k * lb[j];
                break; // x' = x - l
            case 1:
                addc(j, -k);
                rhs -= k * ub[j];
                break; // x' = u - x
            case 2:
                addc(j, k);
                addc(vlb_[j].k, -k * vlb_[j].c);
                rhs += k * vlb_[j].d;
                break; // x' = x - c y - d
            default:
                addc(j, -k);
                addc(vub_[j].k, k * vub_[j].c);
                rhs -= k * vub_[j].d;
                break; // x' = c y + d - x
            }
        }
        // ---- cleanup: tiny coefficients relaxed through the bounds, then
        // checks
        double cmax = 0;
        for (int j : cnz_)
            cmax = std::max(cmax, std::abs(cutv_[j]));
        if (cmax < 1e-9) return false;
        double act = 0, nrm = 0, cmin = 1e300;
        cut.idx.clear();
        cut.coef.clear();
        for (int j : cnz_) {
            double v = cutv_[j];
            if (std::abs(v) < 1e-9 * cmax) {
                // v x_j <= ... : drop the term using the bound that keeps
                // validity
                if (v > 0) {
                    if (!(std::abs(lb[j]) < 1e20)) return false;
                    rhs -= v * lb[j];
                } else if (v < 0) {
                    if (!(std::abs(ub[j]) < 1e20)) return false;
                    rhs -= v * ub[j];
                }
                continue;
            }
            cut.idx.push_back(j);
            cut.coef.push_back(-v); // >= form
            act += v * x[j];
            nrm += v * v;
            cmin = std::min(cmin, std::abs(v));
        }
        if (cut.idx.empty() || cmax / cmin > 1e6) return false;
        const double viol = act - rhs;
        nrm = std::sqrt(nrm);
        cut.efficacy = viol / nrm;
        cut.rhs = -rhs;
        if (!(cut.efficacy > 0) || !std::isfinite(cut.rhs)) return false;
        // scale to a largest coefficient of 1
        const double s = 1.0 / cmax;
        for (double &v : cut.coef)
            v *= s;
        cut.rhs *= s;
        return true;
    }

    // a cut nearly parallel to one already found
    static bool
    duplicate(const Cut &c, const std::vector<Cut> &out)
    {
        for (size_t t = out.size() > 50 ? out.size() - 50 : 0; t < out.size();
            ++t) {
            const Cut &o = out[t];
            if (o.idx.size() != c.idx.size()) continue;
            bool same = true;
            for (size_t k = 0; k < c.idx.size() && same; ++k)
                same = o.idx[k] == c.idx[k] &&
                       std::abs(o.coef[k] - c.coef[k]) <= 1e-9;
            if (same && std::abs(o.rhs - c.rhs) <= 1e-9) return true;
        }
        return false;
    }
};

} // namespace milp
} // namespace Solver
} // namespace AXOS
