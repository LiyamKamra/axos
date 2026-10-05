// SPDX-License-Identifier: BSD-3-Clause
//
// PDHCG: restarted primal-dual hybrid gradient for convex QP with an inexact
// primal subproblem (Y. Huang et al., arXiv:2405.16160), in the Halpern form
// of PDHCG-II.
//
//   saddle  min_{x in C} max_y  1/2 x^T Q x + c^T x - y^T A x + p_K(y)
//   x+  ~ argmin_{x in C} 1/2 x^T Q x + (c - A^T y)^T x + |x - x|^2 / (2 tau)
//   y+  = prox of p_K at y + sigma(... - A(2 x+ - x))       (PDLP dual step)
//   Halpern  z <- (t+1)/(t+2) (2 T(z) - z) + 1/(t+2) z0
//
// Unlike HPR-QP (and rAPDHG/PDQP), Q is not linearized, so the step sizes
// depend on |A| only: tau = eta / omega, sigma = eta omega, eta < 1 / |A|. The
// subproblem is solved by conjugate gradients when x has no finite bounds and
// by projected Barzilai-Borwein otherwise (the paper's switch), warm-started at
// x, up to a tolerance tied to the current KKT error; without Q it is the exact
// projection. Q x, A x and A^T y are carried along, so a termination check
// needs no extra products.
//
// Restarts (every opt.check_every iterations) compare the relative KKT error at
// T(z) with the one at the last restart (0.2 sufficient, 0.8 necessary,
// artificial after 0.2 of all iterations); the primal weight omega is
// re-balanced from the movement since the last restart. Preconditioning and
// termination as in HPR-QP (qp_scaling.h, qp_model.h).
#pragma once

#include "solver/qp/hpr_qp.h"
#include "solver/qp/qp_model.h"
#include "solver/qp/qp_ops.h"
#include "solver/qp/qp_scaling.h"
#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <vector>

namespace AXOS {
namespace Solver {
namespace qp {

template <class B> class Pdhcg {
    using Vec = typename B::Vec;

  public:
    int max_inner = 50; // inner (CG / BB) iterations per outer iteration

    QpSolution
    solve(const QpProblem &p, const QpOptions &opt)
    {
        using namespace axos_qp;
        const auto t_start = std::chrono::steady_clock::now();
        auto elapsed = [&] {
            return std::chrono::duration<double>(
                std::chrono::steady_clock::now() - t_start)
                .count();
        };
        const std::string bad = p.validate();
        if (!bad.empty()) throw std::invalid_argument("Pdhcg: " + bad);
        const size_t n = p.cols(), m = p.rows();

        QpScaling sc;
        QpProblem q;
        if (opt.scaling) {
            sc = compute_qp_scaling(
                p, opt.ruiz_iterations, opt.pock_chambolle_alpha);
            q = apply_qp_scaling(p, sc);
        } else {
            sc.col.assign(n, 1.0);
            sc.row.assign(m, 1.0);
            q = p;
        }
        B be;
        const bool hasQ = q.Q.nnz() > 0;
        bool bounded = false;
        for (size_t j = 0; j < n && !bounded; ++j)
            bounded =
                std::isfinite(q.lp.col_lb[j]) || std::isfinite(q.lp.col_ub[j]);
        const bool use_cg = hasQ && !bounded;
        auto A = be.upload(q.lp.A);
        auto AT = be.upload(q.lp.A.transpose());
        auto Qm = be.upload(q.Q);
        std::vector<double> cinv(n), rinv(m);
        for (size_t j = 0; j < n; ++j)
            cinv[j] = 1.0 / sc.col[j];
        for (size_t i = 0; i < m; ++i)
            rinv[i] = 1.0 / sc.row[i];
        Vec c = be.upload(q.lp.c), lb = be.upload(q.lp.col_lb),
            ub = be.upload(q.lp.col_ub);
        Vec l = be.upload(q.lp.row_lb), u = be.upload(q.lp.row_ub);
        Vec dcinv = be.upload(cinv), drinv = be.upload(rinv);

        // state z = (x, y) with Q x, A x, A^T y; anchors; T(z) outputs
        Vec x = be.vec(n), qx = be.vec(n), aty = be.vec(n), ax = be.vec(m),
            y = be.vec(m);
        Vec x0 = be.vec(n), qx0 = be.vec(n), aty0 = be.vec(n), ax0 = be.vec(m),
            y0 = be.vec(m);
        Vec xn = be.vec(n), qxn = be.vec(n), atyn = be.vec(n), axn = be.vec(m),
            yn = be.vec(m);
        Vec g = be.vec(n), s = be.vec(n), qs = be.vec(n), hp = be.vec(n);
        Vec r = use_cg ? be.vec(n) : be.vec(0),
            pdir = use_cg ? be.vec(n) : be.vec(0);
        Vec xr = be.vec(n),
            yr = be.vec(m); // last restart point (primal weight)
        auto P = [](Vec &v) { return B::ptr(v); };

        const double lamA =
            m ? power_method(be, A, &AT, n, m, opt.power_iterations) : 0.0;
        const double lamQ =
            hasQ ? power_method(be, Qm, nullptr, n, n, opt.power_iterations)
                 : 0.0;
        const double eta = lamA > 0 ? 0.998 / std::sqrt(1.01 * lamA) : 1.0;
        double omega = 1.0;
        {
            double nb = 0, nc = 0;
            for (size_t i = 0; i < m; ++i) {
                if (std::isfinite(q.lp.row_lb[i]))
                    nb += q.lp.row_lb[i] * q.lp.row_lb[i];
                if (std::isfinite(q.lp.row_ub[i]))
                    nb += q.lp.row_ub[i] * q.lp.row_ub[i];
            }
            for (double v : q.lp.c)
                nc += v * v;
            if (nb > 1e-20 && nc > 1e-20) omega = std::sqrt(nc) / std::sqrt(nb);
        }
        double bmax = 0, cmax = 0;
        for (size_t i = 0; i < m; ++i) {
            if (std::isfinite(p.lp.row_lb[i]))
                bmax = std::max(bmax, std::abs(p.lp.row_lb[i]));
            if (std::isfinite(p.lp.row_ub[i]))
                bmax = std::max(bmax, std::abs(p.lp.row_ub[i]));
        }
        for (double v : p.lp.c)
            cmax = std::max(cmax, std::abs(v));
        // x = P_C(0)
        {
            std::vector<double> h(n);
            for (size_t j = 0; j < n; ++j)
                h[j] = std::min(std::max(0.0, q.lp.col_lb[j]), q.lp.col_ub[j]);
            x = be.upload(h);
            if (hasQ) be.spmv(Qm, P(x), P(qx));
            if (m) be.spmv(A, P(x), P(ax));
        }
        auto copy = [&](Vec &dst, Vec &src, size_t len) {
            AXOS_QP_MAP(be, copy, len, (Copy{P(dst), P(src)}));
        };
        auto anchor = [&]() {
            copy(x0, x, n);
            copy(qx0, qx, n);
            copy(aty0, aty, n);
            copy(ax0, ax, m);
            copy(y0, y, m);
            copy(xr, x, n);
            copy(yr, y, m);
        };
        anchor();
        const double setup_s = elapsed();

        struct Kkt {
            double eta_p = 0, eta_d = 0, eta_gap = 0, pobj = 0, dobj = 0;
            double
            err() const
            { return std::max({eta_p, eta_d, eta_gap}); }
        };
        auto eval = [&](Vec &xv, Vec &qxv, Vec &axv, Vec &yv, Vec &atyv) {
            Kkt k;
            double pviol = 0, axmax = 0, drow = 0;
            if (m) {
                const auto rr = AXOS_QP_RED(be, kkt_rows, m,
                    (KktRows{P(axv), P(yv), P(l), P(u), P(drinv)}));
                pviol = rr[0];
                axmax = rr[1];
                drow = rr[2];
            }
            const auto cc = AXOS_QP_RED(be, pd_kkt_cols, n,
                (PdKktCols{
                    P(xv), P(qxv), P(atyv), P(c), P(lb), P(ub), P(dcinv)}));
            k.pobj = 0.5 * cc[3] + cc[4] + q.lp.offset;
            k.dobj = -0.5 * cc[3] + drow + cc[5] + q.lp.offset;
            k.eta_p = pviol / (1 + std::max(bmax, axmax));
            k.eta_d = cc[0] / (1 + std::max({cmax, cc[1], cc[2]}));
            k.eta_gap = std::abs(k.pobj - k.dobj) /
                        (1 + std::max(std::abs(k.pobj), std::abs(k.dobj)));
            return k;
        };
        long inner_total = 0;
        auto finish = [&](Status st, long iters, Vec &xv, Vec &yv,
                          const Kkt &k) {
            std::vector<double> hx, hy;
            be.download(xv, hx);
            be.download(yv, hy);
            for (size_t j = 0; j < n; ++j)
                hx[j] *= sc.col[j];
            for (size_t i = 0; i < m; ++i)
                hy[i] *= sc.row[i];
            QpSolution sol = evaluate_qp(p, hx, hy);
            sol.status = st;
            sol.iterations = iters;
            sol.seconds = elapsed();
            sol.setup_seconds = setup_s;
            sol.method = "pdhcg";
            sol.device = be.name();
            if (opt.verbose)
                std::printf("[pdhcg] %s after %ld iterations (%ld inner), %.3f "
                            "s (setup %.3f s): "
                            "obj %.10e  eta_p %.2e eta_d %.2e gap %.2e "
                            "(internal %.2e)\n",
                    to_string(st), iters, inner_total, sol.seconds, setup_s,
                    sol.primal_objective, sol.rel_primal, sol.rel_dual,
                    sol.rel_gap, k.err());
            return sol;
        };

        if (opt.verbose)
            std::printf("[pdhcg] %s: n %zu m %zu nnz(A) %zu nnz(Q) %zu, |A| "
                        "%.3e lambda_Q %.3e "
                        "omega0 %.3e inner %s, setup %.3f s\n",
                be.name().c_str(), n, m, q.lp.A.nnz(), q.Q.nnz(),
                std::sqrt(lamA), lamQ, omega,
                use_cg ? "CG" : (hasQ ? "projected BB" : "projection"),
                setup_s);

        long it = 0, t = 0;
        double kkt_restart = -1, kkt_prev = 1e300, kkt_last = 1.0;
        Kkt kkt;
        while (true) {
            const double tau = eta / omega, sigma = eta * omega,
                         inv_tau = 1.0 / tau;
            // ---- primal subproblem, warm-started at x
            // -------------------------------
            AXOS_QP_MAP(be, pd_start, n,
                (PdStart{P(x), P(qx), P(c), P(aty), P(xn), P(qxn), P(g),
                    use_cg ? P(r) : nullptr, use_cg ? P(pdir) : nullptr}));
            if (!hasQ) {
                // exact: x+ = P_C(x - tau (c - A^T y))
                AXOS_QP_MAP(be, pd_pg_step, n,
                    (PdPgStep{P(xn), P(s), P(g), P(lb), P(ub), tau}));
                ++inner_total;
            } else if (!use_cg) {
                // projected Barzilai-Borwein on the strongly convex subproblem
                const double L = lamQ * 1.01 + inv_tau, mu = inv_tau;
                double alpha = 1.0 / L;
                const double tol_in = 0.05 * kkt_last;
                for (int k = 0; k < max_inner; ++k) {
                    AXOS_QP_MAP(be, pd_pg_step, n,
                        (PdPgStep{P(xn), P(s), P(g), P(lb), P(ub), alpha}));
                    be.spmv(Qm, P(s), P(qs));
                    const auto rr = AXOS_QP_RED(be, pd_pg_update, n,
                        (PdPgUpdate{
                            P(g), P(qxn), P(s), P(qs), P(xn), inv_tau}));
                    ++inner_total;
                    if (rr[2] <= tol_in * (1.0 + rr[3]) || rr[0] <= 0) break;
                    alpha = rr[1] > 0 ? rr[0] / rr[1] : 1.0 / L;
                    alpha = std::min(std::max(alpha, 1.0 / L), 1.0 / mu);
                }
            } else {
                // conjugate gradients on (Q + I/tau) x = x/tau - c + A^T y
                double rho = AXOS_QP_RED(be, dot2, n, (Dot2{P(r), P(r)}))[1];
                const double tol_in = 0.05 * kkt_last;
                for (int k = 0; k < max_inner && rho > 0; ++k) {
                    be.spmv(Qm, P(pdir), P(qs));
                    const double php = AXOS_QP_RED(be, pd_cg_curv, n,
                        (PdCgCurv{P(pdir), P(qs), P(hp), inv_tau}))[0];
                    if (!(php > 0)) break;
                    const double alpha = rho / php;
                    const auto rr = AXOS_QP_RED(be, pd_cg_step, n,
                        (PdCgStep{P(xn), P(qxn), P(r), P(pdir), P(qs), P(hp),
                            alpha}));
                    ++inner_total;
                    if (rr[1] <= tol_in * (1.0 + rr[2])) break;
                    const double beta = rr[0] / rho;
                    rho = rr[0];
                    AXOS_QP_MAP(
                        be, pd_cg_dir, n, (PdCgDir{P(pdir), P(r), beta}));
                }
            }
            // ---- dual step
            // ---------------------------------------------------------------
            if (m) {
                be.spmv(A, P(xn), P(axn));
                AXOS_QP_MAP(be, pd_dual, m,
                    (PdDual{P(y), P(ax), P(axn), P(l), P(u), P(yn), sigma}));
                be.spmv(AT, P(yn), P(atyn));
            }
            const bool check = it % opt.check_every == 0;
            if (check) {
                kkt = eval(xn, qxn, axn, yn, atyn);
                kkt_last = std::max(kkt.err(), 1e-12);
                if (opt.verbose >= 2)
                    std::printf("[pdhcg] it %-8ld t %-7ld pobj % .10e dobj % "
                                ".10e eta_p %.2e "
                                "eta_d %.2e gap %.2e omega %.3e inner %ld\n",
                        it, t, kkt.pobj, kkt.dobj, kkt.eta_p, kkt.eta_d,
                        kkt.eta_gap, omega, inner_total);
                if (!std::isfinite(kkt.err()))
                    return finish(Status::NumericalError, it, xn, yn, kkt);
                if (kkt.err() <= opt.tol)
                    return finish(Status::Optimal, it + 1, xn, yn, kkt);
                if (it + 1 >= opt.max_iterations)
                    return finish(Status::IterationLimit, it + 1, xn, yn, kkt);
                if (elapsed() > opt.time_limit)
                    return finish(Status::TimeLimit, it + 1, xn, yn, kkt);
                if (kkt_restart < 0) kkt_restart = kkt.err();
                const double e = kkt.err();
                const bool suff = e <= 0.2 * kkt_restart;
                const bool nec = e <= 0.8 * kkt_restart && e > kkt_prev;
                const bool art = t > 0 && static_cast<double>(t) >=
                                              0.2 * static_cast<double>(it);
                kkt_prev = e;
                if (t > 0 && (suff || nec || art)) {
                    // restart at T(z); re-balance the primal weight
                    const double dx = std::sqrt(
                        AXOS_QP_RED(be, diff_sq, n, (DiffSq{P(xn), P(xr)}))[0]);
                    const double dy = m ? std::sqrt(AXOS_QP_RED(be, diff_sq, m,
                                              (DiffSq{P(yn), P(yr)}))[0])
                                        : 0.0;
                    if (dx > 1e-10 && dy > 1e-10)
                        omega = std::exp(
                            0.5 * std::log(dy / dx) + 0.5 * std::log(omega));
                    copy(x, xn, n);
                    copy(y, yn, m);
                    if (hasQ)
                        be.spmv(Qm, P(x), P(qx)); // exact products, no drift
                    if (m) {
                        be.spmv(A, P(x), P(ax));
                        be.spmv(AT, P(y), P(aty));
                    }
                    anchor();
                    kkt_restart = e;
                    kkt_prev = 1e300;
                    t = 0;
                    ++it;
                    continue;
                }
            }
            // ---- reflected Halpern step
            // --------------------------------------------------
            const double tt = static_cast<double>(t);
            const double w1 = (tt + 1.0) / (tt + 2.0), w0c = 1.0 / (tt + 2.0);
            AXOS_QP_MAP(be, pd_halpern_n, n,
                (PdHalpernN{P(x), P(qx), P(aty), P(xn), P(qxn), P(atyn), P(x0),
                    P(qx0), P(aty0), w1, w0c}));
            AXOS_QP_MAP(
                be, halpern, m, (Halpern{P(ax), P(axn), P(ax0), w1, w0c}));
            AXOS_QP_MAP(be, halpern, m, (Halpern{P(y), P(yn), P(y0), w1, w0c}));
            ++it;
            ++t;
        }
    }
};

} // namespace qp
} // namespace Solver
} // namespace AXOS
