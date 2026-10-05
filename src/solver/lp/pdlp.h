// SPDX-License-Identifier: BSD-3-Clause
//
// PDLP: restarted primal-dual hybrid gradient for LP (Applegate et al.,
// "Practical Large-Scale Linear Programming using Primal-Dual Hybrid
// Gradient"), with the KKT-based restart rule used by cuPDLP.
//
//   min c^T x  s.t.  l <= A x <= u,  lb <= x <= ub
//   saddle form  L(x, y) = c^T x - y^T A x + sum_i phi_i(y_i),
//                phi_i(y) = l_i y (y > 0),  u_i y (y < 0)
//
// One iteration is an x projection, an SpMV with A, a closed-form y update
//   y+ = max(0, y + s (l - t)) + min(0, y + s (u - t)),  t = 2 A x+ - A x
// and an SpMV with A^T. Step sizes adapt to the local Lipschitz estimate,
// iterates are averaged, and the solver restarts from the better of the
// current and averaged iterate when the KKT error drops enough. The primal
// weight is rebalanced at every restart. Termination uses the tolerances in
// SolverOptions on the ORIGINAL (unscaled) problem. Infeasibility and
// unboundedness are detected from normalized iterate differences.
//
// Runs on the CPU (Store = Cpu::HostStorage) or the GPU (Cuda::CudaStorage,
// include pdlp_cuda.h first through solver.h).
#pragma once

#include "solver/lp/pdlp_kernels.h"
#include "solver/model.h"
#include "solver/presolve/presolve.h"
#include "solver/scaling.h"
#include "sparse/sparse.h"
#include "tensorET.h"
#include <chrono>
#include <cstdio>

namespace AXOS {
namespace Solver {

template <template <typename> class Store = Cpu::HostStorage> class Pdlp {
    using VS = Store<double>;
    using Vec = tensorET<1, double, VS>;
    using Mat = Sparse::Csr<double, int32_t, Store>;
    using Par = Parallel<typename VS::backend_type>;

    struct Kkt {
        double pr_scaled = 0, pr_orig = 0, dr_scaled = 0, dr_orig = 0;
        double pobj = 0, dobj = 0, gap = 0, err = 0, bound = 0;
    };

  public:
    // Optional starting point (original scale) and termination mode for the
    // polishing sub-solves: 0 full KKT, 1 primal residual only, 2 dual residual only.
    struct Start {
        const std::vector<double> *x = nullptr, *y = nullptr;
        int mode = 0;
    };

    // Solves p (no presolve). Requires at least one row and one column. With
    // opt.pdlp_polish and a tight tolerance, PDLP runs to a looser tolerance
    // first and then polishes: a feasibility solve for x (objective dropped)
    // and one for y (right-hand sides and bounds zeroed) from that point, which
    // converge much faster than continuing on the full problem.
    LpSolution
    solve(const LpProblem &p, const SolverOptions &opt)
    {
        const double tight = std::min({opt.eps_primal, opt.eps_dual, opt.eps_gap});
        if (!opt.pdlp_polish || tight >= 5e-6) return solve_core(p, opt, nullptr);
        const auto t0 = std::chrono::steady_clock::now();
        auto since = [&] {
            return std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count();
        };
        SolverOptions loose = opt;
        loose.set_tolerance(std::max(tight, 1e-4));
        LpSolution main = solve_core(p, loose, nullptr);
        if (main.status != Status::Optimal) return main;
        // primal feasibility from x (objective dropped)
        LpProblem pp = p;
        std::fill(pp.c.begin(), pp.c.end(), 0.0);
        pp.offset = 0;
        SolverOptions o1 = opt;
        o1.time_limit = std::max(1.0, opt.time_limit - since());
        std::vector<double> y0(p.rows(), 0.0);
        Start s1{&main.x, &y0, 1};
        LpSolution a = solve_core(pp, o1, &s1);
        // dual feasibility from y (b and finite bounds zeroed, objective kept)
        LpProblem pd = p;
        for (double &v : pd.row_lb) if (std::isfinite(v)) v = 0;
        for (double &v : pd.row_ub) if (std::isfinite(v)) v = 0;
        for (double &v : pd.col_lb) if (std::isfinite(v)) v = 0;
        for (double &v : pd.col_ub) if (std::isfinite(v)) v = 0;
        pd.offset = 0;
        SolverOptions o2 = opt;
        o2.time_limit = std::max(1.0, opt.time_limit - since());
        std::vector<double> x0(p.cols(), 0.0);
        Start s2{&x0, &main.y, 2};
        LpSolution b = solve_core(pd, o2, &s2);
        long iters = main.iterations + a.iterations + b.iterations;
        if (a.status == Status::Optimal || a.status == Status::IterationLimit ||
            a.status == Status::TimeLimit) {
            // x from a, y from b; keep the better-gap combination
            LpSolution c = evaluate_solution(p, a.x, b.y);
            const double nbn = norm_b(p), ncn = norm_c(p);
            const bool ok = c.primal_residual <= opt.eps_primal * (1 + nbn) &&
                            c.dual_residual <= opt.eps_dual * (1 + ncn) &&
                            c.gap <= opt.eps_gap * (1 + std::abs(c.primal_objective) + std::abs(c.dual_objective)) &&
                            c.error_bound <= 10 * opt.eps_gap * (1 + std::abs(c.primal_objective));
            if (ok) {
                c.status = Status::Optimal;
                c.iterations = iters;
                c.seconds = since();
                return c;
            }
        }
        // polishing did not reach the tolerance: continue from the loose solution
        SolverOptions o3 = opt;
        o3.time_limit = std::max(1.0, opt.time_limit - since());
        Start s3{&main.x, &main.y, 0};
        LpSolution r = solve_core(p, o3, &s3);
        r.iterations += iters;
        r.seconds = since();
        return r;
    }

  private:
    static double
    norm_b(const LpProblem &p)
    {
        double nb = 0;
        for (size_t i = 0; i < p.rows(); ++i) {
            if (std::isfinite(p.row_lb[i])) nb += p.row_lb[i] * p.row_lb[i];
            if (std::isfinite(p.row_ub[i])) nb += p.row_ub[i] * p.row_ub[i];
        }
        return std::sqrt(nb);
    }
    static double
    norm_c(const LpProblem &p)
    {
        double nc = 0;
        for (double v : p.c) nc += v * v;
        return std::sqrt(nc);
    }

    LpSolution
    solve_core(const LpProblem &p, const SolverOptions &opt, const Start *start)
    {
        using namespace pdlp;
        const auto t_start = std::chrono::steady_clock::now();
        auto elapsed = [&] {
            return std::chrono::duration<double>(
                std::chrono::steady_clock::now() - t_start)
                .count();
        };
        const size_t m = p.rows(), n = p.cols();
        if (m == 0 || n == 0)
            throw std::invalid_argument(
                "Pdlp::solve needs rows and columns; use solve_lp()");
        std::string bad = p.validate();
        if (!bad.empty()) throw std::invalid_argument("Pdlp::solve: " + bad);

        // ---- scaling and device data ------------------------------------
        Scaling sc;
        LpProblem q;
        if (opt.scaling) {
            sc = compute_scaling(p.A, opt.ruiz_iterations, opt.pock_chambolle_alpha);
            q = apply_scaling(p, sc);
        } else {
            sc.row.assign(m, 1.0);
            sc.col.assign(n, 1.0);
            q = p;
        }
        Mat A(q.A);
        const Mat &AT = A.transposed();

        auto upload = [](const std::vector<double> &h) {
            tensorET<1, double> t({h.size()}, 0.0);
            std::memcpy(t.data, h.data(), h.size() * sizeof(double));
            return Vec(t);
        };
        std::vector<double> cinv(n), rinv(m);
        for (size_t j = 0; j < n; ++j) cinv[j] = 1.0 / sc.col[j];
        for (size_t i = 0; i < m; ++i) rinv[i] = 1.0 / sc.row[i];
        Vec c = upload(q.c), lb = upload(q.col_lb), ub = upload(q.col_ub);
        Vec l = upload(q.row_lb), u = upload(q.row_ub);
        Vec dcinv = upload(cinv), drinv = upload(rinv);
        Vec dcol = upload(sc.col), drow = upload(sc.row);

        Vec X0({n}, 0.0), X1({n}, 0.0), Xr({n}, 0.0), Sx({n}, 0.0);
        Vec Aty0({n}, 0.0), Aty1({n}, 0.0), Atyr({n}, 0.0), Saty({n}, 0.0);
        Vec Y0({m}, 0.0), Y1({m}, 0.0), Yr({m}, 0.0), Sy({m}, 0.0);
        Vec Ax0({m}, 0.0), Ax1({m}, 0.0), Axr({m}, 0.0), Sax({m}, 0.0);
        Vec *x = &X0, *xn = &X1, *y = &Y0, *yn = &Y1;
        Vec *ax = &Ax0, *axn = &Ax1, *aty = &Aty0, *atyn = &Aty1;

        Par par;

        // ---- problem norms for tolerances and the initial primal weight ---
        double nc = 0, nb = 0;
        for (size_t j = 0; j < n; ++j) nc += p.c[j] * p.c[j];
        for (size_t i = 0; i < m; ++i) {
            if (std::isfinite(p.row_lb[i])) nb += p.row_lb[i] * p.row_lb[i];
            if (std::isfinite(p.row_ub[i])) nb += p.row_ub[i] * p.row_ub[i];
        }
        nc = std::sqrt(nc);
        nb = std::sqrt(nb);
        double qc = 0, qb = 0, amax = 0; // scaled-space norms
        for (size_t j = 0; j < n; ++j) qc += q.c[j] * q.c[j];
        for (size_t i = 0; i < m; ++i) {
            if (std::isfinite(q.row_lb[i])) qb += q.row_lb[i] * q.row_lb[i];
            if (std::isfinite(q.row_ub[i])) qb += q.row_ub[i] * q.row_ub[i];
        }
        for (size_t k = 0; k < q.A.nnz(); ++k)
            amax = std::max(amax, std::abs(q.A.values()[k]));
        qc = std::sqrt(qc);
        qb = std::sqrt(qb);
        double omega = (qc > 1e-10 && qb > 1e-10) ? qc / qb : 1.0;
        double eta = amax > 0 ? 1.0 / amax : 1.0;

        // Halpern PDHG uses fixed steps: eta = 0.998 / ||A||_2 (power iteration)
        if (opt.pdlp_halpern) {
            Vec &v = X1, &w = Ax1, &t = Aty1;
            par.for_each(n, pdlp::Fill{v.data, 1.0});
            double sig = 1.0;
            for (int k = 0; k < 60; ++k) {
                Sparse::spmv(A, v, w);
                Sparse::spmv(AT, w, t);
                par.zero();
                par.reduce(0, n, SqNormN{t.data});
                par.reduce(1, n, SqNormN{v.data});
                const double *s = par.fetch();
                if (s[1] <= 0 || s[0] <= 0) break;
                sig = std::sqrt(std::sqrt(s[0] / s[1])); // ||A^T A v|| / ||v|| ~ sigma_max^2
                const double inv = 1.0 / std::sqrt(s[0]);
                par.for_each(n, pdlp::ScaleCopy{v.data, t.data, inv});
            }
            eta = 0.998 / (1.05 * sig);
            par.for_each(n, pdlp::Fill{X1.data, 0.0});
            par.for_each(m, pdlp::Fill{Ax1.data, 0.0});
            par.for_each(n, pdlp::Fill{Aty1.data, 0.0});
        }

        // ---- initial point: x = proj(0), y = 0, or the given start -----------
        {
            tensorET<1, double> h({n}, 0.0);
            for (size_t j = 0; j < n; ++j) {
                const double v = (start && start->x) ? (*start->x)[j] / sc.col[j] : 0.0;
                h.data[j] = std::min(std::max(v, q.col_lb[j]), q.col_ub[j]);
            }
            Vec init(h);
            par.copy(x->data, init.data, n);
            if (start && start->y) {
                tensorET<1, double> hy({m}, 0.0);
                for (size_t i = 0; i < m; ++i) hy.data[i] = (*start->y)[i] / sc.row[i];
                Vec inity(hy);
                par.copy(y->data, inity.data, m);
            }
        }
        Sparse::spmv(A, *x, *ax);
        if (start && start->y) Sparse::spmv(AT, *y, *aty);
        snapshot(par, n, m, *x, *y, *ax, *aty, Xr, Yr, Axr, Atyr);
        double se = 0;

        auto eval = [&](const Vec &xv, const Vec &yv, const Vec &axv,
                        const Vec &atyv) {
            par.zero();
            par.reduce(0, m, KktM{axv.data, yv.data, l.data, u.data, drinv.data});
            par.reduce(3, n, KktN{xv.data, atyv.data, c.data, lb.data, ub.data,
                                dcinv.data});
            const double *s = par.fetch();
            Kkt k;
            k.pr_scaled = std::sqrt(s[0]);
            k.pr_orig = std::sqrt(s[1]);
            k.dr_scaled = std::sqrt(s[3]);
            k.dr_orig = std::sqrt(s[4]);
            k.pobj = s[6] + p.offset;
            k.dobj = s[2] + s[5] + p.offset;
            k.gap = std::abs(k.pobj - k.dobj);
            // objective-error bound in the original scale (see LpSolution)
            par.zero();
            par.reduce(0, n, WSqNorm{xv.data, dcol.data});
            par.reduce(1, m, WSqNorm{yv.data, drow.data});
            const double *w = par.fetch();
            k.bound = k.gap + k.dr_orig * std::sqrt(w[0]) + k.pr_orig * std::sqrt(w[1]);
            k.err = std::sqrt(omega * omega * k.pr_scaled * k.pr_scaled +
                              k.dr_scaled * k.dr_scaled / (omega * omega) +
                              k.gap * k.gap);
            return k;
        };
        const int mode = start ? start->mode : 0;
        auto converged = [&](const Kkt &k) {
            if (mode == 1) return k.pr_orig <= opt.eps_primal * (1 + nb);
            if (mode == 2) return k.dr_orig <= opt.eps_dual * (1 + nc);
            return k.pr_orig <= opt.eps_primal * (1 + nb) &&
                   k.dr_orig <= opt.eps_dual * (1 + nc) &&
                   k.gap <= opt.eps_gap * (1 + std::abs(k.pobj) + std::abs(k.dobj)) &&
                   k.bound <= 10 * opt.eps_gap * (1 + std::abs(k.pobj));
        };

        auto finish = [&](Status st, const Vec &fx, const Vec &fy, long iters) {
            tensorET<1, double> hx(fx), hy(fy);
            std::vector<double> xv(hx.data, hx.data + n), yv(hy.data, hy.data + m);
            unscale_solution(xv, yv, sc);
            LpSolution s = evaluate_solution(p, xv, yv);
            s.status = st;
            s.iterations = iters;
            s.seconds = elapsed();
            return s;
        };

        Kkt cur = eval(*x, *y, *ax, *aty);
        double last_restart_err = cur.err;
        double prev_cand_err = kInf;
        long it = 0, since_restart = 0;
        const double eps_inf = 1e-8;
        // Minimum objective improvement of a certificate ray (unit norm),
        // relative to the problem's scale and the requested accuracy.
        const double cert_margin_dual = 10 * opt.eps_primal * (1 + qb);
        const double cert_margin_primal = 10 * opt.eps_dual * (1 + qc);

        while (true) {
            // Halpern PDHG: the output point is T(z) (feasible for the bounds), not z
            Vec *cx = x, *cy = y, *cax = ax, *caty = aty;
            if (opt.pdlp_halpern && it > 0) { cx = xn; cy = yn; cax = axn; caty = atyn; }
            if (it % opt.check_frequency == 0) {
                cur = eval(*cx, *cy, *cax, *caty);
                if (opt.verbose)
                    std::printf("[pdlp] it %-8ld pobj % .8e dobj % .8e "
                                "pres %.2e dres %.2e gap %.2e omega %.2e eta %.2e\n",
                        it, cur.pobj, cur.dobj, cur.pr_orig, cur.dr_orig, cur.gap,
                        omega, eta);
                if (!std::isfinite(cur.err))
                    return finish(Status::NumericalError, *cx, *cy, it);

                Kkt avg;
                const bool have_avg = se > 0;
                if (have_avg) {
                    const double inv = 1.0 / se;
                    par.for_each(n, ScaleCopy{xn->data, Sx.data, inv});
                    par.for_each(n, ScaleCopy{atyn->data, Saty.data, inv});
                    par.for_each(m, ScaleCopy{yn->data, Sy.data, inv});
                    par.for_each(m, ScaleCopy{axn->data, Sax.data, inv});
                    avg = eval(*xn, *yn, *axn, *atyn);
                }
                if (have_avg && converged(avg) &&
                    (!converged(cur) || avg.err < cur.err))
                    return finish(Status::Optimal, *xn, *yn, it);
                if (converged(cur)) return finish(Status::Optimal, *cx, *cy, it);

                // infeasibility / unboundedness certificates
                if (since_restart > 0) {
                    par.zero();
                    par.reduce(0, m, DualRayM{cy->data, Yr.data, l.data, u.data});
                    par.reduce(3, n, DualRayN{caty->data, Atyr.data, lb.data, ub.data});
                    par.reduce(5, n, PrimalRayN{cx->data, Xr.data, c.data, lb.data, ub.data});
                    par.reduce(8, m, PrimalRayM{cax->data, Axr.data, l.data, u.data});
                    const double *s = par.fetch();
                    const double dy2 = s[0], dx2 = s[5];
                    if (dy2 > 1e-20) {
                        const double ny = std::sqrt(dy2);
                        const double rho = (s[2] + s[4]) / ny;
                        const double viol = std::sqrt(s[1] + s[3]) / ny;
                        if (opt.verbose)
                            std::printf("[pdlp]   dual ray: |dy| %.3e rho %.3e viol %.3e margin %.3e\n",
                                ny, rho, viol, cert_margin_dual);
                        // The ray must improve the homogeneous dual objective by
                        // a real margin: redundant constraints give null
                        // directions whose objective is only round-off.
                        if (rho > cert_margin_dual && viol <= eps_inf * rho)
                            return finish(Status::Infeasible, *cx, *cy, it);
                    }
                    if (dx2 > 1e-20) {
                        const double nx = std::sqrt(dx2);
                        const double slope = s[6] / nx;
                        const double viol = std::sqrt(s[7] + s[8]) / nx;
                        if (-slope > cert_margin_primal && viol <= eps_inf * (-slope))
                            return finish(Status::Unbounded, *cx, *cy, it);
                    }
                }

                if (it >= opt.max_iterations)
                    return finish(Status::IterationLimit, *cx, *cy, it);
                if (elapsed() > opt.time_limit)
                    return finish(Status::TimeLimit, *cx, *cy, it);

                // restart?
                const bool use_avg = have_avg && avg.err < cur.err;
                const double cand_err = use_avg ? avg.err : cur.err;
                const bool suff = cand_err <= 0.2 * last_restart_err;
                const bool nec = cand_err <= 0.8 * last_restart_err &&
                                 cand_err > prev_cand_err;
                const bool art = since_restart >= 0.36 * static_cast<double>(it);
                if (it > 0 && (suff || nec || art)) {
                    if (opt.pdlp_halpern) { // restart from T(z)
                        par.copy(x->data, cx->data, n);
                        par.copy(y->data, cy->data, m);
                        par.copy(ax->data, cax->data, m);
                        par.copy(aty->data, caty->data, n);
                    } else if (use_avg) {
                        std::swap(x, xn);
                        std::swap(y, yn);
                        std::swap(ax, axn);
                        std::swap(aty, atyn);
                    }
                    // primal weight: balance the movement since last restart
                    par.zero();
                    par.reduce(0, n, DiffSq{x->data, Xr.data});
                    par.reduce(1, m, DiffSq{y->data, Yr.data});
                    const double *s = par.fetch();
                    const double dx = std::sqrt(s[0]), dy = std::sqrt(s[1]);
                    if (dx > 1e-10 && dy > 1e-10)
                        omega = std::exp(0.5 * std::log(dy / dx) + 0.5 * std::log(omega));
                    snapshot(par, n, m, *x, *y, *ax, *aty, Xr, Yr, Axr, Atyr);
                    par.for_each(n, pdlp::Fill{Sx.data, 0.0});
                    par.for_each(n, pdlp::Fill{Saty.data, 0.0});
                    par.for_each(m, pdlp::Fill{Sy.data, 0.0});
                    par.for_each(m, pdlp::Fill{Sax.data, 0.0});
                    se = 0;
                    since_restart = 0;
                    last_restart_err = eval(*x, *y, *ax, *aty).err;
                    prev_cand_err = kInf;
                } else {
                    prev_cand_err = cand_err;
                }
            }

            if (opt.pdlp_halpern) {
                // ---- reflected Halpern PDHG step (fixed step size) ---------
                const double tau = eta / omega, sigma = eta * omega;
                par.zero();
                par.reduce(0, n, XUpdateStep{c.data, aty->data, lb.data, ub.data,
                                    x->data, xn->data, tau});
                Sparse::spmv(A, *xn, *axn);
                par.reduce(1, m, YUpdateStep{l.data, u.data, y->data, ax->data,
                                    axn->data, yn->data, sigma});
                Sparse::spmv(AT, *yn, *atyn);
                const double k1 = static_cast<double>(since_restart);
                const double w1 = (k1 + 1.0) / (k1 + 2.0), w0 = 1.0 / (k1 + 2.0);
                // z <- w1 (2 T(z) - z) + w0 z0, in place; T(z) stays in xn/yn/axn/atyn
                par.for_each(n, pdlp::HalpernMix{x->data, xn->data, Xr.data, w1, w0});
                par.for_each(m, pdlp::HalpernMix{y->data, yn->data, Yr.data, w1, w0});
                par.for_each(m, pdlp::HalpernMix{ax->data, axn->data, Axr.data, w1, w0});
                par.for_each(n, pdlp::HalpernMix{aty->data, atyn->data, Atyr.data, w1, w0});
                ++it;
                ++since_restart;
                continue;
            }

            // ---- one PDHG iteration with adaptive step size ---------------
            double eta_used = 0;
            bool accepted = false;
            for (int tries = 0; tries < 60 && !accepted; ++tries) {
                const double tau = eta / omega, sigma = eta * omega;
                par.zero();
                par.reduce(0, n, XUpdateStep{c.data, aty->data, lb.data, ub.data,
                                    x->data, xn->data, tau});
                Sparse::spmv(A, *xn, *axn);
                par.reduce(1, m, YUpdateStep{l.data, u.data, y->data, ax->data,
                                    axn->data, yn->data, sigma});
                Sparse::spmv(AT, *yn, *atyn);
                const double *s = par.fetch();
                const double dz = omega * s[0] + s[1] / omega;
                const double denom = 2.0 * std::abs(s[2]);
                if (!std::isfinite(dz) || !std::isfinite(denom))
                    return finish(Status::NumericalError, *x, *y, it);
                const double eta_bar = denom > 0 ? dz / denom : kInf;
                const double kk = static_cast<double>(it + 2);
                const double eta_prime =
                    std::min((1.0 - std::pow(kk, -0.3)) * eta_bar,
                        (1.0 + std::pow(kk, -0.6)) * eta);
                if (eta <= eta_bar) {
                    accepted = true;
                    eta_used = eta;
                }
                eta = eta_prime;
                if (!(eta > 0) || !std::isfinite(eta))
                    return finish(Status::NumericalError, *x, *y, it);
            }
            if (!accepted) return finish(Status::NumericalError, *x, *y, it);

            se += eta_used;
            par.for_each(n, AvgN{eta_used, xn->data, atyn->data, Sx.data, Saty.data});
            par.for_each(m, AvgM{eta_used, yn->data, axn->data, Sy.data, Sax.data});
            std::swap(x, xn);
            std::swap(y, yn);
            std::swap(ax, axn);
            std::swap(aty, atyn);
            ++it;
            ++since_restart;
        }
    }

  private:
    static void
    snapshot(Par &par, size_t n, size_t m, const Vec &x, const Vec &y,
        const Vec &ax, const Vec &aty, Vec &xr, Vec &yr, Vec &axr, Vec &atyr)
    {
        par.copy(xr.data, x.data, n);
        par.copy(yr.data, y.data, m);
        par.copy(axr.data, ax.data, m);
        par.copy(atyr.data, aty.data, n);
    }
};

} // namespace Solver
} // namespace AXOS
