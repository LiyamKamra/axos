// SPDX-License-Identifier: BSD-3-Clause
//
// Primal-dual interior-point method (Mehrotra predictor-corrector) for LP,
// with a regularized quasi-definite augmented KKT system solved by the sparse
// LDL^T solver (cuDSS on the GPU, up-looking LDL^T with AMD on the CPU).
//
//   min c^T x  s.t.  A x - w = 0,  lb <= x <= ub,  l <= w <= u
//
// with z = (x, w), lambda for the equalities and bound duals sl, su. Each
// iteration factors
//        [ -(Theta_x^{-1} + rho I)        A^T          ]
//    K = [           A            Theta_w + delta I    ]
// (Theta = (sl/tl + su/tu)^{-1}; rows with equal bounds have Theta_w = 0),
// whose pattern is fixed, so only the diagonal changes between iterations and
// the symbolic analysis is done once. Both the predictor and corrector solves
// reuse one factorization, and each solve is refined against the
// unregularized K. Free columns need no special handling (rho keeps K
// quasi-definite), which is why the augmented rather than the normal-equation
// system is used.
//
// Limitations: no infeasibility / unboundedness certificates (a failed run
// reports NumericalError or IterationLimit; use PDLP or LpMethod::Auto for
// those), no crossover.
#pragma once

#include "solver/lp/ipm_kernels.h"
#include "solver/lp/normal_kkt.h"
#include "solver/model.h"
#include "solver/presolve/presolve.h"
#include "solver/scaling.h"
#include "sparse/sparse.h"
#include "tensorET.h"
#include <chrono>
#include <cstdio>

namespace AXOS {
namespace Solver {

template <template <typename> class Store = Cpu::HostStorage> class Ipm {
    using VS = Store<double>;
    using Vec = tensorET<1, double, VS>;
    using Mat = Sparse::Csr<double, int32_t, Store>;
    using Par = Parallel<typename VS::backend_type>;
    using Ker = Sparse::Kernels<typename VS::backend_type>;
    using Ldl = Sparse::SparseLdlt<double, int32_t, Store>;

  public:
    // Solves p (no presolve). Requires at least one row and one column, no
    // free rows and no fixed columns (solve_lp() takes care of that).
    LpSolution
    solve(const LpProblem &p, const SolverOptions &opt)
    {
        using namespace ipm;
        const auto t_start = std::chrono::steady_clock::now();
        auto elapsed = [&] {
            return std::chrono::duration<double>(
                std::chrono::steady_clock::now() - t_start)
                .count();
        };
        const size_t m = p.rows(), n = p.cols(), N = n + m, N2 = n + m;
        if (m == 0 || n == 0)
            throw std::invalid_argument("Ipm::solve needs rows and columns");
        std::string bad = p.validate();
        if (!bad.empty()) throw std::invalid_argument("Ipm::solve: " + bad);

        // ---- scaling ----------------------------------------------------
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

        // ---- KKT pattern (host), positions of the diagonals -------------
        HostMatrix At = q.A.transpose();
        std::vector<int32_t> krp(N2 + 1, 0), kci, dxp(n), dwp(m);
        std::vector<double> kv;
        kci.reserve(2 * q.A.nnz() + N2);
        kv.reserve(2 * q.A.nnz() + N2);
        for (size_t j = 0; j < n; ++j) {
            dxp[j] = static_cast<int32_t>(kci.size());
            kci.push_back(static_cast<int32_t>(j));
            kv.push_back(0.0);
            for (int k = At.row_ptr()[j]; k < At.row_ptr()[j + 1]; ++k) {
                kci.push_back(static_cast<int32_t>(n) + At.col_ind()[k]);
                kv.push_back(At.values()[k]);
            }
            krp[j + 1] = static_cast<int32_t>(kci.size());
        }
        for (size_t i = 0; i < m; ++i) {
            for (int k = q.A.row_ptr()[i]; k < q.A.row_ptr()[i + 1]; ++k) {
                kci.push_back(q.A.col_ind()[k]);
                kv.push_back(q.A.values()[k]);
            }
            dwp[i] = static_cast<int32_t>(kci.size());
            kci.push_back(static_cast<int32_t>(n + i));
            kv.push_back(0.0);
            krp[n + i + 1] = static_cast<int32_t>(kci.size());
        }
        Mat K(Sparse::Csr<double, int32_t, Cpu::HostStorage>(N2, N2, krp, kci, kv));
        Mat A(q.A);
        const Mat &AT = A.transposed();
        Store<int32_t> dxpos(n), dwpos(m);
        Sparse::detail::host_to_store(dxpos, dxp.data(), n);
        Sparse::detail::host_to_store(dwpos, dwp.data(), m);

        // ---- host-side starting point --------------------------------------
        std::vector<double> hlb(N), hub(N), hz(N), htl(N, 1.0), htu(N, 1.0),
            hsl(N, 0.0), hsu(N, 0.0);
        for (size_t j = 0; j < n; ++j) { hlb[j] = q.col_lb[j]; hub[j] = q.col_ub[j]; }
        for (size_t i = 0; i < m; ++i) { hlb[n + i] = q.row_lb[i]; hub[n + i] = q.row_ub[i]; }
        double cinf = 0;
        for (double v : q.c) cinf = std::max(cinf, std::abs(v));
        // Starting scale: slacks to finite bounds and bound duals start at
        // about M (large starts keep the early steps from being blocked).
        const double M = 1e2;
        const double s0 = std::max(M, cinf);
        auto interior = [&](double lo, double hi, double target) {
            if (lo == hi) return lo;
            const bool lf = std::isfinite(lo), uf = std::isfinite(hi);
            if (lf && uf) {
                const double d = std::min(M, 0.5 * (hi - lo));
                return std::min(std::max(target, lo + d), hi - d);
            }
            if (lf) return std::max(target, lo + M);
            if (uf) return std::min(target, hi - M);
            return target;
        };
        for (size_t j = 0; j < n; ++j) hz[j] = interior(hlb[j], hub[j], 0.0);
        {
            std::vector<double> ax(m, 0.0);
            for (size_t i = 0; i < m; ++i)
                for (int k = q.A.row_ptr()[i]; k < q.A.row_ptr()[i + 1]; ++k)
                    ax[i] += q.A.values()[k] * hz[q.A.col_ind()[k]];
            for (size_t i = 0; i < m; ++i) hz[n + i] = interior(hlb[n + i], hub[n + i], ax[i]);
        }
        for (size_t j = 0; j < N; ++j) {
            if (bl(hlb[j], hub[j])) { htl[j] = hz[j] - hlb[j]; hsl[j] = s0; }
            if (bu(hlb[j], hub[j])) { htu[j] = hub[j] - hz[j]; hsu[j] = s0; }
        }

        auto upload = [](const std::vector<double> &h) {
            tensorET<1, double> t({h.size()}, 0.0);
            std::memcpy(t.data, h.data(), h.size() * sizeof(double));
            return Vec(t);
        };
        Vec z = upload(hz), tl = upload(htl), tu = upload(htu), sl = upload(hsl),
            su = upload(hsu), lbz = upload(hlb), ubz = upload(hub);
        Vec c = upload(q.c);
        Vec lam({m}, 0.0), dlam({m}, 0.0), dlam_a({m}, 0.0), rp({m}, 0.0),
            ax({m}, 0.0), thw({m}, 0.0), tmpm({m}, 0.0);
        Vec aty({n}, 0.0), tmpn({n}, 0.0);
        Vec rd({N}, 0.0), hzv({N}, 0.0), rcl({N}, 0.0), rcu({N}, 0.0), g({N}, 0.0),
            dz({N}, 0.0), dsl({N}, 0.0), dsu({N}, 0.0), dz_a({N}, 0.0),
            dsl_a({N}, 0.0), dsu_a({N}, 0.0);
        Vec rhs({N2}, 0.0), sol({N2}, 0.0), res({N2}, 0.0), dsol({N2}, 0.0);
        Par par;
        Ldl ldl(Sparse::Symmetry::Symmetric,
            opt.ipm_ordering == 1 ? Sparse::Ordering::NestedDissection
                                  : Sparse::Ordering::MinDegree);
        bool analyzed = false;
        std::unique_ptr<NormalKkt> nk; // normal-equations alternative (CPU)
        bool use_normal = false;

        // ---- norms for the relative termination tests --------------------
        double nb = 0, nc = 0;
        for (size_t i = 0; i < m; ++i) {
            if (std::isfinite(p.row_lb[i])) nb += p.row_lb[i] * p.row_lb[i];
            if (std::isfinite(p.row_ub[i])) nb += p.row_ub[i] * p.row_ub[i];
        }
        for (double v : p.c) nc += v * v;
        nb = std::sqrt(nb);
        nc = std::sqrt(nc);
        double qb = 0, qc = 0;
        for (size_t i = 0; i < m; ++i) {
            if (std::isfinite(q.row_lb[i])) qb += q.row_lb[i] * q.row_lb[i];
            if (std::isfinite(q.row_ub[i])) qb += q.row_ub[i] * q.row_ub[i];
        }
        for (double v : q.c) qc += v * v;
        qb = std::sqrt(qb);
        qc = std::sqrt(qc);

        double rho = 1e-8, delta = 1e-8; // static primal / dual regularization
        const int ref_steps = 3;         // refinement steps per KKT solve
        double tol_eff = std::min({opt.eps_primal, opt.eps_dual, opt.eps_gap});
        int tightened = 0, stall = 0;

        auto finish = [&](Status st, long iters) {
            tensorET<1, double> hx(z), hy(lam);
            std::vector<double> xv(hx.data, hx.data + n), yv(hy.data, hy.data + m);
            unscale_solution(xv, yv, sc);
            LpSolution s = evaluate_solution(p, xv, yv);
            s.status = st;
            s.iterations = iters;
            s.seconds = elapsed();
            return s;
        };

        // Solve K sol = r with the current factorization, refining against the
        // unregularized system.
        double last_ref_res = 0;
        auto solve_kkt = [&](Vec &r, Vec &out) {
            if (use_normal) nk->solve(r.data, out.data); else ldl.solve(r, out);
            for (int step = 0; step < ref_steps; ++step) {
                Ker::spmv(A, out.data, tmpm.data, 1.0, 0.0);
                Ker::spmv(AT, out.data + n, tmpn.data, 1.0, 0.0);
                par.for_each(N2, RefineRes{res.data, r.data, out.data, tmpn.data,
                                      tmpm.data, hzv.data, thw.data, n});
                par.zero();
                par.reduce(0, N2, SqNorm{res.data});
                par.reduce(1, N2, SqNorm{r.data});
                const double *s = par.fetch();
                last_ref_res = std::sqrt(s[0]) / (1 + std::sqrt(s[1]));
                if (std::sqrt(s[0]) <= 1e-13 * (1 + std::sqrt(s[1]))) return;
                if (use_normal) nk->solve(res.data, dsol.data); else ldl.solve(res, dsol);
                par.for_each(N2, Axpy{out.data, dsol.data, 1.0});
            }
        };

        for (long it = 0; it <= opt.ipm_max_iterations; ++it) {
            // ---- residuals, objectives, complementarity ---------------
            Ker::spmv(A, z.data, ax.data, 1.0, 0.0);
            Ker::spmv(AT, lam.data, aty.data, 1.0, 0.0);
            par.for_each(m, ResP{rp.data, z.data + n, ax.data});
            par.for_each(N, ResD{rd.data, c.data, aty.data, lam.data, sl.data,
                                su.data, lbz.data, ubz.data, n});
            par.zero();
            par.reduce(0, N, Stats{rd.data, z.data, c.data, lam.data, tl.data,
                                tu.data, sl.data, su.data, lbz.data, ubz.data, n});
            par.reduce(5, m, SqNorm{rp.data});
            const double *s = par.fetch();
            const double rd2 = s[0], cx = s[1], dobj = s[2] + p.offset,
                         comp = s[3], ncomp = s[4], rp2 = s[5];
            const double pobj = cx + p.offset;
            const double mu = ncomp > 0 ? comp / ncomp : 0.0;
            const double rel_p = std::sqrt(rp2) / (1 + qb);
            const double rel_d = std::sqrt(rd2) / (1 + qc);
            const double rel_g = std::abs(pobj - dobj) /
                                 (1 + std::abs(pobj) + std::abs(dobj));
            if (opt.verbose)
                std::printf("[ipm] it %-3ld pobj % .8e dobj % .8e rp %.1e rd %.1e "
                            "gap %.1e mu %.1e\n",
                    it, pobj, dobj, rel_p, rel_d, rel_g, mu);
            if (!std::isfinite(pobj) || !std::isfinite(dobj) || !std::isfinite(mu))
                return finish(Status::NumericalError, it);

            if (rel_p <= tol_eff && rel_d <= tol_eff && rel_g <= tol_eff) {
                LpSolution cand = finish(Status::Optimal, it);
                // accept only if the ORIGINAL problem meets the tolerances
                const double op = cand.primal_residual / (1 + nb);
                const double od = cand.dual_residual / (1 + nc);
                const double og = cand.gap / (1 + std::abs(cand.primal_objective) +
                                             std::abs(cand.dual_objective));
                const bool trusted = cand.error_bound <=
                    10 * opt.eps_gap * (1 + std::abs(cand.primal_objective));
                if (op <= opt.eps_primal && od <= opt.eps_dual && og <= opt.eps_gap &&
                    trusted)
                    return cand;
                if (tightened >= 3) {
                    // still not trustworthy: do not claim optimality
                    cand.status = Status::NumericalError;
                    return cand;
                }
                tol_eff *= 0.05;
                ++tightened;
                continue;
            }
            if (it >= opt.ipm_max_iterations)
                return finish(Status::IterationLimit, it);
            if (elapsed() > opt.time_limit) return finish(Status::TimeLimit, it);

            if (mu > 0)
                par.for_each(N, Recenter{sl.data, su.data, tl.data, tu.data,
                                    lbz.data, ubz.data, mu, 1e10});

            // ---- factor the KKT matrix ---------------------------------
            par.for_each(N, BuildTheta{hzv.data, tl.data, tu.data, sl.data,
                                  su.data, lbz.data, ubz.data});
            bool factored = false;
            for (int attempt = 0; attempt < 5 && !factored; ++attempt) {
                par.for_each(N, ScatterKkt{K.values_mut(), thw.data, dxpos.data(),
                                      dwpos.data(), hzv.data, rho, delta, n});
                if (!analyzed) {
                    const double ta = elapsed();
                    analyzed = true;
                    double kkt_flops = 0;
                    bool aug_analyzed = false;
                    if constexpr (std::is_same_v<Store<double>, Cpu::HostStorage<double>>) {
                        // the normal equations are analyzed first: when cheap they are
                        // used without the (much larger) augmented analysis
                        if (opt.ipm_normal >= 0) {
                            nk.reset(new NormalKkt(q, opt.ipm_ordering == 1
                                                          ? Sparse::Ordering::NestedDissection
                                                          : Sparse::Ordering::MinDegree));
                            if (opt.verbose) std::printf("[ipm] normal pattern built at %.0f ms, eligible %d\n", (elapsed() - ta) * 1000, (int)nk->eligible());
                            bool nk_ok = nk->eligible();
                            if (nk_ok) {
                                try { nk->analyze(); }
                                catch (const std::bad_alloc &) { nk_ok = false; }
                            }
                            if (nk_ok) {
                                if (opt.verbose) std::printf("[ipm] normal analyzed at %.0f ms\n", (elapsed() - ta) * 1000);
                                kkt_flops = nk->factor_flops();
                                if (opt.ipm_normal == 1 || kkt_flops < 2e9 ||
                                    (opt.ipm_max_flops > 0 && kkt_flops > 50 * opt.ipm_max_flops)) {
                                    // cheap, or hopeless (the augmented system has about
                                    // the same fill): no need to analyze it as well
                                    use_normal = true;
                                } else {
                                    ldl.analyze(K);
                                    aug_analyzed = true;
                                    if (opt.verbose)
                                        std::printf("[ipm] normal equations: %.3e flops vs augmented %.3e\n",
                                            kkt_flops, ldl.factor_flops());
                                    if (kkt_flops < 0.5 * ldl.factor_flops()) use_normal = true;
                                    else kkt_flops = ldl.factor_flops();
                                }
                            }
                        }
                        if (!use_normal && !aug_analyzed) {
                            try {
                                ldl.analyze(K);
                                kkt_flops = ldl.factor_flops();
                            } catch (const std::bad_alloc &) { // factors do not fit in memory
                                LpSolution ns = finish(Status::NotSolved, 0);
                                ns.factor_flops = 1e300;
                                return ns;
                            }
                        }
                    } else {
                        ldl.analyze(K);
                    }
                    if constexpr (std::is_same_v<Store<double>, Cpu::HostStorage<double>>) {
                        if (opt.verbose)
                            std::printf("[ipm] est. factor flops %.3e (%s)\n", kkt_flops,
                                use_normal ? "normal equations" : "augmented");
                        if (opt.ipm_max_flops > 0 && kkt_flops > opt.ipm_max_flops) {
                            if (opt.verbose)
                                std::printf("[ipm] factorization needs %.2e flops (limit %.2e): giving up\n",
                                    kkt_flops, opt.ipm_max_flops);
                            LpSolution ns = finish(Status::NotSolved, 0);
                            ns.factor_flops = kkt_flops;
                            return ns;
                        }
                    }
                    if (opt.verbose)
                        std::printf("[ipm] KKT size %zu, nnz(L) %zu, analyze %.0f ms\n",
                            N2, use_normal ? nk->factor_nnz() : ldl.factor_nnz(),
                            (elapsed() - ta) * 1000);
                }
                const double tf = elapsed();
                if (use_normal) {
                    if constexpr (std::is_same_v<Store<double>, Cpu::HostStorage<double>>)
                        factored = nk->factorize(hzv.data, thw.data, rho, delta);
                } else {
                    factored = ldl.factorize(K);
                }
                if (opt.verbose)
                    std::printf("[ipm]        factorize %.0f ms\n", (elapsed() - tf) * 1000);
                if (opt.verbose && std::getenv("AXOS_MF_PROFILE")) {
                    if (use_normal) nk->print_profile(); else ldl.print_profile();
                }
                if (!factored) { rho *= 100; delta *= 100; }
            }
            if (!factored) return finish(Status::NumericalError, it);

            // ---- predictor -----------------------------------------------
            par.for_each(N, CompAff{rcl.data, rcu.data, tl.data, tu.data, sl.data,
                                su.data, lbz.data, ubz.data});
            par.for_each(N, BuildG{g.data, rd.data, rcl.data, rcu.data, tl.data,
                                tu.data, lbz.data, ubz.data});
            par.for_each(N2, PackRhs{rhs.data, g.data, rp.data, thw.data, n});
            solve_kkt(rhs, sol);
            par.for_each(N, Unpack{dz_a.data, dsl_a.data, dsu_a.data, dlam_a.data,
                                sol.data, g.data, thw.data, rcl.data, rcu.data,
                                tl.data, tu.data, sl.data, su.data, lbz.data,
                                ubz.data, n});

            par.zero();
            par.set_slot(0, 1e300);
            par.set_slot(1, 1e300);
            par.reduce_min(0, N, StepP{tl.data, tu.data, dz_a.data, lbz.data, ubz.data});
            par.reduce_min(1, N, StepD{sl.data, su.data, dsl_a.data, dsu_a.data,
                                     lbz.data, ubz.data});
            const double *sa = par.fetch();
            const double ap_aff = std::min(1.0, sa[0]), ad_aff = std::min(1.0, sa[1]);
            par.zero();
            par.reduce(0, N, MuAff{tl.data, tu.data, sl.data, su.data, dz_a.data,
                                dsl_a.data, dsu_a.data, lbz.data, ubz.data,
                                ap_aff, ad_aff});
            const double mu_aff = ncomp > 0 ? par.fetch()[0] / ncomp : 0.0;
            double sigma = mu > 0 ? std::pow(mu_aff / mu, 3.0) : 0.0;
            sigma = std::min(1.0, std::max(sigma, 0.0));

            // ---- corrector -----------------------------------------------
            par.for_each(N, CompCorr{rcl.data, rcu.data, tl.data, tu.data, sl.data,
                                 su.data, lbz.data, ubz.data, dz_a.data,
                                 dsl_a.data, dsu_a.data, sigma * mu});
            par.for_each(N, BuildG{g.data, rd.data, rcl.data, rcu.data, tl.data,
                                tu.data, lbz.data, ubz.data});
            par.for_each(N2, PackRhs{rhs.data, g.data, rp.data, thw.data, n});
            solve_kkt(rhs, sol);
            par.for_each(N, Unpack{dz.data, dsl.data, dsu.data, dlam.data, sol.data,
                                g.data, thw.data, rcl.data, rcu.data, tl.data,
                                tu.data, sl.data, su.data, lbz.data, ubz.data, n});

            par.zero();
            par.set_slot(0, 1e300);
            par.set_slot(1, 1e300);
            par.reduce_min(0, N, StepP{tl.data, tu.data, dz.data, lbz.data, ubz.data});
            par.reduce_min(1, N, StepD{sl.data, su.data, dsl.data, dsu.data,
                                     lbz.data, ubz.data});
            const double *sb = par.fetch();
            const double eta = 0.995;
            const double ap = std::min(1.0, eta * sb[0]), ad = std::min(1.0, eta * sb[1]);
            if (opt.verbose)
                std::printf("[ipm]        ap %.2e ad %.2e sigma %.2e refine-res %.1e rho %.0e\n",
                    ap, ad, sigma, last_ref_res, rho);
            if (!(ap > 1e-10) && !(ad > 1e-10)) {
                if (++stall >= 3) return finish(Status::NumericalError, it);
            } else {
                stall = 0;
            }
            par.for_each(N, UpdatePrimalDual{z.data, tl.data, tu.data, sl.data,
                                         su.data, dz.data, dsl.data, dsu.data,
                                         lbz.data, ubz.data, ap, ad});
            par.for_each(m, Axpy{lam.data, dlam.data, ad});
        }
        return finish(Status::IterationLimit, opt.ipm_max_iterations);
    }
};

} // namespace Solver
} // namespace AXOS
