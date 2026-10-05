// SPDX-License-Identifier: BSD-3-Clause
//
// Primal-dual interior-point method for convex QP (CPU): the Mehrotra
// predictor-corrector method of the LP solver (solver/lp/ipm.h) extended to
//
//   min 1/2 x^T Q x + c^T x   s.t.  A x - w = 0,  lb <= x <= ub,  l <= w <= u
//
// with z = (x, w), multipliers lam of the equalities and bound duals sl, su.
// Q enters in three places: the dual residual (c + Q x - A^T lam - sl + su),
// the (1,1) block of the regularized quasi-definite augmented system
//        [ -(Q + Theta_x^{-1} + rho I)      A^T          ]
//    K = [            A               Theta_w + delta I  ]
// (its pattern is fixed, so the ordering and symbolic analysis are done
// once, and only the diagonal changes), and the iterative refinement of each
// solve against the unregularized K. Unlike LP, the primal and dual steps
// share one step length: with Q, a longer dual than primal step would break
// the linearization of the dual residual.
//
// This is the accurate complement of the first-order GPU methods
// (hpr_qp.h, pdhcg.h) for small and medium problems whose KKT factorization
// is cheap. Known weakness: the static regularization (1e-8) perturbs the
// constraints at that level, and a few degenerate problems are that
// sensitive (the LISWET family converges to about 25.0 instead of 36.1 with
// 1e-8, to the right value only with 1e-12, which other problems cannot
// take): a proximal method of multipliers with decreasing regularization, as
// in PIQP, is the fix.
//
// Before the solve, fixed columns are substituted out and empty and free
// rows dropped (the IPM needs neither); the problem is scaled like the
// first-order methods (qp_scaling.h). The solution is mapped back and scored
// on the original problem with evaluate_qp(): Optimal is only reported when
// that common metric meets opt.tol (the internal tolerances are tightened
// up to three times when the scaled residuals were met first).
#pragma once

#include "solver/lp/ipm_kernels.h"
#include "solver/qp/qp_model.h"
#include "solver/qp/qp_scaling.h"
#include "sparse/sparse.h"
#include "tensorET.h"
#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <vector>
#if defined(_M_X64) || defined(__SSE2__)
#include <xmmintrin.h>
#endif

namespace AXOS {
namespace Solver {
namespace qp {

namespace qipm {
// out = a + b
struct AddVec {
    double *out;
    const double *a, *b;
    void operator()(size_t i) const { out[i] = a[i] + b[i]; }
};
// y -= x
struct SubVec {
    double *y;
    const double *x;
    void operator()(size_t i) const { y[i] -= x[i]; }
};
// sum a b
struct Dot {
    const double *a, *b;
    static constexpr int K = 1;
    void operator()(size_t i, double *acc) const { acc[0] += a[i] * b[i]; }
};
// KKT diagonal source: Theta^{-1} plus diag(Q) on the columns
struct AddDiag {
    double *out;
    const double *hz, *qd;
    size_t n;
    void operator()(size_t j) const { out[j] = hz[j] + (j < n ? qd[j] : 0.0); }
};
} // namespace qipm

// The problem the IPM solves after fixed columns, empty rows and free rows
// are removed, and the maps back to the original indices.
struct QpReduction {
    QpProblem q;
    std::vector<int> cols;  // reduced column -> original column
    std::vector<int> rows;  // reduced row -> original row (-1: the dummy row)
    std::vector<double> x0; // original-size x with the fixed values
    bool infeasible = false;

    explicit QpReduction(const QpProblem &p)
    {
        const size_t n = p.cols(), m = p.rows();
        x0.assign(n, 0.0);
        std::vector<int> cmap(n, -1);
        for (size_t j = 0; j < n; ++j) {
            if (p.lp.col_lb[j] == p.lp.col_ub[j]) {
                x0[j] = p.lp.col_lb[j];
            } else {
                cmap[j] = static_cast<int>(cols.size());
                cols.push_back(static_cast<int>(j));
            }
        }
        const size_t nr = cols.size();
        // c and the offset with the fixed columns substituted
        std::vector<double> c(nr);
        double offset = p.lp.offset;
        for (size_t k = 0; k < nr; ++k) c[k] = p.lp.c[cols[k]];
        for (size_t j = 0; j < n; ++j)
            if (cmap[j] < 0) offset += p.lp.c[j] * x0[j];
        const bool hasQ = p.Q.nnz() > 0 && p.Q.rows() == n;
        std::vector<int32_t> qrp{0}, qci;
        std::vector<double> qv;
        if (hasQ) {
            const auto *rp = p.Q.row_ptr();
            const auto *ci = p.Q.col_ind();
            const double *v = p.Q.values();
            for (size_t i = 0; i < n; ++i)
                for (auto k = rp[i]; k < rp[i + 1]; ++k)
                    if (cmap[i] < 0 && cmap[ci[k]] < 0) offset += 0.5 * x0[i] * v[k] * x0[ci[k]];
                    else if (cmap[i] >= 0 && cmap[ci[k]] < 0) c[cmap[i]] += v[k] * x0[ci[k]];
            for (size_t k = 0; k < nr; ++k) {
                const int i = cols[k];
                for (auto t = rp[i]; t < rp[i + 1]; ++t)
                    if (cmap[ci[t]] >= 0) {
                        qci.push_back(cmap[ci[t]]);
                        qv.push_back(v[t]);
                    }
                qrp.push_back(static_cast<int32_t>(qci.size()));
            }
        }
        // rows: shift by the fixed columns, drop free and empty ones
        std::vector<int32_t> arp{0}, aci;
        std::vector<double> av, rl, ru;
        const auto *rp = p.lp.A.row_ptr();
        const auto *ci = p.lp.A.col_ind();
        const double *v = p.lp.A.values();
        for (size_t i = 0; i < m; ++i) {
            double shift = 0;
            size_t kept = 0;
            for (auto k = rp[i]; k < rp[i + 1]; ++k)
                if (cmap[ci[k]] < 0) shift += v[k] * x0[ci[k]];
                else if (v[k] != 0) ++kept;
            const double lo = p.lp.row_lb[i] - shift, hi = p.lp.row_ub[i] - shift;
            if (!std::isfinite(lo) && !std::isfinite(hi)) continue;
            if (kept == 0) {
                const double tol = 1e-9 * (1 + std::abs(shift));
                if (lo > tol || hi < -tol) infeasible = true;
                continue;
            }
            for (auto k = rp[i]; k < rp[i + 1]; ++k)
                if (cmap[ci[k]] >= 0 && v[k] != 0) {
                    aci.push_back(cmap[ci[k]]);
                    av.push_back(v[k]);
                }
            arp.push_back(static_cast<int32_t>(aci.size()));
            rl.push_back(lo);
            ru.push_back(hi);
            rows.push_back(static_cast<int>(i));
        }
        if (rows.empty()) { // the IPM needs a row: 0 = 0
            arp.push_back(static_cast<int32_t>(aci.size()));
            rl.push_back(0.0);
            ru.push_back(0.0);
            rows.push_back(-1);
        }
        q.lp.A = HostMatrix(rows.size(), nr, arp, aci, av);
        q.Q = hasQ ? HostMatrix(nr, nr, qrp, qci, qv) : HostMatrix(nr, nr, 0);
        q.lp.c = c;
        q.lp.offset = offset;
        q.lp.row_lb = rl;
        q.lp.row_ub = ru;
        q.lp.col_lb.resize(nr);
        q.lp.col_ub.resize(nr);
        for (size_t k = 0; k < nr; ++k) {
            q.lp.col_lb[k] = p.lp.col_lb[cols[k]];
            q.lp.col_ub[k] = p.lp.col_ub[cols[k]];
        }
    }

    void
    restore(const std::vector<double> &xr, const std::vector<double> &yr, size_t m,
        std::vector<double> &x, std::vector<double> &y) const
    {
        x = x0;
        for (size_t k = 0; k < cols.size(); ++k) x[cols[k]] = xr[k];
        y.assign(m, 0.0);
        for (size_t i = 0; i < rows.size(); ++i)
            if (rows[i] >= 0) y[rows[i]] = yr[i];
    }
};

// Flush-to-zero and denormals-are-zero for the duration of a solve, in the
// calling thread and the OpenMP workers: near the end of an interior-point
// solve the barrier terms span 1e-300 .. 1e300, and a factorization that
// runs into denormal numbers is many times slower on x86.
struct DenormalGuard {
#if defined(_M_X64) || defined(__SSE2__)
    static void
    set(bool on)
    {
        const unsigned mask = 0x8040; // MXCSR FTZ | DAZ
#pragma omp parallel
        _mm_setcsr(on ? (_mm_getcsr() | mask) : (_mm_getcsr() & ~mask));
        _mm_setcsr(on ? (_mm_getcsr() | mask) : (_mm_getcsr() & ~mask));
    }
    DenormalGuard() { set(true); }
    ~DenormalGuard() { set(false); }
#endif
};

class QpIpm {
    // limits on the KKT factor: past them the solve returns NotSolved at once
    static constexpr double kMaxFactorBytes = 2e9;
    using Store = Cpu::HostStorage<double>;
    using Vec = tensorET<1, double, Store>;
    using Mat = Sparse::Csr<double, int32_t, Cpu::HostStorage>;
    using Par = Parallel<Cpu::Backend>;
    using Ker = Sparse::Kernels<Cpu::Backend>;
    using Ldl = Sparse::SparseLdlt<double, int32_t, Cpu::HostStorage>;

  public:
    QpSolution
    solve(const QpProblem &p, const QpOptions &opt)
    {
        using namespace ipm;
        const auto t_start = std::chrono::steady_clock::now();
        auto elapsed = [&] {
            return std::chrono::duration<double>(std::chrono::steady_clock::now() - t_start)
                .count();
        };
        const std::string bad = p.validate();
        if (!bad.empty()) throw std::invalid_argument("QpIpm: " + bad);
        const DenormalGuard ftz;

        // ---- reduction and scaling -------------------------------------------
        const QpReduction red(p);
        auto result = [&](Status st, long iters, const std::vector<double> &xr,
                          const std::vector<double> &yr) {
            std::vector<double> x, y;
            red.restore(xr, yr, p.rows(), x, y);
            QpSolution s = evaluate_qp(p, x, y);
            s.status = st;
            s.iterations = iters;
            s.seconds = elapsed();
            s.method = "ipm";
            s.device = "cpu";
            return s;
        };
        if (red.infeasible)
            return result(Status::Infeasible, 0, std::vector<double>(red.cols.size(), 0.0),
                std::vector<double>(red.rows.size(), 0.0));
        const size_t n = red.q.cols(), m = red.q.rows(), N = n + m;
        if (n == 0) {
            QpSolution s = result(Status::Optimal, 0, {}, std::vector<double>(m, 0.0));
            if (s.kkt() > opt.tol) s.status = Status::NumericalError;
            return s;
        }
        QpScaling sc;
        QpProblem q;
        if (opt.scaling) {
            sc = compute_qp_scaling(red.q, opt.ruiz_iterations, opt.pock_chambolle_alpha);
            q = apply_qp_scaling(red.q, sc);
        } else {
            sc.col.assign(n, 1.0);
            sc.row.assign(m, 1.0);
            q = red.q;
        }
        const bool hasQ = q.Q.nnz() > 0;

        // ---- KKT pattern: x rows (Q, diagonal, A^T), then w rows (A, diagonal) --
        HostMatrix At = q.lp.A.transpose();
        std::vector<int32_t> krp(N + 1, 0), kci, dxp(n), dwp(m);
        std::vector<double> kv, qd(n, 0.0);
        kci.reserve(2 * q.lp.A.nnz() + q.Q.nnz() + N);
        kv.reserve(kci.capacity());
        for (size_t j = 0; j < n; ++j) {
            dxp[j] = static_cast<int32_t>(kci.size());
            kci.push_back(static_cast<int32_t>(j));
            kv.push_back(0.0);
            if (hasQ)
                for (auto k = q.Q.row_ptr()[j]; k < q.Q.row_ptr()[j + 1]; ++k) {
                    const auto col = q.Q.col_ind()[k];
                    if (static_cast<size_t>(col) == j) {
                        qd[j] += q.Q.values()[k];
                    } else {
                        kci.push_back(col);
                        kv.push_back(-q.Q.values()[k]);
                    }
                }
            for (auto k = At.row_ptr()[j]; k < At.row_ptr()[j + 1]; ++k) {
                kci.push_back(static_cast<int32_t>(n) + At.col_ind()[k]);
                kv.push_back(At.values()[k]);
            }
            krp[j + 1] = static_cast<int32_t>(kci.size());
        }
        for (size_t i = 0; i < m; ++i) {
            for (auto k = q.lp.A.row_ptr()[i]; k < q.lp.A.row_ptr()[i + 1]; ++k) {
                kci.push_back(q.lp.A.col_ind()[k]);
                kv.push_back(q.lp.A.values()[k]);
            }
            dwp[i] = static_cast<int32_t>(kci.size());
            kci.push_back(static_cast<int32_t>(n + i));
            kv.push_back(0.0);
            krp[n + i + 1] = static_cast<int32_t>(kci.size());
        }
        Mat K(N, N, krp, kci, kv);
        const Mat &A = q.lp.A;
        const Mat &AT = At;
        const Mat &Qm = q.Q;

        // ---- starting point (as the LP IPM) ------------------------------------
        std::vector<double> hlb(N), hub(N), hz(N), htl(N, 1.0), htu(N, 1.0), hsl(N, 0.0),
            hsu(N, 0.0);
        for (size_t j = 0; j < n; ++j) {
            hlb[j] = q.lp.col_lb[j];
            hub[j] = q.lp.col_ub[j];
        }
        for (size_t i = 0; i < m; ++i) {
            hlb[n + i] = q.lp.row_lb[i];
            hub[n + i] = q.lp.row_ub[i];
        }
        double cinf = 0;
        for (double v : q.lp.c) cinf = std::max(cinf, std::abs(v));
        const double M = 1e2, s0 = std::max(M, cinf);
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
                for (auto k = A.row_ptr()[i]; k < A.row_ptr()[i + 1]; ++k)
                    ax[i] += A.values()[k] * hz[A.col_ind()[k]];
            for (size_t i = 0; i < m; ++i) hz[n + i] = interior(hlb[n + i], hub[n + i], ax[i]);
        }
        for (size_t j = 0; j < N; ++j) {
            if (bl(hlb[j], hub[j])) {
                htl[j] = hz[j] - hlb[j];
                hsl[j] = s0;
            }
            if (bu(hlb[j], hub[j])) {
                htu[j] = hub[j] - hz[j];
                hsu[j] = s0;
            }
        }
        auto upload = [](const std::vector<double> &h) {
            Vec t({std::max<size_t>(h.size(), 1)}, 0.0);
            if (!h.empty()) std::memcpy(t.data, h.data(), h.size() * sizeof(double));
            return t;
        };
        Vec z = upload(hz), tl = upload(htl), tu = upload(htu), sl = upload(hsl),
            su = upload(hsu), lbz = upload(hlb), ubz = upload(hub);
        Vec c = upload(q.lp.c), qdv = upload(qd);
        Vec lam({m}, 0.0), dlam({m}, 0.0), dlam_a({m}, 0.0), rp({m}, 0.0), ax({m}, 0.0),
            thw({m}, 0.0), tmpm({m}, 0.0);
        Vec aty({n}, 0.0), qx({n}, 0.0), ceff({n}, 0.0), tmpn({n}, 0.0), tmpq({n}, 0.0);
        Vec rd({N}, 0.0), hzv({N}, 0.0), hzq({N}, 0.0), rcl({N}, 0.0), rcu({N}, 0.0),
            g({N}, 0.0), dz({N}, 0.0), dsl({N}, 0.0), dsu({N}, 0.0), dz_a({N}, 0.0),
            dsl_a({N}, 0.0), dsu_a({N}, 0.0);
        Vec rhs({N}, 0.0), sol({N}, 0.0), res({N}, 0.0), dsol({N}, 0.0);
        Par par;
        Ldl ldl(Sparse::Symmetry::Symmetric, Sparse::Ordering::MinDegree);
        bool analyzed = false;

        // scaled norms for the internal relative tests
        double qb = 0, qc = 0;
        for (size_t i = 0; i < m; ++i) {
            if (std::isfinite(q.lp.row_lb[i])) qb += q.lp.row_lb[i] * q.lp.row_lb[i];
            if (std::isfinite(q.lp.row_ub[i])) qb += q.lp.row_ub[i] * q.lp.row_ub[i];
        }
        for (double v : q.lp.c) qc += v * v;
        qb = std::sqrt(qb);
        qc = std::sqrt(qc);

        double rho = 1e-8, delta = 1e-8; // static regularization (raised if a pivot fails)
        const int ref_steps = 3;
        double t_factor = 0, t_solve = 0; // accumulated, for the verbose log
        double tol_eff = opt.tol;
        int tightened = 0, stall = 0;
        const int max_it = 300;

        auto finish = [&](Status st, long iters) {
            std::vector<double> xr(n), yr(m);
            for (size_t j = 0; j < n; ++j) xr[j] = z.data[j] * sc.col[j];
            for (size_t i = 0; i < m; ++i) yr[i] = lam.data[i] * sc.row[i];
            QpSolution s = result(st, iters, xr, yr);
            if (opt.verbose)
                std::printf("[qp-ipm] %s after %ld iterations, %.3f s: obj %.10e eta_p %.2e "
                            "eta_d %.2e gap %.2e\n",
                    to_string(s.status), iters, s.seconds, s.primal_objective, s.rel_primal,
                    s.rel_dual, s.rel_gap);
            return s;
        };

        // K sol = r with the current factorization, refined against the
        // unregularized K (whose (1,1) block is -(Q + Theta_x^{-1}))
        auto solve_kkt = [&](Vec &r, Vec &out) {
            ldl.solve(r, out);
            for (int step = 0; step < ref_steps; ++step) {
                Ker::spmv(A, out.data, tmpm.data, 1.0, 0.0);
                Ker::spmv(AT, out.data + n, tmpn.data, 1.0, 0.0);
                if (hasQ) {
                    Ker::spmv(Qm, out.data, tmpq.data, 1.0, 0.0);
                    par.for_each(n, qipm::SubVec{tmpn.data, tmpq.data});
                }
                par.for_each(N, RefineRes{res.data, r.data, out.data, tmpn.data, tmpm.data,
                                    hzv.data, thw.data, n});
                par.zero();
                par.reduce(0, N, SqNorm{res.data});
                par.reduce(1, N, SqNorm{r.data});
                const double *s = par.fetch();
                if (std::sqrt(s[0]) <= 1e-13 * (1 + std::sqrt(s[1]))) return;
                ldl.solve(res, dsol);
                par.for_each(N, Axpy{out.data, dsol.data, 1.0});
            }
        };

        if (opt.verbose)
            std::printf("[qp-ipm] n %zu m %zu nnz(A) %zu nnz(Q) %zu (reduced from n %zu m %zu)\n",
                n, m, A.nnz(), Qm.nnz(), p.cols(), p.rows());

        for (long it = 0; it <= max_it; ++it) {
            // ---- residuals and objectives ------------------------------------------
            Ker::spmv(A, z.data, ax.data, 1.0, 0.0);
            Ker::spmv(AT, lam.data, aty.data, 1.0, 0.0);
            if (hasQ) Ker::spmv(Qm, z.data, qx.data, 1.0, 0.0);
            par.for_each(n, qipm::AddVec{ceff.data, c.data, qx.data});
            par.for_each(m, ResP{rp.data, z.data + n, ax.data});
            par.for_each(N, ResD{rd.data, ceff.data, aty.data, lam.data, sl.data, su.data,
                                lbz.data, ubz.data, n});
            par.zero();
            par.reduce(0, N, Stats{rd.data, z.data, c.data, lam.data, tl.data, tu.data,
                                sl.data, su.data, lbz.data, ubz.data, n});
            par.reduce(5, m, SqNorm{rp.data});
            par.reduce(6, n, qipm::Dot{z.data, qx.data});
            const double *s = par.fetch();
            const double rd2 = s[0], cx = s[1], comp = s[3], ncomp = s[4], rp2 = s[5],
                         xqx = s[6];
            const double pobj = cx + 0.5 * xqx + q.lp.offset;
            const double dobj = s[2] - 0.5 * xqx + q.lp.offset;
            const double mu = ncomp > 0 ? comp / ncomp : 0.0;
            const double rel_p = std::sqrt(rp2) / (1 + qb);
            const double rel_d = std::sqrt(rd2) / (1 + qc);
            const double rel_g = std::abs(pobj - dobj) / (1 + std::abs(pobj) + std::abs(dobj));
            if (opt.verbose >= 2)
                std::printf("[qp-ipm] it %-3ld pobj % .8e dobj % .8e rp %.1e rd %.1e gap %.1e "
                            "mu %.1e\n",
                    it, pobj, dobj, rel_p, rel_d, rel_g, mu);
            if (!std::isfinite(pobj) || !std::isfinite(dobj) || !std::isfinite(mu))
                return finish(Status::NumericalError, it);

            if (rel_p <= tol_eff && rel_d <= tol_eff && rel_g <= tol_eff) {
                QpSolution cand = finish(Status::Optimal, it);
                if (cand.kkt() <= opt.tol) return cand;
                if (tightened >= 3) {
                    cand.status = Status::NumericalError;
                    return cand;
                }
                tol_eff *= 0.05;
                ++tightened;
            }
            if (it >= max_it) return finish(Status::IterationLimit, it);
            if (elapsed() > opt.time_limit) return finish(Status::TimeLimit, it);

            if (mu > 0)
                par.for_each(N, Recenter{sl.data, su.data, tl.data, tu.data, lbz.data, ubz.data,
                                    mu, 1e10});

            // ---- factor K -----------------------------------------------------------
            par.for_each(N, BuildTheta{hzv.data, tl.data, tu.data, sl.data, su.data, lbz.data,
                                ubz.data});
            par.for_each(N, qipm::AddDiag{hzq.data, hzv.data, qdv.data, n});
            bool factored = false;
            for (int attempt = 0; attempt < 5 && !factored; ++attempt) {
                par.for_each(N, ScatterKkt{K.values_mut(), thw.data, dxp.data(), dwp.data(),
                                    hzq.data, rho, delta, n});
                if (!analyzed) {
                    const double ta = elapsed();
                    try {
                        ldl.analyze(K);
                    } catch (const std::bad_alloc &) { // the factors do not fit in memory
                        return finish(Status::NotSolved, it);
                    }
                    analyzed = true;
                    if (opt.verbose)
                        std::printf("[qp-ipm] KKT size %zu, nnz(L) %zu, est. %.2e flops, "
                                    "analyze %.0f ms\n",
                            N, ldl.factor_nnz(), ldl.factor_flops(), (elapsed() - ta) * 1000);
                    // a factorization this large is the first-order methods' case
                    if (8.0 * static_cast<double>(ldl.factor_nnz()) > kMaxFactorBytes ||
                        ldl.factor_flops() > opt.ipm_max_flops) {
                        if (opt.verbose)
                            std::printf("[qp-ipm] factor too large (%.2e bytes, %.2e flops): giving up\n",
                                8.0 * static_cast<double>(ldl.factor_nnz()), ldl.factor_flops());
                        return finish(Status::NotSolved, it);
                    }
                }
                const double tf0 = elapsed();
                factored = ldl.factorize(K);
                t_factor += elapsed() - tf0;
                if (opt.verbose >= 3) ldl.print_profile();
                if (it == 0 && elapsed() + (opt.ipm_project ? 25 : 5) * t_factor > opt.time_limit) {
                    if (opt.verbose)
                        std::printf("[qp-ipm] %.0f ms per factorization: would not finish in "
                                    "%.1f s, giving up\n", 1e3 * t_factor, opt.time_limit);
                    return finish(Status::NotSolved, it);
                }
                if (!factored) {
                    rho *= 100;
                    delta *= 100;
                }
            }
            if (!factored) return finish(Status::NumericalError, it);

            // ---- predictor ------------------------------------------------------------
            par.for_each(N, CompAff{rcl.data, rcu.data, tl.data, tu.data, sl.data, su.data,
                                lbz.data, ubz.data});
            par.for_each(N, BuildG{g.data, rd.data, rcl.data, rcu.data, tl.data, tu.data,
                                lbz.data, ubz.data});
            par.for_each(N, PackRhs{rhs.data, g.data, rp.data, thw.data, n});
            const double ts0 = elapsed();
            solve_kkt(rhs, sol);
            t_solve += elapsed() - ts0;
            par.for_each(N, Unpack{dz_a.data, dsl_a.data, dsu_a.data, dlam_a.data, sol.data,
                                g.data, thw.data, rcl.data, rcu.data, tl.data, tu.data,
                                sl.data, su.data, lbz.data, ubz.data, n});
            par.zero();
            par.set_slot(0, 1e300);
            par.set_slot(1, 1e300);
            par.reduce_min(0, N, StepP{tl.data, tu.data, dz_a.data, lbz.data, ubz.data});
            par.reduce_min(1, N, StepD{sl.data, su.data, dsl_a.data, dsu_a.data, lbz.data,
                                     ubz.data});
            const double *sa = par.fetch();
            double ap_aff = std::min(1.0, sa[0]), ad_aff = std::min(1.0, sa[1]);
            if (hasQ) ap_aff = ad_aff = std::min(ap_aff, ad_aff);
            par.zero();
            par.reduce(0, N, MuAff{tl.data, tu.data, sl.data, su.data, dz_a.data, dsl_a.data,
                                dsu_a.data, lbz.data, ubz.data, ap_aff, ad_aff});
            const double mu_aff = ncomp > 0 ? par.fetch()[0] / ncomp : 0.0;
            double sigma = mu > 0 ? std::pow(mu_aff / mu, 3.0) : 0.0;
            sigma = std::min(1.0, std::max(sigma, 0.0));

            // ---- corrector ------------------------------------------------------------
            par.for_each(N, CompCorr{rcl.data, rcu.data, tl.data, tu.data, sl.data, su.data,
                                 lbz.data, ubz.data, dz_a.data, dsl_a.data, dsu_a.data,
                                 sigma * mu});
            par.for_each(N, BuildG{g.data, rd.data, rcl.data, rcu.data, tl.data, tu.data,
                                lbz.data, ubz.data});
            par.for_each(N, PackRhs{rhs.data, g.data, rp.data, thw.data, n});
            const double ts1 = elapsed();
            solve_kkt(rhs, sol);
            t_solve += elapsed() - ts1;
            par.for_each(N, Unpack{dz.data, dsl.data, dsu.data, dlam.data, sol.data, g.data,
                                thw.data, rcl.data, rcu.data, tl.data, tu.data, sl.data,
                                su.data, lbz.data, ubz.data, n});
            par.zero();
            par.set_slot(0, 1e300);
            par.set_slot(1, 1e300);
            par.reduce_min(0, N, StepP{tl.data, tu.data, dz.data, lbz.data, ubz.data});
            par.reduce_min(1, N, StepD{sl.data, su.data, dsl.data, dsu.data, lbz.data,
                                     ubz.data});
            const double *sb = par.fetch();
            const double eta = 0.995;
            double ap = std::min(1.0, eta * sb[0]), ad = std::min(1.0, eta * sb[1]);
            if (hasQ) ap = ad = std::min(ap, ad);
            if (opt.verbose >= 2)
                std::printf("[qp-ipm]        ap %.2e ad %.2e sigma %.2e rho %.0e (factor %.0f ms, "
                            "solves %.0f ms so far)\n",
                    ap, ad, sigma, rho, 1e3 * t_factor, 1e3 * t_solve);
            if (!(ap > 1e-10) && !(ad > 1e-10)) {
                if (++stall >= 3) return finish(Status::NumericalError, it);
            } else {
                stall = 0;
            }
            par.for_each(N, UpdatePrimalDual{z.data, tl.data, tu.data, sl.data, su.data,
                                         dz.data, dsl.data, dsu.data, lbz.data, ubz.data,
                                         ap, ad});
            par.for_each(m, Axpy{lam.data, dlam.data, ad});
        }
        return finish(Status::IterationLimit, max_it);
    }
};

} // namespace qp
} // namespace Solver
} // namespace AXOS
