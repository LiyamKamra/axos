// SPDX-License-Identifier: BSD-3-Clause
//
// HPR-QP: dual Halpern Peaceman-Rachford for convex QP, Algorithm 4 of
// K. Chen, D. Sun, Y. Yuan, G. Zhang, X. Zhao, arXiv:2507.02470.
//
//   primal   min 1/2 <x,Qx> + <c,x> + delta_C(x)   s.t.  Ax in K
//   dual     min 1/2 <w,Qw> + delta*_K(-y) + delta*_C(-z)
//            s.t. -Q w + A^T y + z = c,  w in range(Q)
//
// Iteration (state u = (y, w, x); only Q w and A^T y are needed, so w is never
// projected onto range(Q) - the paper's "shadow sequence"):
//   z/x    xbar = P_C(x + s(A^T y - Q w - c)),  xhat = 2 xbar - x
//   w      whalf = (s lq w + xhat) / (1 + s lq)                  (sGS forward)
//   y      ybar = (P_K(r) - r) / (s la),  r = A(xhat + s(Qw - Q whalf)) - s la
//   y w      wbar = whalf + s/(1 + s lq) A^T(ybar - y)              (sGS
//   backward) Halpern  u <- (t+1)/(t+2) (2 T(u) - u) + 1/(t+2) u0
// with la >= |A A^T|, lq >= |Q| (power method). For diagonal Q the w step is
// exact (no semi-proximal term), lq replaced per element by q_i, as in
// HPR-QP.jl. When over 80% of the columns are free (opt.hpr_free_variant) the
// HPR-QP.jl free sweep w, x, y is used instead (no second half w step): three
// products per iteration instead of four, up to 1.5x faster.
//
// An iteration is four kernels: two products with Q, one with A, one with A^T,
// each fusing the vector updates into the epilogue of the product that feeds
// them (qp_ops.h). The scalars are read from device memory, so the iterations
// between two checks are queued as blocks of 64, 32, ..., 1 that the GPU
// backend captures once each as a CUDA graph and replays with one launch; a
// check iteration (reductions and KKT) is one more graph plus one sync.
//
// Checks every 10 iterations up to 1000, then every opt.check_every (100).
// Restarts use the merit R = |T(u) - u|_M (metric of the method, x-(w, y)
// coupling terms): the first check restarts; later, restart when R <= 0.2 R0
// (R0 = merit at the epoch's first iteration), or R <= 0.8 R0 and R grew since
// the last check, or the epoch is longer than 1/2 (1/5 once R fell below a
// tenth of the first check's merit) of all iterations; the new anchor is T(u).
// The penalty s minimizes the merit a s + b / s + s^2 g(s) of the epoch's
// displacement (golden section on log s), blended with the best s so far
// (weight exp(-0.05 R / R_best)), and near convergence (residuals below 1e-9)
// scaled by the dual / primal infeasibility ratio. These rules follow
// HPR-QP.jl. Preconditioning: Ruiz (10 passes) and Pock-Chambolle (alpha = 1)
// on [Q A^T; A 0] (qp_scaling.h).
//
// Termination: max(eta_p, eta_d, eta_gap) <= opt.tol at T(u), on the original
// (unscaled) problem with the bound duals of evaluate_qp() (qp_model.h), which
// re-evaluates the solution on the host.
//
// MPI (opt.comm with more than one rank, qp_dist.h): each rank solves its
// block of constraint rows, with the n-sized state and Q replicated; the A^T
// products are summed over the ranks (one MPI_Allreduce of n doubles per
// iteration), the row reductions of a check combined, decisions follow rank 0.
// No CUDA graphs then (host steps between iterations). Every rank returns the
// same solution (y gathered from the blocks).
#pragma once

#include "solver/qp/qp_dist.h"
#include "solver/qp/qp_model.h"
#include "solver/qp/qp_ops.h"
#include "solver/qp/qp_scaling.h"
#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <vector>

// map / reduce / sparse-product epilogue of an op of qp_ops.h on either
// backend (GPU: kernel "k_<op>")
#define AXOS_QP_MAP(be, f, n, args) \
    (be).map("k_" #f, (n), (args), \
        [](axos_qp::idx i_, const auto &a_) { axos_qp::f(i_, a_); })
#define AXOS_QP_RED(be, f, n, args) \
    (be).reduce("k_" #f, (n), (args), \
        [](axos_qp::idx i_, const auto &a_, double *acc_) { \
            axos_qp::f(i_, a_, acc_); \
        })
#define AXOS_QP_RED_TO(be, f, n, args, slot) \
    (be).reduce_to( \
        "k_" #f, (n), (args), \
        [](axos_qp::idx i_, const auto &a_, double *acc_) { \
            axos_qp::f(i_, a_, acc_); \
        }, \
        (slot))
#define AXOS_QP_SPMV(be, f, M, x, args) \
    (be).spmv_epi("k_" #f, (M), (x), (args), \
        [](axos_qp::idx i_, double s_, const auto &a_) { \
            axos_qp::f(i_, s_, a_); \
        })
// the epilogue on row sums s formed elsewhere (GPU: kernel "k_<op>_vec")
#define AXOS_QP_EPI(be, f, n, s, args) \
    (be).epi("k_" #f, (n), (s), (args), \
        [](axos_qp::idx i_, double s_, const auto &a_) { \
            axos_qp::f(i_, s_, a_); \
        })
// A product with M = A^T (n x local rows) and its epilogue. With the rows
// split over MPI ranks (qp_dist.h) the local partial product goes to part
// (n doubles), is summed over the ranks, and the epilogue runs on the sum.
#define AXOS_QP_SPMV_AT(be, cm, f, M, x, part, args) \
    do { \
        if ((cm).distributed()) { \
            (be).spmv((M), (x), (part)); \
            ::AXOS::Solver::qp::allreduce_sum((be), (part), (M).rows(), (cm)); \
            AXOS_QP_EPI(be, f, (M).rows(), (part), args); \
        } else { \
            AXOS_QP_SPMV(be, f, M, x, args); \
        } \
    } while (0)

namespace AXOS {
namespace Solver {
namespace qp {

// Largest eigenvalue of M^T M (MT given) or of symmetric M by the power method
// on the backend, from a fixed generic start vector. The vector is normalized
// on the device, so blocks of 10 iterations run as one graph and convergence
// (relative change 1e-6 over a block) is tested between blocks. Without
// convergence within max_iter the estimate is raised by 5%. With cm, M holds
// this rank's rows of a matrix split over MPI ranks: the products with MT are
// summed over the ranks and rank 0's estimate counts.
template <class B>
double
power_method(B &be, const typename B::Mat &M, const typename B::Mat *MT,
    size_t n, size_t m, int max_iter, const QpComm *cm = nullptr)
{
    using namespace axos_qp;
    if (n == 0) return 0.0;
    std::vector<double> h(n);
    for (size_t j = 0; j < n; ++j)
        h[j] = 1.0 + 0.5 * std::sin(1.7 * double(j) + 0.3);
    double nrm = 0;
    for (double v : h)
        nrm += v * v;
    for (double &v : h)
        v /= std::sqrt(nrm);
    auto v = be.upload(h);
    auto t = be.vec(std::max<size_t>(m, 1));
    auto s = be.vec(n);
    const long key = MT ? -10 : -11;
    const int kBlock = 10;
    auto step = [&] {
        if (MT) {
            be.spmv(M, B::ptr(v), B::ptr(t));
            be.spmv(*MT, B::ptr(t), B::ptr(s));
            if (cm) allreduce_sum(be, B::ptr(s), n, *cm);
        } else {
            be.spmv(M, B::ptr(v), B::ptr(s));
        }
        // slot 0: <s, v> (Rayleigh quotient, |v| = 1), slot 1: |s|^2
        AXOS_QP_RED_TO(be, dot2, n, (Dot2{B::ptr(s), B::ptr(v)}), 0);
        AXOS_QP_MAP(be, scale_norm, n,
            (ScaleNorm{B::ptr(v), B::ptr(s), be.results_dev() + 1}));
    };
    double lam = 0, prev = 0;
    bool converged = false;
    for (int k = 0; k < max_iter && !converged; k += kBlock) {
        be.run_block(key, [&] {
            for (int j = 0; j < kBlock; ++j)
                step();
            be.copy_results(2);
        });
        be.sync();
        double r[2] = {be.results()[0], be.results()[1]};
        if (cm) cm->bcast(r, 2);
        if (!(r[1] > 0) || !std::isfinite(r[0]))
            break; // M v = 0: keep the last estimate
        lam = r[0];
        converged = k > 0 && std::abs(lam - prev) <= 1e-6 * std::abs(lam);
        prev = lam;
    }
    be.forget(key);
    return converged ? lam : 1.05 * lam;
}

template <class B> class HprQp {
    using Vec = typename B::Vec;

  public:
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
        if (!bad.empty()) throw std::invalid_argument("HprQp: " + bad);
        const size_t n = p.cols(), m = p.rows();
        const QpComm one;
        const QpComm &cm = opt.comm ? *opt.comm : one;
        const bool dist = cm.distributed();
        const bool talk = cm.rank() == 0; // the rank that prints

        // ---- preconditioning and device data
        // ----------------------------------
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
        // this rank's block of constraint rows: [r0, r0 + ml) (all of them
        // on one process)
        const auto rows = cm.rows_of(q.lp.A, cm.rank());
        const size_t r0 = rows.first, ml = rows.second - rows.first;
        HostMatrix Ablock;
        if (dist) Ablock = row_block(q.lp.A, r0, r0 + ml);
        const HostMatrix &Al = dist ? Ablock : q.lp.A;
        auto local = [&](const std::vector<double> &v) {
            return std::vector<double>(v.begin() + r0, v.begin() + r0 + ml);
        };
        B be;
        if (dist) be.use_graphs(false); // MPI exchanges inside the iterations
        const bool hasQ = q.Q.nnz() > 0;
        // diagonal Q: exact w step (no semi-proximal term, hpr_factors)
        std::vector<double> hq;
        if (hasQ && q.Q.rows() == n) {
            bool diag = true;
            const auto *qrp = q.Q.row_ptr();
            const auto *qci = q.Q.col_ind();
            for (size_t i = 0; i < n && diag; ++i)
                for (auto k = qrp[i]; k < qrp[i + 1]; ++k)
                    if (static_cast<size_t>(qci[k]) != i) diag = false;
            if (diag) {
                hq.assign(n, 0.0);
                for (size_t i = 0; i < n; ++i)
                    for (auto k = qrp[i]; k < qrp[i + 1]; ++k)
                        hq[i] += q.Q.values()[k];
            }
        }
        const bool qdiag_mode = !hq.empty();
        // sweep order for (mostly) free columns (see HprFreeXW in qp_ops.h)
        size_t nfree = 0;
        for (size_t j = 0; j < n; ++j)
            if (!std::isfinite(q.lp.col_lb[j]) &&
                !std::isfinite(q.lp.col_ub[j]))
                ++nfree;
        const bool free_mode =
            opt.hpr_free_variant == 2 ||
            (opt.hpr_free_variant == 1 && nfree > 0.8 * static_cast<double>(n));
        auto A = be.upload(Al);              // ml x n
        auto AT = be.upload(Al.transpose()); // n x ml (n empty rows if ml == 0)
        // n x n even without Q: the fused steps are epilogues of Q products
        auto Qm = be.upload(q.Q.rows() == n ? q.Q : HostMatrix(n, n, 0));
        std::vector<double> cinv(n), rinv(ml);
        for (size_t j = 0; j < n; ++j)
            cinv[j] = 1.0 / sc.col[j];
        for (size_t i = 0; i < ml; ++i)
            rinv[i] = 1.0 / sc.row[r0 + i];
        Vec c = be.upload(q.lp.c), lb = be.upload(q.lp.col_lb),
            ub = be.upload(q.lp.col_ub);
        Vec l = be.upload(local(q.lp.row_lb)),
            u = be.upload(local(q.lp.row_ub));
        Vec dcinv = be.upload(cinv), drinv = be.upload(rinv);

        Vec x = be.vec(n), w = be.vec(n), qw = be.vec(n), aty = be.vec(n),
            y = be.vec(ml);
        Vec x0 = be.vec(n), w0 = be.vec(n), qw0 = be.vec(n), aty0 = be.vec(n),
            y0 = be.vec(ml);
        Vec xbar = be.vec(n), wbar = be.vec(n), qwbar = be.vec(n);
        Vec atybar = be.vec(n), ybar = be.vec(ml);
        Vec xhat = be.vec(n), whalf = be.vec(n), qwhalf = be.vec(n);
        Vec vv = be.vec(n), d = be.vec(n), qd = be.vec(n);
        Vec tn = be.vec(n), tn2 = be.vec(n),
            tm = be.vec(ml);           // scratch (KKT, restarts)
        Vec sn = be.vec(dist ? n : 0); // partial A^T products (MPI)
        Vec scal = be.vec(kScalars);   // block scalars (HprScalar)
        Vec qdv = be.upload(hq);       // diag(Q) (diagonal Q only)
        Vec qbv = be.vec(hq.size());   // 1 / (1 + sigma diag(Q))
        const double *QD = qdiag_mode ? B::ptr(qbv) : nullptr;
        auto P = [](Vec &v) { return B::ptr(v); };

        // ---- spectral estimates and initial penalty
        // ---------------------------
        const QpComm *pcm = dist ? &cm : nullptr;
        const double lamA = m ? 1.01 * power_method(be, A, &AT, n, ml,
                                           opt.power_iterations, pcm)
                              : 0.0;
        double lamQ = 0.0;
        if (qdiag_mode)
            for (double v : hq)
                lamQ = std::max(lamQ, v);
        else if (hasQ)
            lamQ = 1.01 * power_method(
                              be, Qm, nullptr, n, n, opt.power_iterations, pcm);
        double sigma = 1.0;
        {
            double nb = 0, nc = 0;
            for (size_t i = 0; i < m; ++i) {
                double b = 0;
                if (std::isfinite(q.lp.row_lb[i])) b = std::abs(q.lp.row_lb[i]);
                if (std::isfinite(q.lp.row_ub[i]))
                    b = std::max(b, std::abs(q.lp.row_ub[i]));
                nb += b * b;
            }
            for (double v : q.lp.c)
                nc += v * v;
            nb = std::sqrt(nb);
            nc = std::sqrt(nc);
            if (nb > 1e-16 && nb < 1e16 && nc > 1e-16 && nc < 1e16)
                sigma = nb / nc;
        }
        // original-scale norms for the relative residuals
        double bmax = 0, cmax = 0;
        for (size_t i = 0; i < m; ++i) {
            if (std::isfinite(p.lp.row_lb[i]))
                bmax = std::max(bmax, std::abs(p.lp.row_lb[i]));
            if (std::isfinite(p.lp.row_ub[i]))
                bmax = std::max(bmax, std::abs(p.lp.row_ub[i]));
        }
        for (double v : p.lp.c)
            cmax = std::max(cmax, std::abs(v));
        const double setup_s = elapsed();

        // ---- the iteration, as queued work
        // -------------------------------------
        const double *S = P(scal);
        // scalars of the iterations t0, t0 + 1, ... that follow
        auto set_scalars = [&](long t0) {
            const double sl = sigma * lamQ;
            AXOS_QP_MAP(be, hpr_scalars, kMaxBlock,
                (HprScalars{P(scal), sigma, sl / (1.0 + sl), 1.0 / (1.0 + sl),
                    sigma * lamA, sigma / (1.0 + sl),
                    static_cast<double>(t0)}));
        };
        auto set_qb =
            [&] { // per-element factors of a diagonal Q, after sigma changed
                if (qdiag_mode)
                    AXOS_QP_MAP(be, hpr_qb, n, (HprQb{P(qbv), P(qdv), sigma}));
            };
        set_qb();
        auto primal = [&](double *zb) {
            AXOS_QP_MAP(be, hpr_primal, n,
                (HprPrimal{P(x), P(qw), P(aty), P(c), P(lb), P(ub), P(xbar),
                    P(xhat), zb, S}));
        };
        auto first_half = [&](int j,
                              int halpern_y) { // w half step, y step, A^T ybar
            AXOS_QP_SPMV(be, hpr_whalf, Qm, P(xhat),
                (HprWHalf{
                    P(w), P(qw), P(xhat), P(whalf), P(qwhalf), P(vv), S, QD}));
            AXOS_QP_SPMV(be, hpr_dual, A, P(vv),
                (HprDual{P(y), P(ybar), P(l), P(u), P(y0), S, j, halpern_y}));
            AXOS_QP_SPMV_AT(be, cm, hpr_aty, AT, P(ybar), P(sn),
                (HprAty{P(aty), P(atybar), P(d)}));
        };
        // the free-variable order: A^T y (x, w steps), Q wbar (Q w, point v), A
        // v (y)
        auto free_iteration = [&](int j, int halpern) {
            AXOS_QP_SPMV_AT(be, cm, hpr_free_xw, AT, P(y), P(sn),
                (HprFreeXW{P(x), P(w), P(xbar), P(xhat), P(wbar), P(aty), P(qw),
                    P(c), P(lb), P(ub), P(x0), P(w0), S, QD, j, halpern}));
            AXOS_QP_SPMV(be, hpr_free_q, Qm, P(wbar),
                (HprFreeQ{P(qw), P(vv), halpern ? nullptr : P(qwbar), P(xhat),
                    P(qw0), S, j, halpern}));
            AXOS_QP_SPMV(be, hpr_dual, A, P(vv),
                (HprDual{P(y), P(ybar), P(l), P(u), P(y0), S, j, halpern}));
        };
        // L iterations without a check (offsets j = 0 .. L-1 from the scalars'
        // t0)
        auto block = [&](long L) {
            if (free_mode) {
                for (long j = 0; j < L; ++j)
                    free_iteration(static_cast<int>(j), 1);
                return;
            }
            primal(nullptr);
            for (long j = 0; j < L; ++j) {
                first_half(static_cast<int>(j), 1);
                AXOS_QP_SPMV(be, hpr_halpern, Qm, P(d),
                    (HprHalpern{P(x), P(w), P(qw), P(aty), P(xbar), P(xhat),
                        P(whalf), P(qwhalf), P(d), P(atybar), P(x0), P(w0),
                        P(qw0), P(aty0), P(c), P(lb), P(ub), S, QD,
                        static_cast<int>(j), j + 1 < L ? 1 : 0}));
            }
        };
        // one iteration with the merit terms (slots 0-8) and the KKT residuals
        // at T(u) = (xbar, ybar, atybar) (slots 9-17), with the bound duals z
        // chosen as in evaluate_qp (the projection of Q x + c - A^T y onto
        // their sign cone), so the check measures exactly what the solution is
        // scored with
        auto check_iteration = [&] {
            if (free_mode) {
                // xbar, wbar, Q wbar and ybar without the Halpern steps, then
                // A^T ybar and d = A^T(ybar - y) for the merit and the KKT test
                free_iteration(0, 0);
                AXOS_QP_SPMV_AT(be, cm, hpr_aty, AT, P(ybar), P(sn),
                    (HprAty{P(aty), P(atybar), P(d)}));
            } else {
                primal(nullptr);
                first_half(0, 0);
            }
            be.spmv(Qm, P(d), P(qd));
            // (free order: wbar, Q wbar are whalf, qwhalf; no second half)
            AXOS_QP_RED_TO(be, hpr_halpern_red, n,
                (HprHalpernRed{P(x), P(w), P(qw), P(aty), P(wbar), P(qwbar),
                    P(xbar), free_mode ? P(wbar) : P(whalf),
                    free_mode ? P(qwbar) : P(qwhalf), P(d), P(qd), P(atybar),
                    P(x0), P(w0), P(qw0), P(aty0), S, QD, 0,
                    free_mode ? 0 : 1}),
                0);
            AXOS_QP_RED_TO(be, hpr_halpern_y_red, ml,
                (HprHalpernYRed{P(y), P(ybar), P(y0), S, 0}), 8);
            be.spmv(Qm, P(xbar), P(tn));
            be.spmv(A, P(xbar), P(tm));
            AXOS_QP_RED_TO(be, kkt_rows, ml,
                (KktRows{P(tm), P(ybar), P(l), P(u), P(drinv)}), 9);
            AXOS_QP_RED_TO(be, pd_kkt_cols, n,
                (PdKktCols{
                    P(xbar), P(tn), P(atybar), P(c), P(lb), P(ub), P(dcinv)}),
                12);
            be.copy_results(18);
        };

        struct Kkt {
            double eta_p = 0, eta_d = 0, eta_gap = 0, pobj = 0, dobj = 0;
            double
            err() const
            { return std::max({eta_p, eta_d, eta_gap}); }
        };
        auto kkt_from = [&](const double *r) {
            Kkt k;
            k.pobj = 0.5 * r[15] + r[16] + q.lp.offset;
            k.dobj = -0.5 * r[15] + r[11] + r[17] + q.lp.offset;
            k.eta_p = r[9] / (1 + std::max(bmax, r[10]));
            k.eta_d = r[12] / (1 + std::max({cmax, r[13], r[14]}));
            k.eta_gap = std::abs(k.pobj - k.dobj) /
                        (1 + std::max(std::abs(k.pobj), std::abs(k.dobj)));
            return k;
        };
        auto finish = [&](Status st, long iters, const Kkt &k) {
            std::vector<double> hx, hy;
            be.download(xbar, hx);
            be.download(ybar, hy);
            double drift =
                0; // largest difference of the ranks' copies of x (checksum)
            if (dist) {
                double h[2] = {0, 0};
                for (size_t j = 0; j < n; ++j)
                    h[0] += hx[j] * double(1 + j % 7);
                h[1] = -h[0];
                cm.max(h, 2);
                drift = h[0] + h[1];
                cm.bcast(hx.data(), n); // rank 0's copy
                hy = cm.gather_rows(q.lp.A, hy);
            }
            for (size_t j = 0; j < n; ++j)
                hx[j] *= sc.col[j];
            for (size_t i = 0; i < m; ++i)
                hy[i] *= sc.row[i];
            QpSolution s = evaluate_qp(p, hx, hy);
            s.status = st;
            s.iterations = iters;
            s.seconds = elapsed();
            s.setup_seconds = setup_s;
            s.method =
                dist ? "hpr-qp (mpi, " + std::to_string(cm.size()) + " ranks)"
                     : "hpr-qp";
            s.device = be.name();
            if (opt.verbose && talk)
                std::printf(
                    "[hpr-qp] %s after %ld iterations, %.3f s (setup %.3f s): "
                    "obj %.10e  eta_p %.2e eta_d %.2e gap %.2e (internal "
                    "%.2e)\n",
                    to_string(st), iters, s.seconds, setup_s,
                    s.primal_objective, s.rel_primal, s.rel_dual, s.rel_gap,
                    k.err());
            if (opt.verbose && talk && dist)
                std::printf(
                    "[hpr-qp] mpi: %d ranks, rows %zu..%zu of %zu on rank 0; "
                    "%.3f s in %ld MPI calls (%.3g doubles); copies of x "
                    "differ by %.1e\n",
                    cm.size(), r0, r0 + ml, m, cm.seconds(), cm.calls(),
                    cm.doubles(), drift);
            return s;
        };
        auto copy = [&](Vec &dst, Vec &src, size_t len) {
            AXOS_QP_MAP(be, copy, len, (Copy{P(dst), P(src)}));
        };

        if (opt.verbose && talk)
            std::printf(
                "[hpr-qp] %s: n %zu m %zu nnz(A) %zu nnz(Q) %zu, lambda_A %.3e "
                "lambda_Q %.3e sigma0 %.3e, setup %.3f s\n",
                be.name().c_str(), n, m, q.lp.A.nnz(), q.Q.nnz(), lamA, lamQ,
                sigma, setup_s);

        if (opt.profile) { // time of each step, per call
            const int reps = 200;
            set_scalars(0);
            const double tp = be.time_us(reps, [&] { primal(nullptr); });
            const double t1 = be.time_us(reps, [&] {
                AXOS_QP_SPMV(be, hpr_whalf, Qm, P(xhat),
                    (HprWHalf{P(w), P(qw), P(xhat), P(whalf), P(qwhalf), P(vv),
                        S, QD}));
            });
            const double t2 = be.time_us(reps, [&] {
                AXOS_QP_SPMV(be, hpr_dual, A, P(vv),
                    (HprDual{P(y), P(ybar), P(l), P(u), P(y0), S, 0, 1}));
            });
            const double t3 = be.time_us(reps, [&] {
                AXOS_QP_SPMV(be, hpr_aty, AT, P(ybar),
                    (HprAty{P(aty), P(atybar), P(d)}));
            });
            const double t4 = be.time_us(reps, [&] {
                AXOS_QP_SPMV(be, hpr_halpern, Qm, P(d),
                    (HprHalpern{P(x), P(w), P(qw), P(aty), P(xbar), P(xhat),
                        P(whalf), P(qwhalf), P(d), P(atybar), P(x0), P(w0),
                        P(qw0), P(aty0), P(c), P(lb), P(ub), S, QD, 0, 1}));
            });
            const double tq =
                be.time_us(reps, [&] { be.spmv(Qm, P(xhat), P(tn)); });
            const double tb = be.time_us(20,
                [&] { be.run_block(kMaxBlock, [&] { block(kMaxBlock); }); });
            const double tc = be.time_us(20, [&] {
                be.run_block(-1, check_iteration);
                be.sync();
            });
            if (talk)
                std::printf("[hpr-qp profile] %s n %zu m %zu nnz(A) %zu nnz(Q) "
                            "%zu (us per call)\n"
                            "  z/x step %.1f | Q xhat + w %.1f | A v + y %.1f "
                            "| A^T ybar %.1f | "
                            "Q d + Halpern %.1f | plain Q x %.1f\n"
                            "  iteration in a %d-iteration graph %.1f | check "
                            "iteration %.1f\n",
                    be.name().c_str(), n, m, q.lp.A.nnz(), q.Q.nnz(), tp, t1,
                    t2, t3, t4, tq, int(kMaxBlock), tb / kMaxBlock, tc);
            return finish(Status::NotSolved, 0, Kkt{});
        }

        const long ce = std::max(1, opt.check_every);
        // Checks: every 10 iterations up to 1000, then every ce; restart
        // decisions at the multiples of ce. The first iteration of an epoch (t
        // == 0) is also a check: its merit R0 is what the epoch's later merits
        // are compared with.
        auto is_check = [&](long i, long tt) {
            return tt == 0 || i % ce == 0 ||
                   (i < 1000 && i % std::min<long>(10, ce) == 0);
        };
        long it = 0, t = 0;
        // merits R = |T(u) - u|_M: R0 at the epoch start, Rsave at the previous
        // check, Rw at the first restart check; the best one with its sigma
        double R0 = 0, Rsave = HUGE_VAL, Rw = 0, Rbest = HUGE_VAL,
               sigma_best = sigma;
        bool first = true;
        Kkt kkt;
        int restarts = 0;
        std::vector<double>
            hd; // A^T(ybar - y0) on the host (diagonal-Q penalty update)
        while (true) {
            if (!is_check(it, t)) {
                long L = 1;
                while (!is_check(it + L, t + L))
                    ++L;
                // in blocks of 64, 32, ..., 1 iterations: a few graphs serve
                // every length
                for (long len = kMaxBlock; len >= 1; len /= 2)
                    while (L >= len) {
                        set_scalars(t);
                        be.run_block(len, [&] { block(len); });
                        it += len;
                        t += len;
                        L -= len;
                    }
                continue;
            }

            set_scalars(t);
            be.run_block(-1, check_iteration);
            be.sync();
            double r[19];
            std::copy(be.results(), be.results() + 18, r);
            r[18] = elapsed();
            if (dist) { // rows split: sums and maxima of the row terms over the
                        // ranks
                double add[2] = {r[8], r[11]}, mx[3] = {r[9], r[10], r[18]};
                cm.sum(add, 2);
                cm.max(mx, 3);
                r[8] = add[0], r[11] = add[1];
                r[9] = mx[0], r[10] = mx[1], r[18] = mx[2];
                cm.bcast(
                    r, 19); // rank 0's values: the same decisions on every rank
            }
            // r: 0 |ex|^2, 1 <ew,Q ew>, 2 <Q ew,d>, 3 <d,Qd>, 4 |Q ew|^2, 5 <Q
            // ew,ex>,
            //    6 <d,ex>, 7 sum d (Qd) / (1 + s q), 8 |ybar - y|^2, 9-17 KKT
            //    terms, 18 time (the slowest rank's)
            const double M1 = sigma * (lamA * r[8] - 2.0 * r[2] +
                                          (qdiag_mode ? r[4] : lamQ * r[1]));
            const double M2 =
                r[0] / sigma - 2.0 * r[5] + 2.0 * r[6] + std::max(M1, 0.0);
            const double M3 = sigma * sigma *
                              (qdiag_mode ? r[7] : r[3] / (1.0 + sigma * lamQ));
            const double R = std::sqrt(std::max(M2, 0.0) + std::max(M3, 0.0));
            kkt = kkt_from(r);
            if (opt.verbose >= 2 && talk)
                std::printf("[hpr-qp] it %-8ld t %-7ld pobj % .10e dobj % .10e "
                            "eta_p %.2e eta_d %.2e "
                            "gap %.2e sigma %.3e R %.3e\n",
                    it, t, kkt.pobj, kkt.dobj, kkt.eta_p, kkt.eta_d,
                    kkt.eta_gap, sigma, R);
            if (!std::isfinite(kkt.err()))
                return finish(Status::NumericalError, it, kkt);
            if (kkt.err() <= opt.tol)
                return finish(Status::Optimal, it + 1, kkt);
            if (it + 1 >= opt.max_iterations)
                return finish(Status::IterationLimit, it + 1, kkt);
            if (r[18] > opt.time_limit)
                return finish(Status::TimeLimit, it + 1, kkt);

            // Restart when the merit fell to 0.2 R0 (sufficient), or to 0.8 R0
            // and grew since the last check (necessary), or the epoch is long:
            // half of all iterations, a fifth once R fell below 0.1 of the
            // first check's merit. The first restart check always restarts.
            bool restart = false;
            if (t == 0) {
                R0 = R;
            } else if (it % ce == 0) {
                if (first) {
                    restart = true;
                    first = false;
                    Rw = Rbest = R;
                    sigma_best = sigma;
                } else {
                    const bool suff = R <= 0.2 * R0;
                    const bool nec = R <= 0.8 * R0 && R > Rsave;
                    const double frac = R > 0.1 * Rw ? 0.5 : 0.2;
                    const bool lng = static_cast<double>(t) >=
                                     frac * static_cast<double>(it);
                    restart = suff || nec || lng;
                    if (R < Rbest) {
                        Rbest = R;
                        sigma_best = sigma;
                    }
                    Rsave = R;
                }
            }
            ++it;
            ++t;
            if (!restart) continue;

            // ---- restart at T(u) with a re-balanced penalty
            // ------------------------ sigma minimizes a s + b / s + s^2 g(s)
            // (the merit of the epoch's displacement), g(s) = <d, Q d> / (1 +
            // lq s) or, for diagonal Q, sum q_i d_i^2 / (1 + s q_i), with d =
            // A^T(ybar - y0); blended with the best sigma so far, and balanced
            // between the primal and dual residuals near convergence.
            const auto th = AXOS_QP_RED(be, hpr_theta_n, n,
                (HprThetaN{P(xbar), P(x0), P(wbar), P(w0), P(qwbar), P(qw0),
                    P(atybar), P(aty0), P(tn2)}));
            double dy2 =
                ml ? AXOS_QP_RED(be, diff_sq, ml, (DiffSq{P(ybar), P(y0)}))[0]
                   : 0.0;
            cm.sum(&dy2, 1);
            const double a = std::max(
                lamA * dy2 - 2.0 * th[2] + (qdiag_mode ? th[3] : lamQ * th[1]),
                1e-12);
            const double b = std::max(th[0], 1e-12);
            double s_est;
            if (m == 0) {
                s_est = std::sqrt(b / a);
            } else if (qdiag_mode) {
                be.download(tn2, hd);
                std::vector<std::pair<double, double>> qd2; // (q_i, q_i d_i^2)
                for (size_t j = 0; j < n; ++j)
                    if (hq[j] > 0 && hd[j] != 0)
                        qd2.emplace_back(hq[j], hq[j] * hd[j] * hd[j]);
                s_est = golden_log([&](double s) {
                    double g = 0;
                    for (const auto &e : qd2)
                        g += e.second / (1.0 + s * e.first);
                    return a * s + b / s + s * s * g;
                });
            } else {
                double cq = 0;
                if (hasQ) {
                    be.spmv(Qm, P(tn2), P(tn));
                    cq = AXOS_QP_RED(be, dot2, n, (Dot2{P(tn2), P(tn)}))[0];
                }
                s_est = golden_log([&](double s) {
                    return a * s + b / s + s * s * cq / (1.0 + lamQ * s);
                });
            }
            const double fact = std::exp(-0.05 * (R / Rbest));
            double s_new = std::exp(
                fact * std::log(s_est) + (1.0 - fact) * std::log(sigma_best));
            const double t1 = std::max(
                std::min(kkt.eta_d, kkt.eta_p), std::min(kkt.eta_gap, R));
            if (t1 <= 9e-10 && kkt.eta_p > 0 && kkt.eta_d > 0) {
                const double ratio = kkt.eta_d / kkt.eta_p;
                s_new *= std::min(100.0,
                    std::max(1e-2, t1 > 5e-10 ? std::sqrt(ratio) : ratio));
            }
            if (std::isfinite(s_new) && s_new > 0) sigma = s_new;
            cm.bcast(&sigma, 1);
            set_qb();
            copy(x, xbar, n);
            copy(w, wbar, n);
            copy(y, ybar, ml);
            // recompute Q w and A^T y exactly (no drift from the recurrences)
            be.spmv(Qm, P(w), P(qw));
            be.spmv(AT, P(y), P(aty));
            allreduce_sum(be, P(aty), n, cm);
            copy(x0, x, n);
            copy(w0, w, n);
            copy(qw0, qw, n);
            copy(aty0, aty, n);
            copy(y0, y, ml);
            t = 0;
            Rsave = HUGE_VAL;
            ++restarts;
            if (opt.verbose >= 2 && talk)
                std::printf("[hpr-qp] restart %d at it %ld: sigma %.3e "
                            "(estimate %.3e, a %.2e b %.2e)\n",
                    restarts, it, sigma, s_est, a, b);
        }
    }

  private:
    // argmin over s in [1e-12, 1e12] of a function that is convex in s
    // (so unimodal): golden section on log s.
    template <class F>
    static double
    golden_log(F &&f)
    {
        double lo = std::log(1e-12), hi = std::log(1e12);
        const double g = 0.5 * (std::sqrt(5.0) - 1.0);
        double a = hi - g * (hi - lo), b = lo + g * (hi - lo);
        double fa = f(std::exp(a)), fb = f(std::exp(b));
        for (int k = 0; k < 200 && hi - lo > 1e-10; ++k) {
            if (fa < fb) {
                hi = b;
                b = a;
                fb = fa;
                a = hi - g * (hi - lo);
                fa = f(std::exp(a));
            } else {
                lo = a;
                a = b;
                fa = fb;
                b = lo + g * (hi - lo);
                fb = f(std::exp(b));
            }
        }
        return std::exp(0.5 * (lo + hi));
    }
};

} // namespace qp
} // namespace Solver
} // namespace AXOS
