// SPDX-License-Identifier: BSD-3-Clause
//
// Branch and cut for MILP (CPU orchestration; see solve_milp.h for presolve
// and postsolve).
//
// Root: domain propagation, double probing, feasibility jump (LP-free), the
// LP relaxation by the dual simplex, rounds of Gomory mixed-integer cuts
// (cuts.h) and c-MIR cuts from the model rows (cmir.h), then LP
// based heuristics (rounding with an LP repair of the continuous part,
// fix-and-propagate, fractional diving, the feasibility pump when nothing
// was found yet, RENS, RINS, feasibility jump with an objective cutoff row)
// and reduced-cost fixing. RINS (relaxation induced
// neighborhood search) runs again in the tree.
//
// Tree: a node is the list of bound changes from the root. Processing one
// applies them to the global bounds, propagates (propagate.h), and re-solves
// the LP from the parent's basis with DualSimplex::resolve (the basis stays
// dual feasible under bound changes, so the dual simplex needs no phase 1).
// Nodes whose LP bound reaches the cutoff are pruned; the cutoff is the
// incumbent less the gap tolerance, and incumbent - 1 when every solution
// has an integral objective. Branching: reliability pseudocost branching -
// candidates whose pseudocosts have fewer than opt.reliability samples are
// strong branched (both children, iteration-capped dual simplex); the
// product score of the down and up gains picks the variable; a child that
// strong branching proves infeasible is not created. Node selection: best
// bound, with depth-first plunging into the child the pseudocosts favor
// while its bound stays close to the best.
//
// GPU (opt.gpu, builds with AXOS_ENABLE_CUDA; milp_cuda.h): the root probing
// runs on the GPU (many probes per batch), feasibility-jump walkers run on
// the GPU in a second host thread next to the CPU feasibility jump and the
// root LP. For models with at least opt.gpu_lp_min_nnz nonzeros the root LP
// is first solved approximately by HPR on the GPU (a first-order method,
// tolerance 1e-4): its point feeds rounding, the GPU walkers and RENS. For LPs with at least
// opt.gpu_min_nnz nonzeros the unreliable
// branching candidates are evaluated together by batched PDHG, whose
// Lagrangian bounds are valid child bounds (pruning, scores, pseudocosts).
// The tree itself stays on the CPU.
#pragma once

#include "solver/lp/simplex.h"
#include "solver/milp/cmir.h"
#include "solver/milp/cuts.h"
#include "solver/milp/heuristics.h"
#include "solver/milp/milp_model.h"
#include "solver/milp/probing.h"
#include "solver/milp/propagate.h"
#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <memory>
#include <queue>
#include <random>
#include <string>
#include <vector>
#ifdef AXOS_ENABLE_CUDA
#include "solver/milp/milp_cuda.h"
#include "solver/qp/solve_qp.h"
#include <atomic>
#include <mutex>
#include <thread>
#endif

namespace AXOS {
namespace Solver {

// solve_milp.h: presolve + branch and cut + postsolve (RINS sub-MIPs)
inline MilpSolution solve_milp(const LpProblem &orig, const MilpOptions &opt);

namespace milp {

class BranchAndBound {
    using Basis = std::shared_ptr<const std::vector<VarStatus>>;

  public:
    BranchAndBound(const LpProblem &p, const MilpOptions &opt,
        std::chrono::steady_clock::time_point t0)
        : p_(p), lp_(p), opt_(opt), prop_(p), fj_(p), cmir_(p), t0_(t0)
    {
        const size_t n = p.cols();
        glb_ = p.col_lb;
        gub_ = p.col_ub;
        for (int d = 0; d < 2; ++d) {
            pc_sum_[d].assign(n, 0.0);
            pc_n_[d].assign(n, 0);
        }
        // the objective takes integral values only: c integral on the
        // integer columns, zero on the continuous ones
        obj_integral_ = true;
        for (size_t j = 0; j < n && obj_integral_; ++j) {
            if (p.c[j] == 0) continue;
            if (!is_int(p, j) || p.c[j] != std::round(p.c[j])) obj_integral_ = false;
        }
        sopt_.scaling = true;
        sopt_.max_iterations = 50000000;
        sopt_.verbose = opt.verbose >= 3;
    }

    ~BranchAndBound() { stop_gpu_fj(); }
    BranchAndBound(const BranchAndBound &) = delete;
    BranchAndBound &operator=(const BranchAndBound &) = delete;

    MilpSolution
    solve()
    {
        MilpSolution out;
        const size_t n = p_.cols();
        // ---- root ------------------------------------------------------------
        gpu_init();
        if (!prop_.propagate(glb_, gub_)) return finish(out, Status::Infeasible);
        for (size_t j = 0; j < n; ++j)
            if (glb_[j] > gub_[j]) return finish(out, Status::Infeasible);
        if (opt_.probing && !root_probing()) return finish(out, Status::Infeasible);
        if (opt_.heuristics) {
            // GPU walkers in a second thread, racing the CPU feasibility jump;
            // they keep going during the root LP when the CPU one fails
            start_gpu_fj(std::min(0.2 * opt_.time_limit, 60.0));
            run_fj(nullptr, std::min(opt_.fj_effort * opt_.time_limit, 10.0), "feasibility jump");
            if (!best_x_.empty()) stop_gpu_fj();
        }

        LpSolution gpu_lp; // approximate root LP from the GPU (crossover start)
        const bool have_gpu_lp = gpu_root_lp(gpu_lp);
        set_time();
        SimplexBasis basis;
        LpSolution root = have_gpu_lp ? spx_.solve(lp_, sopt_, nullptr, &basis, &gpu_lp)
                                      : spx_.solve(lp_, sopt_, nullptr, &basis);
        if (have_gpu_lp && root.status != Status::Optimal && root.status != Status::Infeasible &&
            root.status != Status::Unbounded && !time_up()) {
            set_time(); // crossover failed: from scratch
            root = spx_.solve(lp_, sopt_, nullptr, &basis);
        }
        lp_iters_ += root.iterations;
        if (root.status == Status::Infeasible) return finish(out, Status::Infeasible);
        if (root.status == Status::Unbounded) return finish(out, best_x_.empty() ? Status::Unbounded : Status::Optimal);
        if (root.status != Status::Optimal) return finish(out, Status::TimeLimit);
        // the root LP: bounds from the global domain after propagation
        root = spx_.resolve(glb_, gub_, &basis, &basis);
        lp_iters_ += root.iterations;
        if (root.status == Status::Infeasible) return finish(out, Status::Infeasible);
        if (root.status != Status::Optimal) return finish(out, Status::TimeLimit);
        if (opt_.verbose)
            std::printf("[milp] root LP %.10g (%ld iterations, %.2f s)\n", root.primal_objective,
                root.iterations, elapsed());

        if (opt_.cuts) {
            root = cut_loop(root, basis);
            if (root.status == Status::Infeasible) return finish(out, Status::Infeasible);
            drop_inactive_cuts(root, basis);
        }
        out.root_bound = root.primal_objective;
        out.cuts = ncuts_;
        poll_gpu_fj();
        gpu_batch_lp_init();
        // reduced-cost fixing needs the simplex at the root LP: now, and again
        // after the heuristics if they improved the incumbent
        reduced_cost_fixing(root.primal_objective);
        const double inc0 = best_obj_;
        if (opt_.heuristics && !time_up()) {
            lp_heuristics(root.x, glb_, gub_, basis);
            if (!time_up()) dive(root.x, glb_, gub_, basis, 50);
            if (best_x_.empty() && !time_up())
                feasibility_pump(root.x, basis, std::min(0.1 * opt_.time_limit, 30.0));
            if (!time_up()) sub_mip(root.x, true);
            if (!time_up()) rins(root.x);
            if (!time_up()) fj_improve(root.x, std::min(0.08 * opt_.time_limit, 8.0));
        }
        if (best_obj_ < inc0 && !time_up()) {
            set_time();
            LpSolution r2 = spx_.resolve(glb_, gub_, &basis, &basis);
            lp_iters_ += r2.iterations;
            if (r2.status == Status::Optimal) {
                root = r2;
                reduced_cost_fixing(root.primal_objective);
            }
        }

        if (opt_.verbose)
            std::printf("[milp] root done: bound %.10g, %d cuts (%d c-MIR), %ld LP iterations, %.2f s\n",
                root.primal_objective, ncuts_, nmir_, lp_iters_, elapsed());

        // ---- tree ------------------------------------------------------------
        auto root_basis = std::make_shared<const std::vector<VarStatus>>(basis.status);
        Node rn;
        rn.bound = root.primal_objective;
        rn.basis = root_basis;
        rn.x = std::make_shared<const std::vector<double>>(root.x);
        rn.solved = true;
        double root_obj = root.primal_objective;
        // the root's LP is solved: branch it right away
        branch(rn, root_obj, root.x, &root.y, glb_, gub_, basis);
        long last_print = 0;
        while (!open_.empty() || plunge_) {
            if (time_up()) break;
            if (nodes_ >= opt_.node_limit) break;
            Node node;
            if (plunge_) {
                node = std::move(*plunge_);
                plunge_.reset();
            } else {
                node = open_.top();
                open_.pop();
            }
            if (node.bound >= cutoff()) continue;
            process(node);
            ++nodes_;
            if ((nodes_ & 63) == 0) poll_gpu_fj();
            if (opt_.heuristics && nodes_ % 200 == 0 && !time_up() && last_x_)
                dive(*last_x_, last_lb_, last_ub_, last_basis_, 30);
            if (opt_.heuristics && nodes_ >= next_rins_ && !time_up() && last_x_) {
                bool ok = rins(*last_x_);
                if (!ok && !time_up() && fj_time_ < 0.15 * opt_.time_limit) {
                    const double b = elapsed();
                    const double inc = best_obj_;
                    fj_improve(*last_x_, std::min(0.02 * opt_.time_limit, 2.0));
                    fj_time_ += elapsed() - b;
                    ok = best_obj_ < inc;
                }
                rins_fail_ = ok ? 0 : rins_fail_ + 1;
                next_rins_ = nodes_ + (500L << std::min(rins_fail_, 6));
            }
            const double gb = global_bound();
            if (!best_x_.empty() && closed(best_obj_, gb)) break;
            if (opt_.verbose && nodes_ - last_print >= 1000) {
                last_print = nodes_;
                std::printf("[milp] %7ld nodes %7zu open  best %.10g  bound %.10g  %.1f s\n", nodes_,
                    open_.size(), best_obj_, gb, elapsed());
            }
        }
        // optimal: the gap is closed (the tree may still hold nodes that cannot
        // improve the incumbent by more than the tolerance); an exhausted tree
        // with a lost node (LP failure) proves nothing beyond its bound
        Status st;
        if (!best_x_.empty() && closed(best_obj_, global_bound())) st = Status::Optimal;
        else if (open_.empty() && !plunge_)
            st = best_x_.empty() && lost_bound_ == kInf ? Status::Infeasible : Status::NumericalError;
        else st = nodes_ >= opt_.node_limit ? Status::IterationLimit : Status::TimeLimit;
        return finish(out, st);
    }

  private:
    struct Node {
        std::vector<int> var;          // bound changes from the root
        std::vector<double> lo, hi;
        double bound = -kInf;          // lower bound (parent LP or strong branching)
        Basis basis;                   // warm start (the parent's)
        int depth = 0;
        int branch_var = -1, branch_dir = 0; // for pseudocost updates
        double branch_frac = 0, parent_obj = 0;
        bool solved = false;
        std::shared_ptr<const std::vector<double>> x;
    };
    struct WorseBound {
        bool operator()(const Node &a, const Node &b) const { return a.bound > b.bound; }
    };

    const LpProblem &p_;
    LpProblem lp_; // with cuts
    const MilpOptions &opt_;
    SolverOptions sopt_;
    DualSimplex spx_;
    Propagator prop_;
    FeasibilityJump fj_;
    CmirSeparator cmir_;
    std::chrono::steady_clock::time_point t0_;
    std::vector<double> glb_, gub_;
    std::vector<double> best_x_;
    double best_obj_ = kInf, first_time_ = -1;
    std::string best_src_;
    bool obj_integral_ = false;
    std::vector<double> pc_sum_[2];
    std::vector<int> pc_n_[2];
    std::priority_queue<Node, std::vector<Node>, WorseBound> open_;
    std::unique_ptr<Node> plunge_;
    long nodes_ = 0, lp_iters_ = 0;
    double lost_bound_ = kInf; // least bound of a node given up on (LP failure)
    long next_rins_ = 500;     // node count of the next RINS call
    int rins_fail_ = 0;
    double rins_time_ = 0;     // seconds spent in RINS
    double fj_time_ = 0;       // seconds spent in feasibility jump with a cutoff in the tree
    int ncuts_ = 0, nmir_ = 0;
    size_t basis_bytes_ = 0;
    std::string notes_; // root statistics for the output (probing, GPU)
    bool gpu_ = false;   // the GPU is in use
#ifdef AXOS_ENABLE_CUDA
    std::unique_ptr<gpu::DeviceProblem> dev_;   // the problem without cuts
    std::unique_ptr<gpu::GpuPropagator> gprop_;
    std::unique_ptr<gpu::GpuBatchLp> gblp_;     // the LP with the root cuts
    long gpu_sb_calls_ = 0;
    std::thread fj_thread_;
    std::mutex fj_mu_;
    std::atomic<bool> fj_stop_{false};
    std::vector<double> fj_x_;
    bool fj_found_ = false;
    long long fj_moves_ = -1;
#endif
    // last solved node (diving starts there)
    std::shared_ptr<const std::vector<double>> last_x_;
    std::vector<double> last_lb_, last_ub_;
    SimplexBasis last_basis_;
    uint64_t seed_ = 7;

    double elapsed() const { return std::chrono::duration<double>(std::chrono::steady_clock::now() - t0_).count(); }
    bool time_up() const { return elapsed() > opt_.time_limit; }
    void set_time() { sopt_.time_limit = std::max(0.01, opt_.time_limit - elapsed()); }

    bool
    closed(double primal, double dual) const
    {
        const double d = primal - dual;
        return d <= opt_.gap_abs || d <= opt_.gap_rel * std::max(std::abs(primal), 1e-9);
    }
    // nodes with a bound at or above this cannot improve the incumbent enough
    double
    cutoff() const
    {
        if (best_x_.empty()) return kInf;
        const double tol = std::max(opt_.gap_abs, opt_.gap_rel * std::abs(best_obj_));
        if (obj_integral_) return best_obj_ - 1.0 + 1e-6;
        return best_obj_ - tol;
    }
    double
    node_bound(double lp_obj) const
    {
        return obj_integral_ ? std::ceil(lp_obj - 1e-6) : lp_obj;
    }
    double
    global_bound() const
    {
        double b = best_x_.empty() ? kInf : best_obj_;
        if (!open_.empty()) b = std::min(b, open_.top().bound);
        if (plunge_) b = std::min(b, plunge_->bound);
        return std::min(b, lost_bound_);
    }

    MilpSolution
    finish(MilpSolution &out, Status st)
    {
        stop_gpu_fj();
        out.notes = notes_;
#ifdef AXOS_ENABLE_CUDA
        if (gpu_) {
            out.notes += ";gpu_sb_batches=" + std::to_string(gpu_sb_calls_);
            if (fj_moves_ >= 0) out.notes += ";gpu_fj_moves=" + std::to_string(fj_moves_);
        }
#endif
        out.status = st;
        out.nodes = nodes_;
        out.lp_iterations = lp_iters_;
        out.seconds = elapsed();
        out.first_solution_seconds = first_time_;
        out.cuts = ncuts_;
        if (!best_x_.empty()) {
            out.x = best_x_;
            out.objective = best_obj_;
            out.incumbent_source = best_src_;
        }
        if (st == Status::Infeasible) out.bound = kInf;
        else out.bound = std::min(global_bound(), out.objective);
        return out;
    }

    // ---- GPU -----------------------------------------------------------------------
    void
    gpu_init()
    {
#ifdef AXOS_ENABLE_CUDA
        if (!opt_.gpu) return;
        try {
            const double t = elapsed();
            gpu::Module &mod = gpu::Module::get();
            dev_.reset(new gpu::DeviceProblem(p_));
            gprop_.reset(new gpu::GpuPropagator(*dev_));
            gpu_ = true;
            if (opt_.verbose)
                std::printf("[milp] GPU %s (kernels compiled in %.2f s, setup %.2f s)\n", mod.device().c_str(),
                    mod.compile_seconds(), elapsed() - t);
        } catch (const std::exception &e) {
            if (opt_.verbose) std::printf("[milp] GPU unavailable, CPU only: %s\n", e.what());
            gprop_.reset();
            dev_.reset();
            gpu_ = false;
        }
#endif
    }

    // Double probing on the binaries (GPU batches, else CPU); false when it
    // proves the problem infeasible.
    bool
    root_probing()
    {
        const double budget = std::min(opt_.probing_time_frac * opt_.time_limit, 10.0);
        int probes = 0, fixed = 0, tight = 0;
        bool infeasible = false;
        double secs = 0;
        const char *dev = "cpu";
#ifdef AXOS_ENABLE_CUDA
        if (gpu_) {
            try {
                const gpu::ProbeResult r = gpu::probe_binaries(*dev_, p_, glb_, gub_, budget);
                probes = r.probes; fixed = r.fixed; tight = r.tightened; infeasible = r.infeasible; secs = r.seconds;
                dev = "gpu";
            } catch (const std::exception &e) {
                if (opt_.verbose) std::printf("[milp] GPU probing failed: %s\n", e.what());
                gpu_ = false;
            }
        }
#endif
        if (std::string(dev) == "cpu") {
            const ProbingStats r = probe_binaries_cpu(prop_, p_, glb_, gub_, budget);
            probes = r.probes; fixed = r.fixed; tight = r.tightened; infeasible = r.infeasible; secs = r.seconds;
        }
        notes_ += std::string(notes_.empty() ? "" : ";") + "probing_" + dev + "=" + std::to_string(probes) + "/" +
                  std::to_string(fixed) + "/" + std::to_string(tight);
        if (opt_.verbose)
            std::printf("[milp] probing (%s): %d probes, %d binaries fixed, %d bounds tightened, %.3f s\n", dev, probes,
                fixed, tight, secs);
        if (infeasible) return false;
        if (fixed + tight > 0 && !prop_.propagate(glb_, gub_)) return false;
        for (size_t j = 0; j < p_.cols(); ++j)
            if (glb_[j] > gub_[j]) return false;
        return true;
    }

    void
    start_gpu_fj(double seconds, std::vector<std::vector<double>> starts = {})
    {
#ifdef AXOS_ENABLE_CUDA
        if (!gpu_ || fj_thread_.joinable()) return;
        std::vector<uint8_t> isint(p_.cols());
        for (size_t j = 0; j < p_.cols(); ++j) isint[j] = is_int(p_, j) ? 1 : 0;
        fj_stop_ = false;
        fj_thread_ = std::thread([this, lb = glb_, ub = gub_, isint, seconds, starts]() {
            try {
                gpu::GpuFeasibilityJump g(*dev_, opt_.gpu_fj_walkers);
                std::vector<double> x;
                long long moves = 0;
                const bool ok = g.run(lb, ub, starts, isint, seconds, 12345, x, &moves, &fj_stop_);
                std::lock_guard<std::mutex> lk(fj_mu_);
                fj_moves_ = std::max<long long>(fj_moves_, 0) + moves;
                if (ok) {
                    fj_x_ = x;
                    fj_found_ = true;
                }
            } catch (const std::exception &) {
            }
        });
#else
        (void)seconds;
        (void)starts;
#endif
    }

    // takes a solution of the GPU walkers (main thread)
    void
    poll_gpu_fj()
    {
#ifdef AXOS_ENABLE_CUDA
        std::vector<double> x;
        {
            std::lock_guard<std::mutex> lk(fj_mu_);
            if (!fj_found_) return;
            x.swap(fj_x_);
            fj_found_ = false;
        }
        try_solution(x, "feasibility jump (GPU)");
#endif
    }

    void
    stop_gpu_fj()
    {
#ifdef AXOS_ENABLE_CUDA
        if (fj_thread_.joinable()) {
            fj_stop_ = true;
            fj_thread_.join();
        }
        poll_gpu_fj();
#endif
    }

    // The root LP relaxation by HPR on the GPU (tolerance 1e-4), for models
    // with at least opt.gpu_lp_min_nnz nonzeros. Its point is rounded, starts
    // the GPU feasibility-jump walkers (when there is no incumbent yet) and
    // RENS; returns true with out = (x, y) when it converged, as a crossover
    // start for the dual simplex.
    bool
    gpu_root_lp(LpSolution &out)
    {
#ifdef AXOS_ENABLE_CUDA
        if (!gpu_ || static_cast<long>(p_.A.nnz()) < opt_.gpu_lp_min_nnz || time_up()) return false;
        const size_t n = p_.cols();
        QpProblem q;
        q.lp = p_;
        q.lp.col_lb = glb_;
        q.lp.col_ub = gub_;
        q.lp.is_integer.clear();
        q.Q = HostMatrix(n, n, std::vector<int32_t>(n + 1, 0), std::vector<int32_t>(), std::vector<double>());
        QpOptions qo;
        qo.method = QpMethod::HprQp;
        qo.use_gpu = true;
        qo.tol = 1e-4;
        qo.time_limit = std::min(0.1 * opt_.time_limit, std::max(1.0, opt_.time_limit - elapsed() - 1.0));
        stop_gpu_fj(); // the walkers would share the GPU with the LP
        const double t = elapsed();
        QpSolution s;
        try {
            s = solve_qp(q, qo);
        } catch (const std::exception &e) {
            if (opt_.verbose) std::printf("[milp] GPU LP failed: %s\n", e.what());
            return false;
        }
        notes_ += ";gpu_lp=" + std::string(to_string(s.status)) + "/" + std::to_string(elapsed() - t).substr(0, 5) + "s";
        if (opt_.verbose)
            std::printf("[milp] GPU LP (HPR): %s, objective %.10g, %ld iterations, %.2f s\n", to_string(s.status),
                s.primal_objective, s.iterations, elapsed() - t);
        if (s.x.size() != n || s.y.size() != p_.rows()) return false;
        if (opt_.heuristics) {
            try_solution(s.x, "GPU LP rounding");
            if (best_x_.empty()) start_gpu_fj(std::min(0.2 * opt_.time_limit, 60.0), {s.x});
            if (!time_up()) sub_mip(s.x, true); // RENS around the GPU LP point
        }
        // crossover from this point measured slower than the cold dual simplex
        // (academictimetablesmall 10 s vs 3.4 s, neos-950242 4.9 s vs 2.4 s):
        // the point only seeds the heuristics
        if (s.status != Status::Optimal || !opt_.gpu_lp_crossover) return false;
        out = LpSolution();
        out.x = s.x;
        out.y = s.y;
        out.status = s.status;
        return true;
#else
        (void)out;
        return false;
#endif
    }

    // batched PDHG strong branching for LPs above the size threshold
    void
    gpu_batch_lp_init()
    {
#ifdef AXOS_ENABLE_CUDA
        if (!gpu_ || static_cast<long>(lp_.A.nnz()) < opt_.gpu_min_nnz) return;
        try {
            gblp_.reset(new gpu::GpuBatchLp(lp_));
        } catch (const std::exception &e) {
            if (opt_.verbose) std::printf("[milp] GPU batch LP unavailable: %s\n", e.what());
        }
#endif
    }

    // ---- incumbents ----------------------------------------------------------
    bool
    try_solution(const std::vector<double> &x, const char *src)
    {
        std::vector<double> xr = x;
        for (size_t j = 0; j < xr.size(); ++j)
            if (is_int(p_, j)) xr[j] = std::round(xr[j]);
        const MilpCheck c = check_milp(p_, xr);
        if (!c.ok(opt_.feas_tol, opt_.int_tol)) return false;
        if (c.objective >= best_obj_ - 1e-9 * (1 + std::abs(best_obj_))) return false;
        best_obj_ = c.objective;
        best_x_ = xr;
        best_src_ = src;
        if (first_time_ < 0) first_time_ = elapsed();
        if (opt_.verbose)
            std::printf("[milp] incumbent %.10g (%s, %.2f s, %ld nodes)\n", best_obj_, src, elapsed(), nodes_);
        return true;
    }

    void
    run_fj(const std::vector<double> *x0, double seconds, const char *src)
    {
        std::vector<double> x;
        if (fj_.run(glb_, gub_, x0, seconds, x, seed_++)) try_solution(x, src);
    }

    // ---- root cuts -------------------------------------------------------------
    LpSolution
    cut_loop(LpSolution root, SimplexBasis &basis)
    {
        const size_t n = p_.cols();
        std::vector<int> tidx, touched;
        std::vector<double> tcoef, dense(n, 0.0);
        double last = root.primal_objective;
        int stall = 0;
        const double t_end = elapsed() + std::max(0.5, opt_.cut_time_frac * opt_.time_limit);
        for (int round = 0; round < opt_.cut_rounds && !time_up() && elapsed() < t_end; ++round) {
            std::vector<Cut> cuts;
            Cut cut;
            const int m = spx_.num_rows();
            // fractional basic integers, most fractional first
            std::vector<std::pair<double, int>> rows;
            for (int r = 0; r < m; ++r) {
                const int k = spx_.basic_at(r);
                if (k >= static_cast<int>(n) || !is_int(lp_, k)) continue;
                const double v = spx_.value_of(k), f = v - std::floor(v);
                if (f > 0.01 && f < 0.99) rows.emplace_back(-std::min(f, 1 - f), r);
            }
            std::sort(rows.begin(), rows.end());
            for (const auto &pr : rows) {
                if (cuts.size() >= 100) break;
                if (gmi_cut(spx_, lp_, pr.second, glb_, gub_, root.x, cut, tidx, tcoef, dense, touched))
                    cuts.push_back(cut);
            }
            const size_t ngmi = cuts.size();
            {
                const double left = std::max(0.0, t_end - elapsed());
                const auto deadline = std::chrono::steady_clock::now() +
                                      std::chrono::microseconds(static_cast<long long>(0.5e6 * left));
                cmir_.separate(root.x, glb_, gub_, cuts, 100, deadline);
            }
            const size_t nmir = cuts.size() - ngmi;
            select_cuts(cuts, 150, dense);
            nmir_ += static_cast<int>(nmir);
            if (opt_.verbose >= 2)
                std::printf("[milp]   separated %zu GMI + %zu c-MIR, kept %zu\n", ngmi, nmir, cuts.size());
            if (cuts.empty()) break;
            append_cuts(lp_, cuts);
            ncuts_ += static_cast<int>(cuts.size());
            // warm start: the old basis plus the new rows' slacks basic
            for (size_t c = 0; c < cuts.size(); ++c) basis.status.push_back(VarStatus::Basic);
            set_time();
            LpSolution s = spx_.solve(lp_, sopt_, &basis, &basis);
            lp_iters_ += s.iterations;
            if (s.status != Status::Optimal) break;
            s = spx_.resolve(glb_, gub_, &basis, &basis);
            lp_iters_ += s.iterations;
            if (s.status == Status::Infeasible) {
                root = s;
                break;
            }
            if (s.status != Status::Optimal) break;
            root = s;
            if (opt_.verbose)
                std::printf("[milp] cut round %d: %zu GMI cuts, LP %.10g (%ld iterations, %.2f s)\n", round + 1,
                    cuts.size(), root.primal_objective, s.iterations, elapsed());
            const double gain = root.primal_objective - last;
            last = root.primal_objective;
            // a round counts as stalled below 0.1% of the objective (not the
            // gap to the incumbent: an early incumbent can be far off)
            const double min_gain = 1e-3 * std::max(1.0, std::abs(last));
            if (gain <= min_gain) {
                if (++stall >= 2) break;
            } else {
                stall = 0;
            }
            if (ncuts_ > 4 * static_cast<int>(p_.rows()) + 1000) break;
        }
        return root;
    }

    // Keeps at most max_keep cuts, most efficacious first, skipping cuts nearly
    // parallel (cosine > 0.98) to one already kept. dense: scratch (n), zero.
    static void
    select_cuts(std::vector<Cut> &cuts, size_t max_keep, std::vector<double> &dense)
    {
        std::sort(cuts.begin(), cuts.end(), [](const Cut &a, const Cut &b) { return a.efficacy > b.efficacy; });
        std::vector<Cut> kept;
        std::vector<double> norm;
        for (Cut &c : cuts) {
            if (kept.size() >= max_keep) break;
            double nc = 0;
            for (size_t t = 0; t < c.idx.size(); ++t) {
                dense[c.idx[t]] = c.coef[t];
                nc += c.coef[t] * c.coef[t];
            }
            nc = std::sqrt(nc);
            bool parallel = false;
            const size_t from = kept.size() > 60 ? kept.size() - 60 : 0;
            for (size_t k = from; k < kept.size() && !parallel; ++k) {
                double dot = 0;
                const Cut &o = kept[k];
                for (size_t t = 0; t < o.idx.size(); ++t) dot += dense[o.idx[t]] * o.coef[t];
                parallel = dot > 0.98 * nc * norm[k];
            }
            for (int j : c.idx) dense[j] = 0;
            if (parallel) continue;
            kept.push_back(std::move(c));
            norm.push_back(nc);
        }
        cuts.swap(kept);
    }

    // Removes the cuts whose slack is basic at the root optimum: the basis stays
    // optimal for the smaller LP (those rows have zero duals), and every node
    // re-solves the smaller one. Restores the full LP if the re-solve fails.
    void
    drop_inactive_cuts(LpSolution &root, SimplexBasis &basis)
    {
        const size_t m0 = p_.rows(), m = lp_.rows(), n = lp_.cols();
        if (m == m0 || basis.status.size() != n + m) return;
        std::vector<char> keep(m, 1);
        int dropped = 0;
        for (size_t i = m0; i < m; ++i)
            if (basis.status[n + i] == VarStatus::Basic) {
                keep[i] = 0;
                ++dropped;
            }
        if (!dropped) return;
        const LpProblem full = lp_;
        const SimplexBasis full_basis = basis;
        remove_rows(lp_, keep);
        SimplexBasis b;
        b.status.assign(basis.status.begin(), basis.status.begin() + n);
        for (size_t i = 0; i < m; ++i)
            if (keep[i]) b.status.push_back(basis.status[n + i]);
        set_time();
        LpSolution s = spx_.solve(lp_, sopt_, &b, &b);
        lp_iters_ += s.iterations;
        if (s.status == Status::Optimal) {
            s = spx_.resolve(glb_, gub_, &b, &b);
            lp_iters_ += s.iterations;
        }
        if (s.status == Status::Optimal) {
            if (opt_.verbose)
                std::printf("[milp] dropped %d inactive cuts, %zu left, LP %.10g\n", dropped,
                    lp_.rows() - m0, s.primal_objective);
            root = s;
            basis = b;
            return;
        }
        lp_ = full; // keep everything
        basis = full_basis;
        set_time();
        LpSolution r = spx_.solve(lp_, sopt_, &basis, &basis);
        lp_iters_ += r.iterations;
        if (r.status == Status::Optimal) r = spx_.resolve(glb_, gub_, &basis, &basis);
        if (r.status == Status::Optimal) root = r;
    }

    // ---- LP-based heuristics ----------------------------------------------------
    // Fix the integers (at given values) and solve the LP of the continuous
    // part; an integer-only problem only needs the feasibility check.
    bool
    repair(const std::vector<double> &lb0, const std::vector<double> &ub0, const SimplexBasis &warm, const char *src)
    {
        bool cont = false;
        for (size_t j = 0; j < p_.cols() && !cont; ++j)
            if (!is_int(p_, j) && lb0[j] < ub0[j]) cont = true;
        if (!cont) {
            std::vector<double> x(lb0);
            return try_solution(x, src);
        }
        set_time();
        LpSolution s = spx_.resolve(lb0, ub0, &warm, nullptr);
        lp_iters_ += s.iterations;
        if (s.status != Status::Optimal) return false;
        return try_solution(s.x, src);
    }

    void
    lp_heuristics(const std::vector<double> &x, const std::vector<double> &lb, const std::vector<double> &ub,
        const SimplexBasis &warm)
    {
        const size_t n = p_.cols();
        // 1. simple rounding + LP repair
        {
            std::vector<double> l = lb, u = ub;
            bool ok = true;
            for (size_t j = 0; j < n && ok; ++j)
                if (is_int(p_, j)) {
                    const double v = std::min(std::max(std::round(x[j]), l[j]), u[j]);
                    l[j] = u[j] = v;
                }
            if (ok && prop_.propagate(l, u)) repair(l, u, warm, "rounding");
        }
        if (time_up()) return;
        // 2. fix-and-propagate, most decided integers first
        {
            std::vector<std::pair<double, int>> ord;
            for (size_t j = 0; j < n; ++j)
                if (is_int(p_, j)) ord.emplace_back(std::abs(x[j] - std::round(x[j])), static_cast<int>(j));
            std::sort(ord.begin(), ord.end());
            std::vector<int> order;
            for (auto &pr : ord) order.push_back(pr.second);
            std::vector<double> l = lb, u = ub;
            if (fix_and_propagate(prop_, p_, order, x, l, u, opt_.time_limit, t0_))
                repair(l, u, warm, "fix-and-propagate");
        }
        if (time_up()) return;
        // 3. feasibility jump from the LP point
        if (best_x_.empty()) run_fj(&x, std::min(0.02 * opt_.time_limit, 5.0), "feasibility jump (LP start)");
    }

    // Fractional diving: fix the least fractional integer to its rounding,
    // propagate, re-solve; one backtrack per level.
    void
    dive(const std::vector<double> &x0, const std::vector<double> &lb0, const std::vector<double> &ub0,
        const SimplexBasis &warm0, int max_depth)
    {
        std::vector<double> x = x0, lb = lb0, ub = ub0;
        SimplexBasis b = warm0;
        std::vector<int> changed(1);
        for (int depth = 0; depth < max_depth && !time_up(); ++depth) {
            int jb = -1;
            double fb = 2;
            for (size_t j = 0; j < x.size(); ++j) {
                if (!is_int(p_, j) || lb[j] == ub[j]) continue;
                const double f = std::abs(x[j] - std::round(x[j]));
                if (f > opt_.int_tol && f < fb) {
                    fb = f;
                    jb = static_cast<int>(j);
                }
            }
            if (jb < 0) {
                try_solution(x, "diving");
                return;
            }
            bool ok = false;
            for (int attempt = 0; attempt < 2 && !ok; ++attempt) {
                const double r = std::round(x[jb]);
                const double v = attempt == 0 ? r : (x[jb] > r ? r + 1 : r - 1);
                if (v < lb[jb] || v > ub[jb]) continue;
                std::vector<double> l = lb, u = ub;
                l[jb] = u[jb] = v;
                changed[0] = jb;
                if (!prop_.propagate(l, u, &changed, nullptr, 2.0)) continue;
                set_time();
                SimplexBasis nb;
                LpSolution s = spx_.resolve(l, u, &b, &nb, 2000);
                lp_iters_ += s.iterations;
                if (s.status != Status::Optimal || s.primal_objective >= cutoff()) continue;
                x = s.x;
                lb.swap(l);
                ub.swap(u);
                b = std::move(nb);
                ok = true;
            }
            if (!ok) return;
        }
    }

    // Objective feasibility pump (M. Fischetti, F. Glover, A. Lodi, "The
    // feasibility pump", Math. Prog. 104 (2005); T. Achterberg, T. Berthold,
    // "Improving the feasibility pump", Discrete Optimization 4 (2007)): the
    // integers of the LP point are rounded, then the LP point closest (L1) to
    // that rounding is found, with the original objective mixed in at a
    // weight that decays by 0.9 per round; until the LP point is integral.
    // The distance is linear in the binaries and in general integers at a
    // bound; others are left out. A rounding met again flips the T integers
    // farthest from it (T random in [10, 30)); a longer cycle perturbs at
    // random. Each LP is warm-started by the primal simplex (only the costs
    // change). The integral LP points go through try_solution.
    bool
    feasibility_pump(const std::vector<double> &x0, const SimplexBasis &basis0, double seconds)
    {
        const size_t n = p_.cols();
        std::vector<int> ints;
        for (size_t j = 0; j < n; ++j)
            if (is_int(p_, j) && glb_[j] < gub_[j]) ints.push_back(static_cast<int>(j));
        if (ints.empty()) return false;
        const double t0 = elapsed(), deadline = std::min(t0 + seconds, opt_.time_limit);
        double cn = 0;
        for (size_t j = 0; j < n; ++j) cn += p_.c[j] * p_.c[j];
        cn = std::sqrt(cn);
        const double dn = std::sqrt(static_cast<double>(ints.size()));
        std::vector<double> x = x0, xr(n, 0.0), prev(n, kInf), c(n, 0.0);
        std::vector<uint64_t> seen;
        SimplexBasis b = basis0;
        double alpha = 1.0;
        std::mt19937_64 rng(seed_++);
        bool found = false;
        int it = 0;
        for (; it < 500 && elapsed() < deadline && !found; ++it) {
            // round, and stop at an integral LP point
            double frac = 0;
            for (int j : ints) {
                xr[j] = std::min(std::max(std::round(x[j]), glb_[j]), gub_[j]);
                frac += std::abs(x[j] - xr[j]);
            }
            if (frac < 1e-6 * ints.size()) {
                found = try_solution(x, "feasibility pump");
                break;
            }
            // cycles: the same rounding as last round, or one seen before
            bool same = true;
            for (int j : ints) same = same && xr[j] == prev[j];
            if (same) {
                std::vector<std::pair<double, int>> away;
                for (int j : ints) away.emplace_back(-std::abs(x[j] - xr[j]), j);
                std::sort(away.begin(), away.end());
                const int T = 10 + static_cast<int>(rng() % 20);
                for (int t = 0; t < T && t < static_cast<int>(away.size()); ++t) {
                    const int j = away[t].second;
                    if (away[t].first == 0) break;
                    xr[j] = xr[j] > x[j] ? xr[j] - 1 : xr[j] + 1;
                    xr[j] = std::min(std::max(xr[j], glb_[j]), gub_[j]);
                }
            } else {
                uint64_t h = 1469598103934665603ull;
                for (int j : ints) h = (h ^ static_cast<uint64_t>(xr[j] + 7)) * 1099511628211ull;
                if (std::find(seen.begin(), seen.end(), h) != seen.end()) {
                    std::uniform_real_distribution<double> u(-0.3, 0.7);
                    for (int j : ints) {
                        const double r = std::abs(x[j] - xr[j]) + std::max(u(rng), 0.0);
                        if (r > 0.5) xr[j] = std::min(std::max(xr[j] > x[j] ? xr[j] - 1 : xr[j] + 1, glb_[j]), gub_[j]);
                    }
                }
                seen.push_back(h);
                if (seen.size() > 100) seen.erase(seen.begin());
            }
            for (int j : ints) prev[j] = xr[j];
            // the distance objective (plus the decaying original objective)
            alpha *= 0.9;
            std::fill(c.begin(), c.end(), 0.0);
            for (int j : ints) {
                if (xr[j] <= glb_[j]) c[j] = 1.0;
                else if (xr[j] >= gub_[j]) c[j] = -1.0;
            }
            if (alpha > 1e-3 && cn > 0)
                for (size_t j = 0; j < n; ++j) c[j] = (1 - alpha) * c[j] + alpha * dn / cn * p_.c[j];
            set_time();
            LpSolution s = spx_.reoptimize(c, glb_, gub_, &b, &b, 20000);
            lp_iters_ += s.iterations;
            if (s.status != Status::Optimal) break;
            x = s.x;
        }
        spx_.restore_costs();
        if (opt_.verbose)
            std::printf("[milp] feasibility pump: %s after %d rounds, %.2f s\n", found ? "solution" : "nothing", it,
                elapsed() - t0);
        return found;
    }

    // Feasibility jump as an improvement heuristic: the model plus the cutoff
    // row c^T x <= incumbent - delta, searched from the LP point x; every
    // improving point tightens the cutoff and the search goes on (it uses
    // what an LP-free local search is good at: moving a poor incumbent when
    // too few integers agree with the LP point for RINS).
    void
    fj_improve(const std::vector<double> &x, double seconds)
    {
        if (best_x_.empty() || !opt_.heuristics) return;
        const size_t n = p_.cols();
        Cut obj;
        for (size_t j = 0; j < n; ++j)
            if (p_.c[j] != 0) {
                obj.idx.push_back(static_cast<int>(j));
                obj.coef.push_back(-p_.c[j]);
            }
        if (obj.idx.empty()) return;
        const double t_end = std::min(elapsed() + seconds, opt_.time_limit);
        int found = 0;
        while (elapsed() < t_end) {
            const double delta =
                obj_integral_ ? 1.0 - 1e-6 : std::max(opt_.gap_abs, 1e-3 * std::max(1.0, std::abs(best_obj_)));
            obj.rhs = -(best_obj_ - delta - p_.offset);
            LpProblem pc = p_;
            append_cuts(pc, {obj});
            FeasibilityJump fjc(pc);
            std::vector<double> xs;
            if (!fjc.run(glb_, gub_, &x, t_end - elapsed(), xs, seed_++)) break;
            if (!try_solution(xs, "feasibility jump (cutoff)")) break;
            ++found;
        }
        if (opt_.verbose >= 2) std::printf("[milp] FJ with cutoff: %d improvements\n", found);
    }

    // RINS: the integer columns where the LP point x agrees with the incumbent
    // are fixed, and the rest is solved as a sub-MIP (presolved, node and time
    // limits, no RINS of its own) with a cutoff row c^T x <= incumbent - delta,
    // so only an improving solution comes back. Skipped when less than half
    // of the integers would be fixed; at most a fifth of the time limit in all.
    bool rins(const std::vector<double> &x) { return sub_mip(x, false); }

    // RINS (rens = false) or RENS (rens = true, T. Berthold, "RENS - the
    // optimal rounding", Math. Prog. Comp. 6 (2014)): RENS fixes the integers
    // that are integral at the LP point and restricts the others to their
    // floor and ceiling; it needs no incumbent. With an incumbent a cutoff
    // row keeps only improving solutions.
    bool
    sub_mip(const std::vector<double> &x, bool rens)
    {
        if (!opt_.rins || (!rens && best_x_.empty())) return false;
        const double budget = std::min({0.05 * opt_.time_limit, 10.0, 0.2 * opt_.time_limit - rins_time_,
            opt_.time_limit - elapsed() - 0.5});
        if (budget < 0.2) return false;
        const size_t n = p_.cols();
        LpProblem sub = p_;
        sub.col_lb = glb_;
        sub.col_ub = gub_;
        size_t nint = 0, nfix = 0;
        for (size_t j = 0; j < n; ++j) {
            if (!is_int(p_, j)) continue;
            ++nint;
            if (glb_[j] == gub_[j]) {
                ++nfix;
                continue;
            }
            if (rens) {
                const double r = std::round(x[j]);
                if (std::abs(x[j] - r) < 1e-6) {
                    sub.col_lb[j] = sub.col_ub[j] = std::min(std::max(r, glb_[j]), gub_[j]);
                    ++nfix;
                } else {
                    sub.col_lb[j] = std::max(glb_[j], std::floor(x[j]));
                    sub.col_ub[j] = std::min(gub_[j], std::ceil(x[j]));
                }
            } else if (std::abs(x[j] - best_x_[j]) < 1e-6) {
                sub.col_lb[j] = sub.col_ub[j] = best_x_[j];
                ++nfix;
            }
        }
        if (nint == 0 || nfix < nint / 2 || nfix == nint) return false;
        if (!best_x_.empty()) {
            const double delta =
                obj_integral_ ? 1.0 - 1e-6 : std::max(opt_.gap_abs, opt_.gap_rel * std::abs(best_obj_));
            Cut obj;
            for (size_t j = 0; j < n; ++j)
                if (p_.c[j] != 0) {
                    obj.idx.push_back(static_cast<int>(j));
                    obj.coef.push_back(-p_.c[j]); // -c^T x >= -(best - delta - offset)
                }
            if (obj.idx.empty()) return false;
            obj.rhs = -(best_obj_ - delta - p_.offset);
            append_cuts(sub, {obj});
        }
        MilpOptions so;
        so.time_limit = budget;
        so.node_limit = 1000;
        so.gap_rel = opt_.gap_rel;
        so.gap_abs = opt_.gap_abs;
        so.rins = false;
        so.probing = false;
        so.gpu = false;
        so.verbose = 0;
        so.cut_rounds = 5;
        so.fj_effort = 0.02;
        const double t = elapsed();
        bool improved = false;
        try {
            const MilpSolution r = solve_milp(sub, so);
            if (r.has_solution()) improved = try_solution(r.x, rens ? "RENS" : "RINS");
        } catch (const std::exception &) {
        }
        rins_time_ += elapsed() - t;
        if (opt_.verbose >= 2)
            std::printf("[milp] %s: fixed %zu of %zu integers, %s, %.2f s\n", rens ? "RENS" : "RINS", nfix, nint,
                improved ? "improved" : "no improvement", elapsed() - t);
        return improved;
    }

    // ---- reduced-cost fixing at the root -----------------------------------
    void
    reduced_cost_fixing(double root_obj)
    {
        if (best_x_.empty()) return;
        const double gap = best_obj_ - root_obj;
        if (gap <= 0) return;
        int fixed = 0;
        for (size_t j = 0; j < p_.cols(); ++j) {
            if (!is_int(p_, j) || glb_[j] == gub_[j]) continue;
            const VarStatus st = spx_.status_of(static_cast<int>(j));
            const double d = spx_.reduced_cost(static_cast<int>(j));
            if (st == VarStatus::AtLower && d > 1e-9) {
                const double nu = glb_[j] + std::floor(gap / d + 1e-9);
                if (nu < gub_[j]) { gub_[j] = nu; ++fixed; }
            } else if (st == VarStatus::AtUpper && d < -1e-9) {
                const double nl = gub_[j] - std::floor(gap / -d + 1e-9);
                if (nl > glb_[j]) { glb_[j] = nl; ++fixed; }
            }
        }
        if (opt_.verbose && fixed) std::printf("[milp] reduced-cost fixing tightened %d bounds\n", fixed);
    }

    // ---- nodes ------------------------------------------------------------------
    void
    process(Node &node)
    {
        std::vector<double> lb = glb_, ub = gub_;
        for (size_t t = 0; t < node.var.size(); ++t) {
            const int j = node.var[t];
            lb[j] = std::max(lb[j], node.lo[t]);
            ub[j] = std::min(ub[j], node.hi[t]);
            if (lb[j] > ub[j]) return; // empty
        }
        if (opt_.propagation && !prop_.propagate(lb, ub, &node.var, nullptr, 3.0)) return;
        SimplexBasis warm, outb;
        if (node.basis) warm.status = *node.basis;
        else warm = last_basis_;
        set_time();
        LpSolution s = spx_.resolve(lb, ub, &warm, &outb);
        lp_iters_ += s.iterations;
        if (opt_.verbose >= 2)
            std::printf("[milp] node %ld depth %d: %s %.10g (%ld iterations, %.2f s)\n", nodes_, node.depth,
                to_string(s.status), s.primal_objective, s.iterations, elapsed());
        if (s.status == Status::Infeasible) return;
        if (s.status != Status::Optimal) {
            // retry from the slack basis once (numerical trouble)
            s = spx_.resolve(lb, ub, nullptr, &outb);
            lp_iters_ += s.iterations;
            if (s.status == Status::Infeasible) return;
            if (s.status != Status::Optimal) {
                if (time_up()) { open_.push(node); return; } // keep it open: the bound stays valid
                // give up on the node: its bound caps what the search can prove
                lost_bound_ = std::min(lost_bound_, node.bound);
                if (opt_.verbose) std::printf("[milp] node LP failed (%s), node given up\n", to_string(s.status));
                return;
            }
        }
        const double obj = s.primal_objective;
        // pseudocost update for the branching that created this node
        if (node.branch_var >= 0 && node.branch_frac > 0) {
            const double gain = std::max(0.0, obj - node.parent_obj) / node.branch_frac;
            pc_sum_[node.branch_dir][node.branch_var] += gain;
            pc_n_[node.branch_dir][node.branch_var] += 1;
        }
        if (node_bound(obj) >= cutoff()) return;
        last_x_ = std::make_shared<const std::vector<double>>(s.x);
        last_lb_ = lb;
        last_ub_ = ub;
        last_basis_ = outb;
        node.bound = node_bound(obj);
        node.basis = std::make_shared<const std::vector<VarStatus>>(outb.status);
        branch(node, obj, s.x, &s.y, lb, ub, outb);
    }

    // average gain per unit of a branching on j (unknown: the mean over the
    // columns with samples, refreshed per branching decision, or 1)
    double pc_mean_[2] = {1.0, 1.0};
    void
    refresh_pc_mean()
    {
        for (int d = 0; d < 2; ++d) {
            double s = 0;
            long c = 0;
            for (size_t k = 0; k < pc_n_[d].size(); ++k)
                if (pc_n_[d][k] > 0) {
                    s += pc_sum_[d][k] / pc_n_[d][k];
                    ++c;
                }
            pc_mean_[d] = c ? s / c : 1.0;
        }
    }
    double
    pc_avg(int dir, int j) const
    {
        return pc_n_[dir][j] > 0 ? pc_sum_[dir][j] / pc_n_[dir][j] : pc_mean_[dir];
    }

    // Chooses the branching variable at an LP solution and creates the
    // children (or reports an integral LP solution).
    void
    branch(Node &node, double obj, const std::vector<double> &x, const std::vector<double> *y,
        const std::vector<double> &lb, const std::vector<double> &ub, const SimplexBasis &basis)
    {
        const size_t n = p_.cols();
        std::vector<int> frac;
        for (size_t j = 0; j < n; ++j) {
            if (!is_int(p_, j)) continue;
            const double f = x[j] - std::floor(x[j]);
            if (f > opt_.int_tol && f < 1 - opt_.int_tol) frac.push_back(static_cast<int>(j));
        }
        if (frac.empty()) {
            try_solution(x, "LP at a node");
            return;
        }
        refresh_pc_mean();
        // pseudocost scores
        auto score_of = [](double down, double up) { return std::max(down, 1e-6) * std::max(up, 1e-6); };
        std::vector<std::pair<double, int>> cand;
        for (int j : frac) {
            const double f = x[j] - std::floor(x[j]);
            cand.emplace_back(-score_of(pc_avg(0, j) * f, pc_avg(1, j) * (1 - f)), j);
        }
        std::sort(cand.begin(), cand.end());
        int best = cand[0].second;
        double best_score = -cand[0].first;
        double best_down = obj, best_up = obj; // proven child bounds (strong branching)
        bool down_inf = false, up_inf = false;
        // reliability: strong branch unreliable candidates (a few, best first)
        int tried = 0, no_gain = 0;
        double sb_best = -1;
#ifdef AXOS_ENABLE_CUDA
        if (gblp_ && y && !time_up()) {
            // up to 16 unreliable candidates (32 children) in one batched PDHG
            // run on the GPU; its bounds are valid, so they also prune
            std::vector<int> sel, col;
            std::vector<double> lo, hi, bd;
            for (const auto &c : cand) {
                if (sel.size() >= 16) break;
                if (std::min(pc_n_[0][c.second], pc_n_[1][c.second]) < opt_.reliability) sel.push_back(c.second);
            }
            for (int j : sel) {
                col.push_back(j); lo.push_back(-kInf); hi.push_back(std::floor(x[j]));
                col.push_back(j); lo.push_back(std::ceil(x[j])); hi.push_back(kInf);
            }
            if (!sel.empty()) {
                try {
                    bd = gblp_->child_bounds(x, *y, lb, ub, col, lo, hi, opt_.gpu_sb_iters, 20);
                    ++gpu_sb_calls_;
                } catch (const std::exception &e) {
                    if (opt_.verbose) std::printf("[milp] GPU strong branching failed: %s\n", e.what());
                    gblp_.reset();
                }
            }
            for (size_t k = 0; k < sel.size() && !bd.empty(); ++k) {
                const int j = sel[k];
                const double f = x[j] - std::floor(x[j]);
                const double od = bd[2 * k], ou = bd[2 * k + 1];
                const bool dinf = od > -kInf && node_bound(od) >= cutoff();
                const bool uinf = ou > -kInf && node_bound(ou) >= cutoff();
                if (dinf && uinf) return; // the node is pruned
                const double gd = od > -kInf ? std::max(0.0, od - obj) : 0.0;
                const double gu = ou > -kInf ? std::max(0.0, ou - obj) : 0.0;
                if (od > -kInf) {
                    pc_sum_[0][j] += gd / f;
                    pc_n_[0][j] += 1;
                }
                if (ou > -kInf) {
                    pc_sum_[1][j] += gu / (1 - f);
                    pc_n_[1][j] += 1;
                }
                if (dinf || uinf) {
                    best = j;
                    best_down = std::max(obj, od);
                    best_up = std::max(obj, ou);
                    down_inf = dinf;
                    up_inf = uinf;
                    sb_best = 1e300;
                    break;
                }
                const double sc = score_of(gd, gu);
                if (sc > sb_best) {
                    sb_best = sc;
                    best = j;
                    best_down = std::max(obj, od);
                    best_up = std::max(obj, ou);
                }
            }
            if (!bd.empty()) tried = 1 << 30; // no CPU strong branching on top
        }
#endif
        for (const auto &c : cand) {
            if (tried >= 10 || no_gain >= 4 || time_up()) break;
            const int j = c.second;
            if (std::min(pc_n_[0][j], pc_n_[1][j]) >= opt_.reliability) continue;
            ++tried;
            const double f = x[j] - std::floor(x[j]);
            double od, ou, dbd, ubd;
            bool dinf, uinf;
            strong_branch(j, x[j], lb, ub, basis, od, ou, dinf, uinf, dbd, ubd);
            if (dinf && uinf) return; // the node is infeasible
            if (!dinf) {
                pc_sum_[0][j] += std::max(0.0, od - obj) / f;
                pc_n_[0][j] += 1;
            }
            if (!uinf) {
                pc_sum_[1][j] += std::max(0.0, ou - obj) / (1 - f);
                pc_n_[1][j] += 1;
            }
            if (dinf || uinf) { // one direction only: branch on it right away
                best = j;
                best_down = dbd;
                best_up = ubd;
                down_inf = dinf;
                up_inf = uinf;
                sb_best = 1e300;
                break;
            }
            const double sc = score_of(od - obj, ou - obj);
            if (sc > sb_best) {
                sb_best = sc;
                best = j;
                best_down = dbd;
                best_up = ubd;
                no_gain = 0;
            } else {
                ++no_gain;
            }
        }
        if (sb_best < 0) { // no strong branching: pseudocost choice
            best_down = best_up = obj;
            (void)best_score;
        }
        const double v = x[best], f = v - std::floor(v);
        Node child[2];
        for (int d = 0; d < 2; ++d) {
            Node &c = child[d];
            c.var = node.var;
            c.lo = node.lo;
            c.hi = node.hi;
            c.var.push_back(best);
            c.lo.push_back(d == 0 ? -kInf : std::ceil(v));
            c.hi.push_back(d == 0 ? std::floor(v) : kInf);
            c.bound = node_bound(std::max(obj, d == 0 ? best_down : best_up));
            c.depth = node.depth + 1;
            c.branch_var = best;
            c.branch_dir = d;
            c.branch_frac = d == 0 ? f : 1 - f;
            c.parent_obj = obj;
            c.basis = node.basis;
        }
        const bool keep[2] = {!down_inf && child[0].bound < cutoff(), !up_inf && child[1].bound < cutoff()};
        // plunge into the child with the smaller estimated degradation
        const int pref = pc_avg(0, best) * f <= pc_avg(1, best) * (1 - f) ? 0 : 1;
        const double gb = global_bound();
        for (int k = 0; k < 2; ++k) {
            const int d = k == 0 ? pref : 1 - pref;
            if (!keep[d]) continue;
            const bool close = best_x_.empty() || child[d].bound <= gb + 0.5 * (best_obj_ - gb);
            if (k == 0 && !plunge_ && close && child[d].depth < 1000) plunge_.reset(new Node(std::move(child[d])));
            else open_.push(std::move(child[d]));
        }
    }

    // Both children of x_j with an iteration-capped dual simplex from the
    // node basis. od/ou score the children; db/ub_ are their proven bounds:
    // the objective when the child LP was solved to optimality, else -inf
    // (an early stop is dual feasible only for the shifted costs, so its
    // objective is an estimate, good for scores but not for pruning).
    void
    strong_branch(int j, double v, const std::vector<double> &lb, const std::vector<double> &ub,
        const SimplexBasis &basis, double &od, double &ou, bool &dinf, bool &uinf, double &dbound,
        double &ubound)
    {
        std::vector<double> l = lb, u = ub;
        set_time();
        u[j] = std::floor(v);
        LpSolution s = spx_.resolve(l, u, &basis, nullptr, opt_.strong_branch_iters);
        lp_iters_ += s.iterations;
        dinf = s.status == Status::Infeasible;
        od = (s.status == Status::Optimal || s.status == Status::IterationLimit) ? s.primal_objective : -kInf;
        dbound = s.status == Status::Optimal ? s.primal_objective : -kInf;
        if (s.status == Status::Optimal) try_integral(s.x);
        u[j] = ub[j];
        l[j] = std::ceil(v);
        s = spx_.resolve(l, u, &basis, nullptr, opt_.strong_branch_iters);
        lp_iters_ += s.iterations;
        uinf = s.status == Status::Infeasible;
        ou = (s.status == Status::Optimal || s.status == Status::IterationLimit) ? s.primal_objective : -kInf;
        ubound = s.status == Status::Optimal ? s.primal_objective : -kInf;
        if (s.status == Status::Optimal) try_integral(s.x);
        if (!dinf && node_bound(dbound) >= cutoff()) dinf = true; // pruned: as good as infeasible
        if (!uinf && node_bound(ubound) >= cutoff()) uinf = true;
    }

    void
    try_integral(const std::vector<double> &x)
    {
        for (size_t j = 0; j < x.size(); ++j)
            if (is_int(p_, j) && std::abs(x[j] - std::round(x[j])) > opt_.int_tol) return;
        try_solution(x, "strong branching");
    }
};

} // namespace milp
} // namespace Solver
} // namespace AXOS
