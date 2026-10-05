// SPDX-License-Identifier: BSD-3-Clause
//
// Checks and times the GPU components of the MILP solver against their CPU
// counterparts on MPS files:
//
//   test_gpu <file.mps | dir> [--only a,b] [--fj-seconds 5]
//
//   propagation  CPU queue propagator vs GPU rounds: same fixpoint?
//   probing      CPU vs GPU double probing on the binaries: time, fixings,
//                tightened bounds
//   fj           CPU vs GPU feasibility jump: time to a feasible point
//   batch LP     GPU Lagrangian bounds of 2K strong-branching children vs
//                their exact LP values (dual simplex)
#include "solver/io/mps.h"
#include "solver/lp/simplex.h"
#include "solver/milp/heuristics.h"
#include "solver/milp/milp_cuda.h"
#include "solver/milp/probing.h"
#include "solver/milp/propagate.h"
#include "solver/presolve/presolve.h"

#include <chrono>
#include <cstdio>
#include <filesystem>
#include <set>
#include <sstream>

using namespace AXOS::Solver;
using namespace AXOS::Solver::milp;
namespace fs = std::filesystem;

static double
now()
{
    return std::chrono::duration<double>(std::chrono::steady_clock::now().time_since_epoch()).count();
}

static LpProblem
reduced(const LpProblem &orig)
{
    Presolve pre(orig);
    LpProblem w = pre.reduced();
    w.is_integer.assign(w.cols(), 0);
    if (!orig.is_integer.empty())
        for (size_t k = 0; k < w.cols(); ++k) w.is_integer[k] = orig.is_integer[pre.col_map()[k]];
    for (size_t j = 0; j < w.cols(); ++j)
        if (w.is_integer[j]) {
            w.col_lb[j] = std::ceil(w.col_lb[j] - 1e-9);
            w.col_ub[j] = std::floor(w.col_ub[j] + 1e-9);
        }
    return w;
}

int
main(int argc, char **argv)
{
    if (argc < 2) {
        std::fprintf(stderr, "usage: %s <file.mps|dir> [--only a,b] [--fj-seconds s]\n", argv[0]);
        return 2;
    }
    std::set<std::string> only;
    double fj_s = 5;
    bool probe_only = false;
    for (int i = 2; i < argc; ++i) {
        const std::string a = argv[i];
        if (a == "--only" && i + 1 < argc) {
            std::stringstream ss(argv[++i]);
            std::string t;
            while (std::getline(ss, t, ',')) only.insert(t);
        } else if (a == "--probe-only") {
            probe_only = true;
        } else if (a == "--fj-seconds" && i + 1 < argc) {
            fj_s = std::stod(argv[++i]);
        }
    }
    std::vector<fs::path> files;
    const fs::path target(argv[1]);
    if (fs::is_directory(target)) {
        for (auto &e : fs::directory_iterator(target))
            if (e.path().extension() == ".mps" && (only.empty() || only.count(e.path().stem().string())))
                files.push_back(e.path());
        std::sort(files.begin(), files.end());
    } else {
        files.push_back(target);
    }
    {
        const double t0 = now();
        gpu::Module::get();
        std::printf("GPU %s, kernels compiled in %.2f s\n", gpu::Module::get().device().c_str(), now() - t0);
    }
    for (const auto &f : files) {
        const std::string name = f.stem().string();
        LpProblem p;
        try {
            p = reduced(read_mps_file(f.string()));
        } catch (const std::exception &e) {
            std::printf("%s: %s\n", name.c_str(), e.what());
            continue;
        }
        const size_t n = p.cols(), m = p.rows();
        if (n == 0 || m == 0) continue;
        size_t nbin = 0;
        for (size_t j = 0; j < n; ++j) nbin += (is_int(p, j) && p.col_lb[j] == 0 && p.col_ub[j] == 1) ? 1 : 0;
        std::printf("== %s  n %zu  m %zu  nnz %zu  binaries %zu\n", name.c_str(), n, m, p.A.nnz(), nbin);
        gpu::DeviceProblem dp(p);
        gpu::GpuPropagator gp(dp);
        Propagator cp(p);
        // ---- propagation
        std::vector<double> l1 = p.col_lb, u1 = p.col_ub, l2 = l1, u2 = u1;
        double t = now();
        const bool f1 = cp.propagate(l1, u1, nullptr, nullptr, 1e9);
        const double tc = now() - t;
        t = now();
        const bool f2 = gp.propagate(l2, u2);
        const double tg = now() - t;
        int diff = 0;
        double dmax = 0;
        for (size_t j = 0; j < n; ++j) {
            const double a = std::abs(l1[j] - l2[j]), b = std::abs(u1[j] - u2[j]);
            const double d = std::max(std::isfinite(a) ? a : (l1[j] == l2[j] ? 0 : 1e300),
                std::isfinite(b) ? b : (u1[j] == u2[j] ? 0 : 1e300));
            if (d > 1e-6) ++diff;
            dmax = std::max(dmax, d);
        }
        std::printf("  propagation: cpu %.4f s (feasible %d)  gpu %.4f s (feasible %d)  columns differing %d (max %.2e)\n",
            tc, f1, tg, f2, diff, dmax);
        if (!f1) continue;
        // ---- probing
        {
            std::vector<double> la = l1, ua = u1, lb = l1, ub = u1;
            const ProbingStats sc = probe_binaries_cpu(cp, p, la, ua, 30.0);
            const gpu::ProbeResult sg = gpu::probe_binaries(dp, p, lb, ub, 30.0);
            std::printf("  probing: cpu %.3f s  %d probes  fixed %d  tightened %d%s | gpu %.3f s  %d probes  fixed %d  tightened %d%s\n",
                sc.seconds, sc.probes, sc.fixed, sc.tightened, sc.infeasible ? " INFEASIBLE" : "", sg.seconds, sg.probes,
                sg.fixed, sg.tightened, sg.infeasible ? " INFEASIBLE" : "");
        }
        if (probe_only) continue;
        // ---- feasibility jump
        {
            FeasibilityJump fj(p);
            std::vector<double> x;
            t = now();
            const bool ok1 = fj.run(l1, u1, nullptr, fj_s, x, 1);
            const double t1 = now() - t;
            std::vector<uint8_t> isint(n);
            for (size_t j = 0; j < n; ++j) isint[j] = is_int(p, j) ? 1 : 0;
            gpu::GpuFeasibilityJump gfj(dp, 64);
            std::vector<double> xg;
            long long moves = 0;
            t = now();
            const bool ok2 = gfj.run(l1, u1, {}, isint, fj_s, 1, xg, &moves);
            const double t2 = now() - t;
            bool feas2 = false;
            if (ok2) feas2 = check_milp(p, xg).ok(1e-6, 1e-6);
            std::printf("  fj: cpu %s in %.3f s | gpu (64 walkers) %s in %.3f s, %lld moves, check %d\n",
                ok1 ? "found" : "none", t1, ok2 ? "found" : "none", t2, moves, feas2);
        }
        // ---- batch LP bounds vs exact children
        {
            DualSimplex spx;
            SolverOptions so;
            so.time_limit = 30;
            SimplexBasis basis;
            LpProblem q = p;
            q.col_lb = l1;
            q.col_ub = u1;
            LpSolution s = spx.solve(q, so, nullptr, &basis);
            if (s.status != Status::Optimal) {
                std::printf("  batch LP: root LP %s\n", to_string(s.status));
                continue;
            }
            std::vector<int> frac;
            for (size_t j = 0; j < n && frac.size() < 16; ++j) {
                if (!is_int(p, j)) continue;
                const double fr = s.x[j] - std::floor(s.x[j]);
                if (fr > 1e-6 && fr < 1 - 1e-6) frac.push_back(static_cast<int>(j));
            }
            if (frac.empty()) {
                std::printf("  batch LP: root LP integral\n");
                continue;
            }
            std::vector<int> col;
            std::vector<double> lo, hi, exact;
            for (int j : frac) {
                col.push_back(j); lo.push_back(-kInf); hi.push_back(std::floor(s.x[j]));
                col.push_back(j); lo.push_back(std::ceil(s.x[j])); hi.push_back(kInf);
            }
            t = now();
            for (size_t b = 0; b < col.size(); ++b) {
                std::vector<double> l = l1, u = u1;
                l[col[b]] = std::max(l[col[b]], lo[b]);
                u[col[b]] = std::min(u[col[b]], hi[b]);
                LpSolution c = spx.resolve(l, u, &basis, nullptr);
                exact.push_back(c.status == Status::Optimal ? c.primal_objective
                                                           : (c.status == Status::Infeasible ? kInf : -kInf));
            }
            const double te = now() - t;
            gpu::GpuBatchLp blp(q);
            for (int iters : {0, 100, 400, 1600}) {
                t = now();
                const std::vector<double> gb = blp.child_bounds(s.x, s.y, l1, u1, col, lo, hi, iters, 10);
                const double tb = now() - t;
                double frac_sum = 0;
                int cnt = 0, invalid = 0;
                for (size_t b = 0; b < col.size(); ++b) {
                    if (!std::isfinite(exact[b])) continue;
                    if (gb[b] > exact[b] + 1e-6 * (1 + std::abs(exact[b]))) ++invalid;
                    const double gain = exact[b] - s.primal_objective;
                    if (gain > 1e-9) {
                        frac_sum += std::max(0.0, std::min(1.0, (gb[b] - s.primal_objective) / gain));
                        ++cnt;
                    }
                }
                std::printf("  batch LP: %zu children, %4d iters: gpu %.4f s, mean share of the exact gain %.2f "
                            "(%d with gain), bound above exact %d | exact dual simplex %.4f s (root %.6g)\n",
                    col.size(), iters, tb, cnt ? frac_sum / cnt : 0.0, cnt, invalid, te, s.primal_objective);
            }
        }
        std::fflush(stdout);
    }
    return 0;
}
