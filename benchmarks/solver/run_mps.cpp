// Solves MPS files with AXOS's LP solver (CPU and CUDA) and prints one line
// per solve:
//   RESULT mps <solver> <name> <ms> <iterations> <status> <objective> <pres> <dres> <gap>
// The objective is in the model's own sense (max models are un-negated).
//
//   ./build/run_mps [--method pdlp|ipm|simplex|auto] [--eps 1e-6] [--time 120] [--no-cuda]
//                   [--no-presolve] file.mps ...
// The solver column is <method>_cpu / <method>_cuda.

#ifdef AXOS_ENABLE_CUDA
#include "tensorCuda.h"
#endif
#include "solver/solver.h"
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <string>
#include <vector>

using namespace AXOS;
using namespace AXOS::Solver;

template <template <typename> class Store>
static void
run(const std::string &solver, const std::string &name, const LpProblem &p,
    const SolverOptions &o)
{
    auto t0 = std::chrono::steady_clock::now();
    LpSolution s;
    try {
        s = solve_lp<Store>(p, o);
    } catch (const std::exception &e) {
        printf("RESULT mps %s %s 0 0 error nan nan nan nan # %s\n", solver.c_str(),
            name.c_str(), e.what());
        return;
    }
    double ms = std::chrono::duration<double, std::milli>(
        std::chrono::steady_clock::now() - t0).count();
    const double sgn = p.maximize ? -1.0 : 1.0;
    double nb = 0, nc = 0;
    for (size_t i = 0; i < p.rows(); ++i) {
        if (std::isfinite(p.row_lb[i])) nb += p.row_lb[i] * p.row_lb[i];
        if (std::isfinite(p.row_ub[i])) nb += p.row_ub[i] * p.row_ub[i];
    }
    for (double v : p.c) nc += v * v;
    printf("RESULT mps %s %s %.1f %ld %s %.10e %.2e %.2e %.2e\n", solver.c_str(),
        name.c_str(), ms, s.iterations, to_string(s.status),
        sgn * s.primal_objective, s.primal_residual / (1 + std::sqrt(nb)),
        s.dual_residual / (1 + std::sqrt(nc)),
        s.gap / (1 + std::abs(s.primal_objective) + std::abs(s.dual_objective)));
    fflush(stdout);
}

int
main(int argc, char **argv)
{
    SolverOptions o;
    o.set_tolerance(1e-6);
    o.time_limit = 120;
    o.max_iterations = 5000000;
    bool cuda = true;
    std::string mname = "pdlp";
    std::vector<std::string> files;
    for (int i = 1; i < argc; ++i) {
        std::string a = argv[i];
        if (a == "--eps" && i + 1 < argc) o.set_tolerance(atof(argv[++i]));
        else if (a == "--time" && i + 1 < argc) o.time_limit = atof(argv[++i]);
        else if (a == "--method" && i + 1 < argc) {
            mname = argv[++i];
            o.method = mname == "ipm" ? LpMethod::Ipm
                       : mname == "simplex" ? LpMethod::Simplex
                       : mname == "auto" ? LpMethod::Auto : LpMethod::Pdlp;
        }
        else if (a == "--nd") o.ipm_ordering = 1;
        else if (a == "--crossover") o.crossover = true;
        else if (a == "--normal") o.ipm_normal = 1;
        else if (a == "--halpern") o.pdlp_halpern = true;
        else if (a == "--polish") o.pdlp_polish = true;
        else if (a == "--no-halpern") o.pdlp_halpern = false;
        else if (a == "--augmented") o.ipm_normal = -1;
        else if (a == "-v") o.verbose = true;
        else if (a == "--no-cuda") cuda = false;
        else if (a == "--no-presolve") o.presolve = false;
        else files.push_back(a);
    }
    for (const auto &f : files) {
        LpProblem p;
        try {
            p = read_mps_file(f);
        } catch (const std::exception &e) {
            printf("RESULT mps - %s 0 0 readerror nan nan nan nan # %s\n", f.c_str(), e.what());
            continue;
        }
        std::string name = f.substr(f.find_last_of('/') + 1);
        name = name.substr(0, name.rfind('.'));
        run<Cpu::HostStorage>(mname + "_cpu", name, p, o);
#ifdef AXOS_ENABLE_CUDA
        if (cuda) run<Cuda::CudaStorage>(mname + "_cuda", name, p, o);
#endif
    }
    return 0;
}
