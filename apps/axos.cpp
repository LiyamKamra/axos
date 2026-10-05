// SPDX-License-Identifier: BSD-3-Clause
//
// axos: command-line front end of the AXOS solvers (LP, QP, MILP).
//
//   axos <model.mps|model.qps> [options]
//
//   --type auto|lp|qp|milp     problem type (default: from the model; lp on
//                              a MILP solves its LP relaxation)
//   --method NAME              LP: auto simplex ipm pdlp hpr
//                              QP: auto hprqp pdhcg ipm     MILP: auto
//   --device auto|cpu|gpu      (default auto: the GPU where it pays off)
//   --time-limit S             seconds (default 3600)
//   --tol T                    LP/QP relative KKT tolerance (default 1e-6)
//   --gap G                    MILP relative gap (default 1e-4)
//   --no-presolve --no-cuts --no-heuristics   (MILP)
//   --solution FILE            write the solution ("name value" per nonzero)
//   --json FILE                write the result as JSON ("-": standard output)
//   --quiet                    no summary on standard output
//   --verbose N                solver log (0-3)
//   --info                     print the model statistics and exit
//   --version
//
// Exit status: 0 optimal, 1 solution without proof of optimality, 2
// infeasible, 3 unbounded, 4 no solution within the limits, 5 error, 64
// usage.
//
// MPI build (-DAXOS_ENABLE_MPI, bash apps/build.sh mpi axos_mpi):
//   mpiexec -n K axos_mpi model.mps --device gpu
// solves an LP or QP by HPR with the constraint rows split over the K
// ranks, rank r on GPU (r mod GPUs) of its machine (qp_dist.h); rank 0
// prints and writes the files. A MILP needs a single process.
#include "solver/api.h"

#include <cstdio>
#include <cstring>
#include <fstream>
#include <iostream>
#include <string>

using namespace AXOS::Solver;

static void
usage()
{
    std::fprintf(stderr,
        "usage: axos <model.mps|model.qps> [options]\n"
        "  --type auto|lp|qp|milp   --method NAME   --device auto|cpu|gpu\n"
        "  --time-limit S   --tol T (LP/QP)   --gap G (MILP)\n"
        "  --no-presolve --no-cuts --no-heuristics (MILP)\n"
        "  --solution FILE   --json FILE|-   --quiet   --verbose N   --info   --version\n"
        "methods: LP auto simplex ipm pdlp hpr; QP auto hprqp pdhcg ipm; MILP auto\n");
}

static int
exit_code(const Result &r)
{
    switch (r.status) {
    case Status::Optimal: return 0;
    case Status::Infeasible: return 2;
    case Status::Unbounded: return 3;
    case Status::NumericalError: return r.has_solution() ? 1 : 5;
    default: return r.has_solution() ? 1 : 4;
    }
}

static int run(int argc, char **argv, const QpComm &comm);

int
main(int argc, char **argv)
{
#if defined(AXOS_ENABLE_MPI)
    MPI_Init(&argc, &argv);
    int rc = 5;
    {
        const QpComm comm(MPI_COMM_WORLD);
        try {
            rc = run(argc, argv, comm);
        } catch (const std::exception &e) {
            std::fprintf(stderr, "axos: rank %d: %s\n", comm.rank(), e.what());
            if (comm.distributed()) MPI_Abort(MPI_COMM_WORLD, 5); // the others may be waiting
        }
    }
    MPI_Finalize();
    return rc;
#else
    try {
        return run(argc, argv, QpComm());
    } catch (const std::exception &e) {
        std::fprintf(stderr, "axos: %s\n", e.what());
        return 5;
    }
#endif
}

static int
run(int argc, char **argv, const QpComm &comm)
{
    const bool root = comm.rank() == 0; // prints and writes the files
    if (argc >= 2 && std::strcmp(argv[1], "--version") == 0) {
        if (root)
            std::printf("AXOS %s (LP: dual simplex, IPM, PDLP, HPR; QP: HPR-QP, PDHCG, IPM; MILP: branch and cut)%s%s\n",
                kAxosVersion,
#if defined(AXOS_ENABLE_CUDA)
                ", CUDA build",
#else
                ", CPU build",
#endif
#if defined(AXOS_ENABLE_MPI)
                ", MPI (LP / QP by HPR over ranks)"
#else
                ""
#endif
            );
        return 0;
    }
    if (argc < 2 || argv[1][0] == '-') {
        if (root) usage();
        return 64;
    }
    const std::string path = argv[1];
    Options o;
    std::string sol_path, json_path;
    bool quiet = false, info = false;
    try {
        for (int i = 2; i < argc; ++i) {
            const std::string a = argv[i];
            auto next = [&]() -> std::string {
                if (i + 1 >= argc) throw std::invalid_argument("missing value after " + a);
                return argv[++i];
            };
            if (a == "--type") {
                const std::string t = next();
                o.type = t == "lp" ? ProblemType::LP : t == "qp" ? ProblemType::QP
                       : t == "milp" ? ProblemType::MILP : t == "auto" ? ProblemType::Auto
                       : throw std::invalid_argument("--type: auto, lp, qp or milp");
            } else if (a == "--method") o.method = next();
            else if (a == "--device") {
                const std::string d = next();
                o.device = d == "cpu" ? Device::Cpu : d == "gpu" ? Device::Gpu : d == "auto" ? Device::Auto
                         : throw std::invalid_argument("--device: auto, cpu or gpu");
            } else if (a == "--time-limit") o.time_limit = std::stod(next());
            else if (a == "--tol") o.tol = std::stod(next());
            else if (a == "--gap") o.gap = std::stod(next());
            else if (a == "--no-presolve") o.presolve = false;
            else if (a == "--no-cuts") o.cuts = false;
            else if (a == "--no-heuristics") o.heuristics = false;
            else if (a == "--solution") sol_path = next();
            else if (a == "--json") json_path = next();
            else if (a == "--quiet") quiet = true;
            else if (a == "--verbose") o.verbose = std::stoi(next());
            else if (a == "--info") info = true;
            else throw std::invalid_argument("unknown option " + a);
        }
    } catch (const std::exception &e) {
        if (root) {
            std::fprintf(stderr, "axos: %s\n", e.what());
            usage();
        }
        return 64;
    }
    if (!root) quiet = true;
    if (comm.distributed()) {
        o.comm = &comm;
#if defined(AXOS_ENABLE_CUDA)
        int count = 0; // before any other CUDA call of this process
        if (o.device != Device::Cpu && cudaGetDeviceCount(&count) == cudaSuccess && count > 0)
            cudaSetDevice(comm.local_rank() % count);
#endif
    }

    Model m;
    const auto t0 = std::chrono::steady_clock::now();
    try {
        m = read_model(path);
    } catch (const std::exception &e) {
        std::fprintf(stderr, "axos: %s\n", e.what());
        return 5;
    }
    const double read_s = std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count();
    if (!quiet || (info && root))
        std::printf("AXOS %s  model %s (%s%s): %zu columns (%zu integer), %zu rows, %zu nonzeros%s, read in %.2f s\n",
            kAxosVersion, m.name().c_str(), to_string(m.type()), m.maximize() ? ", maximize" : "", m.cols(),
            m.integers(), m.rows(), m.nnz(),
            m.qp.has_quadratic() ? (", " + std::to_string(m.qp.Q.nnz()) + " in Q").c_str() : "", read_s);
    if (info) return 0;

    Result r;
    try {
        r = solve(m, o);
    } catch (const std::invalid_argument &e) { // the same on every rank
        if (root) std::fprintf(stderr, "axos: %s\n", e.what());
        return 64;
    }
    if (!quiet) {
        std::printf("method       %s on %s (%s)\n", r.method.c_str(), r.device.c_str(), to_string(r.type));
        std::printf("status       %s\n", to_string(r.status));
        if (r.has_solution()) {
            std::printf("objective    %.12g\n", r.objective);
            if (std::isfinite(r.bound)) std::printf("bound        %.12g\n", r.bound);
            if (r.type == ProblemType::MILP) std::printf("gap          %.4g%%\n", 100 * r.gap());
            std::printf("violation    %.2e (largest relative bound / row%s violation)\n", r.max_violation,
                r.type == ProblemType::MILP ? " / integrality" : "");
        } else if (std::isfinite(r.bound)) {
            std::printf("bound        %.12g\n", r.bound);
        }
        std::printf("time         %.3f s, %ld iterations%s\n", r.seconds, r.iterations,
            r.type == ProblemType::MILP ? (", " + std::to_string(r.nodes) + " nodes").c_str() : "");
        if (comm.distributed())
            std::printf("mpi          %d ranks, %.3f s in %ld MPI calls on rank 0\n", comm.size(),
                comm.seconds(), comm.calls());
        if (!r.message.empty()) std::printf("note         %s\n", r.message.c_str());
    }
    if (!root) return exit_code(r);
    if (!sol_path.empty()) {
        std::ofstream f(sol_path);
        if (!f) {
            std::fprintf(stderr, "axos: cannot write %s\n", sol_path.c_str());
            return 5;
        }
        write_solution(f, m, r);
    }
    if (!json_path.empty()) {
        const std::string js = result_json(m, r, true);
        if (json_path == "-") {
            std::printf("%s\n", js.c_str());
        } else {
            std::ofstream f(json_path);
            if (!f) {
                std::fprintf(stderr, "axos: cannot write %s\n", json_path.c_str());
                return 5;
            }
            f << js << "\n";
        }
    }
    return exit_code(r);
}
