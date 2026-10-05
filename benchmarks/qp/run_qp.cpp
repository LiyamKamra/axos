// SPDX-License-Identifier: BSD-3-Clause
//
// Runs the AXOS QP solvers on QPS files and prints one CSV line per solve:
//
//   run_qp <file.qps | file.axqp | directory> [options]
//     --device cpu|gpu|auto   (default gpu when available; auto: gpu for
//                             problems of at least --gpu-min-work = nnz(A) +
//                             nnz(Q) + n + m, 100000 by default)
//     --method hprqp|pdhcg|ipm|auto (default hprqp; ipm runs on the CPU; auto:
//                             ipm when cheap, else or on failure hprqp)
//     --tol 1e-6              max relative KKT residual (qp_model.h)
//     --time-limit 600        seconds per problem
//     --max-iter N
//     --check-every 64
//     --no-scaling
//     --profile               HPR-QP: time each step of an iteration and stop
//     --ipm-max-flops F       IPM: give up when a KKT factorization needs more (1e12)
//     --verbose 0|1|2
//     --out results.csv       also append the lines to this file
//     --only NAME[,NAME...]   solve only these problems of a directory
//     --export DIR            write each problem as DIR/<name>.axqp (binary,
//                             read by qpbench.py) instead of solving it
//     --save-sol DIR          write each solution to DIR/<name>.sol (int64 n,
//                             int64 m, x[n], y[m] as float64; for
//                             `qpbench.py score`)
//
// Built with -DAXOS_ENABLE_MPI (bash benchmarks/qp/build.sh mpi) and started
// with mpiexec -n K, HPR-QP runs distributed over the K ranks (qp_dist.h):
// rank r uses GPU (r mod GPUs) of its machine, rank 0 prints and writes the
// files, and the solver column reads axos-hpr-qp-mpiK.
//
// Times: `seconds` is the solve time after the file is read (preconditioning,
// upload, spectral estimates and iterations), `setup` the part before the
// first iteration. GPU kernels are compiled (NVRTC) once, before the first
// solve, and that time is excluded and reported separately.
#include "solver/io/qps.h"
#include "solver/qp/qp_dist.h"
#include "solver/qp/solve_qp.h"

#include <algorithm>
#include <chrono>
#include <cstdio>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <set>
#include <sstream>
#include <string>
#include <vector>

using namespace AXOS::Solver;
namespace fs = std::filesystem;

static std::string
status_name(Status s)
{
    std::string t = to_string(s);
    std::replace(t.begin(), t.end(), ' ', '_');
    return t;
}

// .axqp: "AXQP", int32 version 1, int64 n m nnzA nnzQ, double offset, int32
// maximize, then A (int32 row_ptr[m+1], col[nnzA], double val[nnzA]), Q (same
// with n rows), and double c, col_lb, col_ub (n), row_lb, row_ub (m).
// Infinite bounds are IEEE infinities. Little-endian.
static void
export_axqp(const QpProblem &p, const std::string &path)
{
    std::ofstream f(path, std::ios::binary);
    if (!f) throw std::runtime_error("cannot write " + path);
    auto put = [&](const void *d, size_t bytes) {
        f.write(static_cast<const char *>(d), static_cast<std::streamsize>(bytes));
    };
    const int32_t version = 1, maximize = p.lp.maximize ? 1 : 0;
    const int64_t dims[4] = {int64_t(p.cols()), int64_t(p.rows()), int64_t(p.lp.A.nnz()),
        int64_t(p.Q.nnz())};
    put("AXQP", 4);
    put(&version, 4);
    put(dims, sizeof dims);
    put(&p.lp.offset, 8);
    put(&maximize, 4);
    auto csr = [&](const HostMatrix &M, size_t rows) {
        if (M.rows() == rows) {
            put(M.row_ptr(), (rows + 1) * 4);
        } else { // empty Q
            std::vector<int32_t> z(rows + 1, 0);
            put(z.data(), z.size() * 4);
        }
        put(M.col_ind(), M.nnz() * 4);
        put(M.values(), M.nnz() * 8);
    };
    csr(p.lp.A, p.rows());
    csr(p.Q, p.cols());
    put(p.lp.c.data(), p.cols() * 8);
    put(p.lp.col_lb.data(), p.cols() * 8);
    put(p.lp.col_ub.data(), p.cols() * 8);
    put(p.lp.row_lb.data(), p.rows() * 8);
    put(p.lp.row_ub.data(), p.rows() * 8);
}

// Reads an .axqp file (the format above), e.g. the generated large problems.
static QpProblem
import_axqp(const std::string &path)
{
    std::ifstream f(path, std::ios::binary);
    if (!f) throw std::runtime_error("cannot read " + path);
    auto get = [&](void *d, size_t bytes) {
        f.read(static_cast<char *>(d), static_cast<std::streamsize>(bytes));
        if (!f) throw std::runtime_error(path + ": truncated .axqp file");
    };
    char magic[4];
    int32_t version = 0, maximize = 0;
    int64_t dims[4];
    QpProblem p;
    get(magic, 4);
    if (std::string(magic, 4) != "AXQP") throw std::runtime_error(path + ": not an .axqp file");
    get(&version, 4);
    get(dims, sizeof dims);
    get(&p.lp.offset, 8);
    get(&maximize, 4);
    p.lp.maximize = maximize != 0;
    const size_t n = size_t(dims[0]), m = size_t(dims[1]);
    auto csr = [&](size_t rows, size_t nnz) {
        std::vector<int32_t> rp(rows + 1), ci(nnz);
        std::vector<double> v(nnz);
        get(rp.data(), rp.size() * 4);
        get(ci.data(), nnz * 4);
        get(v.data(), nnz * 8);
        return HostMatrix(rows, n, rp, ci, v);
    };
    p.lp.A = csr(m, size_t(dims[2]));
    p.Q = csr(n, size_t(dims[3]));
    auto vec = [&](size_t k) {
        std::vector<double> v(k);
        get(v.data(), k * 8);
        return v;
    };
    p.lp.c = vec(n);
    p.lp.col_lb = vec(n);
    p.lp.col_ub = vec(n);
    p.lp.row_lb = vec(m);
    p.lp.row_ub = vec(m);
    p.lp.name = fs::path(path).stem().string();
    return p;
}

static QpProblem
read_problem(const fs::path &f)
{
    if (f.extension() == ".axqp") return import_axqp(f.string());
    return read_qps_file(f.string());
}

static int run(int argc, char **argv, const QpComm &comm);

int
main(int argc, char **argv)
{
#if defined(AXOS_ENABLE_MPI)
    MPI_Init(&argc, &argv);
    int rc = 0;
    {
        const QpComm comm(MPI_COMM_WORLD);
        try {
            rc = run(argc, argv, comm);
        } catch (const std::exception &e) {
            std::fprintf(stderr, "rank %d: %s\n", comm.rank(), e.what());
            MPI_Abort(MPI_COMM_WORLD, 5); // the other ranks would wait forever
        }
    }
    MPI_Finalize();
    return rc;
#else
    return run(argc, argv, QpComm());
#endif
}

static int
run(int argc, char **argv, const QpComm &comm)
{
    const bool root = comm.rank() == 0; // prints and writes files
    if (argc < 2) {
        std::fprintf(stderr, "usage: %s <file.qps|dir> [--device cpu|gpu] [--method hprqp|pdhcg] "
                             "[--tol e] [--time-limit s] [--max-iter n] [--check-every k] "
                             "[--no-scaling] [--verbose v] [--out file.csv] [--only a,b] "
                             "[--export dir] [--save-sol dir]\n",
            argv[0]);
        return 2;
    }

    QpOptions opt;
    opt.use_gpu = qp_gpu_available();
    opt.time_limit = 600;
    if (comm.distributed()) opt.comm = &comm;
    std::string out_path, export_dir, sol_dir;
    std::set<std::string> only;
    for (int i = 2; i < argc; ++i) {
        const std::string a = argv[i];
        auto next = [&]() -> std::string {
            if (i + 1 >= argc) throw std::runtime_error("missing value after " + a);
            return argv[++i];
        };
        if (a == "--device") {
            const std::string d = next();
            opt.auto_device = d == "auto";
            opt.use_gpu = d == "gpu" || (opt.auto_device && qp_gpu_available());
        } else if (a == "--gpu-min-work") opt.gpu_min_work = std::stoul(next());
        else if (a == "--method") {
            const std::string m = next();
            opt.method = m == "pdhcg" ? QpMethod::Pdhcg
                         : m == "ipm"  ? QpMethod::Ipm
                         : m == "auto" ? QpMethod::Auto
                                       : QpMethod::HprQp;
        } else if (a == "--tol") opt.tol = std::stod(next());
        else if (a == "--time-limit") opt.time_limit = std::stod(next());
        else if (a == "--max-iter") opt.max_iterations = std::stol(next());
        else if (a == "--check-every") opt.check_every = std::stoi(next());
        else if (a == "--no-scaling") opt.scaling = false;
        else if (a == "--profile") opt.profile = true;
        else if (a == "--hpr-free") opt.hpr_free_variant = std::stoi(next());
        else if (a == "--ipm-max-flops") opt.ipm_max_flops = std::stod(next());
        else if (a == "--verbose") opt.verbose = std::stoi(next());
        else if (a == "--out") out_path = next();
        else if (a == "--export") export_dir = next();
        else if (a == "--save-sol") sol_dir = next();
        else if (a == "--only") {
            std::stringstream ss(next());
            std::string t;
            while (std::getline(ss, t, ',')) only.insert(t);
        } else {
            std::fprintf(stderr, "unknown option %s\n", a.c_str());
            return 2;
        }
    }

    std::vector<fs::path> files;
    const fs::path target(argv[1]);
    if (fs::is_directory(target)) {
        for (auto &e : fs::directory_iterator(target)) {
            const std::string ext = e.path().extension().string();
            if (ext == ".SIF" || ext == ".sif" || ext == ".QPS" || ext == ".qps" ||
                ext == ".mps" || ext == ".MPS" || ext == ".axqp")
                if (only.empty() || only.count(e.path().stem().string()))
                    files.push_back(e.path());
        }
        std::sort(files.begin(), files.end());
    } else {
        files.push_back(target);
    }

    if (!export_dir.empty()) {
        fs::create_directories(export_dir);
        for (const auto &f : files) {
            const QpProblem p = read_problem(f);
            export_axqp(p, (fs::path(export_dir) / (f.stem().string() + ".axqp")).string());
            std::printf("exported %s (n %zu, m %zu)\n", f.stem().string().c_str(), p.cols(), p.rows());
        }
        return 0;
    }

#if defined(AXOS_ENABLE_CUDA)
    if (opt.use_gpu) {
        int count = 0;
        if (comm.distributed() && cudaGetDeviceCount(&count) == cudaSuccess && count > 0)
            cudaSetDevice(comm.local_rank() % count); // before any other CUDA call
        const auto t0 = std::chrono::steady_clock::now();
        qp::CudaBackend warm; // compiles the kernels (NVRTC) once
        if (root || opt.verbose)
            std::fprintf(stderr, "# rank %d of %d: %s, kernels compiled in %.2f s\n", comm.rank(),
                comm.size(), warm.name().c_str(),
                std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count());
    }
#endif
    std::string method_s = opt.method == QpMethod::Pdhcg ? "pdhcg"
                           : opt.method == QpMethod::Ipm  ? "ipm"
                           : opt.method == QpMethod::Auto ? "auto"
                                                          : "hpr-qp";
    if (comm.distributed()) method_s = "hpr-qp-mpi" + std::to_string(comm.size());
    const char *method = method_s.c_str();
    const std::string header =
        "problem,n,m,nnzA,nnzQ,solver,device,tol,status,iterations,seconds,setup_seconds,"
        "objective,rel_primal,rel_dual,rel_gap,read_seconds";
    if (root) std::printf("%s\n", header.c_str());
    std::ofstream out;
    if (!out_path.empty() && root) {
        const bool fresh = !fs::exists(out_path);
        out.open(out_path, std::ios::app);
        if (fresh) out << header << "\n";
    }
    for (const auto &f : files) {
        char line[1024];
        const std::string name = f.stem().string();
        try {
            const auto t0 = std::chrono::steady_clock::now();
            QpProblem p = read_problem(f);
            const double read_s =
                std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count();
            QpSolution s = solve_qp(p, opt);
            if (!sol_dir.empty() && root) {
                fs::create_directories(sol_dir);
                std::ofstream f((fs::path(sol_dir) / (name + ".sol")).string(), std::ios::binary);
                const int64_t dims[2] = {int64_t(s.x.size()), int64_t(s.y.size())};
                f.write(reinterpret_cast<const char *>(dims), sizeof dims);
                f.write(reinterpret_cast<const char *>(s.x.data()),
                    static_cast<std::streamsize>(s.x.size() * sizeof(double)));
                f.write(reinterpret_cast<const char *>(s.y.data()),
                    static_cast<std::streamsize>(s.y.size() * sizeof(double)));
            }
            std::snprintf(line, sizeof line,
                "%s,%zu,%zu,%zu,%zu,axos-%s,%s,%.0e,%s,%ld,%.6f,%.6f,%.12e,%.3e,%.3e,%.3e,%.3f",
                name.c_str(), p.cols(), p.rows(), p.lp.A.nnz(), p.Q.nnz(), method,
                s.device.rfind("gpu", 0) == 0 ? "gpu" : "cpu", opt.tol, status_name(s.status).c_str(),
                s.iterations, s.seconds, s.setup_seconds,
                p.lp.maximize ? -s.primal_objective : s.primal_objective, s.rel_primal,
                s.rel_dual, s.rel_gap, read_s);
        } catch (const std::exception &e) {
            if (comm.distributed()) throw; // the other ranks may be waiting: MPI_Abort
            std::string msg = e.what();
            std::replace(msg.begin(), msg.end(), ',', ';');
            std::replace(msg.begin(), msg.end(), '\n', ' ');
            std::snprintf(line, sizeof line, "%s,,,,,axos-%s,%s,%.0e,error,,,,,,,,%s",
                name.c_str(), method, opt.use_gpu ? "gpu" : "cpu", opt.tol,
                msg.substr(0, 400).c_str());
        }
        if (!root) continue;
        std::printf("%s\n", line);
        std::fflush(stdout);
        if (out) {
            out << line << "\n";
            out.flush();
        }
    }
    return 0;
}
