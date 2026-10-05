// LP benchmark: AXOS PDLP (CPU, CUDA) on generated structured LPs. The
// instances are exported so bench_highs.py solves exactly the same problems.
//
//   make benchmark_lp
//
// Output lines: RESULT lp <solver> <instance> <ms> <iterations> <status> <objective>

#ifdef AXOS_ENABLE_CUDA
#include "tensorCuda.h"
#endif
#include "solver/solver.h"
#include <chrono>
#include <cstdio>
#include <fstream>
#include <random>
#include <string>
#include <sys/stat.h>

using namespace AXOS;
using namespace AXOS::Solver;

struct Inst {
    std::string name;
    LpProblem p;
};

// Transportation: S sources, D sinks, balanced integer supplies/demands.
static LpProblem
gen_transport(size_t S, size_t D, unsigned seed)
{
    std::mt19937_64 g(seed);
    std::uniform_int_distribution<int> cost(1, 100), amt(10, 60);
    std::vector<double> sup(S), dem(D);
    double tot = 0;
    for (auto &s : sup) { s = amt(g); tot += s; }
    double dt = 0;
    for (auto &d : dem) { d = amt(g); dt += d; }
    for (auto &d : dem) d = std::round(d * tot / dt);
    long diff = static_cast<long>(tot);
    for (double d : dem) diff -= static_cast<long>(d);
    for (size_t j = 0; diff != 0; j = (j + 1) % D) { // spread the rounding error
        const long step = diff > 0 ? 1 : -1;
        if (dem[j] + step >= 1) { dem[j] += step; diff -= step; }
    }
    LpProblem p;
    const size_t n = S * D, m = S + D;
    Sparse::CooBuilder<double> b(m, n);
    p.c.resize(n);
    for (size_t i = 0; i < S; ++i)
        for (size_t j = 0; j < D; ++j) {
            size_t v = i * D + j;
            b.add(i, v, 1.0);
            b.add(S + j, v, 1.0);
            p.c[v] = cost(g);
        }
    p.A = b.build();
    p.col_lb.assign(n, 0);
    p.col_ub.assign(n, kInf);
    p.row_lb.resize(m);
    for (size_t i = 0; i < S; ++i) p.row_lb[i] = sup[i];
    for (size_t j = 0; j < D; ++j) p.row_lb[S + j] = dem[j];
    p.row_ub = p.row_lb;
    return p;
}

// Min-cost flow on a random graph with a known feasible flow.
static LpProblem
gen_mcf(size_t V, size_t E, unsigned seed)
{
    std::mt19937_64 g(seed);
    std::uniform_int_distribution<size_t> node(0, V - 1);
    std::uniform_int_distribution<int> cost(1, 50), cap(5, 40);
    LpProblem p;
    Sparse::CooBuilder<double> b(V, E);
    std::vector<double> bal(V, 0.0);
    p.c.resize(E);
    p.col_lb.assign(E, 0);
    p.col_ub.resize(E);
    for (size_t e = 0; e < E; ++e) {
        size_t u = node(g), v = node(g);
        while (v == u) v = node(g);
        int cp = cap(g);
        double f = std::floor(cp * std::uniform_real_distribution<double>(0, 1)(g));
        b.add(u, e, 1.0);   // out of u
        b.add(v, e, -1.0);  // into v
        bal[u] += f;
        bal[v] -= f;
        p.c[e] = cost(g);
        p.col_ub[e] = cp;
    }
    p.A = b.build();
    p.row_lb = bal;
    p.row_ub = bal;
    return p;
}

// Packing LP: max c^T x s.t. A x <= b, x >= 0 (stored as a minimization).
static LpProblem
gen_packing(size_t m, size_t n, int per_col, unsigned seed)
{
    std::mt19937_64 g(seed);
    std::uniform_int_distribution<size_t> row(0, m - 1);
    std::uniform_real_distribution<double> u(0.5, 2.0);
    LpProblem p;
    Sparse::CooBuilder<double> b(m, n);
    for (size_t j = 0; j < n; ++j)
        for (int k = 0; k < per_col; ++k)
            b.add(row(g), j, u(g));
    p.A = b.build();
    p.c.resize(n);
    for (auto &v : p.c) v = -u(g);
    p.col_lb.assign(n, 0);
    p.col_ub.assign(n, kInf);
    p.row_lb.assign(m, -kInf);
    p.row_ub.resize(m);
    for (auto &v : p.row_ub) v = 10 * per_col * u(g) * n / m / per_col + 1;
    return p;
}

static void
export_lp(const std::string &dir, const std::string &name, const LpProblem &p)
{
    std::ofstream f(dir + "/" + name + ".lp", std::ios::binary);
    int64_t h[3] = {(int64_t)p.rows(), (int64_t)p.cols(), (int64_t)p.A.nnz()};
    f.write((const char *)h, sizeof(h));
    f.write((const char *)p.A.row_ptr(), (p.rows() + 1) * 4);
    f.write((const char *)p.A.col_ind(), p.A.nnz() * 4);
    f.write((const char *)p.A.values(), p.A.nnz() * 8);
    for (const auto *v : {&p.c, &p.col_lb, &p.col_ub})
        f.write((const char *)v->data(), p.cols() * 8);
    for (const auto *v : {&p.row_lb, &p.row_ub})
        f.write((const char *)v->data(), p.rows() * 8);
    f.write((const char *)&p.offset, 8);
    std::ofstream m(dir + "/" + name + ".mps");
    write_mps(m, p); // for run_mps and other solvers
}

template <template <typename> class Store>
static void
run(const char *solver, const Inst &in, double eps, double tlimit,
    LpMethod method = LpMethod::Pdlp)
{
    SolverOptions o;
    o.method = method;
    o.set_tolerance(eps);
    o.time_limit = tlimit;
    o.max_iterations = 5000000;
    auto t0 = std::chrono::steady_clock::now();
    LpSolution s = solve_lp<Store>(in.p, o);
    double ms = std::chrono::duration<double, std::milli>(
        std::chrono::steady_clock::now() - t0).count();
    std::string tag = std::string(solver) + "_" + (eps <= 1e-8 ? "1e-8" : eps <= 1e-6 ? "1e-6" : "1e-4");
    printf("RESULT lp %s %s %.1f %ld %s %.9e\n", tag.c_str(), in.name.c_str(), ms,
        s.iterations, to_string(s.status), s.primal_objective);
    fflush(stdout);
}

int
main(int argc, char **argv)
{
    std::string dir = "build/lp_bench";
    double tlimit = 120;
    bool big = false, cuda = true, ipm = true, export_only = false;
    for (int i = 1; i < argc; ++i) {
        std::string a = argv[i];
        if (a == "--big") big = true;
        if (a == "--no-cuda") cuda = false;
        if (a == "--no-ipm") ipm = false;
        if (a == "--export-only") export_only = true;
        if (a == "--time" && i + 1 < argc) tlimit = atof(argv[++i]);
    }
    mkdir("build", 0755);
    mkdir(dir.c_str(), 0755);

    std::vector<Inst> insts;
    insts.push_back({"transport100", gen_transport(100, 100, 1)});
    insts.push_back({"transport300", gen_transport(300, 300, 2)});
    insts.push_back({"mcf10k", gen_mcf(10000, 50000, 3)});
    insts.push_back({"packing20k", gen_packing(20000, 50000, 4, 4)});
    if (big) {
        insts.push_back({"transport700", gen_transport(700, 700, 5)});
        insts.push_back({"mcf50k", gen_mcf(50000, 250000, 6)});
        insts.push_back({"packing200k", gen_packing(200000, 500000, 4, 7)});
    }
    for (auto &in : insts) {
        export_lp(dir, in.name, in.p);
        printf("# %s: %zu rows, %zu cols, %zu nnz\n", in.name.c_str(), in.p.rows(),
            in.p.cols(), in.p.A.nnz());
    }
    if (export_only) return 0;
    if (ipm)
        for (auto &in : insts) {
            run<Cpu::HostStorage>("ipm_cpu", in, 1e-8, tlimit, LpMethod::Ipm);
#ifdef AXOS_ENABLE_CUDA
            if (cuda) run<Cuda::CudaStorage>("ipm_cuda", in, 1e-8, tlimit, LpMethod::Ipm);
#endif
        }
    for (auto &in : insts)
        for (double eps : {1e-4, 1e-6}) {
            run<Cpu::HostStorage>("pdlp_cpu", in, eps, tlimit);
#ifdef AXOS_ENABLE_CUDA
            if (cuda) run<Cuda::CudaStorage>("pdlp_cuda", in, eps, tlimit);
#endif
        }
    return 0;
}
