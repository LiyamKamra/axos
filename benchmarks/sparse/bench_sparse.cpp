// Sparse benchmark: AXOS (CPU, CUDA) vs Eigen. Also exports the test
// matrices so bench_torch.py measures exactly the same inputs.
//
//   make benchmark_sparse
//   OMP_NUM_THREADS=1 ./build/bench_sparse     # single-thread AXOS
//
// Output lines:  RESULT <lib> <op> <matrix> <ms>
// (compare with benchmarks/sparse/report.py)

#ifdef AXOS_ENABLE_CUDA
#include "tensorCuda.h"
#endif
#include "sparse/sparse.h"
#include <Eigen/Dense>
#include <Eigen/Sparse>
#include <algorithm>
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <fstream>
#include <functional>
#include <random>
#include <string>
#include <sys/stat.h>
#include <vector>

using namespace AXOS;
using namespace AXOS::Sparse;
using HostCsr = Csr<double, int32_t, Cpu::HostStorage>;
using ESp = Eigen::SparseMatrix<double, Eigen::RowMajor, int>;
using EMapSp = Eigen::Map<const ESp>;

static double
now_ms()
{
    return std::chrono::duration<double, std::milli>(
        std::chrono::steady_clock::now().time_since_epoch())
        .count();
}

static void
sync_device()
{
#ifdef AXOS_ENABLE_CUDA
    cudaDeviceSynchronize();
#endif
}

// Median wall time in ms: warm up, then repeat for ~min_s seconds.
static double
bench(const std::function<void()> &f, double min_s = 0.4, int min_reps = 5,
    int max_reps = 400)
{
    f();
    sync_device();
    f();
    sync_device();
    std::vector<double> t;
    double start = now_ms();
    while ((int)t.size() < min_reps ||
           ((now_ms() - start) < min_s * 1000 && (int)t.size() < max_reps)) {
        double a = now_ms();
        f();
        sync_device();
        t.push_back(now_ms() - a);
    }
    std::sort(t.begin(), t.end());
    return t[t.size() / 2];
}

static void
result(const char *lib, const char *op, const std::string &m, double ms)
{
    printf("RESULT %s %s %s %.4f\n", lib, op, m.c_str(), ms);
    fflush(stdout);
}

// ─── matrix generators ───────────────────────────────────────────────────────
static HostCsr
gen_random(size_t n, int per_row, unsigned seed)
{
    std::mt19937_64 g(seed);
    std::uniform_int_distribution<size_t> col(0, n - 1);
    std::uniform_real_distribution<double> val(-1, 1);
    CooBuilder<double> b(n, n);
    b.reserve(n * per_row);
    for (size_t i = 0; i < n; ++i)
        for (int k = 0; k < per_row; ++k)
            b.add(i, col(g), val(g));
    return b.build();
}

// Row lengths follow a heavy-tailed distribution (a few very long rows).
static HostCsr
gen_skewed(size_t n, double avg, unsigned seed)
{
    std::mt19937_64 g(seed);
    std::uniform_int_distribution<size_t> col(0, n - 1);
    std::uniform_real_distribution<double> val(-1, 1), u(0, 1);
    CooBuilder<double> b(n, n);
    for (size_t i = 0; i < n; ++i) {
        // Pareto(alpha = 1.5), scaled so the mean is ~avg
        double len = avg / 3.0 * std::pow(1.0 - u(g), -1.0 / 1.5);
        size_t k = std::min<size_t>(std::max(1.0, len), n / 4);
        for (size_t c = 0; c < k; ++c)
            b.add(i, col(g), val(g));
    }
    return b.build();
}

static HostCsr
gen_laplacian(size_t g)
{
    const size_t n = g * g;
    CooBuilder<double> b(n, n);
    b.reserve(5 * n);
    for (size_t i = 0; i < g; ++i)
        for (size_t j = 0; j < g; ++j) {
            size_t p = i * g + j;
            b.add(p, p, 4.0);
            if (i > 0) b.add(p, p - g, -1.0);
            if (i + 1 < g) b.add(p, p + g, -1.0);
            if (j > 0) b.add(p, p - 1, -1.0);
            if (j + 1 < g) b.add(p, p + 1, -1.0);
        }
    return b.build();
}

static void
export_matrix(const std::string &dir, const std::string &name, const HostCsr &A)
{
    std::ofstream f(dir + "/" + name + ".bin", std::ios::binary);
    int64_t h[3] = {(int64_t)A.rows(), (int64_t)A.cols(), (int64_t)A.nnz()};
    f.write((const char *)h, sizeof(h));
    f.write((const char *)A.row_ptr(), (A.rows() + 1) * sizeof(int32_t));
    f.write((const char *)A.col_ind(), A.nnz() * sizeof(int32_t));
    f.write((const char *)A.values(), A.nnz() * sizeof(double));
}

// ─── operation benchmarks ────────────────────────────────────────────────────
template <typename Csr_, typename VS>
static void
bench_axos_ops(const char *lib, const std::string &name, const HostCsr &H,
    bool do_spgemm)
{
    using S = tensorET<1, double, VS>;
    Csr_ A(H);
    const size_t m = H.rows(), n = H.cols(), k = 16;
    tensorET<1, double> hx({n}, 0.0);
    for (size_t i = 0; i < n; ++i) hx.data[i] = std::sin(0.001 * i);
    S x(hx), y({m}, 0.0), yt({n}, 0.0);
    tensorET<2, double> hX({n, k}, 0.5);
    tensorET<2, double, VS> X(hX), Y({m, k}, 0.0);

    result(lib, "spmv", name, bench([&] { spmv(A, x, y); }));
    (void)A.transposed(); // cached transpose built outside the timing
    result(lib, "spmv_t", name, bench([&] { spmv_t(A, y, yt); }));
    result(lib, "spmm16", name, bench([&] { spmm(A, X, Y); }));
    result(lib, "transpose", name, bench([&] {
        Csr_ T = A.transpose();
        (void)T;
    }));
    if (do_spgemm)
        result(lib, "spgemm", name, bench([&] {
            Csr_ C = spgemm(A, A);
            (void)C;
        }, 0.4, 3, 50));
}

static void
bench_eigen_ops(const std::string &name, const HostCsr &H, bool do_spgemm)
{
    EMapSp A(H.rows(), H.cols(), H.nnz(), H.row_ptr(), H.col_ind(), H.values());
    const size_t m = H.rows(), n = H.cols(), k = 16;
    Eigen::VectorXd x(n), y(m), yt(n);
    for (size_t i = 0; i < n; ++i) x(i) = std::sin(0.001 * i);
    Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor> X =
        Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic,
            Eigen::RowMajor>::Constant(n, k, 0.5),
        Y(m, k);
    result("eigen", "spmv", name, bench([&] { y.noalias() = A * x; }));
    result("eigen", "spmv_t", name,
        bench([&] { yt.noalias() = A.transpose() * y; }));
    result("eigen", "spmm16", name, bench([&] { Y.noalias() = A * X; }));
    result("eigen", "transpose", name, bench([&] {
        ESp T = A.transpose();
        (void)T;
    }));
    if (do_spgemm)
        result("eigen", "spgemm", name, bench([&] {
            ESp C = A * A;
            (void)C;
        }, 0.4, 3, 50));
}

// Cross-check AXOS and Eigen results on one matrix so the timings compare
// the same computation.
static void
check_agreement(const std::string &name, const HostCsr &H, bool do_spgemm)
{
    EMapSp A(H.rows(), H.cols(), H.nnz(), H.row_ptr(), H.col_ind(), H.values());
    const size_t n = H.cols();
    tensorET<1, double> x({n}, 0.0), y({H.rows()}, 0.0);
    Eigen::VectorXd ex(n);
    for (size_t i = 0; i < n; ++i) x.data[i] = ex(i) = std::sin(0.001 * i);
    spmv(H, x, y);
    Eigen::VectorXd ey = A * ex;
    double e = 0;
    for (size_t i = 0; i < H.rows(); ++i)
        e = std::max(e, std::abs(ey(i) - y.data[i]));
    double e2 = 0;
    if (do_spgemm) {
        HostCsr C = spgemm(H, H);
        ESp EC = A * A;
        e2 = (double)C.nnz() - (double)EC.nonZeros();
    }
    printf("CHECK %s spmv max|diff|=%.2e spgemm nnz diff=%.0f\n", name.c_str(),
        e, e2);
}

template <typename Solver, typename Vec>
static void
bench_ldl(const char *lib, const std::string &name, const HostCsr &H,
    typename Solver::matrix_type &A, bool time_analyze)
{
    const size_t n = H.rows();
    tensorET<1, double> hb({n}, 1.0);
    Vec b(hb), x({n}, 0.0);
    Solver s(Symmetry::SPD);
    double t0 = now_ms();
    s.analyze(A);
    sync_device();
    if (time_analyze) result(lib, "ldl_analyze", name, now_ms() - t0);
    result(lib, "ldl_factor", name, bench([&] { s.factorize(A); }, 0.4, 3, 30));
    result(lib, "ldl_solve", name, bench([&] { s.solve(b, x); }));
}

static void
bench_eigen_ldl(const std::string &name, const HostCsr &H)
{
    Eigen::SparseMatrix<double> A = EMapSp(H.rows(), H.cols(), H.nnz(),
        H.row_ptr(), H.col_ind(), H.values());
    Eigen::VectorXd b = Eigen::VectorXd::Ones(H.rows()), x;
    Eigen::SimplicialLDLT<Eigen::SparseMatrix<double>, Eigen::Lower,
        Eigen::AMDOrdering<int>>
        s;
    double t0 = now_ms();
    s.analyzePattern(A);
    result("eigen", "ldl_analyze", name, now_ms() - t0);
    result("eigen", "ldl_factor", name, bench([&] { s.factorize(A); }, 0.4, 3, 30));
    result("eigen", "ldl_solve", name, bench([&] { x = s.solve(b); }));
}

int
main(int argc, char **argv)
{
    std::string dir = "build/sparse_bench";
    bool run_cuda = true, ldl = true;
    for (int i = 1; i < argc; ++i) {
        std::string a = argv[i];
        if (a == "--no-cuda") run_cuda = false;
        if (a == "--no-ldl") ldl = false;
        if (a == "--dir" && i + 1 < argc) dir = argv[++i];
    }
    mkdir("build", 0755);
    mkdir(dir.c_str(), 0755);

    struct Item { std::string name; HostCsr A; bool spgemm; };
    std::vector<Item> items;
    items.push_back({"rand20k", gen_random(20000, 10, 1), true});
    items.push_back({"lap300", gen_laplacian(300), true});
    items.push_back({"rand100k", gen_random(100000, 10, 2), false});
    items.push_back({"skew200k", gen_skewed(200000, 8, 3), false});
    items.push_back({"lap1000", gen_laplacian(1000), false});
    items.push_back({"rand1m", gen_random(1000000, 10, 4), false});
    std::vector<Item> ldl_items;
    ldl_items.push_back({"lap100", gen_laplacian(100), false});
    ldl_items.push_back({"lap200", gen_laplacian(200), false});
    ldl_items.push_back({"lap400", gen_laplacian(400), false});

    for (auto &it : items) export_matrix(dir, it.name, it.A);
    for (auto &it : ldl_items) export_matrix(dir, it.name, it.A);

    printf("# matrices:");
    for (auto &it : items)
        printf(" %s(%zux%zu nnz=%zu)", it.name.c_str(), it.A.rows(),
            it.A.cols(), it.A.nnz());
    printf("\n");

    for (auto &it : items) {
        check_agreement(it.name, it.A, it.spgemm);
        bench_axos_ops<HostCsr, Cpu::HostStorage<double>>(
            "axos_cpu", it.name, it.A, it.spgemm);
        bench_eigen_ops(it.name, it.A, it.spgemm);
#ifdef AXOS_ENABLE_CUDA
        if (run_cuda)
            bench_axos_ops<CsrCuda<double>, Cuda::CudaStorage<double>>(
                "axos_cuda", it.name, it.A, it.spgemm);
#endif
    }

    if (ldl) {
        for (auto &it : ldl_items) {
            HostCsr &A = it.A;
            bench_ldl<SparseLdlt<double, int32_t, Cpu::HostStorage>,
                tensorET<1, double>>("axos_cpu", it.name, A, A, true);
            {   // the scalar up-looking factorization, for comparison
                using Simp = SparseLdlt<double, int32_t, Cpu::HostStorage>;
                const size_t n = A.rows();
                tensorET<1, double> hb({n}, 1.0), x({n}, 0.0);
                Simp s(Symmetry::SPD, Ordering::MinDegree, Factorization::Simplicial);
                s.analyze(A);
                if (n <= 50000) {
                    result("axos_cpu_simplicial", "ldl_factor", it.name,
                        bench([&] { s.factorize(A); }, 0.4, 3, 30));
                    s.factorize(A);
                    result("axos_cpu_simplicial", "ldl_solve", it.name,
                        bench([&] { s.solve(hb, x); }));
                }
            }
            bench_eigen_ldl(it.name, A);
#ifdef AXOS_ENABLE_CUDA
            if (run_cuda) {
                CsrCuda<double> dA(A);
                bench_ldl<SparseLdlt<double, int32_t, Cuda::CudaStorage>,
                    tensorET<1, double, Cuda::CudaStorage<double>>>(
                    "axos_cuda", it.name, A, dA, true);
            }
#endif
        }
    }
    return 0;
}
