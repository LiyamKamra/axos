// SPDX-License-Identifier: BSD-3-Clause
//
// Dense layer (Tier 2) against Eigen, for the targets in
// docs/TENSOR_SPEC.md §11.2. Every row is the median of 5 timed runs after a
// warm-up run, in milliseconds; "speedup" is Eigen's time / AXOS's time
// (above 1 means AXOS is faster).
//
//   make benchmark_dense                        # all threads, then 1 thread
//   OMP_NUM_THREADS=1 ./build/bench_dense       # single-threaded rows
//   ./build/bench_dense quick                   # smaller sizes
#include "axos.h"
#include <Eigen/Dense>
#include <Eigen/IterativeLinearSolvers>
#if __has_include(<unsupported/Eigen/MatrixFunctions>)
#include <unsupported/Eigen/MatrixFunctions>
#define AXOS_BENCH_EIGEN_EXPM 1
#endif

#include <algorithm>
#include <chrono>
#include <cstdio>
#include <cstring>
#include <functional>
#include <string>
#include <vector>

#ifdef _OPENMP
#include <omp.h>
#endif

using namespace AXOS;
using EMat = Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>;
using EMatF = Eigen::Matrix<float, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>;

static double
median_ms(const std::function<void()> &f, int runs = 5)
{
    f(); // warm-up
    std::vector<double> t;
    for (int r = 0; r < runs; ++r) {
        const auto s = std::chrono::steady_clock::now();
        f();
        t.push_back(std::chrono::duration<double, std::milli>(
            std::chrono::steady_clock::now() - s).count());
    }
    std::sort(t.begin(), t.end());
    return t[t.size() / 2];
}

// eigen < 0: no Eigen counterpart
static void
row(const char *op, size_t n, double axos, double eigen)
{
    if (eigen < 0) {
        std::printf("| %-26s | %6zu | %10.3f | %10s | %7s |\n", op, n, axos, "n/a", "");
        std::fflush(stdout);
        return;
    }
    std::printf("| %-26s | %6zu | %10.3f | %10.3f | %6.2fx |\n", op, n, axos, eigen,
        eigen / axos);
    std::fflush(stdout);
}

template <class T>
static tensorET<2, T>
rnd(size_t m, size_t n, unsigned seed)
{
    return tensorET<2, T>::uniform({m, n}, T(-1), T(1), seed);
}

static EMat
emat(const tensorET<2, double> &t)
{
    return Eigen::Map<const EMat>(t.data, Eigen::Index(t.size(0)), Eigen::Index(t.size(1)));
}

int
main(int argc, char **argv)
{
    const bool quick = argc > 1 && std::string(argv[1]) == "quick";
    int threads = 1;
#ifdef _OPENMP
    threads = omp_get_max_threads();
#endif
    std::printf("AXOS dense layer vs Eigen %d.%d.%d, %d thread(s), SIMD: %s\n\n",
        EIGEN_WORLD_VERSION, EIGEN_MAJOR_VERSION, EIGEN_MINOR_VERSION, threads,
#if defined(AXOS_SIMD_AVX512)
        "AVX-512"
#elif defined(AXOS_SIMD_AVX2)
        "AVX2+FMA"
#elif defined(AXOS_SIMD_SSE2)
        "SSE2"
#else
        "none"
#endif
    );
    std::printf("| %-26s | %6s | %10s | %10s | %7s |\n", "operation", "n", "axos ms",
        "eigen ms", "speedup");
    std::printf("|%s|%s|%s|%s|%s|\n", std::string(28, '-').c_str(), std::string(8, '-').c_str(),
        std::string(12, '-').c_str(), std::string(12, '-').c_str(), std::string(9, '-').c_str());

    // GEMM
    const std::vector<size_t> gemm_n = quick ? std::vector<size_t>{256, 1024}
                                             : std::vector<size_t>{256, 512, 1024, 2048, 3072};
    for (size_t n : gemm_n) {
        auto A = rnd<double>(n, n, 1), B = rnd<double>(n, n, 2);
        tensorET<2, double> C({n, n});
        EMat eA = emat(A), eB = emat(B), eC(n, n);
        row("gemm f64 (matMul)", n, median_ms([&] { C = matMul(A, B); }),
            median_ms([&] { eC.noalias() = eA * eB; }));
    }
    for (size_t n : gemm_n) {
        auto A = rnd<float>(n, n, 1), B = rnd<float>(n, n, 2);
        tensorET<2, float> C({n, n});
        EMatF eA = Eigen::Map<EMatF>(A.data, n, n), eB = Eigen::Map<EMatF>(B.data, n, n), eC(n, n);
        row("gemm f32 (matMul)", n, median_ms([&] { C = matMul(A, B); }),
            median_ms([&] { eC.noalias() = eA * eB; }));
    }
    // GEMV
    for (size_t n : quick ? std::vector<size_t>{4096} : std::vector<size_t>{4096, 8192, 16384}) {
        auto A = rnd<double>(n, n, 3);
        auto x = tensorET<1, double>::uniform({n}, -1.0, 1.0, 4);
        tensorET<1, double> y({n});
        EMat eA = emat(A);
        Eigen::VectorXd ex = Eigen::Map<Eigen::VectorXd>(x.data, n), ey(n);
        row("gemv f64", n, median_ms([&] { gemv(A, x, y); }),
            median_ms([&] { ey.noalias() = eA * ex; }));
    }
    // element-wise
    {
        const size_t n = 10000000;
        auto A = tensorET<1, double>::uniform({n}, 0.1, 2.0, 5);
        auto B = tensorET<1, double>::uniform({n}, 0.1, 2.0, 6);
        tensorET<1, double> C({n});
        Eigen::Map<Eigen::VectorXd> eA(A.data, n), eB(B.data, n);
        Eigen::VectorXd eC(n);
        row("C = A + B", n, median_ms([&] { C = A + B; }), median_ms([&] { eC = eA + eB; }));
        row("C = A + B * 2", n, median_ms([&] { C = A + B * 2.0; }),
            median_ms([&] { eC = eA + eB * 2.0; }));
        row("C = sqrt(A)", n, median_ms([&] { C = sqrt(A); }),
            median_ms([&] { eC = eA.array().sqrt().matrix(); }));
        const size_t m = 1000000;
        auto a = tensorET<1, double>::uniform({m}, -1.0, 1.0, 7);
        tensorET<1, double> c({m});
        Eigen::Map<Eigen::VectorXd> ea(a.data, m);
        Eigen::VectorXd ec(m);
        row("C = exp(sin(A) + cos(A))", m, median_ms([&] { c = exp(sin(a) + cos(a)); }),
            median_ms([&] { ec = (ea.array().sin() + ea.array().cos()).exp().matrix(); }));
    }
    // transpose
    {
        const size_t n = 2048;
        auto A = rnd<double>(n, n, 8);
        tensorET<2, double> T({n, n});
        EMat eA = emat(A), eT(n, n);
        row("transpose f64", n,
            median_ms([&] { kernels::transpose<double>(n, n, A.data, n, T.data, n); }),
            median_ms([&] { eT = eA.transpose(); }));
    }
    // LU, solve, inverse
    for (size_t n : quick ? std::vector<size_t>{500} : std::vector<size_t>{500, 1000, 2000}) {
        auto A = rnd<double>(n, n, 9);
        EMat eA = emat(A);
        row("luDcmpPivoted", n, median_ms([&] { auto r = luDcmpPivoted(A); }),
            median_ms([&] { Eigen::PartialPivLU<EMat> lu(eA); }));
        auto lu = luDcmpPivoted(A);
        auto b = tensorET<1, double>::uniform({n}, -1.0, 1.0, 10);
        Eigen::PartialPivLU<EMat> elu(eA);
        Eigen::VectorXd eb = Eigen::Map<Eigen::VectorXd>(b.data, n), ex(n);
        row("luSolve, 1 rhs", n,
            median_ms([&] { auto x = luSolve(lu.first.first, lu.first.second, lu.second, b); }),
            median_ms([&] { ex = elu.solve(eb); }));
    }
    for (size_t n : quick ? std::vector<size_t>{256} : std::vector<size_t>{256, 512, 1024}) {
        auto A = rnd<double>(n, n, 11);
        for (size_t i = 0; i < n; ++i) A(i, i) += 4.0;
        EMat eA = emat(A);
        row("inverse (general)", n, median_ms([&] { auto X = inverse(A); }),
            median_ms([&] { EMat X = eA.inverse(); }));
    }
    // upper-triangular inverse
    for (size_t n : quick ? std::vector<size_t>{512} : std::vector<size_t>{512, 1024, 2048}) {
        auto A = rnd<double>(n, n, 12);
        for (size_t i = 0; i < n; ++i) {
            A(i, i) = 2.0 + double(i % 7);
            for (size_t j = 0; j < i; ++j) A(i, j) = 0.0;
        }
        EMat eU = emat(A);
        row("inverse_backs (upper)", n, median_ms([&] { auto X = inverse_backs(A, 64); }),
            median_ms([&] {
                EMat X = EMat::Identity(n, n);
                eU.triangularView<Eigen::Upper>().solveInPlace(X);
            }));
    }
    // QR
    for (size_t n : quick ? std::vector<size_t>{500} : std::vector<size_t>{500, 1000}) {
        auto A = rnd<double>(n, n, 13);
        EMat eA = emat(A);
        row("qrDecompositionTile", n,
            median_ms([&] {
                auto Q = A.clone();
                auto R = qrDecompositionTile(Q);
            }),
            median_ms([&] {
                Eigen::HouseholderQR<EMat> qr(eA);
                EMat Q = qr.householderQ() * EMat::Identity(n, n);
                EMat R = qr.matrixQR().triangularView<Eigen::Upper>();
            }));
    }
    // CG: 20 iterations
    {
        const size_t n = 2000;
        auto M = rnd<double>(n, n, 14);
        tensorET<2, double> S = matMul(M, transpose(M));
        for (size_t i = 0; i < n; ++i) S(i, i) += double(n);
        // A large right-hand side keeps p^T A p far above the spec'd 1e-12
        // stop, so both solvers run exactly 20 iterations.
        auto b = tensorET<1, double>::uniform({n}, -1e6, 1e6, 15);
        tensorET<1, double> x0({n}, 0.0);
        EMat eS = emat(S);
        Eigen::VectorXd eb = Eigen::Map<Eigen::VectorXd>(b.data, n);
        Eigen::ConjugateGradient<EMat, Eigen::Lower | Eigen::Upper, Eigen::IdentityPreconditioner> ecg;
        ecg.setMaxIterations(20);
        ecg.setTolerance(1e-30);
        row("conjugateGradient, 20 it", n,
            median_ms([&] { auto x = conjugateGradient(S, b, x0, 20, 1e-30); }),
            median_ms([&] {
                ecg.compute(eS);
                Eigen::VectorXd x = ecg.solve(eb);
            }));
    }
    // expm (Eigen's is in unsupported/; compared when available)
    for (size_t n : {size_t(100), size_t(200)}) {
        auto A = rnd<double>(n, n, 16);
        const double ms = median_ms([&] { auto E = expm(A); });
#ifdef AXOS_BENCH_EIGEN_EXPM
        EMat eA = emat(A);
        row("expm", n, ms, median_ms([&] { EMat E = eA.exp(); }));
#else
        row("expm", n, ms, -1.0);
#endif
    }
    return 0;
}
