#pragma once
// ─── Shared test infrastructure (no CUDA headers) ────────────────────────────
// Included by test_cpu.cpp (GCC) and test_cuda.cpp (NVCC).
// CUDA-specific includes go into test_cuda.cpp directly.

#include "axos.h" // dense tensor layer, see docs/TENSOR_SPEC.md
#include <Eigen/Dense>
#include <algorithm>
#include <chrono>
#include <cmath>
#include <complex>
#include <cstring>
#include <functional>
#include <random>
#include <sstream>
#include <string>
#include <unistd.h>
#include <vector>

using namespace AXOS;

struct TestResult {
    std::string name, category;
    bool pass;
    double err, time_ms;
    std::string detail;
};

static std::vector<TestResult> results;
static bool headless = false;

static void
tlog(const char *cat, const char *name, bool pass, double err, double ms,
    const char *det = "")
{
    results.push_back({name, cat, pass, err, ms, det});
    if (headless) {
        printf(" [%s] %-30s %s  err=%.2e  %.3fms  %s\n", cat, name,
            pass ? "PASS" : "FAIL", err, ms, det);
    }
}

static double
timer_ms(std::chrono::steady_clock::time_point s)
{
    return std::chrono::duration<double, std::milli>(
        std::chrono::steady_clock::now() - s)
        .count();
}
#define DECLARE_TIMER std::chrono::steady_clock::time_point _t
#define TIC _t = std::chrono::steady_clock::now()
#define TOC timer_ms(_t)
#define THRESH 1e-8

static void
fill_rand(tensorET<2, double> &A, Eigen::MatrixXd &E, size_t N, std::mt19937 &g)
{
    std::uniform_real_distribution<double> d(-5, 5);
    for (size_t i = 0; i < N; i++)
        for (size_t j = 0; j < N; j++) {
            double v = d(g);
            A(i, j) = v;
            E(i, j) = v;
        }
}

static void
fill_diag_dom(
    tensorET<2, double> &A, Eigen::MatrixXd &E, size_t N, std::mt19937 &g)
{
    std::uniform_real_distribution<double> d(-5, 5);
    for (size_t i = 0; i < N; i++) {
        double s = 0;
        for (size_t j = 0; j < N; j++) {
            if (i != j) {
                double v = d(g);
                A(i, j) = v;
                E(i, j) = v;
                s += std::abs(v);
            }
        }
        double dv = s + 1 + std::abs(d(g));
        A(i, i) = dv;
        E(i, i) = dv;
    }
}

static void
make_sym(tensorET<2, double> &S, Eigen::MatrixXd &eS,
    const tensorET<2, double> &A, size_t N)
{
    for (size_t i = 0; i < N; i++)
        for (size_t j = 0; j < N; j++) {
            double v = (A(i, j) + A(j, i)) * 0.5;
            S(i, j) = v;
            eS(i, j) = v;
        }
}

// ─── headless / interactive main helpers ─────────────────────────────────────

static void
print_summary()
{
    int pass = 0, fail = 0;
    for (auto &r : results) {
        if (r.pass)
            pass++;
        else
            fail++;
    }
    printf("\nSUMMARY: %d passed, %d failed out of %d total\n", pass, fail,
        pass + fail);
    for (auto &r : results)
        if (!r.pass)
            fprintf(stderr, "FAIL: [%s] %s err=%.2e\n", r.category.c_str(),
                r.name.c_str(), r.err);
}

static bool
parse_headless(int argc, char *argv[])
{
    for (int i = 1; i < argc; ++i)
        if (std::string(argv[i]) == "--headless") return true;
    return !isatty(1) || getenv("HEADLESS");
}
