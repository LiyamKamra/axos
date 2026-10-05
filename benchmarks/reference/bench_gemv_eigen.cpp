#include <Eigen/Dense>
#include <chrono>
#include <iostream>
#include <string>
#include <type_traits>

using Clock = std::chrono::steady_clock;

template <typename T>
void
run_gemv_eigen(size_t N, int iters)
{
    using Mat = typename std::conditional<std::is_same<T, double>::value,
        Eigen::MatrixXd, Eigen::MatrixXf>::type;
    using Vec = typename std::conditional<std::is_same<T, double>::value,
        Eigen::VectorXd, Eigen::VectorXf>::type;

    Mat A = Mat::Constant(N, N, (T)1.0);
    Vec x = Vec::Constant(N, (T)2.0);
    Vec y(N);

    // Warmup
    y.noalias() = A * x;

    double total_ms = 0.0;
    for (int i = 0; i < iters; ++i) {
        auto t0 = Clock::now();
        y.noalias() = A * x;
        auto t1 = Clock::now();
        total_ms += std::chrono::duration<double, std::milli>(t1 - t0).count();
    }

    double ms = total_ms / iters;
    if (y(0) == (T)0.0) std::cout << ""; // prevent DCE
    std::cout << ms << std::endl;
}

int
main(int argc, char *argv[])
{
    if (argc < 2) {
        std::cerr << "Usage: " << argv[0] << " <size> [double|float] [iters]\n";
        return 1;
    }
    size_t N = std::stoull(argv[1]);
    std::string precision = "double";
    if (argc >= 3) precision = argv[2];
    int iters = 5;
    if (argc >= 4) iters = std::stoi(argv[3]);

    bool use_float = (precision == "float" || precision == "single");
    if (use_float)
        run_gemv_eigen<float>(N, iters);
    else
        run_gemv_eigen<double>(N, iters);

    return 0;
}
