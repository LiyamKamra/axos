#include <Eigen/Dense>
#include <chrono>
#include <iostream>
#include <string>
#include <type_traits>

using Clock = std::chrono::steady_clock;

template <typename T>
void
run_eigen(size_t N, int iters)
{
    using MatrixType = typename std::conditional<std::is_same<T, double>::value,
        Eigen::MatrixXd, Eigen::MatrixXf>::type;
    MatrixType A = MatrixType::Constant(N, N, (T)1.0);
    MatrixType B = MatrixType::Constant(N, N, (T)2.0);
    MatrixType C = MatrixType::Zero(N, N);

    // Warmup
    C.noalias() = A * B;

    double total_ms = 0.0;
    for (int i = 0; i < iters; ++i) {
        auto start = Clock::now();
        C.noalias() = A * B;
        auto end = Clock::now();
        total_ms +=
            std::chrono::duration<double, std::milli>(end - start).count();
    }

    double ms = total_ms / iters;
    if (C(0, 0) == (T)0.0) std::cout << "";
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
    if (argc >= 3) { precision = argv[2]; }
    int iters = 5;
    if (argc >= 4) { iters = std::stoi(argv[3]); }

    bool use_float = (precision == "float" || precision == "single");
    if (use_float)
        run_eigen<float>(N, iters);
    else
        run_eigen<double>(N, iters);

    return 0;
}
