// Sparse (Csr) test suite.
//
// CPU build:   make test_sparse_cpu
// CUDA build:  make test_sparse      (adds the CUDA kernels and cuDSS tests)
//
// Every test is templated on the matrix storage, builds its inputs on the
// host, moves them to the storage under test, and compares the downloaded
// result with Eigen.

#ifdef AXOS_ENABLE_CUDA
#include "tensorCuda.h"
#endif
#include "sparse/sparse.h"
#include "test_common.h"

#include <Eigen/Dense>
#include <numeric>

using namespace AXOS;
using namespace AXOS::Sparse;

static const int RUNS = 3;

template <typename T>
using EMat = Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>;
template <typename T> using EVec = Eigen::Matrix<T, Eigen::Dynamic, 1>;

template <typename T>
static T
rand_val(std::mt19937 &g)
{
    std::uniform_real_distribution<double> d(-2, 2);
    if constexpr (is_complex_value<T>::value)
        return T(d(g), d(g));
    else
        return static_cast<T>(d(g));
}

// Host CSR plus its dense Eigen equivalent. Entries are generated with some
// duplicates so that the builder's duplicate summation is exercised.
template <typename T, typename Idx> struct RandSparse {
    Csr<T, Idx, Cpu::HostStorage> csr;
    EMat<T> dense;
};

template <typename T, typename Idx>
static RandSparse<T, Idx>
make_random(size_t m, size_t n, double density, unsigned seed)
{
    std::mt19937 g(seed);
    std::uniform_real_distribution<double> u(0, 1);
    CooBuilder<T, Idx> b(m, n);
    EMat<T> D = EMat<T>::Zero(m, n);
    for (size_t i = 0; i < m; ++i)
        for (size_t j = 0; j < n; ++j) {
            if (u(g) < density) {
                T v = rand_val<T>(g);
                b.add(i, j, v);
                D(i, j) += v;
                if (u(g) < 0.2) { // duplicate entry
                    T w = rand_val<T>(g);
                    b.add(i, j, w);
                    D(i, j) += w;
                }
            }
        }
    return {b.build(), D};
}

template <typename T>
static double
tol()
{
    return std::is_same_v<real_of_t<T>, float> ? 2e-4 : 1e-11;
}

template <typename T, typename Idx, template <typename> class S>
static EMat<T>
dense_of(const Csr<T, Idx, S> &A)
{
    tensorET<2, T> D = A.template to_dense<>();
    EMat<T> E(A.rows(), A.cols());
    for (size_t i = 0; i < A.rows(); ++i)
        for (size_t j = 0; j < A.cols(); ++j)
            E(i, j) = D.data[i * A.cols() + j];
    return E;
}

template <typename T>
static double
rel_err(const EMat<T> &a, const EMat<T> &b)
{
    if (a.size() == 0 && b.size() == 0) return 0;
    if (a.rows() != b.rows() || a.cols() != b.cols()) return 1e30;
    double scale = 1.0 + b.cwiseAbs().maxCoeff();
    return (a - b).cwiseAbs().maxCoeff() / scale;
}

template <typename T, template <typename> class S>
static tensorET<1, T, S<T>>
to_dev(const EVec<T> &v)
{
    tensorET<1, T> h({static_cast<size_t>(v.size())}, T(0));
    for (long i = 0; i < v.size(); ++i)
        h.data[i] = v(i);
    return tensorET<1, T, S<T>>(h);
}

template <typename T, typename S>
static EVec<T>
from_dev(const tensorET<1, T, S> &x)
{
    tensorET<1, T> h(x);
    EVec<T> v(h.size());
    for (size_t i = 0; i < h.size(); ++i)
        v(i) = h.data[i];
    return v;
}

template <typename T, template <typename> class S>
static tensorET<2, T, S<T>>
to_dev2(const EMat<T> &M)
{
    tensorET<2, T> h({static_cast<size_t>(M.rows()),
        static_cast<size_t>(M.cols())}, T(0));
    for (long i = 0; i < M.rows(); ++i)
        for (long j = 0; j < M.cols(); ++j)
            h.data[i * M.cols() + j] = M(i, j);
    return tensorET<2, T, S<T>>(h);
}

template <typename T, typename S>
static EMat<T>
from_dev2(const tensorET<2, T, S> &X)
{
    tensorET<2, T> h(X);
    EMat<T> M(h.size(0), h.size(1));
    for (size_t i = 0; i < h.size(0); ++i)
        for (size_t j = 0; j < h.size(1); ++j)
            M(i, j) = h.data[i * h.size(1) + j];
    return M;
}

// ─── builder / structure ─────────────────────────────────────────────────────
template <typename T, typename Idx, template <typename> class S>
static void
test_structure(const char *cat, const char *tag)
{
    std::string t = tag;
    // Random matrices with duplicate triplets.
    for (auto shape : {std::pair<size_t, size_t>{40, 30},
             std::pair<size_t, size_t>{1, 25}, std::pair<size_t, size_t>{25, 1},
             std::pair<size_t, size_t>{17, 17}}) {
        auto r = make_random<T, Idx>(shape.first, shape.second, 0.15, 7);
        Csr<T, Idx, S> A(r.csr);
        std::string v = A.validate();
        double e = rel_err<T>(dense_of(A), r.dense);
        std::string nm = t + " build " + std::to_string(shape.first) + "x" +
                         std::to_string(shape.second);
        tlog(cat, nm.c_str(), v.empty() && e < tol<T>(), e, 0, v.c_str());
    }

    // nnz == 0 and rows == 0.
    {
        CooBuilder<T, Idx> b(5, 4);
        Csr<T, Idx, S> A(b.build());
        bool ok = A.nnz() == 0 && A.rows() == 5 && A.validate().empty();
        EMat<T> D = dense_of(A);
        ok = ok && D.cwiseAbs().maxCoeff() == 0;
        tlog(cat, (t + " empty matrix nnz=0").c_str(), ok, 0, 0);
        Csr<T, Idx, S> Z(CooBuilder<T, Idx>(0, 3).build());
        tlog(cat, (t + " zero rows").c_str(),
            Z.rows() == 0 && Z.validate().empty(), 0, 0);
    }

    // drop_zeros and duplicate cancellation.
    {
        CooBuilder<T, Idx> b(3, 3);
        b.add(0, 0, T(1));
        b.add(0, 0, T(-1)); // cancels to zero
        b.add(1, 2, T(3));
        b.add(2, 1, T(0));
        auto keep = b.build(false);
        auto drop = b.build(true);
        bool ok = keep.nnz() == 3 && drop.nnz() == 1 && drop.validate().empty();
        tlog(cat, (t + " builder drop_zeros").c_str(), ok, 0, 0);
    }

    // validate() must reject broken patterns.
    {
        Csr<T, Idx, S> bad(2, 3, {0, 2, 3}, {1, 0, 2}, {T(1), T(2), T(3)});
        bool ok = !bad.validate().empty(); // columns not increasing in row 0
        tlog(cat, (t + " validate rejects unsorted").c_str(), ok, 0, 0);
    }

    // dense round trip, drop tolerance, append_rows.
    {
        auto r = make_random<T, Idx>(12, 9, 0.3, 21);
        Csr<T, Idx, S> A(r.csr);
        tensorET<2, T, S<T>> Dd = to_dev2<T, S>(r.dense);
        auto back = Csr<T, Idx, Cpu::HostStorage>::from_dense(Dd);
        double e = rel_err<T>(dense_of(back), r.dense);
        tlog(cat, (t + " dense round trip").c_str(),
            e < tol<T>() && back.validate().empty(), e, 0);

        auto big = make_random<T, Idx>(4, 9, 0.4, 22);
        Csr<T, Idx, S> B(big.csr);
        Csr<T, Idx, S> C = A.append_rows(B);
        EMat<T> ref(16, 9);
        ref << r.dense, big.dense;
        double e2 = rel_err<T>(dense_of(C), ref);
        tlog(cat, (t + " append_rows").c_str(),
            e2 < tol<T>() && C.validate().empty(), e2, 0);
    }

    // copy is deep; move empties the source.
    {
        auto r = make_random<T, Idx>(10, 10, 0.3, 5);
        Csr<T, Idx, S> A(r.csr);
        Csr<T, Idx, S> B(A);
        Csr<T, Idx, S> C(std::move(A));
        bool ok = A.nnz() == 0 && A.rows() == 0 && C.nnz() == B.nnz() &&
                  B.values() != C.values() &&
                  rel_err<T>(dense_of(B), dense_of(C)) == 0;
        tlog(cat, (t + " copy/move semantics").c_str(), ok, 0, 0);
    }
}

// ─── spmv / spmm / scaling / norms ───────────────────────────────────────────
template <typename T, typename Idx, template <typename> class S>
static void
test_products(const char *cat, const char *tag)
{
    std::string t = tag;
    std::mt19937 g(3);
    for (auto shape : {std::pair<size_t, size_t>{50, 35},
             std::pair<size_t, size_t>{1, 20}, std::pair<size_t, size_t>{20, 1},
             std::pair<size_t, size_t>{60, 60}}) {
        const size_t m = shape.first, n = shape.second;
        auto r = make_random<T, Idx>(m, n, 0.12, 11);
        Csr<T, Idx, S> A(r.csr);
        std::string sh = std::to_string(m) + "x" + std::to_string(n);

        EVec<T> x(n), y0(m), z(m), w(n);
        for (size_t i = 0; i < n; ++i) x(i) = rand_val<T>(g);
        for (size_t i = 0; i < m; ++i) y0(i) = rand_val<T>(g);
        for (size_t i = 0; i < m; ++i) z(i) = rand_val<T>(g);
        const T alpha = T(1.5), beta = T(-0.5);

        auto dx = to_dev<T, S>(x);
        auto dy = to_dev<T, S>(y0);
        spmv(A, dx, dy, alpha, beta);
        EVec<T> ref = alpha * (r.dense * x) + beta * y0;
        double e1 = (from_dev(dy) - ref).cwiseAbs().maxCoeff() /
                    (1 + ref.cwiseAbs().maxCoeff());
        tlog(cat, (t + " spmv " + sh).c_str(), e1 < tol<T>(), e1, 0);

        // beta == 0 must not read y (NaN-filled).
        EVec<T> nanv(m);
        nanv.setConstant(T(std::numeric_limits<real_of_t<T>>::quiet_NaN()));
        auto dn = to_dev<T, S>(nanv);
        spmv(A, dx, dn);
        EVec<T> ref0 = r.dense * x;
        double e2 = (from_dev(dn) - ref0).cwiseAbs().maxCoeff() /
                    (1 + ref0.cwiseAbs().maxCoeff());
        tlog(cat, (t + " spmv beta=0 ignores y " + sh).c_str(), e2 < tol<T>(),
            e2, 0);

        // A^T z
        auto dz = to_dev<T, S>(z);
        auto dw = to_dev<T, S>(EVec<T>::Zero(n));
        spmv_t(A, dz, dw);
        EVec<T> refT = r.dense.transpose() * z;
        double e3 = (from_dev(dw) - refT).cwiseAbs().maxCoeff() /
                    (1 + refT.cwiseAbs().maxCoeff());
        tlog(cat, (t + " spmv_t " + sh).c_str(), e3 < tol<T>(), e3, 0);

        // spmm
        const size_t k = 5;
        EMat<T> X(n, k), Y0(m, k);
        for (size_t i = 0; i < n; ++i)
            for (size_t c = 0; c < k; ++c) X(i, c) = rand_val<T>(g);
        for (size_t i = 0; i < m; ++i)
            for (size_t c = 0; c < k; ++c) Y0(i, c) = rand_val<T>(g);
        auto dX = to_dev2<T, S>(X);
        auto dY = to_dev2<T, S>(Y0);
        spmm(A, dX, dY, alpha, beta);
        EMat<T> refM = alpha * (r.dense * X) + beta * Y0;
        double e4 = rel_err<T>(from_dev2(dY), refM);
        tlog(cat, (t + " spmm " + sh).c_str(), e4 < tol<T>(), e4, 0);
    }

    // Row/column scaling and norms (real scale vectors).
    if constexpr (true) {
        using R = real_of_t<T>;
        const size_t m = 30, n = 22;
        auto r = make_random<T, Idx>(m, n, 0.2, 13);
        Csr<T, Idx, S> A(r.csr);
        std::mt19937 gg(9);
        std::uniform_real_distribution<double> u(0.5, 2);
        std::vector<R> rs(m), cs(n);
        for (auto &v : rs) v = static_cast<R>(u(gg));
        for (auto &v : cs) v = static_cast<R>(u(gg));
        tensorET<1, R> hr({m}, R(0)), hc({n}, R(0));
        for (size_t i = 0; i < m; ++i) hr.data[i] = rs[i];
        for (size_t j = 0; j < n; ++j) hc.data[j] = cs[j];
        tensorET<1, R, S<R>> dr(hr), dc(hc);

        // norms of the unscaled matrix
        tensorET<1, R, S<R>> nr({m}, R(0)), nc({n}, R(0));
        double worst = 0;
        for (Norm p : {Norm::L1, Norm::L2, Norm::Linf}) {
            row_norms(A, nr.data, p);
            col_norms(A, nc.data, p);
            tensorET<1, R> hnr(nr), hnc(nc);
            for (size_t i = 0; i < m; ++i) {
                double ref = 0;
                for (size_t j = 0; j < n; ++j) {
                    double a = std::abs(r.dense(i, j));
                    ref = p == Norm::L1    ? ref + a
                          : p == Norm::L2  ? ref + a * a
                                           : std::max(ref, a);
                }
                if (p == Norm::L2) ref = std::sqrt(ref);
                worst = std::max(worst, std::abs(hnr.data[i] - ref) / (1 + ref));
            }
            for (size_t j = 0; j < n; ++j) {
                double ref = 0;
                for (size_t i = 0; i < m; ++i) {
                    double a = std::abs(r.dense(i, j));
                    ref = p == Norm::L1    ? ref + a
                          : p == Norm::L2  ? ref + a * a
                                           : std::max(ref, a);
                }
                if (p == Norm::L2) ref = std::sqrt(ref);
                worst = std::max(worst, std::abs(hnc.data[j] - ref) / (1 + ref));
            }
        }
        tlog(cat, (t + " row/col norms L1,L2,Linf").c_str(), worst < tol<T>(),
            worst, 0);

        scale_rows_cols(A, dr.data, dc.data);
        EMat<T> refS = r.dense;
        for (size_t i = 0; i < m; ++i)
            for (size_t j = 0; j < n; ++j)
                refS(i, j) *= T(rs[i] * cs[j]);
        double es = rel_err<T>(dense_of(A), refS);
        tlog(cat, (t + " scale_rows_cols").c_str(),
            es < tol<T>() && A.validate().empty(), es, 0);
    }
}

// ─── transpose / spgemm ──────────────────────────────────────────────────────
template <typename T, typename Idx, template <typename> class S>
static void
test_transpose_spgemm(const char *cat, const char *tag)
{
    std::string t = tag;
    for (auto shape : {std::pair<size_t, size_t>{35, 20},
             std::pair<size_t, size_t>{1, 15}, std::pair<size_t, size_t>{15, 1}}) {
        auto r = make_random<T, Idx>(shape.first, shape.second, 0.2, 31);
        Csr<T, Idx, S> A(r.csr);
        Csr<T, Idx, S> At = A.transpose();
        double e = rel_err<T>(dense_of(At), EMat<T>(r.dense.transpose()));
        std::string nm = t + " transpose " + std::to_string(shape.first) + "x" +
                         std::to_string(shape.second);
        tlog(cat, nm.c_str(), e < tol<T>() && At.validate().empty(), e, 0,
            At.validate().c_str());
    }

    // Cached transpose is dropped when values are written.
    {
        using R = real_of_t<T>;
        auto r = make_random<T, Idx>(20, 14, 0.25, 32);
        Csr<T, Idx, S> A(r.csr);
        (void)A.transposed(); // populate the cache
        tensorET<1, R> h2({20}, R(2));
        tensorET<1, R, S<R>> two(h2);
        scale_rows_cols(A, two.data, static_cast<const R *>(nullptr));
        EMat<T> ref = r.dense * T(2);
        double e = rel_err<T>(dense_of(A.transposed()),
            EMat<T>(ref.transpose()));
        tlog(cat, (t + " transposed() refreshes after value write").c_str(),
            e < tol<T>(), e, 0);
        auto view = A.values_view();
        tlog(cat, (t + " values_view size").c_str(), view.size() == A.nnz(), 0,
            0);
    }

    struct Case { size_t m, k, n; };
    for (Case c : {Case{25, 18, 22}, Case{1, 10, 1}, Case{12, 12, 12},
             Case{20, 15, 0 + 9}}) {
        auto ra = make_random<T, Idx>(c.m, c.k, 0.2, 41);
        auto rb = make_random<T, Idx>(c.k, c.n, 0.2, 42);
        Csr<T, Idx, S> A(ra.csr), B(rb.csr);
        Csr<T, Idx, S> C = spgemm(A, B);
        EMat<T> ref = ra.dense * rb.dense;
        double e = rel_err<T>(dense_of(C), ref);
        std::string nm = t + " spgemm " + std::to_string(c.m) + "x" +
                         std::to_string(c.k) + "x" + std::to_string(c.n);
        tlog(cat, nm.c_str(), e < tol<T>() && C.validate().empty(), e, 0,
            C.validate().c_str());
    }

    // Normal equations A diag(d) A^T
    if constexpr (!is_complex_value<T>::value) {
        using R = real_of_t<T>;
        const size_t m = 18, n = 30;
        auto r = make_random<T, Idx>(m, n, 0.2, 51);
        Csr<T, Idx, S> A(r.csr);
        tensorET<1, R> hd({n}, R(0));
        std::mt19937 g(2);
        std::uniform_real_distribution<double> u(0.1, 3);
        for (size_t j = 0; j < n; ++j) hd.data[j] = static_cast<R>(u(g));
        tensorET<1, R, S<R>> dd(hd);
        Csr<T, Idx, S> N = AdAt(A, dd.data);
        EMat<T> ref = r.dense;
        for (size_t j = 0; j < n; ++j)
            ref.col(j) *= T(hd.data[j]);
        ref = ref * r.dense.transpose();
        double e = rel_err<T>(dense_of(N), ref);
        tlog(cat, (t + " AdAt normal equations").c_str(),
            e < tol<T>() * 10 && N.validate().empty(), e, 0);
    }
}

// ─── LDL^T solver ────────────────────────────────────────────────────────────
// B B^T + n I for a sparse random B: symmetric positive definite.
template <typename T, typename Idx>
static RandSparse<T, Idx>
make_spd(size_t n, double density, unsigned seed)
{
    auto B = make_random<T, Idx>(n, n, density, seed);
    EMat<T> M = B.dense * B.dense.transpose();
    M.diagonal().array() += T(n);
    CooBuilder<T, Idx> b(n, n);
    for (size_t i = 0; i < n; ++i)
        for (size_t j = 0; j < n; ++j)
            if (M(i, j) != T(0)) b.add(i, j, M(i, j));
    return {b.build(), M};
}

// Quasi-definite KKT matrix [[H, A^T], [A, -delta I]].
template <typename T, typename Idx>
static RandSparse<T, Idx>
make_kkt(size_t n, size_t m, double density, unsigned seed)
{
    auto Ac = make_random<T, Idx>(m, n, density, seed);
    std::mt19937 g(seed + 1);
    std::uniform_real_distribution<double> u(0.5, 3);
    EMat<T> K = EMat<T>::Zero(n + m, n + m);
    for (size_t i = 0; i < n; ++i) K(i, i) = T(u(g));
    K.block(n, 0, m, n) = Ac.dense;
    K.block(0, n, n, m) = Ac.dense.transpose();
    for (size_t i = 0; i < m; ++i) K(n + i, n + i) = T(-1e-2);
    CooBuilder<T, Idx> b(n + m, n + m);
    for (size_t i = 0; i < n + m; ++i)
        for (size_t j = 0; j < n + m; ++j)
            if (K(i, j) != T(0)) b.add(i, j, K(i, j));
    return {b.build(), K};
}

template <typename T, typename Idx, template <typename> class S>
static double
solve_residual(SparseLdlt<T, Idx, S> &ldl, const Csr<T, Idx, S> &A,
    const EMat<T> &dense, unsigned seed)
{
    const size_t n = A.rows();
    std::mt19937 g(seed);
    EVec<T> b(n);
    for (size_t i = 0; i < n; ++i) b(i) = rand_val<T>(g);
    auto db = to_dev<T, S>(b);
    auto dx = to_dev<T, S>(EVec<T>::Zero(n));
    ldl.solve(db, dx);
    EVec<T> x = from_dev(dx);
    return (dense * x - b).norm() / (1 + b.norm());
}

template <typename T, typename Idx, template <typename> class S>
static void
test_ldlt(const char *cat, const char *tag)
{
    if constexpr (is_complex_value<T>::value) {
        return;
    } else {
        std::string t = tag;
        const double rtol = std::is_same_v<T, float> ? 5e-3 : 1e-9;
        using Solver = SparseLdlt<T, Idx, S>;

        for (size_t n : {1, 6, 40, 120}) {
            auto r = make_spd<T, Idx>(n, 0.08, 61 + n);
            Csr<T, Idx, S> A(r.csr);
            Solver ldl(Symmetry::SPD);
            ldl.analyze(A);
            bool ok = ldl.factorize(A);
            double res = ok ? solve_residual(ldl, A, r.dense, 5) : 1e30;
            tlog(cat, (t + " ldlt SPD n=" + std::to_string(n)).c_str(),
                ok && res < rtol, res, 0);
        }

        // Refactorization with new values, same pattern.
        {
            auto r = make_spd<T, Idx>(50, 0.08, 71);
            Csr<T, Idx, S> A(r.csr);
            Solver ldl(Symmetry::SPD);
            ldl.analyze(A);
            bool ok1 = ldl.factorize(A);
            double r1 = solve_residual(ldl, A, r.dense, 6);
            // A2 = A + 3 I (diagonal always present)
            EMat<T> D2 = r.dense;
            D2.diagonal().array() += T(3);
            auto h = r.csr;
            for (size_t i = 0; i < h.rows(); ++i)
                for (Idx k = h.row_ptr()[i]; k < h.row_ptr()[i + 1]; ++k)
                    if (static_cast<size_t>(h.col_ind()[k]) == i)
                        h.values_mut()[k] += T(3);
            Csr<T, Idx, S> A2(h);
            bool ok2 = ldl.factorize(A2);
            double r2 = solve_residual(ldl, A2, D2, 7);
            tlog(cat, (t + " ldlt refactorize").c_str(),
                ok1 && ok2 && r1 < rtol && r2 < rtol, std::max(r1, r2), 0);
        }

        // Quasi-definite KKT with the no-pivot symmetric mode.
        {
            const size_t n = 30, m = 12;
            auto r = make_kkt<T, Idx>(n, m, 0.25, 81);
            Csr<T, Idx, S> A(r.csr);
            Solver ldl(Symmetry::Symmetric);
            ldl.analyze(A);
            bool ok = ldl.factorize(A);
            double res = ok ? solve_residual(ldl, A, r.dense, 8) : 1e30;
            tlog(cat, (t + " ldlt KKT symmetric").c_str(),
                ok && res < rtol * 10, res, 0);
            size_t pos, neg, zero;
            ldl.inertia(pos, neg, zero);
            tlog(cat, (t + " ldlt KKT inertia (n+, m-)").c_str(),
                pos == n && neg == m && zero == 0, 0, 0,
                (std::to_string(pos) + "/" + std::to_string(neg) + "/" +
                    std::to_string(zero))
                    .c_str());
        }

        // An indefinite matrix must be rejected in SPD mode.
        {
            auto r = make_kkt<T, Idx>(10, 4, 0.3, 91);
            Csr<T, Idx, S> A(r.csr);
            Solver ldl(Symmetry::SPD);
            ldl.analyze(A);
            bool ok = ldl.factorize(A);
            tlog(cat, (t + " ldlt SPD rejects indefinite").c_str(), !ok, 0, 0);
        }

        // 2-D Laplacian: solve accuracy on a larger, structured system.
        {
            const size_t g = 30, n = g * g;
            CooBuilder<T, Idx> b(n, n);
            EMat<T> L = EMat<T>::Zero(n, n);
            auto id = [&](size_t i, size_t j) { return i * g + j; };
            for (size_t i = 0; i < g; ++i)
                for (size_t j = 0; j < g; ++j) {
                    size_t p = id(i, j);
                    b.add(p, p, T(4));
                    L(p, p) = T(4);
                    auto edge = [&](size_t q) {
                        b.add(p, q, T(-1));
                        L(p, q) = T(-1);
                    };
                    if (i > 0) edge(id(i - 1, j));
                    if (i + 1 < g) edge(id(i + 1, j));
                    if (j > 0) edge(id(i, j - 1));
                    if (j + 1 < g) edge(id(i, j + 1));
                }
            Csr<T, Idx, S> A(b.build());
            Solver ldl(Symmetry::SPD);
            ldl.analyze(A);
            bool ok = ldl.factorize(A);
            double res = ok ? solve_residual(ldl, A, L, 9) : 1e30;
            tlog(cat, (t + " ldlt 2D Laplacian n=900").c_str(),
                ok && res < rtol, res, 0);
        }
    }
}

// The CPU fallback's own options (ordering); not applicable to cuDSS.
template <typename T, typename Idx>
static void
test_ldlt_orderings(const char *cat, const char *tag)
{
    if constexpr (is_complex_value<T>::value) {
        return;
    } else {
        using Mat = Csr<T, Idx, Cpu::HostStorage>;
        using Solver = SparseLdlt<T, Idx, Cpu::HostStorage>;
        const size_t g = 25, n = g * g;
        CooBuilder<T, Idx> b(n, n);
        EMat<T> L = EMat<T>::Zero(n, n);
        for (size_t i = 0; i < g; ++i)
            for (size_t j = 0; j < g; ++j) {
                size_t p = i * g + j;
                b.add(p, p, T(4));
                L(p, p) = T(4);
                auto edge = [&](size_t q) {
                    b.add(p, q, T(-1));
                    L(p, q) = T(-1);
                };
                if (i > 0) edge(p - g);
                if (i + 1 < g) edge(p + g);
                if (j > 0) edge(p - 1);
                if (j + 1 < g) edge(p + 1);
            }
        Mat A(b.build());
        Solver nat(Symmetry::SPD, Ordering::Natural);
        Solver md(Symmetry::SPD, Ordering::MinDegree);
        nat.analyze(A);
        md.analyze(A);
        bool ok = nat.factorize(A) && md.factorize(A);
        double r1 = solve_residual(nat, A, L, 3);
        double r2 = solve_residual(md, A, L, 3);
        const double rtol = std::is_same_v<T, float> ? 5e-3 : 1e-9;
        std::string det = "fill natural=" + std::to_string(nat.factor_nnz()) +
                          " mindeg=" + std::to_string(md.factor_nnz());
        tlog(cat, (std::string(tag) + " ldlt min-degree reduces fill").c_str(),
            ok && r1 < rtol && r2 < rtol && md.factor_nnz() < nat.factor_nnz(),
            std::max(r1, r2), 0, det.c_str());
    }
}

// ─── AMD ordering ────────────────────────────────────────────────────────────
static bool
is_permutation_of(const std::vector<int> &p, size_t n)
{
    if (p.size() != n) return false;
    std::vector<char> seen(n, 0);
    for (int v : p) {
        if (v < 0 || static_cast<size_t>(v) >= n || seen[v]) return false;
        seen[v] = 1;
    }
    return true;
}

static size_t
fill_of(const Csr<double, int32_t> &A, Ordering ord)
{
    SparseLdlt<double, int32_t, Cpu::HostStorage> s(Symmetry::Symmetric, ord);
    s.analyze(A);
    return s.factor_nnz();
}

static Csr<double, int32_t>
grid_laplacian(size_t g)
{
    const size_t n = g * g;
    CooBuilder<double> b(n, n);
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
test_amd()
{
    const char *cat = "SPARSE_AMD";
    // valid permutations on assorted graphs
    bool ok = true;
    for (size_t n : {1, 2, 5, 37, 200}) {
        auto r = make_spd<double, int32_t>(n, 0.05, 900 + n);
        auto p = amd_order<int32_t>(n, r.csr.row_ptr(), r.csr.col_ind());
        ok = ok && is_permutation_of(p, n);
    }
    tlog(cat, "amd returns a valid permutation (random SPD)", ok, 0, 0);
    {
        CooBuilder<double> b(50, 50); // diagonal: no edges
        for (size_t i = 0; i < 50; ++i) b.add(i, i, 1.0);
        auto A = b.build();
        auto p = amd_order<int32_t>(50, A.row_ptr(), A.col_ind());
        tlog(cat, "amd on a diagonal matrix", is_permutation_of(p, 50), 0, 0);
        std::vector<int32_t> none;
        tlog(cat, "amd on n = 0", amd_order<int32_t>(0, A.row_ptr(), A.col_ind()).empty(), 0, 0);
    }
    {   // arrow matrix: one dense row/column that must be ordered last
        const size_t n = 400;
        CooBuilder<double> b(n, n);
        for (size_t i = 0; i < n; ++i) b.add(i, i, 4.0);
        for (size_t i = 1; i < n; ++i) { b.add(0, i, 1.0); b.add(i, 0, 1.0); }
        auto A = b.build();
        auto p = amd_order<int32_t>(n, A.row_ptr(), A.col_ind());
        size_t nat = fill_of(A, Ordering::Natural), amd = fill_of(A, Ordering::MinDegree);
        tlog(cat, "amd orders the dense row of an arrow matrix last",
            is_permutation_of(p, n) && p.back() == 0 && amd < nat, 0, 0,
            (std::to_string(nat) + " -> " + std::to_string(amd)).c_str());
    }
    // fill quality on grids: close to the exact minimum degree, far better
    // than natural
    for (size_t g : {20, 60}) {
        auto A = grid_laplacian(g);
        size_t nat = fill_of(A, Ordering::Natural);
        size_t amd = fill_of(A, Ordering::MinDegree);
        size_t exact = g <= 20 ? fill_of(A, Ordering::ExactMinDegree) : amd;
        std::string det = "natural " + std::to_string(nat) + " amd " +
                          std::to_string(amd) + " exact " + std::to_string(exact);
        tlog(cat, ("amd fill on " + std::to_string(g) + "x" + std::to_string(g) + " grid").c_str(),
            amd < nat / 2 && amd <= exact * 1.35, 0, 0, det.c_str());
    }
    // the factorization with the AMD ordering still solves accurately
    {
        auto A = grid_laplacian(40);
        SparseLdlt<double, int32_t, Cpu::HostStorage> s(Symmetry::SPD, Ordering::MinDegree);
        s.analyze(A);
        bool f = s.factorize(A);
        const size_t n = A.rows();
        tensorET<1, double> b({n}, 1.0), x({n}, 0.0);
        s.solve(b, x);
        tensorET<1, double> r({n}, 0.0);
        spmv(A, x, r);
        double e = 0;
        for (size_t i = 0; i < n; ++i) e = std::max(e, std::abs(r.data[i] - 1.0));
        tlog(cat, "solve with the AMD ordering", f && e < 1e-9, e, 0);
    }
    // timing on a larger grid (CPU only; informational)
    {
        auto A = grid_laplacian(150);
        DECLARE_TIMER;
        TIC;
        auto p = amd_order<int32_t>(A.rows(), A.row_ptr(), A.col_ind());
        double ms = TOC;
        tlog(cat, "amd on a 150x150 grid (22.5k rows)", is_permutation_of(p, A.rows()) && ms < 2000, 0, ms);
    }
}

// ─── multifrontal (supernodal) factorization ─────────────────────────────────
using HostLdl = SparseLdlt<double, int32_t, Cpu::HostStorage>;

static double
solve_err(HostLdl &ldl, const Csr<double, int32_t> &A, unsigned seed, size_t *nfail = nullptr)
{
    const size_t n = A.rows();
    std::mt19937 g(seed);
    tensorET<1, double> b({n}, 0.0), x({n}, 0.0), r({n}, 0.0);
    double bn = 0;
    for (size_t i = 0; i < n; ++i) { b.data[i] = rand_val<double>(g); bn += b.data[i] * b.data[i]; }
    ldl.solve(b, x);
    spmv(A, x, r);
    double e = 0;
    for (size_t i = 0; i < n; ++i) e += (r.data[i] - b.data[i]) * (r.data[i] - b.data[i]);
    (void)nfail;
    return std::sqrt(e) / (1 + std::sqrt(bn));
}

static void
test_multifrontal()
{
    const char *cat = "SPARSE_MF";
    struct Case { std::string name; RandSparse<double, int32_t> m; Symmetry kind; };
    std::vector<Case> cases;
    for (size_t n : {1, 2, 7, 60, 200, 500})
        cases.push_back({"spd n=" + std::to_string(n), make_spd<double, int32_t>(n, 0.06, 500 + n), Symmetry::SPD});
    cases.push_back({"kkt 60+25", make_kkt<double, int32_t>(60, 25, 0.12, 601), Symmetry::Symmetric});
    cases.push_back({"kkt 300+120", make_kkt<double, int32_t>(300, 120, 0.03, 602), Symmetry::Symmetric});
    for (auto &c : cases) {
        auto &A = c.m.csr;
        HostLdl mf(c.kind, Ordering::MinDegree, Factorization::Supernodal);
        HostLdl sp(c.kind, Ordering::MinDegree, Factorization::Simplicial);
        mf.analyze(A);
        sp.analyze(A);
        bool f1 = mf.factorize(A), f2 = sp.factorize(A);
        double e1 = f1 ? solve_err(mf, A, 3) : 1e30, e2 = f2 ? solve_err(sp, A, 3) : 1e30;
        size_t p1, n1, z1, p2, n2, z2;
        mf.inertia(p1, n1, z1);
        sp.inertia(p2, n2, z2);
        bool ok = f1 && f2 && mf.supernodal() && !sp.supernodal() && e1 < 1e-9 &&
                  mf.factor_nnz() == sp.factor_nnz() && p1 == p2 && n1 == n2 && z1 == z2;
        tlog(cat, ("supernodal == simplicial: " + c.name).c_str(), ok, std::max(e1, e2), 0,
            (std::to_string(mf.factor_nnz()) + " nnz(L)").c_str());
    }
    // bipartite matrices with a dense coupling block (transportation-problem normal
    // equations): the leaf supernodes of one side are merged (sibling merge)
    for (size_t k : {5, 40, 300}) {
        std::mt19937 g(900 + k);
        CooBuilder<double> b(2 * k, 2 * k);
        for (size_t i = 0; i < 2 * k; ++i) b.add(i, i, 4.0 * k + rand_val<double>(g) * 0.1);
        for (size_t i = 0; i < k; ++i)
            for (size_t j = 0; j < k; ++j) {
                if ((i * 7 + j * 3 + k) % 5 == 0) continue; // a few structural zeros
                const double v = rand_val<double>(g);
                b.add(i, k + j, v);
                b.add(k + j, i, v);
            }
        auto A = b.build();
        HostLdl mf(Symmetry::SPD, Ordering::MinDegree, Factorization::Supernodal);
        HostLdl sp(Symmetry::SPD, Ordering::MinDegree, Factorization::Simplicial);
        mf.analyze(A);
        sp.analyze(A);
        bool f1 = mf.factorize(A), f2 = sp.factorize(A);
        double e1 = f1 ? solve_err(mf, A, 11) : 1e30, e2 = f2 ? solve_err(sp, A, 11) : 1e30;
        tlog(cat, ("bipartite dense block k=" + std::to_string(k)).c_str(),
            f1 && f2 && e1 < 1e-9 && e2 < 1e-9, std::max(e1, e2), 0);
    }
    // structured matrices and the automatic choice
    for (size_t g : {30, 120}) {
        auto A = grid_laplacian(g);
        HostLdl s(Symmetry::SPD, Ordering::MinDegree, Factorization::Auto);
        s.analyze(A);
        bool f = s.factorize(A);
        double e = f ? solve_err(s, A, 5) : 1e30;
        bool want_mf = s.factor_nnz() > 200000;
        tlog(cat, ("grid " + std::to_string(g) + "x" + std::to_string(g) + " (auto picks " +
                   (s.supernodal() ? "supernodal" : "simplicial") + ")").c_str(),
            f && e < 1e-9 && s.supernodal() == want_mf, e, 0);
    }
    // refactorization with new values, failure detection, regularization
    {
        auto r = make_spd<double, int32_t>(300, 0.03, 700);
        HostLdl s(Symmetry::SPD, Ordering::MinDegree, Factorization::Supernodal);
        s.analyze(r.csr);
        bool f1 = s.factorize(r.csr);
        auto h = r.csr;
        for (size_t i = 0; i < h.rows(); ++i)
            for (int k = h.row_ptr()[i]; k < h.row_ptr()[i + 1]; ++k)
                if (static_cast<size_t>(h.col_ind()[k]) == i) h.values_mut()[k] += 5.0;
        bool f2 = s.factorize(h);
        double e = f2 ? solve_err(s, h, 6) : 1e30;
        tlog(cat, "supernodal refactorize", f1 && f2 && e < 1e-9, e, 0);

        auto k = make_kkt<double, int32_t>(40, 15, 0.2, 701);
        HostLdl bad(Symmetry::SPD, Ordering::MinDegree, Factorization::Supernodal);
        bad.analyze(k.csr);
        tlog(cat, "supernodal SPD mode rejects an indefinite matrix", !bad.factorize(k.csr), 0, 0);
    }
    {   // dynamic pivot regularization: a singular (zero) diagonal entry
        CooBuilder<double> b(3, 3);
        b.add(0, 0, 0.0); b.add(0, 1, 1.0); b.add(1, 0, 1.0); b.add(1, 1, 2.0); b.add(2, 2, 3.0);
        auto A = b.build();
        for (Factorization fac : {Factorization::Supernodal, Factorization::Simplicial}) {
            HostLdl s(Symmetry::Symmetric, Ordering::Natural, fac);
            s.analyze(A);
            bool plain = s.factorize(A); // zero pivot
            s.set_pivot_regularization({-1, 1, 1}, 1e-8);
            bool reg = s.factorize(A);
            tlog(cat, (std::string("pivot regularization ") + (fac == Factorization::Supernodal ? "supernodal" : "simplicial")).c_str(),
                !plain && reg && s.regularized_pivots() == 1, 0, 0);
        }
    }
}

// ─── driver ──────────────────────────────────────────────────────────────────
template <typename T, typename Idx, template <typename> class S>
static void
run_suite(const char *cat, const char *tag)
{
    test_structure<T, Idx, S>(cat, tag);
    test_products<T, Idx, S>(cat, tag);
    test_transpose_spgemm<T, Idx, S>(cat, tag);
    test_ldlt<T, Idx, S>(cat, tag);
    if constexpr (Csr<T, Idx, S>::on_host)
        test_ldlt_orderings<T, Idx>(cat, tag);
}

int
main(int argc, char *argv[])
{
    headless = parse_headless(argc, argv);
    if (headless) printf("--- AXOS sparse test suite ---\n");

    for (int run = 0; run < RUNS; ++run) {
        if (headless) printf("\n═══ RUN %d/%d ═══\n", run + 1, RUNS);
        test_amd();
        test_multifrontal();
        run_suite<double, int32_t, Cpu::HostStorage>("SPARSE_CPU", "d/i32");
        run_suite<float, int32_t, Cpu::HostStorage>("SPARSE_CPU", "f/i32");
        run_suite<double, int64_t, Cpu::HostStorage>("SPARSE_CPU", "d/i64");
        run_suite<std::complex<double>, int32_t, Cpu::HostStorage>(
            "SPARSE_CPU", "z/i32");
#ifdef AXOS_ENABLE_CUDA
        run_suite<double, int32_t, Cuda::CudaStorage>("SPARSE_CUDA", "d/i32");
        run_suite<float, int32_t, Cuda::CudaStorage>("SPARSE_CUDA", "f/i32");
        run_suite<double, int64_t, Cuda::CudaStorage>("SPARSE_CUDA", "d/i64");
        run_suite<std::complex<double>, int32_t, Cuda::CudaStorage>(
            "SPARSE_CUDA", "z/i32");
#endif
    }

    print_summary();
    int fail = 0;
    for (auto &r : results)
        if (!r.pass) fail++;
    return fail > 0 ? 1 : 0;
}
