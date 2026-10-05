// SPDX-License-Identifier: BSD-3-Clause
//
// Tests for the dense tensor layer (docs/TENSOR_SPEC.md §12):
//   Tier 1  storage policies, tensorET construction / copy / move / views /
//           tiling / conversions, element types, compile-time access rules,
//           and with CUDA the host<->device round trips and the GPU pool
//   Tier 2  element-wise expressions (bit-identical to the scalar
//           expression, no heap allocation), matrix products, and the dense
//           linear algebra against Eigen at the §7 accuracy
//
//   make test_tensor_cpu         host only
//   make test_tensor             host + CUDA
//
// Prints one line per check and a SUMMARY; the exit code is the number of
// failed checks (capped at 255).
#ifdef AXOS_ENABLE_CUDA
#include "tensorCuda.h"
#endif
#include "axos.h"
#include <Eigen/Dense>

#include <algorithm>
#include <atomic>
#include <chrono>
#include <cmath>
#include <complex>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <limits>
#include <new>
#include <random>
#include <sstream>
#include <string>
#include <type_traits>
#include <vector>
#ifdef _WIN32
#include <malloc.h>
#endif

using namespace AXOS;

// ---- allocation counting (replaces the global operator new) ---------------

static std::atomic<long> g_news{0}, g_deletes{0};

static void *
counted_alloc(std::size_t n, std::size_t align)
{
    ++g_news;
    void *p = nullptr;
    if (n == 0) n = 1;
#ifdef _WIN32
    p = _aligned_malloc(n, align < sizeof(void *) ? sizeof(void *) : align);
#else
    if (posix_memalign(&p, align < sizeof(void *) ? sizeof(void *) : align, n)) p = nullptr;
#endif
    if (!p) throw std::bad_alloc();
    return p;
}
static void
counted_free(void *p) noexcept
{
    if (!p) return;
    ++g_deletes;
#ifdef _WIN32
    _aligned_free(p);
#else
    std::free(p);
#endif
}
void *operator new(std::size_t n) { return counted_alloc(n, 16); }
void *operator new[](std::size_t n) { return counted_alloc(n, 16); }
void *operator new(std::size_t n, std::align_val_t a) { return counted_alloc(n, std::size_t(a)); }
void *operator new[](std::size_t n, std::align_val_t a) { return counted_alloc(n, std::size_t(a)); }
void operator delete(void *p) noexcept { counted_free(p); }
void operator delete[](void *p) noexcept { counted_free(p); }
void operator delete(void *p, std::size_t) noexcept { counted_free(p); }
void operator delete[](void *p, std::size_t) noexcept { counted_free(p); }
void operator delete(void *p, std::align_val_t) noexcept { counted_free(p); }
void operator delete[](void *p, std::align_val_t) noexcept { counted_free(p); }
void operator delete(void *p, std::size_t, std::align_val_t) noexcept { counted_free(p); }
void operator delete[](void *p, std::size_t, std::align_val_t) noexcept { counted_free(p); }

// ---- tiny test harness ------------------------------------------------------

static int g_pass = 0, g_fail = 0;

static void
check(const char *cat, const std::string &name, bool ok, double err = 0)
{
    (ok ? g_pass : g_fail)++;
    std::printf(" [%s] %-52s %s  err=%.2e\n", cat, name.c_str(), ok ? "PASS" : "FAIL", err);
}

template <class F>
static bool
throws(F &&f)
{
    try {
        f();
    } catch (const std::runtime_error &) {
        return true;
    } catch (...) {
        return false;
    }
    return false;
}

template <class T>
static bool
bits_equal(const T *a, const T *b, size_t n)
{
    return n == 0 || std::memcmp(a, b, n * sizeof(T)) == 0;
}

using EMat = Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>;

static EMat
to_eigen(const tensorET<2, double> &t)
{
    const tensorET<2, double> c = t.clone();
    return Eigen::Map<const EMat>(c.data, Eigen::Index(c.size(0)), Eigen::Index(c.size(1)));
}

static tensorET<2, double>
from_eigen(const EMat &m)
{
    tensorET<2, double> t({size_t(m.rows()), size_t(m.cols())});
    Eigen::Map<EMat>(t.data, m.rows(), m.cols()) = m;
    return t;
}

// Random n x n matrix with condition number `cond` (singular values
// log-spaced in [1, cond], random orthogonal factors).
static EMat
conditioned(size_t n, double cond, unsigned seed)
{
    std::mt19937 g(seed);
    std::normal_distribution<double> nd;
    auto orth = [&]() {
        EMat G(n, n);
        for (Eigen::Index i = 0; i < G.size(); ++i) G.data()[i] = nd(g);
        Eigen::HouseholderQR<EMat> qr(G);
        return EMat(qr.householderQ());
    };
    EMat S = EMat::Zero(n, n);
    for (size_t i = 0; i < n; ++i)
        S(i, i) = n > 1 ? std::pow(cond, double(i) / double(n - 1)) : 1.0;
    return orth() * S * orth();
}

// ---- compile-time rules (TENSOR_SPEC §5.5, §10) ----------------------------

template <class X, class = void> struct writable_index : std::false_type {};
template <class X>
struct writable_index<X, std::void_t<decltype(std::declval<X &>()[size_t(0)] =
                                                  std::declval<typename X::value_type>())>>
    : std::true_type {};
template <class X, class = void> struct writable_value : std::false_type {};
template <class X>
struct writable_value<X, std::void_t<decltype(std::declval<X &>().value(size_t(0)) =
                                                  std::declval<typename X::value_type>())>>
    : std::true_type {};
template <class X, class = void> struct writable_call2 : std::false_type {};
template <class X>
struct writable_call2<X, std::void_t<decltype(std::declval<X &>()(size_t(0), size_t(0)) =
                                                  std::declval<typename X::value_type>())>>
    : std::true_type {};
template <class X, class = void> struct callable3 : std::false_type {};
template <class X>
struct callable3<X, std::void_t<decltype(std::declval<X &>()(0, 0, 0))>> : std::true_type {};
template <class X, class = void> struct readable_index : std::false_type {};
template <class X>
struct readable_index<X, std::void_t<decltype(std::declval<const X &>()[size_t(0)])>>
    : std::true_type {};

using HD = tensorET<1, double>;
using DD = tensorET<1, double, Cuda::CudaStorage<double>>;
using DC = tensorET<1, std::complex<double>, Cuda::CudaStorage<std::complex<double>>>;
using HM = tensorET<2, double>;
using DM = tensorET<2, double, Cuda::CudaStorage<double>>;
static_assert(writable_index<HD>::value && writable_value<HD>::value && writable_call2<HM>::value,
    "host element access must be writable");
static_assert(!writable_index<DD>::value && !writable_index<DC>::value,
    "device operator[] must not be writable");
static_assert(!writable_value<DD>::value && !writable_call2<DM>::value,
    "device value()/operator() must not be writable");
static_assert(readable_index<DD>::value, "device operator[] reads by value");
static_assert(!callable3<HM>::value, "operator() with the wrong index count must not compile");
static_assert(!is_tensor_element_v<bool> && !is_tensor_element_v<unsigned char> &&
                  !is_tensor_element_v<std::string>,
    "other element types are rejected");
static_assert(is_tensor_element_v<half> && is_tensor_element_v<std::complex<float>>, "");
static_assert(std::is_nothrow_move_constructible_v<HM> && std::is_nothrow_move_assignable_v<HM>, "");
static_assert(std::is_same_v<backend_of_t<Cpu::HostStorage<float>>, Cpu::Backend> &&
                  std::is_same_v<backend_of_t<int>, Cpu::Backend> &&
                  std::is_same_v<backend_of_t<Cuda::CudaStorage<float>>, Cuda::Backend>,
    "backend_of_t");
static_assert(is_host_storage_v<Cpu::HostStorage<double>> &&
                  !is_host_storage_v<Cuda::CudaStorage<double>>,
    "is_host_storage");
static_assert(HM::rank == 2 && std::is_same_v<HM::value_type, double> &&
                  std::is_same_v<HM::data_type, double> &&
                  std::is_same_v<HM::backend_type, Cpu::Backend>,
    "member types");

// =============================================================================
// Tier 1: host
// =============================================================================

static void
test_storage()
{
    const char *C = "T1_STORAGE";
    Cpu::HostStorage<double> e;
    check(C, "empty policy", e.data() == nullptr && e.size() == 0 && !e.owns());
    Cpu::HostStorage<double> a(1000);
    check(C, "(n): 64-byte aligned, owning",
        a.data() && reinterpret_cast<std::uintptr_t>(a.data()) % 64 == 0 && a.owns());
    Cpu::HostStorage<double> z(0);
    check(C, "(0) is empty", z.data() == nullptr && z.size() == 0);
    Cpu::HostStorage<int64_t> f(1 << 20, -7);
    bool ok = true;
    for (size_t i = 0; i < f.size(); ++i) ok &= f.data()[i] == -7;
    check(C, "(n, x) fills (large, parallel)", ok);
    Cpu::HostStorage<int64_t> c(f);
    check(C, "copy of owning is deep", c.data() != f.data() && c.data()[123] == -7);
    double raw[4] = {1, 2, 3, 4};
    Cpu::HostStorage<double> b(raw, 4);
    Cpu::HostStorage<double> bc(b);
    check(C, "copy of borrowed aliases", bc.data() == raw && !bc.owns());
    Cpu::HostStorage<double> m(std::move(a));
    check(C, "move leaves source empty", a.data() == nullptr && a.size() == 0 && m.owns());
    double *keep = m.data();
    Cpu::HostStorage<double> same(1000, 3.0);
    m = same;
    check(C, "same-size copy-assign reuses the buffer", m.data() == keep && m.data()[999] == 3.0);
    m = m;
    check(C, "self-assignment is a no-op", m.data() == keep);
    m.allocate(10);
    check(C, "allocate(n)", m.size() == 10 && m.owns());
    const long news = g_news, dels = g_deletes;
    {
        double *p = Cpu::HostStorage<double>::allocate_raw(64);
        Cpu::HostStorage<double> own(p, 64, true);
    }
    check(C, "own == true frees through the policy",
        g_news - news == g_deletes - dels && g_news - news >= 1);
    check(C, "allocation failure throws bad_alloc", [] {
        try {
            Cpu::HostStorage<double> huge(std::numeric_limits<size_t>::max() / 4);
        } catch (const std::bad_alloc &) {
            return true;
        }
        return false;
    }());
}

static void
test_construction()
{
    const char *C = "T1_TENSOR";
    tensorET<1, double> e;
    check(C, "default is empty", e.size() == 0 && e.data == nullptr && e.size(0) == 0);
    tensorET<2, double> a({3, 4});
    check(C, "{m, n} shape and row-major strides",
        a.size() == 12 && a.size(0) == 3 && a.size(1) == 4 && a.stride(0) == 4 && a.stride(1) == 1);
    tensorET<2, double> b({3, 4}, 1.5);
    bool ok = true;
    for (size_t i = 0; i < b.size(); ++i) ok &= b[i] == 1.5;
    check(C, "{m, n}, x fills", ok);
    tensorET<3, float> c(7);
    check(C, "(n) puts n in the last dimension", c.size(0) == 1 && c.size(1) == 1 && c.size(2) == 7);
    tensorET<1, int64_t> d(5, 9);
    check(C, "(n, x)", d.size() == 5 && d[4] == 9);
    size_t dims[2] = {2, 5};
    tensorET<2, double> f(dims), g(dims, 2.0);
    check(C, "array forms", f.size(1) == 5 && g(1, 4) == 2.0);
    double buf[10] = {};
    tensorET<2, double> v(buf, dims);
    v(1, 2) = 4.5;
    check(C, "raw view writes through", v.data == buf && buf[7] == 4.5 && !v.owns_memory());
    size_t n1[1] = {3};
    const tensorET<1, double> csr_view(buf, n1, false); // Csr::values_view form
    check(C, "raw view with own == false", csr_view.size() == 3 && csr_view.data == buf);
    Cpu::HostStorage<double> s(10, 1.0);
    tensorET<2, double> adopt(std::move(s), dims);
    check(C, "adopts a storage", adopt.data && adopt(1, 4) == 1.0 && s.data() == nullptr);
    size_t dd[2] = {4, 5}, st[2] = {1, 1}, en[2] = {2, 3};
    tensorET<2, double> dom(dd, st, en);
    const auto D = dom.getIndex(), D0 = a.getIndex();
    check(C, "domain constructor / getIndex",
        D.extent(1) == 5 && D.start(0) == 1 && D.end(1) == 3 && D0.start(1) == 0 && D0.end(1) == 3);
    check(C, "wrong dimension count throws", throws([] { tensorET<2, double> x({3}); }));
    check(C, "size(DIM) throws", throws([&] { (void)a.size(2); }));
    check(C, "stride(-1) throws", throws([&] { (void)a.stride(-1); }));
    check(C, "bad reshape throws", throws([&] { (void)a.reshape<1>({5}); }));
}

static void
test_copy_move_views()
{
    const char *C = "T1_SEMANTICS";
    tensorET<2, double> a({3, 4});
    for (size_t i = 0; i < 12; ++i) a[i] = double(i);
    tensorET<2, double> c = a;
    c(0, 0) = 100;
    check(C, "copying an owning tensor does not alias", a(0, 0) == 0 && c.data != a.data);
    auto v = a.slice({1, 1}, {2, 2});
    auto vc = v;
    vc(0, 0) = -1;
    check(C, "copying a view aliases", a(1, 1) == -1 && vc.data == v.data);
    a(1, 1) = 5;
    tensorET<2, double> m(std::move(c));
    check(C, "move: source empty, destination has the data",
        c.size() == 0 && c.data == nullptr && c.size(0) == 0 && m(0, 0) == 100);
    tensorET<2, double> m2;
    m2 = std::move(m);
    check(C, "move-assign", m.data == nullptr && m2(0, 0) == 100);
    double *keep = m2.data;
    m2 = a;
    check(C, "copy-assign into same size keeps the buffer", m2.data == keep && m2(2, 3) == 11);
    check(C, "slice strides are the parent's",
        v.stride(0) == 4 && v.stride(1) == 1 && v.size(0) == 2 && v.data == a.data + 5 &&
            !v.is_contiguous());
    check(C, "slice element access", v(1, 1) == a(2, 2) && v.value(0) == a[5]);
    auto r = a.reshape<3>({2, 3, 2});
    r(1, 2, 1) = 77;
    check(C, "reshape is a view", a[11] == 77 && r.stride(0) == 6);
    auto fl = a.flatten();
    check(C, "flatten", fl.size() == 12 && fl.data == a.data);
    check(C, "reshape of a slice throws", throws([&] { (void)v.reshape<1>({4}); }));
    auto cl = v.clone();
    check(C, "clone of a view is contiguous and owning",
        cl.is_contiguous() && cl.owns_memory() && cl(1, 1) == a(2, 2));
    tensorET<2, double> big({4, 6}, 0.0);
    auto bv = big.slice({1, 2}, {2, 3});
    bv.fill(9.0);
    double sum = 0;
    for (size_t i = 0; i < big.size(); ++i) sum += big[i];
    check(C, "fill on a view touches only the view", sum == 54.0 && big(1, 2) == 9 && big(3, 5) == 0);
    tensorET<2, double> into({2, 3});
    Cpu::Backend::copy_data(into, bv);
    check(C, "copy_data from a view", into(1, 2) == 9.0);
    std::vector<float> host(6);
    Cpu::Backend::copy_to_host(bv, host.data());
    check(C, "copy_to_host converts (double -> float)", host[5] == 9.0f);
    const float src[6] = {1, 2, 3, 4, 5, 6};
    Cpu::Backend::copy_from_host(bv, src);
    check(C, "copy_from_host into a view", big(1, 2) == 1 && big(2, 4) == 6 && big(1, 1) == 0);
    check(C, "read_element", Cpu::Backend::read_element(big, 1 * 6 + 2) == 1.0);
}

static void
test_tiles()
{
    const char *C = "T1_TILES";
    // 5 x 7 matrix inside an 8 x 10 parent (to catch out-of-range writes)
    tensorET<2, double> parent({8, 10}, -5.0);
    auto A = parent.slice({0, 0}, {5, 7});
    for (size_t i = 0; i < 5; ++i)
        for (size_t j = 0; j < 7; ++j) A(i, j) = double(10 * i + j);
    std::vector<double> buf(16, 99.0);
    auto t = A.blockCopy(1, 1, 4, buf.data());
    // tile (1, 1): rows [4, 8), cols [4, 8) -> only row 4, cols 4..6 exist
    bool ok = t.data == buf.data() && t(0, 0) == 44 && t(0, 2) == 46 && t(0, 3) == 0 &&
              t(1, 0) == 0 && t(3, 3) == 0;
    check(C, "blockCopy zero-pads right and bottom", ok);
    auto tt = A.blockCopy(0, 1, 4, buf.data(), true);
    // transpose of tile (1, 0): rows [4, 8) x cols [0, 4) -> buf[r][c] = A[4 + c][r]
    ok = tt(0, 0) == 40 && tt(3, 0) == 43 && tt(0, 1) == 0;
    check(C, "blockCopy transpose", ok);
    auto own = A.blockCopy(0, 1, 4);
    check(C, "owning blockCopy", own.owns_memory() && own(0, 0) == 4 && own(4 - 1, 2) == 36 && own(0, 3) == 0);
    tensorET<2, double> tile({4, 4}, 1.0);
    A.blockWriteBack(1, 1, 4, tile);
    A.blockAddWriteBack(1, 1, 4, tile);
    ok = A(4, 4) == 2 && A(4, 6) == 2 && A(3, 3) == 33;
    for (size_t i = 0; i < 8; ++i)
        for (size_t j = 0; j < 10; ++j)
            if (i >= 5 || j >= 7) ok &= parent(i, j) == -5.0;
    check(C, "write-back touches only in-range elements", ok);
    auto bv = A.blockView(0, 0, 2);
    check(C, "blockView", bv.size(0) == 2 && bv(1, 1) == 11 && bv.data == A.data);
}

static void
test_types_and_random()
{
    const char *C = "T1_TYPES";
    tensorET<1, std::complex<double>> z({3}, {1.0, -2.0});
    tensorET<1, std::complex<float>> zf(4, {0.5f, 0.25f});
    tensorET<1, int32_t> i32({5}, -3);
    check(C, "complex / integer element types",
        z[2].imag() == -2.0 && zf[3].real() == 0.5f && i32[4] == -3);
    tensorET<1, half> h({4}, half(1.5f));
    check(C, "half element type", float(h[3]) == 1.5f);
    const float probe[] = {1.0f, 0.1f, 65504.0f, -2.5f, 6.0e-8f, 1e-9f, 70000.0f};
    const float expect[] = {1.0f, 0.0999755859375f, 65504.0f, -2.5f, 5.9604644775390625e-8f, 0.0f,
        std::numeric_limits<float>::infinity()};
    bool ok = true;
    for (int i = 0; i < 7; ++i) ok &= float(half_fallback(probe[i])) == expect[i];
    ok &= std::isnan(float(half_fallback(std::nanf(""))));
    ok &= float(half_fallback(1.5f) * half_fallback(2.0f) + 1) == 4.0f;
    check(C, "half_fallback rounding and arithmetic", ok);
    auto u1 = tensorET<2, double>::uniform({100, 50}, -2.0, 3.0, 42);
    auto u2 = tensorET<2, double>::uniform({100, 50}, -2.0, 3.0, 42);
    double lo = 1e9, hi = -1e9;
    for (size_t i = 0; i < u1.size(); ++i) lo = std::min(lo, u1[i]), hi = std::max(hi, u1[i]);
    check(C, "uniform: seeded is deterministic, in range",
        bits_equal(u1.data, u2.data, u1.size()) && lo >= -2.0 && hi < 3.0);
    auto gs = tensorET<1, double>::gaussian({200000}, 1.0, 2.0, 7);
    double mean = 0, var = 0;
    for (size_t i = 0; i < gs.size(); ++i) mean += gs[i];
    mean /= double(gs.size());
    for (size_t i = 0; i < gs.size(); ++i) var += (gs[i] - mean) * (gs[i] - mean);
    var /= double(gs.size());
    check(C, "gaussian moments", std::abs(mean - 1.0) < 0.02 && std::abs(std::sqrt(var) - 2.0) < 0.02,
        std::abs(mean - 1.0));
    auto ui = tensorET<1, int64_t>::uniform({1000}, int64_t(-3), int64_t(3), 1);
    ok = true;
    for (size_t i = 0; i < ui.size(); ++i) ok &= ui[i] >= -3 && ui[i] <= 3;
    check(C, "integer uniform is inclusive", ok);
    auto uz = tensorET<1, std::complex<double>>::uniform({64}, {0, 0}, {1, 0}, 3);
    check(C, "complex uniform", uz[5].imag() >= 0 && uz[5].imag() < 1 && uz[5].real() >= 0);
    std::ostringstream os;
    tensorET<2, double> p({2, 3});
    for (int i = 0; i < 6; ++i) p[size_t(i)] = i + 1;
    os << p;
    check(C, "operator<< format", os.str() == "[\n  [1  2  3]\n  [4  5  6]\n]");
    std::ostringstream os3;
    tensorET<3, int32_t> q({2, 1, 2});
    for (int i = 0; i < 4; ++i) q[size_t(i)] = i;
    os3 << q;
    check(C, "operator<< rank 3",
        os3.str() == "[\n  [\n    [0  1]\n  ]\n  [\n    [2  3]\n  ]\n]");
}

// =============================================================================
// Tier 1: CUDA
// =============================================================================

#ifdef AXOS_ENABLE_CUDA
template <class T>
static bool
round_trip(size_t n, unsigned seed)
{
    tensorET<1, T> h({n});
    std::mt19937 g(seed);
    for (size_t i = 0; i < n; ++i) {
        if constexpr (is_complex_v<T>)
            h[i] = T(typename T::value_type(g() % 1000) / 7, typename T::value_type(g() % 97) / 3);
        else
            h[i] = T(g() % 100000) / T(3);
    }
    tensorETCuda<1, T> d(h);
    tensorET<1, T> back(d);
    return back.size() == n && bits_equal(h.data, back.data, n);
}

static void
test_cuda()
{
    const char *C = "T1_CUDA";
    bool ok = true;
    for (size_t n : {size_t(0), size_t(1), size_t(1000)}) {
        ok &= round_trip<double>(n, 1) && round_trip<float>(n, 2);
        ok &= round_trip<int32_t>(n, 3) && round_trip<int64_t>(n, 4);
        ok &= round_trip<std::complex<double>>(n, 5) && round_trip<std::complex<float>>(n, 6);
    }
    check(C, "host->device->host bit-identical (6 types, n = 0/1/1000)", ok);
    tensorETCuda<1, double> z({1000}, 0.0), f({5000}, 2.5);
    tensorET<1, double> hz(z), hf(f);
    ok = true;
    for (size_t i = 0; i < 1000; ++i) ok &= hz[i] == 0.0;
    for (size_t i = 0; i < 5000; ++i) ok &= hf[i] == 2.5;
    check(C, "device fill (zero and non-zero)", ok);
    tensorETCuda<1, int32_t> m1({777}, -1);
    tensorET<1, int32_t> hm1(m1);
    check(C, "device fill with a uniform byte pattern", hm1[776] == -1 && hm1[0] == -1);
    check(C, "device read_element / const operator[]", f[4999] == 2.5 && f.value(3) == 2.5);
    tensorETCuda<1, double> fc = f;
    check(C, "device copy is deep", fc.data != f.data && tensorET<1, double>(fc)[17] == 2.5);
    size_t n5[1] = {5};
    tensorETCuda<1, double> view(f.data + 10, n5);
    tensorETCuda<1, double> vcopy = view;
    check(C, "device view copy aliases", vcopy.data == f.data + 10 && !vcopy.owns_memory());
    auto H = tensorET<2, double>::uniform({6, 9}, -1.0, 1.0, 11);
    tensorETCuda<2, double> DM2(H);
    auto dsl = DM2.slice({1, 2}, {3, 4});
    tensorET<2, double> hsl(dsl), ref = H.slice({1, 2}, {3, 4}).clone();
    check(C, "download of a device slice (2-D copy)", bits_equal(hsl.data, ref.data, 12));
    tensorETCuda<2, double> DM3({3, 4}, 0.0);
    Cuda::Backend::copy_data(DM3, dsl);
    check(C, "device copy_data from a slice", bits_equal(tensorET<2, double>(DM3).data, ref.data, 12));
    tensorET<1, double> reuse({5000});
    double *keep = reuse.data;
    reuse = f;
    check(C, "cross-storage assign reuses a same-size buffer", reuse.data == keep && reuse[3] == 2.5);
    tensorETCuda<2, double> dview = DM2.slice({0, 0}, {2, 9});
    dview.fill(7.0);
    tensorET<2, double> after(DM2);
    check(C, "device fill on a view", after(1, 8) == 7.0 && after(2, 0) == H(2, 0));

    auto &pool = GPUMemoryPool::get();
    pool.clear();
    const auto s0 = pool.stats();
    void *p = pool.allocate(1 << 20);
    check(C, "pool: 256-byte aligned", reinterpret_cast<std::uintptr_t>(p) % 256 == 0);
    pool.deallocate(p, 1 << 20);
    void *q = pool.allocate(1 << 20);
    const auto s1 = pool.stats();
    check(C, "pool reuses a freed block (no new cudaMalloc)",
        q == p && s1.device_mallocs == s0.device_mallocs + 1 && s1.reuses == s0.reuses + 1);
    pool.deallocate(q, 1 << 20);
    check(C, "pool: allocate(0) is nullptr", pool.allocate(0) == nullptr);
    const size_t held = pool.stats().bytes_in_use;
    pool.set_limit(held + (size_t(5) << 19)); // room for 2.5 MiB, 1 MiB cached
    void *r = nullptr;
    const bool recovered = !throws([&] { r = pool.allocate(size_t(2) << 20); });
    check(C, "pool recovers from out-of-memory by releasing the cache",
        recovered && r && pool.stats().oom_recoveries == s1.oom_recoveries + 1);
    std::string msg;
    try {
        (void)pool.allocate(size_t(100) << 20);
    } catch (const std::runtime_error &e) {
        msg = e.what();
    }
    check(C, "pool throws with sizes in the message", msg.find("MB") != std::string::npos);
    pool.deallocate(r, size_t(2) << 20);
    pool.set_limit(0);
#ifdef CUSPARSE_WITH
    check(C, "cuSPARSE handle is created once per thread",
        pool.get_cusparse() != nullptr && pool.get_cusparse() == pool.get_cusparse());
#endif
#ifdef CUDSS_WITH
    check(C, "cuDSS handle is created once per thread", pool.get_cudss() == pool.get_cudss());
#endif
    auto Hb = tensorET<2, double>::uniform({5, 7}, -1.0, 1.0, 13);
    tensorETCuda<2, double> Db(Hb);
    auto dt = blockCopyCuda(Db, 1, 1, 4, cudaStream_t(0));
    cudaStreamSynchronize(0);
    tensorET<2, double> gotd(dt), want = Hb.blockCopy(1, 1, 4);
    check(C, "blockCopyCuda matches blockCopy", bits_equal(gotd.data, want.data, 16));
    tensorETCuda<2, double> ones(tensorET<2, double>({4, 4}, 1.0));
    blockWriteBackCuda(Db, 1, 1, 4, ones, cudaStream_t(0));
    cudaStreamSynchronize(0);
    tensorET<2, double> wb(Db);
    check(C, "blockWriteBackCuda writes only in range", wb(4, 4) == 1 && wb(4, 6) == 1 && wb(3, 3) == Hb(3, 3));
}
#endif

// =============================================================================
// Tier 2: element-wise expressions
// =============================================================================

template <class T>
static void
test_elementwise(const char *C, size_t n)
{
    auto A = tensorET<1, T>::uniform({n}, T(0.1), T(2.0), 1);
    auto B = tensorET<1, T>::uniform({n}, T(0.5), T(1.5), 2);
    tensorET<1, T> R({n});
    const T *a = A.data, *b = B.data;
    std::vector<T> ref(n);
    auto same = [&](const std::string &name, auto scalar) {
        for (size_t i = 0; i < n; ++i) ref[i] = scalar(a[i], b[i]);
        check(C, name + " (n=" + std::to_string(n) + ")", bits_equal(R.data, ref.data(), n));
    };
    R = A + B;
    same("A + B", [](T x, T y) { return x + y; });
    R = A - B;
    same("A - B", [](T x, T y) { return x - y; });
    R = A * B;
    same("A * B", [](T x, T y) { return x * y; });
    R = A / B;
    same("A / B", [](T x, T y) { return x / y; });
    R = A + B * T(2);
    same("A + B * 2", [](T x, T y) { return x + y * T(2); });
    R = T(3) - A / T(4) + T(1) / B;
    same("3 - A / 4 + 1 / B", [](T x, T y) { return T(3) - x / T(4) + T(1) / y; });
    R = -A * T(2);
    same("-A * 2", [](T x, T) { return -x * T(2); });
    R = sqrt(A);
    same("sqrt", [](T x, T) { return std::sqrt(x); });
    R = fabs(A - B);
    same("fabs", [](T x, T y) { return std::abs(x - y); });
    R = exp(sin(A) + cos(A)) * T(2) + B;
    same("exp(sin(A) + cos(A)) * 2 + B", [](T x, T y) { return std::exp(std::sin(x) + std::cos(x)) * T(2) + y; });
    R = Log(A) + tan(B) - sinh(A) * cosh(B) / tanh(A);
    same("Log tan sinh cosh tanh", [](T x, T y) {
        return std::log(x) + std::tan(y) - std::sinh(x) * std::cosh(y) / std::tanh(x);
    });
}

static void
test_expressions()
{
    const char *C = "T2_EXPR";
    test_elementwise<double>(C, 37);
    test_elementwise<double>(C, 1000003);
    test_elementwise<float>(C, 1000003);
    const size_t n = 1 << 21; // above the streaming-store threshold
    auto A = tensorET<1, double>::uniform({n}, -1.0, 1.0, 5);
    auto B = tensorET<1, double>::uniform({n}, -1.0, 1.0, 6);
    tensorET<1, double> Cc({n});
    tensorET<1, double> small_a({100}, 1.0), small_c({100});
    Cc = A * B + A; // warm-up (thread pool)
    const long before = g_news;
    Cc = exp(sin(A) + cos(A)) * 2.0 + B; // parallel path
    Cc = A + B * 2.0;                    // parallel, streaming stores
    small_c = sqrt(small_a) * 3.0 - small_a; // serial path
    check(C, "expression chains allocate nothing", g_news == before, double(g_news - before));
    double err = 0;
    Cc = A * B + A;
    for (size_t i = 0; i < n; i += 101)
        err = std::max(err, std::abs(Cc[i] - (A[i] * B[i] + A[i])));
    check(C, "A * B + A (within 1 ulp, FMA contraction allowed)", err <= 4.5e-16, err);
    tensorET<1, double> Al = A.clone();
    Al = Al + B;
    bool ok = true;
    for (size_t i = 0; i < n; i += 997) ok &= Al[i] == A[i] + B[i];
    check(C, "A = A + B (aliasing)", ok);
    tensorET<2, double> M = tensorET<2, double>::uniform({6, 8}, -1.0, 1.0, 9);
    tensorET<2, double> out({6, 8}, 0.0);
    auto sv = out.slice({1, 2}, {3, 4});
    sv = M.slice({0, 0}, {3, 4}) * 2.0 + M.slice({2, 3}, {3, 4});
    ok = out(0, 0) == 0.0 && out(1, 6) == 0.0;
    for (size_t i = 0; i < 3; ++i)
        for (size_t j = 0; j < 4; ++j) ok &= out(1 + i, 2 + j) == M(i, j) * 2.0 + M(2 + i, 3 + j);
    check(C, "expressions over strided views", ok);
    tensorET<2, double> made = M * M - 1.0;
    check(C, "tensor constructed from an expression", made.size(0) == 6 && made(5, 7) == M(5, 7) * M(5, 7) - 1.0);
    tensorET<2, double> empty;
    empty = M + M;
    check(C, "assigning to an empty tensor allocates", empty.size() == 48 && empty(1, 1) == 2 * M(1, 1));
    tensorET<2, double> wrong({3, 3});
    check(C, "shape mismatch throws", throws([&] { wrong = M + M; }) &&
                                          throws([&] { (void)(M + wrong); }));
    tensorET<1, int64_t> ia({10}, 7), ib({10}, 3);
    tensorET<1, int64_t> ic = ia * ib - ia / ib;
    tensorET<1, std::complex<double>> za({4}, {1.0, 2.0});
    tensorET<1, std::complex<double>> zc = za * za + std::complex<double>(0, 1);
    check(C, "integer and complex expressions", ic[9] == 19 && zc[2] == std::complex<double>(-3, 5));
}

// =============================================================================
// Tier 2: products
// =============================================================================

static double
max_abs_diff(const EMat &a, const EMat &b)
{
    return a.rows() == b.rows() && a.cols() == b.cols() ? (a - b).cwiseAbs().maxCoeff() : 1e300;
}

static void
test_products()
{
    const char *C = "T2_PRODUCTS";
    for (auto s : {std::array<size_t, 3>{1, 1, 1}, {7, 5, 3}, {64, 64, 64}, {130, 257, 71},
             {300, 200, 513}}) {
        const size_t m = s[0], k = s[1], n = s[2];
        auto A = tensorET<2, double>::uniform({m, k}, -1.0, 1.0, 1);
        auto B = tensorET<2, double>::uniform({k, n}, -1.0, 1.0, 2);
        const EMat ref = to_eigen(A) * to_eigen(B);
        const std::string tag = std::to_string(m) + "x" + std::to_string(k) + "x" + std::to_string(n);
        tensorET<2, double> P = matMul(A, B);
        const double e1 = max_abs_diff(to_eigen(P), ref);
        check(C, "matMul " + tag, e1 < 1e-12 * double(k + 1), e1);
        auto lazy = matMul(A, B);
        const size_t probe = (m * n) / 2;
        check(C, "matMul value(i) without evaluation " + tag,
            std::abs(lazy.value(probe) - ref(Eigen::Index(probe / n), Eigen::Index(probe % n))) < 1e-12 * double(k + 1));
        tensorET<2, double> T2 = matMulTile(A, B);
        check(C, "matMulTile " + tag, max_abs_diff(to_eigen(T2), ref) < 1e-12 * double(k + 1));
        tensorET<2, double> Cc = matMul(A, B) * 2.0 - T2;
        check(C, "matMul inside an expression " + tag, max_abs_diff(to_eigen(Cc), ref) < 1e-12 * double(k + 1));
    }
    auto A = tensorET<2, double>::uniform({90, 70}, -1.0, 1.0, 3);
    auto B = tensorET<2, double>::uniform({70, 50}, -1.0, 1.0, 4);
    auto bias = tensorET<2, double>::uniform({90, 50}, -1.0, 1.0, 5);
    const EMat eA = to_eigen(A), eB = to_eigen(B), eBias = to_eigen(bias);
    tensorET<2, double> Cb;
    matMulBlocked(A, B, Cb, 64, 0.5, &bias, -2.0);
    check(C, "matMulBlocked alpha, bias, beta", max_abs_diff(to_eigen(Cb), 0.5 * eA * eB - 2.0 * eBias) < 1e-12);
    const EMat before = to_eigen(Cb);
    matMulBlocked(A, B, Cb, 64, 1.0, nullptr, 1.0, false, false, true);
    check(C, "matMulBlocked accumulate", max_abs_diff(to_eigen(Cb), before + eA * eB) < 1e-12);
    tensorET<2, double> At = transpose(A), Bt = transpose(B), Ct;
    matMulBlocked(At, Bt, Ct, 32, 1.0, nullptr, 1.0, true, true);
    check(C, "matMulBlocked transA, transB", max_abs_diff(to_eigen(Ct), eA * eB) < 1e-12);
    check(C, "transpose", max_abs_diff(to_eigen(At), eA.transpose()) == 0.0);
    tensorET<2, double> sq = tensorET<2, double>::uniform({40, 40}, -1.0, 1.0, 6);
    const EMat esq = to_eigen(sq);
    sq = matMul(sq, sq); // aliasing: evaluated through the cache
    check(C, "matMul assigned to its own operand", max_abs_diff(to_eigen(sq), esq * esq) < 1e-12);
    auto big = tensorET<2, double>::uniform({50, 60}, -1.0, 1.0, 7);
    auto va = big.slice({5, 3}, {20, 30}), vb = big.slice({10, 20}, {30, 25});
    tensorET<2, double> pv = matMul(va, vb);
    check(C, "products of strided views", max_abs_diff(to_eigen(pv), to_eigen(va) * to_eigen(vb)) < 1e-12);
    auto x = tensorET<1, double>::uniform({70}, -1.0, 1.0, 8);
    auto y = tensorET<1, double>::uniform({90}, -1.0, 1.0, 9);
    const Eigen::VectorXd ex = Eigen::Map<const Eigen::VectorXd>(x.data, 70);
    const Eigen::VectorXd ey = Eigen::Map<const Eigen::VectorXd>(y.data, 90);
    tensorET<1, double> mv = matMulTile(A, x);
    check(C, "matMulTile with a vector (GEMV)", (Eigen::Map<Eigen::VectorXd>(mv.data, 90) - eA * ex).cwiseAbs().maxCoeff() < 1e-13);
    tensorET<1, double> yy = y.clone();
    gemv(A, x, yy, 2.0, 0.5);
    check(C, "gemv alpha, beta", (Eigen::Map<Eigen::VectorXd>(yy.data, 90) - (2.0 * eA * ex + 0.5 * ey)).cwiseAbs().maxCoeff() < 1e-13);
    tensorET<1, double> tv = matVec(A, y, true);
    check(C, "gemv transposed", (Eigen::Map<Eigen::VectorXd>(tv.data, 70) - eA.transpose() * ey).cwiseAbs().maxCoeff() < 1e-13);
    auto tall = tensorET<2, double>::uniform({20000, 3}, -1.0, 1.0, 10);
    auto tx = tensorET<1, double>::uniform({20000}, -1.0, 1.0, 11);
    tensorET<1, double> ty = matVec(tall, tx, true);
    double tref[3] = {0, 0, 0};
    for (size_t i = 0; i < 20000; ++i)
        for (size_t j = 0; j < 3; ++j) tref[j] += tall(i, j) * tx[i];
    check(C, "gemv transposed, tall and thin", std::abs(ty[2] - tref[2]) < 1e-11 && std::abs(ty[0] - tref[0]) < 1e-11);
    check(C, "dot", std::abs(dot(x, x) - ex.squaredNorm()) < 1e-12);
    auto nd_a = tensorET<3, double>::uniform({4, 6, 5}, -1.0, 1.0, 12);
    auto nd_b = tensorET<3, double>::uniform({4, 5, 7}, -1.0, 1.0, 13);
    tensorET<3, double> nd = matMulND(nd_a, nd_b);
    bool ok = nd.size(0) == 4 && nd.size(1) == 6 && nd.size(2) == 7;
    for (size_t bt = 0; bt < 4; ++bt) {
        tensorET<2, double> a2 = extractSlice(nd_a, bt), b2 = extractSlice(nd_b, bt), c2 = extractSlice(nd, bt);
        ok &= max_abs_diff(to_eigen(c2), to_eigen(a2) * to_eigen(b2)) < 1e-13;
    }
    check(C, "matMulND / extractSlice", ok);
    tensorET<2, double> put({6, 7}, 3.0);
    insertSlice(nd, 2, put);
    size_t idx[3] = {2, 5, 6};
    check(C, "insertSlice / getValue", getValue(nd, idx) == 3.0);
    setValue(nd, idx, -1.0);
    check(C, "setValue", nd(2, 5, 6) == -1.0);
    for (size_t n : {size_t(512), size_t(1024)}) {
        auto Fa = tensorET<2, float>::uniform({n, n}, -1.0f, 1.0f, 14);
        auto Fb = tensorET<2, float>::uniform({n, n}, -1.0f, 1.0f, 15);
        tensorET<2, float> Fc = matMul(Fa, Fb);
        Eigen::MatrixXf ea = Eigen::Map<Eigen::Matrix<float, -1, -1, Eigen::RowMajor>>(Fa.data, n, n);
        Eigen::MatrixXf eb = Eigen::Map<Eigen::Matrix<float, -1, -1, Eigen::RowMajor>>(Fb.data, n, n);
        Eigen::MatrixXf ec = ea * eb;
        float e = 0;
        for (size_t i = 0; i < n; i += 7)
            for (size_t j = 0; j < n; j += 5) e = std::max(e, std::abs(Fc(i, j) - ec(Eigen::Index(i), Eigen::Index(j))));
        check(C, "float GEMM n=" + std::to_string(n), e < 1e-6f * float(n), e);
    }
    auto ia = tensorET<2, int64_t>::uniform({9, 11}, int64_t(-5), int64_t(5), 16);
    auto ib = tensorET<2, int64_t>::uniform({11, 4}, int64_t(-5), int64_t(5), 17);
    tensorET<2, int64_t> ic = matMulTile(ia, ib);
    int64_t s = 0;
    for (size_t p = 0; p < 11; ++p) s += ia(8, p) * ib(p, 3);
    check(C, "integer GEMM (generic kernel)", ic(8, 3) == s);
}

// =============================================================================
// Tier 2: linear algebra
// =============================================================================

static void
test_linear_algebra()
{
    const char *C = "T2_LINALG";
    for (size_t n : {size_t(1), size_t(2), size_t(7), size_t(64), size_t(211), size_t(600), size_t(2000)}) {
        const std::string tag = " n=" + std::to_string(n);
        const double tol = 1e-12 * double(n);
        const EMat eA = conditioned(n, 1e3, unsigned(n));
        const tensorET<2, double> A = from_eigen(eA);
        const double na = eA.norm();
        auto lu = luDcmpPivoted(A);
        const EMat L = to_eigen(lu.first.first), U = to_eigen(lu.first.second), P = to_eigen(lu.second);
        const double r_lu = (P * eA - L * U).norm() / na;
        bool shape = L.isLowerTriangular() && U.isUpperTriangular() && (L.diagonal().array() == 1.0).all();
        check(C, "luDcmpPivoted P A = L U" + tag, r_lu < tol && shape, r_lu);
        const Eigen::VectorXd eb = Eigen::VectorXd::LinSpaced(Eigen::Index(n), -1.0, 2.0);
        tensorET<1, double> b({n});
        Eigen::Map<Eigen::VectorXd>(b.data, Eigen::Index(n)) = eb;
        tensorET<1, double> x = luSolve(lu.first.first, lu.first.second, lu.second, b);
        const double r_x = (eA * Eigen::Map<Eigen::VectorXd>(x.data, Eigen::Index(n)) - eb).norm() /
                           (na * Eigen::Map<Eigen::VectorXd>(x.data, Eigen::Index(n)).norm() + eb.norm());
        check(C, "luSolve (vector)" + tag, r_x < tol, r_x);
        const tensorET<2, double> Pcopy = lu.second.clone(); // not remembered: P is scanned
        tensorET<1, double> x2 = luSolve(lu.first.first, lu.first.second, Pcopy, b);
        check(C, "luSolve with a copied P" + tag, bits_equal(x.data, x2.data, n));
        if (n > 600) continue;
        const EMat eB = EMat::Random(Eigen::Index(n), 5);
        tensorET<2, double> X = luSolve(lu.first.first, lu.first.second, lu.second, from_eigen(eB));
        const double r_X = (eA * to_eigen(X) - eB).norm() / (na * to_eigen(X).norm() + eB.norm());
        check(C, "luSolve (matrix)" + tag, r_X < tol, r_X);
        const tensorET<2, double> Ai = inverse(A);
        const double r_inv = (eA * to_eigen(Ai) - EMat::Identity(Eigen::Index(n), Eigen::Index(n))).norm() /
                             (na * to_eigen(Ai).norm());
        check(C, "inverse" + tag, r_inv < tol, r_inv);
        tensorET<2, double> Q = A.clone();
        const tensorET<2, double> R = qrDecompositionTile(Q);
        const EMat eQ = to_eigen(Q), eR = to_eigen(R);
        const double r_qr = (eQ * eR - eA).norm() / na;
        const double r_or = (eQ.transpose() * eQ - EMat::Identity(Eigen::Index(n), Eigen::Index(n))).norm();
        check(C, "qrDecompositionTile A = Q R, Q^T Q = I" + tag,
            r_qr < tol && r_or < tol && eR.isUpperTriangular() && (eR.diagonal().array() >= 0).all(),
            std::max(r_qr, r_or));
        const EMat eU = to_eigen(lu.first.second);
        const tensorET<2, double> Ui = inverse_backs(lu.first.second, 4);
        const double r_ui = (eU * to_eigen(Ui) - EMat::Identity(Eigen::Index(n), Eigen::Index(n))).norm() /
                            (eU.norm() * to_eigen(Ui).norm());
        check(C, "inverse_backs" + tag, r_ui < tol && to_eigen(Ui).isUpperTriangular(), r_ui);
        auto nopiv = luDcmp(from_eigen(eA + double(n) * 10.0 * EMat::Identity(Eigen::Index(n), Eigen::Index(n))));
        const double r_np = (to_eigen(nopiv.first) * to_eigen(nopiv.second) - (eA + double(n) * 10.0 * EMat::Identity(Eigen::Index(n), Eigen::Index(n)))).norm() / (na + double(n) * 10.0);
        check(C, "luDcmp (no pivoting)" + tag, r_np < tol, r_np);
    }
    check(C, "singular matrix throws", throws([] {
        tensorET<2, double> S({3, 3}, 1.0);
        (void)luDcmpPivoted(S);
    }));
    check(C, "luDcmp tiny pivot throws", throws([] {
        tensorET<2, double> S({2, 2}, 0.0);
        S(0, 1) = S(1, 0) = 1.0;
        (void)luDcmp(S);
    }));
    check(C, "non-square throws", throws([] { (void)inverse(tensorET<2, double>({2, 3}, 1.0)); }));

    // Gauss-Jordan
    const EMat G = conditioned(40, 10, 99);
    tensorET<2, double> rref = gaussJordanElimination(from_eigen(G));
    check(C, "gaussJordanElimination of a regular matrix is I",
        max_abs_diff(to_eigen(rref), EMat::Identity(40, 40)) < 1e-10);
    tensorET<2, double> rd({3, 4});
    const double rdv[12] = {1, 2, 3, 4, 2, 4, 6, 8, 1, 0, 1, 0};
    std::copy(rdv, rdv + 12, rd.data);
    tensorET<2, double> rr = gaussJordanElimination(rd);
    const double rexp[12] = {1, 0, 1, 0, 0, 1, 1, 2, 0, 0, 0, 0};
    double e = 0;
    for (int i = 0; i < 12; ++i) e = std::max(e, std::abs(rr[size_t(i)] - rexp[i]));
    check(C, "gaussJordanElimination of a rank-deficient matrix", e < 1e-14, e);
    const EMat Gb = conditioned(700, 10, 98);
    tensorET<2, double> rrb = gaussJordanElimination(from_eigen(Gb));
    check(C, "gaussJordanElimination n=700 (parallel)", max_abs_diff(to_eigen(rrb), EMat::Identity(700, 700)) < 1e-9);
    tensorET<1, double> col({3}, 5.0);
    tensorET<2, double> aug = augmentMatrix(rd, col), aug2 = augmentMatrix(rd, rd);
    check(C, "augmentMatrix", aug.size(1) == 5 && aug(2, 4) == 5.0 && aug(1, 3) == 8 && aug2.size(1) == 8 && aug2(2, 4) == 1);

    // Gram-Schmidt
    const EMat V = EMat::Random(50, 12);
    const EMat Qg = to_eigen(gramSchmidtOrthogonalization(from_eigen(V)));
    EMat Qref = V;
    for (int j = 0; j < 12; ++j) { // classical GS reference for the column order / signs
        for (int i = 0; i < j; ++i) Qref.col(j) -= Qref.col(i).dot(V.col(j)) * Qref.col(i);
        Qref.col(j).normalize();
    }
    check(C, "gramSchmidtOrthogonalization", max_abs_diff(Qg, Qref) < 1e-10 &&
                                                 (Qg.transpose() * Qg - EMat::Identity(12, 12)).norm() < 1e-13);

    // norms
    tensorET<2, double> na({2, 2}), nb({2, 2}, 0.0);
    na[0] = 3, na[1] = -4, na[2] = 0, na[3] = 12;
    check(C, "norm p = 1, 2, 3 and named",
        norm(na, nb, 1) == 19 && norm(na, nb, 2) == 13 && std::abs(norm(na, nb, 3) - std::cbrt(27.0 + 64 + 1728)) < 1e-12 &&
            norm(na, nb, "fro") == 13 && norm(na, nb, "inf") == 12 && norm(na, nb, "l1") == 19);
    check(C, "unknown norm name throws", throws([&] { (void)norm(na, nb, "spectral"); }));

    // conjugate gradient
    const size_t ncg = 400;
    const EMat M = EMat::Random(ncg, ncg);
    const EMat S = M * M.transpose() / double(ncg) + EMat::Identity(ncg, ncg);
    tensorET<1, double> rhs = tensorET<1, double>::uniform({ncg}, -1.0, 1.0, 21);
    // Default tol 1e-6. (The spec'd stop on |p^T A p| < 1e-12 caps the
    // reachable residual near 1e-6 for this scaling.)
    tensorET<1, double> xcg = conjugateGradient(from_eigen(S), rhs, tensorET<1, double>({ncg}, 0.0));
    const double r_cg = (S * Eigen::Map<Eigen::VectorXd>(xcg.data, ncg) - Eigen::Map<Eigen::VectorXd>(rhs.data, ncg)).norm();
    check(C, "conjugateGradient residual < tol", r_cg < 2e-6, r_cg);

    // Lanczos
    const size_t nl = 120, ml = 30;
    const EMat Sl = S.topLeftCorner(nl, nl);
    auto [al, be, Ql] = lanczos(from_eigen(Sl), ml, tensorET<1, double>({nl}, 1.0));
    const EMat eQl = to_eigen(Ql);
    EMat Tm = EMat::Zero(ml, ml);
    for (size_t i = 0; i < ml; ++i) Tm(Eigen::Index(i), Eigen::Index(i)) = al[i];
    for (size_t i = 0; i + 1 < ml; ++i)
        Tm(Eigen::Index(i), Eigen::Index(i + 1)) = Tm(Eigen::Index(i + 1), Eigen::Index(i)) = be[i];
    const double r_l1 = (eQl * eQl.transpose() - EMat::Identity(ml, ml)).norm();
    const double r_l2 = (eQl * Sl * eQl.transpose() - Tm).norm() / Sl.norm();
    check(C, "lanczos: orthonormal basis, Q A Q^T = T", r_l1 < 1e-12 && r_l2 < 1e-12 && be.size() == ml - 1,
        std::max(r_l1, r_l2));

    // expm
    check(C, "expm(0) = I", max_abs_diff(to_eigen(expm(tensorET<2, double>({4, 4}, 0.0))), EMat::Identity(4, 4)) == 0);
    for (double scale : {0.01, 0.5, 3.0, 40.0}) {
        const size_t ne = 60;
        const EMat Ae = EMat::Random(ne, ne) * (scale / std::sqrt(double(ne)));
        const EMat E1 = to_eigen(expm(from_eigen(Ae))), E2 = to_eigen(expm(from_eigen(-Ae)));
        const double r = (E1 * E2 - EMat::Identity(ne, ne)).norm() / (E1.norm() * E2.norm());
        // commuting check against a diagonal: exp(D) exactly
        check(C, "expm(A) expm(-A) = I, |A| ~ " + std::to_string(scale), r < 1e-13, r);
    }
    const EMat Dg = Eigen::VectorXd::LinSpaced(8, -3.0, 5.0).asDiagonal();
    const EMat Ed = to_eigen(expm(from_eigen(Dg)));
    double rd_e = 0;
    for (int i = 0; i < 8; ++i) rd_e = std::max(rd_e, std::abs(Ed(i, i) - std::exp(Dg(i, i))) / std::exp(Dg(i, i)));
    check(C, "expm of a diagonal matrix (relative 1e-12)", rd_e < 1e-12, rd_e);
    EMat rot(2, 2);
    rot << 0, -1.2, 1.2, 0;
    const EMat Er = to_eigen(expm(from_eigen(rot)));
    check(C, "expm of a rotation generator", std::abs(Er(0, 0) - std::cos(1.2)) < 1e-14 && std::abs(Er(1, 0) - std::sin(1.2)) < 1e-14);

    // symmetric eigenproblems
    const size_t ne = 300;
    const EMat Se = S.topLeftCorner(ne, ne);
    Eigen::SelfAdjointEigenSolver<Eigen::MatrixXd> es(Se);
    tensorET<1, double> ev;
    tensorET<2, double> Xv;
    const int nev = 4;
    const int its = lobpcg(from_eigen(Se), nev, tensorET<2, double>::uniform({ne, 8}, -1.0, 1.0, 5), ev, Xv, 500, 1e-9);
    double eerr = 0;
    for (int i = 0; i < nev; ++i) eerr = std::max(eerr, std::abs(ev[size_t(i)] - es.eigenvalues()(i)));
    const EMat eX = to_eigen(Xv);
    const double rres = (Se * eX - eX * Eigen::Map<Eigen::VectorXd>(ev.data, nev).asDiagonal()).norm();
    check(C, "lobpcg smallest eigenpairs vs Eigen (" + std::to_string(its) + " iterations)",
        eerr < 1e-9 && rres < 1e-7 && ev.size() == size_t(nev), eerr);
    tensorET<1, double> w;
    tensorET<2, double> Vw;
    symmetricEigen(from_eigen(Se.topLeftCorner(40, 40)), w, Vw);
    Eigen::SelfAdjointEigenSolver<Eigen::MatrixXd> es40(Se.topLeftCorner(40, 40));
    check(C, "symmetricEigen (Jacobi) vs Eigen",
        (Eigen::Map<Eigen::VectorXd>(w.data, 40) - es40.eigenvalues()).cwiseAbs().maxCoeff() < 1e-12);

    // tournament elimination and history replay
    for (size_t n : {size_t(9), size_t(64), size_t(150)}) {
        const EMat Ae = conditioned(n, 50, unsigned(7 * n));
        tensorET<2, double> Ut = from_eigen(Ae);
        const History<double> H = revEl(Ut, 16);
        const EMat eUt = to_eigen(Ut);
        const double upper = (eUt - EMat(eUt.triangularView<Eigen::Upper>())).cwiseAbs().maxCoeff();
        const double r_h = (to_eigen(matMul(H, from_eigen(Ae), 16)) - eUt).norm() / Ae.norm();
        const tensorET<2, double> Ainv = matMul(inverse_backs(Ut, 16), H, 16);
        const double r_ai = (to_eigen(Ainv) * Ae - EMat::Identity(Eigen::Index(n), Eigen::Index(n))).norm();
        check(C, "revEl: U upper, H A = U, U^-1 H = A^-1 n=" + std::to_string(n),
            upper == 0.0 && r_h < 1e-12 && r_ai < 1e-10, std::max(r_h, r_ai));
    }
    tensorET<2, double> st = from_eigen(conditioned(8, 5, 3));
    const auto one = elimStep(st, 1, 0, 4);
    size_t zeros = 0;
    for (size_t i = 1; i < 8; i += 2) zeros += st(i, 0) == 0.0;
    check(C, "elimStep eliminates the odd rows of one level", zeros == 4 && !one.empty());
}

int
main()
{
    auto t0 = std::chrono::steady_clock::now();
    test_storage();
    test_construction();
    test_copy_move_views();
    test_tiles();
    test_types_and_random();
#ifdef AXOS_ENABLE_CUDA
    test_cuda();
#endif
    test_expressions();
    test_products();
    test_linear_algebra();
    const double secs = std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count();
    std::printf("\nSUMMARY: %d passed, %d failed out of %d total (%.1f s)\n", g_pass, g_fail,
        g_pass + g_fail, secs);
    return g_fail > 255 ? 255 : g_fail;
}
