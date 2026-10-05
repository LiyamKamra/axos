// SPDX-License-Identifier: BSD-3-Clause
//
// Threading helpers for the dense layer: OpenMP when the translation unit is
// compiled with it, serial otherwise. Every helper opens at most one parallel
// region per call, runs small work inline (no fork/join), and never nests:
// inside an existing parallel region everything runs on the calling thread.
//
//   parallel_for(n, grain, f)     f(begin, end) over chunks of [0, n)
//   parallel_sum<R>(n, grain, f)  sum of f(begin, end) over the chunks, added
//                                 in chunk order (deterministic for a fixed
//                                 thread count)
//   copy_bytes / fill_elems       memcpy / fill, parallel above a few MiB
#pragma once

#include <algorithm>
#include <cstddef>
#include <cstring>
#include <vector>

#ifdef _OPENMP
#include <omp.h>
#endif

namespace AXOS {
namespace detail {

inline int &
serial_depth()
{
    thread_local int depth = 0;
    return depth;
}

// While alive, the dense kernels called from this thread run on it alone
// (for small, latency-bound phases where fork/join costs more than it saves).
struct SerialScope {
    SerialScope() { ++serial_depth(); }
    ~SerialScope() { --serial_depth(); }
    SerialScope(const SerialScope &) = delete;
    SerialScope &operator=(const SerialScope &) = delete;
};

// Threads a new parallel region would get from this point (1 inside one).
inline int
max_threads()
{
#ifdef _OPENMP
    if (serial_depth() > 0) return 1;
    return omp_in_parallel() ? 1 : omp_get_max_threads();
#else
    return 1;
#endif
}

// Number of chunks for n items with at least `grain` items per chunk: up to
// 4 per thread, so the dynamic schedule can balance hybrid (P/E) cores.
inline size_t
chunk_count(size_t n, size_t grain, int threads)
{
    if (threads <= 1 || grain == 0 || n < 2 * grain) return 1;
    return std::max<size_t>(1, std::min(n / grain, size_t(4) * threads));
}

// Start of chunk k of c over [0, n), rounded down to a multiple of `align`.
inline size_t
chunk_begin(size_t n, size_t c, size_t k, size_t align)
{
    if (k == 0) return 0;
    if (k >= c) return n;
    const size_t b = (n / c) * k + std::min(k, n % c);
    return align > 1 ? b / align * align : b;
}

template <class F>
void
parallel_for(size_t n, size_t grain, F &&f, size_t align = 16)
{
    if (n == 0) return;
    const int nt = max_threads();
    const size_t c = chunk_count(n, grain, nt);
    if (c <= 1) {
        f(size_t(0), n);
        return;
    }
#ifdef _OPENMP
    const int use = static_cast<int>(std::min<size_t>(nt, c));
#pragma omp parallel for schedule(dynamic, 1) num_threads(use)
    for (long long k = 0; k < static_cast<long long>(c); ++k) {
        const size_t b = chunk_begin(n, c, size_t(k), align);
        const size_t e = chunk_begin(n, c, size_t(k) + 1, align);
        if (b < e) f(b, e);
    }
#else
    f(size_t(0), n);
#endif
}

template <class R, class F>
R
parallel_sum(size_t n, size_t grain, F &&f, size_t align = 16)
{
    if (n == 0) return R(0);
    const int nt = max_threads();
    const size_t c = chunk_count(n, grain, nt);
    if (c <= 1) return f(size_t(0), n);
#ifdef _OPENMP
    std::vector<R> part(c, R(0));
    const int use = static_cast<int>(std::min<size_t>(nt, c));
#pragma omp parallel for schedule(dynamic, 1) num_threads(use)
    for (long long k = 0; k < static_cast<long long>(c); ++k) {
        const size_t b = chunk_begin(n, c, size_t(k), align);
        const size_t e = chunk_begin(n, c, size_t(k) + 1, align);
        if (b < e) part[size_t(k)] = f(b, e);
    }
    R s(0);
    for (const R &v : part)
        s += v;
    return s;
#else
    return f(size_t(0), n);
#endif
}

// Bytes per chunk for parallel copies and fills; below 4 chunks' worth a
// single memcpy/memset is as fast as a parallel one.
inline constexpr size_t kCopyGrain = size_t(1) << 20;

inline void
copy_bytes(void *dst, const void *src, size_t bytes)
{
    if (bytes == 0) return;
    if (bytes < 4 * kCopyGrain) {
        std::memcpy(dst, src, bytes);
        return;
    }
    char *d = static_cast<char *>(dst);
    const char *s = static_cast<const char *>(src);
    parallel_for(
        bytes, kCopyGrain,
        [=](size_t b, size_t e) { std::memcpy(d + b, s + b, e - b); }, 4096);
}

// True when every byte of x is the same, so a fill is a memset of that byte
// (covers 0, 0.0 and -1).
template <class T>
bool
uniform_bytes(const T &x, unsigned char &byte)
{
    unsigned char b[sizeof(T)];
    std::memcpy(b, &x, sizeof(T));
    for (size_t i = 1; i < sizeof(T); ++i)
        if (b[i] != b[0]) return false;
    byte = b[0];
    return true;
}

template <class T>
void
fill_elems(T *p, size_t n, const T &x)
{
    if (n == 0) return;
    unsigned char byte;
    const bool bytes = uniform_bytes(x, byte);
    const size_t grain = std::max<size_t>(1, kCopyGrain / sizeof(T));
    if (n < 4 * grain) {
        if (bytes)
            std::memset(static_cast<void *>(p), byte, n * sizeof(T));
        else
            std::fill_n(p, n, x);
        return;
    }
    parallel_for(
        n, grain,
        [=](size_t b, size_t e) {
            if (bytes)
                std::memset(
                    static_cast<void *>(p + b), byte, (e - b) * sizeof(T));
            else
                std::fill(p + b, p + e, x);
        },
        std::max<size_t>(1, 4096 / sizeof(T)));
}

} // namespace detail
} // namespace AXOS
