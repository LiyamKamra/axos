// SPDX-License-Identifier: BSD-3-Clause
//
// CUDA execution of the PDLP kernels (Parallel<Cuda::Backend>). Reductions
// are two-stage (per-block partials, then one finishing block), so they need
// no atomics; the host reads all slots back with one copy in fetch().
#pragma once

#include "solver/lp/pdlp_kernels.h"
#include "storage/cuda_storage.h"
#include <cuda_runtime.h>
#include <stdexcept>
#include <string>

namespace AXOS {
namespace Solver {
namespace pdlp {

template <typename F>
__global__ void
for_each_kernel(size_t n, F f)
{
    const size_t stride = static_cast<size_t>(gridDim.x) * blockDim.x;
    for (size_t i = static_cast<size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
         i < n; i += stride)
        f(i);
}

// Stage 1: each block writes its partial sums to part[block * K + k].
template <int K, typename F>
__global__ void
reduce_kernel(size_t n, F f, double *part)
{
    __shared__ double sh[K][32];
    double acc[K];
#pragma unroll
    for (int k = 0; k < K; ++k)
        acc[k] = 0.0;
    const size_t stride = static_cast<size_t>(gridDim.x) * blockDim.x;
    for (size_t i = static_cast<size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
         i < n; i += stride)
        f(i, acc);
    const int lane = threadIdx.x & 31, warp = threadIdx.x >> 5;
    const int nwarps = (blockDim.x + 31) >> 5;
#pragma unroll
    for (int k = 0; k < K; ++k) {
        double v = acc[k];
        for (int off = 16; off > 0; off >>= 1)
            v += __shfl_down_sync(0xffffffffu, v, off);
        if (lane == 0) sh[k][warp] = v;
    }
    __syncthreads();
    if (warp == 0) {
#pragma unroll
        for (int k = 0; k < K; ++k) {
            double v = lane < nwarps ? sh[k][lane] : 0.0;
            for (int off = 16; off > 0; off >>= 1)
                v += __shfl_down_sync(0xffffffffu, v, off);
            if (lane == 0) part[static_cast<size_t>(blockIdx.x) * K + k] = v;
        }
    }
}

// Stage 2 (one block): out[k] += sum over blocks of part[b * K + k].
template <int K>
__global__ void
reduce_finish_kernel(const double *part, int nblocks, double *out)
{
    __shared__ double sh[K][32];
    const int lane = threadIdx.x & 31, warp = threadIdx.x >> 5;
    const int nwarps = (blockDim.x + 31) >> 5;
#pragma unroll
    for (int k = 0; k < K; ++k) {
        double v = 0.0;
        for (int b = threadIdx.x; b < nblocks; b += blockDim.x)
            v += part[static_cast<size_t>(b) * K + k];
        for (int off = 16; off > 0; off >>= 1)
            v += __shfl_down_sync(0xffffffffu, v, off);
        if (lane == 0) sh[k][warp] = v;
    }
    __syncthreads();
    if (warp == 0) {
#pragma unroll
        for (int k = 0; k < K; ++k) {
            double v = lane < nwarps ? sh[k][lane] : 0.0;
            for (int off = 16; off > 0; off >>= 1)
                v += __shfl_down_sync(0xffffffffu, v, off);
            if (lane == 0) out[k] += v;
        }
    }
}

// Minimum reduction: stage 1 writes one minimum per block, stage 2 folds
// them into out[0] (which the caller preloads, see set_slot).
template <typename F>
__global__ void
reduce_min_kernel(size_t n, F f, double *part)
{
    __shared__ double sh[32];
    double m = INFINITY;
    const size_t stride = static_cast<size_t>(gridDim.x) * blockDim.x;
    for (size_t i = static_cast<size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
         i < n; i += stride) {
        const double v = f(i);
        m = v < m ? v : m;
    }
    for (int off = 16; off > 0; off >>= 1) {
        const double o = __shfl_down_sync(0xffffffffu, m, off);
        m = o < m ? o : m;
    }
    const int lane = threadIdx.x & 31, warp = threadIdx.x >> 5;
    if (lane == 0) sh[warp] = m;
    __syncthreads();
    if (warp == 0) {
        double v = lane < ((blockDim.x + 31) >> 5) ? sh[lane] : INFINITY;
        for (int off = 16; off > 0; off >>= 1) {
            const double o = __shfl_down_sync(0xffffffffu, v, off);
            v = o < v ? o : v;
        }
        if (lane == 0) part[blockIdx.x] = v;
    }
}

__global__ void
reduce_min_finish_kernel(const double *part, int nblocks, double *out)
{
    __shared__ double sh[32];
    double m = INFINITY;
    for (int b = threadIdx.x; b < nblocks; b += blockDim.x)
        m = part[b] < m ? part[b] : m;
    for (int off = 16; off > 0; off >>= 1) {
        const double o = __shfl_down_sync(0xffffffffu, m, off);
        m = o < m ? o : m;
    }
    const int lane = threadIdx.x & 31, warp = threadIdx.x >> 5;
    if (lane == 0) sh[warp] = m;
    __syncthreads();
    if (warp == 0) {
        double v = lane < ((blockDim.x + 31) >> 5) ? sh[lane] : INFINITY;
        for (int off = 16; off > 0; off >>= 1) {
            const double o = __shfl_down_sync(0xffffffffu, v, off);
            v = o < v ? o : v;
        }
        if (lane == 0) out[0] = v < out[0] ? v : out[0];
    }
}

} // namespace pdlp

template <> struct Parallel<Cuda::Backend> {
    double *dev = nullptr;  // accumulated slots
    double *part = nullptr; // per-block partial sums (kMaxBlocks * kMaxK)
    double host[pdlp::kSlots] = {};
    static constexpr int kBlock = 256;
    static constexpr int kMaxBlocks = 1024;
    static constexpr int kMaxK = 8;

    Parallel()
    {
        check(cudaMalloc(&dev, pdlp::kSlots * sizeof(double)), "cudaMalloc");
        check(cudaMalloc(&part, kMaxBlocks * kMaxK * sizeof(double)), "cudaMalloc");
        zero();
    }
    Parallel(const Parallel &) = delete;
    Parallel &operator=(const Parallel &) = delete;
    ~Parallel()
    {
        if (dev) cudaFree(dev);
        if (part) cudaFree(part);
    }

    static void
    check(cudaError_t e, const char *what)
    {
        if (e != cudaSuccess)
            throw std::runtime_error(std::string("CUDA error in ") + what +
                                     ": " + cudaGetErrorString(e));
    }

    static int
    grid_for(size_t n)
    {
        size_t g = (n + kBlock - 1) / kBlock;
        return static_cast<int>(g > kMaxBlocks ? kMaxBlocks : (g == 0 ? 1 : g));
    }

    template <typename F>
    void
    for_each(size_t n, F f)
    {
        if (n == 0) return;
        pdlp::for_each_kernel<<<grid_for(n), kBlock>>>(n, f);
    }

    void
    zero()
    {
        check(cudaMemset(dev, 0, pdlp::kSlots * sizeof(double)), "zero slots");
    }

    template <typename F>
    void
    reduce(int slot, size_t n, F f)
    {
        if (n == 0) return;
        static_assert(F::K <= kMaxK, "too many reduction values");
        const int blocks = grid_for(n);
        pdlp::reduce_kernel<F::K, F><<<blocks, kBlock>>>(n, f, part);
        pdlp::reduce_finish_kernel<F::K><<<1, kBlock>>>(part, blocks, dev + slot);
    }

    void
    set_slot(int slot, double v)
    {
        check(cudaMemcpy(dev + slot, &v, sizeof(double), cudaMemcpyHostToDevice),
            "set_slot");
    }

    // slots[slot] = min(slots[slot], min over i of f(i)).
    template <typename F>
    void
    reduce_min(int slot, size_t n, F f)
    {
        if (n == 0) return;
        const int blocks = grid_for(n);
        pdlp::reduce_min_kernel<<<blocks, kBlock>>>(n, f, part);
        pdlp::reduce_min_finish_kernel<<<1, kBlock>>>(part, blocks, dev + slot);
    }

    const double *
    fetch()
    {
        check(cudaGetLastError(), "kernel launch");
        check(cudaMemcpy(host, dev, sizeof(host), cudaMemcpyDeviceToHost),
            "fetch");
        return host;
    }

    void
    copy(double *dst, const double *src, size_t n)
    {
        if (n) check(cudaMemcpy(dst, src, n * sizeof(double),
                         cudaMemcpyDeviceToDevice), "copy");
    }
};

} // namespace Solver
} // namespace AXOS
