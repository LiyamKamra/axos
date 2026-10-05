// SPDX-License-Identifier: BSD-3-Clause
//
// CUDA side of the dense tensor layer (docs/TENSOR_SPEC.md §2, §5.8, §9):
// Cuda::CudaStorage, Cuda::Backend, GPUMemoryPool, the tensorETCuda alias and
// asynchronous 2-D tile helpers. Everything here is host code (no kernels),
// so it compiles with nvcc or with g++ and -DAXOS_ENABLE_CUDA. Without CUDA
// only the alias is provided (its storage cannot allocate).
#pragma once

#include "storage/cuda_storage.h"
#include "tensorET.h"
#include <algorithm>
#include <array>
#include <cstddef>

#if defined(AXOS_ENABLE_CUDA) || defined(__CUDACC__)
#include "tensorcuda/gpu_pool.h"
#include <cuda_runtime.h>
#endif

namespace AXOS {

template <int DIM, typename T>
using tensorETCuda = tensorET<DIM, T, Cuda::CudaStorage<T>>;

#if defined(AXOS_ENABLE_CUDA) || defined(__CUDACC__)

// Tile (i, j) of size m covers rows [i*m, i*m+m) and columns [j*m, j*m+m),
// clipped to the matrix. All helpers are asynchronous on `stream`; the
// buffers involved must stay alive until the stream reaches them.

// Writes tile (i, j) into the device buffer buf (m*m, row-major), zero outside
// the matrix, and returns a non-owning m x m view of buf.
template <class T>
tensorETCuda<2, T>
blockCopyCuda(const tensorETCuda<2, T> &t, size_t i, size_t j, size_t m,
    T *buf, cudaStream_t stream)
{
    Cuda::cuda_detail::check(cudaMemsetAsync(buf, 0, m * m * sizeof(T), stream),
        "blockCopyCuda: cudaMemsetAsync");
    const size_t r0 = i * m, c0 = j * m;
    const size_t nr = r0 < t.size(0) ? std::min(m, t.size(0) - r0) : 0;
    const size_t nc = c0 < t.size(1) ? std::min(m, t.size(1) - c0) : 0;
    if (nr && nc)
        Cuda::cuda_detail::check(
            cudaMemcpy2DAsync(buf, m * sizeof(T),
                t.data + r0 * t.stride(0) + c0, t.stride(0) * sizeof(T),
                nc * sizeof(T), nr, cudaMemcpyDeviceToDevice, stream),
            "blockCopyCuda: cudaMemcpy2DAsync");
    size_t d[2] = {m, m};
    return tensorETCuda<2, T>(buf, d, false);
}

// Owning, zero-padded copy of tile (i, j) (memory from the GPU pool).
template <class T>
tensorETCuda<2, T>
blockCopyCuda(const tensorETCuda<2, T> &t, size_t i, size_t j, size_t m,
    cudaStream_t stream)
{
    tensorETCuda<2, T> tile(std::array<size_t, 2>{m, m});
    blockCopyCuda(t, i, j, m, tile.data, stream);
    return tile;
}

// Copies the in-range part of an m x m device tile into tile (i, j) of t.
template <class T>
void
blockWriteBackCuda(tensorETCuda<2, T> &t, size_t i, size_t j, size_t m,
    const tensorETCuda<2, T> &tile, cudaStream_t stream)
{
    const size_t r0 = i * m, c0 = j * m;
    const size_t nr = r0 < t.size(0) ? std::min(m, t.size(0) - r0) : 0;
    const size_t nc = c0 < t.size(1) ? std::min(m, t.size(1) - c0) : 0;
    if (nr && nc)
        Cuda::cuda_detail::check(
            cudaMemcpy2DAsync(t.data + r0 * t.stride(0) + c0,
                t.stride(0) * sizeof(T), tile.data, tile.stride(0) * sizeof(T),
                nc * sizeof(T), nr, cudaMemcpyDeviceToDevice, stream),
            "blockWriteBackCuda: cudaMemcpy2DAsync");
}

#endif

} // namespace AXOS
