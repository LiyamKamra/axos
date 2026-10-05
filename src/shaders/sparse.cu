// SPDX-License-Identifier: BSD-3-Clause
//
// CUDA kernels for Csr: row/column scaling and row norms. One warp handles
// one row. Real value types only (float, double); complex matrices use the
// host kernels or cuSPARSE for the products.
#ifndef SHADERS_SPARSE_H
#define SHADERS_SPARSE_H

#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cuda_runtime.h>

namespace AXOS {
namespace Sparse {
namespace cuda_kernels {

inline constexpr int SPARSE_WARPS_PER_BLOCK = 8;

template <typename T, typename Idx>
__global__ void
scaleKernel(const Idx *__restrict__ rp, const Idx *__restrict__ ci,
    T *__restrict__ vals, const T *__restrict__ r, const T *__restrict__ c,
    long rows)
{
    const long warp =
        (static_cast<long>(blockIdx.x) * blockDim.x + threadIdx.x) / 32;
    const int lane = threadIdx.x & 31;
    if (warp >= rows) return;
    const T ri = r ? r[warp] : T(1);
    for (Idx k = rp[warp] + lane; k < rp[warp + 1]; k += 32)
        vals[k] *= c ? ri * c[ci[k]] : ri;
}

// norm: 0 = L1, 1 = L2, 2 = Linf
template <typename T, typename Idx>
__global__ void
rowNormKernel(const Idx *__restrict__ rp, const T *__restrict__ vals,
    T *__restrict__ out, long rows, int norm)
{
    const long warp =
        (static_cast<long>(blockIdx.x) * blockDim.x + threadIdx.x) / 32;
    const int lane = threadIdx.x & 31;
    if (warp >= rows) return; // whole warp exits together
    T acc = 0;
    for (Idx k = rp[warp] + lane; k < rp[warp + 1]; k += 32) {
        T a = ::fabs(vals[k]);
        if (norm == 0)
            acc += a;
        else if (norm == 1)
            acc += a * a;
        else
            acc = a > acc ? a : acc;
    }
    for (int off = 16; off > 0; off >>= 1) {
        T other = __shfl_down_sync(0xffffffffu, acc, off);
        if (norm == 2)
            acc = other > acc ? other : acc;
        else
            acc += other;
    }
    if (lane == 0) out[warp] = (norm == 1) ? ::sqrt(acc) : acc;
}

} // namespace cuda_kernels
} // namespace Sparse
} // namespace AXOS

#endif // SHADERS_SPARSE_H
