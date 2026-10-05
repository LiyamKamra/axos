// SPDX-License-Identifier: BSD-3-Clause
//
// GPU kernels of the QP solvers, compiled at run time by NVRTC (see
// qp_backend_cuda.h); not included by host code. Every op of qp_ops.h gets a
// kernel named k_<op>:
//
//   map      k_f(idx n, Args a)                       grid-stride loop over f
//   reduce   k_f(idx n, Args a, double *part)        per-block partials,
//            then k_finish(nblocks, K, kMax, part, out) folds them
//   product  k_f_<T>(rows, rp, ci, v, x, Args a)     s = (A x)_row, then
//            f(row, s, a) (CSR, T = 1..32 threads per row, a power of two
//            chosen from the mean row length, combined with warp shuffles);
//            k_f_fin(rows, rcp, part, a) is the same epilogue for matrices
//            whose long rows are split into chunks (k_chunks_<T>), and
//            k_f_vec(rows, s, a) for row sums s formed elsewhere (summed
//            over MPI ranks, qp_dist.h).
// A plain product y = A x is the epilogue k_store.
#include "solver/qp/qp_ops.h"

using namespace axos_qp;

#define QP_BLOCK 256

#define QP_MAP_KERNEL(f, Args) \
    extern "C" __global__ void k_##f(idx n, Args a) \
    { \
        const idx stride = (idx)gridDim.x * blockDim.x; \
        for (idx i = (idx)blockIdx.x * blockDim.x + threadIdx.x; i < n; \
            i += stride) \
            f(i, a); \
    }

// Block reduction of K accumulators: warp shuffles, then one warp folds the
// per-warp results. Max-accumulators (bit set in kMax) combine with max.
template <int K>
__device__ void
block_reduce(double (&acc)[K], unsigned kmax, double *out)
{
    __shared__ double sh[K][QP_BLOCK / 32];
    const int lane = threadIdx.x & 31, warp = threadIdx.x >> 5;
#pragma unroll
    for (int k = 0; k < K; ++k) {
        double v = acc[k];
        for (int off = 16; off > 0; off >>= 1) {
            const double o = __shfl_down_sync(0xffffffffu, v, off);
            v = ((kmax >> k) & 1u) ? dmax(v, o) : v + o;
        }
        if (lane == 0) sh[k][warp] = v;
    }
    __syncthreads();
    if (warp == 0) {
#pragma unroll
        for (int k = 0; k < K; ++k) {
            double v = lane < QP_BLOCK / 32 ? sh[k][lane] : 0.0;
            for (int off = 16; off > 0; off >>= 1) {
                const double o = __shfl_down_sync(0xffffffffu, v, off);
                v = ((kmax >> k) & 1u) ? dmax(v, o) : v + o;
            }
            if (lane == 0) out[k] = v;
        }
    }
}

#define QP_REDUCE_KERNEL(f, Args) \
    extern "C" __global__ void k_##f(idx n, Args a, double *part) \
    { \
        double acc[Args::K]; \
        _Pragma("unroll") for (int k = 0; k < Args::K; ++k) acc[k] = 0.0; \
        const idx stride = (idx)gridDim.x * blockDim.x; \
        for (idx i = (idx)blockIdx.x * blockDim.x + threadIdx.x; i < n; \
            i += stride) \
            f(i, a, acc); \
        block_reduce<Args::K>( \
            acc, Args::kMax, part + (idx)blockIdx.x * Args::K); \
    }

// One block folds nblocks x K partials into out[0..K).
extern "C" __global__ void
k_finish(int nblocks, int K, unsigned kmax, const double *part, double *out)
{
    __shared__ double sh[QP_BLOCK / 32];
    const int lane = threadIdx.x & 31, warp = threadIdx.x >> 5;
    for (int k = 0; k < K; ++k) {
        const bool mx = (kmax >> k) & 1u;
        double v = 0.0;
        for (int b = threadIdx.x; b < nblocks; b += blockDim.x) {
            const double o = part[(idx)b * K + k];
            v = mx ? dmax(v, o) : v + o;
        }
        for (int off = 16; off > 0; off >>= 1) {
            const double o = __shfl_down_sync(0xffffffffu, v, off);
            v = mx ? dmax(v, o) : v + o;
        }
        if (lane == 0) sh[warp] = v;
        __syncthreads();
        if (warp == 0) {
            v = lane < QP_BLOCK / 32 ? sh[lane] : 0.0;
            for (int off = 16; off > 0; off >>= 1) {
                const double o = __shfl_down_sync(0xffffffffu, v, off);
                v = mx ? dmax(v, o) : v + o;
            }
            if (lane == 0) out[k] = v;
        }
        __syncthreads();
    }
}

// ---- sparse products with epilogues
// -------------------------------------------- s = sum of v[k] x[ci[k]] over
// [b, e), computed by the T threads of a group (T a power of two <= 32); lane 0
// of the group ends up with the sum. Every thread of the warp must call it
// (idle groups pass b == e), so the full-mask shuffles are safe.
template <int T>
__device__ double
group_dot(
    int b, int e, int lane, const int *ci, const double *v, const double *x)
{
    double s = 0.0;
    for (int k = b + lane; k < e; k += T)
        s += v[k] * x[ci[k]];
    for (int off = T / 2; off > 0; off >>= 1)
        s += __shfl_down_sync(0xffffffffu, s, off, T);
    return s;
}

// k_<f>_<T>: s = (A x)_row by T threads per row, then f(row, s, a). A block
// covers QP_BLOCK / T consecutive rows; with T > 1 the row sums go through
// shared memory so that the epilogues (which touch many vectors per row) run
// on consecutive threads, with coalesced loads, instead of on one thread in T.
#define QP_SPMV_EPI_T(f, Args, T) \
    extern "C" __global__ void __launch_bounds__(QP_BLOCK) \
        k_##f##_##T(idx rows, const int *rp, const int *ci, const double *v, \
            const double *x, Args a) \
    { \
        __shared__ double sums[QP_BLOCK / T]; \
        const idx base = (idx)blockIdx.x * (QP_BLOCK / T); \
        const int g = threadIdx.x / T, lane = threadIdx.x % T; \
        const idx row = base + g; \
        int b = 0, e = 0; \
        if (row < rows) { \
            b = rp[row]; \
            e = rp[row + 1]; \
        } \
        const double s = group_dot<T>(b, e, lane, ci, v, x); \
        if (T == 1) { \
            if (row < rows) f(row, s, a); \
            return; \
        } \
        if (lane == 0) sums[g] = s; \
        __syncthreads(); \
        if (threadIdx.x < QP_BLOCK / T && base + threadIdx.x < rows) \
            f(base + threadIdx.x, sums[threadIdx.x], a); \
    }

// k_<f>_fin: rows split into chunks (see k_chunks_<T>): s = sum of the row's
// chunk partials part[rcp[row] .. rcp[row+1]), then f(row, s, a).
// k_<f>_vec: f(row, s[row], a) for given row sums.
#define QP_SPMV_EPI_KERNEL(f, Args) \
    QP_SPMV_EPI_T(f, Args, 1) \
    QP_SPMV_EPI_T(f, Args, 2) \
    QP_SPMV_EPI_T(f, Args, 4) \
    QP_SPMV_EPI_T(f, Args, 8) \
    QP_SPMV_EPI_T(f, Args, 16) \
    QP_SPMV_EPI_T(f, Args, 32) \
    extern "C" __global__ void k_##f##_fin( \
        idx rows, const int *rcp, const double *part, Args a) \
    { \
        const idx stride = (idx)gridDim.x * blockDim.x; \
        for (idx r = (idx)blockIdx.x * blockDim.x + threadIdx.x; r < rows; \
            r += stride) { \
            double s = 0.0; \
            for (int c = rcp[r]; c < rcp[r + 1]; ++c) \
                s += part[c]; \
            f(r, s, a); \
        } \
    } \
    extern "C" __global__ void k_##f##_vec(idx rows, const double *s, Args a) \
    { \
        const idx stride = (idx)gridDim.x * blockDim.x; \
        for (idx r = (idx)blockIdx.x * blockDim.x + threadIdx.x; r < rows; \
            r += stride) \
            f(r, s[r], a); \
    }

// Partial products of the chunks of a row-split matrix: chunk c covers the
// nonzeros [cp[c], cp[c+1]) of one row; T threads per chunk.
#define QP_CHUNK_KERNEL(T) \
    extern "C" __global__ void k_chunks_##T(idx nch, const int *cp, \
        const int *ci, const double *v, const double *x, double *part) \
    { \
        const idx tid = (idx)blockIdx.x * blockDim.x + threadIdx.x; \
        const idx c = tid / T; \
        const int lane = (int)(tid % T); \
        int b = 0, e = 0; \
        if (c < nch) { \
            b = cp[c]; \
            e = cp[c + 1]; \
        } \
        const double s = group_dot<T>(b, e, lane, ci, v, x); \
        if (c < nch && lane == 0) part[c] = s; \
    }
QP_CHUNK_KERNEL(1)
QP_CHUNK_KERNEL(2)
QP_CHUNK_KERNEL(4)
QP_CHUNK_KERNEL(8)
QP_CHUNK_KERNEL(16)
QP_CHUNK_KERNEL(32)

// ---- instantiations
// -----------------------------------------------------------
QP_MAP_KERNEL(fill, Fill)
QP_MAP_KERNEL(copy, Copy)
QP_MAP_KERNEL(scale, Scale)
QP_MAP_KERNEL(scale_norm, ScaleNorm)
QP_MAP_KERNEL(diff, Diff)
QP_MAP_KERNEL(hpr_scalars, HprScalars)
QP_MAP_KERNEL(hpr_qb, HprQb)
QP_REDUCE_KERNEL(dot2, Dot2)
QP_SPMV_EPI_KERNEL(store, Store)
QP_MAP_KERNEL(hpr_primal, HprPrimal)
QP_SPMV_EPI_KERNEL(hpr_whalf, HprWHalf)
QP_SPMV_EPI_KERNEL(hpr_dual, HprDual)
QP_SPMV_EPI_KERNEL(hpr_aty, HprAty)
QP_SPMV_EPI_KERNEL(hpr_halpern, HprHalpern)
QP_SPMV_EPI_KERNEL(hpr_free_xw, HprFreeXW)
QP_SPMV_EPI_KERNEL(hpr_free_q, HprFreeQ)
QP_REDUCE_KERNEL(hpr_halpern_red, HprHalpernRed)
QP_REDUCE_KERNEL(hpr_halpern_y_red, HprHalpernYRed)
QP_REDUCE_KERNEL(hpr_theta_n, HprThetaN)
QP_REDUCE_KERNEL(diff_sq, DiffSq)
QP_REDUCE_KERNEL(kkt_rows, KktRows)
QP_REDUCE_KERNEL(kkt_cols, KktCols)
QP_MAP_KERNEL(pd_start, PdStart)
QP_MAP_KERNEL(pd_pg_step, PdPgStep)
QP_REDUCE_KERNEL(pd_pg_update, PdPgUpdate)
QP_REDUCE_KERNEL(pd_cg_curv, PdCgCurv)
QP_REDUCE_KERNEL(pd_cg_step, PdCgStep)
QP_MAP_KERNEL(pd_cg_dir, PdCgDir)
QP_MAP_KERNEL(pd_dual, PdDual)
QP_MAP_KERNEL(halpern, Halpern)
QP_MAP_KERNEL(pd_halpern_n, PdHalpernN)
QP_REDUCE_KERNEL(pd_kkt_cols, PdKktCols)
