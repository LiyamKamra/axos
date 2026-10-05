// SPDX-License-Identifier: BSD-3-Clause
//
// CUDA sparse kernels built on cuSPARSE, plus small custom kernels for
// scaling and norms (shaders/sparse.cu). Requires -DCUSPARSE_WITH and nvcc.
#pragma once

#include "shaders/sparse.cu"
#include "sparse/csr.h"
#include "sparse/sparse_cpu.h"
#include "tensorcuda/gpu_pool.h"
#include <cusparse.h>

#ifndef CUSPARSE_WITH
#error "sparse_cuda.h requires -DCUSPARSE_WITH (and -lcusparse)"
#endif

namespace AXOS {
namespace Sparse {
namespace cuda_detail {

inline void
check(cusparseStatus_t s, const char *what)
{
    if (s != CUSPARSE_STATUS_SUCCESS)
        throw std::runtime_error(std::string("cuSPARSE error in ") + what +
                                 ": " + cusparseGetErrorString(s));
}

inline void
check(cudaError_t e, const char *what)
{
    if (e != cudaSuccess)
        throw std::runtime_error(std::string("CUDA error in ") + what + ": " +
                                 cudaGetErrorString(e));
}

template <typename T> constexpr cudaDataType
value_type()
{
    if constexpr (std::is_same_v<T, float>) return CUDA_R_32F;
    else if constexpr (std::is_same_v<T, double>) return CUDA_R_64F;
    else if constexpr (std::is_same_v<T, std::complex<float>>) return CUDA_C_32F;
    else return CUDA_C_64F;
}

template <typename Idx> constexpr cusparseIndexType_t
index_type()
{
    return sizeof(Idx) == 4 ? CUSPARSE_INDEX_32I : CUSPARSE_INDEX_64I;
}

// Scratch memory taken from the GPU pool.
struct PoolBuf {
    void *p = nullptr;
    size_t bytes = 0;
    PoolBuf() = default;
    PoolBuf(const PoolBuf &) = delete;
    PoolBuf &operator=(const PoolBuf &) = delete;
    ~PoolBuf() { release(); }
    void
    reserve(size_t n)
    {
        if (n <= bytes) return;
        release();
        p = GPUMemoryPool::get().allocate(n);
        bytes = n;
    }
    void
    release()
    {
        if (p) GPUMemoryPool::get().deallocate(p, bytes);
        p = nullptr;
        bytes = 0;
    }
};

// Per-matrix cuSPARSE state: the sparse-matrix descriptor and the SpMV
// workspace (with preprocessing done once), reused across calls.
struct CusparseCache : Sparse::detail::DeviceCache {
    cusparseSpMatDescr_t mat = nullptr;
    PoolBuf spmv_buf;
    bool spmv_ready = false;
    PoolBuf spmm_buf;
    ~CusparseCache() override
    {
        if (mat) cusparseDestroySpMat(mat);
    }
};

template <typename M>
inline CusparseCache &
cache_of(const M &A)
{
    using T = typename M::value_type;
    using Idx = typename M::index_type;
    auto *c = dynamic_cast<CusparseCache *>(A.device_cache());
    if (!c) {
        auto sp = std::make_shared<CusparseCache>();
        A.set_device_cache(sp);
        c = sp.get();
    }
    if (!c->mat) {
        check(cusparseCreateCsr(&c->mat, A.rows(), A.cols(), A.nnz(),
                  const_cast<Idx *>(A.row_ptr()),
                  const_cast<Idx *>(A.col_ind()), const_cast<T *>(A.values()),
                  index_type<Idx>(), index_type<Idx>(),
                  CUSPARSE_INDEX_BASE_ZERO, value_type<T>()),
            "cusparseCreateCsr");
    }
    return *c;
}

inline cusparseHandle_t
handle()
{
    return GPUMemoryPool::get().get_cusparse();
}

} // namespace cuda_detail

template <> struct Kernels<Cuda::Backend> {
    using HostK = Kernels<Cpu::Backend>;

    template <typename M>
    static void
    spmv(const M &A, const typename M::value_type *x,
        typename M::value_type *y, typename M::value_type alpha,
        typename M::value_type beta)
    {
        using T = typename M::value_type;
        using namespace cuda_detail;
        if (A.nnz() == 0) {
            // y = beta * y (rare edge case, done on the host)
            std::vector<T> h(A.rows(), T(0));
            if (beta != T(0)) {
                check(cudaMemcpy(h.data(), y, A.rows() * sizeof(T),
                          cudaMemcpyDeviceToHost),
                    "spmv copy");
                for (auto &v : h) v *= beta;
            }
            check(cudaMemcpy(y, h.data(), A.rows() * sizeof(T),
                      cudaMemcpyHostToDevice),
                "spmv copy");
            return;
        }
        CusparseCache &c = cache_of(A);
        cusparseDnVecDescr_t vx, vy;
        check(cusparseCreateDnVec(&vx, A.cols(), const_cast<T *>(x),
                  value_type<T>()),
            "cusparseCreateDnVec");
        check(cusparseCreateDnVec(&vy, A.rows(), y, value_type<T>()),
            "cusparseCreateDnVec");
        if (!c.spmv_ready) {
            size_t sz = 0;
            check(cusparseSpMV_bufferSize(handle(),
                      CUSPARSE_OPERATION_NON_TRANSPOSE, &alpha, c.mat, vx,
                      &beta, vy, value_type<T>(), CUSPARSE_SPMV_ALG_DEFAULT,
                      &sz),
                "cusparseSpMV_bufferSize");
            c.spmv_buf.reserve(sz);
#if CUSPARSE_VERSION >= 12400
            check(cusparseSpMV_preprocess(handle(),
                      CUSPARSE_OPERATION_NON_TRANSPOSE, &alpha, c.mat, vx,
                      &beta, vy, value_type<T>(), CUSPARSE_SPMV_ALG_DEFAULT,
                      c.spmv_buf.p),
                "cusparseSpMV_preprocess");
#endif
            c.spmv_ready = true;
        }
        check(cusparseSpMV(handle(), CUSPARSE_OPERATION_NON_TRANSPOSE, &alpha,
                  c.mat, vx, &beta, vy, value_type<T>(),
                  CUSPARSE_SPMV_ALG_DEFAULT, c.spmv_buf.p),
            "cusparseSpMV");
        cusparseDestroyDnVec(vx);
        cusparseDestroyDnVec(vy);
    }

    template <typename M>
    static void
    spmm(const M &A, const typename M::value_type *X, size_t k,
        typename M::value_type *Y, typename M::value_type alpha,
        typename M::value_type beta)
    {
        using T = typename M::value_type;
        using namespace cuda_detail;
        if (A.nnz() == 0) {
            std::vector<T> h(A.rows() * k, T(0));
            if (beta != T(0)) {
                check(cudaMemcpy(h.data(), Y, h.size() * sizeof(T),
                          cudaMemcpyDeviceToHost),
                    "spmm copy");
                for (auto &v : h) v *= beta;
            }
            check(cudaMemcpy(Y, h.data(), h.size() * sizeof(T),
                      cudaMemcpyHostToDevice),
                "spmm copy");
            return;
        }
        CusparseCache &c = cache_of(A);
        cusparseDnMatDescr_t mx, my;
        check(cusparseCreateDnMat(&mx, A.cols(), k, k, const_cast<T *>(X),
                  value_type<T>(), CUSPARSE_ORDER_ROW),
            "cusparseCreateDnMat");
        check(cusparseCreateDnMat(&my, A.rows(), k, k, Y, value_type<T>(),
                  CUSPARSE_ORDER_ROW),
            "cusparseCreateDnMat");
        size_t sz = 0;
        check(cusparseSpMM_bufferSize(handle(), CUSPARSE_OPERATION_NON_TRANSPOSE,
                  CUSPARSE_OPERATION_NON_TRANSPOSE, &alpha, c.mat, mx, &beta,
                  my, value_type<T>(), CUSPARSE_SPMM_ALG_DEFAULT, &sz),
            "cusparseSpMM_bufferSize");
        c.spmm_buf.reserve(sz);
        check(cusparseSpMM(handle(), CUSPARSE_OPERATION_NON_TRANSPOSE,
                  CUSPARSE_OPERATION_NON_TRANSPOSE, &alpha, c.mat, mx, &beta,
                  my, value_type<T>(), CUSPARSE_SPMM_ALG_DEFAULT,
                  c.spmm_buf.p),
            "cusparseSpMM");
        cusparseDestroyDnMat(mx);
        cusparseDestroyDnMat(my);
    }

    // A_ij *= r[i] * c[j]; r, c are device pointers (either may be null).
    template <typename M>
    static void
    scale(M &A, const real_of_t<typename M::value_type> *r,
        const real_of_t<typename M::value_type> *c)
    {
        using T = typename M::value_type;
        using R = real_of_t<T>;
        using namespace cuda_detail;
        if (A.nnz() == 0) return;
        if constexpr (is_complex_value<T>::value) {
            // No complex kernel: scale on the host.
            Csr<T, typename M::index_type, Cpu::HostStorage> h(A);
            std::vector<R> hr(A.rows()), hc(A.cols());
            if (r) check(cudaMemcpy(hr.data(), r, hr.size() * sizeof(R),
                             cudaMemcpyDeviceToHost), "scale copy");
            if (c) check(cudaMemcpy(hc.data(), c, hc.size() * sizeof(R),
                             cudaMemcpyDeviceToHost), "scale copy");
            HostK::scale(h, r ? hr.data() : nullptr, c ? hc.data() : nullptr);
            check(cudaMemcpy(A.values_mut(), h.values(),
                      A.nnz() * sizeof(T), cudaMemcpyHostToDevice),
                "scale copy");
        } else {
            constexpr int W = Sparse::cuda_kernels::SPARSE_WARPS_PER_BLOCK;
            Sparse::cuda_kernels::scaleKernel<T, typename M::index_type>
                <<<static_cast<int>((A.rows() + W - 1) / W), W * 32>>>(
                    A.row_ptr(), A.col_ind(), A.values_mut(), r, c,
                    static_cast<long>(A.rows()));
            check(cudaGetLastError(), "scaleKernel");
        }
    }

    template <typename M>
    static void
    row_norms(const M &A, real_of_t<typename M::value_type> *out, Norm p)
    {
        using T = typename M::value_type;
        using R = real_of_t<T>;
        using namespace cuda_detail;
        if (A.nnz() == 0) {
            check(cudaMemset(out, 0, A.rows() * sizeof(R)), "row_norms");
            return;
        }
        if constexpr (is_complex_value<T>::value) {
            Csr<T, typename M::index_type, Cpu::HostStorage> h(A);
            std::vector<R> hn(A.rows());
            HostK::row_norms(h, hn.data(), p);
            check(cudaMemcpy(out, hn.data(), hn.size() * sizeof(R),
                      cudaMemcpyHostToDevice),
                "row_norms copy");
        } else {
            constexpr int W = Sparse::cuda_kernels::SPARSE_WARPS_PER_BLOCK;
            Sparse::cuda_kernels::rowNormKernel<T, typename M::index_type>
                <<<static_cast<int>((A.rows() + W - 1) / W), W * 32>>>(
                    A.row_ptr(), A.values(), out, static_cast<long>(A.rows()),
                    p == Norm::L1 ? 0 : (p == Norm::L2 ? 1 : 2));
            check(cudaGetLastError(), "rowNormKernel");
        }
    }

    template <typename M>
    static M
    transpose(const M &A)
    {
        using T = typename M::value_type;
        using Idx = typename M::index_type;
        using namespace cuda_detail;
        if constexpr (sizeof(Idx) != 4) {
            // cusparseCsr2cscEx2 is 32-bit only: go through the host.
            Csr<T, Idx, Cpu::HostStorage> h(A);
            return M(HostK::transpose(h));
        } else {
            const size_t m = A.rows(), n = A.cols(), nz = A.nnz();
            M B(n, m, nz);
            if (nz == 0) return B;
            size_t sz = 0;
            check(cusparseCsr2cscEx2_bufferSize(handle(), m, n, nz, A.values(),
                      A.row_ptr(), A.col_ind(), B.values_mut(),
                      B.row_ptr_mut(), B.col_ind_mut(), value_type<T>(),
                      CUSPARSE_ACTION_NUMERIC, CUSPARSE_INDEX_BASE_ZERO,
                      CUSPARSE_CSR2CSC_ALG1, &sz),
                "cusparseCsr2cscEx2_bufferSize");
            PoolBuf buf;
            buf.reserve(sz);
            check(cusparseCsr2cscEx2(handle(), m, n, nz, A.values(),
                      A.row_ptr(), A.col_ind(), B.values_mut(),
                      B.row_ptr_mut(), B.col_ind_mut(), value_type<T>(),
                      CUSPARSE_ACTION_NUMERIC, CUSPARSE_INDEX_BASE_ZERO,
                      CUSPARSE_CSR2CSC_ALG1, buf.p),
                "cusparseCsr2cscEx2");
            return B;
        }
    }

    template <typename M>
    static M
    spgemm(const M &A, const M &B)
    {
        using T = typename M::value_type;
        using Idx = typename M::index_type;
        using namespace cuda_detail;
        const size_t m = A.rows(), n = B.cols();
        if (A.nnz() == 0 || B.nnz() == 0) return M(m, n, 0);
        if constexpr (sizeof(Idx) != 4) {
            Csr<T, Idx, Cpu::HostStorage> ha(A), hb(B);
            return M(HostK::spgemm(ha, hb));
        } else {
            CusparseCache &ca = cache_of(A);
            CusparseCache &cb = cache_of(B);
            cusparseSpMatDescr_t matC;
            check(cusparseCreateCsr(&matC, m, n, 0, nullptr, nullptr, nullptr,
                      CUSPARSE_INDEX_32I, CUSPARSE_INDEX_32I,
                      CUSPARSE_INDEX_BASE_ZERO, value_type<T>()),
                "cusparseCreateCsr(C)");
            cusparseSpGEMMDescr_t d;
            check(cusparseSpGEMM_createDescr(&d), "cusparseSpGEMM_createDescr");
            const T alpha = T(1), beta = T(0);
            const auto op = CUSPARSE_OPERATION_NON_TRANSPOSE;
            const auto vt = value_type<T>();
            const auto alg = CUSPARSE_SPGEMM_DEFAULT;
            size_t s1 = 0, s2 = 0;
            check(cusparseSpGEMM_workEstimation(handle(), op, op, &alpha,
                      ca.mat, cb.mat, &beta, matC, vt, alg, d, &s1, nullptr),
                "SpGEMM workEstimation(size)");
            PoolBuf b1, b2;
            b1.reserve(s1);
            check(cusparseSpGEMM_workEstimation(handle(), op, op, &alpha,
                      ca.mat, cb.mat, &beta, matC, vt, alg, d, &s1, b1.p),
                "SpGEMM workEstimation");
            check(cusparseSpGEMM_compute(handle(), op, op, &alpha, ca.mat,
                      cb.mat, &beta, matC, vt, alg, d, &s2, nullptr),
                "SpGEMM compute(size)");
            b2.reserve(s2);
            check(cusparseSpGEMM_compute(handle(), op, op, &alpha, ca.mat,
                      cb.mat, &beta, matC, vt, alg, d, &s2, b2.p),
                "SpGEMM compute");
            int64_t cr, cc, cn;
            check(cusparseSpMatGetSize(matC, &cr, &cc, &cn),
                "cusparseSpMatGetSize");
            M C(m, n, static_cast<size_t>(cn));
            if (cn > 0) {
                check(cusparseCsrSetPointers(matC, C.row_ptr_mut(),
                          C.col_ind_mut(), C.values_mut()),
                    "cusparseCsrSetPointers");
                check(cusparseSpGEMM_copy(handle(), op, op, &alpha, ca.mat,
                          cb.mat, &beta, matC, vt, alg, d),
                    "cusparseSpGEMM_copy");
            }
            cusparseSpGEMM_destroyDescr(d);
            cusparseDestroySpMat(matC);
            check(cudaDeviceSynchronize(), "SpGEMM sync");
            return C;
        }
    }
};

} // namespace Sparse
} // namespace AXOS
