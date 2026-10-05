// SPDX-License-Identifier: BSD-3-Clause
//
// SparseLdlt on the GPU, built on NVIDIA cuDSS. Same interface as the CPU
// solver in sparse_ldl.h. Requires -DCUDSS_WITH and -lcudss.
//
// analyze() runs cuDSS's reordering + symbolic phases, factorize() the
// numeric phase (the first call factorizes, later calls refactorize with the
// stored analysis), and solve() the triangular/diagonal solves. The matrix
// values are read from the Csr each time, so a new Csr with the same pattern
// can be passed to factorize().
#pragma once

#include "sparse/sparse_ldl.h"
#include "tensorcuda/gpu_pool.h"

#ifndef CUDSS_WITH
#error "cudss_solver.h requires -DCUDSS_WITH (and -lcudss)"
#endif

namespace AXOS {
namespace Sparse {
namespace cudss_detail {

inline void
check(cudssStatus_t s, const char *what)
{
    if (s != CUDSS_STATUS_SUCCESS)
        throw std::runtime_error(
            std::string("cuDSS error in ") + what + " (status " +
            std::to_string(static_cast<int>(s)) + ")");
}

template <typename Idx> constexpr cudssDataType_t
index_dt()
{
    return sizeof(Idx) == 4 ? CUDSS_R_32I : CUDSS_R_64I;
}

template <typename T> constexpr cudssDataType_t
value_dt()
{
    return std::is_same_v<T, float> ? CUDSS_R_32F : CUDSS_R_64F;
}

} // namespace cudss_detail

template <typename T, typename Idx>
class SparseLdlt<T, Idx, Cuda::CudaStorage> {
    static_assert(std::is_floating_point_v<T>,
        "SparseLdlt supports float and double");

  public:
    using matrix_type = Csr<T, Idx, Cuda::CudaStorage>;
    using vector_type = tensorET<1, T, Cuda::CudaStorage<T>>;

    explicit SparseLdlt(Symmetry kind = Symmetry::SPD,
        Ordering ord = Ordering::MinDegree)
        : kind_(kind)
    {
        using namespace cudss_detail;
        check(cudssConfigCreate(&config_), "cudssConfigCreate");
        if (ord == Ordering::MinDegree || ord == Ordering::NestedDissection) {
            cudssReorderingAlg_t alg = ord == Ordering::MinDegree
                                           ? CUDSS_REORDERING_ALG_AMD
                                           : CUDSS_REORDERING_ALG_NESTED_DISSECTION;
            check(cudssConfigSet(config_, CUDSS_CONFIG_REORDERING_ALG, &alg,
                      sizeof(alg)),
                "cudssConfigSet(reordering)");
        }
        check(cudssDataCreate(handle(), &data_), "cudssDataCreate");
    }

    SparseLdlt(const SparseLdlt &) = delete;
    SparseLdlt &operator=(const SparseLdlt &) = delete;

    ~SparseLdlt()
    {
        destroy_matrices();
        if (data_) cudssDataDestroy(handle(), data_);
        if (config_) cudssConfigDestroy(config_);
    }

    void
    analyze(const matrix_type &A)
    {
        using namespace cudss_detail;
        if (A.rows() != A.cols())
            throw std::invalid_argument("SparseLdlt: matrix must be square");
        destroy_matrices();
        n_ = A.rows();
        nnz_ = A.nnz();
        // Placeholder right-hand side / solution for the descriptors; the
        // real vectors are bound in solve() with cudssMatrixSetValues.
        scratch_ = Cuda::CudaStorage<T>(2 * n_);
        T *bp = scratch_.data(), *xp = scratch_.data() + n_;
        check(cudssMatrixCreateDn(&b_, n_, 1, n_, bp, value_dt<T>(),
                  CUDSS_LAYOUT_COL_MAJOR),
            "cudssMatrixCreateDn(b)");
        check(cudssMatrixCreateDn(&x_, n_, 1, n_, xp, value_dt<T>(),
                  CUDSS_LAYOUT_COL_MAJOR),
            "cudssMatrixCreateDn(x)");
        check(cudssMatrixCreateCsr(&a_, n_, n_, nnz_, A.row_ptr(), nullptr,
                  A.col_ind(), const_cast<T *>(A.values()), index_dt<Idx>(),
                  index_dt<Idx>(), value_dt<T>(),
                  kind_ == Symmetry::SPD ? CUDSS_MTYPE_SPD
                                         : CUDSS_MTYPE_SYMMETRIC,
                  CUDSS_MVIEW_FULL, CUDSS_BASE_ZERO),
            "cudssMatrixCreateCsr");
        check(cudssExecute(handle(), CUDSS_PHASE_ANALYSIS, config_, data_, a_,
                  x_, b_),
            "cudssExecute(analysis)");
        analyzed_ = true;
        factored_ = false;
        numeric_done_ = false;
    }

    // Returns false when cuDSS reports a failed factorization.
    bool
    factorize(const matrix_type &A)
    {
        using namespace cudss_detail;
        if (!analyzed_) analyze(A);
        if (A.rows() != n_ || A.nnz() != nnz_)
            throw std::invalid_argument(
                "SparseLdlt::factorize: pattern differs from analyze()");
        check(cudssMatrixSetCsrPointers(a_, A.row_ptr(), nullptr, A.col_ind(),
                  const_cast<T *>(A.values())),
            "cudssMatrixSetCsrPointers");
        check(cudssExecute(handle(),
                  numeric_done_ ? CUDSS_PHASE_REFACTORIZATION
                                : CUDSS_PHASE_FACTORIZATION,
                  config_, data_, a_, x_, b_),
            "cudssExecute(factorization)");
        cudaDeviceSynchronize();
        int info = 0;
        size_t written = 0;
        check(cudssDataGet(handle(), data_, CUDSS_DATA_INFO, &info,
                  sizeof(info), &written),
            "cudssDataGet(info)");
        numeric_done_ = true;
        factored_ = (info == 0);
        failed_ = factored_ ? -1 : info;
        return factored_;
    }

    void
    solve(const vector_type &b, vector_type &x) const
    {
        using namespace cudss_detail;
        if (!factored_)
            throw std::runtime_error("SparseLdlt::solve: not factorized");
        if (b.size() != n_ || x.size() != n_)
            throw std::invalid_argument("SparseLdlt::solve: size mismatch");
        check(cudssMatrixSetValues(b_, const_cast<T *>(b.data)),
            "cudssMatrixSetValues(b)");
        check(cudssMatrixSetValues(x_, x.data), "cudssMatrixSetValues(x)");
        check(cudssExecute(handle(), CUDSS_PHASE_SOLVE, config_, data_, a_,
                  x_, b_),
            "cudssExecute(solve)");
        cudaDeviceSynchronize();
    }

    size_t
    factor_nnz() const
    {
        int64_t v = 0;
        size_t written = 0;
        if (analyzed_)
            cudssDataGet(handle(), data_, CUDSS_DATA_LU_NNZ, &v, sizeof(v),
                &written);
        return static_cast<size_t>(v);
    }

    // Not implemented for cuDSS (it has its own pivot handling); no-op so
    // generic code can call it.
    void set_pivot_regularization(std::vector<signed char>, T) {}
    size_t regularized_pivots() const { return 0; }
    void print_profile() const {}

    // cuDSS reports a status code rather than a pivot index.
    long failed_pivot() const { return failed_; }

    // Positive / negative eigenvalue counts as reported by cuDSS for
    // indefinite matrices (zero is n - pos - neg).
    void
    inertia(size_t &pos, size_t &neg, size_t &zero) const
    {
        Idx v[2] = {0, 0}; // cuDSS: {positive, negative}, same width as the index
        size_t written = 0;
        cudss_detail::check(cudssDataGet(handle(), data_, CUDSS_DATA_INERTIA,
                                v, sizeof(v), &written),
            "cudssDataGet(inertia)");
        pos = static_cast<size_t>(v[0]);
        neg = static_cast<size_t>(v[1]);
        zero = n_ - pos - neg;
    }

  private:
    static cudssHandle_t handle() { return GPUMemoryPool::get().get_cudss(); }

    void
    destroy_matrices()
    {
        if (a_) cudssMatrixDestroy(a_);
        if (b_) cudssMatrixDestroy(b_);
        if (x_) cudssMatrixDestroy(x_);
        a_ = b_ = x_ = nullptr;
    }

    Symmetry kind_;
    size_t n_ = 0, nnz_ = 0;
    bool analyzed_ = false, factored_ = false, numeric_done_ = false;
    long failed_ = -1;
    cudssConfig_t config_ = nullptr;
    cudssData_t data_ = nullptr;
    cudssMatrix_t a_ = nullptr, b_ = nullptr, x_ = nullptr;
    Cuda::CudaStorage<T> scratch_;
};

} // namespace Sparse
} // namespace AXOS
