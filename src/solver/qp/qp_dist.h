// SPDX-License-Identifier: BSD-3-Clause
//
// HPR-QP over several MPI processes, one GPU (or one CPU) per rank.
//
// The constraint rows are split into contiguous blocks, balanced on nonzeros.
// Rank r holds rows [r0, r1) of A, the n x (r1 - r0) matrix A^T of those rows,
// and the m-sized vectors of those rows (y, ybar, y0, the row bounds). Every
// rank keeps a full copy of the n-sized vectors and of Q. Then:
//   * A v needs no communication, because v (n-sized) is replicated.
//   * A^T y is a sum over the row blocks: each rank computes its block's
//     partial product, and one MPI_Allreduce (SUM, n doubles) per iteration
//     adds them up before the fused epilogue runs on the sum.
//   * Q products are repeated on every rank, so the work divided is the work in
//     A: LPs and QPs whose cost is in the constraints.
// Reductions over rows (merit and KKT terms at a check) are combined over the
// ranks: a few doubles every 10 to 100 iterations. Host-side decisions (stop,
// restart, new penalty) use rank 0's values, broadcast, so the ranks never take
// different paths.
//
// GPU vectors are exchanged through pinned host memory (the backend's
// through_host). With a CUDA-aware MPI (Open MPI or MVAPICH2 built with CUDA,
// Cray MPICH), define AXOS_MPI_CUDA_AWARE to pass device pointers to MPI
// directly. MS-MPI on Windows is not CUDA-aware.
//
// Without AXOS_ENABLE_MPI a QpComm is a single process, every operation is a
// no-op, and nothing here needs an MPI library.
#pragma once

#include "solver/model.h"
#include <algorithm>
#include <chrono>
#include <cstdint>
#include <stdexcept>
#include <utility>
#include <vector>

#if defined(AXOS_ENABLE_MPI)
#include <mpi.h>
#endif

namespace AXOS {
namespace Solver {

class QpComm {
  public:
    QpComm() = default; // one process
#if defined(AXOS_ENABLE_MPI)
    explicit QpComm(MPI_Comm c) : comm_(c)
    {
        MPI_Comm_rank(c, &rank_);
        MPI_Comm_size(c, &size_);
        MPI_Comm local;
        MPI_Comm_split_type(
            c, MPI_COMM_TYPE_SHARED, rank_, MPI_INFO_NULL, &local);
        MPI_Comm_rank(local, &local_rank_);
        MPI_Comm_free(&local);
    }
    MPI_Comm
    comm() const
    { return comm_; }
#endif

    int
    rank() const
    { return rank_; }
    int
    size() const
    { return size_; }
    // rank among the processes on this machine: rank r takes GPU r mod count
    int
    local_rank() const
    { return local_rank_; }
    bool
    distributed() const
    { return size_ > 1; }

    // Rows [first, second) of rank r: contiguous blocks with about equal
    // nnz + rows (row_ptr[i] + i is nondecreasing in i)
    std::pair<size_t, size_t>
    rows_of(const HostMatrix &A, int r) const
    {
        const size_t m = A.rows();
        auto start = [&](int k) -> size_t {
            if (k <= 0) return 0;
            if (k >= size_) return m;
            const auto *rp = A.row_ptr();
            const uint64_t total = uint64_t(A.nnz()) + m;
            const uint64_t target = total * uint64_t(k) / uint64_t(size_);
            size_t lo = 0, hi = m; // first i with rp[i] + i >= target
            while (lo < hi) {
                const size_t mid = (lo + hi) / 2;
                if (uint64_t(rp[mid]) + mid < target)
                    lo = mid + 1;
                else
                    hi = mid;
            }
            return lo;
        };
        return {start(r), start(r + 1)};
    }

    // In place over the ranks, on host memory (or device memory with a
    // CUDA-aware MPI)
    void
    sum(double *v, size_t n) const
    { reduce(v, n, true); }
    void
    max(double *v, size_t n) const
    { reduce(v, n, false); }

    void
    bcast(double *v, size_t n, int root = 0) const
    {
#if defined(AXOS_ENABLE_MPI)
        if (!distributed() || n == 0) return;
        const auto t0 = std::chrono::steady_clock::now();
        MPI_Bcast(v, int(n), MPI_DOUBLE, root, comm_);
        account(t0, n);
#else
        (void)v, (void)n, (void)root;
#endif
    }

    // The concatenation of every rank's rows (rows_of(A, r) of a vector
    // with one entry per row of A), in rank order, on every rank.
    std::vector<double>
    gather_rows(const HostMatrix &A, const std::vector<double> &mine) const
    {
        if (!distributed()) return mine;
        std::vector<double> all(A.rows());
#if defined(AXOS_ENABLE_MPI)
        std::vector<int> counts(size_), displs(size_);
        for (int r = 0; r < size_; ++r) {
            const auto b = rows_of(A, r);
            counts[r] = int(b.second - b.first);
            displs[r] = int(b.first);
        }
        if (int(mine.size()) != counts[rank_])
            throw std::logic_error("QpComm::gather_rows: wrong local length");
        const auto t0 = std::chrono::steady_clock::now();
        MPI_Allgatherv(mine.data(), counts[rank_], MPI_DOUBLE, all.data(),
            counts.data(), displs.data(), MPI_DOUBLE, comm_);
        account(t0, A.rows());
#endif
        return all;
    }

    // time spent in MPI calls (incl. waiting for the other ranks), calls,
    // doubles moved per rank
    double
    seconds() const
    { return seconds_; }
    long
    calls() const
    { return calls_; }
    double
    doubles() const
    { return doubles_; }

  private:
    void
    reduce(double *v, size_t n, bool add) const
    {
#if defined(AXOS_ENABLE_MPI)
        if (!distributed() || n == 0) return;
        const auto t0 = std::chrono::steady_clock::now();
        MPI_Allreduce(MPI_IN_PLACE, v, int(n), MPI_DOUBLE,
            add ? MPI_SUM : MPI_MAX, comm_);
        account(t0, n);
#else
        (void)v, (void)n, (void)add;
#endif
    }

    void
    account(std::chrono::steady_clock::time_point t0, size_t n) const
    {
        seconds_ +=
            std::chrono::duration<double>(std::chrono::steady_clock::now() - t0)
                .count();
        ++calls_;
        doubles_ += double(n);
    }

#if defined(AXOS_ENABLE_MPI)
    MPI_Comm comm_ = MPI_COMM_SELF;
#endif
    int rank_ = 0, size_ = 1, local_rank_ = 0;
    mutable double seconds_ = 0, doubles_ = 0;
    mutable long calls_ = 0;
};

// Rows [r0, r1) of A as a matrix of their own (same columns).
inline HostMatrix
row_block(const HostMatrix &A, size_t r0, size_t r1)
{
    const auto *rp = A.row_ptr();
    const auto *ci = A.col_ind();
    const double *v = A.values();
    std::vector<int32_t> brp(r1 - r0 + 1);
    for (size_t i = r0; i <= r1; ++i)
        brp[i - r0] = rp[i] - rp[r0];
    std::vector<int32_t> bci(ci + rp[r0], ci + rp[r1]);
    std::vector<double> bv(v + rp[r0], v + rp[r1]);
    return HostMatrix(r1 - r0, A.cols(), brp, bci, bv);
}

namespace qp {

// v (n doubles of the backend) summed over the ranks, in place, ordered
// with the work queued on the backend
template <class B>
void
allreduce_sum(B &be, double *v, size_t n, const QpComm &c)
{
    if (!c.distributed() || n == 0) return;
#if defined(AXOS_MPI_CUDA_AWARE)
    if constexpr (B::is_gpu) {
        be.sync();
        c.sum(v, n);
        return;
    }
#endif
    be.through_host(v, n, [&](double *h) { c.sum(h, n); });
}

} // namespace qp
} // namespace Solver
} // namespace AXOS
