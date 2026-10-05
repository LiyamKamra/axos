// SPDX-License-Identifier: BSD-3-Clause
//
// Entry point of the QP solvers: solve_qp(problem, options) runs the chosen
// first-order method (QpOptions::method) on the CPU or, when the build has
// CUDA (-DAXOS_ENABLE_CUDA) and opt.use_gpu is set (or opt.auto_device finds
// the problem large enough), on the GPU. With opt.comm (several MPI ranks,
// -DAXOS_ENABLE_MPI) HPR-QP runs distributed, one GPU or CPU per rank.
#pragma once

#include "solver/qp/hpr_qp.h"
#include "solver/qp/qp_dist.h"
#include "solver/qp/pdhcg.h"
#include "solver/qp/qp_ipm.h"
#include "solver/qp/qp_backend_cpu.h"
#include "solver/qp/qp_model.h"
#if defined(AXOS_ENABLE_CUDA)
#include "solver/qp/qp_backend_cuda.h"
#endif
#include <algorithm>
#include <stdexcept>
#include <string>

namespace AXOS {
namespace Solver {

inline bool
qp_gpu_available()
{
#if defined(AXOS_ENABLE_CUDA)
    int count = 0;
    return cudaGetDeviceCount(&count) == cudaSuccess && count > 0;
#else
    return false;
#endif
}

template <class B>
QpSolution
solve_qp_on(const QpProblem &p, const QpOptions &opt)
{
    if (opt.method == QpMethod::Pdhcg) return qp::Pdhcg<B>().solve(p, opt);
    return qp::HprQp<B>().solve(p, opt);
}

// The device solve_qp uses for p under opt.
inline bool
qp_use_gpu(const QpProblem &p, const QpOptions &opt)
{
    if (!opt.auto_device) return opt.use_gpu;
    const size_t work = p.lp.A.nnz() + p.Q.nnz() + p.cols() + p.rows();
    return work >= opt.gpu_min_work && qp_gpu_available();
}

inline QpSolution
solve_qp(const QpProblem &p, const QpOptions &opt)
{
    if (opt.comm && opt.comm->distributed()) {
        // over MPI ranks: HPR-QP (Auto included), on every rank's GPU or CPU
        if (opt.method == QpMethod::Pdhcg || opt.method == QpMethod::Ipm)
            throw std::invalid_argument("solve_qp: only HPR-QP runs over several MPI ranks");
        QpOptions o = opt;
        o.method = QpMethod::HprQp;
        if (qp_use_gpu(p, o)) {
#if defined(AXOS_ENABLE_CUDA)
            return solve_qp_on<qp::CudaBackend>(p, o);
#else
            throw std::runtime_error("solve_qp: this build has no CUDA (-DAXOS_ENABLE_CUDA)");
#endif
        }
        return solve_qp_on<qp::CpuBackend>(p, o);
    }
    if (opt.method == QpMethod::Ipm) return qp::QpIpm().solve(p, opt); // CPU only
    if (opt.method == QpMethod::Auto) {
        QpOptions o = opt;
        o.method = QpMethod::Ipm;
        o.ipm_max_flops = std::min(opt.ipm_max_flops, opt.ipm_auto_flops);
        o.time_limit = 0.25 * opt.time_limit; // the rest is the first-order method's
        o.ipm_project = true;
        const QpSolution s = qp::QpIpm().solve(p, o);
        if (s.status == Status::Optimal || s.status == Status::Infeasible) return s;
        o = opt;
        o.method = QpMethod::HprQp;
        o.auto_device = true;
        o.gpu_min_work = std::min(opt.gpu_min_work, opt.gpu_min_work_hard);
        o.time_limit = opt.time_limit - s.seconds;
        QpSolution t = solve_qp(p, o);
        t.seconds += s.seconds; // the declined or failed interior-point attempt counts
        t.setup_seconds += s.seconds;
        t.method = "auto: hpr-qp after ipm (" + std::string(to_string(s.status)) + ")";
        return t;
    }
    if (qp_use_gpu(p, opt)) {
#if defined(AXOS_ENABLE_CUDA)
        return solve_qp_on<qp::CudaBackend>(p, opt);
#else
        throw std::runtime_error("solve_qp: this build has no CUDA (-DAXOS_ENABLE_CUDA)");
#endif
    }
    return solve_qp_on<qp::CpuBackend>(p, opt);
}

} // namespace Solver
} // namespace AXOS
