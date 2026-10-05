// SPDX-License-Identifier: BSD-3-Clause
//
// Sparse matrices (CSR) and kernels. Include this header for CPU use;
// CUDA kernels are added when compiled with nvcc and CUSPARSE_WITH.
#pragma once

#include "sparse/csr.h"
#include "sparse/sparse_cpu.h"
#include "sparse/sparse_ldl.h"
#include "sparse/sparse_ops.h"

#if (defined(AXOS_ENABLE_CUDA) || defined(__CUDACC__)) && defined(CUSPARSE_WITH)
#include "sparse/sparse_cuda.h"
#endif

#if (defined(AXOS_ENABLE_CUDA) || defined(__CUDACC__)) && defined(CUDSS_WITH)
#include "sparse/cudss_solver.h"
#endif
