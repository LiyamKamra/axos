// SPDX-License-Identifier: BSD-3-Clause
//
// Umbrella header for the dense tensor layer (docs/TENSOR_SPEC.md §2):
// tensorET with its storage policies, element-wise math and matrix
// products, dense linear algebra and printing, plus the CUDA side when the
// build has it.
#pragma once

#include "storage/cuda_storage.h"
#include "storage/host_storage.h"
#include "tensorET.h"
#include "tensorIO.h"
#include "tensorLinearAlgebra.h"
#include "tensorMath.h"

#if defined(AXOS_ENABLE_CUDA) || defined(__CUDACC__)
#include "tensorCuda.h"
#include "tensorcuda/gpu_pool.h"
#endif
