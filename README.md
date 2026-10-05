# AXOS

Sparse linear algebra and optimization solvers in C++17, with CPU (OpenMP) and
CUDA backends.

## What is here

* **Dense tensor layer** (`src/tensorET.h` and friends): `tensorET<DIM, T,
  Storage>` with host and CUDA storage policies, a pooled GPU allocator,
  fused expression templates, GEMM/GEMV kernels (AVX-512 / AVX2 / SSE2), and
  dense linear algebra (blocked LU, Householder QR, triangular solves and
  inverses, CG, Lanczos, LOBPCG, expm). It implements
  [docs/TENSOR_SPEC.md](docs/TENSOR_SPEC.md); see the status section there.
* **Sparse** (`src/sparse/`): CSR matrices with SpMV/SpMM/SpGEMM/transpose
  (cuSPARSE on CUDA), AMD ordering, and a sparse LDLᵀ direct solver: cuDSS on
  CUDA, and on the CPU a multifrontal supernodal code with packed AVX-512 dense
  kernels.
* **LP solver** (`src/solver/`): MPS reader/writer, presolve with postsolve,
  scaling, and four methods:
  * PDLP: reflected Halpern PDHG, CPU and CUDA, with infeasibility detection.
  * A primal-dual interior-point method: cuDSS on the GPU; multifrontal LDLᵀ or
    normal equations on the CPU; optional crossover.
  * A CPU dual simplex: hypersparse LU, dual steepest edge, bound flipping,
    warm starts.
  * `Auto`, which picks and chains the other three.
* **QP solver** (`src/solver/qp/`): convex QP from QPS/MPS files, three
  methods:
  * HPR-QP, a Halpern Peaceman–Rachford method. It runs on the CPU (OpenMP)
    and on the GPU, using its own kernels compiled at run time with NVRTC, so
    no nvcc is needed. Vector updates are fused into the sparse products, and
    iterations replay as CUDA graphs.
  * PDHCG, a primal-dual hybrid conjugate gradient method, on the CPU and the
    GPU.
  * A primal-dual interior-point method on the CPU, built on the multifrontal
    LDLᵀ.

  `Auto` uses the interior point when its factorization is cheap. Otherwise
  it uses HPR-QP, on the GPU for large problems. Benchmarks against
  HPR-QP.jl, PDHCG.jl, OSQP, Clarabel, HiGHS, PIQP and SCS are in
  [benchmarks/qp/README.md](benchmarks/qp/README.md).

  HPR-QP (and HPR for LPs) also runs over several MPI processes, one GPU
  each (`qp_dist.h`, build with `-DAXOS_ENABLE_MPI`). The constraint rows
  are split across the ranks. Each iteration then needs one
  `MPI_Allreduce` of n doubles, which sums the ranks' partial Aᵀy products
  before the fused update runs on every GPU.
* **MILP solver** (`src/solver/milp/`): branch and cut with the tree on the
  CPU and the arithmetic-heavy parts on the GPU:
  * Presolve: LP reductions, MIP coefficient tightening, domain propagation
    and double probing on the binaries. The GPU probes many binaries at once
    with frontier-based propagation.
  * Root: the dual simplex, Gomory mixed-integer cuts from the tableau, and
    c-MIR cuts from the rows (with variable-bound substitution and
    aggregation). Reduced-cost fixing follows.
  * Primal heuristics: feasibility jump (also as many GPU walkers next to the
    CPU), rounding, fix-and-propagate, diving, the objective feasibility
    pump, RENS and RINS.
  * Tree: reliability pseudocost branching with strong branching, best-bound
    search with plunging. Large LPs use batched GPU PDHG, whose Lagrangian
    bounds are valid child bounds.

  Benchmarks against HiGHS, SCIP and CBC on MIPLIB 2017 are in
  [benchmarks/milp/README.md](benchmarks/milp/README.md).
* The plan for MIQP and convex MINLP is in
  [src/solver/ROADMAP.md](src/solver/ROADMAP.md).

## Using AXOS: command line and API

`build/axos` solves LP, QP and MILP models from MPS / QPS files; the problem
type follows from the file (integer markers: MILP, a quadratic section: QP).
Build it with `make axos` (Linux) or `bash apps/build.sh` (Windows Git Bash
with MSVC; `bash apps/build.sh cpu axos_cpu` for a build without CUDA). The
GPU build needs an NVIDIA driver to start (it links the CUDA driver API; the
runtime DLLs are copied next to it) and compiles its kernels at run time with
NVRTC from the source tree: set `AXOS_SRC_DIR` to the `src` directory when
the executable is moved elsewhere. The CPU build runs anywhere.

```sh
axos model.mps                                   # solve, print a summary
axos model.mps --time-limit 60 --gap 1e-4        # MILP with limits
axos model.qps --method hprqp --device gpu       # QP on the GPU
axos model.mps --type lp                         # LP relaxation of a MILP
axos model.mps --solution x.sol --json result.json
axos model.mps --info                            # statistics only
```

Options: `--type auto|lp|qp|milp`, `--method` (LP: `auto simplex ipm pdlp
hpr`; QP: `auto hprqp pdhcg ipm`; MILP: `auto`), `--device auto|cpu|gpu`,
`--time-limit S`, `--tol T` (LP/QP), `--gap G` (MILP), `--no-presolve`,
`--no-cuts`, `--no-heuristics`, `--solution FILE` (`name value` lines),
`--json FILE|-`, `--quiet`, `--verbose N`, `--version`. The exit status is 0
for optimal, 1 for a solution without proof of optimality, 2 infeasible, 3
unbounded, 4 no solution within the limits, 5 error, 64 bad usage. Every
reported solution is checked against the model (largest relative violation
in the summary and the JSON).

C++ (header only, `src/solver/api.h`):

```cpp
#include "solver/api.h"
using namespace AXOS::Solver;

Model m = read_model("model.mps");
Options o;
o.time_limit = 60;                 // also: type, method, device, tol, gap, verbose
Result r = solve(m, o);            // r.status, r.objective, r.bound, r.x, r.y, r.gap()
std::cout << result_json(m, r);    // or write_solution(std::cout, m, r)
```

`make test_cli` (or `python tests/test_cli.py`) checks the command line and
the Python wrapper on LP, QP and MILP models with known answers.

Python (`apps/axos.py` with `apps/` on `PYTHONPATH`; it calls the executable
and parses its JSON):

```python
import axos
r = axos.solve("model.mps", time_limit=60, device="gpu")
print(r["status"], r["objective"], r["x"])
```

Several GPUs (or machines) through MPI: build `build/axos_mpi` with `make
axos_mpi` or `bash apps/build.sh mpi axos_mpi` (MS-MPI SDK on Windows,
mpicxx on Linux) and start it under `mpiexec`. Rank r uses GPU r mod (GPUs
per machine), and rank 0 prints and writes the files. LPs and QPs are
solved by HPR with the constraint rows split over the ranks; a MILP needs a
single process. From Python: `axos.solve(path, ranks=4, device="gpu")`.

```sh
mpiexec -n 4 build/axos_mpi model.qps --device gpu --time-limit 600
```

## Documentation

* [ref-manual.txt](ref-manual.txt): API of the sparse, LP, QP and MILP layers.
* [docs/TENSOR_SPEC.md](docs/TENSOR_SPEC.md): specification and
  implementation status of the dense tensor layer.
* [benchmarks/sparse/RESULTS.md](benchmarks/sparse/RESULTS.md),
  [benchmarks/solver/RESULTS.md](benchmarks/solver/RESULTS.md): results
  against Eigen, PyTorch, SciPy and HiGHS (including HiGHS' cuPDLP-C).

## Building

Requirements: GNU Make, g++ (C++17, OpenMP), Eigen 3 for the tests. For CUDA:
nvcc with a supported host gcc (`make configure` detects gcc-14/13), cuSPARSE,
cuDSS, cuBLAS and cuSolver.

```sh
make test_tensor_cpu                           # dense layer, CPU only
make test_sparse_cpu && make test_solver_cpu   # CPU only
make test_tensor && make test_sparse && make test_solver   # CPU + CUDA
make benchmark_dense                           # dense layer vs Eigen
make benchmark_sparse                          # vs Eigen / torch
make benchmark_lp                              # vs HiGHS (needs scipy/highspy)
make build/run_mps && ./build/run_mps --method auto file.mps
```

## Layout

```
src/tensorET.h     tensorET (dense tensor class); axos.h is the umbrella header
src/tensorMath.h   expressions, matrix products; tensorLinearAlgebra.h, tensorIO.h
src/storage/       Cpu::HostStorage, Cuda::CudaStorage (+ backend tags)
src/tensor/        SIMD layer, threading helpers, GEMM/GEMV/transpose kernels
src/tensorcuda/    GPUMemoryPool (pooled device memory, library handles)
src/sparse/        Csr, CPU/CUDA kernels, AMD, LDLᵀ (multifrontal, cuDSS)
src/shaders/       sparse.cu (CUDA kernels for the sparse layer)
src/solver/        LP model, MPS I/O, presolve, scaling, PDLP, IPM, simplex; api.h (LP/QP/MILP entry point)
src/solver/qp/     HPR-QP, PDHCG, QP interior point; GPU kernels (NVRTC); MPI (qp_dist.h)
src/solver/milp/   branch and cut, cuts, propagation, probing, heuristics; GPU kernels (NVRTC)
apps/              axos (command-line solver), axos.py (Python wrapper)
benchmarks/        drivers and results (qp/, milp/, solver/, sparse/, dense/, reference/)
tests/             test_tensor.cpp, test_sparse.cpp, test_solver.cpp, netlib models
docs/              TENSOR_SPEC.md
```
