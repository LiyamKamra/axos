# QP benchmarks

AXOS's first-order QP solvers (`src/solver/qp/`: HPR-QP and PDHCG, CPU and
GPU backends, no external solver library) against established open-source
solvers, on identical data and scored with one metric.

| solver | kind | device | how it is run |
|---|---|---|---|
| **AXOS HPR-QP** | Halpern Peaceman–Rachford (this project) | GPU, CPU | `build/run_qp` |
| **AXOS PDHCG** | primal–dual hybrid conjugate gradient (this project) | GPU | `build/run_qp --method pdhcg` |
| **AXOS IPM** | Mehrotra interior point on the project's sparse LDLᵀ | CPU | `build/run_qp --method ipm` |
| **AXOS auto** | interior point when its factorization is cheap, else HPR-QP (GPU when large) | CPU + GPU | `build/run_qp --method auto --device auto` |
| HPR-QP.jl | the authors' GPU implementation of HPR-QP | GPU | `run_julia.jl hprqp` |
| PDHCG.jl | the authors' GPU implementation of PDHCG | GPU | `run_julia.jl pdhcg` |
| Clarabel, PIQP | interior point | CPU | `qpbench.py` |
| HiGHS | active set (QP) / simplex (LP) | CPU | `qpbench.py` |
| OSQP, SCS | ADMM (first order) | CPU | `qpbench.py` |

## Setup (everything project-local, git-ignored)

```bash
python -m venv .venv
.venv/Scripts/python -m pip install numpy scipy osqp clarabel highspy piqp scs
bash benchmarks/qp/build.sh                  # build/run_qp (MSVC or g++; NVRTC at run time)
bash benchmarks/qp/setup_comparators.sh all  # Maros-Meszaros (+ .axqp export), Julia, HPR-QP.jl, PDHCG.jl
```

On Windows the GPU build needs only the pip CUDA wheels in `.venv`
(`nvidia-cuda-runtime-cu12`, `nvidia-cuda-nvcc-cu12` for headers,
`nvidia-cuda-nvrtc-cu12`); see `build.sh`.

## Problem sets

* **Maros–Meszaros** (138 convex QPs, `data/maros_meszaros/*.SIF`), exported
  once to `data/mm_axqp/*.axqp` by `run_qp --export`, so every solver reads
  exactly the data AXOS parsed.
* **Large generated** (`gen_large.py`, 14 problems): the seven families of the
  OSQP benchmark suite — random, equality-constrained, portfolio, control (MPC),
  Huber fitting, SVM, Lasso — at about 1 M (`_M`) and 4 M (`_L`) nonzeros,
  seeded and reproducible.

## Running

```bash
bash benchmarks/qp/run_all.sh                       # Maros-Meszaros, 1e-6, 60 s per problem
TOL=1e-9 SOLVERS=piqp bash benchmarks/qp/run_all.sh benchmarks/qp/data/maros_meszaros \
    benchmarks/qp/data/mm_axqp benchmarks/qp/results/mm_ref   # objective reference
TL=120 SOLVERS="axos-hprqp-gpu axos-pdhcg-gpu hprqp-jl pdhcg-jl axos-hprqp-cpu clarabel piqp osqp scs" \
    bash benchmarks/qp/run_all.sh benchmarks/qp/data/large_axqp \
    benchmarks/qp/data/large_axqp benchmarks/qp/results/large
python benchmarks/qp/report.py benchmarks/qp/results/mm --ref benchmarks/qp/results/mm_ref/piqp.csv
```

`run_all.sh` runs one solver at a time (never two on the GPU or the CPU at
once) and writes `results/<set>/<solver>.csv`. The GPU solvers also write
their solutions, which `qpbench.py score` re-evaluates; the CPU solvers are
evaluated in the harness. Each CPU solve runs in its own process, killed at
1.5x the time limit if the solver does not stop by itself, or when its memory
passes 4 GiB (`--mem-limit`; the interior-point factorizations of the random
large problems would otherwise exhaust a 16 GB machine).

## Metric

All solutions are scored with the relative KKT residuals of AXOS
(`src/solver/qp/qp_model.h`, `evaluate_qp`; `qpbench.py`, `evaluate`), after
mapping each solver's duals to one sign convention:

    eta_p   = max(|Ax - P_K(Ax)|, |x - P_C(x)|) / (1 + max(|b|, |Ax|))
    eta_d   = |Qx + c - A^T y - z| / (1 + max(|c|, |A^T y|, |Qx|))
    eta_gap = |P - D| / (1 + max(|P|, |D|))

(infinity norms; z is the projection of Qx + c - A^T y onto the sign cone of
the bounds). A problem counts as **solved** when the solver reports optimal and
max(eta_p, eta_d, eta_gap) <= 10 x tol (each solver measures its own
tolerance in its own norms); the strict count uses 1 x tol. AXOS stops on this
metric itself. Times are wall clock for the solve after the data are in memory
(preconditioning, host-to-GPU transfer, factorizations or spectral estimates,
iterations); JIT and kernel compilation are excluded for every solver (the
Julia solvers are warmed up on each problem first, AXOS compiles its kernels
once per process). **SGM10** is the shifted geometric mean of the times with a
10 s shift, unsolved problems counted at the time limit. Objective errors are
measured against a high-accuracy reference run, PIQP at 1e-9. (Not Clarabel:
it reports optimal with wrong objectives on the ill-conditioned LISWET
problems, e.g. 31.1 for LISWET7 whose optimum is 498.8; the gap term of the
metric catches this.)

## Results (30 September and 5 October 2026)

RTX 3050 A Laptop GPU (4 GB), Core i7-12650H, tolerance 1e-6. Solved = reports
optimal and passes the common check (10 x tol); SGM10 in seconds. Full report
with performance profiles and per-problem tables: `results/qp_report.html`.

| solver | Maros–Meszaros (138, 60 s) | SGM10 | large (14, 120 s) | SGM10 |
|---|---|---|---|---|
| PIQP | 136 | 0.67 | 9 | 46.8 |
| **AXOS auto** | **129** | **1.90** | **13** | 17.2 |
| SCS | 125 | 3.23 | 9 | 31.9 |
| **AXOS IPM (CPU)** | **123** | 2.70 | 9 | 41.3 |
| **AXOS HPR-QP (GPU)** | **123** | 3.75 | **13** | **5.7** |
| Clarabel | 119 | 3.44 | 9 | 44.0 |
| **AXOS HPR-QP (CPU)** | 115 | 5.47 | 13 | 14.0 |
| HPR-QP.jl (GPU) | 114 | 6.98 | 11 | 12.6 |
| **AXOS PDHCG (GPU)** | 100 | 11.24 | 11 | 13.1 |
| OSQP | 95 | 10.71 | 9 | 44.7 |
| HiGHS | 79 | 14.68 | – | – |
| PDHCG.jl (GPU) | 75 | 19.36 | 7 | 52.3 |

On the problems both solve, AXOS HPR-QP on the GPU is 7.8x faster than
HPR-QP.jl on Maros–Meszaros (113 problems) and 17–18x faster than Clarabel
and PIQP on the large problems they can factor (9 of 14). Every answer AXOS
reports as optimal on Maros–Meszaros meets the strict 1e-6 check. All AXOS
HPR-QP entries and AXOS auto were measured with the final build (free-variable
sweep order, final auto rule; re-runs on 5 October); AXOS IPM with the build
just before it (its code path is unchanged), AXOS PDHCG with the first build.

## Several GPUs: HPR-QP over MPI

`src/solver/qp/qp_dist.h` runs HPR-QP (and HPR for LPs) over MPI ranks, one
GPU each. The constraint rows are split into contiguous blocks with about
equal nonzeros. Each rank keeps its rows of A and Aᵀ and their duals, plus a
copy of the n-sized state and of Q. Per iteration:

* the products with A need no communication, because their input is
  replicated;
* the products with Aᵀ are summed over the ranks: each GPU forms its partial
  product, one `MPI_Allreduce` (n doubles) adds them up, and the fused
  update runs on the sum (`k_<op>_vec` kernels).

At the checks, the row terms (a few numbers) are combined and rank 0's
values decide termination, restarts and the penalty, so all ranks take the
same path and return the same solution. Without a CUDA-aware MPI the
exchange goes through pinned host memory; `-DAXOS_MPI_CUDA_AWARE` passes
device pointers to MPI instead. The Q products are repeated on every rank,
so the work that is divided is the work in A.

```bash
bash benchmarks/qp/build.sh mpi       # build/run_qp_mpi (MS-MPI SDK on Windows, mpicxx on Linux)
mpiexec -n 4 build/run_qp_mpi benchmarks/qp/data/large_axqp/HUBER_L.axqp --verbose 1
bash benchmarks/qp/run_mpi.sh 2       # large problems on 2 ranks against the 1-GPU CSV
bash benchmarks/qp/run_mpi.sh 2 benchmarks/qp/data/maros_meszaros benchmarks/qp/data/mm_axqp \
    benchmarks/qp/results/mm/axos-hprqp-gpu.csv mm_n2
```

`run_mpi.sh` solves each problem on K ranks, re-scores the solutions with
the common metric and compares them with the single-GPU run
(`compare_mpi.py`; output in `results/mpi/`).

Checks on the laptop (5 October 2026, tolerance 1e-6), against the
single-GPU runs of the table above. The laptop has one GPU, so all ranks
share it: these runs test the distributed algorithm, not multi-GPU speed.

| problems | ranks | optimal, 1 GPU | optimal, MPI | both | same iterations | largest objective difference |
|---|---|---|---|---|---|---|
| 14 large | 2 | 13 | 13 | 13 | 13 | 0 (13 digits) |
| 14 large | 4 | 13 | 13 | 13 | 13 | 0 (13 digits) |
| 138 Maros–Meszaros (60 s) | 2 | 123 | 114 | 114 | 103 | 1.6e-6 |

All large problems solved by both take the same number of iterations and
give the same objective to 13 digits. So do 103 of the 114
Maros–Meszaros problems solved by both. The other 11 are ill-conditioned
problems of the Q* family: QBORE3D, QFORPLAN, QGFRDXPN, QGROW7, QGROW22,
QSCFXM1–3, QSCORPIO, QSCRS8 and QSHARE2B. There, summing Aᵀy in another
order changes the iteration count by up to 28%. Both runs still meet the
tolerance, with objectives within 1.6e-6. The same happens between the CPU
and GPU builds in one process: QSHARE2B takes 204,301 iterations on the
CPU and 87,501 on the GPU.

After 2,000 iterations, one GPU and 2 ranks agree in the objective to 2e-7
or better (`run_mpi_fixed.sh`) on all 20 problems: these 11 and the 9
below. Every MPI answer reported optimal passes the common check.
In all 166 runs, the ranks' copies of x stay bit-identical. The CPU build
(`bash benchmarks/qp/build.sh mpi-cpu`) also agrees over 3 and 4 ranks,
even with more ranks than constraint rows.

The 9 Maros–Meszaros problems that only the single GPU solved ran out of
time with 2 ranks: CONT-101, CONT-200, EXDATA, LISWET3–6, STADAT2 and
STADAT3. On the shared GPU an MPI iteration of a small problem costs about
15x a single-GPU one. Every iteration waits for a host exchange, and there
are no CUDA graphs.

On the shared GPU the distributed runs are slower than one GPU. On the
large problems they take 2.2x the single-GPU time with 2 ranks and 3.1x
with 4 (geometric means): the ranks take turns on the one GPU, and every Q
product runs once per rank. With one GPU per rank, each holds 1/K of the
nonzeros of A, so larger models fit. Problems whose cost is in Q gain
nothing from more ranks (EQ_L has 3.5 M nonzeros in Q and 0.3 M in A). The
next step is a 2-D partition that also splits Q and the n-sized vectors,
as in D-PDLP and PDHCG-CQP.
