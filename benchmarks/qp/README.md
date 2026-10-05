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
