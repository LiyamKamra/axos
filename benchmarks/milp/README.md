# MILP benchmark

AXOS's branch-and-cut solver (`src/solver/milp/`) against HiGHS, SCIP and
CBC on MIPLIB 2017 instances. Each solver reads the same MPS file and runs
single-threaded with the same time limit and relative gap (1e-4, the MIPLIB
criterion), in its own process per instance. Every reported solution is
checked on the original model before it counts.

## What AXOS runs

The techniques of the SIH problem statement's MILP section (sih.md §2), with
the tree on the CPU and the arithmetic-heavy parts on the GPU:

| technique | where | notes |
|---|---|---|
| presolve | `presolve/presolve.h`, `solve_milp.h` | LP reductions, MIP coefficient tightening |
| domain propagation | `propagate.h` (CPU), `milp_kernels.cuh` `prop_*` (GPU) | activity based, integer rounding, infeasibility detection |
| double probing (GPU presolve) | `probing.h` (CPU), `milp_cuda.h` `GpuProber` | 256 probes at once, frontier propagation (changed constraints only), merge on the device |
| cutting planes | `cuts.h`, `cmir.h` | Gomory mixed-integer from the tableau; c-MIR with variable-bound substitution and aggregation; inactive cuts dropped |
| branching | `bnb.h` | reliability pseudocost branching with strong branching |
| GPU-batched LP relaxations | `milp_cuda.h` `GpuBatchLp` | children of 16 candidates by batched PDHG; Lagrangian bounds are valid, so they prune |
| PDLP-class approximate LP on the GPU | `bnb.h` `gpu_root_lp` | HPR (the QP solver with Q = 0) for the root LP of large models; seeds the heuristics |
| feasibility jump | `heuristics.h` (CPU), `milp_cuda.h` `GpuFeasibilityJump` | 64 GPU walkers in a second host thread, racing the CPU |
| feasibility pump | `bnb.h` | objective FP, warm-started primal simplex (cost changes) |
| fix-and-propagate, rounding, diving | `heuristics.h`, `bnb.h` | |
| RENS, RINS | `bnb.h` | sub-MIPs through `solve_milp`, cutoff row |

The simplex (`lp/simplex.h`) gained what the tree needs: bound-change
re-solves from a parent basis, cost-change re-solves (primal simplex), the
tableau rows for cuts, and protection against cycling.

## Running it

```sh
bash benchmarks/milp/build.sh              # build/run_milp (gpu build; "cpu" for none)
pip install highspy pyscipopt "pulp<3"     # comparators (PuLP 2.x ships cbc)
SOLVERS="highs scip cbc" bash benchmarks/milp/run_all.sh
bash benchmarks/milp/run_axos_final.sh     # AXOS CPU and GPU, both instance sets
python benchmarks/milp/report.py benchmarks/milp/results/miplib50 --json results/miplib50.json
```

`build/test_gpu` measures each GPU component against its CPU counterpart.
CBC must not get `-threads 1`: with it, the bundled CBC 2.10.3 ignores the
time limit.

Instances: `data/set50.txt`, the 50 smallest "easy" instances of the MIPLIB
2017 benchmark set (up to 12,528 nonzeros), and `data/set_large.txt`, 16
larger easy instances (51k–392k nonzeros), from miplib.zib.de. Reference
values: `data/miplib2017-v31.solu`.

## Results (5 October 2026)

Laptop: Core i7-12650H, RTX 3050 Laptop 4 GB. 60 s per instance, one CPU
thread per solver, relative gap 1e-4. Solved = optimal (or infeasible) and
matching the MIPLIB reference; primal gap = |obj − best known| / max(|obj|,
|best known|) at the end, 1 without a solution.

| solver | 50 instances: solved | with a solution | mean primal gap | 16 larger: solved | with a solution | mean primal gap |
|---|---|---|---|---|---|---|
| HiGHS 1.15.1 | 15 | 47 | 0.089 | 4 | 15 | 0.327 |
| SCIP 10.0 | 16 | 46 | 0.135 | 3 | 11 | 0.520 |
| CBC 2.10.3 | 7 | 46 | 0.112 | 3 | 9 | 0.506 |
| **AXOS (CPU + GPU)** | 3 | 44 | 0.196 | 0 | **12** | **0.468** |
| AXOS (CPU only) | 3 | 44 | 0.209 | 0 | 9 | 0.573 |

- On the 16 larger instances, AXOS with its GPU components finds verified
  solutions on more instances than SCIP and CBC, with a lower mean primal
  gap. Per instance it beats SCIP on 6 and loses on 4.
- Best of all four solvers on some instances (60 s, minimization):
  - mzzv11: −8,800 against HiGHS −300 and SCIP 0, CBC none.
  - comp07-2idx: 292 against 504 and 823.
  - cvs16r128-89: −92 against CBC −89, SCIP −67 and HiGHS −3.
- On the 50 smaller instances the established solvers lead clearly. AXOS
  proves 3 optimal, against 15 for HiGHS, 16 for SCIP and 7 for CBC.
- Every AXOS solution passes the independent check (`verify_sol.py`), and no
  solver made a wrong optimality or infeasibility claim.

GPU components against their CPU counterparts (`run_gpu_components.sh`, idle
machine):
- Double probing is faster on 12 of the 16 larger instances. The median is
  2.2× more probes per second, up to 5.8×, and 4–4.7× on the two largest.
- The LP relaxation by HPR on the GPU (tolerance 1e-4) is faster than the
  AXOS dual simplex in the median over the 7 of 8 instances both solve
  (1.6×), and up to 99× faster (cod105: 13.3 s against 0.13 s). It solves
  rail01, where the simplex runs out of time at 30 s.
- The batched PDHG bounds are always valid (never above the exact child
  LP). After 400 iterations they recover 78% of the exact strong-branching
  gain on neos5, but little on the larger models.

The full report with per-instance tables is `results/milp_report.html`.
