# AXOS optimization solver: roadmap

The goal is LP, then MILP, QP, MIQP and (convex) MINLP solvers built on
AXOS tensors, with CPU and CUDA backends. Vulkan is out of scope for now.

Each stage lists what it needs from earlier stages, what gets built, and
what "done" means. The stages are ordered by dependency, and the order
matters: MILP can't start until LP has a warm-startable method (dual
simplex), and QP reuses the LP interior-point code.

```
Stage 0  Csr + sparse kernels + cuDSS wrapper
   │
Stage 1  LP ── 1a model/MPS/presolve/scaling
   │          1b PDLP (GPU first-order)
   │          1c interior point (cuDSS) ───────────┐
   │          1d dual simplex (warm starts) ─┐      │
   │                                         │      │
Stage 2  MILP (branch & bound on 1d) ◄───────┘      │
Stage 3  QP  (interior point from 1c) ◄─────────────┘
Stage 4  MIQP (Stage 2 search + Stage 3 node solver)
Stage 5  NLP + convex MINLP
```

## Layout

```
src/sparse/          Csr<T, Idx = int32_t, Store>, CooBuilder, sparse kernels
src/solver/
  model.h            LP/MILP/QP problem model, status codes, tolerances
  io/mps.h           MPS reader (free and fixed format)
  presolve/          reductions + postsolve stack
  lp/pdlp.h          stage 1b
  lp/ipm.h           stage 1c
  lp/dual_simplex.h  stage 1d
  mip/               stage 2
  qp/                stage 3
  nlp/               stage 5
tests/test_sparse.cpp, tests/test_solver.cpp
```

The solver code is templated on the backend (`backend_of_t`), like the
tensorml ops. Dense vectors are `tensorET<1, T, Storage>`; the constraint
matrix is a `Csr<T, Idx, Store>`.

---

## Stage 0: sparse infrastructure

Needs: nothing new.

- `Csr<T, Idx = int32_t, Store = Cpu::HostStorage>` with
  `CsrCuda<T, Idx>` as an alias. It's 0-based, with column indices sorted
  within each row and no duplicates. The sparsity pattern is fixed once the
  matrix is built and the values can change.
- `CooBuilder`: triplets → CSR (sort, sum duplicates).
- Conversions: `from_dense` / `to_dense` (on device for CUDA),
  `transpose`, `append_rows`, and `values_view()` (a dense 1-D view of the
  stored values). Eigen conversions for tests only.
- cuSPARSE: handle in `GPUMemoryPool` (`CUSPARSE_WITH`), cached matrix
  descriptors, pooled workspace buffers, `cusparseSpMV_preprocess`.
- Kernels, CPU and CUDA:
  - A: `spmv`, `spmv_t` (via a cached Aᵀ), row/column scaling, row/column
    norms.
  - B: `spgemm`, `A·diag(d)·Aᵀ`.
  - C: cuDSS wrapper (analyse once, then refactor and solve many times),
    with a CPU fallback (sparse Cholesky / LDLᵀ, CHOLMOD-style supernodal
    later; simple left-looking first).

Done when: CPU matches Eigen and CUDA matches CPU on random and edge-case
matrices (empty rows, `nnz = 0`, one column, duplicate triplets), for
`int32_t`/`int64_t` and `float`/`double`. The cuDSS solve residual is at or
below 1e-10 on SPD test matrices.

**Status: done** (`tests/test_sparse.cpp`: 1125 checks on CPU and CUDA, clean
under compute-sanitizer memcheck and ASan/UBSan). What was built, and where it
differs from the list above:

- `src/sparse/`: `csr.h` (`Csr`, `CsrCuda`, `CooBuilder`), `sparse_cpu.h` and
  `sparse_cuda.h` (kernels), `sparse_ops.h` (user-facing `spmv`, `spmv_t`,
  `spmm`, `scale_rows_cols`, `row_norms`, `col_norms`, `spgemm`, `AdAt`),
  `sparse_ldl.h` (CPU LDLᵀ) and `cudss_solver.h` (cuDSS LDLᵀ).
  `src/shaders/sparse.cu` holds the scaling and norm kernels.
- Kernels live in `Sparse::Kernels<Backend>` (specialized for
  `Cpu::Backend` and `Cuda::Backend`), not as members of the tensorml backend
  structs, so sparse code doesn't pull in the tensorml headers.
- `SparseLdlt<T, Idx, Store>` has one interface for CPU and GPU. `analyze()`
  once per pattern, `factorize()` per value change (cuDSS refactorization),
  `solve()`. `Symmetry::SPD` needs positive pivots; `Symmetry::Symmetric`
  needs nonzero pivots only (no numerical pivoting, so it is meant for
  quasi-definite KKT systems). `inertia()` reports pivot signs.
- `from_dense` goes through the host; there is no on-device
  `cusparseDenseToSparse` path yet.
- The CPU LDLᵀ orders with approximate minimum degree
  (`src/sparse/amd_order.h`, no supervariable detection); the exact
  minimum-degree code is kept as `Ordering::ExactMinDegree` for comparison.
  cuDSS chooses its own ordering unless `Ordering::MinDegree` (its AMD) or
  `Ordering::NestedDissection` is requested.
- `Csr::transposed()` caches an explicit Aᵀ (extra memory); any mutable
  access to the values (`values_mut`, `values_view`, kernels that write)
  drops it. Scaling and norms on complex matrices run on the host.
- cuSPARSE `Csr2cscEx2` and SpGEMM are 32-bit-index only, so `int64_t`
  matrices transpose and multiply through the host on CUDA.

## Stage 1: LP

### 1a. Model, I/O, presolve, scaling

- Model: minimize cᵀx subject to l_r ≤ Ax ≤ u_r and l_c ≤ x ≤ u_c, with
  infinite bounds allowed. This single form covers equality, inequality and
  free rows without converting between them.
- MPS reader. Status codes: optimal, infeasible, unbounded, iteration or
  time limit, numerical error.
- Presolve (first pass): empty and singleton rows/columns, fixed columns,
  duplicate rows, bound tightening. Every reduction pushes an entry onto a
  postsolve stack, so primal and dual solutions can be mapped back.
- Scaling: Ruiz equilibration and Pock-Chambolle, done on
  `Csr::values_view()`.

### 1b. PDLP (first-order, GPU first)

- A restarted primal-dual hybrid gradient method (the PDLP / cuPDLP
  scheme): adaptive step size, primal weight updates, restarts, and
  infeasibility detection.
- The whole iteration is `spmv` / `spmv_t` plus elementwise vector updates,
  so it is Stage 0-A plus existing dense code.
- Why first: it is the fastest way to a working GPU LP solver and a
  benchmark target. Its accuracy is moderate (about 1e-4 to 1e-6), and it
  can't be warm-started usefully, so it doesn't replace 1c or 1d.

**Status of 1a and 1b: done** (`tests/test_solver.cpp`: 168 checks on CPU,
230 on CPU + CUDA (including the interior-point method and real netlib models); clean under compute-sanitizer memcheck and ASan/UBSan).

- 1a, in `src/solver/`: `model.h` (`LpProblem`, `LpSolution`, `Status`,
  `SolverOptions`, `evaluate_solution`), `io/mps.h` (reader and writer),
  `presolve/presolve.h` (empty rows and columns, fixed columns, singleton
  rows, duplicate/parallel rows, each with a postsolve that also recovers
  duals), `scaling.h` (Ruiz + Pock-Chambolle, done on the host).
  Not done: bound tightening, dominated columns, doubleton equations.
- 1b, in `src/solver/lp/`: `pdlp.h` (adaptive-step restarted PDHG with
  KKT-based restarts, primal-weight updates and infeasibility/unboundedness
  detection), `pdlp_kernels.h` (host+device functors run by `Parallel<Backend>`)
  and `pdlp_cuda.h` (the GPU executor). `solve_lp<Store>()` chains presolve,
  PDLP and postsolve; the same code runs on CPU and CUDA.
  Not done: feasibility polishing, a fixed-step-size mode, a device-side
  step-size loop (there is one host sync per iteration).
- Termination is checked on the original (unscaled) problem with the
  tolerances in `SolverOptions`; the returned `LpSolution` is always
  re-evaluated on the original problem, so its residuals are exact.
- Benchmarks (`make benchmark_lp`, `benchmarks/solver/`): see RESULTS.md
  there. On 2M-nonzero packing LPs the GPU is about 6-8x faster than the
  16-thread CPU at 1e-6.

### 1c. Interior point

- Mehrotra predictor-corrector. Normal equations `A·D·Aᵀ` go through cuDSS
  on GPU (Stage 0-B/C), with a switch to the augmented system for dense
  columns.
- Gives high-accuracy LP solutions (1e-8), and it is the base for QP
  (Stage 3).

**Status of 1c: done, with the limits listed here** (`src/solver/lp/ipm.h`,
`ipm_kernels.h`). Mehrotra predictor-corrector on a regularized quasi-definite
augmented KKT system, solved by `SparseLdlt` (cuDSS on the GPU, up-looking
LDL^T with AMD on the CPU). The KKT pattern is fixed, so the symbolic
analysis happens once and each iteration only rewrites the diagonal and
refactorizes; each solve is iteratively refined against the unregularized
system; every complementarity pair is kept centered (Ipopt's safeguard).
`solve_lp` runs it with `SolverOptions::method = LpMethod::Ipm`, or
`LpMethod::Auto` (see below).

- Real models: on the 25 netlib instances in the HiGHS repository, the
  interior-point method solves 18 of the 19 feasible ones to about 1e-9
  relative objective error in 0.1 ms to 0.4 s (`greenbea` is the exception,
  see below), identical iteration counts on CPU and GPU. The 6 infeasible
  ones are not solved (no certificates); `Auto` recovers 3 of them through
  PDLP. See benchmarks/solver/RESULTS.md.
- No infeasibility/unboundedness certificates; crossover only as described in 1d.
- Bad scaling: `greenbea` has optimal variables around 3e8 and is not
  solved (see the crossover note below). Termination requires an
  objective-error bound (`LpSolution::error_bound` = gap + ||dual res|| ||x|| +
  ||primal res|| ||y||) so such a run cannot be reported as optimal; before
  that check it was, with a 0.13% error, at a 1e-6 tolerance.
- Fill-heavy problems are slow: on random sparse graphs and packing LPs
  the augmented system factors into nearly dense factors. `mcf10k`: 7
  iterations took 121 s with the scalar factorization; with the multifrontal
  one the whole solve (16 iterations) takes 21 s (8 threads), GPU 9 s, HiGHS
  IPM 0.8 s using a different, iterative approach; packing LPs are still out
  of reach. `Auto` therefore predicts the factorization cost from the symbolic
  analysis and lets the dual simplex go first when it is large (mcf10k: 1.0 s).
  See "Numerical linear algebra status".
- Normal equations: when A has no free columns and no dense columns, the KKT system
  is solved through S = A D^-1 A^T + E (m x m, multifrontal Cholesky) instead of the
  augmented system, and the multifrontal code merges identical leaf siblings into
  one supernode. transport700 (1400 x 490k): IPM 14 s -> 1.1 s.
- `greenbea` (optimal variables ~3e8) needs a crossover: HiGHS' own
  interior-point method without crossover fails on it the same way (status
  Unknown, 0.07% off), and only crossover + simplex reach the optimum. The
  static regularization level, the starting scale (M = 1e2..1e8) and dynamic
  pivot regularization were tried and ruled out. `Auto` reaches the optimum
  through the dual simplex (1d), cold started: a crossover start from the failed
  IPM iterate needed more iterations than a cold start.

### 1d. Dual simplex

**Status: done** (`src/solver/lp/simplex.h`, `basis_lu.h`; `LpMethod::Simplex`).
CPU bounded-variable dual simplex:

- Basis: sparse LU (Gilbert-Peierls, threshold pivoting, columns ordered by
  count) with product-form eta updates; FTRAN/BTRAN are hypersparse (symbolic
  DFS reach over L, U and their row-wise copies) when the vector is sparse. The
  factorization is redone every 100-500 updates or when the eta file outgrows
  the factors. (Forrest-Tomlin was not needed to reach the numbers below.)
- Pricing: dual steepest edge with the exact weight of the leaving row taken
  from BTRAN and a rebuild of all weights when the updated one drifts (the weights
  collapsing to their floor was the cause of 3-4x too many iterations).
  Leaving-row selection keeps a candidate list (exact, O(1) typical).
- Ratio test: bound flipping with Harris tolerances; the tolerance widens while the
  best pivot is small relative to the row (tiny pivots blew up B^-1).
- Phase 1: artificial boxed bounds; cost perturbation on stalling; cost
  shifting for small dual infeasibilities; a Devex primal simplex removes
  the remaining dual infeasibilities at the end (so no small dual infeasibility
  is accepted any more).
- Warm start from a `SimplexBasis` (used by MILP later), and a crossover start
  (`SolverOptions::crossover`, `solve(..., start)`): the basis is picked from an
  interior point by the ratio distance-to-bound / |reduced cost| with dependent
  columns replaced. It gives a basic solution but is not yet faster than a cold
  start (25fv47: 1562 iterations vs 2107 cold): the basis from the IPM point is
  ill-conditioned (max primal infeasibility 1e7 on 25fv47). A real crossover
  (primal/dual push phases) is not done.
- Not done: Forrest-Tomlin updates, partial/hyper-sparse pricing beyond the
  CHUZR list, dual phase 1 with subproblem approach, parallelism. Per-iteration
  cost on large network problems is about 3x HiGHS (mcf50k: 180 us vs 55 us).
- Netlib: 25/25 (19 optimal to 1e-7..1e-9, 6 infeasible detected); times within
  1.2-2.4x of HiGHS except greenbea (5.6x). Larger instances: transport700
  2.0 s (HiGHS 1.5 s), mcf10k 0.8 s (0.4 s), mcf50k 18 s (4.8 s).

Done when (Stage 1):
- IPM and dual simplex solve all of netlib to 1e-8 relative tolerance. (Dual simplex: 25/25; IPM: all feasible ones but greenbea.)
- PDLP solves netlib to 1e-4, and its GPU run times on the larger instances
  are recorded against cuPDLP and HiGHS.
- Primal and dual solutions after postsolve pass an independent KKT check.

## Numerical linear algebra status (the KKT machinery)

What exists and is tested (`src/sparse/`): CSR matrices, SpMV/SpMM/SpGEMM,
transposes, scaling and norms on CPU and CUDA; a sparse symmetric LDL^T with
`analyze` / `factorize` (refactorization for changing values) / `solve` /
`inertia` on the CPU (up-looking, AMD ordering) and the GPU (cuDSS);
Symmetry::Symmetric works for quasi-definite KKT systems.

What was added after the first version: a multifrontal LDL^T on relaxed
supernodes for the CPU (`multifrontal_ldl.h`, `dense_ldl.h`; assembly-tree
parallelism plus OpenMP dense updates, dynamic per-pivot regularization),
chosen automatically for factors above 200k entries. It matches the scalar
factorization exactly (same fill, inertia, solutions) and is 3-4x faster than
Eigen's LDLT on grids; on the `mcf10k` interior-point run it cut the
per-iteration cost from about 17 s to 0.9 s.

What is not complete:
- The factorization is flop-bound by the ordering: on random graphs (min-cost
  flow, packing LPs) AMD leaves a very large dense top separator, so a KKT
  factorization is 10^11 flops (0.9 s on 8 cores for `mcf10k`) and problems
  that are 10x larger are out of reach for a direct method. Better orderings
  (a real nested dissection / METIS-class ordering for the CPU) and an
  iterative alternative (preconditioned CG on the normal equations, as HiGHS'
  IPX does) are the remaining options; cuDSS' nested dissection was not a
  consistent win.
- AMD has no supervariable detection and the symbolic phase (two elimination
  tree passes, structure merging) is 3x slower than Eigen's (147 ms vs 41 ms at
  160k rows).
- Tree parallelism is a static split into small subtrees plus a serial top;
  the top-of-tree work is parallel only inside the dense kernels.
- The interior-point method still uses a static regularization plus
  refinement; the LDL^T supports dynamic pivot regularization
  (`set_pivot_regularization`, tested) but the IPM does not use it (it did not
  help `greenbea`, see 1c).
- Not verified: thread-race checking of the parallel factorization under
  ThreadSanitizer (libgomp is not TSan-aware, so it reports false positives
  that swamp the output); ASan/UBSan and the test suite are clean.
- cuDSS is fast for the numeric factorization, but its analysis phase is slow
  (100+ ms), which dominates small problems.

## Stage 2: MILP

Needs: 1a presolve and 1d dual simplex (node LPs); 1b optionally for the
root LP on large instances.

- Branch and bound: node queue with a best-bound / depth-first mix, warm
  start from the parent basis, reduced-cost fixing, pruning on the incumbent.
- Branching: most-infeasible first, then pseudocost, then reliability
  branching.
- MIP presolve: integer bound tightening, coefficient tightening,
  clique detection.
- Cuts at the root: Gomory mixed-integer (from the simplex tableau), MIR,
  knapsack cover. Add rows with `Csr::append_rows`.
- Primal heuristics: rounding, diving, feasibility pump, then RINS.
  GPU-friendly heuristics (feasibility-jump style) can run alongside the
  tree search.
- Parallel tree search with OpenMP (a shared node queue first).

Done when: the MIP gap, number of nodes and time are reported on a subset of
MIPLIB 2017 "easy" instances, and a fixed list of small instances solves to
proven optimality. Every incumbent is checked for feasibility on the
original (non-presolved) model.

Status (October 2026): implemented in `src/solver/milp/`. Everything in the
list above is built except clique detection, knapsack covers as their own
separator (c-MIR covers many of them) and the parallel tree. Additions:
- Double probing at the root.
- c-MIR with variable-bound substitution and row aggregation.
- The objective feasibility pump and RENS.
- A GPU side (milp_cuda.h): frontier-based batched probing, feasibility-jump
  walkers, and batched PDHG Lagrangian bounds for strong branching on large
  LPs.

Results against HiGHS, SCIP and CBC: `benchmarks/milp/README.md`. Missing
next, by the benchmark: flow-cover / path cuts (fixed-charge network
instances), conflict analysis, stronger MIP presolve (dominated columns,
cliques), and a faster node LP.

## Stage 3: QP (convex)

Needs: 1c interior point, Stage 0-C (factorization).

Status (2026-09-30), in `qp/`, benchmarked in `benchmarks/qp/` against
HPR-QP.jl, PDHCG.jl, OSQP, Clarabel, HiGHS, PIQP and SCS:

- Done: first-order GPU methods, as a matrix-free complement to the IPM for
  large problems. HPR-QP (`hpr_qp.h`, Halpern Peaceman–Rachford) and PDHCG
  (`pdhcg.h`) on CPU (OpenMP) and GPU (own NVRTC kernels: SpMV with fused
  epilogues, CUDA graphs). On Maros–Meszaros at 1e-6 AXOS HPR-QP solves 120
  of 138 on a laptop GPU, 6x faster than the authors' HPR-QP.jl on the same
  problems.
- Done, first version: the IPM extended to QP (`qp_ipm.h`, CPU, augmented
  system on the multifrontal LDLᵀ), and `QpMethod::Auto` (IPM when the
  factorization is cheap, else HPR-QP, GPU for large problems).
- Next: proximal (decreasing) regularization in the QP IPM, as in PIQP.
  The static 1e-8 regularization fails on the degenerate LISWET family. Then
  the GPU IPM on cuDSS, and a convexity check that rejects non-PSD Q with a
  clear status.
- Optional: an ADMM solver (OSQP-style) for problems where moderate
  accuracy is enough. It reuses a single KKT factorization.

Done when: the Maros-Meszaros test set is solved to 1e-8, and the number
solved and time taken are compared with OSQP / Clarabel.

## Stage 4: MIQP

Needs: the Stage 2 search, the Stage 3 node solver.

- Plug the QP solver into the branch-and-bound code as the node solver.
  The Stage 2 search, branching and heuristics are written against a node
  solver interface so this is a swap, not a copy.
- Weakness to accept at first: interior-point nodes can't be warm-started,
  so each node costs a full solve. A dual active-set QP method fixes that
  later.

## Stage 5: NLP and convex MINLP

Needs: Stages 3 and 4, and tensorML autograd.

- NLP interior point (IPOPT-style: barrier method, filter line search).
  Gradients come from tensorML reverse mode. Hessians start as L-BFGS,
  then exact Hessians once autograd can do second derivatives
  (forward-over-reverse).
- Convex MINLP: NLP-based branch and bound first, then outer approximation
  (MILP master problem from Stage 2 plus NLP subproblems).
- Out of scope: nonconvex MINLP (spatial branch and bound, global
  optimality).

Done when: a chosen convex subset of MINLPLib is solved, with results
checked against published optimal values.

---

## Cross-cutting

- Tolerances (primal/dual feasibility, optimality gap, integrality) and
  iteration/time limits live in one options struct shared by all solvers.
- Every solver returns a status and verifies its solution against the
  original model before reporting "optimal".
- Benchmarks go in `benchmarks/solver/` with fixed instance lists, so runs
  can be compared across commits. (similar to the suite, which has some good features (idempotence, logging, ...))
- Test instances (netlib, MIPLIB, Maros-Meszaros, MINLPLib) are downloaded
  by a script, not committed.
