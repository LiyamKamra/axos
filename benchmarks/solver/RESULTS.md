# LP solver benchmark results

AXOS's LP solvers (`solve_lp`: presolve + scaling on) against HiGHS on
(1) generated structured LPs and (2) real netlib models. Machine: 16
hardware threads, RTX 4060 Laptop GPU, on AC power, double precision. HiGHS
is the version bundled with SciPy 1.18.1 (generated instances, run through
`scipy.optimize.linprog`) and highspy 1.15.1 (netlib references), at its
default tolerances (about 1e-7). Times are wall-clock ms including presolve
and scaling. `(limit)` is a time/iteration limit, `(killed)` a HiGHS IPM
solve that ignored its time limit and was killed after 120 s.

Methods: `pdlp_*_1e-4` / `1e-6` are PDLP at those relative tolerances;
`ipm_*_1e-8` is the interior-point method at 1e-8; `auto_cpu` is
interior point first with PDLP as fallback (tolerance 1e-6).

Regenerate: `make benchmark_lp` (generated instances, exported to
`build/lp_bench` so both sides solve identical problems; `report.py`
prints the table) and, for netlib, `benchmarks/solver/fetch_netlib.sh`,
`build/run_mps`, `netlib_reference.py`, `netlib_report.py`.

## 1. Generated instances

`transport100/300/700` (dense transportation problems), `mcf10k/50k`
(min-cost flow on a random graph), `packing20k/200k` (sparse packing
LPs; 200k has 2M nonzeros). PDLP and HiGHS were measured before the
kernel fusion was re-timed; the interior-point rows were only run where it
finishes in reasonable time.

```
instance           pdlp_cpu_1e-4    pdlp_cuda_1e-4     pdlp_cpu_1e-6    pdlp_cuda_1e-6      ipm_cpu_1e-8     ipm_cuda_1e-8        highs_auto          highs_ds         highs_ipm
mcf10k                       112                60               264               178    121575 (limit)              8855               444               437               805
mcf50k                      1028               226              5301               842                 -                 -              4976              4654              6008
packing20k                    68                72               996               565    343346 (limit)    126350 (limit)     60106 (limit)     60046 (limit)     61287 (limit)
packing200k                 5718              1097             33101              4825                 -                 -     60381 (limit)     60368 (limit)   120000 (killed)
transport100                  20               246                24                24                24               308                37                37                62
transport300                 159               150               361               169               489               419               262               266               519
transport700                8054               925              6669               934              5627              3386              1874              1862              3369

relative objective difference vs HiGHS (dual simplex / IPM optimum):
instance         pdlp_cpu_1e-4  pdlp_cuda_1e-4   pdlp_cpu_1e-6  pdlp_cuda_1e-6    ipm_cpu_1e-8   ipm_cuda_1e-8
mcf10k                 1.1e-06         1.3e-06         5.8e-08         5.1e-08               -         0.0e+00
mcf50k                 1.5e-06         1.5e-06         1.4e-08         1.4e-08               -               -
packing20k                   -               -               -               -               -               -
packing200k                  -               -               -               -               -               -
transport100           1.4e-06         2.9e-06         4.7e-08         1.5e-07         5.3e-10         5.3e-10
transport300           2.1e-06         6.6e-06         1.2e-08         9.1e-08         0.0e+00         0.0e+00
transport700           3.6e-08         1.8e-07         2.0e-09         3.5e-08         4.0e-10         4.0e-10
```

- Accuracy: PDLP agrees with HiGHS to about 1e-6 relative at 1e-4 and
  1e-8..1e-7 at 1e-6; the interior-point method to 1e-9.
- Packing LPs: neither HiGHS (simplex or IPM) nor a direct-factorization
  interior-point method finishes packing20k, while PDLP solves it in
  0.07-1 s and the 2M-nonzero one in 1-5 s on the GPU. This is the regime
  first-order methods are for.
- Network / transportation LPs: on the GPU PDLP is 2-6x faster than HiGHS
  from about 500k nonzeros (mcf50k: 0.84 s vs 4.7 s; transport700: 0.93 s
  vs 1.9 s); on the CPU HiGHS' simplex wins on transport700 (1.9 s vs 6.7 s).
- Interior point: fine on structured problems (transport700: 5.6 s CPU,
  3.4 s GPU, about HiGHS' own IPM at 3.4 s), but on random sparse graphs
  the KKT factors fill in almost completely: mcf10k takes 7 iterations in
  121 s on the CPU and 9 s on the GPU (HiGHS IPM: 0.8 s with a different,
  iterative method), packing20k does not finish. Sparse factorization
  speed is the limiting factor (see the ROADMAP).
- Small problems (< 100k nonzeros): CPU and GPU take about the same time;
  the GPU pays a one-time context startup (the 246 ms in the first row).

## 2. Netlib (25 models from the HiGHS repository)

19 are feasible (`gt2` is an integer model; its LP relaxation is used) and
6 are netlib's deliberately infeasible ones (bgetam, forest6, galenet,
klein1, refinery, vol1). Limits: 30 s (PDLP), 60 s (IPM), 20 s (Auto).

```
model         HiGHS    verdict |         auto_cpu          ipm_cpu         ipm_cuda         pdlp_cpu        pdlp_cuda
25fv47        104.1    optimal |             51.8             55.3            377.5           1039.8           2979.0
80bau3b        79.3    optimal |            139.8            111.8            284.2           9903.9           6610.9
adlittle        0.9    optimal |              0.6              0.6             15.4             17.0            406.9
afiro           0.3    optimal |              0.2              0.2             12.0              0.7             14.6
avgas           0.1    optimal |              0.2              0.2             11.7              0.2              5.0
bgetam          1.7 infeasible |    20084.2 limit       79.0 limit     326.8 numerr    22551.0 limit    30002.4 limit
blending        0.1    optimal |              0.1              0.1              8.6              0.3              5.1
e226            4.5    optimal |              4.4              4.5             37.1            155.7           1267.2
etamacro        5.8    optimal |              9.4             12.6             67.8             90.4            901.7
forest6         0.4 infeasible |      29.6 infeas       0.9 numerr      25.9 numerr      27.9 infeas     402.1 infeas
galenet         0.0 infeasible |       0.5 infeas       0.4 numerr      25.6 numerr       0.1 infeas       2.7 infeas
greenbea      140.8    optimal |    20379.7 limit     402.6 numerr     475.2 numerr    30001.4 limit    30001.4 limit
gt2             0.6    optimal |              0.6              0.5             12.3              0.4              5.2
israel          2.1    optimal |              2.9              4.3             28.6             25.4            308.4
klein1          1.2 infeasible |    11375.4 limit      12.0 numerr     202.8 numerr    11329.1 limit    30001.6 limit
perold         36.7    optimal |             61.0             64.9            162.7    30000.5 limit    30002.4 limit
refinery        2.6 infeasible |     445.4 infeas       9.5 numerr     100.0 numerr     432.6 infeas    4702.8 infeas
scrs8           5.4    optimal |              7.1              7.0             58.2            606.0           4088.5
sctest          0.1    optimal |              0.1              0.2              9.6              0.5              9.7
shell           6.9    optimal |             17.8             17.9            140.7            127.5            515.3
stair           8.3    optimal |              6.7              6.9             45.1            602.0           4379.4
standata        2.5    optimal |              4.1              4.3             38.0             15.3             99.0
standgub        2.5    optimal |              4.4              4.5             36.7             16.2             74.9
standmps        2.9    optimal |              7.6              8.2             55.4             29.0            190.9
vol1            5.6 infeasible |    20007.3 limit       7.1 numerr      77.5 numerr    20337.0 limit    30001.7 limit

solved correctly (optimal within 1e-4 rel., or infeasible), out of 19 feasible + 6 infeasible:
  auto_cpu: 21, ipm_cpu: 18, ipm_cuda: 18, pdlp_cpu: 20, pdlp_cuda: 20

relative objective error vs HiGHS (optimal runs only):
model      |         auto_cpu          ipm_cpu         ipm_cuda         pdlp_cpu        pdlp_cuda
25fv47     |          4.4e-08          2.2e-10          2.2e-10          1.6e-06          9.7e-07
80bau3b    |          6.3e-07          3.9e-09          3.9e-09          1.5e-06          1.3e-07
adlittle   |          1.5e-07          7.5e-10          7.5e-10          1.5e-06          1.2e-06
afiro      |          1.0e-06          5.1e-09          5.1e-09          2.0e-07          2.0e-07
avgas      |          4.6e-07          2.3e-09          2.3e-09          9.5e-07          9.5e-07
blending   |          1.4e-08          6.2e-11          6.2e-11          0.0e+00          0.0e+00
e226       |          6.4e-07          3.2e-09          3.2e-09          1.7e-07          1.6e-07
etamacro   |          1.3e-07          3.8e-09          3.8e-09          1.7e-08          4.9e-08
greenbea   |                -                -                -                -                -
gt2        |          3.1e-07          1.6e-09          1.6e-09          0.0e+00          0.0e+00
israel     |          1.2e-06          1.1e-10          1.1e-10          3.6e-07          3.6e-07
perold     |          1.2e-06          0.0e+00          0.0e+00                -                -
scrs8      |          1.4e-08          1.4e-08          1.4e-08          6.6e-08          1.0e-07
sctest     |          1.5e-11          0.0e+00          0.0e+00          7.3e-08          7.3e-08
shell      |          1.2e-08          1.2e-08          1.7e-10          4.4e-07          2.2e-09
stair      |          9.0e-08          4.4e-10          4.4e-10          2.5e-09          7.0e-07
standata   |          2.0e-08          7.9e-11          7.9e-11          1.1e-09          9.0e-09
standgub   |          1.2e-09          1.2e-09          1.2e-09          1.1e-09          1.5e-09
standmps   |          1.4e-06          0.0e+00          0.0e+00          3.7e-08          6.5e-08
```

- Interior point: 18 of the 19 feasible models, to about 1e-9 relative
  objective error, in 0.1 ms to 0.4 s, with the same iteration counts on the
  CPU and the GPU (the GPU is slower here: the models are small and cuDSS'
  analysis and launch overhead dominate). It does not detect infeasibility.
- PDLP: 17 of 19 (perold and greenbea hit the 30 s limit) and 3 of the 6
  infeasible models (forest6, galenet, refinery); the other three
  (bgetam, klein1, vol1) are not certified within the limit. It needs 10-1000x
  more time than the interior-point method on these small, ill-conditioned
  models.
- Auto solves the same 18 feasible models as the interior-point method and
  reports the same 3 infeasible ones through PDLP.
- `greenbea` is unsolved by every method (optimal variables near 3e8).
  Earlier versions reported it as optimal with a 0.13% objective error at
  a 1e-6 tolerance, because the residual tolerances are relative to ||c||;
  the `LpSolution::error_bound` check now rejects such answers.

## Mittelmann LPopt/LPfeas subset (auto, CPU, 300 s limit, eps 1e-6)

16 of the smaller instances from plato.asu.edu/ftp/lptestset (integrality relaxed).
Raw output: `build/mittelmann/auto.log`. Not run: the large instances (>9 MB compressed) and the fome/pds/nug/network/rail directories.

| instance | time (s) | status | objective |
|---|---|---|---|
| brazil3 | 0.6 | optimal | 2.0 |
| chromaticindex1024-7 | 0.7 | optimal | 3.0 |
| datt256_lp | 4.4 | optimal | 256 |
| ex10 | 83.0 | optimal | 100 |
| graph40-40 | 3.8 | optimal | -300 |
| irish-electricity | 300 | time limit | (primal infeasible 0.88) |
| neos-5052403-cygnet | 14.1 | optimal | 179.50043 |
| neos-5251015 | 5.5 | optimal | 0.1058 |
| physiciansched3-3 | 217.6 | optimal | 2.4326948e6 |
| qap15 | 12.0 | optimal | 1040.995 |
| rmine15 | 305 | time limit | gap 6e-4 |
| s100 | 300 | time limit | gap 0.49 |
| s250r10 | 277 | reported optimal, but residuals pres 7e-3 / gap 2e-2: a dual-simplex bug (overflowing steepest-edge weights made row selection skip violated rows), fixed; now time limit | -0.17269 |
| savsched1 | 19.6 | optimal | 217.40357 |
| supportcase10 | 95.4 | optimal | 3.3839 |
| woodlands09 | 15.3 | optimal | ~0 |

## Mini benchmarks after the weak-spot work (CPU, 8 threads, on battery, eps 1e-8 unless noted)

Simplex (HiGHS dual simplex with presolve, 1 thread, for comparison), milliseconds:

```
model        ours   HiGHS      | instance     ours (Auto)   HiGHS
25fv47        108      91      | transport100      27         27
80bau3b       184      77      | transport300     300        176
perold         87      37      | transport700    1030       1419
greenbea      820     147      | mcf10k          1200        384
                               | mcf50k         23400       4830
```

- Dual simplex: 3-4x fewer iterations than before (dual steepest-edge weights
  were collapsing), 9x faster per iteration on large sparse models (hypersparse
  solves): savsched1 39k iterations in 20 s (HiGHS: 34k), neos-5251015 solved in
  5.7 s (HiGHS did not finish in 60 s).
- Interior point: the normal-equations path and the sibling supernode merge take
  transport700 from 14 s to 1.1 s; mcf10k (fill-heavy) is 21 s, so Auto lets the
  simplex go first there (1.0 s).
- PDLP at 1e-6, iterations, adaptive+averaging vs reflected Halpern: 25fv47
  78k -> 57k, 80bau3b 264k -> 129k, e226 36k -> 19k, israel 6.3k -> 4.3k, mcf10k
  2.8k -> 2.2k, transport300 1.9k -> 3.5k, packing20k 9.2k -> 10.7k.
- Not solved within 60 s by us or HiGHS: chromaticindex1024-7, qap15, graph40-40,
  datt256_lp, supportcase10, physiciansched3-3, woodlands09 (simplex);
  the IPM path solves several of them (see the Mittelmann table).
- Known gaps: greenbea (5.6x HiGHS), mcf50k (about 5x), crossover from an
  interior point is not faster than a cold simplex, no Forrest-Tomlin update.

## GPU: PDLP on CUDA vs HiGHS 1.15.1 built with cuPDLP-C (RTX 4060 Laptop)

HiGHS built from source with `-DCUPDLP_GPU=ON`, `--solver pdlp --presolve on
--run_crossover off` (its accuracy option had no visible effect: identical
iteration counts at 1e-4 and 1e-6). Ours: `run_mps --method pdlp` (Halpern), times in ms.

```
instance      ours cuda 1e-4   ours cuda 1e-6   ours cpu 1e-6   HiGHS cuPDLP-C
transport300         408              387            353              630
transport700         818              957           6881             1780
mcf10k               272              302            276              750
mcf50k               401              931           4148             1910
packing20k           310              953           1183             2670
packing200k         1247             5130          47391            19250
```

## Dense factorization kernel (CPU multifrontal)

`dense::syrk_d` is now a packed, register-blocked GEMM-style kernel (AVX-512
vectors, 24x8 tile, KC 256, column blocks as independent OpenMP tasks) and the
panel step computes L21 in parallel row chunks (64-wide panels, 128 for fronts of
3000+). Whole partial factorization of a dense front, GFLOPS (8 cores):

```
front (fs x ns)     before    after
1 thread            ~27..47    49
1000 x 1000          123       126
2000 x 2000          190       212..227
4000 x 4000          116       255..269
```

End to end (IPM, CPU, eps 1e-8): mcf10k 21 s -> 13.7 s, ex10 83 s -> 51 s (about
310 GFLOPS sustained, near this machine's peak), supportcase10 95 s -> 67 s.
Update matrices are pooled between supernodes. cuDSS on the GPU takes 7.3 s (mcf10k),
47 s (ex10), 48 s (supportcase10).

## PDLP polishing (`SolverOptions::pdlp_polish`, off by default)

PDLP to 1e-4, then a primal feasibility solve (objective dropped) and a dual feasibility
solve (b and bounds zeroed), each from the current point. Mixed results at a 1e-6
to 1e-8 target: 80bau3b 5.5 s -> 2.5 s, shell 117 -> 80 ms, but qap15 0.56 -> 1.2 s,
stair 0.37 -> 1.3 s, e226 and 25fv47 slower; perold and greenbea still hit the limit.
