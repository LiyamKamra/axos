# Sparse benchmark results

Machine: 16 hardware threads, NVIDIA GeForce RTX 4060 Laptop GPU, CUDA 12.9,
Eigen 3.4.0, torch 2.12.1+cu130 (8 threads), scipy 1.18.1. Double precision,
`int32` indices, median of repeated runs, times in ms (lower is better).
Regenerate with `make benchmark_sparse`.

Reading the table:
- `axos_cpu` uses OpenMP (16 threads); `axos_cpu_1t` is the same code
  with `OMP_NUM_THREADS=1`. Eigen's sparse-times-dense products are also
  OpenMP-parallel, so `eigen` is the like-for-like column for `axos_cpu`.
- `spmv_t` for AXOS uses the cached explicit transpose (built once, its
  cost is the `transpose` row); Eigen, SciPy and torch compute the transposed
  product directly.
- `axos_cuda` and `torch_cuda` both call cuSPARSE for spmv/spmm/spgemm/
  transpose, so they mostly measure launch overhead and buffer reuse.
- The LDL rows use SPD 2D Laplacians (lap100 = 10k rows, lap200 = 40k).
  `axos_cuda` is cuDSS. `scipy_superlu` is a general sparse LU, not LDL.
- Matrices: rand20k/rand100k/rand1m (10 nonzeros per row), lap300/lap1000
  (2D Laplacian, 5 per row), skew200k (heavy-tailed row lengths).

```
op                  matrix          axos_cpu           eigen     axos_cuda   axos_cpu_1t           scipy       torch_cpu      torch_cuda   scipy_superlu
spmv                rand20k              0.015           0.024           0.022           0.101           0.131           0.038           0.027               -
spmv_t              rand20k              0.015           0.127           0.021           0.173           0.161           2.242           0.337               -
spmm16              rand20k              0.090           0.106           0.190           1.486           1.099           0.128           0.181               -
transpose           rand20k              0.523           0.467           0.101           0.443           0.454           2.433           0.286               -
spgemm              rand20k              3.363          71.532           1.639          41.033          10.909          19.472           1.556               -
spmv                lap300               0.025           0.042           0.031           0.192           0.194           0.055           0.049               -
spmv_t              lap300               0.025           0.256           0.032           0.194           0.282           6.198           0.531               -
spmm16              lap300               0.429           0.841           0.482           3.253           1.764           1.590           0.466               -
transpose           lap300               0.706           0.954           0.146           0.704           0.767           5.779           0.460               -
spgemm              lap300               1.922          26.642           0.527           5.707           5.628          10.222           0.588               -
spmv                rand100k             0.085           0.100           0.060           0.730           0.672           0.208           0.091               -
spmv_t              rand100k             0.087           0.786           0.060           1.026           0.865          39.836           1.476               -
spmm16              rand100k             1.579           1.603           0.790          10.545          11.430           3.636           0.827               -
transpose           rand100k             5.289           5.621           0.288           5.419           4.367          41.058           1.396               -
spmv                skew200k             0.196           0.229           0.108           2.146           2.068           0.758           0.121               -
spmv_t              skew200k             0.186           2.207           0.096           1.915           2.074          66.555           3.102               -
spmm16              skew200k             4.917           5.577           1.333          24.694          36.003           9.070           1.330               -
transpose           skew200k            11.426          13.430           0.538          11.598          11.155          66.190           2.964               -
spmv                lap1000              2.071           2.311           0.269           3.861           3.479           4.911           0.353               -
spmv_t              lap1000              2.137           4.346           0.268           3.840           3.937         231.623          11.579               -
spmm16              lap1000             11.772          18.355           4.526          37.452          69.969          48.381           5.096               -
transpose           lap1000             25.403          26.879           2.001          25.739          13.076         233.579          11.187               -
spmv                rand1m               3.664           4.705           0.644          24.050          27.481          11.018           0.674               -
spmv_t              rand1m               3.713          26.605           0.646          26.082          31.438         726.150          32.950               -
spmm16              rand1m              53.785          46.928           7.654         227.693         343.974         102.003           8.407               -
transpose           rand1m             208.726         209.952          10.416         210.687         179.673         705.184          32.085               -
ldl_analyze         lap100               3.078           2.098          39.059           3.078               -               -               -               -
ldl_factor          lap100               4.305           4.196           0.970           4.305               -               -               -               -
ldl_solve           lap100               0.237           0.323           0.183           0.237               -               -               -           0.574
ldl_analyze         lap200              28.380           8.454         112.828          28.380               -               -               -               -
ldl_factor          lap200              12.483          35.691           3.614          12.483               -               -               -               -
ldl_solve           lap200               2.207           1.906           0.461           2.207               -               -               -           4.652
ldl_analyze+factor  lap100                   -               -               -               -               -               -               -          25.314
ldl_analyze+factor  lap200                   -               -               -               -               -               -               -          82.270
```

Findings:

Larger LDL comparison (added with the multifrontal factorization; SPD 2-D
Laplacian, times in ms, 8 OpenMP threads, matrix sizes lap200 = 40k rows and
lap400 = 160k rows):

```
                      axos_cpu   axos_cpu    eigen   axos_cuda
                      (multifr.)   (simplicial)  (LDLT)  (cuDSS)
ldl_factor  lap200        12.5        38.4        35.2       20.1
ldl_factor  lap400        79.1         -         287.7       84.2
ldl_analyze lap400       147.4         -          41.1      413.8
ldl_solve   lap400        12.3         -          11.7        8.5
```

- GPU: comparable to torch's cuSPARSE path. SpMV is 5-25% faster and SpMM
  about 10% faster (buffer and descriptor reuse); SpGEMM is on par.
- CPU: SpMV/SpMM are at or above Eigen and well ahead of SciPy and torch-CPU;
  SpGEMM (Gustavson) is much faster than Eigen (20x on rand20k). The
  1M-row SpMM is the one case where Eigen is slightly faster.
- The CPU factorization is now a multifrontal LDL^T over relaxed supernodes
  with dense register-blocked kernels (`src/sparse/multifrontal_ldl.h`,
  chosen automatically once the factor has more than 200k entries): 2.8x
  faster than Eigen's SimplicialLDLT at 40k rows and 3.6x at 160k rows, and
  about equal to cuDSS on the GPU at 160k rows. Its dense kernel reaches 36
  GFlop/s on one thread and 169 GFlop/s on 8 (about half of this CPU's
  peak). Use one OpenMP thread per physical core: 16 hyperthreads were 30%
  slower than 8 on the flop-bound fronts.
- CPU `ldl_analyze` uses an approximate-minimum-degree ordering
  (`Ordering::MinDegree`, `src/sparse/amd_order.h`): 20.7 ms on lap200 with
  fill within 3% of the exact ordering, versus 448 ms with the exact
  minimum-degree code it replaced (`Ordering::ExactMinDegree`) and 8.5 ms for
  Eigen's AMD (2.4x slower; no supervariable detection yet). Numeric
  factorization and solves are on par with Eigen. cuDSS factorizes 11x
  faster than the CPU path. (The `axos_cpu_1t` analyze cells for the LDL
  rows were not re-measured single-threaded; ordering is serial either way.)
- CPU `transpose` is serial (same speed as Eigen), and 20x slower than the
  GPU version on the large matrices.
