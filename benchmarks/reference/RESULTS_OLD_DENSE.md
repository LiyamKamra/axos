# Dense layer baseline: removed implementation vs Eigen and PyTorch

Measured 2026-09-25 on AC power: Ryzen 7 7840HS (8 cores / 16 threads, AVX-512),
g++ -O3 -march=native -fopenmp, Eigen 3.4, PyTorch 2.12.1 CPU. The old code is
commit 03a4c03 (branch feature/csr), built in a separate worktree. One line in
`inverse_backs` was patched for this measurement: it wrote a block product into an
empty tensor and crashed, so the product tensor was given its m x m size.
Summarized in docs/TENSOR_SPEC.md §11.2. The benchmark suite's own summary
(benchmarks/suite, CPU, timings include allocation) is in
suite_old_dense_summary.csv.

## GEMM / GEMV / triangular inverse (old matops harnesses; times in ms)

`omp`: T = OpenMP (16 threads; PyTorch best of 8/16), F = 1 thread.
`blocked_mX` = `matMulBlocked` with block size X; `gemv_*` = benchmark-local
GEMV kernels except `gemv_tiled` (library `matMulTile`).

### gemm

| prec | omp | n | eigen | blocked_m32 | blocked_m64 | blocked_m128 | tiled | avx512 | lazy | torch |
|---|---|---|---|---|---|---|---|---|---|---|
| double | T | 256 | 0.176 | 0.096 | 0.082 | 0.948 | 3.207 | 0.684 | 28.542 | 0.180 |
| double | F | 256 | 0.590 |  | 0.597 |  | 1.942 | 0.643 | 28.329 | 0.687 |
| double | T | 512 | 2.045 | 1.129 | 1.472 | 1.363 | 8.871 | 2.934 | 805.014 | 0.904 |
| double | F | 512 | 4.236 |  | 4.813 |  | 15.858 | 4.787 | 788.800 | 5.509 |
| double | T | 1024 | 16.740 | 6.331 | 5.648 | 8.431 | 34.492 | 14.713 | 6737.350 | 9.147 |
| double | F | 1024 | 35.575 |  | 41.976 |  | 151.376 | 35.558 | 6713.250 | 40.483 |
| double | T | 2048 | 66.641 | 71.589 | 53.344 | 46.386 | 238.027 | 84.146 |  | 60.567 |
| double | F | 2048 | 273.151 |  | 397.090 |  | 1102.280 | 311.719 |  | 309.926 |
| double | T | 3072 | 138.147 | 260.708 | 171.314 | 156.816 | 557.650 | 225.733 |  | 173.386 |
| double | T | 4096 | 313.484 | 601.332 | 432.625 | 373.408 | 1689.350 | 471.116 |  | 430.184 |
| float | T | 256 | 0.108 | 0.082 | 0.041 | 0.207 | 1.353 | 0.352 | 12.631 | 0.062 |
| float | F | 256 | 0.354 |  | 0.294 |  | 1.197 | 0.311 | 13.298 | 0.352 |
| float | T | 512 | 0.464 | 0.385 | 0.347 | 0.315 | 5.074 | 1.550 | 289.583 | 0.308 |
| float | F | 512 | 2.062 |  | 2.371 |  | 8.523 | 2.222 | 279.742 | 2.155 |
| float | T | 1024 | 2.165 | 3.150 | 3.337 | 2.361 | 20.454 | 7.100 | 6499.750 | 2.530 |
| float | F | 1024 | 16.361 |  | 20.971 |  | 78.400 | 17.886 | 6465.800 | 19.143 |
| float | T | 2048 | 2771.280 | 90.418 | 48.372 | 28.361 | 136.127 | 36.618 |  | 35.266 |
| float | F | 2048 | 134.001 |  | 230.432 |  | 719.452 | 140.829 |  | 159.026 |
| float | T | 3072 | 71.145 | 119.841 | 81.929 | 70.737 | 368.269 | 115.861 |  | 100.090 |
| float | T | 4096 | 197.093 | 291.791 | 193.907 | 167.511 | 949.528 | 229.546 |  | 205.464 |

### gemv

| prec | omp | n | eigen | gemv_tiled | gemv_non_avx | gemv_avx512 | torch |
|---|---|---|---|---|---|---|---|
| double | T | 1024 | 0.174 | 1.013 | 2.691 | 0.164 | 0.119 |
| double | F | 1024 | 0.096 | 1.950 | 0.199 | 0.104 | 0.139 |
| double | T | 4096 | 3.469 | 7.665 | 2.970 | 2.820 | 4.396 |
| double | F | 4096 | 3.329 | 33.059 | 7.486 | 4.934 | 4.295 |
| double | T | 8192 | 12.093 | 23.499 | 10.708 | 11.697 | 15.570 |
| double | F | 8192 | 12.003 | 130.408 | 34.289 | 20.947 | 17.154 |
| double | T | 16384 | 47.164 | 73.050 | 41.350 | 39.291 | 61.949 |
| double | F | 16384 | 47.367 | 561.646 | 141.554 | 80.829 | 65.417 |
| float | T | 1024 | 0.051 | 5.538 | 5.501 | 3.478 | 0.066 |
| float | F | 1024 | 0.068 | 2.157 | 0.146 | 0.056 | 0.080 |
| float | T | 4096 | 2.403 | 7.454 | 2.038 | 1.609 | 2.800 |
| float | F | 4096 | 2.252 | 31.927 | 6.591 | 2.048 | 2.643 |
| float | T | 8192 | 6.388 | 23.422 | 7.106 | 5.276 | 8.968 |
| float | F | 8192 | 6.376 | 128.513 | 26.837 | 10.031 | 9.136 |
| float | T | 16384 | 25.040 | 85.979 | 30.620 | 21.686 | 31.700 |
| float | F | 16384 | 22.667 | 548.449 | 99.119 | 37.616 | 31.429 |

### inverse

| prec | omp | n | eigen | inverse_backs_m32 | inverse_backs_m64 | inverse_backs_m128 | torch |
|---|---|---|---|---|---|---|---|
| double | T | 256 | 2.164 | 0.305 | 0.966 | 0.970 | 0.698 |
| double | F | 256 | 1.983 | 0.410 | 0.458 | 1.005 | 1.827 |
| double | T | 512 | 20.318 | 1.170 | 1.040 | 2.102 | 3.189 |
| double | F | 512 | 19.075 | 2.335 | 2.068 | 3.052 | 14.574 |
| double | T | 1024 | 109.829 | 4.501 | 4.152 | 7.152 | 16.986 |
| double | F | 1024 | 106.270 | 17.060 | 12.800 | 14.812 | 86.032 |
| double | T | 2048 | 715.864 | 31.726 | 34.937 | 34.377 | 120.619 |
| double | F | 2048 | 745.555 | 158.240 | 136.734 | 116.810 | 633.566 |
| double | T | 3072 | 1722.900 | 78.035 | 74.162 | 94.660 | 337.092 |
| double | T | 4096 | 4177.080 | 185.394 | 156.115 | 162.032 | 749.922 |
| float | T | 256 | 1.447 | 0.136 | 0.225 | 0.643 | 0.360 |
| float | F | 256 | 1.217 | 0.271 | 0.306 | 0.801 | 1.830 |
| float | T | 512 | 9.238 | 0.474 | 0.602 | 1.073 | 1.378 |
| float | F | 512 | 8.136 | 1.315 | 1.214 | 2.008 | 10.128 |
| float | T | 1024 | 75.231 | 1.970 | 3.406 | 3.057 | 8.653 |
| float | F | 1024 | 71.984 | 10.182 | 7.404 | 8.027 | 85.213 |
| float | T | 2048 | 375.696 | 12.068 | 11.629 | 12.434 | 66.318 |
| float | F | 2048 | 409.142 | 79.863 | 53.264 | 46.531 | 453.837 |
| float | T | 3072 | 1212.860 | 46.117 | 33.922 | 44.227 | 219.827 |
| float | T | 4096 | 2835.190 | 98.526 | 68.396 | 75.413 | 531.730 |

Note: in the inverse table, `eigen` and `torch` are **general** inverses
(`A.inverse()`, `torch.linalg.inv`); `inverse_backs` inverts an upper-triangular
matrix. The triangular-vs-triangular comparison is in the next section.
Re-measured outliers: gemm f32 n=2048 omp is Eigen 19.5-26.1 ms and blocked
20.2-23.0 ms over 3 repeats (the 2771 ms Eigen entry above is noise); gemv f32
n=1024 omp is too small to time reliably (0.01-0.1 ms).

## Old code vs Eigen in one binary (ms, best of 3; OpenMP 16 threads)

`inverse_backs` here uses block size 128 (the table above has the best block size).

```
add C=A+B                  n=1000000   old      0.680 ms  eigen      0.601 ms  old/eigen-throughput     88%
axpy C=A+B*2               n=1000000   old      0.634 ms  eigen      0.611 ms  old/eigen-throughput     96%
chain exp(sin(A)+cos(A))   n=1000000   old      9.856 ms  eigen      9.807 ms  old/eigen-throughput     99%
sqrt                       n=1000000   old      1.951 ms  eigen      0.338 ms  old/eigen-throughput     17%
add C=A+B                  n=10000000  old      7.068 ms  eigen      7.029 ms  old/eigen-throughput     99%
axpy C=A+B*2               n=10000000  old      7.224 ms  eigen      7.046 ms  old/eigen-throughput     98%
chain exp(sin(A)+cos(A))   n=10000000  old     98.596 ms  eigen     98.075 ms  old/eigen-throughput     99%
sqrt                       n=10000000  old     19.797 ms  eigen      4.803 ms  old/eigen-throughput     24%
luDcmpPivoted              n=500       old     16.338 ms  eigen      2.107 ms  old/eigen-throughput     13%
luSolve (1 rhs)            n=500       old      0.319 ms  eigen      0.025 ms  old/eigen-throughput      8%  err 9.1e-18
luDcmpPivoted              n=1000      old    141.507 ms  eigen      7.690 ms  old/eigen-throughput      5%
luSolve (1 rhs)            n=1000      old      1.453 ms  eigen      0.112 ms  old/eigen-throughput      8%  err 7.2e-18
luDcmpPivoted              n=2000      old   1860.384 ms  eigen     38.107 ms  old/eigen-throughput      2%
luSolve (1 rhs)            n=2000      old      6.019 ms  eigen      0.875 ms  old/eigen-throughput     15%  err 6.6e-18
qrDecompositionTile        n=500       old     20.873 ms  eigen      3.977 ms  old/eigen-throughput     19%
qrDecompositionTile        n=1000      old    164.822 ms  eigen     25.940 ms  old/eigen-throughput     16%
conjugateGradient 50 it    n=2000      old     56.198 ms  eigen     31.222 ms  old/eigen-throughput     56%
expm                       n=100       old      1.970 ms  eigen      0.298 ms  old/eigen-throughput     15%  err 8.8e-12
expm                       n=200       old     15.814 ms  eigen      1.349 ms  old/eigen-throughput      9%  err 1.3e-12
inverse_backs vs tri-solve n=512       old      1.579 ms  eigen      3.930 ms  old/eigen-throughput    249%  err 3.1e-17
inverse_backs vs inverse() n=512       old      1.579 ms  eigen     18.524 ms  old/eigen-throughput   1173%
inverse_backs vs tri-solve n=1024      old      5.465 ms  eigen     26.329 ms  old/eigen-throughput    482%  err 3.1e-17
inverse_backs vs inverse() n=1024      old      5.465 ms  eigen     84.754 ms  old/eigen-throughput   1551%
inverse_backs vs tri-solve n=2048      old     35.451 ms  eigen    182.718 ms  old/eigen-throughput    515%  err 3.1e-17
inverse_backs vs inverse() n=2048      old     35.451 ms  eigen    529.964 ms  old/eigen-throughput   1495%
inverse_backs vs tri-solve n=4096      old    156.392 ms  eigen   1275.881 ms  old/eigen-throughput    816%  err 3.1e-17
inverse_backs vs inverse() n=4096      old    156.392 ms  eigen   3169.087 ms  old/eigen-throughput   2026%
general inv revEl+backs+replay n=64        old     13.074 ms  eigen      0.055 ms  old/eigen-throughput      0%  err 1.2e-15
general inv revEl+backs+replay n=128       old     81.250 ms  eigen      0.300 ms  old/eigen-throughput      0%  err 7.1e-15
general inv revEl+backs+replay n=256       old    346.715 ms  eigen      1.894 ms  old/eigen-throughput      1%  err 9.4e-14
general inv revEl+backs+replay n=512       old   1400.166 ms  eigen     16.811 ms  old/eigen-throughput      1%  err 1.9e-12
```

## PyTorch (ms, best of 3)

```
TORCH th=1 add n=1000000 0.457
TORCH th=1 axpy n=1000000 0.450
TORCH th=1 chain n=1000000 15.172
TORCH th=1 sqrt n=1000000 1.462
TORCH th=1 add n=10000000 6.777
TORCH th=1 axpy n=10000000 6.624
TORCH th=1 chain n=10000000 152.620
TORCH th=1 sqrt n=10000000 15.274
TORCH th=1 lu n=500 2.332
TORCH th=1 lusolve n=500 0.051
TORCH th=1 lu n=1000 15.117
TORCH th=1 lusolve n=1000 0.144
TORCH th=1 lu n=2000 113.265
TORCH th=1 lusolve n=2000 1.142
TORCH th=1 qr n=500 8.639
TORCH th=1 qr n=1000 59.533
TORCH th=1 expm n=100 0.507
TORCH th=1 expm n=200 2.420
TORCH th=1 triinv n=512 4.995
TORCH th=1 triinv n=1024 30.126
TORCH th=1 triinv n=2048 246.509
TORCH th=1 triinv n=4096 1901.222
TORCH th=8 add n=1000000 0.197
TORCH th=8 axpy n=1000000 0.188
TORCH th=8 chain n=1000000 1.998
TORCH th=8 sqrt n=1000000 0.249
TORCH th=8 add n=10000000 8.006
TORCH th=8 axpy n=10000000 7.764
TORCH th=8 chain n=10000000 40.453
TORCH th=8 sqrt n=10000000 5.710
TORCH th=8 lu n=500 0.530
TORCH th=8 lusolve n=500 0.049
TORCH th=8 lu n=1000 2.141
TORCH th=8 lusolve n=1000 0.141
TORCH th=8 lu n=2000 18.378
TORCH th=8 lusolve n=2000 1.029
TORCH th=8 qr n=500 2.207
TORCH th=8 qr n=1000 12.138
TORCH th=8 expm n=100 0.313
TORCH th=8 expm n=200 0.599
TORCH th=8 triinv n=512 0.678
TORCH th=8 triinv n=1024 5.993
TORCH th=8 triinv n=2048 41.843
TORCH th=8 triinv n=4096 282.811
TORCH th=16 add n=1000000 0.224
TORCH th=16 axpy n=1000000 0.187
TORCH th=16 chain n=1000000 2.218
TORCH th=16 sqrt n=1000000 0.188
TORCH th=16 add n=10000000 7.856
TORCH th=16 axpy n=10000000 11.011
TORCH th=16 chain n=10000000 50.849
TORCH th=16 sqrt n=10000000 6.208
TORCH th=16 lu n=500 0.647
TORCH th=16 lusolve n=500 0.062
TORCH th=16 lu n=1000 2.511
TORCH th=16 lusolve n=1000 0.196
TORCH th=16 lu n=2000 21.667
TORCH th=16 lusolve n=2000 1.135
TORCH th=16 qr n=500 2.951
TORCH th=16 qr n=1000 14.844
TORCH th=16 expm n=100 0.403
TORCH th=16 expm n=200 0.784
TORCH th=16 triinv n=512 0.823
TORCH th=16 triinv n=1024 6.784
TORCH th=16 triinv n=2048 53.693
TORCH th=16 triinv n=4096 297.317
```
