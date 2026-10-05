"""Sparse benchmark with PyTorch (CPU + CUDA) and SciPy on the matrices exported
by bench_sparse (build/sparse_bench/*.bin). Prints the same RESULT lines.

    ~/python_junk/.venv/bin/python benchmarks/sparse/bench_torch.py
"""
import statistics
import sys
import time

import numpy as np
import scipy.sparse as sp
import scipy.sparse.linalg as spla
import torch

DIR = sys.argv[1] if len(sys.argv) > 1 else "build/sparse_bench"
SPMM_K = 16


def load(name):
    with open(f"{DIR}/{name}.bin", "rb") as f:
        rows, cols, nnz = np.frombuffer(f.read(24), dtype=np.int64)
        rp = np.frombuffer(f.read(4 * (rows + 1)), dtype=np.int32)
        ci = np.frombuffer(f.read(4 * nnz), dtype=np.int32)
        v = np.frombuffer(f.read(8 * nnz), dtype=np.float64)
    return int(rows), int(cols), rp.copy(), ci.copy(), v.copy()


def sync(dev):
    if dev == "cuda":
        torch.cuda.synchronize()


def bench(fn, dev="cpu", min_s=0.4, min_reps=5, max_reps=400):
    fn(); sync(dev); fn(); sync(dev)
    t = []
    start = time.perf_counter()
    while len(t) < min_reps or ((time.perf_counter() - start) < min_s and len(t) < max_reps):
        a = time.perf_counter()
        fn(); sync(dev)
        t.append((time.perf_counter() - a) * 1000)
    return statistics.median(t)


def result(lib, op, name, ms):
    print(f"RESULT {lib} {op} {name} {ms:.4f}", flush=True)


def try_bench(lib, op, name, fn, dev="cpu", **kw):
    try:
        result(lib, op, name, bench(fn, dev, **kw))
    except Exception as e:  # unsupported op on this backend
        print(f"SKIP {lib} {op} {name}: {type(e).__name__}: {str(e)[:80]}", flush=True)


def scipy_ops(name, A, spgemm):
    n = A.shape[1]
    x = np.sin(0.001 * np.arange(n))
    X = np.full((n, SPMM_K), 0.5)
    y = A @ x
    try_bench("scipy", "spmv", name, lambda: A @ x)
    At = A.T  # CSC view of the transpose (no data movement)
    try_bench("scipy", "spmv_t", name, lambda: At @ y)
    try_bench("scipy", "spmm16", name, lambda: A @ X)
    try_bench("scipy", "transpose", name, lambda: A.T.tocsr())
    if spgemm:
        try_bench("scipy", "spgemm", name, lambda: A @ A, min_reps=3, max_reps=50)


def torch_ops(lib, dev, name, rows, cols, rp, ci, v, spgemm):
    d = torch.device(dev)
    A = torch.sparse_csr_tensor(
        torch.from_numpy(rp).to(torch.int64), torch.from_numpy(ci).to(torch.int64),
        torch.from_numpy(v), size=(rows, cols), dtype=torch.float64).to(d)
    x = torch.sin(0.001 * torch.arange(cols, dtype=torch.float64, device=d))
    X = torch.full((cols, SPMM_K), 0.5, dtype=torch.float64, device=d)
    y = torch.mv(A, x)
    try_bench(lib, "spmv", name, lambda: torch.mv(A, x), dev)
    try_bench(lib, "spmv_t", name, lambda: torch.mv(A.t(), y), dev)
    try_bench(lib, "spmm16", name, lambda: torch.sparse.mm(A, X), dev)
    try_bench(lib, "transpose", name, lambda: A.t().to_sparse_csr(), dev)
    if spgemm:
        try_bench(lib, "spgemm", name, lambda: torch.sparse.mm(A, A), dev,
                  min_reps=3, max_reps=50)


def scipy_ldl(name, A):
    Acsc = A.tocsc()
    b = np.ones(A.shape[0])
    t0 = time.perf_counter()
    lu = spla.splu(Acsc, permc_spec="COLAMD", diag_pivot_thresh=0.0)
    result("scipy_superlu", "ldl_analyze+factor", name, (time.perf_counter() - t0) * 1000)
    try_bench("scipy_superlu", "ldl_solve", name, lambda: lu.solve(b))


def main():
    print(f"# torch {torch.__version__}, threads={torch.get_num_threads()}, "
          f"cuda={torch.cuda.is_available()}")
    for name, spgemm in [("rand20k", True), ("lap300", True), ("rand100k", False),
                         ("skew200k", False), ("lap1000", False), ("rand1m", False)]:
        rows, cols, rp, ci, v = load(name)
        A = sp.csr_matrix((v, ci, rp), shape=(rows, cols))
        scipy_ops(name, A, spgemm)
        torch_ops("torch_cpu", "cpu", name, rows, cols, rp, ci, v, spgemm)
        if torch.cuda.is_available():
            torch_ops("torch_cuda", "cuda", name, rows, cols, rp, ci, v, spgemm)
    for name in ["lap100", "lap200"]:
        rows, cols, rp, ci, v = load(name)
        scipy_ldl(name, sp.csr_matrix((v, ci, rp), shape=(rows, cols)))


if __name__ == "__main__":
    main()
