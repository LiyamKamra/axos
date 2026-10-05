"""Solve the LPs exported by bench_lp with HiGHS (through SciPy) and print
RESULT lines in the same format.

    ~/python_junk/.venv/bin/python benchmarks/solver/bench_highs.py [dir] [time_limit]
"""
import glob
import os
import sys
import time

import numpy as np
import scipy.sparse as sp
from scipy.optimize import linprog

DIR = sys.argv[1] if len(sys.argv) > 1 else "build/lp_bench"
TLIM = float(sys.argv[2]) if len(sys.argv) > 2 else 300


def load(path):
    with open(path, "rb") as f:
        rows, cols, nnz = np.frombuffer(f.read(24), dtype=np.int64)
        rp = np.frombuffer(f.read(4 * (rows + 1)), dtype=np.int32)
        ci = np.frombuffer(f.read(4 * nnz), dtype=np.int32)
        v = np.frombuffer(f.read(8 * nnz), dtype=np.float64)
        rd = lambda k: np.frombuffer(f.read(8 * k), dtype=np.float64)
        c, lb, ub = rd(cols), rd(cols), rd(cols)
        rl, ru = rd(rows), rd(rows)
        off = float(rd(1)[0])
    A = sp.csr_matrix((v, ci, rp), shape=(int(rows), int(cols)))
    return A, c, lb, ub, rl, ru, off


def solve(name, method):
    A, c, lb, ub, rl, ru, off = load(f"{DIR}/{name}.lp")
    eq = rl == ru
    only_u = ~eq & np.isfinite(ru) & ~np.isfinite(rl)
    only_l = ~eq & np.isfinite(rl) & ~np.isfinite(ru)
    both = ~eq & np.isfinite(rl) & np.isfinite(ru)
    ub_rows = [A[only_u], -A[only_l], A[both], -A[both]]
    ub_rhs = [ru[only_u], -rl[only_l], ru[both], -rl[both]]
    A_ub = sp.vstack(ub_rows).tocsr() if any(r.shape[0] for r in ub_rows) else None
    b_ub = np.concatenate(ub_rhs) if A_ub is not None else None
    A_eq = A[eq] if eq.any() else None
    b_eq = rl[eq] if eq.any() else None
    bounds = np.column_stack([np.where(np.isfinite(lb), lb, -np.inf),
                              np.where(np.isfinite(ub), ub, np.inf)])
    t0 = time.perf_counter()
    r = linprog(c, A_ub=A_ub, b_ub=b_ub, A_eq=A_eq, b_eq=b_eq, bounds=bounds,
                method=method, options={"time_limit": TLIM})
    ms = (time.perf_counter() - t0) * 1000
    status = {0: "optimal", 1: "limit", 2: "infeasible", 3: "unbounded"}.get(r.status, str(r.status))
    obj = (r.fun + off) if r.fun is not None else float("nan")
    print(f"RESULT lp {method.replace('highs-', 'highs_').replace('highs', 'highs_auto') if method == 'highs' else method.replace('highs-', 'highs_')} "
          f"{name} {ms:.1f} {getattr(r, 'nit', 0)} {status} {obj:.9e}", flush=True)


def solve_hard_timeout(name, method):
    # HiGHS' IPM can ignore time_limit in some phases: run each solve in a
    # child process and kill it after a hard limit.
    import multiprocessing as mp
    p = mp.Process(target=solve, args=(name, method))
    p.start()
    p.join(TLIM * 1.5 + 30)
    if p.is_alive():
        p.terminate()
        p.join()
        lib = "highs_auto" if method == "highs" else method.replace("highs-", "highs_")
        print(f"RESULT lp {lib} {name} {(TLIM * 1.5 + 30) * 1000:.1f} 0 killed nan", flush=True)


if __name__ == "__main__":
    only = os.environ.get("HIGHS_ONLY")  # comma-separated instance names
    for path in sorted(glob.glob(f"{DIR}/*.lp")):
        name = os.path.basename(path)[:-3]
        if only and name not in only.split(","):
            continue
        for method in ("highs", "highs-ds", "highs-ipm"):
            try:
                solve_hard_timeout(name, method)
            except Exception as e:
                print(f"SKIP {name} {method}: {type(e).__name__}: {str(e)[:80]}", flush=True)
