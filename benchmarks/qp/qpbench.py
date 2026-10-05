#!/usr/bin/env python3
"""QP benchmark harness for the established open-source solvers.

Reads problems exported by `run_qp <dir> --export <out>` (.axqp, identical data
to what AXOS solves), solves them with OSQP, Clarabel, HiGHS, PIQP or SCS at a
given tolerance and time limit, and scores every solution with the same
relative KKT residuals as AXOS (src/solver/qp/qp_model.h, evaluate_qp):

  eta_p   = max(|Ax - P_K(Ax)|, |x - P_C(x)|) / (1 + max(|b|, |Ax|))
  eta_d   = |Qx + c - A^T y - z| / (1 + max(|c|, |A^T y|, |Qx|))
  eta_gap = |P - D| / (1 + max(|P|, |D|))

Each solver's duals are mapped to the AXOS convention (y_i >= 0 at a lower row
bound; Qx + c - A^T y - z = 0). Times are wall clock around the solver's setup
(factorizations included) and solve, after the data is built.

  python qpbench.py run --solver clarabel --data DIR --tol 1e-6 \
      --time-limit 600 --out results.csv [--only A,B]
"""
import argparse
import csv
import math
import os
import sys
import time

import numpy as np
import scipy.sparse as sp

HEADER = ["problem", "n", "m", "nnzA", "nnzQ", "solver", "device", "tol", "status",
          "iterations", "seconds", "setup_seconds", "objective", "rel_primal",
          "rel_dual", "rel_gap", "read_seconds"]


# ---- problem data --------------------------------------------------------------

def read_axqp(path):
    b = open(path, "rb").read()
    if b[:4] != b"AXQP":
        raise ValueError(f"{path}: not an .axqp file")
    off = 8
    n, m, nnza, nnzq = (int(v) for v in np.frombuffer(b, np.int64, 4, off))
    off += 32
    offset = float(np.frombuffer(b, np.float64, 1, off)[0])
    off += 8
    maximize = int(np.frombuffer(b, np.int32, 1, off)[0])
    off += 4

    def csr(rows, nnz):
        nonlocal off
        rp = np.frombuffer(b, np.int32, rows + 1, off).astype(np.int64)
        off += 4 * (rows + 1)
        ci = np.frombuffer(b, np.int32, nnz, off).astype(np.int64)
        off += 4 * nnz
        v = np.frombuffer(b, np.float64, nnz, off).copy()
        off += 8 * nnz
        return sp.csr_matrix((v, ci, rp), shape=(rows, n))

    def vec(k):
        nonlocal off
        v = np.frombuffer(b, np.float64, k, off).copy()
        off += 8 * k
        return v

    A = csr(m, nnza)
    Q = csr(n, nnzq)
    c, lb, ub, l, u = vec(n), vec(n), vec(n), vec(m), vec(m)
    name = os.path.splitext(os.path.basename(path))[0]
    return dict(name=name, n=n, m=m, A=A, Q=Q, c=c, lb=lb, ub=ub, l=l, u=u,
                offset=offset, maximize=maximize)


def evaluate(p, x, y):
    """Relative KKT residuals of (x, y); returns (objective, eta_p, eta_d, eta_gap)."""
    A, Q, c = p["A"], p["Q"], p["c"]
    l, u, lb, ub = p["l"], p["u"], p["lb"], p["ub"]
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float).copy()
    if p["m"]:
        y[(y > 0) & ~np.isfinite(l)] = 0.0
        y[(y < 0) & ~np.isfinite(u)] = 0.0
    ax = A @ x if p["m"] else np.zeros(0)
    aty = A.T @ y if p["m"] else np.zeros(p["n"])
    qx = Q @ x
    finite = lambda v: v[np.isfinite(v)]
    bmax = max([np.abs(finite(l)).max(initial=0.0), np.abs(finite(u)).max(initial=0.0)])
    axmax = np.abs(ax).max(initial=0.0)
    with np.errstate(invalid="ignore"):
        pviol = max(0.0, np.max(l - ax, initial=-np.inf), np.max(ax - u, initial=-np.inf),
                    np.max(lb - x, initial=-np.inf), np.max(x - ub, initial=-np.inf))
    g = qx + c - aty
    lo, hi = np.isfinite(lb), np.isfinite(ub)
    z = np.where(lo & hi, g, np.where(lo, np.maximum(g, 0), np.where(hi, np.minimum(g, 0), 0.0)))
    dres = np.abs(g - z).max(initial=0.0)
    dual = p["offset"]
    if p["m"]:
        dual += np.sum(np.where(y > 0, y * np.where(y > 0, l, 0), 0.0))
        dual += np.sum(np.where(y < 0, y * np.where(y < 0, u, 0), 0.0))
    dual += np.sum(np.where(z > 0, z * np.where(z > 0, lb, 0), 0.0))
    dual += np.sum(np.where(z < 0, z * np.where(z < 0, ub, 0), 0.0))
    xqx = float(x @ qx)
    P = 0.5 * xqx + float(c @ x) + p["offset"]
    D = dual - 0.5 * xqx
    cmax = np.abs(c).max(initial=0.0)
    ep = pviol / (1 + max(bmax, axmax))
    ed = dres / (1 + max(cmax, np.abs(aty).max(initial=0.0), np.abs(qx).max(initial=0.0)))
    eg = abs(P - D) / (1 + max(abs(P), abs(D)))
    return P, float(ep), float(ed), float(eg)


def cone_form(p):
    """Rows for Ax + s = b with s in {0}^meq x R+^mineq (Clarabel, SCS).

    Returns (Acone, b, meq, row map) where the map gives, per original row i,
    the cone rows of its equality / upper / lower pieces (-1 if absent), so
    y_axos[i] = z[low] - z[up] - z[eq]."""
    A, n, m = p["A"].tocsr(), p["n"], p["m"]
    l, u, lb, ub = p["l"], p["u"], p["lb"], p["ub"]
    eq = np.isfinite(l) & np.isfinite(u) & (l == u)
    up = np.isfinite(u) & ~eq
    low = np.isfinite(l) & ~eq
    bup, blo = np.isfinite(ub), np.isfinite(lb)
    blocks, rhs = [], []
    ieq, iup, ilo = np.where(eq)[0], np.where(up)[0], np.where(low)[0]
    blocks += [A[ieq], A[iup], -A[ilo]]
    rhs += [u[ieq], u[iup], -l[ilo]]
    I = sp.identity(n, format="csr")
    jb_up, jb_lo = np.where(bup)[0], np.where(blo)[0]
    blocks += [I[jb_up], -I[jb_lo]]
    rhs += [ub[jb_up], -lb[jb_lo]]
    Acone = sp.vstack(blocks, format="csc")
    b = np.concatenate(rhs)
    meq = len(ieq)
    rows = {"eq": np.full(m, -1), "up": np.full(m, -1), "lo": np.full(m, -1)}
    k = 0
    rows["eq"][ieq] = np.arange(k, k + len(ieq)); k += len(ieq)
    rows["up"][iup] = np.arange(k, k + len(iup)); k += len(iup)
    rows["lo"][ilo] = np.arange(k, k + len(ilo)); k += len(ilo)
    return Acone, b, meq, rows


def y_from_cone(z, rows, m):
    y = np.zeros(m)
    for key, sign in (("lo", 1.0), ("up", -1.0), ("eq", -1.0)):
        idx = rows[key]
        sel = idx >= 0
        y[sel] += sign * z[idx[sel]]
    return y


# ---- solvers -------------------------------------------------------------------
# Each returns (status, x, y (AXOS convention), iterations, seconds, setup_seconds).

def solve_osqp(p, tol, tl):
    import osqp
    n, m = p["n"], p["m"]
    bnd = np.where(np.isfinite(p["lb"]) | np.isfinite(p["ub"]))[0]
    A = sp.vstack([p["A"], sp.identity(n, format="csr")[bnd]], format="csc")
    l = np.concatenate([p["l"], p["lb"][bnd]])
    u = np.concatenate([p["u"], p["ub"][bnd]])
    P = sp.triu(p["Q"], format="csc")
    t0 = time.perf_counter()
    s = osqp.OSQP()
    s.setup(P, p["c"], A, l, u, eps_abs=tol, eps_rel=tol, max_iter=2_000_000_000,
            time_limit=tl, verbose=False, polishing=False)
    t1 = time.perf_counter()
    r = s.solve()
    t2 = time.perf_counter()
    st = str(r.info.status).lower()
    x = r.x if r.x is not None else np.zeros(n)
    y = -r.y[:m] if r.y is not None else np.zeros(m)
    return st, x, y, int(r.info.iter), t2 - t0, t1 - t0


def solve_clarabel(p, tol, tl):
    import clarabel
    n, m = p["n"], p["m"]
    Acone, b, meq, rows = cone_form(p)
    P = sp.triu(p["Q"], format="csc")
    cones = []
    if meq:
        cones.append(clarabel.ZeroConeT(meq))
    if Acone.shape[0] - meq:
        cones.append(clarabel.NonnegativeConeT(Acone.shape[0] - meq))
    st = clarabel.DefaultSettings()
    st.verbose = False
    st.tol_gap_abs = st.tol_gap_rel = st.tol_feas = tol
    st.tol_ktratio = min(1e-6, tol)
    st.time_limit = tl
    st.max_iter = 5000
    t0 = time.perf_counter()
    solver = clarabel.DefaultSolver(P, p["c"], Acone, b, cones, st)
    t1 = time.perf_counter()
    sol = solver.solve()
    t2 = time.perf_counter()
    x = np.array(sol.x)
    y = y_from_cone(np.array(sol.z), rows, m)
    return str(sol.status).lower(), x, y, int(sol.iterations), t2 - t0, t1 - t0


def solve_highs(p, tol, tl):
    import highspy
    n, m = p["n"], p["m"]
    inf = highspy.kHighsInf
    h = highspy.Highs()
    h.setOptionValue("output_flag", False)
    h.setOptionValue("time_limit", float(tl))
    h.setOptionValue("primal_feasibility_tolerance", max(tol, 1e-10))
    h.setOptionValue("dual_feasibility_tolerance", max(tol, 1e-10))
    model = highspy.HighsModel()
    lp = model.lp_
    lp.num_col_, lp.num_row_ = n, m
    fix = lambda v: np.where(np.isfinite(v), v, np.where(v > 0, inf, -inf))
    lp.col_cost_ = p["c"]
    lp.col_lower_, lp.col_upper_ = fix(p["lb"]), fix(p["ub"])
    lp.row_lower_, lp.row_upper_ = fix(p["l"]), fix(p["u"])
    lp.offset_ = p["offset"]
    Ac = p["A"].tocsc()
    lp.a_matrix_.format_ = highspy.MatrixFormat.kColwise
    lp.a_matrix_.start_, lp.a_matrix_.index_, lp.a_matrix_.value_ = Ac.indptr, Ac.indices, Ac.data
    lp.a_matrix_.num_col_, lp.a_matrix_.num_row_ = n, m
    if p["Q"].nnz:
        L = sp.tril(p["Q"], format="csc")
        hs = model.hessian_
        hs.dim_ = n
        hs.format_ = highspy.HessianFormat.kTriangular
        hs.start_, hs.index_, hs.value_ = L.indptr, L.indices, L.data
    t0 = time.perf_counter()
    h.passModel(model)
    t1 = time.perf_counter()
    h.run()
    t2 = time.perf_counter()
    status = h.modelStatusToString(h.getModelStatus()).lower()
    sol = h.getSolution()
    x = np.array(sol.col_value) if sol.value_valid else np.zeros(n)
    y = np.array(sol.row_dual) if sol.dual_valid else np.zeros(m)
    info = h.getInfo()
    it = int(info.qp_iteration_count if p["Q"].nnz else info.simplex_iteration_count)
    return status, x, y, it, t2 - t0, t1 - t0


def solve_piqp(p, tol, tl):
    import piqp
    n, m = p["n"], p["m"]
    A = p["A"].tocsr()
    l, u = p["l"], p["u"]
    eq = np.isfinite(l) & np.isfinite(u) & (l == u)
    ie = np.where(eq)[0]
    ii = np.where(~eq & (np.isfinite(l) | np.isfinite(u)))[0]
    P = sp.triu(p["Q"], format="csc")
    s = piqp.SparseSolver()
    s.settings.verbose = False
    s.settings.eps_abs = tol
    s.settings.eps_rel = tol
    s.settings.max_iter = 1000
    t0 = time.perf_counter()
    s.setup(P, p["c"], A[ie].tocsc() if len(ie) else None, u[ie] if len(ie) else None,
            A[ii].tocsc() if len(ii) else None, l[ii] if len(ii) else None,
            u[ii] if len(ii) else None, p["lb"], p["ub"])
    t1 = time.perf_counter()
    st = s.solve()
    t2 = time.perf_counter()
    r = s.result
    x = np.array(r.x)
    y = np.zeros(m)
    if len(ie):
        y[ie] = -np.array(r.y)
    if len(ii):
        y[ii] = np.array(r.z_l) - np.array(r.z_u)
    return str(st).split(".")[-1].lower(), x, y, int(r.info.iter), t2 - t0, t1 - t0


def solve_scs(p, tol, tl):
    import scs
    n, m = p["n"], p["m"]
    Acone, b, meq, rows = cone_form(p)
    data = dict(P=sp.triu(p["Q"], format="csc"), A=Acone, b=b, c=p["c"])
    cone = dict(z=meq, l=Acone.shape[0] - meq)
    t0 = time.perf_counter()
    s = scs.SCS(data, cone, eps_abs=tol, eps_rel=tol, max_iters=2_000_000_000,
                time_limit_secs=tl, verbose=False)
    t1 = time.perf_counter()
    sol = s.solve()
    t2 = time.perf_counter()
    info = sol["info"]
    y = y_from_cone(np.array(sol["y"]), rows, m)
    return str(info["status"]).lower(), np.array(sol["x"]), y, int(info["iter"]), t2 - t0, t1 - t0


SOLVERS = dict(osqp=solve_osqp, clarabel=solve_clarabel, highs=solve_highs,
               piqp=solve_piqp, scs=solve_scs)
def normalize_status(s):
    s = s.lower().replace(" ", "_")
    if "inaccurate" in s or "almost" in s:
        return "inaccurate"
    if s in ("solved", "optimal") or s.endswith("solved"):
        return "optimal"
    if "time" in s:
        return "time_limit"
    if "iter" in s:
        return "iteration_limit"
    if "infeas" in s:
        return "infeasible"
    return s


def read_sol(path):
    b = open(path, "rb").read()
    n, m = (int(v) for v in np.frombuffer(b, np.int64, 2, 0))
    x = np.frombuffer(b, np.float64, n, 16)
    y = np.frombuffer(b, np.float64, m, 16 + 8 * n)
    return x, y


def score(a):
    rows = list(csv.reader(open(a.csv, newline="")))
    head, body = rows[0], rows[1:]
    col = {k: i for i, k in enumerate(head)}
    for r in body:
        path = os.path.join(a.sols, r[col["problem"]] + ".sol")
        if r[col["status"]] == "error" or not os.path.exists(path):
            continue
        p = read_axqp(os.path.join(a.data, r[col["problem"]] + ".axqp"))
        x, y = read_sol(path)
        obj, ep, ed, eg = evaluate(p, x, y)
        r[col["objective"]] = f"{-obj if p['maximize'] else obj:.12e}"
        r[col["rel_primal"]], r[col["rel_dual"]], r[col["rel_gap"]] = (
            f"{ep:.3e}", f"{ed:.3e}", f"{eg:.3e}")
    with open(a.out or a.csv, "w", newline="") as f:
        csv.writer(f, lineterminator="\n").writerows([head] + body)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["run", "score"])
    ap.add_argument("--solver", choices=sorted(SOLVERS))
    ap.add_argument("--csv")
    ap.add_argument("--sols")
    ap.add_argument("--data", required=True)
    ap.add_argument("--tol", type=float, default=1e-6)
    ap.add_argument("--time-limit", type=float, default=600)
    ap.add_argument("--out")
    ap.add_argument("--only")
    ap.add_argument("--mem-limit", type=float, default=4.0, help="GiB per solve")
    a = ap.parse_args()
    if a.cmd == "score":
        return score(a)
    if not a.solver:
        ap.error("run needs --solver")
    files = sorted(f for f in os.listdir(a.data) if f.endswith(".axqp"))
    if a.only:
        want = set(a.only.split(","))
        files = [f for f in files if os.path.splitext(f)[0] in want]
    out = None
    if a.out:
        fresh = not os.path.exists(a.out)
        out = open(a.out, "a", newline="")
        if fresh:
            out.write(",".join(HEADER) + "\n")
    print(",".join(HEADER), flush=True)
    import multiprocessing as mp
    import queue
    ctx = mp.get_context("spawn")
    for f in files:
        # Each solve runs in its own process, killed when it overruns the time
        # limit by more than half (some solvers check their limit rarely) or
        # its memory passes --mem-limit (a factorization's fill-in can exhaust
        # the machine).
        path = os.path.join(a.data, f)
        q = ctx.Queue()
        proc = ctx.Process(target=solve_one, args=(path, a.solver, a.tol, a.time_limit, q))
        t0 = time.perf_counter()
        proc.start()
        row, reason = None, "time_limit"
        while row is None:
            try:
                row = q.get(timeout=0.5)
            except queue.Empty:
                if time.perf_counter() - t0 > 1.5 * a.time_limit + 30:
                    break
                if process_memory(proc.pid) > a.mem_limit * 2**30:
                    reason = "memory_limit"
                    break
                if not proc.is_alive():
                    try:
                        row = q.get(timeout=5)  # a result still in the pipe
                    except queue.Empty:
                        reason = "error"  # crashed without a result
                        break
        proc.join(5)
        if proc.is_alive():
            proc.kill()
            proc.join()
        if row is None:
            name = os.path.splitext(f)[0]
            row = [name, "", "", "", "", a.solver, "cpu", f"{a.tol:.0e}", reason, "",
                   f"{time.perf_counter() - t0:.6f}", "", "", "", "", "", "killed"]
        line = ",".join(str(v) for v in row)
        print(line, flush=True)
        if out:
            out.write(line + "\n")
            out.flush()


def process_memory(pid):
    """Private memory of process pid in bytes (0 when unknown)."""
    try:
        if sys.platform == "win32":
            import ctypes
            from ctypes import wintypes

            class Counters(ctypes.Structure):
                _fields_ = [("cb", wintypes.DWORD), ("PageFaultCount", wintypes.DWORD)] + [
                    (k, ctypes.c_size_t) for k in (
                        "PeakWorkingSetSize", "WorkingSetSize", "QuotaPeakPagedPoolUsage",
                        "QuotaPagedPoolUsage", "QuotaPeakNonPagedPoolUsage",
                        "QuotaNonPagedPoolUsage", "PagefileUsage", "PeakPagefileUsage",
                        "PrivateUsage")]

            k32, psapi = ctypes.windll.kernel32, ctypes.windll.psapi
            k32.OpenProcess.restype = wintypes.HANDLE
            h = k32.OpenProcess(0x1000 | 0x0010, False, pid)  # QUERY_LIMITED_INFORMATION | VM_READ
            if not h:
                return 0
            c = Counters()
            c.cb = ctypes.sizeof(c)
            ok = psapi.GetProcessMemoryInfo(wintypes.HANDLE(h), ctypes.byref(c), c.cb)
            k32.CloseHandle(wintypes.HANDLE(h))
            return c.PrivateUsage if ok else 0
        with open(f"/proc/{pid}/status") as f:
            for line in f:
                if line.startswith("VmRSS:"):
                    return int(line.split()[1]) * 1024
    except Exception:  # noqa: BLE001 - the guard is best effort
        pass
    return 0


def solve_one(path, solver, tol, tl, q):
    t0 = time.perf_counter()
    p = read_axqp(path)
    read_s = time.perf_counter() - t0
    try:
        st, x, y, it, secs, setup = SOLVERS[solver](p, tol, tl)
        obj, ep, ed, eg = evaluate(p, x, y)
        if p["maximize"]:
            obj = -obj
        row = [p["name"], p["n"], p["m"], p["A"].nnz, p["Q"].nnz, solver, "cpu",
               f"{tol:.0e}", normalize_status(st), it, f"{secs:.6f}", f"{setup:.6f}",
               f"{obj:.12e}", f"{ep:.3e}", f"{ed:.3e}", f"{eg:.3e}", f"{read_s:.3f}"]
    except Exception as e:  # noqa: BLE001 - record the failure and go on
        msg = str(e).replace(",", ";").replace("\n", " ")[:300]
        row = [p["name"], p["n"], p["m"], p["A"].nnz, p["Q"].nnz, solver, "cpu",
               f"{tol:.0e}", "error", "", "", "", "", "", "", "", msg]
    q.put(row)


if __name__ == "__main__":
    main()
