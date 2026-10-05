#!/usr/bin/env python3
"""MILP benchmark harness for the established solvers: HiGHS (highspy), SCIP
(pyscipopt) and CBC (the binary bundled with PuLP).

Every solver reads the same MPS file, runs single-threaded with the same time
limit and relative gap (1e-4, the MIPLIB criterion), in its own process
(killed at 1.5x the time limit or past --mem-limit GiB). Each reported
solution is checked on the original model (bounds, rows, integrality, read
with HiGHS) before it counts.

  python milpbench.py run --solver highs --data DIR --time-limit 60 --out res.csv [--only a,b]

CSV columns match run_milp (AXOS): problem, n, m, nnz, nint, solver,
status, seconds, objective, bound, gap, nodes, lp_iterations,
first_solution_seconds, root_bound, cuts, incumbent_source, feasible,
read_seconds.
"""
import argparse
import math
import os
import queue
import subprocess
import sys
import tempfile
import time

HEADER = ["problem", "n", "m", "nnz", "nint", "solver", "status", "seconds", "objective", "bound",
          "gap", "nodes", "lp_iterations", "first_solution_seconds", "root_bound", "cuts",
          "incumbent_source", "feasible", "read_seconds"]


def load_model(path):
    """Problem data through HiGHS: dense row/column bounds, CSC matrix, integrality, names."""
    import highspy
    import numpy as np
    h = highspy.Highs()
    h.setOptionValue("output_flag", False)
    h.readModel(path)
    lp = h.getLp()
    n, m = lp.num_col_, lp.num_row_
    a = lp.a_matrix_
    integ = list(lp.integrality_) if len(lp.integrality_) else [highspy.HighsVarType.kContinuous] * n
    isint = [t != highspy.HighsVarType.kContinuous for t in integ]
    return dict(n=n, m=m, start=np.array(a.start_), index=np.array(a.index_), value=np.array(a.value_),
                lb=np.array(lp.col_lower_), ub=np.array(lp.col_upper_), rl=np.array(lp.row_lower_),
                ru=np.array(lp.row_upper_), c=np.array(lp.col_cost_), offset=lp.offset_,
                sense=-1.0 if lp.sense_ == highspy.ObjSense.kMaximize else 1.0,
                isint=isint, names=list(lp.col_names_), nnz=len(a.value_))


def check(md, x):
    """Largest relative violation of bounds/rows and integrality; (ok, objective)."""
    import numpy as np
    x = np.asarray(x, dtype=float)
    if len(x) != md["n"] or not np.all(np.isfinite(x)):
        return False, math.nan
    inf = 1e20
    lb, ub = md["lb"], md["ub"]
    bv = np.maximum(np.where(lb > -inf, lb - x, 0), np.where(ub < inf, x - ub, 0)).max(initial=0)
    ax = np.zeros(md["m"])
    s, idx, val = md["start"], md["index"], md["value"]
    for j in range(md["n"]):
        if x[j] != 0:
            ax[idx[s[j]:s[j + 1]]] += val[s[j]:s[j + 1]] * x[j]
    rl, ru = md["rl"], md["ru"]
    rv = np.maximum(np.where(rl > -inf, (rl - ax) / (1 + np.abs(rl)), 0),
                    np.where(ru < inf, (ax - ru) / (1 + np.abs(ru)), 0)).max(initial=0)
    iv = max([abs(x[j] - round(x[j])) for j in range(md["n"]) if md["isint"][j]], default=0)
    obj = float(md["c"] @ x) + md["offset"]
    return bool(bv <= 1e-5 * (1 + 0) or bv <= 1e-5) and rv <= 1e-5 and iv <= 1e-5, obj


def solve_highs(path, tl, gap):
    import highspy
    h = highspy.Highs()
    h.setOptionValue("output_flag", False)
    h.setOptionValue("time_limit", float(tl))
    h.setOptionValue("mip_rel_gap", gap)
    h.setOptionValue("threads", 1)
    h.readModel(path)
    t0 = time.perf_counter()
    h.run()
    secs = time.perf_counter() - t0
    st = h.modelStatusToString(h.getModelStatus()).lower()
    info = h.getInfo()
    sol = h.getSolution()
    x = list(sol.col_value) if info.primal_solution_status >= 2 else None
    return dict(status=st, seconds=secs, objective=info.objective_function_value if x else math.nan,
                bound=info.mip_dual_bound, nodes=info.mip_node_count, x=x)


def solve_scip(path, tl, gap):
    from pyscipopt import Model
    md = load_model(path)
    mdl = Model()
    mdl.hideOutput()
    mdl.readProblem(path)
    mdl.setParam("limits/time", float(tl))
    mdl.setParam("limits/gap", gap)
    t0 = time.perf_counter()
    mdl.optimize()
    secs = time.perf_counter() - t0
    st = mdl.getStatus().lower()
    x = None
    obj = math.nan
    if mdl.getNSols() > 0:
        best = mdl.getBestSol()
        val = {v.name: mdl.getSolVal(best, v) for v in mdl.getVars()}
        x = [val.get(nm, 0.0) for nm in md["names"]]
        obj = mdl.getObjVal()
    return dict(status=st, seconds=secs, objective=obj, bound=mdl.getDualbound(),
                nodes=mdl.getNNodes(), x=x)


def solve_cbc(path, tl, gap):
    import pulp
    md = load_model(path)
    cbc = pulp.PULP_CBC_CMD().path
    with tempfile.TemporaryDirectory() as d:
        solf = os.path.join(d, "sol.txt")
        # As PuLP runs it: wall-clock limit, branch, and "-solution" after it, which
        # writes the incumbent ("-solve ... -solu" wrote an intermediate LP point after
        # a time stop). Serial by default: "-threads 1" makes this build ignore the
        # time limit.
        cmd = [cbc, path, "-seconds", str(tl), "-timeMode", "elapsed", "-ratioGap", str(gap), "-branch",
               "-printingOptions", "all", "-solution", solf]
        t0 = time.perf_counter()
        r = subprocess.run(cmd, capture_output=True, text=True, timeout=1.5 * tl + 60)
        secs = time.perf_counter() - t0
        log = r.stdout
        st, obj, bound, nodes = "unknown", math.nan, math.nan, 0
        for line in log.splitlines():
            t = line.strip()
            if t.startswith("Result - "):
                st = t[len("Result - "):].lower()
            elif t.startswith("Objective value:"):
                obj = float(t.split(":")[1])
                if abs(obj) >= 1e49:  # CBC prints 1e50 when it has no solution
                    obj = math.nan
            elif t.startswith("Lower bound:"):
                bound = float(t.split(":")[1])
            elif t.startswith("Enumerated nodes:"):
                nodes = int(float(t.split(":")[1]))
        x = None
        if os.path.exists(solf) and math.isfinite(obj):
            lines = open(solf).read().splitlines()
            if lines and "infeasible" not in lines[0].lower():
                # "-printingOptions all": the rows, then the columns, each numbered from 0
                sections, cur = [], []
                for ln in lines[1:]:
                    parts = ln.replace("**", " ").split()
                    if len(parts) < 3:
                        continue
                    if parts[0] == "0" and cur:
                        sections.append(cur)
                        cur = []
                    cur.append(parts)
                if cur:
                    sections.append(cur)
                pos = {nm: j for j, nm in enumerate(md["names"])}
                x = [0.0] * md["n"]
                for parts in (sections[-1] if sections else []):
                    if parts[1] in pos:
                        x[pos[parts[1]]] = float(parts[2])
        if "optimal" in st:
            st = "optimal"
        elif "infeasible" in st:
            st = "infeasible"
        elif "stopped on time" in st or "time" in st:
            st = "time_limit"
        return dict(status=st, seconds=secs, objective=obj, bound=bound, nodes=nodes, x=x)


SOLVERS = dict(highs=solve_highs, scip=solve_scip, cbc=solve_cbc)


def normalize(st):
    s = st.lower()
    if s in ("optimal",) or "optimal" in s and "not" not in s:
        return "optimal"
    if "infeasible" in s:
        return "infeasible"
    if "time" in s:
        return "time_limit"
    if "gaplimit" in s.replace(" ", ""):
        return "optimal"
    return s.replace(" ", "_")


def solve_one(path, solver, tl, gap, q):
    name = os.path.splitext(os.path.basename(path))[0]
    t0 = time.perf_counter()
    md = load_model(path)
    read_s = time.perf_counter() - t0
    try:
        r = SOLVERS[solver](path, tl, gap)
        feas, obj = (False, math.nan)
        if r["x"] is not None:
            feas, obj = check(md, r["x"])
        objective = r["objective"] if r["x"] is not None else math.nan
        bound = r["bound"]
        gap_v = math.nan
        if r["x"] is not None and math.isfinite(objective) and math.isfinite(bound):
            gap_v = abs(objective - bound) / max(abs(objective), abs(bound), 1e-9)
        nint = sum(md["isint"])
        row = [name, md["n"], md["m"], md["nnz"], nint, solver, normalize(r["status"]),
               f"{r['seconds']:.4f}", f"{objective:.12e}", f"{bound:.12e}", f"{gap_v:.3e}",
               r["nodes"], "", "", "", "", "", int(feas), f"{read_s:.3f}"]
    except Exception as e:  # noqa: BLE001
        msg = str(e).replace(",", ";").replace("\n", " ")[:200]
        row = [name, md["n"], md["m"], md["nnz"], "", solver, "error", "", "", "", "", "", "", "",
               "", "", msg, 0, f"{read_s:.3f}"]
    q.put(row)


def process_memory(pid):
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
            h = k32.OpenProcess(0x1000 | 0x0010, False, pid)
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
    except Exception:  # noqa: BLE001
        pass
    return 0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["run"])
    ap.add_argument("--solver", required=True, choices=sorted(SOLVERS))
    ap.add_argument("--data", required=True)
    ap.add_argument("--time-limit", type=float, default=60)
    ap.add_argument("--gap", type=float, default=1e-4)
    ap.add_argument("--out")
    ap.add_argument("--only")
    ap.add_argument("--mem-limit", type=float, default=4.0)
    a = ap.parse_args()
    files = sorted(f for f in os.listdir(a.data) if f.endswith(".mps"))
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
    ctx = mp.get_context("spawn")
    for f in files:
        path = os.path.join(a.data, f)
        q = ctx.Queue()
        proc = ctx.Process(target=solve_one, args=(path, a.solver, a.time_limit, a.gap, q))
        t0 = time.perf_counter()
        proc.start()
        row, reason = None, "time_limit"
        while row is None:
            try:
                row = q.get(timeout=0.5)
            except queue.Empty:
                if time.perf_counter() - t0 > 1.5 * a.time_limit + 60:
                    break
                if process_memory(proc.pid) > a.mem_limit * 2**30:
                    reason = "memory_limit"
                    break
                if not proc.is_alive():
                    try:
                        row = q.get(timeout=5)
                    except queue.Empty:
                        reason = "error"
                        break
        proc.join(5)
        if proc.is_alive():
            proc.kill()
            proc.join()
        if row is None:
            name = os.path.splitext(f)[0]
            row = [name, "", "", "", "", a.solver, reason, f"{time.perf_counter() - t0:.4f}", "",
                   "", "", "", "", "", "", "", "killed", 0, ""]
        line = ",".join(str(v) for v in row)
        print(line, flush=True)
        if out:
            out.write(line + "\n")
            out.flush()


if __name__ == "__main__":
    main()
