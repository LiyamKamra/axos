#!/usr/bin/env python3
"""Regression test of the AXOS command-line solver and its Python wrapper
(apps/axos.py): LP, QP and MILP models with known answers, statuses and exit
codes, solution and JSON output.

  python tests/test_cli.py            (after apps/build.sh; AXOS_EXE overrides)
"""
import json
import os
import subprocess
import sys
import tempfile

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, os.path.join(ROOT, "apps"))
import axos  # noqa: E402

DATA = os.path.join(HERE, "data")
MM = os.path.join(ROOT, "benchmarks", "qp", "data", "maros_meszaros_mps")
MILP = os.path.join(ROOT, "benchmarks", "milp", "data", "mps")

failures = []


def check(name, cond, detail=""):
    print(("ok   " if cond else "FAIL ") + name + ("" if cond else "  " + detail))
    if not cond:
        failures.append(name)


def close(a, b, rel):
    return a is not None and abs(a - b) <= rel * max(1.0, abs(b))


# LP: Netlib optima
r = axos.solve(os.path.join(DATA, "afiro.mps"), time_limit=30)
check("LP afiro simplex", r["status"] == "optimal" and close(r["objective"], -464.7531428571, 1e-9), str(r))
r = axos.solve(os.path.join(DATA, "adlittle.mps"), method="ipm", time_limit=30)
check("LP adlittle ipm", r["status"] == "optimal" and close(r["objective"], 225494.96316, 1e-6), str(r))
r = axos.solve(os.path.join(DATA, "adlittle.mps"), method="hpr", time_limit=30, tol=1e-8)
check("LP adlittle hpr", r["status"] == "optimal" and close(r["objective"], 225494.96316, 1e-6), str(r))
r = axos.solve(os.path.join(DATA, "infeasible_lp.mps"))
check("LP infeasible", r["status"] == "infeasible" and r["exit_code"] == 2, str(r))

# MILP: a maximization model and its LP relaxation
r = axos.solve(os.path.join(DATA, "tinymax_milp.mps"), time_limit=10)
check("MILP maximize", r["status"] == "optimal" and close(r["objective"], 20, 1e-9) and r["x"] == {"x": 4.0}, str(r))
r = axos.solve(os.path.join(DATA, "tinymax_milp.mps"), type="lp")
check("MILP LP relaxation", r["status"] == "optimal" and close(r["objective"], 22.5, 1e-9), str(r))
if os.path.exists(os.path.join(MILP, "markshare_4_0.mps")):
    r = axos.solve(os.path.join(MILP, "markshare_4_0.mps"), time_limit=2)
    check("MILP time limit", r["status"] == "time limit" and r["exit_code"] == 1 and r["max_violation"] <= 1e-6, str(r))

# QP (Maros-Meszaros QSHARE1B, optimum 7.2007832e5)
if os.path.exists(os.path.join(MM, "QSHARE1B.mps")):
    for m in ("ipm", "hprqp"):
        r = axos.solve(os.path.join(MM, "QSHARE1B.mps"), method=m, time_limit=60)
        check("QP QSHARE1B " + m, r["status"] == "optimal" and close(r["objective"], 720078.32, 2e-5), str(r))

# files and usage errors
exe = axos._exe()
with tempfile.TemporaryDirectory() as d:
    sol, js = os.path.join(d, "x.sol"), os.path.join(d, "r.json")
    p = subprocess.run([exe, os.path.join(DATA, "afiro.mps"), "--solution", sol, "--json", js, "--quiet"])
    ok = p.returncode == 0 and os.path.exists(sol) and json.load(open(js))["status"] == "optimal"
    check("solution and JSON files", ok)
p = subprocess.run([exe, os.path.join(DATA, "afiro.mps"), "--bogus"], capture_output=True)
check("bad option -> 64", p.returncode == 64)
p = subprocess.run([exe, os.path.join(DATA, "missing.mps")], capture_output=True)
check("missing file -> 5", p.returncode == 5)

print("%d failure(s)" % len(failures))
sys.exit(1 if failures else 0)
