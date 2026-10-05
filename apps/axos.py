#!/usr/bin/env python3
"""Python interface to AXOS through its command-line solver (build/axos).

    import axos
    r = axos.solve("problem.mps", time_limit=60)        # LP, QP or MILP
    print(r["status"], r["objective"])
    x = r["x"]                                           # {column name: value}, nonzero entries

    r = axos.solve("model.qps", method="hprqp", device="gpu", tol=1e-6)
    r = axos.solve("model.mps", type="lp")               # LP relaxation of a MILP

Every keyword maps to a command-line option (see `axos --help`): type,
method, device, time_limit, tol, gap, presolve, cuts, heuristics, verbose.
The result is the solver's JSON report as a dict: status, objective, bound,
gap, seconds, iterations, nodes, method, device, max_violation, message and
the solution x. `exit_code` holds the process status (0 optimal, 1 solution
without proof of optimality, 2 infeasible, 3 unbounded, 4 no solution, 5
error).

The executable is found from AXOS_EXE, else next to this file in ../build.
"""
import json
import os
import subprocess
import sys

__all__ = ["solve", "info", "version", "AxosError"]


class AxosError(RuntimeError):
    pass


def _exe():
    env = os.environ.get("AXOS_EXE")
    if env:
        return env
    here = os.path.dirname(os.path.abspath(__file__))
    for name in ("axos.exe", "axos"):
        p = os.path.join(here, "..", "build", name)
        if os.path.exists(p):
            return os.path.normpath(p)
    raise AxosError("axos executable not found: build it (apps/build.sh) or set AXOS_EXE")


def _args(type=None, method=None, device=None, time_limit=None, tol=None, gap=None, presolve=True, cuts=True,
          heuristics=True, verbose=None):
    a = []
    if type: a += ["--type", type]
    if method: a += ["--method", method]
    if device: a += ["--device", device]
    if time_limit is not None: a += ["--time-limit", str(time_limit)]
    if tol is not None: a += ["--tol", str(tol)]
    if gap is not None: a += ["--gap", str(gap)]
    if not presolve: a.append("--no-presolve")
    if not cuts: a.append("--no-cuts")
    if not heuristics: a.append("--no-heuristics")
    if verbose is not None: a += ["--verbose", str(verbose)]
    return a


def solve(model, **options):
    """Solves the model file (MPS or QPS); returns the result dict."""
    cmd = [_exe(), str(model)] + _args(**options) + ["--json", "-", "--quiet"]
    p = subprocess.run(cmd, capture_output=True, text=True)
    if p.returncode in (5, 64) and not p.stdout.strip():
        raise AxosError(p.stderr.strip() or "axos failed with status %d" % p.returncode)
    line = p.stdout.strip().splitlines()[-1] if p.stdout.strip() else "{}"
    try:
        r = json.loads(line)
    except json.JSONDecodeError as e:
        raise AxosError("unexpected output from axos: %s" % p.stdout[-500:]) from e
    r["exit_code"] = p.returncode
    return r


def info(model):
    """The model statistics line printed by `axos --info`."""
    p = subprocess.run([_exe(), str(model), "--info"], capture_output=True, text=True)
    if p.returncode != 0:
        raise AxosError(p.stderr.strip())
    return p.stdout.strip()


def version():
    return subprocess.run([_exe(), "--version"], capture_output=True, text=True).stdout.strip()


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("usage: axos.py <model.mps|qps> [time_limit]")
        sys.exit(64)
    res = solve(sys.argv[1], time_limit=float(sys.argv[2]) if len(sys.argv) > 2 else None)
    x = res.pop("x", {})
    print(json.dumps(res, indent=1))
    print("%d nonzero columns" % len(x))
