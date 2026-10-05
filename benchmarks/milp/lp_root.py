#!/usr/bin/env python3
"""The LP relaxation of MILP models solved by the AXOS dual simplex (CPU,
exact vertex) and by HPR on the GPU (first order, relative KKT 1e-4), through
the axos command line; writes a JSON list for report_html.py.

  python lp_root.py data/mps_large a,b,c --time-limit 30 --out results/lp_root.json
"""
import argparse
import json
import os
import subprocess
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "..", "apps"))
import axos  # noqa: E402


def run(path, method, device, tl):
    r = axos.solve(path, type="lp", method=method, device=device, time_limit=tl, tol=1e-4)
    return dict(status=r.get("status"), objective=r.get("objective"), seconds=r.get("seconds"),
                iterations=r.get("iterations"))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("data")
    ap.add_argument("names")
    ap.add_argument("--time-limit", type=float, default=30)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    out = []
    for name in a.names.split(","):
        path = os.path.join(a.data, name + ".mps")
        info = subprocess.run([axos._exe(), path, "--info"], capture_output=True, text=True).stdout
        cpu = run(path, "simplex", "cpu", a.time_limit)
        gpu = run(path, "hpr", "gpu", a.time_limit)
        out.append(dict(name=name, info=info.strip(), simplex=cpu, hpr_gpu=gpu))
        print(name, cpu, gpu, flush=True)
    json.dump(out, open(a.out, "w"), indent=1)


if __name__ == "__main__":
    main()
