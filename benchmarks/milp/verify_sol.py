#!/usr/bin/env python3
"""Independent check of the AXOS solutions of a benchmark run: every solution
file AXOS wrote (results/<set>/sol/<solver>/<problem>.sol, "name value"
lines) is re-evaluated on the original model read by HiGHS' MPS reader, with
the same check the comparators get (milpbench.check: bounds, rows relative
1e-5, integrality 1e-5), and its objective compared with the CSV.

  python verify_sol.py results/miplib50 [--data data/mps] [--fix]

--fix rewrites the CSV's feasible column with the independent verdict.
"""
import argparse
import csv
import math
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import milpbench  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("dir")
    ap.add_argument("--data", default=os.path.join(os.path.dirname(os.path.abspath(__file__)), "data", "mps"))
    ap.add_argument("--fix", action="store_true")
    a = ap.parse_args()
    bad = 0
    for solver in ("axos", "axos-gpu"):
        path = os.path.join(a.dir, solver + ".csv")
        if not os.path.exists(path):
            continue
        rows = list(csv.DictReader(open(path)))
        fields = list(rows[0].keys()) if rows else []
        checked = 0
        for r in rows:
            sol = os.path.join(a.dir, "sol", solver, r["problem"] + ".sol")
            claimed = r.get("feasible") == "1"
            if not os.path.exists(sol):
                if claimed:
                    print(f"{solver} {r['problem']}: claims a solution but wrote no file")
                    bad += 1
                continue
            md = milpbench.load_model(os.path.join(a.data, r["problem"] + ".mps"))
            pos = {nm: j for j, nm in enumerate(md["names"])}
            x = [0.0] * md["n"]
            for line in open(sol):
                t = line.split()
                if len(t) == 2 and t[0] in pos:
                    x[pos[t[0]]] = float(t[1])
            ok, obj = milpbench.check(md, x)
            checked += 1
            try:
                rep = float(r["objective"])
            except ValueError:
                rep = math.nan
            same = math.isfinite(rep) and abs(obj - rep) <= 1e-6 * max(1.0, abs(obj))
            if ok != claimed or not same:
                print(f"{solver} {r['problem']}: independent check {ok} (claimed {claimed}), objective {obj} "
                      f"vs reported {rep}")
                bad += 1
            if a.fix:
                r["feasible"] = "1" if ok else "0"
        print(f"{solver}: {checked} solutions checked")
        if a.fix and rows:
            with open(path, "w", newline="") as f:
                w = csv.DictWriter(f, fieldnames=fields)
                w.writeheader()
                w.writerows(rows)
    print(f"{bad} mismatch(es)")
    return 1 if bad else 0


if __name__ == "__main__":
    sys.exit(main())
