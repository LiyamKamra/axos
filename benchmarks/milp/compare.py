#!/usr/bin/env python3
"""Head-to-head of one solver against the others in a benchmark directory:
per instance, whose final solution has the smaller primal gap to the MIPLIB
reference (ties within 1e-6), and the instances where it is better.

  python compare.py results/miplib50 --solver axos [--json out.json]
"""
import argparse
import csv
import glob
import json
import os

import report


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("dir")
    ap.add_argument("--solver", default="axos")
    ap.add_argument("--time-limit", type=float, default=60.0)
    ap.add_argument("--json")
    a = ap.parse_args()
    ref = report.load_solu(os.path.join(os.path.dirname(os.path.abspath(__file__)), "data", "miplib2017-v31.solu"))
    per = {}
    for f in glob.glob(os.path.join(a.dir, "*.csv")):
        s = os.path.splitext(os.path.basename(f))[0]
        per[s] = report.summarize(list(csv.DictReader(open(f))), ref, a.time_limit)["per"]
    me = per[a.solver]
    out = {}
    for other, rows in sorted(per.items()):
        if other == a.solver:
            continue
        better, tie, worse, bl = 0, 0, 0, []
        for k, e in me.items():
            o = rows.get(k)
            if not o or e["ref_kind"] == "inf":
                continue
            if abs(e["gap"] - o["gap"]) <= 1e-6:
                tie += 1
            elif e["gap"] < o["gap"]:
                better += 1
                bl.append(dict(name=k, mine=e["objective"], theirs=o["objective"], ref=e["ref"]))
            else:
                worse += 1
        out[other] = dict(better=better, tie=tie, worse=worse, better_list=bl)
        print(f"{a.solver} vs {other}: better {better}, equal {tie}, worse {worse}")
        for b in bl:
            print(f"   {b['name']}: {b['mine']} vs {b['theirs']} (best known {b['ref']})")
    if a.json:
        json.dump(out, open(a.json, "w"), indent=1)


if __name__ == "__main__":
    main()
