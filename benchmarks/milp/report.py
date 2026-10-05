#!/usr/bin/env python3
"""Summary of a MILP benchmark directory (one CSV per solver, run_milp /
milpbench.py format) against the MIPLIB 2017 reference values.

  python report.py results/miplib50 [--solu data/miplib2017-v31.solu]
                   [--time-limit 60] [--json out.json] [--title T]

Per solver:
  solved     reported optimal, the solution passes the feasibility check on
             the original model, and its objective matches the reference
             (relative 1e-4, absolute 1e-6); or reported infeasible on an
             infeasible instance
  wrong      reported optimal (or infeasible) against the reference
  feasible   a solution that passes the check
  SGM10      shifted geometric mean of the time (shift 10 s; time limit for
             unsolved)
  primal gap |obj - ref| / max(|obj|, |ref|) of the final solution (0 when
             both are 0; 1 without a solution or with opposite signs),
             mean over the instances; and how many are within 1e-4 / 1%
"""
import argparse
import csv
import glob
import json
import math
import os


def load_solu(path):
    ref = {}
    for line in open(path):
        t = line.split()
        if len(t) < 2:
            continue
        tag, name = t[0], t[1]
        if tag == "=inf=":
            ref[name] = ("inf", None)
        elif tag in ("=opt=", "=best="):
            ref[name] = ("opt" if tag == "=opt=" else "best", float(t[2]))
    return ref


def fnum(v):
    try:
        x = float(v)
        return x if math.isfinite(x) else None
    except (TypeError, ValueError):
        return None


def primal_gap(obj, refv):
    if obj is None or refv is None:
        return 1.0
    if abs(obj) < 1e-9 and abs(refv) < 1e-9:
        return 0.0
    if obj * refv < 0:
        return 1.0
    return abs(obj - refv) / max(abs(obj), abs(refv))


def summarize(rows, ref, tl):
    out = dict(n=len(rows), solved=0, wrong=0, feasible=0, gap_mean=0.0, gap_1e4=0, gap_1pct=0, sgm10=0.0,
               wrong_list=[], per={})
    logs = []
    for r in rows:
        name = r["problem"]
        kind, refv = ref.get(name, (None, None))
        st = r.get("status", "")
        feas = r.get("feasible", "0") in ("1", "True", "true")
        obj = fnum(r.get("objective")) if feas else None
        secs = fnum(r.get("seconds")) or tl
        ok = False
        wrong = False
        if st == "optimal":
            if kind in ("opt", "best") and obj is not None:
                ok = abs(obj - refv) <= max(1e-6, 1e-4 * max(abs(obj), abs(refv)))
                wrong = not ok and kind == "opt" and obj is not None and \
                    abs(obj - refv) > max(1e-6, 1e-4 * max(abs(obj), abs(refv)))
            elif kind == "inf":
                wrong = True
        elif st == "infeasible":
            ok = kind == "inf"
            wrong = kind in ("opt", "best")
        if feas:
            out["feasible"] += 1
        gap = 0.0 if kind == "inf" else primal_gap(obj, refv)
        out["gap_mean"] += gap
        if kind != "inf" and gap <= 1e-4:
            out["gap_1e4"] += 1
        if kind != "inf" and gap <= 1e-2:
            out["gap_1pct"] += 1
        out["solved"] += ok
        out["wrong"] += wrong
        if wrong:
            out["wrong_list"].append(name)
        t = min(secs, tl) if ok else tl
        logs.append(math.log(t + 10))
        out["per"][name] = dict(status=st, ok=ok, wrong=wrong, feasible=feas, objective=obj, gap=gap,
                                seconds=secs, ref=refv, ref_kind=kind,
                                first=fnum(r.get("first_solution_seconds")),
                                source=r.get("incumbent_source", ""), notes=r.get("notes", ""))
    if rows:
        out["gap_mean"] /= len(rows)
        out["sgm10"] = math.exp(sum(logs) / len(logs)) - 10
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("dir")
    ap.add_argument("--solu", default=os.path.join(os.path.dirname(__file__), "data", "miplib2017-v31.solu"))
    ap.add_argument("--time-limit", type=float, default=60.0)
    ap.add_argument("--json")
    ap.add_argument("--title", default="MILP benchmark")
    args = ap.parse_args()
    ref = load_solu(args.solu)
    res = {}
    for f in sorted(glob.glob(os.path.join(args.dir, "*.csv"))):
        name = os.path.splitext(os.path.basename(f))[0]
        rows = list(csv.DictReader(open(f)))
        if rows:
            res[name] = summarize(rows, ref, args.time_limit)
    order = sorted(res, key=lambda s: (-res[s]["solved"], res[s]["sgm10"]))
    print(f"### {args.title}\n")
    print("| solver | instances | solved | wrong | feasible | SGM10 [s] | mean primal gap | gap <= 1e-4 | gap <= 1% |")
    print("|---|---|---|---|---|---|---|---|---|")
    for s in order:
        r = res[s]
        print(f"| {s} | {r['n']} | {r['solved']} | {r['wrong']} | {r['feasible']} | {r['sgm10']:.2f} | "
              f"{r['gap_mean']:.4f} | {r['gap_1e4']} | {r['gap_1pct']} |")
    for s in order:
        if res[s]["wrong_list"]:
            print(f"\n{s} wrong on: {', '.join(res[s]['wrong_list'])}")
    if args.json:
        json.dump(dict(title=args.title, time_limit=args.time_limit, order=order, solvers=res), open(args.json, "w"),
                  indent=1)


if __name__ == "__main__":
    main()
