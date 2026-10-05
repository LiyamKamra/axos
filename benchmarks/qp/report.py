#!/usr/bin/env python3
"""Summarizes a QP benchmark directory written by run_all.sh.

For every solver (one CSV per solver, same problems):
  solved      the solver reports optimal AND the common relative KKT error
              max(eta_p, eta_d, eta_gap) of its solution is <= slack * tol
              (slack 10 by default: every solver measures its own tolerance
              in its own norms and scaling; AXOS checks the common metric
              itself). The strict count uses slack 1.
  SGM10       shifted geometric mean of the solve times, shift 10 s, with the
              time limit for problems not solved (as in H. Mittelmann's
              benchmarks)
  obj. error  |f - f_ref| / (1 + |f_ref|) on the solved problems, f_ref from
              a high-accuracy reference run (--ref, e.g. PIQP at 1e-9)

  python report.py DIR [--ref ref.csv] [--tol 1e-6] [--time-limit 60]
      [--json out.json]

Prints a markdown summary; --json writes everything the HTML report needs
(per-problem times, statuses and errors, and the performance profiles).
"""
import argparse
import csv
import glob
import json
import math
import os


def load(path):
    rows = {}
    with open(path, newline="") as f:
        for r in csv.DictReader(f):
            rows[r["problem"]] = r
    return rows


def fnum(s, default=math.nan):
    try:
        return float(s)
    except (TypeError, ValueError):
        return default


def kkt(r):
    v = [fnum(r.get(k)) for k in ("rel_primal", "rel_dual", "rel_gap")]
    return max(v) if all(math.isfinite(x) for x in v) else math.inf


def sgm(times, shift=10.0):
    return math.exp(sum(math.log(t + shift) for t in times) / len(times)) - shift


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("dir")
    ap.add_argument("--ref")
    ap.add_argument("--tol", type=float, default=1e-6)
    ap.add_argument("--slack", type=float, default=10.0)
    ap.add_argument("--time-limit", type=float, default=60.0)
    ap.add_argument("--json")
    ap.add_argument("--title", default="QP benchmark")
    a = ap.parse_args()

    runs = {}
    for p in sorted(glob.glob(os.path.join(a.dir, "*.csv"))):
        name = os.path.splitext(os.path.basename(p))[0]
        if a.ref and os.path.abspath(p) == os.path.abspath(a.ref):
            continue
        runs[name] = load(p)
    ref = load(a.ref) if a.ref else {}
    problems = sorted(set().union(*[set(r) for r in runs.values()]))
    sizes = {}
    for rows in runs.values():
        for pb, r in rows.items():
            if r.get("n") and pb not in sizes:
                sizes[pb] = int(r["n"])
    TL = a.time_limit

    def refobj(pb):
        r = ref.get(pb)
        if r and r["status"] == "optimal" and kkt(r) <= 1e-7:
            return fnum(r["objective"])
        return math.nan

    summary, per = {}, {}
    for s, rows in runs.items():
        times, solved, strict, errs = [], 0, 0, []
        per[s] = {}
        for pb in problems:
            r = rows.get(pb)
            ok = ok_strict = False
            t = TL
            e = math.inf
            if r is not None:
                e = kkt(r)
                claimed = r["status"] == "optimal"
                ok = claimed and e <= a.slack * a.tol
                ok_strict = claimed and e <= a.tol
                if ok:
                    t = min(fnum(r["seconds"], TL), TL)
            times.append(t if ok else TL)
            solved += ok
            strict += ok_strict
            oe = math.nan
            fr = refobj(pb)
            if ok and math.isfinite(fr):
                oe = abs(fnum(r["objective"]) - fr) / (1 + abs(fr))
                errs.append(oe)
            per[s][pb] = dict(ok=ok, strict=ok_strict, time=t if ok else None,
                              status=r["status"] if r else "missing",
                              kkt=e if math.isfinite(e) else None,
                              iters=r.get("iterations") if r else None,
                              objerr=oe if math.isfinite(oe) else None)
        errs.sort()
        summary[s] = dict(
            device=next(iter(rows.values()))["device"] if rows else "",
            solved=solved, strict=strict, total=len(problems), sgm10=sgm(times),
            objerr_median=errs[len(errs) // 2] if errs else None,
            objerr_max=errs[-1] if errs else None,
            objerr_gt_1e4=sum(e > 1e-4 for e in errs))

    # performance profile: fraction of problems solved within tau x best time
    best = {}
    for pb in problems:
        ts = [per[s][pb]["time"] for s in runs if per[s][pb]["ok"]]
        best[pb] = min(ts) if ts else None
    taus = [1, 1.5, 2, 3, 5, 10, 20, 50, 100, 1000]
    profile = {}
    for s in runs:
        ratios = []
        for pb in problems:
            if per[s][pb]["ok"] and best[pb]:
                ratios.append(max(per[s][pb]["time"], 1e-6) / max(best[pb], 1e-6))
        profile[s] = [sum(r <= tau for r in ratios) / len(problems) for tau in taus]

    order = sorted(runs, key=lambda s: (-summary[s]["solved"], summary[s]["sgm10"]))
    print(f"## {a.title}: {len(problems)} problems, tol {a.tol:g}, time limit {TL:g} s\n")
    print("| solver | device | solved (<= %g) | strict (<= %g) | SGM10 [s] | median obj. err. | obj. err. > 1e-4 |"
          % (a.slack * a.tol, a.tol))
    print("|---|---|---|---|---|---|---|")
    for s in order:
        v = summary[s]
        med = "-" if v["objerr_median"] is None else f"{v['objerr_median']:.1e}"
        print(f"| {s} | {v['device']} | {v['solved']} | {v['strict']} | {v['sgm10']:.2f} | {med} | "
              f"{v['objerr_gt_1e4']} |")
    if a.json:
        with open(a.json, "w") as f:
            json.dump(dict(title=a.title, tol=a.tol, slack=a.slack, time_limit=TL,
                           problems=problems, sizes=sizes, order=order, summary=summary, per=per,
                           taus=taus, profile=profile), f, indent=1)


if __name__ == "__main__":
    main()
