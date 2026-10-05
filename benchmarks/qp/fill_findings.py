#!/usr/bin/env python3
"""Fills the numbers of results/findings.html (placeholders __NAME__) from the
result CSVs, so the text of the report always matches the data.

  python fill_findings.py results/findings.html results/findings_filled.html
"""
import csv
import math
import sys

R = "results/"


def load(p):
    return {r["problem"]: r for r in csv.DictReader(open(p))}


def ok(r):
    try:
        return r["status"] == "optimal" and max(
            float(r["rel_primal"]), float(r["rel_dual"]), float(r["rel_gap"])) <= 1e-5
    except (TypeError, ValueError, KeyError):
        return False


def solved(rows):
    return sum(ok(r) for r in rows.values())


def sgm(rows, tl):
    t = [min(float(r["seconds"]), tl) if ok(r) else tl for r in rows.values()]
    return math.exp(sum(math.log(x + 10) for x in t) / len(t)) - 10


def speedup(fast, slow):
    both = [p for p in fast if ok(fast[p]) and p in slow and ok(slow[p])]
    r = [float(slow[p]["seconds"]) / max(float(fast[p]["seconds"]), 1e-6) for p in both]
    return len(both), math.exp(sum(map(math.log, r)) / len(r))


def main():
    src, dst = sys.argv[1], sys.argv[2]
    g, jl, auto = (load(R + "mm/" + s + ".csv") for s in ("axos-hprqp-gpu", "hprqp-jl", "axos-auto"))
    lg = load(R + "large/axos-hprqp-gpu.csv")
    ljl = load(R + "large/hprqp-jl.csv")
    common, sp = speedup(g, jl)
    ipm_sp = []
    for s in ("clarabel", "piqp"):
        _, x = speedup(lg, load(R + "large/" + s + ".csv"))
        ipm_sp.append(x)
    vals = {
        "G_MM": str(solved(g)),
        "G_COMMON": str(common),
        "G_SPEED": "%.1f" % sp,
        "G_LG": str(solved(lg)),
        "G_LG_SGM": "%.1f" % sgm(lg, 120),
        "LG_IPM_SPEED": "%.0f–%.0f" % (min(ipm_sp), max(ipm_sp)),
        "JL_LG": str(solved(ljl)),
        "AUTO_MM": str(solved(auto)),
        "AUTO_MM_SGM": "%.2f" % sgm(auto, 60),
    }
    text = open(src, encoding="utf-8").read()
    for k, v in vals.items():
        text = text.replace("__%s__" % k, v)
    assert "__" not in text.replace("__init__", ""), "unfilled placeholder"
    open(dst, "w", encoding="utf-8").write(text)
    print(vals)


if __name__ == "__main__":
    main()
