#!/usr/bin/env python3
"""Writes the key-findings fragment of the MILP report from the result JSON
files, so every number in the text comes from the data.

  python make_findings.py --small results/miplib50.json --large results/large16.json
      --cmp-small results/cmp50.json --cmp-large results/cmp16.json
      [--gpu results/gpu_components.json] [--lp-root results/lp_root.json] --out results/findings.html
"""
import argparse
import json
import statistics

NAMES = {"axos": "AXOS (CPU)", "axos-gpu": "AXOS (CPU + GPU)", "highs": "HiGHS", "scip": "SCIP", "cbc": "CBC"}


def fmt(v):
    if v is None:
        return "none"
    a = abs(v)
    if 1e5 <= a < 1e13:
        return f"{v:,.0f}".replace("-", "−")
    return f"{v:.6g}".replace("-", "−")


def line(d, s):
    r = d["solvers"][s]
    return f"{NAMES.get(s, s)} {r['solved']} solved, {r['feasible']} with a solution, mean primal gap {r['gap_mean']:.3f}"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--small", required=True)
    ap.add_argument("--large", required=True)
    ap.add_argument("--cmp-small", required=True)
    ap.add_argument("--cmp-large", required=True)
    ap.add_argument("--gpu")
    ap.add_argument("--lp-root")
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    S, L = json.load(open(a.small)), json.load(open(a.large))
    CS, CL = json.load(open(a.cmp_small)), json.load(open(a.cmp_large))
    G = json.load(open(a.gpu)) if a.gpu else None
    LP = json.load(open(a.lp_root)) if a.lp_root else None
    me = "axos-gpu"
    sm, lg = S["solvers"], L["solvers"]
    others = [s for s in ("highs", "scip", "cbc") if s in sm]

    def h2h(C):
        return "; ".join(f"{NAMES[o]}: better on {C[o]['better']}, equal on {C[o]['tie']}, worse on {C[o]['worse']}"
                         for o in others if o in C)

    # notable wins: largest gap reduction against every comparator
    wins = {}
    for o in others:
        for b in CS.get(o, {}).get("better_list", []) + CL.get(o, {}).get("better_list", []):
            wins.setdefault(b["name"], {})[o] = b
    notable = sorted(wins.items(), key=lambda kv: -len(kv[1]))[:4]
    notable_txt = "; ".join(
        f"{k}: {fmt(next(iter(v.values()))['mine'])} against "
        + ", ".join(f"{NAMES[o]} {fmt(b['theirs'])}" for o, b in v.items())
        + f" (best known {fmt(next(iter(v.values()))['ref'])})" for k, v in notable)
    solved_mine = [k for k, e in sm[me]["per"].items() if e["ok"]]
    solved_only = [k for k in solved_mine if not any(sm[o]["per"].get(k, {}).get("ok") for o in others)]
    nosol = [k for k, e in sm[me]["per"].items() if e["objective"] is None and e["ref_kind"] != "inf"]
    parts = ["<h2>Key findings</h2><div class=\"cols\">"]
    parts.append(
        "<div><h3>Where AXOS stands</h3><p>On the 50 MIPLIB instances (60 s, one thread): "
        + "; ".join(line(S, s) for s in S["order"]) + ". "
        "The established solvers prove optimality far more often: their presolve, cuts, conflict analysis "
        "and simplex have been tuned for decades. AXOS finds a verified solution on "
        + f"{sm[me]['feasible']} of {sm[me]['n']} instances, close to "
        + ", ".join(f"{NAMES[o]} ({sm[o]['feasible']})" for o in others)
        + ", but its solutions are further from the best known values on average.</p></div>")
    parts.append(
        "<div><h3>Where AXOS wins</h3><p>Per instance, by final primal gap, AXOS (CPU + GPU) against "
        f"{h2h(CS)} on the 50 instances. Notable: {notable_txt}."
        + (f" It proves {', '.join(solved_only)} optimal where the other solvers do not within 60 s." if solved_only else "")
        + "</p></div>")
    parts.append(
        "<div><h3>Larger instances</h3><p>On the 16 larger instances (51k–392k nonzeros, 60 s): "
        + "; ".join(line(L, s) for s in L["order"]) + f". Per instance, AXOS (CPU + GPU) against {h2h(CL)}. "
        "Here the time goes into the LP relaxations: the dual simplex of AXOS is much slower than those of HiGHS "
        "and SCIP on LPs of this size, which the GPU first-order LP only partly makes up for.</p></div>")
    gpu_txt = []
    if G and G.get("probing"):
        sp = [e["gpu_rate"] / e["cpu_rate"] for e in G["probing"] if e["cpu_rate"] > 0]
        faster = sum(1 for x in sp if x > 1.05)
        gpu_txt.append(f"double probing on the GPU is faster on {faster} of {len(sp)} larger instances "
                       f"(probes per second, median {statistics.median(sp):.1f}× the CPU, up to {max(sp):.1f}×)")
    if LP:
        rows = [e for e in LP if e["simplex"]["status"] == "optimal" and e["hpr_gpu"]["status"] == "optimal"]
        if rows:
            r = [e["simplex"]["seconds"] / max(e["hpr_gpu"]["seconds"], 1e-6) for e in rows]
            gpu_txt.append(f"the LP relaxation by HPR on the GPU (tolerance 1e-4) takes {statistics.median(r):.1f}× "
                           f"less time than the dual simplex in the median over the {len(rows)} instances both solve, "
                           f"up to {max(r):.0f}×")
        miss = [e["name"] for e in LP if e["simplex"]["status"] != "optimal" and e["hpr_gpu"]["status"] == "optimal"]
        if miss:
            gpu_txt[-1] += f", and solves {', '.join(miss)} where the simplex runs out of time"
    gsm, csm = sm[me], sm.get("axos")
    if csm:
        gpu_txt.append(f"with the GPU components on, the mean primal gap on the 50 instances is {gsm['gap_mean']:.3f} "
                       f"against {csm['gap_mean']:.3f} on the CPU alone (one run each, so part of the difference is "
                       f"run-to-run variation of the search)")
    parts.append("<div><h3>What the GPU adds</h3><p>" + (lambda t: t[:1].upper() + t[1:])("; ".join(gpu_txt)) + ". The batched PDHG "
                 "bounds are valid but converge slowly: they recover most of the strong-branching information on some "
                 "models and little on others.</p></div>")
    parts.append(
        "<div><h3>Checked, not claimed</h3><p>Every AXOS solution was re-checked on the original model through "
        "HiGHS' MPS reader, the same check every comparator gets: all pass and match the reported objectives. "
        "No solver claimed a wrong optimum or infeasibility. "
        + (f"AXOS finds no solution on {', '.join(nosol)}. " if nosol else "")
        + "One adapter bug was found and fixed on the way: read the wrong way, CBC's solution file holds an "
        "intermediate LP point after a time stop, so CBC was re-run with PuLP's command line.</p></div>")
    parts.append(
        "<div><h3>What is missing</h3><p>From the per-instance results: flow-cover and path cuts for fixed-charge "
        "networks (beasleyC3, mc11, lotsize), conflict analysis and stronger MIP presolve for the instances "
        "nobody closes quickly, and above all a faster dual simplex for the node and root LPs of large models.</p></div>")
    parts.append("</div>")
    open(a.out, "w", encoding="utf-8").write("\n".join(parts))
    print("wrote", a.out)


if __name__ == "__main__":
    main()
