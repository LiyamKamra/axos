#!/usr/bin/env python3
"""Renders the MILP benchmark (report.py --json summaries, the GPU component
measurements of test_gpu) as one self-contained HTML page.

  python report_html.py --set results/miplib50.json --set results/large16.json
      [--gpu results/gpu_components.json] [--notes findings.html]
      [--machine key=value ...] --out results/milp_report.html
"""
import argparse
import html
import json
import math

LABEL = {
    "axos": "AXOS (CPU)", "axos-gpu": "AXOS (CPU + GPU)", "highs": "HiGHS", "scip": "SCIP", "cbc": "CBC",
}
VERSION = {"highs": "1.15.1 (highspy)", "scip": "10.0 (PySCIPOpt 6.2)", "cbc": "2.10.3 (PuLP)"}

PAGE_HEAD = r"""<title>AXOS MILP Benchmarks</title>
<link rel="preconnect" href="https://fonts.googleapis.com">
<link rel="stylesheet" href="https://fonts.googleapis.com/css2?family=Archivo:wght@500;700;800&family=Source+Sans+3:wght@400;600&family=JetBrains+Mono:wght@400;600&display=swap">
<style>
/* layout: one reading column; wide tables scroll inside their own box. Tokens shared with the QP report. */
:root {
  --bg: #f6f7f8; --panel: #ffffff; --fg: #17202a; --muted: #5b6672; --rule: #d9dee3;
  --accent: #0a7c86; --accent-soft: #d8eef0; --good: #1f7a3f; --warn: #a15c00; --bad: #b3261e;
  --display: "Archivo", "Arial Narrow", Arial, sans-serif;
  --body: "Source Sans 3", "Segoe UI", system-ui, sans-serif;
  --mono: "JetBrains Mono", ui-monospace, Consolas, monospace;
}
@media (prefers-color-scheme: dark) { :root:not([data-theme="light"]) {
  --bg: #11161b; --panel: #182028; --fg: #e4e9ee; --muted: #9aa6b2; --rule: #2c3743;
  --accent: #4cc3cc; --accent-soft: #173a3e; --good: #5cc98a; --warn: #e3a452; --bad: #f2837a; color-scheme: dark } }
:root[data-theme="dark"] {
  --bg: #11161b; --panel: #182028; --fg: #e4e9ee; --muted: #9aa6b2; --rule: #2c3743;
  --accent: #4cc3cc; --accent-soft: #173a3e; --good: #5cc98a; --warn: #e3a452; --bad: #f2837a; color-scheme: dark }
body { background: var(--bg); color: var(--fg); font: 16px/1.55 var(--body); }
.wrap { max-width: 1080px; margin: 0 auto; padding-inline: 20px; padding-block: 36px 64px; display: grid; gap: 44px; }
header { display: grid; gap: 10px; }
.eyebrow { font: 600 12px/1 var(--mono); letter-spacing: .08em; text-transform: uppercase; color: var(--accent); }
h1 { font: 800 clamp(30px, 5vw, 46px)/1.05 var(--display); margin: 0; text-wrap: balance; letter-spacing: -.01em; }
h2 { font: 700 24px/1.2 var(--display); margin: 0; text-wrap: balance; }
h3 { font: 700 17px/1.3 var(--display); margin: 0; }
p { margin: 0; max-width: 70ch; }
.lede { color: var(--muted); font-size: 17px; }
section { display: grid; gap: 16px; min-width: 0; }
.meta { display: flex; flex-wrap: wrap; gap: 8px 18px; font: 13px/1.4 var(--mono); color: var(--muted); }
.meta b { color: var(--fg); font-weight: 600; }
.scroll { overflow-x: auto; min-width: 0; }
table { border-collapse: collapse; font-variant-numeric: tabular-nums; width: 100%; }
th, td { padding: 7px 10px; text-align: right; border-bottom: 1px solid var(--rule); white-space: nowrap; }
th { font: 600 12px/1.3 var(--mono); color: var(--muted); letter-spacing: .03em; text-transform: uppercase; vertical-align: bottom; }
th:first-child, td:first-child { text-align: left; }
td { font: 14px/1.35 var(--mono); }
tr.axos td { background: var(--accent-soft); }
td.best { color: var(--good); font-weight: 600; }
td.fail { color: var(--bad); }
td.loose { color: var(--warn); }
.grid-wrap { max-height: 640px; overflow: auto; border: 1px solid var(--rule); border-radius: 6px; }
.grid-wrap thead th { position: sticky; top: env(safe-area-inset-top, 0px); background: var(--bg); }
.cols { display: grid; grid-template-columns: repeat(2, minmax(0, 1fr)); gap: 20px 32px; }
@media (max-width: 640px) { .cols { grid-template-columns: minmax(0, 1fr); } }
.cols > * { min-width: 0; display: grid; gap: 6px; align-content: start; }
#findings h3 { color: var(--accent); }
.small { font-size: 14px; color: var(--muted); }
ul.plain { margin: 0; padding-left: 20px; display: grid; gap: 8px; }
ul.plain li { max-width: 74ch; }
code { font-family: var(--mono); font-size: 13px; }
.tag { display: inline-block; padding: 1px 8px; border-radius: 999px; font: 600 11px/1.6 var(--mono); background: var(--accent-soft); color: var(--accent); margin-right: 6px; }
</style>
"""


def esc(s):
    return html.escape(str(s))


def label(s):
    return LABEL.get(s, s)


def fmt_num(v, digits=6):
    if v is None or (isinstance(v, float) and not math.isfinite(v)):
        return "–"
    a = abs(v)
    if a != 0 and (a >= 1e7 or a < 1e-3):
        return f"{v:.{digits - 1}e}"
    return f"{v:.{digits}g}"


def summary_table(d):
    rows = []
    order = d["order"]
    for s in order:
        r = d["solvers"][s]
        cls = ' class="axos"' if s.startswith("axos") else ""
        rows.append(
            f"<tr{cls}><td>{esc(label(s))}</td><td>{r['solved']}</td><td>{r['wrong']}</td><td>{r['feasible']}</td>"
            f"<td>{r['sgm10']:.1f}</td><td>{r['gap_mean']:.3f}</td><td>{r['gap_1e4']}</td><td>{r['gap_1pct']}</td></tr>")
    n = d["solvers"][order[0]]["n"] if order else 0
    return (f"<div class=\"scroll\"><table><thead><tr><th>solver</th><th>solved</th><th>wrong</th>"
            f"<th>with a solution</th><th>SGM10 [s]</th><th>mean primal gap</th><th>gap ≤ 1e-4</th>"
            f"<th>gap ≤ 1%</th></tr></thead><tbody>{''.join(rows)}</tbody></table></div>"
            f"<p class=\"small\">{n} instances, {d['time_limit']:.0f} s each. Solved: reported optimal (or infeasible), "
            f"the solution passes the feasibility check on the original model, and the objective matches the MIPLIB "
            f"reference within 1e-4 (relative). Primal gap: |obj − ref| / max(|obj|, |ref|) at the end, 1 without a "
            f"solution. SGM10: shifted geometric mean of the time, unsolved counted at the limit.</p>")


def per_instance(d):
    order = d["order"]
    names = sorted({p for s in order for p in d["solvers"][s]["per"]})
    head = "".join(f"<th>{esc(label(s))}</th>" for s in order)
    body = []
    for p in names:
        cells = []
        ref = None
        best_gap = min((d["solvers"][s]["per"].get(p, {}).get("gap", 1.0) for s in order), default=1.0)
        for s in order:
            e = d["solvers"][s]["per"].get(p)
            if not e:
                cells.append("<td>–</td>")
                continue
            ref = e.get("ref") if e.get("ref") is not None else ref
            if e["ref_kind"] == "inf":
                txt = "infeasible ✓" if e["ok"] else ("TL" if e["status"] == "time_limit" else esc(e["status"]))
                cls = "best" if e["ok"] else ("fail" if e["wrong"] else "")
            elif e["objective"] is None:
                txt, cls = "no solution", "fail"
            else:
                txt = fmt_num(e["objective"], 7)
                if e["ok"]:
                    txt += f" ✓ {e['seconds']:.1f}s"
                cls = "best" if e["gap"] <= best_gap + 1e-12 and e["gap"] <= 1e-4 else (
                    "loose" if e["gap"] > 0.01 else "")
                if e["wrong"]:
                    cls = "fail"
                    txt += " (wrong)"
            cells.append(f"<td class=\"{cls}\">{txt}</td>")
        refs = "infeasible" if any(d["solvers"][s]["per"].get(p, {}).get("ref_kind") == "inf" for s in order) \
            else fmt_num(ref, 7)
        body.append(f"<tr><td>{esc(p)}</td><td>{refs}</td>{''.join(cells)}</tr>")
    return (f"<div class=\"grid-wrap\"><table><thead><tr><th>instance</th><th>best known</th>{head}</tr></thead>"
            f"<tbody>{''.join(body)}</tbody></table></div>"
            "<p class=\"small\">Final objective of each solver; ✓ with the time when solved to optimality. "
            "<span style=\"color:var(--good);font-weight:600\">Green</span>: matches the best known value; "
            "<span style=\"color:var(--warn)\">amber</span>: primal gap above 1%; "
            "<span style=\"color:var(--bad)\">red</span>: no solution or a wrong claim.</p>")


def lp_root_table(lp):
    import re
    rows = []
    for e in lp:
        m = re.search(r"([\d]+) nonzeros", e.get("info", ""))
        nnz = m.group(1) if m else "–"
        c, g = e["simplex"], e["hpr_gpu"]
        def cell(r):
            if r["status"] != "optimal":
                return f"{esc(r['status'])} ({r['seconds']:.1f} s)" if r.get("seconds") else esc(r["status"])
            return f"{r['seconds']:.2f}"
        sp = (c["seconds"] / g["seconds"]) if c["status"] == "optimal" and g["status"] == "optimal" else None
        cls = "best" if sp and sp > 1.05 else ""
        rows.append(f"<tr><td>{esc(e['name'])}</td><td>{nnz}</td><td>{cell(c)}</td><td>{cell(g)}</td>"
                    f"<td>{fmt_num(g.get('objective'), 8)}</td><td class=\"{cls}\">{'%.1f×' % sp if sp else '–'}</td></tr>")
    return ("<h3>LP relaxation: dual simplex vs HPR on the GPU</h3><div class=\"scroll\"><table><thead><tr>"
            "<th>instance</th><th>nonzeros</th><th>dual simplex [s]</th><th>HPR on GPU [s]</th><th>HPR objective</th>"
            "<th>speedup</th></tr></thead><tbody>" + "".join(rows) + "</tbody></table></div>"
            "<p class=\"small\">The root LP relaxation through the axos command line (30 s limit): the AXOS dual "
            "simplex (exact vertex) and HPR, AXOS's first-order solver, on the GPU at relative KKT tolerance 1e-4. "
            "In the MILP solver the GPU LP seeds rounding, the GPU walkers and RENS on models from 50,000 nonzeros; "
            "crossover from its point to a simplex basis measured slower than the cold dual simplex, so the tree "
            "still starts from the simplex.</p>")


def gpu_section(g, lproot=None):
    if not g and not lproot:
        return ""
    g = g or {}
    out = ["<section id=\"gpu\"><h2>GPU components</h2>",
           "<p class=\"small\">Measured with <code>test_gpu</code> on the same machine with nothing else running: "
           "each GPU component against its CPU counterpart on the presolved model.</p>"]
    pr = g.get("probing", [])
    if pr:
        rows = []
        for e in pr:
            sp = e["cpu_rate"] and e["gpu_rate"] and e["gpu_rate"] / e["cpu_rate"]
            cls = "best" if sp and sp > 1.05 else ("fail" if sp and sp < 0.95 else "")
            rows.append(f"<tr><td>{esc(e['name'])}</td><td>{e['nnz']}</td><td>{e['binaries']}</td>"
                        f"<td>{e['cpu_s']:.3f}</td><td>{e['cpu_probes']}</td><td>{e['gpu_s']:.3f}</td>"
                        f"<td>{e['gpu_probes']}</td><td class=\"{cls}\">{sp:.1f}×</td>"
                        f"<td>{e['cpu_fixed']} / {e['gpu_fixed']}</td></tr>")
        out.append("<h3>Double probing on the binaries</h3>"
                   "<div class=\"scroll\"><table><thead><tr><th>instance</th><th>nonzeros</th><th>binaries</th>"
                   "<th>CPU [s]</th><th>CPU probes</th><th>GPU [s]</th><th>GPU probes</th>"
                   "<th>probes/s GPU ÷ CPU</th><th>fixed CPU / GPU</th></tr></thead><tbody>" + "".join(rows) +
                   "</tbody></table></div>"
                   "<p class=\"small\">Every binary propagated at 0 and at 1 (30 s cap). CPU: the queue-based "
                   "propagator, one probe at a time. GPU: 256 probes at once, each propagated over row and column "
                   "frontiers (only the rows a change touches), so a probe costs what its propagation touches; "
                   "conclusions are merged on the device. Speed is compared as probes per second.</p>")
    lp = g.get("batch_lp", [])
    if lp:
        rows = [f"<tr><td>{esc(e['name'])}</td><td>{e['children']}</td><td>{e['iters']}</td><td>{e['gpu_s']:.4f}</td>"
                f"<td>{e['share']:.2f}</td><td>{e['exact_s']:.4f}</td><td>{e['above']}</td></tr>" for e in lp]
        out.append("<h3>Batched LP bounds for strong branching</h3>"
                   "<div class=\"scroll\"><table><thead><tr><th>instance</th><th>children</th><th>PDHG iterations</th>"
                   "<th>GPU [s]</th><th>share of the exact gain</th><th>exact dual simplex [s]</th>"
                   "<th>bounds above exact</th></tr></thead><tbody>" + "".join(rows) + "</tbody></table></div>"
                   "<p class=\"small\">All children of up to 16 branching candidates solved together by PDHG on the "
                   "GPU, warm-started from the parent; any dual iterate gives a valid Lagrangian bound (never above "
                   "the exact child LP, last column). The share is how much of the exact bound increase the GPU "
                   "bound recovers, mean over children whose bound increases.</p>")
    fj = g.get("fj", [])
    if fj:
        rows = [f"<tr><td>{esc(e['name'])}</td><td>{esc(e['cpu'])}</td><td>{esc(e['gpu'])}</td><td>{e['moves']}</td></tr>"
                for e in fj]
        out.append("<h3>Feasibility jump</h3><div class=\"scroll\"><table><thead><tr><th>instance</th>"
                   "<th>CPU (1 walker)</th><th>GPU (64 walkers)</th><th>GPU moves</th></tr></thead><tbody>" +
                   "".join(rows) + "</tbody></table></div>")
    if lproot:
        out.append(lp_root_table(lproot))
    out.append("</section>")
    return "".join(out)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--set", action="append", required=True)
    ap.add_argument("--gpu")
    ap.add_argument("--lp-root")
    ap.add_argument("--notes")
    ap.add_argument("--machine", action="append", default=[])
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    sets = [json.load(open(p)) for p in a.set]
    gpu = json.load(open(a.gpu)) if a.gpu else None
    lproot = json.load(open(a.lp_root)) if a.lp_root else None
    notes = open(a.notes, encoding="utf-8").read() if a.notes else ""
    meta = "".join(f"<span>{esc(k)} <b>{esc(v)}</b></span>" for k, v in (m.split("=", 1) for m in a.machine))
    parts = [PAGE_HEAD, "<div class=\"wrap\"><header>",
             "<div class=\"eyebrow\">SIH 2026 · PS 119 · sovereign optimization solver</div>",
             "<h1>AXOS MILP Benchmarks</h1>",
             "<p class=\"lede\">AXOS's branch-and-cut solver, written from scratch with no solver library "
             "underneath, against HiGHS, SCIP and CBC on MIPLIB 2017 instances. Every solver gets the same MPS "
             "file, one thread, the same time limit and gap, and every reported solution is checked on the "
             "original model.</p>",
             f"<div class=\"meta\">{meta}</div></header>"]
    if notes:
        parts.append(f"<section id=\"findings\">{notes}</section>")
    for d in sets:
        parts.append(f"<section><h2>{esc(d['title'])}</h2>{summary_table(d)}</section>")
    if gpu or lproot:
        parts.append(gpu_section(gpu, lproot))
    for d in sets:
        parts.append(f"<section><h2>Per instance: {esc(d['title'])}</h2>{per_instance(d)}</section>")
    vers = "; ".join(f"{label(k)} {v}" for k, v in VERSION.items())
    parts.append("<section><h2>Method</h2><ul class=\"plain\">"
                 f"<li>Solvers: {esc(vers)}; AXOS 1.0, CPU build and the same build with the GPU components on. "
                 "Every solver single-threaded, relative gap 1e-4 (the MIPLIB criterion), its own process per "
                 "instance, killed at 1.5× the time limit + 60 s.</li>"
                 "<li>Instances: the 50 smallest 'easy' instances of the MIPLIB 2017 benchmark set (up to 12,528 "
                 "nonzeros) and 16 larger easy instances (51k–392k nonzeros). Reference values: the MIPLIB 2017 "
                 "solution file (v31).</li>"
                 "<li>Feasibility check of every reported solution on the original model, read independently with "
                 "HiGHS' MPS reader: bounds, rows (relative 1e-6) and integrality.</li>"
                 "<li>The AXOS runs and the comparator runs overlapped in time (each single-threaded on a 16-thread "
                 "machine); nothing else ran.</li></ul></section>")
    parts.append("</div>")
    open(a.out, "w", encoding="utf-8").write("\n".join(parts))
    print("wrote", a.out)


if __name__ == "__main__":
    main()
