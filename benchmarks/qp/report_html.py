#!/usr/bin/env python3
"""Renders the QP benchmark summaries (report.py --json) as one HTML page.

  python report_html.py --set mm.json --set large.json [--speedup speedup.csv]
      [--machine "..."] --out report.html

The page is self-contained (data embedded, charts drawn by inline script).
"""
import argparse
import csv
import json

PAGE = r"""<title>AXOS QP Benchmarks</title>
<link rel="preconnect" href="https://fonts.googleapis.com">
<link rel="stylesheet" href="https://fonts.googleapis.com/css2?family=Archivo:wght@500;700;800&family=Source+Sans+3:wght@400;600&family=JetBrains+Mono:wght@400;600&display=swap">
<style>
/* layout: one reading column (tables and charts may run wider, each in its own scroller) */
:root {
  --bg: #f6f7f8; --panel: #ffffff; --fg: #17202a; --muted: #5b6672; --rule: #d9dee3;
  --accent: #0a7c86; --accent-soft: #d8eef0; --good: #1f7a3f; --warn: #a15c00; --bad: #b3261e;
  --s0: #0a7c86; --s1: #3fb3bd; --s2: #86d3d9; --s3: #c2410c; --s4: #f28c4b;
  --s5: #4b5563; --s6: #7c8795; --s7: #1d4ed8; --s8: #7c3aed; --s9: #a8a29e;
  --display: "Archivo", "Arial Narrow", Arial, sans-serif;
  --body: "Source Sans 3", "Segoe UI", system-ui, sans-serif;
  --mono: "JetBrains Mono", ui-monospace, Consolas, monospace;
}
@media (prefers-color-scheme: dark) { :root:not([data-theme="light"]) {
  --bg: #11161b; --panel: #182028; --fg: #e4e9ee; --muted: #9aa6b2; --rule: #2c3743;
  --accent: #4cc3cc; --accent-soft: #173a3e; --good: #5cc98a; --warn: #e3a452; --bad: #f2837a;
  --s0: #4cc3cc; --s1: #2f8f97; --s2: #9adfe4; --s3: #f07a3c; --s4: #f7b27f;
  --s5: #aab4bf; --s6: #748290; --s7: #7aa2f7; --s8: #b196f5; --s9: #d6d3d1; color-scheme: dark } }
:root[data-theme="dark"] {
  --bg: #11161b; --panel: #182028; --fg: #e4e9ee; --muted: #9aa6b2; --rule: #2c3743;
  --accent: #4cc3cc; --accent-soft: #173a3e; --good: #5cc98a; --warn: #e3a452; --bad: #f2837a;
  --s0: #4cc3cc; --s1: #2f8f97; --s2: #9adfe4; --s3: #f07a3c; --s4: #f7b27f;
  --s5: #aab4bf; --s6: #748290; --s7: #7aa2f7; --s8: #b196f5; --s9: #d6d3d1; color-scheme: dark }
body { background: var(--bg); color: var(--fg); font: 16px/1.55 var(--body); }
.wrap { max-width: 1080px; margin: 0 auto; padding-inline: 20px; padding-block: 36px 64px;
  display: grid; gap: 44px; }
header { display: grid; gap: 10px; }
.eyebrow { font: 600 12px/1 var(--mono); letter-spacing: .08em; text-transform: uppercase; color: var(--accent); }
h1 { font: 800 clamp(30px, 5vw, 46px)/1.05 var(--display); margin: 0; text-wrap: balance; letter-spacing: -.01em; }
h2 { font: 700 24px/1.2 var(--display); margin: 0; text-wrap: balance; }
h3 { font: 700 17px/1.3 var(--display); margin: 0; }
p { margin: 0; max-width: 68ch; }
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
td .dot { display: inline-block; width: 9px; height: 9px; border-radius: 50%; margin-right: 8px; vertical-align: 1px; }
.pill { display: inline-block; padding: 1px 8px; border-radius: 999px; font: 600 11px/1.6 var(--mono); }
.pill.gpu { background: var(--accent-soft); color: var(--accent); }
.pill.cpu { background: color-mix(in srgb, var(--muted) 18%, transparent); color: var(--muted); }
.chart { background: var(--panel); border: 1px solid var(--rule); border-radius: 6px; padding: 14px 8px 8px; }
.chart svg { display: block; width: 100%; height: auto; }
.legend { display: flex; flex-wrap: wrap; gap: 6px 16px; font: 13px/1.4 var(--mono); padding: 4px 8px 0; }
.legend span { display: inline-flex; align-items: center; gap: 6px; }
.legend i { width: 18px; height: 3px; border-radius: 2px; display: inline-block; }
.controls { display: flex; flex-wrap: wrap; gap: 10px 18px; align-items: center; font-size: 14px; }
.controls input[type=search] { font: 14px var(--mono); padding: 6px 10px; border: 1px solid var(--rule);
  border-radius: 4px; background: var(--panel); color: var(--fg); min-width: 0; width: 220px; max-width: 100%; }
.controls select { font: 14px var(--mono); padding: 5px 8px; border: 1px solid var(--rule); border-radius: 4px; background: var(--panel); color: var(--fg); }
.controls label { display: inline-flex; gap: 6px; align-items: center; }
:focus-visible { outline: 2px solid var(--accent); outline-offset: 2px; }
.grid td.ok { color: var(--fg); }
.grid td.best { color: var(--good); font-weight: 600; }
.grid td.fail { color: var(--bad); }
.grid td.loose { color: var(--warn); }
.grid thead th { position: sticky; top: env(safe-area-inset-top, 0px); background: var(--bg); }
.grid-wrap { max-height: 640px; overflow: auto; border: 1px solid var(--rule); border-radius: 6px; }
.notes { display: grid; gap: 10px; }
.notes li { max-width: 72ch; }
code, pre { font-family: var(--mono); font-size: 13px; }
pre { background: var(--panel); border: 1px solid var(--rule); border-radius: 6px; padding: 12px 14px; overflow-x: auto; margin: 0; }
.cols { display: grid; grid-template-columns: repeat(auto-fit, minmax(300px, 1fr)); gap: 24px; }
#findings .cols { grid-template-columns: repeat(2, minmax(0, 1fr)); gap: 20px 32px; }
@media (max-width: 640px) { #findings .cols { grid-template-columns: minmax(0, 1fr); } }
#findings h3 { color: var(--accent); }
.cols > * { min-width: 0; }
.small { font-size: 14px; color: var(--muted); }
</style>

<div class="wrap">
<header>
  <div class="eyebrow">SIH 2026 · PS 119 · sovereign optimization solver</div>
  <h1>AXOS QP Benchmarks</h1>
  <p class="lede">AXOS's own HPR-QP and PDHCG solvers, written from the published algorithms with no solver library underneath, against the authors' GPU implementations and five established open-source solvers. Every solver gets the same data and is scored with the same KKT metric.</p>
  <div class="meta" id="meta"></div>
</header>

<section id="findings" hidden></section>

<section id="summary"></section>

<section>
  <h2>Performance profile</h2>
  <p class="small">Share of problems a solver solves within a factor τ of the fastest solver on that problem (solved = reported optimal and common KKT error ≤ 10 × tol). Higher and further left is better.</p>
  <div class="controls"><label for="pset">Problem set <select id="pset"></select></label></div>
  <div class="chart"><svg id="profile" viewBox="0 0 900 380" role="img" aria-label="Performance profile"></svg>
  <div class="legend" id="legend"></div></div>
</section>

<section id="speedup-sec" hidden>
  <h2>GPU engineering</h2>
  <p class="small">Microseconds per HPR-QP iteration on the GPU at each stage (solve time without setup, divided by iterations; problems that ran at least 20,000 iterations, so check iterations are amortized). Stage 1: one kernel per vector operation, about 10 launches per iteration. Stage 2: vector updates fused into the sparse products (4 kernels per iteration), iteration scalars in device memory, iterations between checks replayed as CUDA graphs, dense rows split across warps. Stage 3: no FP64 division left in the per-element steps (this GPU runs FP64 at 1/64 of its FP32 rate), and the epilogues of multi-thread rows moved to consecutive threads for coalesced loads.</p>
  <div class="scroll"><table id="speedup"></table></div>
</section>

<section>
  <h2>Per problem</h2>
  <p class="small">Solve time in seconds when solved; <span style="color:var(--good);font-weight:600">green</span> is the fastest solver on that problem. <span style="color:var(--warn)">≈</span> reported optimal but misses the common metric by more than 10×; TL time limit; ✗ other failure.</p>
  <div class="controls">
    <label for="gset">Problem set <select id="gset"></select></label>
    <input type="search" id="q" placeholder="Filter problems" aria-label="Filter problems">
    <label><input type="checkbox" id="hard"> Only problems some solver failed</label>
  </div>
  <div class="grid-wrap"><table class="grid" id="grid"></table></div>
</section>

<section class="notes">
  <h2>How it was measured</h2>
  <ul>
    <li><b>Metric.</b> η<sub>p</sub> = ‖Ax − P<sub>K</sub>(Ax)‖∞ and bound violation over 1 + max(‖b‖∞, ‖Ax‖∞); η<sub>d</sub> = ‖Qx + c − Aᵀy − z‖∞ over 1 + max(‖c‖∞, ‖Aᵀy‖∞, ‖Qx‖∞) with z the projection onto the bounds' sign cone; η<sub>gap</sub> = |P − D| / (1 + max(|P|, |D|)). Each solver's duals are mapped to one sign convention first.</li>
    <li><b>Solved</b> means the solver reports optimal and max(η<sub>p</sub>, η<sub>d</sub>, η<sub>gap</sub>) ≤ 10 × tol (every solver measures its own tolerance in its own norms). The strict column uses 1 × tol; AXOS stops on this metric itself.</li>
    <li><b>Time</b> is wall clock of the solve after the data are in memory: preconditioning, transfer to the GPU, factorizations or spectral estimates, iterations. Kernel and JIT compilation are excluded for every solver (the Julia solvers are warmed up on each problem first). <b>SGM10</b> is the shifted geometric mean with a 10 s shift, unsolved problems at the time limit.</li>
    <li><b>Objective error</b> is |f − f<sub>ref</sub>| / (1 + |f<sub>ref</sub>|) against PIQP run to 1e-9.</li>
    <li>One solver runs at a time; CPU solvers run in a separate process that is killed at 1.5 × the time limit.</li>
  </ul>
  <pre>bash benchmarks/qp/run_all.sh
python benchmarks/qp/report.py benchmarks/qp/results/mm --ref benchmarks/qp/results/mm_ref/piqp.csv</pre>
</section>
</div>

<script>
const DATA = __DATA__;
const LABEL = {"axos-hprqp-gpu": "AXOS HPR-QP", "axos-hprqp-cpu": "AXOS HPR-QP", "axos-hprqp-auto": "AXOS HPR-QP auto",
  "axos-pdhcg-gpu": "AXOS PDHCG", "axos-ipm-cpu": "AXOS IPM", "axos-auto": "AXOS auto", "hprqp-jl": "HPR-QP.jl", "pdhcg-jl": "PDHCG.jl", "clarabel": "Clarabel",
  "piqp": "PIQP", "highs": "HiGHS", "osqp": "OSQP", "scs": "SCS"};
const COLOR = {"axos-hprqp-gpu": "--s0", "axos-hprqp-auto": "--s1", "axos-hprqp-cpu": "--s2", "axos-pdhcg-gpu": "--s1", "axos-ipm-cpu": "--s2", "axos-auto": "--s0",  "hprqp-jl": "--s3", "pdhcg-jl": "--s4", "clarabel": "--s7", "piqp": "--s8", "highs": "--s5", "osqp": "--s6", "scs": "--s9"};
const col = s => `var(${COLOR[s] || "--s9"})`;
const lab = s => LABEL[s] || s;
const isAxos = s => s.startsWith("axos");
const fmt = (v, d = 2) => v == null ? "–" : (v < 0.01 ? v.toExponential(1) : v.toFixed(d));
const el = (t, a = {}, ...kids) => { const e = document.createElement(t);
  for (const [k, v] of Object.entries(a)) (k === "class" ? e.className = v : e.setAttribute(k, v));
  for (const k of kids) e.append(k); return e; };

document.getElementById("meta").innerHTML = DATA.machine.map(([k, v]) => `<span>${k} <b>${v}</b></span>`).join("");

// summary tables
const sum = document.getElementById("summary");
for (const set of DATA.sets) {
  const sec = el("div", {class: "cols"});
  const box = el("div", {}, el("h2", {}, set.title));
  box.append(el("p", {class: "small"}, `${set.problems.length} problems · tol ${set.tol} · time limit ${set.time_limit} s per problem`));
  const t = el("table"); const sc = el("div", {class: "scroll"}, t);
  t.innerHTML = `<thead><tr><th>solver</th><th>device</th><th>solved</th><th>strict</th><th>SGM10 s</th><th>median obj. err.</th></tr></thead>`;
  const tb = el("tbody");
  for (const s of set.order) {
    const v = set.summary[s];
    const tr = el("tr", {class: isAxos(s) ? "axos" : ""});
    tr.innerHTML = `<td><span class="dot" style="background:${col(s)}"></span>${lab(s)}</td>
      <td>${s === "axos-auto" ? '<span class="pill gpu">CPU + GPU</span>' : `<span class="pill ${v.device.startsWith("gpu") ? "gpu" : "cpu"}">${v.device.startsWith("gpu") ? "GPU" : "CPU"}</span>`}</td>
      <td>${v.solved} / ${v.total}</td><td>${v.strict}</td><td>${v.sgm10.toFixed(2)}</td>
      <td>${v.objerr_median == null ? "–" : v.objerr_median.toExponential(1)}</td>`;
    tb.append(tr);
  }
  t.append(tb); box.append(sc); sec.append(box); sum.append(sec);
}

// performance profile (log2 tau axis)
function profile(set) {
  const svg = document.getElementById("profile"); svg.innerHTML = "";
  const W = 900, H = 380, L = 56, R = 16, T = 12, B = 44;
  const tmax = 1024, x = t => L + (W - L - R) * Math.log2(t) / Math.log2(tmax), y = f => T + (H - T - B) * (1 - f);
  const ns = "http://www.w3.org/2000/svg", mk = (t, a) => { const e = document.createElementNS(ns, t); for (const k in a) e.setAttribute(k, a[k]); return e; };
  for (const f of [0, .25, .5, .75, 1]) {
    svg.append(mk("line", {x1: L, x2: W - R, y1: y(f), y2: y(f), stroke: "var(--rule)", "stroke-width": 1}));
    const tx = mk("text", {x: L - 8, y: y(f) + 4, "text-anchor": "end", fill: "var(--muted)", "font-size": 12, "font-family": "var(--mono)"}); tx.textContent = f.toFixed(2); svg.append(tx);
  }
  for (const t of [1, 2, 4, 8, 16, 32, 64, 128, 256, 512, 1024]) {
    svg.append(mk("line", {x1: x(t), x2: x(t), y1: T, y2: H - B, stroke: "var(--rule)", "stroke-width": 1, "stroke-dasharray": "2 4"}));
    const tx = mk("text", {x: x(t), y: H - B + 18, "text-anchor": "middle", fill: "var(--muted)", "font-size": 12, "font-family": "var(--mono)"}); tx.textContent = t; svg.append(tx);
  }
  const xl = mk("text", {x: (L + W - R) / 2, y: H - 6, "text-anchor": "middle", fill: "var(--muted)", "font-size": 13, "font-family": "var(--mono)"}); xl.textContent = "τ (time relative to the fastest solver)"; svg.append(xl);
  const best = {};
  for (const p of set.problems) { let b = Infinity; for (const s of set.order) { const r = set.per[s][p]; if (r.ok) b = Math.min(b, Math.max(r.time, 1e-4)); } best[p] = b; }
  const legend = document.getElementById("legend"); legend.innerHTML = "";
  for (const s of [...set.order].reverse()) {
    const ratios = set.problems.filter(p => set.per[s][p].ok).map(p => Math.max(set.per[s][p].time, 1e-4) / best[p]).sort((a, b) => a - b);
    let d = `M ${x(1)} ${y(0)}`, cnt = 0;
    for (const r of ratios) { const rr = Math.min(r, tmax); d += ` H ${x(rr)}`; cnt++; d += ` V ${y(cnt / set.problems.length)}`; }
    d += ` H ${x(tmax)}`;
    svg.append(mk("path", {d, fill: "none", stroke: col(s), "stroke-width": isAxos(s) ? 3 : 1.75, "stroke-linejoin": "round"}));
  }
  for (const s of set.order) legend.append(el("span", {}, el("i", {style: `background:${col(s)}`}), lab(s) + (s.endsWith("-cpu") ? " (CPU)" : "")));
}

// per-problem grid
function grid(set) {
  const q = document.getElementById("q").value.trim().toUpperCase(), hard = document.getElementById("hard").checked;
  const t = document.getElementById("grid");
  let h = `<thead><tr><th>problem</th><th>n</th>` + set.order.map(s => `<th style="color:${col(s)}">${lab(s)}${s.endsWith("-cpu") ? "<br>CPU" : ""}</th>`).join("") + `</tr></thead><tbody>`;
  for (const p of set.problems) {
    if (q && !p.includes(q)) continue;
    const rs = set.order.map(s => set.per[s][p]);
    if (hard && rs.every(r => r.ok)) continue;
    const b = Math.min(...rs.filter(r => r.ok).map(r => r.time));
    h += `<tr><td>${p}</td><td>${set.sizes[p] ?? ""}</td>` + rs.map(r => {
      if (r.ok) return `<td class="${r.time === b ? "best" : "ok"}">${fmt(r.time)}</td>`;
      if (r.status === "optimal") return `<td class="loose" title="KKT ${r.kkt}">≈</td>`;
      return `<td class="fail" title="${r.status}">${r.status === "time_limit" ? "TL" : "✗"}</td>`;
    }).join("") + `</tr>`;
  }
  t.innerHTML = h + "</tbody>";
}

for (const id of ["pset", "gset"]) {
  const s = document.getElementById(id);
  DATA.sets.forEach((set, i) => s.append(el("option", {value: i}, set.title)));
}
const cur = id => DATA.sets[+document.getElementById(id).value];
document.getElementById("pset").onchange = () => profile(cur("pset"));
for (const id of ["gset", "q", "hard"]) document.getElementById(id).addEventListener("input", () => grid(cur("gset")));
profile(DATA.sets[0]); grid(DATA.sets[0]);

if (DATA.notes) {
  const f = document.getElementById("findings");
  f.innerHTML = DATA.notes;
  f.hidden = false;
}

if (DATA.speedup.rows && DATA.speedup.rows.length) {
  document.getElementById("speedup-sec").hidden = false;
  const t = document.getElementById("speedup"), st = DATA.speedup.stages;
  t.innerHTML = `<thead><tr><th>problem</th><th>n</th><th>nnz(A) + nnz(Q)</th>` +
    st.map(x => `<th>${x}<br>µs / it</th>`).join("") + `<th>speed-up</th></tr></thead><tbody>` +
    DATA.speedup.rows.map(r => `<tr><td>${r.problem}</td><td>${r.n}</td><td>${r.nnz}</td>` +
      r.us.map(u => `<td>${u == null ? "–" : u.toFixed(1)}</td>`).join("") +
      `<td>${(r.us[0] / r.us[r.us.length - 1]).toFixed(1)}×</td></tr>`).join("") + "</tbody>";
}
</script>
"""


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--set", action="append", required=True, help="report.py --json output")
    ap.add_argument("--stage", action="append", default=[],
                    help="label=csv of an AXOS GPU run, in order (engineering table)")
    ap.add_argument("--min-iters", type=int, default=20000)
    ap.add_argument("--machine", action="append", default=[], help="key=value")
    ap.add_argument("--notes", help="HTML fragment shown above the tables (key findings)")
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    sets = []
    for path in a.set:
        sets.append(json.load(open(path)))
    speed = dict(stages=[], rows=[])
    if a.stage:
        runs = []
        for st in a.stage:
            label, path = st.split("=", 1)
            speed["stages"].append(label)
            runs.append({r["problem"]: r for r in csv.DictReader(open(path))})

        def us(r):
            try:
                it = int(r["iterations"])
                return 1e6 * (float(r["seconds"]) - float(r["setup_seconds"])) / it if it else None
            except (TypeError, ValueError, KeyError):
                return None
        last = runs[-1]
        for pb in sorted(last):
            vals = [us(run.get(pb, {})) for run in runs]
            r = last[pb]
            if None in vals or int(r["iterations"] or 0) < a.min_iters:
                continue
            speed["rows"].append(dict(problem=pb, n=r["n"], nnz=int(r["nnzA"]) + int(r["nnzQ"]), us=vals))
        speed["rows"].sort(key=lambda x: x["nnz"])
        rows = speed["rows"]
        if len(rows) > 12:  # a dozen, evenly spread over the problem sizes
            speed["rows"] = [rows[round(k * (len(rows) - 1) / 11)] for k in range(12)]
    machine = [kv.split("=", 1) for kv in a.machine]
    notes = open(a.notes, encoding="utf-8").read() if a.notes else ""
    data = dict(sets=sets, speedup=speed, machine=machine, notes=notes)
    txt = json.dumps(data, separators=(",", ":")).replace("</", "<\\/")
    with open(a.out, "w", encoding="utf-8") as f:
        f.write(PAGE.replace("__DATA__", txt))
    print("wrote", a.out)


if __name__ == "__main__":
    main()
