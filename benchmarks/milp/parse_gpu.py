#!/usr/bin/env python3
"""Turns test_gpu output (probing / fj / batch LP lines) into the JSON that
report_html.py --gpu reads.

  test_gpu data/mps_large --fj-seconds 10 > gpu.txt
  python parse_gpu.py gpu.txt --out results/gpu_components.json
"""
import argparse
import json
import re

HEAD = re.compile(r"^== (\S+)\s+n (\d+)\s+m (\d+)\s+nnz (\d+)(?:\s+binaries (\d+))?")
PROBE = re.compile(r"probing: cpu ([\d.]+) s\s+(\d+) probes\s+fixed (\d+)\s+tightened (\d+).*\| gpu ([\d.]+) s\s+(\d+) probes"
                   r"\s+fixed (\d+)\s+tightened (\d+)")
FJ = re.compile(r"fj: cpu (found|none) in ([\d.]+) s \| gpu \(\d+ walkers\) (found|none) in ([\d.]+) s, (\d+) moves")
LP = re.compile(r"batch LP: (\d+) children,\s+(\d+) iters: gpu ([\d.]+) s, mean share of the exact gain ([\d.]+) "
                r"\((\d+) with gain\), bound above exact (\d+) \| exact dual simplex ([\d.]+) s")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("inputs", nargs="+")
    ap.add_argument("--out", required=True)
    ap.add_argument("--lp-iters", type=int, default=400, help="batch LP row to keep per instance")
    a = ap.parse_args()
    out = dict(probing=[], fj=[], batch_lp=[])
    for path in a.inputs:
        cur = None
        for line in open(path, encoding="utf-8", errors="replace"):
            m = HEAD.match(line)
            if m:
                cur = dict(name=m.group(1), n=int(m.group(2)), m=int(m.group(3)), nnz=int(m.group(4)),
                           binaries=int(m.group(5) or 0))
                continue
            if cur is None:
                continue
            m = PROBE.search(line)
            if m and cur["binaries"] > 0 and all(e["name"] != cur["name"] for e in out["probing"]):
                cs, cp, cf, gs, gp, gf = float(m.group(1)), int(m.group(2)), int(m.group(3)), float(m.group(5)), \
                    int(m.group(6)), int(m.group(7))
                out["probing"].append(dict(name=cur["name"], nnz=cur["nnz"], binaries=cur["binaries"], cpu_s=cs,
                                           cpu_probes=cp, cpu_fixed=cf, gpu_s=gs, gpu_probes=gp, gpu_fixed=gf,
                                           cpu_rate=cp / max(cs, 1e-6), gpu_rate=gp / max(gs, 1e-6)))
                continue
            m = FJ.search(line)
            if m:
                out["fj"].append(dict(name=cur["name"], cpu=f"{m.group(1)} in {float(m.group(2)):.2f} s",
                                      gpu=f"{m.group(3)} in {float(m.group(4)):.2f} s", moves=int(m.group(5))))
                continue
            m = LP.search(line)
            if m and int(m.group(2)) == a.lp_iters:
                out["batch_lp"].append(dict(name=cur["name"], children=int(m.group(1)), iters=int(m.group(2)),
                                            gpu_s=float(m.group(3)), share=float(m.group(4)),
                                            with_gain=int(m.group(5)), above=int(m.group(6)),
                                            exact_s=float(m.group(7))))
    json.dump(out, open(a.out, "w"), indent=1)
    print("probing", len(out["probing"]), "fj", len(out["fj"]), "batch LP", len(out["batch_lp"]))


if __name__ == "__main__":
    main()
