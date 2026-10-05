#!/usr/bin/env bash
# Builds results/milp_report.html from the benchmark results (after
# run_all.sh, run_axos_final.sh, verify_sol.py and run_gpu_components.sh).
#   bash benchmarks/milp/build_report.sh
set -eu
cd "$(dirname "$0")"
PY=../../.venv/Scripts/python.exe
[ -x "$PY" ] || PY=../../.venv/bin/python
$PY report.py results/miplib50 --time-limit 60 --title "50 MIPLIB 2017 instances (60 s)" --json results/miplib50.json
$PY report.py results/large16 --time-limit 60 --title "16 larger MIPLIB 2017 instances (60 s)" --json results/large16.json
$PY compare.py results/miplib50 --solver axos-gpu --json results/cmp50.json > /dev/null
$PY compare.py results/large16 --solver axos-gpu --json results/cmp16.json > /dev/null
GPU=""
[ -f results/gpu_components.json ] && GPU="--gpu results/gpu_components.json"
LPR=""
[ -f results/lp_root.json ] && LPR="--lp-root results/lp_root.json"
# shellcheck disable=SC2086
$PY make_findings.py --small results/miplib50.json --large results/large16.json --cmp-small results/cmp50.json \
    --cmp-large results/cmp16.json $GPU $LPR --out results/findings.html
# shellcheck disable=SC2086
$PY report_html.py --set results/miplib50.json --set results/large16.json $GPU $LPR --notes results/findings.html \
    --machine "CPU=Core i7-12650H (one thread per solver)" --machine "GPU=RTX 3050 A Laptop, 4 GB" \
    --machine "RAM=16 GB" --machine "time limit=60 s" --machine "gap=1e-4 (relative)" \
    --out results/milp_report.html
