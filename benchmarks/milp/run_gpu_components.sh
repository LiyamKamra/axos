#!/usr/bin/env bash
# GPU component measurements for the report (run on an otherwise idle
# machine): double probing on the 16 larger instances, and feasibility jump
# plus batched LP bounds on a few small and large ones.
#   bash benchmarks/milp/run_gpu_components.sh
set -u
ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
cd "$ROOT"
OUT=benchmarks/milp/results
PY="$ROOT/.venv/Scripts/python.exe"
[ -x "$PY" ] || PY="$ROOT/.venv/bin/python"
T="$ROOT/build/test_gpu.exe"
[ -x "$T" ] || T="$ROOT/build/test_gpu"
echo "[$(date '+%H:%M:%S')] probing (16 larger instances)"
"$T" benchmarks/milp/data/mps_large --probe-only > "$OUT/gpu_probing.txt" 2>&1
echo "[$(date '+%H:%M:%S')] feasibility jump and batched LP bounds"
"$T" benchmarks/milp/data/mps --only neos5,mas76,pk1,gen-ip002 --fj-seconds 10 > "$OUT/gpu_small.txt" 2>&1
"$T" benchmarks/milp/data/mps_large --only air05,neos-950242,n2seq36q,academictimetablesmall,cod105 \
    --fj-seconds 10 > "$OUT/gpu_large.txt" 2>&1
echo "[$(date '+%H:%M:%S')] LP relaxations: dual simplex vs HPR on the GPU"
"$PY" benchmarks/milp/lp_root.py benchmarks/milp/data/mps_large     air05,cod105,neos-950242,n2seq36q,academictimetablesmall,nursesched-sprint02,sp97ar,rail01     --time-limit 30 --out "$OUT/lp_root.json"
"$PY" benchmarks/milp/parse_gpu.py "$OUT/gpu_probing.txt" "$OUT/gpu_small.txt" "$OUT/gpu_large.txt" \
    --out "$OUT/gpu_components.json"
echo "[$(date '+%H:%M:%S')] done"
