#!/usr/bin/env bash
# Second pass of the QP benchmark (after run_all.sh on Maros-Meszaros):
#   1. keeps the first build's AXOS rows (results/mm_build1) for the
#      engineering comparison, re-runs the AXOS HPR-QP entries with the final
#      build and adds the interior point and auto entries (AXOS PDHCG keeps
#      its first-build results: later kernel work only makes it faster)
#   2. the objective reference: PIQP at 1e-9 (results/mm_ref)
#   3. the generated large problems, all solvers but HiGHS (results/large)
set -u
ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
cd "$ROOT"
R=benchmarks/qp/results
mkdir -p "$R/mm_build1"
for f in "$R"/mm/axos-*.csv; do
    [ -f "$f" ] && cp "$f" "$R/mm_build1/"
done
SOLVERS="axos-hprqp-gpu axos-hprqp-cpu axos-ipm-cpu axos-auto" \
    bash benchmarks/qp/run_all.sh
TOL=1e-9 SOLVERS=piqp bash benchmarks/qp/run_all.sh benchmarks/qp/data/maros_meszaros \
    benchmarks/qp/data/mm_axqp "$R/mm_ref"
TL=120 SOLVERS="axos-hprqp-gpu axos-pdhcg-gpu hprqp-jl pdhcg-jl axos-hprqp-cpu axos-ipm-cpu axos-auto clarabel piqp osqp scs" \
    bash benchmarks/qp/run_all.sh benchmarks/qp/data/large_axqp benchmarks/qp/data/large_axqp "$R/large"
echo "[$(date +%H:%M:%S)] follow-up done"
