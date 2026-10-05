#!/usr/bin/env bash
# The remaining AXOS entries with the final build (free-variable sweep order,
# auto fallback rule): HPR-QP on the CPU and auto on Maros-Meszaros, HPR-QP on
# GPU and CPU and auto on the large problems. The previous rows are kept in
# results/final_backup. Needs the machine awake: a standby during a solve
# makes its time meaningless (check the System log, Kernel-Power 506/507).
set -u
ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
cd "$ROOT"
echo "[$(date '+%Y-%m-%d %H:%M:%S')] start"
SOLVERS="axos-hprqp-cpu axos-auto" bash benchmarks/qp/run_all.sh
TL=120 SOLVERS="axos-hprqp-gpu axos-hprqp-cpu axos-auto" bash benchmarks/qp/run_all.sh \
    benchmarks/qp/data/large_axqp benchmarks/qp/data/large_axqp benchmarks/qp/results/large
echo "[$(date '+%Y-%m-%d %H:%M:%S')] final done"
