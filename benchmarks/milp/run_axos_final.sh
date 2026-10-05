#!/usr/bin/env bash
# The AXOS entries of the MILP benchmark: CPU and GPU builds of the same
# solver on the 50 MIPLIB instances, then on the 16 larger ones.
#   bash benchmarks/milp/run_axos_final.sh
set -u
ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
cd "$ROOT"
TL="${TL:-60}"
echo "[$(date '+%Y-%m-%d %H:%M:%S')] start"
TL=$TL SOLVERS="axos" bash benchmarks/milp/run_all.sh benchmarks/milp/data/mps benchmarks/milp/results/miplib50
TL=$TL SOLVERS="axos-gpu" AXOS_FLAGS="--gpu" bash benchmarks/milp/run_all.sh benchmarks/milp/data/mps benchmarks/milp/results/miplib50
TL=$TL SOLVERS="axos" bash benchmarks/milp/run_all.sh benchmarks/milp/data/mps_large benchmarks/milp/results/large16
TL=$TL SOLVERS="axos-gpu" AXOS_FLAGS="--gpu" bash benchmarks/milp/run_all.sh benchmarks/milp/data/mps_large benchmarks/milp/results/large16
echo "[$(date '+%Y-%m-%d %H:%M:%S')] done"
