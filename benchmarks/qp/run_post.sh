#!/usr/bin/env bash
# Third pass, after run_followup.sh:
#   1. PDHCG.jl on the large problems one Julia process per problem (it runs
#      out of host memory on CONTROL_L, which ended the single-process run)
#   2. the HPR-QP-based AXOS entries again with the final build (free-variable
#      sweep order, auto fallback rule), on both problem sets
set -u
ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
cd "$ROOT"
R=benchmarks/qp/results
export JULIA_DEPOT_PATH="$ROOT/.tools/julia-depot"
JULIA="$ROOT/.tools/julia-1.10.10/bin/julia.exe"
PY="$ROOT/.venv/Scripts/python.exe"
echo "[$(date +%H:%M:%S)] pdhcg-jl (one process per problem)"
for f in benchmarks/qp/data/large_axqp/*.axqp; do
    p="$(basename "$f" .axqp)"
    grep -q "^$p," "$R/large/pdhcg-jl.csv" 2>/dev/null && continue
    before=$(wc -l < "$R/large/pdhcg-jl.csv")
    "$JULIA" --project=.tools/HPR-QP benchmarks/qp/run_julia.jl pdhcg benchmarks/qp/data/large_axqp \
        1e-6 120 "$R/large/pdhcg-jl.csv" "$R/large/sol/pdhcg-jl" "$p" >> "$R/large/pdhcg-jl.log" 2>&1
    if [ "$(wc -l < "$R/large/pdhcg-jl.csv")" -eq "$before" ]; then # the process died
        echo "$p,,,,,pdhcg-jl,gpu,1e-06,error,,,,,,,,process died" >> "$R/large/pdhcg-jl.csv"
    fi
done
"$PY" benchmarks/qp/qpbench.py score --data benchmarks/qp/data/large_axqp \
    --csv "$R/large/pdhcg-jl.csv" --sols "$R/large/sol/pdhcg-jl"
echo "[$(date +%H:%M:%S)] pdhcg-jl: $(awk -F, 'NR>1{print $9}' "$R/large/pdhcg-jl.csv" | sort | uniq -c | tr '\n' ' ')"
cp "$R/mm/axos-auto.csv" "$R/mm_build1/axos-auto-fallback100k.csv" # the earlier rule, for the record
# the final build (free-variable sweep order, fallback rule) for every
# HPR-QP-based entry, on both problem sets
mkdir -p "$R/large_build1"
for s in axos-hprqp-gpu axos-hprqp-cpu axos-auto; do
    cp "$R/mm/$s.csv" "$R/mm_build1/$s-before-free-order.csv"
    cp "$R/large/$s.csv" "$R/large_build1/$s.csv"
done
SOLVERS="axos-hprqp-gpu axos-hprqp-cpu axos-auto" bash benchmarks/qp/run_all.sh
TL=120 SOLVERS="axos-hprqp-gpu axos-hprqp-cpu axos-auto" bash benchmarks/qp/run_all.sh \
    benchmarks/qp/data/large_axqp benchmarks/qp/data/large_axqp "$R/large"
echo "[$(date +%H:%M:%S)] post done"
