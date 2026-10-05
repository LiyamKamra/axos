#!/usr/bin/env bash
# Runs the QP benchmark on one problem set, one solver at a time (never two
# solvers on the GPU or the CPU cores at once), writing
#   $OUT/<solver>.csv          one line per problem (run_qp / qpbench.py columns)
#   $OUT/sol/<solver>/*.sol    solutions of the GPU solvers, re-scored by
#                              `qpbench.py score` with the common metric
#
#   bash benchmarks/qp/run_all.sh [problem dir (.SIF/.QPS)] [axqp dir] [out dir]
# Environment: TOL (1e-6), TL (time limit per problem, 60 s), SOLVERS (list).
set -u
ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
cd "$ROOT"
DATA="${1:-benchmarks/qp/data/maros_meszaros}"
AXQP="${2:-benchmarks/qp/data/mm_axqp}"
OUT="${3:-benchmarks/qp/results/mm}"
TOL="${TOL:-1e-6}"
TL="${TL:-60}"
SOLVERS="${SOLVERS:-axos-hprqp-gpu hprqp-jl pdhcg-jl axos-pdhcg-gpu axos-hprqp-cpu clarabel piqp highs osqp scs}"
PY="$ROOT/.venv/Scripts/python.exe"
[ -x "$PY" ] || PY="$ROOT/.venv/bin/python"
JULIA="$ROOT/.tools/julia-1.10.10/bin/julia.exe"
export JULIA_DEPOT_PATH="$ROOT/.tools/julia-depot"
RUN_QP="$ROOT/build/run_qp.exe"
[ -x "$RUN_QP" ] || RUN_QP="$ROOT/build/run_qp"
mkdir -p "$OUT/sol"

for s in $SOLVERS; do
    csv="$OUT/$s.csv"
    rm -f "$csv"
    rm -rf "$OUT/sol/$s"
    echo "[$(date +%H:%M:%S)] $s"
    case "$s" in
    axos-auto) # interior point when cheap, else HPR-QP on the GPU (large) or CPU
        "$RUN_QP" "$DATA" --method auto --device auto --tol "$TOL" --time-limit "$TL"             --out "$csv" --save-sol "$OUT/sol/$s" > /dev/null 2> "$OUT/$s.log"
        ;;
    axos-hprqp-gpu | axos-hprqp-cpu | axos-pdhcg-gpu | axos-pdhcg-cpu | axos-ipm-cpu)
        dev="${s##*-}"
        m="${s#axos-}"
        m="${m%-*}"
        "$RUN_QP" "$DATA" --device "$dev" --method "$m" --tol "$TOL" --time-limit "$TL" \
            --out "$csv" --save-sol "$OUT/sol/$s" > /dev/null 2> "$OUT/$s.log"
        ;;
    hprqp-jl | pdhcg-jl)
        "$JULIA" --project=.tools/HPR-QP benchmarks/qp/run_julia.jl "${s%-jl}" "$AXQP" "$TOL" "$TL" \
            "$csv" "$OUT/sol/$s" > "$OUT/$s.log" 2>&1
        ;;
    *)
        "$PY" benchmarks/qp/qpbench.py run --solver "$s" --data "$AXQP" --tol "$TOL" \
            --time-limit "$TL" --out "$csv" > "$OUT/$s.log" 2>&1
        ;;
    esac
    if [ -d "$OUT/sol/$s" ]; then
        "$PY" benchmarks/qp/qpbench.py score --data "$AXQP" --csv "$csv" --sols "$OUT/sol/$s"
    fi
    echo "[$(date +%H:%M:%S)] $s: $(awk -F, 'NR>1{print $9}' "$csv" | sort | uniq -c | tr '\n' ' ')"
done
