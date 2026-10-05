#!/usr/bin/env bash
# Runs the MILP benchmark, one solver at a time:
#   bash benchmarks/milp/run_all.sh [mps dir] [out dir]
# Environment: TL (time limit per problem, 60 s), GAP (1e-4),
# SOLVERS (axos highs scip cbc), AXOS_FLAGS (extra run_milp options).
set -u
ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
cd "$ROOT"
DATA="${1:-benchmarks/milp/data/mps}"
OUT="${2:-benchmarks/milp/results/miplib50}"
TL="${TL:-60}"
GAP="${GAP:-1e-4}"
SOLVERS="${SOLVERS:-axos highs scip cbc}"
PY="$ROOT/.venv/Scripts/python.exe"
[ -x "$PY" ] || PY="$ROOT/.venv/bin/python"
RUN="$ROOT/build/run_milp.exe"
[ -x "$RUN" ] || RUN="$ROOT/build/run_milp"
mkdir -p "$OUT"
for s in $SOLVERS; do
    csv="$OUT/$s.csv"
    rm -f "$csv"
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] $s"
    case "$s" in
    axos*)
        # shellcheck disable=SC2086
        "$RUN" "$DATA" --time-limit "$TL" --gap "$GAP" ${AXOS_FLAGS:-} --out "$csv" \
            --sol "$OUT/sol/$s" > /dev/null 2> "$OUT/$s.log"
        ;;
    *)
        "$PY" benchmarks/milp/milpbench.py run --solver "$s" --data "$DATA" --time-limit "$TL" \
            --gap "$GAP" --out "$csv" > "$OUT/$s.log" 2>&1
        ;;
    esac
    echo "[$(date '+%H:%M:%S')] $s: $(awk -F, 'NR>1{print $7}' "$csv" | sort | uniq -c | tr '\n' ' ')"
done
