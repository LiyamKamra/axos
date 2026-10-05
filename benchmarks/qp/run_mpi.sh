#!/usr/bin/env bash
# Checks HPR-QP distributed over MPI ranks (src/solver/qp/qp_dist.h) against
# the single-GPU runs of the benchmark: solves every problem of a set with
# build/run_qp_mpi on K ranks (one mpiexec per problem), re-scores the
# solutions with the benchmark's common metric (qpbench.py score), and
# compares status, iterations and objective with a reference CSV
# (compare_mpi.py).
#
#   bash benchmarks/qp/run_mpi.sh K [problem dir] [axqp dir] [reference csv] [name]
#
# Defaults: the 14 large problems against results/large/axos-hprqp-gpu.csv.
# Output: benchmarks/qp/results/mpi/<name>.csv, .log, sol/<name>/.
# Environment: TOL (1e-6), TL (seconds per problem, 300), MPIEXEC.
# On one GPU all K ranks share it: this checks the distributed algorithm,
# it does not measure multi-GPU speed.
set -u
ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
cd "$ROOT"
K="${1:-2}"
DATA="${2:-benchmarks/qp/data/large_axqp}"
AXQP="${3:-benchmarks/qp/data/large_axqp}"
REF="${4:-benchmarks/qp/results/large/axos-hprqp-gpu.csv}"
NAME="${5:-$(basename "$DATA")_n$K}"
TOL="${TOL:-1e-6}"
TL="${TL:-300}"
OUT=benchmarks/qp/results/mpi
PY="$ROOT/.venv/Scripts/python.exe"
[ -x "$PY" ] || PY="$ROOT/.venv/bin/python"
EXE="$ROOT/build/run_qp_mpi.exe"
[ -x "$EXE" ] || EXE="$ROOT/build/run_qp_mpi"
if [ -z "${MPIEXEC:-}" ]; then
    MPIEXEC=mpiexec
    [ -x "/c/Program Files/Microsoft MPI/Bin/mpiexec.exe" ] &&
        MPIEXEC="/c/Program Files/Microsoft MPI/Bin/mpiexec.exe"
fi
mkdir -p "$OUT/sol/$NAME"
rm -f "$OUT/$NAME.csv" "$OUT/$NAME.log"
rm -f "$OUT/sol/$NAME/"*.sol

echo "[$(date +%H:%M:%S)] $NAME: $K ranks, $DATA, tol $TOL, $TL s"
for f in "$DATA"/*; do
    case "$f" in
    *.SIF | *.sif | *.QPS | *.qps | *.mps | *.MPS | *.axqp) ;;
    *) continue ;;
    esac
    "$MPIEXEC" -n "$K" "$EXE" "$f" --method hprqp --device gpu --tol "$TOL" --time-limit "$TL" \
        --verbose 1 --out "$OUT/$NAME.csv" --save-sol "$OUT/sol/$NAME" >> "$OUT/$NAME.log" 2>&1 ||
        echo "$(basename "$f"): mpiexec exit $?" >> "$OUT/$NAME.log"
done
"$PY" benchmarks/qp/qpbench.py score --data "$AXQP" --csv "$OUT/$NAME.csv" --sols "$OUT/sol/$NAME"
"$PY" benchmarks/qp/compare_mpi.py "$REF" "$OUT/$NAME.csv" --tol "$TOL" | tee "$OUT/$NAME.cmp.txt"
echo "[$(date +%H:%M:%S)] $NAME done"
