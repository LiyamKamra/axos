#!/usr/bin/env bash
# Same-trajectory check of HPR-QP over MPI for problems too slow to finish on
# K ranks sharing one GPU: runs each listed problem for a fixed number of
# iterations on one GPU (build/run_qp) and on K ranks (build/run_qp_mpi),
# and compares the iterates at the end (objective and residuals, compare_mpi.py
# columns), which agree only if the two runs took the same path.
#
#   bash benchmarks/qp/run_mpi_fixed.sh K ITERS "NAME NAME ..." [problem dir] [name]
#
# Output: benchmarks/qp/results/mpi/<name>_1gpu.csv and <name>_mpiK.csv.
set -u
ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
cd "$ROOT"
K="${1:-2}"
ITERS="${2:-20000}"
LIST="${3:?problem names}"
DATA="${4:-benchmarks/qp/data/maros_meszaros}"
NAME="${5:-fixed$ITERS}"
OUT=benchmarks/qp/results/mpi
EXE1="$ROOT/build/run_qp.exe"
[ -x "$EXE1" ] || EXE1="$ROOT/build/run_qp"
EXE="$ROOT/build/run_qp_mpi.exe"
[ -x "$EXE" ] || EXE="$ROOT/build/run_qp_mpi"
if [ -z "${MPIEXEC:-}" ]; then
    MPIEXEC=mpiexec
    [ -x "/c/Program Files/Microsoft MPI/Bin/mpiexec.exe" ] &&
        MPIEXEC="/c/Program Files/Microsoft MPI/Bin/mpiexec.exe"
fi
mkdir -p "$OUT"
rm -f "$OUT/${NAME}_1gpu.csv" "$OUT/${NAME}_mpi$K.csv"
for p in $LIST; do
    f=$(ls "$DATA/$p".* 2>/dev/null | head -1)
    [ -n "$f" ] || { echo "$p: not found"; continue; }
    "$EXE1" "$f" --method hprqp --device gpu --max-iter "$ITERS" --time-limit 3600 \
        --out "$OUT/${NAME}_1gpu.csv" > /dev/null 2>&1
    "$MPIEXEC" -n "$K" "$EXE" "$f" --method hprqp --device gpu --max-iter "$ITERS" --time-limit 3600 \
        --out "$OUT/${NAME}_mpi$K.csv" > /dev/null 2>&1
done
python - "$OUT/${NAME}_1gpu.csv" "$OUT/${NAME}_mpi$K.csv" <<'EOF'
import csv, sys
a = {r["problem"]: r for r in csv.DictReader(open(sys.argv[1]))}
b = {r["problem"]: r for r in csv.DictReader(open(sys.argv[2]))}
worst = 0.0
print(f"{'problem':12s} {'status':>16s} {'iters':>8s}  rel. differences: objective, eta_p, eta_d, gap")
for p in sorted(b):
    r, m = a[p], b[p]
    d = []
    for k in ("objective", "rel_primal", "rel_dual", "rel_gap"):
        x, y = float(r[k]), float(m[k])
        d.append(abs(x - y) / max(abs(x), 1e-300) if k != "objective" else abs(x - y) / max(1.0, abs(x)))
    worst = max(worst, d[0])
    same = r["status"] == m["status"] and r["iterations"] == m["iterations"]
    print(f"{p:12s} {m['status']:>16s} {m['iterations']:>8s}  " + " ".join(f"{v:.1e}" for v in d)
          + ("" if same else f"   (1 GPU: {r['status']} {r['iterations']})"))
print(f"largest relative objective difference {worst:.1e}")
EOF
