#!/usr/bin/env bash
# Real axos sessions (LP, QP, MILP) for the documentation.
#   bash benchmarks/cli_demo.sh > benchmarks/cli_demo.txt
set -u
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"
AX=./build/axos.exe
[ -x "$AX" ] || AX=./build/axos
run() {
    echo "\$ axos $*"
    "$AX" "$@"
    echo "(exit status $?)"
    echo
}
run tests/data/afiro.mps
run benchmarks/qp/data/maros_meszaros_mps/QSHARE1B.mps
run benchmarks/milp/data/mps/exp-1-500-5-5.mps --time-limit 60 --solution exp.sol
