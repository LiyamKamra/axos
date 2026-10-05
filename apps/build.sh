#!/usr/bin/env bash
# Builds build/axos, the AXOS command-line solver (LP, QP, MILP), with GPU
# support unless "cpu" is given. Same toolchains as the benchmark drivers:
# MSVC + pip CUDA wheels on Windows (no CUDA toolkit: NVRTC compiles the
# kernels at run time), g++ (+ CUDA_HOME) on Linux.
#
#   bash apps/build.sh [gpu|cpu] [name]     (default: gpu, build/axos)
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
bash "$ROOT/benchmarks/milp/build.sh" "${1:-gpu}" "$ROOT/apps/axos.cpp" "${2:-axos}"
