#!/usr/bin/env bash
# Builds build/run_milp (the AXOS MILP benchmark driver), with GPU support
# unless "cpu" is given.
#
# Windows (Git Bash): MSVC 2019+ (vcvars64.bat) and the pip CUDA wheels in
#   .venv (nvidia-cuda-runtime-cu12, nvidia-cuda-nvcc-cu12 for the crt headers,
#   nvidia-cuda-nvrtc-cu12), as for the QP driver: the kernels are compiled at
#   run time by NVRTC, no CUDA toolkit is needed.
# Linux: g++ (with CUDA_HOME for the GPU build).
# "mpi" / "mpi-cpu" add MPI (-DAXOS_ENABLE_MPI): the MS-MPI SDK on Windows
#   (MSMPI_INC / MSMPI_LIB64, else its default install path), mpicxx
#   (MPICXX) on Linux. Used for the MPI build of the command-line solver.
#
# Usage: bash benchmarks/milp/build.sh [gpu|cpu|mpi|mpi-cpu] [source.cpp] [out-name]
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
mkdir -p "$ROOT/build"
mode="${1:-gpu}"
src="${2:-$ROOT/benchmarks/milp/run_milp.cpp}"
name="${3:-run_milp}"
case "$mode" in
gpu) gpu=1 mpi=0 ;;
cpu) gpu=0 mpi=0 ;;
mpi) gpu=1 mpi=1 ;;
mpi-cpu) gpu=0 mpi=1 ;;
*)
    echo "usage: $0 [gpu|cpu|mpi|mpi-cpu] [source.cpp] [out-name]" >&2
    exit 2
    ;;
esac

case "$(uname -s)" in
MINGW* | MSYS* | CYGWIN*)
    VCVARS="${VCVARS:-C:\\Program Files (x86)\\Microsoft Visual Studio\\2019\\BuildTools\\VC\\Auxiliary\\Build\\vcvars64.bat}"
    NV="$(cygpath -w "$ROOT/.venv/Lib/site-packages/nvidia")"
    SRC="$(cygpath -w "$ROOT/src")"
    OUT="$(cygpath -w "$ROOT/build")"
    FLAGS=""
    LINK=""
    if [ $gpu = 1 ]; then
        FLAGS="/DAXOS_ENABLE_CUDA /I \"$NV\\cuda_runtime\\include\" /I \"$NV\\cuda_nvcc\\include\""
        LINK="/LIBPATH:\"$NV\\cuda_runtime\\lib\\x64\" cuda.lib cudart.lib"
    fi
    if [ $mpi = 1 ]; then
        MPI_INC="${MSMPI_INC:-C:\\Program Files (x86)\\Microsoft SDKs\\MPI\\Include\\}"
        MPI_LIB="${MSMPI_LIB64:-C:\\Program Files (x86)\\Microsoft SDKs\\MPI\\Lib\\x64\\}"
        # no trailing backslash before a closing quote (cl would read \" as a quote)
        FLAGS="$FLAGS /DAXOS_ENABLE_MPI /I \"${MPI_INC%\\}\""
        LINK="$LINK /LIBPATH:\"${MPI_LIB%\\}\" msmpi.lib"
    fi
    mkdir -p "$ROOT/build/obj_$name"
    cat > "$ROOT/build/build_$name.bat" <<EOF
@echo off
call "$VCVARS" >nul
cl /nologo /std:c++17 /O2 /arch:AVX2 /openmp:llvm /EHsc /Zc:__cplusplus /bigobj /W3 ^
   /DNDEBUG /I "$SRC" $FLAGS ^
   "$(cygpath -w "$src")" /Fo"$OUT\\obj_$name\\\\" /Fe"$OUT\\$name.exe" ^
   /link $LINK
EOF
    cmd //c "$(cygpath -w "$ROOT/build/build_$name.bat")"
    if [ $gpu = 1 ]; then
        # run-time DLLs next to the executable
        cp -u "$ROOT/.venv/Lib/site-packages/nvidia/cuda_runtime/bin/"*.dll "$ROOT/build/"
        cp -u "$ROOT/.venv/Lib/site-packages/nvidia/cuda_nvrtc/bin/"*.dll "$ROOT/build/"
    fi
    ;;
*)
    CUDA_HOME="${CUDA_HOME:-/usr/local/cuda}"
    CXX=g++
    FLAGS=()
    LIBS=()
    if [ $mpi = 1 ]; then
        CXX="${MPICXX:-mpicxx}"
        FLAGS+=(-DAXOS_ENABLE_MPI)
    fi
    if [ $gpu = 1 ]; then
        FLAGS+=(-DAXOS_ENABLE_CUDA -I"$CUDA_HOME/include")
        LIBS+=(-L"$CUDA_HOME/lib64" -lcudart -lcuda -ldl)
    fi
    "$CXX" -std=c++17 -O3 -march=native -fopenmp -DNDEBUG -I"$ROOT/src" "${FLAGS[@]}" "$src" \
        "${LIBS[@]}" -o "$ROOT/build/$name"
    ;;
esac
echo "built $ROOT/build/$name"
