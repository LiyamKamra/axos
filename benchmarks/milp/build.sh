#!/usr/bin/env bash
# Builds build/run_milp (the AXOS MILP benchmark driver), with GPU support
# unless "cpu" is given.
#
# Windows (Git Bash): MSVC 2019+ (vcvars64.bat) and the pip CUDA wheels in
#   .venv (nvidia-cuda-runtime-cu12, nvidia-cuda-nvcc-cu12 for the crt headers,
#   nvidia-cuda-nvrtc-cu12), as for the QP driver: the kernels are compiled at
#   run time by NVRTC, no CUDA toolkit is needed.
# Linux: g++ (with CUDA_HOME for the GPU build).
#
# Usage: bash benchmarks/milp/build.sh [gpu|cpu] [source.cpp] [out-name]
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
mkdir -p "$ROOT/build"
mode="${1:-gpu}"
src="${2:-$ROOT/benchmarks/milp/run_milp.cpp}"
name="${3:-run_milp}"

case "$(uname -s)" in
MINGW* | MSYS* | CYGWIN*)
    VCVARS="${VCVARS:-C:\\Program Files (x86)\\Microsoft Visual Studio\\2019\\BuildTools\\VC\\Auxiliary\\Build\\vcvars64.bat}"
    NV="$(cygpath -w "$ROOT/.venv/Lib/site-packages/nvidia")"
    SRC="$(cygpath -w "$ROOT/src")"
    OUT="$(cygpath -w "$ROOT/build")"
    CUDA_FLAGS=""
    CUDA_LINK=""
    if [ "$mode" != cpu ]; then
        CUDA_FLAGS="/DAXOS_ENABLE_CUDA /I \"$NV\\cuda_runtime\\include\" /I \"$NV\\cuda_nvcc\\include\""
        CUDA_LINK="/LIBPATH:\"$NV\\cuda_runtime\\lib\\x64\" cuda.lib cudart.lib"
    fi
    mkdir -p "$ROOT/build/obj_$name"
    cat > "$ROOT/build/build_$name.bat" <<EOF
@echo off
call "$VCVARS" >nul
cl /nologo /std:c++17 /O2 /arch:AVX2 /openmp:llvm /EHsc /Zc:__cplusplus /bigobj /W3 ^
   /DNDEBUG /I "$SRC" $CUDA_FLAGS ^
   "$(cygpath -w "$src")" /Fo"$OUT\\obj_$name\\\\" /Fe"$OUT\\$name.exe" ^
   /link $CUDA_LINK
EOF
    cmd //c "$(cygpath -w "$ROOT/build/build_$name.bat")"
    if [ "$mode" != cpu ]; then
        # run-time DLLs next to the executable
        cp -u "$ROOT/.venv/Lib/site-packages/nvidia/cuda_runtime/bin/"*.dll "$ROOT/build/"
        cp -u "$ROOT/.venv/Lib/site-packages/nvidia/cuda_nvrtc/bin/"*.dll "$ROOT/build/"
    fi
    ;;
*)
    CUDA_HOME="${CUDA_HOME:-/usr/local/cuda}"
    if [ "$mode" != cpu ]; then
        g++ -std=c++17 -O3 -march=native -fopenmp -DNDEBUG -DAXOS_ENABLE_CUDA \
            -I"$ROOT/src" -I"$CUDA_HOME/include" "$src" \
            -L"$CUDA_HOME/lib64" -lcudart -lcuda -ldl -o "$ROOT/build/$name"
    else
        g++ -std=c++17 -O3 -march=native -fopenmp -DNDEBUG -I"$ROOT/src" "$src" -o "$ROOT/build/$name"
    fi
    ;;
esac
echo "built $ROOT/build/$name"
