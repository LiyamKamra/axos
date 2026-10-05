#!/usr/bin/env bash
# Builds build/run_qp (the AXOS QP benchmark driver) with GPU support.
#
# Windows (Git Bash): MSVC 2019+ (vcvars64.bat) and the pip CUDA wheels in
#   .venv (nvidia-cuda-runtime-cu12, nvidia-cuda-nvcc-cu12 for the crt headers,
#   nvidia-cuda-nvrtc-cu12). No CUDA toolkit or nvcc is needed: the kernels
#   are compiled at run time by NVRTC.
# Linux: g++ with CUDA_HOME (or /usr/local/cuda) for the headers, libcudart,
#   libcuda and libnvrtc.
#
# Usage: bash benchmarks/qp/build.sh [cpu]    ("cpu" builds without CUDA)
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
mkdir -p "$ROOT/build"
mode="${1:-gpu}"

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
    cat > "$ROOT/build/build_run_qp.bat" <<EOF
@echo off
call "$VCVARS" >nul
cl /nologo /std:c++17 /O2 /arch:AVX2 /openmp:llvm /EHsc /Zc:__cplusplus /bigobj /W3 ^
   /DNDEBUG /I "$SRC" $CUDA_FLAGS ^
   "$(cygpath -w "$ROOT/benchmarks/qp/run_qp.cpp")" /Fo"$OUT\\\\" /Fe"$OUT\\run_qp.exe" ^
   /link $CUDA_LINK
EOF
    cmd //c "$(cygpath -w "$ROOT/build/build_run_qp.bat")"
    if [ "$mode" != cpu ]; then
        # run-time DLLs next to the executable
        cp "$ROOT/.venv/Lib/site-packages/nvidia/cuda_runtime/bin/"*.dll "$ROOT/build/"
        cp "$ROOT/.venv/Lib/site-packages/nvidia/cuda_nvrtc/bin/"*.dll "$ROOT/build/"
    fi
    ;;
*)
    CUDA_HOME="${CUDA_HOME:-/usr/local/cuda}"
    if [ "$mode" != cpu ]; then
        g++ -std=c++17 -O3 -march=native -fopenmp -DNDEBUG -DAXOS_ENABLE_CUDA \
            -I"$ROOT/src" -I"$CUDA_HOME/include" "$ROOT/benchmarks/qp/run_qp.cpp" \
            -L"$CUDA_HOME/lib64" -lcudart -lcuda -ldl -o "$ROOT/build/run_qp"
    else
        g++ -std=c++17 -O3 -march=native -fopenmp -DNDEBUG -I"$ROOT/src" \
            "$ROOT/benchmarks/qp/run_qp.cpp" -o "$ROOT/build/run_qp"
    fi
    ;;
esac
echo "built $ROOT/build/run_qp"
