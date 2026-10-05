#!/usr/bin/env bash
# Fetches the QP benchmark data and the GPU comparison solvers, all into
# git-ignored, project-local folders (nothing is installed system-wide):
#
#   benchmarks/qp/data/maros_meszaros/   the 138 Maros-Meszaros QPS files
#   .tools/julia-1.10.10/                 portable Julia (HPR-QP recommends 1.10)
#   .tools/julia-depot/                   Julia packages + CUDA.jl artifacts
#   .tools/HPR-QP/                        github.com/PolyU-IOR/HPR-QP
#   .tools/PDHCG-jl/                      github.com/INFORMSJoC/2024.0983 (PDHCG, Julia)
#
# The CPU comparison solvers (OSQP, Clarabel, HiGHS, PIQP, SCS) come from pip:
#   python -m venv .venv && .venv/Scripts/python -m pip install osqp clarabel highspy piqp scs numpy scipy
#
# Usage: bash benchmarks/qp/setup_comparators.sh [data|julia|all]
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
what="${1:-all}"

fetch_data() {
    local dst="$ROOT/benchmarks/qp/data/maros_meszaros"
    if [ ! -f "$dst/AUG2D.SIF" ]; then
        rm -rf "$dst"
        git clone --depth 1 https://github.com/optimizers/maros-meszaros-mirror "$dst"
    fi
    echo "Maros-Meszaros: $(ls "$dst"/*.SIF | wc -l) files in $dst"
    # the same data in the binary .axqp format, read by qpbench.py and run_julia.jl
    local run_qp="$ROOT/build/run_qp.exe"
    [ -x "$run_qp" ] || run_qp="$ROOT/build/run_qp"
    if [ -x "$run_qp" ]; then
        "$run_qp" "$dst" --export "$ROOT/benchmarks/qp/data/mm_axqp" > /dev/null
    else
        echo "build run_qp (bash benchmarks/qp/build.sh), then rerun: $0 data"
    fi
}

fetch_julia() {
    local tools="$ROOT/.tools" ver=1.10.10
    mkdir -p "$tools"
    case "$(uname -s)" in
    MINGW* | MSYS* | CYGWIN*)
        local jl="$tools/julia-$ver/bin/julia.exe"
        if [ ! -x "$jl" ]; then
            curl -L --fail -o "$tools/julia.zip" \
                "https://julialang-s3.julialang.org/bin/winnt/x64/1.10/julia-$ver-win64.zip"
            (cd "$tools" && unzip -q julia.zip && rm julia.zip)
        fi ;;
    *)
        local jl="$tools/julia-$ver/bin/julia"
        if [ ! -x "$jl" ]; then
            curl -L --fail -o "$tools/julia.tar.gz" \
                "https://julialang-s3.julialang.org/bin/linux/x64/1.10/julia-$ver-linux-x86_64.tar.gz"
            (cd "$tools" && tar xzf julia.tar.gz && rm julia.tar.gz)
        fi ;;
    esac
    export JULIA_DEPOT_PATH="$tools/julia-depot"
    [ -d "$tools/HPR-QP" ] || git clone --depth 1 https://github.com/PolyU-IOR/HPR-QP "$tools/HPR-QP"
    [ -d "$tools/PDHCG-jl" ] || git clone --depth 1 https://github.com/INFORMSJoC/2024.0983 "$tools/PDHCG-jl"
    # HPR-QP.jl needs CUDA.jl 5 (6.x changed the API it uses); PDHCG.jl is run
    # from its sources in the same environment and needs a few more packages.
    # CUDA.jl's runtime is pinned to one the installed driver supports
    # (CUDA_RUNTIME, default 12.9: a 13.1 driver rejects the 13.2 default).
    "$jl" --project="$tools/HPR-QP" -e 'using Pkg; Pkg.instantiate();
        Pkg.add(Pkg.PackageSpec(name="CUDA", version="5")); Pkg.pin("CUDA");
        Pkg.add(["ArgParse", "GZip", "StatsBase", "StructTypes", "Statistics"]; preserve=Pkg.PRESERVE_ALL)'
    "$jl" --project="$tools/HPR-QP" -e "using CUDA; CUDA.set_runtime_version!(v\"${CUDA_RUNTIME:-12.9}\")"
    "$jl" --project="$tools/HPR-QP" -e 'using Pkg, Dates; Pkg.precompile(); using CUDA; CUDA.versioninfo();
        Pkg.gc(collect_delay=Dates.Hour(0))'
    echo "Julia: $("$jl" --version)"
}

case "$what" in
data) fetch_data ;;
julia) fetch_julia ;;
all) fetch_data; fetch_julia ;;
*) echo "usage: $0 [data|julia|all]"; exit 2 ;;
esac
