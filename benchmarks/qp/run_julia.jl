# SPDX-License-Identifier: BSD-3-Clause
#
# Runs the reference GPU implementations of the QP methods AXOS implements,
# HPR-QP.jl (Chen et al., github.com/PolyU-IOR/HPR-QP) and PDHCG.jl (Huang et
# al., github.com/INFORMSJoC/2024.0983), on the .axqp exports of run_qp, so
# every solver sees identical data. Prints one CSV line per problem (same
# columns as run_qp / qpbench.py) and writes each solution to
# <sols>/<problem>.sol (int64 n, int64 m, then x[n], y[m] as float64, y in the
# AXOS sign convention) for `qpbench.py score`, which recomputes objective and
# residuals with the common metric.
#
#   julia --project=.tools/HPR-QP benchmarks/qp/run_julia.jl hprqp|pdhcg \
#       <axqp dir> <tol> <time limit> <out.csv> <sols dir> [NAME,NAME...]
#
# Each problem is first solved with a 200-iteration cap (untimed) so that JIT
# compilation never lands in the timed solve; `seconds` is the wall clock of
# the second, full solve (preconditioning, host-to-GPU transfer, spectral
# estimates and iterations, as for AXOS).
using SparseArrays, Printf, CUDA

const ROOT = normpath(joinpath(@__DIR__, "..", ".."))
const SOLVER = ARGS[1]
if SOLVER == "hprqp"
    import HPRQP
elseif SOLVER == "pdhcg"
    include(joinpath(ROOT, ".tools", "PDHCG-jl", "src", "PDHCG.jl"))
else
    error("solver must be hprqp or pdhcg")
end

struct Axqp
    name::String
    n::Int
    m::Int
    offset::Float64
    A::SparseMatrixCSC{Float64,Int}
    Q::SparseMatrixCSC{Float64,Int}
    c::Vector{Float64}
    lb::Vector{Float64}
    ub::Vector{Float64}
    l::Vector{Float64}
    u::Vector{Float64}
end

function read_axqp(path)
    open(path) do io
        String(read(io, 4)) == "AXQP" || error("$path: not an .axqp file")
        read(io, Int32)
        n, m, nnzA, nnzQ = (read(io, Int64) for _ in 1:4)
        offset = read(io, Float64)
        read(io, Int32)
        vec(T, k) = (v = Vector{T}(undef, k); read!(io, v); v)
        function csr(rows, cols, nnz)
            rp = vec(Int32, rows + 1)
            ci = vec(Int32, nnz)
            va = vec(Float64, nnz)
            ri = Vector{Int}(undef, nnz)
            for r in 1:rows, k in rp[r]+1:rp[r+1]
                ri[k] = r
            end
            sparse(ri, Int.(ci) .+ 1, va, rows, cols)
        end
        A = csr(m, n, nnzA)
        Q = csr(n, n, nnzQ)
        c, lb, ub = vec(Float64, n), vec(Float64, n), vec(Float64, n)
        l, u = vec(Float64, m), vec(Float64, m)
        Axqp(splitext(basename(path))[1], n, m, offset, A, Q, c, lb, ub, l, u)
    end
end

# ---- HPR-QP.jl -----------------------------------------------------------------

# qp_formulation drops the empty and the free rows of A (and nothing else);
# the duals come back for the kept rows, in the AXOS convention.
function solve_hprqp(p::Axqp, tol, tl)
    model = HPRQP.build_from_QAbc(p.Q, p.c, p.A, p.l, p.u, p.lb, p.ub, p.offset)
    rowabs = zeros(p.m)
    for (r, v) in zip(rowvals(p.A), nonzeros(p.A))
        rowabs[r] += abs(v)
    end
    kept = [i for i in 1:p.m if rowabs[i] != 0 && !(p.l[i] == -Inf && p.u[i] == Inf)]
    function params(maxit)
        s = HPRQP.HPRQP_parameters()
        s.stoptol = tol
        s.time_limit = tl
        s.max_iter = maxit
        s.warm_up = false
        s.verbose = false
        s.use_gpu = true
        s.device_number = 0
        s
    end
    HPRQP.solve(model, params(200))
    CUDA.synchronize()
    t0 = time()
    r = HPRQP.solve(model, params(typemax(Int32)))
    CUDA.synchronize()
    secs = time() - t0
    y = zeros(p.m)
    length(r.y) == length(kept) || error("unexpected dual length $(length(r.y))")
    y[kept] .= r.y
    status = r.status == "OPTIMAL" ? "optimal" : lowercase(r.status)
    return status, r.x, y, r.iter, secs, r.power_time
end

# ---- PDHCG.jl ------------------------------------------------------------------

# PDHCG's standard form: free rows dropped (by us), two-sided rows become
# equalities with a slack column, <= rows are negated, and equalities are
# moved first. The map below undoes all of that for y.
function solve_pdhcg(p::Axqp, tol, tl)
    kept = [i for i in 1:p.m if !(p.l[i] == -Inf && p.u[i] == Inf)]
    l, u = p.l[kept], p.u[kept]
    two = isfinite.(l) .& isfinite.(u) .& (l .!= u)
    is_eq = (l .== u) .| two
    is_leq = .!is_eq .& isfinite.(u)
    new_to_old = [findall(is_eq); findall(.!is_eq)]
    function problem()
        t = PDHCG.TwoSidedQpProblem(copy(p.lb), copy(p.ub), copy(l), copy(u),
            p.A[kept, :], p.offset, copy(p.c), copy(p.Q))
        PDHCG.transform_to_standard_form(t)
    end
    run(maxit) = PDHCG.pdhcgSolve(problem(); gpu_flag = true, warm_up_flag = false,
        verbose_level = 0, time_limit = Float64(tl), relat_error_tolerance = tol,
        iteration_limit = maxit)
    run(200)
    CUDA.synchronize()
    t0 = time()
    log = run(Int64(typemax(Int32)))
    CUDA.synchronize()
    secs = time() - t0
    yk = zeros(length(kept))
    for (k, old) in enumerate(new_to_old)
        yk[old] = is_leq[old] ? -log.dual_solution[k] : log.dual_solution[k]
    end
    y = zeros(p.m)
    y[kept] .= yk
    x = log.primal_solution[1:p.n]
    reason = string(log.termination_reason)
    status = reason == "TERMINATION_REASON_OPTIMAL" ? "optimal" :
             occursin("TIME", reason) ? "time_limit" :
             occursin("ITERATION", reason) ? "iteration_limit" : lowercase(reason)
    return status, x, y, log.iteration_count, secs, 0.0
end

# ---- driver --------------------------------------------------------------------

function main()
    dir, tol, tl, out_path, sols = ARGS[2], parse(Float64, ARGS[3]), parse(Float64, ARGS[4]),
        ARGS[5], ARGS[6]
    only = length(ARGS) >= 7 ? Set(split(ARGS[7], ",")) : nothing
    mkpath(sols)
    files = sort([f for f in readdir(dir) if endswith(f, ".axqp")])
    only === nothing || filter!(f -> splitext(f)[1] in only, files)
    header = "problem,n,m,nnzA,nnzQ,solver,device,tol,status,iterations,seconds," *
             "setup_seconds,objective,rel_primal,rel_dual,rel_gap,read_seconds"
    fresh = !isfile(out_path)
    out = open(out_path, "a")
    fresh && println(out, header)
    println(header)
    name = SOLVER == "hprqp" ? "hprqp-jl" : "pdhcg-jl"
    solve = SOLVER == "hprqp" ? solve_hprqp : solve_pdhcg
    for f in files
        t0 = time()
        p = read_axqp(joinpath(dir, f))
        read_s = time() - t0
        head = @sprintf("%s,%d,%d,%d,%d,%s,gpu,%.0e", p.name, p.n, p.m, nnz(p.A), nnz(p.Q),
            name, tol)
        line = try
            st, x, y, it, secs, setup = solve(p, tol, tl)
            open(joinpath(sols, p.name * ".sol"), "w") do io
                write(io, Int64(p.n), Int64(p.m), Float64.(x), Float64.(y))
            end
            obj = 0.5 * dot(x, p.Q * x) + dot(p.c, x) + p.offset
            # residual columns are filled in by `qpbench.py score`
            @sprintf("%s,%s,%d,%.6f,%.6f,%.12e,,,,%.3f", head, st, it, secs, setup, obj, read_s)
        catch e
            msg = replace(first(sprint(showerror, e), 300), "," => ";", "\n" => " ")
            "$head,error,,,,,,,,$msg"
        end
        println(line)
        println(out, line)
        flush(out)
        flush(stdout)
    end
    close(out)
end

using LinearAlgebra: dot
main()
