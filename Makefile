ifndef MAKE_VERSION
$(error This Makefile requires GNU Make)
endif

# The dense tensor layer (tensorET, storage policies, GPU memory pool, dense
# math and linear algebra) lives in src/*.h, src/storage, src/tensor and
# src/tensorcuda; its specification is docs/TENSOR_SPEC.md.

CACHE_MK := .make-cache.mk

-include $(CACHE_MK)

# nvcc needs old gcc; let the user override,
# otherwise try to find something nvcc is likely to accept.
$(CACHE_MK):
	@echo "Detecting toolchain..."
	@echo "NVCC_CCBIN := $$(command -v gcc-14 2>/dev/null || \
	                     command -v gcc-13 2>/dev/null || \
	                     command -v gcc 2>/dev/null)" > $@

configure:
	$(MAKE) $(CACHE_MK)

reconfigure:
	rm -f $(CACHE_MK)
	$(MAKE) configure

# ─── Toolchain ────────────────────────────────────────────────────────────────
CXX  ?= g++
NVCC ?= nvcc

MAKEFLAGS += -j12 --output-sync=target

ifeq ($(NVCC_CCBIN),)
$(warning No suitable host compiler found for nvcc. Set NVCC_CCBIN=/path/to/gcc)
endif

# ─── Compiler flags ───────────────────────────────────────────────────────────
CPPFLAGS := -Isrc -I/usr/include/eigen3 -DCUBLAS_WITH -DCUSOLVER_WITH -DCUSPARSE_WITH -DCUDSS_WITH

CFLAGS := -O3 -march=native
CXXFLAGS := -fopenmp -DAXOS_ENABLE_OMP

CUDA_LDLIBS := $(shell pkg-config --libs cudart 2>/dev/null) -lcublas -lcusolver -lcusparse -lcudss

NVCCFLAGS := $(if $(NVCC_CCBIN),-ccbin=$(NVCC_CCBIN),)

build:
	@mkdir -p $@

TENSOR_HDRS := $(wildcard src/*.h src/storage/*.h src/tensor/*.h src/tensorcuda/*.h)
SPARSE_HDRS := $(wildcard src/sparse/*.h) src/shaders/sparse.cu $(TENSOR_HDRS)
SOLVER_HDRS := $(SPARSE_HDRS) $(wildcard src/solver/*.h src/solver/*/*.h)

# ── test_tensor (dense tensor layer, docs/TENSOR_SPEC.md §12) ────────────────
build/test_tensor_cpu: tests/test_tensor.cpp $(TENSOR_HDRS) | build
	$(CXX) $(CPPFLAGS) $(CFLAGS) $(CXXFLAGS) $< -o $@

test_tensor_cpu: build/test_tensor_cpu
	./build/test_tensor_cpu

build/test_tensor.o: tests/test_tensor.cpp $(TENSOR_HDRS) | build
	$(NVCC) $(NVCCFLAGS) $(CPPFLAGS) -DAXOS_ENABLE_CUDA $(subst -march=native,,$(CFLAGS)) \
	        -Xcompiler "$(CXXFLAGS)" -x cu -c $< -o $@

build/test_tensor: build/test_tensor.o | build
	$(CXX) $(CPPFLAGS) $(CFLAGS) $(CXXFLAGS) $(CUDA_LDLIBS) $^ -o $@

test_tensor: build/test_tensor
	./build/test_tensor

# ── benchmark_dense (dense layer vs Eigen, docs/TENSOR_SPEC.md §11.2) ────────
build/bench_dense: benchmarks/dense/bench_dense.cpp $(TENSOR_HDRS) | build
	$(CXX) $(CPPFLAGS) $(CFLAGS) $(CXXFLAGS) $< -o $@

benchmark_dense: build/bench_dense
	./build/bench_dense | tee build/bench_dense_mt.txt
	OMP_NUM_THREADS=1 ./build/bench_dense | tee build/bench_dense_st.txt

# ── test_sparse_cpu (Csr, CPU kernels) ────────────────────────────────────────
build/test_sparse_cpu: tests/test_sparse.cpp tests/test_common.h $(SPARSE_HDRS) | build
	$(CXX) $(CPPFLAGS) $(CFLAGS) $(CXXFLAGS) -Itests $< -o $@

test_sparse_cpu: build/test_sparse_cpu
	./build/test_sparse_cpu

# ── test_sparse (Csr on CPU and CUDA, cuSPARSE/cuDSS) ─────────────────────────
build/test_sparse.o: tests/test_sparse.cpp tests/test_common.h $(SPARSE_HDRS) | build
	$(NVCC) $(NVCCFLAGS) $(CPPFLAGS) -DAXOS_ENABLE_CUDA -Itests $(subst -march=native,,$(CFLAGS)) \
	        -Xcompiler "$(CXXFLAGS)" -x cu -c $< -o $@

build/test_sparse: build/test_sparse.o | build
	$(CXX) $(CPPFLAGS) $(CFLAGS) $(CXXFLAGS) $(CUDA_LDLIBS) $^ -o $@

test_sparse: build/test_sparse
	./build/test_sparse

# ── test_solver (LP solvers: model, MPS, presolve, PDLP, IPM, simplex) ───────
build/test_solver_cpu: tests/test_solver.cpp tests/test_common.h $(SOLVER_HDRS) | build
	$(CXX) $(CPPFLAGS) $(CFLAGS) $(CXXFLAGS) -Itests $< -o $@

test_solver_cpu: build/test_solver_cpu
	./build/test_solver_cpu

build/test_solver.o: tests/test_solver.cpp tests/test_common.h $(SOLVER_HDRS) | build
	$(NVCC) $(NVCCFLAGS) $(CPPFLAGS) -DAXOS_ENABLE_CUDA -Itests $(subst -march=native,,$(CFLAGS)) \
	        -Xcompiler "$(CXXFLAGS)" -x cu -c $< -o $@

build/test_solver: build/test_solver.o | build
	$(CXX) $(CPPFLAGS) $(CFLAGS) $(CXXFLAGS) $(CUDA_LDLIBS) $^ -o $@

test_solver: build/test_solver
	./build/test_solver

# ── benchmark_lp (PDLP / IPM vs HiGHS via SciPy) ─────────────────────────────
build/bench_lp.o: benchmarks/solver/bench_lp.cpp $(SOLVER_HDRS) | build
	$(NVCC) $(NVCCFLAGS) $(CPPFLAGS) -DAXOS_ENABLE_CUDA $(subst -march=native,,$(CFLAGS)) \
	        -Xcompiler "$(CXXFLAGS) -march=native" -x cu -c $< -o $@

build/bench_lp: build/bench_lp.o | build
	$(CXX) $(CPPFLAGS) $(CFLAGS) $(CXXFLAGS) $(CUDA_LDLIBS) $^ -o $@

benchmark_lp: build/bench_lp
	./build/bench_lp | tee build/bench_lp_axos.txt
	~/python_junk/.venv/bin/python benchmarks/solver/bench_highs.py | tee build/bench_lp_highs.txt

# ── run_mps (real LP instances: netlib, Mittelmann) ──────────────────────────
build/run_mps.o: benchmarks/solver/run_mps.cpp $(SOLVER_HDRS) | build
	$(NVCC) $(NVCCFLAGS) $(CPPFLAGS) -DAXOS_ENABLE_CUDA $(subst -march=native,,$(CFLAGS)) \
	        -Xcompiler "$(CXXFLAGS) -march=native" -x cu -c $< -o $@

build/run_mps: build/run_mps.o | build
	$(CXX) $(CPPFLAGS) $(CFLAGS) $(CXXFLAGS) $(CUDA_LDLIBS) $^ -o $@

# ── benchmark_sparse (CPU/CUDA vs Eigen; torch via bench_torch.py) ───────────
build/bench_sparse.o: benchmarks/sparse/bench_sparse.cpp $(SPARSE_HDRS) | build
	$(NVCC) $(NVCCFLAGS) $(CPPFLAGS) -DAXOS_ENABLE_CUDA $(subst -march=native,,$(CFLAGS)) \
	        -Xcompiler "$(CXXFLAGS) -march=native" -x cu -c $< -o $@

build/bench_sparse: build/bench_sparse.o | build
	$(CXX) $(CPPFLAGS) $(CFLAGS) $(CXXFLAGS) $(CUDA_LDLIBS) $^ -o $@

benchmark_sparse: build/bench_sparse
	./build/bench_sparse | tee build/bench_sparse_axos.txt
	~/python_junk/.venv/bin/python benchmarks/sparse/bench_torch.py | tee build/bench_sparse_torch.txt
	~/python_junk/.venv/bin/python benchmarks/sparse/report.py build/bench_sparse_axos.txt build/bench_sparse_torch.txt

# ── QP: HPR-QP / PDHCG (GPU kernels compiled at run time by NVRTC, no nvcc) ──
# run_qp: bash benchmarks/qp/build.sh (also on Windows / Git Bash with MSVC).
# benchmark_qp: Maros-Meszaros, all solvers one at a time (benchmarks/qp/README.md);
# benchmark_qp_large: the generated large problems.
run_qp:
	bash benchmarks/qp/build.sh

benchmark_qp: run_qp
	bash benchmarks/qp/run_all.sh
	TOL=1e-9 SOLVERS=piqp bash benchmarks/qp/run_all.sh benchmarks/qp/data/maros_meszaros \
	    benchmarks/qp/data/mm_axqp benchmarks/qp/results/mm_ref
	python benchmarks/qp/report.py benchmarks/qp/results/mm --ref benchmarks/qp/results/mm_ref/piqp.csv

benchmark_qp_large: run_qp
	python benchmarks/qp/gen_large.py --out benchmarks/qp/data/large_axqp
	TL=120 SOLVERS="axos-hprqp-gpu axos-pdhcg-gpu hprqp-jl pdhcg-jl axos-hprqp-cpu clarabel piqp osqp scs" \
	    bash benchmarks/qp/run_all.sh benchmarks/qp/data/large_axqp benchmarks/qp/data/large_axqp \
	    benchmarks/qp/results/large
	python benchmarks/qp/report.py benchmarks/qp/results/large --time-limit 120

# ── MILP: branch and cut (CPU tree; GPU probing, feasibility jump, batched LP bounds)
# run_milp: bash benchmarks/milp/build.sh [gpu|cpu]; benchmark_milp: the 50 MIPLIB 2017
# instances of benchmarks/milp/data/set50.txt against HiGHS, SCIP and CBC (one at a time).
run_milp:
	bash benchmarks/milp/build.sh

benchmark_milp: run_milp
	SOLVERS="highs scip cbc" bash benchmarks/milp/run_all.sh
	bash benchmarks/milp/run_axos_final.sh
	python benchmarks/milp/report.py benchmarks/milp/results/miplib50

# ── axos: the command-line solver (LP, QP, MILP); apps/axos.py wraps it for Python
axos:
	bash apps/build.sh

test_cli: axos
	python tests/test_cli.py

# ── Eigen reference harnesses for the dense-layer targets (docs/TENSOR_SPEC.md §11.2)
build/ref_%_omp: benchmarks/reference/bench_%.cpp | build
	$(CXX) $(CPPFLAGS) $(CFLAGS) $(CXXFLAGS) $< -o $@

build/ref_%_no_omp: benchmarks/reference/bench_%.cpp | build
	$(CXX) $(CPPFLAGS) $(CFLAGS) $< -o $@

reference_benchmarks: build/ref_eigen_omp build/ref_eigen_no_omp \
                      build/ref_gemv_eigen_omp build/ref_gemv_eigen_no_omp

# ════════════════════════════════════════════════════════════════════════════════
clean:
	rm -rf build/*

.PHONY: configure reconfigure clean test_sparse test_sparse_cpu test_solver test_solver_cpu \
        test_tensor test_tensor_cpu benchmark_lp benchmark_sparse benchmark_dense \
        reference_benchmarks run_qp benchmark_qp benchmark_qp_large run_milp benchmark_milp axos test_cli
