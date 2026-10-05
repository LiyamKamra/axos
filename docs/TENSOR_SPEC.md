# AXOS dense tensor layer: behavioural specification

Status: **implemented** (Tier 1 and Tier 2); see §16 for where the code
lives, the decisions taken on §14, deviations, and what has been verified.
This document specifies what the layer must do so that the code that remains
(sparse matrices, LP solvers, their tests and benchmarks) builds and performs
as before. It describes interfaces, observable behaviour and performance
targets.

Namespace: everything below lives in `namespace AXOS` (sub-namespaces `AXOS::Cpu`,
`AXOS::Cuda`). Macros use the `AXOS_` prefix.

Contents

1. Scope and priorities
2. Headers and build flags
3. Storage policies (`Cpu::HostStorage`, `Cuda::CudaStorage`)
4. Backend tags and transfer functions
5. The tensor class `tensorET<DIM, T, Storage>`
6. Expressions and element-wise math (Tier 2)
7. Dense linear algebra (Tier 2)
8. Stream output (Tier 2)
9. GPU memory pool and library handles
10. Element types
11. Performance targets
12. Acceptance tests
13. Usage inventory of the retained code
14. Open decisions for the implementers
15. Design notes: how the performance was obtained
16. Implementation status

---

## 1. Scope and priorities

**Tier 1: required to build and run the retained code.** Storage policies,
backend tags and transfer functions, the tensor class for rank 1 and 2 with the
constructors, accessors and conversions listed in §5, and the GPU memory pool
with cuSPARSE and cuDSS handles (§9). Tier 1 is small (see the inventory in §13)
and must be finished first. Until it exists, `make test_sparse` and
`make test_solver` do not compile.

**Tier 2: the general dense library.** Lazy element-wise expressions, math
functions, matrix products, dense factorizations and eigen-solvers, and printing.
The retained code does not use these. They are specified so the library stays a
general tensor library, and they have performance targets against Eigen (§11).

**Out of scope.** The ML stack (autograd, layers, optimizers, ONNX, data loading,
MPI) and the Vulkan backend were removed and are not specified here.

## 2. Headers and build flags

| Header (suggested)          | Provides                                            | Tier |
|-----------------------------|-----------------------------------------------------|------|
| `tensorET.h`                | the tensor class, `Cpu::HostStorage`, `Cpu::Backend`, `backend_of_t` | 1 |
| `tensorCuda.h`              | `Cuda::CudaStorage`, `Cuda::Backend`, `tensorETCuda` alias; includes the pool | 1 |
| `storage/host_storage.h`, `storage/cuda_storage.h` | the storage policies on their own (the sparse code includes these directly) | 1 |
| `tensorcuda/gpu_pool.h`     | `GPUMemoryPool` (§9)                                | 1 |
| `tensorMath.h`, `tensorLinearAlgebra.h`, `tensorIO.h` | Tier 2 functions   | 2 |
| `axos.h`                    | umbrella header: all of the above that apply to the build | 2 |

The retained code includes exactly these names: `tensorET.h`, `tensorCuda.h`,
`storage/host_storage.h`, `storage/cuda_storage.h`, `tensorcuda/gpu_pool.h`.
`tests/test_common.h` includes the umbrella header (it was `tensorPanini.h`; it
must be renamed there).

Build flags (defined by the Makefile, consumed by the headers):

- `AXOS_ENABLE_CUDA`: CUDA storage and backend are available. Without it,
  `tensorCuda.h` need not compile, but `storage/cuda_storage.h` must still
  compile, as a type with the same interface that can never be allocated.
  Host-only builds instantiate templates that mention `Cuda::CudaStorage`
  without using it.
- `CUSPARSE_WITH`, `CUDSS_WITH`, `CUBLAS_WITH`, `CUSOLVER_WITH`: enable the
  corresponding handle getters on the pool (§9).
- `AXOS_ENABLE_OMP`: OpenMP is available (Tier 2 parallel kernels).
- The code is compiled with `-std=c++17` or later, both by g++ and by nvcc
  (host compiler g++-14). Everything in Tier 1 must compile under nvcc.

## 3. Storage policies

A storage policy owns (or borrows) a flat, contiguous array of `n` elements in
one memory space. `Csr<T, Idx, Store>` in `src/sparse/csr.h` uses the policies
directly as `template <typename> class Store` parameters, independently of the
tensor class. So the policies are a public interface in their own right.

### 3.1 Required interface (both policies)

For `template <typename T> struct Policy`:

| Member                             | Behaviour |
|------------------------------------|-----------|
| `using backend_type = <tag>;`      | `Cpu::Backend` for host, `Cuda::Backend` for CUDA (§4). |
| `Policy()`                         | Empty: no memory, `data() == nullptr`, size 0. |
| `explicit Policy(size_t n)`        | Owns `n` elements, **contents unspecified** (not initialized, not touched on the host). `n == 0` gives the empty state. |
| `Policy(size_t n, T x)`            | Owns `n` elements, all equal to `x`. |
| `Policy(T *raw, size_t n, bool own = false)` | Wraps an existing buffer in the policy's memory space. With `own == false` the buffer is never freed by the policy. With `own == true` the policy frees it as if it had allocated it (so it must have come from the same allocator). |
| copy constructor / copy assignment | If the source owns its memory: **deep copy** into a new allocation in the same memory space. If the source is a borrowed wrapper: the copy borrows the same pointer (shallow). Self-assignment is a no-op. |
| move constructor / move assignment | Transfers the pointer and ownership; the source becomes empty and non-owning. `noexcept`. |
| `~Policy()`                        | Frees owned memory; never frees borrowed memory. |
| `void allocate(size_t n)`          | Releases current contents (per ownership), then owns `n` unspecified elements. |
| `T *data()`, `const T *data() const` | Pointer to element 0 in the policy's memory space (a device pointer for CUDA), `nullptr` when empty. |
| public `size_t numElem`, `bool ownmem`, `T *ptr` | Present in the removed version; the retained code does **not** use them, so they are optional. |

Element order is the order of the flat array; there is no padding between
elements. `data()` must be aligned to at least `alignof(T)`. For the CUDA
policy, pooled allocations are 256-byte aligned (§9).

### 3.2 Host policy specifics

- Allocation must not touch memory in `Policy(n)`. Large uninitialized vectors
  are used as scratch in the solvers, and first-touch cost belongs to the caller.
- Allocation failure throws `std::bad_alloc`. The IPM relies on catching it
  for over-large factorizations.

### 3.3 CUDA policy specifics

- Memory comes from `GPUMemoryPool::get()` (§9) and is returned to it on
  destruction. There is no `cudaMalloc`/`cudaFree` per object in steady state.
- `Policy(n, x)` fills on the device. Any method is acceptable, but for `x`
  equal to zero it must cost no more than one `cudaMemset`-equivalent pass.
- Deep copies are device-to-device.
- Constructors and assignments are synchronous with respect to the default stream:
  when they return, the data is in place.
- Must compile in translation units that are not compiled by nvcc (plain g++
  with `AXOS_ENABLE_CUDA`), because host code declares CUDA vectors.

## 4. Backend tags and transfer functions

`Cpu::Backend` and `Cuda::Backend` are structs used as tags (template
arguments of `Parallel<Backend>` and `Sparse::Kernels<Backend>` in the retained
code) and as holders of static transfer functions:

```
template <class Tensor>          static T    read_element(const Tensor &t, size_t i);
template <class Tensor>          static void copy_data(Tensor &dst, const Tensor &src);   // same size, same space
template <class Tensor, class U> static void copy_to_host(const Tensor &t, U *host_dst); // t.size() elements
template <class Tensor, class U> static void copy_from_host(Tensor &t, const U *host_src);
```

For the host tag these are plain copies. For CUDA they are blocking copies of
exactly `t.size()` elements. `read_element` returns one element by value, so on
CUDA it is a one-element device-to-host transfer.

`backend_of_t<X>` is `typename X::backend_type` when `X` has one, otherwise
`Cpu::Backend`.

`is_host_storage<S>` is a trait that is true exactly when
`S::backend_type` is `Cpu::Backend`.

## 5. The tensor class

```
template <int DIM, typename T, typename Storage = Cpu::HostStorage<T>>
class tensorET;

template <int DIM, typename T>
using tensorETCuda = tensorET<DIM, T, Cuda::CudaStorage<T>>;   // tensorCuda.h
```

A `tensorET` is a dense, **row-major**, rank-`DIM` array whose elements live in
the memory of `Storage`. `DIM >= 1`. Ranks 1 and 2 are Tier 1; higher ranks
are Tier 2.

### 5.1 Member types and constants

`value_type` and `data_type` (= `T`), `storage_type` (= `Storage`),
`backend_type` (= `Storage::backend_type`), `static constexpr int rank = DIM`.

### 5.2 Required public data

- `T *data`: a **public data member** (not a function) pointing at element 0,
  in the storage's memory space. The retained code reads and writes through
  `x.data` in host loops, passes it to CUDA kernels, and passes it to cuSPARSE and
  cuDSS. It must equal `storage_.data()` for an owning tensor, and it may be
  offset into another tensor's buffer for views.
- `Storage storage_`: public, the owning/borrowing policy object.

Other bookkeeping (sizes, strides, ownership flags) may be public or private at
the implementers' choice. The retained code accesses only `data` and
`storage_`, and the latter only in documentation.

### 5.3 Constructors

| Signature | Behaviour | Tier |
|-----------|-----------|------|
| `tensorET()` | Empty: `size() == 0`, `data == nullptr`, every `size(d) == 0`. | 1 |
| `tensorET(std::initializer_list<size_t> dims)` | Shape `dims`, contents unspecified. Throws `std::runtime_error` if `dims.size() != DIM`. | 1 |
| `tensorET(std::initializer_list<size_t> dims, T x)` | Same, every element equal to `x`. | 1 |
| `tensorET(size_t n)` | `n` elements, contents unspecified. For `DIM == 1` shape `{n}`; for `DIM > 1` shape `{1, ..., 1, n}` (the size goes to the **last** dimension). | 1 |
| `tensorET(size_t n, T x)` | Same, filled with `x`. | 1 |
| `tensorET(size_t dims[DIM])`, `tensorET(size_t dims[DIM], T x)` | Array forms of the initializer-list constructors. | 1 |
| `tensorET(T *raw, size_t dims[DIM])` | Non-owning view of an existing buffer in `Storage`'s memory space; never frees it. | 1 |
| `tensorET(T *raw, size_t dims[DIM], bool own)` | As above; with `own == true` the tensor frees the buffer through `Storage`. | 2 |
| `tensorET(Storage &&s, size_t dims[DIM])` | Adopts an existing storage object; `s.size` must be the product of `dims` (undefined otherwise). | 2 |
| `tensorET(size_t dims[DIM], size_t start[DIM], size_t end[DIM])` | Shape `dims`, and additionally records a sub-domain `[start, end]` (inclusive) per dimension that can be read back (§5.7). Contents unspecified. | 2 |
| copy constructor / assignment | Same semantics as the storage policy: owning tensors deep-copy; **views stay views** (the copy aliases the same memory). | 1 |
| move constructor / assignment | Steals the buffer; the source becomes empty (`size() == 0`, `data == nullptr`). `noexcept`. | 1 |
| `template <class S2> tensorET(const tensorET<DIM, T, S2> &o)` and `operator=(const tensorET<DIM, T, S2> &)` | **Cross-storage conversion.** Allocates a new owning tensor of `o`'s shape in *this* storage and copies all elements across memory spaces with the backend transfer functions (§4). Host↔host is a straight copy; host↔device is one transfer; device↔device through host staging is acceptable. After assignment the target's previous contents are released. | 1 |
| `template <class E> tensorET(const Expr<T, E> &e)` | Host only: allocates the expression's shape and evaluates it element by element (§6). | 2 |

Host-to-device and device-to-host conversions are how the solvers upload and download
vectors (`Vec d(h)` / `tensorET<1,double> h(d)`). They must be one blocking copy
and nothing more (see the §11 targets).

### 5.4 Size queries

- `size_t size() const`: total element count (product of the shape).
- `size_t size(int d) const`: extent of dimension `d`, for `0 <= d < DIM`.
  Throws `std::runtime_error` otherwise.
- `size_t stride(int d) const`: element stride of dimension `d` (row-major:
  the last dimension has stride 1). Throws `std::runtime_error` for out-of-range `d`.
  For views created by `slice`, the strides are the **parent's** strides.

### 5.5 Element access

Host storage:

- `T &operator[](size_t i)` and `const T` overload: flat element `i` of the
  underlying buffer (no bounds check). For contiguous tensors this is the
  row-major linear index.
- `T &value(size_t i)` and `T value(size_t i) const`: same as `operator[]`.
- `T &operator()(i0, ..., i_{DIM-1})` and const overload: element at the
  coordinates, computed from the strides. A call with the wrong number of indices
  must fail to compile.

Non-host storage (CUDA):

- Only the **const, by-value** forms exist: `operator[](i)`, `value(i)` and
  `operator()(...)` return a copy of the element via `read_element`. Writable
  element access must fail to compile. This is intentional: silent per-element
  device round trips in loops are a performance bug.

### 5.6 Views

Views never own memory and are valid only while the tensor they came from
lives. They keep the source's `Storage` type.

- `slice(std::array<size_t,DIM> start, std::array<size_t,DIM> length)`: view
  of the box `[start, start+length)`. Shape is `length`, strides are the parent's,
  and `data` points at the first element of the box. No bounds checking.
- `template <int N> reshape(std::array<size_t,N> dims)`: rank-`N` view of the
  same contiguous buffer with fresh row-major strides. Throws
  `std::runtime_error` if the product of `dims` differs from `size()`. It is
  only meaningful on contiguous tensors (not on slices).
- `flatten()`: `reshape<1>({size()})`.

(Tier 2. Slice and reshape work on CUDA tensors as well, since they only move the
pointer.)

### 5.7 Domain

`getIndex()` returns an object from which, for each dimension, the extent, a
start and an end index can be read. By default start is 0 and end is extent-1.
With the domain constructor they are the given values. Tier 2; the retained code
does not use it.

### 5.8 2-D tiling helpers (Tier 2, host only)

For `DIM == 2`, with square tiles of size `m`, where tile `(i, j)` covers rows
`[i*m, i*m+m)` and columns `[j*m, j*m+m)` clipped to the matrix:

- `blockView(i, j, m)`: `slice({i*m, j*m}, {m, m})` (unclipped; caller
  ensures the tile is interior).
- `blockCopy(i, j, m, T *buf, bool transpose = false)`: writes the tile into the
  caller's `m*m` row-major buffer. Parts outside the matrix are **zero**. With
  `transpose`, it writes the transpose of tile `(j, i)`. Returns a non-owning
  `m x m` view of `buf`.
- `blockCopy(i, j, m)`: returns a new owning `m x m` tile, zero-padded.
- `blockWriteBack(i, j, m, tile)`: copies the in-range part of `tile` into the
  matrix (the padding is ignored).
- `blockAddWriteBack(i, j, m, tile)`: same with `+=`.

CUDA equivalents (free functions in `tensorCuda.h`, all on a given
`cudaStream_t`, asynchronous with respect to the host) are
`blockCopyCuda(t, i, j, m, buf, stream)` (non-owning view of `buf`, zero-padded),
`blockCopyCuda(t, i, j, m, stream)` (owning tile), and
`blockWriteBackCuda(t, i, j, m, tile, stream)`.

### 5.9 Random fill (Tier 2, host)

`static tensorET uniform(dims, lo = 0, hi = 1)` and
`static tensorET gaussian(dims, mean = 0, stddev = 1)` return owning host tensors
of i.i.d. samples. Seeding is non-deterministic. Adding an optional seed
parameter is recommended; tests must not depend on the values.

### 5.10 Assignment from a scalar sequence (Tier 2)

The removed version allowed assigning a "place holder" object that fills the
first `N` elements with a constant. It is rarely used. Treat it as optional.

## 6. Expressions and element-wise math (Tier 2)

All element-wise operations are **lazy**. An expression over tensors of equal
shape is an object that can report `size(d)` and compute `value(i)` for any flat
index. It is only evaluated when assigned to a tensor. A chain such as
`C = exp(sin(A) + cos(A)) * 2.0 + B` must run as **one pass** over the data
with **no temporary tensors**. The fusion is required for the targets in §11.

Operations:

- binary, tensor/expression ⊕ tensor/expression, element-wise: `+ - * /`
  (`*` is the element-wise product, not a matrix product);
- scalar, both sides: `x * s`, `s * x`, `x + s`, `s + x`, `x - s`, `s - x`,
  `x / s`, `s / x`;
- unary functions: `sqrt, exp, Log (natural log), sin, cos, tan, sinh, cosh,
  tanh, fabs`;
- `tensorET = expression` (host) evaluates into the **existing** buffer. Sizes
  must match (undefined otherwise; a debug-mode check is recommended). A
  tensor constructed from an expression allocates first.

Users can write their own expression types. An expression type must provide
`value(size_t i)` and `size(int d)` and derive from the library's expression
base so that the operators above accept it. Document the base class's name and
requirements in the new headers.

Matrix products:

- `matMul(A, B)`: lazy product `A (m x k) * B (k x n)`; `value(i)` computes
  entry `(i / n, i % n)`.
- `matMulTile(A, B)`: eager, cache-tiled product returning an `m x n` tensor.
- `matMulBlocked(A, B, C, m, alpha = 1, const tensorET *bias = nullptr, beta = 1,
  bool transA = false, bool transB = false, bool accumulate = false)`, which computes
  `C = alpha * op(A) op(B) + beta * bias` with `op` an optional transpose. With
  `accumulate`, the product is added to `C` instead of overwriting it. `bias`
  (if given) has `C`'s shape. `m` is a tile-size hint. OpenMP-parallel.
- `matMulND(A, B)`: batched product over leading dimensions (rank ≥ 3): the
  last two dimensions multiply, the leading ones must match.
- `extractSlice(t, b)` and `insertSlice(t, b, M)`: get and put the `b`-th 2-D
  matrix of a rank ≥ 3 tensor, with the leading dimensions flattened in row-major
  order.
- `getValue(t, idx*)` and `setValue(t, idx*, v)`: element access by an index array.

## 7. Dense linear algebra (Tier 2, host, `tensorLinearAlgebra.h`)

Matrices are `tensorET<2,T>` and vectors are `tensorET<1,T>`. Errors
(singular pivot, shape mismatch) throw `std::runtime_error`.

| Function | Contract |
|----------|----------|
| `norm(A, B, int p)` | p-norm of `A - B` over all elements (`p >= 1`). |
| `norm(A, B, "fro"/"inf"/...)` | named norms of `A - B`. |
| `conjugateGradient(A, b, x0, maxIter = -1, tol = 1e-6)` | CG for SPD `A`. `maxIter < 0` means `n`. Stops when the absolute residual norm drops below `tol`; also stops (with a message on stderr) when `pᵀAp` falls below 1e-12 in absolute value, which on well-scaled problems means it has converged. Returns `x`. |
| `gaussJordanElimination(A)` | Reduced row-echelon form of `A` (partial pivoting). |
| `augmentMatrix(A, b)` | `[A | b]`. |
| `gramSchmidtOrthogonalization(V)` | Matrix whose columns are an orthonormal basis of `V`'s columns, in order. |
| `luDcmp(A, tol = 1e-12)` | `{L, U}` without pivoting, `A = L U`, unit-diagonal `L`. Throws if a pivot has absolute value at most `tol`. |
| `luDcmpPivoted(A, tol)` | `{{L, U}, P}` with `P A = L U`, `P` a permutation matrix. |
| `luDcmpPivotedTile(A)` | Declared in the removed header but **never implemented** (no definition existed). Optional; if provided, same result as `luDcmpPivoted`. |
| `qrDecompositionTile(A)` | Thin QR of an `m x n` matrix. **Overwrites `A` with `Q`** (orthonormal columns) and **returns `R`** (`n x n`, upper triangular). Columns whose norm falls below 1e-12 are left as is (rank-deficient input is not handled). |
| `luSolve(L, U, P, B)` | Solves `A X = B` for a matrix or vector `B`, given the factors of `luDcmpPivoted`. |
| `lanczos(A, m, q0)` | Returns `{alpha (m), beta (m-1), Q (m x n)}`: tridiagonal coefficients and Krylov basis (with reorthogonalization) for symmetric `A`, starting at `q0`. |
| `expm(A)` | Matrix exponential (scaling and squaring with a Padé approximant, or equivalent accuracy: relative error at most 1e-12 on well-conditioned inputs). |
| `lobpcg(A, nev, X0, eigvals, X, maxIter = 100, tol = 1e-8)` | The `nev` smallest eigenpairs of symmetric `A` from the initial block `X0`. Writes `eigvals` (ascending) and `X` (columns = eigenvectors). |
| `inverse_backs(U, m = 4)` | Inverse of an **upper-triangular** `U` (`n` divisible by the block size `m`). This is what the fast benchmark numbers are for. It does not factor a general matrix. The removed version at `03a4c03` crashed (a product was written into an empty tensor); it must return an `n x n` result. |
| `revEl(A, m)`, `elimStep(A, factor, pivot, m)` | Reduces `A` to upper-triangular form **in place** by pairwise ("tournament") elimination. For each pivot column, rows are paired at distances 1, 2, 4, …; in each pair the row with the larger entry in the pivot column is swapped to the top (2-way partial pivoting), and the lower row is eliminated against it. Every row operation is recorded and returned as a history (`Op` = tagged union of `ElimOp{target_row, source_row, alpha}` and `PermOp{target_row, source_row}`); `elimStep` does one level for one pivot. Entries whose magnitude is below 1e-10 times the pivot are flushed to zero. |
| `matMul(history, A, m)`, `matMul(A, history, m)` | Replay a history on `A` from the left (row operations) or from the right (column operations), most recent operation first. **General inverse** = `revEl` on a copy of `A` giving `U` and history `H`, then `matMul(inverse_backs(U, m), H, m)`. It is correct (max error vs Eigen 1e-15 … 2e-12 for n = 64 … 512) but about 100x slower than Eigen in the removed version (§11.2). |

Accuracy requirement for the factorizations and solves: on random
well-conditioned matrices (condition number up to 1e3, `n` up to 2000), relative
residuals below `1e-12 * n` in double precision.

## 8. Stream output (Tier 2)

`std::ostream &operator<<(std::ostream&, const tensorET<DIM,T>&)` prints nested
brackets, one level per dimension, with the innermost elements separated by two
spaces and each sub-array on its own line, indented by two spaces per level. It
is for debugging only, not a serialization format.

## 9. GPU memory pool and library handles (Tier 1 with CUDA)

`GPUMemoryPool` (in `tensorcuda/gpu_pool.h`) is a per-thread singleton
obtained with `GPUMemoryPool::get()`. The retained code calls:

- `void *allocate(size_t bytes)`: returns device memory of at least `bytes`,
  aligned to 256 bytes; `nullptr` for `bytes == 0`. Freed blocks are reused for
  later requests that fit, preferring the smallest block that fits, so a loop
  that repeatedly allocates the same sizes performs no `cudaMalloc` after the
  first iteration. If the device is out of memory, the pool first returns all
  cached blocks to the driver and retries once. If that fails, it throws
  `std::runtime_error` with a message that includes the MB currently held and the
  requested size.
- `void deallocate(void *p, size_t bytes)`: returns a block to the pool
  (`bytes` is the size originally requested); no-op for `nullptr`.
- `void clear()`: returns all cached blocks to the driver.
- `cusparseHandle_t get_cusparse()` (with `CUSPARSE_WITH`) and
  `cudssHandle_t get_cudss()` (with `CUDSS_WITH`): lazily created, one per
  thread, reused for the thread's lifetime. They throw `std::runtime_error` if
  creation fails. `get_cublas()` and `get_cusolver()` follow the same pattern
  under their flags.
- Allocation mode is a compile-time choice: `Pool` (default, described above),
  `Raw` (direct `cudaMalloc`/`cudaFree` per call), `Async`
  (`cudaMallocAsync`/`cudaFreeAsync` on the default stream). Only `Pool` is
  required.

Handles and memory may be leaked at process exit (destruction order with the
CUDA runtime is not guaranteed), but not during normal operation.

## 10. Element types

Tier 1 needs `double`, `float`, `int32_t` and `int64_t` in both storage
policies (the sparse code stores indices in them). Tier 2 must also accept
`std::complex<float|double>` and a half-precision type: the compiler's
`_Float16` where available, with a 16-bit fallback struct convertible to and
from `float` otherwise, and CUDA's `__half` under nvcc. Other element types
should be rejected at compile time.

## 11. Performance targets

Machine for all targets: the development laptop (AMD Ryzen 7 7840HS, 8 cores /
16 threads, AVX-512; RTX 4060 Laptop GPU, CUDA 12.9), **on AC power**, g++
`-O3 -march=native -fopenmp`, double precision unless noted. Report the median of at
least 5 runs after a warm-up.

### 11.1 Tier 1: no regression of the retained code

The tensor layer sits under every solver vector operation, so the end-to-end
benchmarks are the real test. With the new implementation:

- `make benchmark_sparse`: every row of the table in
  `benchmarks/sparse/RESULTS.md` within **±5 %** of the recorded value (for example
  `spmv rand1m` CPU 3.66 ms, Eigen 4.71 ms; `ldl_factor lap200` CPU 12.5 ms;
  cuDSS 3.6 ms).
- `build/run_mps` on the netlib models and `build/bench_lp`: times within
  **±5 %** of `benchmarks/solver/RESULTS.md` (for example IPM 25fv47 about 55 ms
  CPU; PDLP CUDA transport700 at 1e-6 about 0.96 s, mcf50k about 0.93 s,
  packing200k about 5.1 s; HiGHS cuPDLP-C 1.8 / 1.9 / 19.3 s).
- Iteration counts **identical** to the recorded ones (the container must not
  change any arithmetic).

Micro-targets (these explain regressions when the above misses):

| Operation | Target |
|-----------|--------|
| `.data` access, `size()` | compile to a load; no call, no branch |
| `tensorET<1,double> v({n})` (uninitialized, host) | O(1) apart from the allocator; no page touched |
| `tensorET<1,double> v({n}, 0.0)`, n = 1e7 | ≤ 1.05 × `std::vector<double>(n, 0.0)` |
| host deep copy, n = 1e7 | ≤ 1.05 × `std::memcpy` |
| host→device / device→host conversion, n = 1e7 doubles | ≥ 90 % of a bare `cudaMemcpy` of the same pageable buffer |
| `Cuda` `({n}, 0.0)`, n = 1e7 | ≤ 1.1 × `cudaMemset` of the same size + launch |
| pool: `allocate`+`deallocate` of a previously seen size | ≤ 2 µs host time, zero `cudaMalloc` calls (check with `nsys`) |
| `read_element` on CUDA | one 8-byte transfer (≈10 µs); documented as slow |

### 11.2 Tier 2: measured baseline and targets

All numbers below were **measured** on 2026-09-25 (AC power) with the removed
implementation at commit `03a4c03` (`feature/csr`). They use the old harnesses
(`benchmarks/matops/bench_*.cpp`, `benchmarks/suite/tp_bench.cpp`) plus a
comparison program that links the old code and Eigen into one binary. Setup:
Eigen 3.4 (OpenMP where it parallelizes), PyTorch 2.12.1 CPU (best of 8 and 16
threads, or 1 thread for the single-thread rows), g++ `-O3 -march=native
-fopenmp`, OpenMP default 16 threads. Times are best or mean of 3–5 runs after a
warm-up. **% = the other library's time / the old code's time** (above 100 %
means the old code was faster).

**Rule for the targets:** the minimum acceptable is the old result (no
regression). The target is at least 101 % of the faster of Eigen and PyTorch.
Where the old code was far behind, the target is Eigen parity and the old
result is only the floor.

#### GEMM, `matMulBlocked` (best block size of 32/64/128), square n x n

| prec | threads | n | old ms | Eigen ms | PyTorch ms | vs Eigen | vs PyTorch |
|------|---------|---|--------|----------|------------|----------|------------|
| f64 | 16 | 256 | 0.08 | 0.18 | 0.18 | 215 % | 220 % |
| f64 | 16 | 512 | 1.13 | 2.04 | 0.90 | 181 % | 80 % |
| f64 | 16 | 1024 | 5.65 | 16.7 | 9.15 | 296 % | 162 % |
| f64 | 16 | 2048 | 46.4 | 66.6 | 60.6 | 144 % | 131 % |
| f64 | 16 | 3072 | 157 | 138 | 173 | 88 % | 111 % |
| f64 | 16 | 4096 | 373 | 313 | 430 | 84 % | 115 % |
| f64 | 1 | 256 | 0.60 | 0.59 | 0.69 | 99 % | 115 % |
| f64 | 1 | 1024 | 42.0 | 35.6 | 40.5 | 85 % | 96 % |
| f64 | 1 | 2048 | 397 | 273 | 310 | 69 % | 78 % |
| f32 | 16 | 256 | 0.04 | 0.11 | 0.06 | 264 % | 153 % |
| f32 | 16 | 1024 | 2.36 | 2.16 | 2.53 | 92 % | 107 % |
| f32 | 16 | 2048 | 21 ± 1 | 20 ± 1 | 35.3 | ≈ 95 % | 124 % |
| f32 | 16 | 4096 | 168 | 197 | 205 | 118 % | 123 % |
| f32 | 1 | 2048 | 230 | 134 | 159 | 58 % | 69 % |

The other old GEMM paths (`matMulTile`, lazy `matMul`) are much slower
(f64, 16 threads, n = 1024: tiled 34.5 ms, lazy 6.7 s). They exist for generality,
not speed.

Targets: ≥ 101 % of the faster of Eigen/PyTorch at every size and thread count
above. Minimum: the old column. The known gaps are n ≥ 3072 and single thread;
§15.2 explains why and how to close them.

#### GEMV, y = A x, n x n (f64)

| threads | n | best old kernel ms | Eigen ms | PyTorch ms | vs Eigen | vs PyTorch | library `matMulTile` ms |
|---------|---|--------------------|----------|------------|----------|------------|--------------------------|
| 16 | 4096 | 2.82 | 3.47 | 4.40 | 123 % | 156 % | 7.67 |
| 16 | 8192 | 10.7 | 12.1 | 15.6 | 113 % | 145 % | 23.5 |
| 16 | 16384 | 39.3 | 47.2 | 62.0 | 120 % | 158 % | 73.1 |
| 1 | 4096 | 4.93 | 3.33 | 4.30 | 67 % | 87 % | 33.1 |
| 1 | 16384 | 80.8 | 47.4 | 65.4 | 59 % | 81 % | 562 |

The fast GEMV kernels lived only in the benchmark programs; the library's
matrix-vector path was `matMulTile` (last column). f32 behaves the same (16
threads: 110–149 % of Eigen). Target: a library GEMV at ≥ 101 % of Eigen and
PyTorch with threads, and at least Eigen parity single-threaded. GEMV is
memory-bound, so this means running at the machine's memory bandwidth.

#### Triangular inverse, `inverse_backs` (upper triangular, f64, 16 threads)

| n | old ms (best m) | Eigen triangular solve with I | PyTorch `solve_triangular` | Eigen general `inverse()` | PyTorch general `inv` |
|---|-----------------|-------------------------------|----------------------------|---------------------------|-----------------------|
| 512 | 1.04 | 3.93 ms (378 %) | 0.68 ms (65 %) | 20.3 ms (1954 %) | 3.19 ms (307 %) |
| 1024 | 4.15 | 26.3 ms (634 %) | 5.99 ms (144 %) | 110 ms (2645 %) | 17.0 ms (409 %) |
| 2048 | 31.7 | 183 ms (576 %) | 41.8 ms (132 %) | 716 ms (2256 %) | 121 ms (380 %) |
| 4096 | 156 | 1276 ms (817 %) | 283 ms (181 %) | 4177 ms (2676 %) | 750 ms (480 %) |

The earlier "200–300 %" came from comparing against the general inverse. The fair
comparison is the triangular one, and the old code wins it from n = 1024 up.
Targets: ≥ 101 % of the faster triangular solve (PyTorch) at every size, which
closes the n = 512 gap. Minimum: the old column.

The **general inverse** built from the history (`revEl` + `inverse_backs` +
replay) was correct but slow: n = 256: 347 ms vs Eigen 1.9 ms; n = 512: 1400 ms vs
16.8 ms. Target: Eigen parity (see §15.4 for how).

#### Everything else (f64; Eigen and PyTorch as noted)

| Operation | n | old | Eigen | PyTorch (8 thr) | old vs Eigen |
|-----------|---|-----|-------|-----------------|--------------|
| `C = A + B` (1 thread) | 1e7 | 7.07 ms | 7.03 ms | 8.01 ms | 99 % |
| `C = A + B * 2.0` | 1e7 | 7.22 ms | 7.05 ms | 7.76 ms | 98 % |
| `C = exp(sin(A) + cos(A))` | 1e6 | 9.86 ms | 9.81 ms | 2.00 ms | 99 % |
| `C = sqrt(A)` | 1e7 | 19.8 ms | 4.80 ms | 5.71 ms | 24 % |
| `luDcmpPivoted` | 500 / 1000 / 2000 | 16.3 / 142 / 1860 ms | 2.1 / 7.7 / 38.1 ms | 0.53 / 2.1 / 18.4 ms | 13 / 5 / 2 % |
| `luSolve`, 1 rhs | 1000 / 2000 | 1.45 / 6.02 ms | 0.11 / 0.88 ms | 0.14 / 1.03 ms | 8 / 15 % |
| `qrDecompositionTile` | 500 / 1000 | 20.9 / 165 ms | 4.0 / 25.9 ms | 2.2 / 12.1 ms | 19 / 16 % |
| `conjugateGradient`, per iteration | 2000 | 3.5 ms | 0.62 ms | – | 18 % |
| `expm` | 100 / 200 | 1.97 / 15.8 ms | 0.30 / 1.35 ms | 0.31 / 0.60 ms | 15 / 9 % |
| `gaussJordanElimination` (suite) | 512 | 35.5 ms | – | – | – |
| transpose (suite) | 2048 | 60 ms | – | – | – |

The element-wise rows are single-threaded on both sides. PyTorch is multi-threaded,
so it wins the compute-bound chain by 5x; the old assignment loop had no threading.
Targets: element-wise ≥ 100 % of Eigen single-threaded and ≥ 101 % of PyTorch
threaded (the assignment must parallelize above a size threshold). The rest must
reach Eigen parity; the old values are only the floor. `expm` inherits the GEMM
speed once it uses the fast product. Transpose must run at least at half of copy
bandwidth.

Raw results: `benchmarks/reference/RESULTS_OLD_DENSE.md` (tables above plus the
float rows and every block size).

## 12. Acceptance tests

1. `make test_sparse_cpu` and `make test_solver_cpu` compile and pass (last
   recorded: 630 and 268 checks). With CUDA, `make test_sparse` and
   `make test_solver` pass (1176 and 330 checks).
2. The §11.1 benchmarks are within tolerance and the iteration counts are identical.
3. A new tensor test suite (to be written with the implementation) covering
   every row of §5, including:
   - shape errors throw (`{..}` with the wrong count, `size(DIM)`, `stride(-1)`, bad `reshape`);
   - copying a view aliases, and copying an owning tensor does not (mutate one, check the other);
   - after a move, the source is empty and the destination holds the data;
   - host→device→host round trip is bit-identical for all Tier 1 element types, including n = 0 and n = 1;
   - `slice` strides equal the parent's; `reshape` of a slice is not required to work;
   - `blockCopy` zero padding at the right and bottom edges; `transpose=true`;
     write-back and add-write-back touch only in-range elements;
   - CUDA writable element access does not compile (a negative compile test, or a
     `static_assert` check via detection idiom);
   - the pool reuses blocks (allocate/free/allocate same size returns a block
     without a new `cudaMalloc`) and recovers from a simulated OOM.
4. Tier 2: each function in §6–§7 against an Eigen reference: element-wise results
   bit-identical to the same scalar expression; factorizations to the §7 accuracy;
   expression chains produce no heap allocation (check with a counting allocator).

## 13. Usage inventory of the retained code

What the kept files actually use, so Tier 1 can be sized precisely:

- Types: `tensorET<1, T, S>` and `tensorET<2, T, S>` for `T` in {double, float,
  int32_t, int64_t}, `S` in {`Cpu::HostStorage<T>`, `Cuda::CudaStorage<T>`};
  `Cpu::Backend`, `Cuda::Backend` as tags; `value_type`, `backend_type`.
- Constructors: `({n}, x)`, `({m, k}, x)`, cross-storage `Vec d(host)` /
  `tensorET<1,T> h(device)`; copy and move of vectors (members of solver objects).
- Members: `.data` (field), `.size()`, `.size(0)`, `.size(1)`.
- Storage policies used directly by `Csr`: `Store<T>(n)`, `.data()`, copy,
  move, default construction.
- `GPUMemoryPool::get().allocate/deallocate/get_cusparse/get_cudss`.
- Macros: `AXOS_ENABLE_CUDA`, `CUSPARSE_WITH`, `CUDSS_WITH` (Makefile);
  `AXOS_HD` is defined by the solver itself in `pdlp_kernels.h`.

Nothing else in §5–§10 is used by the retained code.

## 14. Open decisions for the implementers

- Names: `tensorET` and the header names are kept so that the retained code
  compiles unchanged. Renaming them (like the namespace was) is a mechanical
  follow-up.
- Whether bookkeeping fields stay public (§5.2).
- Seeds for `uniform`/`gaussian` (§5.9).
- Whether the placeholder assignment (§5.10) and the domain constructor
  (§5.3, §5.7) are worth keeping.

## 15. Design notes: how the performance was obtained

This section explains how the removed implementation got the §11.2 numbers,
where it lost performance, and what a rewrite should do. It gives ideas and
parameters, not code. The retained `src/sparse/dense_ldl.h` (a packed,
register-blocked `L D Lᵀ` update written for this project, not part of the
removed code) is a working in-tree example of the kernel techniques in §15.2 and
may be read freely.

### 15.1 Expression templates (element-wise math)

- **Static polymorphism via CRTP.** Every expression node (a tensor, a
  sum, a scalar product, `exp(...)`, …) derives from a common base template that
  is parameterized on the node's own type. The operators and math functions accept
  "anything derived from the base", then downcast statically to the concrete type.
  There are no virtual functions and no runtime graph: the full expression tree
  is a single C++ type known at compile time.
- **Nodes are tiny and hold references.** A binary node stores const
  references to its two operands (a scalar node stores the scalar by value); it
  owns no data. Building `A + B * 2.0` allocates nothing.
- **Evaluation is pull-based per element.** Each node answers `value(i)` by
  asking its children for `value(i)` and combining them. Assignment to a tensor is
  one loop over `i` that calls the root's `value(i)`. With everything inlined, the
  compiler flattens the tree into a single loop body and vectorizes it. This is
  why the old code matched Eigen (98–99 %) on `A + B`, `A + B * 2` and the
  `exp(sin + cos)` chain.
- **Where it lost:**
  - The assignment loop was single-threaded; add an OpenMP split above roughly
    32–64 K elements. This is how PyTorch won the compute-bound chain by 5x.
  - `sqrt` ran at about 20 % of Eigen because the scalar `std::sqrt` call with
    `errno` semantics blocks vectorization. Use `-fno-math-errno`, or map
    `sqrt` (and `fabs`, `min`, `max`) to vector instructions explicitly.
  - Transcendentals rely on libm auto-vectorization (glibc `libmvec`), which is
    what makes `exp/sin/cos` competitive. Keep the loop simple enough that the
    compiler still emits the vector calls.
- **Aliasing:** `A = A + B` is safe element-wise because element `i` depends
  only on inputs at `i`. The lazy `matMul` is not element-local and must not be
  assigned to one of its own operands.

### 15.2 GEMM (`matMulBlocked`)

What made it beat Eigen and PyTorch up to n ≈ 2048 with threads:

1. **Square tiling of the output.** C is cut into m x m tiles (m = 64 was best
   up to n = 1024, and m = 128 at n ≥ 2048). **Each output tile is one parallel
   task**, and the tiles are handed out dynamically. The task loops over all k
   tiles and accumulates into a thread-private m x m buffer, then writes the tile
   back once. No two threads ever write the same part of C, so there are no
   reductions or atomics. The fused `alpha`/bias epilogue is applied to the
   private buffer before the write-back.
2. **Packing.** For each (i, k) and (k, j) step, the A and B tiles are copied
   into 64-byte-aligned, contiguous m x m buffers. The copy zero-pads edges and
   applies a requested transpose during packing, so the kernel only ever sees
   aligned, unit-stride, square data.
3. **Register-blocked micro-kernel.** The f64 AVX-512 inner kernel keeps a
   **4-row x 16-column block of C in 8 zmm registers** for the whole k loop.
   Each step loads two vectors of a B row, broadcasts four A elements (one per
   row) and issues 8 FMAs, then stores the block once at the end. The f32 kernel
   has the same shape (16 lanes per vector). AVX2 and scalar fallbacks are chosen
   by tile size (m ≥ 64 and a multiple of 16 for f64 AVX-512).
4. **Compile-time dispatch** on element type and ISA (`if constexpr` plus
   preprocessor feature tests), so there is no runtime branching in the hot loop.

Why it fell to 84–88 % of Eigen at n ≥ 3072, and to 58–85 % single-threaded:

- **Every output tile re-packs its whole row of A tiles and column of B
  tiles.** Total packing traffic grows as n³/m instead of n². Eigen and GotoBLAS
  pack a B panel once and reuse it across all row blocks.
- **Three `malloc`/`free` calls per tile task** (a heap allocation inside the
  parallel loop).
- **The micro-tile is small.** 4 x 16 f64 is 8 accumulators, while AVX-512 has
  32 registers. A wider tile (for example 8 x 24 or 12 x 16) raises the FMA to
  load ratio. `dense_ldl.h` uses 24 x 8 with KC = 256.
- **One tile size m serves as cache block, micro-kernel shape and task size
  at once.** A rewrite should separate these:
  - an MC x KC block of A kept in L2;
  - a KC x NC panel of B kept in L3 and shared by all threads;
  - an MR x NR register tile;
  - tasks formed over the MC blocks inside one NC panel.
  Keep the "one task owns one C region" property, which is what made it scale.
- **Square-only, n divisible by m.** Rectangular shapes and remainder tiles
  should be handled without falling back to the scalar kernel.

### 15.3 GEMV

GEMV is memory-bound: the matrix is read once. The fastest old kernels, which
lived only in benchmark programs:

- split **rows** across threads (contiguous row ranges, static schedule);
- for each row, run a dot product over the contiguous row with **several
  independent vector accumulators** (4 zmm), combining them only at the end;
  a single accumulator is latency-bound on the FMA chain.

That reached 113–123 % of Eigen with threads. Single-threaded it reached only
57–67 %: one core cannot saturate memory bandwidth, and Eigen single-threaded
processes several rows at once to reuse the `x` loads. A rewrite should:
- do several rows per pass (4–8) to amortize loading `x`;
- prefetch;
- make the **library** GEMV (and `matMulTile` with a vector) use this kernel. The
  old library path was 2–10x slower than the benchmark kernel.

### 15.4 Inverses

**Upper-triangular inverse (`inverse_backs`), 3.8–8x Eigen, 1.3–1.8x PyTorch:**

1. Invert all m x m diagonal blocks **independently in parallel** (small dense
   triangular inverses).
2. Then, for each block column j, **in parallel over j**, go up the column:
   `X_ij = −X_ii · Σ_{k=i+1..j} A_ik X_kj`. The block products use the fast GEMM.
   Block columns are independent, so the parallelism is embarrassing. The work
   is about n³/3 flops, all in GEMM.
3. It beats Eigen and PyTorch because they solve U X = I as a general
   triangular solve with a dense right-hand side. They do not exploit that the
   result is triangular (half the work), and Eigen 3.4 only threads its GEMM, not
   its triangular solve. The n = 512 loss to PyTorch is probably from too few
   block columns for 16 threads plus the GEMM re-packing overhead of §15.2
   (not profiled).

**General inverse:** the old path (tournament elimination with a recorded
history, then `inverse_backs`, then replaying the history) was ~100x slower than
Eigen. Every recorded row operation became a separate OpenMP parallel region
over column blocks, with heap-allocated tile buffers per operation: about n²
parallel regions for an n x n matrix. The pairwise elimination idea is
numerically fine (2-way partial pivoting, errors 1e-15 … 1e-12). Its execution
granularity is the problem. To reach Eigen parity:
- factor with a **blocked, right-looking LU**: a panel factorization of m
  columns, a triangular solve for the block row, and a GEMM trailing update
  (> 90 % of the flops in the fast GEMM);
- then either invert `U` with the `inverse_backs` scheme and `L` with its lower
  mirror, and multiply (`A⁻¹ = U⁻¹ L⁻¹ P`), or solve `A X = I` with blocked
  triangular solves that skip the known-zero part of the right-hand side;
- if the recorded-history replay is kept for other uses, apply the operations
  in **batches** (all ops of one elimination level touch disjoint row pairs and
  can go in one parallel pass), not one parallel region per operation.

### 15.5 Factorizations, CG, expm

The old implementations were textbook scalar algorithms (floors in §11.2). What
to build instead:

- **LU (2–13 % of Eigen):** it was unblocked and serial. It also swapped rows in
  full `L`, `U` and `P` matrices. Use the blocked right-looking scheme of §15.4,
  store the permutation as a vector, and keep `P` as a matrix only in the
  public result.
- **luSolve (8–15 %):** forward/back substitution on the dense `L`/`U`
  matrices with a permutation-matrix multiply. Apply the permutation as an
  index vector, and use row-major substitution with vectorized dot products.
- **QR (16–19 %):** tiled modified Gram–Schmidt (column by column). Use
  blocked Householder with the compact-WY representation, so most of the work
  becomes GEMM. This changes the numerics (better orthogonality) but not the
  interface (A becomes Q, R is returned).
- **CG (18 % per iteration):** each iteration allocated a new vector for
  `A p` and used the slow matrix-vector path. Preallocate all work vectors and use
  the fast GEMV; fuse the vector updates (`x += αp`, `r −= αAp`, `rᵀr`) into one
  pass.
- **expm (9–15 %):** scaling and squaring with a degree-13 Padé approximant
  (Higham's method), with the 1-norm choosing the scaling. The algorithm is the
  right one; the time goes into its ~6 matrix products, the LU solve and the
  squarings, all on the slow paths. With the fast GEMM and blocked LU it reaches
  Eigen parity.
- **Lanczos / LOBPCG:** Lanczos with DGKS re-orthogonalization. LOBPCG solves
  the Rayleigh–Ritz step with a Jacobi eigensolver on the small Gram matrix.
  Both are dominated by GEMV/GEMM and inherit their speed.

### 15.6 Memory and threading conventions

- Never allocate inside a parallel loop or per task. Give each thread its
  packing and scratch buffers once per call; `thread_local` buffers that grow
  on demand are fine.
- Aligned (64-byte) packing buffers, unaligned loads on user tensors.
- Use one OpenMP parallel region per operation with work-sharing inside,
  rather than one region per small step (see §15.4).
- Parallelize only above a work threshold. For example `dense_ldl.h` gates on
  about 2e6 flops; smaller problems run serially to avoid fork/join cost.
- **Transpose** (60 ms at n = 2048, about 1 GB/s of read + write traffic): use cache-tiled 32 x 32 or
  64 x 64 blocks, in parallel over tiles. The target is at least half of copy
  bandwidth.

## 16. Implementation status

### 16.1 Where the code is

| Header | Contents |
|--------|----------|
| `src/storage/host_storage.h` | `Cpu::Backend`, `Cpu::HostStorage`, `is_host_storage`, `backend_of_t` |
| `src/storage/cuda_storage.h` | `Cuda::Backend`, `Cuda::CudaStorage` (a never-allocating stub without CUDA) |
| `src/tensorcuda/gpu_pool.h` | `GPUMemoryPool` (§9) |
| `src/tensorET.h` | `tensorET`, `Domain` |
| `src/tensorCuda.h` | `tensorETCuda`, `blockCopyCuda`, `blockWriteBackCuda` |
| `src/tensorMath.h` | expression nodes and operators, math functions, `matMul*`, `gemv`, `transpose` |
| `src/tensorLinearAlgebra.h` | §7 |
| `src/tensorIO.h` | §8 |
| `src/axos.h` | umbrella header |
| `src/tensor/` | internals: `element.h` (element types, `half`), `expr.h` (expression base and evaluator), `simd.h` (AVX-512 / AVX2 / SSE2 / scalar packs), `parallel.h` (OpenMP helpers), `blas.h` (GEMM, GEMV, transpose, dot, axpy on raw arrays) |

Tests: `tests/test_tensor.cpp` (`make test_tensor_cpu`, `make test_tensor`).
Benchmark: `benchmarks/dense/bench_dense.cpp` (`make benchmark_dense`).

### 16.2 Decisions on §14 and deviations

- Names and headers are the ones in §2; the retained code is unchanged.
- Bookkeeping (shape, strides, domain) is private; `data` and `storage_` are
  public as required. The storage policies expose `data()`, `size()`,
  `owns()`, `allocate_raw()` / `free_raw()` instead of public fields.
- `uniform` / `gaussian` take an optional seed; with a seed the values do not
  depend on the thread count.
- The placeholder assignment (§5.10) is not provided; `fill(x)` sets all
  elements (views included). The domain constructor (§5.3, §5.7) is provided.
- `tensorET(T *raw, size_t dims[DIM], bool own)` is Tier 1 in practice:
  `Csr::values_view()` uses it. `std::complex` element types are needed in
  both storages by the sparse tests; both are supported.
- `tensorET(size_t n)` and the array-form constructors are `explicit`
  (the literal `0` would otherwise be ambiguous with a null pointer).
- Copy-assigning an owning tensor into an owning tensor of the same size
  reuses the target's buffer (same observable result as a new allocation);
  the same holds for host <-> device assignment.
- Element-wise evaluation uses explicit SIMD for `+ - * /`, `sqrt`, `fabs`
  and negation, and the scalar `std::` functions for transcendentals, so
  results are bit-identical to the scalar expression; outputs of 8 MiB and
  more are written with non-temporal stores. A lazy `matMul` inside a larger
  expression is computed once into a cache (by GEMM) before the element pass.
- `matMulBlocked`'s `m` and `inverse_backs`' `m` are accepted as legacy hints
  and not used (the kernels choose their own blocking); `inverse_backs` works
  for any n, not only multiples of m.
- `inverse(A)` (general inverse via blocked LU and triangular solves) is an
  addition; so are `gemv`, `matVec`, `transpose`, `dot`, `symmetricEigen`,
  `clone`, `fill`, `slice`/`reshape` on device tensors and the `Stats` /
  `set_limit` testing hooks of the pool.
- `norm(A, B, name)`: `"fro"`, `"l2"`, `"2"` (Frobenius), `"l1"`, `"1"`,
  `"inf"`, `"max"` (largest absolute difference); other names throw.
- `lanczos` returns `std::tuple{alpha, beta, Q}`; `lobpcg` returns the number
  of iterations; `luDcmp*` return `std::pair`s.
- `revEl` history: `Op<T> = std::variant<ElimOp<T>, PermOp>`. With H the
  product of the recorded operations, `matMul(H, A)` computes H A (row
  operations, oldest first) and `matMul(A, H)` computes A H (column
  operations, most recent first), so `matMul(inverse_backs(U), H)` is A^-1.
- `luSolve` takes P as a matrix (§7). Scanning a dense P costs more than the
  triangular solves, so `luDcmpPivoted` remembers the last few permutations
  per thread; `luSolve` uses one only after checking P(i, perm[i]) == 1 for
  every row (which determines a permutation matrix), and otherwise scans P.
- `conjugateGradient` implements the stop on |p^T A p| < 1e-12 as specified.
  Because the threshold is absolute, it limits the reachable residual to
  about sqrt(1e-12 / lambda_min) (around 1e-6 for unit-scale SPD matrices);
  a smaller `tol` has no further effect.
- The linear algebra (§7) is implemented for `float` and `double`; GEMM and
  the expressions also work for integers, complex and half (scalar kernels).

### 16.3 Verification so far

The development machine used for this implementation is not the one in §11
(it has an Intel i7-12650H, 6 performance + 4 efficiency cores, AVX2 only, and
MSVC 2019 but no g++, nvcc or Linux), so:

- `tests/test_tensor.cpp` passes on the host (210 checks) and, with the CUDA
  runtime replaced by a host-memory mock, the CUDA paths too (229 checks):
  round trips of all Tier 1 element types, fills, strided 2-D transfers, pool
  reuse and out-of-memory recovery, handles, the tile helpers.
- The retained headers that do not need g++ vector extensions (`csr.h`,
  `sparse_cpu.h`, `sparse_ops.h`, `model.h`, `mps.h`, `presolve.h`,
  `pdlp_kernels.h`) were compiled against the new layer and their tensor
  usage exercised. `dense_ldl.h` (and so the LDLᵀ, the IPM, PDLP and the
  simplex) needs g++; `make test_sparse_cpu test_solver_cpu` (§12.1) and the
  CUDA targets have **not been run yet**, nor the §11.1 end-to-end
  benchmarks. The AVX-512 GEMM micro-kernel (8 x 24 / 8 x 48) is compiled only
  on AVX-512 machines and has not been executed yet.

Measured on the development machine (MSVC `/O2 /arch:AVX2`, median of 5,
`make benchmark_dense` equivalent), single-threaded against single-threaded
Eigen 3.4 (MSVC's multi-threaded Eigen is not a usable reference on this
hybrid CPU): GEMM f64 0.97-1.38x Eigen's speed (n = 256..3072), f32
0.92-1.27x; GEMV 0.95-1.01x; `A + B` / `A + B * 2` 1.33x; `sqrt` 1.05x;
`exp(sin + cos)` 1.02x; transpose 9.5x; `luDcmpPivoted` 0.83-1.08x (it also
builds the dense L, U and P that Eigen does not); `luSolve` 0.87-0.97x;
general inverse 0.81-1.03x; `inverse_backs` 1.11-1.39x; QR 1.19-1.20x; CG
0.96x; expm 1.44-1.52x. With 16 threads: GEMM f64 about 220 GF/s (n = 2048),
`exp(sin + cos)` of 1e6 elements 2.1 ms, `A + B` of 1e7 5.8 ms. On this
laptop, spin-waiting OpenMP workers slow the serial phases of small
factorizations (LU at n = 500: 5.7 ms with 16 threads, 3.0 ms with 6, 3.9 ms
with `OMP_WAIT_POLICY=PASSIVE`); symmetric CPUs such as the §11 machine are
not expected to show this.

Still to do on the §11 machine: `make test_tensor test_sparse test_solver`
(and the `_cpu` variants), `make benchmark_sparse benchmark_lp` against the
recorded results (§11.1), and `make benchmark_dense` for §11.2.
