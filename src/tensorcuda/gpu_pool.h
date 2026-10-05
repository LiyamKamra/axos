// SPDX-License-Identifier: BSD-3-Clause
//
// GPUMemoryPool: per-thread caching allocator for device memory, plus the
// per-thread cuSPARSE / cuDSS / cuBLAS / cuSOLVER handles
// (docs/TENSOR_SPEC.md §9). Obtain it with GPUMemoryPool::get().
//
//   allocate(bytes)     device memory, 256-byte aligned (512-byte granules);
//                       nullptr for 0 bytes. A freed block is reused for a
//                       later request that fits (smallest first) as long as
//                       it is at most twice the request, so a loop that
//                       allocates the same sizes calls cudaMalloc only in its
//                       first iteration. On out-of-memory the cached blocks
//                       go back to the driver and the allocation is retried
//                       once; after that it throws std::runtime_error with
//                       the requested size and the MB the pool held.
//   deallocate(p, n)    returns a block to the cache (the pool tracks block
//                       sizes itself, so n is informational). A pointer the
//                       pool did not hand out (adopted memory, or a block
//                       from another thread's pool) is freed directly.
//   clear()             returns all cached blocks to the driver.
//   get_cusparse() ...  lazily created, one per thread, reused for the
//                       thread's lifetime; creation failure throws.
//
// The allocation mode is fixed at compile time by AXOS_GPU_ALLOC_MODE:
// 0 = Pool (default, described above), 1 = Raw (cudaMalloc/cudaFree per
// call), 2 = Async (cudaMallocAsync/cudaFreeAsync on the default stream).
// A thread's pool and handles are released when the thread exits; objects
// destroyed after that (static storage) free their memory directly.
#pragma once

#if !defined(AXOS_ENABLE_CUDA) && !defined(__CUDACC__)
#error "tensorcuda/gpu_pool.h needs a CUDA build (define AXOS_ENABLE_CUDA)"
#endif

#include <cuda_runtime.h>
#if defined(CUSPARSE_WITH)
#include <cusparse.h>
#endif
#if defined(CUDSS_WITH)
#include <cudss.h>
#endif
#if defined(CUBLAS_WITH)
#include <cublas_v2.h>
#endif
#if defined(CUSOLVER_WITH)
#include <cusolverDn.h>
#endif

#include <cstddef>
#include <cstdio>
#include <map>
#include <stdexcept>
#include <string>
#include <unordered_map>

#ifndef AXOS_GPU_ALLOC_MODE
#define AXOS_GPU_ALLOC_MODE 0
#endif

namespace AXOS {

class GPUMemoryPool;

namespace detail {
struct PoolSlot {
    GPUMemoryPool *pool = nullptr;
    bool torn_down = false;
};
inline thread_local PoolSlot pool_slot; // trivially destructible
struct PoolReaper {
    ~PoolReaper();
};
} // namespace detail

class GPUMemoryPool {
  public:
    enum class Mode { Pool = 0, Raw = 1, Async = 2 };
    static constexpr Mode mode = static_cast<Mode>(AXOS_GPU_ALLOC_MODE);
    static constexpr size_t kGranule = 512;

    struct Stats {
        size_t device_mallocs = 0; // successful cudaMalloc(Async) calls
        size_t device_frees = 0;   // cudaFree(Async) calls
        size_t reuses = 0;         // requests served from the cache
        size_t bytes_in_use = 0;   // handed out, not yet returned
        size_t bytes_cached = 0;   // returned, kept for reuse
        size_t oom_recoveries = 0; // out-of-memory retries after clear()
    };

    static GPUMemoryPool &
    get()
    {
        detail::PoolSlot &s = detail::pool_slot;
        if (s.pool) return *s.pool;
        s.pool = new GPUMemoryPool();
        if (!s.torn_down) {
            thread_local detail::PoolReaper reaper;
            (void)reaper;
        }
        return *s.pool;
    }

    GPUMemoryPool(const GPUMemoryPool &) = delete;
    GPUMemoryPool &operator=(const GPUMemoryPool &) = delete;

    void *
    allocate(size_t bytes)
    {
        if (bytes == 0) return nullptr;
        const size_t need = (bytes + kGranule - 1) / kGranule * kGranule;
        if (mode == Mode::Pool) {
            auto it = free_.lower_bound(need);
            if (it != free_.end() && it->first <= 2 * need) {
                const size_t sz = it->first;
                void *p = it->second;
                free_.erase(it);
                st_.bytes_cached -= sz;
                track(p, sz);
                ++st_.reuses;
                return p;
            }
        }
        void *p = device_malloc(need);
        track(p, need);
        return p;
    }

    void
    deallocate(void *p, size_t bytes) noexcept
    {
        (void)bytes;
        if (!p) return;
        auto it = used_.find(p);
        if (it == used_.end()) {
            device_free(p);
            return;
        }
        const size_t sz = it->second;
        used_.erase(it);
        st_.bytes_in_use -= sz;
        if (mode == Mode::Pool) {
            try {
                free_.emplace(sz, p);
                st_.bytes_cached += sz;
                return;
            } catch (...) {
            }
        }
        device_free(p);
    }

    void
    clear() noexcept
    {
        for (auto &kv : free_)
            device_free(kv.second);
        free_.clear();
        st_.bytes_cached = 0;
    }

    Stats
    stats() const
    { return st_; }

    // Testing hook: treat any allocation that would make this pool hold more
    // than `bytes` of device memory as out of memory (0 = no limit).
    void
    set_limit(size_t bytes)
    { limit_ = bytes; }

#if defined(CUSPARSE_WITH)
    cusparseHandle_t
    get_cusparse()
    {
        if (!cusparse_ &&
            cusparseCreate(&cusparse_) != CUSPARSE_STATUS_SUCCESS) {
            cusparse_ = nullptr;
            throw std::runtime_error("GPUMemoryPool: cusparseCreate failed");
        }
        return cusparse_;
    }
#endif
#if defined(CUDSS_WITH)
    cudssHandle_t
    get_cudss()
    {
        if (!cudss_ && cudssCreate(&cudss_) != CUDSS_STATUS_SUCCESS) {
            cudss_ = nullptr;
            throw std::runtime_error("GPUMemoryPool: cudssCreate failed");
        }
        return cudss_;
    }
#endif
#if defined(CUBLAS_WITH)
    cublasHandle_t
    get_cublas()
    {
        if (!cublas_ && cublasCreate(&cublas_) != CUBLAS_STATUS_SUCCESS) {
            cublas_ = nullptr;
            throw std::runtime_error("GPUMemoryPool: cublasCreate failed");
        }
        return cublas_;
    }
#endif
#if defined(CUSOLVER_WITH)
    cusolverDnHandle_t
    get_cusolver()
    {
        if (!cusolver_ &&
            cusolverDnCreate(&cusolver_) != CUSOLVER_STATUS_SUCCESS) {
            cusolver_ = nullptr;
            throw std::runtime_error("GPUMemoryPool: cusolverDnCreate failed");
        }
        return cusolver_;
    }
#endif

  private:
    friend struct detail::PoolReaper;

    GPUMemoryPool() = default;

    ~GPUMemoryPool()
    {
        clear();
#if defined(CUSPARSE_WITH)
        if (cusparse_) cusparseDestroy(cusparse_);
#endif
#if defined(CUDSS_WITH)
        if (cudss_) cudssDestroy(cudss_);
#endif
#if defined(CUBLAS_WITH)
        if (cublas_) cublasDestroy(cublas_);
#endif
#if defined(CUSOLVER_WITH)
        if (cusolver_) cusolverDnDestroy(cusolver_);
#endif
    }

    void
    track(void *p, size_t sz)
    {
        auto r = used_.emplace(p, sz);
        if (!r.second) { // stale entry: freed directly by another thread
            st_.bytes_in_use -= r.first->second;
            r.first->second = sz;
        }
        st_.bytes_in_use += sz;
    }

    bool
    try_malloc(void **p, size_t need)
    {
        if (limit_ && st_.bytes_in_use + st_.bytes_cached + need > limit_)
            return false;
        const cudaError_t e = mode == Mode::Async ? cudaMallocAsync(p, need, 0)
                                                  : cudaMalloc(p, need);
        if (e == cudaSuccess) {
            ++st_.device_mallocs;
            return true;
        }
        if (e == cudaErrorMemoryAllocation) {
            (void)cudaGetLastError(); // reset the error state
            return false;
        }
        throw std::runtime_error(
            std::string("GPUMemoryPool: cudaMalloc failed: ") +
            cudaGetErrorString(e));
    }

    void *
    device_malloc(size_t need)
    {
        void *p = nullptr;
        if (try_malloc(&p, need)) return p;
        const size_t held = st_.bytes_in_use + st_.bytes_cached;
        clear();
        ++st_.oom_recoveries;
        if (try_malloc(&p, need)) return p;
        char msg[192];
        std::snprintf(msg, sizeof msg,
            "GPUMemoryPool: out of device memory allocating %.1f MB "
            "(the pool held %.1f MB, %.1f MB still in use)",
            need / 1048576.0, held / 1048576.0, st_.bytes_in_use / 1048576.0);
        throw std::runtime_error(msg);
    }

    void
    device_free(void *p) noexcept
    {
        if (mode == Mode::Async)
            (void)cudaFreeAsync(p, 0);
        else
            (void)cudaFree(p);
        ++st_.device_frees;
    }

    std::multimap<size_t, void *> free_;      // cached blocks by size
    std::unordered_map<void *, size_t> used_; // live blocks -> size
    Stats st_;
    size_t limit_ = 0;
#if defined(CUSPARSE_WITH)
    cusparseHandle_t cusparse_ = nullptr;
#endif
#if defined(CUDSS_WITH)
    cudssHandle_t cudss_ = nullptr;
#endif
#if defined(CUBLAS_WITH)
    cublasHandle_t cublas_ = nullptr;
#endif
#if defined(CUSOLVER_WITH)
    cusolverDnHandle_t cusolver_ = nullptr;
#endif
};

inline detail::PoolReaper::~PoolReaper()
{
    PoolSlot &s = pool_slot;
    delete s.pool;
    s.pool = nullptr;
    s.torn_down = true;
}

} // namespace AXOS
