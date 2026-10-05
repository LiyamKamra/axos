// SPDX-License-Identifier: BSD-3-Clause
//
// Multifrontal LDL^T on relaxed supernodes for symmetric matrices with a fixed
// pivot order (no numerical pivoting: SPD and quasi-definite systems).
//
// analyze(): given an elimination order it computes the elimination tree,
// postorders it, finds supernodes (columns with nested structure, merged when
// the explicit zeros this adds are cheap), and each supernode's row
// structure. factorize(): for every supernode in postorder it assembles the
// original entries and the children's update matrices into a dense front,
// factors the leading columns with dense kernels (dense_ldl.h) and passes the
// Schur complement on to the parent. Large fronts use OpenMP inside the
// dense update. solve(): supernodal forward / diagonal / backward sweeps.
#pragma once

#include "sparse/dense_ldl.h"
#include <algorithm>
#include <atomic>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <map>
#include <memory>
#include <mutex>
#include <new>
#include <numeric>
#include <stdexcept>
#include <vector>

namespace AXOS {
namespace Sparse {

template <typename T, typename Idx> class MultifrontalLdl {
  public:
    // pivot check: SPD needs > 0, otherwise nonzero and finite.
    enum class Kind { SPD, Symmetric };

    // (rp, ci): CSR pattern of the full symmetric matrix; perm: elimination
    // order (perm[k] = original index eliminated k-th).
    void
    analyze(
        size_t n, const Idx *rp, const Idx *ci, const std::vector<Idx> &perm0)
    {
        n_ = n;
        // ---- etree of the given order, then postorder ------------------
        std::vector<Idx> pinv0(n), parent0, cnt0;
        for (size_t k = 0; k < n; ++k)
            pinv0[perm0[k]] = static_cast<Idx>(k);
        etree_counts(n, rp, ci, perm0, pinv0, parent0, cnt0);
        std::vector<Idx> post = postorder(parent0);
        perm_.resize(n);
        for (size_t k = 0; k < n; ++k)
            perm_[k] = perm0[post[k]];
        pinv_.assign(n, 0);
        for (size_t k = 0; k < n; ++k)
            pinv_[perm_[k]] = static_cast<Idx>(k);
        std::vector<Idx> parent, cnt;
        etree_counts(n, rp, ci, perm_, pinv_, parent, cnt);

        factor_nnz_ = 0;
        for (size_t j = 0; j < n; ++j)
            factor_nnz_ += static_cast<size_t>(cnt[j]);

        // ---- lower CSC of the permuted matrix, with value positions -----
        std::vector<Idx> count(n + 1, 0);
        for (size_t r = 0; r < n; ++r) {
            const Idx pr = pinv_[r];
            for (Idx p = rp[r]; p < rp[r + 1]; ++p) {
                const Idx pc = pinv_[ci[p]];
                if (pc <= pr) count[pc + 1]++;
            }
        }
        for (size_t j = 0; j < n; ++j)
            count[j + 1] += count[j];
        acp_ = count;
        ari_.assign(acp_[n], 0);
        asrc_.assign(acp_[n], 0);
        std::vector<Idx> nextp(acp_.begin(), acp_.end() - 1);
        for (size_t r = 0; r < n; ++r) {
            const Idx pr = pinv_[r];
            for (Idx p = rp[r]; p < rp[r + 1]; ++p) {
                const Idx pc = pinv_[ci[p]];
                if (pc <= pr) {
                    Idx d = nextp[pc]++;
                    ari_[d] = pr;
                    asrc_[d] = p;
                }
            }
        }

        // ---- fundamental supernodes ---------------------------------------
        struct Sn {
            Idx start, width;
            size_t nnz;
        };
        std::vector<Sn> sns;
        {
            Idx s0 = 0;
            for (size_t j = 1; j <= n; ++j) {
                bool cont = j < n && parent[j - 1] == static_cast<Idx>(j) &&
                            cnt[j - 1] == cnt[j] + 1;
                if (!cont) {
                    Sn s;
                    s.start = s0;
                    s.width = static_cast<Idx>(j) - s0;
                    s.nnz = 0;
                    for (Idx c = s0; c < static_cast<Idx>(j); ++c)
                        s.nnz += static_cast<size_t>(cnt[c]) + 1;
                    sns.push_back(s);
                    s0 = static_cast<Idx>(j);
                }
            }
        }
        // ---- relaxed amalgamation (merge a child into an adjacent parent) --
        {
            std::vector<Sn> out;
            for (const Sn &cur0 : sns) {
                Sn cur = cur0;
                while (!out.empty()) {
                    const Sn &prev = out.back();
                    const Idx last = prev.start + prev.width - 1;
                    const Idx pc = parent[last];
                    if (!(pc >= cur.start && pc < cur.start + cur.width &&
                            prev.start + prev.width == cur.start))
                        break;
                    const size_t ns =
                        static_cast<size_t>(prev.width) + cur.width;
                    const size_t below =
                        static_cast<size_t>(cnt[cur.start + cur.width - 1]);
                    const size_t dense = ns * below + ns * (ns + 1) / 2;
                    const size_t truenz = prev.nnz + cur.nnz;
                    const double z =
                        dense > truenz
                            ? static_cast<double>(dense - truenz) / dense
                            : 0.0;
                    const bool merge = ns <= 4 || (ns <= 16 && z < 0.8) ||
                                       (ns <= 48 && z < 0.1) || z < 0.05;
                    if (!merge) break;
                    cur.start = prev.start;
                    cur.width = static_cast<Idx>(ns);
                    cur.nnz = truenz;
                    out.pop_back();
                }
                out.push_back(cur);
            }
            sns = std::move(out);
        }
        // ---- sibling merge: adjacent leaf-like supernodes with the same
        // parent and the same number of rows below (typically identical
        // structure, e.g. the source rows of a transportation problem) become
        // one supernode: an explicit-zero diagonal block instead of one update
        // matrix each.
        {
            std::vector<Sn> out;
            for (const Sn &cur0 : sns) {
                Sn cur = cur0;
                if (!out.empty()) {
                    const Sn &prev = out.back();
                    const Idx lastp = prev.start + prev.width - 1;
                    const Idx lastc = cur.start + cur.width - 1;
                    const size_t ns =
                        static_cast<size_t>(prev.width) + cur.width;
                    if (prev.start + prev.width == cur.start &&
                        parent[lastp] >= 0 && parent[lastp] == parent[lastc] &&
                        cnt[lastp] == cnt[lastc] && ns <= 1024) {
                        const size_t below = static_cast<size_t>(cnt[lastc]);
                        const size_t dense = ns * below + ns * (ns + 1) / 2;
                        const size_t truenz = prev.nnz + cur.nnz;
                        const double z =
                            dense > truenz
                                ? static_cast<double>(dense - truenz) / dense
                                : 0.0;
                        if (z < 0.5) {
                            cur.start = prev.start;
                            cur.width = static_cast<Idx>(ns);
                            cur.nnz = truenz;
                            out.pop_back();
                        }
                    }
                }
                out.push_back(cur);
            }
            sns = std::move(out);
        }
        const size_t S = sns.size();
        sn_start_.resize(S + 1);
        for (size_t s = 0; s < S; ++s)
            sn_start_[s] = sns[s].start;
        sn_start_[S] = static_cast<Idx>(n);
        std::vector<Idx> sn_of(n);
        for (size_t s = 0; s < S; ++s)
            for (Idx c = sn_start_[s]; c < sn_start_[s + 1]; ++c)
                sn_of[c] = static_cast<Idx>(s);
        sn_parent_.assign(S, -1);
        for (size_t s = 0; s < S; ++s) {
            const Idx last = sn_start_[s + 1] - 1;
            if (parent[last] >= 0) sn_parent_[s] = sn_of[parent[last]];
        }
        children_.assign(S, {});
        for (size_t s = 0; s < S; ++s)
            if (sn_parent_[s] >= 0)
                children_[sn_parent_[s]].push_back(static_cast<Idx>(s));

        // ---- row structure below each supernode ----------------------------
        below_.assign(S, {});
        std::vector<Idx> mark(n, -1);
        std::vector<Idx> rows;
        loff_.assign(S + 1, 0);
        max_front_ = 0;
        for (size_t s = 0; s < S; ++s) {
            const Idx f = sn_start_[s], l = sn_start_[s + 1] - 1;
            rows.clear();
            for (Idx c = f; c <= l; ++c)
                for (Idx e = acp_[c]; e < acp_[c + 1]; ++e) {
                    const Idx r = ari_[e];
                    if (r > l && mark[r] != static_cast<Idx>(s)) {
                        mark[r] = static_cast<Idx>(s);
                        rows.push_back(r);
                    }
                }
            for (Idx t : children_[s])
                for (Idx r : below_[t])
                    if (r > l && mark[r] != static_cast<Idx>(s)) {
                        mark[r] = static_cast<Idx>(s);
                        rows.push_back(r);
                    }
            std::sort(rows.begin(), rows.end());
            below_[s] = rows;
            const size_t ns = static_cast<size_t>(l - f + 1);
            const size_t fs = ns + rows.size();
            loff_[s + 1] = loff_[s] + fs * ns;
            max_front_ = std::max(max_front_, fs);
        }
        lvals_.assign(loff_[S], T(0));
        d_.assign(n, T(0));
        plan_tasks();
        analyzed_ = true;
        factored_ = false;
    }

    // Numeric factorization from the values of the ORIGINAL matrix (same CSR
    // pattern as analyze()). Returns false on a failed pivot.
    //
    // Independent small subtrees of the assembly tree are factored in parallel
    // (one OpenMP thread each, private front buffers); the remaining top of the
    // tree is processed in order with the dense updates and assembly loops
    // parallelized inside each large front.
    bool
    factorize(const T *values, Kind kind, const signed char *signs, T eps)
    {
        if (!analyzed_)
            throw std::runtime_error("MultifrontalLdl: not analyzed");
        const size_t S = sn_start_.size() - 1;
        factored_ = false;
        failed_ = -1;
        n_reg_ = 0;
        prof_ = {};
        std::vector<Block> U(
            S); // uninitialized update matrices (pooled buffers)
        std::atomic<long> fail_col(-1);
        std::atomic<size_t> nreg(0);
        std::mutex prof_mu;

        struct Buffers {
            std::vector<T> F, work;
            std::vector<Idx> relpos;
            std::vector<signed char> sgn;
            Profile prof;
        };
        auto make_buffers = [&](size_t max_fs) {
            Buffers b;
            b.F.resize(max_fs * max_fs);
            b.work.resize(max_fs * static_cast<size_t>(dense::kPanel) + 8);
            b.relpos.resize(n_);
            return b;
        };
        auto check = [&](int, T piv) {
            if (!std::isfinite(piv)) return false;
            return kind == Kind::SPD ? piv > T(0) : piv != T(0);
        };
        // Factors supernode s; false on a failed pivot.
        auto process = [&](size_t s, Buffers &bf, bool par) {
            using clock = std::chrono::steady_clock;
            auto ms = [](clock::time_point a) {
                return std::chrono::duration<double, std::milli>(
                    clock::now() - a)
                    .count();
            };
            auto t0 = clock::now();
            const Idx f = sn_start_[s];
            const int ns = static_cast<int>(sn_start_[s + 1] - f);
            const std::vector<Idx> &bl = below_[s];
            const int nb = static_cast<int>(bl.size());
            const int fs = ns + nb;
            Idx *relpos = bf.relpos.data();
            for (int c = 0; c < ns; ++c)
                relpos[f + c] = c;
            for (int i = 0; i < nb; ++i)
                relpos[bl[i]] = ns + i;
            T *F = bf.F.data();
            const bool big = par && fs >= 256;
            if (big) {
#pragma omp parallel for schedule(static)
                for (int c = 0; c < fs; ++c)
                    std::memset(
                        F + static_cast<size_t>(c) * fs, 0, fs * sizeof(T));
            } else {
                std::memset(F, 0, static_cast<size_t>(fs) * fs * sizeof(T));
            }
            for (int c = 0; c < ns; ++c) { // original entries
                T *col = F + static_cast<size_t>(c) * fs;
                for (Idx e = acp_[f + c]; e < acp_[f + c + 1]; ++e)
                    col[relpos[ari_[e]]] += values[asrc_[e]];
            }
            bf.prof.zero_assemble += ms(t0);
            t0 = clock::now();
            for (Idx t : children_[s]) { // children's update matrices
                Block &Ut = U[t];
                const std::vector<Idx> &bt = below_[t];
                const int m = static_cast<int>(bt.size());
                auto add_col = [&](int jj) {
                    const int lj = relpos[bt[jj]];
                    T *col = F + static_cast<size_t>(lj) * fs;
                    const T *uc = Ut.p + static_cast<size_t>(jj) * m;
                    for (int ii = jj; ii < m; ++ii)
                        col[relpos[bt[ii]]] += uc[ii];
                };
                if (par && m >= 256) {
#pragma omp parallel for schedule(dynamic, 16)
                    for (int jj = 0; jj < m; ++jj)
                        add_col(jj);
                } else {
                    for (int jj = 0; jj < m; ++jj)
                        add_col(jj);
                }
                pool_.release(Ut);
            }
            bf.prof.extend_add += ms(t0);
            t0 = clock::now();
            const signed char *fsign = nullptr;
            if (signs) {
                bf.sgn.resize(ns);
                for (int c = 0; c < ns; ++c)
                    bf.sgn[c] = signs[perm_[f + c]];
                fsign = bf.sgn.data();
            }
            int fail = -1;
            nreg += dense::partial_ldlt<T>(fs, ns, F, fs, d_.data() + f, fsign,
                eps, check, &fail, bf.work.data(), par);
            bf.prof.dense += ms(t0);
            t0 = clock::now();
            if (fail >= 0) {
                long expected = -1;
                fail_col.compare_exchange_strong(
                    expected, static_cast<long>(f) + fail);
                return false;
            }
            T *L = lvals_.data() + loff_[s];
            auto copy_l = [&](int c) {
                std::memcpy(L + static_cast<size_t>(c) * fs,
                    F + static_cast<size_t>(c) * fs, fs * sizeof(T));
            };
            if (nb > 0) U[s] = pool_.acquire(static_cast<size_t>(nb) * nb);
            auto copy_u = [&](int jj) {
                std::memcpy(U[s].p + static_cast<size_t>(jj) * nb + jj,
                    F + static_cast<size_t>(ns + jj) * fs + ns + jj,
                    (nb - jj) * sizeof(T));
            };
            if (big) {
#pragma omp parallel for schedule(static)
                for (int c = 0; c < ns; ++c)
                    copy_l(c);
#pragma omp parallel for schedule(dynamic, 16)
                for (int jj = 0; jj < nb; ++jj)
                    copy_u(jj);
            } else {
                for (int c = 0; c < ns; ++c)
                    copy_l(c);
                for (int jj = 0; jj < nb; ++jj)
                    copy_u(jj);
            }
            bf.prof.store += ms(t0);
            return true;
        };
        auto add_prof = [&](const Profile &p) {
            std::lock_guard<std::mutex> lk(prof_mu);
            prof_.zero_assemble += p.zero_assemble;
            prof_.extend_add += p.extend_add;
            prof_.dense += p.dense;
            prof_.store += p.store;
        };

        auto wall0 = std::chrono::steady_clock::now();
        auto wall_ms = [&] {
            return std::chrono::duration<double, std::milli>(
                std::chrono::steady_clock::now() - wall0)
                .count();
        };
        // ---- parallel phase: independent small subtrees ------------------
        if (!tasks_.empty()) {
#pragma omp parallel
            {
                Buffers bf = make_buffers(task_max_front_);
#pragma omp for schedule(dynamic, 1) nowait
                for (long ti = 0; ti < static_cast<long>(tasks_.size()); ++ti) {
                    if (fail_col.load() >= 0) continue;
                    const Idx root = tasks_[ti];
                    for (Idx s = first_desc_[root]; s <= root; ++s)
                        if (!process(static_cast<size_t>(s), bf, false)) break;
                }
                add_prof(bf.prof);
            }
        }
        prof_.wall_tasks = wall_ms();
        prof_.ntasks = tasks_.size();
        prof_.ntop = top_nodes_.size();
        wall0 = std::chrono::steady_clock::now();
        // ---- serial phase: the top of the tree ---------------------------
        if (fail_col.load() < 0) {
            Buffers bf = make_buffers(max_front_top_);
            for (Idx s : top_nodes_)
                if (!process(static_cast<size_t>(s), bf, true)) break;
            add_prof(bf.prof);
        }
        prof_.wall_top = wall_ms();
        n_reg_ = nreg.load();
        if (fail_col.load() >= 0) {
            failed_ = fail_col.load();
            return false;
        }
        factored_ = true;
        return true;
    }

    // Milliseconds spent per phase in the last factorize() (diagnostics).
    struct Profile {
        double zero_assemble = 0, extend_add = 0, dense = 0, store = 0,
               wall_tasks = 0, wall_top = 0;
        size_t ntasks = 0, ntop = 0;
    };
    const Profile &
    profile() const
    { return prof_; }

    // x = A^{-1} b (b and x are in the original ordering; may not alias).
    void
    solve(const T *b, T *x) const
    {
        const size_t S = sn_start_.size() - 1;
        std::vector<T> y(n_);
        for (size_t k = 0; k < n_; ++k)
            y[k] = b[perm_[k]];
        for (size_t s = 0; s < S; ++s) { // L y = b
            const Idx f = sn_start_[s];
            const int ns = static_cast<int>(sn_start_[s + 1] - f);
            const std::vector<Idx> &bl = below_[s];
            const int nb = static_cast<int>(bl.size());
            const int fs = ns + nb;
            const T *L = lvals_.data() + loff_[s];
            for (int c = 0; c < ns; ++c) {
                const T xc = y[f + c];
                const T *col = L + static_cast<size_t>(c) * fs;
                for (int i = c + 1; i < ns; ++i)
                    y[f + i] -= col[i] * xc;
                for (int i = 0; i < nb; ++i)
                    y[bl[i]] -= col[ns + i] * xc;
            }
        }
        for (size_t k = 0; k < n_; ++k)
            y[k] /= d_[k];
        for (size_t s = S; s-- > 0;) { // L^T x = y
            const Idx f = sn_start_[s];
            const int ns = static_cast<int>(sn_start_[s + 1] - f);
            const std::vector<Idx> &bl = below_[s];
            const int nb = static_cast<int>(bl.size());
            const int fs = ns + nb;
            const T *L = lvals_.data() + loff_[s];
            for (int c = ns - 1; c >= 0; --c) {
                const T *col = L + static_cast<size_t>(c) * fs;
                T sum = 0;
                for (int i = c + 1; i < ns; ++i)
                    sum += col[i] * y[f + i];
                for (int i = 0; i < nb; ++i)
                    sum += col[ns + i] * y[bl[i]];
                y[f + c] -= sum;
            }
        }
        for (size_t k = 0; k < n_; ++k)
            x[perm_[k]] = y[k];
    }

    size_t
    factor_nnz() const
    { return factor_nnz_; }
    size_t
    stored_entries() const
    { return lvals_.size(); }
    size_t
    supernodes() const
    { return sn_start_.empty() ? 0 : sn_start_.size() - 1; }
    long
    failed_pivot() const
    { return failed_; }
    size_t
    regularized_pivots() const
    { return n_reg_; }
    const std::vector<T> &
    pivots() const
    { return d_; }
    size_t
    max_front() const
    { return max_front_; }
    bool
    factored() const
    { return factored_; }

  private:
    // Recycles the update-matrix buffers between supernodes and factorizations
    // (fresh allocations of large blocks are dominated by page faults).
    struct Block {
        T *p = nullptr;
        size_t cap = 0;
    };
    // Buffers are kept ordered by capacity, so the best fit (smallest buffer >=
    // n, taken only if at most 2n + 1024) is a lookup, not a scan: a
    // factorization can hold tens of thousands of update matrices (many small
    // fronts under one root), and a linear scan per acquire made every
    // factorization after the first quadratic in their number.
    struct Pool {
        std::mutex mu;
        std::multimap<size_t, T *> free_; // capacity -> buffer
        size_t held = 0;
        Pool() = default;
        Pool(const Pool &) {}
        Pool &
        operator=(const Pool &)
        { return *this; }
        ~Pool()
        {
            for (auto &e : free_)
                std::free(e.second);
        }
        Block
        acquire(size_t n)
        {
            {
                std::lock_guard<std::mutex> lk(mu);
                auto it = free_.lower_bound(n);
                if (it != free_.end() && it->first <= 2 * n + 1024) {
                    Block b;
                    b.cap = it->first;
                    b.p = it->second;
                    free_.erase(it);
                    held -= b.cap;
                    return b;
                }
            }
            Block b;
            b.cap = n;
            b.p = static_cast<T *>(std::malloc(n * sizeof(T)));
            if (!b.p) throw std::bad_alloc();
            return b;
        }
        void
        release(Block &b)
        {
            if (!b.p) return;
            std::lock_guard<std::mutex> lk(mu);
            constexpr size_t kMaxHeld = size_t(1)
                                        << 28; // elements (2 GB of doubles)
            if (held + b.cap > kMaxHeld)
                std::free(b.p);
            else {
                free_.emplace(b.cap, b.p);
                held += b.cap;
            }
            b = Block();
        }
    };
    Pool pool_;
    // Splits the assembly tree into independent subtrees (processed in
    // parallel) and the remaining top nodes (processed serially). A subtree
    // is a task when its estimated work is a small fraction of the total.
    void
    plan_tasks()
    {
        const size_t S = sn_start_.size() - 1;
        first_desc_.assign(S, 0);
        std::vector<double> sub(S, 0.0);
        for (size_t s = 0; s < S; ++s) {
            const double ns =
                static_cast<double>(sn_start_[s + 1] - sn_start_[s]);
            const double fs = ns + static_cast<double>(below_[s].size());
            sub[s] +=
                ns * fs * fs + 100.0 * fs; // flops-ish plus per-node overhead
            Idx fd = static_cast<Idx>(s);
            for (Idx c : children_[s])
                fd = std::min(fd, first_desc_[c]);
            first_desc_[s] = fd;
            if (sn_parent_[s] >= 0) sub[sn_parent_[s]] += sub[s];
        }
        double total = 0;
        for (size_t s = 0; s < S; ++s)
            if (sn_parent_[s] < 0) total += sub[s];
        int P = 1;
#ifdef _OPENMP
        P = omp_get_max_threads();
#endif
        tasks_.clear();
        top_nodes_.clear();
        task_max_front_ = 1;
        max_front_top_ = 1;
        if (P <= 1 || S < 64) {
            for (size_t s = 0; s < S; ++s)
                top_nodes_.push_back(static_cast<Idx>(s));
            max_front_top_ = max_front_;
            return;
        }
        const double thresh = total / (4.0 * P);
        std::vector<char> in_task(S, 0);
        // descend from the roots: a node whose subtree is small is a task
        std::vector<Idx> stack;
        for (size_t s = S; s-- > 0;)
            if (sn_parent_[s] < 0) stack.push_back(static_cast<Idx>(s));
        while (!stack.empty()) {
            Idx s = stack.back();
            stack.pop_back();
            if (sub[s] <= thresh) {
                tasks_.push_back(s);
                for (Idx t = first_desc_[s]; t <= s; ++t) {
                    in_task[t] = 1;
                    const size_t fs =
                        static_cast<size_t>(sn_start_[t + 1] - sn_start_[t]) +
                        below_[t].size();
                    task_max_front_ = std::max(task_max_front_, fs);
                }
            } else {
                for (Idx c : children_[s])
                    stack.push_back(c);
            }
        }
        for (size_t s = 0; s < S; ++s)
            if (!in_task[s]) {
                top_nodes_.push_back(static_cast<Idx>(s));
                const size_t fs =
                    static_cast<size_t>(sn_start_[s + 1] - sn_start_[s]) +
                    below_[s].size();
                max_front_top_ = std::max(max_front_top_, fs);
            }
    }

    // Elimination tree and strictly-below column counts of the matrix
    // ordered by perm (up-looking symbolic factorization).
    static void
    etree_counts(size_t n, const Idx *rp, const Idx *ci,
        const std::vector<Idx> &perm, const std::vector<Idx> &pinv,
        std::vector<Idx> &parent, std::vector<Idx> &cnt)
    {
        parent.assign(n, -1);
        cnt.assign(n, 0);
        std::vector<Idx> flag(n, -1);
        for (size_t k = 0; k < n; ++k) {
            flag[k] = static_cast<Idx>(k);
            const Idx kk = perm[k];
            for (Idx p = rp[kk]; p < rp[kk + 1]; ++p) {
                Idx i = pinv[ci[p]];
                if (static_cast<size_t>(i) < k) {
                    for (; flag[i] != static_cast<Idx>(k); i = parent[i]) {
                        if (parent[i] == -1) parent[i] = static_cast<Idx>(k);
                        cnt[i]++;
                        flag[i] = static_cast<Idx>(k);
                    }
                }
            }
        }
    }

    // Postorder of a forest given by parent[]: children before parents,
    // subtrees contiguous. Returns post with post[k] = old index of the k-th.
    static std::vector<Idx>
    postorder(const std::vector<Idx> &parent)
    {
        const size_t n = parent.size();
        std::vector<Idx> first(n, -1), next(n, -1), post;
        post.reserve(n);
        for (size_t j = n; j-- > 0;) {
            if (parent[j] >= 0) {
                next[j] = first[parent[j]];
                first[parent[j]] = static_cast<Idx>(j);
            }
        }
        std::vector<Idx> stack;
        for (size_t r = 0; r < n; ++r) {
            if (parent[r] >= 0) continue;
            stack.push_back(static_cast<Idx>(r));
            while (!stack.empty()) {
                Idx v = stack.back();
                if (first[v] >= 0) {
                    Idx c = first[v];
                    first[v] = next[c];
                    stack.push_back(c);
                } else {
                    post.push_back(v);
                    stack.pop_back();
                }
            }
        }
        return post;
    }

    Profile prof_;
    size_t n_ = 0, factor_nnz_ = 0, max_front_ = 0, n_reg_ = 0;
    bool analyzed_ = false, factored_ = false;
    long failed_ = -1;
    std::vector<Idx> perm_, pinv_;
    std::vector<Idx> acp_, ari_, asrc_; // permuted lower pattern
    std::vector<Idx> sn_start_, sn_parent_;
    std::vector<std::vector<Idx>> children_, below_;
    std::vector<size_t> loff_;
    std::vector<T> lvals_, d_;
    std::vector<Idx> first_desc_, tasks_, top_nodes_;
    size_t task_max_front_ = 1, max_front_top_ = 1;
};

} // namespace Sparse
} // namespace AXOS
