// SPDX-License-Identifier: BSD-3-Clause
//
// Sparse symmetric direct solver: A = P^T L D L^T P.
//
//   analyze(A)    ordering + symbolic factorization (once per sparsity pattern)
//   factorize(A)  numeric factorization (repeat as the values change)
//   solve(b, x)   solves A x = b
//
// A must be square and symmetric with full storage (both triangles); it is
// read through row k's entries with permuted column index <= k, so only one
// triangle's values matter. No numerical pivoting is done: Symmetry::SPD
// requires all pivots D > 0 (Cholesky-equivalent); Symmetry::Symmetric only
// requires nonzero pivots, which suffices for quasi-definite systems such as
// interior-point KKT matrices.
//
// This header is the CPU implementation (up-looking LDL^T after T. Davis's
// LDL, with a minimum-degree ordering). The CUDA specialization on top of
// cuDSS is in cudss_solver.h.
#pragma once

#include "sparse/amd_order.h"
#include "sparse/csr.h"
#include "sparse/multifrontal_ldl.h"
#include <algorithm>
#include <cmath>
#include <set>

namespace AXOS {
namespace Sparse {

enum class Symmetry { SPD, Symmetric };

// Fill-reducing orderings. MinDegree is approximate minimum degree (AMD);
// ExactMinDegree is a slow reference implementation. The cuDSS backend
// picks its own default ordering unless told otherwise: it maps MinDegree to
// its AMD, NestedDissection to its nested dissection, and Natural /
// ExactMinDegree to its default. On the CPU, NestedDissection falls back to
// MinDegree.
enum class Ordering { Natural, MinDegree, ExactMinDegree, NestedDissection };

// CPU factorization algorithm. Simplicial is the scalar up-looking LDL^T;
// Supernodal is the multifrontal code (dense kernels, much faster once the
// factor has a few hundred thousand entries); Auto picks by factor size.
enum class Factorization { Auto, Simplicial, Supernodal };

template <typename T, typename Idx = int32_t,
    template <typename> class Store = Cpu::HostStorage>
class SparseLdlt;

namespace detail {

// Exact minimum-degree ordering on the graph of A (both triangles). This is
// a straightforward elimination-graph implementation: fine for the CPU
// fallback, but its cost grows with the largest cliques it forms.
template <typename Idx>
std::vector<Idx>
min_degree_order(size_t n, const Idx *rp, const Idx *ci)
{
    std::vector<std::vector<Idx>> adj(n);
    for (size_t i = 0; i < n; ++i)
        for (Idx k = rp[i]; k < rp[i + 1]; ++k) {
            Idx j = ci[k];
            if (static_cast<size_t>(j) != i) {
                adj[i].push_back(j);
                adj[j].push_back(static_cast<Idx>(i));
            }
        }
    for (auto &a : adj) {
        std::sort(a.begin(), a.end());
        a.erase(std::unique(a.begin(), a.end()), a.end());
    }
    std::set<std::pair<size_t, Idx>> pq;
    for (size_t v = 0; v < n; ++v)
        pq.insert({adj[v].size(), static_cast<Idx>(v)});
    std::vector<Idx> perm;
    perm.reserve(n);
    std::vector<Idx> merged;
    while (!pq.empty()) {
        Idx v = pq.begin()->second;
        pq.erase(pq.begin());
        perm.push_back(v);
        std::vector<Idx> nb = std::move(adj[v]);
        adj[v].clear();
        for (Idx u : nb) {
            pq.erase({adj[u].size(), u});
            // adj[u] = (adj[u] union nb) minus {u, v}
            merged.clear();
            std::set_union(adj[u].begin(), adj[u].end(), nb.begin(), nb.end(),
                std::back_inserter(merged));
            merged.erase(
                std::remove_if(merged.begin(), merged.end(),
                    [&](Idx w) { return w == u || w == v; }),
                merged.end());
            adj[u] = merged;
            pq.insert({adj[u].size(), u});
        }
    }
    return perm;
}

} // namespace detail

template <typename T, typename Idx> class SparseLdlt<T, Idx, Cpu::HostStorage> {
    static_assert(std::is_floating_point_v<T>,
        "SparseLdlt supports float and double");

  public:
    using matrix_type = Csr<T, Idx, Cpu::HostStorage>;
    using vector_type = tensorET<1, T, Cpu::HostStorage<T>>;

    explicit SparseLdlt(Symmetry kind = Symmetry::SPD,
        Ordering ord = Ordering::MinDegree,
        Factorization fact = Factorization::Auto)
        : kind_(kind), ord_(ord), fact_(fact)
    {
    }

    // Ordering and symbolic factorization for A's sparsity pattern.
    void
    analyze(const matrix_type &A)
    {
        if (A.rows() != A.cols())
            throw std::invalid_argument("SparseLdlt: matrix must be square");
        n_ = A.rows();
        nnz_ = A.nnz();
        const Idx *rp = A.row_ptr(), *ci = A.col_ind();

        perm_.resize(n_);
        if ((ord_ == Ordering::MinDegree || ord_ == Ordering::NestedDissection) && n_ > 0)
            perm_ = amd_order<Idx>(n_, rp, ci);
        else if (ord_ == Ordering::ExactMinDegree && n_ > 0)
            perm_ = detail::min_degree_order<Idx>(n_, rp, ci);
        else
            std::iota(perm_.begin(), perm_.end(), Idx(0));
        pinv_.assign(n_, 0);
        for (size_t k = 0; k < n_; ++k)
            pinv_[perm_[k]] = static_cast<Idx>(k);

        parent_.assign(n_, -1);
        lnz_.assign(n_, 0);
        flag_.assign(n_, 0);
        for (size_t k = 0; k < n_; ++k) {
            parent_[k] = -1;
            flag_[k] = static_cast<Idx>(k);
            lnz_[k] = 0;
            const Idx kk = perm_[k];
            for (Idx p = rp[kk]; p < rp[kk + 1]; ++p) {
                Idx i = pinv_[ci[p]];
                if (static_cast<size_t>(i) < k) {
                    for (; flag_[i] != static_cast<Idx>(k); i = parent_[i]) {
                        if (parent_[i] == -1) parent_[i] = static_cast<Idx>(k);
                        lnz_[i]++;
                        flag_[i] = static_cast<Idx>(k);
                    }
                }
            }
        }
        size_t total = 0;
        double flops = 0;
        for (size_t k = 0; k < n_; ++k) {
            total += static_cast<size_t>(lnz_[k]);
            flops += static_cast<double>(lnz_[k]) * static_cast<double>(lnz_[k]);
        }
        factor_flops_ = flops;
        use_mf_ = fact_ == Factorization::Supernodal ||
                  (fact_ == Factorization::Auto && total > 200000);
        if (use_mf_) {
            mf_ = MultifrontalLdl<T, Idx>();
            mf_.analyze(n_, rp, ci, perm_);
            lp_.assign(n_ + 1, 0); // simplicial storage is not used
            li_.clear();
            lx_.clear();
            d_.assign(n_, T(0));
            factor_nnz_ = total;
        } else {
            lp_.assign(n_ + 1, 0);
            for (size_t k = 0; k < n_; ++k)
                lp_[k + 1] = lp_[k] + lnz_[k];
            li_.assign(lp_[n_], 0);
            lx_.assign(lp_[n_], T(0));
            d_.assign(n_, T(0));
            factor_nnz_ = total;
        }
        analyzed_ = true;
        factored_ = false;
    }

    // Numeric factorization. A must have the pattern given to analyze().
    // Returns false when a pivot fails (SPD: D <= 0, Symmetric: D == 0);
    // failed_pivot() then gives the (permuted) index.
    bool
    factorize(const matrix_type &A)
    {
        if (!analyzed_) analyze(A);
        if (A.rows() != n_ || A.nnz() != nnz_)
            throw std::invalid_argument(
                "SparseLdlt::factorize: pattern differs from analyze()");
        if (use_mf_) {
            n_reg_ = 0;
            factored_ = mf_.factorize(A.values(),
                kind_ == Symmetry::SPD ? MultifrontalLdl<T, Idx>::Kind::SPD
                                       : MultifrontalLdl<T, Idx>::Kind::Symmetric,
                signs_.empty() ? nullptr : signs_.data(), pivot_eps_);
            n_reg_ = mf_.regularized_pivots();
            failed_ = factored_ ? -1 : mf_.failed_pivot();
            return factored_;
        }
        const Idx *rp = A.row_ptr(), *ci = A.col_ind();
        const T *ax = A.values();
        std::vector<T> y(n_, T(0));
        std::vector<Idx> pattern(n_), flag(n_), lnz(n_, 0);
        failed_ = -1;
        factored_ = false;
        n_reg_ = 0;
        for (size_t k = 0; k < n_; ++k) {
            y[k] = T(0);
            size_t top = n_;
            flag[k] = static_cast<Idx>(k);
            lnz[k] = 0;
            const Idx kk = perm_[k];
            for (Idx p = rp[kk]; p < rp[kk + 1]; ++p) {
                Idx i = pinv_[ci[p]];
                if (static_cast<size_t>(i) <= k) {
                    y[i] += ax[p];
                    size_t len = 0;
                    for (; flag[i] != static_cast<Idx>(k); i = parent_[i]) {
                        pattern[len++] = i;
                        flag[i] = static_cast<Idx>(k);
                    }
                    while (len > 0)
                        pattern[--top] = pattern[--len];
                }
            }
            d_[k] = y[k];
            y[k] = T(0);
            for (; top < n_; ++top) {
                const Idx i = pattern[top];
                const T yi = y[i];
                y[i] = T(0);
                const Idx p2 = lp_[i] + lnz[i];
                for (Idx p = lp_[i]; p < p2; ++p)
                    y[li_[p]] -= lx_[p] * yi;
                const T lki = yi / d_[i];
                d_[k] -= lki * yi;
                li_[p2] = static_cast<Idx>(k);
                lx_[p2] = lki;
                lnz[i]++;
            }
            if (!signs_.empty() && signs_[kk] != 0) {
                const T sg = static_cast<T>(signs_[kk]);
                if (!(sg * d_[k] >= pivot_eps_)) { d_[k] = sg * pivot_eps_; ++n_reg_; }
            }
            const bool bad = (kind_ == Symmetry::SPD) ? !(d_[k] > T(0))
                                                        : (d_[k] == T(0));
            if (bad || !std::isfinite(d_[k])) {
                failed_ = static_cast<long>(k);
                return false;
            }
        }
        factored_ = true;
        return true;
    }

    // Solves A x = b (host tensors).
    void
    solve(const vector_type &b, vector_type &x) const
    {
        if (!factored_)
            throw std::runtime_error("SparseLdlt::solve: not factorized");
        if (b.size() != n_ || x.size() != n_)
            throw std::invalid_argument("SparseLdlt::solve: size mismatch");
        if (use_mf_) {
            mf_.solve(b.data, x.data); // b and x must not alias
            return;
        }
        std::vector<T> w(n_);
        for (size_t k = 0; k < n_; ++k)
            w[k] = b.data[perm_[k]];
        for (size_t j = 0; j < n_; ++j) // L w = w
            for (Idx p = lp_[j]; p < lp_[j + 1]; ++p)
                w[li_[p]] -= lx_[p] * w[j];
        for (size_t j = 0; j < n_; ++j) // D w = w
            w[j] /= d_[j];
        for (size_t j = n_; j-- > 0;) // L^T w = w
            for (Idx p = lp_[j]; p < lp_[j + 1]; ++p)
                w[j] -= lx_[p] * w[li_[p]];
        for (size_t k = 0; k < n_; ++k)
            x.data[perm_[k]] = w[k];
    }

    // Dynamic pivot regularization (Symmetric kind): entry i of `signs` is the
    // expected sign (+1 / -1, 0 = none) of the pivot of variable i. A pivot
    // whose value times its sign is below `eps` is replaced by sign * eps, so
    // only pivots that really need it are perturbed. Returns via
    // regularized_pivots() how many were.
    void
    set_pivot_regularization(std::vector<signed char> signs, T eps)
    {
        signs_ = std::move(signs);
        pivot_eps_ = eps;
    }
    size_t regularized_pivots() const { return n_reg_; }

    size_t factor_nnz() const { return analyzed_ ? factor_nnz_ : 0; }
    // approximate flop count of one numeric factorization (sum of squared column counts)
    double factor_flops() const { return analyzed_ ? factor_flops_ : 0.0; }
    bool supernodal() const { return use_mf_; }
    void
    print_profile() const
    {
        if (!use_mf_) return;
        const auto &p = mf_.profile();
        std::printf("[ldl]        zero+assemble %.1f  extend-add %.1f  dense %.1f  store %.1f ms (max front %zu, %zu supernodes)\n",
            p.zero_assemble, p.extend_add, p.dense, p.store, mf_.max_front(), mf_.supernodes());
        std::printf("[ldl]        wall: %zu parallel subtrees %.1f ms, %zu top nodes %.1f ms\n",
            p.ntasks, p.wall_tasks, p.ntop, p.wall_top);
    }
    long failed_pivot() const { return failed_; }

    // Counts of positive / negative / zero pivots after factorize().
    void
    inertia(size_t &pos, size_t &neg, size_t &zero) const
    {
        pos = neg = zero = 0;
        const std::vector<T> &d = use_mf_ ? mf_.pivots() : d_;
        for (size_t k = 0; k < n_; ++k) {
            if (d[k] > T(0)) ++pos;
            else if (d[k] < T(0)) ++neg;
            else ++zero;
        }
    }

  private:
    Symmetry kind_;
    Ordering ord_;
    Factorization fact_;
    bool use_mf_ = false;
    MultifrontalLdl<T, Idx> mf_;
    size_t factor_nnz_ = 0;
    double factor_flops_ = 0;
    size_t n_ = 0, nnz_ = 0;
    bool analyzed_ = false, factored_ = false;
    long failed_ = -1;
    std::vector<signed char> signs_;
    T pivot_eps_ = 0;
    size_t n_reg_ = 0;
    std::vector<Idx> perm_, pinv_, parent_, lnz_, flag_, lp_, li_;
    std::vector<T> lx_, d_;
};

} // namespace Sparse
} // namespace AXOS
