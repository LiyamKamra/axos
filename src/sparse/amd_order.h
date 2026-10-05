// SPDX-License-Identifier: BSD-3-Clause
//
// Approximate minimum degree ordering on the graph of a symmetric sparse
// matrix, written from the published algorithm (Amestoy, Davis, Duff,
// "An approximate minimum degree ordering algorithm", SIMAX 1996):
//
//   * quotient graph: every variable keeps a list of adjacent variables (A_i)
//     and adjacent elements (E_i); an eliminated pivot becomes an element
//     whose variable list L_e replaces the clique it would create
//   * approximate external degree with the |L_e \ L_p| trick
//   * element absorption (aggressive) and mass elimination of variables that
//     become indistinguishable from the pivot
//   * dense variables (degree > 10 sqrt(n)) are ordered last
//
// Supervariable detection is not implemented, so matrices with many
// identical columns order somewhat slower than a full AMD.
#pragma once

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <vector>

namespace AXOS {
namespace Sparse {

// Graph of the symmetric pattern of the n x n matrix given by CSR (rp, ci);
// both triangles need not be present (the pattern is symmetrized). Returns
// perm with perm[k] = the k-th eliminated variable.
template <typename Idx>
std::vector<Idx>
amd_order(size_t n, const Idx *rp, const Idx *ci)
{
    std::vector<Idx> perm;
    perm.reserve(n);
    if (n == 0) return perm;

    // ---- build symmetric adjacency (no self loops, no duplicates) --------
    std::vector<std::vector<Idx>> A(n), E(n), L(n);
    for (size_t i = 0; i < n; ++i)
        for (Idx k = rp[i]; k < rp[i + 1]; ++k) {
            Idx j = ci[k];
            if (static_cast<size_t>(j) != i) {
                A[i].push_back(j);
                A[j].push_back(static_cast<Idx>(i));
            }
        }
    for (auto &a : A) {
        std::sort(a.begin(), a.end());
        a.erase(std::unique(a.begin(), a.end()), a.end());
    }

    enum : char { Live = 0, Element = 1, Dead = 2, Dense = 3 };
    std::vector<char> state(n, Live);
    std::vector<Idx> deg(n);
    const size_t dense_thresh = std::max<size_t>(
        16, static_cast<size_t>(10.0 * std::sqrt(static_cast<double>(n))));
    std::vector<Idx> dense_vars;
    size_t nlive = 0;
    for (size_t i = 0; i < n; ++i) {
        if (A[i].size() > dense_thresh) {
            state[i] = Dense;
            dense_vars.push_back(static_cast<Idx>(i));
        } else {
            deg[i] = static_cast<Idx>(A[i].size());
            ++nlive;
        }
    }
    // Dense variables no longer count as neighbours of the rest.
    if (!dense_vars.empty()) {
        for (size_t i = 0; i < n; ++i) {
            if (state[i] == Dense) { A[i].clear(); continue; }
            auto &a = A[i];
            a.erase(std::remove_if(a.begin(), a.end(),
                        [&](Idx j) { return state[j] == Dense; }),
                a.end());
            deg[i] = static_cast<Idx>(a.size());
        }
    }

    // ---- degree buckets ------------------------------------------------
    std::vector<Idx> head(n + 1, -1), nxt(n, -1), prv(n, -1);
    auto bucket_insert = [&](Idx i) {
        Idx d = deg[i];
        nxt[i] = head[d];
        prv[i] = -1;
        if (head[d] >= 0) prv[head[d]] = i;
        head[d] = i;
    };
    auto bucket_remove = [&](Idx i) {
        Idx d = deg[i];
        if (prv[i] >= 0) nxt[prv[i]] = nxt[i];
        else head[d] = nxt[i];
        if (nxt[i] >= 0) prv[nxt[i]] = prv[i];
    };
    for (size_t i = 0; i < n; ++i)
        if (state[i] == Live) bucket_insert(static_cast<Idx>(i));
    size_t mindeg = 0;

    std::vector<Idx> mark(n, -1);      // membership in Lp
    std::vector<long> w(n, 0);         // |L_e \ L_p| workspace (per element)
    std::vector<long> wstamp(n, -1);
    std::vector<Idx> lp;               // L_p being built
    size_t remaining = nlive;

    while (nlive > 0) {
        while (mindeg <= n && head[mindeg] < 0) ++mindeg;
        if (mindeg > n) break;
        const Idx p = head[mindeg];
        bucket_remove(p);
        perm.push_back(p);
        --nlive;
        mark[p] = p; // p itself is never a member of L_p

        // ---- form L_p = (A_p U union of L_e, e in E_p) minus p ---------
        lp.clear();
        for (Idx j : A[p])
            if (state[j] == Live && mark[j] != p) { mark[j] = p; lp.push_back(j); }
        for (Idx e : E[p]) {
            if (state[e] != Element) continue;
            for (Idx j : L[e])
                if (state[j] == Live && mark[j] != p) {
                    mark[j] = p;
                    lp.push_back(j);
                }
            state[e] = Dead; // absorbed into p
            L[e].clear();
            L[e].shrink_to_fit();
        }
        A[p].clear();
        E[p].clear();
        A[p].shrink_to_fit();

        // ---- |L_e \ L_p| for elements touching L_p --------------------
        const long stamp = static_cast<long>(perm.size());
        for (Idx i : lp)
            for (Idx e : E[i]) {
                if (state[e] != Element) continue;
                if (wstamp[e] != stamp) {
                    wstamp[e] = stamp;
                    w[e] = static_cast<long>(L[e].size());
                }
                --w[e];
            }

        // ---- update every variable of L_p ------------------------------
        std::vector<Idx> mass; // variables eliminated together with p
        size_t lp_live = lp.size();
        for (Idx i : lp) {
            bucket_remove(i);
            // prune E_i: drop absorbed elements, aggressive absorption
            auto &Ei = E[i];
            size_t out = 0;
            long ext = 0; // sum of |L_e \ L_p| over surviving elements
            for (size_t k = 0; k < Ei.size(); ++k) {
                Idx e = Ei[k];
                if (state[e] != Element) continue;
                if (wstamp[e] == stamp && w[e] == 0) { // L_e subset of L_p
                    state[e] = Dead;
                    L[e].clear();
                    L[e].shrink_to_fit();
                    continue;
                }
                ext += (wstamp[e] == stamp) ? w[e] : static_cast<long>(L[e].size());
                Ei[out++] = e;
            }
            Ei.resize(out);
            // prune A_i: variables in L_p (or p) are covered by element p
            auto &Ai = A[i];
            out = 0;
            for (size_t k = 0; k < Ai.size(); ++k) {
                Idx j = Ai[k];
                if (state[j] != Live || mark[j] == p) continue;
                Ai[out++] = j;
            }
            Ai.resize(out);
            Ei.push_back(p);

            if (Ai.empty() && Ei.size() == 1) {
                // only adjacent to element p: eliminate together with p
                mass.push_back(i);
                continue;
            }
            const long dp = static_cast<long>(lp.size()) - 1;
            long d = static_cast<long>(Ai.size()) + dp + ext;
            d = std::min<long>(d, static_cast<long>(deg[i]) + dp);
            d = std::min<long>(d, static_cast<long>(remaining) - 2);
            if (d < 1) d = 1;
            deg[i] = static_cast<Idx>(d);
        }

        // ---- finalize the new element ---------------------------------
        for (Idx i : mass) {
            state[i] = Dead;
            perm.push_back(i);
            --nlive;
            --lp_live;
            A[i].clear();
            E[i].clear();
        }
        remaining -= (1 + mass.size());
        L[p].clear();
        for (Idx i : lp)
            if (state[i] == Live) L[p].push_back(i);
        state[p] = L[p].empty() ? Dead : Element;
        for (Idx i : L[p]) {
            bucket_insert(i);
            if (static_cast<size_t>(deg[i]) < mindeg) mindeg = deg[i];
        }
        (void)lp_live;
    }
    for (size_t i = 0; i < n; ++i) // safety net: never lose a variable
        if (state[i] == Live) perm.push_back(static_cast<Idx>(i));
    for (Idx v : dense_vars) perm.push_back(v);
    return perm;
}

} // namespace Sparse
} // namespace AXOS
