// SPDX-License-Identifier: BSD-3-Clause
//
// Root probing on binary columns (CPU; the GPU version, which probes many
// binaries at once, is gpu::probe_binaries in milp_cuda.h).
//
// Each binary x_j is propagated at 0 and at 1 (propagate.h). A probe that
// empties the domain fixes x_j the other way (and keeps that probe's
// implications); otherwise, for every other column, the weaker of the two
// probes' bounds holds whichever value x_j takes ("double probing") and
// replaces the global bound when tighter. Both probes empty: infeasible.
// The work per probe is that of its propagation: only the columns a probe
// changed (its trail) are compared.
#pragma once

#include "solver/milp/milp_model.h"
#include "solver/milp/propagate.h"
#include <chrono>
#include <vector>

namespace AXOS {
namespace Solver {
namespace milp {

struct ProbingStats {
    int probes = 0, fixed = 0, tightened = 0;
    bool infeasible = false;
    double seconds = 0;
};

inline ProbingStats
probe_binaries_cpu(Propagator &prop, const LpProblem &p, std::vector<double> &lb, std::vector<double> &ub,
    double seconds)
{
    const auto t0 = std::chrono::steady_clock::now();
    auto elapsed = [&] { return std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count(); };
    ProbingStats st;
    const size_t n = p.cols();
    std::vector<double> wl = lb, wu = ub; // working domain, equal to lb/ub between probes
    std::vector<int> mark(n, -1), c0;
    std::vector<double> pl(n), pu(n);    // probe 0 bounds of the columns it changed
    std::vector<int> changed(1);
    std::vector<std::pair<int, std::pair<double, double>>> upd;
    Trail tr;
    for (size_t jj = 0; jj < n && elapsed() < seconds; ++jj) {
        const int j = static_cast<int>(jj);
        if (!is_int(p, jj) || lb[jj] != 0 || ub[jj] != 1) continue;
        changed[0] = j;
        // probe x_j = 0
        tr.undo(0, wl, wu);
        tr.record(j, wl[j], wu[j]);
        wu[j] = 0;
        const bool f0 = prop.propagate(wl, wu, &changed, &tr, 25.0);
        c0.clear();
        for (size_t k = 0; k < tr.col.size(); ++k) {
            const int t = tr.col[k];
            if (mark[t] != j) {
                mark[t] = j;
                c0.push_back(t);
            }
        }
        for (int t : c0) {
            pl[t] = wl[t];
            pu[t] = wu[t];
        }
        tr.undo(0, wl, wu);
        // probe x_j = 1
        tr.record(j, wl[j], wu[j]);
        wl[j] = 1;
        const bool f1 = prop.propagate(wl, wu, &changed, &tr, 25.0);
        st.probes += 2;
        if (!f0 && !f1) {
            tr.undo(0, wl, wu);
            st.infeasible = true;
            break;
        }
        if (!f0) { // x_j = 1: keep probe 1
            for (int t : tr.col) {
                lb[t] = wl[t];
                ub[t] = wu[t];
            }
            tr.col.clear();
            tr.old_lb.clear();
            tr.old_ub.clear();
            ++st.fixed;
            continue;
        }
        if (!f1) { // x_j = 0: restore probe 0
            tr.undo(0, wl, wu);
            for (int t : c0) {
                lb[t] = wl[t] = pl[t];
                ub[t] = wu[t] = pu[t];
            }
            ++st.fixed;
            continue;
        }
        // both feasible: the weaker bound of the two probes, where both moved it
        upd.clear();
        for (int t : tr.col) {
            if (t == j || mark[t] != j) continue;
            const double l2 = std::min(pl[t], wl[t]), u2 = std::max(pu[t], wu[t]);
            if (l2 > lb[t] + 1e-9 || u2 < ub[t] - 1e-9) upd.push_back({t, {l2, u2}});
        }
        tr.undo(0, wl, wu);
        for (auto &e : upd) {
            const int t = e.first;
            if (mark[t] == -2 - j) continue; // the trail may list a column twice
            mark[t] = -2 - j;
            if (e.second.first > lb[t] + 1e-9) {
                lb[t] = wl[t] = e.second.first;
                ++st.tightened;
            }
            if (e.second.second < ub[t] - 1e-9) {
                ub[t] = wu[t] = e.second.second;
                ++st.tightened;
            }
        }
    }
    st.seconds = elapsed();
    return st;
}

} // namespace milp
} // namespace Solver
} // namespace AXOS
