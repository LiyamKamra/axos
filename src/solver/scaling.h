// SPDX-License-Identifier: BSD-3-Clause
//
// Diagonal scaling for LPs: Ruiz equilibration followed by Pock-Chambolle.
//
// With row factors r and column factors s, the scaled problem is
//   A~ = diag(r) A diag(s),  c~ = s .* c,  x~ = x ./ s,
//   lb~ = lb ./ s,  ub~ = ub ./ s,  row bounds~ = r .* row bounds.
// Duals map back as y = r .* y~ and z = z~ ./ s; objective values (and the
// offset) are unchanged.
#pragma once

#include "solver/model.h"
#include "sparse/sparse.h"
#include <cmath>

namespace AXOS {
namespace Solver {

struct Scaling {
    std::vector<double> row; // r
    std::vector<double> col; // s
};

// Computes scaling factors for A (does not modify A).
inline Scaling
compute_scaling(const HostMatrix &A0, int ruiz_iterations,
    double pock_chambolle_alpha)
{
    const size_t m = A0.rows(), n = A0.cols();
    Scaling sc;
    sc.row.assign(m, 1.0);
    sc.col.assign(n, 1.0);
    if (m == 0 || n == 0) return sc;
    HostMatrix A(A0);
    std::vector<double> rn(m), cn(n);

    for (int it = 0; it < ruiz_iterations; ++it) {
        Sparse::row_norms(A, rn.data(), Sparse::Norm::Linf);
        Sparse::col_norms(A, cn.data(), Sparse::Norm::Linf);
        for (size_t i = 0; i < m; ++i) rn[i] = rn[i] > 0 ? 1.0 / std::sqrt(rn[i]) : 1.0;
        for (size_t j = 0; j < n; ++j) cn[j] = cn[j] > 0 ? 1.0 / std::sqrt(cn[j]) : 1.0;
        Sparse::scale_rows_cols(A, rn.data(), cn.data());
        for (size_t i = 0; i < m; ++i) sc.row[i] *= rn[i];
        for (size_t j = 0; j < n; ++j) sc.col[j] *= cn[j];
    }

    if (pock_chambolle_alpha >= 0) {
        // rows: l_{2-alpha} norms, columns: l_alpha norms (both l1 for alpha=1)
        const double alpha = pock_chambolle_alpha;
        const double pr = 2.0 - alpha, pc = alpha;
        std::fill(rn.begin(), rn.end(), 0.0);
        std::fill(cn.begin(), cn.end(), 0.0);
        const auto *rp = A.row_ptr();
        const auto *ci = A.col_ind();
        const double *v = A.values();
        for (size_t i = 0; i < m; ++i)
            for (int k = rp[i]; k < rp[i + 1]; ++k) {
                double a = std::abs(v[k]);
                rn[i] += std::pow(a, pr);
                cn[ci[k]] += std::pow(a, pc);
            }
        for (size_t i = 0; i < m; ++i)
            rn[i] = rn[i] > 0 ? 1.0 / std::sqrt(std::pow(rn[i], 1.0 / pr)) : 1.0;
        for (size_t j = 0; j < n; ++j)
            cn[j] = cn[j] > 0 ? 1.0 / std::sqrt(std::pow(cn[j], 1.0 / std::max(pc, 1e-12))) : 1.0;
        for (size_t i = 0; i < m; ++i) sc.row[i] *= rn[i];
        for (size_t j = 0; j < n; ++j) sc.col[j] *= cn[j];
    }
    return sc;
}

// Returns the scaled copy of `p`.
inline LpProblem
apply_scaling(const LpProblem &p, const Scaling &sc)
{
    LpProblem q = p;
    HostMatrix A(p.A);
    Sparse::scale_rows_cols(A, sc.row.data(), sc.col.data());
    q.A = std::move(A);
    for (size_t j = 0; j < p.cols(); ++j) {
        q.c[j] = p.c[j] * sc.col[j];
        q.col_lb[j] = p.col_lb[j] / sc.col[j];
        q.col_ub[j] = p.col_ub[j] / sc.col[j];
    }
    for (size_t i = 0; i < p.rows(); ++i) {
        q.row_lb[i] = p.row_lb[i] * sc.row[i];
        q.row_ub[i] = p.row_ub[i] * sc.row[i];
    }
    return q;
}

// Maps a solution of the scaled problem back to the original variables.
inline void
unscale_solution(std::vector<double> &x, std::vector<double> &y,
    const Scaling &sc)
{
    for (size_t j = 0; j < x.size(); ++j) x[j] *= sc.col[j];
    for (size_t i = 0; i < y.size(); ++i) y[i] *= sc.row[i];
}

} // namespace Solver
} // namespace AXOS
