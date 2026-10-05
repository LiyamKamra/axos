// SPDX-License-Identifier: BSD-3-Clause
//
// Diagonal preconditioning for QPs, as in HPR-QP and PDHCG: Ruiz
// equilibration followed by Pock-Chambolle, applied to the symmetric KKT
// matrix
//
//   K = [ Q  A^T ]      with column factors d (n) and row factors e (m):
//       [ A  0   ]      Q~ = D Q D,  A~ = E A D,  c~ = D c,
//                       x-bounds~ = bounds / d,  row bounds~ = e .* bounds.
//
// A solution maps back as x = D x~, y = E y~, z = z~ / d; objective values
// are unchanged.
#pragma once

#include "solver/qp/qp_model.h"
#include <cmath>
#include <vector>

namespace AXOS {
namespace Solver {

struct QpScaling {
    std::vector<double> col; // d
    std::vector<double> row; // e
};

namespace qp_scaling_detail {

// Row / column infinity (p = 0) or l1 (p = 1) norms of the scaled K blocks.
inline void
kkt_norms(const HostMatrix &Q, const HostMatrix &A,
    const std::vector<double> &d, const std::vector<double> &e, bool l1,
    std::vector<double> &cn, std::vector<double> &rn)
{
    const size_t n = d.size(), m = e.size();
    cn.assign(n, 0.0);
    rn.assign(m, 0.0);
    auto acc = [l1](double &s, double v) { s = l1 ? s + v : std::max(s, v); };
    if (Q.nnz()) {
        const auto *rp = Q.row_ptr();
        const auto *ci = Q.col_ind();
        const double *v = Q.values();
        for (size_t i = 0; i < n; ++i)
            for (int k = rp[i]; k < rp[i + 1]; ++k)
                acc(cn[i], std::abs(v[k]) * d[i] * d[ci[k]]);
    }
    const auto *rp = A.row_ptr();
    const auto *ci = A.col_ind();
    const double *v = A.values();
    for (size_t i = 0; i < m; ++i)
        for (int k = rp[i]; k < rp[i + 1]; ++k) {
            const double a = std::abs(v[k]) * e[i] * d[ci[k]];
            acc(rn[i], a);
            acc(cn[ci[k]], a);
        }
}

} // namespace qp_scaling_detail

inline QpScaling
compute_qp_scaling(
    const QpProblem &p, int ruiz_iterations, double pock_chambolle_alpha)
{
    using namespace qp_scaling_detail;
    const size_t n = p.cols(), m = p.rows();
    QpScaling s;
    s.col.assign(n, 1.0);
    s.row.assign(m, 1.0);
    std::vector<double> cn, rn;
    for (int it = 0; it < ruiz_iterations; ++it) {
        kkt_norms(p.Q, p.lp.A, s.col, s.row, false, cn, rn);
        for (size_t j = 0; j < n; ++j)
            if (cn[j] > 0) s.col[j] /= std::sqrt(cn[j]);
        for (size_t i = 0; i < m; ++i)
            if (rn[i] > 0) s.row[i] /= std::sqrt(rn[i]);
    }
    if (pock_chambolle_alpha >= 0) {
        // alpha = 1: l1 norms of rows and columns (K is symmetric)
        kkt_norms(p.Q, p.lp.A, s.col, s.row, true, cn, rn);
        for (size_t j = 0; j < n; ++j)
            if (cn[j] > 0) s.col[j] /= std::sqrt(cn[j]);
        for (size_t i = 0; i < m; ++i)
            if (rn[i] > 0) s.row[i] /= std::sqrt(rn[i]);
    }
    return s;
}

inline QpProblem
apply_qp_scaling(const QpProblem &p, const QpScaling &s)
{
    QpProblem q = p;
    const size_t n = p.cols(), m = p.rows();
    {
        double *v = q.lp.A.values_mut();
        const auto *rp = q.lp.A.row_ptr();
        const auto *ci = q.lp.A.col_ind();
        for (size_t i = 0; i < m; ++i)
            for (int k = rp[i]; k < rp[i + 1]; ++k)
                v[k] *= s.row[i] * s.col[ci[k]];
    }
    if (q.Q.nnz()) {
        double *v = q.Q.values_mut();
        const auto *rp = q.Q.row_ptr();
        const auto *ci = q.Q.col_ind();
        for (size_t i = 0; i < n; ++i)
            for (int k = rp[i]; k < rp[i + 1]; ++k)
                v[k] *= s.col[i] * s.col[ci[k]];
    }
    for (size_t j = 0; j < n; ++j) {
        q.lp.c[j] *= s.col[j];
        q.lp.col_lb[j] /= s.col[j];
        q.lp.col_ub[j] /= s.col[j];
    }
    for (size_t i = 0; i < m; ++i) {
        q.lp.row_lb[i] *= s.row[i];
        q.lp.row_ub[i] *= s.row[i];
    }
    return q;
}

} // namespace Solver
} // namespace AXOS
