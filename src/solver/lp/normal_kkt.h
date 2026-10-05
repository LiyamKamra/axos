// SPDX-License-Identifier: BSD-3-Clause
//
// Normal-equations solver for the interior-point KKT system (CPU).
//
// The augmented system used by Ipm,
//        [ -D    A^T ] [dx]   [r1]
//        [  A     E  ] [dy] = [r2],     D = Theta_x^{-1} + rho,  E = Theta_w + delta,
// is solved by eliminating dx:  S dy = r2 + A D^{-1} r1  with S = A D^{-1} A^T + E
// (symmetric positive definite), then dx = D^{-1}(A^T dy - r1). This wins when
// A has many more columns than rows and no dense columns (transportation and
// assignment LPs, set covering): S is only m x m and is factored with the
// multifrontal Cholesky. It is not used with free columns (D -> rho makes S
// badly conditioned) or when a column touches many rows (S would be dense).
#pragma once

#include "solver/model.h"
#include "sparse/sparse.h"
#include "tensorET.h"
#include <algorithm>
#include <cstdint>
#include <vector>

namespace AXOS {
namespace Solver {

class NormalKkt {
    using Ldl = Sparse::SparseLdlt<double, int32_t, Cpu::HostStorage>;
    using Mat = Sparse::Csr<double, int32_t, Cpu::HostStorage>;
    using Vec = tensorET<1, double, Cpu::HostStorage<double>>;

  public:
    // Eligibility: no free columns, columns not too dense, S not too large.
    // On success the pattern of S is built and eligible() is true.
    NormalKkt(const LpProblem &q, Sparse::Ordering ord)
        : ldl_(Sparse::Symmetry::SPD, ord)
    {
        m_ = q.rows();
        n_ = q.cols();
        for (size_t j = 0; j < n_; ++j)
            if (!std::isfinite(q.col_lb[j]) && !std::isfinite(q.col_ub[j])) return;
        A_ = &q.A;
        AT_ = q.A.transpose();
        const int *cp = AT_.row_ptr();
        double work = 0;
        for (size_t j = 0; j < n_; ++j) {
            const double c = cp[j + 1] - cp[j];
            if (c > 60) return; // dense column
            work += c * c;
        }
        if (work > 30.0 * (q.A.nnz() + m_)) return; // S would fill in too much
        par_ = work > 2e5;
        // pattern of S row by row (sorted, with the diagonal)
        std::vector<int32_t> rp(m_ + 1, 0), ci;
        std::vector<int32_t> mark(m_, -1), tmp;
        const int *ar = A_->row_ptr(), *ac = A_->col_ind();
        const int *tr = AT_.row_ptr(), *tc = AT_.col_ind();
        for (size_t i = 0; i < m_; ++i) {
            tmp.clear();
            mark[i] = static_cast<int32_t>(i);
            tmp.push_back(static_cast<int32_t>(i));
            for (int k = ar[i]; k < ar[i + 1]; ++k) {
                const int j = ac[k];
                for (int t = tr[j]; t < tr[j + 1]; ++t) {
                    const int r = tc[t];
                    if (mark[r] != static_cast<int32_t>(i)) {
                        mark[r] = static_cast<int32_t>(i);
                        tmp.push_back(r);
                    }
                }
            }
            std::sort(tmp.begin(), tmp.end());
            ci.insert(ci.end(), tmp.begin(), tmp.end());
            rp[i + 1] = static_cast<int32_t>(ci.size());
            if (ci.size() > 60u * m_ + 4000000u) return; // too large
        }
        srp_ = rp;
        sci_ = ci;
        diag_.resize(m_);
        for (size_t i = 0; i < m_; ++i)
            for (int32_t p = rp[i]; p < rp[i + 1]; ++p)
                if (static_cast<size_t>(ci[p]) == i) diag_[i] = p;
        S_ = Mat(m_, m_, rp, ci, std::vector<double>(ci.size(), 0.0));
        dinv_.assign(n_, 0.0);
        y_ = Vec(m_);
        rhs_ = Vec(m_);
        ok_ = true;
    }

    bool eligible() const { return ok_; }

    void analyze()
    {
        ldl_.analyze(S_);
        analyzed_ = true;
    }
    double factor_flops() const { return ldl_.factor_flops(); }
    size_t factor_nnz() const { return ldl_.factor_nnz(); }
    void print_profile() const { ldl_.print_profile(); }

    // hz: Theta_x^{-1} for the n columns; thw: Theta_w for the m rows.
    bool factorize(const double *hz, const double *thw, double rho, double delta)
    {
        for (size_t j = 0; j < n_; ++j) dinv_[j] = 1.0 / (hz[j] + rho);
        const int *ar = A_->row_ptr(), *ac = A_->col_ind();
        const double *av = A_->values();
        const int *tr = AT_.row_ptr(), *tc = AT_.col_ind();
        const double *tv = AT_.values();
        double *sv = S_.values_mut();
#pragma omp parallel if (par_)
        {
            std::vector<double> acc(m_, 0.0);
#pragma omp for schedule(dynamic, 64)
            for (long i = 0; i < static_cast<long>(m_); ++i) {
                for (int k = ar[i]; k < ar[i + 1]; ++k) {
                    const int j = ac[k];
                    const double f = av[k] * dinv_[j];
                    for (int t = tr[j]; t < tr[j + 1]; ++t) acc[tc[t]] += f * tv[t];
                }
                for (int32_t p = srp_[i]; p < srp_[i + 1]; ++p) {
                    sv[p] = acc[sci_[p]];
                    acc[sci_[p]] = 0.0;
                }
                sv[diag_[i]] += thw[i] + delta;
            }
        }
        return ldl_.factorize(S_);
    }

    // Solves the (regularized) augmented system: rhs = [r1 (n); r2 (m)].
    void solve(const double *r, double *out)
    {
        const double *r1 = r, *r2 = r + n_;
        // y := r2 + A D^{-1} r1
        std::vector<double> t(n_);
        for (size_t j = 0; j < n_; ++j) t[j] = dinv_[j] * r1[j];
        for (size_t i = 0; i < m_; ++i) {
            double s = r2[i];
            for (int k = A_->row_ptr()[i]; k < A_->row_ptr()[i + 1]; ++k)
                s += A_->values()[k] * t[A_->col_ind()[k]];
            rhs_.data[i] = s;
        }
        ldl_.solve(rhs_, y_);
        double *dx = out, *dy = out + n_;
        for (size_t i = 0; i < m_; ++i) dy[i] = y_.data[i];
        // dx = D^{-1} (A^T dy - r1)
        for (size_t j = 0; j < n_; ++j) {
            double s = 0;
            for (int k = AT_.row_ptr()[j]; k < AT_.row_ptr()[j + 1]; ++k)
                s += AT_.values()[k] * dy[AT_.col_ind()[k]];
            dx[j] = dinv_[j] * (s - r1[j]);
        }
    }

  private:
    Ldl ldl_;
    const HostMatrix *A_ = nullptr;
    HostMatrix AT_;
    Mat S_;
    std::vector<int32_t> srp_, sci_, diag_;
    std::vector<double> dinv_;
    Vec y_, rhs_;
    size_t m_ = 0, n_ = 0;
    bool ok_ = false, analyzed_ = false, par_ = false;
};

} // namespace Solver
} // namespace AXOS
