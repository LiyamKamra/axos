// SPDX-License-Identifier: BSD-3-Clause
//
// Sparse LU of a simplex basis with product-form (eta) updates.
//
//   factor()   Gilbert-Peierls left-looking LU with threshold partial
//              pivoting: columns are ordered by increasing nonzero count
//              (singletons, i.e. slacks, go first) and among acceptable pivots
//              the row with the fewest original nonzeros is preferred.
//              Singular columns are reported so the caller can replace them
//              by slack columns.
//   ftran()    solves B x = b     (b indexed by row, x by basis position)
//   btran()    solves B^T y = c   (c indexed by basis position, y by row)
//   update()   replaces the column at basis position r (eta file), so
//              B_new^{-1} = E B_old^{-1}
//
// All vectors are dense (length m); the triangular solves cost
// O(nnz(L) + nnz(U) + nnz(etas)).
#pragma once

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <vector>

namespace AXOS {
namespace Solver {

// Sparse work vector: dense values plus the list of positions that may be
// nonzero (each position appears at most once).
struct SVec {
    std::vector<double> v;
    std::vector<int> idx;
    std::vector<char> flag;

    void init(int n) { v.assign(n, 0.0); flag.assign(n, 0); idx.clear(); }
    int size() const { return static_cast<int>(v.size()); }
    void clear() {
        for (int i : idx) { v[i] = 0.0; flag[i] = 0; }
        idx.clear();
    }
    // v[i] += a, tracking the pattern
    void add(int i, double a) {
        if (!flag[i]) { flag[i] = 1; idx.push_back(i); }
        v[i] += a;
    }
    void set(int i, double a) {
        if (!flag[i]) { flag[i] = 1; idx.push_back(i); }
        v[i] = a;
    }
    // rebuild the pattern from the dense values
    void rebuild_pattern() {
        for (int i : idx) flag[i] = 0;
        idx.clear();
        for (int i = 0; i < size(); ++i)
            if (v[i] != 0.0) { flag[i] = 1; idx.push_back(i); }
    }
};

class BasisFactor {
  public:
    // A sparse column: parallel arrays of row indices and values.
    struct Column {
        const int *idx;
        const double *val;
        int nnz;
    };

    // Factor the m x m matrix whose k-th column is cols[k]. Returns the number
    // of singular pivots; for each, singular_pos[i] is the basis position of the
    // dependent column and free_row[i] a row that was left without a pivot
    // (the caller substitutes the slack of that row).
    int
    factor(int m, const std::vector<Column> &cols, std::vector<int> &singular_pos,
        std::vector<int> &free_row)
    {
        m_ = m;
        n_updates_ = 0;
        eta_ptr_.assign(1, 0);
        eta_idx_.clear();
        eta_val_.clear();
        eta_pos_.clear();
        eta_diag_.clear();
        singular_pos.clear();
        free_row.clear();

        // row counts of the original matrix (sparsity tie-break)
        std::vector<int> rowcnt(m, 0);
        for (int k = 0; k < m; ++k)
            for (int t = 0; t < cols[k].nnz; ++t) rowcnt[cols[k].idx[t]]++;
        // column order: fewest nonzeros first
        q_.resize(m);
        for (int k = 0; k < m; ++k) q_[k] = k;
        std::stable_sort(q_.begin(), q_.end(),
            [&](int a, int b) { return cols[a].nnz < cols[b].nnz; });

        lp_.assign(1, 0); li_.clear(); lx_.clear();
        up_.assign(1, 0); ui_.clear(); ux_.clear();
        udiag_.assign(m, 1.0);
        prow_.assign(m, -1);       // prow[k] = row pivoted at step k
        std::vector<int> pinv(m, -1); // pinv[row] = step, -1 if not pivoted
        std::vector<double> x(m, 0.0);
        std::vector<int> xi;       // nonzero pattern in topological order
        std::vector<int> stack, pstack, mark(m, -1);
        std::vector<int> lcol_row;  // L entries use ORIGINAL row ids until the end
        int nsing = 0;
        std::vector<int> failed_steps;
        // L columns are stored by step; entries below the pivot in original rows
        std::vector<int> lstart{0};
        std::vector<int> l_orig_idx;
        std::vector<double> l_val;

        for (int k = 0; k < m; ++k) {
            const Column &c = cols[q_[k]];
            // ---- reach of the column pattern in the graph of L ----------
            xi.clear();
            for (int t = 0; t < c.nnz; ++t) {
                int i = c.idx[t];
                if (mark[i] == k) continue;
                // iterative DFS from row i
                stack.assign(1, i);
                pstack.assign(1, 0);
                mark[i] = k;
                while (!stack.empty()) {
                    int r = stack.back();
                    int step = pinv[r];
                    int &pp = pstack.back();
                    bool pushed = false;
                    if (step >= 0) {
                        int beg = lstart[step], end = lstart[step + 1];
                        for (; beg + pp < end;) {
                            int ch = l_orig_idx[beg + pp];
                            ++pp;
                            if (mark[ch] != k) {
                                mark[ch] = k;
                                stack.push_back(ch);
                                pstack.push_back(0);
                                pushed = true;
                                break;
                            }
                        }
                    }
                    if (!pushed) {
                        xi.push_back(r);
                        stack.pop_back();
                        pstack.pop_back();
                    }
                }
            }
            // xi is in reverse topological order (finish order): process reversed
            for (int t = 0; t < c.nnz; ++t) x[c.idx[t]] = c.val[t];
            for (int t = static_cast<int>(xi.size()) - 1; t >= 0; --t) {
                int r = xi[t];
                int step = pinv[r];
                if (step < 0) continue;
                double xr = x[r];
                if (xr == 0.0) continue;
                for (int p = lstart[step]; p < lstart[step + 1]; ++p)
                    x[l_orig_idx[p]] -= l_val[p] * xr;
            }
            // ---- pivot selection among unpivoted rows ---------------------
            double amax = 0;
            for (int r : xi)
                if (pinv[r] < 0) amax = std::max(amax, std::abs(x[r]));
            int piv = -1;
            if (amax > 1e-11) {
                const double thresh = 0.1 * amax;
                int best_cnt = 1 << 30;
                double best_abs = 0;
                for (int r : xi) {
                    if (pinv[r] >= 0) continue;
                    double a = std::abs(x[r]);
                    if (a < thresh) continue;
                    if (rowcnt[r] < best_cnt || (rowcnt[r] == best_cnt && a > best_abs)) {
                        best_cnt = rowcnt[r];
                        best_abs = a;
                        piv = r;
                    }
                }
            }
            if (piv < 0) { // singular column
                ++nsing;
                failed_steps.push_back(k);
                singular_pos.push_back(q_[k]);
                for (int r : xi) x[r] = 0.0;
                for (int t = 0; t < c.nnz; ++t) x[c.idx[t]] = 0.0;
                // record an empty step; pinv stays -1 (row remains free)
                lstart.push_back(static_cast<int>(l_orig_idx.size()));
                up_.push_back(static_cast<int>(ui_.size()));
                prow_[k] = -1;
                udiag_[k] = 1.0;
                continue;
            }
            const double pv = x[piv];
            udiag_[k] = pv;
            pinv[piv] = k;
            prow_[k] = piv;
            // U column: entries at already pivoted rows (steps < k)
            for (int r : xi) {
                if (r == piv) { x[r] = 0.0; continue; }
                if (pinv[r] >= 0 && pinv[r] < k) {
                    if (x[r] != 0.0) {
                        ui_.push_back(pinv[r]);
                        ux_.push_back(x[r]);
                    }
                } else if (x[r] != 0.0) { // below the pivot: L entry
                    l_orig_idx.push_back(r);
                    l_val.push_back(x[r] / pv);
                }
                x[r] = 0.0;
            }
            for (int t = 0; t < c.nnz; ++t) x[c.idx[t]] = 0.0;
            up_.push_back(static_cast<int>(ui_.size()));
            lstart.push_back(static_cast<int>(l_orig_idx.size()));
        }
        // ---- repair: pair every singular column with an unpivoted row -----
        {
            std::vector<int> unp;
            for (int r = 0; r < m; ++r)
                if (pinv[r] < 0) unp.push_back(r);
            for (size_t t = 0; t < unp.size() && t < failed_steps.size(); ++t) {
                free_row.push_back(unp[t]);
            }
        }
        // ---- convert L row ids to pivot steps ---------------------------------
        lp_ = lstart;
        li_.resize(l_orig_idx.size());
        for (size_t p = 0; p < l_orig_idx.size(); ++p) li_[p] = pinv[l_orig_idx[p]];
        lx_ = l_val;
        // entries whose row is still unpivoted (only when singular) are dropped
        if (nsing > 0) {
            for (size_t p = 0; p < li_.size(); ++p)
                if (li_[p] < 0) lx_[p] = 0.0, li_[p] = 0;
        }
        singular_ = nsing;
        pinv_ = pinv;
        qinv_.assign(m, 0);
        for (int k = 0; k < m; ++k) qinv_[q_[k]] = k;
        mark_.assign(m, 0);
        stamp_ = 0;
        w_.assign(m, 0.0);
        // row-wise copies of L and U so that BTRAN can skip zero entries
        transpose(lp_, li_, lx_, lrp_, lci_, lrx_);
        transpose(up_, ui_, ux_, urp_, uci_, urx_);
        return nsing;
    }

    // Solve B x = b in place: on input b is indexed by row, on output by basis
    // position. `work` must have m entries.
    void
    ftran(std::vector<double> &b) const
    {
        const int m = m_;
        w_.assign(m, 0.0);
        for (int k = 0; k < m; ++k) w_[k] = prow_[k] >= 0 ? b[prow_[k]] : 0.0;
        for (int k = 0; k < m; ++k) { // L
            double yk = w_[k];
            if (yk == 0.0) continue;
            for (int p = lp_[k]; p < lp_[k + 1]; ++p) w_[li_[p]] -= lx_[p] * yk;
        }
        for (int k = m - 1; k >= 0; --k) { // U (upper triangular in step order)
            double zk = w_[k] / udiag_[k];
            w_[k] = zk;
            if (zk == 0.0) continue;
            for (int p = up_[k]; p < up_[k + 1]; ++p) w_[ui_[p]] -= ux_[p] * zk;
        }
        for (int k = 0; k < m; ++k) b[q_[k]] = w_[k];
        std::fill(w_.begin(), w_.end(), 0.0); // the sparse solves expect w_ == 0
        // eta file, oldest first
        for (int e = 0; e < n_updates_; ++e) {
            const int r = eta_pos_[e];
            const double tr = b[r];
            if (tr == 0.0) continue;
            b[r] = tr * eta_diag_[e];
            for (int p = eta_ptr_[e]; p < eta_ptr_[e + 1]; ++p)
                b[eta_idx_[p]] += eta_val_[p] * tr;
        }
    }

    // Solve B^T y = c: c indexed by basis position, y by row.
    void
    btran(std::vector<double> &c) const
    {
        const int m = m_;
        for (int e = n_updates_ - 1; e >= 0; --e) { // eta file, newest first
            const int r = eta_pos_[e];
            double s = c[r] * eta_diag_[e];
            for (int p = eta_ptr_[e]; p < eta_ptr_[e + 1]; ++p)
                s += eta_val_[p] * c[eta_idx_[p]];
            c[r] = s;
        }
        w_.assign(m, 0.0);
        for (int k = 0; k < m; ++k) w_[k] = c[q_[k]];
        for (int i = 0; i < m; ++i) { // U^T z = w, row-wise sweep
            const double zi = w_[i] / udiag_[i];
            w_[i] = zi;
            if (zi == 0.0) continue;
            for (int p = urp_[i]; p < urp_[i + 1]; ++p) w_[uci_[p]] -= urx_[p] * zi;
        }
        for (int i = m - 1; i >= 0; --i) { // L^T v = z, row-wise sweep
            const double vi = w_[i];
            if (vi == 0.0) continue;
            for (int p = lrp_[i]; p < lrp_[i + 1]; ++p) w_[lci_[p]] -= lrx_[p] * vi;
        }
        for (int k = 0; k < m; ++k)
            if (prow_[k] >= 0) c[prow_[k]] = w_[k];
        std::fill(w_.begin(), w_.end(), 0.0);
    }

    // ---- hypersparse solves ---------------------------------------------------
    // Same operations as ftran/btran, on a sparse vector: the triangular solves
    // visit only the nonzero pattern of the result (symbolic reach by DFS), and
    // fall back to the dense loops when the vector is not sparse.

    // Solve B x = b: b indexed by row on input, by basis position on output.
    void
    ftran(SVec &b) const
    {
        const int m = m_;
        if (static_cast<int>(b.idx.size()) > 0.1 * m) {
            ftran(b.v);
            b.rebuild_pattern();
            return;
        }
        // rows -> steps
        starts_.clear();
        for (int i : b.idx) {
            const int k = pinv_[i];
            const double val = b.v[i];
            b.v[i] = 0.0; b.flag[i] = 0;
            if (k < 0 || val == 0.0) continue;
            w_[k] = val;
            starts_.push_back(k);
        }
        b.idx.clear();
        reach(lp_, li_, starts_, order_);       // L: edges step -> larger steps
        for (int t = static_cast<int>(order_.size()) - 1; t >= 0; --t) {
            const int k = order_[t];
            const double yk = w_[k];
            if (yk == 0.0) continue;
            for (int p = lp_[k]; p < lp_[k + 1]; ++p) w_[li_[p]] -= lx_[p] * yk;
        }
        starts_.assign(order_.begin(), order_.end());
        reach(up_, ui_, starts_, order2_);      // U: edges step -> smaller steps
        for (int t = static_cast<int>(order2_.size()) - 1; t >= 0; --t) {
            const int k = order2_[t];
            const double zk = w_[k] / udiag_[k];
            w_[k] = zk;
            if (zk == 0.0) continue;
            for (int p = up_[k]; p < up_[k + 1]; ++p) w_[ui_[p]] -= ux_[p] * zk;
        }
        for (int k : order2_) {
            const double val = w_[k];
            w_[k] = 0.0;
            if (val != 0.0) b.set(q_[k], val);
        }
        for (int e = 0; e < n_updates_; ++e) { // eta file, oldest first
            const int r = eta_pos_[e];
            const double tr = b.v[r];
            if (tr == 0.0) continue;
            b.v[r] = tr * eta_diag_[e];
            for (int p = eta_ptr_[e]; p < eta_ptr_[e + 1]; ++p)
                b.add(eta_idx_[p], eta_val_[p] * tr);
        }
    }

    // Solve B^T y = c: c indexed by basis position on input, by row on output.
    void
    btran(SVec &c) const
    {
        const int m = m_;
        if (static_cast<int>(c.idx.size()) > 0.1 * m && n_updates_ == 0) {
            btran(c.v);
            c.rebuild_pattern();
            return;
        }
        for (int e = n_updates_ - 1; e >= 0; --e) { // eta file, newest first
            const int r = eta_pos_[e];
            double sum = c.v[r] * eta_diag_[e];
            for (int p = eta_ptr_[e]; p < eta_ptr_[e + 1]; ++p)
                sum += eta_val_[p] * c.v[eta_idx_[p]];
            if (sum != 0.0 || c.v[r] != 0.0) c.set(r, sum);
        }
        if (static_cast<int>(c.idx.size()) > 0.1 * m) {
            btran_lu_dense(c);
            return;
        }
        starts_.clear();
        for (int pos : c.idx) {
            const double val = c.v[pos];
            c.v[pos] = 0.0; c.flag[pos] = 0;
            if (val == 0.0) continue;
            const int k = qinv_[pos];
            w_[k] = val;
            starts_.push_back(k);
        }
        c.idx.clear();
        reach(urp_, uci_, starts_, order_);     // U^T: row-wise U, edges to larger steps
        for (int t = static_cast<int>(order_.size()) - 1; t >= 0; --t) {
            const int i = order_[t];
            const double zi = w_[i] / udiag_[i];
            w_[i] = zi;
            if (zi == 0.0) continue;
            for (int p = urp_[i]; p < urp_[i + 1]; ++p) w_[uci_[p]] -= urx_[p] * zi;
        }
        starts_.assign(order_.begin(), order_.end());
        reach(lrp_, lci_, starts_, order2_);    // L^T: row-wise L, edges to smaller steps
        for (int t = static_cast<int>(order2_.size()) - 1; t >= 0; --t) {
            const int i = order2_[t];
            const double vi = w_[i];
            if (vi == 0.0) continue;
            for (int p = lrp_[i]; p < lrp_[i + 1]; ++p) w_[lci_[p]] -= lrx_[p] * vi;
        }
        for (int k : order2_) {
            const double val = w_[k];
            w_[k] = 0.0;
            if (val != 0.0 && prow_[k] >= 0) c.set(prow_[k], val);
        }
    }

    // Column at basis position r is replaced; alpha = B_old^{-1} a_q (indexed
    // by basis position). Returns false if the pivot is unusably small.
    bool
    update(int r, const std::vector<double> &alpha)
    {
        const double ar = alpha[r];
        if (std::abs(ar) < 1e-11) return false;
        eta_pos_.push_back(r);
        eta_diag_.push_back(1.0 / ar);
        for (int i = 0; i < m_; ++i) {
            if (i == r) continue;
            const double v = alpha[i];
            if (v != 0.0 && std::abs(v) > 1e-14) {
                eta_idx_.push_back(i);
                eta_val_.push_back(-v / ar);
            }
        }
        eta_ptr_.push_back(static_cast<int>(eta_idx_.size()));
        ++n_updates_;
        return true;
    }

    // Same as update(r, alpha) with a sparse alpha.
    bool
    update(int r, const SVec &alpha)
    {
        const double ar = alpha.v[r];
        if (std::abs(ar) < 1e-11) return false;
        eta_pos_.push_back(r);
        eta_diag_.push_back(1.0 / ar);
        for (int i : alpha.idx) {
            if (i == r) continue;
            const double v = alpha.v[i];
            if (v != 0.0 && std::abs(v) > 1e-14) {
                eta_idx_.push_back(i);
                eta_val_.push_back(-v / ar);
            }
        }
        eta_ptr_.push_back(static_cast<int>(eta_idx_.size()));
        ++n_updates_;
        return true;
    }

    int updates() const { return n_updates_; }
    // true when the eta file has grown larger than the factors it extends
    bool eta_heavy(double factor) const { return eta_idx_.size() > factor * (li_.size() + ui_.size() + m_); }
    size_t nnz() const { return li_.size() + ui_.size() + eta_idx_.size() + m_; }
    int size() const { return m_; }

  private:
    // The L/U triangular part of btran on a dense-ish SVec (no eta file).
    void
    btran_lu_dense(SVec &c) const
    {
        const int m = m_;
        w_.assign(m, 0.0);
        for (int k = 0; k < m; ++k) w_[k] = c.v[q_[k]];
        for (int i = 0; i < m; ++i) {
            const double zi = w_[i] / udiag_[i];
            w_[i] = zi;
            if (zi == 0.0) continue;
            for (int p = urp_[i]; p < urp_[i + 1]; ++p) w_[uci_[p]] -= urx_[p] * zi;
        }
        for (int i = m - 1; i >= 0; --i) {
            const double vi = w_[i];
            if (vi == 0.0) continue;
            for (int p = lrp_[i]; p < lrp_[i + 1]; ++p) w_[lci_[p]] -= lrx_[p] * vi;
        }
        c.clear();
        for (int k = 0; k < m; ++k)
            if (prow_[k] >= 0 && w_[k] != 0.0) c.set(prow_[k], w_[k]);
        std::fill(w_.begin(), w_.end(), 0.0);
    }

    // Nodes reachable from `starts` in the graph with adjacency (ptr, idx), in
    // DFS finishing order (so the reverse is a topological order).
    void
    reach(const std::vector<int> &ptr, const std::vector<int> &idx,
        const std::vector<int> &starts, std::vector<int> &out) const
    {
        out.clear();
        if (++stamp_ == 0x7fffffff) { std::fill(mark_.begin(), mark_.end(), 0); stamp_ = 1; }
        for (int s0 : starts) {
            if (mark_[s0] == stamp_) continue;
            mark_[s0] = stamp_;
            stk_.assign(1, s0);
            pst_.assign(1, ptr[s0]);
            while (!stk_.empty()) {
                const int node = stk_.back();
                int &pp = pst_.back();
                bool pushed = false;
                while (pp < ptr[node + 1]) {
                    const int ch = idx[pp++];
                    if (mark_[ch] != stamp_) {
                        mark_[ch] = stamp_;
                        stk_.push_back(ch);
                        pst_.push_back(ptr[ch]);
                        pushed = true;
                        break;
                    }
                }
                if (!pushed) { out.push_back(node); stk_.pop_back(); pst_.pop_back(); }
            }
        }
    }

    // CSR transpose of a column-wise triangular factor (column k holds row
    // indices idx[ptr[k]..ptr[k+1])).
    void
    transpose(const std::vector<int> &ptr, const std::vector<int> &idx,
        const std::vector<double> &val, std::vector<int> &rp, std::vector<int> &ci,
        std::vector<double> &rx) const
    {
        const int m = m_;
        rp.assign(m + 1, 0);
        for (size_t p = 0; p < idx.size(); ++p) ++rp[idx[p] + 1];
        for (int i = 0; i < m; ++i) rp[i + 1] += rp[i];
        ci.resize(idx.size());
        rx.resize(idx.size());
        std::vector<int> pos(rp.begin(), rp.end() - 1);
        for (int k = 0; k < m; ++k)
            for (int p = ptr[k]; p < ptr[k + 1]; ++p) {
                const int q = pos[idx[p]]++;
                ci[q] = k;
                rx[q] = val[p];
            }
    }

    int m_ = 0, n_updates_ = 0, singular_ = 0;
    std::vector<int> q_, prow_;
    std::vector<int> lp_, li_, up_, ui_;
    std::vector<double> lx_, ux_, udiag_;
    std::vector<int> lrp_, lci_, urp_, uci_;
    std::vector<double> lrx_, urx_;
    std::vector<int> eta_ptr_{0}, eta_idx_, eta_pos_;
    std::vector<double> eta_val_, eta_diag_;
    mutable std::vector<double> w_;
    std::vector<int> pinv_, qinv_;
    mutable std::vector<int> mark_, starts_, order_, order2_, stk_, pst_;
    mutable int stamp_ = 0;
};

} // namespace Solver
} // namespace AXOS
