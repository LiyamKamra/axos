// SPDX-License-Identifier: BSD-3-Clause
//
// GPU kernels of the MILP solver, compiled at run time by NVRTC (see
// milp_cuda.h); not included by host code. Three families:
//
//   prop_*  batched activity-based domain propagation (bound tightening):
//           B independent bound vectors (layout b * n + j) over one CSR/CSC
//           matrix. A round computes every row's min/max activity from the
//           bounds at the start of the round (one warp per (row, batch)),
//           then every column's implied bounds from its rows (one thread per
//           (column, batch), CSC, no atomics: a thread writes only its own
//           column). Bounds implied by valid bounds are valid, so these
//           Jacobi-style rounds reach the same fixpoint as the sequential
//           propagator. Batches are probes, fix-and-propagate attempts, or a
//           single domain (B = 1).
//   fj_*    feasibility jump (Luteberget & Sartor 2023), one walker per
//           thread block: the block picks a violated row, its warps evaluate
//           the jump values of the row's columns in parallel (lanes over the
//           breakpoints), the best move is applied with the row updates
//           spread over the threads. Walkers differ in start point and seed.
//   pd_*    batched PDHG for LP relaxations that share the matrix and costs
//           but differ in column bounds (children of a node). Interleaved
//           layout j * B + b with B <= 32, one warp per row / column and one
//           lane per LP. Every dual iterate y yields the Lagrangian bound
//               g(y) = sum_i p_i(y_i) + sum_j min_{l_j <= x_j <= u_j} r_j x_j,
//               r = c - A^T y,  p_i(y) = rl_i y (y > 0), ru_i y (y < 0),
//           a valid lower bound on that LP however far PDHG has converged.
#define MILP_BLOCK 256

typedef long long i64;

__device__ __forceinline__ bool
fin(double v)
{ return fabs(v) < 1e20; }

__device__ __forceinline__ double
warp_sum(double v)
{
    for (int off = 16; off > 0; off >>= 1)
        v += __shfl_down_sync(0xffffffffu, v, off);
    return v;
}

__device__ __forceinline__ int
warp_sum_i(int v)
{
    for (int off = 16; off > 0; off >>= 1)
        v += __shfl_down_sync(0xffffffffu, v, off);
    return v;
}

// ---- domain propagation
// ------------------------------------------------------

// Min / max activity of every (row, batch), with counts of infinite terms.
// A row whose finite min activity exceeds its upper side (or max activity is
// below its lower side) marks its batch infeasible.
extern "C" __global__ void
prop_activity(int m, int n, int B, const int *rp, const int *ci,
    const double *va, const double *rlb, const double *rub, const double *lb,
    const double *ub, double *minact, double *maxact, int *ninfmin,
    int *ninfmax, const int *active, int *infeas)
{
    const int lane = threadIdx.x & 31;
    const i64 g = ((i64)blockIdx.x * blockDim.x + threadIdx.x) >> 5;
    if (g >= (i64)m * B) return;
    const int b = (int)(g / m), i = (int)(g % m);
    if (!active[b]) return;
    const double *L = lb + (i64)b * n, *U = ub + (i64)b * n;
    double smin = 0, smax = 0;
    int cmin = 0, cmax = 0;
    for (int k = rp[i] + lane; k < rp[i + 1]; k += 32) {
        const double a = va[k];
        const int j = ci[k];
        const double lo = a > 0 ? L[j] : U[j], hi = a > 0 ? U[j] : L[j];
        if (fin(lo))
            smin += a * lo;
        else
            ++cmin;
        if (fin(hi))
            smax += a * hi;
        else
            ++cmax;
    }
    smin = warp_sum(smin);
    smax = warp_sum(smax);
    cmin = warp_sum_i(cmin);
    cmax = warp_sum_i(cmax);
    if (lane == 0) {
        const i64 o = (i64)b * m + i;
        minact[o] = smin;
        maxact[o] = smax;
        ninfmin[o] = cmin;
        ninfmax[o] = cmax;
        const double u = rub[i], l = rlb[i];
        if ((fin(u) && cmin == 0 && smin > u + 1e-6 * (1 + fabs(u))) ||
            (fin(l) && cmax == 0 && smax < l - 1e-6 * (1 + fabs(l))))
            infeas[b] = 1;
    }
}

// Implied bounds of every (column, batch) from the activities of its rows;
// the same rules and tolerances as propagate.h (integer rounding, no
// creeping of continuous bounds by less than 1e-3 of their range).
extern "C" __global__ void
prop_tighten(int n, int m, int B, const int *cp, const int *ri,
    const double *cv, const double *rlb, const double *rub,
    const double *minact, const double *maxact, const int *ninfmin,
    const int *ninfmax, double *lb, double *ub, const unsigned char *isint,
    const int *active, int *changed, int *infeas)
{
    const i64 t = (i64)blockIdx.x * blockDim.x + threadIdx.x;
    if (t >= (i64)n * B) return;
    const int b = (int)(t / n), j = (int)(t % n);
    if (!active[b]) return;
    const i64 o = (i64)b * n + j;
    const double L = lb[o], U = ub[o];
    if (L == U) return;
    double nl = L, nu = U;
    for (int k = cp[j]; k < cp[j + 1]; ++k) {
        const double a = cv[k];
        if (fabs(a) < 1e-9) continue;
        const int i = ri[k];
        const i64 q = (i64)b * m + i;
        const double u = rub[i], l = rlb[i];
        if (fin(u)) {
            const double own = a > 0 ? L : U;
            const int c = ninfmin[q];
            double rest = 0;
            bool ok = true;
            if (c == 0)
                rest = minact[q] - a * own;
            else if (c == 1 && !fin(own))
                rest = minact[q];
            else
                ok = false;
            if (ok) {
                const double v = (u - rest) / a;
                if (a > 0)
                    nu = fmin(nu, v);
                else
                    nl = fmax(nl, v);
            }
        }
        if (fin(l)) {
            const double own = a > 0 ? U : L;
            const int c = ninfmax[q];
            double rest = 0;
            bool ok = true;
            if (c == 0)
                rest = maxact[q] - a * own;
            else if (c == 1 && !fin(own))
                rest = maxact[q];
            else
                ok = false;
            if (ok) {
                const double v = (l - rest) / a;
                if (a > 0)
                    nl = fmax(nl, v);
                else
                    nu = fmin(nu, v);
            }
        }
    }
    if (isint[j]) {
        nl = ceil(nl - 1e-6);
        nu = floor(nu + 1e-6);
    }
    const double range = fin(L) && fin(U) ? U - L : 1e300;
    const double step =
        isint[j] ? 0.5 : 1e-3 * fmax(1.0, fmin(range, fabs(nl) + fabs(nu)));
    const bool up_l = nl > L + step && fabs(nl) < 1e15;
    const bool up_u = nu < U - step && fabs(nu) < 1e15;
    if (!up_l && !up_u) return;
    double NL = up_l ? nl : L, NU = up_u ? nu : U;
    if (NL > NU + 1e-6 * (1 + fabs(NU))) {
        infeas[b] = 1;
        return;
    }
    if (NL > NU) NL = NU;
    lb[o] = NL;
    ub[o] = NU;
    changed[b] = 1;
}

// Batch b <- base bounds with column fix_col[b] set to [fix_lo[b], fix_hi[b]]
// (fix_col < 0: unchanged): the start of a probe or child.
extern "C" __global__ void
prop_init(int n, int B, const double *lb0, const double *ub0,
    const int *fix_col, const double *fix_lo, const double *fix_hi, double *lb,
    double *ub, int *active, int *changed, int *infeas)
{
    const i64 t = (i64)blockIdx.x * blockDim.x + threadIdx.x;
    if (t >= (i64)n * B) return;
    const int b = (int)(t / n), j = (int)(t % n);
    double L = lb0[j], U = ub0[j];
    if (fix_col[b] == j) {
        L = fmax(L, fix_lo[b]);
        U = fmin(U, fix_hi[b]);
    }
    lb[t] = L;
    ub[t] = U;
    if (j == 0) {
        active[b] = 1;
        changed[b] = 0;
        infeas[b] = 0;
    }
}

// Double probing: batches 2k and 2k+1 probed one binary at 0 and at 1. For
// every column, the bounds valid whatever the binary takes: those of the
// feasible probe when the other emptied the domain, else the weaker of the
// two; the tightest over all pairs (each is valid on its own).
extern "C" __global__ void
probe_merge(int n, int K, const double *lb0, const double *ub0,
    const double *lb, const double *ub, const int *infeas, double *out_lb,
    double *out_ub)
{
    const int t = blockIdx.x * blockDim.x + threadIdx.x;
    if (t >= n) return;
    double L = lb0[t], U = ub0[t];
    for (int k = 0; k < K; ++k) {
        const int i0 = infeas[2 * k], i1 = infeas[2 * k + 1];
        const i64 o0 = (i64)(2 * k) * n + t, o1 = (i64)(2 * k + 1) * n + t;
        if (i0 && i1)
            continue; // reported by the host: the problem is infeasible
        if (i0) {
            L = fmax(L, lb[o1]);
            U = fmin(U, ub[o1]);
        } else if (i1) {
            L = fmax(L, lb[o0]);
            U = fmin(U, ub[o0]);
        } else {
            L = fmax(L, fmin(lb[o0], lb[o1]));
            U = fmin(U, fmax(ub[o0], ub[o1]));
        }
    }
    out_lb[t] = L;
    out_ub[t] = U;
}

// active[b] = !infeas[b] && changed[b]; changed[b] = 0 (between rounds).
extern "C" __global__ void
prop_next(int B, int *active, int *changed, const int *infeas)
{
    const int b = blockIdx.x * blockDim.x + threadIdx.x;
    if (b >= B) return;
    active[b] = active[b] && !infeas[b] && changed[b];
    changed[b] = 0;
}

// ---- probing with frontiers
// -----------------------------------------------------
//
// B probe slots share one CSR/CSC matrix and a base domain. A slot's bounds
// and row activities live in dense B x n / B x m arrays that hold the base
// values except where the slot changed them. A round recomputes only the
// rows on the row frontier (rows of columns that changed) and then tightens
// only the columns of those rows (the column frontier); a tightened column
// puts its rows on the next row frontier. Bitsets keep every frontier free
// of duplicates; touched lists let a batch restore just the entries it
// changed. The work per probe is that of the sequential queue propagator.

struct ProbeData {
    int n, m, B;
    const int *rp, *ci;
    const double *va;
    const int *cp, *ri;
    const double *cv;
    const double *rlb, *rub;
    const unsigned char *isint;
    const double *blb, *bub;   // base bounds (n)
    const double *bmin, *bmax; // base activities (m)
    const int *bcmin, *bcmax;
    double *lb, *ub; // B x n
    double *mn, *mx; // B x m
    int *cmn, *cmx;
    unsigned *rbits, *cbits;   // frontier membership (B m, B n bits)
    unsigned *trbits, *tcbits; // touched membership
    unsigned *cf, *tc,
        *tr; // column frontier, touched columns / rows (codes b*n+j, b*m+i)
    unsigned *cnt; // [1] next row frontier, [2] column frontier, [3] tc, [4] tr
    int *infeas;   // B
    double *out_lb, *out_ub; // merged double-probing bounds (n)
};

__device__ __forceinline__ bool
set_bit(unsigned *bits, unsigned long long k) // true if it was clear
{
    const unsigned mask = 1u << (k & 31);
    return !(atomicOr(&bits[k >> 5], mask) & mask);
}

__device__ __forceinline__ void
clear_bit(unsigned *bits, unsigned long long k)
{ atomicAnd(&bits[k >> 5], ~(1u << (k & 31))); }

__device__ __forceinline__ void
push_rows_of(const ProbeData &d, int b, int j, unsigned *rf_next)
{
    for (int k = d.cp[j]; k < d.cp[j + 1]; ++k) {
        const unsigned long long code = (unsigned long long)b * d.m + d.ri[k];
        if (set_bit(d.rbits, code))
            rf_next[atomicAdd(&d.cnt[1], 1u)] = (unsigned)code;
    }
}

__device__ __forceinline__ void
touch_col(const ProbeData &d, unsigned long long code)
{
    if (set_bit(d.tcbits, code))
        d.tc[atomicAdd(&d.cnt[3], 1u)] = (unsigned)code;
}

// Slot b probes column pcol[b] within [plo[b], phi[b]]: its rows start the
// row frontier (rf_next).
extern "C" __global__ void
pb_start(ProbeData d, const int *pcol, const double *plo, const double *phi,
    unsigned *rf_next)
{
    const int b = blockIdx.x * blockDim.x + threadIdx.x;
    if (b >= d.B) return;
    const int j = pcol[b];
    const unsigned long long o = (unsigned long long)b * d.n + j;
    d.lb[o] = fmax(d.lb[o], plo[b]);
    d.ub[o] = fmin(d.ub[o], phi[b]);
    touch_col(d, o);
    push_rows_of(d, b, j, rf_next);
}

// One warp per row-frontier entry: the slot's activity of the row; the row
// may empty the domain; its columns join the column frontier when the row
// can imply bounds.
extern "C" __global__ void
pb_rows(ProbeData d, const unsigned *rf, unsigned nrf)
{
    const int lane = threadIdx.x & 31;
    const unsigned long long e =
        ((unsigned long long)blockIdx.x * blockDim.x + threadIdx.x) >> 5;
    if (e >= nrf) return;
    const unsigned code = rf[e];
    const int b = (int)(code / (unsigned)d.m), i = (int)(code % (unsigned)d.m);
    int dead = 0;
    if (lane == 0) {
        clear_bit(d.rbits, code);
        dead = d.infeas[b];
    }
    if (__shfl_sync(0xffffffffu, dead, 0))
        return; // uniform: the shuffles below need the whole warp
    const double *L = d.lb + (unsigned long long)b * d.n,
                 *U = d.ub + (unsigned long long)b * d.n;
    double smin = 0, smax = 0;
    int cmin = 0, cmax = 0;
    for (int k = d.rp[i] + lane; k < d.rp[i + 1]; k += 32) {
        const double a = d.va[k];
        const int j = d.ci[k];
        const double lo = a > 0 ? L[j] : U[j], hi = a > 0 ? U[j] : L[j];
        if (fin(lo))
            smin += a * lo;
        else
            ++cmin;
        if (fin(hi))
            smax += a * hi;
        else
            ++cmax;
    }
    smin = warp_sum(smin);
    smax = warp_sum(smax);
    cmin = warp_sum_i(cmin);
    cmax = warp_sum_i(cmax);
    smin = __shfl_sync(0xffffffffu, smin, 0);
    smax = __shfl_sync(0xffffffffu, smax, 0);
    cmin = __shfl_sync(0xffffffffu, cmin, 0);
    cmax = __shfl_sync(0xffffffffu, cmax, 0);
    const double u = d.rub[i], l = d.rlb[i];
    if (lane == 0) {
        d.mn[code] = smin;
        d.mx[code] = smax;
        d.cmn[code] = cmin;
        d.cmx[code] = cmax;
        if (set_bit(d.trbits, code)) d.tr[atomicAdd(&d.cnt[4], 1u)] = code;
        if ((fin(u) && cmin == 0 && smin > u + 1e-6 * (1 + fabs(u))) ||
            (fin(l) && cmax == 0 && smax < l - 1e-6 * (1 + fabs(l))))
            d.infeas[b] = 1;
    }
    const bool use = (fin(u) && cmin <= 1) || (fin(l) && cmax <= 1);
    if (!use) return;
    for (int k = d.rp[i] + lane; k < d.rp[i + 1]; k += 32) {
        const unsigned long long c = (unsigned long long)b * d.n + d.ci[k];
        if (d.lb[c] == d.ub[c]) continue;
        if (set_bit(d.cbits, c)) d.cf[atomicAdd(&d.cnt[2], 1u)] = (unsigned)c;
    }
}

// One thread per column-frontier entry: implied bounds from the slot's row
// activities (rules of prop_tighten); a tightened column puts its rows on
// the next row frontier.
extern "C" __global__ void
pb_cols(ProbeData d, unsigned ncf, unsigned *rf_next)
{
    const unsigned long long e =
        (unsigned long long)blockIdx.x * blockDim.x + threadIdx.x;
    if (e >= ncf) return;
    const unsigned code = d.cf[e];
    clear_bit(d.cbits, code);
    const int b = (int)(code / (unsigned)d.n), j = (int)(code % (unsigned)d.n);
    if (d.infeas[b]) return;
    const double Lj = d.lb[code], Uj = d.ub[code];
    if (Lj == Uj) return;
    double nl = Lj, nu = Uj;
    for (int k = d.cp[j]; k < d.cp[j + 1]; ++k) {
        const double a = d.cv[k];
        if (fabs(a) < 1e-9) continue;
        const int i = d.ri[k];
        const unsigned long long q = (unsigned long long)b * d.m + i;
        const double u = d.rub[i], l = d.rlb[i];
        if (fin(u)) {
            const double own = a > 0 ? Lj : Uj;
            const int c = d.cmn[q];
            double rest = 0;
            bool ok = true;
            if (c == 0)
                rest = d.mn[q] - a * own;
            else if (c == 1 && !fin(own))
                rest = d.mn[q];
            else
                ok = false;
            if (ok) {
                const double v = (u - rest) / a;
                if (a > 0)
                    nu = fmin(nu, v);
                else
                    nl = fmax(nl, v);
            }
        }
        if (fin(l)) {
            const double own = a > 0 ? Uj : Lj;
            const int c = d.cmx[q];
            double rest = 0;
            bool ok = true;
            if (c == 0)
                rest = d.mx[q] - a * own;
            else if (c == 1 && !fin(own))
                rest = d.mx[q];
            else
                ok = false;
            if (ok) {
                const double v = (l - rest) / a;
                if (a > 0)
                    nl = fmax(nl, v);
                else
                    nu = fmin(nu, v);
            }
        }
    }
    if (d.isint[j]) {
        nl = ceil(nl - 1e-6);
        nu = floor(nu + 1e-6);
    }
    const double range = fin(Lj) && fin(Uj) ? Uj - Lj : 1e300;
    const double step =
        d.isint[j] ? 0.5 : 1e-3 * fmax(1.0, fmin(range, fabs(nl) + fabs(nu)));
    const bool up_l = nl > Lj + step && fabs(nl) < 1e15;
    const bool up_u = nu < Uj - step && fabs(nu) < 1e15;
    if (!up_l && !up_u) return;
    double NL = up_l ? nl : Lj, NU = up_u ? nu : Uj;
    if (NL > NU + 1e-6 * (1 + fabs(NU))) {
        d.infeas[b] = 1;
        return;
    }
    if (NL > NU) NL = NU;
    d.lb[code] = NL;
    d.ub[code] = NU;
    touch_col(d, code);
    push_rows_of(d, b, j, rf_next);
}

__device__ __forceinline__ void
atomic_max_d(double *a, double v)
{
    unsigned long long *p = (unsigned long long *)a, old = *p;
    while (__longlong_as_double((long long)old) < v) {
        const unsigned long long prev =
            atomicCAS(p, old, (unsigned long long)__double_as_longlong(v));
        if (prev == old) break;
        old = prev;
    }
}

__device__ __forceinline__ void
atomic_min_d(double *a, double v)
{
    unsigned long long *p = (unsigned long long *)a, old = *p;
    while (__longlong_as_double((long long)old) > v) {
        const unsigned long long prev =
            atomicCAS(p, old, (unsigned long long)__double_as_longlong(v));
        if (prev == old) break;
        old = prev;
    }
}

// Double probing over the touched columns: slots 2k and 2k+1 probed one
// binary at 0 and 1. A feasible slot whose partner emptied its domain fixes
// the binary, so all its bounds hold; two feasible slots give the weaker of
// their bounds. out_lb / out_ub collect the tightest over all pairs.
extern "C" __global__ void
pb_merge(ProbeData d, unsigned ntc)
{
    const unsigned long long e =
        (unsigned long long)blockIdx.x * blockDim.x + threadIdx.x;
    if (e >= ntc) return;
    const unsigned code = d.tc[e];
    const int b = (int)(code / (unsigned)d.n), j = (int)(code % (unsigned)d.n);
    if (d.infeas[b]) return;
    const int o = b ^ 1;
    const unsigned long long oc = (unsigned long long)o * d.n + j;
    double L = d.lb[code], U = d.ub[code];
    if (!d.infeas[o]) {
        L = fmin(L, d.lb[oc]);
        U = fmax(U, d.ub[oc]);
    }
    if (L > d.blb[j]) atomic_max_d(&d.out_lb[j], L);
    if (U < d.bub[j]) atomic_min_d(&d.out_ub[j], U);
}

// Restores the touched entries of every slot to the base domain.
extern "C" __global__ void
pb_reset(ProbeData d, unsigned ntc, unsigned ntr)
{
    const unsigned long long e =
        (unsigned long long)blockIdx.x * blockDim.x + threadIdx.x;
    if (e < ntc) {
        const unsigned code = d.tc[e];
        const int j = (int)(code % (unsigned)d.n);
        d.lb[code] = d.blb[j];
        d.ub[code] = d.bub[j];
        clear_bit(d.tcbits, code);
    }
    if (e < ntr) {
        const unsigned code = d.tr[e];
        const int i = (int)(code % (unsigned)d.m);
        d.mn[code] = d.bmin[i];
        d.mx[code] = d.bmax[i];
        d.cmn[code] = d.bcmin[i];
        d.cmx[code] = d.bcmax[i];
        clear_bit(d.trbits, code);
    }
}

// Fills the B slots with the base domain and activities (once per pass).
extern "C" __global__ void
pb_fill(ProbeData d)
{
    const unsigned long long t =
        (unsigned long long)blockIdx.x * blockDim.x + threadIdx.x;
    const unsigned long long nn = (unsigned long long)d.B * d.n,
                             mm = (unsigned long long)d.B * d.m;
    if (t < nn) {
        const int j = (int)(t % (unsigned long long)d.n);
        d.lb[t] = d.blb[j];
        d.ub[t] = d.bub[j];
    }
    if (t < mm) {
        const int i = (int)(t % (unsigned long long)d.m);
        d.mn[t] = d.bmin[i];
        d.mx[t] = d.bmax[i];
        d.cmn[t] = d.bcmin[i];
        d.cmx[t] = d.bcmax[i];
    }
}

// ---- feasibility jump
// ------------------------------------------------------------

struct FjData {
    int n, m;
    const int *rp, *ci; // CSR
    const double *va;
    const int *cp, *ri; // CSC
    const double *cv;
    const double *rlb, *rub, *lb, *ub;
    const unsigned char *isint;
    // per walker w: x[w n + j], act/wt/vpos[w m + i], vl[w m + k], vcnt[w]
    double *x, *act, *wt;
    int *vl, *vpos, *vcnt, *status;
    unsigned long long *rng;
    long long *moves;
};

__device__ __forceinline__ unsigned long long
xorshift(unsigned long long &s)
{
    s ^= s << 13;
    s ^= s >> 7;
    s ^= s << 17;
    return s;
}

__device__ __forceinline__ double
row_viol(double a, double l, double u)
{
    if (a > u) return a - u;
    if (a < l) return l - a;
    return 0.0;
}

__device__ __forceinline__ bool
violated(double a, double l, double u)
{ return a > u + 1e-6 * (1 + fabs(u)) || a < l - 1e-6 * (1 + fabs(l)); }

// Activities and violated-row lists of every walker (one block per walker).
extern "C" __global__ void
fj_init(FjData d)
{
    const int w = blockIdx.x;
    const double *x = d.x + (i64)w * d.n;
    double *act = d.act + (i64)w * d.m, *wt = d.wt + (i64)w * d.m;
    int *vl = d.vl + (i64)w * d.m, *vpos = d.vpos + (i64)w * d.m;
    if (threadIdx.x == 0) d.vcnt[w] = 0;
    __syncthreads();
    for (int i = threadIdx.x; i < d.m; i += blockDim.x) {
        double a = 0;
        for (int k = d.rp[i]; k < d.rp[i + 1]; ++k)
            a += d.va[k] * x[d.ci[k]];
        act[i] = a;
        wt[i] = 1.0;
        if (violated(a, d.rlb[i], d.rub[i])) {
            const int p = atomicAdd(&d.vcnt[w], 1);
            vl[p] = i;
            vpos[i] = p;
        } else {
            vpos[i] = -1;
        }
    }
}

#define FJ_CAND 32
#define FJ_VALS 128

// Up to `steps` moves of every walker that is still searching.
extern "C" __global__ void
fj_run(FjData d, int steps)
{
    const int w = blockIdx.x;
    const int lane = threadIdx.x & 31, warp = threadIdx.x >> 5,
              nwarps = blockDim.x >> 5;
    double *x = d.x + (i64)w * d.n;
    double *act = d.act + (i64)w * d.m, *wt = d.wt + (i64)w * d.m;
    int *vl = d.vl + (i64)w * d.m, *vpos = d.vpos + (i64)w * d.m;
    __shared__ int s_row, s_ncand, s_cand[FJ_CAND], s_done;
    __shared__ double s_score[FJ_CAND], s_val[FJ_CAND];
    __shared__ int s_best;
    __shared__ unsigned long long s_rng;
    if (threadIdx.x == 0) s_rng = d.rng[w];
    if (d.status[w] != 0) return;
    for (int step = 0; step < steps; ++step) {
        // 1. a violated row, dropping satisfied entries met on the way
        if (threadIdx.x == 0) {
            s_done = 0;
            s_row = -1;
            while (d.vcnt[w] > 0) {
                const int p =
                    (int)(xorshift(s_rng) % (unsigned long long)d.vcnt[w]);
                const int r = vl[p];
                if (violated(act[r], d.rlb[r], d.rub[r])) {
                    s_row = r;
                    break;
                }
                const int last = vl[d.vcnt[w] - 1];
                vl[p] = last;
                vpos[last] = p;
                vpos[r] = -1;
                d.vcnt[w] -= 1;
            }
            if (s_row < 0) s_done = 1;
            // 2. candidate columns of the row (a sample of a long row)
            if (s_row >= 0) {
                const int b0 = d.rp[s_row], len = d.rp[s_row + 1] - b0;
                int k = 0;
                if (len <= FJ_CAND) {
                    for (int t = 0; t < len; ++t)
                        s_cand[k++] = d.ci[b0 + t];
                } else {
                    for (int t = 0; t < FJ_CAND; ++t)
                        s_cand[k++] = d.ci[b0 + (int)(xorshift(s_rng) %
                                                      (unsigned long long)len)];
                }
                s_ncand = k;
            }
        }
        __syncthreads();
        if (s_done) {
            if (threadIdx.x == 0) d.status[w] = 1;
            break;
        }
        // 3. jump value and score of each candidate (one warp per candidate)
        for (int c = warp; c < s_ncand; c += nwarps) {
            const int j = s_cand[c];
            const double xj = x[j], lj = d.lb[j], uj = d.ub[j];
            double best_cost = 1e300, best_v = xj, base = 0;
            if (lj < uj) {
                const int c0 = d.cp[j], c1 = d.cp[j + 1];
                // the cost at the current value
                double part = 0;
                for (int k = c0 + lane; k < c1; k += 32) {
                    const int i = d.ri[k];
                    part += wt[i] * row_viol(act[i], d.rlb[i], d.rub[i]);
                }
                base = warp_sum(part);
                base = __shfl_sync(0xffffffffu, base, 0);
                // breakpoints: row k of the column made tight at its lower or
                // upper side, rounded for an integer column (4 values per
                // entry), plus the two bounds; lanes over the values
                const int nent = min(c1 - c0, FJ_VALS / 4);
                const int nval = 4 * nent + 2;
                for (int v0 = lane; v0 < nval; v0 += 32) {
                    double v;
                    if (v0 >= 4 * nent) {
                        v = v0 == 4 * nent ? lj : uj;
                    } else {
                        const int k = c0 + v0 / 4, side = (v0 / 2) & 1,
                                  rnd = v0 & 1;
                        const int i = d.ri[k];
                        const double a = d.cv[k], rest = act[i] - a * xj;
                        const double rhs = side ? d.rub[i] : d.rlb[i];
                        v = fin(rhs) && fabs(a) > 1e-12 ? (rhs - rest) / a : xj;
                        if (d.isint[j])
                            v = rnd ? ceil(v - 1e-9) : floor(v + 1e-9);
                        else if (rnd)
                            v = xj; // continuous: one value per side
                    }
                    v = fmin(fmax(v, lj), uj);
                    if (!fin(v) || v == xj) continue;
                    double cost = 0;
                    for (int k = c0; k < c1; ++k) {
                        const int i = d.ri[k];
                        cost += wt[i] * row_viol(act[i] + d.cv[k] * (v - xj),
                                            d.rlb[i], d.rub[i]);
                    }
                    if (cost < best_cost) {
                        best_cost = cost;
                        best_v = v;
                    }
                }
            }
            // warp minimum of (cost, value)
            for (int off = 16; off > 0; off >>= 1) {
                const double oc = __shfl_down_sync(0xffffffffu, best_cost, off);
                const double ov = __shfl_down_sync(0xffffffffu, best_v, off);
                if (oc < best_cost) {
                    best_cost = oc;
                    best_v = ov;
                }
            }
            if (lane == 0) {
                s_score[c] = best_cost < 1e299 ? base - best_cost : -1.0;
                s_val[c] = best_v;
            }
        }
        __syncthreads();
        // 4. the best candidate (ties broken at random)
        if (threadIdx.x == 0) {
            int bi = -1;
            double bs = 1e-9;
            for (int c = 0; c < s_ncand; ++c)
                if (s_score[c] > bs ||
                    (s_score[c] == bs && bi >= 0 && (xorshift(s_rng) & 1))) {
                    bs = s_score[c];
                    bi = c;
                }
            s_best = bi;
            if (bi < 0) wt[s_row] += 1.0; // local minimum: weigh the row up
        }
        __syncthreads();
        // 5. the move: x_j and the activities of its rows
        if (s_best >= 0) {
            const int j = s_cand[s_best];
            const double delta = s_val[s_best] - x[j];
            __syncthreads();
            for (int k = d.cp[j] + threadIdx.x; k < d.cp[j + 1];
                k += blockDim.x) {
                const int i = d.ri[k];
                const double a = act[i] + d.cv[k] * delta;
                act[i] = a;
                if (vpos[i] < 0 && violated(a, d.rlb[i], d.rub[i])) {
                    const int p = atomicAdd(&d.vcnt[w], 1);
                    vl[p] = i;
                    vpos[i] = p;
                }
            }
            __syncthreads();
            if (threadIdx.x == 0) {
                x[j] = s_val[s_best];
                d.moves[w] += 1;
            }
        }
        __syncthreads();
    }
    if (threadIdx.x == 0) d.rng[w] = s_rng;
}

// ---- batched PDHG
// ----------------------------------------------------------------

// Primal step of every LP: x <- proj_[l,u](x - tau (c - A^T y)), xbar = 2 x+ -
// x, and (eval) the column part of the Lagrangian bound of y into acc[b]
// (bad[b] set when a needed bound is infinite). One warp per column.
extern "C" __global__ void
pd_primal(int n, int B, const int *cp, const int *ri, const double *cv,
    const double *c, const double *L, const double *U, const double *y,
    double *x, double *xbar, double tau, int eval, double *acc, int *bad)
{
    __shared__ double sh[MILP_BLOCK / 32][32];
    const int lane = threadIdx.x & 31, warp = threadIdx.x >> 5;
    const i64 j64 = ((i64)blockIdx.x * blockDim.x + threadIdx.x) >> 5;
    double contrib = 0;
    if (j64 < n && lane < B) {
        const int j = (int)j64;
        double s = 0;
        for (int k = cp[j]; k < cp[j + 1]; ++k)
            s += cv[k] * y[(i64)ri[k] * B + lane];
        const double r = c[j] - s;
        const i64 o = (i64)j * B + lane;
        const double l = L[o], u = U[o];
        if (eval) {
            if (r > 0) {
                if (fin(l))
                    contrib = r * l;
                else
                    atomicOr(&bad[lane], 1);
            } else if (r < 0) {
                if (fin(u))
                    contrib = r * u;
                else
                    atomicOr(&bad[lane], 1);
            }
        }
        const double xo = x[o];
        const double xn = fmin(fmax(xo - tau * r, l), u);
        x[o] = xn;
        xbar[o] = 2 * xn - xo;
    }
    if (!eval) return;
    sh[warp][lane] = contrib;
    __syncthreads();
    if (warp == 0 && lane < B) {
        double s = 0;
        for (int w = 0; w < (int)(blockDim.x >> 5); ++w)
            s += sh[w][lane];
        atomicAdd(&acc[lane], s);
    }
}

// Dual step of every LP: y <- prox(y - sigma A xbar) for rl <= A x <= ru, and
// (eval) the row part of the Lagrangian bound of the old y into acc[b].
extern "C" __global__ void
pd_dual(int m, int B, const int *rp, const int *ci, const double *va,
    const double *rl, const double *ru, const double *xbar, double *y,
    double sigma, int eval, double *acc)
{
    __shared__ double sh[MILP_BLOCK / 32][32];
    const int lane = threadIdx.x & 31, warp = threadIdx.x >> 5;
    const i64 i64r = ((i64)blockIdx.x * blockDim.x + threadIdx.x) >> 5;
    double contrib = 0;
    if (i64r < m && lane < B) {
        const int i = (int)i64r;
        double s = 0;
        for (int k = rp[i]; k < rp[i + 1]; ++k)
            s += va[k] * xbar[(i64)ci[k] * B + lane];
        const i64 o = (i64)i * B + lane;
        const double yo = y[o], l = rl[i], u = ru[i];
        if (eval) contrib = yo > 0 ? yo * l : (yo < 0 ? yo * u : 0.0);
        const double v = yo - sigma * s;
        double yn = 0;
        if (fin(l) && v + sigma * l > 0)
            yn = v + sigma * l;
        else if (fin(u) && v + sigma * u < 0)
            yn = v + sigma * u;
        y[o] = yn;
    }
    if (!eval) return;
    sh[warp][lane] = contrib;
    __syncthreads();
    if (warp == 0 && lane < B) {
        double s = 0;
        for (int w = 0; w < (int)(blockDim.x >> 5); ++w)
            s += sh[w][lane];
        atomicAdd(&acc[lane], s);
    }
}

// best[b] = max(best[b], acc[b]) unless bad[b]; acc = 0, bad = 0.
extern "C" __global__ void
pd_take(int B, double *acc, int *bad, double *best)
{
    const int b = threadIdx.x;
    if (b >= B) return;
    if (!bad[b] && acc[b] > best[b]) best[b] = acc[b];
    acc[b] = 0;
    bad[b] = 0;
}
