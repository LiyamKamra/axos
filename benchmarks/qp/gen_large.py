#!/usr/bin/env python3
"""Large generated convex QPs for the GPU benchmark, written as .axqp.

The seven problem families of the OSQP benchmark suite (B. Stellato et al.,
"OSQP: an operator splitting solver for quadratic programs", Math. Prog.
Comp. 2020, Appendix), which the HPR-QP and PDHCG papers also generate:
random, eq (equality constrained), portfolio, control (MPC), huber, svm and
lasso. The data are sparser than in the paper so that the larger sizes fit
in a 4 GB GPU and in the memory of the CPU interior-point comparators.

  min 1/2 x^T Q x + c^T x   s.t.  l <= A x <= u,  lb <= x <= ub

  python gen_large.py --out DIR [--sizes M,L] [--only random,svm]

Everything is seeded, so the files are reproducible. Sizes: M about 1e6 and
L about 4e6 nonzeros in A and Q together.
"""
import argparse
import os
import struct

import numpy as np
import scipy.sparse as sp

INF = np.inf


def sprandn(rng, m, n, density):
    """m x n sparse matrix, `density` fraction of N(0, 1) entries."""
    return sp.random(m, n, density=density, format="csr", random_state=rng,
                     data_rvs=rng.standard_normal)


def write_axqp(path, Q, c, A, l, u, lb, ub, offset=0.0):
    A = sp.csr_matrix(A, dtype=np.float64)
    Q = sp.csr_matrix(Q, dtype=np.float64)
    Q = sp.csr_matrix((Q + Q.T) * 0.5)  # exactly symmetric, both triangles
    A.sort_indices()
    Q.sort_indices()
    m, n = A.shape
    assert Q.shape == (n, n) and len(c) == n and len(l) == m
    with open(path, "wb") as f:
        f.write(b"AXQP")
        f.write(struct.pack("<i4qdi", 1, n, m, A.nnz, Q.nnz, offset, 0))
        for M in (A, Q):
            f.write(M.indptr.astype("<i4").tobytes())
            f.write(M.indices.astype("<i4").tobytes())
            f.write(M.data.astype("<f8").tobytes())
        for v in (c, lb, ub, l, u):
            f.write(np.asarray(v, dtype="<f8").tobytes())
    return n, m, A.nnz, Q.nnz


# ---- families (OSQP benchmark suite, Appendix A) --------------------------------

def random_qp(rng, n, dens):
    # P = M M^T + 1e-2 I, m = 10 n constraints l <= A x <= u
    m = 10 * n
    M = sprandn(rng, n, n, dens)
    Q = M @ M.T + 1e-2 * sp.identity(n)
    A = sprandn(rng, m, n, dens)
    c = rng.standard_normal(n)
    l, u = -rng.random(m), rng.random(m)
    return Q, c, A, l, u, np.full(n, -INF), np.full(n, INF)


def eq_qp(rng, n, dens):
    # P = M M^T + 1e-2 I, A x = b with m = n / 2 (b from a random point)
    m = n // 2
    M = sprandn(rng, n, n, dens)
    Q = M @ M.T + 1e-2 * sp.identity(n)
    A = sprandn(rng, m, n, dens)
    b = A @ rng.standard_normal(n)
    return Q, rng.standard_normal(n), A, b, b.copy(), np.full(n, -INF), np.full(n, INF)


def portfolio(rng, n, k, dens):
    # min x^T D x + y^T y - mu^T x / gamma  s.t.  y = F^T x, 1^T x = 1, 0 <= x <= 1
    F = sprandn(rng, n, k, dens)
    D = sp.diags(rng.random(n) * np.sqrt(k))
    mu = rng.standard_normal(n)
    gamma = 1.0
    Q = 2.0 * sp.block_diag([D, sp.identity(k)], format="csr")
    c = np.concatenate([-mu / gamma, np.zeros(k)])
    A = sp.vstack([sp.hstack([F.T, -sp.identity(k)]),
                   sp.hstack([np.ones((1, n)), sp.csr_matrix((1, k))])], format="csr")
    l = np.concatenate([np.zeros(k), [1.0]])
    lb = np.concatenate([np.zeros(n), np.full(k, -INF)])
    ub = np.concatenate([np.ones(n), np.full(k, INF)])
    return Q, c, A, l, l.copy(), lb, ub


def control(rng, nx, T, row_nnz):
    # MPC over T steps: x_{t+1} = Ad x_t + Bd u_t, x_0 given, box bounds;
    # cost sum x^T Qx x + u^T R u + x_T^T QT x_T (nu = nx / 2 inputs)
    nu = nx // 2
    # stable dynamics (spectral radius about 0.95 + 0.04), so x_0 is steerable
    # within the bounds and the problem is feasible
    Ad = 0.95 * sp.identity(nx) + sprandn(rng, nx, nx, row_nnz / nx) * (0.04 / np.sqrt(row_nnz))
    Bd = sprandn(rng, nx, nu, row_nnz / nx)
    Qx = sp.diags(rng.random(nx) * 10.0)
    R = 0.1 * sp.identity(nu)
    QT = Qx * 10.0
    Q = sp.block_diag([sp.kron(sp.identity(T), Qx), QT, sp.kron(sp.identity(T), R)],
                      format="csr") * 2.0
    Ax = sp.kron(sp.identity(T + 1), -sp.identity(nx)) + \
        sp.kron(sp.eye(T + 1, k=-1), Ad)
    Bu = sp.kron(sp.vstack([sp.csr_matrix((1, T)), sp.identity(T)]), Bd)
    A = sp.hstack([Ax, Bu], format="csr")
    x0 = rng.standard_normal(nx) * 0.5
    b = np.concatenate([-x0, np.zeros(T * nx)])
    xmax, umax = 5.0, 1.0
    N = (T + 1) * nx + T * nu
    lb = np.concatenate([np.full((T + 1) * nx, -xmax), np.full(T * nu, -umax)])
    ub = -lb
    lb[:nx], ub[:nx] = -np.inf, np.inf  # x_0 is fixed by the dynamics rows
    return Q, np.zeros(N), A, b, b.copy(), lb, ub


def huber(rng, n, m, dens):
    # min u^T u + 2 1^T (r + s)  s.t.  A x - u - r + s = b,  r, s >= 0
    Ad = sprandn(rng, m, n, dens)
    xt = rng.standard_normal(n) / np.sqrt(n)
    noise = np.where(rng.random(m) < 0.95, rng.standard_normal(m) / 4.0,
                     rng.standard_normal(m) * 10.0)
    b = Ad @ xt + noise
    Im = sp.identity(m)
    A = sp.hstack([Ad, -Im, -Im, Im], format="csr")
    N = n + 3 * m
    Q = sp.block_diag([sp.csr_matrix((n, n)), 2.0 * Im, sp.csr_matrix((2 * m, 2 * m))],
                      format="csr")
    c = np.concatenate([np.zeros(n + m), 2.0 * np.ones(2 * m)])
    lb = np.concatenate([np.full(n + m, -INF), np.zeros(2 * m)])
    return Q, c, A, b, b.copy(), lb, np.full(N, INF)


def svm(rng, n, m, dens):
    # min x^T x + lam 1^T t  s.t.  t >= diag(b) A x + 1, t >= 0
    # two classes: entries N(+-1/n, 1/n) on a random pattern
    half = m // 2
    S = sp.random(m, n, density=dens, format="coo", random_state=rng,
                  data_rvs=rng.standard_normal)
    S.data = S.data / np.sqrt(n) + np.where(S.row < half, 1.0, -1.0) / n
    Ad = S.tocsr()
    bl = np.concatenate([np.ones(half), -np.ones(m - half)])
    lam = 1.0
    A = sp.hstack([-sp.diags(bl) @ Ad, sp.identity(m)], format="csr")
    Q = sp.block_diag([2.0 * sp.identity(n), sp.csr_matrix((m, m))], format="csr")
    c = np.concatenate([np.zeros(n), lam * np.ones(m)])
    lb = np.concatenate([np.full(n, -INF), np.zeros(m)])
    return Q, c, A, np.ones(m), np.full(m, INF), lb, np.full(n + m, INF)


def lasso(rng, n, m, dens):
    # min y^T y + lam 1^T t  s.t.  y = A x - b, -t <= x <= t
    Ad = sprandn(rng, m, n, dens)
    xt = np.where(rng.random(n) < 0.5, rng.standard_normal(n) / np.sqrt(n), 0.0)
    b = Ad @ xt + rng.standard_normal(m)
    lam = 0.2 * np.abs(Ad.T @ b).max()
    In, Im = sp.identity(n), sp.identity(m)
    A = sp.vstack([sp.hstack([Ad, -Im, sp.csr_matrix((m, n))]),
                   sp.hstack([In, sp.csr_matrix((n, m)), -In]),
                   sp.hstack([In, sp.csr_matrix((n, m)), In])], format="csr")
    l = np.concatenate([b, np.full(n, -INF), np.zeros(n)])
    u = np.concatenate([b, np.zeros(n), np.full(n, INF)])
    N = 2 * n + m
    Q = sp.block_diag([sp.csr_matrix((n, n)), 2.0 * Im, sp.csr_matrix((n, n))], format="csr")
    c = np.concatenate([np.zeros(n + m), lam * np.ones(n)])
    return Q, c, A, l, u, np.full(N, -INF), np.full(N, INF)


# name -> size -> generator call
FAMILIES = {
    "random": {"M": lambda r: random_qp(r, 20000, 2e-4), "L": lambda r: random_qp(r, 40000, 1.5e-4)},
    "eq": {"M": lambda r: eq_qp(r, 70000, 6e-5), "L": lambda r: eq_qp(r, 140000, 3.5e-5)},
    "portfolio": {"M": lambda r: portfolio(r, 20000, 200, 0.25),
                  "L": lambda r: portfolio(r, 40000, 400, 0.25)},
    "control": {"M": lambda r: control(r, 400, 100, 20), "L": lambda r: control(r, 800, 150, 20)},
    "huber": {"M": lambda r: huber(r, 1000, 50000, 0.02), "L": lambda r: huber(r, 2000, 100000, 0.02)},
    "svm": {"M": lambda r: svm(r, 1000, 50000, 0.02), "L": lambda r: svm(r, 2000, 100000, 0.02)},
    "lasso": {"M": lambda r: lasso(r, 2000, 20000, 0.02), "L": lambda r: lasso(r, 4000, 40000, 0.02)},
}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--sizes", default="M,L")
    ap.add_argument("--only")
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    fams = a.only.split(",") if a.only else list(FAMILIES)
    for size in a.sizes.split(","):
        for k, name in enumerate(fams):
            rng = np.random.default_rng(1000 * (k + 1) + ord(size[0]))
            Q, c, A, l, u, lb, ub = FAMILIES[name][size](rng)
            path = os.path.join(a.out, f"{name.upper()}_{size}.axqp")
            n, m, na, nq = write_axqp(path, Q, c, A, l, u, lb, ub)
            print(f"{os.path.basename(path)}: n {n} m {m} nnz(A) {na} nnz(Q) {nq}", flush=True)


if __name__ == "__main__":
    main()
