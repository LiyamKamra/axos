"""Compares a distributed HPR-QP run (run_mpi.sh) with a single-GPU run of the
same problems: status, iterations and objective per problem, and whether
every solution reported optimal passes the common metric (the CSVs hold the
residuals re-scored by `qpbench.py score`).

    python benchmarks/qp/compare_mpi.py reference.csv mpi.csv [--tol 1e-6]
"""
import argparse
import csv


def load(path):
    return {r["problem"]: r for r in csv.DictReader(open(path, newline=""))}


def num(r, k):
    try:
        return float(r[k])
    except (KeyError, TypeError, ValueError):
        return float("nan")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("reference")
    ap.add_argument("mpi")
    ap.add_argument("--tol", type=float, default=1e-6)
    a = ap.parse_args()
    ref, mpi = load(a.reference), load(a.mpi)
    names = sorted(mpi)
    solver = next(iter(mpi.values()))["solver"] if mpi else "?"
    both, same_it, its, objd = 0, 0, [], []
    only_ref, only_mpi, failed_check = [], [], []
    print(f"{'problem':14s} {'reference':>22s} {solver:>24s}  rel. obj. diff")
    for p in names:
        r, m = ref.get(p), mpi[p]
        res = max(num(m, "rel_primal"), num(m, "rel_dual"), num(m, "rel_gap"))
        if m["status"] == "optimal" and not res <= 1.1 * a.tol:
            failed_check.append(p)
        if r is None:
            print(f"{p:14s} {'(not in reference)':>22s} {m['status']:>14s} {m['iterations']:>9s}")
            continue
        rs, ms = r["status"], m["status"]
        d = ""
        if rs == "optimal" and ms == "optimal":
            both += 1
            ri, mi = int(r["iterations"]), int(m["iterations"])
            same_it += ri == mi
            its.append(abs(mi - ri) / max(ri, 1))
            ro, mo = num(r, "objective"), num(m, "objective")
            od = abs(ro - mo) / max(1.0, abs(ro))
            objd.append(od)
            d = f"{od:.1e}"
        elif rs == "optimal":
            only_ref.append(p)
        elif ms == "optimal":
            only_mpi.append(p)
        print(f"{p:14s} {rs:>12s} {r['iterations']:>9s} {ms:>14s} {m['iterations']:>9s}  {d}")
    print()
    print(f"{len(names)} problems; optimal in both runs: {both}")
    if both:
        print(f"  same iteration count: {same_it} of {both}; "
              f"largest relative iteration difference {max(its):.1%}")
        print(f"  largest relative objective difference {max(objd):.1e} "
              f"(median {sorted(objd)[len(objd) // 2]:.1e})")
    print(f"optimal only in the reference: {len(only_ref)} {' '.join(only_ref)}")
    print(f"optimal only with MPI: {len(only_mpi)} {' '.join(only_mpi)}")
    print(f"reported optimal but above 1.1 x tol on the common metric: "
          f"{len(failed_check)} {' '.join(failed_check)}")


if __name__ == "__main__":
    main()
