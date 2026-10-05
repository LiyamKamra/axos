"""Compare run_mps output (any mix of methods) with HiGHS references.

    netlib_report.py reference.txt run1.txt [run2.txt ...]

One row per model: HiGHS time, then for every solver column its time in ms
(with a short status if it is not optimal) and, in the second table, its
relative objective error against HiGHS."""
import collections
import sys

ref = {}
for l in open(sys.argv[1]):
    _, n, st, obj, ms = l.split()
    ref[n] = (st, float(obj), float(ms))
rows = collections.OrderedDict()
solvers = []
for path in sys.argv[2:]:
    for l in open(path):
        if not l.startswith("RESULT mps"):
            continue
        t = l.split("#")[0].split()
        solver, name, ms, it = t[2], t[3], float(t[4]), int(t[5])
        st = "_".join(t[6:-4])
        obj, pr, dr, gap = map(float, t[-4:])
        if solver not in solvers:
            solvers.append(solver)
        rows.setdefault(name, {})[solver] = (ms, it, st, obj)

short = {"optimal": "", "infeasible": " infeas", "iteration_limit": " limit",
         "time_limit": " limit", "numerical_error": " numerr", "error": " error"}
solvers.sort(key=lambda s: (s.split("_")[0], s.split("_")[1] if "_" in s else ""))
W = 17
print(f"{'model':<10}{'HiGHS':>9}{'verdict':>11} |" + "".join(f"{s:>{W}}" for s in solvers))
score = collections.Counter()
for n in sorted(rows):
    r = ref.get(n, ("?", float("nan"), 0))
    cells = []
    for s in solvers:
        x = rows[n].get(s)
        if not x:
            cells.append(f"{'-':>{W}}")
            continue
        cells.append(f"{x[0]:.1f}{short.get(x[2], ' ' + x[2]):>{0}}".rjust(W))
        if r[0] == "optimal" and x[2] == "optimal" and abs(x[3] - r[1]) / (1 + abs(r[1])) < 1e-4:
            score[s] += 1
        elif r[0] == "infeasible" and x[2] == "infeasible":
            score[s] += 1
    print(f"{n:<10}{r[2]:>9.1f}{r[0][:10]:>11} |" + "".join(cells))
nfeas = sum(1 for v in ref.values() if v[0] == "optimal")
ninf = sum(1 for v in ref.values() if v[0] == "infeasible")
print(f"\nsolved correctly (optimal within 1e-4 rel., or infeasible), out of "
      f"{nfeas} feasible + {ninf} infeasible:")
print("  " + ", ".join(f"{s}: {score[s]}" for s in solvers))
print("\nrelative objective error vs HiGHS (optimal runs only):")
print(f"{'model':<10} |" + "".join(f"{s:>{W}}" for s in solvers))
for n in sorted(rows):
    r = ref.get(n, ("?", float("nan"), 0))
    if r[0] != "optimal":
        continue
    cells = []
    for s in solvers:
        x = rows[n].get(s)
        if x and x[2] == "optimal":
            cells.append(f"{abs(x[3] - r[1]) / (1 + abs(r[1])):.1e}".rjust(W))
        else:
            cells.append(f"{'-':>{W}}")
    print(f"{n:<10} |" + "".join(cells))
