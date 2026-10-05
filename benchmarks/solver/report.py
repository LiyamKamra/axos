"""Merge bench_lp / bench_highs RESULT lines into one table.

    report.py axos.txt highs1.txt [highs2.txt ...]

Columns are ms (or ms+status when not optimal); the last columns give the
relative objective difference of each PDLP run against the HiGHS reference."""
import collections
import sys

rows = collections.OrderedDict()   # instance -> {solver: (ms, it, status, obj)}
for path in sys.argv[1:]:
    for line in open(path):
        if not line.startswith("RESULT lp"):
            continue
        t = line.split()
        solver, inst, ms, it = t[2], t[3], t[4], t[5]
        status, obj = "_".join(t[6:-1]), t[-1]
        rows.setdefault(inst, {})[solver] = (float(ms), int(it), status, float(obj))

order = ["pdlp_cpu_1e-4", "pdlp_cuda_1e-4", "pdlp_cpu_1e-6", "pdlp_cuda_1e-6",
         "ipm_cpu_1e-8", "ipm_cuda_1e-8", "highs_auto", "highs_ds", "highs_ipm"]
insts = sorted(rows, key=lambda k: (k.rstrip("0123456789km"), int("".join(c for c in k if c.isdigit()) or 0)))
print(f"{'instance':<14}" + "".join(f"{s:>18}" for s in order))
for inst in insts:
    r = rows[inst]
    cells = []
    for s in order:
        if s not in r:
            cells.append(f"{'-':>18}")
            continue
        ms, it, st, obj = r[s]
        st = {"time_limit": "limit", "iteration_limit": "limit"}.get(st, st)
        txt = f"{ms:.0f}" if st == "optimal" else f"{ms:.0f} ({st})"
        cells.append(f"{txt:>18}")
    print(f"{inst:<14}" + "".join(cells))

print("\nrelative objective difference vs HiGHS (dual simplex / IPM optimum):")
print(f"{'instance':<14}" + "".join(f"{s:>16}" for s in order[:6]))
for inst in insts:
    r = rows[inst]
    ref = None
    for h in ("highs_ds", "highs_auto", "highs_ipm"):
        if h in r and r[h][2] == "optimal":
            ref = r[h][3]
            break
    cells = []
    for s in order[:6]:
        if ref is None or s not in r or r[s][2] != "optimal":
            cells.append(f"{'-':>16}")
        else:
            cells.append(f"{abs(r[s][3] - ref) / (1 + abs(ref)):>16.1e}")
    print(f"{inst:<14}" + "".join(cells))
