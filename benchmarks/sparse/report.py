"""Merge RESULT lines from several benchmark runs into one table (ms, lower is
better). Usage: report.py run1.txt [run2.txt ...]"""
import collections
import sys

res = collections.OrderedDict()  # (op, matrix) -> {lib: ms}
libs = []
for path in sys.argv[1:]:
    for line in open(path):
        if not line.startswith("RESULT"):
            continue
        _, lib, op, matrix, ms = line.split()
        if lib not in libs:
            libs.append(lib)
        res.setdefault((op, matrix), {})[lib] = float(ms)

w = max(len(x) for x in libs) + 1
print(f"{'op':<20}{'matrix':<10}" + "".join(f"{l:>{w + 2}}" for l in libs))
for (op, matrix), row in res.items():
    cells = "".join(
        f"{row[l]:>{w + 2}.3f}" if l in row else f"{'-':>{w + 2}}" for l in libs)
    print(f"{op:<20}{matrix:<10}{cells}")
