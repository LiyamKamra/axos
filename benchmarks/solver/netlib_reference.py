"""Reference optimal values for MPS files, from HiGHS (highspy).

    python netlib_reference.py [dir]   ->  lines: REF <name> <status> <objective> <ms>
"""
import glob
import os
import sys
import time

import highspy
import numpy as np

d = sys.argv[1] if len(sys.argv) > 1 else "build/netlib"
for path in sorted(glob.glob(os.path.join(d, "*.mps"))):
    name = os.path.basename(path)[:-4]
    h = highspy.Highs()
    h.setOptionValue("output_flag", False)
    h.setOptionValue("time_limit", 120.0)
    if h.readModel(path) != highspy.HighsStatus.kOk:
        print(f"REF {name} readfail nan 0", flush=True)
        continue
    lp = h.getLp()
    if any(v != highspy.HighsVarType.kContinuous for v in lp.integrality_):
        # solve the LP relaxation of integer models
        n = lp.num_col_
        h.changeColsIntegrality(n, np.arange(n, dtype=np.int32),
                                np.array([highspy.HighsVarType.kContinuous] * n))
    t0 = time.perf_counter()
    h.run()
    ms = (time.perf_counter() - t0) * 1000
    st = h.modelStatusToString(h.getModelStatus()).replace(" ", "_").lower()
    print(f"REF {name} {st} {h.getInfo().objective_function_value:.10e} {ms:.1f}", flush=True)
