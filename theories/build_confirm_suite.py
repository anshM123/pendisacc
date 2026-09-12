"""Freeze a fresh Isaac confirmation suite BEFORE the theory screen's results are read.

Whatever theory the screen nominates, it has to predict Isaac outcomes it has
never seen. This writes 16 new conditions on the corrected asset and the list
of corrected-asset policies to score on them, with hashes. It refuses to
overwrite. No condition is chosen by looking at any theory's output.

  run.cmd theories/build_confirm_suite.py
"""

from __future__ import annotations

import glob
import hashlib
import json
import os
import sys

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT = os.path.join(ROOT, "xi", "confirm_theory")
MAN = os.path.join(OUT, "manifest.json")
LOGS = os.path.join(ROOT, "logs", "rsl_rl", "tip_swingup")
SEED = 20260913


def main() -> int:
    if os.path.exists(MAN):
        print("REFUSING: %s exists; the confirmation suite is frozen" % MAN)
        return 1
    os.makedirs(OUT, exist_ok=True)
    rng = np.random.default_rng(SEED)
    conds = []
    for i in range(6):                                    # arm-mass ratios only
        f = np.exp(rng.normal(0.0, 0.20, 3))
        conds.append(("CT_mass_%d" % i, {"mass_scale": [round(float(x), 4) for x in f]}))
    for name, xi in (("CT_tau_070", {"tau": 0.07}), ("CT_tau_135", {"tau": 0.135}),
                     ("CT_delay_8ms", {"delay_s": 0.008}), ("CT_delay_16ms", {"delay_s": 0.016})):
        conds.append((name, xi))
    for i in range(4):                                    # combined
        f = np.exp(rng.normal(0.0, 0.15, 3))
        conds.append(("CT_combo_%d" % i, {"mass_scale": [round(float(x), 4) for x in f],
                                          "tau": round(float(rng.uniform(0.08, 0.13)), 4),
                                          "delay_s": float(rng.choice([0.0, 0.004, 0.008]))}))
    # negative controls: quantities the theories say should not matter
    conds.append(("CT_ctrl_kv", {"kv": 240.0}))
    conds.append(("CT_ctrl_cart", {"cart_mass_scale": 1.2}))

    runs = []
    for pat in ("*_CORR_s1", "*_CORR_s2", "*_CORR_s3", "*_T5_orbit_s1", "*_T5_orbit_s2",
                "*_VOIDT5_orbit_s1", "*_VOIDT5_orbit_s2", "*_VOIDT5_orbit_s3", "*_VOIDT5_orbit_s4",
                "*_VOIDT5_orbit_s5", "*_VOIDT5_transverse_s1", "*_VOIDT5_transverse_s2"):
        ds = [d for d in sorted(glob.glob(os.path.join(LOGS, pat))) if glob.glob(os.path.join(d, "model_*.pt"))]
        if ds:
            d = max(ds, key=lambda x: len(glob.glob(os.path.join(x, "model_*.pt"))))
            runs.append(d)
    man = {"seed": SEED, "asset": "corrected", "conditions": [], "runs": [os.path.relpath(r, ROOT) for r in runs],
           "checkpoint_rule": "final checkpoint of each run (evaluate.py --run_list)"}
    for name, xi in conds:
        blob = json.dumps(xi, indent=1, sort_keys=True).encode("utf-8")
        with open(os.path.join(OUT, name + ".json"), "wb") as fh:
            fh.write(blob)
        man["conditions"].append({"name": name, "xi": xi, "sha256": hashlib.sha256(blob).hexdigest()})
    with open(os.path.join(OUT, "run_list.txt"), "w", encoding="utf-8") as fh:
        fh.write("\n".join(runs) + "\n")
    with open(MAN, "w", encoding="utf-8") as fh:
        json.dump(man, fh, indent=1)
    print("%d conditions x %d policies frozen in %s" % (len(conds), len(runs), OUT))
    return 0


if __name__ == "__main__":
    sys.exit(main())
