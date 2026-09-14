"""Separation growth of parameter-close twins (experiments/fractal_ftle.py output). CPU only.

For each eps, the median over twin pairs of log d(t) is examined. Sensitive
dependence shows up as (i) a window where log d grows linearly in time at a rate
lambda that does not depend on eps, and (ii) curves for different eps offset by
roughly log of the eps ratio before saturating. eps = 0 twins show what the
simulator's own slot-to-slot nondeterminism produces.

  run.cmd tools/fractal_ftle_analyze.py
"""

from __future__ import annotations

import json
import os

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def main():
    z = np.load(os.path.join(ROOT, "results", "fractal", "ftle_CORR_s1.npz"))
    D, eps, dt = z["D"], z["eps"], float(z["dt"])          # D: [steps, n_eps, pairs]
    t = np.arange(D.shape[0]) * dt
    out = {"eps": eps.tolist(), "per_eps": []}
    for i, e in enumerate(eps):
        ld = np.log10(np.maximum(D[:, i, :], 1e-12))
        med = np.median(ld, axis=1)
        rec = {"eps": float(e), "t_first_1e-3": None, "t_first_1e-1": None, "lambda_per_s": None}
        above3 = np.where(med > -3)[0]
        above1 = np.where(med > -1)[0]
        if len(above3):
            rec["t_first_1e-3"] = float(t[above3[0]])
        if len(above1):
            rec["t_first_1e-1"] = float(t[above1[0]])
        # growth rate between the first crossings of 1e-5 and 1e-2 (exponential phase, before saturation)
        a = np.where(med > -5)[0]
        b = np.where(med > -2)[0]
        if len(a) and len(b) and b[0] > a[0] + 5:
            seg = slice(a[0], b[0])
            rec["lambda_per_s"] = float(np.polyfit(t[seg], med[seg] * np.log(10), 1)[0])
        rec["median_log10_d_at"] = {str(s): float(med[min(int(s / dt), len(med) - 1)]) for s in (0.5, 1, 2, 4, 8, 11.9)}
        out["per_eps"].append(rec)
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        fig, ax = plt.subplots(figsize=(6, 4))
        for i, e in enumerate(eps):
            ax.plot(t, np.median(np.log10(np.maximum(D[:, i, :], 1e-12)), axis=1), label="eps=%g" % e)
        ax.set_xlabel("time [s]"); ax.set_ylabel("median log10 twin separation")
        ax.legend(); ax.set_title("parameter-close twins, frozen CORR_s1")
        fig.savefig(os.path.join(ROOT, "figures", "fractal_ftle.png"), dpi=150, bbox_inches="tight")
    except Exception as ex:
        out["figure_error"] = str(ex)
    json.dump(out, open(os.path.join(ROOT, "results", "fractal", "ftle_analysis.json"), "w"), indent=1)
    print(json.dumps(out, indent=1))


if __name__ == "__main__":
    main()
