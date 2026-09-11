"""Is a single per-body mass factor good enough for the density correction?

experiments/train.py applies the CAD correction as one scale factor per body,
multiplying that body's mass and inertia together. That is exact for a uniform
density error. This one is not uniform: inside a single link, steel parts get
7.85x heavier while PETG parts get 0.48x lighter. The centre of mass and the
inertia distribution therefore move, not just the total.

This recomputes each link properly -- per-part density correction, then
composition by the parallel-axis theorem -- and compares against the scaled
approximation actually being trained on, so the size of the shortcut is known
rather than assumed.

  run.cmd tools/corrected_inertia.py
"""

from __future__ import annotations

import csv
import json
import os
import re

import numpy as np
import yaml

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CSV = os.path.join(ROOT, "assets", "triple_pendulum", "cad", "bodies_raw.csv")
GROUP = os.path.join(ROOT, "configs", "robot", "body_grouping.yaml")
OUT = os.path.join(ROOT, "results", "corrected_inertia.json")

RHO_STEEL, RHO_PETG, RHO_PCB = 7850.0, 480.0, 1800.0
STEEL = (r"5972K91", r"6056N16", r"Ball Bea", r"Shaft", r"Part14", r"Part31",
         r"Steel", r"SBR20", r"Motor", r"Pulley", r"Belt", r"Bearing", r"Screw",
         r"Nut", r"Bolt", r"LM\d")
PCB = (r"MT6701",)


def target_density(name, rho_cad):
    if abs(rho_cad - 1000.0) > 1e-6:
        return rho_cad                       # already assigned; leave alone
    if any(re.search(p, name, re.I) for p in STEEL):
        return RHO_STEEL
    if any(re.search(p, name, re.I) for p in PCB):
        return RHO_PCB
    return RHO_PETG


def match(spec, n):
    if isinstance(spec, str):
        spec = {"exact": [spec]}
    return n in spec.get("exact", []) or any(n.startswith(p) for p in spec.get("prefix", []))


def composite(parts):
    """(mass, com, I_about_com) for a list of (m, com, I_own) tuples."""
    M = sum(p[0] for p in parts)
    com = sum(p[0] * p[1] for p in parts) / M
    I = np.zeros((3, 3))
    for m, c, Iown in parts:
        d = c - com
        I += Iown + m * (float(d @ d) * np.eye(3) - np.outer(d, d))
    return M, com, I


def main() -> int:
    rows = list(csv.DictReader(open(CSV, encoding="utf-8")))
    g = yaml.safe_load(open(GROUP, encoding="utf-8"))["bodies"]
    out = {}

    print("  body    mass CAD   mass true   factor |  COM shift | Ixx scaled vs true")
    print("  " + "-" * 76)
    for b in ("link1", "link2", "link3", "cart"):
        old, new = [], []
        for r in rows:
            if not match(g[b], r["name"]):
                continue
            m, v = float(r["mass"]), float(r["volume"])
            rho_cad = m / v if v > 0 else 1000.0
            f = target_density(r["name"], rho_cad) / rho_cad
            com = np.array([float(r["com_x"]), float(r["com_y"]), float(r["com_z"])])
            I = np.array([[float(r["Ixx"]), float(r["Ixy"]), float(r["Ixz"])],
                          [float(r["Ixy"]), float(r["Iyy"]), float(r["Iyz"])],
                          [float(r["Ixz"]), float(r["Iyz"]), float(r["Izz"])]])
            old.append((m, com, I))
            new.append((m * f, com, I * f))
        Mo, Co, Io = composite(old)
        Mn, Cn, In = composite(new)
        fac = Mn / Mo
        # the approximation currently being trained on: scale the OLD composite
        I_approx = Io * fac
        shift = float(np.linalg.norm(Cn - Co))
        err = float(abs(In[0, 0] - I_approx[0, 0]) / max(abs(In[0, 0]), 1e-12))
        out[b] = {"mass_cad": Mo, "mass_true": Mn, "factor": fac,
                  "com_cad": Co.tolist(), "com_true": Cn.tolist(),
                  "com_shift_m": shift,
                  "Ixx_true": In[0, 0], "Ixx_scaled_approx": I_approx[0, 0],
                  "Ixx_rel_error": err}
        print("  %-6s %9.4f %11.4f %8.3f | %8.4f m | %.3e vs %.3e  (%.1f%% err)"
              % (b, Mo, Mn, fac, shift, In[0, 0], I_approx[0, 0], 100 * err))

    print("\n  COM shift is in ASSEMBLY coordinates; what matters dynamically is")
    print("  the shift along the link, i.e. the change in l_com.")
    worst = max(out[b]["Ixx_rel_error"] for b in out)
    print("\n  worst inertia error from the single-factor shortcut: %.1f%%" % (100 * worst))
    if worst > 0.15:
        print("  -> LARGE. The retraining currently running uses the shortcut, so")
        print("     its plant is not the fully corrected one. Reported, not hidden.")
    else:
        print("  -> small enough that the shortcut is defensible.")

    json.dump(out, open(OUT, "w", encoding="utf-8"), indent=1)
    print("\n[out] %s" % OUT)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
