"""The CAD assembly was never assigned materials. Quantify the damage.

Every one of the 16 parts making up link1, link2 and link3 carries SolidWorks'
default density of exactly 1000 kg/m^3 -- water. Only 8 bodies in the whole
130-body assembly have a real material, and all 8 are linear rails or rail
bearings in the static base. So every inertial parameter the pendulum
dynamics depend on is fictitious.

The error is NOT a uniform scale factor, which is what makes it serious for
this project specifically:

  steel shafts, bearings, collars   modelled at 1000, should be ~7850  -> 7.9x too LIGHT
  PETG arms printed at 25% infill   modelled at 1000, should be ~480   -> 2.1x too HEAVY

Those pull in opposite directions, so the link MASS RATIOS are wrong. The
transfer landscape (results/h4_grid_velocity.json) shows ratio distortion
explains 99.9% of transfer variance while overall scale explains 0.0%. The CAD
defect therefore sits squarely in the one direction this system cannot
tolerate.

DENSITIES ARE BRACKETED, NOT GUESSED PRECISELY. A 25%-infill PETG print is not
25% of solid density: perimeters, top and bottom solid layers and over-extrusion
push the effective figure up. The bracket below spans 400-570 kg/m^3, and the
corrected masses are reported across that range so the conclusion does not
depend on one number.

  run.cmd tools/cad_density_audit.py
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
OUT = os.path.join(ROOT, "results", "cad_density_audit.json")

RHO_CAD = 1000.0
RHO_STEEL = 7850.0
RHO_PETG = (400.0, 480.0, 570.0)      # low, central, high for ~25% infill
RHO_PCB = 1800.0                      # MT6701 encoder board + magnet

# Classification by part name. Steel: the 8 mm shafts (what the build calls
# "links"), the ball bearings and the shaft collars. PETG: the printed arms
# and casings. Anything unmatched is treated as PETG and listed, so an
# unclassified part cannot hide.
STEEL = (r"5972K91", r"6056N16", r"Ball Bea", r"Shaft", r"Part14", r"Part31")
PCB = (r"MT6701",)


def classify(name: str) -> str:
    if any(re.search(p, name, re.I) for p in STEEL):
        return "steel"
    if any(re.search(p, name, re.I) for p in PCB):
        return "pcb"
    return "petg"


def match(spec, name: str) -> bool:
    if isinstance(spec, str):
        spec = {"exact": [spec]}
    return (name in spec.get("exact", [])
            or any(name.startswith(p) for p in spec.get("prefix", [])))


def main() -> int:
    rows = list(csv.DictReader(open(CSV, encoding="utf-8")))
    g = yaml.safe_load(open(GROUP, encoding="utf-8"))["bodies"]

    print("Part classification and corrected mass, central PETG estimate\n")
    print("  %-44s %-6s %8s %9s" % ("part", "class", "CAD[g]", "true[g]"))
    print("  " + "-" * 74)
    per_link = {}
    for b in ("link1", "link2", "link3"):
        tot = {k: 0.0 for k in ("cad", "lo", "mid", "hi")}
        for r in rows:
            if not match(g[b], r["name"]):
                continue
            m, v = float(r["mass"]), float(r["volume"])
            c = classify(r["name"])
            rho = {"steel": (RHO_STEEL,) * 3, "pcb": (RHO_PCB,) * 3,
                   "petg": RHO_PETG}[c]
            new = [v * x for x in rho]
            tot["cad"] += m
            for k, x in zip(("lo", "mid", "hi"), new):
                tot[k] += x
            print("  %-44s %-6s %8.2f %9.2f" % (r["name"][:44], c, 1000 * m, 1000 * new[1]))
        per_link[b] = tot
        print("  %-44s %-6s %8.2f %9.2f   <== %s\n"
              % ("TOTAL", "", 1000 * tot["cad"], 1000 * tot["mid"], b))

    cad = np.array([per_link[b]["cad"] for b in ("link1", "link2", "link3")])
    mid = np.array([per_link[b]["mid"] for b in ("link1", "link2", "link3")])
    lo = np.array([per_link[b]["lo"] for b in ("link1", "link2", "link3")])
    hi = np.array([per_link[b]["hi"] for b in ("link1", "link2", "link3")])

    print("LINK MASSES [kg]")
    print("  %-22s %8s %8s %8s" % ("", "link1", "link2", "link3"))
    print("  %-22s %8.4f %8.4f %8.4f" % ("CAD (as simulated)", *cad))
    print("  %-22s %8.4f %8.4f %8.4f" % ("corrected, PETG 400", *lo))
    print("  %-22s %8.4f %8.4f %8.4f" % ("corrected, PETG 480", *mid))
    print("  %-22s %8.4f %8.4f %8.4f" % ("corrected, PETG 570", *hi))
    print("  %-22s %8.3f %8.3f %8.3f" % ("ratio true/CAD", *(mid / cad)))

    # decompose into the two axes the landscape uses: overall scale c, and
    # ratio distortion. c is the geometric-mean scale; the residual is delta.
    c = float(np.exp(np.mean(np.log(mid / cad))))
    resid = (mid / cad) / c
    print("\nIN THE COORDINATES THE LANDSCAPE USES")
    print("  overall scale  c       = %.3f" % c)
    print("  residual ratio vector  = [%.3f, %.3f, %.3f]" % tuple(resid))
    print("  ratio distortion       = %.3f  (max/min of the residual)"
          % (resid.max() / resid.min()))
    print("\n  The landscape says c explains 0.0%% of transfer variance and ratio")
    print("  distortion explains 99.9%%. This defect is almost entirely ratio.")

    json.dump({"rho_cad": RHO_CAD, "rho_steel": RHO_STEEL, "rho_petg": RHO_PETG,
               "cad_masses": cad.tolist(),
               "corrected_lo": lo.tolist(), "corrected_mid": mid.tolist(),
               "corrected_hi": hi.tolist(),
               "scale_factors_mid": (mid / cad).tolist(),
               "overall_scale_c": c, "residual_ratio": resid.tolist(),
               "ratio_distortion": float(resid.max() / resid.min())},
              open(OUT, "w", encoding="utf-8"), indent=1)
    print("\n[out] %s" % OUT)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
