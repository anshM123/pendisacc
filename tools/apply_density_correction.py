"""Write real material densities into the CAD export, at the source.

The SolidWorks assembly was exported with no materials assigned: 122 of its 130
bodies carry the default 1000 kg/m^3, including every one of the 16 bodies that
make up the three pendulum links. Patching the resulting masses at runtime is
not enough, because the error is not uniform within a body -- steel hardware
goes up 7.85x while PETG goes down to 0.48x -- so the centre of mass and the
inertia distribution move too. Measured: l_com shifts by +53.6% on link1 and
+28.3% on link2.

So the correction is applied here, to bodies_raw.csv, and the whole downstream
pipeline is re-run: aggregate_bodies.py -> params yaml -> build_urdf.py ->
build_usd.py. That way the simulated robot has the right masses, the right
centres of mass and the right inertias, rather than the right masses and
nothing else.

Per part: mass and the full inertia tensor are multiplied by
rho_true / rho_cad. Volume, centre of mass and pose are geometry and are left
alone. Bodies that already carry a real material (the 8 rail and rail-bearing
parts) are untouched.

  run.cmd tools/apply_density_correction.py            # dry run, reports only
  run.cmd tools/apply_density_correction.py --write    # edits bodies_raw.csv
"""

from __future__ import annotations

import argparse
import csv
import os
import re
import shutil

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CSV = os.path.join(ROOT, "assets", "triple_pendulum", "cad", "bodies_raw.csv")
BACKUP = os.path.join(ROOT, "assets", "triple_pendulum", "cad",
                      "bodies_raw_ORIGINAL_default_density.csv")

RHO_STEEL, RHO_PETG, RHO_PCB = 7850.0, 480.0, 1800.0
STEEL = (r"5972K91", r"6056N16", r"Ball Bea", r"Shaft", r"Part14", r"Part31",
         r"Steel", r"SBR20", r"Motor", r"Pulley", r"Belt", r"Bearing", r"Screw",
         r"Nut", r"Bolt", r"LM\d")
PCB = (r"MT6701",)
INERTIA = ("Ixx", "Iyy", "Izz", "Ixy", "Ixz", "Iyz")


def target(name: str, rho_cad: float) -> tuple:
    if abs(rho_cad - 1000.0) > 1e-6:
        return rho_cad, "already-assigned"
    if any(re.search(p, name, re.I) for p in STEEL):
        return RHO_STEEL, "steel"
    if any(re.search(p, name, re.I) for p in PCB):
        return RHO_PCB, "pcb"
    return RHO_PETG, "petg"


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--write", action="store_true")
    a = ap.parse_args()

    rows = list(csv.DictReader(open(CSV, encoding="utf-8")))
    fields = list(rows[0].keys())
    counts, m_old, m_new = {}, 0.0, 0.0

    for r in rows:
        m = float(r["mass"])
        v = float(r["volume"])
        rho_cad = m / v if v > 0 else 1000.0
        rho_new, cls = target(r["name"], rho_cad)
        f = rho_new / rho_cad
        counts[cls] = counts.get(cls, 0) + 1
        m_old += m
        m_new += m * f
        if a.write and abs(f - 1.0) > 1e-12:
            r["mass"] = "%.12g" % (m * f)
            r["density"] = "%.12g" % rho_new
            for k in INERTIA:
                if k in r and r[k] not in ("", None):
                    r[k] = "%.12g" % (float(r[k]) * f)

    print("parts by class:", counts)
    print("assembly mass  %.4f kg  ->  %.4f kg  (x%.3f)"
          % (m_old, m_new, m_new / m_old))

    if not a.write:
        print("\ndry run. re-run with --write to edit bodies_raw.csv")
        return 0

    if not os.path.exists(BACKUP):
        shutil.copy2(CSV, BACKUP)
        print("\noriginal preserved at %s" % os.path.basename(BACKUP))
    with open(CSV, "w", encoding="utf-8", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=fields)
        w.writeheader()
        w.writerows(rows)
    print("wrote %s" % CSV)
    print("\nnow re-run, in order:")
    print("  run.cmd tools/aggregate_bodies.py      # -> params yaml")
    print("  powershell tools/rebuild_asset.ps1     # -> urdf -> usd -> validate")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
