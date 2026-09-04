"""
Collapse the 130 CAD leaf bodies into the 5 rigid bodies of the pendulum, using
configs/robot/body_grouping.yaml.

Composite mass properties via the parallel-axis theorem:

    M   = sum_i m_i
    c   = (1/M) sum_i m_i c_i
    I_c = sum_i [ I_i + m_i ( |d_i|^2 E - d_i d_i^T ) ],   d_i = c_i - c

Each leaf's I_i is already about ITS OWN centre of mass and already rotated into
the assembly frame by sw_dump_assembly.py, so no further rotation is needed here.

Also derives, per link, the quantities the analytical model actually wants:
    L      distance between the link's proximal and distal joints
    l_c    distance from the proximal joint to the centre of mass
    I_pivot inertia about the proximal joint axis (x), via parallel axis

Outputs configs/robot/triple_pendulum_params.yaml -- the single source of truth
for the Isaac asset, the analytical model, and the hardware interface.
"""

from __future__ import annotations

import os
import sys

import sys as _sys, os as _os
_sys.path.insert(0, _os.path.dirname(_os.path.abspath(__file__)))
from grouping import body_of as _body_of
import numpy as np
import pandas as pd
import yaml

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CSV = os.path.join(ROOT, "assets", "triple_pendulum", "cad", "bodies_raw.csv")
GROUPING = os.path.join(ROOT, "configs", "robot", "body_grouping.yaml")
OUT = os.path.join(ROOT, "configs", "robot", "triple_pendulum_params.yaml")

# which joint is proximal / distal for each moving link
CHAIN = {
    "base": (None, None),
    "cart": (None, "J1"),
    "link1": ("J1", "J2"),
    "link2": ("J2", "J3"),
    "link3": ("J3", None),
}


def assign(name: str, rules: dict) -> bool:
    for p in rules.get("prefix", []) or []:
        if name.startswith(p):
            return True
    for e in rules.get("exact", []) or []:
        if name == e:
            return True
    return False


def composite(df: pd.DataFrame):
    """(mass, com[3], I_about_com[3x3]) for a set of leaf bodies."""
    m = df["mass"].to_numpy()
    total = float(m.sum())
    c = np.array([np.average(df["com_%s" % a].to_numpy(), weights=m) for a in "xyz"])
    inertia = np.zeros((3, 3))
    for _, r in df.iterrows():
        i_own = np.array([
            [r["Ixx"], r["Ixy"], r["Ixz"]],
            [r["Ixy"], r["Iyy"], r["Iyz"]],
            [r["Ixz"], r["Iyz"], r["Izz"]],
        ])
        d = np.array([r["com_x"], r["com_y"], r["com_z"]]) - c
        inertia += i_own + r["mass"] * (float(d @ d) * np.eye(3) - np.outer(d, d))
    return total, c, inertia


def main():
    df = pd.read_csv(CSV)
    with open(GROUPING, encoding="utf-8") as fh:
        cfg = yaml.safe_load(fh)

    joints = {k: np.array(v["pos_yz"], dtype=float) for k, v in cfg["joints"].items()}

    # longest-prefix-wins, shared with every other consumer (tools/grouping.py)
    df["body"] = [_body_of(n, cfg) for n in df["name"]]

    unassigned = df[df["body"].isna()]
    if len(unassigned):
        print("!! %d UNASSIGNED bodies (%.6f kg) -- fix body_grouping.yaml:"
              % (len(unassigned), unassigned["mass"].sum()))
        for _, r in unassigned.iterrows():
            print("     %-58s %.6f kg" % (r["name"][:58], r["mass"]))
        sys.exit(1)

    out = {
        "_source": "derived by tools/aggregate_bodies.py from CAD; do not hand-edit",
        "frame": cfg["frame"],
        "prismatic": cfg["prismatic"],
        "joints": {k: {"pos_yz": [float(x) for x in v]} for k, v in joints.items()},
        "bodies": {},
    }

    print("=== composite rigid bodies ===")
    print("%-8s %5s %10s   %-28s %-28s" % ("body", "n", "mass[kg]", "com (x,y,z) [m]", "I_com diag [kg m^2]"))
    total_mass = 0.0
    for body in ["base", "cart", "link1", "link2", "link3"]:
        sub = df[df["body"] == body]
        mass, com, inertia = composite(sub)
        total_mass += mass
        print("%-8s %5d %10.6f   (%7.4f,%7.4f,%7.4f)  (%.3e,%.3e,%.3e)"
              % (body, len(sub), mass, *com, inertia[0, 0], inertia[1, 1], inertia[2, 2]))

        entry = {
            "n_cad_bodies": int(len(sub)),
            "mass": float(mass),
            "com": [float(x) for x in com],
            "inertia_com": {
                "ixx": float(inertia[0, 0]), "iyy": float(inertia[1, 1]), "izz": float(inertia[2, 2]),
                "ixy": float(inertia[0, 1]), "ixz": float(inertia[0, 2]), "iyz": float(inertia[1, 2]),
            },
        }

        prox, dist = CHAIN[body]
        if prox is not None:
            p = joints[prox]
            com_yz = np.array([com[1], com[2]])
            lc = float(np.linalg.norm(com_yz - p))
            # inertia about the proximal joint's x axis (revolute axis)
            i_pivot = float(inertia[0, 0] + mass * lc ** 2)
            entry["proximal_joint"] = prox
            entry["l_com"] = lc
            entry["inertia_about_proximal_x"] = i_pivot
            if dist is not None:
                entry["distal_joint"] = dist
                entry["length"] = float(np.linalg.norm(joints[dist] - p))
        out["bodies"][body] = entry

    print("%-8s %5d %10.6f" % ("TOTAL", len(df), total_mass))

    print("\n=== link parameters for the analytical model ===")
    print("%-7s %9s %9s %9s %14s" % ("link", "m[kg]", "L[m]", "l_c[m]", "I_pivot[kgm^2]"))
    for body in ["link1", "link2", "link3"]:
        e = out["bodies"][body]
        print("%-7s %9.5f %9s %9.5f %14.6e"
              % (body, e["mass"],
                 ("%.5f" % e["length"]) if "length" in e else "  (free)",
                 e["l_com"], e["inertia_about_proximal_x"]))

    moving = sum(out["bodies"][b]["mass"] for b in ["cart", "link1", "link2", "link3"])
    print("\nmoving mass (cart + 3 links) = %.6f kg" % moving)
    print("static base                  = %.6f kg" % out["bodies"]["base"]["mass"])

    with open(OUT, "w", encoding="utf-8") as fh:
        yaml.safe_dump(out, fh, sort_keys=False, default_flow_style=False)
    print("\n[out]", OUT)


if __name__ == "__main__":
    main()
