"""
Merge the exported part STLs into ONE mesh per rigid body, expressed in that
body's URDF LINK frame, so the Isaac asset can show the real machine instead of
capsules.

Frames, in order:
  part STL  --rotate R, re-anchor on recorded bbox centre-->  assembly (CAD)
  assembly  --R_link . (p - pivot)-->                         URDF link frame

R_link rows are the link's local axes in CAD coordinates, identical to the
construction in build_urdf.py: local Z along the link, local Y the revolute
axis, local X = Y x Z. The cart uses the plain CAD->URDF axis permutation since
it does not rotate.

Re-anchoring is required because SolidWorks' STL export shifts geometry into
positive space; see the note in build_link_outlines.py.

Output: assets/triple_pendulum/meshes/visual/<body>.obj
"""

from __future__ import annotations

import os
import re

import sys as _sys, os as _os
_sys.path.insert(0, _os.path.dirname(_os.path.abspath(__file__)))
from grouping import body_of as _body_of
import numpy as np
import pandas as pd
import trimesh
import yaml

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CSV = os.path.join(ROOT, "assets", "triple_pendulum", "cad", "bodies_raw.csv")
MESHDIR = os.path.join(ROOT, "assets", "triple_pendulum", "meshes")
OUTDIR = os.path.join(MESHDIR, "visual")
PARAMS = os.path.join(ROOT, "configs", "robot", "triple_pendulum_params.yaml")
GROUPING = os.path.join(ROOT, "configs", "robot", "body_grouping.yaml")

MOVING = ["base", "cart", "link1", "link2", "link3"]
MIN_MASS = 1e-4
MIN_MASS_BY_BODY = {"base": 2e-3}
HULL_ABOVE = 20000

SUBSTITUTE = {"Major Assembly parts - R6PEndulum_Med-1.step.SLDPRT": "R6PEndulum_Med.STL"}

JOINT_BEARING = {
    "J1": ("R65972K91_Ball Bearing.SLDPRT", None),
    "J2": ("R65972K91_Steel Ball Bearing(1).SLDPRT", "R6PEndulum assembly-1"),
    "J3": ("R65972K91_Steel Ball Bearing(1).SLDPRT", "R6PEndulum assembly-2"),
}
PIVOT_OF = {"link1": "J1", "link2": "J2", "link3": "J3"}


def safe(n):
    return re.sub(r"[^A-Za-z0-9_.-]", "_", n)[:90]


def stl_for(part_path):
    base = os.path.basename(part_path)
    c = os.path.join(MESHDIR, safe(base).replace(".SLDPRT", "") + ".STL")
    if os.path.exists(c):
        return c
    sub = SUBSTITUTE.get(base)
    return os.path.join(MESHDIR, sub) if sub and os.path.exists(os.path.join(MESHDIR, sub)) else None


def main() -> int:
    os.makedirs(OUTDIR, exist_ok=True)
    df = pd.read_csv(CSV)
    cfg = yaml.safe_load(open(GROUPING, encoding="utf-8"))
    P = yaml.safe_load(open(PARAMS, encoding="utf-8"))

    df["body"] = [_body_of(n, cfg) for n in df["name"]]

    # joint positions in full 3-D CAD coordinates
    jx = {}
    for j, (part, prefix) in JOINT_BEARING.items():
        sel = df["part_file"] == part
        if prefix:
            sel &= df["name"].str.startswith(prefix)
        jx[j] = float(df[sel]["com_x"].mean())
    Jyz = {k: np.array(v["pos_yz"], float) for k, v in P["joints"].items()}
    Jcad = {k: np.array([jx[k], Jyz[k][0], Jyz[k][1]]) for k in Jyz}

    # link local frames, same construction as build_urdf.py
    def link_dir(name, prox, dist):
        a = Jyz[prox]
        b = Jyz[dist] if dist else np.array(P["bodies"][name]["com"])[[1, 2]]
        d = b - a
        return d / np.linalg.norm(d)

    dirs = {"link1": link_dir("link1", "J1", "J2"),
            "link2": link_dir("link2", "J2", "J3"),
            "link3": link_dir("link3", "J3", None)}
    Rlink = {}
    for name, d in dirs.items():
        ez = np.array([0.0, d[0], d[1]])
        ey = np.array([1.0, 0.0, 0.0])
        Rlink[name] = np.vstack([np.cross(ey, ez), ey, ez])
    P_CAD2URDF = np.array([[0, 0, 1], [1, 0, 0], [0, 1, 0]], float)

    report = []
    for body in MOVING:
        parts = []
        sub = df[(df["body"] == body) & (df["mass"] >= MIN_MASS_BY_BODY.get(body, MIN_MASS))]
        for _, r in sub.iterrows():
            path = stl_for(r["part_path"])
            if path is None:
                continue
            bbox = np.array([r["bbox_xmin"], r["bbox_ymin"], r["bbox_zmin"],
                             r["bbox_xmax"], r["bbox_ymax"], r["bbox_zmax"]], float)
            if np.isnan(bbox).any():
                continue
            try:
                m = trimesh.load_mesh(path)
                if len(m.faces) > HULL_ABOVE:
                    m = m.convex_hull
                v = np.asarray(m.vertices, float)
                R = np.array([[r["R00"], r["R01"], r["R02"]],
                              [r["R10"], r["R11"], r["R12"]],
                              [r["R20"], r["R21"], r["R22"]]], float)
                want_ext = bbox[3:] - bbox[:3]
                ext = v.max(axis=0) - v.min(axis=0)
                s = 0.001 if (ext.max() > 0 and 0.0005 < want_ext.max() / ext.max() < 0.002) else 1.0
                v = (R @ (v * s).T).T
                v += 0.5 * (bbox[:3] + bbox[3:]) - 0.5 * (v.min(axis=0) + v.max(axis=0))
                parts.append(trimesh.Trimesh(vertices=v, faces=np.asarray(m.faces), process=False))
            except Exception as exc:
                print("   ! %s: %s" % (os.path.basename(path)[:36], str(exc)[:44]))

        if not parts:
            print("   %-6s no meshes" % body)
            continue
        merged = trimesh.util.concatenate(parts)

        v = np.asarray(merged.vertices, float)
        if body == "base":
            # base_link sits at the world origin, which is the CAD assembly
            # origin (rail centre); only the axis permutation is needed
            v = (P_CAD2URDF @ v.T).T
        elif body == "cart":
            v = (P_CAD2URDF @ (v - Jcad["J1"]).T).T
        else:
            v = (Rlink[body] @ (v - Jcad[PIVOT_OF[body]]).T).T
        out = trimesh.Trimesh(vertices=v, faces=merged.faces, process=False)

        dst = os.path.join(OUTDIR, "%s.obj" % body)
        out.export(dst)
        ext = v.max(axis=0) - v.min(axis=0)
        report.append((body, len(parts), len(out.faces), ext))
        print("   %-6s %2d parts, %6d faces, local extent (%.3f, %.3f, %.3f) m"
              % (body, len(parts), len(out.faces), *ext))

    print("[mesh] -> %s" % OUTDIR)
    return 0 if report else 1


if __name__ == "__main__":
    raise SystemExit(main())
