"""
Turn the exported part STLs into 2-D silhouettes of the five rigid bodies,
in the assembly frame, ready to animate.

Pipeline per part instance:
  1. load its STL (part coordinates)
  2. detect the STL's unit scale by comparing its extent to the bounding box
     recorded in bodies_raw.csv -- SolidWorks' STL unit preference is
     version-dependent, so it is measured rather than assumed
  3. place it with the SAME R / t used for the mass properties (validated to
     1.4 um against SolidWorks' own centre of mass)
  4. project to the (y, z) swing plane and union the triangles into a polygon

Parts with no STL fall back to the rectangle of their recorded bounding box, so
nothing silently disappears; the report says which.

Output: assets/triple_pendulum/meshes/body_outlines.npz
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
from shapely.geometry import Polygon
from shapely.ops import unary_union

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CSV = os.path.join(ROOT, "assets", "triple_pendulum", "cad", "bodies_raw.csv")
MESHDIR = os.path.join(ROOT, "assets", "triple_pendulum", "meshes")
GROUPING = os.path.join(ROOT, "configs", "robot", "body_grouping.yaml")
OUT = os.path.join(MESHDIR, "body_outlines.npz")

BODIES = ["base", "cart", "link1", "link2", "link3"]
MIN_MASS = 1e-4
MAX_FACES = 20000         # above this fall back to the convex hull (bearings,
                          # collars: round parts whose hull IS their silhouette).
                          # Below it keep every triangle, so the printed trusses
                          # keep their holes.

# link2's arm is an imported STEP copy that SolidWorks will not export
# standalone. It is the same physical part as link3's arm, so borrow that mesh
# and let the recorded transform place it; the bbox check below reports how well
# that substitution actually lands.
SUBSTITUTE = {
    "Major Assembly parts - R6PEndulum_Med-1.step.SLDPRT": "R6PEndulum_Med.STL",
}


def safe(name: str) -> str:
    return re.sub(r"[^A-Za-z0-9_.-]", "_", name)[:90]


def stl_for(part_path: str) -> str | None:
    base = os.path.basename(part_path)
    cand = os.path.join(MESHDIR, safe(base).replace(".SLDPRT", "") + ".STL")
    if os.path.exists(cand):
        return cand
    sub = SUBSTITUTE.get(base)
    if sub and os.path.exists(os.path.join(MESHDIR, sub)):
        return os.path.join(MESHDIR, sub)
    return None



def main() -> int:
    df = pd.read_csv(CSV)
    with open(GROUPING, encoding="utf-8") as fh:
        cfg = yaml.safe_load(fh)
    df["body"] = [_body_of(n, cfg) for n in df["name"]]

    polys: dict[str, list] = {b: [] for b in BODIES}
    used, fallback = 0, 0
    scales: list[float] = []
    mism: list = []

    for _, r in df.iterrows():
        body = r["body"]
        if body not in BODIES or r["mass"] < MIN_MASS:
            continue
        bbox = np.array([r["bbox_xmin"], r["bbox_ymin"], r["bbox_zmin"],
                         r["bbox_xmax"], r["bbox_ymax"], r["bbox_zmax"]], dtype=float)
        if np.isnan(bbox).any():
            continue
        want_ext = bbox[3:] - bbox[:3]

        path = stl_for(r["part_path"])
        placed = None
        if path is not None:
            try:
                m = trimesh.load_mesh(path)
                if len(m.faces) > MAX_FACES:
                    m = m.convex_hull
                v = np.asarray(m.vertices, dtype=float)

                R = np.array([[r["R00"], r["R01"], r["R02"]],
                              [r["R10"], r["R11"], r["R12"]],
                              [r["R20"], r["R21"], r["R22"]]], dtype=float)
                t = np.array([r["tx"], r["ty"], r["tz"]], dtype=float)

                # measure the STL's units against the recorded bbox extent
                ext = v.max(axis=0) - v.min(axis=0)
                s = 1.0
                if ext.max() > 0 and want_ext.max() > 0:
                    ratio = want_ext.max() / ext.max()
                    s = 0.001 if 0.0005 < ratio < 0.002 else (1.0 if 0.5 < ratio < 2.0 else ratio)
                scales.append(s)

                # Rotate only, then RE-ANCHOR. SolidWorks' STL exporter shifts
                # meshes into positive space (the DontTranslateToPositive
                # preference id did not take), so the exported origin is not the
                # part origin and R*v + t lands in the wrong place -- measured
                # offsets up to 0.31 m. The recorded bounding box is trustworthy
                # (same extraction validated to 1.4 um on centre of mass), so
                # align the rotated mesh's bbox centre onto the recorded one.
                v = (R @ (v * s).T).T
                have_c = 0.5 * (v.min(axis=0) + v.max(axis=0))
                want_c = 0.5 * (bbox[:3] + bbox[3:])
                v = v + (want_c - have_c)

                # sanity: extents should now agree, which also validates the
                # rotation and the detected scale
                got_ext = v.max(axis=0) - v.min(axis=0)
                rel = np.abs(got_ext - want_ext) / np.maximum(want_ext, 1e-6)
                if rel.max() > 0.25:
                    mism.append((os.path.basename(r["part_file"])[:38], float(rel.max())))

                f = np.asarray(m.faces)
                tris = v[f][:, :, [1, 2]]                   # project to (y, z)
                good = [Polygon(tr) for tr in tris]
                good = [g for g in good if g.is_valid and g.area > 1e-9]
                if good:
                    placed = unary_union(good)
                    used += 1
            except Exception as exc:
                print("   ! %s: %s" % (os.path.basename(str(path))[:40], str(exc)[:50]))

        if placed is None:      # bbox rectangle so nothing vanishes silently
            placed = Polygon([(bbox[1], bbox[2]), (bbox[4], bbox[2]),
                              (bbox[4], bbox[5]), (bbox[1], bbox[5])])
            fallback += 1
        polys[body].append(placed)

    print("[outline] %d parts from real geometry, %d from bbox fallback" % (used, fallback))
    if mism:
        print("[outline] %d parts whose extents disagree with the CSV bbox by >25%%:" % len(mism))
        for nm, rel in mism[:8]:
            print("     %-40s %.0f%%" % (nm, 100 * rel))
    if scales:
        uniq = sorted(set(round(s, 6) for s in scales))
        print("[outline] detected STL scale factors: %s" % uniq)

    out: dict[str, np.ndarray] = {}
    for body, ps in polys.items():
        if not ps:
            continue
        merged = unary_union(ps)
        geoms = list(merged.geoms) if merged.geom_type == "MultiPolygon" else [merged]
        rings, k = [], 0
        for g in geoms:
            g = g.simplify(0.0008)          # ~0.8 mm, keeps truss holes readable
            if g.is_empty or g.area < 2e-5:
                continue
            rings.append(np.asarray(g.exterior.coords, dtype=float))
            k += 1
            for hole in g.interiors:
                h = np.asarray(hole.coords, dtype=float)
                if Polygon(h).area > 2e-5:
                    rings.append(h)
                    k += 1
        out["%s_n" % body] = np.array([len(rings)])
        for i, ring in enumerate(rings):
            out["%s_%d" % (body, i)] = ring
        print("   %-6s %3d rings, %5d vertices" % (body, len(rings), sum(len(x) for x in rings)))

    np.savez(OUT, **out)
    print("[out]", OUT)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
