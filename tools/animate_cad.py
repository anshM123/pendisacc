"""
Animate a recorded rollout using the ACTUAL CAD silhouettes.

Each body's outline (from build_link_outlines.py) sits in the assembly frame in
its CAD pose. To draw frame k, each body is rotated about its own pivot by the
difference between the commanded angle and its CAD rest angle, then carried by
whatever its parent is doing:

    link i:   p' = pivot_i(t) + R(theta_i(t) - theta_i^CAD) . (p - pivot_i^CAD)
    cart:     p' = p + (s(t), 0)

Rest angles are derived from the joint positions, not hard-coded, so this stays
correct if the CAD changes.

Plot frame is (horizontal = assembly z = rail axis, vertical = assembly y).

  run.cmd tools\\animate_cad.py --in results\\rollout_swingup2.npz --out figures\\swingup_cad.gif
"""

from __future__ import annotations

import argparse
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import yaml  # noqa: E402
from matplotlib.animation import FuncAnimation, PillowWriter  # noqa: E402
from matplotlib.path import Path  # noqa: E402
from matplotlib.patches import PathPatch  # noqa: E402

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

ap = argparse.ArgumentParser()
ap.add_argument("--in", dest="src", default=os.path.join(ROOT, "results", "rollout_swingup2.npz"))
ap.add_argument("--out", dest="dst", default=os.path.join(ROOT, "figures", "swingup_cad.gif"))
ap.add_argument("--fps", type=int, default=40)
ap.add_argument("--trail", type=int, default=70)
args = ap.parse_args()

# ---- geometry -------------------------------------------------------------
with open(os.path.join(ROOT, "configs", "robot", "triple_pendulum_params.yaml"), encoding="utf-8") as fh:
    P = yaml.safe_load(fh)
J = {k: np.array(v["pos_yz"], dtype=float) for k, v in P["joints"].items()}   # (y, z)
RAIL = float(P["prismatic"]["rail_span"])

def yz_to_hv(p):
    """(y, z) -> (horizontal, vertical) = (z, y)."""
    p = np.atleast_2d(np.asarray(p, dtype=float))
    return np.column_stack([p[:, 1], p[:, 0]])

PIV = {k: yz_to_hv(v)[0] for k, v in J.items()}

def rest_angle(a_hv, b_hv):
    d = b_hv - a_hv
    return float(np.arctan2(d[0], d[1]))          # angle from +vertical toward +horizontal

L, TH0, PIVOT_OF = {}, {}, {"link1": "J1", "link2": "J2", "link3": "J3"}
com3 = np.array(P["bodies"]["link3"]["com"])[[1, 2]]
ends = {"link1": PIV["J2"], "link2": PIV["J3"], "link3": yz_to_hv(com3)[0]}
for name, pj in PIVOT_OF.items():
    TH0[name] = rest_angle(PIV[pj], ends[name])
    e = P["bodies"][name]
    L[name] = float(e.get("length", 2.0 * e["l_com"]))
print("[cad] CAD rest angles (rad): " + ", ".join("%s=%.4f" % (k, v) for k, v in TH0.items()))

# ---- outlines -------------------------------------------------------------
OUT = np.load(os.path.join(ROOT, "assets", "triple_pendulum", "meshes", "body_outlines.npz"))
def rings(body):
    n = int(OUT["%s_n" % body][0]) if ("%s_n" % body) in OUT else 0
    return [yz_to_hv(OUT["%s_%d" % (body, i)]) for i in range(n)]

BODY_RINGS = {b: rings(b) for b in ("cart", "link1", "link2", "link3")}
print("[cad] rings: " + ", ".join("%s=%d" % (b, len(r)) for b, r in BODY_RINGS.items()))

# ---- trajectory -----------------------------------------------------------
d = np.load(args.src, allow_pickle=True)
t, q, dt = d["t"], d["q"], float(d["dt"])
stride = max(1, int(round(1.0 / (args.fps * dt))))
t, q = t[::stride], q[::stride]

def rot(dth):
    c, s = np.cos(dth), np.sin(dth)
    return np.array([[c, s], [-s, c]])            # +theta tips toward +horizontal

def pose(qk):
    """Per-body (rotation, pivot_now, pivot_cad) for one frame."""
    s = qk[0]
    th = qk[1:]
    p1 = PIV["J1"] + np.array([s, 0.0])
    p2 = p1 + L["link1"] * np.array([np.sin(th[0]), np.cos(th[0])])
    p3 = p2 + L["link2"] * np.array([np.sin(th[1]), np.cos(th[1])])
    now = {"link1": p1, "link2": p2, "link3": p3}
    return s, {k: (rot(th[i] - TH0[k]), now[k], PIV[PIVOT_OF[k]])
               for i, k in enumerate(("link1", "link2", "link3"))}, (p1, p2, p3)

# ---- figure ---------------------------------------------------------------
reach = sum(L.values())
# Frame tightly on the region the cart actually uses. The full 1.524 m rail
# makes the machine tiny; the policy only travels about +-0.5 m.
fig, ax = plt.subplots(figsize=(7.6, 8.2))
ax.set_aspect("equal")
ax.set_xlim(-0.70, 0.70)
ax.set_ylim(-reach - 0.06, reach + 0.10)
ax.set_xlabel("rail position [m]")
ax.set_ylabel("height [m]")
ax.grid(alpha=0.2, lw=0.5)
ax.plot([-RAIL / 2, RAIL / 2], [0, 0], color="0.35", lw=5, solid_capstyle="butt", zorder=1)
for xe in (-0.60, 0.60):            # soft rail limits the task terminates on
    ax.axvline(xe, color="0.75", lw=1, ls="--", zorder=0)

COL = {"cart": "#6a4c93", "link1": "#c1272d", "link2": "#1565c0", "link3": "#2e7d32"}
patches = {}
for b, col in COL.items():
    p = PathPatch(Path([[0, 0]]), facecolor=col, edgecolor="0.12", lw=0.9, alpha=0.95, zorder=3)
    ax.add_patch(p)
    patches[b] = p
pivots, = ax.plot([], [], "o", ms=5, color="0.1", zorder=6)
trail, = ax.plot([], [], "-", lw=1.3, color=COL["link3"], alpha=0.55, zorder=2)
label = ax.text(0.015, 0.965, "", transform=ax.transAxes, va="top", family="monospace", fontsize=10)
ax.set_title("triple inverted pendulum - CAD geometry, PPO swing-up")

def build_path(ring_list):
    verts, codes = [], []
    for r in ring_list:
        if len(r) < 3:
            continue
        verts.extend(r)
        verts.append(r[0])
        codes.extend([Path.MOVETO] + [Path.LINETO] * (len(r) - 1) + [Path.CLOSEPOLY])
    if not verts:
        return Path([[0, 0]])
    return Path(np.asarray(verts), codes)

tip_hist: list[np.ndarray] = []

def update(k):
    s, bodies, (p1, p2, p3) = pose(q[k])
    patches["cart"].set_path(build_path([r + np.array([s, 0.0]) for r in BODY_RINGS["cart"]]))
    for b in ("link1", "link2", "link3"):
        R, now, cad = bodies[b]
        patches[b].set_path(build_path([(R @ (r - cad).T).T + now for r in BODY_RINGS[b]]))
    pivots.set_data([p1[0], p2[0], p3[0]], [p1[1], p2[1], p3[1]])
    tip = p3 + L["link3"] * np.array([np.sin(q[k, 3]), np.cos(q[k, 3])])
    tip_hist.append(tip)
    if len(tip_hist) > args.trail:
        del tip_hist[0]
    tr = np.array(tip_hist)
    trail.set_data(tr[:, 0], tr[:, 1])
    label.set_text("t = %5.2f s\ncart = %+6.3f m" % (t[k], s))
    return list(patches.values()) + [pivots, trail, label]

os.makedirs(os.path.dirname(args.dst), exist_ok=True)
anim = FuncAnimation(fig, update, frames=len(t), interval=1000 / args.fps, blit=True)
anim.save(args.dst, writer=PillowWriter(fps=args.fps))
print("[cad] wrote %s (%d frames, %.1f s)" % (args.dst, len(t), t[-1]))
