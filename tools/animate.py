"""
Turn a recorded rollout into a watchable animation, using matplotlib only.

No Isaac renderer involved, so this works even when the GUI/hydra path is
unusable. Link geometry is read from the CAD-derived parameters, so what you see
is the real machine's proportions.

  run.cmd tools\\animate.py --in results\\rollout_passive.npz --out figures\\passive.gif
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

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

parser = argparse.ArgumentParser()
parser.add_argument("--in", dest="src", default=os.path.join(ROOT, "results", "rollout_passive.npz"))
parser.add_argument("--out", dest="dst", default=None)
parser.add_argument("--fps", type=int, default=50)
parser.add_argument("--trail", type=int, default=60, help="tip trail length in frames")
args = parser.parse_args()

d = np.load(args.src, allow_pickle=True)
t, q = d["t"], d["q"]
mode = str(d["mode"]) if "mode" in d else "rollout"

with open(os.path.join(ROOT, "configs", "robot", "triple_pendulum_params.yaml"), encoding="utf-8") as fh:
    P = yaml.safe_load(fh)
L = []
for name in ("link1", "link2", "link3"):
    e = P["bodies"][name]
    L.append(float(e.get("length", 2.0 * e["l_com"])))
RAIL = float(P["prismatic"]["rail_span"])

# decimate to the requested frame rate
dt = float(d["dt"])
stride = max(1, int(round(1.0 / (args.fps * dt))))
t, q = t[::stride], q[::stride]


def joints(qk):
    """Cart + the three link endpoints, in the (rail, vertical) plane."""
    s, th = qk[0], qk[1:]
    pts = [(s, 0.0)]
    x, y = s, 0.0
    for i in range(3):
        x += L[i] * np.sin(th[i])
        y += L[i] * np.cos(th[i])
        pts.append((x, y))
    return np.array(pts)

reach = sum(L)
fig, ax = plt.subplots(figsize=(9, 5))
ax.set_aspect("equal")
ax.set_xlim(-RAIL / 2 - 0.05, RAIL / 2 + 0.05)
ax.set_ylim(-reach - 0.08, reach + 0.12)
ax.axhline(0.0, color="0.75", lw=1)
ax.plot([-RAIL / 2, RAIL / 2], [0, 0], color="0.4", lw=3, solid_capstyle="butt")
ax.set_xlabel("rail position [m]")
ax.set_ylabel("height [m]")
ax.grid(alpha=0.25)

colors = ["#c62828", "#1565c0", "#2e7d32"]
segments = [ax.plot([], [], lw=5, color=c, solid_capstyle="round")[0] for c in colors]
pivots, = ax.plot([], [], "o", ms=7, color="0.15", zorder=5)
cart, = ax.plot([], [], "s", ms=14, color="#6a1b9a", zorder=4)
trail, = ax.plot([], [], "-", lw=1.2, color="#2e7d32", alpha=0.5)
label = ax.text(0.02, 0.96, "", transform=ax.transAxes, va="top", family="monospace")
ax.set_title("triple inverted pendulum -- %s" % mode)

tip_hist: list[tuple[float, float]] = []


def update(k):
    pts = joints(q[k])
    for i, seg in enumerate(segments):
        seg.set_data(pts[i:i + 2, 0], pts[i:i + 2, 1])
    pivots.set_data(pts[:, 0], pts[:, 1])
    cart.set_data([pts[0, 0]], [pts[0, 1]])
    tip_hist.append((pts[3, 0], pts[3, 1]))
    if len(tip_hist) > args.trail:
        del tip_hist[0]
    tr = np.array(tip_hist)
    trail.set_data(tr[:, 0], tr[:, 1])
    label.set_text("t = %5.2f s\ncart = %+6.3f m" % (t[k], q[k, 0]))
    return segments + [pivots, cart, trail, label]


dst = args.dst or os.path.join(ROOT, "figures", "rollout_%s.gif" % mode)
os.makedirs(os.path.dirname(dst), exist_ok=True)
anim = FuncAnimation(fig, update, frames=len(t), interval=1000 / args.fps, blit=True)
anim.save(dst, writer=PillowWriter(fps=args.fps))
print("[animate] wrote %s  (%d frames, %.1f s)" % (dst, len(t), t[-1]))
