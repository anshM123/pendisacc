"""Render a 60 s highlight montage from the recorded Isaac clips.

  run.cmd tools/make_montage.py            -> figures/montage_swingup.mp4

Input: results/fractal/montage_clips.npz (experiments/record_montage.py). Every trajectory is
real Isaac data; nothing here is re-simulated or idealised.
"""

from __future__ import annotations

import json
import os

import imageio.v2 as imageio
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CLIPS = os.path.join(ROOT, "results", "fractal", "montage_clips.npz")
OUT = os.path.join(ROOT, "figures", "montage_swingup.mp4")
FPS, W, H = 30, 1280, 720
GREEN, RED, BLUE, GREY, INK = "#2e7d32", "#c62828", "#1f5fa8", "#777777", "#111111"
plt.rcParams.update({"font.family": "DejaVu Sans", "text.color": INK,
                     "axes.edgecolor": "#cccccc", "axes.labelcolor": INK,
                     "xtick.color": GREY, "ytick.color": GREY})

d = np.load(CLIPS, allow_pickle=True)
TH, CART, SUCC = d["theta"], d["cart"], d["success"]
LAB = [str(x) for x in d["labels"]]
L, DT = d["link_lengths"], float(d["dt"])
STEPS = TH.shape[0]
TIP = (L[None, None, :] * np.cos(TH)).sum(-1) / L.sum()


def joints(k, i):
    x, y = float(CART[k, i]), 0.0
    xs, ys = [x], [y]
    for j in range(3):
        x = x + L[j] * np.sin(TH[k, i, j])
        y = y + L[j] * np.cos(TH[k, i, j])
        xs.append(x)
        ys.append(y)
    return np.array(xs), np.array(ys)


def draw_robot(ax, k, i, color, lw=3.0, trail=90, label=None, badge=None, small=False):
    ax.set_xlim(-0.78, 0.78)
    ax.set_ylim(-0.92, 0.92)
    ax.set_aspect("equal")
    ax.set_xticks([])
    ax.set_yticks([])
    for s in ax.spines.values():
        s.set_color("#dddddd")
    ax.plot([-0.6, 0.6], [0, 0], color="#bbbbbb", lw=1.4, zorder=1)
    ax.plot([-0.6, -0.6], [-0.03, 0.03], color="#bbbbbb", lw=1.4)
    ax.plot([0.6, 0.6], [-0.03, 0.03], color="#bbbbbb", lw=1.4)
    if trail:
        k0 = max(0, k - trail)
        tx, ty = [], []
        for kk in range(k0, k + 1, 2):
            xs, ys = joints(kk, i)
            tx.append(xs[-1])
            ty.append(ys[-1])
        ax.plot(tx, ty, color=color, alpha=0.25, lw=1.2, zorder=2)
    xs, ys = joints(k, i)
    ax.plot([xs[0] - 0.05, xs[0] + 0.05], [0, 0], color=INK, lw=7 if not small else 5,
            solid_capstyle="butt", zorder=4)
    ax.plot(xs, ys, color=color, lw=lw, solid_capstyle="round", zorder=5)
    ax.plot(xs[1:], ys[1:], "o", color=color, ms=5 if not small else 3.5, zorder=6)
    if label:
        ax.text(0, 0.86, label, ha="center", va="top", fontsize=11 if not small else 9, color=INK)
    if badge:
        txt, col = badge
        ax.text(0, -0.84, txt, ha="center", va="bottom", fontsize=13 if not small else 10,
                color="white", weight="bold",
                bbox=dict(boxstyle="round,pad=0.28", fc=col, ec=col))


def frame_to_rgb(fig):
    fig.canvas.draw()
    buf = np.asarray(fig.canvas.buffer_rgba())[:, :, :3]
    return buf.copy()


def sim_index(t, speed=1.0):
    """Simulation step for montage time t (s)."""
    return int(min(STEPS - 1, t * speed / DT))


def main():
    fig = plt.figure(figsize=(W / 100, H / 100), dpi=100)
    writer = imageio.get_writer(OUT, fps=FPS, codec="libx264", quality=8,
                               macro_block_size=8, ffmpeg_log_level="error")
    nom = LAB.index("nominal")
    holds = [i for i, s in enumerate(LAB) if s.startswith("holds")]
    falls = [i for i, s in enumerate(LAB) if s.startswith("falls")]
    twinA, twinB = LAB.index("twin A"), LAB.index("twin B (+0.1% mass)")
    grid = json.load(open(os.path.join(ROOT, "results", "fractal", "grid_A.json"), encoding="utf-8"))
    n = grid["meta"]["n"]
    M = np.array(grid["success"][: n * n]).reshape(n, n)
    gx, gy = np.array(grid["meta"]["xs"]), np.array(grid["meta"]["ys"])
    an = json.load(open(os.path.join(ROOT, "results", "fractal", "analysis_gate.json"), encoding="utf-8"))
    rows = [r for r in an["pairs"]["rows"] if r["eps"] > 0]
    ep = np.array([r["eps"] for r in rows])
    ff = np.array([r["f"] for r in rows]) - an["pairs"]["p0"]

    scenes = [("title", 4.0), ("hero", 16.0), ("grid", 16.0), ("twins", 16.0), ("result", 8.0)]
    t = 0.0
    total = sum(s[1] for s in scenes)
    nframe = 0
    for name, dur in scenes:
        for f in range(int(dur * FPS)):
            tt = f / FPS
            fig.clf()
            fig.patch.set_facecolor("white")

            if name == "title":
                ax = fig.add_axes([0.06, 0.1, 0.42, 0.8])
                k = sim_index(min(tt * 3.0, 11.9))
                draw_robot(ax, k, nom, BLUE, trail=200)
                ax.set_frame_on(False)
                fig.text(0.54, 0.66, "Triple inverted pendulum", fontsize=30, weight="bold")
                fig.text(0.54, 0.585, "learned swing-up, Isaac Lab", fontsize=20, color=GREY)
                fig.text(0.54, 0.47, "frozen PPO policy  ·  1 motor, 3 passive joints", fontsize=15, color=INK)
                fig.text(0.54, 0.41, "12 s episodes  ·  every clip is recorded simulation", fontsize=15, color=GREY)

            elif name == "hero":
                k = sim_index(tt * 0.78)
                ax = fig.add_axes([0.05, 0.08, 0.5, 0.86])
                badge = ("SUCCESS", GREEN) if tt > dur - 3.5 else None
                draw_robot(ax, k, nom, BLUE, trail=140, badge=badge)
                fig.text(0.06, 0.95, "Nominal robot: hang → pump → catch → hold", fontsize=19, weight="bold")
                ax2 = fig.add_axes([0.62, 0.56, 0.33, 0.33])
                ax2.plot(np.arange(k + 1) * DT, TIP[: k + 1, nom], color=BLUE, lw=1.6)
                ax2.axhline(0.9, color=GREEN, ls="--", lw=1.0)
                ax2.set_xlim(0, STEPS * DT)
                ax2.set_ylim(-1.05, 1.15)
                ax2.set_xlabel("time [s]", fontsize=10)
                ax2.set_title("tip height (1 = fully upright)", fontsize=11)
                ax3 = fig.add_axes([0.62, 0.12, 0.33, 0.3])
                ax3.plot(np.arange(k + 1) * DT, CART[: k + 1, nom], color=INK, lw=1.4)
                ax3.axhline(0.6, color=RED, ls="--", lw=1.0)
                ax3.axhline(-0.6, color=RED, ls="--", lw=1.0)
                ax3.set_xlim(0, STEPS * DT)
                ax3.set_ylim(-0.72, 0.72)
                ax3.set_xlabel("time [s]", fontsize=10)
                ax3.set_title("cart position [m] (rail limits dashed)", fontsize=11)

            elif name == "grid":
                k = sim_index(tt * 0.78)
                fig.text(0.5, 0.955, "Same policy, different robots: link masses changed by a few percent",
                         fontsize=19, weight="bold", ha="center")
                order = holds + falls
                for p, i in enumerate(order):
                    r, c = divmod(p, 3)
                    ax = fig.add_axes([0.035 + c * 0.325, 0.50 - r * 0.45, 0.30, 0.40])
                    col = GREEN if SUCC[i] else RED
                    lab = LAB[i].split(": ")[1]
                    badge = (("SUCCESS", GREEN) if SUCC[i] else ("FAIL", RED)) if tt > dur - 4.0 else None
                    draw_robot(ax, k, i, col, lw=2.2, trail=90, label=lab, badge=badge, small=True)
                fig.text(0.5, 0.025, "top row: still works   ·   bottom row: same policy, same start, robot fails",
                         fontsize=13, color=GREY, ha="center")

            elif name == "twins":
                k = sim_index(tt * 0.78)
                fig.text(0.5, 0.955, "Twin simulators: arm 1 heavier by 0.1%", fontsize=20, weight="bold", ha="center")
                for p, (i, nm, col) in enumerate(((twinA, "twin A", GREEN), (twinB, "twin B  (+0.1% mass)", RED))):
                    ax = fig.add_axes([0.03 + p * 0.33, 0.30, 0.31, 0.58])
                    badge = ((("SUCCESS", GREEN) if SUCC[i] else ("FAIL", RED)) if tt > dur - 4.5 else None)
                    draw_robot(ax, k, i, col, lw=2.6, trail=110, label=nm, badge=badge)
                ax = fig.add_axes([0.71, 0.36, 0.26, 0.46])
                sep = np.abs(TH[:, twinA, :] - TH[:, twinB, :]).max(-1)
                ax.semilogy(np.arange(k + 1) * DT, np.maximum(sep[: k + 1], 1e-8), color=INK, lw=1.5)
                ax.set_xlim(0, STEPS * DT)
                ax.set_ylim(1e-8, 10)
                ax.set_xlabel("time [s]", fontsize=10)
                ax.set_title("angle difference [rad]", fontsize=11)
                fig.text(0.5, 0.11, "identical policy, identical start, identical rail",
                         fontsize=14, color=GREY, ha="center")
                fig.text(0.5, 0.045, "a 0.1% model error decides success", fontsize=16, ha="center", weight="bold")

            else:  # result
                a = min(1.0, tt / 1.2)
                fig.text(0.5, 0.955, "Why that matters", fontsize=20, weight="bold", ha="center")
                ax = fig.add_axes([0.07, 0.24, 0.36, 0.6])
                ax.imshow(M.T, origin="lower", cmap="gray", extent=[gx[0], gx[-1], gy[0], gy[-1]],
                          aspect="auto", interpolation="nearest", alpha=a)
                ax.set_xlabel("log arm-1 mass", fontsize=11)
                ax.set_ylabel("log arm-3 mass", fontsize=11)
                ax.set_title("4096 robots: white = success", fontsize=12)
                ax2 = fig.add_axes([0.56, 0.24, 0.36, 0.6])
                m = max(2, int(len(ep) * min(1.0, tt / 2.2)))
                ax2.loglog(ep[:m], ff[:m], "o-", color=BLUE, lw=1.6, ms=5)
                ax2.loglog([1e-3, 0.3], [0.004, 1.2], "k:", lw=1.0)
                ax2.set_xlim(7e-4, 0.45)
                ax2.set_ylim(3e-3, 1.5)
                ax2.set_xlabel("model error", fontsize=11)
                ax2.set_ylabel("chance the outcome flips", fontsize=11)
                ax2.set_title("measured slope 0.28 (smooth would be 1)", fontsize=12)
                if tt > 2.6:
                    fig.text(0.5, 0.12, "halving the uncertainty about success needs ~12x better models",
                             fontsize=17, ha="center", weight="bold")
                if tt > 4.4:
                    fig.text(0.5, 0.055, "github.com/anshM123/pendisacc", fontsize=13, color=GREY, ha="center")

            fig.text(0.985, 0.015, "%02d:%02d" % (int(t + tt) // 60, int(t + tt) % 60),
                     fontsize=9, color="#cccccc", ha="right")
            writer.append_data(frame_to_rgb(fig))
            nframe += 1
        t += dur
    writer.close()
    plt.close(fig)
    print("wrote %s  (%d frames, %.1f s, %d fps)" % (OUT, nframe, nframe / FPS, FPS))
    print("size: %.1f MB" % (os.path.getsize(OUT) / 1e6))


if __name__ == "__main__":
    main()
