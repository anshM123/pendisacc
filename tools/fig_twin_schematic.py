"""Schematic of the twin-simulator measurement: same frozen policy, simulators theta and theta + eps*u,
opposite outcomes, f(eps) = probability that outcomes disagree.

  run.cmd tools/fig_twin_schematic.py   -> figures/fig_twin_schematic.{png,pdf}
"""

import os

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch  # noqa: E402

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT = os.path.join(ROOT, "figures", "fig_twin_schematic")
plt.rcParams.update({"font.size": 8, "font.family": "serif", "mathtext.fontset": "cm"})

GREEN, RED, BLUE, GREY = "#2e7d32", "#c62828", "#1f5fa8", "#555555"


def box(ax, x, y, w, h, text, fc, ec, fs=8, weight="normal", color="k"):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.012,rounding_size=0.02",
                                fc=fc, ec=ec, lw=1.0))
    ax.text(x + w / 2, y + h / 2, text, ha="center", va="center", fontsize=fs, weight=weight, color=color)


def arrow(ax, x0, y0, x1, y1, color=GREY):
    ax.add_patch(FancyArrowPatch((x0, y0), (x1, y1), arrowstyle="-|>", mutation_scale=9, lw=1.0, color=color))


def pendulum(ax, cx, cy, angles, color, s=0.045):
    """Cart plus three links; angles are absolute, 0 = upright."""
    ax.plot([cx - 0.022, cx + 0.022], [cy, cy], color="k", lw=3, solid_capstyle="butt")
    x, y = cx, cy
    for a in angles:
        nx, ny = x + s * np.sin(a), y + s * np.cos(a)
        ax.plot([x, nx], [y, ny], color=color, lw=1.8, solid_capstyle="round")
        ax.plot(nx, ny, "o", color=color, ms=2.2)
        x, y = nx, ny


def main():
    fig = plt.figure(figsize=(7.1, 2.45))
    ax = fig.add_axes([0, 0, 1, 1])
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 0.345)
    ax.set_axis_off()

    # parameter space inset: a rough boundary with theta and theta + eps u straddling it
    ins = fig.add_axes([0.012, 0.16, 0.16, 0.62])
    rng = np.random.default_rng(3)
    t = np.linspace(0, 1, 400)
    yb = 0.5 + 0.18 * np.sin(5 * t) + 0.06 * np.cumsum(rng.normal(0, 0.18, t.size)) / np.sqrt(t.size) * 6
    ins.fill_between(t, yb, 1, color="#f3f3f3")
    ins.fill_between(t, 0, yb, color="#d9ead3")
    ins.plot(t, yb, color="k", lw=0.7)
    th = np.array([0.52, float(np.interp(0.52, t, yb)) - 0.03])
    tb = th + np.array([0.05, 0.075])
    ins.plot(*th, "o", color=GREEN, ms=4)
    ins.plot(*tb, "o", color=RED, ms=4)
    ins.annotate("", xy=tb, xytext=th, arrowprops=dict(arrowstyle="-|>", lw=0.9, color="k"))
    ins.text(th[0] - 0.03, th[1] - 0.07, r"$\theta$", ha="right", fontsize=9, color=GREEN)
    ins.text(tb[0] + 0.03, tb[1] + 0.02, r"$\theta+\epsilon u$", ha="left", fontsize=9, color=RED)
    ins.text(0.05, 0.08, "success region", fontsize=6.5, color=GREEN)
    ins.text(0.05, 0.9, "failure region", fontsize=6.5, color=GREY)
    ins.set_xticks([])
    ins.set_yticks([])
    ins.set_xlim(0, 1)
    ins.set_ylim(0, 1)
    ins.set_title("simulator-parameter space", fontsize=7.5, pad=3)
    for sp in ins.spines.values():
        sp.set_lw(0.6)

    # the two rows
    rows = [(0.235, r"simulator A:  $\theta$", GREEN, "SUCCESS", [0.04, -0.06, 0.05]),
            (0.075, r"simulator B:  $\theta+\epsilon u$", RED, "FAIL", [2.3, 2.9, 3.5])]
    box(ax, 0.2, 0.13, 0.115, 0.085, "same frozen\nPPO policy", "#e8eef8", BLUE, fs=8, weight="bold", color=BLUE)
    hang = [np.pi + 0.03, np.pi + 0.04, np.pi + 0.05]
    for y, label, col, verdict, pose in rows:
        arrow(ax, 0.315, 0.1725, 0.365, y + 0.03, color=GREY)
        box(ax, 0.365, y, 0.19, 0.06, label, "white", col, fs=8.5)
        arrow(ax, 0.555, y + 0.03, 0.6, y + 0.03)
        ax.add_patch(FancyBboxPatch((0.6, y - 0.012), 0.16, 0.084, boxstyle="round,pad=0.004",
                                    fc="#fafafa", ec="#bbbbbb", lw=0.6))
        pendulum(ax, 0.627, y + 0.064, hang, "#888888", s=0.013)
        ax.text(0.68, y + 0.042, "12 s", ha="center", va="center", fontsize=6.5, color=GREY)
        arrow(ax, 0.66, y + 0.028, 0.702, y + 0.028, color="#aaaaaa")
        base = y - 0.004 if verdict == "SUCCESS" else y + 0.064
        pendulum(ax, 0.735, base, pose, col, s=0.013)
        arrow(ax, 0.76, y + 0.03, 0.795, y + 0.03)
        box(ax, 0.795, y + 0.004, 0.08, 0.052, verdict, col, col, fs=8.5, weight="bold", color="white")
    ax.text(0.68, 0.183, "same initial state, same policy", ha="center", va="center", fontsize=6.3, color=GREY)

    # the measured quantity
    ax.plot([0.9, 0.9], [0.07, 0.29], color="#bbbbbb", lw=0.8)
    ax.text(0.953, 0.25, r"$f(\epsilon)=$", ha="center", fontsize=9)
    ax.text(0.953, 0.215, r"$\Pr[S_A\neq S_B]$", ha="center", fontsize=8)
    ax.text(0.953, 0.155, r"$\propto\epsilon^{\alpha}$", ha="center", fontsize=10)
    ax.text(0.953, 0.113, r"$\alpha\approx0.28$", ha="center", fontsize=8, color=RED)
    ax.text(0.953, 0.085, r"smooth: $\alpha=1$", ha="center", fontsize=6, color=GREY)

    ax.text(0.5, 0.318, r"Twin simulators: repeat for many $\theta$, random directions $u$, and scales $\epsilon$",
            ha="center", fontsize=8.5, weight="bold")
    ax.text(0.5, 0.022, r"$\epsilon=10^{-3}$ (0.1% mass change) still flips 5% of outcomes; "
                        r"halving $f$ needs $2^{1/\alpha}\approx12\times$ better model accuracy",
            ha="center", fontsize=7.2, color=GREY)

    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    fig.savefig(OUT + ".png", dpi=300, bbox_inches="tight")
    fig.savefig(OUT + ".pdf", bbox_inches="tight")
    print("wrote", OUT + ".png", OUT + ".pdf")


if __name__ == "__main__":
    main()
