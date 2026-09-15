"""Paper figures for the predictability study, from results/fractal/*.json only.

  fig_pred_main.png   A: robot  B: m1 x m3 transfer map  C: servo-lag map  D: f(eps) log-log with fits
  fig_pred_repl.png   f(eps) for every replication (policy, planes, ICs, CPU dynamics, state, resolution)
  fig_pred_mech.png   twin separation, small-eps floor, cartpole map

  run.cmd tools/fractal_figures.py
"""

from __future__ import annotations

import json
import os
import sys

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "tools"))
from fractal_analyze import pair_stats  # noqa: E402

R = os.path.join(ROOT, "results", "fractal")
FIG = os.path.join(ROOT, "figures")
plt.rcParams.update({"font.size": 8, "axes.titlesize": 8, "axes.labelsize": 8, "legend.fontsize": 6.5})


def J(n):
    p = os.path.join(R, n)
    return json.load(open(p, encoding="utf-8")) if os.path.exists(p) else None


def curve(ax, fname, label, color, marker="o", fit=True):
    d = J(fname)
    if d is None:
        return None
    s = pair_stats(d)
    rows = [r for r in s["rows"] if r["eps"] > 0]
    e = np.array([r["eps"] for r in rows])
    f = np.array([r["f"] for r in rows]) - s["p0"]
    se = np.array([r["se"] for r in rows])
    ok = f > 0
    ax.errorbar(e[ok], f[ok], yerr=se[ok], fmt=marker, ms=3, lw=0.8, color=color, capsize=1.5,
                label="%s  $\\alpha$=%.2f" % (label, s["alpha"]) if s["alpha"] else label)
    if fit and s["alpha"] and s["usable_eps"]:
        ue = np.array(s["usable_eps"])
        fu = np.array([r["f"] for r in rows if r["eps"] in s["usable_eps"]]) - s["p0"]
        b = np.polyfit(np.log(ue), np.log(fu), 1)
        xx = np.logspace(np.log10(ue.min()), np.log10(ue.max()), 20)
        ax.plot(xx, np.exp(np.polyval(b, np.log(xx))), "-", lw=0.8, color=color, alpha=0.7)
    return s


def grid_img(ax, fname, xlabel, ylabel, title, ytrans=None):
    g = J(fname)
    if g is None:
        ax.set_axis_off()
        return
    n = g["meta"]["n"]
    M = np.array(g["success"][: n * n]).reshape(n, n)
    xs, ys = g["meta"]["xs"], g["meta"]["ys"]
    ext = [xs[0], xs[-1], ys[0], ys[-1]] if ytrans is None else [xs[0], xs[-1], ytrans(ys[0]), ytrans(ys[-1])]
    ax.imshow(M.T, origin="lower", cmap="gray", extent=ext, aspect="auto", interpolation="nearest")
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_title(title)


def main():
    os.makedirs(FIG, exist_ok=True)
    fig = plt.figure(figsize=(7.1, 2.25))
    gs = fig.add_gridspec(1, 4, width_ratios=[1, 1, 1, 1.25], wspace=0.45)
    ax = fig.add_subplot(gs[0])
    hero = os.path.join(FIG, "hero.png")
    if os.path.exists(hero):
        img = plt.imread(hero)
        ax.imshow(img)
    ax.set_axis_off()
    ax.set_title("(a) triple pendulum, frozen PPO")
    grid_img(fig.add_subplot(gs[1]), "grid_A.json", "$\\log(m_1/m_1^0)$", "$\\log(m_3/m_3^0)$", "(b) mass plane (white = success)")
    grid_img(fig.add_subplot(gs[2]), "P2_grid.json", "$\\log(m_1/m_1^0)$", "servo lag $\\tau$ [s]", "(c) mass $\\times$ servo lag",
             ytrans=lambda y: 0.1 * np.exp(y))
    ax = fig.add_subplot(gs[3])
    curve(ax, "pairs.json", "mass plane", "C0")
    curve(ax, "P2_pairs.json", "mass$\\times$lag", "C3", "s")
    ax.plot([1e-3, 0.3], [0.004, 1.2], "k:", lw=0.8, label="smooth boundary ($\\alpha$=1)")
    ax.set_xscale("log"); ax.set_yscale("log")
    ax.set_xlabel("perturbation size $\\epsilon$ (log units)")
    ax.set_ylabel("$P$(outcome flips) $-$ floor")
    ax.set_title("(d) outcome predictability")
    ax.legend(loc="lower right", frameon=False)
    fig.savefig(os.path.join(FIG, "fig_pred_main.png"), dpi=300, bbox_inches="tight")
    plt.close(fig)

    fig, axs = plt.subplots(1, 3, figsize=(7.1, 2.2))
    ax = axs[0]
    curve(ax, "pairs.json", "Isaac, policy A", "C0")
    curve(ax, "pairs_T5orbit.json", "Isaac, policy B", "C1", "^")
    curve(ax, "R1_cpu_pairs.json", "CPU dynamics, policy A", "C2", "D")
    curve(ax, "R4_fine_pairs.json", "Isaac, dt/2", "C4", "v")
    ax.set_title("(a) policies, simulators, resolution")
    ax = axs[1]
    for i in range(5):
        curve(ax, "R2_ic%d_pairs.json" % i, "IC%d" % i, "C%d" % i, "o", fit=False)
    ax.set_title("(b) five initial conditions")
    ax = axs[2]
    curve(ax, "R3_state_pairs.json", "initial-state plane", "C5", "o")
    curve(ax, "pairs_zoomZ2.json", "zoom, main boundary", "C6", "s")
    curve(ax, "pairs_zoom.json", "zoom, isolated island", "C7", "x", fit=False)
    ax.set_title("(c) state space and zoom windows")
    for ax in axs:
        ax.set_xscale("log"); ax.set_yscale("log")
        ax.set_xlabel("$\\epsilon$")
        ax.legend(loc="lower right", frameon=False)
    axs[0].set_ylabel("$P$(flip) $-$ floor")
    fig.tight_layout()
    fig.savefig(os.path.join(FIG, "fig_pred_repl.png"), dpi=300, bbox_inches="tight")
    plt.close(fig)

    fig, axs = plt.subplots(1, 3, figsize=(7.1, 2.1))
    z = os.path.join(R, "ftle_CORR_s1.npz")
    if os.path.exists(z):
        d = np.load(z)
        D, eps, dt = d["D"], d["eps"], float(d["dt"])
        t = np.arange(D.shape[0]) * dt
        for i, e in enumerate(eps):
            fin = D[-1, i, :]
            div = fin > 0.3
            if div.sum():
                axs[0].plot(t, np.median(np.log10(np.maximum(D[:, i, div], 1e-12)), axis=1), lw=0.9,
                            label="$\\epsilon$=%g (%d%% diverge)" % (e, round(100 * div.mean())))
        axs[0].set_xlabel("time [s]"); axs[0].set_ylabel("median $\\log_{10}$ twin separation")
        axs[0].set_title("(a) outcome-divergent twins"); axs[0].legend(frameon=False)
    for fname, lab, c in (("pairs.json", "$\\epsilon\\geq10^{-3}$", "C0"), ("pairs_small_EXPLORATORY.json", "$\\epsilon\\leq10^{-3}$", "C1")):
        dd = J(fname)
        if dd:
            s = pair_stats(dd)
            rows = [r for r in s["rows"] if r["eps"] > 0]
            axs[1].errorbar([r["eps"] for r in rows], [r["f"] for r in rows], yerr=[r["se"] for r in rows],
                            fmt="o", ms=3, color=c, capsize=1.5, label=lab)
            axs[1].axhline(s["p0"], color=c, ls=":", lw=0.8)
    axs[1].set_xscale("log"); axs[1].set_yscale("log"); axs[1].set_xlabel("$\\epsilon$")
    axs[1].set_ylabel("$P$(flip)"); axs[1].set_title("(b) resolved range and floor"); axs[1].legend(frameon=False)
    grid_img(axs[2], "cartpole/grid.json", "$\\log$ pole mass", "$\\log$ cart mass", "(c) stock cartpole control")
    fig.tight_layout()
    fig.savefig(os.path.join(FIG, "fig_pred_mech.png"), dpi=300, bbox_inches="tight")
    plt.close(fig)
    print("wrote figures/fig_pred_main.png, fig_pred_repl.png, fig_pred_mech.png")


if __name__ == "__main__":
    main()
