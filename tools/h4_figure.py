"""The H4 figure: direction matters under one interface and not the other.

Two panels sharing an axis. x is ||dm||, the Euclidean distance of the model
error in link-mass space -- so moving right means a BIGGER error, and the two
curves within a panel are always at the same x, i.e. the same magnitude. The
only thing that differs between the curves is DIRECTION.

If the interface account is right the left panel (cart velocity) shows a flat
uniform curve and a collapsing transverse curve, while the right panel (cart
force) shows the two curves lying on top of each other.

Reads only results/h4_sweep_*.json. Draws whatever is there; it does not know
what the answer is supposed to be.

  run.cmd tools/h4_figure.py
"""

from __future__ import annotations

import json
import os
import sys

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT = os.path.join(ROOT, "figures", "h4_interface_anisotropy.png")

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402


def series(rows, direction_pred):
    """mean +- spread over policies, as a function of ||dm||."""
    by = {}
    for r in rows:
        if not direction_pred(r["direction"]):
            continue
        by.setdefault(round(r["dm_kg"], 9), []).append(100.0 * r["success"])
    xs = sorted(by)
    mean = [float(np.mean(by[x])) for x in xs]
    lo = [float(np.min(by[x])) for x in xs]
    hi = [float(np.max(by[x])) for x in xs]
    return np.array(xs), np.array(mean), np.array(lo), np.array(hi)


def panel(ax, data, title):
    rows = data["rows"]
    for pred, label, colour in (
        (lambda d: d == "uniform", "uniform  $[c,c,c]$  (equivalence direction)", "#1a7f37"),
        (lambda d: d.startswith("transverse"), "transverse  (one link)", "#cf222e"),
    ):
        x, m, lo, hi = series(rows, pred)
        if len(x) == 0:
            continue
        ax.plot(x, m, "o-", color=colour, lw=2, ms=5, label=label, zorder=3)
        ax.fill_between(x, lo, hi, color=colour, alpha=0.18, lw=0, zorder=2)
    ax.set_title(title, fontsize=11)
    ax.set_xlabel(r"model error $\|\Delta m\|$  [kg]")
    ax.set_ylim(-3, 103)
    ax.grid(alpha=0.25, zorder=0)


def main() -> int:
    paths = {k: os.path.join(ROOT, "results", "h4_sweep_%s.json" % k)
             for k in ("velocity", "force")}
    missing = [k for k, p in paths.items() if not os.path.exists(p)]
    if missing:
        print("missing sweep results for: %s" % ", ".join(missing))
        return 1
    data = {k: json.load(open(p, encoding="utf-8")) for k, p in paths.items()}

    fig, axes = plt.subplots(1, 2, figsize=(9.6, 3.9), sharey=True)
    panel(axes[0], data["velocity"], "cart VELOCITY command")
    panel(axes[1], data["force"], "cart FORCE command")
    axes[0].set_ylabel("swing-up success  [%]")
    axes[0].legend(loc="lower left", fontsize=8.5, framealpha=0.95)
    fig.suptitle("Same plant, same model errors, same magnitudes -- only the "
                 "commanded variable differs", fontsize=11.5, y=1.0)
    fig.tight_layout()
    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    fig.savefig(OUT, dpi=200, bbox_inches="tight")
    print("[out] %s" % OUT)

    # the same numbers as text, so the figure is never the only record
    for k in ("velocity", "force"):
        print("\n%s interface" % k)
        for pred, name in ((lambda d: d == "uniform", "uniform"),
                           (lambda d: d.startswith("transverse"), "transverse")):
            x, m, lo, hi = series(data[k]["rows"], pred)
            print("  %-11s %s" % (name, "  ".join("%.3f:%5.1f%%" % (a, b)
                                                  for a, b in zip(x, m))))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
