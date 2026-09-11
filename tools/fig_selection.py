"""Figure: in-simulator success carries no information about transfer.

Left  -- every H3 policy, own-simulator success against success in R*. The
         x-axis spans 2 points and the y-axis spans 100.
Right -- the controlled version: the nominal arm alone, where simulator,
         protocol, reward, architecture and iteration count are identical and
         only the random seed differs.

Reads results/selection_blindness.json only.

  run.cmd tools/fig_selection.py
"""

from __future__ import annotations

import json
import os

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SRC = os.path.join(ROOT, "results", "selection_blindness.json")
OUT = os.path.join(ROOT, "figures", "selection_blindness.png")

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402


def main() -> int:
    if not os.path.exists(SRC):
        print("missing %s" % SRC)
        return 1
    d = json.load(open(SRC, encoding="utf-8"))
    rows = d["rows"]

    fig, ax = plt.subplots(1, 2, figsize=(7.2, 3.0))

    fam = {}
    for r in rows:
        fam.setdefault(r["policy"].rsplit("_s", 1)[0], []).append(r)
    cmap = plt.get_cmap("tab10")
    for i, (k, v) in enumerate(sorted(fam.items())):
        ax[0].scatter([r["own_sim"] for r in v], [r["in_R_star"] for r in v],
                      s=34, color=cmap(i), label=k.replace("S_", ""), zorder=3)
    ax[0].set_xlabel("success in its OWN training simulator [%]")
    ax[0].set_ylabel(r"success in $R^\star$ [%]")
    ax[0].set_xlim(96, 101)
    ax[0].set_ylim(-5, 105)
    ax[0].grid(alpha=0.25, zorder=0)
    ax[0].legend(fontsize=6.4, loc="center left", framealpha=0.95)
    ax[0].set_title(r"$\rho = %+.3f$,  $p = %.2f$,  $n = %d$"
                    % (d["spearman_rho"], d["permutation_p"], d["n"]),
                    fontsize=9)

    nom = d["nominal_arm"]["rows"]
    x = np.arange(len(nom))
    ax[1].bar(x - 0.19, [r["own_sim"] for r in nom], 0.38,
              label="own simulator", color="#1a7f37")
    ax[1].bar(x + 0.19, [r["in_R_star"] for r in nom], 0.38,
              label=r"deployed in $R^\star$", color="#cf222e")
    ax[1].set_xticks(x)
    ax[1].set_xticklabels(["seed %d" % (i + 1) for i in range(len(nom))], fontsize=8)
    # headroom so the legend cannot sit on top of a full-height bar and make
    # a 100% result read as truncated
    ax[1].set_ylim(0, 132)
    ax[1].set_yticks([0, 25, 50, 75, 100])
    ax[1].set_ylabel("success [%]")
    ax[1].grid(alpha=0.25, axis="y", zorder=0)
    for i, r in enumerate(nom):
        ax[1].text(i - 0.19, r["own_sim"] + 2, "%.0f" % r["own_sim"],
                   ha="center", fontsize=7)
        ax[1].text(i + 0.19, r["in_R_star"] + 2, "%.0f" % r["in_R_star"],
                   ha="center", fontsize=7)
    ax[1].legend(fontsize=7.2, loc="upper center", ncol=2, framealpha=0.95)
    ax[1].set_title("identical settings; only the seed differs\n"
                    "own-sim spread %.1f pts, $R^\\star$ spread %.1f pts"
                    % (d["nominal_arm"]["own_sim_spread"],
                       d["nominal_arm"]["R_star_spread"]), fontsize=9)

    fig.tight_layout()
    fig.savefig(OUT, dpi=200, bbox_inches="tight")
    print("[out] %s" % OUT)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
