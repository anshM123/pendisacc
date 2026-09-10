"""The two-heatmap figure: transfer geometry under two action interfaces.

Reads results/h4_grid_{velocity,force}.json and draws P_success(c, delta) for
each, on a shared colour scale. It does not know what the answer is supposed to
be; it draws whatever is in the files.

Cells where the drive spent a large fraction of the episode against its force
clamp or speed limit are HATCHED. In those cells the measurement is about
saturation rather than about the interface, and hatching them keeps that
visible instead of letting it be read as geometry.

  run.cmd tools/h4_landscape.py
"""

from __future__ import annotations

import json
import os

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT = os.path.join(ROOT, "figures", "h4_transfer_landscape.png")
SAT = 0.20          # a cell spending more than this against a limit is hatched

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402


def grid(data):
    cs = data["c_values"]
    ds = data["deltas"]
    P = np.full((len(ds), len(cs)), np.nan)
    S = np.zeros((len(ds), len(cs)))
    for r in data["rows"]:
        i, j = ds.index(r["delta"]), cs.index(r["c"])
        vals = [x["success"] for x in data["rows"]
                if x["c"] == r["c"] and x["delta"] == r["delta"]]
        sats = [max(x["force_sat_frac"], x["speed_sat_frac"]) for x in data["rows"]
                if x["c"] == r["c"] and x["delta"] == r["delta"]]
        P[i, j] = 100.0 * float(np.mean(vals))
        S[i, j] = float(np.mean(sats))
    return np.array(cs, float), np.array(ds, float), P, S


def panel(ax, data, title):
    cs, ds, P, S = grid(data)
    im = ax.imshow(P, origin="lower", aspect="auto", cmap="viridis",
                   vmin=0, vmax=100, interpolation="nearest")
    ax.set_xticks(range(len(cs)))
    ax.set_xticklabels(["%g" % c for c in cs])
    ax.set_yticks(range(len(ds)))
    ax.set_yticklabels(["%g" % d for d in ds])
    ax.set_xlabel(r"common scale $c$   (along the equivalence direction)")
    ax.set_title(title, fontsize=11)
    for i in range(P.shape[0]):
        for j in range(P.shape[1]):
            if S[i, j] > SAT:
                ax.add_patch(plt.Rectangle((j - .5, i - .5), 1, 1, fill=False,
                                           hatch="///", edgecolor="white",
                                           linewidth=0.0, alpha=0.55))
            ax.text(j, i, "%.0f" % P[i, j], ha="center", va="center",
                    fontsize=7.5,
                    color="white" if P[i, j] < 55 else "black")
    return im


def main() -> int:
    paths = {k: os.path.join(ROOT, "results", "h4_grid_%s.json" % k)
             for k in ("velocity", "force")}
    missing = [k for k, p in paths.items() if not os.path.exists(p)]
    if missing:
        print("missing grid results for: %s" % ", ".join(missing))
        return 1
    data = {k: json.load(open(p, encoding="utf-8")) for k, p in paths.items()}

    fig, axes = plt.subplots(1, 2, figsize=(10.4, 4.1), sharey=True)
    im = panel(axes[0], data["velocity"], "cart VELOCITY command")
    panel(axes[1], data["force"], "cart FORCE command")
    axes[0].set_ylabel(r"asymmetry $\delta$   (transverse)")
    fig.colorbar(im, ax=axes, label="swing-up success [%]", pad=0.02)
    fig.suptitle("Same plant, same model errors. Only the commanded variable differs. "
                 "Hatched = drive saturated.", fontsize=10.5, y=1.01)
    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    fig.savefig(OUT, dpi=200, bbox_inches="tight")
    print("[out] %s" % OUT)

    # the anisotropy each landscape implies, as text
    for k in ("velocity", "force"):
        cs, ds, P, S = grid(data[k])
        along = P[0, :]                      # delta = 0: pure equivalence direction
        across = P[:, 0]                     # c = 1:     pure transverse
        print("")
        print("%s interface" % k)
        print("  along the equivalence direction (delta=0): %s"
              % "  ".join("c=%g:%.0f%%" % (c, v) for c, v in zip(cs, along)))
        print("  transverse only (c=1):                     %s"
              % "  ".join("d=%g:%.0f%%" % (d, v) for d, v in zip(ds, across)))
        print("  mean along %.1f%%   mean across %.1f%%   anisotropy %+.1f points"
              % (np.nanmean(along), np.nanmean(across),
                 np.nanmean(along) - np.nanmean(across)))
        n_sat = int((S > SAT).sum())
        if n_sat:
            print("  %d of %d cells saturated the drive and are hatched"
                  % (n_sat, S.size))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
