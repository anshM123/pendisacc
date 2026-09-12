"""Figure 1: one large panel that states the paper's claim, not eight small ones.

The previous figure was eight postage-stamp frames of a swing-up. It
established that a pendulum exists and little else. This draws the claim
itself: two plants whose parameters differ enormously along the symmetry
generator produce the SAME trajectory, while a far smaller error transverse to
it produces a different one that fails.

Left  -- tip trace of the nominal plant and of the plant with every inertial
         and dissipative parameter, the loop gain and the force limit scaled by
         1000. They are indistinguishable.
Right -- the same overlay for a transverse error a thousand times smaller in
         every norm.

CPU only, via the standalone closed loop.

  run.cmd tools/fig_hero.py
"""

from __future__ import annotations

import os
import sys
from dataclasses import replace

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from dynamics.closed_loop import IQ, NZ, ClosedLoop, SimCfg  # noqa: E402
from dynamics.policy import Actor  # noqa: E402

import matplotlib  # noqa: E402
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

OUT = os.path.join(ROOT, "paper", "figures", "hero.png")
CKPT = os.path.join(ROOT, "logs", "rsl_rl", "tip_swingup",
                    "2026-09-05_21-02-09_rel1", "model_800.pt")
HORIZON = 4.0
C_GROUP = 1000.0
C_TRANS = 1.5          # link 1 only, a far smaller error


def link_geom():
    import yaml
    p = yaml.safe_load(open(os.path.join(ROOT, "configs", "robot",
                                         "triple_pendulum_params.yaml"), encoding="utf-8"))
    b = p["bodies"]
    return [float(b[n].get("length", 2.0 * b[n]["l_com"])) for n in ("link1", "link2", "link3")]


def tip_xy(Z, L):
    """Absolute tip position over time, from absolute link angles."""
    q = Z[:, IQ]
    x = q[:, 0].copy()
    y = np.zeros_like(x)
    for i in range(3):
        x = x + L[i] * np.sin(q[:, 1 + i])
        y = y + L[i] * np.cos(q[:, 1 + i])
    return x, y


def main() -> int:
    if not os.path.exists(CKPT):
        print("missing checkpoint:", CKPT)
        return 1
    actor = Actor(CKPT)
    L = link_geom()
    base = SimCfg()
    base = replace(base, friction=replace(base.friction, model="coulomb",
                                          b_joint=0.004, fc_joint=0.004, b_cart=0.5))
    n = int(HORIZON / base.dt_ctrl)
    z0 = np.zeros(NZ)
    z0[IQ] = [0.0, np.pi, 0.0, 0.0]

    def roll(cfg):
        return ClosedLoop(actor, cfg).rollout(z0, n)

    Zn = roll(base)
    f = base.friction
    grp = replace(base,
                  mass_scale=(C_GROUP,) * 3, inertia_scale=(C_GROUP,) * 3,
                  cart_mass_scale=C_GROUP,
                  drive=replace(base.drive, kv=base.drive.kv * C_GROUP,
                                f_clamp=base.drive.f_clamp * C_GROUP),
                  friction=replace(f, b_joint=f.b_joint * C_GROUP,
                                   fc_joint=f.fc_joint * C_GROUP,
                                   b_cart=f.b_cart * C_GROUP))
    tra = replace(base, mass_scale=(C_TRANS, 1.0, 1.0),
                  inertia_scale=(C_TRANS, 1.0, 1.0))
    Zg, Zt = roll(grp), roll(tra)

    fig, ax = plt.subplots(1, 2, figsize=(7.4, 3.1), sharey=True)
    for a, Z2, ttl, col, note in (
        (ax[0], Zg,
         r"every parameter $\times\,1000$, along $G$",
         "#1a7f37", "indistinguishable"),
        (ax[1], Zt,
         r"one link $\times\,1.5$, transverse to $G$",
         "#cf222e", "diverges, then falls")):
        xn, yn = tip_xy(Zn, L)
        x2, y2 = tip_xy(Z2, L)
        a.plot(xn, yn, lw=2.6, color="0.55", label="nominal", zorder=2)
        a.plot(x2, y2, lw=1.2, color=col, label="perturbed", zorder=3)
        a.set_title(ttl, fontsize=10)
        a.set_xlabel("tip position along rail [m]")
        a.grid(alpha=0.25, zorder=0)
        a.axhline(0, color="0.3", lw=0.8, zorder=1)
        a.text(0.03, 0.04, note, transform=a.transAxes, fontsize=8.5,
               color=col, style="italic")
        a.legend(fontsize=8, loc="upper right", framealpha=0.95)
    ax[0].set_ylabel("tip height [m]")
    d_g = float(np.abs(Zg[:, IQ][:, 1:] - Zn[:, IQ][:, 1:]).max())
    d_t = float(np.abs(Zt[:, IQ][:, 1:] - Zn[:, IQ][:, 1:]).max())
    fig.suptitle(r"max $|\Delta\theta|$ = %.1e rad (left) vs %.2f rad (right)"
                 % (d_g, d_t), fontsize=10.5, y=1.02)
    fig.tight_layout()
    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    fig.savefig(OUT, dpi=220, bbox_inches="tight")
    print("[out] %s" % OUT)
    print("  along the group, c=%g   : max |dtheta| = %.3e rad" % (C_GROUP, d_g))
    print("  transverse, link1 x%.1f  : max |dtheta| = %.3f rad" % (C_TRANS, d_t))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
