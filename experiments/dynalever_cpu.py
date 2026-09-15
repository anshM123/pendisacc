"""DynaLever, cheap stage (CPU): does the swing-up contain usable dynamical leverage, and which
actuator limit actually binds?

Frozen CORR_s1 on the analytical closed loop (the one policy the CPU model reproduces):

 1. nominal swing-up from the fixed hanging IC; record the full closed-loop state, the cart
    force F = kv (v_ref - xdot) and the action every control step
 2. leverage map: at pulse times every 40 ms, add a small action pulse for ONE step and
    measure the state change after a window (0.25 s and 0.5 s), both
      open loop  (recorded actions replayed after the pulse: the plant's own amplification)
      closed loop (policy keeps running: what the controller actually experiences)
    J(t, T) = d x(t+T) / d a(t), central differences
 3. spectrum: W = sum_t J J^T over windows of pulse times; effective rank
 4. authority sweeps: swing-up success vs force clamp, vs commanded-velocity clip v_max,
    and vs servo lag tau (5 fixed ICs each)

  run.cmd experiments/dynalever_cpu.py
"""

from __future__ import annotations

import json
import os
import sys
from concurrent.futures import ProcessPoolExecutor
from dataclasses import replace

os.environ.setdefault("OMP_NUM_THREADS", "1")
import numpy as np
import torch

torch.set_num_threads(1)
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "tools"))

from dynamics.closed_loop import IA, IQD, IV, NZ, ClosedLoop, DriveCfg, SimCfg  # noqa: E402
from dynamics.policy import Actor, observation  # noqa: E402
from pi_prove import L, ckpt  # noqa: E402

CK = os.path.join(ROOT, "logs", "rsl_rl", "tip_swingup", "2026-09-11_22-27-54_CORR_s1", "model_800.pt")
STEPS, DT = 3000, 1 / 250
OUT = os.path.join(ROOT, "results", "dynalever")
_ACT = {}


def actor():
    if "a" not in _ACT:
        _ACT["a"] = Actor(CK)
    return _ACT["a"]


def ic(k=0):
    z = np.zeros(NZ)
    r = np.random.default_rng(700 + k)
    z[1] = np.pi + 0.03 + (0 if k == 0 else r.uniform(-0.02, 0.02))
    z[2:4] = np.pi + 0.01
    return z


class Replay:
    """Plays recorded actions (open loop), optionally with a one-step pulse."""

    def __init__(self, acts, k0=0, pulse_at=None, delta=0.0):
        self.acts, self.k, self.p, self.d = acts, k0, pulse_at, delta

    def __call__(self, _obs):
        a = self.acts[min(self.k, len(self.acts) - 1)] + (self.d if self.k == self.p else 0.0)
        self.k += 1
        return np.array([[a]])


class Pulsed:
    """Closed loop: the policy, plus a one-step additive pulse at step p."""

    def __init__(self, k0=0, pulse_at=None, delta=0.0):
        self.k, self.p, self.d = k0, pulse_at, delta

    def __call__(self, obs):
        a = float(actor()(obs)[0, 0]) + (self.d if self.k == self.p else 0.0)
        self.k += 1
        return np.array([[a]])


def success_of(Z):
    tip = (L[None, :] * np.cos(Z[:, 1:4])).sum(1) / L.sum()
    hold = np.mean(tip[int(0.75 * len(tip)):] > 0.9)
    return bool((np.abs(Z[:, 0]) <= 0.6).all() and (tip > 0.9).any() and hold > 0.95), float(hold)


def nominal(cfg=None, k=0):
    cfg = cfg or SimCfg()
    act = actor()
    loop = ClosedLoop(act, cfg)
    z = ic(k)
    Z, A, F = [z.copy()], [], []
    for _ in range(STEPS):
        a = float(act(observation(z[0:4], z[4:8], z[IA]))[0, 0])
        A.append(a)
        z = loop.step(z)
        Z.append(z.copy())
        F.append(cfg.drive.kv * (z[IV] - z[IQD][0]))
        if abs(z[0]) > 0.6:
            break
    return np.array(Z), np.array(A), np.array(F)


def leverage(job):
    kind, k, win, Zs, As = job
    cfg = SimCfg()
    delta = 1e-3
    n = int(win / DT)
    outs = []
    for sgn in (1, -1):
        pol = Replay(As, k0=k, pulse_at=k, delta=sgn * delta) if kind == "open" else Pulsed(k0=k, pulse_at=k, delta=sgn * delta)
        loop = ClosedLoop(pol, cfg)
        outs.append(loop.rollout(Zs[k].copy(), n)[-1, :8])
    return kind, k, win, ((outs[0] - outs[1]) / (2 * delta)).tolist()


def sweep_one(job):
    name, val, k = job
    d = DriveCfg()
    if name == "f_clamp":
        d = replace(d, f_clamp=val)
    elif name == "v_max":
        d = replace(d, v_max=val)
    elif name == "tau":
        d = replace(d, tau=val)
    Z, _, _ = nominal(SimCfg(drive=d), k)
    ok, hold = success_of(Z) if len(Z) == STEPS + 1 else (False, 0.0)
    return name, val, k, ok


def main():
    os.makedirs(OUT, exist_ok=True)
    Z, A, F = nominal()
    ok, hold = success_of(Z)
    tip = (L[None, :] * np.cos(Z[:, 1:4])).sum(1) / L.sum()
    t_up = float(np.argmax(tip > 0.9) * DT) if (tip > 0.9).any() else None
    print("nominal CPU swing-up: success %s hold %.3f, first upright %.2f s, peak |F| %.1f N (clamp 349.5), "
          "peak |a| %.2f, peak |v_ref| %.2f m/s" % (ok, hold, t_up or -1, np.abs(F).max(), np.abs(A).max(),
                                                      np.abs(Z[:, IV]).max()), flush=True)
    pulse_steps = list(range(0, int((t_up or 6.0) / DT) + 250, 10))    # every 40 ms through swing-up and capture
    jobs = [(kind, k, win, Z, A) for kind in ("open", "closed") for win in (0.25, 0.5) for k in pulse_steps
            if k + int(win / DT) < len(Z)]
    with ProcessPoolExecutor(16) as ex:
        lev = list(ex.map(leverage, jobs, chunksize=8))
        sw = list(ex.map(sweep_one, [(nm, v, k) for nm, vals in (
            ("f_clamp", (349.5, 200, 120, 80, 60, 40, 30, 20, 15, 10)),
            ("v_max", (4.0, 3.0, 2.5, 2.0, 1.5, 1.25, 1.0, 0.75, 0.5)),
            ("tau", (0.05, 0.1, 0.13, 0.16, 0.2, 0.25, 0.3))) for v in vals for k in range(5)]))
    res = {"checkpoint": CK, "nominal": {"success": ok, "hold": hold, "t_first_upright_s": t_up,
                                        "peak_abs_force_N": float(np.abs(F).max()),
                                        "p99_abs_force_N": float(np.percentile(np.abs(F), 99)),
                                        "peak_abs_action": float(np.abs(A).max()),
                                        "frac_action_at_clip": float(np.mean(np.abs(A) >= 0.999)),
                                        "peak_abs_vref": float(np.abs(Z[:, IV]).max())}}
    for kind in ("open", "closed"):
        for win in (0.25, 0.5):
            rows = sorted([(k, np.array(j)) for kd, k, w, j in lev if kd == kind and w == win])
            ks = np.array([k for k, _ in rows]) * DT
            Js = np.array([j for _, j in rows])
            Lnorm = np.linalg.norm(Js[:, 1:4], axis=1)          # task-relevant: link angles
            U = Js[:, 1:4] / np.maximum(np.linalg.norm(Js[:, 1:4], axis=1, keepdims=True), 1e-12)
            W = Js[:, 1:4].T @ Js[:, 1:4]
            ev = np.sort(np.linalg.eigvalsh(W))[::-1]
            eff_rank = float(ev.sum() ** 2 / (ev ** 2).sum())
            res["%s_%.2fs" % (kind, win)] = {
                "t": ks.tolist(), "angle_leverage": Lnorm.tolist(),
                "leverage_max_over_median": float(Lnorm.max() / np.median(Lnorm)),
                "leverage_p90_over_p10": float(np.percentile(Lnorm, 90) / max(np.percentile(Lnorm, 10), 1e-12)),
                "W_angle_eigs": ev.tolist(), "effective_rank_of_3": eff_rank,
                "direction_mean_abs_cos_between_top_windows": float(np.mean(np.abs(U[np.argsort(-Lnorm)[:10]] @ U[np.argsort(-Lnorm)[:10]].T)))}
            print("%-6s window %.2fs: angle leverage max/median %.1f, p90/p10 %.1f, effective rank %.2f of 3, eigs %s"
                  % (kind, win, res["%s_%.2fs" % (kind, win)]["leverage_max_over_median"],
                     res["%s_%.2fs" % (kind, win)]["leverage_p90_over_p10"], eff_rank, np.round(ev / ev[0], 4)), flush=True)
    for nm in ("f_clamp", "v_max", "tau"):
        vals = sorted({v for n_, v, _, _ in sw if n_ == nm}, reverse=(nm != "tau"))
        tab = {v: float(np.mean([o for n_, vv, _, o in sw if n_ == nm and vv == v])) for v in vals}
        res["sweep_" + nm] = tab
        print("sweep %-7s: %s" % (nm, "  ".join("%g:%d%%" % (v, round(100 * s)) for v, s in tab.items())), flush=True)
    json.dump(res, open(os.path.join(OUT, "cpu_stage.json"), "w"), indent=1)
    print("wrote results/dynalever/cpu_stage.json")


if __name__ == "__main__":
    main()
