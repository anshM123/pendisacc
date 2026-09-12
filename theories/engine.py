"""Fast exact linearisation of the CPU closed loop at the upright, for bulk theory screening.

dynamics/closed_loop.py rebuilds a sympy model per plant (1.5 s). Screening
dozens of theories over thousands of plants needs microseconds. The mass matrix
and the gravity stiffness are affine in the seven inertial parameters at fixed
geometry, so eight model builds per parameter file give every plant exactly.
The discrete map then reproduces closed_loop.ClosedLoop.step's own integrator
(semi-implicit Euler substeps, first-order lag, delay line) linearised, not a
continuous-time approximation of it. validate() checks this against central
differences of the real step.
"""

from __future__ import annotations

import glob
import os
import sys
from functools import lru_cache

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from dynamics.closed_loop import DMAX, IA, IDLY, IQ, IQD, IV, IVD, NZ, DriveCfg, FrictionCfg, SimCfg  # noqa: E402

PARAMS = {
    "corrected": os.path.join(ROOT, "configs", "robot", "triple_pendulum_params.yaml"),
    "original": os.path.join(ROOT, "configs", "robot", "triple_pendulum_params_ORIGINAL_default_density.yaml"),
}
CACHE = os.path.join(ROOT, "theories", "cache")
YIDX = list(range(IQ.start, IQ.stop)) + list(range(IQD.start, IQD.stop)) + [IA]
NAMES7 = ["m1", "m2", "m3", "I1", "I2", "I3", "mc"]


def _build(asset, ms=(1, 1, 1), is_=(1, 1, 1), cs=1.0):
    from dataclasses import replace
    from dynamics.analytical.triple_pendulum import PendulumParams, TriplePendulumModel
    p = PendulumParams.from_yaml(PARAMS[asset])
    links = [replace(l, m=l.m * a, I=l.I * b) for l, a, b in zip(p.links, ms, is_)]
    return TriplePendulumModel(replace(p, links=links, m_cart=p.m_cart * cs,
                                       b_cart=0.0, fc_cart=0.0, b_joint=[0.0] * 3, fc_joint=[0.0] * 3)), p


def _mk(model, q0):
    M = np.asarray(model.M(q0), float)
    e = 1e-6
    K = np.empty((4, 4))
    for j in range(4):
        d = np.zeros(4)
        d[j] = e
        K[:, j] = (np.asarray(model.G(q0 + d), float) - np.asarray(model.G(q0 - d), float)) / (2 * e)
    return M, K


@lru_cache(maxsize=None)
def basis(asset="corrected"):
    """Affine basis: M(s) = M0 + sum_j (s_j - 1) dM_j, same for gravity stiffness, at upright and hanging."""
    os.makedirs(CACHE, exist_ok=True)
    path = os.path.join(CACHE, "basis_%s.npz" % asset)
    if os.path.exists(path) and os.path.getmtime(path) > os.path.getmtime(PARAMS[asset]):
        z = np.load(path)
        return {k: z[k] for k in z.files}
    up, hang = np.zeros(4), np.array([0.0, np.pi, np.pi, np.pi])
    m0, p = _build(asset)
    out = {}
    for tag, q0 in (("up", up), ("hang", hang)):
        M0, K0 = _mk(m0, q0)
        dM, dK = [], []
        for j in range(7):
            ms, is_, cs = [1, 1, 1], [1, 1, 1], 1.0
            if j < 3:
                ms[j] = 2
            elif j < 6:
                is_[j - 3] = 2
            else:
                cs = 2.0
            mj, _ = _build(asset, ms, is_, cs)
            Mj, Kj = _mk(mj, q0)
            dM.append(Mj - M0)
            dK.append(Kj - K0)
        out["M0_" + tag], out["K0_" + tag] = M0, K0
        out["dM_" + tag], out["dK_" + tag] = np.array(dM), np.array(dK)
    out["B"] = np.asarray(m0.B(), float).ravel()
    out["L"] = np.array([l.L for l in p.links])
    out["lc"] = np.array([l.lc for l in p.links])
    out["m"] = np.array([l.m for l in p.links] + [p.m_cart])
    out["I"] = np.array([l.I for l in p.links])
    np.savez(path, **out)
    return out


def plant_mats(s7, asset="corrected", at="up"):
    b = basis(asset)
    s = np.asarray(s7, float) - 1.0
    M = b["M0_" + at] + np.tensordot(s, b["dM_" + at], 1)
    K = b["K0_" + at] + np.tensordot(s, b["dK_" + at], 1)
    return M, K


def joint_damping_matrix(b_joint):
    """Viscous friction on RELATIVE joint rates, as closed_loop.friction_forces."""
    D = np.zeros((4, 4))
    for i in range(3):
        g = np.zeros(4)
        g[1 + i] = 1.0
        if i > 0:
            g[i] = -1.0
        D += b_joint * np.outer(g, g)
    return D


def open_loop(s7=(1,) * 7, asset="corrected", kv=400.0, tau=0.100, delay=0, dt=1 / 250,
              substeps=None, b_joint=0.0, b_cart=0.0, action_scale=4.0, extra_D=None,
              order=1, omega_n=0.0, zeta=0.9):
    """Discrete (A, B) with z_{k+1} = A z_k + B a_k, state layout of closed_loop (NZ=15)."""
    M, Kg = plant_mats(s7, asset)
    Bf = basis(asset)["B"]
    D = joint_damping_matrix(b_joint)
    D[0, 0] += b_cart
    if extra_D is not None:
        D = D + extra_D
    Mi = np.linalg.inv(M)
    if substeps is None:
        substeps = max(1, int(round(dt / 0.0004)))
    h = dt / substeps
    # substep on x = [q, qd] with v_ref held: qd' = qd + h Mi (Bf kv (v - qd0) - Kg q - D qd); q' = q + h qd'
    Ad = np.zeros((8, 8))
    Ad[4:, 4:] = -Mi @ (np.outer(Bf, np.eye(4)[0]) * kv + D)
    Ad[4:, :4] = -Mi @ Kg
    bv = Mi @ Bf * kv
    S = np.eye(8)
    Sv = np.zeros(8)
    step = np.eye(8)
    step[4:, :] += h * Ad[4:, :]
    stepv = np.zeros(8)
    stepv[4:] = h * bv
    # q' = q + h qd'
    step[:4, :] += h * step[4:, :]
    stepv[:4] += h * stepv[4:]
    for _ in range(substeps):
        S = step @ S
        Sv = step @ Sv + stepv
    alpha = 1.0 if tau <= 0 else dt / (dt + tau)
    A = np.zeros((NZ, NZ))
    B = np.zeros(NZ)
    # v_eff: from action directly (delay 0) or from the queue
    if delay <= 0:
        veff_z, veff_a = np.zeros(NZ), action_scale
    else:
        veff_z = np.zeros(NZ)
        veff_z[IDLY.start + min(delay, DMAX) - 1] = 1.0
        veff_a = 0.0
    if order == 2:
        # closed_loop: vdd = wn^2 (v_eff - v) - 2 zeta wn vd; vd' = vd + dt vdd; v' = v + dt vd'
        wn = float(omega_n) if omega_n else 1.0 / max(tau, 1e-9)
        vd_z = np.zeros(NZ)
        vd_z[IVD] = 1.0 - 2.0 * zeta * wn * dt
        vd_z[IV] = -dt * wn ** 2
        vd_z += dt * wn ** 2 * veff_z
        vd_a = dt * wn ** 2 * veff_a
        vr_z = np.zeros(NZ)
        vr_z[IV] = 1.0
        vr_z += dt * vd_z
        vr_a = dt * vd_a
        A[IVD, :], B[IVD] = vd_z, vd_a
    else:
        # v_ref' = (1-alpha) v_ref + alpha v_eff
        vr_z = np.zeros(NZ)
        vr_z[IV] = 1.0 - alpha
        vr_z += alpha * veff_z
        vr_a = alpha * veff_a
    x_idx = list(range(8))
    A[np.ix_(x_idx, x_idx)] = S
    A[:8, :] += np.outer(Sv, vr_z)
    B[:8] += Sv * vr_a
    A[IV, :], B[IV] = vr_z, vr_a
    B[IA] = 1.0
    qi = IDLY.start
    B[qi] = action_scale
    for j in range(1, DMAX):
        A[qi + j, qi + j - 1] = 1.0
    return A, B


def closed(A, B, K):
    C = np.zeros((len(YIDX), NZ))
    C[np.arange(len(YIDX)), YIDX] = 1.0
    return A + np.outer(B, K) @ C


def rho(A, B, K):
    return float(np.max(np.abs(np.linalg.eigvals(closed(A, B, K)))))


def run_dir(tag):
    ds = sorted(d for d in glob.glob(os.path.join(ROOT, "logs", "rsl_rl", "tip_swingup", "*" + tag))
                if "VOID" not in os.path.basename(d) or "VOID" in tag)
    return ds[-1]


def checkpoints(tag):
    fs = glob.glob(os.path.join(run_dir(tag), "model_*.pt"))
    return sorted(fs, key=lambda f: int("".join(c for c in os.path.basename(f) if c.isdigit())))


@lru_cache(maxsize=None)
def policy_gain(ckpt_path):
    from dynamics.policy import Actor, observation
    act = Actor(ckpt_path)
    z0 = np.zeros(NZ)

    def a(z):
        return float(act(observation(z[IQ], z[IQD], z[IA]))[0, 0])

    K = np.empty(len(YIDX))
    for j, i in enumerate(YIDX):
        dz = np.zeros(NZ)
        dz[i] = 1e-5
        K[j] = (a(z0 + dz) - a(z0 - dz)) / 2e-5
    return tuple(K), a(z0)


def validate():
    """Engine vs central differences of the real nonlinear step, several plants."""
    sys.path.insert(0, os.path.join(ROOT, "tools"))
    import dr_certificate as Dc
    from pi_prove import cfg_of
    from dynamics.closed_loop import ClosedLoop
    K = np.array(policy_gain(checkpoints("CORR_s1")[-1])[0])
    worst = 0.0
    for d, delay, tau, order in ((np.zeros(6), 0, 0.1, 1), (np.array([0.3, -0.2, 0.1, 0.2, 0, 0]), 0, 0.1, 1),
                                 (np.array([-0.3, 0.3, -0.3, -0.1, 0, 0]), 2, 0.13, 1), (np.zeros(6), 1, 0.1, 2)):
        cfg = cfg_of(d)
        cfg.drive.delay_steps, cfg.drive.tau = delay, tau
        if order == 2:
            cfg.drive.order, cfg.drive.omega_n, cfg.drive.zeta = 2, 28.0, 0.5
        eps, z0 = 1e-6, np.zeros(NZ)
        lp = ClosedLoop(Dc.Const(0.0), cfg)
        A1 = np.empty((NZ, NZ))
        for i in range(NZ):
            dz = np.zeros(NZ)
            dz[i] = eps
            A1[:, i] = (lp.step(z0 + dz) - lp.step(z0 - dz)) / (2 * eps)
        B1 = (ClosedLoop(Dc.Const(eps), cfg, model=lp.model).step(z0)
              - ClosedLoop(Dc.Const(-eps), cfg, model=lp.model).step(z0)) / (2 * eps)
        e = np.exp(d)
        s7 = (e[0], e[1], e[2], e[0], e[1], e[2], e[3])
        A2, B2 = open_loop(s7, kv=400 * e[4], tau=tau, delay=delay, order=order, omega_n=28.0, zeta=0.5)
        err = max(np.abs(A1 - A2).max(), np.abs(B1 - B2).max())
        print("  plant %s delay %d tau %.3f: max|dA,dB| %.2e  rho fd %.6f engine %.6f"
              % (np.round(d, 2), delay, tau, err, rho(A1, B1, K), rho(A2, B2, K)))
        worst = max(worst, abs(rho(A1, B1, K) - rho(A2, B2, K)))
    return worst


if __name__ == "__main__":
    print("worst rho mismatch", validate())
