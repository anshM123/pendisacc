"""The robot-policy closed loop, and its variational dynamics.

This is the object the project is actually about. A simulator and reality can
produce nearly the same trajectory under the same policy and still transfer
differently, because what reaches the task is not the model error itself but
the model error *after the closed loop has amplified it*.

Write the closed loop as a discrete map at the control rate,

    z_{k+1} = F(z_k ; xi)

with z the FULL closed-loop state -- not just the plant, because the servo lag
and the policy's own action memory are states too:

    z = [ s, th1, th2, th3, sdot, w1, w2, w3, v_ref, a_prev ]   (10)
        |------- plant (absolute angles) -------| |lag| |policy|

Then for a simulator S and a reality R driven by the same policy from the same
initial condition, the first-order error obeys

    e_{k+1} = A_k e_k + d_k ,    A_k = dF_S/dz|_{z_k} ,
                                 d_k = F_R(z_k) - F_S(z_k)

so that

    e_N = Phi(N,0) e_0 + sum_k Phi(N,k+1) d_k ,   Phi(N,j) = A_{N-1} ... A_j.

Two quantities follow directly and are what the analysis uses:

  G_T   = sigma_max(Phi(N,0))
          finite-time amplification of the closed loop. How much an initial
          discrepancy is magnified over the horizon, worst case over directions.

  D_SW  = sum_k || Phi(N,k+1) d_k ||
          the STABILITY-WEIGHTED model discrepancy: per-step model error
          weighted by how much the closed loop will amplify it before the
          horizon ends. This is the quantity that ordinary trajectory error
          fails to capture -- two models with the same ||d|| can have wildly
          different D_SW.

  e_pred = || sum_k Phi(N,k+1) d_k ||
          the same sum WITHOUT taking norms term by term, i.e. the actual
          predicted end-state discrepancy including cancellation. D_SW is its
          triangle-inequality upper bound. Reporting both is honest: if they
          differ by orders of magnitude, cancellation matters and the bound is
          loose.

Jacobians are taken by central differences on the closed-loop map. That is
deliberate: it differentiates THROUGH the policy network, the action clip, the
servo filter and the integrator together, so nothing about the real control
path is quietly linearised away.

Self-check: `validate()` compares the first-order prediction e_N against a
directly simulated R trajectory. If the variational machinery is wrong, that is
where it shows.
"""

from __future__ import annotations

from dataclasses import dataclass, field, replace

import numpy as np

from .policy import Actor, observation

NZ = 10                       # closed-loop state dimension
IQ = slice(0, 4)              # q  = cart, th1, th2, th3
IQD = slice(4, 8)             # qd = cart_vel, w1, w2, w3
IV = 8                        # servo filter state (commanded cart velocity)
IA = 9                        # previous action, as the policy observes it


@dataclass
class DriveCfg:
    """Everything between the policy output and the cart force."""

    action_scale: float = 4.0        # action 1.0 -> 4.0 m/s commanded
    v_max: float = 4.0               # physical clip on the commanded velocity
    tau: float = 0.100               # first-order servo lag [s]
    delay_steps: int = 0             # pure transport delay, in control steps
    kv: float = 400.0                # inner velocity-loop gain [N s/m]
    f_clamp: float = 349.5           # drive current limit, at the cart [N]
    obs_action_clip: float = 1.0     # what last_action_clipped reports


@dataclass
class SimCfg:
    """One simulator: a plant, a drive, and a control rate. This is `xi`."""

    name: str = "nominal"
    dt_ctrl: float = 1.0 / 250.0
    # Physics steps per control step. NOT 2, which is what Isaac uses.
    #
    # The inner velocity loop has time constant m_cart / k_v = 0.79 / 400 =
    # 2.0 ms, exactly Isaac's physics step. Isaac survives that because
    # ImplicitActuatorCfg solves the damping term implicitly and is stable at
    # any step size; the explicit integration here is not. Measured: at
    # substeps = 2 the reproduced loop swings up and then fails to hold at all
    # (0% of the final quarter upright, tip oscillating -0.99 to +0.93), while
    # at substeps >= 5 it holds 100% with tip +1.000. 10 gives h = 0.4 ms, a
    # 5x margin on the stiff mode, for 5x the cost of a step that was never the
    # bottleneck.
    substeps: int = 10
    drive: DriveCfg = field(default_factory=DriveCfg)
    # plant overrides applied to the CAD-derived PendulumParams
    mass_scale: tuple = (1.0, 1.0, 1.0)     # per link
    inertia_scale: tuple = (1.0, 1.0, 1.0)  # per link
    cart_mass_scale: float = 1.0
    b_cart: float = 0.0              # viscous cart friction [N s/m]
    fc_cart: float = 0.0             # Coulomb cart friction [N]
    b_joint: tuple = (0.0, 0.0, 0.0)
    fc_joint: tuple = (0.0, 0.0, 0.0)
    gravity: float = 9.80665


def build_model(cfg: SimCfg):
    """Instantiate the analytical plant for one simulator config."""
    from .analytical.triple_pendulum import PendulumParams, TriplePendulumModel

    p = PendulumParams.from_yaml()
    links = [replace(l, m=l.m * ms, I=l.I * is_)
             for l, ms, is_ in zip(p.links, cfg.mass_scale, cfg.inertia_scale)]
    p = replace(p, links=links, m_cart=p.m_cart * cfg.cart_mass_scale, g=cfg.gravity,
                b_cart=cfg.b_cart, fc_cart=cfg.fc_cart,
                b_joint=list(cfg.b_joint), fc_joint=list(cfg.fc_joint))
    return TriplePendulumModel(p)


class ClosedLoop:
    """The map z -> F(z) for one simulator under one policy."""

    def __init__(self, actor: Actor, cfg: SimCfg, model=None):
        self.actor = actor
        self.cfg = cfg
        self.model = build_model(cfg) if model is None else model
        d = cfg.drive
        self.alpha = 1.0 if d.tau <= 0 else cfg.dt_ctrl / (cfg.dt_ctrl + d.tau)

    # ---------------------------------------------------------------- one step
    def step(self, z: np.ndarray) -> np.ndarray:
        """Advance one CONTROL step: policy, drive, servo, then the plant."""
        cfg, d = self.cfg, self.cfg.drive
        q, qd = z[IQ].copy(), z[IQD].copy()

        obs = observation(q, qd, z[IA])
        a = float(self.actor(obs)[0, 0])

        v_cmd = np.clip(a * d.action_scale, -d.v_max, d.v_max)
        v_ref = z[IV] + self.alpha * (v_cmd - z[IV])

        h = cfg.dt_ctrl / cfg.substeps
        for _ in range(cfg.substeps):
            F = np.clip(d.kv * (v_ref - qd[0]), -d.f_clamp, d.f_clamp)
            # semi-implicit Euler at a substep well inside the velocity loop's
            # 2 ms time constant; see the note on SimCfg.substeps
            acc = self.model.accel(q, qd, F)
            qd = qd + h * acc
            q = q + h * qd

        out = np.empty(NZ)
        out[IQ], out[IQD] = q, qd
        out[IV] = v_ref
        out[IA] = np.clip(a, -d.obs_action_clip, d.obs_action_clip)
        return out

    def rollout(self, z0: np.ndarray, n: int) -> np.ndarray:
        """(n+1, NZ) trajectory."""
        Z = np.empty((n + 1, NZ))
        Z[0] = z0
        for k in range(n):
            Z[k + 1] = self.step(Z[k])
        return Z

    # ------------------------------------------------------------- variational
    def jacobian(self, z: np.ndarray, eps: float = 1e-6) -> np.ndarray:
        """dF/dz by central differences, through policy + clip + servo + plant."""
        A = np.empty((NZ, NZ))
        for i in range(NZ):
            dz = np.zeros(NZ)
            dz[i] = eps
            A[:, i] = (self.step(z + dz) - self.step(z - dz)) / (2 * eps)
        return A


def transition_matrices(loop: ClosedLoop, Z: np.ndarray) -> np.ndarray:
    """A_k = dF/dz at each visited state. Shape (n, NZ, NZ)."""
    return np.stack([loop.jacobian(Z[k]) for k in range(len(Z) - 1)])


def amplification(A: np.ndarray) -> tuple:
    """Phi(N,0) accumulated backwards, plus sigma_max at every horizon.

    Returns (Phi_N0, G_curve) where G_curve[k] = sigma_max(Phi(k,0)), i.e. the
    growth achieved by step k. The curve is what localises WHEN the closed loop
    is fragile -- for swing-up the interesting window is the capture, not the
    steady hold.
    """
    n = len(A)
    Phi = np.eye(NZ)
    G = np.empty(n + 1)
    G[0] = 1.0
    for k in range(n):
        Phi = A[k] @ Phi
        G[k + 1] = np.linalg.svd(Phi, compute_uv=False)[0]
    return Phi, G


def stability_weighted_gap(loop_s: ClosedLoop, loop_r: ClosedLoop,
                           z0: np.ndarray, n: int) -> dict:
    """The central measurement: model discrepancy weighted by amplification.

    Everything is evaluated ALONG THE SIMULATOR'S trajectory, which is the only
    one available in practice -- you do not have reality's trajectory when you
    are choosing a simulator.
    """
    Zs = loop_s.rollout(z0, n)
    A = transition_matrices(loop_s, Zs)

    # per-step model discrepancy d_k = F_R(z_k) - F_S(z_k)
    D = np.stack([loop_r.step(Zs[k]) - Zs[k + 1] for k in range(n)])

    # accumulate Phi(N, k+1) d_k without ever forming all the Phi's:
    # walk backwards, carrying P = Phi(N, k+1)
    P = np.eye(NZ)
    terms = np.empty((n, NZ))
    for k in range(n - 1, -1, -1):
        terms[k] = P @ D[k]
        P = P @ A[k]
    Phi_N0 = P

    contrib = np.linalg.norm(terms, axis=1)
    return {
        "Phi_N0": Phi_N0,
        "G_T": float(np.linalg.svd(Phi_N0, compute_uv=False)[0]),
        "D_SW": float(contrib.sum()),          # triangle-inequality bound
        "e_pred": float(np.linalg.norm(terms.sum(axis=0))),   # with cancellation
        "raw_gap": float(np.linalg.norm(D, axis=1).sum()),    # unweighted model error
        "per_step_contrib": contrib,
        "Zs": Zs,
        "D": D,
    }
