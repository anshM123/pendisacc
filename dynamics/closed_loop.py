"""The robot-policy closed loop, and its variational dynamics.

This is the object the project is actually about. A simulator and reality can
produce nearly the same trajectory under the same policy and still transfer
differently, because what reaches the task is not the model error itself but
the model error *after the closed loop has amplified it*.

Write the closed loop as a discrete map at the control rate,

    z_{k+1} = F(z_k ; xi)

with z the FULL closed-loop state. Not just the plant: the servo lag, its rate
(for a second-order drive), the policy's own action memory and the transport
delay line are all states, and leaving any of them out silently changes what is
being linearised.

    z = [ s th1 th2 th3 | sdot w1 w2 w3 | v_ref v_refdot | a_prev | queue(4) ]
        |--- plant, ABSOLUTE angles ---| |--- actuator ---| |policy| |delay|

Then for a simulator S and a reality R driven by the same policy from the same
initial condition, the first-order error obeys

    e_{k+1} = A_k e_k + d_k ,    A_k = dF_S/dz|_{z_k} ,
                                 d_k = F_R(z_k) - F_S(z_k)

so that

    e_N = Phi(N,0) e_0 + sum_k Phi(N,k+1) d_k ,   Phi(N,j) = A_{N-1} ... A_j.

Two quantities follow directly:

  G_T   = sigma_max(Phi(N,0))
          finite-time amplification. How much an initial discrepancy is
          magnified over the horizon, worst case over directions.

  D_SW  = sum_k || Phi(N,k+1) d_k ||
          the STABILITY-WEIGHTED model discrepancy: per-step model error
          weighted by how much the closed loop will amplify it before the
          horizon ends. This is what ordinary trajectory error misses -- two
          models with the same ||d|| can have wildly different D_SW.

  e_pred = || sum_k Phi(N,k+1) d_k ||
          the same sum without per-term norms, i.e. the predicted end-state
          discrepancy including cancellation. D_SW is its triangle-inequality
          upper bound; reporting both shows whether the bound is loose.

Jacobians are central differences on the closed-loop map, which differentiates
THROUGH the policy network, the deadband, the clip, the actuator and the
integrator together. Nothing about the real control path is linearised away by
hand.

Model-form variation is first-class here, not just parameter scaling: friction
can be absent, viscous, viscous+Coulomb or Stribeck; the drive can be first- or
second-order, with deadband and transport delay. A simulator population built
only from parameter randomisation would let a reviewer object that reality was
drawn from the training distribution.
"""

from __future__ import annotations

from dataclasses import dataclass, field, replace

import numpy as np

from .policy import Actor, observation

DMAX = 4                      # delay line length, i.e. up to 16 ms at 250 Hz
NZ = 15
IQ = slice(0, 4)              # cart, th1, th2, th3   (absolute angles)
IQD = slice(4, 8)             # cart_vel, w1, w2, w3  (absolute rates)
IV = 8                        # actuator state: commanded cart velocity
IVD = 9                       # its rate; inert for a first-order drive
IA = 10                       # previous action, as the policy observes it
IDLY = slice(11, 11 + DMAX)   # transport delay line, newest first


@dataclass
class FrictionCfg:
    """Dissipation. `model` selects the FORM, not merely the coefficients."""

    model: str = "none"          # none | viscous | coulomb | stribeck
    b_cart: float = 0.0          # viscous, N s/m
    fc_cart: float = 0.0         # Coulomb, N
    fs_cart: float = 0.0         # static/breakaway, N  (stribeck only)
    vs_cart: float = 0.02        # Stribeck velocity, m/s
    b_joint: float = 0.0         # N m s/rad, applied per joint
    fc_joint: float = 0.0        # N m
    fs_joint: float = 0.0        # N m  (stribeck only)
    vs_joint: float = 0.05       # rad/s
    tanh_k: float = 100.0        # smoothing of the sign function


@dataclass
class DriveCfg:
    """Everything between the policy output and the cart force."""

    action_scale: float = 4.0    # action 1.0 -> 4.0 m/s commanded
    v_max: float = 4.0           # physical clip on the commanded velocity
    order: int = 1               # 1 = first-order lag, 2 = second-order
    tau: float = 0.100           # first-order time constant [s]
    zeta: float = 0.9            # damping ratio, second-order only
    omega_n: float = 0.0         # rad/s, second order; 0 means use 1/tau
    deadband: float = 0.0        # m/s of commanded velocity ignored
    delay_steps: int = 0         # pure transport delay, in control steps
    kv: float = 400.0            # inner velocity-loop gain [N s/m]
    f_clamp: float = 349.5       # drive current limit, at the cart [N]
    obs_action_clip: float = 1.0


@dataclass
class SimCfg:
    """One simulator. This is `xi`."""

    name: str = "nominal"
    dt_ctrl: float = 1.0 / 250.0
    # Physics steps per control step. NOT 2, which is what Isaac uses.
    #
    # The inner velocity loop has time constant m_cart / k_v = 0.79 / 400 =
    # 2.0 ms, exactly Isaac's physics step. Isaac survives that because
    # ImplicitActuatorCfg solves the damping term implicitly and is stable at
    # any step size; explicit integration here is not. Measured: at substeps =
    # 2 the reproduced loop swings up and then fails to hold at all (0% of the
    # final quarter upright, tip oscillating -0.99 to +0.93), while at
    # substeps >= 5 it holds 100% with tip +1.000. 10 gives h = 0.4 ms.
    substeps: int = 10
    drive: DriveCfg = field(default_factory=DriveCfg)
    friction: FrictionCfg = field(default_factory=FrictionCfg)
    mass_scale: tuple = (1.0, 1.0, 1.0)
    inertia_scale: tuple = (1.0, 1.0, 1.0)
    cart_mass_scale: float = 1.0
    gravity: float = 9.80665


def build_model(cfg: SimCfg):
    """Analytical plant for one simulator. Friction is handled OUTSIDE the
    model, so that its form can be varied; the model's own coefficients are
    therefore left at zero to avoid counting it twice."""
    from .analytical.triple_pendulum import PendulumParams, TriplePendulumModel

    p = PendulumParams.from_yaml()
    links = [replace(l, m=l.m * ms, I=l.I * is_)
             for l, ms, is_ in zip(p.links, cfg.mass_scale, cfg.inertia_scale)]
    return TriplePendulumModel(replace(
        p, links=links, m_cart=p.m_cart * cfg.cart_mass_scale, g=cfg.gravity,
        b_cart=0.0, fc_cart=0.0, b_joint=[0.0] * 3, fc_joint=[0.0] * 3))


def friction_forces(fc: FrictionCfg, q, qd) -> np.ndarray:
    """Generalised friction, shape (4,), opposing motion.

    Joint friction acts on the RELATIVE rate across each joint and appears with
    opposite sign on the two bodies it couples -- it is not diagonal damping on
    absolute angles. Same structure as the analytical model's own friction, so
    the two agree when the form is 'coulomb'.
    """
    f = np.zeros(4)
    if fc.model == "none":
        return f
    k = fc.tanh_k
    v = qd[0]

    def slide(vel, b, coul, stat, vstr):
        out = b * vel
        if fc.model in ("coulomb", "stribeck"):
            mag = coul
            if fc.model == "stribeck" and stat > coul:
                # breakaway force decaying to Coulomb as speed rises
                mag = coul + (stat - coul) * np.exp(-(vel / max(vstr, 1e-9)) ** 2)
            out = out + mag * np.tanh(k * vel)
        return out

    f[0] = slide(v, fc.b_cart, fc.fc_cart, fc.fs_cart, fc.vs_cart)
    rel = [qd[1], qd[2] - qd[1], qd[3] - qd[2]]
    for i, w in enumerate(rel):
        t = slide(w, fc.b_joint, fc.fc_joint, fc.fs_joint, fc.vs_joint)
        f[1 + i] += t
        if i > 0:
            f[i] -= t
    return f


class ClosedLoop:
    """The map z -> F(z) for one simulator under one policy."""

    def __init__(self, actor: Actor, cfg: SimCfg, model=None):
        self.actor = actor
        self.cfg = cfg
        self.model = build_model(cfg) if model is None else model
        d, dt = cfg.drive, cfg.dt_ctrl
        self.alpha = 1.0 if d.tau <= 0 else dt / (dt + d.tau)
        self.wn = float(d.omega_n) if d.omega_n else 1.0 / max(d.tau, 1e-9)

    def step(self, z: np.ndarray) -> np.ndarray:
        cfg, d, fr = self.cfg, self.cfg.drive, self.cfg.friction
        dt = cfg.dt_ctrl
        q, qd = z[IQ].copy(), z[IQD].copy()

        a = float(self.actor(observation(q, qd, z[IA]))[0, 0])
        v_cmd = a * d.action_scale

        if d.deadband > 0.0:
            # commands below the deadband produce no motion at all
            v_cmd = np.sign(v_cmd) * max(abs(v_cmd) - d.deadband, 0.0)
        v_cmd = np.clip(v_cmd, -d.v_max, d.v_max)

        queue = z[IDLY].copy()
        v_eff = v_cmd if d.delay_steps <= 0 else queue[min(d.delay_steps, DMAX) - 1]
        queue = np.concatenate([[v_cmd], queue[:-1]])

        if d.order == 2:
            vdd = self.wn ** 2 * (v_eff - z[IV]) - 2.0 * d.zeta * self.wn * z[IVD]
            v_refdot = z[IVD] + dt * vdd
            v_ref = z[IV] + dt * v_refdot
        else:
            v_ref = z[IV] + self.alpha * (v_eff - z[IV])
            v_refdot = 0.0

        h = dt / cfg.substeps
        M, C, G, B = self.model.M, self.model.C, self.model.G, self.model.B()
        for _ in range(cfg.substeps):
            F = np.clip(d.kv * (v_ref - qd[0]), -d.f_clamp, d.f_clamp)
            rhs = B * F - C(q, qd) @ qd - G(q) - friction_forces(fr, q, qd)
            qd = qd + h * np.linalg.solve(M(q), rhs)
            q = q + h * qd

        out = np.zeros(NZ)
        out[IQ], out[IQD] = q, qd
        out[IV], out[IVD] = v_ref, v_refdot
        out[IA] = np.clip(a, -d.obs_action_clip, d.obs_action_clip)
        out[IDLY] = queue
        return out

    def rollout(self, z0: np.ndarray, n: int) -> np.ndarray:
        Z = np.empty((n + 1, NZ))
        Z[0] = z0
        for k in range(n):
            Z[k + 1] = self.step(Z[k])
        return Z

    def jacobian(self, z: np.ndarray, eps: float = 1e-6) -> np.ndarray:
        A = np.empty((NZ, NZ))
        for i in range(NZ):
            dz = np.zeros(NZ)
            dz[i] = eps
            A[:, i] = (self.step(z + dz) - self.step(z - dz)) / (2 * eps)
        return A


def transition_matrices(loop: ClosedLoop, Z: np.ndarray) -> np.ndarray:
    return np.stack([loop.jacobian(Z[k]) for k in range(len(Z) - 1)])


def stability_weighted_gap(loop_s: ClosedLoop, loop_r: ClosedLoop,
                           z0: np.ndarray, n: int, A=None, Zs=None) -> dict:
    """Model discrepancy weighted by closed-loop amplification.

    Everything is evaluated ALONG THE SIMULATOR'S trajectory, which is the only
    one available in practice: you do not have reality's trajectory when you are
    choosing which simulator to trust. `A` and `Zs` may be passed in when many
    realities are compared against one simulator, since they depend only on S.
    """
    if Zs is None:
        Zs = loop_s.rollout(z0, n)
    if A is None:
        A = transition_matrices(loop_s, Zs)

    D = np.stack([loop_r.step(Zs[k]) - Zs[k + 1] for k in range(n)])

    P = np.eye(NZ)
    terms = np.empty((n, NZ))
    for k in range(n - 1, -1, -1):
        terms[k] = P @ D[k]
        P = P @ A[k]

    contrib = np.linalg.norm(terms, axis=1)
    return {
        "G_T": float(np.linalg.svd(P, compute_uv=False)[0]),
        "D_SW": float(contrib.sum()),
        "e_pred": float(np.linalg.norm(terms.sum(axis=0))),
        "raw_gap": float(np.linalg.norm(D, axis=1).sum()),
        "per_step_contrib": contrib,
        "Zs": Zs,
        "D": D,
    }


# --------------------------------------------------------- task-margin risk
RAIL_LIMIT = 0.60          # |cart| beyond this terminates the episode
LINK_LENGTHS = np.array([0.24999, 0.25000, 0.31575])


def task_margins(z: np.ndarray) -> tuple:
    """Task/safety margins g_j(x) > 0, and their gradients w.r.t. the state.

    A generic Euclidean state norm throws away the thing that actually decides
    the episode. Every measured failure of this policy is a rail excursion --
    under randomised conditions it reaches upright in 100.00% of trials and
    every loss is |cart| > 0.60 m during the transient -- so the margin that
    matters is distance to that boundary, and the component of a predicted
    deviation that matters is the part pointing across it.

    Returns (g, grad) with g shape (m,) and grad shape (m, NZ).
    """
    x = z[0]
    g = np.array([RAIL_LIMIT - abs(x)])
    grad = np.zeros((1, NZ))
    grad[0, 0] = -np.sign(x) if x != 0.0 else 0.0
    return g, grad


def transfer_critical_risk(loop_s: "ClosedLoop", loop_r: "ClosedLoop",
                           z0: np.ndarray, n: int, eps: float = 1e-3,
                           A=None, Zs=None) -> dict:
    """R_TC: what fraction of the task margin does the model error consume?

    The deviation obeys the same first-order recursion as everything else,

        e_{t+1} = A_t e_t + d_t ,   e_0 = 0,

    so no O(N^2) accumulation of transition matrices is needed -- this is
    CHEAPER than D_SW, not more expensive. Projecting onto the margin gives

        dg_j(t) = grad g_j(x_t)^T e_t

    and the risk is the worst fraction of remaining margin that a predicted
    deviation eats, over margins and over time:

        R_TC = max_{j,t}  [ -grad g_j(x_t)^T e_t ]_+ / ( g_j(x_t) + eps )

    R_TC < 1 predicts margin left; R_TC > 1 predicts a boundary crossing under
    the local approximation. The point of the projection is orientation: a large
    deviation parallel to the boundary is harmless, a small one across it is not,
    and a norm cannot tell the difference.
    """
    if Zs is None:
        Zs = loop_s.rollout(z0, n)
    if A is None:
        A = transition_matrices(loop_s, Zs)
    e = np.zeros(NZ)
    worst, worst_t, erosion = 0.0, -1, np.zeros(n + 1)
    margin_min = np.inf
    for k in range(n):
        g, grad = task_margins(Zs[k])
        proj = -(grad @ e)
        r = np.max(np.maximum(proj, 0.0) / (np.maximum(g, 0.0) + eps))
        erosion[k] = r
        margin_min = min(margin_min, float(g.min()))
        if r > worst:
            worst, worst_t = r, k
        e = A[k] @ e + (loop_r.step(Zs[k]) - Zs[k + 1])
    g, grad = task_margins(Zs[n])
    erosion[n] = np.max(np.maximum(-(grad @ e), 0.0) / (np.maximum(g, 0.0) + eps))
    if erosion[n] > worst:
        worst, worst_t = float(erosion[n]), n
    return {"R_TC": float(worst), "t_worst": worst_t,
            "erosion": erosion, "min_margin": float(margin_min),
            "e_final_norm": float(np.linalg.norm(e))}
