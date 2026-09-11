"""Discover the reality-gap geometry instead of being told it.

Everything in this project so far has tested a null direction we already knew
analytically: uniform inertial scaling. That is a demonstration, not a method.
A method has to take a parameter vector and find the structure itself.

For a fixed policy pi and task T, let b(xi) be a behavioural feature of the
closed loop -- here the link-angle trajectory over a short window, which is the
quantity short-horizon trajectory RMSE already showed to be the best aggregate
predictor of transfer. Its sensitivity to the simulator parameters is

    J(xi) = d b / d xi

and the local reality-gap metric tensor is

    G = E_{x0} [ J^T W J ]

over initial conditions x0. Eigenvectors of G with small eigenvalues are
directions in physical parameter space that the closed loop cannot see;
eigenvectors with large eigenvalues are directions that destroy it.

G is LOCAL and the measured success boundaries are nonlinear and sometimes
discontinuous, so G is not a theory of transfer. It is a cheap linear picture
of the geometry near nominal, to be used for allocating budget and then
checked against the nonlinear boundary further out.

PARAMETERISATION. Directions are only comparable if the coordinates are. Each
parameter is perturbed in units of its own plausible scale: multiplicatively
for quantities with a non-zero nominal (masses, inertias, tau, kv, gravity),
additively against a stated plausible magnitude for those whose nominal is
zero (joint damping, joint friction). The resulting xi is dimensionless, and a
direction in it is a recipe like "increase all three link masses by 1% each".

THE FALSIFIABLE CHECK. The uniform inertial direction is an exact symmetry of
the passive dynamics under prescribed base motion (dynamics/symmetry.py). If
this estimator works, it must recover that direction as a small-eigenvalue
eigenvector WITHOUT being told it exists. tools/geometry_discover.py runs that
check.
"""

from __future__ import annotations

import os
import sys
from dataclasses import replace

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from dynamics.closed_loop import IQ, IQD, NZ, ClosedLoop, SimCfg  # noqa: E402


# name, kind, plausible scale. "rel" perturbs multiplicatively, "abs" adds.
PARAMS = (
    ("m1",        "rel", 1.0),
    ("m2",        "rel", 1.0),
    ("m3",        "rel", 1.0),
    ("I1",        "rel", 1.0),
    ("I2",        "rel", 1.0),
    ("I3",        "rel", 1.0),
    ("cart_mass", "rel", 1.0),
    ("gravity",   "rel", 1.0),
    ("tau",       "rel", 1.0),
    ("kv",        "rel", 1.0),
    ("b_joint",   "abs", 0.01),     # N m s/rad, a plausible bearing drag
    ("fc_joint",  "abs", 0.01),     # N m, a plausible Coulomb torque
)
NAMES = [p[0] for p in PARAMS]
NP = len(PARAMS)


def apply(base: SimCfg, theta: np.ndarray) -> SimCfg:
    """theta is a dimensionless perturbation in the coordinates of PARAMS."""
    ms = list(base.mass_scale)
    isc = list(base.inertia_scale)
    cart = base.cart_mass_scale
    grav = base.gravity
    drive = base.drive
    tau, kv = drive.tau, drive.kv
    fr = base.friction
    bj, fcj = fr.b_joint, fr.fc_joint

    for k, (name, kind, scale) in enumerate(PARAMS):
        t = float(theta[k])
        if t == 0.0:
            continue
        if kind == "rel":
            f = 1.0 + t
            if name in ("m1", "m2", "m3"):
                ms[int(name[1]) - 1] *= f
            elif name in ("I1", "I2", "I3"):
                isc[int(name[1]) - 1] *= f
            elif name == "cart_mass":
                cart *= f
            elif name == "gravity":
                grav *= f
            elif name == "tau":
                tau *= f
            elif name == "kv":
                kv *= f
        else:
            if name == "b_joint":
                bj = bj + t * scale
            elif name == "fc_joint":
                fcj = fcj + t * scale

    model = "none"
    if fcj != 0.0:
        model = "coulomb"
    elif bj != 0.0:
        model = "viscous"
    return replace(
        base,
        mass_scale=tuple(ms), inertia_scale=tuple(isc),
        cart_mass_scale=cart, gravity=grav,
        drive=replace(drive, tau=tau, kv=kv),
        friction=replace(fr, model=model, b_joint=bj, fc_joint=fcj),
    )


def feature(actor, cfg: SimCfg, z0: np.ndarray, n_steps: int) -> np.ndarray:
    """Behavioural feature: the link-angle trajectory of the closed loop."""
    loop = ClosedLoop(actor, cfg)
    Z = loop.rollout(z0, n_steps)
    return Z[:, IQ][:, 1:].reshape(-1)       # th1..th3 over time, flattened


def jacobian(actor, base: SimCfg, z0: np.ndarray, n_steps: int,
             h: float = 0.02) -> np.ndarray:
    """Central-difference d(feature)/d(theta) at theta = 0."""
    cols = []
    for k in range(NP):
        e = np.zeros(NP)
        e[k] = h
        fp = feature(actor, apply(base, +e), z0, n_steps)
        fm = feature(actor, apply(base, -e), z0, n_steps)
        cols.append((fp - fm) / (2.0 * h))
    return np.column_stack(cols)


def metric(actor, base: SimCfg, z0s, n_steps: int, h: float = 0.02):
    """G = E_{x0}[ J^T J ], plus the per-initial-condition Jacobians."""
    G = np.zeros((NP, NP))
    Js = []
    for z0 in z0s:
        J = jacobian(actor, base, z0, n_steps, h)
        Js.append(J)
        G += J.T @ J
    G /= max(len(z0s), 1)
    return G, Js


def spectrum(G: np.ndarray):
    """Eigenpairs of G, ascending: the first is the most null direction."""
    w, V = np.linalg.eigh(G)
    order = np.argsort(w)
    return w[order], V[:, order]


def uniform_inertial_direction() -> np.ndarray:
    """The analytically exact null direction, for scoring the estimator only.

    Scaling every link mass AND inertia by the same relative amount. It is
    never given to the estimator -- it exists so the discovered subspace can
    be checked against a known answer.
    """
    u = np.zeros(NP)
    for i, n in enumerate(NAMES):
        if n in ("m1", "m2", "m3", "I1", "I2", "I3"):
            u[i] = 1.0
    return u / np.linalg.norm(u)


def subspace_alignment(V: np.ndarray, k: int, u: np.ndarray) -> float:
    """How much of u lies in the span of the k most-null eigenvectors."""
    P = V[:, :k]
    return float(np.linalg.norm(P.T @ u))
