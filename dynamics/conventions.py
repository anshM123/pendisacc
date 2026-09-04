"""
The one place that knows how Isaac's joint coordinates relate to the analytical
model's.

    Isaac / URDF : joint angles are RELATIVE to the parent link
    analytical   : angles are ABSOLUTE, measured from the upward vertical

    th_abs[i] = sum_{k<=i} th_rel[k]
    th_rel[i] = th_abs[i] - th_abs[i-1]

Absolute angles are kept for the analytical side on purpose: the upright
equilibrium is then simply q = 0, which keeps the linearisation, the Poincare
sections and the continuation code clean.

Mixing the two conventions is silent and catastrophic. Both trajectories agree
near q = 0 (relative == absolute to first order), so a comparison looks roughly
right at small angles and only degrades as the pendulum moves -- exactly the
regime the whole project is about. It cost a real debugging cycle here: a 105%
trajectory mismatch that looked like a physics error and was purely coordinates.
Route every crossing between the two worlds through these functions.

Layout of q in both conventions: [cart, joint1, joint2, joint3]; element 0 is the
cart's prismatic coordinate and is passed through untouched.
"""

from __future__ import annotations

import numpy as np


def rel_to_abs(q):
    """Isaac/URDF relative joint angles -> absolute angles from vertical."""
    q = np.asarray(q, dtype=float)
    out = q.copy()
    out[..., 1:] = np.cumsum(q[..., 1:], axis=-1)
    return out


def abs_to_rel(q):
    """Absolute angles from vertical -> Isaac/URDF relative joint angles."""
    q = np.asarray(q, dtype=float)
    out = q.copy()
    out[..., 1:] = np.diff(q[..., 1:], prepend=0.0, axis=-1)
    return out


# Angular VELOCITIES transform with the same linear map as the angles.
rel_to_abs_vel = rel_to_abs
abs_to_rel_vel = abs_to_rel


HANGING_REL = np.array([0.0, np.pi, 0.0, 0.0])
"""All three links hanging straight down, in Isaac's relative coordinates.

Note this is NOT [0, pi, pi, pi]. That would put link2 at an absolute angle of
2*pi -- pointing straight UP -- which is a folded configuration, not a hanging
one. Getting this wrong drives the links through the rail and injects energy.
"""

UPRIGHT_REL = np.zeros(4)
"""All three links upright. Identical in both conventions."""


def _self_test():
    rng = np.random.default_rng(0)
    for _ in range(200):
        q = rng.normal(size=4)
        assert np.allclose(abs_to_rel(rel_to_abs(q)), q, atol=1e-12)
        assert np.allclose(rel_to_abs(abs_to_rel(q)), q, atol=1e-12)
    batch = rng.normal(size=(7, 4))
    assert np.allclose(abs_to_rel(rel_to_abs(batch)), batch, atol=1e-12)
    assert np.allclose(rel_to_abs(HANGING_REL)[1:], [np.pi, np.pi, np.pi])
    assert np.allclose(rel_to_abs(UPRIGHT_REL)[1:], [0.0, 0.0, 0.0])
    print("conventions self-test OK")


if __name__ == "__main__":
    _self_test()
