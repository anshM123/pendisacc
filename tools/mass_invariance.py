"""Why 2.5x on every link is harmless and 1.5x on one link is fatal.

The measured result that motivates this: scaling all three link masses by 2.5
leaves success at 100.0%, while scaling link 1 alone by 1.5 -- a model error
five times smaller in magnitude -- gives 0.0%. Parameter distance gets that
backwards in sign, so it is worth asking whether mechanics predicts it.

CLAIM. The policy commands cart VELOCITY through a stiff inner loop, so while
the drive's force clamp is not binding the cart is effectively kinematically
prescribed. The pendulum subsystem is then

    M_p(theta) thetaddot + C_p(theta,thetadot) thetadot + G_p(theta) = -h(theta) xddot

and every one of M_p, C_p, G_p and h is LINEAR in the link masses and inertias.
Scaling all of them by a common factor c multiplies both sides by c, which
cancels. Uniform mass scaling is therefore an exact equivalence direction of
the pendulum dynamics under prescribed cart motion -- not an approximation, and
not a property of the policy.

Asymmetric scaling changes the ratios inside M_p and G_p, so nothing cancels
and the trajectory changes.

The claim has a stated breakdown: the cart force needed to prescribe the same
motion scales with c, so the invariance must fail once c times the nominal
force demand reaches the drive clamp. That is a quantitative prediction, and it
is checked here too.

  run.cmd tools/mass_invariance.py
"""

from __future__ import annotations

import json
import os
import sys
from dataclasses import replace

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from dynamics.analytical.triple_pendulum import PendulumParams, TriplePendulumModel  # noqa: E402

OUT = os.path.join(ROOT, "results", "mass_invariance.json")


def scaled(p: PendulumParams, c, cart_c=1.0) -> PendulumParams:
    c = np.atleast_1d(c)
    if c.size == 1:
        c = np.repeat(c, 3)
    links = [replace(l, m=l.m * ci, I=l.I * ci) for l, ci in zip(p.links, c)]
    return replace(p, links=links, m_cart=p.m_cart * cart_c)


def pendulum_accel(model: TriplePendulumModel, q, qd, xddot):
    """theta-ddot with the cart acceleration PRESCRIBED, not the force.

    Solve the 4-DOF system for the link rows only, treating xddot as given:
        M[1:,1:] thddot = -M[1:,0] xddot - (C qd)[1:] - G[1:]
    """
    M = model.M(q)
    rhs = -M[1:, 0] * xddot - (model.C(q, qd) @ np.asarray(qd))[1:] - model.G(q)[1:]
    return np.linalg.solve(M[1:, 1:], rhs)


def cart_force(model: TriplePendulumModel, q, qd, xddot, thddot):
    """Force the drive must supply to hold that prescribed cart acceleration."""
    M = model.M(q)
    acc = np.concatenate([[xddot], thddot])
    return float((M @ acc + model.C(q, qd) @ np.asarray(qd) + model.G(q))[0])


def main() -> int:
    p = PendulumParams.from_yaml()
    base = TriplePendulumModel(p)

    rng = np.random.default_rng(0)
    states = [(np.concatenate([[rng.uniform(-0.3, 0.3)], rng.uniform(-np.pi, np.pi, 3)]),
               np.concatenate([[rng.uniform(-2, 2)], rng.uniform(-8, 8, 3)]),
               rng.uniform(-20, 20)) for _ in range(200)]

    res = {"uniform": [], "asymmetric": [], "force": []}

    print("CLAIM 1: uniform link scaling leaves theta-ddot EXACTLY unchanged")
    print("         (cart acceleration prescribed)\n")
    print("   scale c    max |thddot(c) - thddot(1)|   relative")
    for c in (1.3, 2.5, 5.0, 10.0, 100.0):
        m = TriplePendulumModel(scaled(p, c))
        err = mag = 0.0
        for q, qd, xdd in states:
            a0 = pendulum_accel(base, q, qd, xdd)
            a1 = pendulum_accel(m, q, qd, xdd)
            err = max(err, float(np.max(np.abs(a1 - a0))))
            mag = max(mag, float(np.max(np.abs(a0))))
        res["uniform"].append({"c": c, "abs": err, "rel": err / mag})
        print("   %7.1f    %.3e                  %.3e" % (c, err, err / mag))

    print("\nCLAIM 2: asymmetric scaling does NOT cancel")
    print("\n   scaling            max |thddot - thddot(1)|   relative")
    for name, c in (("link1 x1.5", [1.5, 1.0, 1.0]),
                    ("link2 x1.5", [1.0, 1.5, 1.0]),
                    ("link3 x1.5", [1.0, 1.0, 1.5]),
                    ("all x1.5", [1.5, 1.5, 1.5])):
        m = TriplePendulumModel(scaled(p, c))
        err = mag = 0.0
        for q, qd, xdd in states:
            a0 = pendulum_accel(base, q, qd, xdd)
            a1 = pendulum_accel(m, q, qd, xdd)
            err = max(err, float(np.max(np.abs(a1 - a0))))
            mag = max(mag, float(np.max(np.abs(a0))))
        res["asymmetric"].append({"scaling": name, "abs": err, "rel": err / mag})
        print("   %-18s %.4e                %.4e" % (name, err, err / mag))

    print("\nCLAIM 3: the invariance must break when the force clamp binds")
    print("         (force scales with c; clamp is 349.5 N)\n")
    print("   scale c    max |cart force| [N]    clamp binds?")
    for c in (1.0, 2.5, 10.0, 30.0, 40.0, 60.0):
        m = TriplePendulumModel(scaled(p, c))
        f = 0.0
        for q, qd, xdd in states:
            thdd = pendulum_accel(m, q, qd, xdd)
            f = max(f, abs(cart_force(m, q, qd, xdd, thdd)))
        res["force"].append({"c": c, "max_force_N": f, "binds": bool(f > 349.5)})
        print("   %7.1f    %10.1f              %s" % (c, f, "YES" if f > 349.5 else "no"))

    print("\nThe cart mass does NOT scale, so the FULL 4-DOF system is not")
    print("invariant -- only the pendulum subsystem under prescribed cart motion.")
    print("That is exactly the regime a stiff velocity loop creates, and it is")
    print("why the invariance is a property of the ARCHITECTURE (velocity")
    print("command) as much as of the mechanics.")

    json.dump(res, open(OUT, "w", encoding="utf-8"), indent=1)
    print("\n[out] %s" % OUT)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
