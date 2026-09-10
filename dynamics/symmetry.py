"""The uniform-inertia symmetry, proved symbolically and for general N.

The measured result is that scaling all three link masses by 2.5 leaves transfer
at 100% while scaling one link by 1.5 destroys it. The explanation should not be
a property of this particular triple pendulum, so it is stated and proved for a
planar N-link chain on a PRESCRIBED base, for N = 1..4, exactly rather than
numerically.

PROPOSITION. Let a mechanical system have coordinates (x, q) where x is
kinematically prescribed -- driven by an actuator that imposes x(t) regardless
of reaction -- and q is passive, carrying no generalised force. Let the kinetic
and potential energies of the passive bodies be

    T = sum_i [ (1/2) m_i |v_i(q, qdot, x, xdot)|^2 + (1/2) I_i omega_i^2 ]
    V = sum_i m_i g h_i(q, x)

Both are LINEAR in the inertial parameters {m_i, I_i}, so scaling every one of
them by a common c > 0 gives L -> cL. The Euler-Lagrange equations for the
passive coordinates are homogeneous,

    d/dt (dL/dqdot) - dL/dq = 0,

so they become c times themselves, and c divides out. Therefore

    q(t; c) = q(t; 1)

for the same prescribed x(t) and the same initial conditions. The result is
EXACT, holds for any N, and does not involve the base mass at all -- the base
never enters the passive equations once x is prescribed.

THE HYPOTHESES ARE THE INTERESTING PART. The symmetry fails if any of them
breaks, and each failure is a real physical mechanism:

  * x is not truly prescribed. A real actuator has finite authority, and the
    reaction from heavier links grows with c. This is what actually breaks the
    equivalence on the hardware model -- verified: uniform scaling at c=20 goes
    from 0% to 100% success when the drive's gain AND force clamp are both
    lifted, and neither alone suffices.
  * a generalised force on q that does not scale with c. Joint friction is the
    obvious case: a fixed Coulomb torque does not scale, so the passive equation
    picks up a term that survives the division. Verified below as a deliberate
    counterexample.
  * non-uniform scaling. Scaling a subset changes the ratios inside the mass
    matrix and nothing cancels. Also verified below.

  run.cmd dynamics/symmetry.py
"""

from __future__ import annotations

import sympy as sp


def chain_lagrangian(n: int, with_joint_friction: bool = False):
    """Planar n-link chain hanging from a base whose position x(t) is prescribed.

    Absolute angles measured from the upward vertical, so theta = 0 is up and
    theta = pi is hanging -- the same convention as the rest of the project.
    Returns (residual, params) with residual = 0 the passive equations of motion.
    """
    t = sp.symbols("t", real=True)
    g = sp.symbols("g", positive=True)
    m = sp.symbols("m1:%d" % (n + 1), positive=True)
    I = sp.symbols("I1:%d" % (n + 1), positive=True)
    L = sp.symbols("L1:%d" % (n + 1), positive=True)
    lc = sp.symbols("lc1:%d" % (n + 1), positive=True)

    x = sp.Function("x")(t)                       # PRESCRIBED base coordinate
    th = [sp.Function("th%d" % (i + 1))(t) for i in range(n)]

    # centre-of-mass positions: base offset plus the chain of link vectors
    T = sp.Integer(0)
    V = sp.Integer(0)
    px, py = x, sp.Integer(0)
    for i in range(n):
        cx = px + lc[i] * sp.sin(th[i])
        cy = py + lc[i] * sp.cos(th[i])
        vx, vy = sp.diff(cx, t), sp.diff(cy, t)
        T += sp.Rational(1, 2) * m[i] * (vx ** 2 + vy ** 2)
        T += sp.Rational(1, 2) * I[i] * sp.diff(th[i], t) ** 2
        V += m[i] * g * cy
        px = px + L[i] * sp.sin(th[i])
        py = py + L[i] * sp.cos(th[i])

    lag = T - V
    res = []
    for i in range(n):
        e = sp.diff(sp.diff(lag, sp.diff(th[i], t)), t) - sp.diff(lag, th[i])
        if with_joint_friction:
            # a Coulomb-like torque that does NOT scale with the inertias
            b = sp.symbols("b", positive=True)
            e = e + b * sp.diff(th[i], t)
        res.append(sp.simplify(e))
    return res, (m, I)


def scales_exactly(res, params, c, subset=None) -> bool:
    """Is every passive equation multiplied by exactly c under the scaling?"""
    m, I = params
    idx = range(len(m)) if subset is None else subset
    sub = {}
    for i in idx:
        sub[m[i]] = c * m[i]
        sub[I[i]] = c * I[i]
    for e in res:
        if sp.simplify(e.subs(sub) - c * e) != 0:
            return False
    return True


def main() -> int:
    c = sp.symbols("c", positive=True)

    print("PROPOSITION: uniform inertial scaling multiplies the passive")
    print("equations by c, hence leaves the passive trajectory invariant.\n")
    print("  N   uniform scaling     one link only        + joint friction")
    print("  " + "-" * 62)
    for n in (1, 2, 3, 4):
        res, params = chain_lagrangian(n)
        uni = scales_exactly(res, params, c)
        one = scales_exactly(res, params, c, subset=[0]) if n > 1 else None
        resf, paramsf = chain_lagrangian(n, with_joint_friction=True)
        fri = scales_exactly(resf, paramsf, c)
        print("  %d   %-19s %-20s %s"
              % (n,
                 "EXACT" if uni else "fails",
                 ("-" if one is None else ("EXACT" if one else "fails, as expected")),
                 "EXACT" if fri else "fails, as expected"))

    print("\n  Column 1 is the proposition: it holds for every N, symbolically,")
    print("  not to within a tolerance.")
    print("  Column 2 breaks the hypothesis of UNIFORMITY.")
    print("  Column 3 breaks the hypothesis that q carries no non-scaling force.")
    print("\n  The base mass never appears: once x is prescribed, the base is")
    print("  outside the passive equations entirely. The symmetry is therefore a")
    print("  property of the ACTUATION INTERFACE as much as of the mechanics --")
    print("  under torque command x is not prescribed and the cancellation fails.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
