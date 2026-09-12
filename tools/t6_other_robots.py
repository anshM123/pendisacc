"""T6: does the symmetry solver generalise to other robot classes?

Pre-registered in PREREGISTRATION_OVERNIGHT.md. CPU only, symbolic.

The solver needs only a Lagrangian and a control law, so pointing it at other
systems is cheap. Three are run here:

  cartpole_torque       cart-pole, cart driven by commanded FORCE
  cartpole_velocity     cart-pole, cart driven by a velocity loop
  arm2_payload_position 2-link arm carrying a payload, joint POSITION control
  tip_force             the triple pendulum under force command

The second claim is the interesting one. H4 tested whether the action interface
changes which errors matter BEHAVIOURALLY and failed (gap +4.3 against a
threshold of 40). This tests whether the interface changes the symmetry group
STRUCTURALLY, which is a different and much sharper question: a group either
contains a generator or it does not.

  run.cmd tools/t6_other_robots.py
"""

from __future__ import annotations

import itertools
import json
import os
import sys

import sympy as sp

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
OUT = os.path.join(ROOT, "results", "T6", "other_robots.json")


def solve(eqs, params, deriv, freeze_time=True):
    """Null space of the exponent constraints; see dynamics/symmetry_group.py."""
    names = list(params) + ["q", "t"]
    k = {nm: sp.Symbol("k_%s" % nm) for nm in names}
    k["ang"] = sp.Integer(0)
    if freeze_time:
        k["t"] = sp.Integer(0)

    def weight(term):
        w = sp.Integer(0)
        for nm, sym in params.items():
            w += sp.degree(sp.Poly(term, sym), sym) * k[nm]
        for sym, (base, order) in deriv.items():
            if base == "ang" and order == 0:
                continue
            d = sp.degree(sp.Poly(term, sym), sym)
            if d:
                w += d * (k[base] - order * k["t"])
        return sp.expand(w)

    cons = []
    for e in eqs:
        ws = [weight(tm) for tm in sp.Add.make_args(sp.expand(e))]
        for A, B in itertools.combinations(ws, 2):
            c = sp.expand(A - B)
            if c != 0:
                cons.append(c)
    free = [nm for nm in names if not isinstance(k[nm], sp.Integer)]
    vars_ = [k[nm] for nm in free]
    M = sp.Matrix([[sp.expand(c).coeff(v) for v in vars_] for c in cons])
    return M.nullspace(), free


def cartpole(drive: str):
    g, mp, mc, lc, I = sp.symbols("g m_pole m_cart lc I_pole", positive=True)
    kv, vc, u = sp.symbols("kv v_cmd u_force", positive=True)
    th, w, a = sp.symbols("th w a")
    X, V, A = sp.symbols("X V A")
    deriv = {th: ("ang", 0), w: ("ang", 1), a: ("ang", 2),
             X: ("q", 0), V: ("q", 1), A: ("q", 2)}
    vx = V + lc * sp.cos(th) * w
    vy = -lc * sp.sin(th) * w
    T = sp.Rational(1, 2) * mp * (vx ** 2 + vy ** 2) + sp.Rational(1, 2) * I * w ** 2
    T += sp.Rational(1, 2) * mc * V ** 2
    Vp = mp * g * lc * sp.cos(th)
    lag = T - Vp

    def ddt(e):
        return (sp.diff(e, th) * w + sp.diff(e, w) * a
                + sp.diff(e, X) * V + sp.diff(e, V) * A)

    force = kv * (vc - V) if drive == "velocity" else u
    eqs = [sp.expand(ddt(sp.diff(lag, V)) - sp.diff(lag, X) - force),
           sp.expand(ddt(sp.diff(lag, w)) - sp.diff(lag, th))]
    params = {"g": g, "m_pole": mp, "m_cart": mc, "lc": lc, "I_pole": I}
    params.update({"kv": kv, "v_cmd": vc} if drive == "velocity" else {"u_force": u})
    return eqs, params, deriv


def arm2_payload():
    """2-link arm, joint POSITION control through PD loops, payload at the tip."""
    g = sp.Symbol("g", positive=True)
    m1, m2, mp = sp.symbols("m1 m2 m_pay", positive=True)
    I1, I2 = sp.symbols("I1 I2", positive=True)
    L1, lc1, lc2 = sp.symbols("L1 lc1 lc2", positive=True)
    kp1, kd1, kp2, kd2 = sp.symbols("kp1 kd1 kp2 kd2", positive=True)
    q1d, q2d = sp.symbols("q1_des q2_des", positive=True)
    t1, t2, w1, w2, a1, a2 = sp.symbols("th1 th2 w1 w2 a1 a2")
    deriv = {t1: ("ang", 0), t2: ("ang", 0), w1: ("ang", 1), w2: ("ang", 1),
             a1: ("ang", 2), a2: ("ang", 2)}
    # planar, absolute angles; payload rides at the tip of link 2
    v1x, v1y = -lc1 * sp.sin(t1) * w1, lc1 * sp.cos(t1) * w1
    e1x, e1y = -L1 * sp.sin(t1) * w1, L1 * sp.cos(t1) * w1
    v2x = e1x - lc2 * sp.sin(t2) * w2
    v2y = e1y + lc2 * sp.cos(t2) * w2
    T = (sp.Rational(1, 2) * m1 * (v1x ** 2 + v1y ** 2) + sp.Rational(1, 2) * I1 * w1 ** 2
         + sp.Rational(1, 2) * m2 * (v2x ** 2 + v2y ** 2) + sp.Rational(1, 2) * I2 * w2 ** 2
         + sp.Rational(1, 2) * mp * (v2x ** 2 + v2y ** 2))
    Vp = m1 * g * lc1 * sp.sin(t1) + (m2 + mp) * g * (L1 * sp.sin(t1) + lc2 * sp.sin(t2))
    lag = T - Vp

    def ddt(e):
        return (sp.diff(e, t1) * w1 + sp.diff(e, w1) * a1
                + sp.diff(e, t2) * w2 + sp.diff(e, w2) * a2)

    eqs = [sp.expand(ddt(sp.diff(lag, w1)) - sp.diff(lag, t1) - (kp1 * q1d - kd1 * w1)),
           sp.expand(ddt(sp.diff(lag, w2)) - sp.diff(lag, t2) - (kp2 * q2d - kd2 * w2))]
    params = {"g": g, "m1": m1, "m2": m2, "m_pay": mp, "I1": I1, "I2": I2,
              "L1": L1, "lc1": lc1, "lc2": lc2,
              "kp1": kp1, "kd1": kd1, "kp2": kp2, "kd2": kd2,
              "q1_des": q1d, "q2_des": q2d}
    return eqs, params, deriv


def show(tag, eqs, params, deriv):
    ns, free = solve(eqs, params, deriv)
    print("%s" % tag)
    print("  %d params, group dimension %d" % (len(free), len(ns)))
    basis = []
    for j, v in enumerate(ns):
        nz = [abs(x) for x in v if x != 0]
        v = v / min(nz) if nz else v
        d = {nm: sp.nsimplify(v[i]) for i, nm in enumerate(free)
             if sp.nsimplify(v[i]) != 0}
        basis.append({a: str(b) for a, b in d.items()})
        print("    g%-2d %s" % (j + 1, ",  ".join("%s^%s" % (a, b) for a, b in d.items())))
    print("")
    return {"dim": len(ns), "free": free, "basis": basis}


def main() -> int:
    out = {}
    print("T6: does the solver generalise to other robot classes?\n")
    e, p, d = cartpole("velocity")
    out["cartpole_velocity"] = show("cart-pole, VELOCITY loop", e, p, d)
    e, p, d = cartpole("torque")
    out["cartpole_torque"] = show("cart-pole, FORCE command", e, p, d)
    e, p, d = arm2_payload()
    out["arm2_payload_position"] = show("2-link arm + payload, POSITION control", e, p, d)

    print("P1 (every system has dim >= 1): %s"
          % ("SUPPORTED" if all(v["dim"] >= 1 for v in out.values()) else "FAIL"))
    same = out["cartpole_velocity"]["basis"] == out["cartpole_torque"]["basis"]
    print("P2 (interface changes the group): %s"
          % ("FAIL -- identical" if same else "SUPPORTED -- groups differ"))
    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    json.dump(out, open(OUT, "w", encoding="utf-8"), indent=1)
    print("\n[out] %s" % OUT)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
