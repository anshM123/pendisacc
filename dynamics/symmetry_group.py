"""Find ALL exact scaling symmetries of the driven chain, not just the one we knew.

dynamics/symmetry.py proves a single symmetry -- uniform inertial scaling --
because somebody guessed it. tools/geom_confirm.py then found a second one by
measurement, cart mass together with loop gain, with no proof at all. Guessing
does not scale and measurement is discovery-grade. This searches for the group.

METHOD. Write the equations of motion algebraically, with a symbol for each
coordinate and each of its time derivatives. Give every parameter, every
coordinate and time itself a scaling exponent,

    p -> lambda^{k_p} p,    x -> lambda^{k_x} x,    t -> lambda^{k_t} t,

so that a derivative of order r carries lambda^{k_coord - r k_t}. Invariance
means every additive term in an equation picks up the SAME power of lambda, so
the equation is multiplied by a common factor and its solution set is
unchanged. Each pair of terms gives one linear constraint on the exponents,
and the null space of the resulting matrix is the Lie algebra of the scaling
symmetry group. This is Buckingham-Pi applied to the equations rather than to
the units.

WHY IT MATTERS FOR TRANSFER. A generator with k_t = 0 maps a trajectory to the
same trajectory at the same instants, so a policy sampling at a fixed rate
cannot distinguish the two plants: the symmetry is REALISABLE. A generator with
k_t != 0 reparametrises time, and realising it would require rescaling the
control rate, which a 250 Hz policy cannot do. The group therefore splits into
directions the interface can express and directions it cannot -- the same
distinction SS3 found empirically for actuator authority.

  run.cmd dynamics/symmetry_group.py
"""

from __future__ import annotations

import itertools
import json
import os

import sympy as sp

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT = os.path.join(ROOT, "results", "symmetry_group.json")


def build(n: int, with_drive: bool):
    """Algebraic equations of motion for an n-link chain on a cart.

    Coordinates and their derivatives are plain symbols, so the scaling
    analysis is polynomial. Angles are dimensionless; x is a length.
    """
    t = sp.Symbol("t")
    g = sp.Symbol("g", positive=True)
    m = sp.symbols("m1:%d" % (n + 1), positive=True)
    I = sp.symbols("I1:%d" % (n + 1), positive=True)
    L = sp.symbols("L1:%d" % (n + 1), positive=True)
    lc = sp.symbols("lc1:%d" % (n + 1), positive=True)
    mc, kv, vc = sp.symbols("m_cart kv v_cmd", positive=True)

    th = sp.symbols("th1:%d" % (n + 1))
    w = sp.symbols("w1:%d" % (n + 1))          # dth/dt
    a = sp.symbols("a1:%d" % (n + 1))          # d2th/dt2
    X, V, A = sp.symbols("X V A")              # x, dx/dt, d2x/dt2

    # exponent bookkeeping: (parameter or coordinate) -> (base name, deriv order)
    deriv = {}
    for i in range(n):
        deriv[th[i]] = ("th", 0)
        deriv[w[i]] = ("th", 1)
        deriv[a[i]] = ("th", 2)
    deriv[X], deriv[V], deriv[A] = ("x", 0), ("x", 1), ("x", 2)

    # positions of each link centre of mass, and their time derivatives by the
    # chain rule expressed in the symbols above
    T = sp.Integer(0)
    Vpot = sp.Integer(0)
    px, py = X, sp.Integer(0)
    pvx, pvy = V, sp.Integer(0)
    for i in range(n):
        cx = px + lc[i] * sp.sin(th[i])
        cy = py + lc[i] * sp.cos(th[i])
        vx = pvx + lc[i] * sp.cos(th[i]) * w[i]
        vy = pvy - lc[i] * sp.sin(th[i]) * w[i]
        T += sp.Rational(1, 2) * m[i] * (vx ** 2 + vy ** 2)
        T += sp.Rational(1, 2) * I[i] * w[i] ** 2
        Vpot += m[i] * g * cy
        px = px + L[i] * sp.sin(th[i])
        py = py + L[i] * sp.cos(th[i])
        pvx = pvx + L[i] * sp.cos(th[i]) * w[i]
        pvy = pvy - L[i] * sp.sin(th[i]) * w[i]
    if with_drive:
        T += sp.Rational(1, 2) * mc * V ** 2

    def ddt(expr):
        """Total time derivative, promoting each symbol one derivative order."""
        out = sp.Integer(0)
        for i in range(n):
            out += sp.diff(expr, th[i]) * w[i] + sp.diff(expr, w[i]) * a[i]
        out += sp.diff(expr, X) * V + sp.diff(expr, V) * A
        return out

    lag = T - Vpot
    eqs = []
    if with_drive:
        e = ddt(sp.diff(lag, V)) - sp.diff(lag, X)
        eqs.append(sp.expand(e - kv * (vc - V)))
    for i in range(n):
        e = ddt(sp.diff(lag, w[i])) - sp.diff(lag, th[i])
        eqs.append(sp.expand(e))

    params = {"g": g}
    for i in range(n):
        params["m%d" % (i + 1)] = m[i]
        params["I%d" % (i + 1)] = I[i]
        params["L%d" % (i + 1)] = L[i]
        params["lc%d" % (i + 1)] = lc[i]
    if with_drive:
        params.update({"m_cart": mc, "kv": kv, "v_cmd": vc})
    return eqs, params, deriv


def constraints(eqs, params, deriv, freeze_time: bool):
    """Linear constraints on the scaling exponents."""
    names = list(params) + ["x", "t"]
    k = {nm: sp.Symbol("k_%s" % nm) for nm in names}
    k["th"] = sp.Integer(0)                       # angles are dimensionless
    if freeze_time:
        k["t"] = sp.Integer(0)

    def weight(term):
        """Total lambda-exponent of one additive term."""
        w = sp.Integer(0)
        for nm, sym in params.items():
            w += sp.degree(sp.Poly(term, sym), sym) * k[nm]
        for sym, (base, order) in deriv.items():
            # angles themselves appear inside sin/cos, so they are not
            # polynomial generators -- and they are dimensionless, so their
            # exponent is zero and they can be skipped outright. Their
            # DERIVATIVES (w, a) are polynomial and do carry time exponents.
            if base == "th" and order == 0:
                continue
            d = sp.degree(sp.Poly(term, sym), sym)
            if d:
                w += d * (k[base] - order * k["t"])
        return sp.expand(w)

    cons = []
    for e in eqs:
        terms = sp.Add.make_args(e)
        ws = [weight(tm) for tm in terms]
        for A, B in itertools.combinations(ws, 2):
            c = sp.expand(A - B)
            if c != 0:
                cons.append(c)
    free = [nm for nm in names if not isinstance(k[nm], sp.Integer)]
    return cons, k, free


def report(tag, eqs, params, deriv, freeze_time):
    cons, k, free = constraints(eqs, params, deriv, freeze_time)
    vars_ = [k[nm] for nm in free]
    M = sp.Matrix([[sp.expand(c).coeff(v) for v in vars_] for c in cons])
    ns = M.nullspace()
    print("  %d params + coords, %d constraints  ->  group dimension %d"
          % (len(free), len(cons), len(ns)))
    basis = []
    for j, v in enumerate(ns):
        nz = [abs(x) for x in v if x != 0]
        v = v / min(nz) if nz else v
        d = {nm: sp.nsimplify(v[i]) for i, nm in enumerate(free)
             if sp.nsimplify(v[i]) != 0}
        basis.append({a: str(b) for a, b in d.items()})
        print("    g%-2d  %s" % (j + 1, ",  ".join("%s^%s" % (a, b)
                                                   for a, b in d.items())))
    return basis, free


def main() -> int:
    out = {}
    cases = (("prescribed_base_fixed_rate", 3, False, True),
             ("prescribed_base_free_time", 3, False, False),
             ("velocity_loop_fixed_rate", 3, True, True),
             ("velocity_loop_free_time", 3, True, False))
    for tag, n, drive, freeze in cases:
        print("=" * 70)
        print("%s   (n=%d, drive=%s, time %s)"
              % (tag, n, drive, "FROZEN" if freeze else "free"))
        eqs, params, deriv = build(n, drive)
        basis, free = report(tag, eqs, params, deriv, freeze)
        out[tag] = {"n_links": n, "with_drive": drive,
                    "time_frozen": freeze, "dim": len(basis),
                    "free_exponents": free, "basis": basis}
    json.dump(out, open(OUT, "w", encoding="utf-8"), indent=1)
    print("\n[out] %s" % OUT)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
