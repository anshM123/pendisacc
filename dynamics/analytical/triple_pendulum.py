"""
Analytical planar dynamics of the triple pendulum on a cart.

    M(q) qdd + C(q, qd) qd + G(q) + f(q, qd) = B u

Generalised coordinates      q  = [s, th1, th2, th3]
    s    cart position along the rail (the assembly z axis), metres
    th_i absolute angle of link i from the UPWARD vertical, radians,
         positive tilting toward +z

Absolute (not relative) angles are used because the closed-loop analysis this
project is built around -- eigenvalues at the upright equilibrium, Floquet
multipliers, continuation -- is cleaner when the upright state is simply
q = 0. Joint friction still acts on RELATIVE velocity, which is handled
explicitly in the friction term.

The model is derived symbolically once and lambdified. Every physical parameter
is read from configs/robot/triple_pendulum_params.yaml, which is itself derived
from CAD. Nothing here hard-codes a mass or a length: this file and the Isaac
asset must stay two views of the same numbers, otherwise the whole sim-to-real
comparison is meaningless.

Sign/axis convention matches the CAD extraction: y is vertical, gravity -y.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field

import numpy as np
import sympy as sp

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
PARAMS = os.path.join(ROOT, "configs", "robot", "triple_pendulum_params.yaml")

G_ACC = 9.80665


@dataclass
class LinkParams:
    m: float          # mass, kg
    L: float          # pivot-to-pivot (or pivot-to-tip) length, m
    lc: float         # pivot to centre of mass, m
    I: float          # inertia about the COM, about the revolute (x) axis, kg m^2


@dataclass
class PendulumParams:
    m_cart: float
    links: list[LinkParams]
    g: float = G_ACC

    # dissipation -- zero until identified from hardware
    b_cart: float = 0.0                                   # N s/m
    fc_cart: float = 0.0                                  # N
    b_joint: list[float] = field(default_factory=lambda: [0.0, 0.0, 0.0])    # N m s/rad
    fc_joint: list[float] = field(default_factory=lambda: [0.0, 0.0, 0.0])   # N m
    tanh_k: float = 100.0     # smoothing of the Coulomb term; sharper = stiffer

    @staticmethod
    def from_yaml(path: str = PARAMS) -> "PendulumParams":
        import yaml

        with open(path, encoding="utf-8") as fh:
            p = yaml.safe_load(fh)
        b = p["bodies"]

        links = []
        for name in ("link1", "link2", "link3"):
            e = b[name]
            # the distal link has no distal joint; fall back to the uniform-bar
            # estimate of pivot-to-tip length implied by its centre of mass
            length = e.get("length", 2.0 * e["l_com"])
            links.append(
                LinkParams(
                    m=float(e["mass"]),
                    L=float(length),
                    lc=float(e["l_com"]),
                    # inertia about the COM about x; the stored value is about
                    # the proximal joint, so undo the parallel-axis shift
                    I=float(e["inertia_about_proximal_x"]) - float(e["mass"]) * float(e["l_com"]) ** 2,
                )
            )
        # The cart mass must match the URDF exactly, which means including the
        # reflected drivetrain inertia (J_rotor / r_pulley^2). If the two models
        # disagree here, every eigenvalue comparison downstream is meaningless.
        m_refl = 0.0
        hw_path = os.path.join(os.path.dirname(path), "hardware.yaml")
        if os.path.exists(hw_path):
            with open(hw_path, encoding="utf-8") as fh:
                hw = yaml.safe_load(fh)
            m_refl = float((hw.get("drive") or {}).get("reflected_mass_kg") or 0.0)
        return PendulumParams(m_cart=float(b["cart"]["mass"]) + m_refl, links=links)


class TriplePendulumModel:
    """Symbolically derived, numerically evaluated planar model."""

    def __init__(self, params: PendulumParams):
        self.p = params
        self._build()

    # ------------------------------------------------------------------ build
    def _build(self):
        p = self.p
        t = sp.symbols("t")
        q = sp.Matrix(sp.symbols("s th1 th2 th3", real=True))
        qd = sp.Matrix(sp.symbols("sd th1d th2d th3d", real=True))
        s, th = q[0], [q[1], q[2], q[3]]
        sd, thd = qd[0], [qd[1], qd[2], qd[3]]

        m = [sp.Float(l.m) for l in p.links]
        L = [sp.Float(l.L) for l in p.links]
        lc = [sp.Float(l.lc) for l in p.links]
        Ic = [sp.Float(l.I) for l in p.links]
        g = sp.Float(p.g)

        # positions in the (z, y) plane; th measured from +y, positive toward +z
        pivot = sp.Matrix([s, 0])
        coms, T, V = [], sp.Float(0), sp.Float(0)

        # cart
        T += sp.Rational(1, 2) * sp.Float(p.m_cart) * sd ** 2

        for i in range(3):
            u = sp.Matrix([sp.sin(th[i]), sp.cos(th[i])])
            com = pivot + lc[i] * u
            coms.append(com)

            v = com.jacobian(q) * qd          # chain rule: d(com)/dt
            T += sp.Rational(1, 2) * m[i] * (v.T * v)[0, 0]
            T += sp.Rational(1, 2) * Ic[i] * thd[i] ** 2
            V += m[i] * g * com[1]

            pivot = pivot + L[i] * u

        T = sp.simplify(sp.expand_trig(sp.expand(T)))

        # mass matrix from the quadratic form of T
        M = sp.zeros(4, 4)
        for i in range(4):
            for j in range(4):
                M[i, j] = sp.simplify(sp.diff(T, qd[i], qd[j]))

        # Coriolis/centrifugal via Christoffel symbols of the first kind
        C = sp.zeros(4, 4)
        for i in range(4):
            for j in range(4):
                expr = sp.Float(0)
                for k in range(4):
                    expr += sp.Rational(1, 2) * (
                        sp.diff(M[i, j], q[k])
                        + sp.diff(M[i, k], q[j])
                        - sp.diff(M[j, k], q[i])
                    ) * qd[k]
                C[i, j] = sp.simplify(expr)

        Gv = sp.Matrix([sp.simplify(sp.diff(V, q[i])) for i in range(4)])

        self._sym = dict(q=q, qd=qd, M=M, C=C, G=Gv, T=T, V=V, coms=coms, t=t)

        # NOTE the nesting: [list(q)] makes the generated function take ONE
        # sequence argument. Passing list(q) directly would make it take four
        # separate scalars, which silently mismatches every call site below.
        qa, qda = list(q), list(qd)
        self._M = sp.lambdify([qa], M, "numpy")
        self._C = sp.lambdify([qa, qda], C, "numpy")
        self._G = sp.lambdify([qa], Gv, "numpy")
        self._T = sp.lambdify([qa, qda], T, "numpy")
        self._V = sp.lambdify([qa], V, "numpy")

    # -------------------------------------------------------------- evaluate
    def M(self, q) -> np.ndarray:
        return np.asarray(self._M(list(q)), dtype=float)

    def C(self, q, qd) -> np.ndarray:
        return np.asarray(self._C(list(q), list(qd)), dtype=float)

    def G(self, q) -> np.ndarray:
        return np.asarray(self._G(list(q)), dtype=float).reshape(4)

    def energy(self, q, qd) -> float:
        """Total mechanical energy -- the passive conservation check."""
        return float(self._T(list(q), list(qd))) + float(self._V(list(q)))

    def friction(self, q, qd) -> np.ndarray:
        """Generalised friction forces.

        Cart friction opposes cart velocity. Joint friction opposes the RELATIVE
        angular velocity across each joint and appears with opposite sign on the
        two bodies it couples, which is why this cannot be folded into a simple
        diagonal damping matrix on absolute angles.
        """
        p = self.p
        f = np.zeros(4)
        k = p.tanh_k

        f[0] = p.b_cart * qd[0] + p.fc_cart * np.tanh(k * qd[0])

        rel = [qd[1], qd[2] - qd[1], qd[3] - qd[2]]
        for i, w in enumerate(rel):
            tau = p.b_joint[i] * w + p.fc_joint[i] * np.tanh(k * w)
            f[1 + i] += tau
            if i > 0:
                f[i] -= tau      # reaction on the proximal link
        return f

    def B(self) -> np.ndarray:
        """Input matrix: the cart force is the only actuation."""
        return np.array([1.0, 0.0, 0.0, 0.0])

    def accel(self, q, qd, u: float) -> np.ndarray:
        rhs = self.B() * u - self.C(q, qd) @ np.asarray(qd) - self.G(q) - self.friction(q, qd)
        return np.linalg.solve(self.M(q), rhs)

    def rhs(self, _t, x, u: float = 0.0) -> np.ndarray:
        """State derivative for x = [q, qd]; use with scipy.integrate.solve_ivp."""
        q, qd = np.asarray(x[:4]), np.asarray(x[4:])
        return np.concatenate([qd, self.accel(q, qd, u)])

    # ------------------------------------------------------------- linearise
    def linearize(self, q_eq=None, qd_eq=None, u_eq: float = 0.0, eps: float = 1e-6):
        """Numerical Jacobian A = d(rhs)/dx at an operating point.

        This is the entry point for the closed-loop spectrum once a policy is
        wrapped around it: substitute u = pi(x) and linearise the composite.
        """
        q_eq = np.zeros(4) if q_eq is None else np.asarray(q_eq, dtype=float)
        qd_eq = np.zeros(4) if qd_eq is None else np.asarray(qd_eq, dtype=float)
        x0 = np.concatenate([q_eq, qd_eq])
        f0 = self.rhs(0.0, x0, u_eq)
        A = np.zeros((8, 8))
        for i in range(8):
            dx = np.zeros(8)
            dx[i] = eps
            A[:, i] = (self.rhs(0.0, x0 + dx, u_eq) - self.rhs(0.0, x0 - dx, u_eq)) / (2 * eps)
        B = np.zeros((8, 1))
        du = 1e-6
        B[:, 0] = (self.rhs(0.0, x0, u_eq + du) - self.rhs(0.0, x0, u_eq - du)) / (2 * du)
        return A, B, f0


def load_default() -> TriplePendulumModel:
    return TriplePendulumModel(PendulumParams.from_yaml())
