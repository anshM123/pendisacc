"""The equivalence class depends on the ACTION INTERFACE, not only on the plant.

`dynamics/symmetry.py` proves symbolically that uniform inertial scaling of an
N-link chain leaves the passive trajectory invariant PROVIDED the base
coordinate x is kinematically prescribed. That proviso is a statement about the
actuator, not about the mechanics: a stiff velocity loop with authority to
spare prescribes x; a force/torque command does not.

This file measures the consequence on the actual triple pendulum. Identical
plant, identical model error (uniform link-mass scaling by c), two interfaces:

  VELOCITY interface   cart acceleration prescribed. Solve the link rows only,
                       xddot given:
                           M[1:,1:] thddot = -M[1:,0] xddot - (C qd)[1:] - G[1:]
                       Every one of M, C, G is linear in {m_i, I_i}, so both
                       sides scale by c and it cancels. Predicted: EXACTLY zero
                       change, to floating point.

  FORCE interface      cart force prescribed. Solve the full 4-DOF system:
                           M qddot = B u - C qd - G
                       The cart mass does NOT scale with c, so row 0 mixes a
                       scaled block with an unscaled one and nothing cancels.
                       Predicted: error growing with c, and NOT small.

The point is not that force control is worse. It is that the same physical
parameter direction is a null direction of one interface and a sensitive
direction of the other, so "is this simulator accurate enough?" cannot be
answered from the model alone.

  run.cmd tools/interface_symmetry.py
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

OUT = os.path.join(ROOT, "results", "interface_symmetry.json")
C_VALUES = (1.5, 2.0, 2.5, 4.0, 8.0, 20.0)


def scaled(p: PendulumParams, c: float) -> PendulumParams:
    """Uniform scaling of every LINK inertial parameter. The cart is untouched:
    it is not part of the passive subsystem, and on hardware its mass is set by
    the carriage and rotor, not by the links."""
    links = [replace(l, m=l.m * c, I=l.I * c) for l in p.links]
    return replace(p, links=links)


def theta_accel_velocity_interface(model, q, qd, xddot):
    """Link accelerations with the cart acceleration PRESCRIBED."""
    M = model.M(q)
    rhs = -M[1:, 0] * xddot - (model.C(q, qd) @ np.asarray(qd))[1:] - model.G(q)[1:]
    return np.linalg.solve(M[1:, 1:], rhs)


def theta_accel_force_interface(model, q, qd, u):
    """Link accelerations with the cart FORCE prescribed: full 4-DOF solve."""
    return model.accel(q, qd, u)[1:]


def main() -> int:
    p = PendulumParams.from_yaml()
    base = TriplePendulumModel(p)

    rng = np.random.default_rng(0)
    states = [(np.concatenate([[rng.uniform(-0.3, 0.3)], rng.uniform(-np.pi, np.pi, 3)]),
               np.concatenate([[rng.uniform(-2, 2)], rng.uniform(-8, 8, 3)]))
              for _ in range(400)]
    # Matched excitation: for each state the prescribed acceleration and the
    # prescribed force are the SAME nominal motion -- u is the force the
    # nominal plant needs to realise that xddot. Without this the two columns
    # would differ merely because they were driven differently.
    drive = [rng.uniform(-20, 20) for _ in states]
    forces = []
    for (q, qd), xdd in zip(states, drive):
        thdd = theta_accel_velocity_interface(base, q, qd, xdd)
        M = base.M(q)
        acc = np.concatenate([[xdd], thdd])
        forces.append(float((M @ acc + base.C(q, qd) @ np.asarray(qd) + base.G(q))[0]))

    print("Same plant, same model error (uniform link mass x c), two interfaces.")
    print("Relative change in link acceleration, worst over 400 random states.\n")
    print("     c      VELOCITY interface        FORCE interface      ratio")
    print("            (x prescribed)            (u prescribed)")
    print("  " + "-" * 66)

    rows = []
    for c in C_VALUES:
        m = TriplePendulumModel(scaled(p, c))
        ev = ef = mag = 0.0
        for (q, qd), xdd, u in zip(states, drive, forces):
            a0 = theta_accel_velocity_interface(base, q, qd, xdd)
            a1 = theta_accel_velocity_interface(m, q, qd, xdd)
            ev = max(ev, float(np.max(np.abs(a1 - a0))))
            b0 = theta_accel_force_interface(base, q, qd, u)
            b1 = theta_accel_force_interface(m, q, qd, u)
            ef = max(ef, float(np.max(np.abs(b1 - b0))))
            mag = max(mag, float(np.max(np.abs(a0))))
        rv, rf = ev / mag, ef / mag
        rows.append({"c": c, "rel_velocity": rv, "rel_force": rf,
                     "ratio": rf / rv if rv > 0 else float("inf")})
        print("  %6.1f      %.3e               %.3e        %s"
              % (c, rv, rf, "%.1e" % (rf / rv) if rv > 0 else "inf"))

    worst_v = max(r["rel_velocity"] for r in rows)
    print("\n  Velocity interface: worst relative change %.2e over every c up to"
          % worst_v)
    print("  %.0fx -- machine precision. The direction is EXACTLY null." % max(C_VALUES))
    print("  Force interface: the same direction is strongly sensitive, reaching")
    print("  %.0f%% at c = %.0f." % (100 * rows[-1]["rel_force"], C_VALUES[-1]))
    print("\n  The plant is identical in both columns. Only the variable the")
    print("  controller commands has changed. Equivalence classes in physical")
    print("  parameter space are therefore a property of (plant, interface),")
    print("  and a simulator-fidelity judgement made without naming the")
    print("  interface is not well posed.")

    # ---------------------------------------------------------------
    # The 2x2. Three DIRECTIONS in link-mass space, each displaced by the
    # SAME Euclidean distance ||dm||, evaluated under both interfaces.
    # Equal parameter distance is the whole point: with magnitude held
    # fixed, any difference between rows is a property of direction alone.
    #
    # Under the velocity interface the rows are not comparable at all --
    # one direction is exactly null and the others are not. Under the
    # force interface that distinction largely collapses and every
    # direction is simply "wrong masses". The interface does not move the
    # sensitivity boundary; it changes what the boundary is drawn around.
    print("\n\nTHE 2x2: equal parameter distance, three directions, two interfaces\n")
    m0 = np.array([l.m for l in p.links], dtype=float)
    D = 0.5 * float(np.linalg.norm(m0))          # fixed displacement magnitude
    directions = (("uniform    [c,c,c]", m0 / np.linalg.norm(m0)),
                  ("transverse [c,1,1]", np.array([1.0, 0.0, 0.0])),
                  ("transverse [1,1,c]", np.array([0.0, 0.0, 1.0])))
    grid = []
    for label, u in directions:
        dm = D * u
        # inertia follows mass at fixed geometry: a density error, not a
        # geometry error, so I scales by exactly the same factor as m
        links = [replace(l, m=l.m + d, I=l.I * (l.m + d) / l.m)
                 for l, d in zip(p.links, dm)]
        m = TriplePendulumModel(replace(p, links=links))
        ev = ef = mag = 0.0
        for (q, qd), xdd, u_f in zip(states, drive, forces):
            a0 = theta_accel_velocity_interface(base, q, qd, xdd)
            a1 = theta_accel_velocity_interface(m, q, qd, xdd)
            ev = max(ev, float(np.max(np.abs(a1 - a0))))
            b0 = theta_accel_force_interface(base, q, qd, u_f)
            b1 = theta_accel_force_interface(m, q, qd, u_f)
            ef = max(ef, float(np.max(np.abs(b1 - b0))))
            mag = max(mag, float(np.max(np.abs(a0))))
        grid.append({"direction": label, "delta_m_kg": float(np.linalg.norm(dm)),
                     "scale_factors": [round(float(1 + d / l.m), 3)
                                       for d, l in zip(dm, p.links)],
                     "rel_velocity": ev / mag, "rel_force": ef / mag})
    print("   direction            factors             ||dm||   VELOCITY     FORCE")
    print("   " + "-" * 70)
    for g in grid:
        print("   %-20s %-19s %6.3f   %.3e   %.3e"
              % (g["direction"], str(g["scale_factors"]), g["delta_m_kg"],
                 g["rel_velocity"], g["rel_force"]))
    sv = [g["rel_velocity"] for g in grid]
    sf = [g["rel_force"] for g in grid]
    rv_ = max(sv) / max(min(sv), 1e-300)
    rf_ = max(sf) / max(min(sf), 1e-300)
    print("\n   Ratio worst/best direction at EQUAL parameter distance:")
    print("     velocity interface   %.1e   direction is everything" % rv_)
    print("     force interface      %.1f      direction barely matters" % rf_)
    print("\n   Same plant. Same three model errors. Same parameter distances.")
    print("   The only change is which variable the controller commands.")

    json.dump({"c_values": list(C_VALUES), "rows": rows,
               "n_states": len(states),
               "worst_rel_velocity_interface": worst_v,
               "grid": grid,
               "direction_ratio_velocity": rv_,
               "direction_ratio_force": rf_},
              open(OUT, "w", encoding="utf-8"), indent=1)
    print("\n[out] %s" % OUT)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
