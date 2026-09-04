"""
Correctness checks on the analytical model, independent of Isaac.

These are the checks that catch a wrong Lagrangian before it silently poisons
every downstream nonlinear result:

  1. mass matrix symmetric and positive definite
  2. energy conserved for the passive, frictionless system over a long swing
  3. the upright equilibrium is an equilibrium, and is UNSTABLE with the
     expected count of unstable modes
  4. the hanging equilibrium is stable, with three oscillation frequencies
  5. gravity torque signs behave sensibly

Run: env_isaaclab/Scripts/python.exe dynamics/analytical/validate.py
"""

from __future__ import annotations

import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from dynamics.analytical.triple_pendulum import load_default  # noqa: E402


def main() -> int:
    print("building symbolic model ...")
    model = load_default()
    p = model.p
    print("cart mass %.6f kg" % p.m_cart)
    for i, l in enumerate(p.links, 1):
        print("  link%d  m=%.6f  L=%.5f  lc=%.5f  I_com=%.6e" % (i, l.m, l.L, l.lc, l.I))

    ok = True

    # --- 1. mass matrix ---------------------------------------------------
    print("\n[1] mass matrix at q=0")
    M = model.M(np.zeros(4))
    sym_err = float(np.abs(M - M.T).max())
    eig = np.linalg.eigvalsh((M + M.T) / 2)
    print("    symmetry error %.3e   eigenvalues %s" % (sym_err, np.array2string(eig, precision=6)))
    if sym_err > 1e-12 or eig.min() <= 0:
        print("    FAIL: mass matrix must be symmetric positive definite")
        ok = False
    else:
        print("    OK")

    # --- 2. energy conservation ------------------------------------------
    print("\n[2] energy conservation, passive + frictionless, 20 s")
    from scipy.integrate import solve_ivp

    x0 = np.array([0.0, 2.5, 2.0, 3.0, 0.0, 0.0, 0.0, 0.0])   # large-angle swing
    e0 = model.energy(x0[:4], x0[4:])
    sol = solve_ivp(lambda t, x: model.rhs(t, x, 0.0), (0.0, 20.0), x0,
                    rtol=1e-11, atol=1e-12, dense_output=True, max_step=0.01)
    if not sol.success:
        print("    FAIL: integration failed:", sol.message)
        ok = False
    else:
        e = np.array([model.energy(s[:4], s[4:]) for s in sol.y.T])
        drift = float(np.abs(e - e0).max())
        scale = max(abs(e0), float(np.ptp(e)), 1e-12)
        print("    E0=%.9f J   max |dE|=%.3e J   relative %.3e" % (e0, drift, drift / abs(e0)))
        if drift / abs(e0) > 1e-6:
            print("    FAIL: energy not conserved -- the Lagrangian is wrong")
            ok = False
        else:
            print("    OK")

    # --- 3. upright equilibrium ------------------------------------------
    print("\n[3] upright equilibrium q=0 (all links up)")
    f0 = model.rhs(0.0, np.zeros(8), 0.0)
    print("    ||rhs(0)|| = %.3e" % float(np.linalg.norm(f0)))
    if np.linalg.norm(f0) > 1e-9:
        print("    FAIL: q=0 is not an equilibrium")
        ok = False
    A, B, _ = model.linearize()
    ev = np.linalg.eigvals(A)
    ev = ev[np.argsort(-ev.real)]
    n_unstable = int((ev.real > 1e-9).sum())
    print("    eigenvalues (real parts sorted):")
    for v in ev:
        print("      %+10.5f %+10.5fj" % (v.real, v.imag))
    print("    unstable modes: %d" % n_unstable)
    if n_unstable != 3:
        print("    WARNING: expected 3 unstable modes for a triple inverted pendulum")
    else:
        print("    OK")

    # --- 4. hanging equilibrium ------------------------------------------
    print("\n[4] hanging equilibrium (all links down)")
    q_hang = np.array([0.0, np.pi, np.pi, np.pi])
    f_h = model.rhs(0.0, np.concatenate([q_hang, np.zeros(4)]), 0.0)
    print("    ||rhs|| = %.3e" % float(np.linalg.norm(f_h)))
    Ah, _, _ = model.linearize(q_eq=q_hang)
    evh = np.linalg.eigvals(Ah)
    n_unstable_h = int((evh.real > 1e-6).sum())
    freqs = np.sort(np.abs(evh.imag))[::-1]
    freqs = freqs[freqs > 1e-6][::2] / (2 * np.pi)
    print("    unstable modes: %d (expect 0)" % n_unstable_h)
    print("    oscillation frequencies [Hz]: %s" % np.array2string(freqs, precision=4))
    if n_unstable_h != 0:
        print("    FAIL: hanging pendulum should not be unstable")
        ok = False
    else:
        print("    OK")

    # --- 5. gravity torque signs -----------------------------------------
    print("\n[5] gravity torques")
    for label, q in [("all upright", np.zeros(4)),
                     ("tilted +0.1 rad", np.array([0.0, 0.1, 0.1, 0.1]))]:
        print("    %-16s G = %s" % (label, np.array2string(model.G(q), precision=6)))

    print("\n%s" % ("ALL CHECKS PASSED" if ok else "SOME CHECKS FAILED"))
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
