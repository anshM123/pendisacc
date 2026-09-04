"""
Offline analysis of results/diag_dynamics.npz -- no Isaac launch needed.

Tests, in order, the candidate explanations for the D1/D2/D3 failures:
  * did the sim accept the initial condition we asked for?
  * are the "continuous" URDF joints actually limited in the USD?
  * is the disagreement just angle wrapping?
  * is there a one-timestep offset between Isaac's reported state and the
    reference time grid?
"""

from __future__ import annotations

import json
import os
import sys

import numpy as np
from scipy.integrate import solve_ivp

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
from dynamics.analytical.triple_pendulum import load_default  # noqa: E402

d = np.load(os.path.join(ROOT, "results", "diag_dynamics.npz"), allow_pickle=True)
meta = json.loads(str(d["meta"]))
model = load_default()

print("=== joint metadata ===")
print("joint_names       :", meta["joint_names"])
print("default_joint_pos :", np.round(meta["default_joint_pos"], 6))
for k in ("joint_pos_limits", "soft_joint_pos_limits"):
    v = meta.get(k)
    print("%-18s: %s" % (k, np.round(np.asarray(v), 4).tolist() if isinstance(v, list) else v))


def reference(q0, t):
    x0 = np.concatenate([np.asarray(q0, float), np.zeros(4)])
    s = solve_ivp(lambda tt, x: model.rhs(tt, x, 0.0), (0.0, float(t[-1])), x0,
                  t_eval=t, rtol=1e-12, atol=1e-14)
    return s.y[:4].T


for tag in ("upright", "hanging", "tiny"):
    q0r = d[tag + "_q0_req"]
    q0a = d[tag + "_q0_acc"]
    t = d[tag + "_t"]
    q = d[tag + "_q"]
    v = d[tag + "_v"]

    print("\n=== %s ===" % tag.upper())
    print("requested q0 :", np.round(q0r, 6))
    print("accepted  q0 :", np.round(q0a, 6))
    ic_err = float(np.abs(q0a - q0r).max())
    print("initial-condition error: %.3e %s" % (ic_err, "  <-- SIM REJECTED THE IC" if ic_err > 1e-6 else ""))

    # integrate the reference from what the SIM actually started at
    ref = reference(q0a, t)

    raw = float(np.abs(q[:, 1:] - ref[:, 1:]).max())
    # wrapping-insensitive comparison
    wrapped = float(np.abs(np.angle(np.exp(1j * (q[:, 1:] - ref[:, 1:])))).max())
    print("max |err| raw            : %.4e rad" % raw)
    print("max |err| mod 2pi        : %.4e rad" % wrapped)

    # one-timestep offset test
    if len(t) > 3:
        shifted = float(np.abs(np.angle(np.exp(1j * (q[1:, 1:] - ref[:-1, 1:])))).max())
        print("max |err| mod 2pi, shift : %.4e rad" % shifted)

    scale = float(np.abs(np.angle(np.exp(1j * (ref[:, 1:] - ref[0, 1:])))).max())
    print("motion amplitude         : %.4e rad" % scale)
    if scale > 1e-12:
        print("relative (mod 2pi)       : %.3f %%" % (100 * wrapped / scale))

    en = np.array([model.energy(q[k], v[k]) for k in range(len(t))])
    print("energy: E0=%.6f J  drift=%.3e J  rel=%.3e"
          % (en[0], np.abs(en - en[0]).max(), np.abs(en - en[0]).max() / max(abs(en[0]), 1e-12)))
    print("max |joint vel|: %s" % np.round(np.abs(v).max(axis=0), 3))
