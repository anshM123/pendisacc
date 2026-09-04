"""
Load the converted USD, assert the articulation is what the experiment requires,
and cross-check Isaac's physics against the analytical model.

Run tools/build_usd.py first -- conversion and simulation must not share a
process (see the note in that file).

Checks:
  A. exactly 4 DOFs: 1 prismatic (cart) + 3 revolute
  B. the three revolute joints carry ZERO drive stiffness and damping. If Isaac
     quietly adds implicit stiffness or damping to the pendulum joints then every
     "passive plant" claim in the paper is false.
  C. link masses in the sim equal the CAD-derived masses
  D. Isaac and the analytical model are the same dynamical system

On D -- why this is NOT a trajectory comparison
------------------------------------------------
The upright equilibrium has lambda_max = +16.3 rad/s, a 61 ms divergence time
constant. Over 1 s that is ~1e7 amplification, so two integrators of the SAME
system diverge completely; a raw trajectory comparison there measures nothing but
sensitivity to initial conditions. (That is, of course, this project's whole
thesis about trajectory fidelity, arriving early and uninvited.)

So D is split into quantities that are actually well posed:
  D1 short horizon at upright, well inside one divergence time constant
  D2 long horizon about the STABLE hanging equilibrium, which genuinely
     exercises the nonlinear terms without exponential amplification
  D3 energy conservation of the passive system as simulated by Isaac
  D4 the measured growth rate of a small upright perturbation vs the analytical
     lambda_max -- an eigenvalue, which is the kind of quantity this project
     actually cares about

Results are written to results/asset_validation.json because Kit takes over
stdout after launch and console output is not a reliable record.
"""

from __future__ import annotations

import argparse
import json
import os

# Isaac Sim refuses to boot non-interactively without this. It mirrors the
# acceptance already made when the stack was installed (tools/install_isaaclab.ps1),
# so scripts run the same way from any shell without extra setup.
os.environ.setdefault("OMNI_KIT_ACCEPT_EULA", "YES")
import sys

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser()
parser.add_argument("--dt", type=float, default=1.0 / 1000.0)
AppLauncher.add_app_launcher_args(parser)
args = parser.parse_args()
args.headless = True

app_launcher = AppLauncher(args)
simulation_app = app_launcher.app

import numpy as np  # noqa: E402
import torch  # noqa: E402
import yaml  # noqa: E402

import isaaclab.sim as sim_utils  # noqa: E402
from isaaclab.actuators import ImplicitActuatorCfg  # noqa: E402
from isaaclab.assets import Articulation, ArticulationCfg  # noqa: E402

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
USD_DIR = os.path.join(ROOT, "assets", "triple_pendulum", "usd")
PARAMS = os.path.join(ROOT, "configs", "robot", "triple_pendulum_params.yaml")
OUT = os.path.join(ROOT, "results", "asset_validation.json")

REVOLUTE = ["joint1", "joint2", "joint3"]
PRISMATIC = "cart_slide"


def main() -> int:
    usd_path = os.path.join(USD_DIR, "triple_pendulum.usd")
    if not os.path.exists(usd_path):
        raise SystemExit("missing %s -- run tools/build_usd.py first" % usd_path)
    with open(PARAMS, encoding="utf-8") as fh:
        params = yaml.safe_load(fh)

    sim = sim_utils.SimulationContext(
        sim_utils.SimulationCfg(dt=args.dt, device=args.device, gravity=(0.0, 0.0, -9.80665))
    )
    cfg = ArticulationCfg(
        prim_path="/World/Robot",
        spawn=sim_utils.UsdFileCfg(usd_path=usd_path),
        init_state=ArticulationCfg.InitialStateCfg(pos=(0.0, 0.0, 0.0)),
        actuators={
            "cart": ImplicitActuatorCfg(joint_names_expr=[PRISMATIC],
                                        effort_limit_sim=200.0, stiffness=0.0, damping=0.0),
            "passive": ImplicitActuatorCfg(joint_names_expr=REVOLUTE,
                                           effort_limit_sim=0.0, stiffness=0.0, damping=0.0),
        },
    )
    robot = Articulation(cfg)
    sim.reset()

    R: dict = {"checks": {}, "dt": args.dt}
    ok = True

    names = list(robot.joint_names)
    body_names = list(robot.body_names)
    R["joint_names"] = names
    R["body_names"] = body_names

    # ---- A ----
    a_ok = (len(names) == 4 and PRISMATIC in names and all(j in names for j in REVOLUTE))
    R["checks"]["A_dofs"] = {"pass": bool(a_ok), "joints": names}
    ok &= a_ok

    # ---- B ----
    stiff = robot.data.joint_stiffness[0].cpu().numpy()
    damp = robot.data.joint_damping[0].cpu().numpy()
    b_detail, b_ok = {}, True
    for j in REVOLUTE:
        i = names.index(j)
        b_detail[j] = {"stiffness": float(stiff[i]), "damping": float(damp[i])}
        if abs(stiff[i]) > 1e-12 or abs(damp[i]) > 1e-12:
            b_ok = False
    R["checks"]["B_passive_joints"] = {"pass": bool(b_ok), "detail": b_detail}
    ok &= b_ok

    # ---- C ----
    masses = robot.data.default_mass[0].cpu().numpy()
    c_detail, c_ok = {}, True
    # the cart carries the reflected drivetrain inertia on top of its CAD mass
    hw_path = os.path.join(ROOT, "configs", "robot", "hardware.yaml")
    m_refl = 0.0
    if os.path.exists(hw_path):
        with open(hw_path, encoding="utf-8") as fh:
            m_refl = float((yaml.safe_load(fh).get("drive") or {}).get("reflected_mass_kg") or 0.0)
    extra = {"cart": m_refl}
    for body in ["cart", "link1", "link2", "link3"]:
        if body not in body_names:
            c_detail[body] = "MISSING"
            c_ok = False
            continue
        got = float(masses[body_names.index(body)])
        want = float(params["bodies"][body]["mass"]) + extra.get(body, 0.0)
        rel = abs(got - want) / want
        c_detail[body] = {"sim": got, "cad": want, "rel_err": rel}
        if rel > 1e-4:
            c_ok = False
    R["checks"]["C_masses"] = {"pass": bool(c_ok), "detail": c_detail}
    ok &= c_ok

    # ---- helpers ----
    idx = [names.index(PRISMATIC)] + [names.index(j) for j in REVOLUTE]
    zero = torch.zeros((1, len(names)), device=robot.device)

    def roll(q0_rel, seconds, v0_rel=None):
        """Roll the sim from RELATIVE joint angles/velocities; return ABSOLUTE."""
        q0 = q0_rel
        jp = robot.data.default_joint_pos.clone()
        jv = torch.zeros_like(jp)
        for k, i in enumerate(idx):
            jp[0, i] = float(q0[k])
            if v0_rel is not None:
                jv[0, i] = float(v0_rel[k])
        robot.write_joint_state_to_sim(jp, jv)
        robot.reset()
        n = int(seconds / args.dt)
        qs, vs = [], []
        for _ in range(n):
            robot.set_joint_effort_target(zero)
            robot.write_data_to_sim()
            sim.step()
            robot.update(args.dt)
            qs.append(robot.data.joint_pos[0, idx].cpu().numpy().copy())
            vs.append(robot.data.joint_vel[0, idx].cpu().numpy().copy())
        # convert out of Isaac's relative-angle convention exactly once, here
        return (np.arange(1, n + 1) * args.dt,
                rel_to_abs(np.asarray(qs)), rel_to_abs_vel(np.asarray(vs)))

    from scipy.integrate import solve_ivp

    from dynamics.analytical.triple_pendulum import load_default
    from dynamics.conventions import (HANGING_REL, abs_to_rel, abs_to_rel_vel,
                                      rel_to_abs, rel_to_abs_vel)

    model = load_default()

    def reference(q0_rel, t, v0_rel=None):
        """Analytical reference from the same initial condition, in ABSOLUTE angles."""
        q0 = rel_to_abs(np.asarray(q0_rel, dtype=float))
        v0 = np.zeros(4) if v0_rel is None else rel_to_abs_vel(np.asarray(v0_rel, float))
        x0 = np.concatenate([q0, v0])
        s = solve_ivp(lambda tt, x: model.rhs(tt, x, 0.0), (0.0, float(t[-1])), x0,
                      t_eval=t, rtol=1e-11, atol=1e-13)
        return s.y[:4].T

    # ---- D1: short horizon at upright (inside one divergence time constant) ----
    q0 = np.array([0.0, 0.02, 0.0, 0.0])
    t, qs, _ = roll(q0, 0.05)
    ref = reference(q0, t)
    e1 = float(np.abs(qs[:, 1:] - ref[:, 1:]).max())
    s1 = float(np.abs(ref[:, 1:]).max())
    d1_ok = e1 / s1 < 0.02
    R["checks"]["D1_upright_short"] = {
        "pass": bool(d1_ok), "horizon_s": float(t[-1]),
        "max_abs_err_rad": e1, "max_abs_angle_rad": s1, "rel": e1 / s1}
    ok &= d1_ok

    # ---- D2: long horizon about the stable hanging equilibrium ----
    q0 = HANGING_REL + np.array([0.0, 0.15, -0.10, 0.05])
    t, qs, vs = roll(q0, 3.0)
    ref = reference(q0, t)
    e2 = float(np.abs(qs[:, 1:] - ref[:, 1:]).max())
    s2 = float(np.abs(ref[:, 1:] - ref[0, 1:]).max())
    d2_ok = e2 / s2 < 0.10
    R["checks"]["D2_hanging_long"] = {
        "pass": bool(d2_ok), "horizon_s": float(t[-1]),
        "max_abs_err_rad": e2, "osc_amplitude_rad": s2, "rel": e2 / s2}
    ok &= d2_ok

    # ---- D3: energy conservation of the passive system inside Isaac ----
    en = np.array([model.energy(qs[k], vs[k]) for k in range(len(t))])
    drift = float(np.abs(en - en[0]).max())
    span = float(np.ptp(en)) if np.ptp(en) > 0 else 1.0
    d3_rel = drift / max(abs(en[0]), 1e-12)
    d3_ok = d3_rel < 0.02
    R["checks"]["D3_energy"] = {
        "pass": bool(d3_ok), "E0_J": float(en[0]),
        "max_drift_J": drift, "rel_drift": d3_rel}
    ok &= d3_ok

    # ---- D4: perturb ALONG the dominant eigenvector and measure its growth ----
    # Starting on the eigenvector makes the response a pure exponential from t=0,
    # so the fitted rate is not contaminated by other modes. It also tests the
    # eigenVECTOR, not just the eigenvalue: if Isaac's mode shape differed, the
    # motion would not stay on the ray and the fit would degrade.
    A, _, _ = model.linearize()
    w, V = np.linalg.eig(A)
    k = int(np.argmax(w.real))
    lam = float(w[k].real)
    vec = np.real(V[:, k])
    q_part, v_part = vec[:4], vec[4:]
    scale = 1e-4 / np.abs(q_part[1:]).max()
    q0_abs, v0_abs = q_part * scale, v_part * scale
    # a true eigenvector satisfies qd = lambda * q
    eigvec_consistency = float(np.abs(v0_abs - lam * q0_abs).max() / max(np.abs(v0_abs).max(), 1e-30))

    q0_rel = abs_to_rel(q0_abs)
    v0_rel = abs_to_rel_vel(v0_abs)
    t, qs, _ = roll(q0_rel, 0.40, v0_rel)
    amp = np.linalg.norm(qs[:, 1:], axis=1)
    m = amp < 3e-2                      # stay inside the linear regime
    if m.sum() > 20:
        coef = np.polyfit(t[m], np.log(amp[m]), 1)
        slope = float(coef[0])
        resid = np.log(amp[m]) - np.polyval(coef, t[m])
        r2 = float(1.0 - resid.var() / np.log(amp[m]).var())
        d4_rel = abs(slope - lam) / lam
        d4_ok = d4_rel < 0.02 and r2 > 0.999
    else:
        slope, d4_rel, r2, d4_ok = float("nan"), float("nan"), float("nan"), False
    R["checks"]["D4_growth_rate"] = {
        "pass": bool(d4_ok), "analytic_lambda_max": lam,
        "isaac_fitted_rate": slope, "rel": d4_rel,
        "log_fit_r2": r2, "eigvec_consistency": eigvec_consistency,
        "n_fit_points": int(m.sum())}
    ok &= d4_ok

    R["all_passed"] = bool(ok)
    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    with open(OUT, "w", encoding="utf-8") as fh:
        json.dump(R, fh, indent=2)
    print(json.dumps(R, indent=2))
    return 0 if ok else 1


if __name__ == "__main__":
    code = main()
    simulation_app.close()
    raise SystemExit(code)
