"""Generate the heterogeneous xi suite.

Calibrated against Isaac first, because a suite in which every condition scores
100% carries no information for a rank correlation. Measured boundaries for the
frozen policy (128 episodes each):

    joint Coulomb friction   0.02  -> 6.2%     boundary ~0.005-0.02
    transport delay          8 ms  -> 100%,  20 ms -> 0%
    deadband                 0.30  -> 100%,  0.60 -> 0%
    velocity-loop gain       150   -> 100%,  60   -> 47.7%
    joint viscous damping    0.08  -> 100%    (no boundary found)
    link mass                2.5x  -> 100%    (no boundary found, verified applied)
    force clamp              40 N  -> 100%    (task needs ~9 N)

Two classes of condition, and the second is as important as the first:

  TRANSFER-CRITICAL   perturbations near a measured failure boundary. These
                      provide the spread a rank statistic needs.
  NEGATIVE CONTROLS   LARGE model errors that are measurably harmless -- 2.5x
                      link mass, heavy viscous damping, a force clamp cut to a
                      quarter. A predictor that merely tracks "size of model
                      error" must fail on these, because they are large and
                      benign. This is where a norm-based measure should lose and
                      a margin-projected one should not.

Ranges are annotated with whether they lie inside the plausible build envelope,
so a reviewer can see which conditions are realistic and which deliberately sit
past the edge to make the correlation testable.

  run.cmd experiments/suite.py --write
"""

from __future__ import annotations

import argparse
import json
import os

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
XI_DIR = os.path.join(ROOT, "xi", "suite")

# name, xi, family, plausible-for-the-real-build?
CONDITIONS = [
    ("nominal",            {},                                          "none",       True),

    # ---- actuation: order and speed ------------------------------------
    ("act_tau_50",         {"tau": 0.050},                              "actuation",  True),
    ("act_tau_70",         {"tau": 0.070},                              "actuation",  True),
    ("act_tau_130",        {"tau": 0.130},                              "actuation",  True),
    ("act_tau_160",        {"tau": 0.160},                              "actuation",  True),
    ("act_2nd_wn28",       {"order": 2, "omega_n": 28.0, "zeta": 0.9},  "actuation",  True),
    ("act_2nd_wn40",       {"order": 2, "omega_n": 40.0, "zeta": 0.9},  "actuation",  True),
    ("act_2nd_wn20",       {"order": 2, "omega_n": 20.0, "zeta": 0.9},  "actuation",  True),
    ("act_2nd_z05",        {"order": 2, "omega_n": 28.0, "zeta": 0.5},  "actuation",  True),

    # ---- actuation: dead time and deadband -----------------------------
    ("act_delay_4ms",      {"delay_s": 0.004},                          "actuation",  True),
    ("act_delay_12ms",     {"delay_s": 0.012},                          "actuation",  True),
    ("act_delay_16ms",     {"delay_s": 0.016},                          "actuation",  True),
    ("act_db_040",         {"deadband": 0.40},                          "actuation",  False),
    ("act_db_050",         {"deadband": 0.50},                          "actuation",  False),

    # ---- dissipation ----------------------------------------------------
    ("dis_jfric_004",      {"joint_friction": 0.004},                   "dissipation", True),
    ("dis_jfric_008",      {"joint_friction": 0.008},                   "dissipation", True),
    ("dis_jfric_012",      {"joint_friction": 0.012},                   "dissipation", True),
    ("dis_jfric_016",      {"joint_friction": 0.016},                   "dissipation", True),
    ("dis_jdamp_004",      {"joint_damping": 0.004},                    "dissipation", True),

    # ---- drive authority -------------------------------------------------
    ("drv_kv_100",         {"kv": 100.0},                               "actuation",  True),
    ("drv_kv_70",          {"kv": 70.0},                                "actuation",  True),
    ("drv_kv_50",          {"kv": 50.0},                                "actuation",  False),

    # ---- rigid body ------------------------------------------------------
    ("rb_mass_130",        {"mass_scale": [1.3, 1.3, 1.3]},             "rigidbody",  True),
    ("rb_mass_link1_150",  {"mass_scale": [1.5, 1.0, 1.0]},             "rigidbody",  True),
    ("rb_cart_130",        {"cart_mass_scale": 1.3},                    "rigidbody",  True),
    ("rb_grav_990",        {"gravity": 9.90},                           "rigidbody",  True),
    ("rb_grav_1050",       {"gravity": 10.50},                          "rigidbody",  False),
    ("rb_grav_1100",       {"gravity": 11.00},                          "rigidbody",  False),

    # ---- NEGATIVE CONTROLS: large error, measured harmless ---------------
    ("neg_mass_250",       {"mass_scale": [2.5, 2.5, 2.5]},             "control",    False),
    ("neg_jdamp_080",      {"joint_damping": 0.080},                    "control",    False),
    ("neg_fclamp_40",      {"f_clamp": 40.0},                           "control",    False),
    ("neg_kv_150",         {"kv": 150.0},                               "control",    True),

    # ---- COMBINATIONS: reality does not turn one knob at a time ----------
    ("mix_delay_fric",     {"delay_s": 0.008, "joint_friction": 0.006}, "combination", True),
    ("mix_tau_mass",       {"tau": 0.070, "mass_scale": [1.3, 1.3, 1.3]}, "combination", True),
    ("mix_2nd_fric",       {"order": 2, "omega_n": 32.0, "zeta": 0.9,
                            "joint_friction": 0.004},                   "combination", True),
    ("mix_kv_delay",       {"kv": 100.0, "delay_s": 0.008},             "combination", True),
    ("mix_heavy_benign",   {"mass_scale": [2.0, 2.0, 2.0], "joint_damping": 0.04,
                            "f_clamp": 60.0},                           "combination", False),
    ("mix_realistic",      {"tau": 0.085, "joint_friction": 0.003, "delay_s": 0.004,
                            "mass_scale": [1.15, 1.1, 1.12],
                            "cart_mass_scale": 1.08},                   "combination", True),
]


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--write", action="store_true")
    a = ap.parse_args()

    os.makedirs(XI_DIR, exist_ok=True)
    manifest = []
    for name, xi, family, plausible in CONDITIONS:
        manifest.append({"name": name, "xi": xi, "family": family, "plausible": plausible})
        if a.write:
            json.dump(xi, open(os.path.join(XI_DIR, name + ".json"), "w"), indent=1)

    if a.write:
        json.dump(manifest, open(os.path.join(ROOT, "xi", "manifest.json"), "w"), indent=1)

    fams = {}
    for m in manifest:
        fams[m["family"]] = fams.get(m["family"], 0) + 1
    print("%d conditions" % len(manifest))
    for f, n in sorted(fams.items()):
        print("  %-12s %2d" % (f, n))
    print("  plausible for the real build: %d of %d"
          % (sum(1 for m in manifest if m["plausible"]), len(manifest)))
    if a.write:
        print("\nwrote %s and xi/manifest.json" % XI_DIR)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
