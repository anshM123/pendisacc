"""Generate the frozen hidden target suite (PREREGISTRATION_PI.md, section 1).

24 targets, six families of four, fixed seed. Each target is written to
xi/hidden/HT_XX.json and hashed into xi/hidden/manifest.json. If the manifest
already exists this refuses to run: the suite is an instrument, and an
instrument that can be regenerated after seeing results is not frozen.

Only xi keys that experiments/evaluate.py actually applies are used. Centre of
mass shifts are not included because evaluate.py has no way to apply them;
that is a gap in the suite, stated here rather than papered over.

  run.cmd tools/build_hidden_targets.py
"""

from __future__ import annotations

import hashlib
import json
import os
import sys

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT = os.path.join(ROOT, "xi", "hidden")
MANIFEST = os.path.join(OUT, "manifest.json")
SEED = 20260912
PER_FAMILY = 4

# nominal values the non-mass families perturb around
TAU0 = 0.100          # s
KV0 = 400.0           # N s / m
FCLAMP0 = 349.5       # N


def ratio_only(rng, sigma=0.25):
    """Per-link log-normal factors with the common scale removed."""
    f = rng.normal(0.0, sigma, size=3)
    f -= f.mean()
    return [round(float(np.exp(x)), 4) for x in f]


def family_ratio(rng):
    return {"mass_scale": ratio_only(rng)}


def family_ratio_scale(rng):
    s = float(rng.uniform(0.6, 1.6))
    return {"mass_scale": [round(x * s, 4) for x in ratio_only(rng)],
            "cart_mass_scale": round(float(rng.uniform(0.85, 1.25)), 4)}


def family_dissipation(rng):
    return {"joint_damping": round(float(rng.uniform(0.0005, 0.006)), 5),
            "joint_friction": round(float(rng.uniform(0.001, 0.010)), 5)}


def family_actuator(rng):
    return {"tau": round(float(TAU0 * rng.uniform(0.6, 1.3)), 4),
            "delay_s": round(float(rng.choice([0.0, 0.004, 0.008])), 4)}


def family_drive(rng):
    return {"kv": round(float(KV0 * np.exp(rng.normal(0.0, 0.35))), 2),
            "f_clamp": round(float(FCLAMP0 * rng.uniform(0.7, 1.1)), 2)}


def family_build(rng):
    # The CAD-density bracket, re-read from tools/cad_density_audit.py output.
    # Targets are evaluated on the CORRECTED asset, which already sits at the
    # bracket midpoint (PETG 480 kg/m^3), so the factors are lo/mid .. hi/mid,
    # NOT lo/cad .. hi/cad -- the latter would re-apply the defect that was
    # already corrected. The cart is not in the audit; +-10% is an assumption.
    a = json.load(open(os.path.join(ROOT, "results", "cad_density_audit.json"),
                       encoding="utf-8"))
    mid = np.array(a["corrected_mid"])
    lo, hi = np.array(a["corrected_lo"]) / mid, np.array(a["corrected_hi"]) / mid
    t = float(rng.uniform(0.0, 1.0))
    f = lo + t * (hi - lo)
    return {"mass_scale": [round(float(x), 4) for x in f],
            "cart_mass_scale": round(float(rng.uniform(0.9, 1.1)), 4)}


FAMILIES = (("ratio", family_ratio), ("ratio_scale", family_ratio_scale),
            ("dissipation", family_dissipation), ("actuator", family_actuator),
            ("drive", family_drive), ("build", family_build))


def main() -> int:
    if os.path.exists(MANIFEST):
        print("REFUSING: %s already exists. The hidden suite is frozen." % MANIFEST)
        return 1
    os.makedirs(OUT, exist_ok=True)
    rng = np.random.default_rng(SEED)
    manifest = {"seed": SEED, "per_family": PER_FAMILY, "targets": []}
    k = 0
    for fam, fn in FAMILIES:
        for _ in range(PER_FAMILY):
            xi = fn(rng)
            name = "HT_%02d" % k
            path = os.path.join(OUT, name + ".json")
            blob = json.dumps(xi, indent=1, sort_keys=True).encode("utf-8")
            with open(path, "wb") as fh:
                fh.write(blob)
            manifest["targets"].append({"name": name, "family": fam, "xi": xi,
                                        "sha256": hashlib.sha256(blob).hexdigest()})
            print("  %s  %-12s %s" % (name, fam, json.dumps(xi, sort_keys=True)))
            k += 1
    with open(MANIFEST, "w", encoding="utf-8") as fh:
        json.dump(manifest, fh, indent=1)
    print("\n%d targets written; manifest %s" % (k, MANIFEST))
    return 0


if __name__ == "__main__":
    sys.exit(main())
