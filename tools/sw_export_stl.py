"""
Export an STL for every part belonging to a MOVING body (cart, link1..3).

Exported in each part's OWN coordinates. Placement into the assembly reuses the
R / t already stored in bodies_raw.csv -- the same transforms that were validated
against SolidWorks' own centre of mass to 1.4 um, so the meshes land exactly
where the mass properties said they would.

Units are not assumed: SolidWorks' STL unit preference is version-dependent, so
tools/build_link_outlines.py detects the scale by comparing each mesh's extent
against the bounding box recorded in the CSV.
"""

from __future__ import annotations

import os
import re

import pandas as pd
import win32com.client as win32

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CSV = os.path.join(ROOT, "assets", "triple_pendulum", "cad", "bodies_raw.csv")
OUTDIR = os.path.join(ROOT, "assets", "triple_pendulum", "meshes")
GROUPING = os.path.join(ROOT, "configs", "robot", "body_grouping.yaml")

# base included for visuals only -- it is static, so it cannot affect dynamics
MOVING = {"base", "cart", "link1", "link2", "link3"}
# per-body mass floor: the base has 72 parts, most of them fasteners, so use a
# coarser cut there and keep the rails, extrusion and motor housings
MIN_MASS_BY_BODY = {"base": 2e-3}
MIN_MASS = 1e-4          # skip sub-0.1 g parts; they are invisible at this scale
swDocPART = 1


def value(obj, name, *args):
    v = getattr(obj, name)
    if args:
        return v(*args)
    if callable(v):
        try:
            return v()
        except TypeError:
            return v
    return v


def connect_solidworks(timeout_s: int = 240):
    """Attach to SolidWorks, launching and WAITING for it if necessary.

    Dispatch() returns as soon as the COM object exists, which is well before
    SLDWORKS.exe is ready to serve calls -- every subsequent call then fails with
    "The RPC server is unavailable". So poll a cheap property until it answers.
    """
    import subprocess
    import time

    try:
        sw = win32.GetActiveObject("SldWorks.Application")
        sw.Visible = True
        print("[sw] attached to running instance")
        return sw
    except Exception:
        pass

    exe = r"C:\Program Files\SOLIDWORKS Corp\SOLIDWORKS\SLDWORKS.exe"
    if os.path.exists(exe):
        print("[sw] launching SolidWorks; this takes a while...")
        subprocess.Popen([exe])

    deadline = time.time() + timeout_s
    last = None
    while time.time() < deadline:
        time.sleep(5)
        try:
            sw = win32.GetActiveObject("SldWorks.Application")
            _ = value(sw, "RevisionNumber")      # prove it actually answers
            sw.Visible = True
            print("[sw] ready after %.0f s" % (timeout_s - (deadline - time.time())))
            return sw
        except Exception as exc:
            last = exc
    raise SystemExit(
        "SolidWorks did not become responsive within %d s (%s). "
        "Open SolidWorks manually, then re-run this script." % (timeout_s, last)
    )


def safe(name: str) -> str:
    return re.sub(r"[^A-Za-z0-9_.-]", "_", name)[:90]


def main() -> int:
    import yaml

    os.makedirs(OUTDIR, exist_ok=True)
    df = pd.read_csv(CSV)
    with open(GROUPING, encoding="utf-8") as fh:
        cfg = yaml.safe_load(fh)

    df["body"] = [_body_of(n, cfg) for n in df["name"]]
    floors = df["body"].map(lambda b: MIN_MASS_BY_BODY.get(b, MIN_MASS))
    want = df[df["body"].isin(MOVING) & (df["mass"] >= floors)]
    parts = sorted(set(want["part_path"]))
    print("[stl] %d parts to export (of %d moving bodies' parts)" % (len(parts), len(df[df.body.isin(MOVING)])))

    sw = connect_solidworks()

    # keep STL in the part's own frame -- do NOT shift into positive octant
    for pref, val in ((224, True),):        # swSTLDontTranslateToPositive
        try:
            sw.SetUserPreferenceToggle(pref, val)
        except Exception as exc:
            print("[stl] warn: could not set preference %d: %s" % (pref, exc))

    written, failed = 0, []
    for path in parts:
        out = os.path.join(OUTDIR, safe(os.path.basename(path)).replace(".SLDPRT", "") + ".stl")
        if os.path.exists(out):
            written += 1
            continue
        try:
            spec = sw.GetOpenDocSpec(path)
            spec.DocumentType = swDocPART
            spec.ReadOnly = True
            spec.Silent = True
            doc = sw.OpenDoc7(spec)
            if doc is None:
                failed.append((path, "open failed"))
                continue
            ok = False
            for attempt in (lambda: doc.SaveAs4(out, 0, 1, 0, 0, 0),
                            lambda: doc.SaveAs3(out, 0, 1),
                            lambda: doc.SaveAs2(out, 0, True, False)):
                try:
                    attempt()
                    ok = os.path.exists(out)
                    if ok:
                        break
                except Exception:
                    continue
            if ok:
                written += 1
            else:
                failed.append((path, "save failed"))
        except Exception as exc:
            failed.append((path, str(exc)[:60]))

    # Safety net: saving certain components makes SolidWorks emit one STL per
    # sub-component, which once produced 2080 stray files. Our own names never
    # contain spaces (safe() strips them), so anything with a space is junk.
    junk = [f for f in os.listdir(OUTDIR) if f.lower().endswith(".stl") and " " in f]
    for f in junk:
        os.remove(os.path.join(OUTDIR, f))
    if junk:
        print("[stl] removed %d stray per-component files" % len(junk))

    print("[stl] wrote/kept %d, failed %d" % (written, len(failed)))
    for p, why in failed[:12]:
        print("   FAIL %-58s %s" % (os.path.basename(p)[:58], why))
    print("[stl] ->", OUTDIR)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
