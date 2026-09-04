"""
Dump every rigid body of the SolidWorks assembly with mass properties expressed
in ASSEMBLY (global) coordinates -- SI units (kg, m, kg*m^2).

This is the ground-truth source for the parameter registry xi. Nothing downstream
(URDF/USD/analytical model) may hard-code masses; they all read this dump.

Why it is written this way
--------------------------
IAssemblyDoc.GetComponents returns LATE-BOUND dispatch objects. On those:
  * GetPathName / IsSuppressed / GetChildren / GetSuppression come back as
    PROPERTIES (already-evaluated values), not callables;
  * GetModelDoc2 raises DISP_E_MEMBERNOTFOUND outright;
  * CastTo("IComponent2") fails because the CoClass exposes no type info.
So instead of reaching through the component to its part document, we open each
unique (part file, configuration) once by path -- GetPathName does work -- and
compute mass properties in the PART frame there. The component only has to
supply its placement, via Transform2, which is available.

  c_a = R c_p + t         I_a = R I_p R^T

Self-validation: the summed bodies must reproduce the assembly's own mass and
centre of mass as reported by SolidWorks. The rotation convention of
Transform2.ArrayData is *chosen* by whichever of R / R^T reproduces the
assembly centre of mass, and the residual is recorded in dump_meta.yaml.

Output: assets/triple_pendulum/cad/bodies_raw.csv
        assets/triple_pendulum/cad/dump_meta.yaml
"""

from __future__ import annotations

import csv
import math
import os

import numpy as np
import win32com.client as win32

ASSEMBLY = r"C:\Users\anshm\Downloads\Major Assembly\Major Assembly.SLDASM"
OUTDIR = r"C:\Users\anshm\Downloads\pendisaac\assets\triple_pendulum\cad"

swDocPART = 1
swDocASSEMBLY = 2
swMassPropertyMoment_AboutCenterOfMass = 0


def value(obj, name, *args):
    """Read a COM member that may be exposed as a property or as a method.

    Late-bound SolidWorks objects hand back GetXxx() members as plain values;
    early-bound ones hand back callables. Accept either.
    """
    v = getattr(obj, name)
    if args:
        return v(*args)
    if callable(v):
        try:
            return v()
        except TypeError:
            return v
    return v


def connect():
    try:
        sw = win32.GetActiveObject("SldWorks.Application")
        print("[sw] attached to running instance")
    except Exception:
        sw = win32.Dispatch("SldWorks.Application")
        print("[sw] dispatched new instance")
    sw.Visible = True
    return sw


def open_doc(sw, path, doctype, config=None):
    spec = sw.GetOpenDocSpec(path)
    spec.DocumentType = doctype
    spec.ReadOnly = True  # never write to the user's CAD
    spec.Silent = True
    if config:
        try:
            spec.ConfigurationName = config
        except Exception:
            pass
    return sw.OpenDoc7(spec)


def mass_props(doc):
    """(mass, volume, density, com[3], I_about_com[3x3]) in the doc's own frame, SI."""
    mp = doc.Extension.CreateMassProperty()
    if mp is None:
        return None
    try:
        mp.UseSystemUnits = True
    except Exception:
        pass
    mass = float(value(mp, "Mass"))
    if mass <= 0.0:
        return None
    com = np.array([float(x) for x in value(mp, "CenterOfMass")], dtype=float)
    moi = value(mp, "GetMomentOfInertia", swMassPropertyMoment_AboutCenterOfMass)
    inertia = np.array([float(x) for x in moi], dtype=float).reshape(3, 3)
    return mass, float(value(mp, "Volume")), float(value(mp, "Density")), com, inertia


def main():
    os.makedirs(OUTDIR, exist_ok=True)
    sw = connect()
    print("[sw] revision:", value(sw, "RevisionNumber"))

    model = open_doc(sw, ASSEMBLY, swDocASSEMBLY) or sw.ActiveDoc
    if model is None:
        raise SystemExit("could not open the assembly")
    print("[sw] document:", value(model, "GetTitle"))

    asm_mass, _v, _d, asm_com, _i = mass_props(model)
    print("[sw] ASSEMBLY total: mass=%.6f kg  com=(%.6f, %.6f, %.6f) m" % (asm_mass, *asm_com))

    assy = win32.CastTo(model, "IAssemblyDoc")
    for meth in ("ResolveAllLightWeightComponents", "ResolveAllLightWeight"):
        try:
            getattr(assy, meth)(True)
            print("[sw] resolved lightweight components via %s" % meth)
            break
        except Exception:
            continue

    comps = list(value(assy, "GetComponents", False) or [])
    print("[sw] %d components in tree" % len(comps))

    part_cache: dict[tuple[str, str], object] = {}
    rows = []
    skipped = []

    for comp in comps:
        try:
            name = value(comp, "Name2")
            path = value(comp, "GetPathName") or ""
            if bool(value(comp, "IsSuppressed")):
                skipped.append((name, "suppressed"))
                continue
            children = value(comp, "GetChildren")
            if children and len(children) > 0:
                continue  # sub-assembly: its leaves carry the mass
            if not path or not path.lower().endswith(".sldprt"):
                skipped.append((name, "leaf is not a part file: %r" % path))
                continue

            config = value(comp, "ReferencedConfiguration") or ""
            key = (os.path.normcase(path), config)
            if key not in part_cache:
                doc = open_doc(sw, path, swDocPART, config)
                part_cache[key] = mass_props(doc) if doc is not None else None
            mp = part_cache[key]
            if mp is None:
                skipped.append((name, "no mass properties for %s" % os.path.basename(path)))
                continue
            mass, vol, dens, com_p, inertia_p = mp

            # Transform2 raises DISP_E_MEMBERNOTFOUND on late-bound leaf
            # components; GetXform hands back the same 16 doubles directly
            # (0-8 rotation, 9-11 translation, 12 scale).
            a = None
            try:
                a = [float(x) for x in value(comp, "GetXform")]
            except Exception:
                try:
                    a = [float(x) for x in value(value(comp, "Transform2"), "ArrayData")]
                except Exception:
                    a = None
            if a is None or len(a) < 12:
                skipped.append((name, "no usable placement transform"))
                continue
            rot = np.array(a[0:9], dtype=float).reshape(3, 3)
            trans = np.array(a[9:12], dtype=float)

            try:
                box = [float(x) for x in (value(comp, "GetBox", False, False) or [])]
            except Exception:
                box = []
            if len(box) != 6:
                box = [math.nan] * 6

            rows.append(
                dict(
                    name=name,
                    depth=name.count("/"),
                    parent="/".join(name.split("/")[:-1]),
                    part_file=os.path.basename(path),
                    part_path=path,
                    config=config,
                    mass=mass,
                    volume=vol,
                    density=dens,
                    com_p=com_p,
                    I_p=inertia_p,
                    R=rot,
                    t=trans,
                    box=box,
                )
            )
        except Exception as exc:
            try:
                skipped.append((value(comp, "Name2"), "error: %s" % exc))
            except Exception:
                skipped.append(("<unknown>", "error: %s" % exc))

    print("[sw] %d solid leaf bodies from %d unique parts, %d skipped"
          % (len(rows), len(part_cache), len(skipped)))
    if skipped:
        from collections import Counter
        print("[sw] skip reasons: %s" % dict(Counter(w.split(":")[0] for _n, w in skipped)))
        for nm, why in skipped[:12]:
            print("      %-52s %s" % (nm[:52], why[:70]))
    if not rows:
        raise SystemExit("no bodies extracted -- aborting")

    def assemble(transpose):
        total, moment, out = 0.0, np.zeros(3), []
        for r in rows:
            rot = r["R"].T if transpose else r["R"]
            com_a = rot @ r["com_p"] + r["t"]
            out.append((com_a, rot @ r["I_p"] @ rot.T))
            total += r["mass"]
            moment += r["mass"] * com_a
        return total, (moment / total if total > 0 else moment), out

    m0, c0, out0 = assemble(False)
    m1, c1, out1 = assemble(True)
    err0 = float(np.linalg.norm(c0 - asm_com))
    err1 = float(np.linalg.norm(c1 - asm_com))
    transpose = err1 < err0
    total_mass, total_com, out = (m1, c1, out1) if transpose else (m0, c0, out0)

    print("[chk] convention R   : com err = %.3e m" % err0)
    print("[chk] convention R^T : com err = %.3e m" % err1)
    print("[chk] chose %s" % ("R^T" if transpose else "R"))
    print("[chk] mass summed=%.6f  solidworks=%.6f  rel err=%.3e"
          % (total_mass, asm_mass, abs(total_mass - asm_mass) / max(asm_mass, 1e-12)))
    print("[chk] com  summed=(%.6f, %.6f, %.6f)" % tuple(total_com))
    print("[chk] com  sw    =(%.6f, %.6f, %.6f)" % tuple(asm_com))

    fcsv = os.path.join(OUTDIR, "bodies_raw.csv")
    cols = [
        "name", "depth", "parent", "part_file", "config", "part_path",
        "mass", "volume", "density",
        "com_x", "com_y", "com_z",
        "Ixx", "Iyy", "Izz", "Ixy", "Ixz", "Iyz",
        "bbox_xmin", "bbox_ymin", "bbox_zmin", "bbox_xmax", "bbox_ymax", "bbox_zmax",
        "tx", "ty", "tz",
        "R00", "R01", "R02", "R10", "R11", "R12", "R20", "R21", "R22",
    ]
    with open(fcsv, "w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=cols)
        writer.writeheader()
        for r, (com_a, inertia_a) in zip(rows, out):
            rot = r["R"].T if transpose else r["R"]
            d = dict(
                name=r["name"], depth=r["depth"], parent=r["parent"],
                part_file=r["part_file"], config=r["config"], part_path=r["part_path"],
                mass=r["mass"], volume=r["volume"], density=r["density"],
                com_x=com_a[0], com_y=com_a[1], com_z=com_a[2],
                Ixx=inertia_a[0, 0], Iyy=inertia_a[1, 1], Izz=inertia_a[2, 2],
                Ixy=inertia_a[0, 1], Ixz=inertia_a[0, 2], Iyz=inertia_a[1, 2],
                tx=r["t"][0], ty=r["t"][1], tz=r["t"][2],
            )
            for i in range(3):
                for j in range(3):
                    d["R%d%d" % (i, j)] = rot[i, j]
            for key, val in zip(
                ["bbox_xmin", "bbox_ymin", "bbox_zmin",
                 "bbox_xmax", "bbox_ymax", "bbox_zmax"], r["box"]
            ):
                d[key] = val
            writer.writerow(d)
    print("[out]", fcsv)

    with open(os.path.join(OUTDIR, "dump_meta.yaml"), "w", encoding="utf-8") as fh:
        fh.write("assembly: %s\n" % ASSEMBLY.replace("\\", "/"))
        fh.write("solidworks_revision: '%s'\n" % value(sw, "RevisionNumber"))
        fh.write("units: {mass: kg, length: m, inertia: kg*m^2}\n")
        fh.write("frame: assembly_global\n")
        fh.write("inertia_about: center_of_mass\n")
        fh.write("rotation_convention: %s\n"
                 % ("transpose_of_ArrayData" if transpose else "ArrayData_rowmajor"))
        fh.write("n_bodies: %d\n" % len(rows))
        fh.write("n_unique_parts: %d\n" % len(part_cache))
        fh.write("n_skipped: %d\n" % len(skipped))
        fh.write("assembly_mass_solidworks: %.9f\n" % asm_mass)
        fh.write("assembly_mass_summed: %.9f\n" % total_mass)
        fh.write("assembly_com_solidworks: [%.9f, %.9f, %.9f]\n" % tuple(asm_com))
        fh.write("assembly_com_summed: [%.9f, %.9f, %.9f]\n" % tuple(total_com))
        fh.write("com_match_error_m: %.3e\n" % float(np.linalg.norm(total_com - asm_com)))
        fh.write("mass_match_rel_error: %.3e\n"
                 % (abs(total_mass - asm_mass) / max(asm_mass, 1e-12)))
    print("[out] dump_meta.yaml")


if __name__ == "__main__":
    main()
