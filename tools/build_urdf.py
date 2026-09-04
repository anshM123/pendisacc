"""
Generate the triple-pendulum URDF from the CAD-derived parameters.

Frame convention (URDF / Isaac, Z-up) vs the SolidWorks assembly frame:

    URDF x  =  CAD z      rail axis, cart travel
    URDF y  =  CAD x      revolute axis (all three joints are parallel to it)
    URDF z  =  CAD y      vertical, gravity -z

That is a cyclic permutation, so handedness is preserved.

Each link's frame has its origin ON its proximal joint with +Z pointing along the
link toward the distal joint. Therefore all joint angles zero == fully upright,
which is exactly the convention the analytical model uses, so the two stay
directly comparable.

The three revolute joints are PASSIVE: no drive, no stiffness, no damping is
written here. Only the cart's prismatic joint is actuated. If Isaac ever adds
implicit stiffness or damping to the pendulum joints the experiment is invalid,
so `validate_asset.py` asserts this after conversion.

Inertia is rotated from the CAD assembly frame into each link's own frame:

    I_local = R I_cad R^T,   rows of R = the link frame's axes in CAD coords

Geometry is VISUAL ONLY -- see add_capsule(). The analytical model has no
contacts, so collision shapes would make the two plants different systems.

Visuals use the real CAD meshes when tools/build_body_meshes.py has produced
them (assets/triple_pendulum/meshes/visual/<body>.obj, already in each link's
own frame); otherwise they fall back to capsules and boxes.

IMPORTANT -- joint angle convention. URDF/Isaac joint angles are RELATIVE to the
parent link, while the analytical model uses ABSOLUTE angles from vertical. They
coincide only near q = 0. Everything crossing between the two must go through
dynamics/conventions.py.
"""

from __future__ import annotations

import os
import xml.dom.minidom as minidom
import xml.etree.ElementTree as ET

import numpy as np
import pandas as pd
import yaml

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PARAMS = os.path.join(ROOT, "configs", "robot", "triple_pendulum_params.yaml")
HARDWARE = os.path.join(ROOT, "configs", "robot", "hardware.yaml")
CSV = os.path.join(ROOT, "assets", "triple_pendulum", "cad", "bodies_raw.csv")
OUT = os.path.join(ROOT, "assets", "triple_pendulum", "urdf", "triple_pendulum.urdf")
VISUAL = os.path.join(ROOT, "assets", "triple_pendulum", "meshes", "visual")

# bearing parts that mark each joint's axis position along CAD x
JOINT_BEARING = {
    "J1": ("R65972K91_Ball Bearing.SLDPRT", None),
    "J2": ("R65972K91_Steel Ball Bearing(1).SLDPRT", "R6PEndulum assembly-1"),
    "J3": ("R65972K91_Steel Ball Bearing(1).SLDPRT", "R6PEndulum assembly-2"),
}


def cad_to_urdf(v3):
    """(x, y, z)_CAD -> (x, y, z)_URDF."""
    x, y, z = v3
    return np.array([z, x, y], dtype=float)


def joint_x_positions(df: pd.DataFrame) -> dict[str, float]:
    """CAD x of each revolute axis, from the ball bearings that define it."""
    out = {}
    for jname, (part, prefix) in JOINT_BEARING.items():
        sel = df["part_file"] == part
        if prefix:
            sel &= df["name"].str.startswith(prefix)
        rows = df[sel]
        if not len(rows):
            raise SystemExit("no bearing found for %s (%s)" % (jname, part))
        out[jname] = float(rows["com_x"].mean())
    return out


def sub(parent, tag, **attrs):
    return ET.SubElement(parent, tag, {k: str(v) for k, v in attrs.items()})


def xyz(v):
    return "%.9g %.9g %.9g" % tuple(v)


def add_inertial(link, mass, com_local, inertia_local):
    node = sub(link, "inertial")
    sub(node, "origin", xyz=xyz(com_local), rpy="0 0 0")
    sub(node, "mass", value="%.9g" % mass)
    sub(node, "inertia",
        ixx="%.9g" % inertia_local[0, 0], ixy="%.9g" % inertia_local[0, 1],
        ixz="%.9g" % inertia_local[0, 2], iyy="%.9g" % inertia_local[1, 1],
        iyz="%.9g" % inertia_local[1, 2], izz="%.9g" % inertia_local[2, 2])


def add_mesh_visual(link, body, rgba):
    """Reference the merged CAD mesh for this body, if tools/build_body_meshes.py
    has produced one. The mesh is already expressed in the link frame, so the
    origin is identity. Visual only -- collision stays off (see add_capsule)."""
    path = os.path.join(VISUAL, "%s.obj" % body)
    if not os.path.exists(path):
        return False
    node = sub(link, "visual")
    sub(node, "origin", xyz="0 0 0", rpy="0 0 0")
    geom = sub(node, "geometry")
    sub(geom, "mesh", filename=path.replace("\\", "/"), scale="1 1 1")
    mat = sub(node, "material", name=body)
    sub(mat, "color", rgba=rgba)
    return True


def add_capsule(link, length, radius, name, rgba, collision=False):
    """Capsule running from the link origin along +Z.

    Collision geometry is OFF by default. The analytical model has no contacts,
    so giving the links collision shapes makes the two plants different systems:
    hanging links intersect the rail, PhysX resolves the contact, and energy is
    injected (measured: +8.1 J on a 1.1 J system). Prismatic joint limits already
    provide the cart end stops, so nothing here needs to collide.
    """
    for tag in (("visual", "collision") if collision else ("visual",)):
        node = sub(link, tag)
        sub(node, "origin", xyz=xyz([0, 0, length / 2.0]), rpy="0 0 0")
        geom = sub(node, "geometry")
        sub(geom, "cylinder", radius="%.6g" % radius, length="%.6g" % length)
        if tag == "visual":
            mat = sub(node, "material", name=name)
            sub(mat, "color", rgba=rgba)


def add_box(link, size, origin, name, rgba, collision=False):
    for tag in (("visual", "collision") if collision else ("visual",)):
        node = sub(link, tag)
        sub(node, "origin", xyz=xyz(origin), rpy="0 0 0")
        geom = sub(node, "geometry")
        sub(geom, "box", size=xyz(size))
        if tag == "visual":
            mat = sub(node, "material", name=name)
            sub(mat, "color", rgba=rgba)


def main():
    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    with open(PARAMS, encoding="utf-8") as fh:
        p = yaml.safe_load(fh)
    df = pd.read_csv(CSV)
    jx = joint_x_positions(df)
    print("[urdf] revolute axis CAD-x: %s" % {k: round(v, 5) for k, v in jx.items()})

    B = p["bodies"]
    Jyz = {k: np.array(v["pos_yz"], dtype=float) for k, v in p["joints"].items()}
    # full 3-D joint positions in CAD coords
    Jcad = {k: np.array([jx[k], Jyz[k][0], Jyz[k][1]]) for k in Jyz}

    # link direction unit vectors in CAD (y, z); link3 has no distal joint so its
    # direction is taken from its own centre of mass
    def link_dir(name, prox, dist):
        a = Jyz[prox]
        b = Jyz[dist] if dist else np.array(B[name]["com"])[[1, 2]]
        d = b - a
        return d / np.linalg.norm(d)

    dirs = {
        "link1": link_dir("link1", "J1", "J2"),
        "link2": link_dir("link2", "J2", "J3"),
        "link3": link_dir("link3", "J3", None),
    }

    # rotation from CAD frame into each link frame:
    #   local Z = along the link, local Y = revolute axis (CAD +x), local X = Y x Z
    rot = {}
    for name, d in dirs.items():
        ez = np.array([0.0, d[0], d[1]])          # CAD (x, y, z)
        ey = np.array([1.0, 0.0, 0.0])            # revolute axis
        ex = np.cross(ey, ez)
        R = np.vstack([ex, ey, ez])               # rows = local axes in CAD coords
        assert abs(np.linalg.det(R) - 1.0) < 1e-9, "link frame must be right-handed"
        rot[name] = R

    robot = ET.Element("robot", {"name": "triple_pendulum"})
    ET.Comment("generated by tools/build_urdf.py from CAD -- do not hand-edit")

    # ---- world / base ----
    base = sub(robot, "link", name="base_link")
    # real extrusion + SBR20 rails + risers + motor mounts when the meshes have
    # been built; the plain bar is only a placeholder. base_link is fixed to the
    # world, so this is purely cosmetic and cannot affect the dynamics.
    if not add_mesh_visual(base, "base", "0.55 0.56 0.60 1"):
        add_box(base, [p["prismatic"]["rail_span"], 0.06, 0.02], [0, 0, -0.02],
                "rail", "0.35 0.35 0.38 1")
    node = sub(base, "inertial")
    sub(node, "origin", xyz="0 0 0", rpy="0 0 0")
    sub(node, "mass", value="%.9g" % B["base"]["mass"])
    sub(node, "inertia", ixx="1", ixy="0", ixz="0", iyy="1", iyz="0", izz="1")

    # ---- cart ----
    cart = sub(robot, "link", name="cart")
    cart_com_local = cad_to_urdf(np.array(B["cart"]["com"]) - Jcad["J1"])
    Ic = B["cart"]["inertia_com"]
    # cart does not rotate, so its CAD inertia only needs the axis permutation
    I_cad = np.array([[Ic["ixx"], Ic["ixy"], Ic["ixz"]],
                      [Ic["ixy"], Ic["iyy"], Ic["iyz"]],
                      [Ic["ixz"], Ic["iyz"], Ic["izz"]]])
    P = np.array([[0, 0, 1], [1, 0, 0], [0, 1, 0]], dtype=float)  # CAD -> URDF

    # The motor rotor, seen through the GT2 pulley, adds m = J_rotor / r^2 to
    # everything the cart force has to accelerate. With r = 12.7 mm that is
    # ~0.74 kg against a 0.145 kg CAD cart -- the drivetrain DOMINATES the
    # translating inertia, and leaving it out makes the simulated cart roughly
    # 6x too easy to accelerate.
    with open(HARDWARE, encoding="utf-8") as fh:
        hw = yaml.safe_load(fh)
    m_refl = float(hw["drive"].get("reflected_mass_kg") or 0.0)
    cart_mass = B["cart"]["mass"] + m_refl
    print("[urdf] cart: CAD %.5f kg + reflected drivetrain %.5f kg = %.5f kg"
          % (B["cart"]["mass"], m_refl, cart_mass))
    add_inertial(cart, cart_mass, cart_com_local, P @ I_cad @ P.T)
    if not add_mesh_visual(cart, "cart", "0.62 0.62 0.66 1"):
        add_box(cart, [0.06, 0.05, 0.04], cart_com_local, "cart", "0.45 0.30 0.60 1")

    j = sub(robot, "joint", name="cart_slide", type="prismatic")
    sub(j, "parent", link="base_link")
    sub(j, "child", link="cart")
    sub(j, "origin", xyz=xyz(cad_to_urdf(Jcad["J1"] * np.array([1, 1, 0]))), rpy="0 0 0")
    sub(j, "axis", xyz="1 0 0")
    # usable stroke, kept inside the rail span so the cart never reaches a hard stop
    # ~1.2 m of usable travel on a 1.524 m rail; keep the joint limit just
    # outside the task's own bound so terminations come from the task, not PhysX
    half = 0.62
    sub(j, "limit", lower="%.6g" % -half, upper="%.6g" % half,
        effort="400", velocity="20")

    # ---- three passive links ----
    prev = "cart"
    prev_joint = "J1"
    radii = {"link1": 0.012, "link2": 0.010, "link3": 0.010}
    colors = {"link1": "0.78 0.16 0.16 1", "link2": "0.08 0.43 0.78 1",
              "link3": "0.08 0.59 0.24 1"}
    for idx, name in enumerate(["link1", "link2", "link3"], start=1):
        e = B[name]
        R = rot[name]
        link = sub(robot, "link", name=name)

        Ie = e["inertia_com"]
        I_cad = np.array([[Ie["ixx"], Ie["ixy"], Ie["ixz"]],
                          [Ie["ixy"], Ie["iyy"], Ie["iyz"]],
                          [Ie["ixz"], Ie["iyz"], Ie["izz"]]])
        I_local = R @ I_cad @ R.T
        L = e.get("length", 2.0 * e["l_com"])
        add_inertial(link, e["mass"], [0.0, 0.0, e["l_com"]], I_local)
        if not add_mesh_visual(link, name, colors[name]):
            add_capsule(link, L, radii[name], name, colors[name])

        jt = sub(robot, "joint", name="joint%d" % idx, type="continuous")
        sub(jt, "parent", link=prev)
        sub(jt, "child", link=name)
        if prev == "cart":
            origin = np.zeros(3)                       # cart frame already sits on J1
        else:
            origin = np.array([0.0, 0.0, B[prev].get("length", 0.0)])
        sub(jt, "origin", xyz=xyz(origin), rpy="0 0 0")
        sub(jt, "axis", xyz="0 1 0")
        # NO <dynamics> block: friction and damping are applied by the environment
        # from the parameter vector xi, never baked into the asset.
        prev, prev_joint = name, ("J%d" % (idx + 1))

    xml = minidom.parseString(ET.tostring(robot)).toprettyxml(indent="  ")
    xml = "\n".join(line for line in xml.split("\n") if line.strip())
    with open(OUT, "w", encoding="utf-8") as fh:
        fh.write(xml)
    print("[out]", OUT)

    print("\n[urdf] link frames (local Z along link, local Y = revolute axis):")
    for name in ["link1", "link2", "link3"]:
        e = B[name]
        L = e.get("length", 2.0 * e["l_com"])
        I_local = rot[name] @ np.array(
            [[e["inertia_com"]["ixx"], e["inertia_com"]["ixy"], e["inertia_com"]["ixz"]],
             [e["inertia_com"]["ixy"], e["inertia_com"]["iyy"], e["inertia_com"]["iyz"]],
             [e["inertia_com"]["ixz"], e["inertia_com"]["iyz"], e["inertia_com"]["izz"]]]
        ) @ rot[name].T
        print("  %-6s m=%.5f L=%.5f lc=%.5f  I_yy(about revolute)=%.6e"
              % (name, e["mass"], L, e["l_com"], I_local[1, 1]))


if __name__ == "__main__":
    main()
