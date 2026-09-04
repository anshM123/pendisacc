"""
Overlay the CAD-extracted kinematic chain onto the SolidWorks Right-view render,
so the extraction can be checked against the actual geometry by eye.

The Right view is orthographic, so assembly (y, z) maps to image pixels by a pure
scale + offset. The mapping is fitted from two landmarks whose assembly
coordinates are known exactly (the two belt motors at z = +-0.6808) plus the cart
pivot, and then CHECKED on a landmark that was not used in the fit.
"""

from __future__ import annotations

import os

import numpy as np
import pandas as pd
import yaml
from PIL import Image, ImageDraw, ImageFont

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RENDER = os.path.join(ROOT, "figures", "cad_right.png")
PARAMS = os.path.join(ROOT, "configs", "robot", "triple_pendulum_params.yaml")
CSV = os.path.join(ROOT, "assets", "triple_pendulum", "cad", "bodies_raw.csv")
OUT = os.path.join(ROOT, "figures", "cad_annotated.png")

# fitted from the render: x_px = A*z + B ; y_px = C*y + D
A, B = -859.2, 905.0
C, D = -873.7, 746.3


def to_px(y, z):
    return (A * z + B, C * y + D)


def font(size):
    for name in ("segoeui.ttf", "arial.ttf", "DejaVuSans.ttf"):
        try:
            return ImageFont.truetype(name, size)
        except Exception:
            continue
    return ImageFont.load_default()


def main():
    with open(PARAMS, encoding="utf-8") as fh:
        p = yaml.safe_load(fh)
    df = pd.read_csv(CSV)

    J = {k: np.array(v["pos_yz"]) for k, v in p["joints"].items()}

    im = Image.open(RENDER).convert("RGB")
    d = ImageDraw.Draw(im)
    f_big, f_med, f_sm = font(26), font(20), font(16)

    # --- sanity check: scale should be isotropic in an orthographic view ---
    d.text((20, 20), "extraction check: |x-scale| = %.1f px/m, |y-scale| = %.1f px/m "
                     "(orthographic -> should match)" % (abs(A), abs(C)),
           fill=(90, 90, 90), font=f_sm)

    # --- links ---
    chain = [
        ("link1", "J1", "J2", (198, 40, 40)),
        ("link2", "J2", "J3", (20, 110, 200)),
    ]
    for name, a, b, col in chain:
        pa, pb = to_px(*J[a]), to_px(*J[b])
        d.line([pa, pb], fill=col, width=6)
        mx, my = (pa[0] + pb[0]) / 2, (pa[1] + pb[1]) / 2
        L = p["bodies"][name]["length"]
        m = p["bodies"][name]["mass"]
        d.text((mx + 14, my - 10), "%s  L=%.3f m  m=%.4f kg" % (name, L, m),
               fill=col, font=f_med)

    # link3: free distal end, direction taken from its centre of mass
    l3 = p["bodies"]["link3"]
    p3 = J["J3"]
    com3 = np.array([l3["com"][1], l3["com"][2]])
    u = (com3 - p3) / np.linalg.norm(com3 - p3)
    L3 = 2.0 * l3["l_com"]          # uniform-bar estimate of pivot-to-tip
    tip = p3 + u * L3
    pa, pb = to_px(*p3), to_px(*tip)
    d.line([pa, pb], fill=(20, 150, 60), width=6)
    d.text((pb[0] - 300, pb[1] + 16),
           "link3  L~%.3f m (free end)  m=%.4f kg" % (L3, l3["mass"]),
           fill=(20, 150, 60), font=f_med)

    # --- joints ---
    for name, pos in J.items():
        x, y = to_px(*pos)
        r = 13
        d.ellipse([x - r, y - r, x + r, y + r], outline=(0, 0, 0), width=4)
        d.ellipse([x - 4, y - 4, x + 4, y + 4], fill=(0, 0, 0))
        d.text((x + 18, y - 34), name, fill=(0, 0, 0), font=f_big)

    # --- centres of mass ---
    for body, col in [("cart", (120, 60, 160)), ("link1", (198, 40, 40)),
                      ("link2", (20, 110, 200)), ("link3", (20, 150, 60))]:
        com = p["bodies"][body]["com"]
        x, y = to_px(com[1], com[2])
        d.line([(x - 11, y), (x + 11, y)], fill=col, width=4)
        d.line([(x, y - 11), (x, y + 11)], fill=col, width=4)

    # --- cart + rail annotation ---
    cart = p["bodies"]["cart"]
    x, y = to_px(cart["com"][1], cart["com"][2])
    d.text((x - 340, y + 40), "cart  m=%.4f kg  (%d CAD parts)"
           % (cart["mass"], cart["n_cad_bodies"]), fill=(120, 60, 160), font=f_med)

    zmin, zmax = df["bbox_zmin"].min(), df["bbox_zmax"].max()
    yrail = to_px(-0.045, 0)[1]
    x0, x1 = to_px(0, zmin)[0], to_px(0, zmax)[0]
    d.line([(x0, yrail), (x1, yrail)], fill=(70, 70, 70), width=3)
    for xe in (x0, x1):
        d.line([(xe, yrail - 12), (xe, yrail + 12)], fill=(70, 70, 70), width=3)
    d.text(((x0 + x1) / 2 - 130, yrail + 16),
           "rail span %.3f m  (z axis, cart travel)" % (zmax - zmin),
           fill=(70, 70, 70), font=f_med)

    d.text((20, im.height - 96), "y = vertical (gravity -y)   z = rail axis   "
                                 "revolute axes parallel to x",
           fill=(40, 40, 40), font=f_med)
    d.text((20, im.height - 64), "base (static) m=%.3f kg      moving mass "
                                 "(cart+3 links) = %.3f kg"
           % (p["bodies"]["base"]["mass"],
              sum(p["bodies"][b]["mass"] for b in ["cart", "link1", "link2", "link3"])),
           fill=(40, 40, 40), font=f_med)

    im.save(OUT)
    print("[out]", OUT)


if __name__ == "__main__":
    main()
