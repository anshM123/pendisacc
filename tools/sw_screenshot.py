"""
Save SolidWorks viewport renders of the assembly from the views that matter for
this project.

The links swing in the y-z plane, so the RIGHT view (looking down -x) is the one
that shows the kinematic chain undistorted. Isometric is included for part
recognition.
"""

from __future__ import annotations

import os

import win32com.client as win32

ASSEMBLY = r"C:\Users\anshm\Downloads\Major Assembly\Major Assembly.SLDASM"
OUTDIR = r"C:\Users\anshm\Downloads\pendisaac\figures"

# swStandardViews_e
VIEWS = {
    "right": ("*Right", 4),
    "front": ("*Front", 1),
    "iso": ("*Isometric", 7),
}


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


def main():
    os.makedirs(OUTDIR, exist_ok=True)
    sw = win32.GetActiveObject("SldWorks.Application")
    sw.Visible = True

    spec = sw.GetOpenDocSpec(ASSEMBLY)
    spec.DocumentType = 2
    spec.ReadOnly = True
    spec.Silent = True
    model = sw.OpenDoc7(spec) or sw.ActiveDoc
    print("[sw] document:", value(model, "GetTitle"))

    # a large viewport gives a higher-resolution save
    try:
        sw.FrameState = 1  # swWindowMaximized
    except Exception:
        pass

    for label, (view_name, view_id) in VIEWS.items():
        ok = False
        for call in (
            lambda: model.ShowNamedView2(view_name, view_id),
            lambda: model.ShowNamedView2("", view_id),
        ):
            try:
                call()
                ok = True
                break
            except Exception:
                continue
        try:
            model.ViewZoomtofit2()
        except Exception:
            pass
        try:
            model.GraphicsRedraw2()
        except Exception:
            pass

        out = os.path.join(OUTDIR, "cad_%s.png" % label)
        saved = False
        for attempt in (
            lambda: model.SaveAs4(out, 0, 1, 0, 0, 0),
            lambda: model.SaveAs3(out, 0, 1),
            lambda: model.SaveAs2(out, 0, True, False),
        ):
            try:
                attempt()
                saved = os.path.exists(out)
                if saved:
                    break
            except Exception:
                continue
        print("  %-6s view=%s  saved=%s  %s" % (label, ok, saved, out))


if __name__ == "__main__":
    main()
