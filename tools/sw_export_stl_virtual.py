"""
Second pass: export the parts that cannot be opened standalone.

Virtual components (names containing '^') and imported .step parts live inside
the parent assembly, so OpenDoc7 on their path fails. But once the assembly is
open they ARE loaded documents, so enumerate the open documents with
GetDocuments() and save each one from there.

GetDocuments() is used rather than walking GetFirstDocument/GetNext because the
latter hands back late-bound objects whose GetNext raises "Member not found".
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
ASSEMBLY = r"C:\Users\anshm\Downloads\Major Assembly\Major Assembly.SLDASM"

MOVING = {"cart", "link1", "link2", "link3"}
MIN_MASS = 1e-4


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


def safe(name: str) -> str:
    return re.sub(r"[^A-Za-z0-9_.-]", "_", name)[:90]


def out_path(part_path: str) -> str:
    return os.path.join(OUTDIR, safe(os.path.basename(part_path)).replace(".SLDPRT", "") + ".stl")


def main() -> int:
    import yaml

    df = pd.read_csv(CSV)
    with open(GROUPING, encoding="utf-8") as fh:
        cfg = yaml.safe_load(fh)

    def body_of(name):
        for body, rules in cfg["bodies"].items():
            for p in rules.get("prefix", []) or []:
                if name.startswith(p):
                    return body
            for e in rules.get("exact", []) or []:
                if name == e:
                    return body
        return None

    df["body"] = [body_of(n) for n in df["name"]]
    want = df[df["body"].isin(MOVING) & (df["mass"] >= MIN_MASS)]
    missing = {p for p in set(want["part_path"]) if not os.path.exists(out_path(p))}
    print("[stl2] %d parts still missing" % len(missing))
    if not missing:
        return 0

    sw = win32.GetActiveObject("SldWorks.Application")
    sw.Visible = True

    # make sure the assembly (and therefore its virtual components) is loaded
    spec = sw.GetOpenDocSpec(ASSEMBLY)
    spec.DocumentType = 2
    spec.ReadOnly = True
    spec.Silent = True
    sw.OpenDoc7(spec)

    docs = value(sw, "GetDocuments") or []
    print("[stl2] %d documents open" % len(docs))

    # Virtual components live under ...\Temp\swxNNNNN\VC~~\ and imported ones
    # under IC~~\, so their DOCUMENT path never equals the COMPONENT path the
    # assembly reports. Match on the file name instead, which is stable.
    by_name = {}
    for d in docs:
        try:
            title = value(d, "GetTitle") or ""
            path = value(d, "GetPathName") or ""
            for key in {os.path.basename(path).lower(), title.lower()}:
                if key.endswith(".sldprt"):
                    by_name.setdefault(key, d)
        except Exception:
            continue

    written, still = 0, []
    for p in sorted(missing):
        doc = by_name.get(os.path.basename(p).lower())
        if doc is None:
            still.append((p, "not among open documents"))
            continue
        out = out_path(p)
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
            print("   ok   %s" % os.path.basename(p)[:60])
        else:
            still.append((p, "save failed"))

    print("[stl2] wrote %d, still missing %d" % (written, len(still)))
    for p, why in still:
        print("   MISS %-58s %s" % (os.path.basename(p)[:58], why))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
