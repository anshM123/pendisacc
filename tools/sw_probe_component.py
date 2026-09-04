"""Find which Component2 accessor is throwing DISP_E_MEMBERNOTFOUND."""

import win32com.client as win32

ASSEMBLY = r"C:\Users\anshm\Downloads\Major Assembly\Major Assembly.SLDASM"

sw = win32.GetActiveObject("SldWorks.Application")
spec = sw.GetOpenDocSpec(ASSEMBLY)
spec.DocumentType = 2
spec.ReadOnly = True
spec.Silent = True
model = sw.OpenDoc7(spec) or sw.ActiveDoc
assy = win32.CastTo(model, "IAssemblyDoc")

comps = list(assy.GetComponents(False))
print("components:", len(comps))
comp = comps[0]
print("python class:", type(comp).__name__)

probes = [
    ("Name2", lambda c: c.Name2),
    ("Name", lambda c: c.Name),
    ("GetPathName", lambda c: c.GetPathName()),
    ("IsSuppressed", lambda c: c.IsSuppressed()),
    ("GetSuppression", lambda c: c.GetSuppression()),
    ("GetChildren", lambda c: c.GetChildren()),
    ("GetModelDoc2", lambda c: c.GetModelDoc2()),
    ("ReferencedConfiguration", lambda c: c.ReferencedConfiguration),
    ("Transform2", lambda c: c.Transform2),
    ("GetBox", lambda c: c.GetBox(False, False)),
    ("IsHidden", lambda c: c.IsHidden(True)),
    ("GetBodies3", lambda c: c.GetBodies3(0, None)),
]

for label, fn in probes:
    try:
        val = fn(comp)
        text = repr(val)
        print("  OK   %-26s %s" % (label, text[:90]))
    except Exception as exc:
        print("  FAIL %-26s %s" % (label, exc))

print("\n--- generated wrapper members containing 'Child'/'Model'/'Suppress' ---")
for attr in sorted(dir(comp)):
    if any(k in attr for k in ("Child", "Model", "Suppress", "Name", "Box", "Transform")):
        print("   ", attr)
