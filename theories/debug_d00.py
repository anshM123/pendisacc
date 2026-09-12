"""Why did the light-budget certificate fail its calibration (D00)?"""
import os, sys, time
os.environ.setdefault("OMP_NUM_THREADS", "1")
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import screen as S
import engine as E

K0 = S.K_of(E.checkpoints("CORR_s1")[-1])
fam, full = S.family(0.25), S.family(0.25, full=True)
print("corner plants", len(fam), "full plants", len(full))
As = np.array([p[0] for p in fam]); Bs = np.array([p[1] for p in fam])
t = time.time(); r = S.lin_terms(K0, As, Bs, S.CMAT, S.ICS, 300); print("terms at policy gain", r, "%.3fs/eval" % (time.time() - t))
import json
v2 = json.load(open(os.path.join(S.ROOT, "results", "dr_certificate", "cert_v2.json")))
Kv2 = np.array(v2["boxes"][1]["K"])
print("v2 +-25% gain on light family:", S.lin_terms(Kv2, As, Bs, S.CMAT, S.ICS, 300))
Af = np.array([p[0] for p in full]); Bf = np.array([p[1] for p in full])
print("v2 +-25% gain on full family:", S.lin_terms(Kv2, Af, Bf, S.CMAT, S.ICS, 300))
for fev, starts in ((1500, 3), (6000, 4)):
    t = time.time(); ok, terms = S.certify(fam, full, K0, fev=fev, starts=starts)
    print("certify fev=%d starts=%d -> %s %s in %.0fs" % (fev, starts, ok, terms, time.time() - t), flush=True)
