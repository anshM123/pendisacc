"""Are the CPU-vs-Isaac disagreements (CORR_s2, s3 at 0% on CPU) swing-up only?

Runs each frozen policy from a near-upright start on the nominal CPU plant and
reports whether it balances, and its local spectral radius.
"""
import os, sys
from concurrent.futures import ProcessPoolExecutor
os.environ.setdefault("OMP_NUM_THREADS", "1")
import numpy as np
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT); sys.path.insert(0, os.path.join(ROOT, "tools"))
import dr_certificate as D
from dynamics.closed_loop import NZ, ClosedLoop
from dynamics.policy import Actor
from pi_prove import cfg_of, ckpt, L

def run(job):
    tag, k = job
    act = Actor(ckpt(tag)); loop = ClosedLoop(act, cfg_of(np.zeros(6)))
    rng = np.random.default_rng(900 + k)
    z = np.zeros(NZ); z[1:4] = rng.uniform(-0.03, 0.03, 3); z[5:8] = rng.uniform(-0.05, 0.05, 3)
    held = 0; mode = "ok"
    for s in range(3000):
        z = loop.step(z)
        if abs(z[0]) > 0.6: mode = "cart_bound@%.1fs" % (s / 250); break
        if s >= 2250: held += float(L @ np.cos(z[1:4]) / L.sum()) > 0.9
    if mode == "ok" and held / 750 <= 0.95: mode = "fell"
    return tag, k, mode

def gain_rho(tag):
    D.POLICY = tag
    K, a0 = D.policy_gain()
    _, A, B = D.plant(np.zeros(3))
    return tag, D.rho(A, B, K), a0

if __name__ == "__main__":
    tags = ["CORR_s1", "CORR_s2", "CORR_s3"]
    with ProcessPoolExecutor(12) as ex:
        res = list(ex.map(run, [(t, k) for t in tags for k in range(4)]))
        rh = list(ex.map(gain_rho, tags))
    for t in tags:
        modes = [m for tt, _, m in res if tt == t]
        r = [x for x in rh if x[0] == t][0]
        print("%s  near-upright balance: %s | local rho %.5f  a(upright) %+.3f" % (t, modes, r[1], r[2]))
