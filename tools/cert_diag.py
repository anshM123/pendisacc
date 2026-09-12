"""Why does the certified gain fail where it does? Failure mode, saturation, linear rho."""
import itertools, os, sys
from collections import Counter
from concurrent.futures import ProcessPoolExecutor
os.environ.setdefault("OMP_NUM_THREADS", "1")
import numpy as np
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT); sys.path.insert(0, os.path.join(ROOT, "tools"))
import cert_validate as V
import dr_certificate as D
from dynamics.closed_loop import NZ, ClosedLoop
from pi_prove import cfg_of, L

K = np.array(V.BOX["K"]); W = V.W

def diag(job):
    g3, k, stress = job
    d = np.zeros(6); d[0:3] = g3
    cfg = cfg_of(d)
    if stress:
        cfg.drive.delay_steps = 2; cfg.drive.tau *= 1.3
    loop = ClosedLoop(V.Linear(K), cfg)
    rng = np.random.default_rng(900 + k)
    z = np.zeros(NZ); z[1:4] = rng.uniform(-0.03, 0.03, 3); z[5:8] = rng.uniform(-0.05, 0.05, 3)
    sat = 0; maxx = 0.0; mode = "ok"; held = 0; s = 0
    for s in range(V.STEPS):
        z = loop.step(z)
        sat += abs(z[10]) >= 0.999
        maxx = max(maxx, abs(z[0]))
        if abs(z[0]) > V.BOUND: mode = "cart_bound@%.1fs" % (s / 250); break
        if not np.all(np.isfinite(z)): mode = "nan"; break
        if s >= 3 * V.STEPS // 4: held += float(L @ np.cos(z[1:4]) / L.sum()) > 0.9
    if mode == "ok" and held / (V.STEPS // 4) <= 0.95: mode = "fell"
    eps = 1e-6; z0 = np.zeros(NZ); lp = ClosedLoop(D.Const(0.0), cfg)
    A = np.empty((NZ, NZ))
    for i in range(NZ):
        dz = np.zeros(NZ); dz[i] = eps
        A[:, i] = (lp.step(z0 + dz) - lp.step(z0 - dz)) / (2 * eps)
    B = (ClosedLoop(D.Const(eps), cfg, model=lp.model).step(z0)
         - ClosedLoop(D.Const(-eps), cfg, model=lp.model).step(z0)) / (2 * eps)
    return g3, k, stress, mode, sat / (s + 1), maxx, D.rho(A, B, K)

if __name__ == "__main__":
    pts = list(itertools.product((-W, 0.0, W), repeat=3))
    jobs = [(p, k, st) for st in (False, True) for p in pts for k in range(2)]
    with ProcessPoolExecutor(20) as ex:
        res = list(ex.map(diag, jobs, chunksize=2))
    for st in (False, True):
        R = [r for r in res if r[2] == st]
        print("STRESS (8ms, tau x1.3)" if st else "AS CERTIFIED", Counter(r[3].split("@")[0] for r in R))
        print("  linear rho max %.4f | failing with rho<1: %d | ok with rho>=1: %d" % (
            max(r[6] for r in R), sum(1 for r in R if r[3] != "ok" and r[6] < 1),
            sum(1 for r in R if r[3] == "ok" and r[6] >= 1)))
        print("  mean saturation: ok %.3f  failed %.3f" % (
            np.mean([r[4] for r in R if r[3] == "ok"] or [0]), np.mean([r[4] for r in R if r[3] != "ok"] or [0])))
        for r in R:
            if r[3] != "ok" and r[1] == 0:
                print("   fail x%-18s %-16s sat %.2f max|x| %.2f rho %.4f" % (
                    np.round(np.exp(r[0]), 2), r[3], r[4], r[5], r[6]))
