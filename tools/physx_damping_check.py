"""Hypothesis: Isaac balances policies the CPU model calls unstable because PhysX
applies default per-link velocity damping that the analytical model omits.

PhysX damps every articulation link's COM linear velocity and angular velocity
(defaults believed to be 0.05 each; not set anywhere in this project's asset).
Equivalent generalised force: sum_i c_lin m_i J_i^T J_i qd + c_ang I_i w_i.
This adds it to the CPU loop and asks, for CORR_s1..s3: local rho at the
upright, near-upright balance, and full swing-up success from hanging.
CPU only.
"""
import os, sys
from concurrent.futures import ProcessPoolExecutor
os.environ.setdefault("OMP_NUM_THREADS", "1")
import numpy as np
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT); sys.path.insert(0, os.path.join(ROOT, "tools"))

VARIANTS = [(0.0, 0.0), (0.0, 0.05), (0.05, 0.05)]      # (c_lin, c_ang)
TAGS = ["CORR_s1", "CORR_s2", "CORR_s3"]


def patch(c_lin, c_ang):
    import dynamics.closed_loop as CL
    from dynamics.analytical.triple_pendulum import PendulumParams
    p = PendulumParams.from_yaml()
    lk = p.links
    orig = CL.friction_forces if not hasattr(CL, "_orig_ff") else CL._orig_ff
    CL._orig_ff = orig

    def ff(fc, q, qd):
        f = orig(fc, q, qd)
        if c_lin == 0.0 and c_ang == 0.0:
            return f
        th = q[1:4]
        D = np.zeros((4, 4))
        D[0, 0] += c_lin * p.m_cart
        for i in range(3):
            J = np.zeros((2, 4))
            J[0, 0] = 1.0
            for j in range(i):
                J[0, 1 + j] = lk[j].L * np.cos(th[j]); J[1, 1 + j] = -lk[j].L * np.sin(th[j])
            J[0, 1 + i] = lk[i].lc * np.cos(th[i]); J[1, 1 + i] = -lk[i].lc * np.sin(th[i])
            D += c_lin * lk[i].m * J.T @ J
            D[1 + i, 1 + i] += c_ang * lk[i].I
        return f + D @ qd
    CL.friction_forces = ff


def job(args):
    c_lin, c_ang, tag, kind, k = args
    patch(c_lin, c_ang)
    import dr_certificate as D
    import pi_prove as PP
    from dynamics.closed_loop import NZ, ClosedLoop
    from dynamics.policy import Actor
    if kind == "rho":
        D.POLICY = tag
        K, _ = D.policy_gain()
        _, A, B = D.plant(np.zeros(3))
        return args, D.rho(A, B, K)
    if kind == "swingup":
        return args, PP.episode((tag, np.zeros(6), k, False))["success"]
    loop = ClosedLoop(Actor(PP.ckpt(tag)), PP.cfg_of(np.zeros(6)))
    rng = np.random.default_rng(900 + k)
    z = np.zeros(NZ); z[1:4] = rng.uniform(-0.03, 0.03, 3); z[5:8] = rng.uniform(-0.05, 0.05, 3)
    held = 0
    for s in range(3000):
        z = loop.step(z)
        if abs(z[0]) > 0.6:
            return args, False
        if s >= 2250:
            held += float(PP.L @ np.cos(z[1:4]) / PP.L.sum()) > 0.9
    return args, held / 750 > 0.95


if __name__ == "__main__":
    jobs = []
    for cl, ca in VARIANTS:
        for t in TAGS:
            jobs.append((cl, ca, t, "rho", 0))
            jobs += [(cl, ca, t, "balance", k) for k in range(4)]
            jobs += [(cl, ca, t, "swingup", k) for k in range(8)]
    with ProcessPoolExecutor(14) as ex:
        res = dict(ex.map(job, jobs, chunksize=1))
    for cl, ca in VARIANTS:
        print("link damping lin %.2f ang %.2f" % (cl, ca))
        for t in TAGS:
            r = res[(cl, ca, t, "rho", 0)]
            b = np.mean([res[(cl, ca, t, "balance", k)] for k in range(4)]) * 100
            s = np.mean([res[(cl, ca, t, "swingup", k)] for k in range(8)]) * 100
            print("  %s  rho %.5f  near-upright balance %5.1f%%  swing-up from hang %5.1f%%" % (t, r, b, s))
