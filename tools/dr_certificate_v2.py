"""DR feasibility certificate, v2: physically realisable, over actuator uncertainty too.

v1 (tools/dr_certificate.py) certified spectral radius < 1 only. Its gains
saturated the action 80-95% of the time and ran the cart into the rail within
2 s in the nonlinear loop (tools/cert_diag.py), so v1 certified nothing
physical. v2 asks for one static gain K on the policy's observation such that,
for EVERY plant in the set and every initial perturbation in the IC set, the
LINEAR closed loop

  * is stable with margin (rho <= RHO_MAX),
  * never commands |a| > 1 (so the clip never engages and the linearisation
    stays valid), and
  * keeps the cart within |x| <= X_MAX.

Plant set = arm-mass box {-w, 0, +w}^3 in log(m_i) x three drive variants
(nominal; 8 ms delay with tau x1.3; tau x0.7). Found K are then run in the
NONLINEAR CPU loop on the same set. A K found is a certificate for the linear
problem on the sampled set; not finding one is evidence, not proof.

  run.cmd tools/dr_certificate_v2.py
"""
import itertools, json, os, sys
from concurrent.futures import ProcessPoolExecutor
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
import numpy as np
from scipy.optimize import minimize

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT); sys.path.insert(0, os.path.join(ROOT, "tools"))
from dynamics.closed_loop import NZ, ClosedLoop  # noqa: E402
import dr_certificate as D  # noqa: E402
import cert_validate as V  # noqa: E402
from pi_prove import cfg_of, L  # noqa: E402

WIDTHS = (0.125, 0.25, 0.375, 0.5)
VARIANTS = ((0, 1.0), (2, 1.3), (0, 0.7))        # (delay steps, tau factor)
RHO_MAX, X_MAX, HORIZON = 0.998, 0.40, 750
YI = D.YIDX
ICS = []
for k in range(4):
    r = np.random.default_rng(900 + k)
    e = np.zeros(NZ); e[1:4] = r.uniform(-0.03, 0.03, 3); e[5:8] = r.uniform(-0.05, 0.05, 3)
    ICS.append(e)
ICS = np.array(ICS)


def make_cfg(g3, var):
    d = np.zeros(6); d[0:3] = g3
    cfg = cfg_of(d)
    cfg.drive.delay_steps = var[0]
    cfg.drive.tau *= var[1]
    return cfg


def lin(job):
    g3, var = job
    cfg = make_cfg(g3, var)
    eps = 1e-6; z0 = np.zeros(NZ); lp = ClosedLoop(D.Const(0.0), cfg)
    A = np.empty((NZ, NZ))
    for i in range(NZ):
        dz = np.zeros(NZ); dz[i] = eps
        A[:, i] = (lp.step(z0 + dz) - lp.step(z0 - dz)) / (2 * eps)
    B = (ClosedLoop(D.Const(eps), cfg, model=lp.model).step(z0)
         - ClosedLoop(D.Const(-eps), cfg, model=lp.model).step(z0)) / (2 * eps)
    return (tuple(g3), var), A, B


def terms(K, As, Bs):
    Acl = As + Bs[:, :, None] * K[None, None, :] @ D.C          # (P, NZ, NZ)
    rho = np.max(np.abs(np.linalg.eigvals(Acl)), axis=1)
    X = np.broadcast_to(ICS, (len(As),) + ICS.shape).copy()    # (P, I, NZ)
    AT = np.transpose(Acl, (0, 2, 1))
    pa = px = 0.0
    for _ in range(HORIZON):
        X = X @ AT
        pa = max(pa, float(np.abs(X[:, :, YI] @ K).max()))
        px = max(px, float(np.abs(X[:, :, 0]).max()))
        if not np.isfinite(pa) or pa > 1e6:
            return float(rho.max()), 1e6, 1e6
    return float(rho.max()), pa, px


def objective(K, As, Bs):
    r, pa, px = terms(K, As, Bs)
    return max(r - RHO_MAX, 0) * 100 + max(pa - 1.0, 0) + max(px - X_MAX, 0) * 2.5 + 1e-3 * r


def search(args):
    K0, As, Bs, seed = args
    rng = np.random.default_rng(seed)
    scale = np.maximum(np.abs(K0), 0.05)
    x0 = K0 / scale * (1.0 if seed == 0 else rng.uniform(0.3, 1.2) * (1 + 0.3 * rng.standard_normal(len(K0))))
    f = lambda x: objective(x * scale, As, Bs)  # noqa: E731
    r = minimize(f, x0, method="Nelder-Mead", options={"maxfev": 3000, "adaptive": True, "fatol": 1e-9})
    r = minimize(f, r.x, method="Powell", options={"maxfev": 3000, "ftol": 1e-10})
    K = r.x * scale
    return float(f(r.x)), K.tolist(), terms(K, As, Bs)


def nl(job):
    K, g3, var, k = job
    loop = ClosedLoop(V.Linear(K), make_cfg(g3, var))
    z = ICS[k % len(ICS)].copy(); held = 0
    for s in range(V.STEPS):
        z = loop.step(z)
        if abs(z[0]) > V.BOUND or not np.all(np.isfinite(z)):
            return False
        if s >= 3 * V.STEPS // 4:
            held += float(L @ np.cos(z[1:4]) / L.sum()) > 0.9
    return held / (V.STEPS // 4) > 0.95


if __name__ == "__main__":
    K_pol, _ = D.policy_gain()
    keys = set()
    for w in WIDTHS:
        for p in itertools.product((-w, 0.0, w), repeat=3):
            for var in VARIANTS:
                keys.add((tuple(float(v) for v in p), var))
    keys = sorted(keys)
    with ProcessPoolExecutor(20) as pool:
        L_ = dict((k, (A, B)) for k, A, B in pool.map(lin, keys, chunksize=4))
        print("linearised %d plants" % len(L_), flush=True)
        out = {"rho_max": RHO_MAX, "x_max": X_MAX, "variants": VARIANTS, "boxes": []}
        Kprev = K_pol
        for w in WIDTHS:
            sel = [k for k in keys if max(abs(v) for v in k[0]) in (0.0, w) and
                   all(abs(v) in (0.0, w) for v in k[0])]
            As = np.array([L_[k][0] for k in sel]); Bs = np.array([L_[k][1] for k in sel])
            pol = terms(K_pol, As, Bs)
            starts = [(K_pol, As, Bs, s) for s in range(8)] + [(np.array(Kprev), As, Bs, 0)]
            res = sorted(pool.map(search, starts), key=lambda r: r[0])
            val, K, (r, pa, px) = res[0]
            Kprev = K
            feas = r <= RHO_MAX and pa <= 1.0 and px <= X_MAX
            jobs = [(K, k[0], k[1], i) for k in sel for i in range(2)]
            ok = 100 * float(np.mean(list(pool.map(nl, jobs, chunksize=2))))
            rec = {"half_width_log": w, "range": [float(np.exp(-w)), float(np.exp(w))], "n_plants": len(sel),
                   "K": K, "rho": r, "peak_action": pa, "peak_cart_m": px, "certified": bool(feas),
                   "nonlinear_hold_pct": ok, "policy_terms": {"rho": pol[0], "peak_action": pol[1], "peak_cart_m": pol[2]}}
            out["boxes"].append(rec)
            print("box x%.2f..x%.2f (%d plants incl. delay/lag): rho %.4f  peak|a| %.2f  peak|x| %.2f m -> %s | "
                  "nonlinear hold %.1f%% | trained policy: rho %.4f peak|a| %.2f peak|x| %.2f"
                  % (np.exp(-w), np.exp(w), len(sel), r, pa, px, "CERTIFIED" if feas else "not certified",
                     ok, pol[0], pol[1], pol[2]), flush=True)
    with open(os.path.join(ROOT, "results", "dr_certificate", "cert_v2.json"), "w", encoding="utf-8") as fh:
        json.dump(out, fh, indent=1)
    print("wrote results/dr_certificate/cert_v2.json")
