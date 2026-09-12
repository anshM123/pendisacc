"""D-group re-run with the v2 initial-condition set. PRE-REGISTRATION (new ids D10-D19).

Why: D00 (calibration) failed, which invalidates D01-D09 as run. Diagnosis in
theories/debug_d00.py: the screen's certificate used a fixed pair of initial
perturbations about twice the size of v2's seeded draws. At that size even the
v2 +-25% gain violates the action and rail limits (peak |a| 2.0, |x| 0.52 m),
so no optimiser budget could pass D00 (1500 and 6000 evaluations both fail).
Same claims and thresholds as D00-D09; only the IC set (v2's four seeded
draws) and v2's 750-step horizon change, with a larger optimiser budget.

  run.cmd theories/certs_v2ics.py
"""
import json, os, sys, time, traceback
from concurrent.futures import ProcessPoolExecutor, as_completed
os.environ.setdefault("OMP_NUM_THREADS", "1")
import numpy as np
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE)
import screen as S
from dynamics.closed_loop import NZ

_ics = []
for k in range(4):
    r = np.random.default_rng(900 + k)
    e = np.zeros(NZ); e[1:4] = r.uniform(-0.03, 0.03, 3); e[5:8] = r.uniform(-0.05, 0.05, 3)
    _ics.append(e)
S.ICS = np.array(_ics)
KW = dict(horizon=750, fev=3000, starts=4)

def aniso():
    widths = {}
    for ax in range(3):
        w_ok = 0.0
        for w in (0.25, 0.5, 0.75, 1.0):
            ok, _ = S.cert_at(w, axes=(ax,), **KW)
            if not ok:
                break
            w_ok = w
        widths[ax] = w_ok
    return int(widths[1] < widths[0] and widths[1] < widths[2]), widths

TASKS = {
    "D10": ("calibration", "CALIBRATION: reproduces v2 (+-25% certified)", lambda: S.cert_at(0.25, **KW)),
    "D11": ("D", "Rail is binding: 0.55 m cart travel certifies +-37%", lambda: S.cert_at(0.375, x_max=0.55, **KW)),
    "D12": ("D", "Motor is binding: 1.5x drive certifies +-37%", lambda: S.cert_at(0.375, a_max=1.5, **KW)),
    "D13": ("D", "Actuator uncertainty is binding: nominal drive only certifies +-50%", lambda: S.cert_at(0.5, variants=((0, 1.0),), **KW)),
    "D14": ("D", "Anisotropy: link 2 has the smallest certifiable one-axis range", aniso),
    "D15": ("D", "Measurement priority: m2 known exactly certifies +-50% on m1, m3", lambda: S.cert_at(0.5, axes=(0, 2), **KW)),
    "D16": ("D", "Hardware lever: 50 ms servo certifies +-50%", lambda: S.cert_at(0.5, tau0=0.05, **KW)),
    "D17": ("D", "Memory: one step of observation history certifies +-37%", lambda: S.cert_at(0.375, memory=True, **KW)),
    "D18": ("D", "100 Hz controller certifies +-12%", lambda: S.cert_at(0.125, dt=1 / 100, **KW)),
    "D19": ("D", "Nominal masses: one gain tolerates 0-16 ms delay", lambda: S.cert_at(0.0, variants=((0, 1.0), (2, 1.0), (4, 1.0)), **KW)),
}

def run(tid):
    g, claim, fn = TASKS[tid]
    t0 = time.time()
    try:
        ok, det = fn()
        v = float(ok)
        return {"id": tid, "group": g, "claim": claim, "value": v, "passed": v >= 1, "detail": det, "seconds": round(time.time() - t0)}
    except Exception:
        return {"id": tid, "group": g, "claim": claim, "value": None, "passed": False, "error": traceback.format_exc()[-800:], "seconds": round(time.time() - t0)}

if __name__ == "__main__":
    S.E.basis("corrected")
    out = []
    with ProcessPoolExecutor(10) as ex:
        for f in as_completed([ex.submit(run, t) for t in TASKS]):
            r = f.result(); out.append(r)
            print("%-4s %-11s %s  %4ss  %s  %s" % (r["id"], r["group"], "PASS" if r["passed"] else ("ERR " if r.get("error") else "fail"),
                                               r["seconds"], r["claim"], json.dumps(r.get("detail", r.get("error")), default=str)[:160]), flush=True)
    json.dump(out, open(os.path.join(S.OUT, "certs_v2ics.json"), "w", encoding="utf-8"), indent=1, default=str)
    print("wrote results/theories/certs_v2ics.json")
