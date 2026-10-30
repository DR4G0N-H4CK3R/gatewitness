"""Main GateWitness experiments: benign calibration, seven classes with placebo,
long-horizon suppression, predictions.  Writes out/*.csv."""
from __future__ import annotations

import math
import sys
import zlib
import time
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent))
import folds as FO                 # noqa: E402
import gw_core as C                # noqa: E402
import gw_eval as E                # noqa: E402

OUT = Path(__file__).parent / "out"
OUT.mkdir(exist_ok=True)
FP = E.FP
OPT_IN = ("LIVE", "VOLR")
CORE = lambda al: {k: v for k, v in al.items() if not k.startswith(OPT_IN)}
QUAL = {   # class -> qualifying test prefixes (device-exact where marked with {d})
    "hide": ["SUP[{d}]"], "delay": ["SUP[{d}]"], "lh08": ["SUP[{d}]"], "lh10": ["SUP[{d}]"],
    "change": ["SIZE[{d}]"], "swap": ["SIZE[{d}]", "SIZE[{d2}]"], "invent": ["FAB"],
    "leak": ["NEWEP", "VOL[*"], "forge": ["EMIT[{d}]"], "forgeX": ["EMIT[{d}]"],
}


def qualifying(cls, d, d2=None):
    return [q.format(d=d, d2=d2) for q in QUAL[cls]]


def hit(al, quals, t0, t1):
    best = None
    for q in quals:
        keys = [k for k in al if (k.startswith(q[:-1]) if q.endswith("*") else k == q)]
        for k in keys:
            a = al[k]
            a = a[(a >= t0) & (a <= t1 + E.TAIL)]
            if len(a):
                best = a.min() if best is None else min(best, a.min())
    return (best is not None), (None if best is None else best - t0)


def episode_windows(sess, rng, n, lo=180, hi=270):
    a, b = sess.segs[0][1], sess.segs[-1][2]
    out = []
    for _ in range(n):
        L = rng.uniform(lo, hi)
        t0 = rng.uniform(a + 10, b - L - 5)
        out.append((t0, t0 + L))
    return out


def attack(cls, sess, d, d2, t0, t1, rng, M):
    A = FP["attacks"]
    if cls in ("hide", "lh08"):
        return E.inj_hide(sess, d, A["hide_f"], t0, t1, rng)[0]
    if cls == "lh10":
        return E.inj_hide(sess, d, 1.0, t0, t1, rng)[0]
    if cls == "delay":
        return E.inj_delay(sess, d, A["delay_s"], t0, t1)
    if cls == "change":
        return E.inj_change(sess, d, A["change_bytes"], t0, t1)
    if cls == "swap":
        return E.inj_swap(sess, d, d2, t0, t1, rng)
    if cls == "invent":
        return E.inj_invent(sess, A["invent_rate"], t0, t1, rng)
    if cls == "leak":
        return E.inj_leak(sess, A["leak_rate"], t0, t1, rng)
    if cls == "forge":
        return E.inj_forge(sess, d, A["forge_rate"], t0, t1, rng, ack="host", ack_us=A["forge_ack_us"])[0]
    if cls == "forgeX":
        return E.inj_forge(sess, d, A["forge_rate"], t0, t1, rng, ack="exact", band=M.ack_band,
                           counter="continue")[0]
    raise KeyError(cls)


def predicted_uplinks(M, d, f):
    """Expected CUSUM delay in units under hiding fraction f (i.i.d. model)."""
    T = M.tests.get(f"SUP[{d}]")
    if T is None:
        return math.inf
    c = T["cal"]
    eps, s = c["eps"], T.get("s", 0.2)
    pf = min(eps + s * f, 0.999)
    mu = pf * c["l1"] + (1 - pf) * c["l0"]
    return c["h_bern"] / mu if mu > 0 else math.inf


def run(folds, classes, tag, seeds=5, eplen=(180, 270), whole=False):
    rows, ind_rows, benign = [], [], []
    for f in folds:
        t_fold = time.time()
        M = E.commission(f["comm"])
        Mm = {B: E.recalibrate(M, B) for B in FP_B}
        held = f["held"]
        top = (f["comm"].uplinks.dev.value_counts().index[:2]).tolist()
        # benign alarms per budget and calibration (core and opt-in)
        for B, MB in Mm.items():
            for which in ("bern", "disp", "markov"):
                al = E.alarm_times(MB, held, which)
                benign.append(dict(set=tag, home=f["home"], fold=f["fold"], B=B, calib=which,
                                   core=E.count_alarms(CORE(al)), optin=E.count_alarms(al) - E.count_alarms(CORE(al)),
                                   exposure_days=E.exposure_days(held),
                                   tests=";".join(f"{k}:{len(v)}" for k, v in al.items() if len(v))))
        # indicator export (benign streams)
        if tag == "main":
            for split, sess in (("commission", f["comm"]), ("heldout", held)):
                sup, credit, cdev = E.sup_streams(sess)
                streams = {f"SUP[{d}]": v for d, v in sup.items() if f"SUP[{d}]" in M.tests}
                streams.update({f"SIZE[{d}]": v for d, v in E.size_streams(sess, M.bands, cdev).items()})
                em = E.emit_streams(sess.frames, M.ack_band, E.coverage_series(sess.frames), segs=sess.segs)
                streams.update({f"EMIT[{a}]": v for a, v in em.items() if f"EMIT[{a}]" in M.tests})
                tf, xf = E.fab_stream(sess, credit)
                streams["FAB"] = (tf, xf)
                for tid, (t, x) in streams.items():
                    T = M.tests.get(tid)
                    de = T["cal"]["eps"] if T else np.nan
                    dp = T["cal"]["p1"] if T else np.nan
                    for ti, xi in zip(t, x):
                        ind_rows.append((f["home"], f["fold"], split, tid, "bernoulli", float(ti), int(xi), de, dp))
        clean = {B: E.alarm_times(MB, held, "bern") for B, MB in Mm.items()}
        for cls in classes:
            for di, d in enumerate(top):
                d2 = top[1 - di] if len(top) > 1 else d
                for seed in range(seeds):
                    rng = np.random.default_rng(zlib.crc32(repr((tag, f["home"], f["fold"], cls, int(d), seed)).encode()))
                    if whole:
                        a, b = held.segs[0][1], held.segs[-1][2]
                        t0, t1 = a + 5, b - 5
                    else:
                        t0, t1 = episode_windows(held, rng, 1, *eplen)[0]
                    att = attack(cls, held, d, d2, t0, t1, rng, M)
                    quals = qualifying(cls, d, d2)
                    n_up = int(((held.uplinks.dev == d) & (held.uplinks.ts >= t0) & (held.uplinks.ts <= t1)).sum())
                    for B, MB in Mm.items():
                        only = set(q for q in quals)
                        al = E.alarm_times(MB, att, "bern", only=only)
                        det, lat = hit(al, quals, t0, t1)
                        plc, _ = hit(clean[B], quals, t0, t1)
                        r = dict(experiment=cls, dataset="ziotp", set=tag, home=f["home"], fold=f["fold"],
                                 seed=seed, attack=cls, device=d, B=B, detected=int(det),
                                 placebo=int(plc), latency_s=lat, t0=t0, t1=t1, n_uplinks=n_up,
                                 pred_uplinks=predicted_uplinks(MB, d, 0.8 if cls in ("hide", "lh08") else 1.0)
                                 if cls in ("hide", "lh08", "lh10", "delay") else np.nan)
                        if B == 1.0 and cls in ("forge", "change", "forgeX"):
                            alm = E.alarm_times(MB, att, "markov", only=only)
                            r["detected_markov"] = int(hit(alm, quals, t0, t1)[0])
                        rows.append(r)
        print(f"  fold {f['home']}{f['fold']} ({tag}) done in {time.time() - t_fold:.0f}s", flush=True)
    return pd.DataFrame(rows), pd.DataFrame(ind_rows, columns="home fold split test_id kind t x design_eps design_p1".split()), pd.DataFrame(benign)


FP_B = [0.1, 1.0, 10.0, 100.0, 1000.0, 10000.0]

if __name__ == "__main__":
    which = sys.argv[1] if len(sys.argv) > 1 else "main"
    if which == "main":
        F = FO.folds()
        ep, ind, ben = run(F, ["hide", "delay", "change", "swap", "invent", "leak", "forge", "forgeX"], "main")
        ep.to_csv(OUT / "episodes_main.csv", index=False)
        ind.to_csv(OUT / "indicators.csv", index=False)
        ben.to_csv(OUT / "benign_main.csv", index=False)
    else:
        F = FO.folds(long=True)
        ep, _, ben = run(F, ["lh08", "lh10"], "long", whole=True)
        ep.to_csv(OUT / "episodes_long.csv", index=False)
        ben.to_csv(OUT / "benign_long.csv", index=False)
    print("done")
