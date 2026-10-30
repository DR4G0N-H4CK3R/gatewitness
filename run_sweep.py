"""Matched-rate comparison: GateWitness (swept budget B) and seven baselines
(swept theta) on identical episodes.  Writes out/sweep.csv and
out/sweep_episodes.csv."""
from __future__ import annotations

import sys
import time
import zlib
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent))
import baselines as BL            # noqa: E402
import folds as FO                # noqa: E402
import gw_eval as E               # noqa: E402
import run_main as RM             # noqa: E402

OUT = Path(__file__).parent / "out"
CLASSES = ["hide", "delay", "change", "forge", "leak", "a5"]
GW_QUAL = {"hide": ["SUP[{d}]"], "a5": ["SUP[{d}]"], "delay": ["SUP[{d}]"], "change": ["SIZE[{d}]"],
           "forge": ["EMIT[{d}]"], "leak": ["NEWEP", "VOL[*"]}


def scored(al, cls, d, t0, t1, device_level_prefixes):
    """Device-exact for device-level alarms; gateway-level alarms count for any class."""
    keys = []
    for k in al:
        if "[" in k:
            if cls != "leak" and k.endswith(f"[{d}]"):
                keys.append(k)
        else:
            keys.append(k)
    for k in keys:
        a = al[k]
        if len(a) and ((a >= t0) & (a <= t1 + E.TAIL)).any():
            return 1
    return 0


def main():
    F = FO.folds()
    rows, eprows = [], []
    for f in F:
        t_f = time.time()
        comm, held = f["comm"], f["held"]
        M = E.commission(comm)
        top = comm.uplinks.dev.value_counts().index[:2].tolist()
        devs = [int(k[4:-1]) for k in M.tests if k.startswith("SUP[")]
        dets = {cls.name: cls().fit(comm, devs) for cls in BL.ALL}
        MB = {B: E.recalibrate(M, B) for B in RM.FP_B}
        exp = E.exposure_days(held)
        held_log = BL.label_units(held)
        # benign
        clean = {}
        for B, m in MB.items():
            al = RM.CORE(E.alarm_times(m, held, "bern"))
            clean[("GateWitness", B)] = al
            rows.append(dict(dataset="ziotp", home=f["home"], fold=f["fold"], detector="GateWitness",
                             threshold=B, kind="benign", benign_alarms=E.count_alarms(al), exposure_days=exp))
        for name, det in dets.items():
            for th in BL.THETAS:
                al = det.alarms(held, th, held_log)
                clean[(name, th)] = al
                rows.append(dict(dataset="ziotp", home=f["home"], fold=f["fold"], detector=name,
                                 threshold=th, kind="benign", benign_alarms=E.count_alarms(al), exposure_days=exp))
        # episodes
        for cls in CLASSES:
            for di, d in enumerate(top):
                for seed in range(5):
                    rng = np.random.default_rng(zlib.crc32(repr(("sweep", f["home"], f["fold"], cls, int(d), seed)).encode()))
                    t0, t1 = RM.episode_windows(held, rng, 1)[0]
                    log = None
                    if cls in ("hide", "a5"):
                        att, nd, nc = E.inj_hide(held, d, 0.8, t0, t1, rng)
                        log = BL.label_units(att)
                        if cls == "a5":
                            hidden = held_log[(held_log.dev_true == d) & ~held_log.ts.isin(att.units.ts)
                                              & (held_log.ts >= t0) & (held_log.ts <= t1)]
                            rep = hidden.copy(); rep["ts"] = rep.ts + rng.uniform(-20, 20, len(rep))
                            log = pd.concat([log, rep], ignore_index=True).sort_values("ts")
                    else:
                        att = RM.attack(cls, held, d, top[1 - di], t0, t1, rng, M)
                        lu = BL.label_units(att) if cls != "leak" else held_log
                        log = lu
                    for B, m in MB.items():
                        quals = [q.format(d=d) for q in GW_QUAL[cls]]
                        al = E.alarm_times(m, att, "bern", only=set(quals))
                        det_ = RM.hit(al, quals, t0, t1)[0]
                        plc = RM.hit(clean[("GateWitness", B)], quals, t0, t1)[0]
                        eprows.append(dict(home=f["home"], fold=f["fold"], detector="GateWitness", threshold=B,
                                           attack=cls, device=d, seed=seed, det=int(det_), placebo=int(plc)))
                    for name, det in dets.items():
                        for th in BL.THETAS:
                            al = det.alarms(att, th, log)
                            eprows.append(dict(home=f["home"], fold=f["fold"], detector=name, threshold=th,
                                               attack=cls, device=d, seed=seed,
                                               det=scored(al, cls, d, t0, t1, None),
                                               placebo=scored(clean[(name, th)], cls, d, t0, t1, None)))
        print(f"  sweep fold {f['home']}{f['fold']} {time.time() - t_f:.0f}s", flush=True)
    ben = pd.DataFrame(rows)
    ep = pd.DataFrame(eprows)
    ep.to_csv(OUT / "sweep_episodes.csv", index=False)
    b = ben.groupby(["detector", "threshold"]).agg(benign_alarms=("benign_alarms", "sum"),
                                                     exposure_days=("exposure_days", "sum")).reset_index()
    e = ep.groupby(["detector", "threshold", "attack"]).agg(det=("det", "mean"), placebo=("placebo", "mean"),
                                                             n=("det", "size")).reset_index()
    sw = e.merge(b, on=["detector", "threshold"])
    sw.insert(0, "dataset", "ziotp")
    sw.to_csv(OUT / "sweep.csv", index=False)
    print("done")


if __name__ == "__main__":
    main()
