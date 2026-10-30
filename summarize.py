"""Turn out/*.csv into the numbers the paper reports (results/summary.json and
results/RESULTS.md).  Every number is traceable to one output file."""
from __future__ import annotations

import json
import math
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent))
import gw_stats as rv     # noqa: E402

OUT = Path(__file__).parent / "out"
RES = Path(__file__).parent / "results"
RES.mkdir(exist_ok=True)
S: dict = {}


def kn(g, col="detected"):
    k, n = int(g[col].sum()), len(g)
    lo, hi = rv.clopper_pearson(k, n)
    per = g.groupby(["home", "fold"])[col].mean()
    return dict(k=k, n=n, rate=k / n if n else float("nan"), ci=[lo, hi],
                folds=[float(per.min()), float(per.max())] if len(per) else None)


def rate(k, days):
    r, lo, hi = rv.garwood(int(k), days)
    return dict(k=int(k), days=days, minutes=days * 1440, per_day=r, ci=[lo, hi])


def main():
    # ── seven classes and floor check
    ep = pd.read_csv(OUT / "episodes_main.csv")
    b1 = ep[ep.B == 1.0]
    S["classes"] = {}
    for cls, g in b1.groupby("experiment"):
        d = kn(g)
        d["placebo"] = float(g.placebo.mean())
        d["latency_median_s"] = float(g.latency_s.median()) if g.latency_s.notna().any() else None
        if "detected_markov" in g and g.detected_markov.notna().any():
            d["markov"] = kn(g.dropna(subset=["detected_markov"]), "detected_markov")
        S["classes"][cls] = d
    h = b1[b1.experiment == "hide"]
    pred_miss = h.pred_uplinks > h.n_uplinks
    S["floor_check"] = dict(episodes=len(h), predicted_miss=int(pred_miss.sum()),
                            missed_of_predicted=int((pred_miss & (h.detected == 0)).sum()),
                            uplinks_range=[int(h.n_uplinks.min()), int(h.n_uplinks.max())],
                            required_median=float(np.median(h.pred_uplinks[np.isfinite(h.pred_uplinks)])),
                            required_range=[float(np.nanmin(h.pred_uplinks)), float(np.nanmax(h.pred_uplinks[np.isfinite(h.pred_uplinks)]))])
    # ── long horizon
    lp = OUT / "episodes_long.csv"
    if lp.exists():
        el = pd.read_csv(lp); el1 = el[el.B == 1.0]
        S["long"] = {}
        for cls, g in el1.groupby("experiment"):
            d = kn(g); d["placebo"] = float(g.placebo.mean())
            d["predicted_detect"] = float((g.pred_uplinks <= g.n_uplinks).mean())
            d["latency_median_s"] = float(g.latency_s.median()) if g.latency_s.notna().any() else None
            d["uplinks_range"] = [int(g.n_uplinks.min()), int(g.n_uplinks.max())]
            d["episode_minutes"] = [float(((g.t1 - g.t0) / 60).min()), float(((g.t1 - g.t0) / 60).max())]
            S["long"][cls] = d
        bl = pd.read_csv(OUT / "benign_long.csv")
        bb = bl[(bl.B == 1.0)]
        S["long_benign"] = {c: rate(g.core.sum(), g.exposure_days.sum()) for c, g in bb.groupby("calib")}
    # ── benign calibration
    ben = pd.read_csv(OUT / "benign_main.csv")
    S["benign"] = {}
    for (B, calib), g in ben.groupby(["B", "calib"]):
        S["benign"].setdefault(str(B), {})[calib] = dict(core=rate(g.core.sum(), g.exposure_days.sum()),
                                                         optin=rate(g.optin.sum(), g.exposure_days.sum()))
    tests = ben[(ben.B == 1.0) & (ben.calib == "bern")].tests.fillna("")
    S["benign_B1_tests"] = ";".join(t for t in tests if t)
    # ── adaptive
    for name in ("episodes_adaptive.csv", "episodes_adaptive_long.csv"):
        p = OUT / name
        if not p.exists():
            continue
        ad = pd.read_csv(p)
        for exp, g in ad.groupby("experiment"):
            if exp == "cov_ref":
                continue
            d = kn(g); d["placebo"] = float(g.placebo.mean())
            d["latency_median_s"] = float(g.latency_s.median()) if g.latency_s.notna().any() else None
            for c in ("hidden", "candidates", "mismatch", "mismatch_benign", "coverage"):
                if c in g and g[c].notna().any():
                    d[c] = float(g[c].sum()) if c in ("hidden", "candidates") else float(g[c].mean())
            S.setdefault("adaptive", {})[exp] = d
    # ── counters, sweep, cold start
    if (OUT / "counters.json").exists():
        S["counters"] = json.loads((OUT / "counters.json").read_text())
    if (OUT / "coldstart.csv").exists():
        cs = pd.read_csv(OUT / "coldstart.csv")
        S["coldstart"] = cs.groupby(["minutes", "prior"]).agg(tests=("tests", "mean"), detection=("detection", "mean"),
                                                             benign=("benign_alarms", "sum")).reset_index().to_dict("records")
    if (OUT / "sweep.csv").exists():
        sw = pd.read_csv(OUT / "sweep.csv")
        sw["rate"] = sw.benign_alarms / sw.exposure_days
        sw["gain"] = sw.det - sw.placebo
        S["matched"] = {}
        for att, g in sw.groupby("attack"):
            for fa in (50, 100, 1000):
                ok = g[g.rate <= fa]
                best = ok.groupby("detector").gain.max().to_dict()
                S["matched"].setdefault(att, {})[fa] = best
    (RES / "summary.json").write_text(json.dumps(S, indent=1, default=float))
    print(json.dumps(S, indent=1, default=lambda v: round(float(v), 3))[:6000])


if __name__ == "__main__":
    main()
