"""Write the five CSVs that gw_stats.py reads, from out/*.csv."""
from pathlib import Path

import pandas as pd

OUT = Path(__file__).parent / "out"
DST = Path(__file__).parent / "export"
DST.mkdir(exist_ok=True)
COLS = ["experiment", "dataset", "home", "fold", "seed", "attack", "device", "detected", "placebo",
        "latency_s", "n_uplinks", "coverage", "delay_s"]

parts = []
m = pd.read_csv(OUT / "episodes_main.csv"); parts.append(m[m.B == 1.0])
if (OUT / "episodes_long.csv").exists():
    l = pd.read_csv(OUT / "episodes_long.csv"); parts.append(l[l.B == 1.0])
for f in ("episodes_adaptive.csv", "episodes_adaptive_long.csv"):
    if (OUT / f).exists():
        a = pd.read_csv(OUT / f); parts.append(a[a.experiment != "cov_ref"])
ep = pd.concat(parts, ignore_index=True)
for c in COLS:
    if c not in ep:
        ep[c] = pd.NA
ep[COLS].to_csv(DST / "episodes.csv", index=False)

ind = pd.read_csv(OUT / "indicators.csv"); ind["set"] = "main"
ind.to_csv(DST / "indicators.csv", index=False)
if (OUT / "sweep.csv").exists():
    pd.read_csv(OUT / "sweep.csv")[["dataset", "detector", "attack", "threshold", "benign_alarms",
                                    "exposure_days", "det", "placebo"]].to_csv(DST / "sweep.csv", index=False)
for f in ("coldstart.csv", "forge_frames.csv"):
    if (OUT / f).exists():
        pd.read_csv(OUT / f).to_csv(DST / f, index=False)
print(sorted(p.name for p in DST.iterdir()))
