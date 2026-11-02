"""Cross-platform version of run_all.sh (works in Windows PowerShell).

    $env:GW_DATA = "C:\\path\\to\\ZIOTP2025"      # PowerShell
    python run_all.py
"""
import os
import subprocess
import sys
from pathlib import Path

HERE = Path(__file__).parent
STEPS = [("0/8 self-test of the statistics code", ["gw_stats.py", "selftest", "--config", "configs/fill.json"]),
         ("1/8 parse the 12 paired captures", ["gw_load.py"]),
         ("2/8 channel + commissioning test", ["channel.py"]),
         ("3/8 seven attack classes (held-out folds)", ["run_main.py", "main"]),
         ("4/8 long-horizon suppression", ["run_main.py", "long"]),
         ("5/8 adaptive attacks A1-A10", ["run_adaptive.py", "adaptive"]),
         ("", ["run_adaptive.py", "long"]), ("", ["run_adaptive.py", "counters"]),
         ("", ["run_adaptive.py", "coldstart"]),
         ("6/8 matched-rate baseline sweeps", ["run_sweep.py"]),
         ("7/8 summary + derived numbers", ["summarize.py"]), ("", ["derive.py"]),
         ("8/8 intervals, calibration table, Fig. 2", ["export_csvs.py"]),
         ("", ["gw_stats.py", "all", "--config", "configs/fill.json"]),
         ("check against the paper", ["check_paper.py"])]

if __name__ == "__main__":
    os.chdir(HERE)
    (HERE / "out").mkdir(exist_ok=True); (HERE / "results").mkdir(exist_ok=True)
    quiet = {"channel.py", "summarize.py", "derive.py"}
    for title, cmd in STEPS:
        if title:
            print(f"\n== {title}", flush=True)
        out = subprocess.DEVNULL if cmd[0] in quiet else None
        r = subprocess.run([sys.executable] + cmd, stdout=out)
        if r.returncode:
            sys.exit(f"step failed: {' '.join(cmd)}")
    print("\ndone: results/summary.json, results/derived.json, results/gw_results.tex, "
          "results/figs/fig5_matched_coldstart.pdf")
