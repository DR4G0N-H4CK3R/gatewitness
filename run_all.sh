#!/usr/bin/env bash
# Reproduce every number and figure in the paper from ZIOTP2025.
# Usage:  GW_DATA=/path/to/ZIOTP2025 ./run_all.sh        (~20 min on one core)
# GW_DATA must contain the dataset's Data/ folder (Data/Web_Interaction, Data/Extra/IP-Traffic).
set -euo pipefail
cd "$(dirname "$0")"
mkdir -p out results
step() { echo; echo "== $*"; }
step "0/8 self-test of the statistics code";      python3 gw_stats.py selftest --config configs/fill.json
step "1/8 parse the 12 paired captures";           python3 gw_load.py
step "2/8 channel + commissioning test";           python3 channel.py > /dev/null
step "3/8 seven attack classes (held-out folds)";  python3 run_main.py main
step "4/8 long-horizon suppression";               python3 run_main.py long
step "5/8 adaptive attacks A1-A10";                for w in adaptive long counters coldstart; do python3 run_adaptive.py $w; done
step "6/8 matched-rate baseline sweeps";           python3 run_sweep.py
step "7/8 summary + derived numbers";              python3 summarize.py > /dev/null; python3 derive.py > /dev/null
step "8/8 intervals, calibration table, Fig. 2";   python3 export_csvs.py; python3 gw_stats.py all --config configs/fill.json
step "check against the paper";                   python3 check_paper.py
echo; echo "done: results/summary.json, results/derived.json, results/gw_results.tex, results/figs/fig5_matched_coldstart.pdf"
