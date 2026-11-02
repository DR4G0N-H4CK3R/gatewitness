# GateWitness — artifact

Code, frozen parameters, attack-injection scripts, episode labels and all
outputs for the paper

> **What a Passive Radio Witness Can Prove About an Untrusted Smart-Home
> Gateway: Detectability Limits, Adaptive Attacks and a Budget-Calibrated
> Verifier**

GateWitness checks a smart-home gateway it does not trust. It uses a passive
IEEE 802.15.4 receiver (the *radio witness*) and an IP tap, and needs no keys
and no help from the devices. It tests, sequentially, whether the gateway's
outbound reports match the radio traffic it received. The tests cover report
existence, count and timing, the gateway's egress, and its own use of the
radio. Each test is a CUSUM whose threshold is derived from a declared alarm
budget. Thresholds are never tuned.

Every number in the paper can be regenerated from the public
[ZIOTP2025](https://github.com/antonio-boiano/ZIOTP2025) dataset with one
command. The outputs of our own run are already committed in `out/` and
`results/`, so each number can also be checked without re-running anything.

## Quick start

```bash
pip install -r requirements.txt                       # Python ≥ 3.10
python3 gw_stats.py selftest --config configs/fill.json   # statistics self-test, no data needed

# get ZIOTP2025 (only Data/Web_Interaction and Data/Extra/IP-Traffic are read)
git clone https://github.com/antonio-boiano/ZIOTP2025 data
GW_DATA=data ./run_all.sh                             # 10–15 min on one core
```

On Windows (PowerShell) use `python` and the Python runner instead:

```powershell
$env:GW_DATA = "C:\path\to\ZIOTP2025"
python run_all.py
```

`GW_DATA` must point to the folder that **contains** the dataset's `Data/`
directory. It defaults to `./data`.

## Live demo and paper check

```bash
python demo.py                 # ≈ 30 s: load, commissioning test, learn tests, benign run, live attacks
python demo.py --all           # replay all 80 Table II episodes of one fold and compare with out/
python demo.py --fold B2 --attack forge --device 1 --seed 3     # replay any single episode
python check_paper.py          # every number in Tables II–IV and Sec. V vs. the result files
python dashboard.py            # browser dashboard at http://127.0.0.1:8050: replay any episode on a
                               # timeline (uplinks, reports, attack window, statistic vs. threshold,
                               # alarm) and the paper-vs-code table
```

`docs/DEMO.md` is a step-by-step script for showing this to someone.

## What `run_all.sh` does

| Step | Script | Writes | Paper |
|---|---|---|---|
| Parse the 12 paired radio + IP captures (2 homes) | `gw_load.py` | `cache/pairs.pkl` | §IV |
| Commissioning permutation test; per-device ε, 1−γ, s by self-injection | `channel.py` | `results/channel.json` | §III-B, §V |
| Seven attack classes on held-out folds, with placebo | `run_main.py main` | `out/episodes_main.csv`, `out/indicators.csv`, `out/benign_main.csv` | Tables II, III |
| Long-horizon suppression | `run_main.py long` | `out/episodes_long.csv`, `out/benign_long.csv` | Tables II, III |
| Adaptive attacks A1–A10, counters, cold start | `run_adaptive.py {adaptive,long,counters,coldstart}` | `out/episodes_adaptive*.csv`, `out/counters.json`, `out/forge_frames.csv`, `out/coldstart.csv` | Table IV, Fig. 2e |
| Threshold sweeps of GateWitness and seven baselines | `run_sweep.py` | `out/sweep.csv`, `out/sweep_episodes.csv` | Fig. 2a–d |
| Tables, intervals and derived text numbers | `summarize.py`, `derive.py` | `results/summary.json`, `results/derived.json` | §V |
| Exact intervals, calibration table, Fig. 2 | `export_csvs.py`, `gw_stats.py all` | `results/gw_results.{json,tex}`, `results/figs/fig5_matched_coldstart.pdf` | Tables II–IV, Fig. 2 |
| Compare with the paper | `check_paper.py` | (prints) | all tables |

`docs/RESULTS.md` maps each number quoted in the paper to the file and field
it comes from.

A clean run reproduces `out/` and `results/` byte for byte, with one exception.
`derived.json["log_us_per_decision"]` is a wall-clock timing and depends on the
machine. Fig. 2 is pixel-identical.

## Repository layout

```
frozen_params.json   every parameter, fixed before any attack was injected
gw_load.py           802.15.4 parser (adds 32-bit NWK frame counter, ACK-request bit, PSDU hash)
gw_ip.py             hub IP records (Home Assistant websocket tcp/8123, egress)
gw_core.py           sessions, eligible uplinks, report matching, channel statistics, commissioning
gw_eval.py           tests (SUP, SIZE, FAB, EMIT, VOL, NEWEP, LIVE, VOLR, COVERAGE),
                     budget-derived CUSUM calibration, and all attack injectors (inj_*)
gw_stats.py          Clopper–Pearson / Garwood intervals, Brook–Evans and Markov-exact
                     calibration, Fig. 2; `selftest` checks it on synthetic data
baselines.py         IsolationForest, IP rules, PingPong-style lengths, fixed count test,
                     radio-only, radio-to-hub, HAWatcher-style log monitor
folds.py             leave-two-captures-out folds per home
run_*.py             experiment drivers (run_all.sh / run_all.py run everything)
demo.py              live walk-through; replays any Table II episode
check_paper.py       compares every reported number with the result files
dashboard.py, web/   replay dashboard (standard library server + one HTML page, works offline)
gwio/                pcap/pcapng readers and the ZIOTP2025 adapter
out/, results/       outputs of our run (committed)
docs/CHANGES.md      every code change made after the parameters were frozen
```

## Labels and evaluation protocol

Misbehaviour is injected into held-out traffic with exact labels. Each row of
`out/episodes_*.csv` is one scored episode: attack, device, home, fold, seed,
start and end time, whether it was detected, and latency. Every detector also
runs on the clean held-out traffic, scored against the same windows (the
`placebo` column). That way chance detections are not credited. All
randomness is seeded from `frozen_params.json`. Fixes made after the
parameters were frozen, and the reruns they caused, are listed in
`docs/CHANGES.md`.

## Scope and limitations

- Attacks are injected into real traces, not produced by a compromised hub,
  so the second-order effects of a real compromise are absent.
- ZIOTP2025 carries no RSSI/LQI. The emitter evidence is ACK timing plus
  frame counters.
- The 36.9 minutes of held-out benign traffic is enough to reject a
  one-alarm-per-day budget, but not to confirm one.

## Data

ZIOTP2025 (Boiano et al., arXiv:2602.03140) is not redistributed here. Please
cite it and follow its licence. The CSVs in `out/` are derived metadata
(timestamps, lengths, labels), not packet contents.

## License

MIT (see `LICENSE`).
