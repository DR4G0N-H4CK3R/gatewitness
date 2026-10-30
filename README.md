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
GW_DATA=data ./run_all.sh                             # ≈ 20 min on one core
```

`GW_DATA` must point to the folder that **contains** the dataset's `Data/`
directory. It defaults to `./data`.

## What `run_all.sh` does

| Step | Script | Writes | Paper |
|---|---|---|---|
| Parse the 12 paired radio + IP captures (2 homes) | `gw_load.py` | `cache/pairs.pkl` | §VI |
| Commissioning permutation test; per-device ε, 1−γ, s by self-injection | `channel.py` | `results/channel.json` | §III–IV |
| Seven attack classes on held-out folds, with placebo | `run_main.py main` | `out/episodes_main.csv`, `out/indicators.csv`, `out/benign_main.csv` | Table III |
| Long-horizon suppression | `run_main.py long` | `out/episodes_long.csv`, `out/benign_long.csv` | §VII |
| Adaptive attacks A1–A10, counters, cold start | `run_adaptive.py {adaptive,long,counters,coldstart}` | `out/episodes_adaptive*.csv`, `out/counters.json`, `out/forge_frames.csv`, `out/coldstart.csv` | Table V, Fig. 5e |
| Threshold sweeps of GateWitness and seven baselines | `run_sweep.py` | `out/sweep.csv`, `out/sweep_episodes.csv` | Fig. 5a–d |
| Tables, intervals and derived text numbers | `summarize.py`, `derive.py` | `results/summary.json`, `results/derived.json` | §VII |
| Exact intervals, calibration table, Fig. 5 | `export_csvs.py`, `gw_stats.py all` | `results/gw_results.{json,tex}`, `results/figs/fig5_matched_coldstart.pdf` | Tables III–V, Fig. 5 |

`docs/RESULTS.md` maps each number quoted in the paper to the file and field
it comes from.

A clean run reproduces `out/` and `results/` byte for byte, with one exception.
`derived.json["log_us_per_decision"]` is a wall-clock timing and depends on the
machine. Fig. 5 is pixel-identical.

## Repository layout

```
frozen_params.json   every parameter, fixed before any attack was injected
gw_load.py           802.15.4 parser (adds 32-bit NWK frame counter, ACK-request bit, PSDU hash)
gw_ip.py             hub IP records (Home Assistant websocket tcp/8123, egress)
gw_core.py           sessions, eligible uplinks, report matching, channel statistics, commissioning
gw_eval.py           tests (SUP, SIZE, FAB, EMIT, VOL, NEWEP, LIVE, VOLR, COVERAGE),
                     budget-derived CUSUM calibration, and all attack injectors (inj_*)
gw_stats.py          Clopper–Pearson / Garwood intervals, Brook–Evans and Markov-exact
                     calibration, Fig. 5; `selftest` checks it on synthetic data
baselines.py         IsolationForest, IP rules, PingPong-style lengths, fixed count test,
                     radio-only, radio-to-hub, HAWatcher-style log monitor
folds.py             leave-two-captures-out folds per home
run_*.py             experiment drivers
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
