"""Live demonstration of GateWitness on real Home Assistant traffic.

    python demo.py                         # the default walk-through (about 1-2 min)
    python demo.py --fold B1 --attack forgeX --device 1 --seed 3
                                           # replay any single episode of Table II

Steps
  1. load the 12 paired radio + IP captures of ZIOTP2025
  2. commissioning: is the IP tap really carrying the hub's reports?  (paper Sec. V)
  3. learn every test on one fold's commissioning traffic, thresholds from the budget
  4. run the verifier on the clean held-out traffic: benign alarms
  5. inject attacks into the held-out traffic and run the verifier again; each
     episode is re-scored live and compared with the committed row of
     out/episodes_main.csv, the file Table II is computed from

Nothing here is precomputed: every alarm is produced by the same functions that
produced the paper's numbers (gw_eval.alarm_times on an injected session).
"""
from __future__ import annotations

import argparse
import sys
import time
import zlib
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).parent
sys.path.insert(0, str(HERE))

NAMES = {"hide": "hide 80% of reports", "delay": "delay reports by 30 s", "change": "add 60 B to reports",
         "swap": "swap two devices' reports", "invent": "invent reports (1/15 s)",
         "leak": "leak to a new host (1/20 s)", "forge": "forge frames, host-timed ACK",
         "forgeX": "forge frames, firmware-exact ACK"}
DEFAULT_SHOW = ["forge", "leak", "hide", "change", "forgeX", "swap"]


def say(msg=""):
    print(msg, flush=True)


def header(n, title):
    say(); say(f"[{n}] {title}"); say("-" * (len(title) + 4))


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--fold", default="A0", help="home letter + fold index, e.g. A0, A2, B1")
    ap.add_argument("--attack", choices=list(NAMES), help="replay only this attack class")
    ap.add_argument("--device", type=int, choices=[0, 1], help="0/1: the two top devices of the fold")
    ap.add_argument("--seed", type=int, help="seed 0-4 (Table II uses 5 seeds per device)")
    ap.add_argument("--all", action="store_true",
                    help="replay every Table II episode of the fold (8 classes x 2 devices x 5 seeds)")
    args = ap.parse_args()

    t_start = time.time()
    header(1, "Load the paired captures (ZIOTP2025, Home Assistant + ZHA)")
    import gw_load
    P = gw_load.load_pairs()
    for topo in "AB":
        pp = [p for p in P if p["topology"] == topo]
        fr = sum(len(p["frames"]) for p in pp); ip = sum(len(p["ip"]) for p in pp)
        say(f"  home {topo}: {len(pp)} paired captures, {fr:,} radio frames, {ip:,} hub IP records")

    import folds as FO
    import gw_core as C
    import gw_eval as E
    import run_main as RM

    header(2, "Commissioning test: does the IP tap carry the hub's reports?")
    for topo in "AB":
        S = C.build_session([p for p in P if p["topology"] == topo], topo)
        t = C.commissioning(S)
        cs = C.channel_stats(S); cs = cs[cs.n >= 20]
        say(f"  home {topo}: coupling D = {t['D']:.3f}, permutation p = {t['p']:.3f} -> "
            f"{'ADMITTED' if t['admitted'] else 'REFUSED'};  per-event signal s median {cs.s.median():.2f}")
    say("  paper: 'Both homes are admitted (D = 0.466 and 0.379, p = 0.002)'")

    header(3, f"Learn the tests on fold {args.fold}'s commissioning traffic")
    home, k = args.fold[0].upper(), int(args.fold[1:])
    f = next(x for x in FO.folds() if x["home"] == home and x["fold"] == k)
    M = E.commission(f["comm"])
    MB = E.recalibrate(M, 1.0)
    kinds = pd.Series([t.split("[")[0] for t in MB.tests]).value_counts().to_dict()
    say(f"  commissioning on {', '.join(n.split('/')[-1] for n in f['comm_names'])}")
    say(f"  {len(MB.tests)} tests admitted: " + ", ".join(f"{v} {k}" for k, v in sorted(kinds.items())))
    if MB.ack_band:
        say(f"  ACK-turnaround band learned: {MB.ack_band['centre']:.1f} us +/- {MB.ack_band['half']:.0f} us "
            f"from {MB.ack_band['n']} ACKs")
    say("  thresholds: CUSUM, derived from a declared budget of B = 1 benign alarm/day (none tuned)")

    header(4, "Run the verifier on the clean held-out traffic")
    held = f["held"]
    clean = E.alarm_times(MB, held, "bern")
    core = {kk: v for kk, v in clean.items() if not kk.startswith(RM.OPT_IN)}
    n = E.count_alarms(core); days = E.exposure_days(held)
    fired = [kk for kk, v in core.items() if len(v)]
    say(f"  held out: {', '.join(x.split('/')[-1] for x in f['held_names'])} ({days * 1440:.1f} min)")
    say(f"  benign alarms: {n}" + (f" ({', '.join(fired)})" if fired else "") +
        f"  -> {n / days:.0f}/day on this fold")
    say("  paper: 234 benign alarms/day pooled over all six folds at B = 1 (the budget fails, Table III)")

    header(5, "Inject attacks into the held-out traffic and run the verifier again")
    ref = pd.read_csv(HERE / "out" / "episodes_main.csv")
    ref = ref[(ref.B == 1.0) & (ref.home == home) & (ref.fold == k)]
    top = f["comm"].uplinks.dev.value_counts().index[:2].tolist()
    classes = [args.attack] if args.attack else (list(NAMES) if args.all else DEFAULT_SHOW)
    devs = [args.device] if args.device is not None else ([0, 1] if args.all else [0])
    seeds = [args.seed] if args.seed is not None else (list(range(5)) if args.all else [0])
    say(f"  {'attack':34s} {'device':>6s} {'seed':>4s}  {'live result':32s} {'committed row':16s} match")
    agree = total = 0
    for cls in classes:
        for di in devs:
            d = top[di]; d2 = top[1 - di] if len(top) > 1 else d
            for seed in seeds:
                rng = np.random.default_rng(zlib.crc32(repr(("main", home, k, cls, int(d), seed)).encode()))
                t0, t1 = RM.episode_windows(held, rng, 1)[0]
                att = RM.attack(cls, held, d, d2, t0, t1, rng, M)
                quals = RM.qualifying(cls, d, d2)
                al = E.alarm_times(MB, att, "bern", only=set(quals))
                det, lat = RM.hit(al, quals, t0, t1)
                plc, _ = RM.hit(clean, quals, t0, t1)
                live = (f"DETECTED in {lat:5.1f} s" if det else "missed") + (" (placebo)" if plc else "")
                r = ref[(ref.experiment == cls) & (ref.device == d) & (ref.seed == seed)]
                if len(r):
                    r = r.iloc[0]
                    rd, rl = int(r.detected), r.latency_s
                    comm = (f"detected {rl:5.1f} s" if rd else "missed")
                    ok = (rd == int(det)) and (not det or abs(rl - lat) < 1e-6)
                    agree += ok; total += 1
                    mark = "yes" if ok else "NO"
                else:
                    comm, mark = "(no row)", "-"
                say(f"  {NAMES[cls]:34s} {d:>6d} {seed:>4d}  {live:32s} {comm:16s} {mark}")
    say()
    if total:
        say(f"  {agree}/{total} live episodes identical to the committed rows behind Table II")
    say("  '(placebo)': the clean traffic also alarmed in that window; Table II reports this rate separately")
    say(f"  done in {time.time() - t_start:.0f} s")


if __name__ == "__main__":
    main()
