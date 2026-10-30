"""Adaptive attacks A1-A10, counter evidence (8-bit vs 32-bit), forgery
shortfall diagnostics and cold start.  Writes out/episodes_adaptive.csv,
out/counters.json, out/forge_frames.csv, out/coldstart.csv."""
from __future__ import annotations

import json
import math
import sys
import time
import zlib
from dataclasses import replace
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent))
import folds as FO          # noqa: E402
import gw_core as C         # noqa: E402
import gw_eval as E         # noqa: E402
import run_main as RM       # noqa: E402

OUT = Path(__file__).parent / "out"
FP = E.FP


def rng_for(*key):
    return np.random.default_rng(zlib.crc32(repr(key).encode()))


def device_coverage(sess):
    """C_d per device from MAC sequence gaps (witness completeness)."""
    f = sess.frames[(sess.frames.ftype == 1) & sess.frames.src.notna() & (sess.frames.src != 0)]
    f = f.drop_duplicates(["src", "seq", "psdu_hash"])
    out = {}
    for s, g in f.groupby("src"):
        sq = g.seq.values.astype(int)
        gaps = (np.diff(sq) - 1) % 256
        gaps = np.where(gaps > 30, 0, gaps)
        out[int(s)] = len(sq) / (len(sq) + gaps.sum()) if len(sq) > 1 else np.nan
    return out


def score(al, quals, t0, t1):
    return RM.hit(al, quals, t0, t1)


def adaptive(F):
    rows = []
    for f in F:
        t_f = time.time()
        comm, held = f["comm"], f["held"]
        M = E.commission(comm)
        top = comm.uplinks.dev.value_counts().index[:2].tolist()
        clean = E.alarm_times(M, held, "bern")
        clean8 = E.alarm_times(M, held, "bern", counter="8")
        busiest_ep = tuple(held.egress.groupby(["remote", "port"]).size().idxmax()) if len(held.egress) else None

        def rec(name, tier, d, seed, t0, t1, att, quals, counter="32", extra=None, base=None):
            al = E.alarm_times(M, att, "bern", only=set(quals), counter=counter)
            det, lat = score(al, quals, t0, t1)
            plc, _ = score(base if base is not None else (clean8 if counter == "8" else clean), quals, t0, t1)
            r = dict(experiment=name, dataset="ziotp", home=f["home"], fold=f["fold"], seed=seed,
                     attack=name, tier=tier, device=d, detected=int(det), placebo=int(plc), latency_s=lat,
                     t0=t0, t1=t1)
            if extra:
                r.update(extra)
            rows.append(r)

        for di, d in enumerate(top):
            for seed in range(5):
                rng = rng_for("adaptive", f["home"], f["fold"], int(d), seed)
                t0, t1 = RM.episode_windows(held, rng, 1)[0]
                hideq = [f"SUP[{d}]"]
                # A1 spurious ACKs + hide
                att, _, _ = E.inj_hide(held, d, 0.8, t0, t1, rng)
                n = int(t1 - t0)
                acks = pd.DataFrame(dict(ts=np.sort(rng.uniform(t0, t1, n)), ftype=2, seq=rng.integers(0, 256, n),
                                         src=None, dst=None, pan=None, ack_req=False, mac_sec=False, length=5,
                                         nwk_src=None, nwk_dst=None, nwk_type=None, nwk_seq=None, fc32=None,
                                         sec_src=None, psdu_hash="ack"))
                a1 = replace(att, frames=pd.concat([att.frames, acks], ignore_index=True).sort_values("ts").reset_index(drop=True))
                rec("a1", 2, d, seed, t0, t1, a1, hideq)
                # A2a ACK withholding + hide
                a2a = E.inj_ack_withhold(att, d, t0, t1, rng)
                rec("a2a", 2, d, seed, t0, t1, a2a, [f"EMIT[{d}]", f"SUP[{d}]", "COVERAGE"])
                # A2b jamming to C~0.4, A2c channel change
                rec("a2b", 2, d, seed, t0, t1, E.inj_witness_loss(att, 0.4, t0, t1, rng),
                    [f"EMIT[{d}]", f"SUP[{d}]", "COVERAGE"])
                rec("a2c", 2, d, seed, t0, t1, E.inj_witness_loss(att, 0.0, t0, t1, rng),
                    [f"EMIT[{d}]", f"SUP[{d}]", "COVERAGE"])
                # A3 known-endpoint leak
                if busiest_ep is not None and di == 0:
                    for rate, nm in ((1 / 60, "a3_60"), (1 / 20, "a3_20"), (1 / 5, "a3_5")):
                        rec(nm, 2, d, seed, t0, t1, E.inj_leak(held, rate, t0, t1, rng, busiest_ep), ["VOL[*", "NEWEP"])
                # A4 counter-continuing forgery, 32-bit and 8-bit evidence
                for ack, nm in (("none", "no"), ("host", "sw"), ("exact", "ex")):
                    att4, nf, _ = E.inj_forge(held, d, FP["attacks"]["forge_rate"], t0, t1, rng, ack=ack,
                                              ack_us=FP["attacks"]["forge_ack_us"], band=M.ack_band, counter="continue")
                    rec(f"a4_32_{nm}", 3 if ack == "exact" else 2, d, seed, t0, t1, att4, [f"EMIT[{d}]"])
                    rec(f"a4_8_{nm}", 3 if ack == "exact" else 2, d, seed, t0, t1, att4, [f"EMIT[{d}]"], counter="8")
                # random-counter host-timed forgery with 8-bit evidence (the earlier design)
                att_f, _, _ = E.inj_forge(held, d, FP["attacks"]["forge_rate"], t0, t1, rng, ack="host",
                                          ack_us=FP["attacks"]["forge_ack_us"])
                rec("forge_8", 2, d, seed, t0, t1, att_f, [f"EMIT[{d}]"], counter="8")
                # A5 rate-preserving hide: wire identical to hide for the witness
                rec("a5", 2, d, seed, t0, t1, att, hideq)
                # A7 omniscient masked hiding (f = 1 within the window)
                a7, nd, nc = E.inj_hide(held, d, 1.0, t0, t1, rng, policy="masked")
                X0 = C.device_streams(held)[0][held.uplinks.dev.values == d]
                X7 = C.device_streams(a7)[0][a7.uplinks.dev.values == d]
                w = (held.uplinks.ts.values[held.uplinks.dev.values == d] >= t0) & \
                    (held.uplinks.ts.values[held.uplinks.dev.values == d] <= t1)
                rec("a7", 2, d, seed, t0, t1, a7, hideq,
                    extra=dict(hidden=nd, candidates=nc, mismatch=float(X7[w].mean()) if w.any() else np.nan,
                               mismatch_benign=float(X0[w].mean()) if w.any() else np.nan))
                # A8 chaff +50% / +100% with random hiding 0.78
                for frac, nm in ((0.5, "a8_50"), (1.0, "a8_100")):
                    a8, _, _ = E.inj_hide(E.inj_chaff(held, frac, t0, t1, rng), d, 0.78, t0, t1, rng)
                    rec(nm, 2, d, seed, t0, t1, a8, [f"SUP[{d}]", "VOLR"])
                # A9 upstream silencing
                rec("a9", 1, d, seed, t0, t1, E.inj_silence(held, d, t0, t1), [f"LIVE[{d}]"])
                rec("a9_nolive", 1, d, seed, t0, t1, E.inj_silence(held, d, t0, t1), [f"SUP[{d}]", f"EMIT[{d}]"])
        print(f"  adaptive fold {f['home']}{f['fold']} {time.time() - t_f:.0f}s", flush=True)
    return pd.DataFrame(rows)


def adaptive_long(F):
    """A7' realistic masked hiding (witness completeness 0.8 emulated by
    thinning) and A10 coverage-aware hiding, over whole long held-out sessions."""
    rows = []
    for f in F:
        comm, held = f["comm"], f["held"]
        a, b = held.segs[0][1] + 5, held.segs[-1][2] - 5
        # A7': emulate a witness that keeps each uplink with probability 0.8
        rng0 = rng_for("thin", f["home"], f["fold"])
        thin = lambda s, r: replace(s, uplinks=s.uplinks[r.random(len(s.uplinks)) < 0.8].reset_index(drop=True))
        comm_t, held_t = thin(comm, rng0), thin(held, rng0)
        Mt = E.commission(comm_t)
        clean_t = E.alarm_times(Mt, held_t, "bern")
        M = E.commission(comm)
        clean = E.alarm_times(M, held, "bern")
        top = comm.uplinks.dev.value_counts().index[:2].tolist()
        cov = device_coverage(comm)
        tested = [int(k[4:-1]) for k in M.tests if k.startswith("SUP[")]
        low = sorted([d for d in tested if d in cov], key=lambda d: cov[d])[:2]
        for d in top:
            for seed in range(5):
                rng = rng_for("a7p", f["home"], f["fold"], int(d), seed)
                # hub decides on its full view (held), witness sees held_t
                a7p, nd, nc = E.inj_hide(held, d, 1.0, a, b, rng, policy="masked")
                kept = held_t.uplinks
                a7p_w = replace(a7p, uplinks=kept)
                det, lat = RM.hit(E.alarm_times(Mt, a7p_w, "bern", only={f"SUP[{d}]"}), [f"SUP[{d}]"], a, b)
                plc, _ = RM.hit(clean_t, [f"SUP[{d}]"], a, b)
                Xw = C.device_streams(a7p_w)[0][kept.dev.values == d]
                Xb = C.device_streams(held_t)[0][kept.dev.values == d]
                rows.append(dict(experiment="a7p", dataset="ziotp", home=f["home"], fold=f["fold"], seed=seed,
                                 attack="a7p", tier=2, device=d, detected=int(det), placebo=int(plc), latency_s=lat,
                                 hidden=nd, candidates=nc, mismatch=float(Xw.mean()), mismatch_benign=float(Xb.mean()),
                                 coverage=0.8))
        for d in low:
            for seed in range(5):
                rng = rng_for("a10", f["home"], f["fold"], int(d), seed)
                att, _, _ = E.inj_hide(held, d, 0.8, a, b, rng)
                det, lat = RM.hit(E.alarm_times(M, att, "bern", only={f"SUP[{d}]"}), [f"SUP[{d}]"], a, b)
                plc, _ = RM.hit(clean, [f"SUP[{d}]"], a, b)
                rows.append(dict(experiment="a10", dataset="ziotp", home=f["home"], fold=f["fold"], seed=seed,
                                 attack="a10", tier=2, device=d, detected=int(det), placebo=int(plc), latency_s=lat,
                                 coverage=cov.get(d, np.nan),
                                 n_uplinks=int(((held.uplinks.dev == d) & (held.uplinks.ts >= a) & (held.uplinks.ts <= b)).sum())))
        for d in top:
            rows.append(dict(experiment="cov_ref", home=f["home"], fold=f["fold"], device=d, coverage=cov.get(d, np.nan)))
        print(f"  long adaptive fold {f['home']}{f['fold']}", flush=True)
    return pd.DataFrame(rows)


def counters(P):
    """Benign 8-bit and 32-bit counter anomaly rates over all paired captures."""
    n32 = e32 = n8 = e8 = 0
    for p in P:
        fr = p["frames"].sort_values("ts").reset_index(drop=True)
        for counter in ("32", "8"):
            em = E.emit_streams(fr, None, None, counter)
            k = sum(int(x.sum()) for _, x in em.values()); n = sum(len(x) for _, x in em.values())
            if counter == "32":
                e32 += k; n32 += n
            else:
                e8 += k; n8 += n
    return dict(fc32_events=e32, fc32_frames=n32, fc32_rate=e32 / max(n32, 1),
                seq8_events=e8, seq8_frames=n8, seq8_rate=e8 / max(n8, 1))


def forge_frames(F):
    """Labelled frames of host-timed random-counter forgery episodes, for the
    shortfall diagnostics (dedup drop, no genuine frame in window)."""
    out = []
    for f in F:
        held = f["held"]
        top = f["comm"].uplinks.dev.value_counts().index[:2].tolist()
        for d in top:
            for seed in range(5):
                rng = rng_for("adaptive", f["home"], f["fold"], int(d), seed)
                t0, t1 = RM.episode_windows(held, rng, 1)[0]
                att, nf, forged = E.inj_forge(held, d, FP["attacks"]["forge_rate"], t0, t1, rng, ack="host",
                                              ack_us=FP["attacks"]["forge_ack_us"])
                fr = att.frames
                m = (fr.ftype == 1) & (fr.src == d) & (fr.ts >= t0 - 60) & (fr.ts <= t1 + 120)
                for r in fr[m].itertuples():
                    out.append(dict(episode_id=f"{f['home']}{f['fold']}_{d}_{seed}", t=r.ts, mac_src=int(d),
                                    mac_seq=int(r.seq), psdu_hex=r.psdu_hash.encode().hex()[:32],
                                    forged=int(str(r.psdu_hash).startswith("forged")), device=int(d), t_end=t1))
    return pd.DataFrame(out)


def coldstart(F):
    rows = []
    homes = {"A": [f for f in F if f["home"] == "A"], "B": [f for f in F if f["home"] == "B"]}
    # prior: the other home's per-class (device type) eps and s from full commissioning
    import gwio.datasets.ziotp as Z
    prior_by_home = {}
    for h in "AB":
        other = "B" if h == "A" else "A"
        cls = Z.classes(other)
        ch = C.channel_stats(homes[other][0]["comm"])
        ch["cls"] = ch.dev.map(cls)
        prior_by_home[h] = ch.groupby("cls")[["eps", "s"]].median().to_dict("index")
    for f in F:
        cls_map = __import__("gwio.datasets.ziotp", fromlist=["classes"]).classes(f["home"])
        top = f["comm"].uplinks.dev.value_counts().index[:2].tolist()
        for minutes in (1, 2, 3, 5, 8, 11):
            T = minutes * 60
            seg0 = f["comm"].segs[0][1]
            sub = C.Session(f["comm"].name, f["comm"].frames[f["comm"].frames.ts < seg0 + T],
                            f["comm"].uplinks[f["comm"].uplinks.ts < seg0 + T],
                            f["comm"].units[f["comm"].units.ts < seg0 + T],
                            f["comm"].egress[f["comm"].egress.ts < seg0 + T],
                            [(n, a, min(b, seg0 + T)) for n, a, b in f["comm"].segs if a < seg0 + T])
            for use_prior in (0, 1):
                prior = None
                if use_prior:
                    pc = prior_by_home[f["home"]]
                    prior = {"sup": {}}
                    for d in sub.uplinks.dev.unique():
                        c = cls_map.get(int(d))
                        if c in pc:
                            eps = min(max(pc[c]["eps"], 1e-3), 0.95)
                            prior["sup"][int(d)] = (eps, min(eps + 0.5 * max(pc[c]["s"], 0.02), 0.999))
                M = E.commission(sub, prior=prior)
                al = RM.CORE(E.alarm_times(M, f["held"], "bern"))
                det = []
                for cls in ("hide", "change", "forge", "leak"):
                    for di, d in enumerate(top):
                        for seed in range(3):
                            rng = rng_for("cold", f["home"], f["fold"], cls, int(d), seed)
                            t0, t1 = RM.episode_windows(f["held"], rng, 1)[0]
                            att = RM.attack(cls, f["held"], d, top[1 - di], t0, t1, rng, M)
                            q = RM.qualifying(cls, d)
                            det.append(RM.hit(E.alarm_times(M, att, "bern", only=set(q)), q, t0, t1)[0])
                rows.append(dict(home=f["home"], fold=f["fold"], minutes=minutes, prior=use_prior,
                                 tests=len(M.tests), detection=float(np.mean(det)),
                                 benign_alarms=E.count_alarms(al), exposure_days=E.exposure_days(f["held"])))
        print(f"  coldstart fold {f['home']}{f['fold']}", flush=True)
    return pd.DataFrame(rows)


if __name__ == "__main__":
    what = sys.argv[1]
    if what == "adaptive":
        adaptive(FO.folds()).to_csv(OUT / "episodes_adaptive.csv", index=False)
    elif what == "long":
        adaptive_long(FO.folds(long=True)).to_csv(OUT / "episodes_adaptive_long.csv", index=False)
    elif what == "counters":
        (OUT / "counters.json").write_text(json.dumps(counters(FO.P), indent=1))
        forge_frames(FO.folds()).to_csv(OUT / "forge_frames.csv", index=False)
    elif what == "coldstart":
        coldstart(FO.folds()).to_csv(OUT / "coldstart.csv", index=False)
    print("done", what)
