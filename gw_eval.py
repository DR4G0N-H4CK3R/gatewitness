"""GateWitness evaluation on ZIOTP2025 (paper Sections V-VIII).

All parameters come from frozen_params.json.  Every detector runs on the
attacked held-out stream and, as placebo, on the clean held-out stream scored
against the same episode windows.
"""
from __future__ import annotations

import json
import math
import pickle
import sys
from dataclasses import dataclass, replace
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats as sst

HERE = Path(__file__).parent
sys.path.insert(0, str(HERE))
import gw_core as C                       # noqa: E402
import gw_stats as rv           # noqa: E402

FP = json.loads((HERE / "frozen_params.json").read_text())
DT, TAIL = FP["dt_s"], FP["score_tail_s"]
import shelve
_ORIG_T = rv.threshold_for_arl


def _memo_threshold(l1, l0, target, a, b, res=64, max_states=40_000):
    key = repr(tuple(round(float(v), 9) for v in (l1, l0, target, a, b)) + (res, max_states))
    for other in sorted((HERE / "cache").glob("thresholds*")):
        if other.suffix or other.name.endswith(("-shm", "-wal")) or str(other) == _DB:
            continue
        try:
            with shelve.open(str(other), flag="r") as db:
                if key in db:
                    return db[key]
        except Exception:
            pass
    with shelve.open(_DB) as db:
        if key in db:
            return db[key]
        val = _ORIG_T(l1, l0, target, a, b, res, max_states)
        db[key] = val
        return val


_DB = str(HERE / "cache" / ("thresholds_" + Path(sys.argv[0]).stem + "_" + (sys.argv[1] if len(sys.argv) > 1 else "")))
rv.threshold_for_arl = _memo_threshold
P_RV = dict(rv.DEFAULT_CONFIG["params"], beta_upper=FP["beta_upper_q"], f_star=FP["f_star"], lattice_res=32)
COORD = 0
AIR = lambda length: (length + 6) * 32.0          # us, frame length incl. FCS
SPD = 86400.0


# ════════════════════════════════════════════════════════════════════════════
# Evidence streams
# ════════════════════════════════════════════════════════════════════════════
def exposure_days(sess: C.Session) -> float:
    return sum(b - a for _, a, b in sess.segs) / SPD


def sup_streams(sess):
    X, credit, cdev = C.device_streams(sess, DT)
    dev, ts = sess.uplinks.dev.values, sess.uplinks.ts.values
    return {d: (ts[dev == d], X[dev == d]) for d in np.unique(dev)}, credit, cdev


def size_streams(sess, bands, cdev=None):
    if cdev is None:
        _, _, cdev = C.device_streams(sess, DT)
    out = {}
    t, L = sess.units.ts.values, sess.units.length.values
    for d, (lo, hi) in bands.items():
        m = cdev == d
        out[d] = (t[m], ((L[m] < lo) | (L[m] > hi)).astype(int))
    return out


def fab_stream(sess, credit=None):
    if credit is None:
        _, credit = C.match(sess)
    return sess.units.ts.values, (credit < 0).astype(int)


def coverage_series(frames: pd.DataFrame, win=120.0):
    """Completeness per 10 s step from MAC sequence gaps of non-coordinator
    transmitters (retransmissions collapsed).  Returns (t_grid, completeness)."""
    f = frames[(frames.ftype != 2) & frames.src.notna() & (frames.src != COORD)]
    f = f.drop_duplicates(["src", "seq", "psdu_hash"])
    caps, lost = [], []
    for s, g in f.groupby("src"):
        sq = g.seq.values.astype(int)
        tt = g.ts.values
        gaps = (np.diff(sq) - 1) % 256
        gaps = np.where(gaps > 30, 0, gaps)          # resets / long silences are not loss
        caps.append(tt[1:]); lost.append(np.stack([tt[1:], gaps], 1))
    if not caps:
        return np.array([0.0]), np.array([1.0])
    ct = np.sort(np.concatenate(caps))
    L = np.concatenate(lost); L = L[np.argsort(L[:, 0])]
    cl = np.concatenate([[0], np.cumsum(L[:, 1])])
    t_end = frames.ts.max() if len(frames) else 0
    grid = np.arange(win, t_end + 10, 10.0)
    out = []
    for g in grid:
        a, b = np.searchsorted(ct, [g - win, g])
        la, lb = np.searchsorted(L[:, 0], [g - win, g])
        n, ls = b - a, cl[lb] - cl[la]
        out.append(n / (n + ls) if n + ls > 0 else np.nan)
    return grid, np.asarray(out)


def coverage_alarms(frames, sess_segs, thr=0.5, hold=120.0):
    grid, c = coverage_series(frames)
    # inside capture segments only; empty windows (gaps) are silence, not low coverage
    low = np.zeros(len(grid), bool)
    for _, a, b in sess_segs:
        m = (grid >= a + 120) & (grid <= b)
        low[m] = np.nan_to_num(c[m], nan=0.0) < thr
    al, run0 = [], None
    for g, l in zip(grid, low):
        if l:
            run0 = g if run0 is None else run0
            if g - run0 >= hold and (not al or g - al[-1] > hold):
                al.append(g)
        else:
            run0 = None
    return np.asarray(al), grid, c


def emit_streams(frames, band, cov=None, counter="32", segs=None):
    """Per transmitter a: one unit per data frame a sends.  x = 1 on a counter
    collision at that frame, or (frame to coordinator requesting an ACK) an ACK
    outside the band, or a missing ACK while coverage >= 0.8."""
    fr = frames.sort_values("ts", kind="stable").reset_index(drop=True)
    n = len(fr)
    ts, ft, sq, ln = fr.ts.values, fr.ftype.values, fr.seq.values, fr.length.values
    src, dst, ar = fr.src.values, fr.dst.values, fr.ack_req.values
    psdu, fc = fr.psdu_hash.values, fr.fc32.values
    ssrc = fr.sec_src.values
    x = np.zeros(n, int)
    # ACK evidence
    if band is not None:
        cgrid, cval = cov if cov is not None else (None, None)
        for i in np.where((ft == 1) & ar & (dst == COORD) & (src != COORD))[0]:
            j = i + 1
            ack = None
            while j < n and ts[j] - ts[i] < AIR(ln[i]) / 1e6 + 0.002:   # ts = frame start
                if ft[j] == 2 and sq[j] == sq[i]:
                    ack = j; break
                j += 1
            if ack is not None:
                gap = (ts[ack] - ts[i]) * 1e6 - AIR(ln[i])
                if abs(gap - band["centre"]) > band["half"]:
                    x[i] = 1
            elif cgrid is not None:
                k = np.searchsorted(cgrid, ts[i])
                cv = cval[min(k, len(cval) - 1)] if len(cval) else 1.0
                if not np.isnan(cv) and cv >= 0.8:
                    x[i] = 1
    # counter evidence on raw frames (before deduplication)
    seg_starts = np.array(sorted(a for _, a, _ in segs)) if segs else np.array([-np.inf])
    seg_id = np.searchsorted(seg_starts, ts, side="right")      # counters reset per capture
    if counter == "32":
        state = {}
        for i in np.where((ft == 1) & pd.notna(fc))[0]:
            k = (seg_id[i], ssrc[i] if pd.notna(ssrc[i]) else src[i])
            if k in state:
                lf, lp = state[k]
                if (fc[i] == lf and psdu[i] != lp) or fc[i] < lf:
                    x[i] = 1
            if k not in state or fc[i] >= state[k][0]:
                state[k] = (fc[i], psdu[i])
    elif counter == "8":
        # earlier design: MAC sequence steps per transmitter after 8-bit dedup
        keep = {}
        last = {}
        for i in np.where((ft == 1) & pd.notna(src))[0]:
            key = (src[i], sq[i])
            if key in keep and ts[i] - keep[key] <= 0.5:
                continue                              # dropped by (src, seq) dedup
            keep[key] = ts[i]
            lk = (seg_id[i], src[i])
            if lk in last:
                step = (int(sq[i]) - int(last[lk])) % 256
                if step == 0 or step > 128:           # repeat or backward
                    x[i] = 1
            last[lk] = sq[i]
    out = {}
    m = (ft == 1) & pd.notna(src) & (src != COORD)
    for a in np.unique(src[m]):
        mm = m & (src == a)
        out[int(a)] = (ts[mm], x[mm])
    return out


def learn_band(frames):
    fr = frames.sort_values("ts").reset_index(drop=True)
    ts, ft, sq, ln, ar, dst = (fr.ts.values, fr.ftype.values, fr.seq.values, fr.length.values,
                               fr.ack_req.values, fr.dst.values)
    g = []
    for i in np.where((ft[:-1] == 1) & ar[:-1] & (dst[:-1] == COORD))[0]:
        if ft[i + 1] == 2 and sq[i + 1] == sq[i] and ts[i + 1] - ts[i] < AIR(ln[i]) / 1e6 + 0.002:
            g.append((ts[i + 1] - ts[i]) * 1e6 - AIR(ln[i]))
    g = np.asarray(g)
    if len(g) < 50:
        return None
    span = np.percentile(g, 99.5) - np.percentile(g, 0.5)
    return {"centre": float(np.median(g)), "half": 10.0, "span": float(span), "n": len(g)} if span <= 20 else None


def bins(times, t_end, w=30.0, segs=None):
    edges = np.arange(0, t_end + w, w)
    c, _ = np.histogram(times, edges)
    centres = edges[:-1] + w / 2
    if segs is not None:
        inside = np.zeros(len(centres), bool)
        for _, a, b in segs:
            inside |= (centres >= a) & (centres <= b)
        return centres[inside], c[inside]
    return centres, c


# ════════════════════════════════════════════════════════════════════════════
# Commissioning: learn every test
# ════════════════════════════════════════════════════════════════════════════
@dataclass
class Model:
    tests: dict           # tid -> dict(kind, cal or params)
    bands: dict           # SIZE bands per device
    ack_band: dict | None
    known_eps: set
    newep_thr: int
    live_gap: dict
    volr_ratio: float
    fab_admitted: bool
    devices: list
    chan: pd.DataFrame
    N: int


def commission(comm: C.Session, B=1.0, pairs_sessions=None, prior=None, min_units=None) -> Model:
    min_units = min_units or FP["min_units_per_test"]
    sup, credit, cdev = sup_streams(comm)
    chan = C.channel_stats(comm)
    sdict = dict(zip(chan.dev, chan.s)) if len(chan) else {}
    tests = {}
    for d, (t, x) in sup.items():
        if len(x) >= min_units:
            tests[f"SUP[{d}]"] = dict(kind="bernoulli", t=t, x=x, s=max(sdict.get(d, 0.05), 0.02))
        elif prior is not None and len(x) >= 5 and d in prior.get("sup", {}):
            tests[f"SUP[{d}]"] = dict(kind="bernoulli", t=t, x=x, prior=prior["sup"][d])
    bands = {}
    L, tU = comm.units.length.values, comm.units.ts.values
    for d in np.unique(comm.uplinks.dev):
        Ld = L[cdev == d]
        if len(Ld) >= min_units:
            bands[d] = tuple(np.percentile(Ld, FP["size_band_pct"]))
    for d, (t, x) in size_streams(comm, bands, cdev).items():
        tests[f"SIZE[{d}]"] = dict(kind="bernoulli", t=t, x=x)
    # FAB admission: homogeneity of per-capture rates
    t_f, x_f = fab_stream(comm, credit)
    cnt = []
    for _, a, b in comm.segs:
        m = (t_f >= a) & (t_f <= b)
        cnt.append((x_f[m].sum(), m.sum() - x_f[m].sum()))
    cnt = np.asarray(cnt)
    fab_ok = False
    if len(cnt) >= 2 and (cnt.sum(0) > 0).all():
        fab_ok = sst.chi2_contingency(cnt)[1] >= 0.05
    if fab_ok:
        tests["FAB"] = dict(kind="bernoulli", t=t_f, x=x_f)
    ack_band = learn_band(comm.frames)
    cov = coverage_series(comm.frames)
    for a, (t, x) in emit_streams(comm.frames, ack_band, cov, segs=comm.segs).items():
        if len(x) >= min_units:
            tests[f"EMIT[{a}]"] = dict(kind="bernoulli", t=t, x=x)
    # VOL per known endpoint (Poisson on 30 s bins), VOLR
    known = set(map(tuple, comm.egress[["remote", "port"]].drop_duplicates().values))
    for (rem, port), g in comm.egress.groupby(["remote", "port"]):
        c_t, c = bins(g.ts.values, comm.t_end, segs=comm.segs)
        if c.sum() >= 5:
            tests[f"VOL[{rem}:{port}]"] = dict(kind="poisson", t=c_t, x=c, ep=(rem, port))
    u_t, u_c = bins(comm.units.ts.values, comm.t_end, segs=comm.segs)
    _, up_c = bins(comm.uplinks.ts.values, comm.t_end, segs=comm.segs)
    volr = u_c.sum() / max(up_c.sum(), 1)
    # NEWEP threshold: largest new-endpoint volume after the first 10 min of commissioning
    first = comm.egress.groupby(["remote", "port"]).ts.min()
    late = first[first > 600].index
    vmax = 0
    for ep in late:
        tt = np.sort(comm.egress[(comm.egress.remote == ep[0]) & (comm.egress.port == ep[1])].ts.values)
        if len(tt):
            vmax = max(vmax, int(np.max(np.searchsorted(tt, tt + 60) - np.arange(len(tt)))))
    # LIVE: per device gap quantile
    live = {}
    for d, g in comm.uplinks.groupby("dev"):
        fr_d = comm.frames[(comm.frames.src == d) | (comm.frames.nwk_src == d)].ts.values
        gaps = []
        for _, a, b in comm.segs:
            tt = np.sort(fr_d[(fr_d >= a) & (fr_d <= b)])
            gaps += list(np.diff(tt))
        if len(gaps) >= 20:
            live[d] = max(2 * float(np.percentile(gaps, 99)), 30.0)
    # calibrate CUSUM tests
    N = len(tests) + 1   # + VOLR
    for tid, T in tests.items():
        T["cal"] = calibrate(tid, T, N, B)
    return Model(tests, bands, ack_band, known, max(1, 2 * vmax), live, volr, fab_ok,
                 sorted(np.unique(comm.uplinks.dev)), chan, N)


def calibrate(tid, T, N, B):
    if T["kind"] == "poisson":
        lam0 = max(T["x"].mean(), 1e-3)
        u_d = len(T["x"]) / max(T["t"][-1] - T["t"][0], 30.0) * SPD
        target = u_d * N / B
        a_, b_ = math.log(2.0), lam0
        inc = T["x"] * a_ - b_
        phi = rv.dispersion_phi(inc > 0, a_, -b_, phi_max=8.0)
        h = max(math.log(target), 1.0)
        return dict(lam0=lam0, lam1=2 * lam0, h_bern=h, h_disp=h * phi, h_markov=h * phi, phi=phi)
    x, t = T["x"], T["t"]
    design = None
    if "s" in T:
        k, n = int(x.sum()), len(x)
        eps_u = min(max(rv.beta_upper(k, n, FP["beta_upper_q"]), 1e-4), 0.95)
        design = (eps_u, min(eps_u + FP["f_star"] * T["s"], 0.999))
    elif "prior" in T:
        design = T["prior"]
    c = rv.calibrate_test(tid, "bernoulli", x, t, N, B, P_RV, design)
    l1, l0 = rv.cusum_increments(c.eps, c.p1)
    return dict(eps=c.eps, p1=c.p1, l1=l1, l0=l0, h_bern=c.h_bern, h_disp=c.h_disp,
                h_markov=c.h_markov if c.h_markov is not None else c.h_bern, phi=c.phi,
                a=c.a, b=c.b, u_d=c.units_per_day)


def recalibrate(M: Model, B: float) -> Model:
    tests = {k: dict(v) for k, v in M.tests.items()}
    for tid, T in tests.items():
        T["cal"] = calibrate(tid, T, M.N, B)
    return replace(M, tests=tests)


# ════════════════════════════════════════════════════════════════════════════
# Running the verifier on a (possibly attacked) held-out session
# ════════════════════════════════════════════════════════════════════════════
def alarm_times(M: Model, sess: C.Session, which="bern", only=None, counter="32"):
    """dict tid -> np.array of alarm times.  `only`: set of tids to evaluate."""
    want = lambda tid: only is None or tid in only or any(tid.startswith(o[:-1]) for o in only if o.endswith("*"))
    out = {}
    sup, credit, cdev = (None, None, None)
    need_match = any(want(t) for t in M.tests if t.startswith(("SUP", "SIZE", "FAB")))
    if need_match:
        sup, credit, cdev = sup_streams(sess)
    hkey = {"bern": "h_bern", "disp": "h_disp", "markov": "h_markov"}[which]
    emit = None
    for tid, T in M.tests.items():
        if not want(tid):
            continue
        cal = T["cal"]
        if tid.startswith("SUP"):
            d = int(tid[4:-1]); t, x = sup.get(d, (np.array([]), np.array([])))
        elif tid.startswith("SIZE"):
            d = int(tid[5:-1])
            t, x = size_streams(sess, {d: M.bands[d]}, cdev)[d]
        elif tid == "FAB":
            t, x = fab_stream(sess, credit)
        elif tid.startswith("EMIT"):
            if emit is None:
                emit = emit_streams(sess.frames, M.ack_band, coverage_series(sess.frames), counter, segs=sess.segs)
            t, x = emit.get(int(tid[5:-1]), (np.array([]), np.array([])))
        elif tid.startswith("VOL["):
            ep = T["ep"]
            g = sess.egress[(sess.egress.remote == ep[0]) & (sess.egress.port == ep[1])]
            t, x = bins(g.ts.values, sess.t_end, segs=sess.segs)
            idx = rv.run_cusum_poisson(x, cal["lam0"], cal["lam1"], cal[hkey])
            out[tid] = t[idx] if len(idx) else np.array([])
            continue
        else:
            continue
        idx = rv.run_cusum(x, cal["l1"], cal["l0"], cal[hkey]) if len(x) else []
        out[tid] = t[idx] if len(idx) else np.array([])
    # rule-based and opt-in tests
    if want("NEWEP"):
        out["NEWEP"] = newep_alarms(M, sess)
    if only is None or any(o.startswith("LIVE") for o in only):
        for d, a in live_alarms(M, sess).items():
            out[f"LIVE[{d}]"] = a
    if want("VOLR"):
        out["VOLR"] = volr_alarms(M, sess, which)
    if want("COVERAGE"):
        out["COVERAGE"] = coverage_alarms(sess.frames, sess.segs)[0]
    return out


def newep_alarms(M, sess):
    al, seen, known = [], {}, set(M.known_eps)
    for r in sess.egress.itertuples():
        ep = (r.remote, r.port)
        if ep in known:
            continue
        seen.setdefault(ep, []).append(r.ts)
        if sum(1 for x in seen[ep] if x > r.ts - 60) > M.newep_thr:
            al.append(r.ts)
            known.add(ep)          # one alarm per new endpoint
    return np.asarray(al)


def live_alarms(M, sess):
    out = {}
    fr = sess.frames
    allt = np.sort(fr.ts.values)
    for d, thr in M.live_gap.items():
        tt = np.sort(fr[(fr.src == d) | (fr.nwk_src == d)].ts.values)
        al = []
        for _, a, b in sess.segs:
            seg = tt[(tt >= a) & (tt <= b)]
            pts = np.concatenate([[a], seg, [b]])
            for p, q in zip(pts[:-1], pts[1:]):
                if q - p > thr:
                    others = np.sum((allt > p) & (allt < q))
                    if others > 10:
                        al.append(p + thr)
        out[d] = np.asarray(al)
    return out


def volr_alarms(M, sess, which="bern"):
    t, uc = bins(sess.units.ts.values, sess.t_end, segs=sess.segs)
    _, pc = bins(sess.uplinks.ts.values, sess.t_end, segs=sess.segs)
    lam0 = np.maximum(M.volr_ratio * pc, 0.5)
    lam1 = 1.5 * lam0
    h = max(math.log(SPD / 30 * M.N / FP["budget_B_per_day"]), 1.0)
    s, al = 0.0, []
    for ti, x, a0, a1 in zip(t, uc, lam0, lam1):
        s = max(0.0, s + x * math.log(a1 / a0) - (a1 - a0))
        if s >= h:
            al.append(ti); s = 0.0
    return np.asarray(al)


def count_alarms(al: dict) -> int:
    return int(sum(len(v) for v in al.values()))


# ════════════════════════════════════════════════════════════════════════════
# Injections
# ════════════════════════════════════════════════════════════════════════════
def _units(sess, units):
    return replace(sess, units=units.sort_values("ts").reset_index(drop=True))


def credited_mask(sess, d):
    _, _, cdev = C.device_streams(sess, DT)
    return cdev == d


def inj_hide(sess, d, f, t0, t1, rng, policy="random", witness_C=None):
    m = credited_mask(sess, d) & (sess.units.ts.values >= t0) & (sess.units.ts.values <= t1)
    idx = np.where(m)[0]
    if policy == "random":
        drop = idx[rng.random(len(idx)) < f]
        return _units(sess, sess.units.drop(sess.units.index[drop])), len(drop), len(idx)
    # masked hiding: drop a unit only if the device's unmatched count does not rise
    view = sess
    if witness_C is not None:
        # hub decides on its full view; witness later sees a thinned view
        pass
    units = sess.units.copy()
    dropped = 0
    up = view.uplinks
    devmask = up.dev.values == d
    base = rv.match_units(up.ts.values, units.ts.values, DT)[0][devmask].sum()
    for k in idx:
        if rng.random() >= f:
            continue
        trial = units.drop(index=sess.units.index[k], errors="ignore")
        x = rv.match_units(up.ts.values, trial.ts.values, DT)[0][devmask].sum()
        if x <= base:
            units, base, dropped = trial, x, dropped + 1
    out = _units(sess, units)
    if witness_C is not None:
        keep = rng.random(len(sess.uplinks)) < witness_C
        keep |= sess.uplinks.dev.values != d
        out = replace(out, uplinks=sess.uplinks[keep].reset_index(drop=True))
    return out, dropped, len(idx)


def inj_delay(sess, d, delay, t0, t1):
    m = credited_mask(sess, d) & (sess.units.ts.values >= t0) & (sess.units.ts.values <= t1)
    u = sess.units.copy(); u.loc[u.index[m], "ts"] += delay
    return _units(sess, u)


def inj_change(sess, d, nbytes, t0, t1):
    m = credited_mask(sess, d) & (sess.units.ts.values >= t0) & (sess.units.ts.values <= t1)
    u = sess.units.copy(); u.loc[u.index[m], "length"] += nbytes
    return _units(sess, u)


def inj_swap(sess, d, d2, t0, t1, rng):
    _, _, cdev = C.device_streams(sess, DT)
    u = sess.units.copy()
    w = (u.ts.values >= t0) & (u.ts.values <= t1)
    for a, b in ((d, d2), (d2, d)):
        ma, Lb = (cdev == a) & w, u.length.values[cdev == b]
        if len(Lb) and ma.any():
            u.loc[u.index[ma], "length"] = rng.choice(Lb, ma.sum())
    return _units(sess, u)


def inj_invent(sess, rate, t0, t1, rng):
    n = rng.poisson(rate * (t1 - t0))
    add = pd.DataFrame(dict(ts=np.sort(rng.uniform(t0, t1, n)),
                            length=rng.choice(sess.units.length.values, n), flow="invent"))
    return _units(sess, pd.concat([sess.units, add], ignore_index=True))


def inj_chaff(sess, frac, t0, t1, rng):
    m = (sess.units.ts.values >= t0) & (sess.units.ts.values <= t1)
    rate = m.sum() / max(t1 - t0, 1) * frac
    return inj_invent(sess, rate, t0, t1, rng)


def inj_leak(sess, rate, t0, t1, rng, ep=("203.0.113.9", 443)):
    n = rng.poisson(rate * (t1 - t0))
    L = sess.egress.length.values if len(sess.egress) else np.array([100])
    add = pd.DataFrame(dict(ts=np.sort(rng.uniform(t0, t1, n)), remote=ep[0], port=ep[1],
                            length=rng.choice(L, n)))
    return replace(sess, egress=pd.concat([sess.egress, add], ignore_index=True)
                   .sort_values("ts").reset_index(drop=True))


def inj_forge(sess, d, rate, t0, t1, rng, ack="host", ack_us=600.0, band=None,
              counter="random"):
    """Forged data frames under device d's address to the coordinator.
    ack: 'host' (gap ack_us), 'exact' (band centre), 'none'.
    counter: 'random' or 'continue' (A4: next MAC seq and next 32-bit counter)."""
    fr = sess.frames
    mine = fr[(fr.ftype == 1) & (fr.src == d) & (fr.dst == COORD)].sort_values("ts")
    if mine.empty:
        mine = fr[(fr.ftype == 1) & (fr.nwk_src == d) & (fr.dst == COORD)].sort_values("ts")
    if mine.empty:
        return sess, 0, []
    n = max(1, rng.poisson(rate * (t1 - t0)))
    times = np.sort(rng.uniform(t0, t1, n))
    rows, acks, forged = [], [], []
    for t in times:
        prev = mine[mine.ts <= t]
        base = (prev.iloc[-1] if len(prev) else mine.iloc[0]).to_dict()
        r = dict(base)
        r["ts"] = float(t)
        r["psdu_hash"] = f"forged{rng.integers(1 << 60):x}"
        r["src"] = d
        r["ack_req"] = True
        if counter == "continue":
            r["seq"] = (int(base["seq"]) + 1) % 256
            r["fc32"] = (base["fc32"] + 1) if pd.notna(base["fc32"]) else None
        else:
            r["seq"] = int(rng.integers(0, 256))
            r["fc32"] = int(rng.integers(0, 2 ** 32))
        r["nwk_seq"] = int(rng.integers(0, 256))
        rows.append(r); forged.append(r["ts"])
        if ack != "none":
            gap = ack_us if ack == "host" else (band["centre"] if band else 190.9)
            acks.append(dict(ts=t + (AIR(r["length"]) + gap) / 1e6, ftype=2, seq=r["seq"], src=None, dst=None,
                             pan=None, ack_req=False, mac_sec=False, length=5, nwk_src=None, nwk_dst=None,
                             nwk_type=None, nwk_seq=None, fc32=None, sec_src=None, psdu_hash="ack"))
    new = pd.concat([fr, pd.DataFrame(rows), pd.DataFrame(acks)], ignore_index=True)
    new = new.sort_values("ts", kind="stable").reset_index(drop=True)
    return replace(sess, frames=new, uplinks=C.eligible_uplinks(new)), len(rows), forged


def inj_ack_withhold(sess, d, t0, t1, rng):
    """Hub withholds ACKs for d's frames in the window; d retransmits up to 3x."""
    fr = sess.frames.sort_values("ts").reset_index(drop=True)
    w = (fr.ts >= t0) & (fr.ts <= t1)
    tgt = fr[w & (fr.ftype == 1) & (fr.src == d) & (fr.dst == COORD) & fr.ack_req]
    drop = []
    for i in tgt.index:
        if i + 1 < len(fr) and fr.at[i + 1, "ftype"] == 2 and fr.at[i + 1, "seq"] == fr.at[i, "seq"]:
            drop.append(i + 1)
    retx = []
    for i in tgt.index:
        r = fr.loc[i].to_dict()
        for k in range(1, 4):
            q = dict(r); q["ts"] = r["ts"] + k * (0.000864 + rng.uniform(0, 0.0025)); retx.append(q)
    new = pd.concat([fr.drop(index=drop), pd.DataFrame(retx)], ignore_index=True)
    new = new.sort_values("ts", kind="stable").reset_index(drop=True)
    return replace(sess, frames=new)


def inj_witness_loss(sess, keep_frac, t0, t1, rng):
    fr = sess.frames
    w = (fr.ts >= t0) & (fr.ts <= t1)
    keep = ~w | (rng.random(len(fr)) < keep_frac)
    new = fr[keep].reset_index(drop=True)
    return replace(sess, frames=new, uplinks=C.eligible_uplinks(new))


def inj_silence(sess, d, t0, t1):
    fr = sess.frames
    w = (fr.ts >= t0) & (fr.ts <= t1) & ((fr.src == d) | (fr.nwk_src == d))
    new = fr[~w].reset_index(drop=True)
    m = credited_mask(sess, d) & (sess.units.ts.values >= t0) & (sess.units.ts.values <= t1)
    s2 = replace(sess, frames=new, uplinks=C.eligible_uplinks(new))
    return _units(s2, sess.units[~m])


def inj_known_leak(sess, rate, t0, t1, rng, ep):
    return inj_leak(sess, rate, t0, t1, rng, ep)
