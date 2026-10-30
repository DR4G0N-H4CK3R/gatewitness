"""GateWitness on ZIOTP2025: evidence streams per home and fold.

Builds, for a list of paired captures, one continuous session (pairs rebased
and concatenated with 30 s gaps): eligible uplinks per device, reporting
units, egress records and raw frames.  Statistics, calibration and CUSUM come
from gw_stats (the same code that fills the paper's macros).
"""
from __future__ import annotations

import pickle
import sys
from dataclasses import dataclass, field, replace
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).parent
sys.path.insert(0, str(HERE))
import gw_ip                                   # noqa: E402
import gw_stats as rv                # noqa: E402

COORD = 0x0000
GAP = 30.0
DT = 1.0


def collapse_fanout(units: pd.DataFrame, tol: float = 0.05) -> pd.DataFrame:
    """One event pushed to several subscribed clients is one unit: drop a
    message that has the same length as a message on another flow within tol."""
    u = units.sort_values("ts").reset_index(drop=True)
    keep = np.ones(len(u), bool)
    ts, ln, fl = u.ts.values, u.length.values, u.flow.values
    j = 0
    for i in range(len(u)):
        while ts[i] - ts[j] > tol:
            j += 1
        for k in range(j, i):
            if keep[k] and ln[k] == ln[i] and fl[k] != fl[i]:
                keep[i] = False
                break
    return u[keep].reset_index(drop=True)


def load_ip_cached(pair):
    f = HERE / "cache" / (pair["name"].replace("/", "_") + "_ip.pkl")
    if f.exists():
        return pickle.loads(f.read_bytes())
    rec, st = gw_ip.records(Path(pair["ip_path"]))
    f.write_bytes(pickle.dumps((rec, st)))
    return rec, st


@dataclass
class Session:
    name: str
    frames: pd.DataFrame            # raw radio frames (rebased, concatenated)
    uplinks: pd.DataFrame           # ts, dev (eligible uplinks)
    units: pd.DataFrame             # ts, length, flow (reporting units)
    egress: pd.DataFrame            # non-websocket outbound records: ts, remote, port, length
    segs: list = field(default_factory=list)   # (pair name, start, end)

    @property
    def t_end(self):
        return self.segs[-1][2] if self.segs else 0.0


def eligible_uplinks(fr: pd.DataFrame) -> pd.DataFrame:
    """Distinct device-originated NWK data frames into the coordinator: MAC
    retransmissions (bit-identical) and relay hops ((originator, NWK seq)
    within 2 s) removed."""
    d = fr[(fr.ftype == 1) & (fr.dst == COORD) & (fr.nwk_dst == COORD) & (fr.nwk_type == 0)
           & fr.nwk_src.notna() & (fr.nwk_src != COORD)]
    d = d.sort_values("ts")
    last_psdu, last_nwk, keep = {}, {}, []
    for r in d.itertuples():
        k1 = (r.src, r.seq, r.psdu_hash)
        if k1 in last_psdu and r.ts - last_psdu[k1] <= 0.5:
            continue
        last_psdu[k1] = r.ts
        k2 = (int(r.nwk_src), int(r.nwk_seq))
        if k2 in last_nwk and r.ts - last_nwk[k2] <= 2.0:
            continue
        last_nwk[k2] = r.ts
        keep.append((r.ts, int(r.nwk_src)))
    return pd.DataFrame(keep, columns=["ts", "dev"])


def build_session(pairs: list, name: str = "") -> Session:
    F, U, E, segs, off = [], [], [], [], 0.0
    for p in pairs:
        fr = p["frames"].copy()
        rec, _ = load_ip_cached(p)
        t0 = fr.ts.min()
        dur = max(fr.ts.max(), rec.ts.max()) - t0 + 1.0
        fr["ts"] = fr.ts - t0 + off
        rec = rec.copy(); rec["ts"] = rec.ts - t0 + off
        units = collapse_fanout(gw_ip.reporting_units(rec))
        F.append(fr)
        U.append(units[["ts", "length", "flow"]])
        E.append(rec[(rec.direction == "out") & (rec.kind == "seg")][["ts", "remote", "port", "length"]])
        segs.append((p["name"], off, off + dur))
        off += dur + GAP
    frames = pd.concat(F, ignore_index=True).sort_values("ts", kind="stable").reset_index(drop=True)
    return Session(name, frames, eligible_uplinks(frames),
                   pd.concat(U, ignore_index=True).sort_values("ts").reset_index(drop=True),
                   pd.concat(E, ignore_index=True).sort_values("ts").reset_index(drop=True), segs)


# ── matching ───────────────────────────────────────────────────────────────
def match(sess: Session, units_ts=None, dt=DT):
    """Global one-to-one matching; returns (X per uplink, credit per unit)."""
    ut = sess.uplinks.ts.values
    un = sess.units.ts.values if units_ts is None else units_ts
    return rv.match_units(ut, un, dt)


def device_streams(sess: Session, dt=DT):
    X, credit = match(sess, dt=dt)
    dev = sess.uplinks.dev.values
    cdev = np.where(credit >= 0, dev[np.maximum(credit, 0)], -1)
    return X, credit, cdev


def channel_stats(sess: Session, devices=None, dt=DT) -> pd.DataFrame:
    """eps, 1-gamma, s per device by self-injection (paper Section IV)."""
    X, credit, cdev = device_streams(sess, dt)
    dev = sess.uplinks.dev.values
    rows = []
    for d in (devices if devices is not None else np.unique(dev)):
        m = dev == d
        if m.sum() < 10:
            continue
        eps = X[m].mean()
        keep = cdev != d
        X2, _ = rv.match_units(sess.uplinks.ts.values, sess.units.ts.values[keep], dt)
        s = X2[m].mean() - eps
        rows.append(dict(dev=d, n=int(m.sum()), eps=eps, s=s,
                         one_minus_gamma=s / (1 - eps) if eps < 1 else np.nan,
                         rate=m.sum() / max(sess.t_end, 1)))
    return pd.DataFrame(rows)


def commissioning(sess: Session, M=499, seed=0):
    rng = np.random.default_rng(seed)
    up = np.sort(sess.uplinks.ts.values)
    eps = {"ws_push": sess.units.ts.values}
    for (rem, port), g in sess.egress.groupby(["remote", "port"]):
        if len(g) >= 10:
            eps[f"{rem}:{port}"] = g.ts.values
    return rv.commissioning_test(up, eps, DT, M=M, rng=rng)
