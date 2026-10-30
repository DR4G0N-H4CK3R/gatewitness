#!/usr/bin/env python3
"""gw_stats.py -- interval statistics, CUSUM calibration and Fig. 5 for the paper.

Writes <out_dir>/gw_results.json and <tex_dir>/gw_results.tex (lines of
\\gwset{key}{value}), plus <tex_dir>/figs/fig5_matched_coldstart.pdf.

Subcommands
  selftest   validate the statistics, calibration, parsers and pipelines on
             synthetic data (run this first; no datasets needed)
  stats      exact k/n, Clopper-Pearson and per-fold ranges from episodes.csv
  markov     Markov-exact Brook-Evans calibration on exported indicator streams
  counters   32-bit NWK frame-counter evidence + forge-shortfall diagnostics
  songlin    frozen run on the Song & Lin 25-day paired dataset
  sweeps     SIZE bands, delay sweep, CSMA KS, A7', A10, radio-vs-hub
  figs       Fig. 5 (matched-rate curves + cold start)
  crosscheck compare reference tests with the verifier's exports (run before songlin)
  all        everything above that has inputs present, then write macros

Inputs (paths in a JSON config, see DEFAULT_CONFIG; every input is optional and
a subcommand whose inputs are missing is skipped with a message):

  episodes.csv        one row per scored episode, exported by the verifier
      experiment,dataset,home,fold,seed,attack,device,detected,placebo,latency_s
      `experiment` keys the table row (e.g. hide, lh08, forgeX, a1, a4_32_no).
  indicators.csv      benign indicator streams per test, per fold
      home,fold,split,test_id,kind,t,x      split in {commission,heldout}
      kind in {bernoulli,poisson}; x is 0/1 (bernoulli) or a count (poisson)
      optional extra columns design_eps,design_p1 override the design point.
  episode_indicators.csv  attacked streams for detection under new thresholds
      episode_id,experiment,home,fold,test_id,t,x,t_start,t_end
  sweep.csv           threshold sweeps for the matched-rate comparison
      dataset,detector,attack,threshold,benign_alarms,exposure_days,det,placebo
  coldstart.csv       home,minutes,prior,detection,benign_alarms,exposure_days
  forge_frames.csv    labelled frames of the 8-bit forge episodes
      episode_id,t,mac_src,mac_seq,psdu_hex,forged,device,t_end
  Raw captures for `counters`, `songlin`, `sweeps`: 802.15.4 pcap/pcapng
  (linktypes 195, 230, 283) and Ethernet/Linux-cooked pcap/pcapng for the hub.

The reference tests in this file (SUP, SIZE, FAB, EMIT counter, NEWEP) follow
Sections IV-V of the paper but are not the verifier. Before the frozen run,
`crosscheck` compares the reference SUP/SIZE/FAB indicator streams with the
verifier's exported indicators.csv on the same ZIOTP2025 pairs and exits non-zero
if they disagree on more than 1% of units. Tests not re-implemented here (VOL,
LIVE, VOLR, COVERAGE, EMIT ACK timing on Song-Lin) must come from the verifier;
the key `sl_tests_covered` records which tests the Song-Lin alarm rate includes.
"""
from __future__ import annotations

import argparse
import dataclasses
import hashlib
import json
import math
import re
import os
import struct
import subprocess
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

try:
    import pandas as pd
    from scipy import sparse, stats
    from scipy.sparse.linalg import spsolve
except ImportError as e:  # pragma: no cover
    sys.exit(f"needs numpy, scipy, pandas, matplotlib: {e}")

DEFAULT_CONFIG = {
    "out_dir": "results",
    "tex_dir": ".",
    "episodes": "data/episodes.csv",
    "indicators": "data/indicators.csv",
    "episode_indicators": "data/episode_indicators.csv",
    "sweep": "data/sweep.csv",
    "coldstart": "data/coldstart.csv",
    "forge_frames": "data/forge_frames.csv",
    "ziotp_radio": [],            # list of 802.15.4 pcaps (benign) for counters
    "nrf_radio": [],              # optional nRF52840 captures (ACK spread)
    "songlin": {
        "radio": "songlin/ha_zigbee_ch15.pcap",
        "ethernet": "songlin/ha_ethernet.pcap",
        "replay": "songlin/replay_sniffed.pcap",
        "hub_ip": None,           # inferred if None
        "coordinator": 0x0000,
    },
    "params": {                   # frozen parameters (paper Section VII)
        "dt": 1.0, "B": 1.0, "f_star": 0.5, "ack_halfwidth_us": 10.0,
        "budgets": [1, 10, 100, 1000], "commission_min": 60.0,
        "episodes_per_class": 30, "seed": 20261029, "score_tail_s": 60.0,
        "size_band": [0.5, 99.5], "beta_upper": 0.90, "lattice_res": 64,
        "phi_max": 8.0, "dedup_window_s": 0.5, "markov_min_ones": 20,
    },
    "frozen_commit": None,        # filled automatically from git if available
    "artifact_url": None,         # anonymous artifact link for the paper
}

SECONDS_PER_DAY = 86400.0


# ════════════════════════════════════════════════════════════════════════════
# 1. Interval statistics
# ════════════════════════════════════════════════════════════════════════════
def clopper_pearson(k: int, n: int, conf: float = 0.95) -> tuple[float, float]:
    """Exact binomial interval."""
    if n == 0:
        return (float("nan"), float("nan"))
    a = 1 - conf
    lo = 0.0 if k == 0 else stats.beta.ppf(a / 2, k, n - k + 1)
    hi = 1.0 if k == n else stats.beta.ppf(1 - a / 2, k + 1, n - k)
    return float(lo), float(hi)


def garwood(k: int, exposure_days: float, conf: float = 0.95) -> tuple[float, float, float]:
    """Rate per day with exact Poisson (Garwood) interval."""
    a = 1 - conf
    lo = 0.0 if k == 0 else stats.chi2.ppf(a / 2, 2 * k) / 2
    hi = stats.chi2.ppf(1 - a / 2, 2 * k + 2) / 2
    return k / exposure_days, lo / exposure_days, hi / exposure_days


def fmt_rate(k: int, exposure_days: float) -> str:
    r, lo, hi = garwood(k, exposure_days)
    f = lambda v: f"{v:.0f}" if v >= 10 or v == 0 else f"{v:.1f}"
    return f"{f(r)} [{f(lo)}, {f(hi)}]"


def fmt_ci(lo: float, hi: float) -> str:
    return f"[{lo:.2f}, {hi:.2f}]"


# ════════════════════════════════════════════════════════════════════════════
# 2. CUSUM, Brook-Evans (Bernoulli and Markov-exact), dispersion
# ════════════════════════════════════════════════════════════════════════════
def beta_upper(k: int, n: int, q: float = 0.90) -> float:
    """Upper credible bound of a Beta(1,1)-prior posterior.
    VERIFY: match the prior your verifier uses."""
    return float(stats.beta.ppf(q, k + 1, n - k + 1))


def cusum_increments(eps: float, p1: float) -> tuple[float, float]:
    return math.log(p1 / eps), math.log((1 - p1) / (1 - eps))


def run_cusum(x: np.ndarray, l1: float, l0: float, h: float, restart=True) -> list[int]:
    """Indices at which the Bernoulli CUSUM alarms (restarting at 0)."""
    s, alarms = 0.0, []
    for i, xi in enumerate(x):
        s = max(0.0, s + (l1 if xi else l0))
        if s >= h:
            alarms.append(i)
            if restart:
                s = 0.0
    return alarms


def run_cusum_poisson(x: np.ndarray, lam0: float, lam1: float, h: float) -> list[int]:
    s, alarms = 0.0, []
    a, b = math.log(lam1 / lam0), lam1 - lam0
    for i, xi in enumerate(x):
        s = max(0.0, s + xi * a - b)
        if s >= h:
            alarms.append(i)
            s = 0.0
    return alarms


def markov_fit(x: np.ndarray, pseudo: float = 0.5) -> tuple[float, float]:
    """a = P(1|0), b = P(1|1) with add-`pseudo` smoothing."""
    x = np.asarray(x, dtype=int)
    prev, nxt = x[:-1], x[1:]
    n0, n1 = np.sum(prev == 0), np.sum(prev == 1)
    a = (np.sum((prev == 0) & (nxt == 1)) + pseudo) / (n0 + 2 * pseudo)
    b = (np.sum((prev == 1) & (nxt == 1)) + pseudo) / (n1 + 2 * pseudo)
    return float(a), float(b)


def markov_fit_upper(x: np.ndarray, q: float = 0.90) -> tuple[float, float]:
    """Upper credible bounds on (a, b) under Beta(1,1) priors.  Raising either
    transition probability makes 1s more frequent and more persistent, which
    can only shorten the CUSUM run length (monotone coupling of the two-state
    chain), so thresholds computed from the upper bounds are conservative."""
    x = np.asarray(x, dtype=int)
    prev, nxt = x[:-1], x[1:]
    n0, n1 = int(np.sum(prev == 0)), int(np.sum(prev == 1))
    k0 = int(np.sum((prev == 0) & (nxt == 1)))
    k1 = int(np.sum((prev == 1) & (nxt == 1)))
    return beta_upper(k0, n0, q), beta_upper(k1, n1, q)


def arl_lattice(l1: float, l0: float, m: int, w: float, a: float, b: float) -> float:
    """Average run length of S_i=max(0,S_{i-1}+inc), alarm at S>=m*w, for a
    two-state Markov indicator chain (a=P(1|0), b=P(1|1)); a==b is Bernoulli.

    Increments are rounded UP to multiples of w.  The recursion is monotone in
    its increments, so the lattice path dominates the true path and the
    computed ARL is a lower bound on the true ARL (conservative threshold).
    Returns the minimum over the initial indicator."""
    k1 = math.ceil(l1 / w)
    k0 = math.ceil(l0 / w)          # l0 < 0: rounds toward zero (conservative)
    n = 2 * m                        # states (k, xprev), k in [0, m)
    rows, cols, vals = [], [], []
    for xp in (0, 1):
        p = a if xp == 0 else b
        for k in range(m):
            i = xp * m + k
            k_up = max(0, k + k1)
            if k_up < m:
                rows.append(i); cols.append(1 * m + k_up); vals.append(p)
            k_dn = max(0, k + k0)
            rows.append(i); cols.append(0 * m + k_dn); vals.append(1 - p)
    Q = sparse.csr_matrix((vals, (rows, cols)), shape=(n, n))
    A = sparse.identity(n, format="csr") - Q
    L = spsolve(A.tocsc(), np.ones(n))
    return float(min(L[0], L[m]))


def threshold_for_arl(l1: float, l0: float, target: float, a: float, b: float,
                      res: int = 64, max_states: int = 40_000) -> tuple[float, float]:
    """Smallest lattice threshold h with ARL >= target.  Returns (h, ARL).
    If the lattice would exceed `max_states` cells, the cell width is doubled
    and the search restarts; a coarser lattice rounds increments up further,
    so the threshold stays conservative."""
    w = l1 / res
    while True:
        lo, hi = 1, 2
        ok = True
        while arl_lattice(l1, l0, hi, w, a, b) < target:
            lo, hi = hi, hi * 2
            if hi > max_states:
                ok = False
                break
        if ok:
            break
        if w >= l1 * 4:
            raise RuntimeError("ARL target unreachable; check eps/p1")
        w *= 2
    while hi - lo > 1:
        mid = (lo + hi) // 2
        if arl_lattice(l1, l0, mid, w, a, b) >= target:
            hi = mid
        else:
            lo = mid
    return hi * w, arl_lattice(l1, l0, hi, w, a, b)


def dispersion_phi(x: np.ndarray, l1: float, l0: float, batch: int = 50,
                   phi_max: float = 8.0) -> float:
    """Long-run variance inflation of CUSUM increments via batch means."""
    inc = np.where(np.asarray(x) > 0, l1, l0)
    nb = len(inc) // batch
    if nb < 5:
        return 1.0
    sums = inc[: nb * batch].reshape(nb, batch).sum(axis=1)
    v_iid = inc.var(ddof=1) * batch
    if v_iid <= 0:
        return 1.0
    return float(np.clip(sums.var(ddof=1) / v_iid, 1.0, phi_max))


def monte_carlo_arl(l1, l0, h, a, b, reps=2000, rng=None, x0=None) -> float:
    rng = rng or np.random.default_rng(1)
    total = 0
    for _ in range(reps):
        s, xp, t = 0.0, (x0 if x0 is not None else int(rng.random() < a / (1 - b + a))), 0
        while True:
            t += 1
            xp = int(rng.random() < (a if xp == 0 else b))
            s = max(0.0, s + (l1 if xp else l0))
            if s >= h:
                break
        total += t
    return total / reps


@dataclasses.dataclass
class Calibrated:
    test_id: str
    kind: str
    eps: float
    p1: float
    h_bern: float
    h_disp: float
    h_markov: float | None
    phi: float
    a: float
    b: float
    units_per_day: float


def calibrate_test(test_id, kind, x_comm, t_comm, n_tests, B, P, design=None) -> Calibrated:
    """Three calibrations for one test from its commissioning stream."""
    x_comm = np.asarray(x_comm)
    dur = max(t_comm[-1] - t_comm[0], 1.0) if len(t_comm) > 1 else 1.0
    u_d = len(x_comm) / dur * SECONDS_PER_DAY
    target = u_d * n_tests / B
    if kind == "poisson":
        lam0 = max(x_comm.mean(), 1e-3)
        lam1 = 2 * lam0
        # Poisson tests keep the dispersion correction (paper Section IV)
        h = poisson_threshold(lam0, lam1, target)
        a_ = math.log(lam1 / lam0)
        phi = dispersion_phi((x_comm * a_ - (lam1 - lam0)) > 0, a_, -(lam1 - lam0),
                             phi_max=P["phi_max"])
        return Calibrated(test_id, kind, lam0, lam1, h, h * phi, None, phi, 0, 0, u_d)
    k, n = int(x_comm.sum()), len(x_comm)
    if design is not None:
        eps, p1 = design
    else:
        eps = beta_upper(k, n, P["beta_upper"])
        eps = min(max(eps, 1e-4), 0.95)
        # Non-SUP tests: each event fully informative (s = 1 - eps).
        # VERIFY: use your verifier's design point (export design_eps/design_p1).
        p1 = min(eps + P["f_star"] * (1 - eps), 0.999)
    l1, l0 = cusum_increments(eps, p1)
    h_b, _ = threshold_for_arl(l1, l0, target, eps, eps, P["lattice_res"])
    phi = dispersion_phi(x_comm, l1, l0, phi_max=P["phi_max"])
    a, b = markov_fit_upper(x_comm, P["beta_upper"])
    if k < P.get("markov_min_ones", 20):
        # Too few benign 1s to estimate persistence; dependence cannot be
        # assessed, so Markov-exact falls back to Bernoulli-exact.
        h_m = h_b
    else:
        try:
            h_m, _ = threshold_for_arl(l1, l0, target, a, b, P["lattice_res"])
        except RuntimeError:
            h_m = None
    return Calibrated(test_id, kind, eps, p1, h_b, h_b * phi, h_m, phi, a, b, u_d)


def poisson_threshold(lam0, lam1, target, reps=400, rng=None) -> float:
    """First-order (Wald) Poisson CUSUM threshold, h ~ ln(target ARL).
    VERIFY against your verifier's VOL threshold; VOL keeps the dispersion
    correction on top, as in the paper."""
    return max(math.log(target), 1.0)


# ════════════════════════════════════════════════════════════════════════════
# 3. Capture reading and IEEE 802.15.4 / Zigbee NWK parsing
# ════════════════════════════════════════════════════════════════════════════
def read_capture(path: str):
    """Yield (timestamp_s, linktype, bytes) from pcap or pcapng."""
    with open(path, "rb") as f:
        magic = f.read(4)
        f.seek(0)
        if magic in (b"\xd4\xc3\xb2\xa1", b"\xa1\xb2\xc3\xd4",
                     b"\x4d\x3c\xb2\xa1", b"\xa1\xb2\x3c\x4d"):
            yield from _read_pcap(f)
        elif magic == b"\x0a\x0d\x0d\x0a":
            yield from _read_pcapng(f)
        else:
            raise ValueError(f"{path}: not pcap/pcapng")


def _read_pcap(f):
    hdr = f.read(24)
    m = hdr[:4]
    endian = "<" if m in (b"\xd4\xc3\xb2\xa1", b"\x4d\x3c\xb2\xa1") else ">"
    nano = m in (b"\x4d\x3c\xb2\xa1", b"\xa1\xb2\x3c\x4d")
    linktype = struct.unpack(endian + "I", hdr[20:24])[0]
    div = 1e9 if nano else 1e6
    while True:
        rh = f.read(16)
        if len(rh) < 16:
            return
        sec, frac, incl, _ = struct.unpack(endian + "IIII", rh)
        yield sec + frac / div, linktype, f.read(incl)


def _read_pcapng(f):
    endian, ifaces = "<", []
    while True:
        bh = f.read(8)
        if len(bh) < 8:
            return
        btype, blen = struct.unpack(endian + "II", bh)
        if btype == 0x0A0D0D0A:
            bom = f.read(4)
            endian = "<" if bom == b"\x4d\x3c\x2b\x1a" else ">"
            blen = struct.unpack(endian + "I", bh[4:8])[0]
            f.read(blen - 12)
            ifaces = []
            continue
        body = f.read(blen - 8)
        if btype == 1:  # IDB
            lt = struct.unpack(endian + "H", body[:2])[0]
            tsres = 1e-6
            opts = body[8:-4]
            i = 0
            while i + 4 <= len(opts):
                code, ln = struct.unpack(endian + "HH", opts[i:i + 4])
                if code == 0:
                    break
                if code == 9 and ln >= 1:
                    v = opts[i + 4]
                    tsres = 2.0 ** -(v & 0x7F) if v & 0x80 else 10.0 ** -v
                i += 4 + ((ln + 3) & ~3)
            ifaces.append((lt, tsres))
        elif btype == 6:  # EPB
            iid, th, tl, cap = struct.unpack(endian + "IIII", body[:16])
            lt, res = ifaces[iid]
            yield ((th << 32) | tl) * res, lt, body[20:20 + cap]
        elif btype == 3:  # SPB
            lt, _ = ifaces[0]
            yield float("nan"), lt, body[4:]


def strip_154(linktype: int, pkt: bytes) -> bytes | None:
    """Return the MAC frame without FCS."""
    if linktype == 195:            # IEEE802_15_4_WITHFCS
        return pkt[:-2]
    if linktype in (230, 215):     # NOFCS / NONASK_PHY (no FCS assumed)
        return pkt if linktype == 230 else pkt[6:]
    if linktype == 283:            # IEEE802_15_4_TAP
        tlen = struct.unpack("<H", pkt[2:4])[0]
        return pkt[tlen:-2] if len(pkt) > tlen + 2 else None
    return None


@dataclasses.dataclass
class Frame:
    t: float
    ftype: int
    seq: int
    dst: int | None
    src: int | None
    pan: int | None
    ack_req: bool
    psdu: bytes
    nwk_src: int | None = None
    nwk_dst: int | None = None
    nwk_seq: int | None = None
    fc32: int | None = None
    sec_src: int | None = None   # extended source in NWK aux header

    @property
    def length(self):
        return len(self.psdu) + 2  # + FCS


def parse_mac(t: float, mac: bytes) -> Frame | None:
    if len(mac) < 3:
        return None
    fcf = struct.unpack("<H", mac[:2])[0]
    ftype = fcf & 0x7
    ack_req = bool(fcf & (1 << 5))
    panc = bool(fcf & (1 << 6))
    dmode, smode = (fcf >> 10) & 3, (fcf >> 14) & 3
    seq = mac[2]
    i, pan, dst, src = 3, None, None, None
    try:
        if dmode:
            pan = struct.unpack("<H", mac[i:i + 2])[0]; i += 2
            if dmode == 2:
                dst = struct.unpack("<H", mac[i:i + 2])[0]; i += 2
            else:
                dst = struct.unpack("<Q", mac[i:i + 8])[0]; i += 8
        if smode:
            if not panc:
                pan = struct.unpack("<H", mac[i:i + 2])[0]; i += 2
            if smode == 2:
                src = struct.unpack("<H", mac[i:i + 2])[0]; i += 2
            else:
                src = struct.unpack("<Q", mac[i:i + 8])[0]; i += 8
    except struct.error:
        return None
    fr = Frame(t, ftype, seq, dst, src, pan, ack_req, bytes(mac))
    if ftype == 1 and not (fcf & (1 << 3)):   # data frame, no MAC security
        parse_nwk(fr, mac[i:])
    return fr


def parse_nwk(fr: Frame, p: bytes) -> None:
    """Zigbee NWK header and auxiliary security header (frame counter)."""
    if len(p) < 8:
        return
    fc = struct.unpack("<H", p[:2])[0]
    if (fc & 0x3) not in (0, 1):   # data or command
        return
    fr.nwk_dst, fr.nwk_src = struct.unpack("<HH", p[2:6])
    fr.nwk_seq = p[7]
    i = 8
    if fc & (1 << 11):  # destination IEEE
        i += 8
    if fc & (1 << 12):  # source IEEE
        i += 8
    if fc & (1 << 8):   # multicast control
        i += 1
    if fc & (1 << 10):  # source route subframe
        if len(p) < i + 2:
            return
        relays = p[i]
        i += 2 + 2 * relays
    if fc & (1 << 9):   # security: auxiliary header
        if len(p) < i + 5:
            return
        sc = p[i]
        fr.fc32 = struct.unpack("<I", p[i + 1:i + 5])[0]
        if sc & (1 << 5) and len(p) >= i + 13:   # extended nonce: source addr
            fr.sec_src = struct.unpack("<Q", p[i + 5:i + 13])[0]


def load_frames(paths) -> list[Frame]:
    out = []
    for path in paths:
        for t, lt, pkt in read_capture(path):
            mac = strip_154(lt, pkt)
            if mac is None:
                continue
            fr = parse_mac(t, mac)
            if fr is not None:
                out.append(fr)
    out.sort(key=lambda f: f.t)
    return out


def dedup_bit_identical(frames: list[Frame], window: float = 0.5) -> list[Frame]:
    """Drop MAC retransmissions: same MAC seq AND identical PSDU within window."""
    last: dict[tuple, float] = {}
    out = []
    for fr in frames:
        key = (fr.src, fr.seq, hashlib.blake2b(fr.psdu, digest_size=8).digest())
        if key in last and fr.t - last[key] <= window:
            continue
        last[key] = fr.t
        out.append(fr)
    return out


def dedup_seq8(frames: list[Frame], window: float = 0.5) -> list[Frame]:
    """The earlier rule: drop on (MAC src, 8-bit seq) alone."""
    last: dict[tuple, float] = {}
    out = []
    for fr in frames:
        key = (fr.src, fr.seq)
        if key in last and fr.t - last[key] <= window:
            continue
        last[key] = fr.t
        out.append(fr)
    return out


def counter_events(frames: list[Frame]) -> list[tuple[int, str]]:
    """Index and kind of 32-bit counter collisions on RAW frames.
    'repeat': same counter, different PSDU; 'back': counter decreased."""
    state: dict[int, tuple[int, bytes]] = {}
    ev = []
    for i, fr in enumerate(frames):
        if fr.fc32 is None:
            continue
        key = fr.sec_src if fr.sec_src is not None else (fr.pan or 0) << 16 | (fr.src or 0)
        if key in state:
            last_fc, last_psdu = state[key]
            if fr.fc32 == last_fc and fr.psdu != last_psdu:
                ev.append((i, "repeat"))
            elif fr.fc32 < last_fc:
                ev.append((i, "back"))
        if key not in state or fr.fc32 >= state[key][0]:
            state[key] = (fr.fc32, fr.psdu)
    return ev


AIR_US_PER_BYTE = 32.0     # 250 kb/s O-QPSK
PHY_OVERHEAD_BYTES = 6     # preamble 4 + SFD 1 + PHR 1


def ack_gaps_us(frames: list[Frame]) -> list[tuple[int, float]]:
    """(index of acked frame, gap minus payload air time) for each ACK that
    follows an ack-requesting data frame with matching sequence number.
    Assumes capture timestamps mark frame START; set GW_TS_END=1 if they mark
    the end. VERIFY for your sniffer."""
    ts_end = os.environ.get("GW_TS_END") == "1"
    out = []
    for i in range(len(frames) - 1):
        f, g = frames[i], frames[i + 1]
        if f.ftype == 1 and f.ack_req and g.ftype == 2 and g.seq == f.seq:
            gap = (g.t - f.t) * 1e6
            if not ts_end:
                gap -= (f.length + PHY_OVERHEAD_BYTES) * AIR_US_PER_BYTE
            out.append((i, gap))
    return out


def learn_ack_band(gaps: list[float], halfwidth: float) -> dict:
    g = np.asarray(gaps)
    centre = float(np.median(g))
    spread = float(g.max() - g.min()) if len(g) else float("nan")
    return {"centre_us": centre, "spread_us": spread,
            "admitted": bool(len(g) and spread <= halfwidth), "halfwidth_us": halfwidth}


# ════════════════════════════════════════════════════════════════════════════
# 4. IP side: hub reporting records
# ════════════════════════════════════════════════════════════════════════════
@dataclasses.dataclass
class Record:
    t: float
    length: int
    endpoint: tuple   # (remote ip, remote port, proto)
    outbound: bool


def _ip_payload(lt: int, pkt: bytes):
    if lt == 1:
        eth = struct.unpack("!H", pkt[12:14])[0]
        off = 14
        if eth == 0x8100:
            eth = struct.unpack("!H", pkt[16:18])[0]; off = 18
    elif lt == 113:
        eth = struct.unpack("!H", pkt[14:16])[0]; off = 16
    elif lt == 101:
        return pkt
    else:
        return None
    return pkt[off:] if eth == 0x0800 else None


def load_ip_segments(paths):
    """Yield (t, src, dst, sport, dport, proto, seq, payload) for IPv4 TCP/UDP."""
    for path in paths:
        for t, lt, pkt in read_capture(path):
            ip = _ip_payload(lt, pkt)
            if not ip or len(ip) < 20 or ip[0] >> 4 != 4:
                continue
            ihl = (ip[0] & 0xF) * 4
            proto = ip[9]
            src, dst = ".".join(map(str, ip[12:16])), ".".join(map(str, ip[16:20]))
            tot = struct.unpack("!H", ip[2:4])[0]
            l4 = ip[ihl:tot]
            if proto == 6 and len(l4) >= 20:
                sp, dp, seq = struct.unpack("!HHI", l4[:8])
                doff = (l4[12] >> 4) * 4
                yield t, src, dst, sp, dp, 6, seq, l4[doff:]
            elif proto == 17 and len(l4) >= 8:
                sp, dp = struct.unpack("!HH", l4[:4])
                yield t, src, dst, sp, dp, 17, 0, l4[8:]


def extract_records(paths, hub_ip: str | None):
    """Reporting records: TLS records when the stream is TLS (header in clear),
    otherwise one record per TCP segment / UDP datagram with payload.
    Returns (records, hub_ip)."""
    segs = list(load_ip_segments(paths))
    if hub_ip is None:
        cnt = defaultdict(int)
        for s in segs:
            cnt[s[1]] += 1
        hub_ip = max(cnt, key=cnt.get) if cnt else None
    streams = defaultdict(list)
    for t, src, dst, sp, dp, proto, seq, pl in segs:
        if not pl or hub_ip not in (src, dst):
            continue
        outbound = src == hub_ip
        remote = (dst, dp, proto) if outbound else (src, sp, proto)
        streams[(src, dst, sp, dp, proto)].append((t, seq, pl, remote, outbound))
    recs = []
    for key, lst in streams.items():
        proto = key[4]
        if proto == 6:
            lst.sort(key=lambda z: z[1])
            seen, buf, tbuf = set(), b"", []
            for t, seq, pl, remote, ob in lst:
                if seq in seen:
                    continue
                seen.add(seq)
                buf += pl
                tbuf.append((len(buf), t))
            if buf[:1] and buf[0] in (20, 21, 22, 23) and buf[1:2] == b"\x03":
                i = 0
                while i + 5 <= len(buf):
                    ln = struct.unpack("!H", buf[i + 3:i + 5])[0]
                    end = i + 5 + ln
                    tt = next((t for pos, t in tbuf if pos >= min(end, len(buf))), tbuf[-1][1])
                    if buf[i] == 23:
                        recs.append(Record(tt, 5 + ln, lst[0][3], lst[0][4]))
                    i = end
                continue
        for t, _, pl, remote, ob in lst:
            recs.append(Record(t, len(pl), remote, ob))
    recs.sort(key=lambda r: r.t)
    return recs, hub_ip


# ════════════════════════════════════════════════════════════════════════════
# 5. Matching, self-injection, commissioning, clock offset
# ════════════════════════════════════════════════════════════════════════════
def match_units(uplink_t: np.ndarray, unit_t: np.ndarray, dt: float):
    """Credit each unit one-to-one to the most recent unmatched uplink within dt
    before it.  Returns (X per uplink, credited uplink index per unit or -1)."""
    X = np.ones(len(uplink_t), dtype=int)
    credit = np.full(len(unit_t), -1)
    stack: list[int] = []
    j = 0
    for k, tu in enumerate(unit_t):
        while j < len(uplink_t) and uplink_t[j] <= tu:
            stack.append(j); j += 1
        while stack and tu - uplink_t[stack[-1]] > dt:
            stack.pop()
            stack.clear()  # everything older is also out of range
        if stack:
            i = stack.pop()
            X[i] = 0
            credit[k] = i
    return X, credit


def coupling_D(uplink_t, rec_t, dt, rng, n_ctrl=None) -> float:
    if len(uplink_t) == 0 or len(rec_t) == 0:
        return 0.0
    rec_t = np.sort(rec_t)
    def frac_followed(ts):
        idx = np.searchsorted(rec_t, ts, side="right")
        ok = idx < len(rec_t)
        nxt = np.where(ok, rec_t[np.minimum(idx, len(rec_t) - 1)], np.inf)
        return np.mean(nxt - ts <= dt)
    lo, hi = min(uplink_t.min(), rec_t.min()), max(uplink_t.max(), rec_t.max())
    ctrl = rng.uniform(lo, hi, n_ctrl or len(uplink_t))
    return float(frac_followed(uplink_t) - frac_followed(ctrl))


def commissioning_test(uplink_t, recs_by_ep: dict, dt=1.0, M=499, min_shift=30.0,
                       block=30.0, rng=None) -> dict:
    """Permutation test of Section V (circular shifts > ±min_shift)."""
    rng = rng or np.random.default_rng(0)
    T0, T1 = uplink_t.min(), uplink_t.max()
    span = T1 - T0
    Dobs = {e: coupling_D(uplink_t, r, dt, rng) for e, r in recs_by_ep.items()}
    e_star = max(Dobs, key=Dobs.get)
    D_star = Dobs[e_star]
    null = []
    for _ in range(M):
        sh = rng.uniform(min_shift, span - min_shift)
        u = T0 + ((uplink_t - T0 + sh) % span)
        null.append(max(coupling_D(np.sort(u), r, dt, rng) for r in recs_by_ep.values()))
    p = (1 + np.sum(np.asarray(null) >= D_star)) / (M + 1)
    # 30 s block bootstrap of D for the chosen endpoint
    blocks = np.floor((uplink_t - T0) / block).astype(int)
    ub = np.unique(blocks)
    boots = []
    r = recs_by_ep[e_star]
    for _ in range(200):
        pick = rng.choice(ub, len(ub))
        sel = np.concatenate([uplink_t[blocks == b_] for b_ in pick])
        boots.append(coupling_D(np.sort(sel), r, dt, rng))
    q05 = float(np.percentile(boots, 5))
    return {"endpoint": e_star, "D": D_star, "p": float(p), "q05": q05,
            "admitted": bool(p < 0.01 and q05 > 0.10)}


def clock_offset_fft(uplink_t, rec_t, dt=1.0, max_off=600.0, binw=0.01) -> float:
    """argmax_delta of P(record within (u+delta, u+delta+dt]) via FFT."""
    t0 = min(uplink_t.min(), rec_t.min()) - max_off
    t1 = max(uplink_t.max(), rec_t.max()) + max_off
    n = int(math.ceil((t1 - t0) / binw)) + 1
    nfft = 1 << (2 * n - 1).bit_length()
    u = np.zeros(nfft); r = np.zeros(nfft)
    np.add.at(u, ((uplink_t - t0) / binw).astype(int), 1)
    np.add.at(r, ((rec_t - t0) / binw).astype(int), 1)
    box = np.zeros(nfft); box[: max(1, int(dt / binw))] = 1
    rb = np.real(np.fft.ifft(np.fft.fft(r) * np.conj(np.fft.fft(box))))  # records in window starting at bin
    corr = np.real(np.fft.ifft(np.fft.fft(rb) * np.conj(np.fft.fft(u))))
    lags = np.arange(nfft); lags[lags > nfft // 2] -= nfft
    lag_s = lags * binw
    ok = np.abs(lag_s) <= max_off
    return float(lag_s[ok][np.argmax(corr[ok])])


# ════════════════════════════════════════════════════════════════════════════
# 6. Reference tests and injections (Sections V-VI)
# ════════════════════════════════════════════════════════════════════════════
@dataclasses.dataclass
class Site:
    """Paired observations for one site after alignment."""
    uplinks: dict          # device -> np.ndarray of eligible uplink times
    units: np.ndarray      # reporting unit times
    unit_len: np.ndarray
    egress: list           # Record list (all outbound) for NEWEP/VOL
    frames: list           # raw Frame list (for EMIT)


def crop(site: Site, a: float, b: float) -> Site:
    """Restrict a site to [a, b]; matching is local (dt), so this is exact
    away from the edges."""
    m = (site.units >= a) & (site.units <= b)
    return dataclasses.replace(site, units=site.units[m], unit_len=site.unit_len[m],
                               uplinks={d: v[(v >= a) & (v <= b)] for d, v in site.uplinks.items()},
                               frames=[f for f in site.frames if a <= f.t <= b])


def global_match(site: Site, dt):
    """One-to-one matching of all units against the merged uplinks of every
    device (reports are encrypted, so a unit may be credited to any device).
    Returns (uplink_times, uplink_device, X, credit_index_per_unit)."""
    devs = list(site.uplinks)
    if not devs:
        return np.array([]), np.array([]), np.array([], int), np.full(len(site.units), -1)
    t = np.concatenate([site.uplinks[d] for d in devs])
    dv = np.concatenate([np.full(len(site.uplinks[d]), i) for i, d in enumerate(devs)])
    o = np.argsort(t, kind="stable")
    t, dv = t[o], np.asarray(devs, dtype=object)[dv[o]]
    X, credit = match_units(t, site.units, dt)
    return t, dv, X, credit


def sup_stream(site: Site, dev, dt) -> tuple[np.ndarray, np.ndarray]:
    t, dv, X, _ = global_match(site, dt)
    m = dv == dev
    return t[m], X[m]


def credited_mask(site: Site, dev, dt) -> np.ndarray:
    _, dv, _, credit = global_match(site, dt)
    out = np.zeros(len(site.units), bool)
    ok = credit >= 0
    out[ok] = dv[credit[ok]] == dev
    return out


def size_stream(site: Site, dev, dt, band) -> tuple[np.ndarray, np.ndarray]:
    m = credited_mask(site, dev, dt)
    lens = site.unit_len[m]
    lo, hi = band
    return site.units[m], ((lens < lo) | (lens > hi)).astype(int)


def fab_stream(site: Site, dt) -> tuple[np.ndarray, np.ndarray]:
    """FAB indicator per unit: 1 if the unit was credited to no uplink."""
    _, _, _, credit = global_match(site, dt)
    return site.units, (credit < 0).astype(int)


def self_injection_s(site: Site, dev, dt) -> tuple[float, float]:
    """Measured (eps_d, s_d): remove every unit credited to dev, re-match."""
    _, X = sup_stream(site, dev, dt)
    eps = float(X.mean()) if len(X) else float("nan")
    keep = ~credited_mask(site, dev, dt)
    s2 = dataclasses.replace(site, units=site.units[keep], unit_len=site.unit_len[keep])
    _, X2 = sup_stream(s2, dev, dt)
    return eps, float(X2.mean() - eps)


def inject_hide(site: Site, dev, f, t0, t1, dt, rng, policy="random", thin_C=None):
    """Remove reports credited to dev in [t0,t1].
    policy: random | masked_omniscient (A7: decides on the witness's view) |
    masked_realistic (A7': decides on all its own uplinks; the witness view is
    then thinned to completeness thin_C)."""
    cm = credited_mask(site, dev, dt)
    cand = np.where(cm & (site.units >= t0) & (site.units <= t1))[0]
    drop = np.zeros(len(site.units), bool)
    for k in cand:
        if policy == "random":
            drop[k] = rng.random() < f
            continue
        trial = drop.copy(); trial[k] = True
        before = sup_stream(dataclasses.replace(site, units=site.units[~drop],
                                                unit_len=site.unit_len[~drop]), dev, dt)[1].sum()
        after = sup_stream(dataclasses.replace(site, units=site.units[~trial],
                                               unit_len=site.unit_len[~trial]), dev, dt)[1].sum()
        drop[k] = after <= before
    new = dataclasses.replace(site, units=site.units[~drop], unit_len=site.unit_len[~drop])
    if thin_C is not None:
        ut = site.uplinks[dev]
        upl = dict(site.uplinks); upl[dev] = ut[rng.random(len(ut)) < thin_C]
        new = dataclasses.replace(new, uplinks=upl)
    return new, int(drop.sum()), len(cand)


def inject_delay(site: Site, dev, delay, t0, t1, dt):
    m = credited_mask(site, dev, dt) & (site.units >= t0) & (site.units <= t1)
    u = site.units.copy(); u[m] += delay
    o = np.argsort(u)
    return dataclasses.replace(site, units=u[o], unit_len=site.unit_len[o])


def inject_change(site: Site, dev, nbytes, t0, t1, dt):
    m = credited_mask(site, dev, dt) & (site.units >= t0) & (site.units <= t1)
    L = site.unit_len.copy(); L[m] += nbytes
    return dataclasses.replace(site, unit_len=L)


def inject_invent(site: Site, rate, t0, t1, rng):
    n = rng.poisson(rate * (t1 - t0))
    tt = np.sort(rng.uniform(t0, t1, n))
    ln = rng.choice(site.unit_len, n) if len(site.unit_len) else np.full(n, 100)
    u = np.concatenate([site.units, tt]); L = np.concatenate([site.unit_len, ln])
    o = np.argsort(u)
    return dataclasses.replace(site, units=u[o], unit_len=L[o])


def inject_swap(site: Site, dev_a, dev_b, t0, t1, dt, rng):
    """Report of dev_a carries dev_b's content: lengths resampled from dev_b."""
    ma = credited_mask(site, dev_a, dt) & (site.units >= t0) & (site.units <= t1)
    Lb = site.unit_len[credited_mask(site, dev_b, dt)]
    L = site.unit_len.copy()
    if len(Lb):
        L[ma] = rng.choice(Lb, ma.sum())
    return dataclasses.replace(site, unit_len=L)


def newep_alarms(egress: list, t_comm_end: float, t_from: float, extra=()):
    """NEWEP: a never-seen endpoint whose records in any 60 s window exceed
    twice the largest benign new-endpoint volume seen during commissioning."""
    recs = sorted(list(egress) + list(extra), key=lambda r: r.t)
    first = {}
    for r in recs:
        first.setdefault(r.endpoint, r.t)
    t_start = recs[0].t if recs else 0.0
    vols = defaultdict(list)
    for r in recs:
        if r.t < t_comm_end and first[r.endpoint] > t_start + 600:
            vols[r.endpoint].append(r.t)
    def max60(ts):
        ts = np.asarray(ts)
        return int(np.max(np.searchsorted(ts, ts + 60) - np.arange(len(ts)))) if len(ts) else 0
    thr = max(1, 2 * max((max60(v) for v in vols.values()), default=0))
    known = {e for e, t in first.items() if t < t_comm_end}
    alarms, seen = [], defaultdict(list)
    for r in recs:
        if r.t < t_from or r.endpoint in known:
            continue
        seen[r.endpoint].append(r.t)
        w = [x for x in seen[r.endpoint] if x > r.t - 60]
        if len(w) > thr:
            alarms.append((r.t, r.endpoint)); known.add(r.endpoint)
    return alarms


def emit_counter_stream(frames: list, src_key) -> tuple[np.ndarray, np.ndarray]:
    """Per-frame indicator for one transmitter: 1 on a counter collision."""
    ev = {i for i, _ in counter_events(frames)}
    idx = [i for i, f in enumerate(frames) if f.fc32 is not None and
           (f.sec_src if f.sec_src is not None else f.src) == src_key]
    return (np.array([frames[i].t for i in idx]),
            np.array([int(i in ev) for i in idx], dtype=int))


def inject_forge(frames: list, dev_key, rate, t0, t1, rng, continue_counter: bool):
    """Insert forged frames under dev_key: random counter or continuing the
    target's counter (A4).  Copies a genuine PSDU and alters one byte."""
    mine = [f for f in frames if f.fc32 is not None and
            (f.sec_src if f.sec_src is not None else f.src) == dev_key]
    if not mine:
        return frames, 0
    n = rng.poisson(rate * (t1 - t0))
    forged = []
    for tt in np.sort(rng.uniform(t0, t1, n)):
        last = max((f for f in mine if f.t <= tt), key=lambda f: f.t, default=mine[0])
        fc = last.fc32 + 1 if continue_counter else int(rng.integers(0, 2**32))
        ps = bytearray(last.psdu); ps[-1] ^= 0x5A
        forged.append(dataclasses.replace(last, t=float(tt), fc32=fc, psdu=bytes(ps)))
    return sorted(frames + forged, key=lambda f: f.t), len(forged)


def detect_in_window(t, x, l1, l0, h, t0, t1, tail) -> tuple[bool, float | None]:
    alarms = run_cusum(x, l1, l0, h)
    for i in alarms:
        if t0 <= t[i] <= t1 + tail:
            return True, float(t[i] - t0)
    return False, None


# ════════════════════════════════════════════════════════════════════════════
# 7. Results store and macro output
# ════════════════════════════════════════════════════════════════════════════
class Results:
    def __init__(self, out_dir: Path, tex_dir: Path):
        self.out_dir, self.tex_dir = out_dir, tex_dir
        out_dir.mkdir(parents=True, exist_ok=True)
        self.path = out_dir / "gw_results.json"
        self.d = json.loads(self.path.read_text()) if self.path.exists() else {}

    def set(self, key: str, value):
        self.d[key] = value

    def save(self):
        self.path.write_text(json.dumps(self.d, indent=1, sort_keys=True))
        lines = ["% generated by gw_stats.py -- do not edit"]
        for k in sorted(self.d):
            v = re.sub(r"(?<!\\)%", r"\\%", str(self.d[k]))
            lines.append(rf"\gwset{{{k}}}{{{v}}}")
        (self.tex_dir / "gw_results.tex").write_text("\n".join(lines) + "\n")
        print(f"wrote {len(self.d)} keys -> {self.tex_dir / 'gw_results.tex'}")


def _csv(path):
    return pd.read_csv(path) if path and Path(path).exists() else None


# ════════════════════════════════════════════════════════════════════════════
# 8. Subcommands
# ════════════════════════════════════════════════════════════════════════════
def cmd_stats(cfg, R: Results):
    """k/n, Clopper-Pearson CI and per-fold min-max for every experiment.
    Optional episodes.csv columns: n_uplinks (long-horizon rows), coverage
    (per-device witness completeness, for A10), delay_s (delay sweep rows)."""
    ep = _csv(cfg["episodes"])
    if ep is None:
        print("stats: episodes.csv missing, skipped"); return
    for exp, g in ep.groupby("experiment"):
        k, n = int(g.detected.sum()), len(g)
        lo, hi = clopper_pearson(k, n)
        per = g.groupby(["home", "fold"]).detected.mean()
        R.set(f"{exp}_kn", f"{k}/{n}")
        R.set(f"{exp}_ci", fmt_ci(lo, hi))
        R.set(f"{exp}_folds", f"{per.min():.2f}--{per.max():.2f}")
        R.set(f"{exp}_det", f"{k / n:.2f} {fmt_ci(lo, hi)}")
        if exp.startswith("sl_"):
            R.set(exp, f"{k / n:.2f}")
        if "latency_s" in g and g.latency_s.notna().any():
            R.set(f"{exp}_lat", f"{np.nanmedian(g.latency_s):.0f}\\,s")
    a4 = ep[ep.experiment.str.startswith("a4_32_")]
    for v in ("no", "sw", "ex"):
        g = a4[a4.experiment == f"a4_32_{v}"]
        if len(g):
            R.set(f"a4_32_{v}", f"{g.detected.mean():.2f}")
    if len(a4) and a4.latency_s.notna().any():
        q = a4.latency_s.dropna()
        R.set("a4_32_lat", f"{q.min():.0f}--{q.max():.0f}\\,s")
    lh = ep[ep.experiment.isin(["lh08", "lh10"])]
    if "n_uplinks" in ep and len(lh):
        R.set("lh_uplinks_range", f"{int(lh.n_uplinks.min())}--{int(lh.n_uplinks.max())}")
    if "coverage" in ep:
        a10, ref = ep[ep.experiment == "a10"], ep[ep.experiment == "lh08"]
        if len(a10) and len(ref):
            R.set("a10_ratio", f"{np.nanmedian(a10.latency_s) / np.nanmedian(ref.latency_s):.2f}")
            pred = np.nanmedian(ref.coverage) / np.nanmedian(a10.coverage) if ref.coverage.notna().any() else np.nan
            if np.isfinite(pred):
                R.set("a10_pred", f"{pred:.2f}")
    ds = ep[ep.experiment.str.startswith("delay_sweep")]
    if "delay_s" in ep and len(ds):
        r = ds.groupby("delay_s").detected.mean().sort_index()
        R.set("delay_sweep_summary",
              f"{r.iloc[0]:.2f} at {r.index[0]:g}\\,s and {r.iloc[-1]:.2f} at {r.index[-1]:g}\\,s")
    print(f"stats: {ep.experiment.nunique()} experiments")


def cmd_markov(cfg, R: Results):
    """Markov-exact column of Table IV and detection at B=1 under it."""
    P = cfg["params"]
    ind = _csv(cfg["indicators"])
    if ind is None:
        print("markov: indicators.csv missing, skipped"); return
    if "set" not in ind:
        ind["set"] = "main"
    extra = {"radio": "mk_radio", "long": "mk_long"}
    for st, key in extra.items():
        sub = ind[ind.set == st]
        if len(sub):
            R.set(key, fmt_rate(*_markov_alarms(sub, 1.0, P)))
    ind = ind[ind.set == "main"]
    rows = {}
    emit_k, emit_k_mk = [], []
    for B in P["budgets"]:
        total = {"bern": 0, "disp": 0, "markov": 0}
        exposure = 0.0
        for (home, fold), g in ind.groupby(["home", "fold"]):
            comm, held = g[g.split == "commission"], g[g.split == "heldout"]
            tests = sorted(comm.test_id.unique())
            N = len(tests)
            exposure += _exposure_days(held)
            for tid in tests:
                c = comm[comm.test_id == tid].sort_values("t")
                hd = held[held.test_id == tid].sort_values("t")
                kind = c.kind.iloc[0]
                design = None
                if {"design_eps", "design_p1"} <= set(c.columns) and c.design_eps.notna().any():
                    design = (float(c.design_eps.iloc[0]), float(c.design_p1.iloc[0]))
                cal = calibrate_test(tid, kind, c.x.values, c.t.values, N, B, P, design)
                if kind == "poisson":
                    for key, h in (("bern", cal.h_bern), ("disp", cal.h_disp), ("markov", cal.h_disp)):
                        total[key] += len(run_cusum_poisson(hd.x.values, cal.eps, cal.p1, h))
                    continue
                l1, l0 = cusum_increments(cal.eps, cal.p1)
                if tid.startswith("EMIT") and B == 1:
                    # anomalies needed under Bernoulli-exact thresholds (the operating mode of Table III)
                    emit_k.append(math.ceil(cal.h_bern / l1))
                    emit_k_mk.append(math.ceil((cal.h_markov or cal.h_bern) / l1))
                for key, h in (("bern", cal.h_bern), ("disp", cal.h_disp), ("markov", cal.h_markov)):
                    if h is not None:
                        total[key] += len(run_cusum(hd.x.values, l1, l0, h))
        rows[B] = (total, exposure)
        R.set(f"mk_B{B}", fmt_rate(total["markov"], exposure))
    if emit_k:
        R.set("emit_k_required", f"{min(emit_k)}" if min(emit_k) == max(emit_k)
              else f"{min(emit_k)}--{max(emit_k)}")
        R.set("emit_k_required_markov", f"{min(emit_k_mk)}" if min(emit_k_mk) == max(emit_k_mk)
              else f"{min(emit_k_mk)}--{max(emit_k_mk)}")
    # Detection at B=1 with Markov-exact thresholds on attacked streams
    epi = _csv(cfg["episode_indicators"])
    if epi is not None:
        det = defaultdict(list)
        for eid, g in epi.groupby("episode_id"):
            home, fold, exp = g.home.iloc[0], g.fold.iloc[0], g.experiment.iloc[0]
            comm = ind[(ind.home == home) & (ind.fold == fold) & (ind.split == "commission")]
            N = comm.test_id.nunique()
            hit = False
            for tid, gt in g.groupby("test_id"):
                c = comm[comm.test_id == tid].sort_values("t")
                if c.empty or c.kind.iloc[0] != "bernoulli":
                    continue
                cal = calibrate_test(tid, "bernoulli", c.x.values, c.t.values, N, 1.0, P)
                l1, l0 = cusum_increments(cal.eps, cal.p1)
                gt = gt.sort_values("t")
                ok, _ = detect_in_window(gt.t.values, gt.x.values, l1, l0, cal.h_markov or cal.h_bern,
                                         gt.t_start.iloc[0], gt.t_end.iloc[0], P["score_tail_s"])
                hit |= ok
            det[exp].append(hit)
        if "forge" in det:
            R.set("mk_det_B1", f"{np.mean(det['forge']):.2f}")
        if "change" in det:
            R.set("mk_det_B1_change", f"{np.mean(det['change']):.2f}")
    print("markov: done")


def _markov_alarms(ind, B, P) -> tuple[int, float]:
    """Markov-exact benign alarms and exposure (days) for one indicator set."""
    total, exposure = 0, 0.0
    for (home, fold), g in ind.groupby(["home", "fold"]):
        comm, held = g[g.split == "commission"], g[g.split == "heldout"]
        N = comm.test_id.nunique()
        exposure += _exposure_days(held)
        for tid in comm.test_id.unique():
            c = comm[comm.test_id == tid].sort_values("t")
            hd = held[held.test_id == tid].sort_values("t")
            cal = calibrate_test(tid, c.kind.iloc[0], c.x.values, c.t.values, N, B, P)
            if cal.kind == "poisson":
                total += len(run_cusum_poisson(hd.x.values, cal.eps, cal.p1, cal.h_disp))
            else:
                l1, l0 = cusum_increments(cal.eps, cal.p1)
                total += len(run_cusum(hd.x.values, l1, l0, cal.h_markov or cal.h_bern))
    return total, exposure


def _exposure_days(df) -> float:
    if df.empty:
        return 0.0
    spans = df.groupby("test_id").t.agg(lambda s: s.max() - s.min())
    return float(spans.max()) / SECONDS_PER_DAY


def cmd_counters(cfg, R: Results):
    """32-bit counter benign rate, ACK spread on a second sniffer, and the
    three forge-shortfall diagnostics."""
    P = cfg["params"]
    if cfg["ziotp_radio"]:
        frames = load_frames(cfg["ziotp_radio"])
        n = sum(1 for f in frames if f.fc32 is not None)
        ev = counter_events(frames)
        R.set("fc32_benign_rate", f"{100 * len(ev) / max(n, 1):.3f}\\%")
        print(f"counters: {len(ev)} benign counter events in {n} secured frames")
    if cfg["nrf_radio"]:
        frames = load_frames(cfg["nrf_radio"])
        band = learn_ack_band([g for _, g in ack_gaps_us(frames)], P["ack_halfwidth_us"])
        R.set("nrf_ack_spread_us", f"{band['spread_us']:.1f}")
    ff = _csv(cfg["forge_frames"])
    if ff is not None:
        dropped, nogen, total = 0, 0, 0
        for eid, g in ff.groupby("episode_id"):
            g = g.sort_values("t")
            frs = [Frame(r.t, 1, int(r.mac_seq), None, int(r.mac_src), None, True,
                         bytes.fromhex(r.psdu_hex)) for r in g.itertuples()]
            kept = {id(f) for f in dedup_seq8(frs, P["dedup_window_s"])}
            for f, r in zip(frs, g.itertuples()):
                if not r.forged:
                    continue
                total += 1
                dropped += id(f) not in kept
                gen = g[(g.forged == 0) & (g.device == r.device) & (g.t > r.t)
                        & (g.t <= r.t_end + P["score_tail_s"])]
                nogen += gen.empty
        R.set("fd_dedup_frac", f"{100 * dropped / max(total, 1):.0f}\\%")
        R.set("fd_nogenuine_frac", f"{100 * nogen / max(total, 1):.0f}\\%")
        print(f"counters: {total} forged frames diagnosed")


def build_site(radio_paths, eth_paths, cfg, offset=0.0) -> tuple[Site, dict]:
    sl = cfg["songlin"]
    frames = load_frames(radio_paths)
    for f in frames:
        f.t += offset
    dd = dedup_bit_identical(frames, cfg["params"]["dedup_window_s"])
    coord = sl["coordinator"]
    seen, upl = {}, defaultdict(list)
    for f in dd:
        if f.ftype == 1 and f.dst == coord and f.src not in (None, coord) and f.nwk_src not in (None, coord):
            key = (f.nwk_src, f.nwk_seq)          # 8-bit NWK seq wraps: window it
            if key in seen and f.t - seen[key] <= 2.0:   # relay hop of the same NWK frame
                continue
            seen[key] = f.t
            upl[f.nwk_src].append(f.t)
    recs, hub = extract_records(eth_paths, sl["hub_ip"])
    out = [r for r in recs if r.outbound]
    by_ep = defaultdict(list)
    for r in out:
        by_ep[r.endpoint].append(r.t)
    meta = {"hub_ip": hub, "by_ep": {k: np.asarray(v) for k, v in by_ep.items()}}
    return Site({d: np.asarray(v) for d, v in upl.items()}, np.array([]), np.array([]),
                out, frames), meta


def _site_streams(site: Site, tc: float, P: dict):
    """Benign indicator streams of the reference tests, plus design points."""
    streams, design = {}, {}
    comm = dataclasses.replace(site, units=site.units[site.units < tc],
                               unit_len=site.unit_len[site.units < tc],
                               uplinks={d: v[v < tc] for d, v in site.uplinks.items()})
    devs = [d for d in site.uplinks if np.sum(site.uplinks[d] < tc) >= 30]
    gt, gdv, gX, _ = global_match(site, P["dt"])
    for d in devs:
        streams[f"SUP[{d}]"] = (gt[gdv == d], gX[gdv == d])
        eps_d, s_d = self_injection_s(comm, d, P["dt"])
        n_c = int(np.sum(site.uplinks[d] < tc))
        eps_u = min(max(beta_upper(int(round(eps_d * n_c)), n_c, P["beta_upper"]), 1e-4), 0.95)
        design[f"SUP[{d}]"] = (eps_u, min(eps_u + P["f_star"] * max(s_d, 0.02), 0.999))
        L = comm.unit_len[credited_mask(comm, d, P["dt"])]
        if len(L) >= 30:
            band = tuple(np.percentile(L, P["size_band"]))
            streams[f"SIZE[{d}]"] = size_stream(site, d, P["dt"], band)  # one pass per device
            design[f"SIZE[{d}]#band"] = band
    streams["FAB"] = fab_stream(site, P["dt"])
    keys = {f.sec_src if f.sec_src is not None else f.src
            for f in site.frames if f.fc32 is not None}
    for k in keys:
        t, x = emit_counter_stream(site.frames, k)
        if np.sum(t < tc) >= 30:
            streams[f"EMITC[{k}]"] = (t, x)
    return streams, design, devs


_CAL_CACHE: dict = {}


def _alarms(t, x, tc, tid, N, B, P, design, which):
    key = (tid, N, B, len(t))
    if key not in _CAL_CACHE:
        c = t < tc
        _CAL_CACHE[key] = calibrate_test(tid, "bernoulli", x[c], t[c], N, B, P, design.get(tid))
    cal = _CAL_CACHE[key]
    l1, l0 = cusum_increments(cal.eps, cal.p1)
    h = {"be": cal.h_bern, "dc": cal.h_disp, "mk": cal.h_markov or cal.h_bern}[which]
    return cal, l1, l0, h


def cmd_songlin(cfg, R: Results):
    """Frozen pipeline on Song & Lin: align, commission, benign alarms per budget
    and calibration, the recorded replay, and injected classes.  Runs once; all
    parameters come from cfg['params'] and must not change after first run."""
    P, sl = cfg["params"], cfg["songlin"]
    if not (Path(sl["radio"]).exists() and Path(sl["ethernet"]).exists()):
        print("songlin: captures missing, skipped"); return
    commit = cfg.get("frozen_commit") or _git_commit()
    R.set("frozen_commit", commit[:10] if commit else "uncommitted")
    rng = np.random.default_rng(P["seed"])
    site, meta = build_site([sl["radio"]], [sl["ethernet"]], cfg)
    all_up = np.sort(np.concatenate(list(site.uplinks.values())))
    ep_star = max(meta["by_ep"], key=lambda e: len(meta["by_ep"][e]))
    off = clock_offset_fft(all_up, meta["by_ep"][ep_star], P["dt"])
    R.set("sl_clock_offset_s", f"{off:.2f}")
    site, meta = build_site([sl["radio"]], [sl["ethernet"]], cfg, offset=off)
    all_up = np.sort(np.concatenate(list(site.uplinks.values())))
    tc = all_up.min() + P["commission_min"] * 60
    comm_eps = {e: v[v < tc] for e, v in meta["by_ep"].items() if np.sum(v < tc) > 10}
    ct = commissioning_test(all_up[all_up < tc], comm_eps, P["dt"], rng=rng)
    R.set("sl_D", f"{ct['D']:.3f}"); R.set("sl_p", f"{ct['p']:.3f}")
    R.set("sl_admitted", "admitted" if ct["admitted"] else "refused")
    if not ct["admitted"]:
        print("songlin: site refused at commissioning; stopping as pre-registered")
        return
    rep = [r for r in site.egress if r.endpoint == ct["endpoint"]]
    site = dataclasses.replace(site, units=np.array([r.t for r in rep]),
                               unit_len=np.array([r.length for r in rep]))
    t_end = max(all_up.max(), site.units.max())
    days = (t_end - tc) / SECONDS_PER_DAY
    R.set("sl_eval_days", f"{days:.1f}")
    streams, design, devs = _site_streams(site, tc, P)
    N = len(streams) + 1          # + NEWEP
    R.set("sl_tests_covered", "SUP, SIZE, FAB, EMIT counter, NEWEP")
    newep = len(newep_alarms(site.egress, tc, tc))
    for B in (1, 10, 100):
        tot = {"be": newep, "dc": newep, "mk": newep}
        for tid, (t, x) in streams.items():
            if np.sum(t < tc) < 30:
                continue
            for which in tot:
                _, l1, l0, h = _alarms(t, x, tc, tid, N, B, P, design, which)
                tot[which] += len(run_cusum(x[t >= tc], l1, l0, h))
        for which, k in tot.items():
            R.set(f"sl_{which}_B{B}", fmt_rate(k, days))
        if B == 1:
            R.set("sl_alarms_B1", fmt_rate(tot["mk"], days))
    # recorded replay: counter evidence on the sniffed replay capture
    if Path(sl["replay"]).exists():
        rf = load_frames([sl["replay"]])
        ev = counter_events(rf)
        R.set("sl_replay_detected", "detected" if ev else "not detected")
        R.set("sl_replay_steps", str(len(ev)))
        R.set("sl_replay_latency", f"{rf[ev[0][0]].t - rf[0].t:.1f}\\,s" if ev else "--")
    # injected classes, scored with Markov-exact thresholds at B = 1
    two = sorted(devs, key=lambda d: -len(site.uplinks[d]))[:2]
    keyof = {}
    for f in site.frames:
        if f.fc32 is not None and f.nwk_src in two:
            keyof.setdefault(f.nwk_src, f.sec_src if f.sec_src is not None else f.src)
    classes = ("hide", "delay", "change", "invent", "swap", "leak", "forge", "forgeX", "lh08", "lh10")
    for cls in classes:
        hits = []
        for e in range(P["episodes_per_class"]):
            dev = two[e % 2]
            dur = 900.0 if cls.startswith("lh") else 240.0
            s0 = float(rng.uniform(tc + 3600, t_end - dur - 120))
            s1, tail = s0 + dur, P["score_tail_s"]
            ws = crop(site, s0 - 3600, s1 + tail + 5)
            if cls == "leak":
                fake = [Record(float(tt), int(rng.choice(ws.unit_len)), ("203.0.113.9", 443, 6), True)
                        for tt in np.sort(rng.uniform(s0, s1, rng.poisson(dur / 20)))]
                hits.append(any(s0 <= a <= s1 + tail for a, _ in
                                newep_alarms(ws.egress, tc, s0 - 1, fake)))
                continue
            if cls in ("forge", "forgeX"):
                k = keyof.get(dev)
                tid = f"EMITC[{k}]"
                if k is None or tid not in streams:
                    hits.append(False); continue
                frs, _ = inject_forge(ws.frames, k, 1 / 25, s0, s1, rng, cls == "forgeX")
                t, x = emit_counter_stream(frs, k)
            else:
                if cls in ("hide", "lh08"):
                    att, _, _ = inject_hide(ws, dev, 0.8, s0, s1, P["dt"], rng)
                elif cls == "lh10":
                    att, _, _ = inject_hide(ws, dev, 1.0, s0, s1, P["dt"], rng)
                elif cls == "delay":
                    att = inject_delay(ws, dev, 30.0, s0, s1, P["dt"])
                elif cls == "change":
                    att = inject_change(ws, dev, 60, s0, s1, P["dt"])
                elif cls == "swap":
                    att = inject_swap(ws, dev, two[(e + 1) % 2], s0, s1, P["dt"], rng)
                else:
                    att = inject_invent(ws, 1 / 15, s0, s1, rng)
                if cls in ("change", "swap"):
                    tid = f"SIZE[{dev}]"
                    if tid not in streams:
                        hits.append(False); continue
                    t, x = size_stream(att, dev, P["dt"], design[tid + "#band"])
                elif cls == "invent":
                    tid = "FAB"; t, x = fab_stream(att, P["dt"])
                else:
                    tid = f"SUP[{dev}]"; t, x = sup_stream(att, dev, P["dt"])
            bt, bx = streams[tid]
            _, l1, l0, h = _alarms(bt, bx, tc, tid, N, 1.0, P, design, "mk")
            w = t >= s0 - 3600
            ok, _ = detect_in_window(t[w], x[w], l1, l0, h, s0, s1, tail)
            hits.append(ok)
        R.set(f"sl_{cls}", f"{np.mean(hits):.2f}")
    print("songlin: frozen run complete")


def cmd_crosscheck(cfg, R: Results):
    """Compare reference SUP/SIZE/FAB unit counts and mismatch rates with the
    verifier's exported commissioning streams for the configured ZIOTP2025 pairs.
    cfg['ziotp_pairs'][i] needs 'home','fold' keys matching indicators.csv."""
    ind, pairs = _csv(cfg["indicators"]), cfg.get("ziotp_pairs") or []
    if ind is None or not pairs:
        print("crosscheck: needs indicators.csv and ziotp_pairs"); return True
    worst = 0.0
    for pr in pairs:
        site, _ = build_site([pr["radio"]], [pr["ethernet"]], cfg)
        rep = [r for r in site.egress if r.endpoint == tuple(pr["endpoint"])]
        site = dataclasses.replace(site, units=np.array([r.t for r in rep]),
                                   unit_len=np.array([r.length for r in rep]))
        ref = ind[(ind.home == pr["home"]) & (ind.fold == pr["fold"]) & (ind.split == "commission")]
        for tid, g in ref.groupby("test_id"):
            if not tid.startswith("SUP["):
                continue
            dev = int(tid[4:-1], 0)
            if dev not in site.uplinks:
                continue
            _, x = sup_stream(site, dev, cfg["params"]["dt"])
            d_n = abs(len(x) - len(g)) / max(len(g), 1)
            d_x = abs(x.mean() - g.x.mean())
            worst = max(worst, d_n, d_x)
            print(f"  {pr['home']}/{pr['fold']} {tid}: units {len(x)} vs {len(g)}, "
                  f"eps {x.mean():.3f} vs {g.x.mean():.3f}")
    print(f"crosscheck: worst relative disagreement {worst:.3%}")
    return worst <= 0.01


def cmd_sweeps(cfg, R: Results):
    """SIZE band sweep, delay sweep, CSMA KS test, A7', A10 on ZIOTP2025 pairs.
    Needs cfg['ziotp_pairs'] = [{'radio':..., 'ethernet':..., 'endpoint': [ip,port,proto]}]."""
    P = cfg["params"]
    pairs = cfg.get("ziotp_pairs") or []
    if not pairs:
        print("sweeps: ziotp_pairs not configured, skipped"); return
    rng = np.random.default_rng(P["seed"])
    res = defaultdict(list)
    for pr in pairs:
        site, meta = build_site([pr["radio"]], [pr["ethernet"]], cfg, offset=pr.get("offset", 0.0))  # IP clock minus radio clock (s)
        ep = tuple(pr["endpoint"])
        rep = [r for r in site.egress if r.endpoint == ep]
        site = dataclasses.replace(site, units=np.array([r.t for r in rep]),
                                   unit_len=np.array([r.length for r in rep]))
        if len(site.units) < 50:
            continue
        tmid = np.median(site.units)
        devs = sorted(site.uplinks, key=lambda d: -len(site.uplinks[d]))
        for band, tag in (([1, 99], "99"), ([2.5, 97.5], "975")):
            for d in devs[:2]:
                L = site.unit_len[credited_mask(site, d, P["dt"]) & (site.units < tmid)]
                if len(L) < 20:
                    continue
                bnd = tuple(np.percentile(L, band))
                t, x = size_stream(site, d, P["dt"], bnd)
                c = t < tmid
                if c.sum() < 20 or (~c).sum() < 20:
                    continue
                cal = calibrate_test(f"SIZE[{d}]", "bernoulli", x[c], t[c], 2 * len(devs), P["B"], P)
                l1, l0 = cusum_increments(cal.eps, cal.p1)
                h = cal.h_markov or cal.h_bern
                res[f"size_{tag}_fa"].append((len(run_cusum(x[~c], l1, l0, h)),
                                              (t.max() - tmid) / SECONDS_PER_DAY))
                for s0 in np.linspace(tmid + 60, t.max() - 300, 5):
                    att = inject_change(crop(site, s0 - 1800, s0 + 300), d, 20, s0, s0 + 240, P["dt"])
                    t2, x2 = size_stream(att, d, P["dt"], bnd)
                    ok, _ = detect_in_window(t2, x2, l1, l0, h, s0, s0 + 240, P["score_tail_s"])
                    res[f"size_{tag}_20B"].append(ok)
        # A7' realistic masked hiding with emulated witness completeness 0.8
        d = devs[0]
        w7 = crop(site, tmid - 1800, tmid + 960)
        att, ndrop, ncand = inject_hide(w7, d, 1.0, tmid, tmid + 900, P["dt"], rng,
                                        policy="masked_realistic", thin_C=0.8)
        _, Xa = sup_stream(att, d, P["dt"])
        _, Xb = sup_stream(w7, d, P["dt"])
        res["a7p_hidden"].append(ndrop / max(ncand, 1))
        res["a7p_mismatch"].append(Xa.mean()); res["a7p_benign"].append(Xb.mean())
        # CSMA back-off proxy: gaps before coordinator data frames, first vs second half
        fr = [f for f in site.frames if f.ftype == 1]
        gaps = np.diff([f.t for f in fr]) * 1e6
        src = np.array([f.src for f in fr[1:]])
        cg = gaps[src == cfg["songlin"]["coordinator"]]
        if len(cg) > 40:
            res["csma_p"].append(stats.ks_2samp(cg[: len(cg) // 2], cg[len(cg) // 2:]).pvalue)
    for tag in ("99", "975"):
        if res[f"size_{tag}_fa"]:
            k = int(sum(a for a, _ in res[f"size_{tag}_fa"]))
            e = sum(b for _, b in res[f"size_{tag}_fa"])
            R.set(f"size_{tag}_fa", fmt_rate(k, e).split(" ")[0])
            R.set(f"size_{tag}_20B", f"{np.mean(res[f'size_{tag}_20B']):.2f}")
    if res["a7p_hidden"]:
        R.set("a7p_hidden", f"{100 * np.mean(res['a7p_hidden']):.0f}\\%")
        R.set("a7p_mismatch", f"{np.mean(res['a7p_mismatch']):.3f}")
        R.set("a7p_benign", f"{np.mean(res['a7p_benign']):.3f}")
    if res["csma_p"]:
        R.set("csma_ks_p", f"{np.median(res['csma_p']):.2g}")
    print("sweeps: done (A7'/A10/delay detection rates come from episodes.csv via `stats`)")


# ── figures ────────────────────────────────────────────────────────────────
PALETTE = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#6250d6", "#e34948"]
MARKERS = ["o", "s", "^", "D", "v", "P", "X", "*"]
DETECTOR_ORDER = ["GateWitness", "Log monitor", "Fixed count", "Radio vs hub",
                  "Radio only", "IP rules", "PingPong-style", "IsolationForest"]


def matched_rate(sweep: pd.DataFrame, detector, attack, fa):
    """Detection beyond placebo at observed benign rate `fa` (log-interpolated,
    monotone envelope: best detection achievable at or below `fa`)."""
    g = sweep[(sweep.detector == detector) & (sweep.attack == attack)].copy()
    if g.empty:
        return float("nan")
    g["rate"] = g.benign_alarms / g.exposure_days
    g["gain"] = g.det - g.placebo
    ok = g[g.rate <= fa]
    return float(ok.gain.max()) if len(ok) else 0.0


def cmd_figs(cfg, R: Results):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    sweep, cold = _csv(cfg["sweep"]), _csv(cfg["coldstart"])
    if sweep is None:
        print("figs: sweep.csv missing, skipped"); return
    plt.rcParams.update({"font.size": 7, "font.family": "serif", "axes.linewidth": 0.6,
                         "axes.spines.top": False, "axes.spines.right": False})
    fig, axes = plt.subplots(1, 5, figsize=(7.16, 1.85), constrained_layout=True)
    panels = [("hide", "(a) hide"), ("change", "(b) change"), ("forge", "(c) forge"),
              ("a5", "(d) rate-preserving hide")]
    dets = [d for d in DETECTOR_ORDER if d in set(sweep.detector)]
    one_alarm = 1.0 / float(sweep.exposure_days.median())   # rate of a single alarm
    for ax, (att, title) in zip(axes[:4], panels):
        for j, det in enumerate(dets):
            g = sweep[(sweep.detector == det) & (sweep.attack == att)].copy()
            if g.empty:
                continue
            g["rate"] = (g.benign_alarms / g.exposure_days).clip(lower=0.3)
            g = g.sort_values("rate")
            ax.plot(g.rate, g.det - g.placebo, color=PALETTE[j % 8], marker=MARKERS[j % 8],
                    ms=3, lw=2 if det == "GateWitness" else 1, label=det,
                    zorder=3 if det == "GateWitness" else 2)
        ax.set_xscale("log"); ax.set_xlim(0.3, 3000); ax.set_ylim(-0.1, 1.0)
        ax.set_xticks([1, 10, 100, 1000]); ax.set_xticklabels(["1", "10", "100", "1k"])
        ax.minorticks_off()
        ax.axvline(one_alarm, color="#888780", lw=0.6, ls=":")
        ax.axhline(0, color="#888780", lw=0.5)
        ax.grid(axis="y", color="#e6e5e0", lw=0.5)
        ax.set_title(title, fontsize=7, loc="left")
        ax.set_xlabel("benign alarms/day")
    axes[0].set_ylabel("detection beyond placebo")
    h, l = axes[0].get_legend_handles_labels()
    fig.legend(h, l, loc="outside lower center", ncol=8, frameon=False, fontsize=6)
    ax = axes[4]
    if cold is not None:
        for j, (prior, g) in enumerate(cold.groupby("prior")):
            g = g.groupby("minutes").detection.mean()
            # neutral inks: this panel is not a detector comparison
            ax.plot(g.index, g.values, color="#2c2c2a" if prior else "#888780",
                    ls="-" if prior else "--", marker="o" if prior else "s", ms=3, lw=1.5,
                    label="with prior" if prior else "no prior")
        ax.legend(frameon=False, fontsize=6, loc="lower right")
    ax.set_title("(e) cold start", fontsize=7, loc="left")
    ax.set_xlabel("commissioning (min)"); ax.set_ylabel("detection")
    ax.grid(axis="y", color="#e6e5e0", lw=0.5)
    out = Path(cfg["tex_dir"]) / "figs"
    out.mkdir(parents=True, exist_ok=True)
    fig.savefig(out / "fig5_matched_coldstart.pdf")
    fig.savefig(Path(cfg["out_dir"]) / "fig5_matched_coldstart.png", dpi=200)
    # matched-rate margins quoted in Section VIII-G
    for att, key in (("change", "mr100_change"), ("forge", "mr100_forge"),
                     ("hide", "mr100_hide"), ("a5", "mr100_a5")):
        gw = matched_rate(sweep, "GateWitness", att, 100)
        best = max((matched_rate(sweep, d, att, 100) for d in dets if d != "GateWitness"),
                   default=float("nan"))
        if att in ("hide", "a5"):
            best = matched_rate(sweep, "Log monitor", att, 100)
        R.set(key, f"{gw - best:+.2f}")
    print(f"figs: wrote {out / 'fig5_matched_coldstart.pdf'}")


def _git_commit():
    try:
        return subprocess.check_output(["git", "rev-parse", "HEAD"], text=True,
                                       stderr=subprocess.DEVNULL).strip()
    except Exception:
        return None


# ════════════════════════════════════════════════════════════════════════════
# 9. Self-test on synthetic data
# ════════════════════════════════════════════════════════════════════════════
def selftest(tmp: Path):
    tmp.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(7)
    ok = True

    def check(name, cond, detail=""):
        nonlocal ok
        ok &= bool(cond)
        print(f"  [{'PASS' if cond else 'FAIL'}] {name} {detail}")

    print("intervals")
    r, lo, hi = garwood(0, 42.9 / 1440)
    check("Garwood 0 in 42.9 min = [0, 124]", round(hi) == 124, f"(upper {hi:.1f})")
    lo, hi = clopper_pearson(45, 90)
    check("Clopper-Pearson 45/90", abs(lo - 0.3929) < 2e-3 and abs(hi - 0.6071) < 2e-3,
          f"[{lo:.4f}, {hi:.4f}]")

    print("Brook-Evans calibration")
    eps, p1 = 0.37, 0.37 + 0.5 * 0.21
    l1, l0 = cusum_increments(eps, p1)
    for a, b, tag in ((eps, eps, "Bernoulli"), (0.30, 0.49, "Markov rho~0.2"),
                      (0.25, 0.58, "Markov rho~0.33")):
        h, arl = threshold_for_arl(l1, l0, 500, a, b, res=128)
        mc = monte_carlo_arl(l1, l0, h, a, b, reps=3000, rng=np.random.default_rng(3))
        check(f"{tag}: lattice ARL <= MC ARL within 15%", arl <= mc * 1.05 and arl >= mc * 0.85,
              f"(h={h:.2f}, lattice {arl:.0f}, MC {mc:.0f})")
    hb, _ = threshold_for_arl(l1, l0, 5000, eps, eps)
    hm, _ = threshold_for_arl(l1, l0, 5000, 0.25, 0.58)
    check("positive dependence needs a higher threshold", hm > hb, f"({hb:.2f} -> {hm:.2f})")
    x = np.zeros(20000, int)
    for i in range(1, len(x)):
        x[i] = rng.random() < (0.58 if x[i - 1] else 0.25)
    a_, b_ = markov_fit(x)
    check("Markov fit recovers (a,b)", abs(a_ - 0.25) < 0.02 and abs(b_ - 0.58) < 0.02,
          f"({a_:.3f}, {b_:.3f})")
    phi = dispersion_phi(x, l1, l0)
    check("dispersion > 1 on dependent stream", phi > 1.2, f"(phi={phi:.2f})")
    # alarm-rate validation: Markov-exact keeps nominal rate, Bernoulli-exact exceeds it
    target = 2000
    hB, _ = threshold_for_arl(l1, l0, target, eps, eps)
    au, bu = markov_fit_upper(x)
    hM, _ = threshold_for_arl(l1, l0, target, au, bu)
    xs = np.zeros(400000, int)
    for i in range(1, len(xs)):
        xs[i] = rng.random() < (0.58 if xs[i - 1] else 0.25)
    nB, nM = len(run_cusum(xs, l1, l0, hB)), len(run_cusum(xs, l1, l0, hM))
    nominal = len(xs) / target
    check("Bernoulli-exact over-alarms on Markov stream", nB > 1.3 * nominal,
          f"({nB} vs nominal {nominal:.0f})")
    check("Markov-exact (upper-bound fit) at or below nominal (+10%)", nM <= 1.1 * nominal,
          f"({nM} vs nominal {nominal:.0f})")

    print("802.15.4 / Zigbee NWK parsing")
    # data frame, PAN compression, short dst/src, ack request; NWK secured, ext nonce
    fcf = 0x0001 | (1 << 5) | (1 << 6) | (2 << 10) | (2 << 14)
    mac = struct.pack("<HBHHH", fcf, 0x42, 0x1A62, 0x0000, 0x7B3C)
    nfc = 0x0008 | (1 << 9) | (1 << 12)   # data, version 2, security, src IEEE
    nwk = struct.pack("<HHHBB", nfc, 0x0000, 0x7B3C, 30, 0x99) + struct.pack("<Q", 0x00124B0001020304)
    aux = bytes([0x28]) + struct.pack("<I", 0x01020304) + struct.pack("<Q", 0x00124B0001020304) + b"\x00"
    payload = b"\xaa" * 12
    frame = mac + nwk + aux + payload
    fr = parse_mac(1.0, frame)
    check("MAC fields", fr.src == 0x7B3C and fr.dst == 0 and fr.seq == 0x42 and fr.ack_req)
    check("NWK src/seq", fr.nwk_src == 0x7B3C and fr.nwk_seq == 0x99)
    check("32-bit frame counter", fr.fc32 == 0x01020304, f"(got {fr.fc32})")
    check("aux extended source", fr.sec_src == 0x00124B0001020304)
    # pcap round trip (linktype 195 with FCS)
    pc = tmp / "t.pcap"
    with open(pc, "wb") as f:
        f.write(struct.pack("<IHHiIII", 0xA1B2C3D4, 2, 4, 0, 0, 65535, 195))
        for i, ts in enumerate((1.0, 1.0016, 2.0)):
            body = frame + b"\x00\x00"
            if i == 1:  # ACK frame
                body = struct.pack("<HB", 0x0002, 0x42) + b"\x00\x00"
            f.write(struct.pack("<IIII", int(ts), int(round((ts % 1) * 1e6)), len(body), len(body)) + body)
    frs = load_frames([str(pc)])
    check("pcap reader", len(frs) == 3 and frs[1].ftype == 2)
    gaps = ack_gaps_us(frs)
    exp_gap = 1600 - (len(frame) + 2 + 6) * 32
    check("ACK gap minus air time", gaps and abs(gaps[0][1] - exp_gap) < 1.5, f"({gaps[0][1]:.1f} us)")

    print("counter evidence and deduplication")
    base = dict(ftype=1, dst=0, pan=0x1A62, ack_req=True)
    g1 = Frame(1.0, seq=10, src=5, psdu=b"A", fc32=100, sec_src=1, **base)
    g1r = Frame(1.002, seq=10, src=5, psdu=b"A", fc32=100, sec_src=1, **base)   # retransmission
    fake = Frame(1.1, seq=10, src=5, psdu=b"F", fc32=101, sec_src=1, **base)    # forger continues counter
    g2 = Frame(5.0, seq=11, src=5, psdu=b"B", fc32=101, sec_src=1, **base)      # genuine next frame
    raw = [g1, g1r, fake, g2]
    ev = counter_events(raw)
    check("retransmission is not a collision; genuine next frame collides",
          [k for _, k in ev] == ["repeat"] and ev[0][0] == 3, f"{ev}")
    check("bit-identical dedup keeps the forged frame", fake in dedup_bit_identical(raw))
    check("8-bit dedup drops the forged frame (old rule)", fake not in dedup_seq8(raw))

    print("matching, self-injection, commissioning, clock offset")
    T = 3600.0
    up = np.sort(rng.uniform(0, T, 1500))
    kept = up[rng.random(len(up)) > 0.3]
    units = np.sort(np.concatenate([kept + rng.uniform(0.05, 0.4, len(kept)),
                                    rng.uniform(0, T, 400)]))
    site0 = Site({1: up}, units, rng.integers(80, 120, len(units)), [], [])
    eps_m, s_m = self_injection_s(site0, 1, 1.0)
    check("self-injection gives 0 < s < 1-eps", 0 < s_m < 1 - eps_m, f"(eps={eps_m:.2f}, s={s_m:.2f})")
    ct = commissioning_test(up, {"rep": units, "noise": rng.uniform(0, T, 800)}, M=499,
                            rng=np.random.default_rng(1))
    check("coupled endpoint admitted", ct["admitted"] and ct["endpoint"] == "rep",
          f"(D={ct['D']:.2f}, p={ct['p']:.3f})")
    ct2 = commissioning_test(up, {"noise": rng.uniform(0, T, 800)}, M=99, rng=np.random.default_rng(2))
    check("independent endpoint refused", not ct2["admitted"], f"(D={ct2['D']:.3f})")
    off = clock_offset_fft(up, units + 37.25, 1.0, max_off=120)
    check("FFT clock offset recovered", abs(off - 37.25) < 0.5, f"({off:.2f} s)")

    print("injections and detection")
    site = site0
    h0 = sup_stream(site, 1, 1.0)[1].mean()
    att, nd, nc = inject_hide(site, 1, 0.8, 1000, 2000, 1.0, rng)
    t1_, X1 = sup_stream(att, 1, 1.0)
    w = (t1_ >= 1000) & (t1_ <= 2000)
    check("hide raises mismatch in window", X1[w].mean() > h0 + 0.2, f"({h0:.2f} -> {X1[w].mean():.2f})")
    att7, nd7, nc7 = inject_hide(site, 1, 1.0, 1000, 2000, 1.0, rng, policy="masked_omniscient")
    _, X7 = sup_stream(att7, 1, 1.0)
    check("A7 omniscient masked hiding does not raise unmatched count", X7.sum() <= sup_stream(site, 1, 1.0)[1].sum(),
          f"(hid {nd7}/{nc7})")

    print("macros and figure")
    R = Results(tmp / "res", tmp)
    R.set("demo_key", "0.42"); R.set("pct", "37%"); R.save()
    txt = (tmp / "gw_results.tex").read_text()
    check("macro file", r"\gwset{demo_key}{0.42}" in txt and r"37\%" in txt)
    rows = []
    for j, det in enumerate(DETECTOR_ORDER):
        for att in ("hide", "change", "forge", "a5"):
            for thr in (0.5, 1, 2, 4, 8):
                rate = 2000 / thr ** (1.5 + 0.1 * j)
                rows.append(dict(dataset="synthetic", detector=det, attack=att, threshold=thr,
                                 benign_alarms=rate * 0.03, exposure_days=0.03,
                                 det=max(0, 0.9 - 0.1 * j - 0.05 * thr), placebo=0.02))
    pd.DataFrame(rows).to_csv(tmp / "sweep.csv", index=False)
    pd.DataFrame([dict(home=h, minutes=m, prior=p, detection=0.2 + 0.05 * m + 0.1 * p,
                       benign_alarms=1, exposure_days=0.03)
                  for h in (1, 2) for m in (1, 2, 3, 5, 8) for p in (0, 1)]).to_csv(tmp / "cold.csv", index=False)
    cfg = dict(DEFAULT_CONFIG, sweep=str(tmp / "sweep.csv"), coldstart=str(tmp / "cold.csv"),
               tex_dir=str(tmp), out_dir=str(tmp / "res"))
    cmd_figs(cfg, R)
    check("figure written", (tmp / "figs" / "fig5_matched_coldstart.pdf").exists())
    print("SELFTEST", "PASSED" if ok else "FAILED")
    return ok


# ════════════════════════════════════════════════════════════════════════════
def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("cmd", choices=["selftest", "crosscheck", "stats", "markov", "counters",
                                    "songlin", "sweeps", "figs", "all"])
    ap.add_argument("--config", help="JSON config overriding DEFAULT_CONFIG")
    args = ap.parse_args()
    cfg = json.loads(json.dumps(DEFAULT_CONFIG))
    if args.config:
        user = json.loads(Path(args.config).read_text())
        for k, v in user.items():
            if isinstance(v, dict) and isinstance(cfg.get(k), dict):
                cfg[k].update(v)
            else:
                cfg[k] = v
    if args.cmd == "selftest":
        sys.exit(0 if selftest(Path(cfg["out_dir"]) / "selftest") else 1)
    R = Results(Path(cfg["out_dir"]), Path(cfg["tex_dir"]))
    if cfg.get("artifact_url"):
        R.set("artifact_url", rf"\url{{{cfg['artifact_url']}}}")
    if args.cmd == "crosscheck":
        sys.exit(0 if cmd_crosscheck(cfg, R) else 1)
    steps = {"stats": cmd_stats, "markov": cmd_markov, "counters": cmd_counters,
             "songlin": cmd_songlin,
             "sweeps": cmd_sweeps, "figs": cmd_figs}
    for name, fn in steps.items():
        if args.cmd in (name, "all"):
            fn(cfg, R)
    R.save()


if __name__ == "__main__":
    main()
