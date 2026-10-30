"""Seven baselines (paper Section VII), each with one swept threshold theta.

Every baseline is fitted on the commissioning session and returns alarm times
on a held-out session.  Device-level baselines return {f"{name}[{d}]": times};
gateway-level ones return {name: times}.  A HAWatcher-style log monitor reads
the hub's own log, which the adversary controls (A5 adds log-only entries).
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.ensemble import IsolationForest

import gw_core as C

STEP, WIN = 12.0, 60.0


def _grid(sess):
    g = []
    for _, a, b in sess.segs:
        g += list(np.arange(a + WIN, b, STEP))
    return np.asarray(g)


def _count(ts, grid, win=WIN):
    ts = np.sort(ts)
    return np.searchsorted(ts, grid) - np.searchsorted(ts, grid - win)


def _k_of_n(flags, grid, k=3, n=5):
    out, last = [], -1e18
    for i in range(len(flags)):
        if flags[max(0, i - n + 1):i + 1].sum() >= k and grid[i] - last > WIN:
            out.append(grid[i]); last = grid[i]
    return np.asarray(out)


def label_units(sess):
    """Perfect device attribution of the wire units (benign matching)."""
    _, _, cdev = C.device_streams(sess)
    u = sess.units.copy(); u["dev_true"] = cdev
    return u


class FixedCount:
    name = "Fixed count"
    def fit(self, comm, devices):
        self.devs = devices
        X, credit, cdev = C.device_streams(comm)
        self.stats = {}
        g = _grid(comm)
        for d in devices:
            up = comm.uplinks.ts.values[comm.uplinks.dev.values == d]
            mt = comm.units.ts.values[cdev == d]
            deficit = _count(up, g) - _count(mt, g)
            self.stats[d] = (deficit.mean(), deficit.std() + 1.0)
        return self
    def alarms(self, sess, theta, log=None):
        X, credit, cdev = C.device_streams(sess)
        g = _grid(sess); out = {}
        for d in self.devs:
            up = sess.uplinks.ts.values[sess.uplinks.dev.values == d]
            mt = sess.units.ts.values[cdev == d]
            mu, sd = self.stats[d]
            out[f"FC[{d}]"] = _k_of_n((_count(up, g) - _count(mt, g)) > mu + theta * sd, g)
        return out


class LogMonitor:
    """Per-device report rate in the hub's log, perfect attribution."""
    name = "Log monitor"
    def fit(self, comm, devices):
        self.devs = devices
        lu = label_units(comm); g = _grid(comm)
        self.stats = {d: (lambda c: (c.mean(), c.std() + 1.0))(_count(lu.ts.values[lu.dev_true == d], g))
                      for d in devices}
        return self
    def alarms(self, sess, theta, log=None):
        lu = log if log is not None else label_units(sess)
        g = _grid(sess); out = {}
        for d in self.devs:
            c = _count(lu.ts.values[lu.dev_true.values == d], g)
            mu, sd = self.stats[d]
            out[f"LOG[{d}]"] = _k_of_n(np.abs(c - mu) > theta * sd, g, 2, 3)
        return out


class RadioOnly:
    """HoMonit-style: device frame counts and mean lengths, no IP side."""
    name = "Radio only"
    def fit(self, comm, devices):
        self.devs = devices; g = _grid(comm); self.stats = {}
        for d in devices:
            f = comm.frames[(comm.frames.ftype == 1) & ((comm.frames.src == d) | (comm.frames.nwk_src == d))]
            c = _count(f.ts.values, g)
            self.stats[d] = (c.mean(), c.std() + 1.0)
        return self
    def alarms(self, sess, theta, log=None):
        g = _grid(sess); out = {}
        for d in self.devs:
            f = sess.frames[(sess.frames.ftype == 1) & ((sess.frames.src == d) | (sess.frames.nwk_src == d))]
            mu, sd = self.stats[d]
            out[f"RO[{d}]"] = _k_of_n(np.abs(_count(f.ts.values, g) - mu) > theta * sd, g, 2, 3)
        return out


class RadioVsHub:
    """HoMonit-style comparator: radio event bursts vs any hub record within 5 s."""
    name = "Radio vs hub"
    def _bursts(self, sess, d):
        t = np.sort(sess.uplinks.ts.values[sess.uplinks.dev.values == d])
        if len(t) == 0:
            return t
        return t[np.concatenate([[True], np.diff(t) > 2.0])]
    def _miss(self, sess, d):
        b = self._bursts(sess, d); u = np.sort(sess.units.ts.values)
        j = np.searchsorted(u, b)
        nxt = np.where(j < len(u), u[np.minimum(j, len(u) - 1)], np.inf)
        return b, (nxt - b > 5.0).astype(float)
    def fit(self, comm, devices):
        self.devs = devices; self.stats = {}
        for d in devices:
            b, m = self._miss(comm, d)
            r = pd.Series(m).rolling(10, min_periods=5).mean().dropna().values
            self.stats[d] = (r.mean() if len(r) else 0.5, (r.std() if len(r) else 0.2) + 0.05)
        return self
    def alarms(self, sess, theta, log=None):
        out = {}
        for d in self.devs:
            b, m = self._miss(sess, d)
            r = pd.Series(m).rolling(10, min_periods=5).mean().values
            mu, sd = self.stats[d]
            flag = np.nan_to_num(r, nan=0) > mu + theta * sd
            al, last = [], -1e18
            for t, fl in zip(b, flag):
                if fl and t - last > WIN:
                    al.append(t); last = t
            out[f"RVH[{d}]"] = np.asarray(al)
        return out


class IPRules:
    """New-endpoint volume, per-endpoint volume z, unit-length novelty."""
    name = "IP rules"
    def fit(self, comm, devices):
        g = _grid(comm)
        self.known = set(map(tuple, comm.egress[["remote", "port"]].drop_duplicates().values))
        self.vol = {}
        for ep, e in comm.egress.groupby(["remote", "port"]):
            c = _count(e.ts.values, g); self.vol[ep] = (c.mean(), c.std() + 1.0)
        L = comm.units.length.values
        self.lo, self.hi = np.percentile(L, [0.5, 99.5])
        nov = _count(comm.units.ts.values[(L < self.lo) | (L > self.hi)], g)
        self.nov = (nov.mean(), nov.std() + 1.0)
        return self
    def alarms(self, sess, theta, log=None):
        g = _grid(sess); flags = np.zeros(len(g), bool)
        for ep, e in sess.egress.groupby(["remote", "port"]):
            c = _count(e.ts.values, g)
            if ep in self.known:
                mu, sd = self.vol[ep]; flags |= c > mu + theta * sd
            else:
                flags |= c > max(1.0, theta)
        L = sess.units.length.values
        c = _count(sess.units.ts.values[(L < self.lo) | (L > self.hi)], g)
        flags |= c > self.nov[0] + theta * self.nov[1]
        return {"IPR": _k_of_n(flags, g, 1, 1)}


class IsoForest:
    name = "IsolationForest"
    def _feat(self, sess):
        g = _grid(sess)
        u = np.sort(sess.units.ts.values); L = sess.units.length.values[np.argsort(sess.units.ts.values)]
        F = []
        for t in g:
            a, b = np.searchsorted(u, [t - WIN, t])
            seg = L[a:b]
            e = sess.egress[(sess.egress.ts > t - WIN) & (sess.egress.ts <= t)]
            F.append([b - a, seg.mean() if len(seg) else 0, seg.std() if len(seg) > 1 else 0,
                      len(e), e[["remote", "port"]].drop_duplicates().shape[0]])
        return g, np.asarray(F, float)
    def fit(self, comm, devices):
        _, F = self._feat(comm)
        self.m = IsolationForest(n_estimators=200, random_state=0).fit(F)
        self.ref = np.sort(self.m.score_samples(F))
        return self
    def alarms(self, sess, theta, log=None):
        g, F = self._feat(sess)
        s = self.m.score_samples(F)
        thr = np.quantile(self.ref, 0.2 * 2.0 ** (-theta))   # theta sweeps from the 20% benign quantile down
        return {"IF": _k_of_n(s < thr, g, 1, 1)}


class PingPong:
    """Length-signature inference of device events from IP alone."""
    name = "PingPong-style"
    def fit(self, comm, devices):
        lu = label_units(comm); self.devs = devices
        vc = lu[lu.dev_true >= 0].groupby(["length", "dev_true"]).size().reset_index(name="n")
        own = vc.sort_values("n", ascending=False).drop_duplicates("length")
        own = own[own.n >= 3]
        tot = vc.groupby("length").n.sum()
        purity = own.set_index("length").n / tot.loc[own.length].values
        self.sig = {int(l): int(d) for l, d, p in zip(own.length, own.dev_true, purity.values) if p >= 0.8}
        g = _grid(comm); self.stats = {}
        for d in devices:
            t = comm.units.ts.values[np.isin(comm.units.length.values, [l for l, dd in self.sig.items() if dd == d])]
            c = _count(t, g); self.stats[d] = (c.mean(), c.std() + 1.0)
        return self
    def alarms(self, sess, theta, log=None):
        g = _grid(sess); out = {}
        for d in self.devs:
            t = sess.units.ts.values[np.isin(sess.units.length.values, [l for l, dd in self.sig.items() if dd == d])]
            mu, sd = self.stats[d]
            out[f"PP[{d}]"] = _k_of_n(np.abs(_count(t, g) - mu) > theta * sd, g, 2, 3)
        return out


ALL = [FixedCount, LogMonitor, RadioOnly, RadioVsHub, IPRules, IsoForest, PingPong]
THETAS = [0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0, 8.0, 12.0]
