"""GateWitness replay dashboard.

    python dashboard.py                  # opens http://127.0.0.1:8050 in the browser
    python dashboard.py --port 8080 --no-browser

Pick any episode of the paper's Table II (fold, attack, device, seed). The server
re-runs it live with the same code that produced the paper's numbers. The page
then shows, on one timeline:

  * the device's radio uplinks and the hub's reports, with what the attacker
    removed, delayed, altered, invented, forged or leaked;
  * the attack window and the 60 s scoring tail;
  * the qualifying test's CUSUM statistic rising to its threshold, and the alarm;
  * the live result next to the committed row of out/episodes_main.csv.

A second panel compares every number in the paper with the result files
(the same rows as check_paper.py).

Only the Python standard library is used for the server; the page draws its
charts itself, so nothing is fetched from the internet.
"""
from __future__ import annotations

import argparse
import json
import math
import sys
import threading
import webbrowser
import zlib
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import parse_qs, urlparse

import numpy as np
import pandas as pd

HERE = Path(__file__).parent
sys.path.insert(0, str(HERE))

ATTACKS = {"hide": "Hide 80% of reports", "delay": "Delay reports by 30 s", "change": "Add 60 B to reports",
           "swap": "Swap two devices' reports", "invent": "Invent reports (1/15 s)",
           "leak": "Leak to a new host (1/20 s)", "forge": "Forge frames, host-timed ACK",
           "forgeX": "Forge frames, firmware-exact ACK"}
TEST_NAMES = {"SUP": "suppression test", "SIZE": "report-size test", "FAB": "fabrication test",
              "EMIT": "emitter test", "VOL": "endpoint-volume test", "NEWEP": "new-endpoint rule"}

LOCK = threading.Lock()
STATE: dict = {}


# ─────────────────────────────────────────────────────────────── loading
def load():
    import folds as FO
    import gw_core as C
    import gw_eval as E
    import run_main as RM
    from gwio.datasets import ziotp
    STATE.update(FO=FO, C=C, E=E, RM=RM, ziotp=ziotp, models={}, clean={})
    STATE["folds"] = {(f["home"], f["fold"]): f for f in FO.folds()}
    ref = pd.read_csv(HERE / "out" / "episodes_main.csv")
    STATE["ref"] = ref[ref.B == 1.0]


def model(key):
    if key not in STATE["models"]:
        E = STATE["E"]
        f = STATE["folds"][key]
        M = E.commission(f["comm"])
        MB = E.recalibrate(M, 1.0)
        STATE["models"][key] = (M, MB)
        STATE["clean"][key] = E.alarm_times(MB, f["held"], "bern")
    return STATE["models"][key]


def label(home, d):
    dev = STATE["ziotp"].DEVICES_A if home == "A" else STATE["ziotp"].DEVICES_B
    v = dev.get(int(d))
    return f"{v[0]} ({v[1]})" if v else f"0x{int(d):04x}"


def short(name):
    return name.split("/")[-1]


# ─────────────────────────────────────────────────────────────── evidence streams
def stream(MB, sess, tid):
    """(t, x) evidence units of one Bernoulli/Poisson test, exactly as alarm_times builds them."""
    E = STATE["E"]
    if tid.startswith(("SUP", "SIZE", "FAB")):
        sup, credit, cdev = E.sup_streams(sess)
    if tid.startswith("SUP"):
        return sup.get(int(tid[4:-1]), (np.array([]), np.array([])))
    if tid.startswith("SIZE"):
        d = int(tid[5:-1])
        return E.size_streams(sess, {d: MB.bands[d]}, cdev)[d]
    if tid == "FAB":
        return E.fab_stream(sess, credit)
    if tid.startswith("EMIT"):
        em = E.emit_streams(sess.frames, MB.ack_band, E.coverage_series(sess.frames), "32", segs=sess.segs)
        return em.get(int(tid[5:-1]), (np.array([]), np.array([])))
    if tid.startswith("VOL["):
        ep = MB.tests[tid]["ep"]
        g = sess.egress[(sess.egress.remote == ep[0]) & (sess.egress.port == ep[1])]
        return E.bins(g.ts.values, sess.t_end, segs=sess.segs)
    raise KeyError(tid)


def trajectory(MB, sess, tid):
    """Statistic after every evidence unit, threshold and alarm times.  The CUSUM
    restarts at 0 after an alarm, as in gw_stats.run_cusum."""
    if tid == "NEWEP":
        thr = MB.newep_thr
        eg = sess.egress
        new = eg[[(r, p) not in MB.known_eps for r, p in zip(eg.remote, eg.port)]]
        t, S, al, fired = [], [], [], set()
        seen = {}
        for r in new.itertuples():
            ep = (r.remote, r.port)
            seen.setdefault(ep, []).append(r.ts)
            c = sum(1 for x in seen[ep] if x > r.ts - 60)
            t.append(float(r.ts)); S.append(float(c))
            if c > thr and ep not in fired:
                al.append(float(r.ts)); fired.add(ep)
        return dict(t=t, S=S, h=thr + 0.5, alarms=al, kind="count",
                    ylabel="records to a never-seen host in the last 60 s")
    cal = MB.tests[tid]["cal"]
    t, x = stream(MB, sess, tid)
    S, s, al = [], 0.0, []
    if tid.startswith("VOL["):
        a, b = math.log(cal["lam1"] / cal["lam0"]), cal["lam1"] - cal["lam0"]
        h = cal["h_bern"]
        for ti, xi in zip(t, x):
            s = max(0.0, s + xi * a - b)
            S.append(s)
            if s >= h:
                al.append(float(ti)); s = 0.0
    else:
        l1, l0, h = cal["l1"], cal["l0"], cal["h_bern"]
        for ti, xi in zip(t, x):
            s = max(0.0, s + (l1 if xi else l0))
            S.append(s)
            if s >= h:
                al.append(float(ti)); s = 0.0
    return dict(t=[float(v) for v in t], S=[float(v) for v in S], x=[int(v) for v in x], h=float(h),
                alarms=al, kind="cusum", ylabel="CUSUM statistic (nats)")


def window(tr, lo, hi):
    keep = [i for i, ti in enumerate(tr["t"]) if lo <= ti <= hi]
    out = dict(tr)
    for k in ("t", "S", "x"):
        if k in tr:
            out[k] = [tr[k][i] for i in keep]
    out["alarms"] = [a for a in tr["alarms"] if lo <= a <= hi]
    return out


# ─────────────────────────────────────────────────────────────── one episode
def replay(fold, cls, di, seed):
    C, E, RM = STATE["C"], STATE["E"], STATE["RM"]
    home, k = fold[0].upper(), int(fold[1:])
    key = (home, k)
    f = STATE["folds"][key]
    M, MB = model(key)
    held = f["held"]
    top = f["comm"].uplinks.dev.value_counts().index[:2].tolist()
    d = int(top[di]); d2 = int(top[1 - di]) if len(top) > 1 else d
    rng = np.random.default_rng(zlib.crc32(repr(("main", home, k, cls, int(d), seed)).encode()))
    t0, t1 = RM.episode_windows(held, rng, 1)[0]
    att = RM.attack(cls, held, d, d2, t0, t1, rng, M)
    quals = RM.qualifying(cls, d, d2)
    al = E.alarm_times(MB, att, "bern", only=set(quals))
    det, lat = RM.hit(al, quals, t0, t1)
    plc, _ = RM.hit(STATE["clean"][key], quals, t0, t1)

    lo, hi = max(held.segs[0][1], t0 - 90), min(held.t_end, t1 + E.TAIL + 30)
    segname = " + ".join(short(s[0]) for s in held.segs if s[2] >= t0 and s[1] <= t1)
    gaps = [[float(s[2]), float(n[1])] for s, n in zip(held.segs[:-1], held.segs[1:]) if n[1] > lo and s[2] < hi]

    # tests to show: every qualifying test that exists in this fold
    tids = []
    for q in quals:
        if q.endswith("*"):
            tids += [t for t in MB.tests if t.startswith(q[:-1])]
        elif q == "NEWEP" or q in MB.tests:
            tids.append(q)
    traces = []
    for tid in tids:
        ta = trajectory(MB, att, tid)
        tc = trajectory(MB, held, tid)
        first = [a for a in ta["alarms"] if t0 <= a <= t1 + E.TAIL]
        base = tid.split("[")[0]
        traces.append(dict(tid=tid, name=TEST_NAMES.get(base, tid), attacked=window(ta, lo, hi),
                           clean=window(tc, lo, hi), first_alarm=first[0] if first else None,
                           consistent=bool(np.allclose(sorted(ta["alarms"]),
                                                       sorted(map(float, al.get(tid, [])))))
                           if tid in al else True))
    missing = [q for q in quals if not q.endswith("*") and q != "NEWEP" and q not in MB.tests]
    order = sorted(range(len(traces)), key=lambda i: (traces[i]["first_alarm"] is None,
                                                       traces[i]["first_alarm"] or 0))
    traces = [traces[i] for i in order]

    # radio lane
    up = held.uplinks
    upl = up[(up.dev == d) & (up.ts >= lo) & (up.ts <= hi)].ts.tolist()
    fa = att.frames
    forged = fa[fa.psdu_hash.astype(str).str.startswith("forged") & (fa.ts >= lo) & (fa.ts <= hi)].ts.tolist()

    # report lane: diff clean vs attacked units
    _, _, cdev_c = C.device_streams(held)
    _, _, cdev_a = C.device_streams(att)
    uc, ua = held.units, att.units
    mc = (uc.ts >= lo) & (uc.ts <= hi); ma = (ua.ts >= lo) & (ua.ts <= hi)
    cmap = {round(float(t), 6): int(L) for t, L in zip(uc.ts[mc], uc.length[mc])}
    amap = {round(float(t), 6): int(L) for t, L in zip(ua.ts[ma], ua.length[ma])}
    cdc = {round(float(t), 6): int(c) for t, c in zip(uc.ts[mc], cdev_c[mc.values])}
    cda = {round(float(t), 6): int(c) for t, c in zip(ua.ts[ma], cdev_a[ma.values])}
    reports = []
    for t, L in amap.items():
        if t not in cmap:
            st = "inserted"
        elif cmap[t] != L:
            st = "altered"
        else:
            st = "unchanged"
        reports.append(dict(t=t, len=L, status=st, mine=cda.get(t) == d))
    for t, L in cmap.items():
        if t not in amap:
            reports.append(dict(t=t, len=L, status="removed", mine=cdc.get(t) == d))
    if cls == "delay":
        for r in reports:
            if r["status"] == "inserted":
                r["status"] = "delayed"
    if cls == "invent":
        for r in reports:
            if r["status"] == "inserted":
                r["status"] = "invented"

    # egress lane
    eg = att.egress[(att.egress.ts >= lo) & (att.egress.ts <= hi)]
    ceg = held.egress
    clean_keys = set(zip(ceg.ts.round(6), ceg.remote, ceg.port))
    egress = [dict(t=float(t), new=(r, p) not in MB.known_eps, host=f"{r}:{p}",
                   injected=(round(float(t), 6), r, p) not in clean_keys)
              for t, r, p in zip(eg.ts, eg.remote, eg.port)]

    row = STATE["ref"]
    row = row[(row.experiment == cls) & (row.home == home) & (row.fold == k) & (row.device == d) & (row.seed == seed)]
    committed = None
    if len(row):
        r = row.iloc[0]
        committed = dict(detected=int(r.detected), placebo=int(r.placebo),
                         latency=None if pd.isna(r.latency_s) else float(r.latency_s))
    match = committed is not None and committed["detected"] == int(det) and (
        not det or abs(committed["latency"] - lat) < 1e-6) and committed["placebo"] == int(plc)

    return dict(fold=f"{home}{k}", attack=cls, attack_name=ATTACKS[cls], device=d, device_label=label(home, d),
                seed=seed, t0=t0, t1=t1, tail=E.TAIL, lo=lo, hi=hi, segment=segname, gaps=gaps,
                detected=bool(det), latency=lat, placebo=bool(plc), committed=committed, match=bool(match),
                traces=traces, missing=missing, uplinks=upl, forged=forged, reports=reports, egress=egress,
                n_uplinks_in_window=int(sum(t0 <= t <= t1 for t in upl)))


def fold_info(fold):
    E = STATE["E"]
    home, k = fold[0].upper(), int(fold[1:])
    f = STATE["folds"][(home, k)]
    M, MB = model((home, k))
    held = f["held"]
    core = {t: v for t, v in STATE["clean"][(home, k)].items() if not t.startswith(STATE["RM"].OPT_IN)}
    n = E.count_alarms(core)
    top = f["comm"].uplinks.dev.value_counts().index[:2].tolist()
    kinds = pd.Series([t.split("[")[0] for t in MB.tests]).value_counts().to_dict()
    return dict(fold=f"{home}{k}", commission=[short(x) for x in f["comm_names"]],
                held=[short(x) for x in f["held_names"]], tests=len(MB.tests),
                kinds={str(a): int(b) for a, b in kinds.items()},
                ack_band=MB.ack_band, benign=n, benign_tests=[t for t, v in core.items() if len(v)],
                held_minutes=E.exposure_days(held) * 1440,
                devices=[dict(index=i, addr=int(d), label=label(home, d)) for i, d in enumerate(top)])


def fold_all(fold):
    rows = []
    for cls in ATTACKS:
        for di in (0, 1):
            for seed in range(5):
                r = replay(fold, cls, di, seed)
                rows.append(dict(attack=cls, device=r["device"], seed=seed, detected=r["detected"],
                                 latency=r["latency"], match=r["match"]))
    return dict(fold=fold, n=len(rows), agree=sum(r["match"] for r in rows), rows=rows)


def paper_check():
    import importlib
    import check_paper
    importlib.reload(check_paper)
    rows = [dict(where=w, paper=p, code=c, ok=p == c) for w, p, c in check_paper.ROWS]
    return dict(n=len(rows), agree=sum(r["ok"] for r in rows), rows=rows)


# ─────────────────────────────────────────────────────────────── server
def jsonable(o):
    if isinstance(o, (np.integer,)):
        return int(o)
    if isinstance(o, (np.floating,)):
        return None if not np.isfinite(o) else float(o)
    if isinstance(o, float) and not math.isfinite(o):
        return None
    if isinstance(o, (set, tuple)):
        return list(o)
    raise TypeError(type(o))


class Handler(BaseHTTPRequestHandler):
    def log_message(self, *a):
        pass

    def send(self, code, body, ctype="application/json"):
        data = body if isinstance(body, bytes) else body.encode()
        self.send_response(code)
        self.send_header("Content-Type", ctype + "; charset=utf-8")
        self.send_header("Content-Length", str(len(data)))
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        self.wfile.write(data)

    def do_GET(self):
        u = urlparse(self.path)
        q = {k: v[0] for k, v in parse_qs(u.query).items()}
        try:
            if u.path in ("/", "/index.html"):
                return self.send(200, (HERE / "web" / "dashboard.html").read_bytes(), "text/html")
            with LOCK:
                if u.path == "/api/meta":
                    out = dict(folds=[f"{h}{k}" for h, k in sorted(STATE["folds"])], attacks=ATTACKS)
                elif u.path == "/api/fold":
                    out = fold_info(q.get("fold", "A0"))
                elif u.path == "/api/episode":
                    out = replay(q.get("fold", "A0"), q.get("attack", "forge"), int(q.get("device", 0)),
                                 int(q.get("seed", 0)))
                elif u.path == "/api/fold_all":
                    out = fold_all(q.get("fold", "A0"))
                elif u.path == "/api/check":
                    out = paper_check()
                else:
                    return self.send(404, json.dumps({"error": "not found"}))
            self.send(200, json.dumps(out, default=jsonable, allow_nan=False))
        except Exception as e:      # report errors to the page instead of dropping the connection
            import traceback
            traceback.print_exc()
            self.send(500, json.dumps({"error": f"{type(e).__name__}: {e}"}))


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--port", type=int, default=8050)
    ap.add_argument("--no-browser", action="store_true")
    args = ap.parse_args()
    print("loading the paired captures and building the folds ...", flush=True)
    load()
    print("commissioning fold A0 ...", flush=True)
    model(("A", 0))
    url = f"http://127.0.0.1:{args.port}/"
    srv = ThreadingHTTPServer(("127.0.0.1", args.port), Handler)
    print(f"GateWitness dashboard running at {url}  (Ctrl+C to stop)", flush=True)
    if not args.no_browser:
        threading.Timer(0.5, lambda: webbrowser.open(url)).start()
    try:
        srv.serve_forever()
    except KeyboardInterrupt:
        print("\nstopped")


if __name__ == "__main__":
    main()
