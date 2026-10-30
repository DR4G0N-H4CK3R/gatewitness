"""Derived numbers for the paper text (results/derived.json)."""
import json, math, sys, time, hashlib
import numpy as np, pandas as pd
import os; os.chdir(os.path.dirname(os.path.abspath(__file__))); sys.path.insert(0, '.')
import folds as FO, gw_eval as E, gw_core as C, gw_stats as rv
OUT = {}
ch = json.load(open("results/channel.json"))
# pooled device medians
import pickle
P = pickle.load(open("cache/pairs.pkl", "rb"))
cs = pd.concat([C.channel_stats(C.build_session([p for p in P if p["topology"] == t], t)).assign(home=t) for t in "AB"])
cs = cs[cs.n >= 20]
eps, s = float(cs.eps.median()), float(cs.s.median())
OUT["pooled"] = dict(eps=eps, s=s, omg=float(cs.one_minus_gamma.median()), devices=len(cs))
z = 2 * math.sqrt(math.log(20))
OUT["n_certify_f1"] = 2 * math.log(20) / s ** 2                     # Prop 1(a), alpha=beta=0.05
OUT["fmin_100_a"] = z / (math.sqrt(200) * s)                          # achievable with n=100
OUT["fmin_100_b"] = 0.9 * math.sqrt(2 * eps * (1 - eps) / 100) / s    # lower bound, alpha+beta<=0.1
kl = lambda p, q: p * math.log(p / q) + (1 - p) * math.log((1 - p) / (1 - q))
OUT["I_f08"] = kl(eps + 0.8 * s, eps)
OUT["ideal_ratio"] = ((1 - eps) / s) ** 2   # evidence needed vs an unmasked channel (s = 1-eps)
# floor: per-episode nats, u_d, latency vs prediction
ep = pd.read_csv("out/episodes_main.csv"); h = ep[(ep.B == 1.0) & (ep.experiment == "hide")]
el = pd.read_csv("out/episodes_long.csv"); lh = el[el.B == 1.0]
nats, lnud, ratio, phis, ab = [], [], [], [], []
for f in FO.folds():
    M = E.commission(f["comm"])
    for k, T in M.tests.items():
        if k.startswith("SUP["):
            c = T["cal"]; phis.append(c["phi"]); ab.append((c["a"], c["b"]))
    for d in f["comm"].uplinks.dev.value_counts().index[:2]:
        T = M.tests[f"SUP[{d}]"]; c = T["cal"]
        If = kl(min(c["eps"] + 0.8 * T["s"], 0.999), c["eps"])
        lnud.append(math.log(c["u_d"]))
        g = h[(h.home == f["home"]) & (h.fold == f["fold"]) & (h.device == d)]
        nats += list(g.n_uplinks * If)
for f in FO.folds(long=True):
    M = E.commission(f["comm"])
    for d in f["comm"].uplinks.dev.value_counts().index[:2]:
        g = lh[(lh.home == f["home"]) & (lh.fold == f["fold"]) & (lh.device == d) & (lh.detected == 1)]
        for r in g.itertuples():
            rate_d = r.n_uplinks / (r.t1 - r.t0)
            if np.isfinite(r.pred_uplinks) and rate_d > 0:
                ratio.append(r.latency_s / (r.pred_uplinks / rate_d))
OUT["episode_nats"] = [float(np.min(nats)), float(np.max(nats))]
OUT["ln_ud_min"] = float(min(lnud)); OUT["ud_min_per_s"] = float(math.exp(min(lnud)) / 86400)
OUT["long_latency_over_pred_median"] = float(np.median(ratio))
OUT["phi_range"] = [float(min(phis)), float(max(phis))]; OUT["phi_gt1"] = float(np.mean(np.array(phis) > 1.0))
abn = np.array(ab); OUT["persistent_frac"] = float(np.mean(abn[:, 1] > abn[:, 0])); OUT["b_minus_a_median"] = float(np.median(abn[:, 1] - abn[:, 0]))
OUT["tests_per_fold"] = [len(E.commission(f["comm"]).tests) for f in FO.folds()]
# synthetic Markov study at this channel
rng = np.random.default_rng(7)
p1 = eps + 0.5 * s; l1, l0 = rv.cusum_increments(eps, p1); target = 5000
syn = []
for rho in (0.1, 0.2, 0.3, 0.4):
    a, b = eps * (1 - rho), eps + rho * (1 - eps)
    def chain(n):
        x = np.zeros(n, int); u = rng.random(n)
        for i in range(1, n): x[i] = u[i] < (b if x[i - 1] else a)
        return x
    comm = chain(20000); test = chain(1_500_000)
    hB, _ = rv.threshold_for_arl(l1, l0, target, eps, eps)
    au, bu = rv.markov_fit_upper(comm)
    hM, _ = rv.threshold_for_arl(l1, l0, target, au, bu)
    nom = len(test) / target
    syn.append(dict(rho=rho, bern=len(rv.run_cusum(test, l1, l0, hB)) / nom, markov=len(rv.run_cusum(test, l1, l0, hM)) / nom))
OUT["synthetic"] = syn
# evidence log benchmark
from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PrivateKey
key = Ed25519PrivateKey.generate(); pub = key.public_key()
recs = [json.dumps(dict(i=i, t=i * 0.1, test="SUP[1]", x=i % 2)).encode() for i in range(20000)]
t0 = time.perf_counter(); chain_h, sigs, hcur = [], {}, b"\0" * 32
for i, r in enumerate(recs):
    hcur = hashlib.sha256(hcur + r).digest(); chain_h.append(hcur)
    if (i + 1) % 64 == 0: sigs[i] = key.sign(hcur)
OUT["log_us_per_decision"] = (time.perf_counter() - t0) / len(recs) * 1e6
def verify(rs, sg):
    hc = b"\0" * 32
    for i, r in enumerate(rs):
        hc = hashlib.sha256(hc + r).digest()
        if i in sg:
            try: pub.verify(sg[i], hc)
            except Exception: return False
    return True
tam = {"modify": recs[:100] + [b"x"] + recs[101:], "insert": recs[:100] + [b"y"] + recs[100:],
       "delete": recs[:100] + recs[101:], "reorder": recs[:100] + [recs[101], recs[100]] + recs[102:],
       "truncate_before_checkpoint": recs[:130]}
OUT["tamper_detected"] = {k: (not verify(v, sigs)) if k != "truncate_before_checkpoint" else
                          (not verify(v, sigs) or len(v) - 1 < max(i for i in sigs if i < len(v)) + 64) for k, v in tam.items()}
OUT["verify_clean"] = verify(recs, sigs)
# PingPong signature coverage
import baselines as BL
cov = []
for f in FO.folds():
    pp = BL.PingPong().fit(f["comm"], [])
    lu = BL.label_units(f["held"]); cov.append(np.isin(lu.length.values, list(pp.sig)).mean())
OUT["pingpong_unit_coverage"] = [float(min(cov)), float(max(cov))]
json.dump(OUT, open("results/derived.json", "w"), indent=1, default=float)
print(json.dumps(OUT, indent=1, default=float))
