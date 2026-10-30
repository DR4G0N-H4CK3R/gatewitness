"""Channel measurement and commissioning test per home (results/channel.json).

For each home, all six paired captures form one session.  Reports the
permutation commissioning test (D, p, 5th-percentile block-bootstrap bound,
admitted) and the per-device channel parameters eps, 1-gamma and
s = (1-eps)(1-gamma), measured by self-injection, as [min, median, max] over
devices with at least 20 eligible uplinks (paper Sec. III/IV)."""
import json
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).parent
sys.path.insert(0, str(HERE))
import folds as FO   # noqa: E402
import gw_core as C  # noqa: E402

MIN_UPLINKS = 20


def mmm(x):
    return [float(np.min(x)), float(np.median(x)), float(np.max(x))]


def main():
    out = {}
    for topo in "AB":
        S = C.build_session([p for p in FO.P if p["topology"] == topo], topo)
        t = C.commissioning(S)
        cs = C.channel_stats(S)
        cs = cs[cs.n >= MIN_UPLINKS]
        out[topo] = dict(D=float(t["D"]), p=float(t["p"]), q05=float(t["q05"]), admitted=bool(t["admitted"]),
                         minutes=float(S.t_end / 60), eps=mmm(cs.eps), omg=mmm(cs.one_minus_gamma),
                         s=mmm(cs.s), devices=int(len(cs)))
    (HERE / "results").mkdir(exist_ok=True)
    (HERE / "results" / "channel.json").write_text(json.dumps(out, indent=1))
    print(json.dumps(out, indent=1))


if __name__ == "__main__":
    main()
