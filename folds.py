import pickle, sys
from pathlib import Path
HERE = Path(__file__).parent
sys.path.insert(0, str(HERE))
import gw_core as C
import gw_load
P = gw_load.load_pairs()   # parses ZIOTP2025 once, then reads cache/pairs.pkl
def folds(long=False):
    out = []
    for topo in "AB":
        pp = [p for p in P if p["topology"] == topo]
        for k in range(3):
            pair = {2 * k, 2 * k + 1}
            if long:
                comm = [p for i, p in enumerate(pp) if i in pair]; held = [p for i, p in enumerate(pp) if i not in pair]
            else:
                held = [p for i, p in enumerate(pp) if i in pair]; comm = [p for i, p in enumerate(pp) if i not in pair]
            out.append(dict(home=topo, fold=k, comm=C.build_session(comm, f"{topo}{k}c"),
                            held=C.build_session(held, f"{topo}{k}h"),
                            comm_names=[p["name"] for p in comm], held_names=[p["name"] for p in held]))
    return out
