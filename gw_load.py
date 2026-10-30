"""Load the twelve paired ZIOTP2025 captures into cached frame/record tables.

Radio: own parser (adds the 32-bit NWK security frame counter, ACK-request
bit, PSDU hash), reusing the CRC/trailer logic in gwio.capture.radio.  IP: the gwio
`read_pcapng_hub` (hub flows only, TLS records or per-segment records).
Both streams share the laptop clock (whsniff + tcpdump on one machine).
"""
from __future__ import annotations

import hashlib
import os
import pickle
import struct
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent))
from gwio.capture import radio as R            # noqa: E402
from gwio.capture.ip import read_pcapng_hub    # noqa: E402
from gwio.datasets import ziotp                # noqa: E402

# Folder that contains ZIOTP2025's "Data/" directory (set GW_DATA or put it in ./data).
DATA = Path(os.environ.get("GW_DATA", Path(__file__).parent / "data"))
CACHE = Path(__file__).parent / "cache"
HUB_IP = ziotp.HUB_IP


def parse_frame(ts: float, frame: bytes) -> dict | None:
    if len(frame) < 3:
        return None
    fcf = struct.unpack("<H", frame[:2])[0]
    ftype = fcf & 7
    seq = frame[2]
    dm, sm = (fcf >> 10) & 3, (fcf >> 14) & 3
    panc = bool(fcf & 0x40)
    off, pan, dst, src = 3, None, None, None
    try:
        if dm:
            pan = struct.unpack("<H", frame[off:off + 2])[0]; off += 2
            if dm == 2:
                dst = struct.unpack("<H", frame[off:off + 2])[0]; off += 2
            else:
                off += 8
        if sm:
            if not panc or not dm:
                if pan is None:
                    pan = struct.unpack("<H", frame[off:off + 2])[0]
                off += 2
            if sm == 2:
                src = struct.unpack("<H", frame[off:off + 2])[0]; off += 2
            else:
                off += 8
    except struct.error:
        return None
    d = dict(ts=ts, ftype=ftype, seq=seq, src=src, dst=dst, pan=pan,
             ack_req=bool(fcf & 0x20), mac_sec=bool(fcf & 0x08), length=len(frame) + 2,
             nwk_src=None, nwk_dst=None, nwk_type=None, nwk_seq=None, fc32=None, sec_src=None,
             psdu_hash=hashlib.blake2b(frame, digest_size=8).hexdigest())
    p = frame[off:]
    if ftype == 1 and not (fcf & 0x08) and len(p) >= 8:
        nfc = struct.unpack("<H", p[:2])[0]
        if (nfc >> 2) & 0xF == 2:
            d["nwk_type"] = nfc & 3
            d["nwk_dst"], d["nwk_src"] = struct.unpack("<HH", p[2:6])
            d["nwk_seq"] = p[7]
            i = 8
            if nfc & (1 << 11):
                i += 8
            if nfc & (1 << 12):
                i += 8
            if nfc & (1 << 8):
                i += 1
            if nfc & (1 << 10) and len(p) > i + 1:
                i += 2 + 2 * p[i]
            if nfc & (1 << 9) and len(p) >= i + 5:
                sc = p[i]
                d["fc32"] = struct.unpack("<I", p[i + 1:i + 5])[0]
                if sc & 0x20 and len(p) >= i + 13:
                    d["sec_src"] = struct.unpack("<Q", p[i + 5:i + 13])[0]
    return d


def read_radio(path: Path) -> tuple[pd.DataFrame, dict]:
    trailer = R._detect_trailer(path)
    rows, crc_fail, total = [], 0, 0
    for ts, lt, pkt in R._iter_any(path):
        total += 1
        if lt != R.DLT_IEEE802_15_4_WITHFCS or len(pkt) < 5:
            continue
        if trailer == "fcs":
            if R._crc16_802154(pkt[:-2]) != struct.unpack("<H", pkt[-2:])[0]:
                crc_fail += 1
                continue
        elif not (pkt[-1] & 0x80):
            crc_fail += 1
            continue
        d = parse_frame(ts, pkt[:-2])
        if d:
            rows.append(d)
    df = pd.DataFrame(rows).sort_values("ts", kind="stable").reset_index(drop=True)
    return df, {"trailer": trailer, "crc_fail": crc_fail, "total": total}


def read_ip(path: Path) -> pd.DataFrame:
    recs = read_pcapng_hub(path, HUB_IP)
    return pd.DataFrame([dict(ts=r.ts, direction=r.direction, length=r.length,
                              remote=r.dst_ip if r.direction == "out" else r.src_ip,
                              port=r.dst_port, flow=r.flow_id, kind=r.kind, sni=r.sni)
                         for r in recs])


def load_pairs(force: bool = False) -> list[dict]:
    CACHE.mkdir(exist_ok=True)
    f = CACHE / "pairs.pkl"
    if f.exists() and not force:
        return pickle.loads(f.read_bytes())
    out = []
    for cap in ziotp.paired(DATA):
        fr, st = read_radio(cap.radio)
        ip = read_ip(cap.ip)
        out.append(dict(name=cap.name, topology=cap.topology, radio=str(cap.radio),
                        ip_path=str(cap.ip), frames=fr, ip=ip, stats=st))
        print(f"{cap.name:42s} frames={len(fr):6d} crc_fail={st['crc_fail']:4d} ip={len(ip):6d} "
              f"span={fr.ts.max() - fr.ts.min():6.0f}s ip={Path(cap.ip).name}", flush=True)
    f.write_bytes(pickle.dumps(out))
    return out


if __name__ == "__main__":
    load_pairs(force=True)
