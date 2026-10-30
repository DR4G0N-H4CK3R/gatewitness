"""Hub-side IP parsing for ZIOTP2025.

* Every packet in the ZIOTP2025 tcpdumps appears twice (same IP id and TCP
  sequence number, ~50 us apart); duplicates are removed on
  (src, dst, sport, dport, ip_id, tcp_seq, payload length).
* The Home Assistant push channel is a websocket on tcp/8123
  (permessage-deflate).  TCP is reassembled per direction and websocket
  frame headers are walked; each hub->client data frame (opcode 1/2, or a
  continuation that ends a message) is one *message*.  Control frames
  (close/ping/pong) are kept separately.
* Every other hub flow yields one egress record per de-duplicated segment
  (TLS records where the stream is TLS).
Only headers and lengths are read; payloads are never decompressed.
"""
from __future__ import annotations

import struct
from collections import defaultdict
from pathlib import Path

import pandas as pd

HUB = "192.168.8.116"
WS_PORT = 8123


def _packets(path: Path):
    from scapy.all import RawPcapNgReader, RawPcapReader
    magic = open(path, "rb").read(4)
    rd = RawPcapNgReader(str(path)) if magic == b"\x0a\x0d\x0d\x0a" else RawPcapReader(str(path))
    with rd:
        for pkt, meta in rd:
            if hasattr(meta, "tshigh"):
                ts = (meta.tshigh * 2 ** 32 + meta.tslow) / meta.tsresol
                lt = meta.linktype
            else:
                ts = meta.sec + meta.usec / 1e6
                lt = rd.linktype
            yield ts, lt, bytes(pkt)


def _ipv4(lt, pkt):
    if lt == 1:
        et = struct.unpack("!H", pkt[12:14])[0]; off = 14
        if et == 0x8100:
            et = struct.unpack("!H", pkt[16:18])[0]; off = 18
    elif lt == 113:
        et = struct.unpack("!H", pkt[14:16])[0]; off = 16
    elif lt == 276:   # Linux cooked v2
        et = struct.unpack("!H", pkt[0:2])[0]; off = 20
    else:
        return None
    return pkt[off:] if et == 0x0800 else None


def segments(path: Path) -> pd.DataFrame:
    rows, seen = [], set()
    for ts, lt, pkt in _packets(path):
        ip = _ipv4(lt, pkt)
        if ip is None or len(ip) < 20:
            continue
        ihl = (ip[0] & 15) * 4
        tot, ipid = struct.unpack("!HH", ip[2:6])
        proto = ip[9]
        src, dst = ".".join(map(str, ip[12:16])), ".".join(map(str, ip[16:20]))
        if HUB not in (src, dst):
            continue
        l4 = ip[ihl:tot]
        if proto == 6 and len(l4) >= 20:
            sp, dp, seq = struct.unpack("!HHI", l4[:8])
            flags = l4[13]
            pl = l4[(l4[12] >> 4) * 4:]
        elif proto == 17 and len(l4) >= 8:
            sp, dp = struct.unpack("!HH", l4[:4]); seq, flags = 0, 0
            pl = l4[8:]
        else:
            continue
        key = (src, dst, sp, dp, ipid, seq, len(pl))
        if key in seen:
            continue
        seen.add(key)
        rows.append((ts, src, dst, sp, dp, proto, seq, flags, pl))
    return pd.DataFrame(rows, columns="ts src dst sport dport proto seq flags payload".split())


def _ws_walk(chunks):
    """chunks: list of (ts, bytes) in TCP order for one direction after the
    HTTP upgrade.  Yields (ts_end, opcode, fin, payload_len, masked)."""
    buf = b""
    for ts, data, gap in chunks:
        if gap:                      # TCP sequence gap: resynchronise at this segment
            buf = b""
        buf += data
        while True:
            if len(buf) < 2:
                break
            b0, b1 = buf[0], buf[1]
            n, i = b1 & 0x7F, 2
            if n == 126:
                if len(buf) < 4:
                    break
                n = struct.unpack("!H", buf[2:4])[0]; i = 4
            elif n == 127:
                if len(buf) < 10:
                    break
                n = struct.unpack("!Q", buf[2:10])[0]; i = 10
            masked = bool(b1 & 0x80)
            op = b0 & 0x0F
            if op not in (0, 1, 2, 8, 9, 10) or (op >= 8 and n > 125):
                buf = b""            # lost framing: wait for the next segment
                break
            i += 4 if masked else 0
            if len(buf) < i + n:
                break
            yield ts, b0 & 0x0F, bool(b0 & 0x80), n, masked
            buf = buf[i + n:]


def records(path: Path) -> tuple[pd.DataFrame, dict]:
    seg = segments(path)
    out, stats = [], {"segments": len(seg)}
    seg = seg[seg.payload.map(len) > 0]
    for (src, dst, sp, dp, proto), g in seg.groupby(["src", "dst", "sport", "dport", "proto"], sort=False):
        outbound = src == HUB
        remote, rport, lport = (dst, dp, sp) if outbound else (src, sp, dp)
        g = g.sort_values(["seq", "ts"]).drop_duplicates("seq")
        if proto == 6 and lport == WS_PORT:
            seqs, lens = g.seq.values, g.payload.map(len).values
            gaps = [False] + [int(seqs[k]) != int(seqs[k - 1]) + int(lens[k - 1]) for k in range(1, len(g))]
            stats["tcp_gaps"] = stats.get("tcp_gaps", 0) + sum(gaps)
            chunks = list(zip(g.ts, g.payload, gaps))
            # skip the HTTP upgrade
            data = b"".join(c for _, c, _ in chunks)
            k = data.find(b"\r\n\r\n")
            if data.startswith((b"GET ", b"HTTP/")) and k >= 0:
                skip, new = k + 4, []
                for ts, c, gp in chunks:
                    if skip >= len(c):
                        skip -= len(c); continue
                    new.append((ts, c[skip:], gp)); skip = 0
                chunks = new
            elif data[:4] in (b"GET ", b"HTTP", b"POST"):
                for ts, c, _ in chunks:
                    out.append(dict(ts=ts, direction="out" if outbound else "in", remote=remote,
                                    port=WS_PORT, kind="http", opcode=-1, length=len(c), flow=f"{remote}:{rport}"))
                continue
            msg_start = None
            for ts, op, fin, n, masked in _ws_walk(chunks):
                if op in (1, 2) or op == 0:
                    if op in (1, 2):
                        msg_start = ts
                    if fin:
                        out.append(dict(ts=ts, direction="out" if outbound else "in", remote=remote,
                                        port=WS_PORT, kind="ws", opcode=op if op else 1, length=n,
                                        flow=f"{remote}:{rport}"))
                else:
                    out.append(dict(ts=ts, direction="out" if outbound else "in", remote=remote,
                                    port=WS_PORT, kind="ws_ctrl", opcode=op, length=n,
                                    flow=f"{remote}:{rport}"))
        else:
            for ts, pl in zip(g.ts, g.payload):
                out.append(dict(ts=ts, direction="out" if outbound else "in", remote=remote,
                                port=rport if outbound else rport, kind="seg", opcode=-1,
                                length=len(pl), flow=f"{remote}:{rport}/{proto}"))
    df = pd.DataFrame(out).sort_values("ts", kind="stable").reset_index(drop=True)
    stats["records"] = len(df)
    return df, stats


def reporting_units(rec: pd.DataFrame, resp_window: float = 0.5) -> pd.DataFrame:
    """Hub->client websocket data messages that are not responses: a message is
    a response if it is the first outbound message on its flow within
    `resp_window` s after an inbound client data message."""
    ws = rec[(rec.kind == "ws")]
    out = ws[ws.direction == "out"].copy()
    inb = ws[ws.direction == "in"]
    out["response"] = False
    for flow, g in inb.groupby("flow"):
        idx = out.index[out.flow == flow]
        ts = out.loc[idx, "ts"].values
        used = set()
        for t in g.ts.values:
            for j in range(len(ts)):
                if t <= ts[j] <= t + resp_window and j not in used:
                    used.add(j); break
        out.loc[idx[sorted(used)], "response"] = True
    return out[~out.response].drop(columns="response")
