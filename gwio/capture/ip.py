"""Read the IP tap on the gateway's uplink.

The reader segments inside long-lived TLS sessions: each TLS
application-data record has an observable length and arrival time even
though its content is encrypted, and those records — not five-tuples — are
the unit the translation invariant is defined over.  This module reassembles
TCP payload per flow and direction, walks the TLS record layer, and emits one
IPRecord per application-data record (type 0x17).  Non-TLS packets are
emitted one IPRecord per packet (kind="packet").  The SNI is read from the
ClientHello where present and used only as an endpoint label.

Inputs: a pcap (parsed with scapy), or a CSV from tshark
    tshark -r ip.pcap -Y "tls.record.content_type==23" -T fields -E separator=, -E header=y \
      -e frame.time_epoch -e ip.src -e ip.dst -e tcp.srcport -e tcp.dstport \
      -e tls.record.length -e tls.handshake.extensions_server_name
"""
from __future__ import annotations

import struct
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import pandas as pd

from ..schema import DIR_IN, DIR_OUT, KIND_PACKET, KIND_TLS_RECORD, IPRecord

TLS_APP_DATA, TLS_HANDSHAKE = 0x17, 0x16


def _sni_from_client_hello(payload: bytes) -> Optional[str]:
    try:
        if payload[0] != TLS_HANDSHAKE or payload[5] != 0x01:
            return None
        p = 9 + 2 + 32                       # handshake hdr, version, random
        sid_len = payload[p]; p += 1 + sid_len
        cs_len = struct.unpack(">H", payload[p:p + 2])[0]; p += 2 + cs_len
        cm_len = payload[p]; p += 1 + cm_len
        ext_len = struct.unpack(">H", payload[p:p + 2])[0]; p += 2
        end = p + ext_len
        while p + 4 <= end:
            et, el = struct.unpack(">HH", payload[p:p + 4]); p += 4
            if et == 0:                      # server_name
                name_len = struct.unpack(">H", payload[p + 3:p + 5])[0]
                return payload[p + 5:p + 5 + name_len].decode("ascii", "replace")
            p += el
    except (IndexError, struct.error):
        return None
    return None


class _Stream:
    """Byte stream of one flow direction with a TLS record walker."""

    def __init__(self):
        self.buf = b""
        self.tls = False
        self.sni: Optional[str] = None

    def feed(self, ts: float, data: bytes) -> List[Tuple[float, int]]:
        """Return (ts, record_length) for each complete application-data record."""
        self.buf += data
        out = []
        if not self.tls and len(self.buf) >= 5 and self.buf[0] in (TLS_HANDSHAKE, TLS_APP_DATA) \
                and self.buf[1] == 3:
            self.tls = True
        if not self.tls:
            return out
        while len(self.buf) >= 5:
            ctype, _, _, rlen = struct.unpack(">BBBH", self.buf[:5])
            if ctype not in (0x14, 0x15, 0x16, 0x17) or rlen > 1 << 14 + 8:
                self.tls = False                      # lost sync: fall back to packets
                self.buf = b""
                return out
            if len(self.buf) < 5 + rlen:
                break
            rec, self.buf = self.buf[:5 + rlen], self.buf[5 + rlen:]
            if ctype == TLS_HANDSHAKE and self.sni is None:
                self.sni = _sni_from_client_hello(rec)
            if ctype == TLS_APP_DATA:
                out.append((ts, 5 + rlen))
        return out


def read_pcap(path: Path, gateway_ip: str) -> List[IPRecord]:
    from scapy.all import IP, TCP, UDP, PcapReader  # local import: scapy is slow to load

    streams: Dict[Tuple, _Stream] = defaultdict(_Stream)
    out: List[IPRecord] = []
    with PcapReader(str(path)) as rd:
        for pkt in rd:
            if IP not in pkt:
                continue
            ip = pkt[IP]
            ts = float(pkt.time)
            outbound = ip.src == gateway_ip
            if not outbound and ip.dst != gateway_ip:
                continue
            direction = DIR_OUT if outbound else DIR_IN
            remote_ip = ip.dst if outbound else ip.src
            if TCP in pkt:
                tcp = pkt[TCP]
                lport, rport = (tcp.sport, tcp.dport) if outbound else (tcp.dport, tcp.sport)
                flow = f"{gateway_ip}:{lport}->{remote_ip}:{rport}"
                payload = bytes(tcp.payload)
                if not payload:
                    continue
                st = streams[(flow, direction)]
                recs = st.feed(ts, payload)
                sni = streams[(flow, DIR_OUT)].sni
                if st.tls:
                    for rts, rlen in recs:
                        out.append(IPRecord(rts, direction, rlen, remote_ip if outbound else gateway_ip,
                                            rport, ip.src, sni, flow, KIND_TLS_RECORD))
                else:
                    out.append(IPRecord(ts, direction, len(payload), remote_ip if outbound else gateway_ip,
                                        rport, ip.src, sni, flow, KIND_PACKET))
            elif UDP in pkt:
                udp = pkt[UDP]
                rport = udp.dport if outbound else udp.sport
                flow = f"{gateway_ip}->{remote_ip}:{rport}/udp"
                out.append(IPRecord(ts, direction, len(bytes(udp.payload)),
                                    remote_ip if outbound else gateway_ip, rport, ip.src, None, flow,
                                    KIND_PACKET))
    out.sort(key=lambda r: r.ts)
    return out


def read_pcapng_hub(path: Path, gateway_ip: str) -> List[IPRecord]:
    """Read a pcap/pcapng (Ethernet or Linux cooked capture) and keep only the
    flows that have the gateway as an endpoint.  TLS flows are segmented into
    application-data records; other TCP flows yield one record per segment
    with payload; UDP one per datagram.  Payload is never inspected beyond
    the TLS record header and the ClientHello SNI."""
    from scapy.all import IP, TCP, UDP, PcapNgReader, PcapReader
    Reader = PcapNgReader if open(path, "rb").read(4) == b"\x0a\x0d\x0d\x0a" else PcapReader
    streams: Dict[Tuple, _Stream] = defaultdict(_Stream)
    out: List[IPRecord] = []
    with Reader(str(path)) as rd:
        for pkt in rd:
            if IP not in pkt:
                continue
            ip = pkt[IP]
            outbound = ip.src == gateway_ip
            if not outbound and ip.dst != gateway_ip:
                continue
            ts = float(pkt.time)
            direction = DIR_OUT if outbound else DIR_IN
            remote_ip = ip.dst if outbound else ip.src
            if TCP in pkt:
                tcp = pkt[TCP]
                payload = bytes(tcp.payload)
                if not payload:
                    continue
                # the flow is keyed by the *service* port so both directions share it
                lport, rport = (tcp.sport, tcp.dport) if outbound else (tcp.dport, tcp.sport)
                svc_port = lport if lport < rport else rport
                flow = f"{gateway_ip}:{lport}<->{remote_ip}:{rport}"
                st = streams[(flow, direction)]
                recs = st.feed(ts, payload)
                sni = streams[(flow, DIR_OUT)].sni or streams[(flow, DIR_IN)].sni
                if st.tls:
                    for rts, rlen in recs:
                        out.append(IPRecord(rts, direction, rlen, remote_ip if outbound else gateway_ip,
                                            svc_port, ip.src, sni, flow, KIND_TLS_RECORD))
                else:
                    out.append(IPRecord(ts, direction, len(payload), remote_ip if outbound else gateway_ip,
                                        svc_port, ip.src, None, flow, KIND_PACKET))
            elif UDP in pkt:
                udp = pkt[UDP]
                rport = udp.dport if outbound else udp.sport
                lport = udp.sport if outbound else udp.dport
                out.append(IPRecord(ts, direction, len(bytes(udp.payload)),
                                    remote_ip if outbound else gateway_ip, min(lport, rport), ip.src, None,
                                    f"{gateway_ip}<->{remote_ip}:{rport}/udp", KIND_PACKET))
    out.sort(key=lambda r: r.ts)
    return out


def read_tshark_csv(path: Path, gateway_ip: str) -> List[IPRecord]:
    df = pd.read_csv(path)
    col = lambda name: next((c for c in df.columns if c.endswith(name)), None)
    c_ts, c_src, c_dst = col("frame.time_epoch"), col("ip.src"), col("ip.dst")
    c_sp, c_dp, c_len, c_sni = col("tcp.srcport"), col("tcp.dstport"), col("tls.record.length"), col("server_name")
    out = []
    for r in df.itertuples(index=False):
        g = lambda c: getattr(r, c.replace(".", "_")) if c else None
        src, dst = str(g(c_src)), str(g(c_dst))
        outbound = src == gateway_ip
        lengths = str(g(c_len)).split(",") if c_len else ["0"]
        for L in lengths:                      # several records in one segment
            try:
                rlen = int(float(L)) + 5
            except ValueError:
                continue
            out.append(IPRecord(float(g(c_ts)), DIR_OUT if outbound else DIR_IN, rlen, dst,
                                int(g(c_dp)), src, None if not c_sni or pd.isna(g(c_sni)) else str(g(c_sni)),
                                f"{src}:{g(c_sp)}->{dst}:{g(c_dp)}", KIND_TLS_RECORD))
    out.sort(key=lambda r: r.ts)
    return out


def normalise_server_ports(records: List[IPRecord], server_ports: Sequence[int],
                           label: str = "subscribers") -> List[IPRecord]:
    """Where the gateway is a *server* (e.g. Home Assistant on tcp/8123), every
    client that connects is one logical endpoint — the push channel — rather
    than a new host:port per session.  Signatures then transfer across runs."""
    ports = set(server_ports)
    for r in records:
        if r.dst_port in ports:
            r.sni = f"{label}-{r.dst_port}"
    return records


def read_ip(path: Path, gateway_ip: str, server_ports: Sequence[int] = ()) -> List[IPRecord]:
    """Read the IP tap (pcap, pcapng — Ethernet or Linux cooked — or tshark CSV)."""
    path = Path(path)
    if path.suffix.lower() == ".csv":
        recs = read_tshark_csv(path, gateway_ip)
    else:
        recs = read_pcapng_hub(path, gateway_ip)
    return normalise_server_ports(recs, server_ports) if server_ports else recs
