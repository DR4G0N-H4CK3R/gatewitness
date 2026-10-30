"""Read the radio witness capture (nRF52840 + nRF Sniffer for 802.15.4).

Two input forms are supported, both metadata-only (P2: no decryption):

1. pcap written by Wireshark with the nRF extcap plugin.
   * DLT 283 (IEEE802_15_4_TAP): TLV header carrying RSSI (type 1), channel
     (type 3) and LQI (type 10) ahead of the MAC frame.  This is what current
     nRF Sniffer releases produce.
   * DLT 195 (IEEE802_15_4_WITHFCS) with the CC24xx-style two-byte trailer
     (RSSI, then CRC-OK|LQI) that older sniffer builds emit.
2. CSV exported with tshark, e.g.
     tshark -r cap.pcap -T fields -E separator=, -E header=y \
       -e frame.time_epoch -e wpan.frame_type -e frame.len -e wpan.seq_no \
       -e wpan.src16 -e wpan.dst16 -e wpan.dst_pan -e wpan-tap.rss -e wpan-tap.lqi \
       -e wpan-tap.ch_num
   which is the most robust path when the pcap link type is unexpected.

Timestamps are the pcap/pcapng capture timestamps (the verification host's
clock), which is also the clock of the IP tap.
"""
from __future__ import annotations

import struct
from pathlib import Path
from typing import Iterator, List, Optional, Tuple

import pandas as pd

from ..schema import (FRAME_ACK, FRAME_BEACON, FRAME_CMD, FRAME_DATA,
                      RadioFrame)

DLT_IEEE802_15_4_WITHFCS = 195
DLT_IEEE802_15_4_NOFCS = 230
DLT_IEEE802_15_4_TAP = 283
FRAME_TYPE_BY_CODE = {0: FRAME_BEACON, 1: FRAME_DATA, 2: FRAME_ACK, 3: FRAME_CMD}
_TAP_RSS, _TAP_CHANNEL, _TAP_LQI = 1, 3, 10


# --------------------------------------------------------------------------- #
# pcap / pcapng record iteration (tiny reader: no scapy needed for the hot path)
# --------------------------------------------------------------------------- #
def _iter_pcap(path: Path) -> Iterator[Tuple[float, int, bytes]]:
    """Yield (timestamp, linktype, packet bytes) from a classic pcap file."""
    with open(path, "rb") as fh:
        magic = fh.read(4)
        if magic == b"\xa1\xb2\xc3\xd4":
            endian, nano = ">", False
        elif magic == b"\xd4\xc3\xb2\xa1":
            endian, nano = "<", False
        elif magic == b"\xa1\xb2\x3c\x4d":
            endian, nano = ">", True
        elif magic == b"\x4d\x3c\xb2\xa1":
            endian, nano = "<", True
        elif magic == b"\x0a\x0d\x0d\x0a":
            raise ValueError("pcapng")          # handled by _iter_any
        else:
            raise ValueError(f"not a pcap file: {path}")
        _, _, _, _, _, linktype = struct.unpack(endian + "HHiIII", fh.read(20))
        while True:
            hdr = fh.read(16)
            if len(hdr) < 16:
                return
            ts_s, ts_frac, incl, _ = struct.unpack(endian + "IIII", hdr)
            data = fh.read(incl)
            ts = ts_s + ts_frac / (1e9 if nano else 1e6)
            yield ts, linktype, data


def _iter_any(path: Path) -> Iterator[Tuple[float, int, bytes]]:
    """pcap via the built-in reader; pcapng via scapy's raw reader (lazy import)."""
    with open(path, "rb") as fh:
        magic = fh.read(4)
    if magic == b"\x0a\x0d\x0d\x0a":
        from scapy.all import RawPcapNgReader
        with RawPcapNgReader(str(path)) as rd:
            for pkt, meta in rd:
                ts = (meta.tshigh * 2 ** 32 + meta.tslow) / meta.tsresol
                yield ts, meta.linktype, bytes(pkt)
    else:
        yield from _iter_pcap(path)


def _parse_tap(pkt: bytes) -> Tuple[bytes, float, float, int]:
    """802.15.4 TAP: version(1) reserved(1) length(2) then TLVs, then the frame."""
    if len(pkt) < 4:
        raise ValueError("short TAP header")
    hdr_len = struct.unpack("<H", pkt[2:4])[0]
    off, rssi, lqi, ch = 4, float("nan"), float("nan"), 0
    while off + 4 <= hdr_len:
        t, l = struct.unpack("<HH", pkt[off:off + 4])
        val = pkt[off + 4: off + 4 + l]
        if t == _TAP_RSS and l >= 4:
            rssi = struct.unpack("<f", val[:4])[0]
        elif t == _TAP_LQI and l >= 1:
            lqi = float(val[0])
        elif t == _TAP_CHANNEL and l >= 3:
            ch = struct.unpack("<H", val[:2])[0]
        off += 4 + ((l + 3) & ~3)          # TLVs are padded to 4 bytes
    return pkt[hdr_len:], rssi, lqi, ch


def _parse_mac(frame: bytes) -> Optional[dict]:
    """802.15.4 MAC header (type, seq, PAN, 16-bit addresses) plus, for data
    frames, the Zigbee NWK header, which is *cleartext* even on an encrypted
    network: NWK source and destination identify the originating device even
    when a router relayed the frame on its behalf.  Nothing past the NWK
    header (the security aux header and the encrypted payload) is read."""
    if len(frame) < 3:
        return None
    fcf = struct.unpack("<H", frame[:2])[0]
    ftype = fcf & 0x7
    seq = frame[2]
    dst_mode, src_mode = (fcf >> 10) & 0x3, (fcf >> 14) & 0x3
    pan_compress = bool(fcf & 0x40)
    off, pan, dst, src = 3, None, None, None
    if dst_mode:
        if len(frame) < off + 2:
            return None
        pan = struct.unpack("<H", frame[off:off + 2])[0]
        off += 2
        if dst_mode == 2:
            dst = struct.unpack("<H", frame[off:off + 2])[0]
            off += 2
        elif dst_mode == 3:
            off += 8
    if src_mode:
        if not pan_compress or not dst_mode:
            if len(frame) >= off + 2:
                pan = pan if pan is not None else struct.unpack("<H", frame[off:off + 2])[0]
            off += 2
        if src_mode == 2 and len(frame) >= off + 2:
            src = struct.unpack("<H", frame[off:off + 2])[0]
            off += 2
        elif src_mode == 3:
            off += 8
    if fcf & 0x08:                                        # MAC security: aux header, no NWK parse
        return {"frame_type": FRAME_TYPE_BY_CODE.get(ftype, FRAME_CMD), "seq": seq, "pan_id": pan,
                "src": src, "dst": dst, "nwk_src": None, "nwk_dst": None, "nwk_type": None,
                "nwk_seq": None}
    out = {"frame_type": FRAME_TYPE_BY_CODE.get(ftype, FRAME_CMD), "seq": seq, "pan_id": pan,
           "src": src, "dst": dst, "nwk_src": None, "nwk_dst": None, "nwk_type": None, "nwk_seq": None}
    if ftype == 1 and len(frame) >= off + 8:            # data frame -> NWK header
        nfcf = struct.unpack("<H", frame[off:off + 2])[0]
        nwk_type = nfcf & 0x3                             # 0 data, 1 NWK command
        ndst, nsrc = struct.unpack("<HH", frame[off + 2:off + 6])
        nseq = frame[off + 7]
        # inter-PAN / green power frames have protocol version != 2; skip those
        if (nfcf >> 2) & 0xF == 2:
            out.update(nwk_src=nsrc, nwk_dst=ndst, nwk_type=nwk_type, nwk_seq=nseq)
    return out


def _crc16_802154(data: bytes) -> int:
    crc = 0
    for byte in data:
        crc ^= byte
        for _ in range(8):
            crc = (crc >> 1) ^ 0x8408 if crc & 1 else crc >> 1
    return crc


def _detect_trailer(path: Path, n: int = 300) -> str:
    """DLT 195 files end each frame either with the real FCS (whsniff, most
    sniffers) or with the TI CC24xx metadata pair RSSI | CRC_OK+LQI (Wireshark's
    CC2531 plugin, ZigBeeNet).  Decide by checking whether the trailer is a
    valid CRC over the frame for a sample of frames."""
    ok = tot = 0
    for _, lt, pkt in _iter_any(path):
        if lt != DLT_IEEE802_15_4_WITHFCS:
            return "n/a"
        if len(pkt) >= 5:
            tot += 1
            ok += _crc16_802154(pkt[:-2]) == struct.unpack("<H", pkt[-2:])[0]
        if tot >= n:
            break
    return "fcs" if tot and ok / tot > 0.5 else "cc24xx"


def read_pcap(path: Path, rssi_offset: float = -73.0, trailer: Optional[str] = None,
              stats: Optional[dict] = None) -> List[RadioFrame]:
    """Read a pcap/pcapng from an 802.15.4 sniffer.

    `rssi_offset` is the CC24xx trailer's RSSI reference (the CC2531 reports
    RSSI + 73 dBm).  `trailer` forces "fcs" or "cc24xx" for DLT 195 files;
    by default it is detected.  With a real FCS there is no RSSI/LQI in the
    capture and those fields stay NaN; frames failing the CRC are dropped
    (counted in stats["crc_fail"])."""
    out: List[RadioFrame] = []
    path = Path(path)
    if trailer is None:
        trailer = _detect_trailer(path)
    if stats is not None:
        stats.update(trailer=trailer, crc_fail=0, total=0)
    for ts, lt, pkt in _iter_any(path):
        if stats is not None:
            stats["total"] += 1
        rssi = lqi = float("nan")
        ch = 0
        if lt == DLT_IEEE802_15_4_TAP:
            frame, rssi, lqi, ch = _parse_tap(pkt)
            if len(frame) >= 2:
                frame = frame[:-2]                          # FCS
        elif lt == DLT_IEEE802_15_4_WITHFCS:
            if len(pkt) < 5:
                continue
            if trailer == "cc24xx":
                # TI CC24xx metadata in place of the FCS: RSSI (int8), CRC_OK|LQI
                rssi = float(struct.unpack("b", pkt[-2:-1])[0]) + rssi_offset
                lqi = float(pkt[-1] & 0x7F)
                crc_ok = bool(pkt[-1] & 0x80)
            else:
                crc_ok = _crc16_802154(pkt[:-2]) == struct.unpack("<H", pkt[-2:])[0]
            frame = pkt[:-2]
            if not crc_ok:
                if stats is not None:
                    stats["crc_fail"] += 1
                continue                                    # corrupt frame: not evidence
        elif lt == DLT_IEEE802_15_4_NOFCS:
            frame = pkt
        else:
            raise ValueError(f"unsupported link type {lt}; export a tshark CSV instead")
        mac = _parse_mac(frame)
        if mac is None:
            continue
        out.append(RadioFrame(ts=ts, length=len(frame) + 2, rssi=rssi, lqi=lqi, channel=ch, **mac))
    out.sort(key=lambda f: f.ts)
    return out


def read_tshark_csv(path: Path) -> List[RadioFrame]:
    """CSV from the tshark command in the module docstring (names may be prefixed)."""
    df = pd.read_csv(path)
    col = lambda *names: next((c for c in df.columns for n in names if c.endswith(n)), None)
    c_ts, c_type, c_len, c_seq = col("frame.time_epoch"), col("wpan.frame_type"), col("frame.len"), col("wpan.seq_no")
    c_src, c_dst, c_pan = col("wpan.src16"), col("wpan.dst16"), col("wpan.dst_pan")
    c_rss, c_lqi, c_ch = col("wpan-tap.rss", "rssi"), col("wpan-tap.lqi", "lqi"), col("wpan-tap.ch_num", "channel")
    h = lambda v: None if pd.isna(v) else int(str(v), 16) if isinstance(v, str) and str(v).startswith("0x") else int(v)
    out = []
    for r in df.itertuples(index=False):
        g = lambda c: getattr(r, c.replace(".", "_").replace("-", "_")) if c else float("nan")
        ftype = g(c_type)
        ftype = FRAME_TYPE_BY_CODE.get(int(ftype), FRAME_CMD) if not pd.isna(ftype) and str(ftype).isdigit() \
            else str(ftype).lower()
        out.append(RadioFrame(ts=float(g(c_ts)), frame_type=ftype, length=int(g(c_len)),
                              seq=int(g(c_seq)), src=h(g(c_src)) if c_src else None,
                              dst=h(g(c_dst)) if c_dst else None, pan_id=h(g(c_pan)) if c_pan else None,
                              rssi=float(g(c_rss)) if c_rss else float("nan"),
                              lqi=float(g(c_lqi)) if c_lqi else float("nan"),
                              channel=int(g(c_ch)) if c_ch and not pd.isna(g(c_ch)) else 0))
    out.sort(key=lambda f: f.ts)
    return out


def read_radio(path: Path, **kw) -> List[RadioFrame]:
    path = Path(path)
    return read_tshark_csv(path) if path.suffix.lower() == ".csv" else read_pcap(path, **kw)
