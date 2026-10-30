"""ZIOTP2025 adapter (Boiano et al., 2026 — github.com/antonio-boiano/ZIOTP2025).

Real 802.15.4 captures of 21 commercial Zigbee devices behind a Home Assistant
hub (Raspberry Pi 4, ZHA, Sonoff EFR32MG21 coordinator), sniffed by a CC2531
running whsniff, in two network topologies.  The `Extra/IP-Traffic` folder
holds tcpdumps taken during the command-interaction runs; those overlap the
`Web_Interaction` radio captures in wall-clock time, which gives twelve real
*paired* radio + IP captures of the same hub — exactly the benign paired
baseline the evaluation relies on.

What the dataset does and does not give us:
  * radio: real timing, real MAC/NWK headers (cleartext), real sequence
    numbers, real mesh relaying, real CRC failures.  **No RSSI/LQI** — whsniff
    keeps the true FCS, so the emitter-consistency invariant cannot be
    evaluated on this data (ZigBeeNet, captured with the Wireshark CC2531
    plugin, does carry RSSI/LQI and loads through the same reader).
  * IP: the hub's live state-push channel (Home Assistant websocket on
    tcp/8123 to the web or Android client) and, in some runs, the hub's own
    cloud egress on 443.  The tcpdump ran on the laptop, so it also contains
    laptop traffic that does not involve the hub; only flows with the hub as
    an endpoint are kept.
  * ground truth: per-packet device labels (from keys recovered at pairing,
    used by the dataset authors only for labelling) and a "human command"
    flag on the frames that carry an app-issued command.

Hub IP is 192.168.8.116 in both topologies.
"""
from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import pandas as pd

from gwio.capture.ip import read_pcapng_hub
from gwio.capture.radio import read_pcap
from gwio.schema import IPRecord, RadioFrame

HUB_IP = "192.168.8.116"
COORDINATOR = 0x0000

# short address -> (label, class); from 1_Ground-truth-generation.py in the dataset
DEVICES_A: Dict[int, Tuple[str, str]] = {
    0x09ac: ("ledvance_z3_plug", "socket"), 0x5db6: ("aqara_door_1", "door"),
    0xe694: ("smart_socket", "socket"), 0x772c: ("aqara_door_2", "door"),
    0x61af: ("aqara_vibration", "vibration"), 0xe011: ("aqara_button", "button"),
    0x9989: ("aqara_motion", "motion"), 0x265e: ("sonoff_temperature", "temperature"),
    0x7150: ("sonoff_temperature", "temperature"), 0xb815: ("sonoff_motion_1", "motion"),
    0x4e52: ("sonoff_motion_2", "motion"), 0x3181: ("sonoff_door_1", "door"),
    0x0a79: ("sonoff_door_2", "door"), 0x82eb: ("ledvance_smart_plug", "socket"),
    0x6e5f: ("power_plug_1", "socket"), 0xc8f0: ("power_plug_2", "socket"),
    0xebe5: ("moes_bulb", "bulb"), 0x054f: ("ledvance_bulb", "bulb"),
    0x482d: ("philips_lamp_1", "bulb"), 0xa209: ("philips_lamp_2", "bulb"),
    0xe5c4: ("philips_lamp_3", "bulb"), 0x2cae: ("philips_motion", "motion"),
}
DEVICES_B: Dict[int, Tuple[str, str]] = {
    0x6a5e: ("sonoff_temperature", "temperature"), 0x4615: ("sonoff_temperature", "temperature"),
    0x946e: ("sonoff_door_1", "door"), 0x7b10: ("sonoff_door_2", "door"),
    0xd0bb: ("sonoff_motion_1", "motion"), 0x045b: ("sonoff_motion_2", "motion"),
    0x27d7: ("aqara_motion", "motion"), 0x907b: ("aqara_door_1", "door"),
    0xe01d: ("aqara_door_2", "door"), 0x187a: ("aqara_vibration", "vibration"),
    0xc02a: ("aqara_vibration", "vibration"), 0xc31c: ("aqara_button", "button"),
    0xe1a6: ("smart_socket", "socket"), 0x3d95: ("power_plug_1", "socket"),
    0xa706: ("power_plug_2", "socket"), 0x4e11: ("ledvance_z3_plug", "socket"),
    0x0112: ("ledvance_smart_plug", "socket"), 0xec7f: ("ledvance_bulb", "bulb"),
    0x1e15: ("moes_bulb", "bulb"), 0x1cd8: ("philips_lamp_1", "bulb"),
    0x5bb9: ("philips_lamp_2", "bulb"), 0x711c: ("philips_lamp_3", "bulb"),
    0x059b: ("philips_motion", "motion"), 0x48f1: ("philips_motion", "motion"),
}
DEVICES = {"A": DEVICES_A, "B": DEVICES_B}


def labels(topology: str) -> Dict[int, str]:
    return {a: v[0] for a, v in DEVICES[topology].items()}


def classes(topology: str) -> Dict[int, str]:
    return {a: v[1] for a, v in DEVICES[topology].items()}


@dataclass
class Capture:
    name: str
    topology: str
    phase: str
    radio: Path
    ip: Optional[Path] = None
    groundtruth: Optional[Path] = None
    t0: float = 0.0
    t1: float = 0.0


def _span(path: Path) -> Tuple[float, float]:
    from scapy.all import RawPcapNgReader
    t0 = t1 = None
    with RawPcapNgReader(str(path)) as rd:
        for _, meta in rd:
            ts = (meta.tshigh * 2 ** 32 + meta.tslow) / meta.tsresol
            t0 = ts if t0 is None else t0
            t1 = ts
    return float(t0), float(t1)


def catalog(root: Path) -> List[Capture]:
    """Every radio capture, with its IP capture attached when one overlaps it in time."""
    root = Path(root) / "Data"
    caps: List[Capture] = []
    for phase in ("Idle", "Web_Interaction", "Physical_Interaction", "Scenario", "Power"):
        for topo in ("A", "B"):
            d = root / phase / f"Topology_{topo}"
            if not d.exists():
                continue
            for p in sorted(d.glob("*.pcapng")):
                gt = d / "1-Groundtruth" / f"{p.stem}_groundtruth.csv"
                t0, t1 = _span(p)
                caps.append(Capture(f"{topo}/{phase}/{p.stem}", topo, phase, p,
                                    groundtruth=gt if gt.exists() else None, t0=t0, t1=t1))
    ip_dir = root / "Extra" / "IP-Traffic"
    for topo in ("A", "B"):
        for p in sorted((ip_dir / f"Topology_{topo}").glob("*.pcapng")):
            it0, it1 = _span(p)
            best, overlap = None, 0.0
            for c in caps:
                if c.topology != topo or c.ip is not None:
                    continue
                ov = min(c.t1, it1) - max(c.t0, it0)
                if ov > overlap:
                    best, overlap = c, ov
            if best is not None and overlap > 60:
                best.ip = p
    return caps


def paired(root: Path, topology: Optional[str] = None) -> List[Capture]:
    return [c for c in catalog(root) if c.ip is not None and (topology is None or c.topology == topology)]


def _endpoint_map(rec: IPRecord) -> IPRecord:
    """Normalise endpoint labels so signatures transfer across runs and clients.

    The hub is a *server* on tcp/8123: every subscribed app (web client, the
    Android app, whatever ephemeral port) is one logical endpoint, the
    state-push channel.  Everything the hub initiates keeps host:port."""
    if rec.dst_port == 8123 or (rec.direction == "in" and rec.dst_port == 8123):
        rec.sni = "ha-subscribers"
        rec.dst_port = 8123
    return rec


def load(cap: Capture, hub_ip: str = HUB_IP, rebase: bool = True,
         stats: Optional[dict] = None) -> Tuple[List[RadioFrame], List[IPRecord], float, float]:
    """Return (frames, records, duration, t0).  Both streams share the laptop's
    clock (whsniff and tcpdump ran on the same machine) and are rebased to the
    radio capture's first frame."""
    frames = read_pcap(cap.radio, stats=stats)
    records = read_pcapng_hub(cap.ip, hub_ip) if cap.ip else []
    records = [_endpoint_map(r) for r in records]
    t0 = frames[0].ts if frames else (records[0].ts if records else 0.0)
    if rebase:
        for f in frames:
            f.ts -= t0
        for r in records:
            r.ts -= t0
    dur = max([f.ts for f in frames] + [r.ts for r in records], default=0.0) + 1.0
    return frames, records, dur, t0


def concat(root: Path, caps: Sequence[Capture], gap: float = 30.0,
           hub_ip: str = HUB_IP) -> Tuple[List[RadioFrame], List[IPRecord], float, List[Tuple[str, float, float]]]:
    """Concatenate several captures into one continuous stream separated by
    `gap` seconds of silence, so a rolling-horizon verifier sees them as one
    session.  Returns the segment table (name, start, end) as well."""
    frames: List[RadioFrame] = []
    records: List[IPRecord] = []
    segs: List[Tuple[str, float, float]] = []
    offset = 0.0
    for c in caps:
        fr, rc, dur, _ = load(c, hub_ip)
        for f in fr:
            f.ts += offset
        for r in rc:
            r.ts += offset
        frames += fr
        records += rc
        segs.append((c.name, offset, offset + dur))
        offset += dur + gap
    frames.sort(key=lambda f: f.ts)
    records.sort(key=lambda r: r.ts)
    return frames, records, offset, segs


def human_commands(cap: Capture, t0: float) -> pd.DataFrame:
    """App-issued commands from the ground-truth CSV, as (ts, device, command).

    The CSV has no timestamps; packet numbers are matched back to the capture
    (1-based, including CRC-failed frames) to recover them."""
    if cap.groundtruth is None:
        return pd.DataFrame(columns=["ts", "device", "command"])
    gt = pd.read_csv(cap.groundtruth)
    from scapy.all import RawPcapNgReader
    ts_by_no = {}
    with RawPcapNgReader(str(cap.radio)) as rd:
        for i, (_, meta) in enumerate(rd, start=1):
            ts_by_no[i] = (meta.tshigh * 2 ** 32 + meta.tslow) / meta.tsresol - t0
    h = gt[gt["Human Command"] == 1]
    rows = []
    for r in h.itertuples(index=False):
        dst = r._asdict().get("_11") or getattr(r, "Destination_Address_ZigBee", None)
        rows.append({"ts": ts_by_no.get(int(r._0 if hasattr(r, "_0") else r[0]), float("nan")),
                     "device": dst, "command": getattr(r, "Packet_Type", "")})
    return pd.DataFrame(rows)
