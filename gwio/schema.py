"""Core data types shared by every stage of the pipeline.

Everything here is *envelope* metadata only: we never
carry decrypted payload, only addresses, types, lengths, sequence numbers,
timestamps and signal quality.
"""
from __future__ import annotations

import json
import uuid
from dataclasses import asdict, dataclass, field
from typing import Dict, List, Optional

# --------------------------------------------------------------------------- #
# Radio side (IEEE 802.15.4)
# --------------------------------------------------------------------------- #
FRAME_BEACON, FRAME_DATA, FRAME_ACK, FRAME_CMD = "beacon", "data", "ack", "cmd"
FRAME_TYPES = (FRAME_BEACON, FRAME_DATA, FRAME_ACK, FRAME_CMD)

# 802.15.4 turnaround: an ACK follows the frame it acknowledges within
# aTurnaroundTime (12 symbols = 192 us) + ACK duration (~352 us).  We use a
# generous bound because the sniffer's own timestamping jitters.
ACK_TURNAROUND_S = 2.0e-3


@dataclass
class RadioFrame:
    ts: float                       # seconds, witness clock
    frame_type: str                 # one of FRAME_TYPES
    length: int                     # PHY length in bytes
    seq: int                        # 8-bit MAC sequence number
    src: Optional[int] = None       # 16-bit short address (None for ACKs)
    dst: Optional[int] = None
    pan_id: Optional[int] = None
    rssi: float = float("nan")      # dBm as measured at the witness
    lqi: float = float("nan")       # 0..255
    channel: int = 0
    # Zigbee NWK header (cleartext): originating device / final destination
    nwk_src: Optional[int] = None
    nwk_dst: Optional[int] = None
    nwk_type: Optional[int] = None  # 0 data, 1 NWK command
    nwk_seq: Optional[int] = None   # per-originator NWK sequence number
    # simulator-only ground truth, never populated from real captures
    truth: Optional[str] = None     # e.g. "forged" when the hub emitted it

    def is_ack(self) -> bool:
        return self.frame_type == FRAME_ACK


# --------------------------------------------------------------------------- #
# IP side
# --------------------------------------------------------------------------- #
DIR_OUT, DIR_IN = "out", "in"
KIND_TLS_RECORD, KIND_PACKET = "tls_record", "packet"


@dataclass
class IPRecord:
    """One observable unit of gateway egress/ingress.

    For a long-lived TLS session this is one application-data record; for a
    short-lived connection or non-TLS traffic it is one packet.  Either way it
    has a length and an arrival time and nothing else we would need to decrypt.
    """
    ts: float
    direction: str                  # DIR_OUT (gateway -> endpoint) / DIR_IN
    length: int
    dst_ip: str
    dst_port: int
    src_ip: str = ""
    sni: Optional[str] = None
    flow_id: str = ""
    kind: str = KIND_TLS_RECORD
    # simulator-only ground truth
    truth_device: Optional[int] = None   # device whose event this really is
    truth: Optional[str] = None          # "invented", "leaked", "swapped", ...

    @property
    def endpoint(self) -> str:
        """Remote endpoint label: SNI when visible, else the remote host, plus port."""
        remote = self.dst_ip if self.direction == DIR_OUT else (self.src_ip or self.dst_ip)
        return f"{self.sni or remote}:{self.dst_port}"


# --------------------------------------------------------------------------- #
# Windows
# --------------------------------------------------------------------------- #
@dataclass
class Window:
    index: int
    t0: float
    t1: float
    frames: List[RadioFrame] = field(default_factory=list)
    records: List[IPRecord] = field(default_factory=list)

    @property
    def length(self) -> float:
        return self.t1 - self.t0


# --------------------------------------------------------------------------- #
# Alerts (fixed schema)
# --------------------------------------------------------------------------- #
INV_CONSERVATION = "conservation"
INV_PROVENANCE = "provenance"
INV_TRANSLATION = "translation"
INV_EMITTER = "emitter_consistency"
INV_RESIDUAL = "translation_residual"     # learned (R-GCN) translation check
INVARIANTS = (INV_CONSERVATION, INV_PROVENANCE, INV_TRANSLATION, INV_EMITTER)

VERDICT_ALERT, VERDICT_DEGRADED, VERDICT_OK = "alert", "degraded", "ok"


@dataclass
class Alert:
    alert_id: str
    verdict: str                        # alert | degraded
    window: Dict[str, float]            # {"t0", "t1", "length_s", "index"}
    device: Dict[str, Optional[str]]    # {"short_addr": "0x1a2b", "label": ...}
    invariant: str                      # which check failed
    direction: str                      # "high" | "low" | "drift" | "out_of_band" | ""
    observed: Dict[str, float]
    expected: Dict[str, float]          # learned band the observation fell outside
    persistence: Dict[str, int]         # {"agreed": k, "required": k_req, "of": n}
    completeness: float                 # measured capture completeness in the window
    evidence: Optional[str] = None      # path to retained captures
    interpretation: str = ""            # e.g. "suppression", "fabrication"
    created_at: float = 0.0

    @staticmethod
    def new_id() -> str:
        return uuid.uuid4().hex[:16]

    def to_json(self) -> str:
        return json.dumps(asdict(self), sort_keys=True)

    @classmethod
    def from_json(cls, s: str) -> "Alert":
        return cls(**json.loads(s))


def short_addr_hex(addr: Optional[int]) -> Optional[str]:
    return None if addr is None else f"0x{addr:04x}"
