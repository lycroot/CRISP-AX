from __future__ import annotations

import struct
from dataclasses import dataclass
from pathlib import Path
from typing import BinaryIO

import numpy as np

HEADER_SIZE = 272


class FeitCSIFormatError(ValueError):
    """Raised when a FeitCSI binary record is incomplete or inconsistent."""


@dataclass(frozen=True)
class FeitCSIHeader:
    csi_size: int
    ftm_clock: int
    timestamp: int
    num_rx: int
    num_tx: int
    num_subcarriers: int
    rssi1: int
    rssi2: int
    source_mac: str
    rate_flags: int


def _parse_header(raw: bytes, packet_index: int) -> FeitCSIHeader:
    if len(raw) != HEADER_SIZE:
        raise FeitCSIFormatError(
            f"packet {packet_index}: truncated header; "
            f"expected {HEADER_SIZE} bytes, got {len(raw)}"
        )
    csi_size = struct.unpack_from("<I", raw, 0)[0]
    num_rx = raw[46]
    num_tx = raw[47]
    num_subcarriers = struct.unpack_from("<I", raw, 52)[0]
    if num_rx <= 0 or num_tx <= 0 or num_subcarriers <= 0:
        raise FeitCSIFormatError(
            f"packet {packet_index}: invalid CSI dimensions "
            f"rx={num_rx}, tx={num_tx}, subcarriers={num_subcarriers}"
        )
    expected_size = 4 * num_rx * num_tx * num_subcarriers
    if csi_size != expected_size:
        raise FeitCSIFormatError(
            f"packet {packet_index}: payload size {csi_size} does not match "
            f"4*{num_rx}*{num_tx}*{num_subcarriers}={expected_size}"
        )
    mac = ":".join(f"{value:02x}" for value in raw[68:74])
    return FeitCSIHeader(
        csi_size=csi_size,
        ftm_clock=struct.unpack_from("<I", raw, 8)[0],
        timestamp=struct.unpack_from("<Q", raw, 12)[0],
        num_rx=num_rx,
        num_tx=num_tx,
        num_subcarriers=num_subcarriers,
        rssi1=struct.unpack_from("<I", raw, 60)[0],
        rssi2=struct.unpack_from("<I", raw, 64)[0],
        source_mac=mac,
        rate_flags=struct.unpack_from("<I", raw, 92)[0],
    )


def _read_payload(
    handle: BinaryIO, header: FeitCSIHeader, packet_index: int
) -> np.ndarray:
    raw = handle.read(header.csi_size)
    if len(raw) != header.csi_size:
        raise FeitCSIFormatError(
            f"packet {packet_index}: truncated CSI payload; "
            f"expected {header.csi_size} bytes, got {len(raw)}"
        )
    components = np.frombuffer(raw, dtype="<i2").reshape(
        header.num_rx, header.num_tx, header.num_subcarriers, 2
    )
    real = components[..., 0].astype(np.float32)
    imag = components[..., 1].astype(np.float32)
    return (real + 1j * imag).astype(np.complex64, copy=False)


def read_feitcsi_stream(
    handle: BinaryIO,
    *,
    source: str = "<binary stream>",
    max_packets: int | None = None,
) -> tuple[np.ndarray, list[FeitCSIHeader]]:
    """Read FeitCSI records from an open binary stream without closing it."""
    if max_packets is not None and max_packets <= 0:
        raise ValueError("max_packets must be positive when provided")
    packets: list[np.ndarray] = []
    headers: list[FeitCSIHeader] = []
    reference_shape: tuple[int, ...] | None = None
    packet_index = 0
    while max_packets is None or packet_index < max_packets:
        raw_header = handle.read(HEADER_SIZE)
        if not raw_header:
            break
        header = _parse_header(raw_header, packet_index)
        packet = _read_payload(handle, header, packet_index)
        if reference_shape is None:
            reference_shape = packet.shape
        elif packet.shape != reference_shape:
            raise FeitCSIFormatError(
                f"packet {packet_index}: CSI shape changed from "
                f"{reference_shape} to {packet.shape}"
            )
        headers.append(header)
        packets.append(packet)
        packet_index += 1
    if not packets:
        raise FeitCSIFormatError(f"no complete FeitCSI packets found in {source}")
    return np.stack(packets, axis=0), headers


def read_feitcsi(
    path: str | Path, *, max_packets: int | None = None
) -> tuple[np.ndarray, list[FeitCSIHeader]]:
    """Read a FeitCSI file as ``(packet, rx, tx, subcarrier)`` complex CSI.

    The implementation follows FeitCSI's documented little-endian 272-byte
    header and signed int16 real/imaginary payload layout.
    """
    path = Path(path)
    with path.open("rb") as handle:
        return read_feitcsi_stream(
            handle, source=str(path), max_packets=max_packets
        )
