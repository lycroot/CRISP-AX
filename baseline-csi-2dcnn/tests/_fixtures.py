from __future__ import annotations

import csv
import json
import struct
from pathlib import Path

import numpy as np


def write_feitcsi(
    path: Path,
    *,
    packets: int = 4,
    num_rx: int = 2,
    num_tx: int = 1,
    subcarriers: int = 8,
) -> np.ndarray:
    """Write a deterministic, valid FeitCSI fixture and return expected CSI."""
    expected = np.empty(
        (packets, num_rx, num_tx, subcarriers), dtype=np.complex64
    )
    with path.open("wb") as handle:
        for packet_idx in range(packets):
            header = bytearray(272)
            csi_size = 4 * num_rx * num_tx * subcarriers
            struct.pack_into("<I", header, 0, csi_size)
            struct.pack_into("<I", header, 8, 1000 + packet_idx)
            struct.pack_into("<Q", header, 12, 0)
            header[46] = num_rx
            header[47] = num_tx
            struct.pack_into("<I", header, 52, subcarriers)
            struct.pack_into("<I", header, 60, 70)
            struct.pack_into("<I", header, 64, 71)
            struct.pack_into("<I", header, 92, 2 << 11)
            handle.write(header)

            for rx_idx in range(num_rx):
                for tx_idx in range(num_tx):
                    for sc_idx in range(subcarriers):
                        real = packet_idx * 100 + rx_idx * 10 + sc_idx + 1
                        imag = -(sc_idx + 1)
                        handle.write(struct.pack("<hh", real, imag))
                        expected[packet_idx, rx_idx, tx_idx, sc_idx] = complex(
                            real, imag
                        )
    return expected


def write_mini_release(
    root: Path, *, trials_per_action: int = 1
) -> list[dict[str, str]]:
    if trials_per_action <= 0:
        raise ValueError("trials_per_action must be positive")
    manifest_dir = root / "manifests"
    manifest_dir.mkdir(parents=True)
    rows: list[dict[str, str]] = []
    people = ["P01", "P02", "P03"]
    actions = ["walk", "sit_down"]
    for person_idx, person_id in enumerate(people, start=1):
        session_id = f"S{person_idx:02d}"
        for action_idx, action_id in enumerate(actions, start=1):
            for trial in range(1, trials_per_action + 1):
                trial_id = f"R{trial:03d}"
                sample_id = (
                    f"{session_id}_{person_id}_{action_id}_{trial_id}"
                )
                rel_path = (
                    Path("data")
                    / session_id
                    / "csi"
                    / action_id
                    / f"{sample_id}.dat"
                )
                csi_path = root / rel_path
                csi_path.parent.mkdir(parents=True, exist_ok=True)
                write_feitcsi(csi_path, packets=12, subcarriers=16)
                rows.append(
                    {
                        "sample_id": sample_id,
                        "session_id": session_id,
                        "archive_session_id": session_id,
                        "person_id": person_id,
                        "environment_id": "E1",
                        "action_id": action_id,
                        "trial_id": trial_id,
                        "archive_csi_path": rel_path.as_posix(),
                        "csi_packets_written": "12",
                        "status": "ok",
                        "sync_quality_flags": "ok",
                    }
                )

    fieldnames = list(rows[0])
    with (manifest_dir / "archive_index.csv").open(
        "w", newline="", encoding="utf-8"
    ) as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    (root / "DATASET_VERSION.txt").write_text("CRISP-AX\n", encoding="utf-8")
    (root / "fixture.json").write_text(
        json.dumps({"samples": len(rows)}), encoding="utf-8"
    )
    return rows
