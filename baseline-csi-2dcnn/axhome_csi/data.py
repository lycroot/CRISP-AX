from __future__ import annotations

import csv
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import numpy as np


@dataclass(frozen=True)
class SampleRecord:
    sample_id: str
    session_id: str
    archive_session_id: str
    person_id: str
    environment_id: str
    action_id: str
    trial_id: str
    csi_path: Path
    csi_packets_written: int
    status: str
    sync_quality_flags: str


REQUIRED_MANIFEST_FIELDS = {
    "sample_id",
    "session_id",
    "archive_session_id",
    "person_id",
    "environment_id",
    "action_id",
    "trial_id",
    "archive_csi_path",
    "csi_packets_written",
    "status",
    "sync_quality_flags",
}


def _safe_release_path(dataset_root: Path, relative_path: str) -> Path:
    root = dataset_root.resolve()
    candidate = (root / relative_path).resolve()
    if not candidate.is_relative_to(root):
        raise ValueError(f"release-relative path escapes dataset root: {relative_path}")
    return candidate


def load_manifest(
    dataset_root: str | Path,
    *,
    require_ok: bool = True,
    validate_paths: bool = False,
) -> list[SampleRecord]:
    """Load the canonical AXHome-MM-v1 archive index."""
    root = Path(dataset_root).resolve()
    manifest_path = root / "manifests" / "archive_index.csv"
    if not manifest_path.is_file():
        raise FileNotFoundError(f"manifest not found: {manifest_path}")
    output: list[SampleRecord] = []
    with manifest_path.open("r", newline="", encoding="utf-8-sig") as handle:
        reader = csv.DictReader(handle)
        missing = REQUIRED_MANIFEST_FIELDS - set(reader.fieldnames or [])
        if missing:
            raise ValueError(
                "archive_index.csv is missing required fields: "
                + ", ".join(sorted(missing))
            )
        for row_number, row in enumerate(reader, start=2):
            if require_ok and (
                row["status"] != "ok" or row["sync_quality_flags"] != "ok"
            ):
                continue
            csi_path = _safe_release_path(root, row["archive_csi_path"])
            if validate_paths and not csi_path.is_file():
                raise FileNotFoundError(
                    f"row {row_number}, sample {row['sample_id']}: "
                    f"CSI file not found: {csi_path}"
                )
            try:
                packet_count = int(row["csi_packets_written"])
            except ValueError as exc:
                raise ValueError(
                    f"row {row_number}, sample {row['sample_id']}: "
                    "csi_packets_written is not an integer"
                ) from exc
            output.append(
                SampleRecord(
                    sample_id=row["sample_id"],
                    session_id=row["session_id"],
                    archive_session_id=row["archive_session_id"],
                    person_id=row["person_id"],
                    environment_id=row["environment_id"],
                    action_id=row["action_id"],
                    trial_id=row["trial_id"],
                    csi_path=csi_path,
                    csi_packets_written=packet_count,
                    status=row["status"],
                    sync_quality_flags=row["sync_quality_flags"],
                )
            )
    if not output:
        raise ValueError(f"manifest contains no eligible samples: {manifest_path}")
    return output


def _even_indices(length: int, target: int) -> np.ndarray:
    if length <= 0 or target <= 0:
        raise ValueError("length and target must be positive")
    return np.rint(np.linspace(0, length - 1, target)).astype(np.int64)


def interpolate_low_energy_subcarriers(
    amplitude: np.ndarray,
    *,
    low_energy_ratio: float = 0.05,
    epsilon: float = 1e-6,
) -> tuple[np.ndarray, np.ndarray]:
    """Repair structural low-energy carriers while preserving frequency indices."""
    if amplitude.ndim != 3:
        raise ValueError("amplitude must have shape (packets, rx, subcarriers)")
    if not 0 <= low_energy_ratio < 1:
        raise ValueError("low_energy_ratio must be in [0, 1)")
    carrier_energy = np.median(amplitude, axis=(0, 1))
    nonzero = carrier_energy > epsilon
    if not np.any(nonzero):
        raise ValueError("all CSI subcarriers have zero energy")
    reference_energy = float(np.median(carrier_energy[nonzero]))
    valid = nonzero & (carrier_energy >= reference_energy * low_energy_ratio)
    if np.count_nonzero(valid) < 2:
        valid = nonzero
    valid_indices = np.flatnonzero(valid)
    if valid_indices.size < 2:
        raise ValueError("fewer than two usable subcarriers are available")

    positions = np.arange(amplitude.shape[-1])
    right_positions = np.searchsorted(valid_indices, positions, side="left")
    right_positions = np.clip(right_positions, 0, valid_indices.size - 1)
    left_positions = np.clip(right_positions - 1, 0, valid_indices.size - 1)
    left_indices = valid_indices[left_positions]
    right_indices = valid_indices[right_positions]
    spans = right_indices - left_indices
    weights = np.divide(
        positions - left_indices,
        spans,
        out=np.zeros_like(positions, dtype=np.float32),
        where=spans > 0,
    )
    repaired = (
        amplitude[..., left_indices] * (1.0 - weights)
        + amplitude[..., right_indices] * weights
    ).astype(np.float32, copy=False)
    repaired[..., valid] = amplitude[..., valid]
    return repaired, valid


def preprocess_amplitude(
    csi: np.ndarray,
    *,
    target_packets: int = 256,
    target_subcarriers: int = 128,
    low_energy_ratio: float = 0.05,
    epsilon: float = 1e-6,
) -> np.ndarray:
    """Convert complex CSI into a normalized ``(rx, time, frequency)`` map."""
    if csi.ndim != 4:
        raise ValueError(
            "CSI must have shape (packets, rx, tx, subcarriers), "
            f"got {csi.shape}"
        )
    packets, num_rx, num_tx, subcarriers = csi.shape
    if packets <= 0 or num_rx <= 0 or subcarriers <= 0:
        raise ValueError(f"CSI dimensions must be non-zero, got {csi.shape}")
    if num_tx != 1:
        raise ValueError(f"this baseline expects one TX stream, got {num_tx}")
    amplitude = np.log1p(np.abs(csi[:, :, 0, :])).astype(np.float32)
    amplitude, _ = interpolate_low_energy_subcarriers(
        amplitude,
        low_energy_ratio=low_energy_ratio,
        epsilon=epsilon,
    )
    selected_carriers = _even_indices(subcarriers, target_subcarriers)
    selected_packets = _even_indices(packets, target_packets)
    output = amplitude[selected_packets][:, :, selected_carriers]
    output = np.transpose(output, (1, 0, 2)).astype(np.float32, copy=False)

    working = output.astype(np.float64)
    mean = working.mean(axis=(1, 2), keepdims=True)
    std = working.std(axis=(1, 2), keepdims=True)
    output = (working - mean) / np.maximum(std, epsilon)
    if not np.isfinite(output).all():
        raise ValueError("preprocessing produced non-finite values")
    return output.astype(np.float32, copy=False)


def action_counts(samples: Iterable[SampleRecord]) -> dict[str, int]:
    counts: dict[str, int] = {}
    for sample in samples:
        counts[sample.action_id] = counts.get(sample.action_id, 0) + 1
    return dict(sorted(counts.items()))
