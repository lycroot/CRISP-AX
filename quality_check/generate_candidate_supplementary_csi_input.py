from __future__ import annotations

import argparse
import csv
import datetime as dt
import hashlib
import io
import json
import os
import re
import subprocess
import sys
import tempfile
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path

import numpy as np


BASELINE_ROOT = Path(__file__).resolve().parents[1] / "baseline-csi-2dcnn"
if str(BASELINE_ROOT) not in sys.path:
    sys.path.insert(0, str(BASELINE_ROOT))

from axhome_csi.data import (  # noqa: E402
    _even_indices,
    interpolate_low_energy_subcarriers,
    preprocess_amplitude,
)
from axhome_csi.feitcsi import FeitCSIHeader, read_feitcsi_stream  # noqa: E402


MANIFEST_REQUIRED_FIELDS = {
    "sample_id",
    "session_id",
    "archive_session_id",
    "person_id",
    "environment_id",
    "action_id",
    "trial_id",
    "archive_csi_path",
    "archive_metadata_path",
    "csi_size_bytes",
    "csi_packets_written",
    "status",
    "sync_quality_flags",
}

SOURCE_DATA_FIELDS = [
    "sample_id",
    "rx_index",
    "output_packet_index",
    "source_packet_index",
    "output_subcarrier_index",
    "source_subcarrier_index",
    "raw_amplitude",
    "log_repaired_amplitude",
    "processed_zscore",
]

FROZEN_SAMPLE_ID = "S14_P05_walk_face_rx_E1_clean_none_R006"
FROZEN_MANIFEST_SHA256 = (
    "7d7ea605819c2c988b068b3f86ed77419c5a801c0d49b279b74c671cfb654a00"
)
FROZEN_CSI_SHA256 = (
    "a301478f6be1e6ecb340fb6e7ac878851f58208e59e40ff7c6f09a597bbc2606"
)
FROZEN_CSI_SHAPE = (496, 2, 1, 996)
FROZEN_TARGET_PACKETS = 256
FROZEN_TARGET_SUBCARRIERS = 128
FROZEN_LOW_ENERGY_RATIO = 0.05
FROZEN_EPSILON = 1e-6

PNG_NAME = "candidate_supplementary_csi_input.png"
PDF_NAME = "candidate_supplementary_csi_input.pdf"
SOURCE_DATA_NAME = "candidate_supplementary_csi_input_source_data.csv"
DIAGNOSTICS_NAME = "candidate_supplementary_csi_input_diagnostics.json"
CAPTION_NAME = "candidate_supplementary_csi_input_caption.md"
GENERATION_RECORD_NAME = "candidate_supplementary_csi_input_generation_record.json"
STATUS_UPDATE_NAME = "candidate_status_update.md"
DETERMINISTIC_OUTPUT_NAMES = (
    PNG_NAME,
    PDF_NAME,
    SOURCE_DATA_NAME,
    DIAGNOSTICS_NAME,
    CAPTION_NAME,
    STATUS_UPDATE_NAME,
)
PACKAGE_OUTPUT_NAMES = (*DETERMINISTIC_OUTPUT_NAMES, GENERATION_RECORD_NAME)


@dataclass(frozen=True)
class ValidatedSample:
    sample_id: str
    row: dict[str, str]
    manifest_path: Path
    csi_path: Path
    metadata_path: Path
    manifest_sha256: str
    csi_sha256: str
    metadata_sha256: str
    manifest_size_bytes: int
    csi_size_bytes: int
    metadata_size_bytes: int
    csi: np.ndarray
    headers: list[FeitCSIHeader]


@dataclass(frozen=True)
class StageData:
    raw_amplitude: np.ndarray
    log_repaired_amplitude: np.ndarray
    processed_zscore: np.ndarray
    packet_indices: np.ndarray
    subcarrier_indices: np.ndarray
    valid_subcarrier_mask: np.ndarray
    low_energy_indices: np.ndarray
    reference_carrier_energy: float
    zscore_mean: np.ndarray
    zscore_std: np.ndarray


@dataclass(frozen=True)
class SourceData:
    raw_amplitude: np.ndarray
    log_repaired_amplitude: np.ndarray
    processed_zscore: np.ndarray
    packet_indices: np.ndarray
    subcarrier_indices: np.ndarray
    sample_id: str


def sha256_file(path: str | Path, chunk_size: int = 1024 * 1024) -> str:
    if chunk_size <= 0:
        raise ValueError("chunk_size must be positive")
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        while chunk := handle.read(chunk_size):
            digest.update(chunk)
    return digest.hexdigest()


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _safe_release_path(root: Path, relative: str) -> Path:
    resolved_root = root.resolve()
    relative_path = Path(relative)
    if relative_path.is_absolute() or relative_path.drive or relative_path.root:
        raise ValueError(f"release path must be relative: {relative}")
    resolved_path = (resolved_root / relative_path).resolve()
    if not resolved_path.is_relative_to(resolved_root):
        raise ValueError(f"release path escapes dataset root: {relative}")
    return resolved_path


def _manifest_integer(row: dict[str, str], field: str) -> int:
    try:
        value = int(row[field])
    except (KeyError, TypeError, ValueError) as error:
        raise ValueError(f"manifest field {field} must be an integer") from error
    if value < 0:
        raise ValueError(f"manifest field {field} must be non-negative")
    return value


def validate_frozen_sample(
    dataset_root: str | Path,
    *,
    sample_id: str,
    expected_manifest_sha256: str,
    expected_csi_sha256: str,
    expected_shape: tuple[int, ...],
) -> ValidatedSample:
    root = Path(dataset_root).resolve()
    manifest_path = root / "manifests" / "archive_index.csv"
    if not manifest_path.is_file():
        raise ValueError(f"manifest does not exist: {manifest_path}")

    manifest_bytes = manifest_path.read_bytes()
    manifest_sha256 = sha256_bytes(manifest_bytes)
    if manifest_sha256 != expected_manifest_sha256:
        raise ValueError(
            "manifest SHA-256 mismatch: "
            f"expected {expected_manifest_sha256}, got {manifest_sha256}"
        )

    try:
        manifest_text = manifest_bytes.decode("utf-8-sig")
    except UnicodeDecodeError as error:
        raise ValueError("manifest is not valid UTF-8") from error
    reader = csv.DictReader(io.StringIO(manifest_text, newline=""))
    fieldnames = reader.fieldnames or []
    missing = sorted(MANIFEST_REQUIRED_FIELDS.difference(fieldnames))
    if missing:
        raise ValueError(
            "archive_index.csv is missing required fields: " + ", ".join(missing)
        )
    matches = [row for row in reader if row["sample_id"] == sample_id]
    if len(matches) != 1:
        raise ValueError(
            f"sample_id must appear exactly once in manifest: {sample_id}"
        )
    row = matches[0]
    if row["status"] != "ok" or row["sync_quality_flags"] != "ok":
        raise ValueError(
            f"sample {sample_id} is not frozen with ok status and sync quality"
        )

    csi_path = _safe_release_path(root, row["archive_csi_path"])
    metadata_path = _safe_release_path(root, row["archive_metadata_path"])
    if not csi_path.is_file():
        raise ValueError(f"CSI file does not exist: {csi_path}")
    if not metadata_path.is_file():
        raise ValueError(f"metadata file does not exist: {metadata_path}")

    csi_bytes = csi_path.read_bytes()
    expected_size = _manifest_integer(row, "csi_size_bytes")
    actual_size = len(csi_bytes)
    if actual_size != expected_size:
        raise ValueError(
            f"CSI size mismatch: manifest {expected_size}, actual {actual_size}"
        )
    csi_sha256 = sha256_bytes(csi_bytes)
    if csi_sha256 != expected_csi_sha256:
        raise ValueError(
            "CSI SHA-256 mismatch: "
            f"expected {expected_csi_sha256}, got {csi_sha256}"
        )

    manifest_packet_count = _manifest_integer(row, "csi_packets_written")
    metadata_bytes = metadata_path.read_bytes()
    metadata_sha256 = sha256_bytes(metadata_bytes)
    try:
        metadata = json.loads(metadata_bytes.decode("utf-8-sig"))
        source_sample_id = metadata["source_sync_row"]["sample_id"]
        slice_result = metadata["slice_result"]
        slice_sample_id = slice_result["sample_id"]
        metadata_packet_count = slice_result["csi_packets_written"]
    except (KeyError, TypeError, ValueError, json.JSONDecodeError) as error:
        raise ValueError("metadata is missing required sample provenance") from error
    if source_sample_id != sample_id or slice_sample_id != sample_id:
        raise ValueError(
            f"metadata sample_id mismatch for manifest sample {sample_id}"
        )
    if (
        not isinstance(metadata_packet_count, int)
        or isinstance(metadata_packet_count, bool)
        or metadata_packet_count < 0
    ):
        raise ValueError("metadata packet count must be a non-negative integer")
    if metadata_packet_count != manifest_packet_count:
        raise ValueError(
            "metadata packet count mismatch: "
            f"manifest {manifest_packet_count}, metadata {metadata_packet_count}"
        )
    csi, headers = read_feitcsi_stream(
        io.BytesIO(csi_bytes), source=str(csi_path)
    )
    if len(headers) != manifest_packet_count:
        raise ValueError(
            "packet count mismatch: "
            f"manifest {manifest_packet_count}, decoded {len(headers)}"
        )
    if csi.shape != tuple(expected_shape):
        raise ValueError(
            "decoded CSI shape mismatch: "
            f"expected {tuple(expected_shape)}, got {csi.shape}"
        )

    return ValidatedSample(
        sample_id=sample_id,
        row=row,
        manifest_path=manifest_path,
        csi_path=csi_path,
        metadata_path=metadata_path,
        manifest_sha256=manifest_sha256,
        csi_sha256=csi_sha256,
        metadata_sha256=metadata_sha256,
        manifest_size_bytes=len(manifest_bytes),
        csi_size_bytes=len(csi_bytes),
        metadata_size_bytes=len(metadata_bytes),
        csi=csi,
        headers=headers,
    )


def extract_stages(
    csi: np.ndarray,
    *,
    target_packets: int,
    target_subcarriers: int,
    low_energy_ratio: float,
    epsilon: float,
) -> StageData:
    if csi.ndim != 4:
        raise ValueError(
            "CSI must have shape (packets, rx, tx, subcarriers), "
            f"got {csi.shape}"
        )

    packets, num_rx, num_tx, subcarriers = csi.shape
    if packets <= 0 or num_rx <= 0 or subcarriers <= 0:
        raise ValueError(f"CSI dimensions must be non-zero, got {csi.shape}")
    if num_tx != 1:
        raise ValueError(f"CSI must have one TX stream, got shape {csi.shape}")

    magnitude_full = np.abs(csi[:, :, 0, :])
    raw_full = magnitude_full.astype(np.float32)
    logged_full = np.log1p(magnitude_full).astype(np.float32)
    repaired_full, valid = interpolate_low_energy_subcarriers(
        logged_full,
        low_energy_ratio=low_energy_ratio,
        epsilon=epsilon,
    )

    packet_indices = _even_indices(packets, target_packets)
    subcarrier_indices = _even_indices(subcarriers, target_subcarriers)

    raw = raw_full[packet_indices][:, :, subcarrier_indices]
    raw = np.transpose(raw, (1, 0, 2)).astype(np.float32, copy=False)
    repaired = repaired_full[packet_indices][:, :, subcarrier_indices]
    repaired = np.transpose(repaired, (1, 0, 2)).astype(np.float32, copy=False)

    working = repaired.astype(np.float64)
    mean = working.mean(axis=(1, 2), keepdims=True)
    std = working.std(axis=(1, 2), keepdims=True)
    processed = (
        (working - mean) / np.maximum(std, epsilon)
    ).astype(np.float32)

    authoritative = preprocess_amplitude(
        csi,
        target_packets=target_packets,
        target_subcarriers=target_subcarriers,
        low_energy_ratio=low_energy_ratio,
        epsilon=epsilon,
    )
    if not np.array_equal(processed, authoritative):
        raise RuntimeError(
            "diagnostic stages differ from authoritative preprocessing"
        )

    carrier_energy = np.median(logged_full, axis=(0, 1))
    nonzero = carrier_energy > epsilon
    if not np.any(nonzero):
        raise ValueError("all CSI subcarriers have zero energy")
    reference_energy = float(np.median(carrier_energy[nonzero]))
    diagnostic_valid = nonzero & (
        carrier_energy >= reference_energy * low_energy_ratio
    )
    if np.count_nonzero(diagnostic_valid) < 2:
        diagnostic_valid = nonzero
    if not np.array_equal(diagnostic_valid, valid):
        raise RuntimeError(
            "diagnostic carrier mask differs from authoritative interpolation"
        )

    finite_arrays = (raw, repaired, processed, mean, std)
    if not all(np.isfinite(array).all() for array in finite_arrays):
        raise ValueError("stage extraction produced non-finite values")
    if not np.isfinite(reference_energy):
        raise ValueError("stage extraction produced non-finite values")

    return StageData(
        raw_amplitude=raw,
        log_repaired_amplitude=repaired,
        processed_zscore=processed,
        packet_indices=packet_indices.copy(),
        subcarrier_indices=subcarrier_indices.copy(),
        valid_subcarrier_mask=valid.copy(),
        low_energy_indices=np.flatnonzero(~valid).copy(),
        reference_carrier_energy=reference_energy,
        zscore_mean=np.squeeze(mean, axis=(1, 2)).copy(),
        zscore_std=np.squeeze(std, axis=(1, 2)).copy(),
    )


def _validated_indices(
    values: np.ndarray,
    *,
    name: str,
    expected_length: int,
) -> np.ndarray:
    array = np.asarray(values)
    if array.ndim != 1:
        raise ValueError(f"{name} must be one-dimensional")
    if len(array) != expected_length:
        raise ValueError(f"{name} length must be {expected_length}")
    if not np.issubdtype(array.dtype, np.integer) or np.issubdtype(
        array.dtype, np.bool_
    ):
        raise ValueError(f"{name} must have an integer dtype")
    if np.any(array < 0):
        raise ValueError(f"{name} must be non-negative")
    if len(array) > 1 and np.any(array[1:] <= array[:-1]):
        raise ValueError(f"{name} must be strictly increasing")
    if any(int(value) > np.iinfo(np.int64).max for value in array):
        raise ValueError(f"{name} values exceed int64 range")
    return array.astype(np.int64, copy=True)


def _source_stage_arrays(stages: StageData) -> tuple[np.ndarray, ...]:
    source_arrays = (
        np.asarray(stages.raw_amplitude),
        np.asarray(stages.log_repaired_amplitude),
        np.asarray(stages.processed_zscore),
    )
    if any(array.ndim != 3 for array in source_arrays):
        raise ValueError("source stage arrays must have shape (rx, packet, subcarrier)")
    if not all(array.shape == source_arrays[0].shape for array in source_arrays[1:]):
        raise ValueError("source stage array shapes do not match")
    try:
        arrays = tuple(array.astype(np.float32, copy=False) for array in source_arrays)
    except (TypeError, ValueError, OverflowError) as error:
        raise ValueError("source stage arrays must convert to float32") from error
    if not all(np.isfinite(array).all() for array in arrays):
        raise ValueError("source stage data contains non-finite values")
    return arrays


def write_source_data(path: str | Path, sample_id: str, stages: StageData) -> None:
    if not sample_id:
        raise ValueError("sample_id must not be empty")
    arrays = _source_stage_arrays(stages)
    rx_count, target_packets, target_subcarriers = arrays[0].shape
    packet_indices = _validated_indices(
        stages.packet_indices,
        name="packet_indices",
        expected_length=target_packets,
    )
    subcarrier_indices = _validated_indices(
        stages.subcarrier_indices,
        name="subcarrier_indices",
        expected_length=target_subcarriers,
    )

    rows: list[list[str | int]] = []
    for rx in range(rx_count):
        for output_packet in range(target_packets):
            source_packet = int(packet_indices[output_packet])
            for output_subcarrier in range(target_subcarriers):
                source_subcarrier = int(subcarrier_indices[output_subcarrier])
                rows.append(
                    [
                        sample_id,
                        rx + 1,
                        output_packet,
                        source_packet,
                        output_subcarrier,
                        source_subcarrier,
                        format(
                            float(arrays[0][rx, output_packet, output_subcarrier]),
                            ".9g",
                        ),
                        format(
                            float(arrays[1][rx, output_packet, output_subcarrier]),
                            ".9g",
                        ),
                        format(
                            float(arrays[2][rx, output_packet, output_subcarrier]),
                            ".9g",
                        ),
                    ]
                )

    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(SOURCE_DATA_FIELDS)
        writer.writerows(rows)


def _canonical_decimal_integer(value: str, *, field: str, row_number: int) -> int:
    if re.fullmatch(r"0|[1-9][0-9]*", value) is None:
        raise ValueError(
            f"{field} must be a canonical decimal integer at row {row_number}"
        )
    return int(value)


def read_source_data(
    path: str | Path,
    *,
    rx_count: int,
    target_packets: int,
    target_subcarriers: int,
    expected_packet_indices: np.ndarray,
    expected_subcarrier_indices: np.ndarray,
) -> SourceData:
    dimensions = (rx_count, target_packets, target_subcarriers)
    if any(value <= 0 for value in dimensions):
        raise ValueError("source data dimensions must be positive")
    packet_indices = _validated_indices(
        expected_packet_indices,
        name="expected_packet_indices",
        expected_length=target_packets,
    )
    subcarrier_indices = _validated_indices(
        expected_subcarrier_indices,
        name="expected_subcarrier_indices",
        expected_length=target_subcarriers,
    )

    with Path(path).open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        if reader.fieldnames != SOURCE_DATA_FIELDS:
            raise ValueError(
                "source data header mismatch: "
                f"expected {SOURCE_DATA_FIELDS}, got {reader.fieldnames}"
            )
        rows = list(reader)

    expected_count = rx_count * target_packets * target_subcarriers
    if len(rows) != expected_count:
        raise ValueError(
            f"source data row count mismatch: expected {expected_count}, got {len(rows)}"
        )

    raw = np.empty(dimensions, dtype=np.float32)
    repaired = np.empty(dimensions, dtype=np.float32)
    processed = np.empty(dimensions, dtype=np.float32)
    sample_id: str | None = None

    for row_number, row in enumerate(rows, start=2):
        if None in row or any(value is None for value in row.values()):
            raise ValueError(f"source data row width mismatch at row {row_number}")
        flat_index = row_number - 2
        expected_rx = flat_index // (target_packets * target_subcarriers)
        within_rx = flat_index % (target_packets * target_subcarriers)
        expected_packet = within_rx // target_subcarriers
        expected_subcarrier = within_rx % target_subcarriers
        rx_index = _canonical_decimal_integer(
            row["rx_index"], field="rx_index", row_number=row_number
        )
        output_packet = _canonical_decimal_integer(
            row["output_packet_index"],
            field="output_packet_index",
            row_number=row_number,
        )
        source_packet = _canonical_decimal_integer(
            row["source_packet_index"],
            field="source_packet_index",
            row_number=row_number,
        )
        output_subcarrier = _canonical_decimal_integer(
            row["output_subcarrier_index"],
            field="output_subcarrier_index",
            row_number=row_number,
        )
        source_subcarrier = _canonical_decimal_integer(
            row["source_subcarrier_index"],
            field="source_subcarrier_index",
            row_number=row_number,
        )
        try:
            values = np.asarray(
                [
                    row["raw_amplitude"],
                    row["log_repaired_amplitude"],
                    row["processed_zscore"],
                ],
                dtype=np.float32,
            )
        except (KeyError, TypeError, ValueError) as error:
            raise ValueError(f"invalid source data at row {row_number}") from error
        if (
            rx_index != expected_rx + 1
            or output_packet != expected_packet
            or output_subcarrier != expected_subcarrier
        ):
            raise ValueError(f"source data order mismatch at row {row_number}")
        if not np.isfinite(values).all():
            raise ValueError(f"non-finite source data at row {row_number}")

        row_sample_id = row["sample_id"]
        if not row_sample_id:
            raise ValueError(f"empty sample_id at row {row_number}")
        if sample_id is None:
            sample_id = row_sample_id
        elif row_sample_id != sample_id:
            raise ValueError(f"sample_id mismatch at row {row_number}")

        if source_packet != packet_indices[expected_packet]:
            raise ValueError(f"source packet index mismatch at row {row_number}")
        if source_subcarrier != subcarrier_indices[expected_subcarrier]:
            raise ValueError(f"source subcarrier index mismatch at row {row_number}")

        raw[expected_rx, expected_packet, expected_subcarrier] = values[0]
        repaired[expected_rx, expected_packet, expected_subcarrier] = values[1]
        processed[expected_rx, expected_packet, expected_subcarrier] = values[2]

    return SourceData(
        raw_amplitude=raw,
        log_repaired_amplitude=repaired,
        processed_zscore=processed,
        packet_indices=packet_indices,
        subcarrier_indices=subcarrier_indices,
        sample_id=sample_id or "",
    )


def _stage_summary(values: np.ndarray) -> dict[str, bool | float]:
    return {
        "finite": bool(np.isfinite(values).all()),
        "max": float(values.max()),
        "min": float(values.min()),
    }


def write_diagnostics(
    path: str | Path,
    *,
    sample_id: str,
    stages: StageData,
    decoded_shape: tuple[int, ...],
    decoded_dtype: np.dtype | type,
    authoritative_max_abs_difference: float,
    target_packets: int,
    target_subcarriers: int,
    low_energy_ratio: float,
    epsilon: float,
) -> None:
    stage_summaries = {
        "raw_amplitude": _stage_summary(stages.raw_amplitude),
        "log_repaired_amplitude": _stage_summary(
            stages.log_repaired_amplitude
        ),
        "processed_zscore": _stage_summary(stages.processed_zscore),
    }
    finite_scalars = np.asarray(
        [
            stages.reference_carrier_energy,
            authoritative_max_abs_difference,
            low_energy_ratio,
            epsilon,
        ],
        dtype=np.float64,
    )
    if not all(summary["finite"] for summary in stage_summaries.values()):
        raise ValueError("diagnostics contain non-finite stage values")
    if not np.isfinite(finite_scalars).all():
        raise ValueError("diagnostics contain non-finite scalar values")
    if authoritative_max_abs_difference != 0.0:
        raise ValueError("authoritative_max_abs_difference must be zero")
    if not np.isfinite(stages.zscore_mean).all() or not np.isfinite(
        stages.zscore_std
    ).all():
        raise ValueError("diagnostics contain non-finite z-score statistics")

    payload = {
        "authoritative_max_abs_difference": float(
            authoritative_max_abs_difference
        ),
        "carrier_quality": {
            "low_energy_indices": stages.low_energy_indices.tolist(),
            "low_energy_threshold": float(
                stages.reference_carrier_energy * low_energy_ratio
            ),
            "reference_carrier_energy": float(stages.reference_carrier_energy),
            "valid_subcarrier_indices": np.flatnonzero(
                stages.valid_subcarrier_mask
            ).tolist(),
            "valid_subcarrier_mask": stages.valid_subcarrier_mask.tolist(),
        },
        "decoded_csi": {
            "dtype": np.dtype(decoded_dtype).name,
            "shape": [int(value) for value in decoded_shape],
        },
        "processing_parameters": {
            "epsilon": float(epsilon),
            "low_energy_ratio": float(low_energy_ratio),
            "target_packets": int(target_packets),
            "target_subcarriers": int(target_subcarriers),
        },
        "sample_id": sample_id,
        "selection": {
            "packet_indices": stages.packet_indices.tolist(),
            "subcarrier_indices": stages.subcarrier_indices.tolist(),
        },
        "stages": stage_summaries,
        "zscore": {
            "mean_by_rx": stages.zscore_mean.tolist(),
            "std_by_rx": stages.zscore_std.tolist(),
        },
    }

    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def _centers_to_edges(centers: np.ndarray) -> np.ndarray:
    values = np.asarray(centers, dtype=np.float64)
    if values.ndim != 1 or len(values) < 2:
        raise ValueError("centers must be a one-dimensional array of length at least two")
    if not np.isfinite(values).all() or np.any(values[1:] <= values[:-1]):
        raise ValueError("centers must be finite and strictly increasing")
    edges = np.empty(len(values) + 1, dtype=np.float64)
    edges[1:-1] = (values[:-1] + values[1:]) / 2.0
    edges[0] = values[0] - (values[1] - values[0]) / 2.0
    edges[-1] = values[-1] + (values[-1] - values[-2]) / 2.0
    return edges


def _panel_label_style(column_index: int) -> tuple[str, str]:
    if column_index == 0:
        return "white", "black"
    if column_index == 1:
        return "black", "white"
    raise ValueError("column_index must be 0 or 1")


def plot_figure(
    source_data: SourceData,
    png_path: str | Path,
    pdf_path: str | Path,
) -> None:
    import matplotlib

    matplotlib.use("Agg", force=True)
    import matplotlib.patheffects as path_effects
    import matplotlib.pyplot as plt

    raw = np.asarray(source_data.raw_amplitude)
    processed = np.asarray(source_data.processed_zscore)
    if raw.shape != processed.shape or raw.ndim != 3 or raw.shape[0] != 2:
        raise ValueError("figure source data must contain two matching RX arrays")
    if not np.isfinite(raw).all() or not np.isfinite(processed).all():
        raise ValueError("figure source data contains non-finite values")

    packet_indices = _validated_indices(
        source_data.packet_indices,
        name="packet_indices",
        expected_length=raw.shape[1],
    )
    subcarrier_indices = _validated_indices(
        source_data.subcarrier_indices,
        name="subcarrier_indices",
        expected_length=raw.shape[2],
    )
    packet_edges = _centers_to_edges(packet_indices)
    subcarrier_edges = _centers_to_edges(subcarrier_indices)
    raw_min = float(raw.min())
    raw_max = float(raw.max())
    processed_limit = float(np.max(np.abs(processed)))
    if raw_min == raw_max:
        raw_max = raw_min + 1.0
    if processed_limit == 0.0:
        processed_limit = 1.0

    rc_params = {
        "font.family": "sans-serif",
        "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans"],
        "font.size": 8,
        "axes.titlesize": 9,
        "axes.labelsize": 8,
        "xtick.labelsize": 8,
        "ytick.labelsize": 8,
        "savefig.dpi": 300,
        "pdf.fonttype": 42,
    }
    fixed_pdf_date = dt.datetime(2000, 1, 1, tzinfo=dt.timezone.utc)
    with matplotlib.rc_context(rc_params):
        figure, axes = plt.subplots(
            2,
            2,
            figsize=(6.9, 5.6),
            layout="constrained",
            sharex=True,
            sharey=True,
        )
        try:
            raw_image = None
            processed_image = None
            panel_labels = ("(a)", "(b)", "(c)", "(d)")
            for rx_index in range(2):
                raw_image = axes[rx_index, 0].pcolormesh(
                    subcarrier_edges,
                    packet_edges,
                    raw[rx_index],
                    cmap="viridis",
                    vmin=raw_min,
                    vmax=raw_max,
                    shading="flat",
                    antialiased=False,
                )
                processed_image = axes[rx_index, 1].pcolormesh(
                    subcarrier_edges,
                    packet_edges,
                    processed[rx_index],
                    cmap="RdBu_r",
                    vmin=-processed_limit,
                    vmax=processed_limit,
                    shading="flat",
                    antialiased=False,
                )
                axes[rx_index, 0].set_aspect("auto")
                axes[rx_index, 1].set_aspect("auto")
                axes[rx_index, 0].set_ylabel("Source packet index")
                axes[rx_index, 0].text(
                    -0.28,
                    0.5,
                    f"RX {rx_index + 1}",
                    transform=axes[rx_index, 0].transAxes,
                    ha="center",
                    va="center",
                    rotation=90,
                    fontweight="bold",
                )
                for column in range(2):
                    text_color, outline_color = _panel_label_style(column)
                    axes[rx_index, column].text(
                        0.02,
                        0.97,
                        panel_labels[rx_index * 2 + column],
                        transform=axes[rx_index, column].transAxes,
                        ha="left",
                        va="top",
                        color=text_color,
                        fontweight="bold",
                        path_effects=[
                            path_effects.withStroke(
                                linewidth=2,
                                foreground=outline_color,
                            )
                        ],
                    )
            axes[0, 0].invert_yaxis()
            axes[0, 0].set_title("Matched-grid raw amplitude")
            axes[0, 1].set_title("Processed CSI input")
            axes[1, 0].set_xlabel("Source subcarrier index")
            axes[1, 1].set_xlabel("Source subcarrier index")
            assert raw_image is not None and processed_image is not None
            raw_colorbar = figure.colorbar(
                raw_image, ax=axes[:, 0], pad=0.02, shrink=0.90
            )
            raw_colorbar.set_label("|H| (arbitrary units)")
            processed_colorbar = figure.colorbar(
                processed_image, ax=axes[:, 1], pad=0.02, shrink=0.90
            )
            processed_colorbar.set_label("Standardized log-amplitude (z)")

            png_output = Path(png_path)
            pdf_output = Path(pdf_path)
            png_output.parent.mkdir(parents=True, exist_ok=True)
            pdf_output.parent.mkdir(parents=True, exist_ok=True)
            figure.savefig(
                png_output,
                dpi=300,
                metadata={"Software": "AXHome-MM reproducible CSI figure generator"},
            )
            figure.savefig(
                pdf_output,
                metadata={
                    "Creator": "AXHome-MM reproducible CSI figure generator",
                    "Producer": "Matplotlib",
                    "CreationDate": fixed_pdf_date,
                    "ModDate": fixed_pdf_date,
                },
            )
        finally:
            plt.close(figure)


def write_caption(
    path: str | Path,
    validated_sample: ValidatedSample,
    *,
    target_packets: int = FROZEN_TARGET_PACKETS,
    target_subcarriers: int = FROZEN_TARGET_SUBCARRIERS,
) -> None:
    packets, num_rx, num_tx, subcarriers = validated_sample.csi.shape
    row = validated_sample.row
    action = row.get("action_id", "unknown")
    environment = row.get("environment_id", "unknown")
    text = f"""# Candidate supplementary CSI input figure caption

**English.** CSI input construction for sample `{validated_sample.sample_id}` (action `{action}`, environment `{environment}`; {packets} decoded packets, {num_rx} RX, {num_tx} TX, and {subcarriers} source subcarriers). The left column shows decoded `|H|` on the same evenly selected packet/subcarrier indices used by the model input. This matched-grid raw view is a visual alignment only and does not change the authoritative order. The right column shows the actual preprocessing path: `log1p(|H|)`, low-energy-subcarrier repair, even selection to {target_packets} packets by {target_subcarriers} subcarriers, and per-RX z-score standardization. CSI phase is not used because no validated phase-calibration chain is available. This is a single-sample preprocessing illustration, not evidence about the dataset distribution or model performance.

**中文核对译文。** 样本 `{validated_sample.sample_id}`（动作 `{action}`、环境 `{environment}`）共解码 {packets} 个包、{num_rx} 路 RX、{num_tx} 路 TX 和 {subcarriers} 个源子载波。左列是在与模型输入相同的均匀抽取包/子载波索引上显示的解码 `|H|`；该对齐网格原始视图只用于视觉核对，不改变权威预处理顺序。右列展示实际流程：`log1p(|H|)`、低能量子载波修复、均匀抽取为 {target_packets}×{target_subcarriers}，以及逐 RX z-score 标准化。由于没有经验证的相位校准链，本图不使用 CSI phase。本图仅为单样本预处理示意，不代表数据集分布或模型性能。
"""
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(text, encoding="utf-8", newline="\n")


def write_status_update(
    path: str | Path,
    validated_sample: ValidatedSample,
) -> None:
    text = f"""# CSI 输入候选图状态更新

- 溯源状态：`sufficient`。冻结样本为 `{validated_sample.sample_id}`，清单、CSI、元数据、解析器与权威预处理实现均由 generation record 记录并校验。
- 生成入口：`quality_check/generate_candidate_supplementary_csi_input.py`；候选包位置由命令行 `--output-dir` 指定，本状态文本不嵌入机器相关绝对路径。
- 边界说明：由于不存在已验证的相位校准链，当前只展示幅值输入流程，不将 phase 纳入图示或结论。
- 合并状态：这是可审阅的候选补充图包，尚未分配正式图号，也未修改论文正文。
"""
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(text, encoding="utf-8", newline="\n")


def collect_git_state(project_root: str | Path) -> dict[str, str | bool]:
    root = Path(project_root).resolve()

    def run(*arguments: str) -> str:
        completed = subprocess.run(
            ["git", "-C", str(root), *arguments],
            check=True,
            capture_output=True,
            text=True,
            encoding="utf-8",
        )
        return completed.stdout.strip()

    return {
        "commit": run("rev-parse", "HEAD"),
        "branch": run("rev-parse", "--abbrev-ref", "HEAD"),
        "dirty": bool(run("status", "--porcelain")),
    }


def _file_record(path: Path) -> dict[str, str | int]:
    return {
        "path": str(path.resolve()),
        "sha256": sha256_file(path),
        "size_bytes": path.stat().st_size,
    }


def _write_generation_record(
    path: Path,
    *,
    project_root: Path,
    dataset_root: Path,
    output_dir: Path,
    validated_sample: ValidatedSample,
    generated_at: str,
    command_argv: Sequence[str],
    git_state: dict[str, str | bool],
    target_packets: int,
    target_subcarriers: int,
    low_energy_ratio: float,
    epsilon: float,
    staged_dir: Path,
) -> None:
    import matplotlib

    argv = list(command_argv)
    parser_path = BASELINE_ROOT / "axhome_csi" / "feitcsi.py"
    preprocessing_path = BASELINE_ROOT / "axhome_csi" / "data.py"
    generator_path = Path(__file__).resolve()
    code_input_paths = {
        "parser": parser_path,
        "preprocessing": preprocessing_path,
        "generator": generator_path,
    }
    inputs = {
        "manifest": {
            "path": str(validated_sample.manifest_path.resolve()),
            "sha256": validated_sample.manifest_sha256,
            "size_bytes": validated_sample.manifest_size_bytes,
        },
        "csi": {
            "path": str(validated_sample.csi_path.resolve()),
            "sha256": validated_sample.csi_sha256,
            "size_bytes": validated_sample.csi_size_bytes,
        },
        "metadata": {
            "path": str(validated_sample.metadata_path.resolve()),
            "sha256": validated_sample.metadata_sha256,
            "size_bytes": validated_sample.metadata_size_bytes,
        },
        **{name: _file_record(item) for name, item in code_input_paths.items()},
    }
    outputs = {
        name: {
            "sha256": sha256_file(staged_dir / name),
            "size_bytes": (staged_dir / name).stat().st_size,
        }
        for name in DETERMINISTIC_OUTPUT_NAMES
    }
    payload = {
        "schema_version": 1,
        "provenance_status": "sufficient",
        "generated_at_utc": generated_at,
        "argv": argv,
        "command": subprocess.list2cmdline(argv),
        "paths": {
            "project_root": str(project_root),
            "dataset_root": str(dataset_root),
            "output_dir": str(output_dir),
        },
        "git": git_state,
        "sample": {
            "sample_id": validated_sample.sample_id,
            "manifest_row": dict(validated_sample.row),
        },
        "inputs": inputs,
        "decoded_csi": {
            "shape": [int(value) for value in validated_sample.csi.shape],
            "dtype": validated_sample.csi.dtype.name,
            "packet_count": len(validated_sample.headers),
        },
        "parameters": {
            "plotting": {
                "dpi": 300,
                "figure_size_inches": [6.9, 5.6],
                "processed_center": 0,
                "processed_colormap": "RdBu_r",
                "raw_colormap": "viridis",
                "shading": "flat",
            },
            "preprocessing": {
                "target_packets": int(target_packets),
                "target_subcarriers": int(target_subcarriers),
                "low_energy_ratio": float(low_energy_ratio),
                "epsilon": float(epsilon),
            },
        },
        "software": {
            "python": sys.version.split()[0],
            "numpy": np.__version__,
            "matplotlib": matplotlib.__version__,
        },
        "outputs": outputs,
        "limitations": [
            "CSI phase is excluded because no validated phase-calibration chain is available.",
            "The figure is a single-sample preprocessing illustration and does not characterize the dataset distribution.",
            "The figure does not provide evidence of model performance or comparative effectiveness.",
            "Files are replaced atomically one by one; an OS-level failure during publication can require regeneration.",
            "Code-file hashes for the generator, parser, and preprocessing implementation are sampled when the generation record is written; concurrent code modification during a run is outside the reproducibility guarantee.",
        ],
    }
    path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
        newline="\n",
    )


def generate_package(
    *,
    project_root: str | Path,
    dataset_root: str | Path,
    sample_id: str,
    expected_manifest_sha256: str,
    expected_csi_sha256: str,
    expected_shape: tuple[int, ...],
    target_packets: int,
    target_subcarriers: int,
    low_energy_ratio: float,
    epsilon: float,
    output_dir: str | Path,
    generated_at: str,
    command_argv: Sequence[str],
) -> dict[str, object]:
    resolved_project_root = Path(project_root).resolve()
    resolved_dataset_root = Path(dataset_root).resolve()
    resolved_output_dir = Path(output_dir).resolve()
    output_parent = resolved_output_dir.parent
    output_parent.mkdir(parents=True, exist_ok=True)
    git_state = collect_git_state(resolved_project_root)

    with tempfile.TemporaryDirectory(
        prefix=".candidate-csi-", dir=output_parent
    ) as temporary_directory:
        staged_dir = Path(temporary_directory)
        validated = validate_frozen_sample(
            resolved_dataset_root,
            sample_id=sample_id,
            expected_manifest_sha256=expected_manifest_sha256,
            expected_csi_sha256=expected_csi_sha256,
            expected_shape=expected_shape,
        )
        stages = extract_stages(
            validated.csi,
            target_packets=target_packets,
            target_subcarriers=target_subcarriers,
            low_energy_ratio=low_energy_ratio,
            epsilon=epsilon,
        )
        source_path = staged_dir / SOURCE_DATA_NAME
        write_source_data(source_path, validated.sample_id, stages)
        source_data = read_source_data(
            source_path,
            rx_count=stages.raw_amplitude.shape[0],
            target_packets=target_packets,
            target_subcarriers=target_subcarriers,
            expected_packet_indices=stages.packet_indices,
            expected_subcarrier_indices=stages.subcarrier_indices,
        )
        for readback, expected in (
            (source_data.raw_amplitude, stages.raw_amplitude),
            (source_data.log_repaired_amplitude, stages.log_repaired_amplitude),
            (source_data.processed_zscore, stages.processed_zscore),
        ):
            if not np.array_equal(readback, expected):
                raise RuntimeError("strict source-data readback differs from stages")
        authoritative = preprocess_amplitude(
            validated.csi,
            target_packets=target_packets,
            target_subcarriers=target_subcarriers,
            low_energy_ratio=low_energy_ratio,
            epsilon=epsilon,
        )
        difference = float(
            np.max(np.abs(source_data.processed_zscore - authoritative))
        )
        if difference != 0.0:
            raise RuntimeError("source-data readback differs from authoritative preprocessing")
        write_diagnostics(
            staged_dir / DIAGNOSTICS_NAME,
            sample_id=validated.sample_id,
            stages=stages,
            decoded_shape=validated.csi.shape,
            decoded_dtype=validated.csi.dtype,
            authoritative_max_abs_difference=difference,
            target_packets=target_packets,
            target_subcarriers=target_subcarriers,
            low_energy_ratio=low_energy_ratio,
            epsilon=epsilon,
        )
        write_caption(
            staged_dir / CAPTION_NAME,
            validated,
            target_packets=target_packets,
            target_subcarriers=target_subcarriers,
        )
        write_status_update(staged_dir / STATUS_UPDATE_NAME, validated)
        plot_figure(
            source_data,
            staged_dir / PNG_NAME,
            staged_dir / PDF_NAME,
        )
        _write_generation_record(
            staged_dir / GENERATION_RECORD_NAME,
            project_root=resolved_project_root,
            dataset_root=resolved_dataset_root,
            output_dir=resolved_output_dir,
            validated_sample=validated,
            generated_at=generated_at,
            command_argv=command_argv,
            git_state=git_state,
            target_packets=target_packets,
            target_subcarriers=target_subcarriers,
            low_energy_ratio=low_energy_ratio,
            epsilon=epsilon,
            staged_dir=staged_dir,
        )

        resolved_output_dir.mkdir(parents=True, exist_ok=True)
        for name in PACKAGE_OUTPUT_NAMES:
            os.replace(staged_dir / name, resolved_output_dir / name)

    return {
        "provenance_status": "sufficient",
        "artifacts": [str(resolved_output_dir / name) for name in PACKAGE_OUTPUT_NAMES],
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Generate the frozen AXHome-MM candidate CSI input figure package."
    )
    parser.add_argument("--project-root", required=True, type=Path)
    parser.add_argument("--dataset-root", required=True, type=Path)
    parser.add_argument("--sample-id", required=True)
    parser.add_argument("--output-dir", required=True, type=Path)
    return parser


def main(argv: list[str] | None = None) -> int:
    command_arguments = list(sys.argv[1:] if argv is None else argv)
    parser = build_parser()
    args = parser.parse_args(command_arguments)
    if args.sample_id != FROZEN_SAMPLE_ID:
        parser.error(f"--sample-id must be the frozen sample {FROZEN_SAMPLE_ID}")
    generated_at = dt.datetime.now(dt.timezone.utc).isoformat().replace("+00:00", "Z")
    script_relative = "quality_check/generate_candidate_supplementary_csi_input.py"
    command_argv = [sys.executable, script_relative, *command_arguments]
    summary = generate_package(
        project_root=args.project_root,
        dataset_root=args.dataset_root,
        sample_id=args.sample_id,
        expected_manifest_sha256=FROZEN_MANIFEST_SHA256,
        expected_csi_sha256=FROZEN_CSI_SHA256,
        expected_shape=FROZEN_CSI_SHAPE,
        target_packets=FROZEN_TARGET_PACKETS,
        target_subcarriers=FROZEN_TARGET_SUBCARRIERS,
        low_energy_ratio=FROZEN_LOW_ENERGY_RATIO,
        epsilon=FROZEN_EPSILON,
        output_dir=args.output_dir,
        generated_at=generated_at,
        command_argv=command_argv,
    )
    print(json.dumps(summary, ensure_ascii=False, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
