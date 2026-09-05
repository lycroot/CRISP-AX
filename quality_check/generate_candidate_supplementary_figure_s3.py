from __future__ import annotations

import argparse
import csv
import datetime as dt
import gzip
import io
import json
import math
import os
import subprocess
import sys
import tempfile
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path

import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from quality_check.generate_candidate_supplementary_csi_input import (
    StageData,
    _centers_to_edges,
    collect_git_state,
    extract_stages,
    sha256_bytes,
    sha256_file,
    validate_frozen_sample,
)
from axhome_csi.data import preprocess_amplitude


@dataclass(frozen=True)
class FrozenSampleSpec:
    role: str
    sample_id: str
    participant: str
    action: str
    environment: str
    link_configuration: str
    expected_shape: tuple[int, int, int, int]
    expected_csi_sha256: str


@dataclass(frozen=True)
class SampleArrays:
    spec: FrozenSampleSpec
    native_raw_amplitude: np.ndarray
    stages: StageData
    authoritative_max_abs_difference: float


@dataclass(frozen=True)
class ScaleDiagnostics:
    raw_vmin: float
    raw_vmax: float
    raw_percentile: float
    raw_clipped_fraction: float
    processed_limit: float
    processed_percentile: float
    processed_clipped_fraction: float
    processed_clipped_fraction_by_sample: dict[str, float]


@dataclass(frozen=True)
class SourceDataBundle:
    native_raw_amplitude: np.ndarray
    processed_by_sample: dict[str, np.ndarray]
    packet_indices_by_sample: dict[str, np.ndarray]
    subcarrier_indices_by_sample: dict[str, np.ndarray]
    row_count: int


FROZEN_MANIFEST_SHA256 = (
    "7d7ea605819c2c988b068b3f86ed77419c5a801c0d49b279b74c671cfb654a00"
)
FROZEN_EVIDENCE_SHA256 = {
    "exclusion_ledger": (
        "7d147d6a69b990e5c37e1f0c1cbfc1becf8591be4829a8d5d5af7b879894ac20"
    ),
    "active_csi_audit": (
        "23e13556a309005b868e23315fe978b403241d30d208eae18a0ad29146240fa5"
    ),
    "active_sample_audit": (
        "c040678d18fc0d38b12669d9be4d61affe4f683e67079c60ae980168173274dc"
    ),
    "active_metadata_audit": (
        "e4fe32b2830cbe655b35d9a620513d190aaaa5f0d27652c8a4051409f4e9aedb"
    ),
    "independent_ftm_recheck": (
        "899917047ce1cbd08272132e3c66dea86ac4610bdeda7ae05eaae40b64ae0009"
    ),
}
FROZEN_SAMPLES = (
    FrozenSampleSpec(
        role="preprocessing_representative",
        sample_id="S14_P05_walk_face_rx_E1_clean_none_R006",
        participant="P05",
        action="walk",
        environment="E1",
        link_configuration="face_rx",
        expected_shape=(496, 2, 1, 996),
        expected_csi_sha256=(
            "a301478f6be1e6ecb340fb6e7ac878851f58208e59e40ff7c6f09a597bbc2606"
        ),
    ),
    FrozenSampleSpec(
        role="diversity_a",
        sample_id="S13_P04_walk_face_rx_E1_clean_none_R014",
        participant="P04",
        action="walk",
        environment="E1",
        link_configuration="face_rx",
        expected_shape=(495, 2, 1, 996),
        expected_csi_sha256=(
            "23e3191f8d5a06cf7a326866e7951adba6b64018035a65915b45118acbce1623"
        ),
    ),
    FrozenSampleSpec(
        role="diversity_b",
        sample_id="S03_P01_lie_down_side_link_E2_clean_none_R004",
        participant="P01",
        action="lie_down",
        environment="E2",
        link_configuration="side_link",
        expected_shape=(495, 2, 1, 996),
        expected_csi_sha256=(
            "d144b9b6b5140158df7d87b8f44160d59891434942259a28f66cd27489d54777"
        ),
    ),
    FrozenSampleSpec(
        role="diversity_c",
        sample_id="S23_P08_wash_hands_cross_link_E3_clean_none_R025",
        participant="P08",
        action="wash_hands",
        environment="E3",
        link_configuration="cross_link",
        expected_shape=(495, 2, 1, 996),
        expected_csi_sha256=(
            "d2af36f96b87513dcf897f4fab4e539ef7c33f95b11915b4f80e348d326f44e7"
        ),
    ),
)

TARGET_PACKETS = 256
TARGET_SUBCARRIERS = 128
LOW_ENERGY_RATIO = 0.05
EPSILON = 1e-6

PNG_NAME = "candidate_supplementary_figure_s3.png"
PDF_NAME = "candidate_supplementary_figure_s3.pdf"
V2_PNG_NAME = "figure_s3_csi_representation_v2.png"
V2_PDF_NAME = "figure_s3_csi_representation_v2.pdf"
V2_FIGURE_WIDTH_MM = 180.0
V2_FIGURE_HEIGHT_MM = 92.0
V3_PNG_NAME = "figure_s3_csi_representation_v3.png"
V3_PDF_NAME = "figure_s3_csi_representation_v3.pdf"
V3_SVG_NAME = "figure_s3_csi_representation_v3.svg"
V3_CAPTION_NAME = "figure_s3_csi_representation_v3_caption.md"
V3_OUTPUT_NAMES = (
    V3_PNG_NAME,
    V3_PDF_NAME,
    V3_SVG_NAME,
    V3_CAPTION_NAME,
)
V3_FIGURE_WIDTH_MM = 180.0
V3_FIGURE_HEIGHT_MM = 94.0
V4_PNG_NAME = "figure_s3_csi_representation_v4.png"
V4_PDF_NAME = "figure_s3_csi_representation_v4.pdf"
V4_SVG_NAME = "figure_s3_csi_representation_v4.svg"
V4_VALIDATION_NAME = "figure_s3_csi_representation_v4_coordinate_validation.json"
V4_OUTPUT_NAMES = (
    V4_PNG_NAME,
    V4_PDF_NAME,
    V4_SVG_NAME,
    V4_VALIDATION_NAME,
)
FIGURE6_STEM = "figure6_csi_representation"
FIGURE6_PNG_NAME = f"{FIGURE6_STEM}.png"
FIGURE6_PDF_NAME = f"{FIGURE6_STEM}.pdf"
FIGURE6_SVG_NAME = f"{FIGURE6_STEM}.svg"
FIGURE6_SOURCE_DATA_NAME = f"{FIGURE6_STEM}_source_data.csv.gz"
FIGURE6_SOURCE_SCHEMA_NAME = f"{FIGURE6_STEM}_source_data_schema.json"
FIGURE6_DIAGNOSTICS_NAME = f"{FIGURE6_STEM}_diagnostics.json"
FIGURE6_VALIDATION_NAME = f"{FIGURE6_STEM}_coordinate_validation.json"
FIGURE6_GENERATION_RECORD_NAME = f"{FIGURE6_STEM}_generation_record.json"
FIGURE6_CHANGE_NOTE_NAME = "CHANGELOG.md"
FIGURE6_OUTPUT_NAMES = (
    FIGURE6_PNG_NAME,
    FIGURE6_PDF_NAME,
    FIGURE6_SVG_NAME,
    FIGURE6_SOURCE_DATA_NAME,
    FIGURE6_SOURCE_SCHEMA_NAME,
    FIGURE6_DIAGNOSTICS_NAME,
    FIGURE6_VALIDATION_NAME,
    FIGURE6_GENERATION_RECORD_NAME,
    FIGURE6_CHANGE_NOTE_NAME,
)
FIGURE6_DETERMINISTIC_OUTPUT_NAMES = tuple(
    name
    for name in FIGURE6_OUTPUT_NAMES
    if name != FIGURE6_GENERATION_RECORD_NAME
)
FIGURE6_WIDTH_IN = 7.2
FIGURE6_HEIGHT_IN = 3.50
V4_SOURCE_SUBCARRIER_COUNT = 996
V4_MAIN_SOURCE_PACKET_COUNT = 496
V4_REPRESENTATIVE_SOURCE_PACKET_COUNT = 495
V4_X_LIMITS = (-0.5, 995.5)
V4_X_TICKS = (0.0, 500.0)
V4_MAIN_Y_LIMITS = (495.5, -0.5)
V4_REPRESENTATIVE_Y_LIMITS = (494.5, -0.5)
V4_Y_TICKS = (0.0, 200.0, 400.0)
V4_X_FRACTION_COORDINATES = (0.0, 500.0, 995.0)
V4_Y_FRACTION_COORDINATES = (0.0, 200.0, 400.0)
V4_AXIS_IDS = (
    "rx1_raw_main",
    "rx1_processed_main",
    "rx1_walk",
    "rx1_lie_down",
    "rx1_wash_hands",
    "rx2_raw_main",
    "rx2_processed_main",
    "rx2_walk",
    "rx2_lie_down",
    "rx2_wash_hands",
)
PUBLICATION_FONT_CANDIDATES = (
    "Arial",
    "DejaVu Sans",
)
SOURCE_DATA_NAME = "candidate_supplementary_figure_s3_source_data.csv.gz"
SOURCE_SCHEMA_NAME = "candidate_supplementary_figure_s3_source_data_schema.json"
DIAGNOSTICS_NAME = "candidate_supplementary_figure_s3_diagnostics.json"
CAPTION_NAME = "candidate_supplementary_figure_s3_caption.md"
GENERATION_RECORD_NAME = "candidate_supplementary_figure_s3_generation_record.json"
STATUS_UPDATE_NAME = "candidate_status_update.md"
PACKAGE_OUTPUT_NAMES = (
    PNG_NAME,
    PDF_NAME,
    SOURCE_DATA_NAME,
    SOURCE_SCHEMA_NAME,
    DIAGNOSTICS_NAME,
    CAPTION_NAME,
    GENERATION_RECORD_NAME,
    STATUS_UPDATE_NAME,
)
DETERMINISTIC_OUTPUT_NAMES = tuple(
    name for name in PACKAGE_OUTPUT_NAMES if name != GENERATION_RECORD_NAME
)

SOURCE_DATA_FIELDS = (
    "sample_role",
    "sample_id",
    "representation",
    "rx_index",
    "output_packet_index",
    "source_packet_index",
    "output_subcarrier_index",
    "source_subcarrier_index",
    "value",
)


def extract_sample_arrays(
    spec: FrozenSampleSpec,
    csi: np.ndarray,
    *,
    target_packets: int = TARGET_PACKETS,
    target_subcarriers: int = TARGET_SUBCARRIERS,
) -> SampleArrays:
    array = np.asarray(csi)
    if array.ndim != 4:
        raise ValueError(
            f"sample {spec.sample_id}: CSI must have four dimensions, got {array.shape}"
        )
    if array.shape[2] != 1:
        raise ValueError(
            f"sample {spec.sample_id}: CSI must have one TX stream, got {array.shape}"
        )
    if not np.isfinite(array).all():
        raise ValueError(f"sample {spec.sample_id}: CSI contains non-finite values")

    native_raw = np.transpose(np.abs(array[:, :, 0, :]), (1, 0, 2)).astype(
        np.float32
    )
    if not np.isfinite(native_raw).all():
        raise ValueError(
            f"sample {spec.sample_id}: native raw amplitude contains non-finite values"
        )
    try:
        stages = extract_stages(
            array,
            target_packets=target_packets,
            target_subcarriers=target_subcarriers,
            low_energy_ratio=LOW_ENERGY_RATIO,
            epsilon=EPSILON,
        )
    except (RuntimeError, ValueError) as error:
        raise type(error)(f"sample {spec.sample_id}: {error}") from error

    authoritative = preprocess_amplitude(
        array,
        target_packets=target_packets,
        target_subcarriers=target_subcarriers,
        low_energy_ratio=LOW_ENERGY_RATIO,
        epsilon=EPSILON,
    )
    processed = stages.processed_zscore
    if not np.isfinite(processed).all() or not np.isfinite(authoritative).all():
        raise ValueError(
            f"sample {spec.sample_id}: processed CSI contains non-finite values"
        )
    if not np.array_equal(processed, authoritative):
        difference = float(np.max(np.abs(processed - authoritative)))
        raise RuntimeError(
            f"sample {spec.sample_id}: diagnostic stages differ from authoritative "
            f"preprocessing (max abs difference {difference})"
        )

    return SampleArrays(
        spec=spec,
        native_raw_amplitude=native_raw.copy(),
        stages=stages,
        authoritative_max_abs_difference=0.0,
    )


def compute_scale_diagnostics(
    arrays_by_sample: Sequence[SampleArrays],
) -> ScaleDiagnostics:
    arrays = list(arrays_by_sample)
    if len(arrays) != len(FROZEN_SAMPLES):
        raise ValueError("scale policy requires all four frozen samples")
    observed_ids = [item.spec.sample_id for item in arrays]
    expected_ids = [item.sample_id for item in FROZEN_SAMPLES]
    if observed_ids != expected_ids:
        raise ValueError("scale policy requires the frozen sample order")

    raw = np.asarray(arrays[0].native_raw_amplitude)
    if raw.size == 0 or not np.isfinite(raw).all():
        raise ValueError("raw scale input must be non-empty and finite")
    processed_by_sample = [
        np.asarray(item.stages.processed_zscore) for item in arrays
    ]
    if any(item.size == 0 or not np.isfinite(item).all() for item in processed_by_sample):
        raise ValueError("processed scale inputs must be non-empty and finite")

    raw_percentile = 99.5
    processed_percentile = 99.0
    raw_vmax = float(np.percentile(raw, raw_percentile))
    pooled_processed_abs = np.concatenate(
        [np.abs(item).ravel() for item in processed_by_sample]
    )
    processed_limit = float(
        np.percentile(pooled_processed_abs, processed_percentile)
    )
    if raw_vmax <= 0.0 or processed_limit <= 0.0:
        raise ValueError("raw and processed scale limits must be positive")

    return ScaleDiagnostics(
        raw_vmin=0.0,
        raw_vmax=raw_vmax,
        raw_percentile=raw_percentile,
        raw_clipped_fraction=float(np.count_nonzero(raw > raw_vmax) / raw.size),
        processed_limit=processed_limit,
        processed_percentile=processed_percentile,
        processed_clipped_fraction=float(
            np.count_nonzero(pooled_processed_abs > processed_limit)
            / pooled_processed_abs.size
        ),
        processed_clipped_fraction_by_sample={
            item.spec.sample_id: float(
                np.count_nonzero(np.abs(values) > processed_limit) / values.size
            )
            for item, values in zip(arrays, processed_by_sample, strict=True)
        },
    )


def _require_frozen_arrays(
    arrays_by_sample: Sequence[SampleArrays],
) -> list[SampleArrays]:
    arrays = list(arrays_by_sample)
    if len(arrays) != len(FROZEN_SAMPLES):
        raise ValueError("source data requires all four frozen samples")
    if [item.spec.sample_id for item in arrays] != [
        item.sample_id for item in FROZEN_SAMPLES
    ]:
        raise ValueError("source data requires the frozen sample order")

    raw = np.asarray(arrays[0].native_raw_amplitude)
    if raw.ndim != 3 or raw.shape[0] != 2 or raw.dtype != np.float32:
        raise ValueError("native raw source data must be float32 with two RX arrays")
    if raw.size == 0 or not np.isfinite(raw).all():
        raise ValueError("native raw source data must be non-empty and finite")

    for item in arrays:
        processed = np.asarray(item.stages.processed_zscore)
        if processed.ndim != 3 or processed.shape[0] != 2:
            raise ValueError(
                f"sample {item.spec.sample_id}: processed source data must have two RX arrays"
            )
        if processed.dtype != np.float32 or not np.isfinite(processed).all():
            raise ValueError(
                f"sample {item.spec.sample_id}: processed source data must be finite float32"
            )
        packet_indices = np.asarray(item.stages.packet_indices)
        subcarrier_indices = np.asarray(item.stages.subcarrier_indices)
        for name, indices, expected_length in (
            ("packet", packet_indices, processed.shape[1]),
            ("subcarrier", subcarrier_indices, processed.shape[2]),
        ):
            if (
                indices.ndim != 1
                or len(indices) != expected_length
                or not np.issubdtype(indices.dtype, np.integer)
                or np.any(indices < 0)
                or (len(indices) > 1 and np.any(indices[1:] <= indices[:-1]))
            ):
                raise ValueError(
                    f"sample {item.spec.sample_id}: invalid {name} source indices"
                )
    return arrays


def _float32_text(value: np.float32) -> str:
    return format(float(value), ".9g")


def write_source_data(
    path: str | Path,
    arrays_by_sample: Sequence[SampleArrays],
) -> None:
    arrays = _require_frozen_arrays(arrays_by_sample)
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with output_path.open("wb") as raw_file:
        with gzip.GzipFile(
            filename="",
            mode="wb",
            fileobj=raw_file,
            compresslevel=9,
            mtime=0,
        ) as zipped_file:
            with io.TextIOWrapper(
                zipped_file,
                encoding="utf-8",
                newline="",
            ) as text_file:
                writer = csv.writer(text_file, lineterminator="\n")
                writer.writerow(SOURCE_DATA_FIELDS)

                main = arrays[0]
                raw = main.native_raw_amplitude
                for rx_index in range(raw.shape[0]):
                    for packet_index in range(raw.shape[1]):
                        writer.writerows(
                            (
                                main.spec.role,
                                main.spec.sample_id,
                                "raw_amplitude_native",
                                rx_index + 1,
                                packet_index,
                                packet_index,
                                subcarrier_index,
                                subcarrier_index,
                                _float32_text(
                                    raw[rx_index, packet_index, subcarrier_index]
                                ),
                            )
                            for subcarrier_index in range(raw.shape[2])
                        )

                for item in arrays:
                    processed = item.stages.processed_zscore
                    packet_indices = item.stages.packet_indices
                    subcarrier_indices = item.stages.subcarrier_indices
                    for rx_index in range(processed.shape[0]):
                        for output_packet_index, source_packet_index in enumerate(
                            packet_indices
                        ):
                            writer.writerows(
                                (
                                    item.spec.role,
                                    item.spec.sample_id,
                                    "processed_zscore",
                                    rx_index + 1,
                                    output_packet_index,
                                    int(source_packet_index),
                                    output_subcarrier_index,
                                    int(source_subcarrier_index),
                                    _float32_text(
                                        processed[
                                            rx_index,
                                            output_packet_index,
                                            output_subcarrier_index,
                                        ]
                                    ),
                                )
                                for output_subcarrier_index, source_subcarrier_index in enumerate(
                                    subcarrier_indices
                                )
                            )


def _canonical_nonnegative_integer(text: str, *, field: str, row_number: int) -> int:
    if not text or not text.isascii() or not text.isdigit():
        raise ValueError(f"row {row_number}: {field} must be a canonical integer")
    value = int(text)
    if str(value) != text:
        raise ValueError(f"row {row_number}: {field} must be a canonical integer")
    return value


def read_source_data(
    path: str | Path,
    expected_arrays_by_sample: Sequence[SampleArrays],
) -> SourceDataBundle:
    arrays = _require_frozen_arrays(expected_arrays_by_sample)
    main_raw = np.empty_like(arrays[0].native_raw_amplitude)
    raw_seen = np.zeros(main_raw.shape, dtype=bool)
    processed_by_sample = {
        item.spec.sample_id: np.empty_like(item.stages.processed_zscore)
        for item in arrays
    }
    processed_seen = {
        sample_id: np.zeros(values.shape, dtype=bool)
        for sample_id, values in processed_by_sample.items()
    }
    packet_indices_by_sample = {
        item.spec.sample_id: np.asarray(item.stages.packet_indices).copy()
        for item in arrays
    }
    subcarrier_indices_by_sample = {
        item.spec.sample_id: np.asarray(item.stages.subcarrier_indices).copy()
        for item in arrays
    }
    expected_by_id = {item.spec.sample_id: item for item in arrays}
    row_count = 0

    try:
        source_file = gzip.open(path, "rt", encoding="utf-8", newline="")
        with source_file as handle:
            reader = csv.DictReader(handle)
            if tuple(reader.fieldnames or ()) != SOURCE_DATA_FIELDS:
                raise ValueError("source data fields do not match the frozen schema")
            for row_number, row in enumerate(reader, start=2):
                if None in row:
                    raise ValueError(f"row {row_number}: extra columns")
                sample_id = row["sample_id"]
                if sample_id not in expected_by_id:
                    raise ValueError(f"row {row_number}: unknown sample_id {sample_id}")
                item = expected_by_id[sample_id]
                if row["sample_role"] != item.spec.role:
                    raise ValueError(f"row {row_number}: sample role mismatch")
                rx_index = _canonical_nonnegative_integer(
                    row["rx_index"], field="rx_index", row_number=row_number
                )
                output_packet_index = _canonical_nonnegative_integer(
                    row["output_packet_index"],
                    field="output_packet_index",
                    row_number=row_number,
                )
                source_packet_index = _canonical_nonnegative_integer(
                    row["source_packet_index"],
                    field="source_packet_index",
                    row_number=row_number,
                )
                output_subcarrier_index = _canonical_nonnegative_integer(
                    row["output_subcarrier_index"],
                    field="output_subcarrier_index",
                    row_number=row_number,
                )
                source_subcarrier_index = _canonical_nonnegative_integer(
                    row["source_subcarrier_index"],
                    field="source_subcarrier_index",
                    row_number=row_number,
                )
                try:
                    value = np.float32(row["value"])
                except (TypeError, ValueError, OverflowError) as error:
                    raise ValueError(f"row {row_number}: invalid value") from error
                if not np.isfinite(value):
                    raise ValueError(f"row {row_number}: invalid value")
                rx = rx_index - 1

                representation = row["representation"]
                if representation == "raw_amplitude_native":
                    if item is not arrays[0]:
                        raise ValueError(
                            f"row {row_number}: native raw is restricted to the main sample"
                        )
                    if (
                        rx not in range(main_raw.shape[0])
                        or output_packet_index not in range(main_raw.shape[1])
                        or output_subcarrier_index not in range(main_raw.shape[2])
                        or source_packet_index != output_packet_index
                        or source_subcarrier_index != output_subcarrier_index
                    ):
                        raise ValueError(f"row {row_number}: raw coordinate mismatch")
                    coordinate = (rx, output_packet_index, output_subcarrier_index)
                    if raw_seen[coordinate]:
                        raise ValueError(f"row {row_number}: duplicate raw coordinate")
                    expected_value = arrays[0].native_raw_amplitude[coordinate]
                    if value != expected_value:
                        raise ValueError(f"row {row_number}: value mismatch")
                    main_raw[coordinate] = value
                    raw_seen[coordinate] = True
                elif representation == "processed_zscore":
                    output = processed_by_sample[sample_id]
                    if (
                        rx not in range(output.shape[0])
                        or output_packet_index not in range(output.shape[1])
                        or output_subcarrier_index not in range(output.shape[2])
                    ):
                        raise ValueError(
                            f"row {row_number}: processed coordinate mismatch"
                        )
                    if (
                        source_packet_index
                        != int(item.stages.packet_indices[output_packet_index])
                        or source_subcarrier_index
                        != int(
                            item.stages.subcarrier_indices[output_subcarrier_index]
                        )
                    ):
                        raise ValueError(
                            f"row {row_number}: processed source coordinate mismatch"
                        )
                    coordinate = (rx, output_packet_index, output_subcarrier_index)
                    if processed_seen[sample_id][coordinate]:
                        raise ValueError(
                            f"row {row_number}: duplicate processed coordinate"
                        )
                    expected_value = item.stages.processed_zscore[coordinate]
                    if value != expected_value:
                        raise ValueError(f"row {row_number}: value mismatch")
                    output[coordinate] = value
                    processed_seen[sample_id][coordinate] = True
                else:
                    raise ValueError(
                        f"row {row_number}: unknown representation {representation}"
                    )
                row_count += 1
    except (OSError, UnicodeError) as error:
        raise ValueError(f"cannot read source data: {error}") from error

    if not raw_seen.all() or any(
        not seen.all() for seen in processed_seen.values()
    ):
        raise ValueError("source data has missing rows")

    return SourceDataBundle(
        native_raw_amplitude=main_raw,
        processed_by_sample=processed_by_sample,
        packet_indices_by_sample=packet_indices_by_sample,
        subcarrier_indices_by_sample=subcarrier_indices_by_sample,
        row_count=row_count,
    )


def write_source_schema(path: str | Path) -> None:
    payload = {
        "schema_version": 1,
        "fields": list(SOURCE_DATA_FIELDS),
        "rx_index": "one-based",
        "output_indices": "zero-based",
        "source_indices": "zero-based",
        "representations": {
            "raw_amplitude_native": (
                "abs(CSI), native packet and subcarrier resolution; main sample only"
            ),
            "processed_zscore": (
                "exact float32 output of the authoritative amplitude preprocessing"
            ),
        },
        "value": "finite float32 serialized with nine significant decimal digits",
    }
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
        newline="\n",
    )


def _validate_plot_inputs(
    source_data: SourceDataBundle,
    scales: ScaleDiagnostics,
) -> None:
    raw = np.asarray(source_data.native_raw_amplitude)
    if raw.ndim != 3 or raw.shape[0] != 2 or not np.isfinite(raw).all():
        raise ValueError("figure native raw data must contain two finite RX arrays")
    expected_ids = [item.sample_id for item in FROZEN_SAMPLES]
    if list(source_data.processed_by_sample) != expected_ids:
        raise ValueError("figure processed data must follow the frozen sample order")
    for sample_id in expected_ids:
        processed = np.asarray(source_data.processed_by_sample[sample_id])
        if (
            processed.ndim != 3
            or processed.shape[0] != 2
            or not np.isfinite(processed).all()
        ):
            raise ValueError(
                f"sample {sample_id}: figure processed data must contain two finite RX arrays"
            )
        if len(source_data.packet_indices_by_sample[sample_id]) != processed.shape[1]:
            raise ValueError(f"sample {sample_id}: packet coordinate length mismatch")
        if (
            len(source_data.subcarrier_indices_by_sample[sample_id])
            != processed.shape[2]
        ):
            raise ValueError(
                f"sample {sample_id}: subcarrier coordinate length mismatch"
            )
    if (
        not np.isfinite(scales.raw_vmax)
        or not np.isfinite(scales.processed_limit)
        or scales.raw_vmin != 0.0
        or scales.raw_vmax <= 0.0
        or scales.processed_limit <= 0.0
    ):
        raise ValueError("figure scale diagnostics are invalid")


def build_figure(
    source_data: SourceDataBundle,
    scales: ScaleDiagnostics,
):
    import matplotlib

    matplotlib.use("Agg", force=True)
    import matplotlib.pyplot as plt

    _validate_plot_inputs(source_data, scales)
    rc_params = {
        "font.family": "sans-serif",
        "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans"],
        "font.size": 7.5,
        "axes.titlesize": 8.7,
        "axes.labelsize": 8.5,
        "xtick.labelsize": 7.2,
        "ytick.labelsize": 7.2,
        "pdf.fonttype": 42,
        "savefig.dpi": 300,
    }
    with matplotlib.rc_context(rc_params):
        figure = plt.figure(figsize=(10.5, 6.0))
        grid = figure.add_gridspec(
            3,
            6,
            left=0.075,
            right=0.985,
            bottom=0.105,
            top=0.865,
            wspace=0.10,
            hspace=0.10,
            width_ratios=(1.0, 1.0, 0.16, 1.0, 1.0, 1.0),
            height_ratios=(1.0, 1.0, 0.045),
        )
        heatmap_columns = (0, 1, 3, 4, 5)
        axes = [
            figure.add_subplot(grid[row, grid_column])
            for row in range(2)
            for grid_column in heatmap_columns
        ]
        raw_colorbar_axis = figure.add_subplot(grid[2, 0])
        processed_colorbar_axis = figure.add_subplot(grid[2, 1:6])

        raw = source_data.native_raw_amplitude
        raw_packet_edges = np.arange(raw.shape[1] + 1, dtype=np.float64) - 0.5
        raw_subcarrier_edges = (
            np.arange(raw.shape[2] + 1, dtype=np.float64) - 0.5
        )
        titles = (
            "Raw CSI amplitude",
            "2D CNN input\nrepresentation",
            "P04 | walk\nE1 | face_rx",
            "P01 | lie_down\nE2 | side_link",
            "P08 | wash_hands\nE3 | cross_link",
        )
        raw_mesh = None
        processed_mesh = None
        for rx_index in range(2):
            row_axes = axes[rx_index * 5 : (rx_index + 1) * 5]
            raw_mesh = row_axes[0].pcolormesh(
                raw_subcarrier_edges,
                raw_packet_edges,
                raw[rx_index],
                cmap="cividis",
                vmin=scales.raw_vmin,
                vmax=scales.raw_vmax,
                shading="flat",
                antialiased=False,
                rasterized=True,
            )
            for column_index, spec in enumerate(FROZEN_SAMPLES, start=1):
                processed = source_data.processed_by_sample[spec.sample_id]
                packet_edges = _centers_to_edges(
                    source_data.packet_indices_by_sample[spec.sample_id]
                )
                subcarrier_edges = _centers_to_edges(
                    source_data.subcarrier_indices_by_sample[spec.sample_id]
                )
                processed_mesh = row_axes[column_index].pcolormesh(
                    subcarrier_edges,
                    packet_edges,
                    processed[rx_index],
                    cmap="RdBu_r",
                    vmin=-scales.processed_limit,
                    vmax=scales.processed_limit,
                    shading="flat",
                    antialiased=False,
                    rasterized=True,
                )
            for column_index, axis in enumerate(row_axes):
                axis.set_aspect("auto")
                axis.tick_params(
                    axis="x",
                    which="both",
                    labelbottom=rx_index == 1,
                    length=2.5,
                    pad=1.5,
                )
                axis.tick_params(
                    axis="y",
                    which="both",
                    labelleft=column_index == 0,
                    length=2.5,
                    pad=1.5,
                )
                for spine in axis.spines.values():
                    spine.set_linewidth(0.45)
                if rx_index == 0:
                    axis.set_title(titles[column_index], pad=5.0)
                axis.set_ylim(axis.get_ylim()[::-1])

        assert raw_mesh is not None and processed_mesh is not None
        raw_colorbar = figure.colorbar(
            raw_mesh,
            cax=raw_colorbar_axis,
            orientation="horizontal",
        )
        raw_colorbar.set_label("Raw amplitude |H| (a.u.)", labelpad=2.0)
        raw_colorbar.ax.tick_params(labelsize=7.0, length=2.5, pad=1.5)
        processed_colorbar = figure.colorbar(
            processed_mesh,
            cax=processed_colorbar_axis,
            orientation="horizontal",
        )
        processed_colorbar.set_label(
            "Standardized log-amplitude (z)",
            labelpad=2.0,
        )
        processed_colorbar.ax.tick_params(labelsize=7.0, length=2.5, pad=1.5)

        figure.text(
            0.255,
            0.955,
            "A  Preprocessing demonstration",
            ha="center",
            va="top",
            fontsize=9.5,
            fontweight="bold",
        )
        figure.text(
            0.735,
            0.955,
            "B  Representative sample diversity",
            ha="center",
            va="top",
            fontsize=9.5,
            fontweight="bold",
        )
        figure.text(
            0.047,
            0.665,
            "RX1",
            ha="center",
            va="center",
            fontsize=8.5,
            fontweight="bold",
        )
        figure.text(
            0.047,
            0.325,
            "RX2",
            ha="center",
            va="center",
            fontsize=8.5,
            fontweight="bold",
        )
        figure.supxlabel("Source subcarrier index", x=0.53, y=0.012, fontsize=8.5)
        figure.supylabel("Source packet index", x=0.006, y=0.50, fontsize=8.5)
    return figure


def resolve_publication_font() -> tuple[str, str]:
    from matplotlib import font_manager

    for family in PUBLICATION_FONT_CANDIDATES:
        try:
            font_path = font_manager.findfont(family, fallback_to_default=False)
        except ValueError:
            continue
        return family, str(Path(font_path).resolve())
    raise RuntimeError("no stable sans-serif publication font is available")


def build_publication_figure_v2(
    source_data: SourceDataBundle,
    scales: ScaleDiagnostics,
    *,
    edge_policy: str = "extrapolated",
):
    import matplotlib

    matplotlib.use("Agg", force=True)
    import matplotlib.pyplot as plt

    _validate_plot_inputs(source_data, scales)
    font_family, font_path = resolve_publication_font()
    rc_params = {
        "font.family": font_family,
        "font.size": 6.4,
        "axes.titlesize": 7.2,
        "axes.labelsize": 6.8,
        "xtick.labelsize": 6.2,
        "ytick.labelsize": 6.2,
        "axes.linewidth": 0.5,
        "pdf.fonttype": 42,
        "savefig.dpi": 300,
    }
    with matplotlib.rc_context(rc_params):
        figure = plt.figure(
            figsize=(V2_FIGURE_WIDTH_MM / 25.4, V2_FIGURE_HEIGHT_MM / 25.4)
        )
        figure._axhome_publication_font = {  # type: ignore[attr-defined]
            "family": font_family,
            "path": font_path,
        }

        left = 0.105
        right = 0.985
        intra_group_gap = 0.0125
        inter_group_gap = 1.8 * intra_group_gap
        heatmap_width = (
            right - left - 3.0 * intra_group_gap - inter_group_gap
        ) / 5.0
        x_positions = [left]
        for column_index in range(1, 5):
            preceding_gap = (
                inter_group_gap if column_index == 2 else intra_group_gap
            )
            x_positions.append(
                x_positions[-1] + heatmap_width + preceding_gap
            )
        heatmap_height = 0.275
        row_positions = (0.535, 0.205)
        axes = [
            figure.add_axes(
                [x_positions[column], row_positions[row], heatmap_width, heatmap_height]
            )
            for row in range(2)
            for column in range(5)
        ]
        raw_colorbar_axis = figure.add_axes(
            [x_positions[0], 0.092, heatmap_width, 0.024]
        )
        processed_colorbar_axis = figure.add_axes(
            [
                x_positions[1],
                0.092,
                x_positions[4] + heatmap_width - x_positions[1],
                0.024,
            ]
        )

        raw = source_data.native_raw_amplitude
        raw_packet_edges = np.arange(raw.shape[1] + 1, dtype=np.float64) - 0.5
        raw_subcarrier_edges = np.arange(raw.shape[2] + 1, dtype=np.float64) - 0.5
        raw_mesh = None
        processed_mesh = None
        for rx_index in range(2):
            row_axes = axes[rx_index * 5 : (rx_index + 1) * 5]
            raw_mesh = row_axes[0].pcolormesh(
                raw_subcarrier_edges,
                raw_packet_edges,
                raw[rx_index],
                cmap="cividis",
                vmin=scales.raw_vmin,
                vmax=scales.raw_vmax,
                shading="flat",
                antialiased=False,
                rasterized=True,
            )
            for column_index, spec in enumerate(FROZEN_SAMPLES, start=1):
                processed = source_data.processed_by_sample[spec.sample_id]
                packet_indices = source_data.packet_indices_by_sample[
                    spec.sample_id
                ]
                subcarrier_indices = source_data.subcarrier_indices_by_sample[
                    spec.sample_id
                ]
                if edge_policy == "bounded":
                    packet_edges = _bounded_centers_to_edges(
                        packet_indices, int(packet_indices[-1]) + 1
                    )
                    subcarrier_edges = _bounded_centers_to_edges(
                        subcarrier_indices, int(subcarrier_indices[-1]) + 1
                    )
                elif edge_policy == "extrapolated":
                    packet_edges = _centers_to_edges(packet_indices)
                    subcarrier_edges = _centers_to_edges(subcarrier_indices)
                else:
                    raise ValueError(f"unknown edge policy {edge_policy}")
                processed_mesh = row_axes[column_index].pcolormesh(
                    subcarrier_edges,
                    packet_edges,
                    processed[rx_index],
                    cmap="RdBu_r",
                    vmin=-scales.processed_limit,
                    vmax=scales.processed_limit,
                    shading="flat",
                    antialiased=False,
                    rasterized=True,
                )
            for column_index, axis in enumerate(row_axes):
                axis.set_aspect("auto")
                axis.tick_params(
                    axis="x",
                    which="both",
                    labelbottom=rx_index == 1,
                    length=2.0,
                    width=0.5,
                    pad=1.2,
                )
                axis.tick_params(
                    axis="y",
                    which="both",
                    labelleft=column_index == 0,
                    length=2.0,
                    width=0.5,
                    pad=1.2,
                )
                for spine in axis.spines.values():
                    spine.set_linewidth(0.5)
                axis.invert_yaxis()

        assert raw_mesh is not None and processed_mesh is not None
        raw_colorbar = figure.colorbar(
            raw_mesh,
            cax=raw_colorbar_axis,
            orientation="horizontal",
        )
        raw_colorbar.set_label("CSI amplitude, |H| (a.u.)", labelpad=1.6)
        raw_colorbar.ax.tick_params(
            labelsize=6.0,
            length=2.0,
            width=0.5,
            pad=1.0,
        )
        raw_colorbar.outline.set_linewidth(0.5)
        processed_colorbar = figure.colorbar(
            processed_mesh,
            cax=processed_colorbar_axis,
            orientation="horizontal",
        )
        processed_colorbar.set_label(
            "Standardized log-amplitude, z",
            labelpad=1.6,
        )
        processed_colorbar.ax.tick_params(
            labelsize=6.0,
            length=2.0,
            width=0.5,
            pad=1.0,
        )
        processed_colorbar.outline.set_linewidth(0.5)

        figure.text(
            x_positions[0],
            0.968,
            "a  CSI preprocessing",
            ha="left",
            va="top",
            fontsize=8.2,
            fontweight="bold",
        )
        figure.text(
            x_positions[2],
            0.968,
            "b  Representative samples",
            ha="left",
            va="top",
            fontsize=8.2,
            fontweight="bold",
        )
        figure.text(
            x_positions[0] + heatmap_width / 2.0,
            0.850,
            "Raw CSI amplitude",
            ha="center",
            va="center",
            fontsize=7.2,
        )
        figure.text(
            x_positions[1] + heatmap_width / 2.0,
            0.850,
            "2D CNN input representation\n(exact baseline match confirmed)",
            ha="center",
            va="center",
            multialignment="center",
            fontsize=6.6,
            linespacing=1.15,
        )
        representative_titles = (
            ("Walk (P04)", "E1 · face_rx"),
            ("Lie down (P01)", "E2 · side_link"),
            ("Wash hands (P08)", "E3 · cross_link"),
        )
        for column_index, (action, metadata) in enumerate(
            representative_titles, start=2
        ):
            center = x_positions[column_index] + heatmap_width / 2.0
            figure.text(
                center,
                0.862,
                action,
                ha="center",
                va="center",
                fontsize=7.2,
            )
            figure.text(
                center,
                0.827,
                metadata,
                ha="center",
                va="center",
                fontsize=6.2,
            )
        for row_index, label in enumerate(("RX1", "RX2")):
            figure.text(
                left - 0.040,
                row_positions[row_index] + heatmap_height / 2.0,
                label,
                ha="center",
                va="center",
                fontsize=7.2,
                fontweight="bold",
            )
        figure.supxlabel(
            "Source subcarrier index",
            x=(left + right) / 2.0,
            y=0.006,
            fontsize=6.8,
        )
        figure.supylabel(
            "Source packet index",
            x=0.010,
            y=0.50,
            fontsize=6.8,
        )
    return figure


def build_publication_figure_v3(
    source_data: SourceDataBundle,
    scales: ScaleDiagnostics,
    *,
    edge_policy: str = "extrapolated",
):
    figure = build_publication_figure_v2(
        source_data, scales, edge_policy=edge_policy
    )
    figure.set_size_inches(
        V3_FIGURE_WIDTH_MM / 25.4,
        V3_FIGURE_HEIGHT_MM / 25.4,
        forward=True,
    )
    figure._axhome_mathtext_fontset = "stixsans"  # type: ignore[attr-defined]

    left = 0.105
    right = 0.985
    panel_a_gap = 0.0115
    inter_group_gap = 0.0260
    panel_b_gap = 0.0100
    heatmap_width = (
        right
        - left
        - panel_a_gap
        - inter_group_gap
        - 2.0 * panel_b_gap
    ) / 5.0
    x_positions = [left]
    for gap in (panel_a_gap, inter_group_gap, panel_b_gap, panel_b_gap):
        x_positions.append(x_positions[-1] + heatmap_width + gap)
    heatmap_height = 0.260
    row_positions = (0.545, 0.245)

    heatmaps = figure.axes[:10]
    for row_index in range(2):
        for column_index in range(5):
            heatmaps[row_index * 5 + column_index].set_position(
                [
                    x_positions[column_index],
                    row_positions[row_index],
                    heatmap_width,
                    heatmap_height,
                ]
            )
    figure.axes[10].set_position(
        [x_positions[0], 0.065, heatmap_width, 0.022]
    )
    figure.axes[11].set_position(
        [
            x_positions[1],
            0.065,
            x_positions[4] + heatmap_width - x_positions[1],
            0.022,
        ]
    )
    figure.axes[10].set_xlabel("CSI amplitude, $|H|$ (a.u.)", labelpad=1.5)
    figure.axes[11].set_xlabel(
        "Standardized log-amplitude, $z$",
        labelpad=1.5,
    )
    figure.axes[10].xaxis.label.set_math_fontfamily("stixsans")
    figure.axes[11].xaxis.label.set_math_fontfamily("stixsans")

    text_by_value = {text.get_text(): text for text in figure.texts}
    old_processed_title = (
        "2D CNN input representation\n(exact baseline match confirmed)"
    )
    processed_title = text_by_value.pop(old_processed_title)
    processed_title.set_text("2D CNN input\nrepresentation")
    text_by_value["2D CNN input\nrepresentation"] = processed_title

    position_updates = {
        "a  CSI preprocessing": (x_positions[0], 0.966),
        "b  Representative samples": (x_positions[2], 0.966),
        "Raw CSI amplitude": (
            x_positions[0] + heatmap_width / 2.0,
            0.845,
        ),
        "2D CNN input\nrepresentation": (
            x_positions[1] + heatmap_width / 2.0,
            0.845,
        ),
        "Walk (P04)": (x_positions[2] + heatmap_width / 2.0, 0.856),
        "E1 · face_rx": (x_positions[2] + heatmap_width / 2.0, 0.822),
        "Lie down (P01)": (x_positions[3] + heatmap_width / 2.0, 0.856),
        "E2 · side_link": (x_positions[3] + heatmap_width / 2.0, 0.822),
        "Wash hands (P08)": (x_positions[4] + heatmap_width / 2.0, 0.856),
        "E3 · cross_link": (x_positions[4] + heatmap_width / 2.0, 0.822),
        "RX1": (left - 0.036, row_positions[0] + heatmap_height / 2.0),
        "RX2": (left - 0.036, row_positions[1] + heatmap_height / 2.0),
        "Source subcarrier index": ((left + right) / 2.0, 0.153),
    }
    for value, position in position_updates.items():
        text_by_value[value].set_position(position)

    text_by_value["a  CSI preprocessing"].set_fontsize(8.0)
    text_by_value["b  Representative samples"].set_fontsize(8.0)
    text_by_value["2D CNN input\nrepresentation"].set_fontsize(7.1)
    text_by_value["Source subcarrier index"].set_fontsize(6.8)
    return figure


def _bounded_centers_to_edges(
    centers: np.ndarray,
    source_length: int,
) -> np.ndarray:
    """Midpoint interior edges with endpoints clamped to the source domain.

    Interior edges stay at the midpoints between adjacent selected source
    centers (exactly as `_centers_to_edges`); only the two extrapolated
    endpoint edges are replaced by the true source boundaries ``-0.5`` and
    ``source_length - 0.5``.
    """
    if not isinstance(source_length, (int, np.integer)) or source_length < 2:
        raise ValueError("source_length must be an integer of at least two")
    edges = _centers_to_edges(np.asarray(centers)).copy()
    edges[0] = -0.5
    edges[-1] = float(source_length) - 0.5
    if np.any(edges[1:] <= edges[:-1]):
        raise ValueError("bounded source edges must remain strictly increasing")
    return edges


def build_publication_figure_v4(
    source_data: SourceDataBundle,
    scales: ScaleDiagnostics,
):
    """v3 typography/layout with authoritative source-domain axes limits.

    The processed meshes keep their real selected source coordinates; only
    the displayed axes limits/ticks are pinned to the true source domains so
    that one source coordinate maps to the same normalized axes fraction in
    every heatmap column.
    """
    figure = build_publication_figure_v3(
        source_data, scales, edge_policy="bounded"
    )
    for index, axis in enumerate(figure.axes[:10]):
        column_index = index % 5
        axis.set_xlim(*V4_X_LIMITS)
        axis.set_xticks(list(V4_X_TICKS))
        if column_index <= 1:
            axis.set_ylim(*V4_MAIN_Y_LIMITS)
        else:
            axis.set_ylim(*V4_REPRESENTATIVE_Y_LIMITS)
        axis.set_yticks(list(V4_Y_TICKS))
    return figure


def build_figure6(
    source_data: SourceDataBundle,
    scales: ScaleDiagnostics,
):
    """Build the main-text Figure 6 while retaining the v4 data mapping.

    This is deliberately a layout/label revision of the provenance-preserving
    v4 figure.  The frozen samples, extracted arrays, percentile scales, and
    source-coordinate meshes are inherited unchanged.
    """
    from matplotlib.lines import Line2D

    figure = build_publication_figure_v4(source_data, scales)
    figure.set_size_inches(FIGURE6_WIDTH_IN, FIGURE6_HEIGHT_IN, forward=True)

    left = 0.098
    right = 0.985
    panel_a_gap = 0.009
    inter_group_gap = 0.020
    panel_b_gap = 0.009
    heatmap_width = (
        right
        - left
        - panel_a_gap
        - inter_group_gap
        - 2.0 * panel_b_gap
    ) / 5.0
    x_positions = [left]
    for gap in (panel_a_gap, inter_group_gap, panel_b_gap, panel_b_gap):
        x_positions.append(x_positions[-1] + heatmap_width + gap)
    heatmap_height = 0.255
    row_positions = (0.527, 0.229)

    heatmaps = figure.axes[:10]
    for row_index in range(2):
        for column_index in range(5):
            axis = heatmaps[row_index * 5 + column_index]
            axis.set_position(
                [
                    x_positions[column_index],
                    row_positions[row_index],
                    heatmap_width,
                    heatmap_height,
                ]
            )
            axis.tick_params(
                axis="both",
                labelsize=5.4,
                length=1.8,
                width=0.45,
                pad=1.0,
            )
            axis.spines["top"].set_visible(False)
            axis.spines["right"].set_visible(False)
            axis.spines["left"].set_linewidth(0.45)
            axis.spines["bottom"].set_linewidth(0.45)

    colorbar_y = 0.066
    colorbar_height = 0.019
    raw_colorbar_x = x_positions[0]
    raw_colorbar_width = heatmap_width
    processed_colorbar_x = x_positions[1]
    processed_colorbar_width = right - processed_colorbar_x
    figure.axes[10].set_position(
        [
            raw_colorbar_x,
            colorbar_y,
            raw_colorbar_width,
            colorbar_height,
        ]
    )
    figure.axes[11].set_position(
        [
            processed_colorbar_x,
            colorbar_y,
            processed_colorbar_width,
            colorbar_height,
        ]
    )
    for colorbar_axis in figure.axes[10:12]:
        lower, upper = colorbar_axis.get_xlim()
        colorbar_axis.set_xticks(
            [
                tick
                for tick in colorbar_axis.get_xticks()
                if lower <= tick <= upper
            ]
        )
        colorbar_axis.tick_params(
            axis="x",
            labelsize=5.2,
            length=1.8,
            width=0.45,
            pad=0.8,
        )
        colorbar_axis.xaxis.label.set_fontsize(5.8)
        colorbar_axis.xaxis.labelpad = 1.1

    text_by_value = {text.get_text(): text for text in figure.texts}
    raw_title = text_by_value.pop("Raw CSI amplitude")
    raw_title.set_text("Raw CSI")
    processed_title = text_by_value.pop("2D CNN input\nrepresentation")
    processed_title.set_text("Model input")

    publication_font = figure._axhome_publication_font[  # type: ignore[attr-defined]
        "family"
    ]
    figure.text(
        x_positions[0] + heatmap_width / 2.0,
        0.813,
        "amplitude",
        ha="center",
        va="center",
        fontsize=5.9,
        fontfamily=publication_font,
    )
    figure.text(
        x_positions[1] + heatmap_width / 2.0,
        0.813,
        "representation",
        ha="center",
        va="center",
        fontsize=5.9,
        fontfamily=publication_font,
    )

    text_by_value = {text.get_text(): text for text in figure.texts}
    position_updates = {
        "a  CSI preprocessing": (x_positions[0], 0.962),
        "b  Representative samples": (x_positions[2], 0.962),
        "Raw CSI": (x_positions[0] + heatmap_width / 2.0, 0.844),
        "amplitude": (x_positions[0] + heatmap_width / 2.0, 0.813),
        "Model input": (x_positions[1] + heatmap_width / 2.0, 0.844),
        "representation": (x_positions[1] + heatmap_width / 2.0, 0.813),
        "Walk (P04)": (x_positions[2] + heatmap_width / 2.0, 0.844),
        "E1 · face_rx": (x_positions[2] + heatmap_width / 2.0, 0.813),
        "Lie down (P01)": (x_positions[3] + heatmap_width / 2.0, 0.844),
        "E2 · side_link": (x_positions[3] + heatmap_width / 2.0, 0.813),
        "Wash hands (P08)": (x_positions[4] + heatmap_width / 2.0, 0.844),
        "E3 · cross_link": (x_positions[4] + heatmap_width / 2.0, 0.813),
        "RX1": (left - 0.035, row_positions[0] + heatmap_height / 2.0),
        "RX2": (left - 0.035, row_positions[1] + heatmap_height / 2.0),
        "Source subcarrier index": ((left + right) / 2.0, 0.145),
        "Source packet index": (0.012, 0.505),
    }
    for value, position in position_updates.items():
        text_by_value[value].set_position(position)

    for value in ("a  CSI preprocessing", "b  Representative samples"):
        text_by_value[value].set_fontsize(7.3)
        text_by_value[value].set_fontweight("bold")
    for value in (
        "Raw CSI",
        "Model input",
        "Walk (P04)",
        "Lie down (P01)",
        "Wash hands (P08)",
    ):
        text_by_value[value].set_fontsize(6.4)
        text_by_value[value].set_color("#202020")
    for value in ("amplitude", "representation"):
        text_by_value[value].set_fontsize(5.9)
        text_by_value[value].set_color("#202020")
    for value in ("E1 · face_rx", "E2 · side_link", "E3 · cross_link"):
        text_by_value[value].set_fontsize(5.5)
        text_by_value[value].set_color("#666666")
    for value in ("RX1", "RX2"):
        text_by_value[value].set_fontsize(6.2)
    text_by_value["Source subcarrier index"].set_fontsize(5.9)
    text_by_value["Source packet index"].set_fontsize(5.9)

    separator_x = (
        x_positions[1] + heatmap_width + x_positions[2]
    ) / 2.0
    separator_y = (0.188, 0.907)
    separator = Line2D(
        [separator_x, separator_x],
        list(separator_y),
        transform=figure.transFigure,
        color="#A6A6A6",
        linewidth=0.55,
        solid_capstyle="butt",
        clip_on=False,
        zorder=10,
    )
    separator.set_gid("figure6-panel-separator")
    figure.add_artist(separator)
    figure._axhome_figure6_layout = {  # type: ignore[attr-defined]
        "left": left,
        "right": right,
        "heatmap_width": heatmap_width,
        "heatmap_height": heatmap_height,
        "x_positions": tuple(x_positions),
        "row_positions": row_positions,
        "raw_colorbar_scope": (
            raw_colorbar_x,
            raw_colorbar_x + raw_colorbar_width,
        ),
        "processed_colorbar_scope": (
            processed_colorbar_x,
            processed_colorbar_x + processed_colorbar_width,
        ),
        "colorbar_y": colorbar_y,
        "colorbar_height": colorbar_height,
        "separator_x": separator_x,
        "separator_y": separator_y,
    }
    return figure


def _v4_fraction_key(coordinate: float) -> str:
    return str(int(coordinate))


def validate_v4_coordinate_alignment(
    figure,
    source_data: SourceDataBundle,
    *,
    tolerance: float = 1e-12,
) -> dict[str, object]:
    """Record and enforce per-axis source-coordinate alignment diagnostics.

    Fractions use the display-bottom origin convention:
    ``fraction = (display_coordinate - bbox_min) / bbox_extent`` computed
    through the real ``transData`` transform after ``canvas.draw()``. Any
    violated invariant raises ``ValueError``; a returned summary always has
    ``status == "pass"``.
    """
    if not math.isfinite(tolerance) or tolerance <= 0.0:
        raise ValueError("tolerance must be a positive finite value")
    expected_ids = [item.sample_id for item in FROZEN_SAMPLES]
    if list(source_data.processed_by_sample) != expected_ids:
        raise ValueError("v4 validation requires the frozen sample order")
    heatmaps = list(figure.axes[:10])
    if len(heatmaps) != len(V4_AXIS_IDS):
        raise ValueError("v4 figure must contain ten heatmap axes")
    figure.canvas.draw()

    def require_finite(label: str, values: Sequence[float]) -> list[float]:
        result = [float(value) for value in values]
        if not all(math.isfinite(value) for value in result):
            raise ValueError(f"{label}: non-finite diagnostic value")
        return result

    def require_close(
        label: str, actual: Sequence[float], expected: Sequence[float]
    ) -> None:
        if len(actual) != len(expected) or any(
            abs(float(a) - float(e)) > tolerance
            for a, e in zip(actual, expected, strict=True)
        ):
            raise ValueError(
                f"{label}: expected {list(expected)}, got {list(actual)}"
            )

    records = []
    for index, (axis_id, axis) in enumerate(
        zip(V4_AXIS_IDS, heatmaps, strict=True)
    ):
        column_index = index % 5
        if column_index == 0:
            matrix_shape = [
                int(value)
                for value in source_data.native_raw_amplitude.shape[1:]
            ]
        else:
            spec = FROZEN_SAMPLES[column_index - 1]
            matrix_shape = [
                int(value)
                for value in source_data.processed_by_sample[
                    spec.sample_id
                ].shape[1:]
            ]
        bounds = require_finite(
            f"{axis_id} bounds", axis.get_position().bounds
        )
        xlim = require_finite(f"{axis_id} xlim", axis.get_xlim())
        ylim = require_finite(f"{axis_id} ylim", axis.get_ylim())
        xticks = require_finite(f"{axis_id} xticks", axis.get_xticks())
        yticks = require_finite(f"{axis_id} yticks", axis.get_yticks())
        aspect = axis.get_aspect()
        if aspect != "auto":
            raise ValueError(f"{axis_id}: aspect must be auto, got {aspect}")
        require_close(f"{axis_id} xlim", xlim, V4_X_LIMITS)
        require_close(f"{axis_id} xticks", xticks, V4_X_TICKS)
        expected_ylim = (
            V4_MAIN_Y_LIMITS
            if column_index <= 1
            else V4_REPRESENTATIVE_Y_LIMITS
        )
        require_close(f"{axis_id} ylim", ylim, expected_ylim)
        require_close(f"{axis_id} yticks", yticks, V4_Y_TICKS)

        bbox = axis.get_window_extent()
        x_fractions = {}
        for coordinate in V4_X_FRACTION_COORDINATES:
            x_display = axis.transData.transform((coordinate, 0.0))[0]
            x_fractions[_v4_fraction_key(coordinate)] = float(
                (x_display - bbox.x0) / bbox.width
            )
        y_fractions = {}
        for coordinate in V4_Y_FRACTION_COORDINATES:
            y_display = axis.transData.transform((0.0, coordinate))[1]
            y_fractions[_v4_fraction_key(coordinate)] = float(
                (y_display - bbox.y0) / bbox.height
            )
        require_finite(f"{axis_id} x fractions", x_fractions.values())
        require_finite(f"{axis_id} y fractions", y_fractions.values())
        records.append(
            {
                "axis_id": axis_id,
                "matrix_shape": matrix_shape,
                "bounds": bounds,
                "xlim": xlim,
                "ylim": ylim,
                "xticks": xticks,
                "yticks": yticks,
                "aspect": aspect,
                "x_fractions": x_fractions,
                "y_fractions": y_fractions,
            }
        )

    def group_payload(
        axis_indices: Sequence[int],
        fraction_field: str,
        coordinates: Sequence[float],
    ) -> dict[str, object]:
        by_coordinate = {}
        for coordinate in coordinates:
            key = _v4_fraction_key(coordinate)
            values = [
                records[axis_index][fraction_field][key]
                for axis_index in axis_indices
            ]
            by_coordinate[key] = float(max(values) - min(values))
        max_difference = float(max(by_coordinate.values()))
        return {
            "axis_ids": [V4_AXIS_IDS[axis_index] for axis_index in axis_indices],
            "coordinates": [float(coordinate) for coordinate in coordinates],
            "max_difference_by_coordinate": by_coordinate,
            "max_difference": max_difference,
        }

    groups = {
        "x_all": group_payload(
            list(range(10)), "x_fractions", V4_X_FRACTION_COORDINATES
        ),
        "y_main": group_payload(
            [0, 1, 5, 6], "y_fractions", V4_Y_FRACTION_COORDINATES
        ),
        "y_representatives": group_payload(
            [2, 3, 4, 7, 8, 9], "y_fractions", V4_Y_FRACTION_COORDINATES
        ),
    }
    for name, group in groups.items():
        if group["max_difference"] > tolerance:
            raise ValueError(
                f"{name}: normalized fraction max difference "
                f"{group['max_difference']} exceeds tolerance {tolerance}"
            )

    return {
        "schema_version": 1,
        "status": "pass",
        "tolerance": float(tolerance),
        "fraction_origin": "display-bottom",
        "source_domains": {
            "x": [V4_X_LIMITS[0], V4_X_LIMITS[1]],
            "main_y": [-0.5, float(V4_MAIN_SOURCE_PACKET_COUNT) - 0.5],
            "representative_y": [
                -0.5,
                float(V4_REPRESENTATIVE_SOURCE_PACKET_COUNT) - 0.5,
            ],
        },
        "axes": records,
        "groups": groups,
    }


def validate_figure6_layout(figure) -> dict[str, object]:
    """Fail closed on the Figure 6 editorial and layout requirements."""
    from matplotlib.text import Text

    width, height = (float(value) for value in figure.get_size_inches())
    if not math.isclose(width, FIGURE6_WIDTH_IN, abs_tol=1e-12):
        raise ValueError(f"Figure 6 width must be {FIGURE6_WIDTH_IN} in")
    if not math.isclose(height, FIGURE6_HEIGHT_IN, abs_tol=1e-12):
        raise ValueError(f"Figure 6 height must be {FIGURE6_HEIGHT_IN} in")
    if len(figure.axes) != 12:
        raise ValueError("Figure 6 must contain ten heatmaps and two colorbars")

    text_items = list(figure.texts)
    text_values = [item.get_text() for item in text_items]
    if any("2D CNN input" in value for value in text_values):
        raise ValueError("Figure 6 contains the deprecated architecture-specific label")
    required_text = {
        "a  CSI preprocessing",
        "b  Representative samples",
        "Raw CSI",
        "amplitude",
        "Model input",
        "representation",
        "Source packet index",
        "Source subcarrier index",
    }
    missing = required_text.difference(text_values)
    if missing:
        raise ValueError(f"Figure 6 is missing required text: {sorted(missing)}")

    title_rows = {
        "first": (
            "Raw CSI",
            "Model input",
            "Walk (P04)",
            "Lie down (P01)",
            "Wash hands (P08)",
        ),
        "second": (
            "amplitude",
            "representation",
            "E1 · face_rx",
            "E2 · side_link",
            "E3 · cross_link",
        ),
    }
    text_by_value = {item.get_text(): item for item in text_items}
    title_y_positions = {
        name: [float(text_by_value[value].get_position()[1]) for value in values]
        for name, values in title_rows.items()
    }
    if any(
        max(values) - min(values) > 1e-12
        for values in title_y_positions.values()
    ):
        raise ValueError("Figure 6 column-title baselines are not aligned")

    heatmaps = figure.axes[:10]
    for axis in heatmaps:
        if axis.spines["top"].get_visible() or axis.spines["right"].get_visible():
            raise ValueError("Figure 6 heatmaps must omit top and right spines")
    raw_colorbar_bounds = figure.axes[10].get_position()
    processed_colorbar_bounds = figure.axes[11].get_position()
    raw_heatmap_bounds = heatmaps[0].get_position()
    processed_heatmap_bounds = heatmaps[1].get_position()
    rightmost_heatmap_bounds = heatmaps[4].get_position()
    colorbar_widths = [
        float(raw_colorbar_bounds.width),
        float(processed_colorbar_bounds.width),
    ]
    scope_checks = {
        "raw_left": math.isclose(
            raw_colorbar_bounds.x0,
            raw_heatmap_bounds.x0,
            abs_tol=1e-12,
        ),
        "raw_right": math.isclose(
            raw_colorbar_bounds.x1,
            raw_heatmap_bounds.x1,
            abs_tol=1e-12,
        ),
        "processed_left": math.isclose(
            processed_colorbar_bounds.x0,
            processed_heatmap_bounds.x0,
            abs_tol=1e-12,
        ),
        "processed_right": math.isclose(
            processed_colorbar_bounds.x1,
            rightmost_heatmap_bounds.x1,
            abs_tol=1e-12,
        ),
    }
    if not all(scope_checks.values()):
        raise ValueError(
            "Figure 6 colorbars do not match their heatmap jurisdictions"
        )
    if not processed_colorbar_bounds.width > raw_colorbar_bounds.width:
        raise ValueError(
            "Figure 6 processed colorbar must span more columns than raw"
        )

    visible_y_tick_labels_by_axis = {}
    for axis_index, axis in enumerate(heatmaps):
        axis_id = V4_AXIS_IDS[axis_index]
        visible_labels = [
            item.get_text()
            for item in axis.get_yticklabels()
            if item.get_visible()
        ]
        visible_y_tick_labels_by_axis[axis_id] = visible_labels
        expected_labels = (
            ["0", "200", "400"] if axis_index % 5 == 0 else []
        )
        if visible_labels != expected_labels:
            raise ValueError(
                f"{axis_id} y tick labels must follow the shared-axis policy"
            )

    separator = next(
        (
            artist
            for artist in figure.artists
            if getattr(artist, "get_gid", lambda: None)()
            == "figure6-panel-separator"
        ),
        None,
    )
    if separator is None:
        raise ValueError("Figure 6 requires a panel-group separator")
    separator_y = [float(value) for value in separator.get_ydata()]
    colorbar_top = max(
        float(raw_colorbar_bounds.y1),
        float(processed_colorbar_bounds.y1),
    )
    if min(separator_y) <= colorbar_top:
        raise ValueError(
            "Figure 6 separator must stop above the shared z colorbar"
        )

    figure.canvas.draw()
    renderer = figure.canvas.get_renderer()
    figure_bbox = figure.bbox
    text_margins = []
    for item in figure.findobj(match=Text):
        if not item.get_visible() or not item.get_text():
            continue
        bbox = item.get_window_extent(renderer=renderer)
        margins = (
            float(bbox.x0 - figure_bbox.x0),
            float(figure_bbox.x1 - bbox.x1),
            float(bbox.y0 - figure_bbox.y0),
            float(figure_bbox.y1 - bbox.y1),
        )
        if min(margins) < -0.5:
            raise ValueError(
                f"Figure 6 text is clipped outside the canvas: {item.get_text()}"
            )
        text_margins.append(min(margins))

    font_family = figure._axhome_publication_font[  # type: ignore[attr-defined]
        "family"
    ]
    if font_family not in PUBLICATION_FONT_CANDIDATES:
        raise ValueError(f"unsupported Figure 6 font family: {font_family}")
    return {
        "status": "pass",
        "figure_size_inches": [width, height],
        "font_family": font_family,
        "model_input_label": "Model input representation",
        "column_title_baselines": title_y_positions,
        "colorbar_widths": colorbar_widths,
        "colorbars_equal_length": False,
        "colorbar_scope_policy": {
            "raw": "Raw CSI amplitude column only",
            "processed": (
                "Model input representation plus all three panel b columns"
            ),
            "scope_checks": scope_checks,
        },
        "source_packet_range_review": {
            "panel_a": [
                -0.5,
                float(V4_MAIN_SOURCE_PACKET_COUNT) - 0.5,
            ],
            "panel_b": [
                -0.5,
                float(V4_REPRESENTATIVE_SOURCE_PACKET_COUNT) - 0.5,
            ],
            "ranges_identical": False,
            "difference_packets": (
                V4_MAIN_SOURCE_PACKET_COUNT
                - V4_REPRESENTATIVE_SOURCE_PACKET_COUNT
            ),
            "relative_difference_percent": (
                100.0
                * (
                    V4_MAIN_SOURCE_PACKET_COUNT
                    - V4_REPRESENTATIVE_SOURCE_PACKET_COUNT
                )
                / V4_MAIN_SOURCE_PACKET_COUNT
            ),
            "conclusion": (
                "The one-packet difference is visually negligible; show "
                "source-packet tick labels only on the leftmost column"
            ),
            "figure_policy": "shared y tick labels on leftmost column only",
            "visible_y_tick_labels_by_axis": visible_y_tick_labels_by_axis,
        },
        "minimum_text_canvas_margin_pixels": float(min(text_margins)),
        "top_and_right_spines_visible": False,
        "panel_separator": True,
        "panel_separator_colorbar_policy": (
            "separator stops above the colorbar band because the processed "
            "colorbar crosses the panel boundary"
        ),
        "panel_separator_y": separator_y,
    }


def write_v4_coordinate_validation(
    path: str | Path,
    summary: dict[str, object],
) -> None:
    if summary.get("status") != "pass":
        raise ValueError("only a passing v4 validation summary may be written")
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        json.dumps(summary, sort_keys=True, indent=2, allow_nan=False) + "\n",
        encoding="utf-8",
        newline="\n",
    )


def plot_publication_figure_v4(
    source_data: SourceDataBundle,
    scales: ScaleDiagnostics,
    png_path: str | Path,
    pdf_path: str | Path,
    svg_path: str | Path,
    validation_path: str | Path,
) -> dict[str, object]:
    import matplotlib

    figure = build_publication_figure_v4(source_data, scales)
    try:
        summary = validate_v4_coordinate_alignment(figure, source_data)
        png_output = Path(png_path)
        pdf_output = Path(pdf_path)
        svg_output = Path(svg_path)
        validation_output = Path(validation_path)
        for output in (png_output, pdf_output, svg_output, validation_output):
            output.parent.mkdir(parents=True, exist_ok=True)
        figure.savefig(
            png_output,
            dpi=300,
            metadata={"Software": "AXHome-MM Scientific Data S3 v4 generator"},
        )
        fixed_pdf_date = dt.datetime(2000, 1, 1, tzinfo=dt.timezone.utc)
        with matplotlib.rc_context({"pdf.fonttype": 42}):
            figure.savefig(
                pdf_output,
                dpi=300,
                metadata={
                    "Creator": "AXHome-MM Scientific Data S3 v4 generator",
                    "Producer": "Matplotlib",
                    "CreationDate": fixed_pdf_date,
                    "ModDate": fixed_pdf_date,
                },
            )
        with matplotlib.rc_context(
            {"svg.fonttype": "none", "svg.hashsalt": "axhome-s3-v4"}
        ):
            figure.savefig(
                svg_output,
                dpi=300,
                metadata={
                    "Creator": "AXHome-MM Scientific Data S3 v4 generator",
                    "Date": "2000-01-01T00:00:00Z",
                },
            )
        write_v4_coordinate_validation(validation_output, summary)
    finally:
        close_figure(figure)
    return summary


def plot_figure6(
    source_data: SourceDataBundle,
    scales: ScaleDiagnostics,
    png_path: str | Path,
    pdf_path: str | Path,
    svg_path: str | Path,
    validation_path: str | Path,
) -> dict[str, object]:
    import matplotlib

    figure = build_figure6(source_data, scales)
    try:
        coordinate_summary = validate_v4_coordinate_alignment(
            figure, source_data
        )
        layout_summary = validate_figure6_layout(figure)
        summary = {
            "schema_version": 1,
            "status": "pass",
            "coordinate_alignment": coordinate_summary,
            "layout": layout_summary,
        }
        png_output = Path(png_path)
        pdf_output = Path(pdf_path)
        svg_output = Path(svg_path)
        validation_output = Path(validation_path)
        for output in (png_output, pdf_output, svg_output, validation_output):
            output.parent.mkdir(parents=True, exist_ok=True)
        figure.savefig(
            png_output,
            dpi=300,
            metadata={"Software": "AXHome-MM Scientific Data Figure 6 generator"},
        )
        fixed_pdf_date = dt.datetime(2000, 1, 1, tzinfo=dt.timezone.utc)
        with matplotlib.rc_context({"pdf.fonttype": 42}):
            figure.savefig(
                pdf_output,
                dpi=300,
                metadata={
                    "Creator": "AXHome-MM Scientific Data Figure 6 generator",
                    "Producer": "Matplotlib",
                    "CreationDate": fixed_pdf_date,
                    "ModDate": fixed_pdf_date,
                },
            )
        with matplotlib.rc_context(
            {"svg.fonttype": "none", "svg.hashsalt": "axhome-figure6"}
        ):
            figure.savefig(
                svg_output,
                dpi=300,
                metadata={
                    "Creator": "AXHome-MM Scientific Data Figure 6 generator",
                    "Date": "2000-01-01T00:00:00Z",
                },
            )
        write_v4_coordinate_validation(validation_output, summary)
    finally:
        close_figure(figure)
    return summary


def close_figure(figure) -> None:
    import matplotlib.pyplot as plt

    plt.close(figure)


def plot_figure(
    source_data: SourceDataBundle,
    scales: ScaleDiagnostics,
    png_path: str | Path,
    pdf_path: str | Path,
) -> None:
    import matplotlib

    figure = build_figure(source_data, scales)
    try:
        png_output = Path(png_path)
        pdf_output = Path(pdf_path)
        png_output.parent.mkdir(parents=True, exist_ok=True)
        pdf_output.parent.mkdir(parents=True, exist_ok=True)
        figure.savefig(
            png_output,
            dpi=300,
            metadata={"Software": "AXHome-MM reproducible S3 figure generator"},
        )
        fixed_pdf_date = dt.datetime(2000, 1, 1, tzinfo=dt.timezone.utc)
        with matplotlib.rc_context({"pdf.fonttype": 42}):
            figure.savefig(
                pdf_output,
                dpi=300,
                metadata={
                    "Creator": "AXHome-MM reproducible S3 figure generator",
                    "Producer": "Matplotlib",
                    "CreationDate": fixed_pdf_date,
                    "ModDate": fixed_pdf_date,
                },
            )
    finally:
        close_figure(figure)


def plot_publication_figure_v2(
    source_data: SourceDataBundle,
    scales: ScaleDiagnostics,
    png_path: str | Path,
    pdf_path: str | Path,
) -> None:
    import matplotlib

    figure = build_publication_figure_v2(source_data, scales)
    try:
        png_output = Path(png_path)
        pdf_output = Path(pdf_path)
        png_output.parent.mkdir(parents=True, exist_ok=True)
        pdf_output.parent.mkdir(parents=True, exist_ok=True)
        figure.savefig(
            png_output,
            dpi=300,
            metadata={"Software": "AXHome-MM Scientific Data S3 v2 generator"},
        )
        fixed_pdf_date = dt.datetime(2000, 1, 1, tzinfo=dt.timezone.utc)
        with matplotlib.rc_context({"pdf.fonttype": 42}):
            figure.savefig(
                pdf_output,
                dpi=300,
                metadata={
                    "Creator": "AXHome-MM Scientific Data S3 v2 generator",
                    "Producer": "Matplotlib",
                    "CreationDate": fixed_pdf_date,
                    "ModDate": fixed_pdf_date,
                },
            )
    finally:
        close_figure(figure)


def plot_publication_figure_v3(
    source_data: SourceDataBundle,
    scales: ScaleDiagnostics,
    png_path: str | Path,
    pdf_path: str | Path,
    svg_path: str | Path,
) -> None:
    import matplotlib

    figure = build_publication_figure_v3(source_data, scales)
    try:
        png_output = Path(png_path)
        pdf_output = Path(pdf_path)
        svg_output = Path(svg_path)
        for output in (png_output, pdf_output, svg_output):
            output.parent.mkdir(parents=True, exist_ok=True)
        figure.savefig(
            png_output,
            dpi=300,
            metadata={"Software": "AXHome-MM Scientific Data S3 v3 generator"},
        )
        fixed_pdf_date = dt.datetime(2000, 1, 1, tzinfo=dt.timezone.utc)
        with matplotlib.rc_context({"pdf.fonttype": 42}):
            figure.savefig(
                pdf_output,
                dpi=300,
                metadata={
                    "Creator": "AXHome-MM Scientific Data S3 v3 generator",
                    "Producer": "Matplotlib",
                    "CreationDate": fixed_pdf_date,
                    "ModDate": fixed_pdf_date,
                },
            )
        with matplotlib.rc_context(
            {"svg.fonttype": "none", "svg.hashsalt": "axhome-s3-v3"}
        ):
            figure.savefig(
                svg_output,
                dpi=300,
                metadata={
                    "Creator": "AXHome-MM Scientific Data S3 v3 generator",
                    "Date": "2000-01-01T00:00:00Z",
                },
            )
    finally:
        close_figure(figure)


def write_publication_caption_v3(path: str | Path) -> None:
    text = (
        "**Supplementary Figure S3 | CSI preprocessing and representative "
        "processed representations in AXHome.** "
        "**a,** Raw CSI amplitudes from RX1 and RX2 of one representative "
        "AXHome sample and the corresponding processed representations used as "
        "input to the CSI-only 2D CNN baseline. **b,** Processed CSI "
        "representations from three representative samples spanning different "
        "participants, actions, environments and WiFi link configurations. The "
        "raw RX1 and RX2 panels share one raw-amplitude scale, and all processed "
        "panels share one standardized log-amplitude scale. Processed panels are "
        "displayed using the corresponding source packet and subcarrier indices "
        "selected by the preprocessing pipeline. The representative samples "
        "illustrate qualitative diversity; they do not indicate action-specific "
        "separability and do not report classification performance.\n"
    )
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(text, encoding="utf-8", newline="\n")


def _require_report_samples(
    validated_samples: Sequence[object],
    arrays_by_sample: Sequence[SampleArrays] | None = None,
) -> tuple[list[object], list[SampleArrays] | None]:
    validated = list(validated_samples)
    expected_ids = [item.sample_id for item in FROZEN_SAMPLES]
    if [item.sample_id for item in validated] != expected_ids:
        raise ValueError("reports require all four validated samples in frozen order")
    arrays = None
    if arrays_by_sample is not None:
        arrays = _require_frozen_arrays(arrays_by_sample)
    return validated, arrays


def write_caption(
    path: str | Path,
    validated_samples: Sequence[object],
    scales: ScaleDiagnostics,
) -> None:
    validated, _ = _require_report_samples(validated_samples)
    ids = [item.sample_id for item in validated]
    text = f"""# Candidate Supplementary Figure S3 caption

**English.** **Supplementary Figure S3 | Native CSI amplitude, exact model-input construction, and rule-selected sample diversity.** Group A uses the real AXHome sample `{ids[0]}`. The first column shows native-resolution raw amplitude `|H|` for RX1 and RX2 ({validated[0].csi.shape[0]} packets × {validated[0].csi.shape[3]} source subcarriers), before log transformation, repair, resampling, or standardization. The second column shows the exact amplitude-based 2D CNN input after `log1p(|H|)`, low-energy-subcarrier interpolation, even selection to {TARGET_PACKETS} packets × {TARGET_SUBCARRIERS} subcarriers, and per-RX z-score standardization. Group B shows the same exact processed representation for `{ids[1]}`, `{ids[2]}`, and `{ids[3]}`, selected by the frozen validity, robust-centrality, and participant-diversity rule. The two raw RX panels share `cividis` limits from 0 to the pooled 99.5th percentile ({scales.raw_vmax:.4f}); the eight processed panels share `RdBu_r` limits centered at zero and bounded by the pooled absolute-z 99th percentile (±{scales.processed_limit:.4f}). Values beyond these limits are clipped only for display. The panels demonstrate different structured patterns among these four real samples. The figure does not estimate the dataset distribution, does not identify actions from heatmaps, and does not provide model-performance evidence. CSI phase is excluded because no validated phase-calibration chain is available.

**中文核对译文。** **补充图 S3｜原生 CSI 幅值、精确模型输入构造与规则选择样本的结构差异。** A 组使用真实 AXHome 样本 `{ids[0]}`。第一列显示 RX1 和 RX2 的 native-resolution raw amplitude `|H|`（{validated[0].csi.shape[0]} 个包 × {validated[0].csi.shape[3]} 个源子载波），尚未执行对数变换、低能量修复、重采样或标准化。第二列显示实际 amplitude-based 2D CNN 输入：`log1p(|H|)`、低能量子载波插值、均匀选择为 {TARGET_PACKETS} × {TARGET_SUBCARRIERS}，并逐 RX 做 z-score。B 组对 `{ids[1]}`、`{ids[2]}` 和 `{ids[3]}` 展示相同的精确处理结果；三者由冻结的数据有效性、稳健中心性和 participant diversity 规则选择。两个 raw panels 共享 0 到 pooled 99.5th percentile（{scales.raw_vmax:.4f}）的 `cividis` 色标；八个 processed panels 共享以 0 为中心、由 pooled `|z|` 99th percentile 确定的 `RdBu_r` 色标（±{scales.processed_limit:.4f}）。超出范围的数值只在显示时裁剪。本图仅说明四个真实样本中存在不同但结构化的 CSI patterns，不估计总体分布、不根据热力图识别动作，也不提供模型性能证据。由于没有经验证的相位校准链，本图不使用 CSI phase。
"""
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(text, encoding="utf-8", newline="\n")


def write_status_update(
    path: str | Path,
    validated_samples: Sequence[object],
) -> None:
    validated, _ = _require_report_samples(validated_samples)
    sample_ids = ", ".join(f"`{item.sample_id}`" for item in validated)
    text = f"""# Supplementary Figure S3 candidate status

- Provenance status: **sufficient** for the frozen amplitude-based figure. Four active-release samples were hash-checked, decoded, and compared exactly with the authoritative preprocessing: {sample_ids}.
- Package status: **candidate**. The PNG, PDF, compressed source data, schema, diagnostics, caption, and generation record are internally bound by hashes.
- Scientific boundary: phase is excluded; the four panels do not establish dataset-level representativeness, action identification, generalization, or model performance.
- Manuscript status: 尚未自动合并到论文正文，也尚未替换任何正式编号图件；需作者完成最终排版审阅后再引用为 Supplementary Figure S3。
"""
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(text, encoding="utf-8", newline="\n")


def write_figure6_change_note(
    path: str | Path,
    *,
    project_root: str | Path,
    dataset_root: str | Path,
    evidence_root: str | Path,
    output_dir: str | Path,
    scales: ScaleDiagnostics,
) -> None:
    sample_ids = "\n".join(
        f"  - `{item.sample_id}`" for item in FROZEN_SAMPLES
    )
    command = "\n".join(
        (
            "mamba activate pytorch",
            "python quality_check/generate_candidate_supplementary_figure_s3.py `",
            f'  --project-root "{Path(project_root).resolve()}" `',
            f'  --dataset-root "{Path(dataset_root).resolve()}" `',
            f'  --evidence-root "{Path(evidence_root).resolve()}" `',
            f'  --output-dir "{Path(output_dir).resolve()}" `',
            "  --style-profile figure6",
        )
    )
    text = f"""# Figure 6 redraw change note

## What changed

- Replaced the architecture-specific `2D CNN input representation` label with `Model input representation`, consistent with the shared input used by the four CSI baselines.
- Aligned every column heading to the same two-line baseline grid, added a thin grey separator between panels a and b, removed top/right heatmap spines, and tightened typography and spacing.
- Fixed the figure width at {FIGURE6_WIDTH_IN:.1f} in and rendered PNG at 300 dpi; SVG and PDF are produced from the same figure object.

## 2026-09-04 r3 incremental corrections

1. **Colorbar jurisdiction.** The raw-amplitude colorbar now aligns only with the `Raw CSI amplitude` column. The standardized-z colorbar starts at the left edge of `Model input representation` and ends at the right edge of the final panel b column, covering all four processed heatmap columns. The unequal lengths are intentional and encode their different jurisdictions.
2. **Panel separator.** The grey a/b separator stops above the colorbar band because the standardized-z colorbar crosses the a/b boundary; extending the line through that bar would conflict with its four-column jurisdiction.
3. **Source-packet range check and figure policy.** Coordinate validation confirms that panel a uses 496 source packets (`-0.5` to `495.5`), whereas each panel b sample uses 495 (`-0.5` to `494.5`). The one-packet difference is approximately 0.2% of the panel a range and is not visually resolvable here; because all columns use the same displayed `0 / 200 / 400` tick values, the figure restores shared y-axis labeling on the leftmost column only. The exact range difference remains recorded in coordinate validation and the generation record rather than being repeated on the figure face.

## What was preserved

- Frozen sample order:
{sample_ids}
- Raw display scale: `cividis`, 0 to pooled 99.5th percentile (`{scales.raw_vmax:.9g}`).
- Processed display scale: `RdBu_r`, centered at 0 with pooled absolute-z 99th-percentile limit (`±{scales.processed_limit:.9g}`).
- Percentile clipping is display-only; the source-data values are not clipped.
- Processed arrays remain plotted at their selected source packet and source subcarrier coordinates.
- The manuscript source and `baseline-paper-figures` are outside this generation step.

## Generation command

```powershell
{command}
```
"""
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(text, encoding="utf-8", newline="\n")


def _scale_payload(scales: ScaleDiagnostics) -> dict[str, object]:
    return {
        "raw": {
            "vmin": scales.raw_vmin,
            "vmax": scales.raw_vmax,
            "percentile": scales.raw_percentile,
            "clipped_fraction": scales.raw_clipped_fraction,
        },
        "processed": {
            "vmin": -scales.processed_limit,
            "vmax": scales.processed_limit,
            "limit": scales.processed_limit,
            "percentile": scales.processed_percentile,
            "clipped_fraction": scales.processed_clipped_fraction,
            "clipped_fraction_by_sample": dict(
                scales.processed_clipped_fraction_by_sample
            ),
        },
    }


def write_diagnostics(
    path: str | Path,
    validated_samples: Sequence[object],
    arrays_by_sample: Sequence[SampleArrays],
    source_data: SourceDataBundle,
    scales: ScaleDiagnostics,
) -> None:
    validated, arrays_optional = _require_report_samples(
        validated_samples, arrays_by_sample
    )
    assert arrays_optional is not None
    samples = []
    for validated_sample, item in zip(validated, arrays_optional, strict=True):
        stages = item.stages
        samples.append(
            {
                "role": item.spec.role,
                "sample_id": item.spec.sample_id,
                "participant": item.spec.participant,
                "action": item.spec.action,
                "environment": item.spec.environment,
                "link_configuration": item.spec.link_configuration,
                "decoded_shape": [int(value) for value in validated_sample.csi.shape],
                "decoded_dtype": validated_sample.csi.dtype.name,
                "native_raw_shape": [
                    int(value) for value in item.native_raw_amplitude.shape
                ],
                "native_raw_dtype": item.native_raw_amplitude.dtype.name,
                "processed_shape": [
                    int(value) for value in stages.processed_zscore.shape
                ],
                "processed_dtype": stages.processed_zscore.dtype.name,
                "authoritative_max_abs_difference": (
                    item.authoritative_max_abs_difference
                ),
                "packet_indices": stages.packet_indices.tolist(),
                "subcarrier_indices": stages.subcarrier_indices.tolist(),
                "low_energy_indices": stages.low_energy_indices.tolist(),
                "low_energy_count": int(len(stages.low_energy_indices)),
                "reference_carrier_energy": stages.reference_carrier_energy,
                "zscore_mean_by_rx": stages.zscore_mean.tolist(),
                "zscore_std_by_rx": stages.zscore_std.tolist(),
            }
        )
    payload = {
        "schema_version": 1,
        "provenance_status": "sufficient",
        "source_data_row_count": source_data.row_count,
        "scales": _scale_payload(scales),
        "samples": samples,
    }
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
        newline="\n",
    )


def _path_record(path: str | Path) -> dict[str, str | int]:
    resolved = Path(path).resolve()
    if not resolved.is_file():
        raise ValueError(f"provenance input does not exist: {resolved}")
    return {
        "path": str(resolved),
        "sha256": sha256_file(resolved),
        "size_bytes": resolved.stat().st_size,
    }


def _output_record(path: str | Path) -> dict[str, str | int]:
    resolved = Path(path).resolve()
    if not resolved.is_file():
        raise ValueError(f"generated output does not exist: {resolved}")
    return {
        "sha256": sha256_file(resolved),
        "size_bytes": resolved.stat().st_size,
    }


def _expected_path_record(
    name: str,
    path: str | Path,
    expected_sha256: str,
) -> dict[str, str | int]:
    record = _path_record(path)
    if record["sha256"] != expected_sha256:
        raise ValueError(
            f"{name} changed after selection-evidence validation: "
            f"expected {expected_sha256}, got {record['sha256']}"
        )
    return record


def write_generation_record(
    path: str | Path,
    *,
    project_root: str | Path,
    dataset_root: str | Path,
    output_dir: str | Path,
    validated_samples: Sequence[object],
    arrays_by_sample: Sequence[SampleArrays],
    source_data: SourceDataBundle,
    scales: ScaleDiagnostics,
    evidence_paths: dict[str, str | Path],
    expected_evidence_hashes: dict[str, str],
    generated_at: str,
    command_argv: Sequence[str],
    git_state: dict[str, str | bool],
    staged_dir: str | Path,
    deterministic_output_names: Sequence[str] = DETERMINISTIC_OUTPUT_NAMES,
    plotting_parameters: dict[str, object] | None = None,
    artifact_name: str = "candidate Supplementary Figure S3",
) -> None:
    import matplotlib

    validated, arrays_optional = _require_report_samples(
        validated_samples, arrays_by_sample
    )
    assert arrays_optional is not None
    root = Path(project_root).resolve()
    staged = Path(staged_dir).resolve()
    if set(evidence_paths) != set(expected_evidence_hashes):
        raise ValueError("evidence paths and expected hashes do not match")
    code_paths = {
        "parser": root / "baseline-csi-2dcnn" / "axhome_csi" / "feitcsi.py",
        "cli": root / "baseline-csi-2dcnn" / "axhome_csi" / "cli.py",
        "preprocessing": root / "baseline-csi-2dcnn" / "axhome_csi" / "data.py",
        "training": root
        / "baseline-csi-2dcnn"
        / "axhome_csi"
        / "training.py",
        "model_architectures": root
        / "baseline-csi-2dcnn"
        / "axhome_csi"
        / "model.py",
        "model_input": root
        / "baseline-csi-2dcnn"
        / "axhome_csi"
        / "torch_data.py",
        "generator": Path(__file__).resolve(),
        "selection_design": root
        / "manuscript"
        / "04_technical_validation"
        / "S3代表样本筛选与图件设计.md",
    }
    output_names = tuple(deterministic_output_names)
    if not output_names or len(set(output_names)) != len(output_names):
        raise ValueError("deterministic output names must be unique and non-empty")
    default_plotting_parameters = {
        "figure_size_inches": [10.5, 6.0],
        "dpi": 300,
        "layout": "2 RX rows x 5 heatmap columns",
        "raw_colormap": "cividis",
        "raw_percentile": scales.raw_percentile,
        "raw_vmin": scales.raw_vmin,
        "raw_vmax": scales.raw_vmax,
        "processed_colormap": "RdBu_r",
        "processed_center": 0.0,
        "processed_percentile": scales.processed_percentile,
        "processed_limit": scales.processed_limit,
        "display_clipping_only": True,
        "shading": "flat",
        "pdf_heatmaps_rasterized": True,
    }
    if plotting_parameters is not None:
        default_plotting_parameters.update(plotting_parameters)
    release_samples = []
    for validated_sample, item in zip(validated, arrays_optional, strict=True):
        release_samples.append(
            {
                "role": item.spec.role,
                "sample_id": item.spec.sample_id,
                "manifest_row": dict(validated_sample.row),
                "csi": {
                    "path": str(Path(validated_sample.csi_path).resolve()),
                    "sha256": validated_sample.csi_sha256,
                    "size_bytes": validated_sample.csi_size_bytes,
                },
                "metadata": {
                    "path": str(Path(validated_sample.metadata_path).resolve()),
                    "sha256": validated_sample.metadata_sha256,
                    "size_bytes": validated_sample.metadata_size_bytes,
                },
                "decoded_shape": [int(value) for value in validated_sample.csi.shape],
                "decoded_dtype": validated_sample.csi.dtype.name,
                "packet_count": len(validated_sample.headers),
                "authoritative_max_abs_difference": (
                    item.authoritative_max_abs_difference
                ),
            }
        )
    manifest_sample = validated[0]
    payload = {
        "schema_version": 1,
        "provenance_status": "sufficient",
        "artifact": artifact_name,
        "generated_at_utc": generated_at,
        "argv": list(command_argv),
        "command": subprocess.list2cmdline(list(command_argv)),
        "paths": {
            "project_root": str(root),
            "dataset_root": str(Path(dataset_root).resolve()),
            "output_dir": str(Path(output_dir).resolve()),
        },
        "git": dict(git_state),
        "samples": release_samples,
        "inputs": {
            "release": {
                "manifest": {
                    "path": str(Path(manifest_sample.manifest_path).resolve()),
                    "sha256": manifest_sample.manifest_sha256,
                    "size_bytes": manifest_sample.manifest_size_bytes,
                },
                "samples": release_samples,
            },
            "evidence": {
                name: _expected_path_record(
                    name,
                    item,
                    expected_evidence_hashes[name],
                )
                for name, item in sorted(evidence_paths.items())
            },
            "code": {name: _path_record(item) for name, item in code_paths.items()},
        },
        "parameters": {
            "preprocessing": {
                "target_packets": TARGET_PACKETS,
                "target_subcarriers": TARGET_SUBCARRIERS,
                "low_energy_ratio": LOW_ENERGY_RATIO,
                "epsilon": EPSILON,
            },
            "plotting": default_plotting_parameters,
        },
        "scales": _scale_payload(scales),
        "source_data": {
            "format": "deterministic gzip-compressed UTF-8 CSV",
            "fields": list(SOURCE_DATA_FIELDS),
            "row_count": source_data.row_count,
            "float_serialization": "nine significant digits; exact float32 round trip",
        },
        "software": {
            "python": sys.version.split()[0],
            "numpy": np.__version__,
            "matplotlib": matplotlib.__version__,
        },
        "outputs": {
            name: _output_record(staged / name)
            for name in output_names
        },
        "limitations": [
            "CSI phase is excluded because no validated phase-calibration chain is available.",
            "The four samples do not estimate the distribution of the 6,960 human-action windows.",
            "Visual patterns do not establish action identity or a physical causal interpretation.",
            "The figure does not provide model-performance or comparative-effectiveness evidence.",
            "Percentile clipping changes display saturation only; source data retain the original float32 values.",
            "Package files are atomically replaced one by one; regeneration may be required after an OS-level publication failure.",
            "Evidence and code files are hashed while the record is written; concurrent external modification is outside the reproducibility guarantee.",
        ],
    }
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
        newline="\n",
    )


def resolve_evidence_paths(
    dataset_root: str | Path,
    evidence_root: str | Path,
) -> dict[str, Path]:
    dataset = Path(dataset_root).resolve()
    evidence = Path(evidence_root).resolve()
    audit_root = (
        evidence
        / "outputs"
        / "technical_validation_active7024_7d7ea605_full_20260801"
    )
    return {
        "exclusion_ledger": dataset / "manifests" / "excluded_samples.csv",
        "active_csi_audit": audit_root / "csi_audit.csv",
        "active_sample_audit": audit_root / "sample_audit.csv",
        "active_metadata_audit": audit_root / "metadata_audit.csv",
        "independent_ftm_recheck": evidence
        / "outputs"
        / "recheck_csi_continuity_20260807_212045"
        / "recheck_summary.json",
    }


def _read_selected_audit_rows(
    snapshot: bytes,
    *,
    required_fields: set[str],
    audit_name: str,
) -> dict[str, dict[str, str]]:
    try:
        text = snapshot.decode("utf-8-sig")
        with io.StringIO(text, newline="") as handle:
            reader = csv.DictReader(handle)
            fields = set(reader.fieldnames or ())
            missing = sorted(required_fields.difference(fields))
            if missing:
                raise ValueError(
                    f"{audit_name} is missing fields: {', '.join(missing)}"
                )
            selected_ids = {item.sample_id for item in FROZEN_SAMPLES}
            grouped: dict[str, list[dict[str, str]]] = {
                sample_id: [] for sample_id in selected_ids
            }
            for row in reader:
                sample_id = row.get("sample_id", "")
                if sample_id in grouped:
                    grouped[sample_id].append(row)
    except (OSError, UnicodeError) as error:
        raise ValueError(f"cannot read {audit_name}: {error}") from error
    for sample_id, rows in grouped.items():
        if len(rows) != 1:
            raise ValueError(
                f"{audit_name} must contain sample {sample_id} exactly once"
            )
    return {sample_id: rows[0] for sample_id, rows in grouped.items()}


def validate_selection_evidence(
    evidence_paths: dict[str, str | Path],
    expected_hashes: dict[str, str],
) -> list[dict[str, str]]:
    expected_names = set(FROZEN_EVIDENCE_SHA256)
    if set(evidence_paths) != expected_names or set(expected_hashes) != expected_names:
        raise ValueError("selection evidence mapping does not match the frozen schema")
    resolved = {name: Path(item).resolve() for name, item in evidence_paths.items()}
    snapshots = {}
    for name in sorted(expected_names):
        if not resolved[name].is_file():
            raise ValueError(f"selection evidence does not exist: {name}")
        try:
            snapshot = resolved[name].read_bytes()
        except OSError as error:
            raise ValueError(f"cannot snapshot selection evidence: {name}") from error
        observed_hash = sha256_bytes(snapshot)
        if observed_hash != expected_hashes[name]:
            raise ValueError(
                f"{name} SHA-256 mismatch: expected {expected_hashes[name]}, got {observed_hash}"
            )
        snapshots[name] = snapshot

    identity_fields = {
        "sample_id",
        "person_id",
        "environment_id",
        "action_id",
        "link_geometry",
    }
    csi_rows = _read_selected_audit_rows(
        snapshots["active_csi_audit"],
        required_fields=identity_fields
        | {
            "parse_status",
            "quality_flags",
            "ftm_gap_gt_100ms_count",
            "low_energy_subcarrier_count",
        },
        audit_name="active_csi_audit",
    )
    sample_rows = _read_selected_audit_rows(
        snapshots["active_sample_audit"],
        required_fields=identity_fields
        | {
            "metadata_status",
            "csi_parse_status",
            "video_probe_status",
            "quality_flags",
            "final_status",
        },
        audit_name="active_sample_audit",
    )
    metadata_rows = _read_selected_audit_rows(
        snapshots["active_metadata_audit"],
        required_fields=identity_fields
        | {
            "metadata_status",
            "sync_status",
            "source_quality_flags",
            "slice_status",
            "quality_flags",
        },
        audit_name="active_metadata_audit",
    )

    for spec in FROZEN_SAMPLES:
        expected_identity = {
            "sample_id": spec.sample_id,
            "person_id": spec.participant,
            "environment_id": spec.environment,
            "action_id": spec.action,
            "link_geometry": spec.link_configuration,
        }
        for audit_name, rows in (
            ("active_csi_audit", csi_rows),
            ("active_sample_audit", sample_rows),
            ("active_metadata_audit", metadata_rows),
        ):
            row = rows[spec.sample_id]
            if any(row[field] != value for field, value in expected_identity.items()):
                raise ValueError(
                    f"{audit_name} identity mismatch for sample {spec.sample_id}"
                )
        csi = csi_rows[spec.sample_id]
        if (
            csi["parse_status"] != "ok"
            or csi["quality_flags"] != "ok"
            or csi["ftm_gap_gt_100ms_count"] != "0"
            or csi["low_energy_subcarrier_count"] != "16"
        ):
            raise ValueError(f"CSI audit quality mismatch for sample {spec.sample_id}")
        sample = sample_rows[spec.sample_id]
        if any(
            sample[field] != "ok"
            for field in (
                "metadata_status",
                "csi_parse_status",
                "video_probe_status",
                "quality_flags",
                "final_status",
            )
        ):
            raise ValueError(f"sample audit quality mismatch for sample {spec.sample_id}")
        metadata = metadata_rows[spec.sample_id]
        if any(
            metadata[field] != "ok"
            for field in (
                "metadata_status",
                "sync_status",
                "source_quality_flags",
                "slice_status",
                "quality_flags",
            )
        ):
            raise ValueError(
                f"metadata audit quality mismatch for sample {spec.sample_id}"
            )

    try:
        exclusion_text = snapshots["exclusion_ledger"].decode("utf-8-sig")
        with io.StringIO(exclusion_text, newline="") as handle:
            exclusion_reader = csv.DictReader(handle)
            if "sample_id" not in (exclusion_reader.fieldnames or []):
                raise ValueError("exclusion ledger is missing sample_id")
            excluded_ids = {row["sample_id"] for row in exclusion_reader}
        recheck = json.loads(
            snapshots["independent_ftm_recheck"].decode("utf-8-sig")
        )
        warned_ids = set(recheck["files_with_gap_gt_100ms"])
        errors = recheck["errors"]
    except (OSError, UnicodeError, json.JSONDecodeError, KeyError, TypeError) as error:
        raise ValueError("selection warning evidence is invalid") from error
    selected_ids = {item.sample_id for item in FROZEN_SAMPLES}
    if selected_ids.intersection(excluded_ids):
        raise ValueError("a frozen sample appears in the exclusion ledger")
    if errors or selected_ids.intersection(warned_ids):
        raise ValueError("a frozen sample has an independent FTM warning")

    return [dict(csi_rows[item.sample_id]) for item in FROZEN_SAMPLES]


def generate_package(
    *,
    project_root: str | Path,
    dataset_root: str | Path,
    evidence_root: str | Path,
    output_dir: str | Path,
    generated_at: str,
    command_argv: Sequence[str],
) -> dict[str, object]:
    project = Path(project_root).resolve()
    dataset = Path(dataset_root).resolve()
    output = Path(output_dir).resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    evidence_paths = resolve_evidence_paths(dataset, evidence_root)
    validate_selection_evidence(evidence_paths, FROZEN_EVIDENCE_SHA256)
    git_state = collect_git_state(project)
    if git_state.get("dirty"):
        raise ValueError("generation requires a clean committed project snapshot")

    with tempfile.TemporaryDirectory(
        prefix=".candidate-s3-", dir=output.parent
    ) as temporary_directory:
        staged = Path(temporary_directory)
        validated = []
        arrays = []
        for spec in FROZEN_SAMPLES:
            sample = validate_frozen_sample(
                dataset,
                sample_id=spec.sample_id,
                expected_manifest_sha256=FROZEN_MANIFEST_SHA256,
                expected_csi_sha256=spec.expected_csi_sha256,
                expected_shape=spec.expected_shape,
            )
            for field, expected in (
                ("person_id", spec.participant),
                ("action_id", spec.action),
                ("environment_id", spec.environment),
            ):
                if sample.row.get(field) != expected:
                    raise ValueError(
                        f"sample {spec.sample_id}: manifest {field} mismatch"
                    )
            validated.append(sample)
            arrays.append(extract_sample_arrays(spec, sample.csi))

        scales = compute_scale_diagnostics(arrays)
        source_path = staged / SOURCE_DATA_NAME
        write_source_data(source_path, arrays)
        source_data = read_source_data(source_path, arrays)
        write_source_schema(staged / SOURCE_SCHEMA_NAME)
        write_diagnostics(
            staged / DIAGNOSTICS_NAME,
            validated,
            arrays,
            source_data,
            scales,
        )
        write_caption(staged / CAPTION_NAME, validated, scales)
        write_status_update(staged / STATUS_UPDATE_NAME, validated)
        plot_figure(
            source_data,
            scales,
            staged / PNG_NAME,
            staged / PDF_NAME,
        )
        write_generation_record(
            staged / GENERATION_RECORD_NAME,
            project_root=project,
            dataset_root=dataset,
            output_dir=output,
            validated_samples=validated,
            arrays_by_sample=arrays,
            source_data=source_data,
            scales=scales,
            evidence_paths=evidence_paths,
            expected_evidence_hashes=FROZEN_EVIDENCE_SHA256,
            generated_at=generated_at,
            command_argv=command_argv,
            git_state=git_state,
            staged_dir=staged,
        )

        output.mkdir(parents=True, exist_ok=True)
        for name in PACKAGE_OUTPUT_NAMES:
            os.replace(staged / name, output / name)

    return {
        "provenance_status": "sufficient",
        "sample_ids": [item.sample_id for item in FROZEN_SAMPLES],
        "raw_vmax": scales.raw_vmax,
        "processed_limit": scales.processed_limit,
        "artifacts": [str(output / name) for name in PACKAGE_OUTPUT_NAMES],
    }


def generate_publication_layout_v2(
    *,
    project_root: str | Path,
    dataset_root: str | Path,
    evidence_root: str | Path,
    output_dir: str | Path,
) -> dict[str, object]:
    project = Path(project_root).resolve()
    dataset = Path(dataset_root).resolve()
    output = Path(output_dir).resolve()
    if not project.is_dir():
        raise ValueError(f"project root does not exist: {project}")
    output.parent.mkdir(parents=True, exist_ok=True)
    evidence_paths = resolve_evidence_paths(dataset, evidence_root)
    validate_selection_evidence(evidence_paths, FROZEN_EVIDENCE_SHA256)

    with tempfile.TemporaryDirectory(
        prefix=".candidate-s3-v2-", dir=output.parent
    ) as temporary_directory:
        staged = Path(temporary_directory)
        arrays = []
        for spec in FROZEN_SAMPLES:
            sample = validate_frozen_sample(
                dataset,
                sample_id=spec.sample_id,
                expected_manifest_sha256=FROZEN_MANIFEST_SHA256,
                expected_csi_sha256=spec.expected_csi_sha256,
                expected_shape=spec.expected_shape,
            )
            for field, expected in (
                ("person_id", spec.participant),
                ("action_id", spec.action),
                ("environment_id", spec.environment),
            ):
                if sample.row.get(field) != expected:
                    raise ValueError(
                        f"sample {spec.sample_id}: manifest {field} mismatch"
                    )
            arrays.append(extract_sample_arrays(spec, sample.csi))

        scales = compute_scale_diagnostics(arrays)
        source_path = staged / SOURCE_DATA_NAME
        write_source_data(source_path, arrays)
        source_data = read_source_data(source_path, arrays)
        plot_publication_figure_v2(
            source_data,
            scales,
            staged / V2_PNG_NAME,
            staged / V2_PDF_NAME,
        )

        output.mkdir(parents=True, exist_ok=True)
        for name in (V2_PNG_NAME, V2_PDF_NAME):
            os.replace(staged / name, output / name)

    font_family, font_path = resolve_publication_font()
    return {
        "provenance_status": "sufficient",
        "sample_ids": [item.sample_id for item in FROZEN_SAMPLES],
        "raw_vmax": scales.raw_vmax,
        "processed_limit": scales.processed_limit,
        "font_family": font_family,
        "font_path": font_path,
        "figure_size_mm": [V2_FIGURE_WIDTH_MM, V2_FIGURE_HEIGHT_MM],
        "artifacts": [str(output / V2_PNG_NAME), str(output / V2_PDF_NAME)],
    }


def generate_publication_layout_v3(
    *,
    project_root: str | Path,
    dataset_root: str | Path,
    evidence_root: str | Path,
    output_dir: str | Path,
) -> dict[str, object]:
    project = Path(project_root).resolve()
    dataset = Path(dataset_root).resolve()
    output = Path(output_dir).resolve()
    if not project.is_dir():
        raise ValueError(f"project root does not exist: {project}")
    output.parent.mkdir(parents=True, exist_ok=True)
    evidence_paths = resolve_evidence_paths(dataset, evidence_root)
    validate_selection_evidence(evidence_paths, FROZEN_EVIDENCE_SHA256)

    with tempfile.TemporaryDirectory(
        prefix=".candidate-s3-v3-", dir=output.parent
    ) as temporary_directory:
        staged = Path(temporary_directory)
        arrays = []
        for spec in FROZEN_SAMPLES:
            sample = validate_frozen_sample(
                dataset,
                sample_id=spec.sample_id,
                expected_manifest_sha256=FROZEN_MANIFEST_SHA256,
                expected_csi_sha256=spec.expected_csi_sha256,
                expected_shape=spec.expected_shape,
            )
            for field, expected in (
                ("person_id", spec.participant),
                ("action_id", spec.action),
                ("environment_id", spec.environment),
            ):
                if sample.row.get(field) != expected:
                    raise ValueError(
                        f"sample {spec.sample_id}: manifest {field} mismatch"
                    )
            arrays.append(extract_sample_arrays(spec, sample.csi))

        scales = compute_scale_diagnostics(arrays)
        source_path = staged / SOURCE_DATA_NAME
        write_source_data(source_path, arrays)
        source_data = read_source_data(source_path, arrays)
        plot_publication_figure_v3(
            source_data,
            scales,
            staged / V3_PNG_NAME,
            staged / V3_PDF_NAME,
            staged / V3_SVG_NAME,
        )
        write_publication_caption_v3(staged / V3_CAPTION_NAME)

        output.mkdir(parents=True, exist_ok=True)
        for name in V3_OUTPUT_NAMES:
            os.replace(staged / name, output / name)

    font_family, font_path = resolve_publication_font()
    return {
        "provenance_status": "sufficient",
        "sample_ids": [item.sample_id for item in FROZEN_SAMPLES],
        "raw_vmax": scales.raw_vmax,
        "processed_limit": scales.processed_limit,
        "font_family": font_family,
        "font_path": font_path,
        "mathtext_fontset": "stixsans",
        "source_coordinate_mapping": True,
        "figure_size_mm": [V3_FIGURE_WIDTH_MM, V3_FIGURE_HEIGHT_MM],
        "artifacts": [str(output / name) for name in V3_OUTPUT_NAMES],
    }


def generate_publication_layout_v4(
    *,
    project_root: str | Path,
    dataset_root: str | Path,
    evidence_root: str | Path,
    output_dir: str | Path,
) -> dict[str, object]:
    project = Path(project_root).resolve()
    dataset = Path(dataset_root).resolve()
    output = Path(output_dir).resolve()
    if not project.is_dir():
        raise ValueError(f"project root does not exist: {project}")
    output.parent.mkdir(parents=True, exist_ok=True)
    evidence_paths = resolve_evidence_paths(dataset, evidence_root)
    validate_selection_evidence(evidence_paths, FROZEN_EVIDENCE_SHA256)

    with tempfile.TemporaryDirectory(
        prefix=".candidate-s3-v4-", dir=output.parent
    ) as temporary_directory:
        staged = Path(temporary_directory)
        arrays = []
        for spec in FROZEN_SAMPLES:
            sample = validate_frozen_sample(
                dataset,
                sample_id=spec.sample_id,
                expected_manifest_sha256=FROZEN_MANIFEST_SHA256,
                expected_csi_sha256=spec.expected_csi_sha256,
                expected_shape=spec.expected_shape,
            )
            for field, expected in (
                ("person_id", spec.participant),
                ("action_id", spec.action),
                ("environment_id", spec.environment),
            ):
                if sample.row.get(field) != expected:
                    raise ValueError(
                        f"sample {spec.sample_id}: manifest {field} mismatch"
                    )
            arrays.append(extract_sample_arrays(spec, sample.csi))

        scales = compute_scale_diagnostics(arrays)
        source_path = staged / SOURCE_DATA_NAME
        write_source_data(source_path, arrays)
        source_data = read_source_data(source_path, arrays)
        validation_summary = plot_publication_figure_v4(
            source_data,
            scales,
            staged / V4_PNG_NAME,
            staged / V4_PDF_NAME,
            staged / V4_SVG_NAME,
            staged / V4_VALIDATION_NAME,
        )

        output.mkdir(parents=True, exist_ok=True)
        for name in V4_OUTPUT_NAMES:
            os.replace(staged / name, output / name)

    font_family, font_path = resolve_publication_font()
    return {
        "provenance_status": "sufficient",
        "sample_ids": [item.sample_id for item in FROZEN_SAMPLES],
        "raw_vmax": scales.raw_vmax,
        "processed_limit": scales.processed_limit,
        "font_family": font_family,
        "font_path": font_path,
        "mathtext_fontset": "stixsans",
        "source_coordinate_mapping": True,
        "coordinate_policy": (
            "bounded source-domain mesh edges with authoritative axes limits"
        ),
        "coordinate_validation_status": validation_summary["status"],
        "coordinate_validation_tolerance": validation_summary["tolerance"],
        "figure_size_mm": [V3_FIGURE_WIDTH_MM, V3_FIGURE_HEIGHT_MM],
        "artifacts": [str(output / name) for name in V4_OUTPUT_NAMES],
    }


def generate_figure6_layout(
    *,
    project_root: str | Path,
    dataset_root: str | Path,
    evidence_root: str | Path,
    output_dir: str | Path,
    generated_at: str,
    command_argv: Sequence[str],
) -> dict[str, object]:
    """Generate the provenance-bound main-text Figure 6 package."""
    project = Path(project_root).resolve()
    dataset = Path(dataset_root).resolve()
    evidence = Path(evidence_root).resolve()
    output = Path(output_dir).resolve()
    if not project.is_dir():
        raise ValueError(f"project root does not exist: {project}")
    output.parent.mkdir(parents=True, exist_ok=True)
    evidence_paths = resolve_evidence_paths(dataset, evidence)
    validate_selection_evidence(evidence_paths, FROZEN_EVIDENCE_SHA256)
    git_state = collect_git_state(project)

    with tempfile.TemporaryDirectory(
        prefix=".figure6-csi-representation-", dir=output.parent
    ) as temporary_directory:
        staged = Path(temporary_directory)
        validated = []
        arrays = []
        for spec in FROZEN_SAMPLES:
            sample = validate_frozen_sample(
                dataset,
                sample_id=spec.sample_id,
                expected_manifest_sha256=FROZEN_MANIFEST_SHA256,
                expected_csi_sha256=spec.expected_csi_sha256,
                expected_shape=spec.expected_shape,
            )
            for field, expected in (
                ("person_id", spec.participant),
                ("action_id", spec.action),
                ("environment_id", spec.environment),
            ):
                if sample.row.get(field) != expected:
                    raise ValueError(
                        f"sample {spec.sample_id}: manifest {field} mismatch"
                    )
            validated.append(sample)
            arrays.append(extract_sample_arrays(spec, sample.csi))

        scales = compute_scale_diagnostics(arrays)
        source_path = staged / FIGURE6_SOURCE_DATA_NAME
        write_source_data(source_path, arrays)
        source_data = read_source_data(source_path, arrays)
        write_source_schema(staged / FIGURE6_SOURCE_SCHEMA_NAME)
        write_diagnostics(
            staged / FIGURE6_DIAGNOSTICS_NAME,
            validated,
            arrays,
            source_data,
            scales,
        )
        validation_summary = plot_figure6(
            source_data,
            scales,
            staged / FIGURE6_PNG_NAME,
            staged / FIGURE6_PDF_NAME,
            staged / FIGURE6_SVG_NAME,
            staged / FIGURE6_VALIDATION_NAME,
        )
        write_figure6_change_note(
            staged / FIGURE6_CHANGE_NOTE_NAME,
            project_root=project,
            dataset_root=dataset,
            evidence_root=evidence,
            output_dir=output,
            scales=scales,
        )
        font_family, font_path = resolve_publication_font()
        write_generation_record(
            staged / FIGURE6_GENERATION_RECORD_NAME,
            project_root=project,
            dataset_root=dataset,
            output_dir=output,
            validated_samples=validated,
            arrays_by_sample=arrays,
            source_data=source_data,
            scales=scales,
            evidence_paths=evidence_paths,
            expected_evidence_hashes=FROZEN_EVIDENCE_SHA256,
            generated_at=generated_at,
            command_argv=command_argv,
            git_state=git_state,
            staged_dir=staged,
            deterministic_output_names=FIGURE6_DETERMINISTIC_OUTPUT_NAMES,
            artifact_name="main-text Figure 6",
            plotting_parameters={
                "figure_size_inches": [FIGURE6_WIDTH_IN, FIGURE6_HEIGHT_IN],
                "layout": (
                    "2 RX rows x 5 heatmap columns; panel a preprocessing, "
                    "panel b representative samples"
                ),
                "font_family": font_family,
                "font_path": font_path,
                "model_input_label": "Model input representation",
                "model_input_shared_by": [
                    "cnn2d",
                    "resnet18",
                    "bilstm",
                    "transformer",
                ],
                "source_coordinate_mapping": True,
                "coordinate_policy": (
                    "bounded source-domain mesh edges with authoritative "
                    "axes limits"
                ),
                "colorbars_equal_length": False,
                "colorbar_scope_policy": validation_summary["layout"][
                    "colorbar_scope_policy"
                ],
                "source_packet_range_review": validation_summary["layout"][
                    "source_packet_range_review"
                ],
                "column_title_baselines_aligned": True,
                "panel_group_separator": True,
                "panel_separator_colorbar_policy": validation_summary[
                    "layout"
                ]["panel_separator_colorbar_policy"],
                "hidden_heatmap_spines": ["top", "right"],
            },
        )

        output.mkdir(parents=True, exist_ok=True)
        for name in FIGURE6_OUTPUT_NAMES:
            os.replace(staged / name, output / name)

    return {
        "provenance_status": "sufficient",
        "git_dirty_recorded": bool(git_state.get("dirty")),
        "sample_ids": [item.sample_id for item in FROZEN_SAMPLES],
        "raw_vmax": scales.raw_vmax,
        "processed_limit": scales.processed_limit,
        "font_family": font_family,
        "font_path": font_path,
        "source_coordinate_mapping": True,
        "coordinate_validation_status": validation_summary["status"],
        "figure_size_inches": [FIGURE6_WIDTH_IN, FIGURE6_HEIGHT_IN],
        "artifacts": [str(output / name) for name in FIGURE6_OUTPUT_NAMES],
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Generate the frozen AXHome candidate Supplementary Figure S3 "
            "or main-text Figure 6 package."
        )
    )
    parser.add_argument("--project-root", required=True, type=Path)
    parser.add_argument("--dataset-root", required=True, type=Path)
    parser.add_argument("--evidence-root", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument(
        "--style-profile",
        choices=(
            "original-package",
            "scientific-data-v2",
            "scientific-data-v3",
            "scientific-data-v4",
            "figure6",
        ),
        default="original-package",
        help=(
            "Generate the original provenance package or the independent "
            "Scientific Data layout revision."
        ),
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    arguments = list(sys.argv[1:] if argv is None else argv)
    args = build_parser().parse_args(arguments)
    generated_at = dt.datetime.now(dt.timezone.utc).isoformat().replace("+00:00", "Z")
    command_argv = [
        sys.executable,
        "quality_check/generate_candidate_supplementary_figure_s3.py",
        *arguments,
    ]
    if args.style_profile == "figure6":
        summary = generate_figure6_layout(
            project_root=args.project_root,
            dataset_root=args.dataset_root,
            evidence_root=args.evidence_root,
            output_dir=args.output_dir,
            generated_at=generated_at,
            command_argv=command_argv,
        )
    elif args.style_profile == "scientific-data-v4":
        summary = generate_publication_layout_v4(
            project_root=args.project_root,
            dataset_root=args.dataset_root,
            evidence_root=args.evidence_root,
            output_dir=args.output_dir,
        )
    elif args.style_profile == "scientific-data-v3":
        summary = generate_publication_layout_v3(
            project_root=args.project_root,
            dataset_root=args.dataset_root,
            evidence_root=args.evidence_root,
            output_dir=args.output_dir,
        )
    elif args.style_profile == "scientific-data-v2":
        summary = generate_publication_layout_v2(
            project_root=args.project_root,
            dataset_root=args.dataset_root,
            evidence_root=args.evidence_root,
            output_dir=args.output_dir,
        )
    else:
        summary = generate_package(
            project_root=args.project_root,
            dataset_root=args.dataset_root,
            evidence_root=args.evidence_root,
            output_dir=args.output_dir,
            generated_at=generated_at,
            command_argv=command_argv,
        )
    print(json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
