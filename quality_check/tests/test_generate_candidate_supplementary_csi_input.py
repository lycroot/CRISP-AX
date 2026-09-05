import csv
import hashlib
import importlib.util
import inspect
import io
import json
import struct
import subprocess
import sys
import tempfile
import unittest
import warnings
from contextlib import redirect_stdout
from dataclasses import dataclass, replace
from pathlib import Path
from unittest import mock

import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parents[2]
BASELINE_PACKAGE_ROOT = PROJECT_ROOT / "baseline-csi-2dcnn"
sys.path.insert(0, str(BASELINE_PACKAGE_ROOT))

from axhome_csi.data import (  # noqa: E402
    _even_indices,
    interpolate_low_energy_subcarriers,
    preprocess_amplitude as authoritative_preprocess_amplitude,
)


SCRIPT_PATH = (
    PROJECT_ROOT / "quality_check" / "generate_candidate_supplementary_csi_input.py"
)


def load_generator():
    spec = importlib.util.spec_from_file_location("candidate_csi_figure", SCRIPT_PATH)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot load {SCRIPT_PATH}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def synthetic_csi() -> np.ndarray:
    rng = np.random.default_rng(20260808)
    real = rng.integers(-40, 41, size=(12, 2, 1, 16), dtype=np.int16)
    imag = rng.integers(-40, 41, size=(12, 2, 1, 16), dtype=np.int16)
    real[..., 4] = 0
    imag[..., 4] = 0
    return (real.astype(np.float32) + 1j * imag.astype(np.float32)).astype(
        np.complex64
    )


MANIFEST_FIELDS = [
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
]

SOURCE_FIELDS = [
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


@dataclass(frozen=True)
class ReleaseFixture:
    dataset_root: Path
    sample_id: str
    manifest_path: Path
    csi_path: Path
    metadata_path: Path
    csi: np.ndarray


def file_sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def write_feitcsi(path: Path, csi: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    packets, num_rx, num_tx, num_subcarriers = csi.shape
    with path.open("wb") as handle:
        for packet_index in range(packets):
            payload_components = np.empty(
                (num_rx, num_tx, num_subcarriers, 2), dtype="<i2"
            )
            payload_components[..., 0] = csi[packet_index].real.astype(np.int16)
            payload_components[..., 1] = csi[packet_index].imag.astype(np.int16)
            payload = payload_components.tobytes(order="C")
            header = bytearray(272)
            struct.pack_into("<I", header, 0, len(payload))
            struct.pack_into("<I", header, 8, packet_index + 1)
            header[46] = num_rx
            header[47] = num_tx
            struct.pack_into("<I", header, 52, num_subcarriers)
            handle.write(header)
            handle.write(payload)


def make_release(
    root: Path,
    *,
    manifest_packets: int | None = None,
    metadata_packets: int | float | bool | None = None,
    archive_csi_path: str | None = None,
    source_sample_id: str | None = None,
    slice_sample_id: str | None = None,
) -> ReleaseFixture:
    sample_id = "S01_E01_A01_T01"
    csi = synthetic_csi()
    relative_csi = archive_csi_path or f"data/S01/csi/walk/{sample_id}.dat"
    relative_metadata = f"data/S01/metadata/walk/{sample_id}.json"
    csi_path = root / relative_csi
    metadata_path = root / relative_metadata
    manifest_path = root / "manifests" / "archive_index.csv"

    write_feitcsi(csi_path, csi)
    metadata_path.parent.mkdir(parents=True, exist_ok=True)
    packets_written = csi.shape[0] if manifest_packets is None else manifest_packets
    metadata = {
        "source_sync_row": {"sample_id": source_sample_id or sample_id},
        "slice_result": {
            "sample_id": slice_sample_id or sample_id,
            "csi_packets_written": (
                packets_written if metadata_packets is None else metadata_packets
            ),
        },
    }
    metadata_path.write_text(
        json.dumps(metadata, ensure_ascii=False, sort_keys=True) + "\n",
        encoding="utf-8",
    )

    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    row = {
        "sample_id": sample_id,
        "session_id": "session-01",
        "archive_session_id": "archive-session-01",
        "person_id": "S01",
        "environment_id": "E01",
        "action_id": "walk",
        "trial_id": "T01",
        "archive_csi_path": relative_csi,
        "archive_metadata_path": relative_metadata,
        "csi_size_bytes": str(csi_path.stat().st_size),
        "csi_packets_written": str(packets_written),
        "status": "ok",
        "sync_quality_flags": "ok",
    }
    with manifest_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=MANIFEST_FIELDS)
        writer.writeheader()
        writer.writerow(row)

    return ReleaseFixture(
        dataset_root=root,
        sample_id=sample_id,
        manifest_path=manifest_path,
        csi_path=csi_path,
        metadata_path=metadata_path,
        csi=csi,
    )


def rewrite_source_rows(path: Path, mutate) -> None:
    with path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        rows = list(reader)
    mutate(rows)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=SOURCE_FIELDS)
        writer.writeheader()
        writer.writerows(rows)


class StageExtractionTests(unittest.TestCase):
    def test_complex128_follows_authoritative_log_precision(self):
        generator = load_generator()
        values = np.arange(6 * 2 * 1 * 8, dtype=np.float64).reshape(6, 2, 1, 8)
        csi = (
            values * 0.3141592653589793
            + 0.123456789012345
            + 1j * (values * 0.2718281828459045 + 0.987654321098765)
        )

        stages = generator.extract_stages(
            csi,
            target_packets=5,
            target_subcarriers=6,
            low_energy_ratio=0.05,
            epsilon=1e-6,
        )
        expected = authoritative_preprocess_amplitude(
            csi,
            target_packets=5,
            target_subcarriers=6,
            low_energy_ratio=0.05,
            epsilon=1e-6,
        )

        np.testing.assert_array_equal(stages.processed_zscore, expected)

    def test_stage_extraction_matches_authoritative_preprocess(self):
        generator = load_generator()
        csi = synthetic_csi()

        stages = generator.extract_stages(
            csi,
            target_packets=8,
            target_subcarriers=8,
            low_energy_ratio=0.05,
            epsilon=1e-6,
        )
        expected = authoritative_preprocess_amplitude(
            csi,
            target_packets=8,
            target_subcarriers=8,
            low_energy_ratio=0.05,
            epsilon=1e-6,
        )
        packet_indices = _even_indices(csi.shape[0], 8)
        subcarrier_indices = _even_indices(csi.shape[-1], 8)

        raw_full = np.abs(csi[:, :, 0, :]).astype(np.float32)
        expected_raw = raw_full[packet_indices][:, :, subcarrier_indices]
        expected_raw = np.transpose(expected_raw, (1, 0, 2))

        logged_full = np.log1p(raw_full).astype(np.float32)
        repaired_full, _ = interpolate_low_energy_subcarriers(
            logged_full,
            low_energy_ratio=0.05,
            epsilon=1e-6,
        )
        expected_repaired = repaired_full[packet_indices][:, :, subcarrier_indices]
        expected_repaired = np.transpose(expected_repaired, (1, 0, 2))

        np.testing.assert_array_equal(stages.processed_zscore, expected)
        np.testing.assert_array_equal(stages.raw_amplitude, expected_raw)
        np.testing.assert_array_equal(
            stages.log_repaired_amplitude, expected_repaired
        )
        self.assertEqual(stages.raw_amplitude.shape, (2, 8, 8))
        self.assertEqual(stages.log_repaired_amplitude.shape, (2, 8, 8))
        self.assertEqual(stages.packet_indices.tolist(), [0, 2, 3, 5, 6, 8, 9, 11])
        self.assertEqual(
            stages.subcarrier_indices.tolist(), [0, 2, 4, 6, 9, 11, 13, 15]
        )
        self.assertIn(4, stages.low_energy_indices)


class FrozenSampleValidationTests(unittest.TestCase):
    def test_validate_sample_returns_verified_real_parser_data(self):
        generator = load_generator()
        with tempfile.TemporaryDirectory() as tmp:
            fixture = make_release(Path(tmp))

            result = generator.validate_frozen_sample(
                fixture.dataset_root,
                sample_id=fixture.sample_id,
                expected_manifest_sha256=file_sha256(fixture.manifest_path),
                expected_csi_sha256=file_sha256(fixture.csi_path),
                expected_shape=fixture.csi.shape,
            )

            self.assertEqual(result.sample_id, fixture.sample_id)
            self.assertEqual(result.row["sample_id"], fixture.sample_id)
            self.assertEqual(result.manifest_path, fixture.manifest_path.resolve())
            self.assertEqual(result.csi_path, fixture.csi_path.resolve())
            self.assertEqual(result.metadata_path, fixture.metadata_path.resolve())
            self.assertEqual(result.manifest_sha256, file_sha256(fixture.manifest_path))
            self.assertEqual(result.csi_sha256, file_sha256(fixture.csi_path))
            self.assertEqual(
                result.metadata_sha256, file_sha256(fixture.metadata_path)
            )
            np.testing.assert_array_equal(result.csi, fixture.csi)
            self.assertEqual(len(result.headers), fixture.csi.shape[0])
            self.assertEqual(result.headers[0].ftm_clock, 1)

    def test_validate_sample_hash_and_parser_use_the_same_csi_bytes(self):
        generator = load_generator()
        with tempfile.TemporaryDirectory() as tmp:
            fixture = make_release(Path(tmp))
            original_bytes = fixture.csi_path.read_bytes()
            replacement_csi = (fixture.csi + np.complex64(1 + 1j)).astype(
                np.complex64
            )
            replacement_path = Path(tmp) / "replacement.dat"
            write_feitcsi(replacement_path, replacement_csi)
            replacement_bytes = replacement_path.read_bytes()
            original_sha256_bytes = generator.sha256_bytes
            replacement_triggered = False

            def replace_path_after_hash(data: bytes) -> str:
                nonlocal replacement_triggered
                digest = original_sha256_bytes(data)
                if data == original_bytes and not replacement_triggered:
                    fixture.csi_path.write_bytes(replacement_bytes)
                    replacement_triggered = True
                return digest

            with mock.patch.object(
                generator, "sha256_bytes", side_effect=replace_path_after_hash
            ):
                result = generator.validate_frozen_sample(
                    fixture.dataset_root,
                    sample_id=fixture.sample_id,
                    expected_manifest_sha256=file_sha256(fixture.manifest_path),
                    expected_csi_sha256=hashlib.sha256(original_bytes).hexdigest(),
                    expected_shape=fixture.csi.shape,
                )

            self.assertTrue(replacement_triggered)
            self.assertEqual(result.csi_sha256, hashlib.sha256(original_bytes).hexdigest())
            np.testing.assert_array_equal(result.csi, fixture.csi)

    def test_validate_sample_rejects_wrong_csi_hash(self):
        generator = load_generator()
        with tempfile.TemporaryDirectory() as tmp:
            fixture = make_release(Path(tmp))

            with self.assertRaisesRegex(ValueError, "CSI SHA-256 mismatch"):
                generator.validate_frozen_sample(
                    fixture.dataset_root,
                    sample_id=fixture.sample_id,
                    expected_manifest_sha256=file_sha256(fixture.manifest_path),
                    expected_csi_sha256="0" * 64,
                    expected_shape=fixture.csi.shape,
                )

    def test_validate_sample_rejects_packet_count_mismatch(self):
        generator = load_generator()
        with tempfile.TemporaryDirectory() as tmp:
            fixture = make_release(
                Path(tmp), manifest_packets=synthetic_csi().shape[0] - 1
            )

            with self.assertRaisesRegex(ValueError, "packet count mismatch"):
                generator.validate_frozen_sample(
                    fixture.dataset_root,
                    sample_id=fixture.sample_id,
                    expected_manifest_sha256=file_sha256(fixture.manifest_path),
                    expected_csi_sha256=file_sha256(fixture.csi_path),
                    expected_shape=fixture.csi.shape,
                )

    def test_validate_sample_rejects_manifest_hash_mismatch(self):
        generator = load_generator()
        with tempfile.TemporaryDirectory() as tmp:
            fixture = make_release(Path(tmp))

            with self.assertRaisesRegex(ValueError, "manifest SHA-256 mismatch"):
                generator.validate_frozen_sample(
                    fixture.dataset_root,
                    sample_id=fixture.sample_id,
                    expected_manifest_sha256="f" * 64,
                    expected_csi_sha256=file_sha256(fixture.csi_path),
                    expected_shape=fixture.csi.shape,
                )

    def test_validate_sample_rejects_path_that_escapes_dataset_root(self):
        generator = load_generator()
        with tempfile.TemporaryDirectory() as tmp:
            fixture = make_release(Path(tmp), archive_csi_path="../outside.dat")

            with self.assertRaisesRegex(ValueError, "escapes dataset root"):
                generator.validate_frozen_sample(
                    fixture.dataset_root,
                    sample_id=fixture.sample_id,
                    expected_manifest_sha256=file_sha256(fixture.manifest_path),
                    expected_csi_sha256=file_sha256(fixture.csi_path),
                    expected_shape=fixture.csi.shape,
                )

    def test_validate_sample_rejects_absolute_archive_path(self):
        generator = load_generator()
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            absolute_csi = root / "data" / "S01" / "csi" / "walk" / "sample.dat"
            fixture = make_release(root, archive_csi_path=str(absolute_csi))

            with self.assertRaisesRegex(ValueError, "must be relative"):
                generator.validate_frozen_sample(
                    fixture.dataset_root,
                    sample_id=fixture.sample_id,
                    expected_manifest_sha256=file_sha256(fixture.manifest_path),
                    expected_csi_sha256=file_sha256(fixture.csi_path),
                    expected_shape=fixture.csi.shape,
                )

    def test_validate_sample_rejects_metadata_sample_mismatch(self):
        generator = load_generator()
        with tempfile.TemporaryDirectory() as tmp:
            fixture = make_release(Path(tmp), source_sample_id="wrong-sample")

            with self.assertRaisesRegex(ValueError, "metadata sample_id mismatch"):
                generator.validate_frozen_sample(
                    fixture.dataset_root,
                    sample_id=fixture.sample_id,
                    expected_manifest_sha256=file_sha256(fixture.manifest_path),
                    expected_csi_sha256=file_sha256(fixture.csi_path),
                    expected_shape=fixture.csi.shape,
                )

    def test_validate_sample_rejects_non_integer_metadata_packet_count(self):
        generator = load_generator()
        with tempfile.TemporaryDirectory() as tmp:
            fixture = make_release(Path(tmp), metadata_packets=12.9)

            with self.assertRaisesRegex(ValueError, "metadata packet count"):
                generator.validate_frozen_sample(
                    fixture.dataset_root,
                    sample_id=fixture.sample_id,
                    expected_manifest_sha256=file_sha256(fixture.manifest_path),
                    expected_csi_sha256=file_sha256(fixture.csi_path),
                    expected_shape=fixture.csi.shape,
                )

    def test_validate_sample_rejects_decoded_shape_mismatch(self):
        generator = load_generator()
        with tempfile.TemporaryDirectory() as tmp:
            fixture = make_release(Path(tmp))
            wrong_shape = (fixture.csi.shape[0], 1, 1, fixture.csi.shape[-1])

            with self.assertRaisesRegex(ValueError, "decoded CSI shape mismatch"):
                generator.validate_frozen_sample(
                    fixture.dataset_root,
                    sample_id=fixture.sample_id,
                    expected_manifest_sha256=file_sha256(fixture.manifest_path),
                    expected_csi_sha256=file_sha256(fixture.csi_path),
                    expected_shape=wrong_shape,
                )


class SourceDataTests(unittest.TestCase):
    def test_source_csv_round_trips_float32_stages_in_canonical_order(self):
        generator = load_generator()
        stages = generator.extract_stages(
            synthetic_csi(),
            target_packets=8,
            target_subcarriers=8,
            low_energy_ratio=0.05,
            epsilon=1e-6,
        )
        sample_id = "S01_E01_A01_T01"

        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "source.csv"
            generator.write_source_data(path, sample_id, stages)
            loaded = generator.read_source_data(
                path,
                rx_count=2,
                target_packets=8,
                target_subcarriers=8,
                expected_packet_indices=stages.packet_indices,
                expected_subcarrier_indices=stages.subcarrier_indices,
            )
            with path.open("r", encoding="utf-8", newline="") as handle:
                rows = list(csv.DictReader(handle))

        self.assertEqual(len(rows), 2 * 8 * 8)
        self.assertEqual(list(rows[0]), SOURCE_FIELDS)
        np.testing.assert_array_equal(loaded.raw_amplitude, stages.raw_amplitude)
        np.testing.assert_array_equal(
            loaded.log_repaired_amplitude, stages.log_repaired_amplitude
        )
        np.testing.assert_array_equal(
            loaded.processed_zscore, stages.processed_zscore
        )
        np.testing.assert_array_equal(loaded.packet_indices, stages.packet_indices)
        np.testing.assert_array_equal(
            loaded.subcarrier_indices, stages.subcarrier_indices
        )
        self.assertEqual(loaded.sample_id, sample_id)
        observed_order = [
            (
                int(row["rx_index"]),
                int(row["output_packet_index"]),
                int(row["output_subcarrier_index"]),
            )
            for row in rows
        ]
        expected_order = [
            (rx + 1, packet, subcarrier)
            for rx in range(2)
            for packet in range(8)
            for subcarrier in range(8)
        ]
        self.assertEqual(observed_order, expected_order)
        for row in rows:
            self.assertEqual(
                int(row["source_packet_index"]),
                int(stages.packet_indices[int(row["output_packet_index"])]),
            )
            self.assertEqual(
                int(row["source_subcarrier_index"]),
                int(
                    stages.subcarrier_indices[
                        int(row["output_subcarrier_index"])
                    ]
                ),
            )

    def test_source_csv_rejects_extra_data_cell(self):
        generator = load_generator()
        stages = generator.extract_stages(
            synthetic_csi(),
            target_packets=8,
            target_subcarriers=8,
            low_energy_ratio=0.05,
            epsilon=1e-6,
        )

        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "source.csv"
            generator.write_source_data(path, "S01_E01_A01_T01", stages)
            lines = path.read_text(encoding="utf-8").splitlines()
            lines[1] += ",unexpected"
            path.write_text("\n".join(lines) + "\n", encoding="utf-8")

            with self.assertRaisesRegex(ValueError, "row width mismatch"):
                generator.read_source_data(
                    path,
                    rx_count=2,
                    target_packets=8,
                    target_subcarriers=8,
                    expected_packet_indices=stages.packet_indices,
                    expected_subcarrier_indices=stages.subcarrier_indices,
                )

    def test_source_csv_rejects_consistently_wrong_packet_index(self):
        generator = load_generator()
        stages = generator.extract_stages(
            synthetic_csi(),
            target_packets=8,
            target_subcarriers=8,
            low_energy_ratio=0.05,
            epsilon=1e-6,
        )
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "source.csv"
            generator.write_source_data(path, "S01_E01_A01_T01", stages)
            rewrite_source_rows(
                path,
                lambda rows: [
                    row.update({"source_packet_index": "999999"})
                    for row in rows
                    if row["output_packet_index"] == "0"
                ],
            )

            with self.assertRaisesRegex(ValueError, "source packet index mismatch"):
                generator.read_source_data(
                    path,
                    rx_count=2,
                    target_packets=8,
                    target_subcarriers=8,
                    expected_packet_indices=stages.packet_indices,
                    expected_subcarrier_indices=stages.subcarrier_indices,
                )

    def test_source_csv_rejects_unsorted_and_duplicate_source_indices(self):
        generator = load_generator()
        stages = generator.extract_stages(
            synthetic_csi(),
            target_packets=8,
            target_subcarriers=8,
            low_energy_ratio=0.05,
            epsilon=1e-6,
        )
        mutations = {
            "unsorted": {
                "0": str(stages.packet_indices[1]),
                "1": str(stages.packet_indices[0]),
            },
            "duplicate": {"1": str(stages.packet_indices[0])},
        }
        with tempfile.TemporaryDirectory() as tmp:
            for name, replacements in mutations.items():
                with self.subTest(name=name):
                    path = Path(tmp) / f"{name}.csv"
                    generator.write_source_data(path, "S01_E01_A01_T01", stages)

                    def mutate(rows):
                        for row in rows:
                            replacement = replacements.get(
                                row["output_packet_index"]
                            )
                            if replacement is not None:
                                row["source_packet_index"] = replacement

                    rewrite_source_rows(path, mutate)
                    with self.assertRaisesRegex(
                        ValueError, "source packet index mismatch"
                    ):
                        generator.read_source_data(
                            path,
                            rx_count=2,
                            target_packets=8,
                            target_subcarriers=8,
                            expected_packet_indices=stages.packet_indices,
                            expected_subcarrier_indices=stages.subcarrier_indices,
                        )

    def test_source_csv_rejects_noncanonical_integer(self):
        generator = load_generator()
        stages = generator.extract_stages(
            synthetic_csi(),
            target_packets=8,
            target_subcarriers=8,
            low_energy_ratio=0.05,
            epsilon=1e-6,
        )
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "source.csv"
            generator.write_source_data(path, "S01_E01_A01_T01", stages)
            rewrite_source_rows(path, lambda rows: rows[0].update({"rx_index": "+1"}))

            with self.assertRaisesRegex(ValueError, "canonical decimal integer"):
                generator.read_source_data(
                    path,
                    rx_count=2,
                    target_packets=8,
                    target_subcarriers=8,
                    expected_packet_indices=stages.packet_indices,
                    expected_subcarrier_indices=stages.subcarrier_indices,
                )

    def test_source_csv_rejects_nan_value(self):
        generator = load_generator()
        stages = generator.extract_stages(
            synthetic_csi(),
            target_packets=8,
            target_subcarriers=8,
            low_energy_ratio=0.05,
            epsilon=1e-6,
        )
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "source.csv"
            generator.write_source_data(path, "S01_E01_A01_T01", stages)
            rewrite_source_rows(
                path, lambda rows: rows[0].update({"raw_amplitude": "NaN"})
            )

            with self.assertRaisesRegex(ValueError, "non-finite source data"):
                generator.read_source_data(
                    path,
                    rx_count=2,
                    target_packets=8,
                    target_subcarriers=8,
                    expected_packet_indices=stages.packet_indices,
                    expected_subcarrier_indices=stages.subcarrier_indices,
                )

    def test_source_csv_rejects_invalid_expected_indices(self):
        generator = load_generator()
        stages = generator.extract_stages(
            synthetic_csi(),
            target_packets=8,
            target_subcarriers=8,
            low_energy_ratio=0.05,
            epsilon=1e-6,
        )
        packet_cases = {
            "not_1d": stages.packet_indices.reshape(2, 4),
            "wrong_length": stages.packet_indices[:-1],
            "not_increasing": stages.packet_indices[::-1],
            "negative": np.array([-1, 0, 2, 3, 5, 6, 8, 9]),
        }
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "source.csv"
            generator.write_source_data(path, "S01_E01_A01_T01", stages)
            for name, packet_indices in packet_cases.items():
                with self.subTest(name=name):
                    with self.assertRaisesRegex(ValueError, "expected_packet_indices"):
                        generator.read_source_data(
                            path,
                            rx_count=2,
                            target_packets=8,
                            target_subcarriers=8,
                            expected_packet_indices=packet_indices,
                            expected_subcarrier_indices=stages.subcarrier_indices,
                        )

    def test_source_writer_preserves_existing_file_for_invalid_indices(self):
        generator = load_generator()
        stages = generator.extract_stages(
            synthetic_csi(),
            target_packets=8,
            target_subcarriers=8,
            low_energy_ratio=0.05,
            epsilon=1e-6,
        )
        packet = stages.packet_indices
        subcarrier = stages.subcarrier_indices
        invalid_cases = {
            "packet_dtype": replace(stages, packet_indices=packet.astype(np.float64)),
            "packet_ndim": replace(stages, packet_indices=packet.reshape(2, 4)),
            "packet_length": replace(stages, packet_indices=packet[:-1]),
            "packet_duplicate": replace(
                stages, packet_indices=np.array([0, 0, 3, 5, 6, 8, 9, 11])
            ),
            "packet_negative": replace(
                stages, packet_indices=np.array([-1, 2, 3, 5, 6, 8, 9, 11])
            ),
            "subcarrier_dtype": replace(
                stages, subcarrier_indices=subcarrier.astype(np.float64)
            ),
            "subcarrier_ndim": replace(
                stages, subcarrier_indices=subcarrier.reshape(2, 4)
            ),
            "subcarrier_length": replace(stages, subcarrier_indices=subcarrier[:-1]),
            "subcarrier_duplicate": replace(
                stages, subcarrier_indices=np.array([0, 0, 4, 6, 9, 11, 13, 15])
            ),
            "subcarrier_negative": replace(
                stages, subcarrier_indices=np.array([-1, 2, 4, 6, 9, 11, 13, 15])
            ),
        }
        sentinel = "do not truncate\n"
        with tempfile.TemporaryDirectory() as tmp:
            for name, invalid_stages in invalid_cases.items():
                with self.subTest(name=name):
                    path = Path(tmp) / f"{name}.csv"
                    path.write_text(sentinel, encoding="utf-8")
                    with self.assertRaisesRegex(ValueError, "indices"):
                        generator.write_source_data(
                            path, "S01_E01_A01_T01", invalid_stages
                        )
                    self.assertEqual(path.read_text(encoding="utf-8"), sentinel)


class DiagnosticsTests(unittest.TestCase):
    def test_diagnostics_records_full_transform_audit_as_canonical_json(self):
        generator = load_generator()
        csi = synthetic_csi()
        stages = generator.extract_stages(
            csi,
            target_packets=8,
            target_subcarriers=8,
            low_energy_ratio=0.05,
            epsilon=1e-6,
        )

        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "diagnostics.json"
            generator.write_diagnostics(
                path,
                sample_id="S01_E01_A01_T01",
                stages=stages,
                decoded_shape=csi.shape,
                decoded_dtype=csi.dtype,
                authoritative_max_abs_difference=0.0,
                target_packets=8,
                target_subcarriers=8,
                low_energy_ratio=0.05,
                epsilon=1e-6,
            )
            raw_text = path.read_text(encoding="utf-8")
            payload = json.loads(raw_text)

        self.assertEqual(payload["decoded_csi"]["shape"], list(csi.shape))
        self.assertEqual(payload["decoded_csi"]["dtype"], "complex64")
        self.assertEqual(
            payload["selection"]["packet_indices"], stages.packet_indices.tolist()
        )
        self.assertEqual(
            payload["selection"]["subcarrier_indices"],
            stages.subcarrier_indices.tolist(),
        )
        self.assertEqual(
            payload["carrier_quality"]["valid_subcarrier_mask"],
            stages.valid_subcarrier_mask.tolist(),
        )
        self.assertEqual(
            payload["carrier_quality"]["valid_subcarrier_indices"],
            np.flatnonzero(stages.valid_subcarrier_mask).tolist(),
        )
        self.assertEqual(
            payload["carrier_quality"]["low_energy_indices"],
            stages.low_energy_indices.tolist(),
        )
        self.assertEqual(
            payload["carrier_quality"]["reference_carrier_energy"],
            stages.reference_carrier_energy,
        )
        self.assertEqual(
            payload["carrier_quality"]["low_energy_threshold"],
            stages.reference_carrier_energy * 0.05,
        )
        self.assertEqual(payload["zscore"]["mean_by_rx"], stages.zscore_mean.tolist())
        self.assertEqual(payload["zscore"]["std_by_rx"], stages.zscore_std.tolist())
        self.assertEqual(payload["authoritative_max_abs_difference"], 0.0)
        for name, values in (
            ("raw_amplitude", stages.raw_amplitude),
            ("log_repaired_amplitude", stages.log_repaired_amplitude),
            ("processed_zscore", stages.processed_zscore),
        ):
            self.assertTrue(payload["stages"][name]["finite"])
            self.assertEqual(payload["stages"][name]["min"], float(values.min()))
            self.assertEqual(payload["stages"][name]["max"], float(values.max()))
        self.assertEqual(
            payload["processing_parameters"],
            {
                "epsilon": 1e-6,
                "low_energy_ratio": 0.05,
                "target_packets": 8,
                "target_subcarriers": 8,
            },
        )
        self.assertTrue(raw_text.endswith("\n"))
        self.assertEqual(
            raw_text,
            json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        )

    def test_diagnostics_rejects_nonzero_authoritative_difference(self):
        generator = load_generator()
        csi = synthetic_csi()
        stages = generator.extract_stages(
            csi,
            target_packets=8,
            target_subcarriers=8,
            low_energy_ratio=0.05,
            epsilon=1e-6,
        )
        with tempfile.TemporaryDirectory() as tmp:
            for difference in (1e-7, -1e-7):
                with self.subTest(difference=difference):
                    path = Path(tmp) / f"diagnostics-{difference}.json"
                    with self.assertRaisesRegex(
                        ValueError, "authoritative_max_abs_difference must be zero"
                    ):
                        generator.write_diagnostics(
                            path,
                            sample_id="S01_E01_A01_T01",
                            stages=stages,
                            decoded_shape=csi.shape,
                            decoded_dtype=csi.dtype,
                            authoritative_max_abs_difference=difference,
                            target_packets=8,
                            target_subcarriers=8,
                            low_energy_ratio=0.05,
                            epsilon=1e-6,
                        )


class PlotCoordinateTests(unittest.TestCase):
    def test_panel_label_style_contrasts_with_each_column(self):
        generator = load_generator()

        self.assertEqual(generator._panel_label_style(0), ("white", "black"))
        self.assertEqual(generator._panel_label_style(1), ("black", "white"))
        with self.assertRaises(ValueError):
            generator._panel_label_style(2)

    def test_centers_to_edges_uses_midpoints_and_contains_source_centers(self):
        generator = load_generator()
        self.assertTrue(hasattr(generator, "_centers_to_edges"))
        centers = np.array([0, 8, 16, 23], dtype=np.int64)

        edges = generator._centers_to_edges(centers)

        np.testing.assert_array_equal(edges, [-4.0, 4.0, 12.0, 19.5, 26.5])
        self.assertTrue(np.all(np.diff(edges) > 0))
        self.assertTrue(np.all(centers > edges[:-1]))
        self.assertTrue(np.all(centers < edges[1:]))

    def test_plot_uses_coordinate_edges_with_pcolormesh(self):
        generator = load_generator()
        source = inspect.getsource(generator.plot_figure)
        self.assertIn(".pcolormesh(", source)
        self.assertNotIn(".imshow(", source)


class PackageTests(unittest.TestCase):
    ARTIFACT_NAMES = [
        "candidate_supplementary_csi_input.png",
        "candidate_supplementary_csi_input.pdf",
        "candidate_supplementary_csi_input_source_data.csv",
        "candidate_supplementary_csi_input_diagnostics.json",
        "candidate_supplementary_csi_input_caption.md",
        "candidate_supplementary_csi_input_generation_record.json",
        "candidate_status_update.md",
    ]
    DETERMINISTIC_NAMES = [
        name
        for name in ARTIFACT_NAMES
        if name != "candidate_supplementary_csi_input_generation_record.json"
    ]

    def package_kwargs(
        self,
        fixture: ReleaseFixture,
        output_dir: Path,
        *,
        generated_at: str,
    ) -> dict:
        command_argv = [
            sys.executable,
            str(SCRIPT_PATH),
            "--project-root",
            str(PROJECT_ROOT),
            "--dataset-root",
            str(fixture.dataset_root),
            "--sample-id",
            fixture.sample_id,
            "--output-dir",
            str(output_dir),
        ]
        return {
            "project_root": PROJECT_ROOT,
            "dataset_root": fixture.dataset_root,
            "sample_id": fixture.sample_id,
            "expected_manifest_sha256": file_sha256(fixture.manifest_path),
            "expected_csi_sha256": file_sha256(fixture.csi_path),
            "expected_shape": fixture.csi.shape,
            "target_packets": 8,
            "target_subcarriers": 8,
            "low_energy_ratio": 0.05,
            "epsilon": 1e-6,
            "output_dir": output_dir,
            "generated_at": generated_at,
            "command_argv": command_argv,
        }

    def test_generate_package_is_complete_reproducible_and_auditable(self):
        generator = load_generator()
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            fixture = make_release(root / "mini release")
            first = root / "first output"
            second = root / "second output"
            first_kwargs = self.package_kwargs(
                fixture,
                first,
                generated_at="2026-08-08T01:02:03Z",
            )
            second_kwargs = self.package_kwargs(
                fixture,
                second,
                generated_at="2026-08-08T04:05:06Z",
            )

            with warnings.catch_warnings():
                warnings.simplefilter("error")
                generator.generate_package(**first_kwargs)
                generator.generate_package(**second_kwargs)

            for directory in (first, second):
                self.assertEqual(
                    sorted(path.name for path in directory.iterdir()),
                    sorted(self.ARTIFACT_NAMES),
                )
                for name in self.ARTIFACT_NAMES:
                    self.assertTrue((directory / name).is_file())

            for name in self.DETERMINISTIC_NAMES:
                self.assertEqual(file_sha256(first / name), file_sha256(second / name))
            record_name = "candidate_supplementary_csi_input_generation_record.json"
            self.assertNotEqual(
                file_sha256(first / record_name), file_sha256(second / record_name)
            )

            caption = (first / "candidate_supplementary_csi_input_caption.md").read_text(
                encoding="utf-8"
            )
            for phrase in (
                fixture.sample_id,
                "log1p(|H|)",
                "phase",
                "single-sample preprocessing illustration",
                "matched-grid raw view",
                "authoritative order",
            ):
                self.assertIn(phrase, caption)

            record = json.loads((first / record_name).read_text(encoding="utf-8"))
            self.assertEqual(record["schema_version"], 1)
            self.assertEqual(record["provenance_status"], "sufficient")
            self.assertEqual(record["generated_at_utc"], "2026-08-08T01:02:03Z")
            self.assertEqual(record["argv"], first_kwargs["command_argv"])
            self.assertEqual(
                record["command"],
                subprocess.list2cmdline(first_kwargs["command_argv"]),
            )
            self.assertEqual(record["paths"]["project_root"], str(PROJECT_ROOT.resolve()))
            self.assertEqual(
                record["paths"]["dataset_root"], str(fixture.dataset_root.resolve())
            )
            self.assertEqual(record["paths"]["output_dir"], str(first.resolve()))
            self.assertEqual(set(record["git"]), {"commit", "branch", "dirty"})
            self.assertEqual(record["sample"]["sample_id"], fixture.sample_id)
            self.assertEqual(record["sample"]["manifest_row"]["action_id"], "walk")
            self.assertEqual(
                record["sample"]["manifest_row"]["environment_id"], "E01"
            )
            self.assertEqual(record["decoded_csi"]["shape"], list(fixture.csi.shape))
            self.assertEqual(
                record["decoded_csi"]["packet_count"], fixture.csi.shape[0]
            )
            with self.subTest(record_schema="parameters"):
                self.assertEqual(
                    record["parameters"],
                    {
                        "plotting": {
                            "dpi": 300,
                            "figure_size_inches": [6.9, 5.6],
                            "processed_center": 0,
                            "processed_colormap": "RdBu_r",
                            "raw_colormap": "viridis",
                            "shading": "flat",
                        },
                        "preprocessing": {
                            "epsilon": 1e-6,
                            "low_energy_ratio": 0.05,
                            "target_packets": 8,
                            "target_subcarriers": 8,
                        },
                    },
                )
            with self.subTest(record_schema="software"):
                self.assertIn("software", record)
                self.assertEqual(
                    set(record["software"]), {"python", "numpy", "matplotlib"}
                )
                self.assertNotIn("runtime", record)
            self.assertEqual(
                set(record["inputs"]),
                {"manifest", "csi", "metadata", "parser", "preprocessing", "generator"},
            )
            expected_input_paths = {
                "manifest": fixture.manifest_path,
                "csi": fixture.csi_path,
                "metadata": fixture.metadata_path,
                "parser": BASELINE_PACKAGE_ROOT / "axhome_csi" / "feitcsi.py",
                "preprocessing": BASELINE_PACKAGE_ROOT / "axhome_csi" / "data.py",
                "generator": SCRIPT_PATH,
            }
            for key, path in expected_input_paths.items():
                item = record["inputs"][key]
                self.assertEqual(item["path"], str(path.resolve()))
                self.assertEqual(item["sha256"], file_sha256(path))
                self.assertEqual(item["size_bytes"], path.stat().st_size)

            self.assertEqual(set(record["outputs"]), set(self.DETERMINISTIC_NAMES))
            for name, item in record["outputs"].items():
                self.assertEqual(item["sha256"], file_sha256(first / name))
                self.assertEqual(item["size_bytes"], (first / name).stat().st_size)
            self.assertEqual(len(record["limitations"]), 5)
            self.assertIn(
                "Files are replaced atomically one by one; an OS-level failure "
                "during publication can require regeneration.",
                record["limitations"],
            )
            self.assertIn(
                "Code-file hashes for the generator, parser, and preprocessing "
                "implementation are sampled when the generation record is written; "
                "concurrent code modification during a run is outside the "
                "reproducibility guarantee.",
                record["limitations"],
            )

            diagnostics = json.loads(
                (
                    first / "candidate_supplementary_csi_input_diagnostics.json"
                ).read_text(encoding="utf-8")
            )
            self.assertEqual(diagnostics["authoritative_max_abs_difference"], 0.0)
            csv_path = first / "candidate_supplementary_csi_input_source_data.csv"
            with csv_path.open("r", encoding="utf-8", newline="") as handle:
                self.assertEqual(len(list(csv.DictReader(handle))), 2 * 8 * 8)
            loaded = generator.read_source_data(
                csv_path,
                rx_count=2,
                target_packets=8,
                target_subcarriers=8,
                expected_packet_indices=_even_indices(fixture.csi.shape[0], 8),
                expected_subcarrier_indices=_even_indices(fixture.csi.shape[-1], 8),
            )
            expected = generator.extract_stages(
                fixture.csi,
                target_packets=8,
                target_subcarriers=8,
                low_energy_ratio=0.05,
                epsilon=1e-6,
            )
            np.testing.assert_array_equal(loaded.raw_amplitude, expected.raw_amplitude)
            np.testing.assert_array_equal(
                loaded.processed_zscore, expected.processed_zscore
            )
            self.assertGreater(
                (first / "candidate_supplementary_csi_input.png").stat().st_size, 0
            )
            self.assertGreater(
                (first / "candidate_supplementary_csi_input.pdf").stat().st_size, 0
            )

    def test_generation_record_uses_validated_release_input_snapshot(self):
        generator = load_generator()
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            fixture = make_release(root / "release")
            validated = generator.validate_frozen_sample(
                fixture.dataset_root,
                sample_id=fixture.sample_id,
                expected_manifest_sha256=file_sha256(fixture.manifest_path),
                expected_csi_sha256=file_sha256(fixture.csi_path),
                expected_shape=fixture.csi.shape,
            )
            release_paths = {
                "manifest": fixture.manifest_path,
                "csi": fixture.csi_path,
                "metadata": fixture.metadata_path,
            }
            expected = {
                name: {
                    "sha256": file_sha256(path),
                    "size_bytes": path.stat().st_size,
                }
                for name, path in release_paths.items()
            }
            for name, path in release_paths.items():
                path.write_bytes(f"mutated-{name}".encode("ascii"))

            staged_dir = root / "staged"
            staged_dir.mkdir()
            for name in self.DETERMINISTIC_NAMES:
                (staged_dir / name).write_bytes(name.encode("ascii"))
            record_path = staged_dir / generator.GENERATION_RECORD_NAME
            generator._write_generation_record(
                record_path,
                project_root=PROJECT_ROOT.resolve(),
                dataset_root=fixture.dataset_root.resolve(),
                output_dir=(root / "output").resolve(),
                validated_sample=validated,
                generated_at="2026-08-08T01:02:03Z",
                command_argv=["C:\\Program Files\\Python\\python.exe", "generator.py"],
                git_state={"commit": "abc", "branch": "test", "dirty": False},
                target_packets=8,
                target_subcarriers=8,
                low_energy_ratio=0.05,
                epsilon=1e-6,
                staged_dir=staged_dir,
            )
            record = json.loads(record_path.read_text(encoding="utf-8"))

            for name, expected_item in expected.items():
                with self.subTest(input_name=name):
                    self.assertEqual(
                        record["inputs"][name]["sha256"],
                        expected_item["sha256"],
                    )
                    self.assertEqual(
                        record["inputs"][name]["size_bytes"],
                        expected_item["size_bytes"],
                    )

    def test_prepublication_failure_preserves_existing_output(self):
        generator = load_generator()
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            fixture = make_release(root / "release")
            output_dir = root / "candidate output"
            output_dir.mkdir()
            sentinel = output_dir / "keep-me.txt"
            sentinel.write_text("unchanged\n", encoding="utf-8")
            kwargs = self.package_kwargs(
                fixture,
                output_dir,
                generated_at="2026-08-08T01:02:03Z",
            )
            kwargs["expected_csi_sha256"] = "0" * 64

            with self.assertRaisesRegex(ValueError, "CSI SHA-256 mismatch"):
                generator.generate_package(**kwargs)
            self.assertEqual(sentinel.read_text(encoding="utf-8"), "unchanged\n")
            self.assertEqual([path.name for path in output_dir.iterdir()], [sentinel.name])

            generator.generate_package(
                **self.package_kwargs(
                    fixture,
                    output_dir,
                    generated_at="2026-08-08T01:02:03Z",
                )
            )
            self.assertEqual(sentinel.read_text(encoding="utf-8"), "unchanged\n")
            self.assertTrue(set(self.ARTIFACT_NAMES).issubset(
                {path.name for path in output_dir.iterdir()}
            ))

    def test_cli_parser_accepts_required_paths_and_help(self):
        generator = load_generator()
        parser = generator.build_parser()
        args = parser.parse_args(
            [
                "--project-root",
                str(PROJECT_ROOT),
                "--dataset-root",
                "release",
                "--sample-id",
                generator.FROZEN_SAMPLE_ID,
                "--output-dir",
                "candidate",
            ]
        )
        self.assertEqual(args.sample_id, generator.FROZEN_SAMPLE_ID)
        stdout = io.StringIO()
        with redirect_stdout(stdout), self.assertRaises(SystemExit) as raised:
            generator.main(["--help"])
        self.assertEqual(raised.exception.code, 0)
        for option in (
            "--project-root",
            "--dataset-root",
            "--sample-id",
            "--output-dir",
        ):
            self.assertIn(option, stdout.getvalue())

        cli_argv = [
            "--project-root",
            str(PROJECT_ROOT),
            "--dataset-root",
            "mini release with spaces",
            "--sample-id",
            generator.FROZEN_SAMPLE_ID,
            "--output-dir",
            "candidate output with spaces",
        ]
        stdout = io.StringIO()
        summary = {"provenance_status": "sufficient", "artifacts": []}
        with mock.patch.object(
            generator, "generate_package", return_value=summary
        ) as generate_package, redirect_stdout(stdout):
            self.assertEqual(generator.main(cli_argv), 0)
        expected_command_argv = [
            sys.executable,
            "quality_check/generate_candidate_supplementary_csi_input.py",
            *cli_argv,
        ]
        self.assertEqual(
            generate_package.call_args.kwargs["command_argv"],
            expected_command_argv,
        )
        self.assertEqual(json.loads(stdout.getvalue()), summary)


if __name__ == "__main__":
    unittest.main()
