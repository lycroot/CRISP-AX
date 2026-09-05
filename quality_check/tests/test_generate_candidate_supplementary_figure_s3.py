import csv
import gzip
import hashlib
import importlib.util
import io
import json
import subprocess
import sys
import tempfile
import unittest
import warnings
from contextlib import redirect_stdout
from dataclasses import FrozenInstanceError
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parents[2]
SCRIPT_PATH = (
    PROJECT_ROOT / "quality_check" / "generate_candidate_supplementary_figure_s3.py"
)


def load_generator():
    spec = importlib.util.spec_from_file_location("candidate_s3_figure", SCRIPT_PATH)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot load {SCRIPT_PATH}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def synthetic_csi() -> np.ndarray:
    rng = np.random.default_rng(20260808)
    real = rng.integers(-50, 51, size=(12, 2, 1, 16), dtype=np.int16)
    imag = rng.integers(-50, 51, size=(12, 2, 1, 16), dtype=np.int16)
    real[..., 4] = 0
    imag[..., 4] = 0
    return (real.astype(np.float32) + 1j * imag.astype(np.float32)).astype(
        np.complex64
    )


def small_sample_arrays(generator):
    csi = synthetic_csi()
    return [
        generator.extract_sample_arrays(
            spec,
            csi,
            target_packets=8,
            target_subcarriers=10,
        )
        for spec in generator.FROZEN_SAMPLES
    ]


def write_test_gzip(path: Path, text: str) -> None:
    with path.open("wb") as raw:
        with gzip.GzipFile(fileobj=raw, mode="wb", filename="", mtime=0) as zipped_file:
            with io.TextIOWrapper(zipped_file, encoding="utf-8", newline="") as handle:
                handle.write(text)


def small_source_bundle(generator):
    arrays = small_sample_arrays(generator)
    with tempfile.TemporaryDirectory() as temporary_directory:
        path = Path(temporary_directory) / "source.csv.gz"
        generator.write_source_data(path, arrays)
        source = generator.read_source_data(path, arrays)
    return arrays, source, generator.compute_scale_diagnostics(arrays)


def fake_validated_samples(generator, root: Path):
    root.mkdir(parents=True, exist_ok=True)
    manifest = root / "archive_index.csv"
    manifest.write_text("frozen manifest\n", encoding="utf-8")
    validated = []
    for index, spec in enumerate(generator.FROZEN_SAMPLES):
        csi_path = root / f"{index}.dat"
        metadata_path = root / f"{index}.json"
        csi_path.write_bytes(f"csi-{index}".encode())
        metadata_path.write_text(
            json.dumps({"sample_id": spec.sample_id}), encoding="utf-8"
        )
        csi = synthetic_csi()
        validated.append(
            SimpleNamespace(
                sample_id=spec.sample_id,
                row={
                    "sample_id": spec.sample_id,
                    "person_id": spec.participant,
                    "action_id": spec.action,
                    "environment_id": spec.environment,
                    "status": "ok",
                    "sync_quality_flags": "ok",
                },
                manifest_path=manifest,
                csi_path=csi_path,
                metadata_path=metadata_path,
                manifest_sha256=generator.sha256_file(manifest),
                csi_sha256=generator.sha256_file(csi_path),
                metadata_sha256=generator.sha256_file(metadata_path),
                manifest_size_bytes=manifest.stat().st_size,
                csi_size_bytes=csi_path.stat().st_size,
                metadata_size_bytes=metadata_path.stat().st_size,
                csi=csi,
                headers=[object()] * csi.shape[0],
            )
        )
    return validated


def make_evidence_tree(generator, dataset_root: Path, evidence_root: Path):
    manifest_dir = dataset_root / "manifests"
    manifest_dir.mkdir(parents=True, exist_ok=True)
    exclusion = manifest_dir / "excluded_samples.csv"
    exclusion.write_text("sample_id,reason\nnot_selected,example\n", encoding="utf-8")
    audit_root = (
        evidence_root
        / "outputs"
        / "technical_validation_active7024_7d7ea605_full_20260801"
    )
    audit_root.mkdir(parents=True, exist_ok=True)

    csi_fields = [
        "sample_id", "person_id", "environment_id", "action_id",
        "link_geometry", "parse_status", "quality_flags",
        "ftm_gap_gt_100ms_count", "low_energy_subcarrier_count",
    ]
    sample_fields = [
        "sample_id", "person_id", "environment_id", "action_id",
        "link_geometry", "metadata_status", "csi_parse_status",
        "video_probe_status", "quality_flags", "final_status",
    ]
    metadata_fields = [
        "sample_id", "person_id", "environment_id", "action_id",
        "link_geometry", "metadata_status", "sync_status",
        "source_quality_flags", "slice_status", "quality_flags",
    ]

    def write_rows(path, fields, extra):
        with path.open("w", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=fields, lineterminator="\n")
            writer.writeheader()
            for spec in generator.FROZEN_SAMPLES:
                writer.writerow({
                    "sample_id": spec.sample_id,
                    "person_id": spec.participant,
                    "environment_id": spec.environment,
                    "action_id": spec.action,
                    "link_geometry": spec.link_configuration,
                    **extra,
                })

    write_rows(
        audit_root / "csi_audit.csv",
        csi_fields,
        {
            "parse_status": "ok",
            "quality_flags": "ok",
            "ftm_gap_gt_100ms_count": "0",
            "low_energy_subcarrier_count": "16",
        },
    )
    write_rows(
        audit_root / "sample_audit.csv",
        sample_fields,
        {
            "metadata_status": "ok",
            "csi_parse_status": "ok",
            "video_probe_status": "ok",
            "quality_flags": "ok",
            "final_status": "ok",
        },
    )
    write_rows(
        audit_root / "metadata_audit.csv",
        metadata_fields,
        {
            "metadata_status": "ok",
            "sync_status": "ok",
            "source_quality_flags": "ok",
            "slice_status": "ok",
            "quality_flags": "ok",
        },
    )
    recheck_root = (
        evidence_root / "outputs" / "recheck_csi_continuity_20260807_212045"
    )
    recheck_root.mkdir(parents=True, exist_ok=True)
    (recheck_root / "recheck_summary.json").write_text(
        json.dumps({"files_with_gap_gt_100ms": [], "errors": []}),
        encoding="utf-8",
    )
    paths = generator.resolve_evidence_paths(dataset_root, evidence_root)
    hashes = {name: generator.sha256_file(path) for name, path in paths.items()}
    return paths, hashes


class FrozenConfigurationTests(unittest.TestCase):
    def test_four_ordered_sample_roles_and_preprocessing_parameters_are_frozen(self):
        generator = load_generator()

        expected = [
            (
                "preprocessing_representative",
                "S14_P05_walk_face_rx_E1_clean_none_R006",
                "P05",
                "walk",
                "E1",
                "face_rx",
                (496, 2, 1, 996),
                "a301478f6be1e6ecb340fb6e7ac878851f58208e59e40ff7c6f09a597bbc2606",
            ),
            (
                "diversity_a",
                "S13_P04_walk_face_rx_E1_clean_none_R014",
                "P04",
                "walk",
                "E1",
                "face_rx",
                (495, 2, 1, 996),
                "23e3191f8d5a06cf7a326866e7951adba6b64018035a65915b45118acbce1623",
            ),
            (
                "diversity_b",
                "S03_P01_lie_down_side_link_E2_clean_none_R004",
                "P01",
                "lie_down",
                "E2",
                "side_link",
                (495, 2, 1, 996),
                "d144b9b6b5140158df7d87b8f44160d59891434942259a28f66cd27489d54777",
            ),
            (
                "diversity_c",
                "S23_P08_wash_hands_cross_link_E3_clean_none_R025",
                "P08",
                "wash_hands",
                "E3",
                "cross_link",
                (495, 2, 1, 996),
                "d2af36f96b87513dcf897f4fab4e539ef7c33f95b11915b4f80e348d326f44e7",
            ),
        ]
        actual = [
            (
                item.role,
                item.sample_id,
                item.participant,
                item.action,
                item.environment,
                item.link_configuration,
                item.expected_shape,
                item.expected_csi_sha256,
            )
            for item in generator.FROZEN_SAMPLES
        ]

        self.assertEqual(actual, expected)
        self.assertEqual(
            generator.FROZEN_MANIFEST_SHA256,
            "7d7ea605819c2c988b068b3f86ed77419c5a801c0d49b279b74c671cfb654a00",
        )
        self.assertEqual(generator.TARGET_PACKETS, 256)
        self.assertEqual(generator.TARGET_SUBCARRIERS, 128)
        self.assertEqual(generator.LOW_ENERGY_RATIO, 0.05)
        self.assertEqual(generator.EPSILON, 1e-6)
        self.assertEqual(len({item.participant for item in generator.FROZEN_SAMPLES}), 4)
        with self.assertRaises(FrozenInstanceError):
            generator.FROZEN_SAMPLES[0].role = "changed"

    def test_package_output_names_are_distinct_and_complete(self):
        generator = load_generator()

        self.assertEqual(len(generator.PACKAGE_OUTPUT_NAMES), 8)
        self.assertEqual(len(set(generator.PACKAGE_OUTPUT_NAMES)), 8)
        self.assertIn("candidate_supplementary_figure_s3.png", generator.PACKAGE_OUTPUT_NAMES)
        self.assertIn("candidate_supplementary_figure_s3.pdf", generator.PACKAGE_OUTPUT_NAMES)
        self.assertIn(
            "candidate_supplementary_figure_s3_source_data.csv.gz",
            generator.PACKAGE_OUTPUT_NAMES,
        )
        self.assertIn(
            "candidate_supplementary_figure_s3_generation_record.json",
            generator.PACKAGE_OUTPUT_NAMES,
        )


class SampleExtractionTests(unittest.TestCase):
    def test_native_raw_orientation_and_processed_values_match_authoritative_chain(self):
        generator = load_generator()
        csi = synthetic_csi()

        arrays = generator.extract_sample_arrays(
            generator.FROZEN_SAMPLES[0],
            csi,
            target_packets=8,
            target_subcarriers=10,
        )

        expected_raw = np.transpose(np.abs(csi[:, :, 0, :]), (1, 0, 2)).astype(
            np.float32
        )
        expected_processed = generator.preprocess_amplitude(
            csi,
            target_packets=8,
            target_subcarriers=10,
            low_energy_ratio=generator.LOW_ENERGY_RATIO,
            epsilon=generator.EPSILON,
        )
        self.assertEqual(arrays.native_raw_amplitude.shape, (2, 12, 16))
        self.assertEqual(arrays.native_raw_amplitude.dtype, np.float32)
        self.assertTrue(np.array_equal(arrays.native_raw_amplitude, expected_raw))
        self.assertEqual(arrays.stages.processed_zscore.shape, (2, 8, 10))
        self.assertEqual(arrays.stages.processed_zscore.dtype, np.float32)
        self.assertTrue(
            np.array_equal(arrays.stages.processed_zscore, expected_processed)
        )
        self.assertEqual(arrays.authoritative_max_abs_difference, 0.0)
        with self.assertRaises(FrozenInstanceError):
            arrays.authoritative_max_abs_difference = 1.0

    def test_each_processed_array_is_compared_fail_closed_to_authoritative_result(self):
        generator = load_generator()
        csi = synthetic_csi()
        authoritative = generator.preprocess_amplitude(
            csi,
            target_packets=8,
            target_subcarriers=10,
            low_energy_ratio=generator.LOW_ENERGY_RATIO,
            epsilon=generator.EPSILON,
        )
        changed = authoritative.copy()
        changed[0, 0, 0] = np.nextafter(changed[0, 0, 0], np.float32(np.inf))

        with mock.patch.object(generator, "preprocess_amplitude", return_value=changed):
            with self.assertRaisesRegex(
                RuntimeError,
                generator.FROZEN_SAMPLES[0].sample_id,
            ):
                generator.extract_sample_arrays(
                    generator.FROZEN_SAMPLES[0],
                    csi,
                    target_packets=8,
                    target_subcarriers=10,
                )

    def test_invalid_csi_or_non_finite_arrays_are_rejected_with_sample_identity(self):
        generator = load_generator()
        csi = synthetic_csi()
        bad_tx = np.repeat(csi, 2, axis=2)
        with self.assertRaisesRegex(ValueError, generator.FROZEN_SAMPLES[1].sample_id):
            generator.extract_sample_arrays(
                generator.FROZEN_SAMPLES[1],
                bad_tx,
                target_packets=8,
                target_subcarriers=10,
            )

        bad_finite = csi.copy()
        bad_finite[0, 0, 0, 0] = np.complex64(np.nan + 0j)
        with self.assertRaisesRegex(ValueError, generator.FROZEN_SAMPLES[2].sample_id):
            generator.extract_sample_arrays(
                generator.FROZEN_SAMPLES[2],
                bad_finite,
                target_packets=8,
                target_subcarriers=10,
            )


class ScalePolicyTests(unittest.TestCase):
    def make_arrays(self, generator, raw, processed_values):
        return [
            SimpleNamespace(
                spec=spec,
                native_raw_amplitude=raw if index == 0 else np.ones((2, 3, 4)),
                stages=SimpleNamespace(processed_zscore=processed),
            )
            for index, (spec, processed) in enumerate(
                zip(generator.FROZEN_SAMPLES, processed_values, strict=True)
            )
        ]

    def test_percentile_limits_and_pooled_and_per_sample_clipping_are_exact(self):
        generator = load_generator()
        raw = np.arange(200, dtype=np.float32).reshape(2, 10, 10)
        processed = [
            np.linspace(-4.0 + index, 4.0 + index, 160, dtype=np.float32).reshape(
                2, 8, 10
            )
            for index in range(4)
        ]
        arrays = self.make_arrays(generator, raw, processed)

        diagnostics = generator.compute_scale_diagnostics(arrays)

        expected_raw_max = float(np.percentile(raw, 99.5))
        pooled_abs = np.concatenate([np.abs(item).ravel() for item in processed])
        expected_processed_limit = float(np.percentile(pooled_abs, 99.0))
        self.assertEqual(diagnostics.raw_vmin, 0.0)
        self.assertEqual(diagnostics.raw_vmax, expected_raw_max)
        self.assertEqual(diagnostics.raw_percentile, 99.5)
        self.assertEqual(
            diagnostics.raw_clipped_fraction,
            float(np.count_nonzero(raw > expected_raw_max) / raw.size),
        )
        self.assertEqual(diagnostics.processed_limit, expected_processed_limit)
        self.assertEqual(diagnostics.processed_percentile, 99.0)
        self.assertEqual(
            diagnostics.processed_clipped_fraction,
            float(
                np.count_nonzero(pooled_abs > expected_processed_limit)
                / pooled_abs.size
            ),
        )
        self.assertEqual(
            diagnostics.processed_clipped_fraction_by_sample,
            {
                spec.sample_id: float(
                    np.count_nonzero(np.abs(values) > expected_processed_limit)
                    / values.size
                )
                for spec, values in zip(
                    generator.FROZEN_SAMPLES, processed, strict=True
                )
            },
        )
        with self.assertRaises(FrozenInstanceError):
            diagnostics.processed_limit = 1.0

    def test_non_finite_empty_or_non_positive_scale_inputs_fail_closed(self):
        generator = load_generator()
        raw = np.ones((2, 3, 4), dtype=np.float32)
        processed = [np.ones((2, 3, 4), dtype=np.float32) for _ in range(4)]
        arrays = self.make_arrays(generator, raw, processed)

        for label, mutate in (
            ("raw", lambda: setattr(arrays[0], "native_raw_amplitude", np.array([np.nan]))),
            (
                "processed",
                lambda: setattr(
                    arrays[2].stages,
                    "processed_zscore",
                    np.array([np.inf]),
                ),
            ),
        ):
            with self.subTest(label=label):
                local = self.make_arrays(generator, raw, processed)
                arrays = local
                mutate()
                with self.assertRaisesRegex(ValueError, label):
                    generator.compute_scale_diagnostics(arrays)

        zero = self.make_arrays(
            generator,
            np.zeros((2, 3, 4), dtype=np.float32),
            [np.zeros((2, 3, 4), dtype=np.float32) for _ in range(4)],
        )
        with self.assertRaisesRegex(ValueError, "positive"):
            generator.compute_scale_diagnostics(zero)

    def test_scale_policy_requires_all_four_frozen_samples_in_order(self):
        generator = load_generator()
        arrays = self.make_arrays(
            generator,
            np.ones((2, 3, 4), dtype=np.float32),
            [np.arange(24, dtype=np.float32).reshape(2, 3, 4) for _ in range(4)],
        )

        with self.assertRaisesRegex(ValueError, "four frozen samples"):
            generator.compute_scale_diagnostics(arrays[:3])
        arrays[1].spec = generator.FROZEN_SAMPLES[2]
        with self.assertRaisesRegex(ValueError, "frozen sample order"):
            generator.compute_scale_diagnostics(arrays)


class SourceDataTests(unittest.TestCase):
    def test_source_data_is_deterministic_and_round_trips_every_float32_value(self):
        generator = load_generator()
        arrays = small_sample_arrays(generator)

        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            first = root / "first.csv.gz"
            second = root / "second.csv.gz"
            generator.write_source_data(first, arrays)
            generator.write_source_data(second, arrays)

            self.assertEqual(first.read_bytes(), second.read_bytes())
            self.assertEqual(first.read_bytes()[4:8], b"\x00\x00\x00\x00")
            bundle = generator.read_source_data(first, arrays)

        self.assertEqual(bundle.row_count, 2 * 12 * 16 + 4 * 2 * 8 * 10)
        self.assertTrue(
            np.array_equal(
                bundle.native_raw_amplitude,
                arrays[0].native_raw_amplitude,
            )
        )
        for item in arrays:
            sample_id = item.spec.sample_id
            self.assertTrue(
                np.array_equal(
                    bundle.processed_by_sample[sample_id],
                    item.stages.processed_zscore,
                )
            )
            self.assertTrue(
                np.array_equal(
                    bundle.packet_indices_by_sample[sample_id],
                    item.stages.packet_indices,
                )
            )
            self.assertTrue(
                np.array_equal(
                    bundle.subcarrier_indices_by_sample[sample_id],
                    item.stages.subcarrier_indices,
                )
            )

    def test_source_rows_use_frozen_roles_and_exact_output_and_source_coordinates(self):
        generator = load_generator()
        arrays = small_sample_arrays(generator)

        with tempfile.TemporaryDirectory() as temporary_directory:
            path = Path(temporary_directory) / "source.csv.gz"
            generator.write_source_data(path, arrays)
            with gzip.open(path, "rt", encoding="utf-8", newline="") as handle:
                lines = handle.read().splitlines()

        self.assertEqual(lines[0].split(","), list(generator.SOURCE_DATA_FIELDS))
        raw_first = lines[1].split(",")
        self.assertEqual(
            raw_first[:8],
            [
                "preprocessing_representative",
                generator.FROZEN_SAMPLES[0].sample_id,
                "raw_amplitude_native",
                "1",
                "0",
                "0",
                "0",
                "0",
            ],
        )
        self.assertEqual(
            raw_first[8],
            format(float(arrays[0].native_raw_amplitude[0, 0, 0]), ".9g"),
        )
        processed_start = 1 + 2 * 12 * 16
        processed_first = lines[processed_start].split(",")
        self.assertEqual(processed_first[:6], [
            "preprocessing_representative",
            generator.FROZEN_SAMPLES[0].sample_id,
            "processed_zscore",
            "1",
            "0",
            str(arrays[0].stages.packet_indices[0]),
        ])
        self.assertEqual(processed_first[6:8], [
            "0",
            str(arrays[0].stages.subcarrier_indices[0]),
        ])

    def test_source_reader_rejects_extra_columns_duplicates_and_tampered_values(self):
        generator = load_generator()
        arrays = small_sample_arrays(generator)

        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            good = root / "good.csv.gz"
            generator.write_source_data(good, arrays)
            with gzip.open(good, "rt", encoding="utf-8", newline="") as handle:
                text = handle.read()
            lines = text.splitlines()

            cases = {
                "extra columns": text + lines[1] + ",unexpected\n",
                "duplicate": text + lines[1] + "\n",
                "value mismatch": text.replace(lines[1], lines[1][:-1] + "1", 1),
            }
            for label, content in cases.items():
                with self.subTest(label=label):
                    bad = root / f"{hashlib.sha256(label.encode()).hexdigest()}.csv.gz"
                    write_test_gzip(bad, content)
                    with self.assertRaisesRegex(ValueError, label):
                        generator.read_source_data(bad, arrays)

    def test_source_schema_is_deterministic_and_documents_indexing(self):
        generator = load_generator()

        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            first = root / "first.json"
            second = root / "second.json"
            generator.write_source_schema(first)
            generator.write_source_schema(second)
            self.assertEqual(first.read_bytes(), second.read_bytes())
            payload = json.loads(first.read_text(encoding="utf-8"))

        self.assertEqual(payload["schema_version"], 1)
        self.assertEqual(payload["fields"], list(generator.SOURCE_DATA_FIELDS))
        self.assertEqual(payload["rx_index"], "one-based")
        self.assertEqual(payload["output_indices"], "zero-based")
        self.assertEqual(payload["source_indices"], "zero-based")
        self.assertIn("float32", payload["value"])


class FigureLayoutTests(unittest.TestCase):
    def test_build_figure_has_frozen_layout_shared_scales_and_source_edges(self):
        generator = load_generator()
        arrays, source, scales = small_source_bundle(generator)

        figure = generator.build_figure(source, scales)
        try:
            self.assertEqual(tuple(figure.get_size_inches()), (10.5, 6.0))
            self.assertEqual(len(figure.axes), 12)
            heatmaps = figure.axes[:10]
            meshes = [axis.collections[0] for axis in heatmaps]
            self.assertTrue(all(mesh.get_rasterized() for mesh in meshes))
            self.assertEqual([mesh.get_cmap().name for mesh in meshes[:2]], [
                "cividis",
                "RdBu_r",
            ])
            self.assertEqual([mesh.get_cmap().name for mesh in meshes[2:5]], [
                "RdBu_r",
                "RdBu_r",
                "RdBu_r",
            ])
            self.assertEqual([mesh.get_cmap().name for mesh in meshes[5:]], [
                "cividis",
                "RdBu_r",
                "RdBu_r",
                "RdBu_r",
                "RdBu_r",
            ])
            self.assertEqual(meshes[0].get_clim(), (0.0, scales.raw_vmax))
            self.assertEqual(meshes[5].get_clim(), (0.0, scales.raw_vmax))
            for mesh in [*meshes[1:5], *meshes[6:]]:
                self.assertEqual(
                    mesh.get_clim(),
                    (-scales.processed_limit, scales.processed_limit),
                )

            raw_coordinates = meshes[0].get_coordinates()
            self.assertTrue(np.array_equal(raw_coordinates[0, :, 0], np.arange(17) - 0.5))
            self.assertTrue(np.array_equal(raw_coordinates[:, 0, 1], np.arange(13) - 0.5))
            processed_coordinates = meshes[1].get_coordinates()
            self.assertTrue(
                np.array_equal(
                    processed_coordinates[0, :, 0],
                    generator._centers_to_edges(arrays[0].stages.subcarrier_indices),
                )
            )
            self.assertTrue(
                np.array_equal(
                    processed_coordinates[:, 0, 1],
                    generator._centers_to_edges(arrays[0].stages.packet_indices),
                )
            )

            self.assertEqual(heatmaps[0].get_title(), "Raw CSI amplitude")
            self.assertEqual(
                heatmaps[1].get_title(),
                "2D CNN input\nrepresentation",
            )
            self.assertEqual(heatmaps[2].get_title(), "P04 | walk\nE1 | face_rx")
            self.assertEqual(heatmaps[3].get_title(), "P01 | lie_down\nE2 | side_link")
            self.assertEqual(heatmaps[4].get_title(), "P08 | wash_hands\nE3 | cross_link")
            figure_text = {item.get_text() for item in figure.texts}
            self.assertIn("A  Preprocessing demonstration", figure_text)
            self.assertIn("B  Representative sample diversity", figure_text)
            self.assertIn("RX1", figure_text)
            self.assertIn("RX2", figure_text)
            self.assertIn("Source subcarrier index", figure_text)
            self.assertIn("Source packet index", figure_text)
            self.assertGreater(figure.axes[10].get_position().width, figure.axes[10].get_position().height)
            self.assertGreater(figure.axes[11].get_position().width, figure.axes[11].get_position().height)
            self.assertTrue(all(not label.get_visible() for label in heatmaps[0].get_xticklabels()))
            self.assertTrue(all(label.get_visible() for label in heatmaps[5].get_xticklabels()))
            self.assertTrue(all(not label.get_visible() for label in heatmaps[1].get_yticklabels()))
            self.assertTrue(heatmaps[0].yaxis_inverted())
        finally:
            generator.close_figure(figure)

    def test_plot_outputs_are_3150_by_1800_and_byte_deterministic(self):
        generator = load_generator()
        _, source, scales = small_source_bundle(generator)

        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            outputs = []
            with warnings.catch_warnings():
                warnings.simplefilter("error")
                for prefix in ("first", "second"):
                    png = root / f"{prefix}.png"
                    pdf = root / f"{prefix}.pdf"
                    generator.plot_figure(source, scales, png, pdf)
                    outputs.append((png, pdf))
            self.assertEqual(outputs[0][0].read_bytes(), outputs[1][0].read_bytes())
            self.assertEqual(outputs[0][1].read_bytes(), outputs[1][1].read_bytes())
            import matplotlib.image as mpimg

            image = mpimg.imread(outputs[0][0])
            self.assertEqual(image.shape[:2], (1800, 3150))
            self.assertGreater(outputs[0][1].stat().st_size, 1_000)
            pdf_bytes = outputs[0][1].read_bytes()
            self.assertNotIn(b"/Subtype /Type3", pdf_bytes)
            self.assertIn(b"/FontFile2", pdf_bytes)

    def test_figure6_layout_is_architecture_neutral_aligned_and_compact(self):
        generator = load_generator()
        _, source, scales = small_source_bundle(generator)
        figure = generator.build_figure6(source, scales)
        try:
            layout = generator.validate_figure6_layout(figure)
            coordinates = generator.validate_v4_coordinate_alignment(
                figure, source
            )
            self.assertEqual(layout["status"], "pass")
            self.assertEqual(coordinates["status"], "pass")
            self.assertEqual(
                tuple(figure.get_size_inches()),
                (generator.FIGURE6_WIDTH_IN, generator.FIGURE6_HEIGHT_IN),
            )
            text_by_value = {item.get_text(): item for item in figure.texts}
            self.assertIn("Model input", text_by_value)
            self.assertIn("representation", text_by_value)
            self.assertFalse(
                any("2D CNN input" in item.get_text() for item in figure.texts)
            )
            self.assertEqual(
                text_by_value["Source packet index"].get_text(),
                "Source packet index",
            )
            self.assertEqual(
                text_by_value["Source subcarrier index"].get_text(),
                "Source subcarrier index",
            )
            first_title_y = {
                text_by_value[value].get_position()[1]
                for value in (
                    "Raw CSI",
                    "Model input",
                    "Walk (P04)",
                    "Lie down (P01)",
                    "Wash hands (P08)",
                )
            }
            second_title_y = {
                text_by_value[value].get_position()[1]
                for value in (
                    "amplitude",
                    "representation",
                    "E1 · face_rx",
                    "E2 · side_link",
                    "E3 · cross_link",
                )
            }
            self.assertEqual(len(first_title_y), 1)
            self.assertEqual(len(second_title_y), 1)
            raw_colorbar = figure.axes[10].get_position()
            processed_colorbar = figure.axes[11].get_position()
            self.assertAlmostEqual(
                raw_colorbar.x0,
                figure.axes[0].get_position().x0,
            )
            self.assertAlmostEqual(
                raw_colorbar.x1,
                figure.axes[0].get_position().x1,
            )
            self.assertAlmostEqual(
                processed_colorbar.x0,
                figure.axes[1].get_position().x0,
            )
            self.assertAlmostEqual(
                processed_colorbar.x1,
                figure.axes[4].get_position().x1,
            )
            self.assertGreater(
                processed_colorbar.width,
                raw_colorbar.width,
            )
            self.assertFalse(layout["colorbars_equal_length"])
            range_review = layout["source_packet_range_review"]
            self.assertFalse(range_review["ranges_identical"])
            self.assertEqual(range_review["panel_a"], [-0.5, 495.5])
            self.assertEqual(range_review["panel_b"], [-0.5, 494.5])
            self.assertEqual(range_review["difference_packets"], 1)
            self.assertAlmostEqual(
                range_review["relative_difference_percent"],
                100.0 / 496.0,
            )
            self.assertEqual(
                range_review["figure_policy"],
                "shared y tick labels on leftmost column only",
            )
            for axis_index, axis in enumerate(figure.axes[:10]):
                expected_labels = (
                    ["0", "200", "400"]
                    if axis_index % 5 == 0
                    else []
                )
                self.assertEqual(
                    [
                        item.get_text()
                        for item in axis.get_yticklabels()
                        if item.get_visible()
                    ],
                    expected_labels,
                )
            for axis in figure.axes[:10]:
                self.assertFalse(axis.spines["top"].get_visible())
                self.assertFalse(axis.spines["right"].get_visible())
            self.assertTrue(
                any(
                    artist.get_gid() == "figure6-panel-separator"
                    for artist in figure.artists
                )
            )
            self.assertEqual(
                generator.PUBLICATION_FONT_CANDIDATES,
                ("Arial", "DejaVu Sans"),
            )
        finally:
            generator.close_figure(figure)

    def test_figure6_outputs_are_7_point_2_inches_and_keep_vector_text(self):
        generator = load_generator()
        _, source, scales = small_source_bundle(generator)

        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            png = root / "figure6.png"
            pdf = root / "figure6.pdf"
            svg = root / "figure6.svg"
            validation = root / "validation.json"
            with warnings.catch_warnings():
                warnings.simplefilter("error")
                summary = generator.plot_figure6(
                    source, scales, png, pdf, svg, validation
                )
            import matplotlib.image as mpimg

            image = mpimg.imread(png)
            self.assertEqual(image.shape[:2], (1050, 2160))
            self.assertEqual(summary["status"], "pass")
            self.assertEqual(
                json.loads(validation.read_text(encoding="utf-8"))["status"],
                "pass",
            )
            svg_text = svg.read_text(encoding="utf-8")
            self.assertIn("Model input", svg_text)
            self.assertIn("representation", svg_text)
            self.assertNotIn("2D CNN input", svg_text)
            self.assertGreater(pdf.stat().st_size, 1_000)
            self.assertNotIn(b"/Subtype /Type3", pdf.read_bytes())


class ReportTests(unittest.TestCase):
    def test_caption_and_status_state_supported_claims_and_explicit_boundaries(self):
        generator = load_generator()
        arrays, _, scales = small_source_bundle(generator)

        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            validated = fake_validated_samples(generator, root)
            caption = root / "caption.md"
            status = root / "status.md"
            generator.write_caption(caption, validated, scales)
            generator.write_status_update(status, validated)
            caption_text = caption.read_text(encoding="utf-8")
            status_text = status.read_text(encoding="utf-8")

        for item in arrays:
            self.assertIn(item.spec.sample_id, caption_text)
        self.assertIn("native-resolution raw amplitude", caption_text)
        self.assertIn("exact amplitude-based 2D CNN input", caption_text)
        self.assertIn("99.5th percentile", caption_text)
        self.assertIn("99th percentile", caption_text)
        self.assertIn("does not estimate the dataset distribution", caption_text)
        self.assertIn("does not identify actions", caption_text)
        self.assertIn("does not provide model-performance evidence", caption_text)
        self.assertIn("phase", caption_text.lower())
        self.assertIn("中文核对译文", caption_text)
        self.assertIn("provenance", status_text.lower())
        self.assertIn("candidate", status_text.lower())
        self.assertIn("尚未", status_text)

    def test_diagnostics_records_every_stage_shape_coordinate_and_scale(self):
        generator = load_generator()
        arrays, source, scales = small_source_bundle(generator)

        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            validated = fake_validated_samples(generator, root)
            path = root / "diagnostics.json"
            generator.write_diagnostics(path, validated, arrays, source, scales)
            payload = json.loads(path.read_text(encoding="utf-8"))

        self.assertEqual(payload["schema_version"], 1)
        self.assertEqual(payload["provenance_status"], "sufficient")
        self.assertEqual(payload["source_data_row_count"], source.row_count)
        self.assertEqual(payload["scales"]["raw"]["percentile"], 99.5)
        self.assertEqual(payload["scales"]["processed"]["percentile"], 99.0)
        self.assertEqual(
            payload["scales"]["processed"]["limit"], scales.processed_limit
        )
        self.assertEqual(len(payload["samples"]), 4)
        for validated_sample, arrays_item, record in zip(
            validated, arrays, payload["samples"], strict=True
        ):
            self.assertEqual(record["sample_id"], validated_sample.sample_id)
            self.assertEqual(record["decoded_shape"], list(validated_sample.csi.shape))
            self.assertEqual(record["native_raw_shape"], list(arrays_item.native_raw_amplitude.shape))
            self.assertEqual(record["processed_shape"], list(arrays_item.stages.processed_zscore.shape))
            self.assertEqual(record["authoritative_max_abs_difference"], 0.0)
            self.assertEqual(record["packet_indices"], arrays_item.stages.packet_indices.tolist())
            self.assertEqual(record["subcarrier_indices"], arrays_item.stages.subcarrier_indices.tolist())
            self.assertEqual(record["low_energy_indices"], arrays_item.stages.low_energy_indices.tolist())

    def test_generation_record_binds_inputs_code_evidence_outputs_and_argv(self):
        generator = load_generator()
        arrays, source, scales = small_source_bundle(generator)

        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            validated = fake_validated_samples(generator, root)
            evidence_paths = {}
            for name in (
                "active_csi_audit",
                "active_sample_audit",
                "active_metadata_audit",
                "independent_ftm_recheck",
            ):
                path = root / f"{name}.txt"
                path.write_text(name, encoding="utf-8")
                evidence_paths[name] = path
            evidence_hashes = {
                name: generator.sha256_file(path)
                for name, path in evidence_paths.items()
            }
            staged = root / "staged"
            staged.mkdir()
            for name in generator.DETERMINISTIC_OUTPUT_NAMES:
                (staged / name).write_bytes(name.encode())
            record_path = staged / generator.GENERATION_RECORD_NAME
            argv = ["python", "quality_check/generate_candidate_supplementary_figure_s3.py", "--project-root", "X Y"]
            generator.write_generation_record(
                record_path,
                project_root=PROJECT_ROOT,
                dataset_root=root,
                output_dir=root / "output",
                validated_samples=validated,
                arrays_by_sample=arrays,
                source_data=source,
                scales=scales,
                evidence_paths=evidence_paths,
                expected_evidence_hashes=evidence_hashes,
                generated_at="2026-08-08T00:00:00Z",
                command_argv=argv,
                git_state={"commit": "abc", "branch": "codex/test", "dirty": False},
                staged_dir=staged,
            )
            payload = json.loads(record_path.read_text(encoding="utf-8"))

            evidence_paths["active_csi_audit"].write_text(
                "changed", encoding="utf-8"
            )
            with self.assertRaisesRegex(ValueError, "active_csi_audit.*changed"):
                generator.write_generation_record(
                    root / "changed-record.json",
                    project_root=PROJECT_ROOT,
                    dataset_root=root,
                    output_dir=root / "output",
                    validated_samples=validated,
                    arrays_by_sample=arrays,
                    source_data=source,
                    scales=scales,
                    evidence_paths=evidence_paths,
                    expected_evidence_hashes=evidence_hashes,
                    generated_at="2026-08-08T00:00:00Z",
                    command_argv=argv,
                    git_state={"commit": "abc", "branch": "codex/test", "dirty": False},
                    staged_dir=staged,
                )

        self.assertEqual(payload["schema_version"], 1)
        self.assertEqual(payload["provenance_status"], "sufficient")
        self.assertEqual(payload["argv"], argv)
        self.assertEqual(payload["git"]["commit"], "abc")
        self.assertEqual(len(payload["samples"]), 4)
        self.assertEqual(
            [item["sample_id"] for item in payload["samples"]],
            [item.sample_id for item in generator.FROZEN_SAMPLES],
        )
        self.assertEqual(set(payload["inputs"]["evidence"]), set(evidence_paths))
        self.assertEqual(
            set(payload["inputs"]["code"]),
            {
                "parser",
                "cli",
                "preprocessing",
                "training",
                "model_architectures",
                "model_input",
                "generator",
                "selection_design",
            },
        )
        self.assertEqual(
            set(payload["outputs"]), set(generator.DETERMINISTIC_OUTPUT_NAMES)
        )
        self.assertEqual(payload["parameters"]["plotting"]["raw_percentile"], 99.5)
        self.assertEqual(payload["parameters"]["plotting"]["processed_percentile"], 99.0)
        self.assertEqual(payload["parameters"]["plotting"]["figure_size_inches"], [10.5, 6.0])
        self.assertEqual(payload["parameters"]["preprocessing"]["target_packets"], 256)
        self.assertEqual(payload["source_data"]["row_count"], source.row_count)
        self.assertIn("python", payload["software"])
        self.assertIn("numpy", payload["software"])
        self.assertIn("matplotlib", payload["software"])
        self.assertGreaterEqual(len(payload["limitations"]), 5)


class EvidenceValidationTests(unittest.TestCase):
    def test_hash_and_audit_parse_are_bound_to_one_evidence_snapshot(self):
        generator = load_generator()
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            paths, hashes = make_evidence_tree(
                generator, root / "dataset", root / "evidence"
            )
            original_reader = generator._read_selected_audit_rows
            mutated = False

            def mutate_then_read(source, *args, **kwargs):
                nonlocal mutated
                if kwargs["audit_name"] == "active_csi_audit" and not mutated:
                    text = paths["active_csi_audit"].read_text(encoding="utf-8")
                    paths["active_csi_audit"].write_text(
                        text.replace(",ok,ok,0,16", ",ok,warning,0,16", 1),
                        encoding="utf-8",
                    )
                    mutated = True
                return original_reader(source, *args, **kwargs)

            with mock.patch.object(
                generator, "_read_selected_audit_rows", side_effect=mutate_then_read
            ):
                rows = generator.validate_selection_evidence(paths, hashes)

            self.assertEqual(len(rows), 4)
            self.assertNotEqual(
                generator.sha256_file(paths["active_csi_audit"]),
                hashes["active_csi_audit"],
            )

    def test_selection_evidence_hashes_identities_and_quality_fields_are_validated(self):
        generator = load_generator()
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            dataset_root = root / "dataset"
            evidence_root = root / "evidence"
            paths, hashes = make_evidence_tree(generator, dataset_root, evidence_root)

            rows = generator.validate_selection_evidence(paths, hashes)
            self.assertEqual(
                [row["sample_id"] for row in rows],
                [item.sample_id for item in generator.FROZEN_SAMPLES],
            )

            with self.subTest("hash mismatch"):
                bad_hashes = dict(hashes)
                bad_hashes["active_csi_audit"] = "0" * 64
                with self.assertRaisesRegex(ValueError, "active_csi_audit.*SHA-256"):
                    generator.validate_selection_evidence(paths, bad_hashes)

            with self.subTest("quality mismatch"):
                text = paths["active_csi_audit"].read_text(encoding="utf-8")
                paths["active_csi_audit"].write_text(
                    text.replace(",ok,ok,0,16", ",ok,warning,0,16", 1),
                    encoding="utf-8",
                )
                changed_hashes = dict(hashes)
                changed_hashes["active_csi_audit"] = generator.sha256_file(
                    paths["active_csi_audit"]
                )
                with self.assertRaisesRegex(ValueError, "quality"):
                    generator.validate_selection_evidence(paths, changed_hashes)

    def test_selected_sample_in_ftm_warning_or_exclusion_ledger_is_rejected(self):
        generator = load_generator()
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            paths, hashes = make_evidence_tree(generator, root / "dataset", root / "evidence")
            sample_id = generator.FROZEN_SAMPLES[0].sample_id

            paths["independent_ftm_recheck"].write_text(
                json.dumps({"files_with_gap_gt_100ms": [sample_id], "errors": []}),
                encoding="utf-8",
            )
            hashes["independent_ftm_recheck"] = generator.sha256_file(
                paths["independent_ftm_recheck"]
            )
            with self.assertRaisesRegex(ValueError, "FTM warning"):
                generator.validate_selection_evidence(paths, hashes)

            paths, hashes = make_evidence_tree(generator, root / "dataset2", root / "evidence2")
            paths["exclusion_ledger"].write_text(
                f"sample_id,reason\n{sample_id},bad\n", encoding="utf-8"
            )
            hashes["exclusion_ledger"] = generator.sha256_file(
                paths["exclusion_ledger"]
            )
            with self.assertRaisesRegex(ValueError, "exclusion ledger"):
                generator.validate_selection_evidence(paths, hashes)


class PackageAndCliTests(unittest.TestCase):
    def test_direct_script_help_runs_from_project_root(self):
        completed = subprocess.run(
            [sys.executable, str(SCRIPT_PATH), "--help"],
            cwd=PROJECT_ROOT,
            capture_output=True,
            text=True,
            encoding="utf-8",
        )

        self.assertEqual(completed.returncode, 0, completed.stderr)
        self.assertIn("Supplementary Figure S3", completed.stdout)

    def test_package_generates_complete_outputs_after_strict_readback(self):
        generator = load_generator()
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            dataset_root = root / "dataset"
            evidence_root = root / "evidence"
            _, hashes = make_evidence_tree(generator, dataset_root, evidence_root)
            validated = fake_validated_samples(generator, root / "validated")
            extracted = small_sample_arrays(generator)
            output = root / "output"

            with (
                mock.patch.object(
                    generator,
                    "FROZEN_EVIDENCE_SHA256",
                    hashes,
                ),
                mock.patch.object(
                    generator,
                    "validate_frozen_sample",
                    side_effect=validated,
                ) as validate,
                mock.patch.object(
                    generator,
                    "collect_git_state",
                    return_value={"commit": "abc", "branch": "codex/test", "dirty": False},
                ),
                mock.patch.object(
                    generator,
                    "extract_sample_arrays",
                    side_effect=extracted,
                ),
            ):
                summary = generator.generate_package(
                    project_root=PROJECT_ROOT,
                    dataset_root=dataset_root,
                    evidence_root=evidence_root,
                    output_dir=output,
                    generated_at="2026-08-08T00:00:00Z",
                    command_argv=["python", "generator.py"],
                )

            self.assertEqual(validate.call_count, 4)
            self.assertEqual(summary["provenance_status"], "sufficient")
            self.assertEqual(set(path.name for path in output.iterdir()), set(generator.PACKAGE_OUTPUT_NAMES))
            diagnostics = json.loads((output / generator.DIAGNOSTICS_NAME).read_text(encoding="utf-8"))
            self.assertTrue(all(
                item["authoritative_max_abs_difference"] == 0.0
                for item in diagnostics["samples"]
            ))
            record = json.loads((output / generator.GENERATION_RECORD_NAME).read_text(encoding="utf-8"))
            self.assertFalse(record["git"]["dirty"])
            self.assertEqual(set(record["outputs"]), set(generator.DETERMINISTIC_OUTPUT_NAMES))
            self.assertTrue(all("path" not in item for item in record["outputs"].values()))

    def test_failure_preserves_every_existing_candidate_output(self):
        generator = load_generator()
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            dataset_root = root / "dataset"
            evidence_root = root / "evidence"
            _, hashes = make_evidence_tree(generator, dataset_root, evidence_root)
            validated = fake_validated_samples(generator, root / "validated")
            extracted = small_sample_arrays(generator)
            output = root / "output"
            output.mkdir()
            before = {}
            for name in generator.PACKAGE_OUTPUT_NAMES:
                data = f"sentinel-{name}".encode()
                (output / name).write_bytes(data)
                before[name] = data

            with (
                mock.patch.object(generator, "FROZEN_EVIDENCE_SHA256", hashes),
                mock.patch.object(generator, "validate_frozen_sample", side_effect=validated),
                mock.patch.object(
                    generator,
                    "collect_git_state",
                    return_value={"commit": "abc", "branch": "codex/test", "dirty": False},
                ),
                mock.patch.object(
                    generator,
                    "extract_sample_arrays",
                    side_effect=extracted,
                ),
                mock.patch.object(generator, "plot_figure", side_effect=RuntimeError("plot failed")),
            ):
                with self.assertRaisesRegex(RuntimeError, "plot failed"):
                    generator.generate_package(
                        project_root=PROJECT_ROOT,
                        dataset_root=dataset_root,
                        evidence_root=evidence_root,
                        output_dir=output,
                        generated_at="2026-08-08T00:00:00Z",
                        command_argv=["python", "generator.py"],
                    )
            self.assertEqual(
                {name: (output / name).read_bytes() for name in generator.PACKAGE_OUTPUT_NAMES},
                before,
            )

    def test_repeated_package_generation_is_byte_deterministic(self):
        generator = load_generator()
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            dataset_root = root / "dataset"
            evidence_root = root / "evidence"
            _, hashes = make_evidence_tree(generator, dataset_root, evidence_root)
            validated = fake_validated_samples(generator, root / "validated")
            extracted = small_sample_arrays(generator)
            output = root / "output"
            kwargs = dict(
                project_root=PROJECT_ROOT,
                dataset_root=dataset_root,
                evidence_root=evidence_root,
                output_dir=output,
                generated_at="2026-08-08T00:00:00Z",
                command_argv=["python", "generator.py"],
            )
            with (
                mock.patch.object(generator, "FROZEN_EVIDENCE_SHA256", hashes),
                mock.patch.object(generator, "validate_frozen_sample", side_effect=validated * 2),
                mock.patch.object(
                    generator,
                    "collect_git_state",
                    return_value={"commit": "abc", "branch": "codex/test", "dirty": False},
                ),
                mock.patch.object(
                    generator,
                    "extract_sample_arrays",
                    side_effect=extracted * 2,
                ),
            ):
                generator.generate_package(**kwargs)
                first = {name: (output / name).read_bytes() for name in generator.PACKAGE_OUTPUT_NAMES}
                generator.generate_package(**kwargs)
                second = {name: (output / name).read_bytes() for name in generator.PACKAGE_OUTPUT_NAMES}
            self.assertEqual(first, second)

    def test_cli_binds_paths_and_preserves_structured_argv(self):
        generator = load_generator()
        argv = [
            "--project-root", str(PROJECT_ROOT),
            "--dataset-root", "D:/data set",
            "--evidence-root", "D:/evidence root",
            "--output-dir", "D:/output dir",
        ]
        with mock.patch.object(
            generator,
            "generate_package",
            return_value={"provenance_status": "sufficient", "artifacts": []},
        ) as generate:
            stream = io.StringIO()
            with redirect_stdout(stream):
                result = generator.main(argv)

        self.assertEqual(result, 0)
        call = generate.call_args.kwargs
        self.assertEqual(call["dataset_root"], Path("D:/data set"))
        self.assertEqual(call["evidence_root"], Path("D:/evidence root"))
        self.assertEqual(call["output_dir"], Path("D:/output dir"))
        self.assertEqual(
            call["command_argv"][2:],
            argv,
        )
        self.assertEqual(json.loads(stream.getvalue())["provenance_status"], "sufficient")


if __name__ == "__main__":
    unittest.main()
