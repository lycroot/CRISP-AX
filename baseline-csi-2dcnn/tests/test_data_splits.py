from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import numpy as np

from axhome_csi.data import (
    interpolate_low_energy_subcarriers,
    load_manifest,
    preprocess_amplitude,
)
from axhome_csi.splits import HUMAN_ACTIONS, make_subject_fold
from tests._fixtures import write_mini_release


class DataAndSplitTests(unittest.TestCase):
    def test_manifest_resolves_release_relative_paths(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp) / "AXHome-MM-v1"
            rows = write_mini_release(root)

            loaded = load_manifest(root)

        self.assertEqual(len(loaded), len(rows))
        self.assertTrue(loaded[0].csi_path.is_absolute())
        self.assertEqual(loaded[0].sample_id, rows[0]["sample_id"])

    def test_subject_fold_has_no_person_overlap(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp) / "AXHome-MM-v1"
            write_mini_release(root)
            samples = load_manifest(root)

            fold = make_subject_fold(
                samples, test_person="P01", val_person="P02"
            )

        train_people = {sample.person_id for sample in fold.train}
        val_people = {sample.person_id for sample in fold.val}
        test_people = {sample.person_id for sample in fold.test}
        self.assertEqual(train_people, {"P03"})
        self.assertEqual(val_people, {"P02"})
        self.assertEqual(test_people, {"P01"})
        self.assertTrue(train_people.isdisjoint(val_people | test_people))

    def test_preprocess_produces_finite_rx_time_frequency_tensor(self) -> None:
        rng = np.random.default_rng(2026)
        csi = (
            rng.normal(size=(20, 2, 1, 24))
            + 1j * rng.normal(size=(20, 2, 1, 24))
        ).astype(np.complex64)
        csi[..., 0] = 0

        tensor = preprocess_amplitude(
            csi, target_packets=12, target_subcarriers=8
        )

        self.assertEqual(tensor.shape, (2, 12, 8))
        self.assertEqual(tensor.dtype, np.float32)
        self.assertTrue(np.isfinite(tensor).all())
        self.assertTrue(np.allclose(tensor.mean(axis=(1, 2)), 0, atol=1e-5))

    def test_low_energy_repair_preserves_fixed_frequency_positions(self) -> None:
        amplitude = np.array(
            [[[1.0, 2.0, 0.0, 4.0, 5.0]], [[2.0, 3.0, 0.0, 5.0, 6.0]]],
            dtype=np.float32,
        )

        repaired, valid = interpolate_low_energy_subcarriers(
            amplitude, low_energy_ratio=0.1
        )

        self.assertEqual(valid.tolist(), [True, True, False, True, True])
        self.assertTrue(np.allclose(repaired[..., 2].ravel(), [3.0, 4.0]))
        self.assertTrue(np.array_equal(repaired[..., 0], amplitude[..., 0]))

    def test_declares_expected_fourteen_human_actions(self) -> None:
        self.assertEqual(len(HUMAN_ACTIONS), 14)
        self.assertNotIn("background_idle", HUMAN_ACTIONS)
        self.assertIn("fall_like", HUMAN_ACTIONS)


if __name__ == "__main__":
    unittest.main()
