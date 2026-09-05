from __future__ import annotations

import importlib.util
import tempfile
import unittest
from pathlib import Path

from tests._fixtures import write_mini_release


@unittest.skipUnless(importlib.util.find_spec("torch"), "PyTorch not installed locally")
class TrainingSmokeTests(unittest.TestCase):
    def test_one_epoch_creates_checkpoint_history_and_evaluation(self) -> None:
        from axhome_csi.cache import CacheConfig
        from axhome_csi.training import TrainingConfig, train_subject_fold

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp) / "release"
            output = Path(tmp) / "result"
            write_mini_release(root)

            metrics = train_subject_fold(
                root,
                test_person="P01",
                val_person="P02",
                output_dir=output,
                cache_config=CacheConfig(
                    target_packets=8, target_subcarriers=8
                ),
                training_config=TrainingConfig(
                    epochs=1,
                    batch_size=2,
                    patience=1,
                    num_workers=0,
                    device="cpu",
                ),
                class_names=("walk", "sit_down"),
            )

            self.assertIn("macro_f1", metrics)
            self.assertTrue((output / "best_model.pt").is_file())
            self.assertTrue((output / "history.csv").is_file())
            self.assertTrue((output / "metrics.json").is_file())
            self.assertTrue((output / "predictions.csv").is_file())

    def test_explicit_split_training_creates_outputs(self) -> None:
        from axhome_csi.cache import CacheConfig
        from axhome_csi.data import load_manifest
        from axhome_csi.splits import make_subject_fold
        from axhome_csi.training import TrainingConfig, train_sample_split

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp) / "release"
            output = Path(tmp) / "explicit"
            write_mini_release(root)
            fold = make_subject_fold(
                load_manifest(root), test_person="P01", val_person="P02"
            )

            metrics = train_sample_split(
                root,
                train_samples=fold.train,
                val_samples=fold.val,
                test_samples=fold.test,
                output_dir=output,
                run_metadata={"test_protocol": "explicit-smoke"},
                cache_config=CacheConfig(
                    target_packets=8, target_subcarriers=8
                ),
                training_config=TrainingConfig(
                    epochs=1,
                    batch_size=2,
                    patience=1,
                    num_workers=0,
                    device="cpu",
                ),
                class_names=("walk", "sit_down"),
            )

            self.assertIn("macro_f1", metrics)
            self.assertTrue((output / "run_config.json").is_file())
            self.assertTrue((output / "best_model.pt").is_file())
            self.assertTrue((output / "metrics.json").is_file())

    def test_one_epoch_in_domain_split_creates_auditable_outputs(self) -> None:
        import math

        from axhome_csi.cache import CacheConfig
        from axhome_csi.data import load_manifest
        from axhome_csi.in_domain import (
            split_summary,
            write_split_assignments,
        )
        from axhome_csi.splits import make_in_domain_split
        from axhome_csi.training import TrainingConfig, train_sample_split

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp) / "release"
            output = Path(tmp) / "in_domain"
            write_mini_release(root, trials_per_action=5)
            split = make_in_domain_split(load_manifest(root), seed=2026)
            write_split_assignments(
                output / "split_assignments.csv", split
            )

            metrics = train_sample_split(
                root,
                train_samples=split.train,
                val_samples=split.val,
                test_samples=split.test,
                output_dir=output,
                run_metadata={"in_domain_split": split_summary(split)},
                metric_metadata={
                    "seed": 2026,
                    "protocol": "person_action_stratified_in_domain",
                },
                cache_config=CacheConfig(
                    target_packets=8, target_subcarriers=8
                ),
                training_config=TrainingConfig(
                    epochs=1,
                    batch_size=2,
                    patience=1,
                    num_workers=0,
                    seed=2026,
                    device="cpu",
                ),
                class_names=("walk", "sit_down"),
            )

            for name in (
                "accuracy",
                "balanced_accuracy",
                "macro_f1",
                "weighted_f1",
                "test_loss",
            ):
                self.assertTrue(math.isfinite(float(metrics[name])), name)
            for name in (
                "split_assignments.csv",
                "run_config.json",
                "best_model.pt",
                "metrics.json",
            ):
                self.assertTrue((output / name).is_file(), name)


if __name__ == "__main__":
    unittest.main()
