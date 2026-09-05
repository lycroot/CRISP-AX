from __future__ import annotations

import importlib.util
import json
import platform
import tempfile
import unittest
from pathlib import Path

from axhome_csi.model import (
    ARCHITECTURE_DESCRIPTIONS,
    CSI_ARCHITECTURES,
    DEFAULT_ARCHITECTURE,
)
from tests._fixtures import write_mini_release

TORCH_AVAILABLE = importlib.util.find_spec("torch") is not None


class ArchitectureRegistryTests(unittest.TestCase):
    def test_registry_is_unique_and_documented(self) -> None:
        self.assertEqual(len(CSI_ARCHITECTURES), len(set(CSI_ARCHITECTURES)))
        self.assertIn(DEFAULT_ARCHITECTURE, CSI_ARCHITECTURES)
        self.assertEqual(
            set(ARCHITECTURE_DESCRIPTIONS), set(CSI_ARCHITECTURES)
        )

    def test_default_architecture_is_the_published_cnn_baseline(self) -> None:
        self.assertEqual(DEFAULT_ARCHITECTURE, "cnn2d")


@unittest.skipUnless(TORCH_AVAILABLE, "PyTorch not installed locally")
class ArchitectureForwardTests(unittest.TestCase):
    def test_every_architecture_maps_the_frozen_input_to_class_logits(
        self,
    ) -> None:
        import torch

        from axhome_csi.model import build_model, count_parameters

        inputs = torch.randn(2, 2, 32, 16)
        for arch in CSI_ARCHITECTURES:
            with self.subTest(arch=arch):
                model = build_model(
                    num_classes=14, arch=arch, num_subcarriers=16
                )
                model.eval()
                with torch.no_grad():
                    logits = model(inputs)
                self.assertEqual(tuple(logits.shape), (2, 14))
                self.assertTrue(bool(torch.isfinite(logits).all()))
                self.assertGreater(count_parameters(model), 0)

    def test_every_architecture_produces_finite_gradients(self) -> None:
        import torch

        from axhome_csi.model import build_model

        inputs = torch.randn(2, 2, 32, 16)
        labels = torch.tensor([0, 1])
        for arch in CSI_ARCHITECTURES:
            with self.subTest(arch=arch):
                model = build_model(
                    num_classes=14, arch=arch, num_subcarriers=16
                )
                loss = torch.nn.functional.cross_entropy(
                    model(inputs), labels
                )
                loss.backward()
                gradients = [
                    p.grad for p in model.parameters() if p.grad is not None
                ]
                self.assertTrue(gradients)
                self.assertTrue(
                    all(bool(torch.isfinite(g).all()) for g in gradients)
                )

    def test_unknown_architecture_is_rejected(self) -> None:
        from axhome_csi.model import build_model

        with self.assertRaises(ValueError):
            build_model(num_classes=14, arch="not-a-model")

    def test_sequence_models_reject_a_mismatched_subcarrier_count(
        self,
    ) -> None:
        import torch

        from axhome_csi.model import build_model

        for arch in ("bilstm", "transformer"):
            with self.subTest(arch=arch):
                model = build_model(
                    num_classes=14, arch=arch, num_subcarriers=128
                )
                with self.assertRaises(ValueError):
                    model(torch.randn(1, 2, 32, 16))

    def test_training_config_rejects_an_unknown_architecture(self) -> None:
        from axhome_csi.training import TrainingConfig

        with self.assertRaises(ValueError):
            TrainingConfig(arch="not-a-model")

    def test_training_config_defaults_to_the_published_architecture(
        self,
    ) -> None:
        from axhome_csi.training import TrainingConfig

        self.assertEqual(TrainingConfig().arch, DEFAULT_ARCHITECTURE)


@unittest.skipUnless(TORCH_AVAILABLE, "PyTorch not installed locally")
class ArchitectureProvenanceTests(unittest.TestCase):
    def test_run_config_and_metrics_record_the_selected_architecture(
        self,
    ) -> None:
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
                    arch="bilstm",
                ),
                class_names=("walk", "sit_down"),
            )

            self.assertEqual(metrics["arch"], "bilstm")
            run_config = json.loads(
                (output / "run_config.json").read_text(encoding="utf-8")
            )
            self.assertEqual(run_config["model"]["arch"], "bilstm")
            self.assertEqual(run_config["training_config"]["arch"], "bilstm")
            self.assertGreater(
                int(run_config["model"]["trainable_parameters"]), 0
            )
            runtime = run_config["runtime"]
            self.assertEqual(
                runtime["software_versions"]["python"],
                platform.python_version(),
            )
            self.assertTrue(runtime["software_versions"]["torch"])
            for field in (
                "cuda_available",
                "cuda_runtime",
                "cudnn",
                "device_name",
            ):
                self.assertIn(field, runtime["accelerator"])

    def test_summary_records_a_single_architecture(self) -> None:
        from axhome_csi.in_domain import summarize_in_domain_runs

        runs = [
            {
                "seed": 2026,
                "arch": "resnet18",
                "accuracy": 0.5,
                "balanced_accuracy": 0.5,
                "macro_f1": 0.5,
                "weighted_f1": 0.5,
                "test_loss": 1.0,
                "best_val_macro_f1": 0.5,
                "confusion_matrix": [[1, 1], [1, 1]],
            }
        ]
        summary = summarize_in_domain_runs(
            runs,
            requested_seeds=[2026],
            failed_seeds=[],
            class_names=("walk", "sit_down"),
        )
        self.assertEqual(summary["architecture"], "resnet18")


if __name__ == "__main__":
    unittest.main()
