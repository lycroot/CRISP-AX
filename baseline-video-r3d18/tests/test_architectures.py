from __future__ import annotations

import importlib.util
import unittest

from axhome_video.model import (
    ARCHITECTURE_BUILDERS,
    ARCHITECTURE_DESCRIPTIONS,
    ARCHITECTURE_WEIGHTS,
    DEFAULT_ARCHITECTURE,
    VIDEO_ARCHITECTURES,
    validate_architecture,
)

TORCH_AVAILABLE = importlib.util.find_spec("torch") is not None
TORCHVISION_AVAILABLE = importlib.util.find_spec("torchvision") is not None


class ArchitectureRegistryTests(unittest.TestCase):
    def test_registry_is_unique_and_fully_documented(self) -> None:
        self.assertEqual(
            len(VIDEO_ARCHITECTURES), len(set(VIDEO_ARCHITECTURES))
        )
        self.assertIn(DEFAULT_ARCHITECTURE, VIDEO_ARCHITECTURES)
        for mapping in (
            ARCHITECTURE_BUILDERS,
            ARCHITECTURE_WEIGHTS,
            ARCHITECTURE_DESCRIPTIONS,
        ):
            self.assertEqual(set(mapping), set(VIDEO_ARCHITECTURES))

    def test_default_architecture_is_the_published_backbone(self) -> None:
        self.assertEqual(DEFAULT_ARCHITECTURE, "r3d_18")
        self.assertEqual(
            ARCHITECTURE_WEIGHTS["r3d_18"], "R3D_18_Weights.KINETICS400_V1"
        )

    def test_unknown_architecture_is_rejected(self) -> None:
        with self.assertRaises(ValueError):
            validate_architecture("x3d_s")


@unittest.skipUnless(
    TORCH_AVAILABLE and TORCHVISION_AVAILABLE,
    "PyTorch and Torchvision are required",
)
class ArchitectureForwardTests(unittest.TestCase):
    def test_every_backbone_maps_the_frozen_clip_to_class_logits(
        self,
    ) -> None:
        import torch

        from axhome_video.model import build_video_model, count_parameters

        clip = torch.zeros(1, 3, 16, 112, 112)
        for arch in VIDEO_ARCHITECTURES:
            with self.subTest(arch=arch):
                model = build_video_model(
                    num_classes=14, arch=arch, pretrained=False
                )
                model.eval()
                with torch.no_grad():
                    logits = model(clip)
                self.assertEqual(tuple(logits.shape), (1, 14))
                self.assertTrue(bool(torch.isfinite(logits).all()))
                self.assertGreater(count_parameters(model), 0)

    def test_legacy_r3d18_builder_still_returns_the_default_backbone(
        self,
    ) -> None:
        import torch

        from axhome_video.model import build_r3d18

        model = build_r3d18(num_classes=14, pretrained=False)
        model.eval()
        with torch.no_grad():
            logits = model(torch.zeros(1, 3, 16, 112, 112))
        self.assertEqual(tuple(logits.shape), (1, 14))

    def test_training_config_rejects_an_unknown_architecture(self) -> None:
        from axhome_video.training import TrainingConfig

        with self.assertRaises(ValueError):
            TrainingConfig(arch="x3d_s")

    def test_training_config_defaults_to_the_published_backbone(self) -> None:
        from axhome_video.training import TrainingConfig

        self.assertEqual(TrainingConfig().arch, DEFAULT_ARCHITECTURE)


class RuntimeArchitectureTests(unittest.TestCase):
    def test_runtime_metadata_accepts_an_architecture_without_hashing(
        self,
    ) -> None:
        from axhome_video.runtime import collect_runtime_metadata

        metadata = collect_runtime_metadata(
            include_r3d18_weight=False, arch="mc3_18"
        )
        self.assertIsNone(metadata["pretrained_checkpoint"])


class SummaryArchitectureTests(unittest.TestCase):
    def test_summary_records_a_single_architecture(self) -> None:
        from axhome_video.aggregate import summarize_runs

        runs = [
            {
                "test_person": "P01",
                "arch": "r2plus1d_18",
                "accuracy": 0.5,
                "balanced_accuracy": 0.5,
                "macro_f1": 0.5,
                "weighted_f1": 0.5,
                "test_loss": 1.0,
                "best_val_macro_f1": 0.5,
                "confusion_matrix": [[1, 1], [1, 1]],
            }
        ]
        summary = summarize_runs(
            runs,
            requested=["P01"],
            failed=[],
            class_names=("walk", "sit_down"),
            id_field="test_person",
            requested_key="requested_test_people",
            completed_key="completed_test_people",
            failed_key="failed_folds",
            protocol="subject_wise_loso",
        )
        self.assertEqual(summary["architecture"], "r2plus1d_18")


if __name__ == "__main__":
    unittest.main()
