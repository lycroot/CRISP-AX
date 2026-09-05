from __future__ import annotations

import importlib.util
import tempfile
import unittest
from pathlib import Path

import numpy as np

from axhome_csi.metrics import (
    balanced_class_weights,
    classification_metrics,
    write_evaluation,
)


class MetricsTests(unittest.TestCase):
    def test_balanced_weights_are_inverse_to_training_counts(self) -> None:
        labels = np.array([0, 0, 0, 1, 2, 2], dtype=np.int64)

        weights = balanced_class_weights(labels, num_classes=3)

        self.assertEqual(weights.shape, (3,))
        self.assertLess(weights[0], weights[2])
        self.assertLess(weights[2], weights[1])
        self.assertTrue(np.isfinite(weights).all())

    def test_metrics_include_macro_f1_and_confusion_matrix(self) -> None:
        result = classification_metrics(
            np.array([0, 0, 1, 1]),
            np.array([0, 1, 1, 1]),
            class_names=("a", "b"),
        )

        self.assertAlmostEqual(result["accuracy"], 0.75)
        self.assertAlmostEqual(result["balanced_accuracy"], 0.75)
        self.assertAlmostEqual(result["macro_f1"], (2 / 3 + 0.8) / 2)
        self.assertEqual(result["confusion_matrix"], [[1, 1], [0, 2]])
        self.assertEqual(len(result["per_class"]), 2)

    def test_evaluation_writer_creates_machine_readable_outputs(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            output_dir = Path(tmp)
            y_true = np.array([0, 1, 1])
            y_pred = np.array([0, 0, 1])
            sample_ids = ["s1", "s2", "s3"]
            metrics = classification_metrics(
                y_true, y_pred, class_names=("a", "b")
            )

            write_evaluation(
                output_dir,
                metrics=metrics,
                sample_ids=sample_ids,
                y_true=y_true,
                y_pred=y_pred,
                class_names=("a", "b"),
            )

            self.assertTrue((output_dir / "metrics.json").is_file())
            self.assertTrue((output_dir / "classification_report.csv").is_file())
            self.assertTrue((output_dir / "predictions.csv").is_file())
            self.assertTrue((output_dir / "confusion_matrix_counts.csv").is_file())


@unittest.skipUnless(importlib.util.find_spec("torch"), "PyTorch not installed locally")
class ModelTests(unittest.TestCase):
    def test_model_maps_two_rx_channels_to_class_logits(self) -> None:
        import torch

        from axhome_csi.model import CSI2DCNN

        model = CSI2DCNN(num_classes=14, in_channels=2)
        logits = model(torch.randn(3, 2, 64, 32))

        self.assertEqual(tuple(logits.shape), (3, 14))
        self.assertGreater(sum(p.numel() for p in model.parameters()), 0)


if __name__ == "__main__":
    unittest.main()

