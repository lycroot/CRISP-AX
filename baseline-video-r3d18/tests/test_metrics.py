from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

import numpy as np

from axhome_video.metrics import (
    balanced_class_weights,
    classification_metrics,
    write_evaluation,
)


class VideoMetricsTests(unittest.TestCase):
    def test_metrics_include_balanced_accuracy_and_macro_f1(self) -> None:
        result = classification_metrics(
            np.asarray([0, 0, 1, 1]),
            np.asarray([0, 1, 1, 1]),
            class_names=("a", "b"),
        )
        self.assertAlmostEqual(result["accuracy"], 0.75)
        self.assertAlmostEqual(result["balanced_accuracy"], 0.75)
        self.assertAlmostEqual(result["macro_f1"], (2 / 3 + 0.8) / 2)

    def test_balanced_weights_use_training_counts(self) -> None:
        weights = balanced_class_weights(np.asarray([0, 0, 0, 1]), 2)
        np.testing.assert_allclose(weights, [2 / 3, 2.0])

    def test_evaluation_writer_creates_machine_readable_outputs(self) -> None:
        y_true = np.asarray([0, 1])
        y_pred = np.asarray([0, 1])
        metrics = classification_metrics(
            y_true, y_pred, class_names=("a", "b")
        )
        with tempfile.TemporaryDirectory() as tmp:
            output = Path(tmp)
            write_evaluation(
                output,
                metrics=metrics,
                sample_ids=("one", "two"),
                y_true=y_true,
                y_pred=y_pred,
                class_names=("a", "b"),
            )
            saved = json.loads(
                (output / "metrics.json").read_text(encoding="utf-8")
            )
            self.assertTrue((output / "predictions.csv").is_file())
            self.assertTrue(
                (output / "confusion_matrix_counts.csv").is_file()
            )
        self.assertEqual(saved["accuracy"], 1.0)


if __name__ == "__main__":
    unittest.main()
