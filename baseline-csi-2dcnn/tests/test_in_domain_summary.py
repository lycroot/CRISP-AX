from __future__ import annotations

import csv
import json
import tempfile
import unittest
from pathlib import Path

from axhome_csi.in_domain import (
    summarize_in_domain_runs,
    write_in_domain_summary,
)

RUNS = [
    {
        "seed": 2026,
        "accuracy": 0.8,
        "balanced_accuracy": 0.75,
        "macro_f1": 0.7,
        "weighted_f1": 0.72,
        "test_loss": 0.8,
        "best_epoch": 2,
        "best_val_macro_f1": 0.68,
        "confusion_matrix": [[8, 2], [1, 9]],
    },
    {
        "seed": 2027,
        "accuracy": 0.6,
        "balanced_accuracy": 0.65,
        "macro_f1": 0.5,
        "weighted_f1": 0.52,
        "test_loss": 1.0,
        "best_epoch": 3,
        "best_val_macro_f1": 0.48,
        "confusion_matrix": [[6, 4], [3, 7]],
    },
]


class InDomainSummaryTests(unittest.TestCase):
    def test_summary_reports_population_statistics_and_completion(self) -> None:
        summary = summarize_in_domain_runs(
            RUNS,
            requested_seeds=[2026, 2027],
            failed_seeds=[],
            class_names=("a", "b"),
        )

        self.assertTrue(summary["complete"])
        self.assertAlmostEqual(summary["accuracy"]["mean"], 0.7)
        self.assertAlmostEqual(
            summary["accuracy"]["std_population"], 0.1
        )
        self.assertEqual(summary["completed_seeds"], [2026, 2027])
        for actual_row, expected_row in zip(
            summary["confusion_matrix_mean_normalized"],
            [[0.7, 0.3], [0.2, 0.8]],
            strict=True,
        ):
            for actual, expected in zip(
                actual_row, expected_row, strict=True
            ):
                self.assertAlmostEqual(actual, expected)

    def test_failed_seed_marks_summary_incomplete(self) -> None:
        summary = summarize_in_domain_runs(
            RUNS[:1],
            requested_seeds=[2026, 2027],
            failed_seeds=[{"seed": 2027, "error": "RuntimeError: boom"}],
            class_names=("a", "b"),
        )

        self.assertFalse(summary["complete"])
        self.assertEqual(summary["completed_seeds"], [2026])
        self.assertEqual(summary["failed_seeds"][0]["seed"], 2027)

    def test_non_finite_metric_is_rejected(self) -> None:
        invalid = dict(RUNS[0])
        invalid["macro_f1"] = float("nan")

        with self.assertRaisesRegex(ValueError, "non-finite macro_f1"):
            summarize_in_domain_runs(
                [invalid],
                requested_seeds=[2026],
                failed_seeds=[],
                class_names=("a", "b"),
            )

    def test_writer_emits_machine_readable_csv_and_json(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            output = Path(tmp)
            summary = write_in_domain_summary(
                output,
                RUNS,
                requested_seeds=[2026, 2027],
                failed_seeds=[],
                class_names=("a", "b"),
            )
            with (output / "in_domain_runs.csv").open(
                newline="", encoding="utf-8"
            ) as handle:
                rows = list(csv.DictReader(handle))
            saved = json.loads(
                (output / "in_domain_summary.json").read_text(
                    encoding="utf-8"
                )
            )

        self.assertEqual([int(row["seed"]) for row in rows], [2026, 2027])
        self.assertEqual(saved["completed_seeds"], [2026, 2027])
        self.assertEqual(saved, summary)


if __name__ == "__main__":
    unittest.main()
