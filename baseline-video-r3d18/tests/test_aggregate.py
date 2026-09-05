from __future__ import annotations

import csv
import json
import tempfile
import unittest
from pathlib import Path

from axhome_video.aggregate import (
    split_summary,
    summarize_runs,
    validate_reference_assignments,
    write_in_domain_summary,
    write_loso_summary,
    write_split_assignments,
)
from axhome_video.splits import make_in_domain_split
from tests._fixtures import make_records


class VideoAggregateTests(unittest.TestCase):
    def test_assignment_writer_records_every_sample_once(self) -> None:
        split = make_in_domain_split(make_records(), seed=2026)
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "split_assignments.csv"
            write_split_assignments(path, split)
            with path.open(newline="", encoding="utf-8") as handle:
                rows = list(csv.DictReader(handle))

        self.assertEqual(len(rows), 120)
        self.assertEqual(len({row["sample_id"] for row in rows}), 120)
        self.assertEqual(
            {row["split"] for row in rows},
            {"train", "validation", "test"},
        )
        self.assertEqual({int(row["seed"]) for row in rows}, {2026})

    def test_reference_assignments_match_by_sample_id(self) -> None:
        split = make_in_domain_split(make_records(), seed=2026)
        with tempfile.TemporaryDirectory() as tmp:
            video = Path(tmp) / "video.csv"
            reference = Path(tmp) / "reference.csv"
            write_split_assignments(video, split)
            write_split_assignments(reference, split)
            validate_reference_assignments(video, reference, seed=2026)

    def test_reference_assignment_mismatch_is_rejected(self) -> None:
        split = make_in_domain_split(make_records(), seed=2026)
        with tempfile.TemporaryDirectory() as tmp:
            video = Path(tmp) / "video.csv"
            reference = Path(tmp) / "reference.csv"
            write_split_assignments(video, split)
            write_split_assignments(reference, split)
            with reference.open(newline="", encoding="utf-8") as handle:
                rows = list(csv.DictReader(handle))
            rows[0]["split"] = (
                "test" if rows[0]["split"] != "test" else "train"
            )
            with reference.open("w", newline="", encoding="utf-8") as handle:
                writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
                writer.writeheader()
                writer.writerows(rows)

            with self.assertRaisesRegex(ValueError, "assignment mismatch"):
                validate_reference_assignments(video, reference, seed=2026)

    def test_split_summary_reports_counts_and_ratios(self) -> None:
        split = make_in_domain_split(make_records(), seed=2026)
        summary = split_summary(split, seed=2026)
        self.assertEqual(summary["total_samples"], 120)
        self.assertEqual(summary["counts"], {"train": 78, "validation": 18, "test": 24})
        self.assertAlmostEqual(sum(summary["actual_ratios"].values()), 1.0)

    def test_summary_uses_population_standard_deviation(self) -> None:
        summary = summarize_runs(
            _runs(),
            requested=[2026, 2027],
            failed=[],
            class_names=("a", "b"),
        )
        self.assertAlmostEqual(summary["accuracy"]["mean"], 0.7)
        self.assertAlmostEqual(
            summary["accuracy"]["std_population"], 0.1
        )
        self.assertTrue(summary["complete"])
        self.assertEqual(summary["completed_seeds"], [2026, 2027])
        self.assertEqual(
            summary["confusion_matrix_mean_normalized"],
            [[0.7, 0.30000000000000004], [0.2, 0.8]],
        )

    def test_failed_run_keeps_partial_summary_incomplete(self) -> None:
        summary = summarize_runs(
            _runs()[:1],
            requested=[2026, 2027],
            failed=[{"id": 2027, "error": "RuntimeError: boom"}],
            class_names=("a", "b"),
        )
        self.assertFalse(summary["complete"])
        self.assertEqual(summary["completed_seeds"], [2026])

    def test_non_finite_run_metric_is_rejected(self) -> None:
        invalid = dict(_runs()[0])
        invalid["macro_f1"] = float("nan")
        with self.assertRaisesRegex(ValueError, "non-finite macro_f1"):
            summarize_runs(
                [invalid],
                requested=[2026],
                failed=[],
                class_names=("a", "b"),
            )

    def test_summary_writer_emits_csv_json_and_mean_matrix(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            output = Path(tmp)
            summary = write_in_domain_summary(
                output,
                _runs(),
                requested_seeds=[2026, 2027],
                failed_seeds=[],
                class_names=("a", "b"),
            )
            saved = json.loads(
                (output / "in_domain_summary.json").read_text(
                    encoding="utf-8"
                )
            )
            self.assertTrue((output / "in_domain_runs.csv").is_file())
            self.assertTrue(
                (output / "confusion_matrix_mean_normalized.csv").is_file()
            )
        self.assertEqual(saved, summary)

    def test_loso_writer_uses_subject_keys(self) -> None:
        run = dict(_runs()[0])
        run.update({"test_person": "P01", "val_person": "P02"})
        with tempfile.TemporaryDirectory() as tmp:
            summary = write_loso_summary(
                tmp,
                [run],
                requested_people=["P01"],
                failed_folds=[],
                class_names=("a", "b"),
            )
            self.assertTrue((Path(tmp) / "loso_folds.csv").is_file())
        self.assertEqual(summary["completed_test_people"], ["P01"])
        self.assertEqual(summary["failed_folds"], [])


def _runs() -> list[dict[str, object]]:
    return [
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


if __name__ == "__main__":
    unittest.main()
