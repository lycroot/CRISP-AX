from __future__ import annotations

import csv
import tempfile
import unittest
from pathlib import Path

from axhome_video.data import VideoSampleRecord, load_video_manifest
from axhome_video.splits import (
    human_samples,
    make_in_domain_split,
    make_subject_fold,
    rotating_loso_pairs,
)
from tests._fixtures import make_records, write_manifest_fixture


def _ids(samples: tuple[VideoSampleRecord, ...]) -> set[str]:
    return {sample.sample_id for sample in samples}


class VideoDataAndSplitTests(unittest.TestCase):
    def test_manifest_resolves_video_paths_inside_release(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp) / "release"
            rows = write_manifest_fixture(root)
            records = load_video_manifest(root, validate_paths=True)

        self.assertEqual(len(records), len(rows))
        self.assertTrue(records[0].video_path.is_relative_to(root.resolve()))
        self.assertEqual(records[0].video_frames_written, 151)
        self.assertEqual(records[0].video_fps, 30.0)

    def test_manifest_rejects_path_escape(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp) / "release"
            write_manifest_fixture(root, people=("P01",), actions=("walk",), trials=1)
            manifest = root / "manifests" / "archive_index.csv"
            with manifest.open(newline="", encoding="utf-8") as handle:
                rows = list(csv.DictReader(handle))
            rows[0]["archive_video_path"] = "../outside.mp4"
            with manifest.open("w", newline="", encoding="utf-8") as handle:
                writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
                writer.writeheader()
                writer.writerows(rows)

            with self.assertRaisesRegex(ValueError, "escapes dataset root"):
                load_video_manifest(root)

    def test_manifest_rejects_non_finite_video_fps(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp) / "release"
            write_manifest_fixture(
                root, people=("P01",), actions=("walk",), trials=1
            )
            manifest = root / "manifests" / "archive_index.csv"
            with manifest.open(newline="", encoding="utf-8") as handle:
                rows = list(csv.DictReader(handle))
            rows[0]["video_fps"] = "nan"
            with manifest.open("w", newline="", encoding="utf-8") as handle:
                writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
                writer.writeheader()
                writer.writerows(rows)

            with self.assertRaisesRegex(ValueError, "finite and positive"):
                load_video_manifest(root)

    def test_in_domain_split_is_reproducible_and_complete(self) -> None:
        records = make_records(per_stratum=20)
        first = make_in_domain_split(records, seed=2026)
        second = make_in_domain_split(tuple(reversed(records)), seed=2026)

        self.assertEqual(_ids(first.train), _ids(second.train))
        self.assertEqual(_ids(first.val), _ids(second.val))
        self.assertEqual(_ids(first.test), _ids(second.test))
        self.assertEqual(
            (len(first.train), len(first.val), len(first.test)),
            (78, 18, 24),
        )
        self.assertEqual(
            _ids(first.train) | _ids(first.val) | _ids(first.test),
            _ids(records),
        )
        self.assertFalse(_ids(first.train) & _ids(first.val))
        self.assertFalse(_ids(first.train) & _ids(first.test))
        self.assertFalse(_ids(first.val) & _ids(first.test))

    def test_different_seeds_change_assignments(self) -> None:
        records = make_records()
        self.assertNotEqual(
            _ids(make_in_domain_split(records, seed=2026).train),
            _ids(make_in_domain_split(records, seed=2027).train),
        )

    def test_background_is_excluded(self) -> None:
        records = make_records()
        background = VideoSampleRecord(
            sample_id="background",
            session_id="S01",
            archive_session_id="S01",
            person_id="NONE",
            environment_id="E1",
            action_id="background_idle",
            trial_id="R001",
            video_path=Path("/background.mp4"),
            video_frames_written=151,
            video_fps=30.0,
            status="ok",
            sync_quality_flags="ok",
        )
        self.assertEqual(len(human_samples((*records, background))), len(records))
        split = make_in_domain_split((*records, background), seed=2026)
        self.assertNotIn(
            background.sample_id,
            _ids(split.train) | _ids(split.val) | _ids(split.test),
        )

    def test_subject_fold_and_loso_pairs_match_csi_protocol(self) -> None:
        records = make_records(per_stratum=3)
        fold = make_subject_fold(records, test_person="P01", val_person="P02")
        self.assertEqual(fold.train_people, ("P03",))
        self.assertEqual({sample.person_id for sample in fold.test}, {"P01"})
        self.assertEqual({sample.person_id for sample in fold.val}, {"P02"})
        self.assertEqual(
            rotating_loso_pairs(records),
            [("P01", "P02"), ("P02", "P03"), ("P03", "P01")],
        )

    def test_rejects_strata_too_small_for_three_partitions(self) -> None:
        with self.assertRaisesRegex(ValueError, "cannot be split"):
            make_in_domain_split(make_records(per_stratum=2), seed=2026)


if __name__ == "__main__":
    unittest.main()
