from __future__ import annotations

import csv
import tempfile
import unittest
from pathlib import Path

from axhome_csi.data import SampleRecord
from axhome_csi.in_domain import write_split_assignments
from axhome_csi.splits import make_in_domain_split


def _record(
    *, person_id: str, action_id: str, trial: int, session_id: str = "S01"
) -> SampleRecord:
    sample_id = f"{person_id}_{action_id}_R{trial:03d}"
    return SampleRecord(
        sample_id=sample_id,
        session_id=session_id,
        archive_session_id=session_id,
        person_id=person_id,
        environment_id="E1",
        action_id=action_id,
        trial_id=f"R{trial:03d}",
        csi_path=Path(f"/{sample_id}.dat"),
        csi_packets_written=12,
        status="ok",
        sync_quality_flags="ok",
    )


def _records(per_stratum: int = 20) -> tuple[SampleRecord, ...]:
    return tuple(
        _record(person_id=person, action_id=action, trial=trial)
        for person in ("P01", "P02", "P03")
        for action in ("walk", "sit_down")
        for trial in range(1, per_stratum + 1)
    )


class InDomainSplitTests(unittest.TestCase):
    def test_person_action_split_is_complete_and_reproducible(self) -> None:
        samples = _records()

        first = make_in_domain_split(samples, seed=2026)
        second = make_in_domain_split(tuple(reversed(samples)), seed=2026)

        self.assertEqual(
            {sample.sample_id for sample in first.train},
            {sample.sample_id for sample in second.train},
        )
        self.assertEqual(
            {sample.sample_id for sample in first.val},
            {sample.sample_id for sample in second.val},
        )
        self.assertEqual(
            {sample.sample_id for sample in first.test},
            {sample.sample_id for sample in second.test},
        )
        self.assertEqual(
            (len(first.train), len(first.val), len(first.test)),
            (78, 18, 24),
        )
        assigned_ids = [
            sample.sample_id
            for split_samples in (first.train, first.val, first.test)
            for sample in split_samples
        ]
        self.assertEqual(len(assigned_ids), len(set(assigned_ids)))
        self.assertEqual(set(assigned_ids), {sample.sample_id for sample in samples})

    def test_different_seeds_change_assignments(self) -> None:
        first = make_in_domain_split(_records(), seed=2026)
        second = make_in_domain_split(_records(), seed=2027)

        self.assertNotEqual(
            {sample.sample_id for sample in first.train},
            {sample.sample_id for sample in second.train},
        )

    def test_every_person_action_stratum_appears_in_each_split(self) -> None:
        samples = _records()
        background = _record(
            person_id="NONE", action_id="background_idle", trial=1
        )

        split = make_in_domain_split((*samples, background), seed=2026)

        expected_keys = {
            (person, action)
            for person in ("P01", "P02", "P03")
            for action in ("walk", "sit_down")
        }
        for split_samples in (split.train, split.val, split.test):
            self.assertEqual(
                {(sample.person_id, sample.action_id) for sample in split_samples},
                expected_keys,
            )
            self.assertNotIn(background.sample_id, {s.sample_id for s in split_samples})

    def test_rejects_strata_that_cannot_fill_three_splits(self) -> None:
        with self.assertRaisesRegex(ValueError, "cannot be split"):
            make_in_domain_split(_records(per_stratum=2), seed=2026)

    def test_split_assignments_records_every_sample_once(self) -> None:
        split = make_in_domain_split(_records(), seed=2026)

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
        self.assertEqual(
            list(rows[0]),
            [
                "sample_id",
                "person_id",
                "action_id",
                "archive_session",
                "split",
                "seed",
            ],
        )


if __name__ == "__main__":
    unittest.main()
