from __future__ import annotations

import csv
from pathlib import Path
from typing import Sequence

from axhome_video.data import VideoSampleRecord


def make_records(
    *,
    people: Sequence[str] = ("P01", "P02", "P03"),
    actions: Sequence[str] = ("walk", "sit_down"),
    per_stratum: int = 20,
) -> tuple[VideoSampleRecord, ...]:
    records = []
    for person_index, person in enumerate(people, start=1):
        session = f"S{person_index:02d}"
        for action in actions:
            for trial in range(1, per_stratum + 1):
                trial_id = f"R{trial:03d}"
                sample_id = f"{session}_{person}_{action}_{trial_id}"
                records.append(
                    VideoSampleRecord(
                        sample_id=sample_id,
                        session_id=session,
                        archive_session_id=session,
                        person_id=person,
                        environment_id="E1",
                        action_id=action,
                        trial_id=trial_id,
                        video_path=Path(f"/{sample_id}.mp4"),
                        video_frames_written=151,
                        video_fps=30.0,
                        status="ok",
                        sync_quality_flags="ok",
                    )
                )
    return tuple(records)


def write_manifest_fixture(
    root: Path,
    *,
    people: Sequence[str] = ("P01", "P02", "P03"),
    actions: Sequence[str] = ("walk", "sit_down"),
    trials: int = 3,
) -> list[dict[str, str]]:
    manifest_dir = root / "manifests"
    manifest_dir.mkdir(parents=True)
    rows: list[dict[str, str]] = []
    for person_index, person in enumerate(people, start=1):
        session = f"S{person_index:02d}"
        for action in actions:
            for trial in range(1, trials + 1):
                trial_id = f"R{trial:03d}"
                sample_id = f"{session}_{person}_{action}_{trial_id}"
                relative = Path("data") / session / "video" / action / f"{sample_id}.mp4"
                video_path = root / relative
                video_path.parent.mkdir(parents=True, exist_ok=True)
                video_path.touch()
                rows.append(
                    {
                        "sample_id": sample_id,
                        "session_id": session,
                        "archive_session_id": session,
                        "person_id": person,
                        "environment_id": "E1",
                        "action_id": action,
                        "trial_id": trial_id,
                        "archive_video_path": relative.as_posix(),
                        "video_frames_written": "151",
                        "video_fps": "30.0",
                        "status": "ok",
                        "sync_quality_flags": "ok",
                    }
                )
    with (manifest_dir / "archive_index.csv").open(
        "w", newline="", encoding="utf-8"
    ) as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    return rows
