from __future__ import annotations

import csv
import math
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class VideoSampleRecord:
    sample_id: str
    session_id: str
    archive_session_id: str
    person_id: str
    environment_id: str
    action_id: str
    trial_id: str
    video_path: Path
    video_frames_written: int
    video_fps: float
    status: str
    sync_quality_flags: str


REQUIRED_VIDEO_MANIFEST_FIELDS = {
    "sample_id",
    "session_id",
    "archive_session_id",
    "person_id",
    "environment_id",
    "action_id",
    "trial_id",
    "archive_video_path",
    "video_frames_written",
    "video_fps",
    "status",
    "sync_quality_flags",
}


def _safe_release_path(dataset_root: Path, relative_path: str) -> Path:
    root = dataset_root.resolve()
    candidate = (root / relative_path).resolve()
    if not candidate.is_relative_to(root):
        raise ValueError(
            f"release-relative path escapes dataset root: {relative_path}"
        )
    return candidate


def load_video_manifest(
    dataset_root: str | Path,
    *,
    require_ok: bool = True,
    validate_paths: bool = False,
) -> list[VideoSampleRecord]:
    """Load video records from the canonical CRISP-AX archive index."""
    root = Path(dataset_root).resolve()
    manifest_path = root / "manifests" / "archive_index.csv"
    if not manifest_path.is_file():
        raise FileNotFoundError(f"manifest not found: {manifest_path}")

    records: list[VideoSampleRecord] = []
    with manifest_path.open("r", newline="", encoding="utf-8-sig") as handle:
        reader = csv.DictReader(handle)
        missing = REQUIRED_VIDEO_MANIFEST_FIELDS - set(reader.fieldnames or [])
        if missing:
            raise ValueError(
                "archive_index.csv is missing required video fields: "
                + ", ".join(sorted(missing))
            )
        for row_number, row in enumerate(reader, start=2):
            if require_ok and (
                row["status"] != "ok" or row["sync_quality_flags"] != "ok"
            ):
                continue
            video_path = _safe_release_path(root, row["archive_video_path"])
            if validate_paths and not video_path.is_file():
                raise FileNotFoundError(
                    f"row {row_number}, sample {row['sample_id']}: "
                    f"video file not found: {video_path}"
                )
            try:
                frames = int(row["video_frames_written"])
                fps = float(row["video_fps"])
            except ValueError as exc:
                raise ValueError(
                    f"row {row_number}, sample {row['sample_id']}: "
                    "video_frames_written or video_fps is invalid"
                ) from exc
            if frames <= 0 or not math.isfinite(fps) or fps <= 0:
                raise ValueError(
                    f"row {row_number}, sample {row['sample_id']}: "
                    "video frame count and fps must be finite and positive"
                )
            records.append(
                VideoSampleRecord(
                    sample_id=row["sample_id"],
                    session_id=row["session_id"],
                    archive_session_id=row["archive_session_id"],
                    person_id=row["person_id"],
                    environment_id=row["environment_id"],
                    action_id=row["action_id"],
                    trial_id=row["trial_id"],
                    video_path=video_path,
                    video_frames_written=frames,
                    video_fps=fps,
                    status=row["status"],
                    sync_quality_flags=row["sync_quality_flags"],
                )
            )
    if not records:
        raise ValueError(f"manifest contains no eligible video samples: {manifest_path}")
    sample_ids = [record.sample_id for record in records]
    if len(sample_ids) != len(set(sample_ids)):
        raise ValueError("manifest contains duplicate sample IDs")
    return records
