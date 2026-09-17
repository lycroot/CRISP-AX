#!/usr/bin/env python3
"""Build a final by-session archive with anonymized videos and sliced CSI.

The archive is derived from existing checked slice/anonymization indexes. It
does not modify raw CSI/video files or the original sliced window directories.
"""

from __future__ import annotations

import argparse
import csv
import os
import shutil
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Any


PROJECT_ROOT = Path(__file__).resolve().parents[1]

DEFAULT_CLEAN_FINAL_INDEX = (
    "quality_report_deep/"
    "deepmosaics_face_anonymization_clean_6979_preserve8m_final/"
    "reports/final_video_index.csv"
)
DEFAULT_S00_SLICE_INDEX = "sliced_windows_resample_20260705_s00/slice_index.csv"
DEFAULT_S00_ANON_INDEX = (
    "quality_report_deep/"
    "deepmosaics_face_anonymization_s00_resample_46_preserve16m_png_local_gpu/"
    "reports/anonymization_index.csv"
)
DEFAULT_OUTPUT_DIR = "sliced_windows_final_anonymized_by_session_20260705"


ARCHIVE_FIELDS = [
    "sample_id",
    "session_id",
    "archive_session_id",
    "person_id",
    "environment_id",
    "action_id",
    "trial_id",
    "source_collection",
    "final_video_source_type",
    "archive_video_path",
    "archive_csi_path",
    "archive_metadata_path",
    "source_video_path",
    "source_csi_path",
    "source_metadata_path",
    "video_link_mode",
    "csi_link_mode",
    "metadata_link_mode",
    "video_size_bytes",
    "csi_size_bytes",
    "metadata_size_bytes",
    "csi_packets_expected",
    "csi_packets_written",
    "video_frames_expected",
    "video_frames_written",
    "video_fps",
    "sync_quality_flags",
    "status",
    "notes",
]


@dataclass
class ArchiveRecord:
    sample_id: str
    session_id: str
    archive_session_id: str
    person_id: str
    environment_id: str
    action_id: str
    trial_id: str
    source_collection: str
    final_video_source_type: str
    source_video_path: Path
    source_csi_path: Path
    source_metadata_path: Path | None
    csi_packets_expected: str
    csi_packets_written: str
    video_frames_expected: str
    video_frames_written: str
    video_fps: str
    sync_quality_flags: str
    notes: str = ""


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as f:
        return list(csv.DictReader(f))


def write_csv(path: Path, rows: list[dict[str, Any]], fields: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            writer.writerow({field: row.get(field, "") for field in fields})


def project_path(root: Path, value: str) -> Path:
    path = Path(value)
    if not path.is_absolute():
        path = root / path
    return path.resolve()


def project_rel(root: Path, path: Path | None) -> str:
    if path is None:
        return ""
    try:
        return str(path.resolve().relative_to(root))
    except ValueError:
        return str(path)


def metadata_from_csi_path(csi_path: Path, sample_id: str, action_id: str) -> Path | None:
    parts = list(csi_path.parts)
    try:
        csi_idx = parts.index("csi")
    except ValueError:
        return None
    return Path(*parts[:csi_idx]) / "metadata" / action_id / f"{sample_id}.json"


def link_or_copy(src: Path, dst: Path, mode: str) -> str:
    dst.parent.mkdir(parents=True, exist_ok=True)
    if dst.exists():
        raise FileExistsError(f"destination already exists: {dst}")
    if mode == "copy":
        shutil.copy2(src, dst)
        return "copy"
    try:
        os.link(src, dst)
        return "hardlink"
    except OSError:
        shutil.copy2(src, dst)
        return "copy_fallback"


def build_person_environment_session_map(clean_rows: list[dict[str, str]]) -> dict[tuple[str, str], str]:
    session_map: dict[tuple[str, str], set[str]] = {}
    for row in clean_rows:
        session_id = row.get("session_id", "")
        person_id = row.get("person_id", "")
        environment_id = row.get("environment_id", "")
        if not person_id or not environment_id or session_id == "S00":
            continue
        session_map.setdefault((person_id, environment_id), set()).add(session_id)

    canonical: dict[tuple[str, str], str] = {}
    for key, sessions in session_map.items():
        if len(sessions) == 1:
            canonical[key] = next(iter(sessions))
    return canonical


def archive_session_for_row(
    row: dict[str, str],
    s00_placement: str,
    canonical_map: dict[tuple[str, str], str],
) -> str:
    session_id = row.get("session_id", "")
    if session_id != "S00" or s00_placement == "keep_s00":
        return session_id
    key = (row.get("person_id", ""), row.get("environment_id", ""))
    archive_session_id = canonical_map.get(key)
    if not archive_session_id:
        raise ValueError(f"cannot canonicalize S00 sample {row.get('sample_id')} with key={key}")
    return archive_session_id


def clean_record(
    row: dict[str, str],
    root: Path,
    archive_session_id: str,
) -> ArchiveRecord:
    csi_path = project_path(root, row["csi_out_path"])
    metadata_path = metadata_from_csi_path(
        project_path(root, row["csi_out_path"]),
        row["sample_id"],
        row["action_id"],
    )
    return ArchiveRecord(
        sample_id=row["sample_id"],
        session_id=row["session_id"],
        archive_session_id=archive_session_id,
        person_id=row.get("person_id", ""),
        environment_id=row.get("environment_id", ""),
        action_id=row["action_id"],
        trial_id=row.get("trial_id", ""),
        source_collection="clean_6979",
        final_video_source_type=row.get("final_video_source_type", ""),
        source_video_path=project_path(root, row["final_video_path"]),
        source_csi_path=csi_path,
        source_metadata_path=metadata_path,
        csi_packets_expected=row.get("csi_packets_expected", ""),
        csi_packets_written=row.get("csi_packets_written", ""),
        video_frames_expected=row.get("video_frames_expected", ""),
        video_frames_written=row.get("video_frames_written", ""),
        video_fps=row.get("video_fps", ""),
        sync_quality_flags=row.get("sync_quality_flags", ""),
        notes=row.get("notes", ""),
    )


def s00_record(
    slice_row: dict[str, str],
    anon_row: dict[str, str],
    root: Path,
    archive_session_id: str,
) -> ArchiveRecord:
    csi_path = project_path(root, slice_row["csi_out_path"])
    metadata_path = metadata_from_csi_path(
        project_path(root, slice_row["csi_out_path"]),
        slice_row["sample_id"],
        slice_row["action_id"],
    )
    return ArchiveRecord(
        sample_id=slice_row["sample_id"],
        session_id=slice_row["session_id"],
        archive_session_id=archive_session_id,
        person_id=slice_row.get("person_id", ""),
        environment_id=slice_row.get("environment_id", ""),
        action_id=slice_row["action_id"],
        trial_id=slice_row.get("trial_id", ""),
        source_collection="s00_resample_46",
        final_video_source_type="deepmosaics_preserve16m_png",
        source_video_path=project_path(root, anon_row["anonymized_video_path"]),
        source_csi_path=csi_path,
        source_metadata_path=metadata_path,
        csi_packets_expected=slice_row.get("csi_packets_expected", ""),
        csi_packets_written=slice_row.get("csi_packets_written", ""),
        video_frames_expected=slice_row.get("video_frames_expected", ""),
        video_frames_written=slice_row.get("video_frames_written", ""),
        video_fps=slice_row.get("video_fps", ""),
        sync_quality_flags=slice_row.get("sync_quality_flags", ""),
        notes="S00 supplemental resample; anonymized locally with png temp frames and 16M H.264 target.",
    )


def load_records(args: argparse.Namespace, root: Path) -> list[ArchiveRecord]:
    clean_rows = read_csv(project_path(root, args.clean_final_index))
    s00_slice_rows = read_csv(project_path(root, args.s00_slice_index))
    s00_anon_rows = read_csv(project_path(root, args.s00_anonymization_index))

    records: list[ArchiveRecord] = []
    seen: set[str] = set()
    canonical_map = build_person_environment_session_map(clean_rows)

    for row in clean_rows:
        if row.get("integration_status") != "ok":
            raise ValueError(f"clean final row is not ok: {row.get('sample_id')}")
        archive_session_id = archive_session_for_row(row, args.s00_placement, canonical_map)
        record = clean_record(row, root, archive_session_id)
        if record.sample_id in seen:
            raise ValueError(f"duplicate sample_id in clean final index: {record.sample_id}")
        seen.add(record.sample_id)
        records.append(record)

    slice_by_id = {row["sample_id"]: row for row in s00_slice_rows if row.get("status") == "ok"}
    anon_by_id = {row["sample_id"]: row for row in s00_anon_rows if row.get("status") == "ok"}
    missing_anon = sorted(set(slice_by_id) - set(anon_by_id))
    missing_slice = sorted(set(anon_by_id) - set(slice_by_id))
    if missing_anon:
        raise ValueError(f"S00 slice rows missing anonymized outputs: {missing_anon[:10]}")
    if missing_slice:
        raise ValueError(f"S00 anonymized rows missing slice rows: {missing_slice[:10]}")

    for row in s00_slice_rows:
        if row.get("status") != "ok":
            continue
        archive_session_id = archive_session_for_row(row, args.s00_placement, canonical_map)
        record = s00_record(row, anon_by_id[row["sample_id"]], root, archive_session_id)
        if record.sample_id in seen:
            raise ValueError(f"duplicate S00 supplemental sample_id: {record.sample_id}")
        seen.add(record.sample_id)
        records.append(record)

    return records


def validate_sources(records: list[ArchiveRecord]) -> list[str]:
    errors: list[str] = []
    for record in records:
        if not record.source_video_path.is_file():
            errors.append(f"missing video: {record.sample_id} {record.source_video_path}")
        if not record.source_csi_path.is_file():
            errors.append(f"missing csi: {record.sample_id} {record.source_csi_path}")
        if record.source_metadata_path and not record.source_metadata_path.is_file():
            errors.append(f"missing metadata: {record.sample_id} {record.source_metadata_path}")
    return errors


def build_archive(records: list[ArchiveRecord], root: Path, output_dir: Path, link_mode: str) -> list[dict[str, Any]]:
    archive_rows: list[dict[str, Any]] = []
    for record in records:
        video_dst = output_dir / record.archive_session_id / "video" / record.action_id / f"{record.sample_id}.mp4"
        csi_dst = output_dir / record.archive_session_id / "csi" / record.action_id / f"{record.sample_id}.dat"
        metadata_dst = output_dir / record.archive_session_id / "metadata" / record.action_id / f"{record.sample_id}.json"

        video_mode = link_or_copy(record.source_video_path, video_dst, link_mode)
        csi_mode = link_or_copy(record.source_csi_path, csi_dst, link_mode)
        metadata_mode = ""
        metadata_size = ""
        if record.source_metadata_path:
            metadata_mode = link_or_copy(record.source_metadata_path, metadata_dst, link_mode)
            metadata_size = metadata_dst.stat().st_size

        archive_rows.append(
            {
                "sample_id": record.sample_id,
                "session_id": record.session_id,
                "archive_session_id": record.archive_session_id,
                "person_id": record.person_id,
                "environment_id": record.environment_id,
                "action_id": record.action_id,
                "trial_id": record.trial_id,
                "source_collection": record.source_collection,
                "final_video_source_type": record.final_video_source_type,
                "archive_video_path": project_rel(root, video_dst),
                "archive_csi_path": project_rel(root, csi_dst),
                "archive_metadata_path": project_rel(root, metadata_dst),
                "source_video_path": project_rel(root, record.source_video_path),
                "source_csi_path": project_rel(root, record.source_csi_path),
                "source_metadata_path": project_rel(root, record.source_metadata_path),
                "video_link_mode": video_mode,
                "csi_link_mode": csi_mode,
                "metadata_link_mode": metadata_mode,
                "video_size_bytes": video_dst.stat().st_size,
                "csi_size_bytes": csi_dst.stat().st_size,
                "metadata_size_bytes": metadata_size,
                "csi_packets_expected": record.csi_packets_expected,
                "csi_packets_written": record.csi_packets_written,
                "video_frames_expected": record.video_frames_expected,
                "video_frames_written": record.video_frames_written,
                "video_fps": record.video_fps,
                "sync_quality_flags": record.sync_quality_flags,
                "status": "ok",
                "notes": record.notes,
            }
        )
    return archive_rows


def write_file_lists(root: Path, output_dir: Path, rows: list[dict[str, Any]]) -> dict[str, Path]:
    manifest_dir = output_dir / "manifests"
    paths = {
        "sample_ids": manifest_dir / f"final_sample_ids_{len(rows)}.txt",
        "videos": manifest_dir / f"final_video_files_{len(rows)}.txt",
        "csi": manifest_dir / f"final_csi_files_{len(rows)}.txt",
        "metadata": manifest_dir / f"final_metadata_files_{len(rows)}.txt",
        "all": manifest_dir / f"final_all_files_{len(rows)}.txt",
    }
    manifest_dir.mkdir(parents=True, exist_ok=True)
    paths["sample_ids"].write_text("\n".join(row["sample_id"] for row in rows) + "\n", encoding="utf-8")
    paths["videos"].write_text("\n".join(row["archive_video_path"] for row in rows) + "\n", encoding="utf-8")
    paths["csi"].write_text("\n".join(row["archive_csi_path"] for row in rows) + "\n", encoding="utf-8")
    metadata_rows = [row["archive_metadata_path"] for row in rows if row.get("archive_metadata_path")]
    paths["metadata"].write_text("\n".join(metadata_rows) + "\n", encoding="utf-8")
    all_paths = []
    for row in rows:
        all_paths.extend([row["archive_video_path"], row["archive_csi_path"]])
        if row.get("archive_metadata_path"):
            all_paths.append(row["archive_metadata_path"])
    paths["all"].write_text("\n".join(all_paths) + "\n", encoding="utf-8")
    return paths


def write_report(root: Path, output_dir: Path, rows: list[dict[str, Any]], file_lists: dict[str, Path]) -> Path:
    report_path = output_dir / "reports" / "final_archive_report.md"
    report_path.parent.mkdir(parents=True, exist_ok=True)
    by_archive_session = Counter(row["archive_session_id"] for row in rows)
    by_capture_session = Counter(row["session_id"] for row in rows)
    by_action = Counter(row["action_id"] for row in rows)
    by_collection = Counter(row["source_collection"] for row in rows)
    video_modes = Counter(row["video_link_mode"] for row in rows)
    csi_modes = Counter(row["csi_link_mode"] for row in rows)
    metadata_modes = Counter(row["metadata_link_mode"] for row in rows)
    total_video = sum(int(row["video_size_bytes"]) for row in rows)
    total_csi = sum(int(row["csi_size_bytes"]) for row in rows)
    total_metadata = sum(int(row["metadata_size_bytes"] or 0) for row in rows)

    def mib(value: int) -> str:
        return f"{value / 1024 / 1024:.2f}"

    lines = [
        "# Final Anonymized Slice Archive Report",
        "",
        f"- Output directory: `{project_rel(root, output_dir)}`",
        f"- Total samples: {len(rows)}",
        f"- Video files: {len(rows)}",
        f"- CSI files: {len(rows)}",
        f"- Metadata files: {sum(1 for row in rows if row.get('archive_metadata_path'))}",
        f"- Total video size: {mib(total_video)} MiB",
        f"- Total CSI size: {mib(total_csi)} MiB",
        f"- Total metadata size: {mib(total_metadata)} MiB",
        f"- Source collections: {dict(sorted(by_collection.items()))}",
        f"- Capture sessions: {dict(sorted(by_capture_session.items()))}",
        f"- Video link modes: {dict(sorted(video_modes.items()))}",
        f"- CSI link modes: {dict(sorted(csi_modes.items()))}",
        f"- Metadata link modes: {dict(sorted(metadata_modes.items()))}",
        "",
        "## Manifests",
        "",
        f"- Archive index CSV: `{project_rel(root, output_dir / 'manifests/final_archive_index.csv')}`",
        f"- Sample IDs: `{project_rel(root, file_lists['sample_ids'])}`",
        f"- Video file list: `{project_rel(root, file_lists['videos'])}`",
        f"- CSI file list: `{project_rel(root, file_lists['csi'])}`",
        f"- Metadata file list: `{project_rel(root, file_lists['metadata'])}`",
        f"- All file list: `{project_rel(root, file_lists['all'])}`",
        "",
        "## Archive Session Counts",
        "",
        "| session | samples |",
        "|---|---:|",
    ]
    for key, count in sorted(by_archive_session.items()):
        lines.append(f"| {key} | {count} |")
    lines.extend(["", "## Capture Session Counts", "", "| session | samples |", "|---|---:|"])
    for key, count in sorted(by_capture_session.items()):
        lines.append(f"| {key} | {count} |")
    lines.extend(["", "## Action Counts", "", "| action | samples |", "|---|---:|"])
    for key, count in sorted(by_action.items()):
        lines.append(f"| {key} | {count} |")
    report_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return report_path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Create a final by-session archive of anonymized videos plus sliced CSI."
    )
    parser.add_argument("--project-root", default=str(PROJECT_ROOT))
    parser.add_argument("--clean-final-index", default=DEFAULT_CLEAN_FINAL_INDEX)
    parser.add_argument("--s00-slice-index", default=DEFAULT_S00_SLICE_INDEX)
    parser.add_argument("--s00-anonymization-index", default=DEFAULT_S00_ANON_INDEX)
    parser.add_argument("--output-dir", default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--link-mode", choices=["hardlink", "copy"], default="hardlink")
    parser.add_argument(
        "--s00-placement",
        choices=["keep_s00", "canonical_person_environment"],
        default="keep_s00",
        help="Place S00 samples under S00 or under the unique non-S00 session for person_id+environment_id.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    root = Path(args.project_root).resolve()
    output_dir = project_path(root, args.output_dir)
    if output_dir.exists():
        raise FileExistsError(f"output directory already exists: {output_dir}")
    if root != output_dir and root not in output_dir.parents:
        raise ValueError(f"output directory must be inside project root: {output_dir}")

    records = load_records(args, root)
    errors = validate_sources(records)
    if errors:
        joined = "\n".join(errors[:50])
        raise FileNotFoundError(f"source validation failed ({len(errors)} errors):\n{joined}")

    archive_rows = build_archive(records, root, output_dir, args.link_mode)
    manifest_path = output_dir / "manifests" / "final_archive_index.csv"
    write_csv(manifest_path, archive_rows, ARCHIVE_FIELDS)
    file_lists = write_file_lists(root, output_dir, archive_rows)
    report_path = write_report(root, output_dir, archive_rows, file_lists)

    print(f"records={len(archive_rows)}")
    print(f"output_dir={output_dir}")
    print(f"manifest={manifest_path}")
    print(f"report={report_path}")
    print(f"video_files={len(archive_rows)}")
    print(f"csi_files={len(archive_rows)}")
    print(f"metadata_files={sum(1 for row in archive_rows if row.get('archive_metadata_path'))}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
