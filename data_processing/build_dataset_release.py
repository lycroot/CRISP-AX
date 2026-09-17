#!/usr/bin/env python3
"""Create a named dataset release from a canonical anonymized archive."""

from __future__ import annotations

import argparse
import csv
import os
import shutil
from collections import Counter
from datetime import datetime
from pathlib import Path
from typing import Any


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_DATASET_NAME = "AXHome-MM-v1"
DEFAULT_SOURCE_ARCHIVE = "sliced_windows_final_anonymized_canonical_session_20260705"
DEFAULT_OUTPUT_DIR = f"dataset_releases/{DEFAULT_DATASET_NAME}"
OMIT_RELEASE_INDEX_FIELDS = {
    "source_video_path",
    "source_csi_path",
    "source_metadata_path",
}


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


def rel_to(path: Path, base: Path) -> str:
    return str(path.resolve().relative_to(base.resolve()))


def link_or_copy(src: Path, dst: Path, mode: str) -> str:
    dst.parent.mkdir(parents=True, exist_ok=True)
    if dst.exists():
        raise FileExistsError(f"destination exists: {dst}")
    if mode == "copy":
        shutil.copy2(src, dst)
        return "copy"
    try:
        os.link(src, dst)
        return "hardlink"
    except OSError:
        shutil.copy2(src, dst)
        return "copy_fallback"


def source_to_release_path(
    root: Path,
    source_archive: Path,
    release_dir: Path,
    source_rel: str,
) -> tuple[Path, str]:
    src = project_path(root, source_rel)
    source_archive_rel = src.relative_to(source_archive)
    dst = release_dir / "data" / source_archive_rel
    release_rel = rel_to(dst, release_dir)
    return dst, release_rel


def write_list(path: Path, values: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(values) + "\n", encoding="utf-8")


def build_release(
    root: Path,
    source_archive: Path,
    release_dir: Path,
    dataset_name: str,
    link_mode: str,
) -> dict[str, Any]:
    source_index = source_archive / "manifests" / "final_archive_index.csv"
    source_report = source_archive / "reports" / "final_archive_report.md"
    rows = read_csv(source_index)
    fields = [field for field in rows[0].keys() if field not in OMIT_RELEASE_INDEX_FIELDS] if rows else []

    release_rows: list[dict[str, Any]] = []
    link_modes = Counter()
    for row in rows:
        updated = dict(row)
        for key in ("archive_video_path", "archive_csi_path", "archive_metadata_path"):
            src = project_path(root, row[key])
            dst, release_rel = source_to_release_path(root, source_archive, release_dir, row[key])
            mode = link_or_copy(src, dst, link_mode)
            link_modes[(key, mode)] += 1
            updated[key] = release_rel
        release_rows.append(updated)

    manifests_dir = release_dir / "manifests"
    reports_dir = release_dir / "reports"
    manifests_dir.mkdir(parents=True, exist_ok=True)
    reports_dir.mkdir(parents=True, exist_ok=True)

    index_path = manifests_dir / "archive_index.csv"
    write_csv(index_path, release_rows, fields)
    write_list(manifests_dir / "sample_ids.txt", [row["sample_id"] for row in release_rows])
    write_list(manifests_dir / "video_files.txt", [row["archive_video_path"] for row in release_rows])
    write_list(manifests_dir / "csi_files.txt", [row["archive_csi_path"] for row in release_rows])
    write_list(manifests_dir / "metadata_files.txt", [row["archive_metadata_path"] for row in release_rows])
    all_files: list[str] = []
    for row in release_rows:
        all_files.extend([row["archive_video_path"], row["archive_csi_path"], row["archive_metadata_path"]])
    write_list(manifests_dir / "all_files.txt", all_files)

    if source_report.is_file():
        shutil.copy2(source_report, reports_dir / "source_canonical_archive_report.md")

    by_archive_session = Counter(row["archive_session_id"] for row in release_rows)
    by_capture_session = Counter(row["session_id"] for row in release_rows)
    by_action = Counter(row["action_id"] for row in release_rows)
    by_source = Counter(row["source_collection"] for row in release_rows)
    s00_mapping = Counter(
        (row["person_id"], row["environment_id"], row["archive_session_id"])
        for row in release_rows
        if row["session_id"] == "S00"
    )
    total_video = sum(int(row["video_size_bytes"]) for row in release_rows)
    total_csi = sum(int(row["csi_size_bytes"]) for row in release_rows)
    total_metadata = sum(int(row["metadata_size_bytes"] or 0) for row in release_rows)

    summary = {
        "dataset_name": dataset_name,
        "release_dir": rel_to(release_dir, root),
        "source_archive": rel_to(source_archive, root),
        "samples": len(release_rows),
        "video_files": len(release_rows),
        "csi_files": len(release_rows),
        "metadata_files": len(release_rows),
        "total_video_mib": round(total_video / 1024 / 1024, 2),
        "total_csi_mib": round(total_csi / 1024 / 1024, 2),
        "total_metadata_mib": round(total_metadata / 1024 / 1024, 2),
        "archive_sessions": dict(sorted(by_archive_session.items())),
        "capture_sessions": dict(sorted(by_capture_session.items())),
        "actions": dict(sorted(by_action.items())),
        "source_collections": dict(sorted(by_source.items())),
        "s00_mapping": {"/".join(key): value for key, value in sorted(s00_mapping.items())},
        "link_modes": {"/".join(key): value for key, value in sorted(link_modes.items())},
    }

    write_release_report(release_dir, summary)
    write_readme(release_dir, summary)
    (release_dir / "DATASET_VERSION.txt").write_text(
        f"{dataset_name}\ncreated_at={datetime.now().strftime('%Y-%m-%d %H:%M:%S %z')}\n",
        encoding="utf-8",
    )
    return summary


def write_release_report(release_dir: Path, summary: dict[str, Any]) -> None:
    path = release_dir / "reports" / "release_report.md"
    lines = [
        f"# {summary['dataset_name']} Release Report",
        "",
        f"- Release directory: `{summary['release_dir']}`",
        f"- Source archive: `{summary['source_archive']}`",
        f"- Samples: {summary['samples']}",
        f"- Video files: {summary['video_files']}",
        f"- CSI files: {summary['csi_files']}",
        f"- Metadata files: {summary['metadata_files']}",
        f"- Total video size: {summary['total_video_mib']} MiB",
        f"- Total CSI size: {summary['total_csi_mib']} MiB",
        f"- Total metadata size: {summary['total_metadata_mib']} MiB",
        f"- Source collections: {summary['source_collections']}",
        "",
        "## S00 Placement",
        "",
        "S00 is treated as a supplemental capture container. Samples are placed under",
        "the canonical archive session determined by person_id and environment_id.",
        "",
        "| person/environment/archive_session | samples |",
        "|---|---:|",
    ]
    for key, value in summary["s00_mapping"].items():
        lines.append(f"| {key} | {value} |")
    lines.extend(["", "## Archive Session Counts", "", "| session | samples |", "|---|---:|"])
    for key, value in summary["archive_sessions"].items():
        lines.append(f"| {key} | {value} |")
    lines.extend(["", "## Action Counts", "", "| action | samples |", "|---|---:|"])
    for key, value in summary["actions"].items():
        lines.append(f"| {key} | {value} |")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def write_readme(release_dir: Path, summary: dict[str, Any]) -> None:
    path = release_dir / "README.md"
    lines = [
        f"# {summary['dataset_name']}",
        "",
        "AX protocol multimodal home-environment dataset release.",
        "",
        "## Contents",
        "",
        "- `data/`: anonymized action-window videos, sliced CSI files, and metadata.",
        "- `manifests/archive_index.csv`: primary sample index.",
        "- `manifests/video_files.txt`: release-relative video file list.",
        "- `manifests/csi_files.txt`: release-relative CSI file list.",
        "- `manifests/metadata_files.txt`: release-relative metadata file list.",
        "- `reports/release_report.md`: release summary and counts.",
        "",
        "## Layout",
        "",
        "```text",
        "data/",
        "  Sxx/",
        "    video/<action>/<sample_id>.mp4",
        "    csi/<action>/<sample_id>.dat",
        "    metadata/<action>/<sample_id>.json",
        "```",
        "",
        "## Summary",
        "",
        f"- Samples: {summary['samples']}",
        f"- Archive sessions: {len(summary['archive_sessions'])}",
        f"- Actions: {len(summary['actions'])}",
        f"- Video size: {summary['total_video_mib']} MiB",
        f"- CSI size: {summary['total_csi_mib']} MiB",
        "",
        "## Session Fields",
        "",
        "- `session_id`: original capture/session ID encoded in the sample source.",
        "- `archive_session_id`: canonical release session directory.",
        "",
        "S00 supplemental samples are not stored under `data/S00`; they are placed",
        "under their canonical session while retaining `session_id=S00` in the index.",
        "",
        "## Path Semantics",
        "",
        "All paths in `manifests/archive_index.csv` and the `*_files.txt` lists are",
        "relative to this release directory. The portable data paths are",
        "`archive_video_path`, `archive_csi_path`, and `archive_metadata_path`.",
        "Build-workspace source paths are intentionally omitted from the release index.",
        "",
        "## Notes",
        "",
        "- Videos are anonymized face-mosaic outputs.",
        "- CSI and video windows are action-window slices based on synchronized cue times.",
        "- Raw capture files are not included in this release directory.",
    ]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build a named dataset release directory.")
    parser.add_argument("--project-root", default=str(PROJECT_ROOT))
    parser.add_argument("--dataset-name", default=DEFAULT_DATASET_NAME)
    parser.add_argument("--source-archive", default=DEFAULT_SOURCE_ARCHIVE)
    parser.add_argument("--output-dir", default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--link-mode", choices=["hardlink", "copy"], default="hardlink")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    root = Path(args.project_root).resolve()
    source_archive = project_path(root, args.source_archive)
    release_dir = project_path(root, args.output_dir)
    if not source_archive.is_dir():
        raise FileNotFoundError(f"source archive not found: {source_archive}")
    if release_dir.exists():
        raise FileExistsError(f"release directory already exists: {release_dir}")
    if root != release_dir and root not in release_dir.parents:
        raise ValueError(f"release directory must be inside project root: {release_dir}")
    summary = build_release(root, source_archive, release_dir, args.dataset_name, args.link_mode)
    print(f"dataset_name={summary['dataset_name']}")
    print(f"release_dir={release_dir}")
    print(f"samples={summary['samples']}")
    print(f"video_files={summary['video_files']}")
    print(f"csi_files={summary['csi_files']}")
    print(f"metadata_files={summary['metadata_files']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
