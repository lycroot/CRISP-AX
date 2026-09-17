#!/usr/bin/env python3
"""Build a read-only CSI/video action-window synchronization index."""

from __future__ import annotations

import argparse
import bisect
import csv
import re
import math
import struct
from collections import Counter, defaultdict
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from pathlib import Path


SYNC_FIELDS = [
    "sample_id",
    "session_id",
    "person_id",
    "environment_id",
    "action_id",
    "trial_id",
    "action_start_unix_ns",
    "action_end_unix_ns",
    "action_duration_sec",
    "csi_path",
    "marker_path",
    "csi_log_path",
    "csi_packet_count",
    "csi_duration_sec",
    "csi_packet_rate_hz",
    "csi_packet_start_idx",
    "csi_packet_end_idx",
    "csi_action_packet_count",
    "csi_index_method",
    "video_path",
    "video_timestamp_csv",
    "video_frame_count_total",
    "video_effective_fps",
    "frame_start_idx_raw",
    "frame_end_idx_raw",
    "raw_frame_count",
    "start_time_diff_ms",
    "end_time_diff_ms",
    "max_frame_gap_ms_in_raw_window",
    "sync_status",
    "quality_flags",
    "notes",
]


@dataclass
class CsiInfo:
    path: Path | None
    marker_path: Path | None
    log_path: Path | None
    packet_count: int = 0
    duration_sec: float = math.nan
    packet_rate_hz: float = math.nan
    packet_start_idx: int | None = None
    packet_end_idx: int | None = None
    method: str = ""
    flags: list[str] | None = None
    notes: list[str] | None = None


@dataclass
class VideoInfo:
    timestamp_csv: Path
    video_path: Path | None
    frame_indices: list[int]
    unix_ns: list[int]
    effective_fps: float

    @property
    def start_ns(self) -> int:
        return self.unix_ns[0]

    @property
    def end_ns(self) -> int:
        return self.unix_ns[-1]

    @property
    def frame_count(self) -> int:
        return len(self.unix_ns)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project-root", default=".", help="Workspace root.")
    parser.add_argument(
        "--manifest",
        default="CSI-Formal/01_manifest/collection_manifest_raw.csv",
        help="Manifest with action_start_unix_ns/action_end_unix_ns.",
    )
    parser.add_argument(
        "--output-dir",
        default="quality_report_deep/csi_video_sync_index",
        help="Directory for CSV and Markdown report outputs.",
    )
    parser.add_argument("--default-csi-fs", type=float, default=100.0)
    parser.add_argument("--large-frame-diff-ms", type=float, default=100.0)
    parser.add_argument("--large-frame-gap-ms", type=float, default=200.0)
    return parser.parse_args()


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8-sig", newline="") as f:
        return list(csv.DictReader(f))


def write_csv(path: Path, rows: list[dict[str, object]], fields: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            writer.writerow({field: row.get(field, "") for field in fields})


def to_int(value: object) -> int | None:
    try:
        if value is None or value == "":
            return None
        return int(float(str(value)))
    except ValueError:
        return None


def fmt_float(value: float | int | None, digits: int = 3) -> str:
    if value is None:
        return ""
    try:
        number = float(value)
    except (TypeError, ValueError):
        return ""
    if not math.isfinite(number):
        return ""
    return f"{number:.{digits}f}"


def rel(path: Path | None, root: Path) -> str:
    if path is None:
        return ""
    try:
        return str(path.resolve().relative_to(root.resolve()))
    except ValueError:
        return str(path)


def resolve_sample_file(root: Path, base_dir: str, session: str, name: str) -> Path | None:
    direct = root / base_dir / session / name
    if direct.exists():
        return direct
    matches = sorted((root / base_dir).glob(f"*/{name}"))
    return matches[0] if matches else None


def read_header_at(path: Path, offset: int) -> tuple[int, int] | None:
    with path.open("rb") as f:
        f.seek(offset)
        header = f.read(272)
    if len(header) < 272:
        return None
    csi_size = struct.unpack_from("<I", header, 0)[0]
    ftm_clock = struct.unpack_from("<I", header, 8)[0]
    return csi_size, ftm_clock


LOG_PACKET_RE = re.compile(r"^\[(\d\d):(\d\d):(\d\d)\.(\d{1,6})\] Subcarrier count:", re.MULTILINE)
LOCAL_TZ = timezone(timedelta(hours=8))


def parse_log_packet_times(log_path: Path | None, sample_date: str) -> list[int]:
    if log_path is None or not log_path.exists() or not sample_date:
        return []

    try:
        base_date = datetime.strptime(sample_date, "%Y-%m-%d").date()
    except ValueError:
        return []

    text = log_path.read_text(encoding="utf-8", errors="ignore")
    packet_times: list[int] = []
    day_offset = 0
    prev_ns: int | None = None
    for match in LOG_PACKET_RE.finditer(text):
        hour, minute, second, micro = match.groups()
        dt = datetime(
            base_date.year,
            base_date.month,
            base_date.day,
            int(hour),
            int(minute),
            int(second),
            int(micro.ljust(6, "0")),
            tzinfo=LOCAL_TZ,
        ) + timedelta(days=day_offset)
        ns = int(dt.timestamp() * 1_000_000_000)
        if prev_ns is not None and ns < prev_ns:
            day_offset += 1
            dt += timedelta(days=1)
            ns = int(dt.timestamp() * 1_000_000_000)
        packet_times.append(ns)
        prev_ns = ns
    return packet_times


def action_range_from_times(times_ns: list[int], action_start_ns: int, action_end_ns: int) -> tuple[int, int]:
    start_idx = bisect.bisect_left(times_ns, action_start_ns)
    end_idx = bisect.bisect_right(times_ns, action_end_ns) - 1
    start_idx = max(0, min(len(times_ns) - 1, start_idx))
    end_idx = max(0, min(len(times_ns) - 1, end_idx))
    return start_idx, end_idx


def estimate_csi_info(
    root: Path,
    row: dict[str, str],
    action_start_ns: int,
    action_end_ns: int,
    default_fs: float,
) -> CsiInfo:
    sample_id = row["sample_id"]
    session = row["session_id"]
    csi_path = resolve_sample_file(root, "CSI-Formal/00_raw/csi", session, f"{sample_id}.dat")
    marker_path = resolve_sample_file(root, "CSI-Formal/00_raw/logs", session, f"{sample_id}_markers.log")
    log_path = resolve_sample_file(root, "CSI-Formal/00_raw/logs", session, f"{sample_id}.log")
    flags: list[str] = []
    notes: list[str] = []

    if csi_path is None:
        return CsiInfo(csi_path, marker_path, log_path, flags=["missing_csi_file"], notes=notes)

    first = read_header_at(csi_path, 0)
    if first is None:
        return CsiInfo(csi_path, marker_path, log_path, flags=["empty_or_invalid_csi_file"], notes=notes)

    csi_size, _first_ftm = first
    packet_size = 272 + csi_size
    file_size = csi_path.stat().st_size
    if packet_size <= 272:
        return CsiInfo(csi_path, marker_path, log_path, flags=["invalid_csi_packet_size"], notes=notes)

    packet_count = file_size // packet_size
    trailing_bytes = file_size % packet_size
    if trailing_bytes:
        flags.append("csi_trailing_bytes")
        notes.append(f"trailing_bytes={trailing_bytes}")
    if packet_count <= 0:
        flags.append("empty_or_invalid_csi_file")
        return CsiInfo(csi_path, marker_path, log_path, flags=flags, notes=notes)

    log_packet_times = parse_log_packet_times(log_path, row.get("date", ""))
    use_log_times = False
    if len(log_packet_times) == packet_count:
        use_log_times = True
        duration_sec = (log_packet_times[-1] - log_packet_times[0]) / 1e9 if packet_count > 1 else 0.0
        method = "log_subcarrier_timestamp"
    elif log_packet_times:
        flags.append("csi_log_packet_count_mismatch_dat")
        notes.append(f"log_packet_count={len(log_packet_times)}")
    else:
        flags.append("missing_csi_packet_log_times")

    sample_start_ns = to_int(row.get("sample_start_unix_ns"))
    sample_end_ns = to_int(row.get("sample_end_unix_ns"))
    if sample_start_ns is None:
        sample_start_ns = action_start_ns
        flags.append("missing_sample_start_unix_ns")

    if use_log_times:
        pass
    elif sample_end_ns is not None and sample_end_ns > sample_start_ns:
        duration_sec = (sample_end_ns - sample_start_ns) / 1e9
        method = "manifest_sample_window_linear"
    else:
        flags.append("missing_sample_end_unix_ns")
        duration_sec = packet_count / default_fs
        method = f"default_{default_fs:g}hz"

    manifest_packet_count = to_int(row.get("frame_count"))
    if manifest_packet_count is not None and manifest_packet_count > 0 and abs(manifest_packet_count - packet_count) > 1:
        flags.append("csi_packet_count_mismatch_manifest")
        notes.append(f"manifest_frame_count={manifest_packet_count}")

    packet_rate_hz = (packet_count - 1) / duration_sec if duration_sec > 0 and packet_count > 1 else math.nan

    if packet_count <= 1:
        start_idx = 0
        end_idx = 0
    elif use_log_times:
        if action_end_ns < log_packet_times[0] or action_start_ns > log_packet_times[-1]:
            flags.append("action_outside_csi_range")
        start_idx, end_idx = action_range_from_times(log_packet_times, action_start_ns, action_end_ns)
        if end_idx < start_idx:
            flags.append("invalid_csi_action_range")
    else:
        csi_start_ns = sample_start_ns
        csi_end_ns = int(csi_start_ns + duration_sec * 1e9)
        if action_end_ns < csi_start_ns or action_start_ns > csi_end_ns:
            flags.append("action_outside_csi_range")

        rel_start_sec = (action_start_ns - csi_start_ns) / 1e9
        rel_end_sec = (action_end_ns - csi_start_ns) / 1e9
        start_float = rel_start_sec / duration_sec * (packet_count - 1) if duration_sec > 0 else 0
        end_float = rel_end_sec / duration_sec * (packet_count - 1) if duration_sec > 0 else packet_count - 1
        start_idx = max(0, min(packet_count - 1, math.ceil(start_float)))
        end_idx = max(0, min(packet_count - 1, math.floor(end_float)))
        if end_idx < start_idx:
            flags.append("invalid_csi_action_range")

    if marker_path is None:
        flags.append("missing_marker_file")
    if log_path is None:
        flags.append("missing_csi_log_file")

    return CsiInfo(
        path=csi_path,
        marker_path=marker_path,
        log_path=log_path,
        packet_count=int(packet_count),
        duration_sec=duration_sec,
        packet_rate_hz=packet_rate_hz,
        packet_start_idx=int(start_idx),
        packet_end_idx=int(end_idx),
        method=method,
        flags=flags,
        notes=notes,
    )


def load_video_infos(root: Path) -> list[VideoInfo]:
    infos: list[VideoInfo] = []
    for timestamp_csv in sorted((root / "Video").glob("**/*video_timestamps.csv")):
        frame_indices: list[int] = []
        unix_ns: list[int] = []
        with timestamp_csv.open("r", encoding="utf-8-sig", newline="") as f:
            reader = csv.DictReader(f)
            for row in reader:
                frame_idx = to_int(row.get("frame_idx"))
                ts = to_int(row.get("unix_time_ns"))
                if frame_idx is None or ts is None:
                    continue
                frame_indices.append(frame_idx)
                unix_ns.append(ts)

        if not unix_ns:
            continue

        duration_sec = (unix_ns[-1] - unix_ns[0]) / 1e9
        effective_fps = (len(unix_ns) - 1) / duration_sec if duration_sec > 0 and len(unix_ns) > 1 else math.nan
        video_path = timestamp_csv.with_name(timestamp_csv.name.replace("_timestamps.csv", ".mkv"))
        infos.append(
            VideoInfo(
                timestamp_csv=timestamp_csv,
                video_path=video_path if video_path.exists() else None,
                frame_indices=frame_indices,
                unix_ns=unix_ns,
                effective_fps=effective_fps,
            )
        )
    return infos


def nearest_frame(video: VideoInfo, target_ns: int) -> tuple[int, int, float]:
    pos = bisect.bisect_left(video.unix_ns, target_ns)
    candidates = []
    if pos < len(video.unix_ns):
        candidates.append(pos)
    if pos > 0:
        candidates.append(pos - 1)
    best_pos = min(candidates, key=lambda idx: abs(video.unix_ns[idx] - target_ns))
    diff_ms = (video.unix_ns[best_pos] - target_ns) / 1e6
    return video.frame_indices[best_pos], best_pos, diff_ms


def max_gap_ms(video: VideoInfo, start_pos: int, end_pos: int) -> float:
    if end_pos <= start_pos:
        return 0.0
    values = video.unix_ns[start_pos : end_pos + 1]
    gaps = [(values[i] - values[i - 1]) / 1e6 for i in range(1, len(values))]
    return max(gaps) if gaps else 0.0


def match_video(videos: list[VideoInfo], action_start_ns: int, action_end_ns: int) -> VideoInfo | None:
    containing = [v for v in videos if v.start_ns <= action_start_ns and action_end_ns <= v.end_ns]
    if containing:
        return min(containing, key=lambda v: (v.end_ns - v.start_ns, str(v.timestamp_csv)))
    overlapping = [v for v in videos if v.start_ns <= action_end_ns and action_start_ns <= v.end_ns]
    if overlapping:
        return max(overlapping, key=lambda v: min(action_end_ns, v.end_ns) - max(action_start_ns, v.start_ns))
    return None


def build_rows(root: Path, manifest_rows: list[dict[str, str]], videos: list[VideoInfo], args: argparse.Namespace) -> list[dict[str, object]]:
    output_rows: list[dict[str, object]] = []
    for row in manifest_rows:
        sample_id = row.get("sample_id", "")
        action_start_ns = to_int(row.get("action_start_unix_ns"))
        action_end_ns = to_int(row.get("action_end_unix_ns"))
        flags: list[str] = []
        notes: list[str] = []

        if action_start_ns is None or action_end_ns is None:
            flags.append("missing_action_time")
            action_start_ns = action_start_ns or 0
            action_end_ns = action_end_ns or action_start_ns
        if action_end_ns <= action_start_ns:
            flags.append("invalid_action_time")

        csi = estimate_csi_info(root, row, action_start_ns, action_end_ns, args.default_csi_fs)
        flags.extend(csi.flags or [])
        notes.extend(csi.notes or [])

        video = match_video(videos, action_start_ns, action_end_ns)
        frame_start = frame_end = ""
        raw_frame_count = ""
        start_diff_ms = end_diff_ms = ""
        max_gap = ""
        video_path = ""
        timestamp_csv = ""
        total_video_frames = ""
        video_fps = ""
        if video is None:
            flags.append("no_matching_video_timestamp_csv")
        else:
            timestamp_csv = rel(video.timestamp_csv, root)
            video_path = rel(video.video_path, root)
            total_video_frames = video.frame_count
            video_fps = fmt_float(video.effective_fps)
            frame_start, start_pos, start_diff_ms_f = nearest_frame(video, action_start_ns)
            frame_end, end_pos, end_diff_ms_f = nearest_frame(video, action_end_ns)
            start_diff_ms = fmt_float(start_diff_ms_f)
            end_diff_ms = fmt_float(end_diff_ms_f)
            raw_frame_count = int(frame_end) - int(frame_start) + 1
            max_gap_f = max_gap_ms(video, min(start_pos, end_pos), max(start_pos, end_pos))
            max_gap = fmt_float(max_gap_f)
            if raw_frame_count <= 0:
                flags.append("invalid_video_frame_range")
            if abs(start_diff_ms_f) > args.large_frame_diff_ms:
                flags.append("large_start_time_diff")
            if abs(end_diff_ms_f) > args.large_frame_diff_ms:
                flags.append("large_end_time_diff")
            if max_gap_f > args.large_frame_gap_ms:
                flags.append("large_frame_gap_in_raw_window")

        csi_action_packet_count = ""
        if csi.packet_start_idx is not None and csi.packet_end_idx is not None:
            csi_action_packet_count = csi.packet_end_idx - csi.packet_start_idx + 1
            if csi_action_packet_count <= 0:
                flags.append("invalid_csi_action_count")

        sync_status = "ok" if not flags else "warn"
        output_rows.append(
            {
                "sample_id": sample_id,
                "session_id": row.get("session_id", ""),
                "person_id": row.get("person_id", ""),
                "environment_id": row.get("environment_id", ""),
                "action_id": row.get("action_id", ""),
                "trial_id": row.get("trial_id", ""),
                "action_start_unix_ns": action_start_ns,
                "action_end_unix_ns": action_end_ns,
                "action_duration_sec": fmt_float((action_end_ns - action_start_ns) / 1e9),
                "csi_path": rel(csi.path, root),
                "marker_path": rel(csi.marker_path, root),
                "csi_log_path": rel(csi.log_path, root),
                "csi_packet_count": csi.packet_count or "",
                "csi_duration_sec": fmt_float(csi.duration_sec),
                "csi_packet_rate_hz": fmt_float(csi.packet_rate_hz),
                "csi_packet_start_idx": csi.packet_start_idx if csi.packet_start_idx is not None else "",
                "csi_packet_end_idx": csi.packet_end_idx if csi.packet_end_idx is not None else "",
                "csi_action_packet_count": csi_action_packet_count,
                "csi_index_method": csi.method,
                "video_path": video_path,
                "video_timestamp_csv": timestamp_csv,
                "video_frame_count_total": total_video_frames,
                "video_effective_fps": video_fps,
                "frame_start_idx_raw": frame_start,
                "frame_end_idx_raw": frame_end,
                "raw_frame_count": raw_frame_count,
                "start_time_diff_ms": start_diff_ms,
                "end_time_diff_ms": end_diff_ms,
                "max_frame_gap_ms_in_raw_window": max_gap,
                "sync_status": sync_status,
                "quality_flags": ";".join(sorted(set(flags))) if flags else "ok",
                "notes": ";".join(notes),
            }
        )
    return output_rows


def summarize(rows: list[dict[str, object]]) -> dict[str, object]:
    flag_counter: Counter[str] = Counter()
    session_counter: Counter[str] = Counter()
    action_counter: Counter[str] = Counter()
    env_counter: Counter[str] = Counter()
    frame_counts: list[int] = []
    csi_counts: list[int] = []
    start_diffs: list[float] = []
    end_diffs: list[float] = []

    for row in rows:
        session_counter[str(row.get("session_id", ""))] += 1
        action_counter[str(row.get("action_id", ""))] += 1
        env_counter[str(row.get("environment_id", ""))] += 1
        for flag in str(row.get("quality_flags", "")).split(";"):
            if flag:
                flag_counter[flag] += 1
        for src, dest in [
            ("raw_frame_count", frame_counts),
            ("csi_action_packet_count", csi_counts),
        ]:
            value = to_int(row.get(src))
            if value is not None:
                dest.append(value)
        for src, dest in [
            ("start_time_diff_ms", start_diffs),
            ("end_time_diff_ms", end_diffs),
        ]:
            try:
                value = float(str(row.get(src, "")))
            except ValueError:
                continue
            if math.isfinite(value):
                dest.append(value)

    return {
        "flag_counter": flag_counter,
        "session_counter": session_counter,
        "action_counter": action_counter,
        "env_counter": env_counter,
        "frame_counts": frame_counts,
        "csi_counts": csi_counts,
        "start_diffs": start_diffs,
        "end_diffs": end_diffs,
    }


def describe(values: list[float | int]) -> str:
    if not values:
        return "n/a"
    sorted_values = sorted(float(v) for v in values)
    n = len(sorted_values)

    def pct(p: float) -> float:
        idx = min(n - 1, max(0, int(round((n - 1) * p))))
        return sorted_values[idx]

    return (
        f"count={n}, min={sorted_values[0]:.3f}, "
        f"p50={pct(0.5):.3f}, p95={pct(0.95):.3f}, max={sorted_values[-1]:.3f}"
    )


def write_report(path: Path, rows: list[dict[str, object]], videos: list[VideoInfo], summary: dict[str, object], root: Path, manifest: Path) -> None:
    flag_counter: Counter[str] = summary["flag_counter"]
    ok_count = flag_counter.get("ok", 0)
    warn_count = len(rows) - ok_count
    warn_rows = [row for row in rows if row.get("quality_flags") != "ok"]
    lines = [
        "# CSI/Video Synchronization Index Report",
        "",
        "## Inputs",
        "",
        f"- Project root: `{root}`",
        f"- Manifest: `{rel(manifest, root)}`",
        f"- Video timestamp CSV files: {len(videos)}",
        f"- Manifest samples: {len(rows)}",
        "",
        "## Outputs",
        "",
        f"- Sync index: `{rel(path.with_name('csi_video_sync_index.csv'), root)}`",
        f"- Report: `{rel(path, root)}`",
        "",
        "## Status Summary",
        "",
        f"- `ok`: {ok_count}",
        f"- `warn`: {warn_count}",
        "",
        "## Quality Flags",
        "",
    ]
    for flag, count in flag_counter.most_common():
        lines.append(f"- `{flag}`: {count}")
    lines.extend(
        [
            "",
            "## Distributions",
            "",
            f"- Raw video action frames: {describe(summary['frame_counts'])}",
            f"- CSI action packets: {describe(summary['csi_counts'])}",
            f"- Video start nearest-frame diff ms: {describe(summary['start_diffs'])}",
            f"- Video end nearest-frame diff ms: {describe(summary['end_diffs'])}",
            "",
            "## Warning Samples",
            "",
        ]
    )
    if warn_rows:
        for row in warn_rows[:50]:
            lines.append(
                f"- `{row.get('sample_id')}`: `{row.get('quality_flags')}`"
            )
        if len(warn_rows) > 50:
            lines.append(f"- ... {len(warn_rows) - 50} additional warning rows omitted")
    else:
        lines.append("- None")
    lines.extend(
        [
            "",
            "## Notes",
            "",
            "- Raw CSI, marker, timestamp, and video files were not modified.",
            "- `MARKER_ACTION_START` and `MARKER_ACTION_END` are treated as capture-script cue times, not manually verified motion boundaries.",
            "- Video frame ranges are selected from per-frame `unix_time_ns` values, not from encoded-video offsets.",
            "- CSI packet ranges prefer per-packet `Subcarrier count` timestamps from CSI `.log` files.",
            "- If CSI log packet timestamps are missing or do not match `.dat` packet count, the script falls back to manifest `sample_start_unix_ns/sample_end_unix_ns` linear mapping.",
        ]
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    args = parse_args()
    root = Path(args.project_root).resolve()
    manifest = (root / args.manifest).resolve()
    output_dir = (root / args.output_dir).resolve()

    manifest_rows = read_csv(manifest)
    videos = load_video_infos(root)
    rows = build_rows(root, manifest_rows, videos, args)

    index_path = output_dir / "csi_video_sync_index.csv"
    report_path = output_dir / "csi_video_sync_index_report.md"
    write_csv(index_path, rows, SYNC_FIELDS)
    write_report(report_path, rows, videos, summarize(rows), root, manifest)

    flag_counter = summarize(rows)["flag_counter"]
    print(f"wrote {index_path}")
    print(f"wrote {report_path}")
    print(f"samples={len(rows)} ok={flag_counter.get('ok', 0)} warn={len(rows) - flag_counter.get('ok', 0)}")


if __name__ == "__main__":
    main()
