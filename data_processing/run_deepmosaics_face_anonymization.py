#!/usr/bin/env python3
"""DeepMosaics face-anonymization batch wrapper.

This script wraps ``/data/DeepMosaics-master/deepmosaic.py`` via ``subprocess``
to add face mosaics to the sliced window videos referenced by
``sliced_windows_by_session/slice_index.csv``.

Design constraints (see AGENTS.md and docs/PROJECT_LAYOUT_AFTER_CLEANUP.zh.md):

* Never modify, move, or delete raw data (``Video/``, ``CSI-Formal/00_raw/``,
  ``sliced_windows_by_session/``, ``sliced_windows_holdout/``).
* Never modify the DeepMosaics source tree; only call ``deepmosaic.py``.
* Never write outputs into the DeepMosaics tree (``/data/DeepMosaics-master/result``).
* All outputs land under
  ``quality_report_deep/deepmosaics_face_anonymization/{outputs,staging,logs,previews,reports}``.

The script supports three modes:

* ``--dry-run``  : plan only, no subprocess, no video output (default).
* ``--preview``  : process a small sample (default 3) and emit start/mid/end
                   thumbnails for source + anonymized videos.
* ``--apply``    : process every selected sample; resumable (existing outputs
                   are skipped unless ``--overwrite`` is given).

Every DeepMosaics invocation runs with ``cwd = deepmosaics_root`` and
``stdin = DEVNULL`` so the interactive ``input()`` prompts in
``deepmosaic.py`` cannot hang the batch.
"""

from __future__ import annotations

import argparse
import csv
import inspect
import os
import re
import select
import shutil
import subprocess
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Iterable

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

PROJECT_ROOT = Path(__file__).resolve().parent.parent

DEFAULT_DEEPMOSAICS_ROOT = Path("/data/DeepMosaics-master")
DEFAULT_MODEL_PATH = DEFAULT_DEEPMOSAICS_ROOT / "pretrained_models" / "mosaic" / "add_face.pth"
DEFAULT_INPUT_INDEX = "sliced_windows_by_session/slice_index.csv"
DEFAULT_OUTPUT_BASE_REL = "quality_report_deep/deepmosaics_face_anonymization"
DEFAULT_TIMEOUT_SECONDS = 1800
PREVIEW_DEFAULT_LIMIT = 3

# Raw / protected data zones — writing here is forbidden.
FORBIDDEN_WRITE_PREFIXES: tuple[str, ...] = (
    "Video/",
    "CSI-Formal/00_raw/",
    "CSI-Formal/01_manifest/",
    "sliced_windows_by_session/",
    "sliced_windows_holdout/",
    "sliced_windows_quarantine/",
)

# The only place this script is allowed to write.
ALLOWED_OUTPUT_PREFIXES: tuple[str, ...] = (
    DEFAULT_OUTPUT_BASE_REL + "/",
)

# Status vocabulary (spec, fixed order).
STATUS_MISSING_SOURCE = "missing_source"
STATUS_MISSING_MODEL = "missing_model"
STATUS_SKIPPED_EXISTS = "skipped_exists"
STATUS_FAILED = "deepmosaics_failed"
STATUS_MISSING_OUTPUT = "missing_output"
STATUS_MULTIPLE_OUTPUTS = "multiple_outputs"
STATUS_INVALID_OUTPUT = "invalid_output"
STATUS_OK = "ok"

ALL_STATUSES: tuple[str, ...] = (
    STATUS_OK,
    STATUS_MISSING_SOURCE,
    STATUS_MISSING_MODEL,
    STATUS_SKIPPED_EXISTS,
    STATUS_FAILED,
    STATUS_MISSING_OUTPUT,
    STATUS_MULTIPLE_OUTPUTS,
    STATUS_INVALID_OUTPUT,
)

FAILED_PROGRESS_STATUSES: tuple[str, ...] = (
    STATUS_FAILED,
    STATUS_MISSING_OUTPUT,
    STATUS_MULTIPLE_OUTPUTS,
    STATUS_INVALID_OUTPUT,
    STATUS_MISSING_SOURCE,
    STATUS_MISSING_MODEL,
)

# CSV index schema (spec, fixed order).
INDEX_FIELDS: list[str] = [
    "sample_id",
    "session_id",
    "action_id",
    "source_video_path",
    "anonymized_video_path",
    "status",
    "returncode",
    "frame_count",
    "fps",
    "width",
    "height",
    "command_log_path",
    "notes",
]

# Canonical slice_index.csv schema (used to validate / preserve rows).
SLICE_INDEX_FIELDS: list[str] = [
    "sample_id",
    "session_id",
    "person_id",
    "environment_id",
    "action_id",
    "trial_id",
    "sync_quality_flags",
    "csi_index_method",
    "output_group",
    "csi_in_path",
    "csi_out_path",
    "csi_packet_start_idx",
    "csi_packet_end_idx",
    "csi_packets_expected",
    "csi_packets_written",
    "video_in_path",
    "video_out_path",
    "frame_start_idx_raw",
    "frame_end_idx_raw",
    "frame_start_idx_padded",
    "frame_end_idx_padded",
    "video_frames_expected",
    "video_frames_written",
    "video_fps",
    "status",
    "notes",
]

# Source prefixes that the clean and reviewed derivative slice indexes use.
SOURCE_PREFIXES: tuple[str, ...] = (
    "sliced_windows_by_session/",
    "sliced_windows_holdout/",
    "sliced_windows_resample_20260705_s00/",
    "sliced_windows_resample_",
)


# ---------------------------------------------------------------------------
# Errors
# ---------------------------------------------------------------------------


class PathSafetyError(ValueError):
    """Raised when an output path would violate the raw-data safety rules."""


class DeepMosaicsOutputError(Exception):
    """Raised when DeepMosaics output discovery fails.

    ``status`` is one of ``missing_output`` / ``multiple_outputs`` so the
    caller can surface it directly in the run index.
    """

    def __init__(self, status: str, message: str, candidates: list[Path] | None = None) -> None:
        super().__init__(message)
        self.status = status
        self.message = message
        self.candidates = candidates or []


# ---------------------------------------------------------------------------
# Path safety
# ---------------------------------------------------------------------------


def relative_to_root(path: Path | str, project_root: Path = PROJECT_ROOT) -> str:
    """Return the forward-slash relative path of *path* under *project_root*.

    Absolute paths are accepted only if they resolve under *project_root*.
    Relative paths containing ``..`` are rejected.
    """
    root = project_root.resolve()
    p = Path(path)
    if p.is_absolute():
        resolved = p.resolve()
        try:
            rel = resolved.relative_to(root)
        except ValueError as exc:
            raise PathSafetyError(
                f"absolute path outside project root: {p} "
                f"(resolved {resolved}, root {root})"
            ) from exc
    else:
        path_str = str(p)
        if path_str.startswith("/"):
            raise PathSafetyError(f"absolute path strings are not allowed: {path_str}")
        if ".." in p.parts:
            raise PathSafetyError(f"'..' is not allowed in path: {path_str}")
        resolved = (root / p).resolve()
        try:
            rel = resolved.relative_to(root)
        except ValueError as exc:
            raise PathSafetyError(
                f"path resolves outside project root: {p} "
                f"(resolved {resolved}, root {root})"
            ) from exc
    rel_str = str(rel).replace("\\", "/")
    if ".." in rel_str.split("/"):
        raise PathSafetyError(f"'..' in resolved path: {p} -> {rel_str}")
    return rel_str


def validate_output_path(path: Path | str, project_root: Path = PROJECT_ROOT) -> str:
    """Validate that *path* is a safe place to write an output artifact.

    Returns the relative path string.  Raises :class:`PathSafetyError` if the
    path lands in a forbidden (raw) zone or outside the allowed output prefix.
    """
    rel_str = relative_to_root(path, project_root)

    for prefix in FORBIDDEN_WRITE_PREFIXES:
        if rel_str == prefix.rstrip("/") or rel_str.startswith(prefix):
            raise PathSafetyError(
                f"writing to forbidden zone '{prefix}' is not allowed: {path} "
                f"(resolved to {rel_str})"
            )

    for prefix in ALLOWED_OUTPUT_PREFIXES:
        if rel_str == prefix.rstrip("/") or rel_str.startswith(prefix):
            return rel_str

    raise PathSafetyError(
        f"output path not in any allowed directory: {path} "
        f"(resolved to {rel_str}; allowed prefixes: "
        f"{', '.join(ALLOWED_OUTPUT_PREFIXES)})"
    )


# ---------------------------------------------------------------------------
# CSV I/O
# ---------------------------------------------------------------------------


def read_csv(path: Path) -> list[dict[str, str]]:
    """Read a CSV with header into a list of dict rows (utf-8-sig tolerant)."""
    with path.open("r", encoding="utf-8-sig", newline="") as f:
        return list(csv.DictReader(f))


def write_csv(path: Path, rows: list[dict[str, Any]], fields: list[str]) -> None:
    """Write *rows* to *path* using *fields* as the column order."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            writer.writerow({field: row.get(field, "") for field in fields})


# ---------------------------------------------------------------------------
# Dataclasses
# ---------------------------------------------------------------------------


@dataclass
class DeepMosaicsConfig:
    """Immutable DeepMosaics invocation parameters."""

    deepmosaics_root: Path
    model_path: Path
    gpu_id: str = "0"
    mosaic_size: int = 0
    mask_extend: int = 10
    mosaic_mod: str = "squa_avg"
    temp_image_type: str = "jpg"

    def __post_init__(self) -> None:
        # Normalize to Path for convenient joining downstream.
        self.deepmosaics_root = Path(self.deepmosaics_root)
        self.model_path = Path(self.model_path)


@dataclass
class RunResult:
    """Per-sample result row, serialized into the run index CSV."""

    sample_id: str
    session_id: str
    action_id: str
    source_video_path: str
    anonymized_video_path: str
    status: str
    returncode: str | None = None
    frame_count: str | None = None
    fps: str | None = None
    width: str | None = None
    height: str | None = None
    command_log_path: str = ""
    notes: str = ""

    def to_row(self) -> dict[str, str]:
        return {field: str(getattr(self, field, "") or "") for field in INDEX_FIELDS}


# A subprocess runner callable.  Returns a subprocess.CompletedProcess-like
# object (attributes: returncode, stdout, stderr).
ProgressCallback = Callable[[str], None]
SubprocessRunner = Callable[..., subprocess.CompletedProcess]


def default_runner(
    cmd: list[str],
    cwd: Path,
    timeout: int,
    progress_callback: ProgressCallback | None = None,
) -> subprocess.CompletedProcess:
    """Run *cmd* with stdin=DEVNULL and captured stdout/stderr."""
    env = os.environ.copy()
    env["PYTHONUNBUFFERED"] = "1"
    if progress_callback is not None:
        proc = subprocess.Popen(
            cmd,
            cwd=str(cwd),
            stdin=subprocess.DEVNULL,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            env=env,
            bufsize=0,
        )
        chunks: list[bytes] = []
        started = time.monotonic()
        assert proc.stdout is not None
        fd = proc.stdout.fileno()
        while True:
            if timeout and (time.monotonic() - started) > timeout:
                proc.kill()
                proc.wait()
                output = b"".join(chunks)
                raise subprocess.TimeoutExpired(cmd, timeout, output=output, stderr=b"")
            ready, _, _ = select.select([fd], [], [], 0.1)
            if ready:
                chunk = os.read(fd, 4096)
                if chunk:
                    chunks.append(chunk)
                    progress_callback(chunk.decode("utf-8", errors="replace"))
                    continue
            if proc.poll() is not None:
                rest = proc.stdout.read()
                if rest:
                    chunks.append(rest)
                    progress_callback(rest.decode("utf-8", errors="replace"))
                break
        return subprocess.CompletedProcess(cmd, proc.returncode, b"".join(chunks), b"")
    return subprocess.run(
        cmd,
        cwd=str(cwd),
        stdin=subprocess.DEVNULL,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        env=env,
        timeout=timeout,
    )


# ---------------------------------------------------------------------------
# Index loading / selection
# ---------------------------------------------------------------------------


def load_slice_index(index_path: Path) -> list[dict[str, str]]:
    """Load the slice_index.csv referenced by *index_path*.

    Rows missing ``sample_id`` or ``video_out_path`` are dropped with a warning
    so a single malformed line never breaks the batch.
    """
    if not index_path.exists():
        raise FileNotFoundError(f"slice index not found: {index_path}")
    rows = read_csv(index_path)
    cleaned: list[dict[str, str]] = []
    for row in rows:
        if not row.get("sample_id") or not row.get("video_out_path"):
            print(
                f"[warn] dropping index row without sample_id/video_out_path: {row}",
                file=sys.stderr,
            )
            continue
        cleaned.append(row)
    return cleaned


def select_records(
    rows: list[dict[str, str]],
    sample_ids: list[str] | None = None,
    limit: int | None = None,
) -> list[dict[str, str]]:
    """Filter *rows* by ``--sample-id`` list then take the first ``limit``."""
    selected = list(rows)
    if sample_ids:
        id_set = set(sample_ids)
        selected = [r for r in selected if r.get("sample_id") in id_set]
    if limit is not None and limit > 0:
        selected = selected[:limit]
    return selected


# ---------------------------------------------------------------------------
# Output path construction
# ---------------------------------------------------------------------------


def build_output_path(
    record: dict[str, str],
    output_base: Path,
    project_root: Path = PROJECT_ROOT,
) -> Path:
    """Build the final anonymized-video path for *record*.

    The source ``video_out_path`` (e.g.
    ``sliced_windows_by_session/S01/video/walk/foo.mp4``) is mapped to
    ``<output_base>/outputs/S01/video/walk/foo_add.mp4`` and validated against
    the path-safety rules.
    """
    video_out_path = (record.get("video_out_path") or "").replace("\\", "/").lstrip("./")
    if not video_out_path:
        raise PathSafetyError("record has empty video_out_path")

    stripped = None
    for prefix in SOURCE_PREFIXES:
        if video_out_path.startswith(prefix):
            stripped = video_out_path[len(prefix):]
            break
    if stripped is None:
        # Not under a known source root — still map it, but the safety check
        # below will reject it if it lands in a forbidden zone.
        stripped = video_out_path

    src = Path(stripped)
    rel_under_outputs = str(src.parent / (src.stem + "_add.mp4")).replace("\\", "/")
    if rel_under_outputs in (".", ""):
        rel_under_outputs = src.stem + "_add.mp4"

    output_video = (Path(output_base) / "outputs" / rel_under_outputs).resolve()
    rel_output = relative_to_root(output_video, project_root)
    for prefix in FORBIDDEN_WRITE_PREFIXES:
        if rel_output == prefix.rstrip("/") or rel_output.startswith(prefix):
            raise PathSafetyError(
                f"writing to forbidden zone {prefix!r} is not allowed: {output_video} "
                f"(resolved to {rel_output})"
            )
    return output_video


def validate_source_video_path(video_out_path: str, project_root: Path = PROJECT_ROOT) -> Path:
    path_text = (video_out_path or "").replace("\\", "/").strip()
    if not path_text:
        raise PathSafetyError("record has empty video_out_path")

    p = Path(path_text)
    if ".." in p.parts:
        raise PathSafetyError(f"'..' is not allowed in source path: {path_text}")

    rel_str = relative_to_root(p, project_root)
    if ".." in Path(rel_str).parts:
        raise PathSafetyError(f"'..' is not allowed in resolved source path: {path_text}")

    for prefix in SOURCE_PREFIXES:
        if rel_str == prefix.rstrip("/") or rel_str.startswith(prefix):
            return (project_root.resolve() / rel_str).resolve()

    raise PathSafetyError(
        f"source video path not in allowed sliced-window directories: {path_text} "
        f"(resolved to {rel_str}; allowed prefixes: {', '.join(SOURCE_PREFIXES)})"
    )


# ---------------------------------------------------------------------------
# DeepMosaics command construction
# ---------------------------------------------------------------------------


def build_deepmosaics_command(
    source_video: Path,
    model_path: Path,
    result_dir: Path,
    temp_dir: Path,
    config: DeepMosaicsConfig,
) -> list[str]:
    """Build the argv list for ``sys.executable deepmosaic.py``.

    The path arguments are absolute so DeepMosaics writes into the project
    staging tree rather than ``/data/DeepMosaics-master/result``.
    """
    return [
        sys.executable,
        "deepmosaic.py",
        "--media_path", str(source_video.resolve()),
        "--model_path", str(Path(model_path).resolve()),
        "--result_dir", str(Path(result_dir).resolve()),
        "--temp_dir", str(Path(temp_dir).resolve()),
        "--mode", "add",
        "--gpu_id", str(config.gpu_id),
        "--no_preview",
        "--mosaic_mod", str(config.mosaic_mod),
        "--mosaic_size", str(config.mosaic_size),
        "--mask_extend", str(config.mask_extend),
        "--tempimage_type", str(config.temp_image_type),
    ]


# ---------------------------------------------------------------------------
# DeepMosaics output discovery
# ---------------------------------------------------------------------------


def discover_deepmosaics_output(
    result_dir: Path,
    source_stem: str | None = None,
) -> Path:
    """Find the single ``<stem>_add.mp4`` produced by DeepMosaics.

    * If ``source_stem`` is provided, return exactly ``<source_stem>_add.mp4``.
    * If ``source_stem`` is omitted and exactly one ``*_add.mp4`` exists, return it.
    * If none exists, raise ``DeepMosaicsOutputError(missing_output)``.
    * If fallback discovery sees more than one output, raise
      ``DeepMosaicsOutputError(multiple_outputs)``.
    """
    result_dir = Path(result_dir)
    if not result_dir.exists():
        raise DeepMosaicsOutputError(
            STATUS_MISSING_OUTPUT,
            f"result_dir does not exist: {result_dir}",
        )
    matches = sorted(result_dir.glob("*_add.mp4"))
    if source_stem:
        expected = result_dir / f"{source_stem}_add.mp4"
        if expected.exists():
            return expected
        if matches:
            names = ", ".join(p.name for p in matches)
            raise DeepMosaicsOutputError(
                STATUS_MISSING_OUTPUT,
                f"expected DeepMosaics output not found: {expected.name}; "
                f"candidates in {result_dir}: {names}",
                candidates=matches,
            )
        raise DeepMosaicsOutputError(
            STATUS_MISSING_OUTPUT,
            f"no *_add.mp4 output found in {result_dir} (expected {source_stem}_add.mp4)",
        )

    if not matches:
        raise DeepMosaicsOutputError(
            STATUS_MISSING_OUTPUT,
            f"no *_add.mp4 output found in {result_dir}",
        )
    if len(matches) > 1:
        names = ", ".join(p.name for p in matches)
        raise DeepMosaicsOutputError(
            STATUS_MULTIPLE_OUTPUTS,
            f"multiple *_add.mp4 outputs found in {result_dir}: {names}",
            candidates=matches,
        )
    return matches[0]


# ---------------------------------------------------------------------------
# Video probing / thumbnails
# ---------------------------------------------------------------------------


def probe_video(path: Path) -> dict[str, Any]:
    """Probe a video for frame_count / fps / width / height via cv2.

    Returns a dict with ``None`` values when the file cannot be opened.
    """
    try:
        import cv2
    except ImportError:  # pragma: no cover - cv2 is available in the project env
        return {"frame_count": None, "fps": None, "width": None, "height": None}

    cap = cv2.VideoCapture(str(path))
    if not cap.isOpened():
        return {"frame_count": None, "fps": None, "width": None, "height": None}
    try:
        frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        fps = float(cap.get(cv2.CAP_PROP_FPS))
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    finally:
        cap.release()
    return {
        "frame_count": frame_count,
        "fps": fps,
        "width": width,
        "height": height,
    }


def validate_probe_result(probe: dict[str, Any]) -> tuple[bool, str]:
    frame_count = probe.get("frame_count")
    fps = probe.get("fps")
    width = probe.get("width")
    height = probe.get("height")

    def _positive(value: Any) -> bool:
        try:
            return value is not None and float(value) > 0
        except (TypeError, ValueError):
            return False

    ok = all(_positive(v) for v in (frame_count, fps, width, height))
    if ok:
        return True, ""
    return (
        False,
        "invalid output metadata: "
        f"frame_count={frame_count} fps={fps} width={width} height={height}",
    )


def make_thumbnails(
    source_video: Path,
    anonymized_video: Path,
    preview_dir: Path,
    project_root: Path,
    positions: tuple[str, ...] = ("start", "middle", "end"),
) -> list[str]:
    """Write start/middle/end PNG thumbnails for source + anonymized videos.

    Returns the list of relative (to *project_root*) thumbnail paths actually
    written.  Silently skips videos that cannot be opened.
    """
    try:
        import cv2
    except ImportError:  # pragma: no cover
        return []

    preview_dir.mkdir(parents=True, exist_ok=True)
    written: list[str] = []

    for label, video in (("source", source_video), ("anonymized", anonymized_video)):
        cap = cv2.VideoCapture(str(video))
        if not cap.isOpened():
            continue
        try:
            n = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            for pos in positions:
                if pos == "start":
                    idx = 0
                elif pos == "middle":
                    idx = max(0, n // 2)
                else:
                    idx = max(0, n - 1)
                cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
                ok, frame = cap.read()
                if not ok or frame is None:
                    continue
                out = preview_dir / f"{label}_{pos}.png"
                if cv2.imwrite(str(out), frame):
                    try:
                        written.append(str(out.resolve().relative_to(project_root.resolve())).replace("\\", "/"))
                    except ValueError:
                        written.append(str(out).replace("\\", "/"))
        finally:
            cap.release()
    return written


# ---------------------------------------------------------------------------
# Per-video runner
# ---------------------------------------------------------------------------


def _resolve_source(record: dict[str, str], project_root: Path) -> Path:
    return validate_source_video_path(record.get("video_out_path", ""), project_root)


def _log_command_header(
    log_path: Path,
    cmd: list[str],
    cwd: Path,
    timeout: int,
) -> None:
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with log_path.open("w", encoding="utf-8") as f:
        f.write("command: " + " ".join(cmd) + "\n")
        f.write(f"cwd: {cwd}\n")
        f.write(f"timeout: {timeout}s\n")


def _log_command_result(
    log_path: Path,
    returncode: int,
    elapsed: float,
    stdout: bytes | str | None,
    stderr: bytes | str | None,
) -> None:
    def _to_text(buf: bytes | str | None) -> str:
        if buf is None:
            return ""
        if isinstance(buf, bytes):
            return buf.decode("utf-8", errors="replace")
        return str(buf)

    with log_path.open("a", encoding="utf-8") as f:
        f.write(f"returncode: {returncode}\n")
        f.write(f"elapsed: {elapsed:.2f}s\n")
        f.write("--- stdout ---\n")
        f.write(_to_text(stdout))
        f.write("--- stderr ---\n")
        f.write(_to_text(stderr))


def _call_runner(
    runner: SubprocessRunner,
    cmd: list[str],
    cwd: Path,
    timeout: int,
    progress_callback: ProgressCallback | None,
) -> subprocess.CompletedProcess:
    if progress_callback is None:
        return runner(cmd, cwd, timeout)
    try:
        sig = inspect.signature(runner)
    except (TypeError, ValueError):
        return runner(cmd, cwd, timeout)
    params = sig.parameters.values()
    if any(p.kind == inspect.Parameter.VAR_KEYWORD for p in params) or "progress_callback" in sig.parameters:
        return runner(cmd, cwd, timeout, progress_callback=progress_callback)
    if any(p.kind == inspect.Parameter.VAR_POSITIONAL for p in sig.parameters.values()):
        return runner(cmd, cwd, timeout, progress_callback)
    return runner(cmd, cwd, timeout)


def run_one_video(
    record: dict[str, str],
    config: DeepMosaicsConfig,
    output_base: Path,
    project_root: Path,
    *,
    mode: str = "dry-run",
    overwrite: bool = False,
    keep_staging: bool = False,
    make_thumbs: bool = False,
    runner: SubprocessRunner = default_runner,
    progress_callback: ProgressCallback | None = None,
    timeout: int = DEFAULT_TIMEOUT_SECONDS,
) -> RunResult:
    """Run the full anonymization pipeline for a single record.

    *mode* is one of ``dry-run`` / ``preview`` / ``apply``.  ``runner`` is
    injectable so tests can fake the subprocess call.
    """
    sample_id = record.get("sample_id", "")
    session_id = record.get("session_id", "")
    action_id = record.get("action_id", "")
    source_rel = record.get("video_out_path", "")

    log_path = output_base / "logs" / f"{sample_id}.log"
    try:
        log_rel = str(log_path.resolve().relative_to(project_root.resolve())).replace("\\", "/")
    except ValueError:
        log_rel = str(log_path).replace("\\", "/")

    def _result(status: str, *, returncode: str | None = None, anon: str = "",
                frame_count=None, fps=None, width=None, height=None, notes: str = "") -> RunResult:
        return RunResult(
            sample_id=sample_id,
            session_id=session_id,
            action_id=action_id,
            source_video_path=source_rel,
            anonymized_video_path=anon,
            status=status,
            returncode=returncode,
            frame_count=frame_count,
            fps=fps,
            width=width,
            height=height,
            command_log_path=log_rel,
            notes=notes,
        )

    # ---- pre-checks ------------------------------------------------------
    try:
        source_video = _resolve_source(record, project_root)
    except PathSafetyError as exc:
        return _result(STATUS_INVALID_OUTPUT, notes=f"unsafe source path: {exc}")

    if not source_video.exists():
        return _result(STATUS_MISSING_SOURCE, notes=f"source not found: {source_video}")
    if not config.model_path.exists():
        return _result(STATUS_MISSING_MODEL, notes=f"model not found: {config.model_path}")

    # ---- output path (safety-validated) ---------------------------------
    try:
        output_video = build_output_path(record, output_base, project_root)
        output_rel = str(output_video.resolve().relative_to(project_root.resolve())).replace("\\", "/")
    except PathSafetyError as exc:
        return _result(STATUS_INVALID_OUTPUT, notes=f"unsafe output path: {exc}")

    # ---- skip if already done -------------------------------------------
    if output_video.exists() and not overwrite:
        probe = probe_video(output_video) if mode != "dry-run" else {"frame_count": None, "fps": None, "width": None, "height": None}
        if mode != "dry-run":
            probe_ok, probe_note = validate_probe_result(probe)
            if not probe_ok:
                return _result(
                    STATUS_INVALID_OUTPUT,
                    anon=output_rel,
                    frame_count=probe["frame_count"],
                    fps=probe["fps"],
                    width=probe["width"],
                    height=probe["height"],
                    notes=f"existing output invalid; pass --overwrite to regenerate; {probe_note}",
                )
        return _result(
            STATUS_SKIPPED_EXISTS,
            anon=output_rel,
            frame_count=probe["frame_count"],
            fps=probe["fps"],
            width=probe["width"],
            height=probe["height"],
            notes="output exists; pass --overwrite to regenerate",
        )

    # ---- dry-run: plan only ---------------------------------------------
    if mode == "dry-run":
        return _result(STATUS_OK, returncode="0", anon=output_rel, notes="dry_run_planned")

    # ---- preview / apply: actually run DeepMosaics ----------------------
    staging_root = output_base / "staging" / sample_id
    result_dir = staging_root / "result"
    temp_dir = staging_root / "tmp"
    # Clean any previous staging so discovery sees a clean slate.
    if staging_root.exists():
        shutil.rmtree(staging_root)
    result_dir.mkdir(parents=True, exist_ok=True)
    temp_dir.mkdir(parents=True, exist_ok=True)

    cmd = build_deepmosaics_command(source_video, config.model_path, result_dir, temp_dir, config)
    _log_command_header(log_path, cmd, config.deepmosaics_root, timeout)

    t0 = time.time()
    try:
        cp = _call_runner(runner, cmd, config.deepmosaics_root, timeout, progress_callback)
        elapsed = time.time() - t0
    except subprocess.TimeoutExpired:
        elapsed = time.time() - t0
        _log_command_result(log_path, -1, elapsed, None, "TIMEOUT")
        return _result(STATUS_FAILED, returncode="-1",
                       notes=f"timeout after {timeout}s; staging retained")
    except Exception as exc:  # pragma: no cover - defensive
        elapsed = time.time() - t0
        _log_command_result(log_path, -1, elapsed, None, f"runner exception: {exc}")
        return _result(STATUS_FAILED, returncode="-1", notes=f"runner error: {exc}; staging retained")

    _log_command_result(log_path, cp.returncode, elapsed, cp.stdout, cp.stderr)

    if cp.returncode != 0:
        return _result(
            STATUS_FAILED,
            returncode=str(cp.returncode),
            notes=f"deepmosaics exit {cp.returncode}; see log; staging retained",
        )

    # ---- discover + validate output -------------------------------------
    try:
        anon_file = discover_deepmosaics_output(result_dir, source_video.stem)
    except DeepMosaicsOutputError as exc:
        return _result(exc.status, returncode=str(cp.returncode),
                       notes=f"{exc.message}; staging retained")

    try:
        rel_checked_output = relative_to_root(output_video, project_root)
        for prefix in FORBIDDEN_WRITE_PREFIXES:
            if rel_checked_output == prefix.rstrip("/") or rel_checked_output.startswith(prefix):
                raise PathSafetyError(
                    f"writing to forbidden zone {prefix!r} is not allowed: {output_video} "
                    f"(resolved to {rel_checked_output})"
                )
    except PathSafetyError as exc:
        return _result(STATUS_INVALID_OUTPUT, returncode=str(cp.returncode),
                       notes=f"unsafe output: {exc}; staging retained")

    # ---- move to final location -----------------------------------------
    output_video.parent.mkdir(parents=True, exist_ok=True)
    if output_video.exists():
        if overwrite:
            output_video.unlink()
        else:
            # Race / stale staging; treat as skip.
            shutil.rmtree(staging_root, ignore_errors=True)
            probe = probe_video(output_video)
            return _result(
                STATUS_SKIPPED_EXISTS,
                anon=output_rel,
                frame_count=probe["frame_count"],
                fps=probe["fps"],
                width=probe["width"],
                height=probe["height"],
                notes="output appeared during run; skipped",
            )
    shutil.move(str(anon_file), str(output_video))

    # ---- probe + thumbnails ---------------------------------------------
    probe = probe_video(output_video)
    probe_ok, probe_note = validate_probe_result(probe)
    thumb_notes = ""
    if make_thumbs and probe_ok:
        thumb_dir = output_base / "previews" / sample_id
        thumbs = make_thumbnails(source_video, output_video, thumb_dir, project_root)
        if thumbs:
            thumb_notes = "; thumbs=" + ",".join(thumbs)

    # ---- staging cleanup ------------------------------------------------
    if not keep_staging:
        shutil.rmtree(staging_root, ignore_errors=True)

    notes = f"elapsed={elapsed:.2f}s" + thumb_notes
    if not probe_ok:
        return _result(
            STATUS_INVALID_OUTPUT,
            returncode="0",
            anon=output_rel,
            frame_count=probe["frame_count"],
            fps=probe["fps"],
            width=probe["width"],
            height=probe["height"],
            notes=f"{probe_note}; elapsed={elapsed:.2f}s",
        )

    return _result(
        STATUS_OK,
        returncode="0",
        anon=output_rel,
        frame_count=probe["frame_count"],
        fps=probe["fps"],
        width=probe["width"],
        height=probe["height"],
        notes=notes,
    )


# ---------------------------------------------------------------------------
# Index + report writers
# ---------------------------------------------------------------------------


def write_index_csv(path: Path, results: list[RunResult]) -> None:
    """Write the run-index CSV (one row per processed sample)."""
    rows = [r.to_row() for r in results]
    write_csv(path, rows, INDEX_FIELDS)


def write_report(
    path: Path,
    results: list[RunResult],
    mode: str,
    config: DeepMosaicsConfig,
    index_path: Path,
    output_base: Path,
    project_root: Path,
    selected_count: int,
    total_count: int,
    run_id: str | None = None,
    run_report_dir: Path | None = None,
) -> None:
    """Write a Markdown run report summarising *results*."""
    path.parent.mkdir(parents=True, exist_ok=True)

    def _rel(p: Path) -> str:
        try:
            return str(p.resolve().relative_to(project_root.resolve())).replace("\\", "/")
        except ValueError:
            return str(p)

    counts: dict[str, int] = {s: 0 for s in ALL_STATUSES}
    for r in results:
        counts[r.status] = counts.get(r.status, 0) + 1

    failures = [r for r in results if r.status not in (STATUS_OK, STATUS_SKIPPED_EXISTS)]

    lines: list[str] = []
    lines.append("# DeepMosaics Face Anonymization Run Report")
    lines.append("")
    lines.append(f"- Generated at: {time.strftime('%Y-%m-%d %H:%M:%S %z')}")
    lines.append(f"- Run mode: `{mode}`")
    if run_id:
        lines.append(f"- Run ID: `{run_id}`")
    if run_report_dir:
        lines.append(f"- Run report directory: `{_rel(run_report_dir)}`")
    lines.append(f"- Project root: `{project_root}`")
    lines.append(f"- DeepMosaics root: `{config.deepmosaics_root}`")
    lines.append(f"- Model path: `{config.model_path}`")
    lines.append(f"- Input index: `{_rel(index_path)}` (total rows: {total_count}; selected: {selected_count})")
    lines.append(f"- Output base: `{_rel(output_base)}`")
    lines.append(
        f"- DeepMosaics params: gpu_id={config.gpu_id} mosaic_mod={config.mosaic_mod} "
        f"mosaic_size={config.mosaic_size} mask_extend={config.mask_extend} "
        f"temp_image_type={config.temp_image_type}"
    )
    lines.append("")
    lines.append("## Status summary")
    lines.append("")
    lines.append("| Status | Count |")
    lines.append("|---|---|")
    for s in ALL_STATUSES:
        lines.append(f"| {s} | {counts.get(s, 0)} |")
    lines.append(f"| **total** | **{len(results)}** |")
    lines.append("")
    lines.append("## Per-sample details")
    lines.append("")
    lines.append("| sample_id | session_id | action_id | status | returncode | frames | fps | wxh | notes |")
    lines.append("|---|---|---|---|---|---|---|---|---|")
    for r in results:
        wxh = ""
        if r.width and r.height:
            wxh = f"{r.width}x{r.height}"
        lines.append(
            f"| {r.sample_id} | {r.session_id} | {r.action_id} | {r.status} | "
            f"{r.returncode or ''} | {r.frame_count or ''} | {r.fps or ''} | {wxh} | {r.notes} |"
        )
    lines.append("")
    if failures:
        lines.append("## Failures")
        lines.append("")
        for r in failures:
            lines.append(
                f"- `{r.sample_id}` ({r.session_id}/{r.action_id}): **{r.status}** — {r.notes}"
            )
        lines.append("")
    else:
        lines.append("No failures recorded.")
        lines.append("")
    lines.append("## Output paths")
    lines.append("")
    lines.append(f"- Anonymized videos: `{_rel(output_base / 'outputs')}/`")
    lines.append(f"- Run index CSV: `{_rel(path.parent / 'anonymization_index.csv')}`")
    lines.append(f"- Per-sample logs: `{_rel(output_base / 'logs')}/`")
    lines.append(f"- Preview thumbnails: `{_rel(output_base / 'previews')}/`")
    lines.append(f"- Staging (temp): `{_rel(output_base / 'staging')}/`")
    lines.append("")
    path.write_text("\n".join(lines), encoding="utf-8")


def make_run_id(mode: str, reports_root: Path) -> str:
    base = time.strftime("%Y%m%d_%H%M%S") + f"_{mode}"
    runs_root = reports_root / "runs"
    candidate = base
    suffix = 1
    while (runs_root / candidate).exists():
        candidate = f"{base}_{suffix:03d}"
        suffix += 1
    return candidate


def prepare_report_paths(output_base: Path, mode: str) -> dict[str, Path | str]:
    reports_root = output_base / "reports"
    run_id = make_run_id(mode, reports_root)
    run_dir = reports_root / "runs" / run_id
    latest_dir = reports_root / "latest"
    run_dir.mkdir(parents=True, exist_ok=False)
    latest_dir.mkdir(parents=True, exist_ok=True)
    return {
        "run_id": run_id,
        "run_dir": run_dir,
        "run_csv": run_dir / "anonymization_index.csv",
        "run_report": run_dir / "run_report.md",
        "latest_dir": latest_dir,
        "latest_csv": latest_dir / "anonymization_index.csv",
        "latest_report": latest_dir / "run_report.md",
        "compat_csv": reports_root / "anonymization_index.csv",
        "compat_report": reports_root / "run_report.md",
    }


def copy_latest_report(run_csv: Path, run_report: Path, latest_dir: Path, compat_csv: Path, compat_report: Path) -> None:
    latest_dir.mkdir(parents=True, exist_ok=True)
    latest_csv = latest_dir / "anonymization_index.csv"
    latest_report = latest_dir / "run_report.md"
    shutil.copyfile(run_csv, latest_csv)
    shutil.copyfile(run_report, latest_report)
    compat_csv.parent.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(latest_csv, compat_csv)
    shutil.copyfile(latest_report, compat_report)


def format_duration(seconds: float | int | None) -> str:
    if seconds is None:
        return "?"
    total = max(0, int(seconds))
    hours, rem = divmod(total, 3600)
    minutes, secs = divmod(rem, 60)
    if hours:
        return f"{hours}h{minutes:02d}m{secs:02d}s"
    if minutes:
        return f"{minutes}m{secs:02d}s"
    return f"{secs}s"


def format_progress_bar(done: int, total: int, width: int = 30) -> str:
    safe_width = max(1, int(width))
    if total <= 0:
        filled = 0
    else:
        filled = round((max(0, min(done, total)) / total) * safe_width)
    return "[" + ("#" * filled) + ("-" * (safe_width - filled)) + "]"


def parse_deepmosaics_progress(text: str) -> dict[str, int]:
    parsed: dict[str, int] = {}
    step_match = re.search(r"\bStep\s*:?\s*(\d+)\s*/\s*(\d+)\b", text, flags=re.IGNORECASE)
    if step_match:
        parsed["step_current"] = int(step_match.group(1))
        parsed["step_total"] = int(step_match.group(2))
    for match in re.finditer(r"\b(\d+)\s*/\s*(\d+)\b", text):
        before = text[max(0, match.start() - 12):match.start()].lower()
        if "step" in before:
            continue
        current = int(match.group(1))
        total = int(match.group(2))
        if total > 0 and current <= total:
            parsed["frame_current"] = current
            parsed["frame_total"] = total
    return parsed


class ProgressTracker:
    def __init__(self, total: int, mode: str, width: int = 30, stream=None, enabled: bool = True) -> None:
        self.total = max(0, int(total))
        self.mode = mode
        self.width = max(1, int(width))
        self.stream = stream if stream is not None else sys.stderr
        self.enabled = enabled
        self.done = 0
        self.counts = {"ok": 0, "skipped": 0, "failed": 0, "invalid": 0}
        self.started_at = time.monotonic()
        self.current_sample_id = ""
        self.last_status = "pending"
        self.file_started_at: float | None = None
        self.file_step_current: int | None = None
        self.file_step_total: int | None = None
        self.file_frame_current: int | None = None
        self.file_frame_total: int | None = None
        self._last_file_render_key: tuple[int | None, int | None, int | None] | None = None
        self.is_tty = bool(getattr(self.stream, "isatty", lambda: False)())
        self._rendered = False

    def start(self, sample_id: str) -> None:
        if not self.enabled:
            return
        self.current_sample_id = sample_id
        self.last_status = "running"
        self.file_started_at = time.monotonic()
        self.file_step_current = None
        self.file_step_total = None
        self.file_frame_current = None
        self.file_frame_total = None
        self._last_file_render_key = None
        if self.is_tty:
            self._write(self._line(), end="")
        else:
            next_i = min(self.done + 1, self.total) if self.total else 0
            self._write(f"[{self.mode}] start overall={next_i}/{self.total} current={sample_id}")

    def update(self, sample_id: str, status: str) -> None:
        if not self.enabled:
            return
        self.done += 1
        self.current_sample_id = sample_id
        self.last_status = status
        if status == STATUS_OK:
            self.counts["ok"] += 1
        if status == STATUS_SKIPPED_EXISTS:
            self.counts["skipped"] += 1
        if status in FAILED_PROGRESS_STATUSES:
            self.counts["failed"] += 1
        if status == STATUS_INVALID_OUTPUT:
            self.counts["invalid"] += 1
        if self.is_tty:
            self._write(self._line(), end="")
        else:
            self._write(self._line())

    def update_file_progress(
        self,
        sample_id: str,
        step_current: int | None = None,
        step_total: int | None = None,
        frame_current: int | None = None,
        frame_total: int | None = None,
    ) -> None:
        if not self.enabled:
            return
        self.current_sample_id = sample_id
        self.last_status = "running"
        if self.file_started_at is None:
            self.file_started_at = time.monotonic()
        if step_current is not None:
            self.file_step_current = step_current
        if step_total is not None:
            self.file_step_total = step_total
        if frame_current is not None:
            self.file_frame_current = frame_current
        if frame_total is not None:
            self.file_frame_total = frame_total
        file_bucket = None
        if self.file_frame_current is not None and self.file_frame_total:
            file_bucket = int(((self.file_frame_current / self.file_frame_total) * 100.0) // 5)
        render_key = (self.file_step_current, self.file_step_total, file_bucket)
        if not self.is_tty and render_key == self._last_file_render_key:
            return
        self._last_file_render_key = render_key
        if self.is_tty:
            self._write(self._line(), end="")
        else:
            self._write(self._line())

    def finish(self) -> None:
        if self.enabled and self.is_tty and self._rendered:
            self.stream.write("\n")
            self.stream.flush()

    def _line(self) -> str:
        overall_pct = 100.0 if self.total == 0 else (self.done / self.total) * 100.0
        elapsed = time.monotonic() - self.started_at
        eta = None
        if self.done > 0:
            eta = (elapsed / self.done) * max(0, self.total - self.done)
        counts = (
            f"ok={self.counts['ok']} skipped={self.counts['skipped']} "
            f"failed={self.counts['failed']} invalid={self.counts['invalid']}"
        )
        file_bits: list[str] = []
        if self.file_step_current is not None and self.file_step_total is not None:
            file_bits.append(f"step={self.file_step_current}/{self.file_step_total}")
        if self.file_frame_current is not None and self.file_frame_total is not None:
            file_pct = (self.file_frame_current / self.file_frame_total) * 100.0 if self.file_frame_total else 0.0
            file_bits.append(f"frame={self.file_frame_current}/{self.file_frame_total} {file_pct:.1f}%")
        if self.file_started_at is not None:
            file_bits.append(f"file_elapsed={format_duration(time.monotonic() - self.file_started_at)}")
        file_text = (" " + " ".join(file_bits)) if file_bits else ""
        prefix = "\r" if self.is_tty else ""
        return (
            f"{prefix}[{self.mode}] overall={self.done}/{self.total} {overall_pct:5.1f}% "
            f"{format_progress_bar(self.done, self.total, self.width)} "
            f"current={self.current_sample_id} status={self.last_status} "
            f"elapsed={format_duration(elapsed)} eta={format_duration(eta)}{file_text} "
            f"counts {counts}"
        )

    def _write(self, text: str, end: str = "\n") -> None:
        if self.is_tty:
            # Clear the current terminal line before rewriting it; otherwise a
            # shorter update can leave stale suffixes such as "invalid=0alid=0".
            if text.startswith("\r"):
                text = "\r\033[2K" + text[1:]
            else:
                text = "\r\033[2K" + text
        self.stream.write(text + end)
        self.stream.flush()
        self._rendered = True


def should_use_progress(args: argparse.Namespace, mode: str) -> bool:
    return mode in ("preview", "apply") and not bool(getattr(args, "no_progress", False))


# ---------------------------------------------------------------------------
# Pipeline orchestration
# ---------------------------------------------------------------------------


def run_pipeline(
    args: argparse.Namespace,
    project_root: Path,
) -> int:
    """Top-level driver: load index, select records, run, write artifacts."""
    output_base_arg = Path(args.output_base)
    if not output_base_arg.is_absolute():
        output_base = project_root / output_base_arg
    else:
        output_base = output_base_arg
    output_base = output_base.resolve()
    project_root_resolved = project_root.resolve()
    if output_base != project_root_resolved and project_root_resolved not in output_base.parents:
        raise PathSafetyError(f"output_base must be inside project root: {output_base}")
    output_base.mkdir(parents=True, exist_ok=True)

    index_path = Path(args.input_index)
    if not index_path.is_absolute():
        index_path = project_root / index_path
    index_path = index_path.resolve()

    config = DeepMosaicsConfig(
        deepmosaics_root=Path(args.deepmosaics_root),
        model_path=Path(args.model_path),
        gpu_id=str(args.gpu_id),
        mosaic_size=int(args.mosaic_size),
        mask_extend=int(args.mask_extend),
        mosaic_mod=str(args.mosaic_mod),
        temp_image_type=str(args.temp_image_type),
    )

    rows = load_slice_index(index_path)
    total_count = len(rows)

    limit = args.limit
    if args.preview and (limit is None):
        limit = PREVIEW_DEFAULT_LIMIT
    selected = select_records(rows, sample_ids=args.sample_ids, limit=limit)
    selected_count = len(selected)

    if args.dry_run:
        mode = "dry-run"
    elif args.preview:
        mode = "preview"
    else:
        mode = "apply"

    print(f"[{mode}] index: {index_path} (total {total_count}, selected {selected_count})")
    print(f"[{mode}] model: {config.model_path}")
    print(f"[{mode}] output: {output_base}")

    results: list[RunResult] = []
    progress_enabled = should_use_progress(args, mode)
    progress = ProgressTracker(
        total=selected_count,
        mode=mode,
        width=int(getattr(args, "progress_width", 30)),
        stream=sys.stderr,
        enabled=progress_enabled,
    )
    for i, record in enumerate(selected, 1):
        current_sample_id = record.get("sample_id", "")
        progress.start(current_sample_id)

        def _progress_from_output(text: str, sample_id: str = current_sample_id) -> None:
            if getattr(args, "no_file_progress", False):
                return
            parsed = parse_deepmosaics_progress(text)
            if parsed:
                progress.update_file_progress(
                    sample_id,
                    step_current=parsed.get("step_current"),
                    step_total=parsed.get("step_total"),
                    frame_current=parsed.get("frame_current"),
                    frame_total=parsed.get("frame_total"),
                )

        result = run_one_video(
            record,
            config,
            output_base,
            project_root,
            mode=mode,
            overwrite=args.overwrite,
            keep_staging=args.keep_staging,
            make_thumbs=(mode == "preview"),
            runner=default_runner,
            progress_callback=_progress_from_output if progress_enabled and not getattr(args, "no_file_progress", False) else None,
            timeout=int(args.timeout),
        )
        results.append(result)
        if progress_enabled:
            progress.update(result.sample_id, result.status)
        else:
            print(
                f"[{mode}] {i}/{selected_count} {result.sample_id}: {result.status}"
                + (f" ({result.notes})" if result.notes else "")
            )
    progress.finish()

    report_paths = prepare_report_paths(output_base, mode)
    csv_path = report_paths["run_csv"]
    report_path = report_paths["run_report"]
    write_index_csv(csv_path, results)
    write_report(
        report_path,
        results,
        mode,
        config,
        index_path,
        output_base,
        project_root,
        selected_count=selected_count,
        total_count=total_count,
        run_id=str(report_paths["run_id"]),
        run_report_dir=report_paths["run_dir"],
    )
    copy_latest_report(
        report_paths["run_csv"],
        report_paths["run_report"],
        report_paths["latest_dir"],
        report_paths["compat_csv"],
        report_paths["compat_report"],
    )

    # Summary print.
    counts: dict[str, int] = {}
    for r in results:
        counts[r.status] = counts.get(r.status, 0) + 1
    summary = ", ".join(f"{s}={counts.get(s, 0)}" for s in ALL_STATUSES if counts.get(s, 0))
    print(f"[{mode}] summary: {summary or 'no rows'}")
    print(f"[{mode}] index csv: {csv_path}")
    print(f"[{mode}] report: {report_path}")
    print(f"[{mode}] latest report: {report_paths['latest_report']}")

    # Non-zero exit if anything failed (useful for CI / scripts).
    if any(r.status not in (STATUS_OK, STATUS_SKIPPED_EXISTS) for r in results):
        return 1
    return 0


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Wrap /data/DeepMosaics-master/deepmosaic.py to add face mosaics "
            "to the sliced window videos referenced by slice_index.csv."
        ),
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--project-root", default=str(PROJECT_ROOT))
    parser.add_argument("--input-index", default=DEFAULT_INPUT_INDEX)
    parser.add_argument(
        "--output-base",
        default=DEFAULT_OUTPUT_BASE_REL,
        help="Output/report base directory, relative to project root unless absolute.",
    )
    parser.add_argument("--deepmosaics-root", default=str(DEFAULT_DEEPMOSAICS_ROOT))
    parser.add_argument("--model-path", default=str(DEFAULT_MODEL_PATH))
    parser.add_argument("--gpu-id", default="0", help="GPU id (-1 = CPU).")
    parser.add_argument("--mosaic-size", type=int, default=0, help="0 = auto size.")
    parser.add_argument("--mask-extend", type=int, default=10)
    parser.add_argument("--mosaic-mod", default="squa_avg",
                        choices=["squa_avg", "squa_random", "squa_avg_circle_edge", "rect_avg", "random"])
    parser.add_argument("--temp-image-type", default="jpg", choices=["jpg", "png"],
                        help="Temporary frame format passed to DeepMosaics.")
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--sample-id", action="append", dest="sample_ids", default=[],
                        help="Restrict to this sample_id; may be repeated.")
    parser.add_argument("--timeout", type=int, default=DEFAULT_TIMEOUT_SECONDS,
                        help="Per-video subprocess timeout in seconds.")
    parser.add_argument("--overwrite", action="store_true",
                        help="Re-run even when the output already exists.")
    parser.add_argument("--keep-staging", action="store_true",
                        help="Keep staging dirs after a successful run (for debugging).")
    parser.add_argument("--no-progress", action="store_true",
                        help="Disable preview/apply terminal progress output.")
    parser.add_argument("--no-file-progress", action="store_true",
                        help="Disable current-file DeepMosaics progress output.")
    parser.add_argument("--progress-width", type=int, default=30,
                        help="Progress bar width in characters.")
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--dry-run", action="store_true",
                      help="Plan only, write no videos (default).")
    mode.add_argument("--preview", action="store_true",
                      help="Process a small sample and emit thumbnails.")
    mode.add_argument("--apply", action="store_true",
                      help="Process all selected samples (resumable).")
    args = parser.parse_args(argv)
    if not (args.dry_run or args.preview or args.apply):
        args.dry_run = True
    return args


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    project_root = Path(args.project_root).resolve()
    return run_pipeline(args, project_root)


if __name__ == "__main__":
    raise SystemExit(main())
