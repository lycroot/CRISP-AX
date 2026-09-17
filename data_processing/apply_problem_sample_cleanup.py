#!/usr/bin/env python3
"""Apply problem-sample cleanup plan produced by build_problem_sample_cleanup_index.py.

Default mode is **dry-run**: prints the planned moves and index updates without touching
anything. Pass `--apply` to actually move files — but only after a full **preflight** check
passes for every plan.

With `--apply` (only when preflight passes):
  1. Back up `sliced_windows_by_session/slice_index.csv` and every affected per-session
     `slice_index.csv` to `backups/problem_sample_cleanup/<timestamp>/`.
  2. For operation=quarantine: move csi/video/metadata files to
     `sliced_windows_quarantine/problem_samples/<session>/<csi|video|metadata>/<action>/`.
  3. For operation=holdout_no_shower: move csi/video/metadata files to
     `sliced_windows_holdout/no_shower_bend_pick/S28/<csi|video|metadata>/bend_pick/`.
  4. Rewrite `sliced_windows_by_session/slice_index.csv` to drop quarantine + holdout rows.
  5. Rewrite each affected per-session `slice_index.csv` likewise.
  6. Write `apply_cleanup_log.csv` and `apply_cleanup_report.md` under
     `quality_report_deep/problem_sample_cleanup/`.

Safety (defense in depth):
  - Path validation: rejects absolute paths, rejects any path containing a `..` segment,
    and uses `Path.resolve()` to verify the final location is inside the allowed root
    (catches symlink escapes too).
  - Source must resolve under `<root>/sliced_windows_by_session/` and must NOT resolve
    under any raw-protected dir (`CSI-Formal/00_raw/`, `Video/`, `CSI-Formal/01_manifest/`).
  - Destination must resolve under the operation's quarantine/holdout root and must NOT
    resolve under any raw-protected dir.
  - Preflight: before `--apply` moves a single file or rewrites a single index, ALL plans
    are checked. If any plan is safety_blocked / missing_source / destination_exists, the
    apply ABORTS: no backup is created, no file is moved, no index is rewritten, and the
    process returns a non-zero exit code. This prevents a half-cleaned state.
  - Never deletes: source files are `shutil.move`-ed, not removed.
  - Skips samples whose operation is `unmatched_review_issue` (no file move for those).
"""
from __future__ import annotations

import argparse
import csv
import shutil
import sys
from collections import Counter
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from data_processing.build_problem_sample_cleanup_index import (  # noqa: E402
    CLEAN_INDEX_CSV_REL,
    OPERATIONS_CSV_REL,
    OUTPUT_DIR_REL,
    OPERATION_HOLDOUT_NO_SHOWER,
    OPERATION_QUARANTINE,
    OPERATION_UNMATCHED,
    QUARANTINE_ROOT_REL,
    HOLDOUT_ROOT_REL,
    SLICE_INDEX_REL,
    read_csv,
    write_csv,
)

DEFAULT_PROJECT_ROOT = ROOT

APPLY_LOG_CSV_REL = OUTPUT_DIR_REL / "apply_cleanup_log.csv"
APPLY_REPORT_MD_REL = OUTPUT_DIR_REL / "apply_cleanup_report.md"
BACKUPS_REL = Path("backups/problem_sample_cleanup")

RAW_PROTECTED_DIRS = (
    "CSI-Formal/00_raw",
    "Video",
    "CSI-Formal/01_manifest",
)
SOURCE_ALLOWED_DIR = "sliced_windows_by_session"
RAW_PROTECTED_PREFIXES = tuple(f"{d}/" for d in RAW_PROTECTED_DIRS)

KIND_TO_FIELD = {
    "csi": ("csi_out_path", "destination_csi_path"),
    "video": ("video_out_path", "destination_video_path"),
    "metadata": ("metadata_path", "destination_metadata_path"),
}

PLAN_WOULD_MOVE = "would_move"
PLAN_SAFETY_BLOCKED = "safety_blocked"
PLAN_MISSING_SOURCE = "missing_source"
PLAN_DESTINATION_EXISTS = "destination_exists"


# ---------------------------------------------------------------------------
# Strict path safety helpers
# ---------------------------------------------------------------------------


def path_has_parent_ref(rel_path: str) -> bool:
    """True if the path contains a ``..`` segment (path-traversal attempt)."""
    if not rel_path:
        return False
    return ".." in Path(rel_path).parts


def is_relative_to(path: Path, parent: Path) -> bool:
    """True if ``path`` resolves to a location inside ``parent`` (or equal to it)."""
    try:
        path_r = path.resolve()
        parent_r = parent.resolve()
    except (OSError, RuntimeError):
        return False
    try:
        path_r.relative_to(parent_r)
        return True
    except ValueError:
        return False


def resolve_under_root(root: Path, rel_path: str) -> Path:
    """Resolve a relative path under ``root``; reject absolute paths and ``..`` traversal.

    Returns the resolved absolute Path. The caller must still verify the resolved path is
    inside the intended allowed root (symlink escapes are caught by that follow-up check).
    """
    if not rel_path:
        raise ValueError("empty path")
    p = Path(rel_path)
    if p.is_absolute():
        raise ValueError(f"absolute paths are not allowed: {rel_path}")
    if path_has_parent_ref(rel_path):
        raise ValueError(f"path traversal ('..') is not allowed: {rel_path}")
    return (root / p).resolve()


def is_raw_path(rel_path: str) -> bool:
    """Quick string prefix check used in reports; real safety uses resolve-based checks."""
    if not rel_path:
        return False
    normalized = rel_path.replace("\\", "/")
    return any(normalized == d or normalized.startswith(p) for d, p in
               zip(RAW_PROTECTED_DIRS, RAW_PROTECTED_PREFIXES))


def _raw_resolved_roots(root: Path) -> list[Path]:
    return [(root / d).resolve() for d in RAW_PROTECTED_DIRS]


def assert_safe_source(root: Path, rel_path: str) -> None:
    """Source must resolve under ``sliced_windows_by_session/`` and not under any raw dir."""
    resolved = resolve_under_root(root, rel_path)
    allowed = (root / SOURCE_ALLOWED_DIR).resolve()
    if not is_relative_to(resolved, allowed):
        raise PermissionError(
            f"REFUSING to move source outside {SOURCE_ALLOWED_DIR}/: {rel_path} "
            f"(resolved: {resolved})"
        )
    for raw_r in _raw_resolved_roots(root):
        if is_relative_to(resolved, raw_r):
            raise PermissionError(
                f"REFUSING to move raw file: {rel_path} (protected raw directory)"
            )


def assert_safe_destination(root: Path, rel_path: str, operation: str) -> None:
    """Destination must resolve under the operation's root and not under any raw dir."""
    resolved = resolve_under_root(root, rel_path)
    if operation == OPERATION_QUARANTINE:
        allowed_root = (root / QUARANTINE_ROOT_REL).resolve()
    elif operation == OPERATION_HOLDOUT_NO_SHOWER:
        allowed_root = (root / HOLDOUT_ROOT_REL).resolve()
    else:
        raise ValueError(f"no destination defined for operation={operation}")
    if not is_relative_to(resolved, allowed_root):
        raise PermissionError(
            f"REFUSING to write destination outside allowed root for {operation}: "
            f"{rel_path} (resolved: {resolved})"
        )
    for raw_r in _raw_resolved_roots(root):
        if is_relative_to(resolved, raw_r):
            raise PermissionError(
                f"REFUSING to write into raw dir: {rel_path} (protected raw directory)"
            )


# ---------------------------------------------------------------------------
# Plan construction
# ---------------------------------------------------------------------------


@dataclass
class MovePlan:
    sample_id: str
    operation: str
    kind: str
    source_rel: str
    destination_rel: str
    source_abs: Path
    destination_abs: Path
    source_exists: bool
    destination_exists: bool
    status: str
    note: str


def _make_plan(
    sample_id: str,
    operation: str,
    kind: str,
    src_rel: str,
    dst_rel: str,
    *,
    status: str,
    note: str,
    source_abs: Path = Path(""),
    destination_abs: Path = Path(""),
    source_exists: bool = False,
    destination_exists: bool = False,
) -> MovePlan:
    return MovePlan(
        sample_id=sample_id,
        operation=operation,
        kind=kind,
        source_rel=src_rel,
        destination_rel=dst_rel,
        source_abs=source_abs,
        destination_abs=destination_abs,
        source_exists=source_exists,
        destination_exists=destination_exists,
        status=status,
        note=note,
    )


def classify_plan(
    root: Path,
    sample_id: str,
    operation: str,
    kind: str,
    src_rel: str,
    dst_rel: str,
) -> MovePlan:
    if not src_rel or not dst_rel:
        return _make_plan(
            sample_id, operation, kind, src_rel, dst_rel,
            status=PLAN_MISSING_SOURCE,
            note="empty source or destination path in operations csv",
        )
    try:
        assert_safe_source(root, src_rel)
        assert_safe_destination(root, dst_rel, operation)
    except (PermissionError, ValueError) as exc:
        return _make_plan(
            sample_id, operation, kind, src_rel, dst_rel,
            status=PLAN_SAFETY_BLOCKED,
            note=f"SAFETY_BLOCKED: {exc}",
        )
    src_abs = root / src_rel
    dst_abs = root / dst_rel
    src_exists = src_abs.exists()
    dst_exists = dst_abs.exists()
    if not src_exists:
        return _make_plan(
            sample_id, operation, kind, src_rel, dst_rel,
            status=PLAN_MISSING_SOURCE,
            note="source file does not exist",
            source_abs=src_abs, destination_abs=dst_abs,
            source_exists=False, destination_exists=dst_exists,
        )
    if dst_exists:
        return _make_plan(
            sample_id, operation, kind, src_rel, dst_rel,
            status=PLAN_DESTINATION_EXISTS,
            note="destination file already exists",
            source_abs=src_abs, destination_abs=dst_abs,
            source_exists=True, destination_exists=True,
        )
    return _make_plan(
        sample_id, operation, kind, src_rel, dst_rel,
        status=PLAN_WOULD_MOVE,
        note="",
        source_abs=src_abs, destination_abs=dst_abs,
        source_exists=True, destination_exists=False,
    )


def build_move_plan(root: Path, operations: list[dict[str, object]]) -> list[MovePlan]:
    plans: list[MovePlan] = []
    for op in operations:
        sample_id = str(op.get("sample_id") or "")
        operation = str(op.get("operation") or "")
        if operation not in (OPERATION_QUARANTINE, OPERATION_HOLDOUT_NO_SHOWER):
            continue
        for kind, (src_field, dst_field) in KIND_TO_FIELD.items():
            src_rel = str(op.get(src_field) or "").strip()
            dst_rel = str(op.get(dst_field) or "").strip()
            plans.append(classify_plan(root, sample_id, operation, kind, src_rel, dst_rel))
    return plans


# ---------------------------------------------------------------------------
# Preflight
# ---------------------------------------------------------------------------


@dataclass
class PreflightResult:
    total: int
    would_move: int
    safety_blocked: int
    missing_source: int
    destination_exists: int

    @property
    def ok(self) -> bool:
        return (
            self.safety_blocked == 0
            and self.missing_source == 0
            and self.destination_exists == 0
        )


def preflight(plans: list[MovePlan]) -> PreflightResult:
    counts: Counter[str] = Counter(p.status for p in plans)
    return PreflightResult(
        total=len(plans),
        would_move=counts.get(PLAN_WOULD_MOVE, 0),
        safety_blocked=counts.get(PLAN_SAFETY_BLOCKED, 0),
        missing_source=counts.get(PLAN_MISSING_SOURCE, 0),
        destination_exists=counts.get(PLAN_DESTINATION_EXISTS, 0),
    )


# ---------------------------------------------------------------------------
# Index update helpers
# ---------------------------------------------------------------------------


def affected_sessions(operations: list[dict[str, object]]) -> set[str]:
    sessions: set[str] = set()
    for op in operations:
        if op.get("operation") in (OPERATION_QUARANTINE, OPERATION_HOLDOUT_NO_SHOWER):
            sessions.add(str(op.get("session_id") or ""))
    return sessions


def removed_sample_ids(operations: list[dict[str, object]]) -> set[str]:
    removed: set[str] = set()
    for op in operations:
        if op.get("operation") in (OPERATION_QUARANTINE, OPERATION_HOLDOUT_NO_SHOWER):
            removed.add(str(op.get("sample_id") or ""))
    return removed


def session_slice_index_path(root: Path, session_id: str) -> Path:
    return root / "sliced_windows_by_session" / session_id / "slice_index.csv"


def filter_slice_index(
    path: Path,
    removed: set[str],
) -> tuple[int, int, list[str], list[dict[str, str]]] | None:
    """Read a slice_index.csv and return (original_count, kept_count, fields, kept_rows)."""
    if not path.exists():
        return None
    fields, rows = read_csv(path)
    kept = [r for r in rows if (r.get("sample_id") or "").strip() not in removed]
    return len(rows), len(kept), fields, kept


# ---------------------------------------------------------------------------
# Backup
# ---------------------------------------------------------------------------


def backup_indices(root: Path, sessions: set[str], timestamp: str) -> Path:
    backup_dir = root / BACKUPS_REL / timestamp
    backup_dir.mkdir(parents=True, exist_ok=True)
    main_src = root / SLICE_INDEX_REL
    if main_src.exists():
        shutil.copy2(main_src, backup_dir / "slice_index.csv")
    sess_dir = backup_dir / "per_session"
    sess_dir.mkdir(parents=True, exist_ok=True)
    for session in sorted(sessions):
        src = session_slice_index_path(root, session)
        if src.exists():
            shutil.copy2(src, sess_dir / f"{session}_slice_index.csv")
    return backup_dir


# ---------------------------------------------------------------------------
# Execution
# ---------------------------------------------------------------------------


@dataclass
class ApplyResult:
    planned_moves: int
    executed_moves: int
    skipped_missing_source: int
    skipped_destination_exists: int
    safety_blocked: int
    index_rows_removed: int
    per_session_removed: dict[str, int]
    backup_dir: Path | None
    preflight_ok: bool
    aborted: bool
    log_rows: list[dict[str, object]] = field(default_factory=list)


def _log_row(plan: MovePlan, status: str) -> dict[str, object]:
    return {
        "sample_id": plan.sample_id,
        "operation": plan.operation,
        "kind": plan.kind,
        "source_path": plan.source_rel,
        "destination_path": plan.destination_rel,
        "source_exists": plan.source_exists,
        "destination_exists": plan.destination_exists,
        "status": status,
        "note": plan.note,
    }


def execute_plan(
    root: Path,
    operations: list[dict[str, object]],
    plans: list[MovePlan],
    apply: bool,
) -> ApplyResult:
    pre = preflight(plans)
    sessions = affected_sessions(operations)
    removed = removed_sample_ids(operations)
    log_rows: list[dict[str, object]] = []

    # --- Apply with preflight failure: ABORT before touching anything. ---
    if apply and not pre.ok:
        for plan in plans:
            if plan.status == PLAN_WOULD_MOVE:
                log_status = "preflight_aborted"
            else:
                log_status = plan.status
            log_rows.append(_log_row(plan, log_status))
        return ApplyResult(
            planned_moves=len(plans),
            executed_moves=0,
            skipped_missing_source=pre.missing_source,
            skipped_destination_exists=pre.destination_exists,
            safety_blocked=pre.safety_blocked,
            index_rows_removed=0,
            per_session_removed={},
            backup_dir=None,
            preflight_ok=False,
            aborted=True,
            log_rows=log_rows,
        )

    # --- Apply with preflight pass: backup, move, rewrite indexes. ---
    backup_dir: Path | None = None
    executed = 0
    index_rows_removed = 0
    per_session_removed: dict[str, int] = {}

    if apply and pre.ok:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        backup_dir = backup_indices(root, sessions, timestamp)
        for plan in plans:
            if plan.status == PLAN_WOULD_MOVE:
                plan.destination_abs.parent.mkdir(parents=True, exist_ok=True)
                shutil.move(str(plan.source_abs), str(plan.destination_abs))
                executed += 1
                log_rows.append(_log_row(plan, "moved"))
            else:
                log_rows.append(_log_row(plan, plan.status))
        main_path = root / SLICE_INDEX_REL
        res = filter_slice_index(main_path, removed)
        if res is not None:
            original, kept_n, fields, kept = res
            index_rows_removed = original - kept_n
            write_csv(main_path, kept, fields)
        for session in sorted(sessions):
            spath = session_slice_index_path(root, session)
            res = filter_slice_index(spath, removed)
            if res is not None:
                original, kept_n, fields, kept = res
                per_session_removed[session] = original - kept_n
                write_csv(spath, kept, fields)
    else:
        # --- Dry-run: generate log only, no moves/backup/rewrites. ---
        for plan in plans:
            log_status = plan.status if plan.status != PLAN_WOULD_MOVE else "would_move"
            log_rows.append(_log_row(plan, log_status))
        executed = pre.would_move

    return ApplyResult(
        planned_moves=len(plans),
        executed_moves=executed,
        skipped_missing_source=pre.missing_source,
        skipped_destination_exists=pre.destination_exists,
        safety_blocked=pre.safety_blocked,
        index_rows_removed=index_rows_removed,
        per_session_removed=per_session_removed,
        backup_dir=backup_dir,
        preflight_ok=pre.ok,
        aborted=False,
        log_rows=log_rows,
    )


# ---------------------------------------------------------------------------
# Report
# ---------------------------------------------------------------------------


LOG_FIELDS = [
    "sample_id",
    "operation",
    "kind",
    "source_path",
    "destination_path",
    "source_exists",
    "destination_exists",
    "status",
    "note",
]


def write_apply_report(
    root: Path,
    result: ApplyResult,
    operations: list[dict[str, object]],
    apply: bool,
) -> tuple[Path, Path]:
    out_dir = root / OUTPUT_DIR_REL
    out_dir.mkdir(parents=True, exist_ok=True)
    log_path = out_dir / "apply_cleanup_log.csv"
    report_path = out_dir / "apply_cleanup_report.md"

    write_csv(log_path, result.log_rows, LOG_FIELDS)

    op_counts: Counter[str] = Counter()
    for op in operations:
        op_counts[str(op.get("operation") or "")] += 1
    status_counts: Counter[str] = Counter()
    for row in result.log_rows:
        status_counts[str(row.get("status"))] += 1

    mode_label: str
    if apply and result.aborted:
        mode_label = "APPLY ABORTED (preflight failed; nothing moved)"
    elif apply and result.preflight_ok:
        mode_label = "APPLY (files moved)"
    else:
        mode_label = "DRY-RUN (no files moved)"

    lines: list[str] = []
    lines.append("# Problem Sample Cleanup — Apply Report")
    lines.append("")
    lines.append(f"- Mode: **{mode_label}**")
    lines.append(f"- Preflight OK: {result.preflight_ok}")
    lines.append(f"- Aborted: {result.aborted}")
    lines.append(
        f"- Backup directory: `{result.backup_dir if result.backup_dir else 'N/A (not created)'}`"
    )
    lines.append(f"- Operations loaded: {len(operations)}")
    lines.append(
        f"  - quarantine: {op_counts.get(OPERATION_QUARANTINE, 0)}, "
        f"holdout_no_shower: {op_counts.get(OPERATION_HOLDOUT_NO_SHOWER, 0)}, "
        f"unmatched (no move): {op_counts.get(OPERATION_UNMATCHED, 0)}"
    )
    lines.append(f"- Move plans: {result.planned_moves}")
    lines.append(
        f"- {'Executed' if apply else 'Would-execute'} moves: {result.executed_moves}"
    )
    lines.append(f"- Skipped (missing source): {result.skipped_missing_source}")
    lines.append(f"- Skipped (destination exists): {result.skipped_destination_exists}")
    lines.append(f"- Safety-blocked: {result.safety_blocked}")
    lines.append("")
    if apply and result.aborted:
        lines.append("### Preflight abort")
        lines.append("")
        lines.append(
            "Apply was requested but preflight found blocking plans. No backup was "
            "created, no files were moved, and no index files were rewritten. Fix the "
            "blocking plans (safety / missing source / destination exists) and re-run."
        )
        lines.append("")
    lines.append("### Move status breakdown")
    lines.append("")
    lines.append("| status | count |")
    lines.append("|---|---:|")
    for status, count in sorted(status_counts.items()):
        lines.append(f"| {status} | {count} |")
    lines.append("")
    if apply and result.preflight_ok and not result.aborted:
        lines.append("### Index updates")
        lines.append("")
        lines.append("| target | rows_removed |")
        lines.append("|---|---:|")
        lines.append(
            f"| `sliced_windows_by_session/slice_index.csv` | {result.index_rows_removed} |"
        )
        for session, removed in sorted(result.per_session_removed.items()):
            lines.append(
                f"| `sliced_windows_by_session/{session}/slice_index.csv` | {removed} |"
            )
        lines.append("")
        main_path = root / SLICE_INDEX_REL
        _, post_rows = read_csv(main_path) if main_path.exists() else ([], [])
        post_removed_leak = sum(
            1 for r in post_rows
            if (r.get("sample_id") or "").strip() in removed_sample_ids(operations)
        )
        lines.append("### Post-apply verification")
        lines.append("")
        lines.append(f"- clean slice_index rows after apply: {len(post_rows)}")
        lines.append(
            f"- quarantine/holdout samples remaining in main index (must be 0): "
            f"{post_removed_leak}"
        )
        lines.append("")
    lines.append("## Safety")
    lines.append("")
    lines.append("- Protected raw dirs (resolve-based): " + ", ".join(
        f"`{d}`" for d in RAW_PROTECTED_DIRS))
    lines.append(f"- Source must resolve under `{SOURCE_ALLOWED_DIR}/`.")
    lines.append("- Destinations must resolve under the operation's quarantine/holdout root.")
    lines.append("- Absolute paths and any path containing `..` are rejected before resolve.")
    lines.append("- Preflight aborts `--apply` if any plan is safety_blocked / "
                 "missing_source / destination_exists.")
    lines.append("- No `rm` is performed; files are `shutil.move`-ed to quarantine/holdout.")
    lines.append("")

    report_path.write_text("\n".join(lines), encoding="utf-8")
    return log_path, report_path


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--project-root",
        type=Path,
        default=DEFAULT_PROJECT_ROOT,
        help="Project root containing sliced_windows_by_session/ etc.",
    )
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument(
        "--dry-run",
        action="store_true",
        default=True,
        help="Print planned moves and index updates without moving files (default).",
    )
    mode.add_argument(
        "--apply",
        action="store_true",
        default=False,
        help="Preflight then move files and rewrite slice index files.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    root: Path = args.project_root
    apply_mode = bool(args.apply)

    ops_path = root / OPERATIONS_CSV_REL
    if not ops_path.exists():
        print(
            f"ERROR: operations file not found: {ops_path}\n"
            "Run build_problem_sample_cleanup_index.py first.",
            file=sys.stderr,
        )
        return 2

    _, operations = read_csv(ops_path)
    operations = [{k: ("" if v is None else v) for k, v in row.items()} for row in operations]

    plans = build_move_plan(root, operations)
    pre = preflight(plans)

    print(f"[apply] mode                    : {'APPLY' if apply_mode else 'DRY-RUN'}")
    print(f"[apply] operations loaded       : {len(operations)}")
    print(f"[apply] move plans              : {len(plans)}")
    print(f"[apply] preflight ok            : {pre.ok}")
    print(f"[apply]   would_move            : {pre.would_move}")
    print(f"[apply]   safety_blocked        : {pre.safety_blocked}")
    print(f"[apply]   missing_source        : {pre.missing_source}")
    print(f"[apply]   destination_exists    : {pre.destination_exists}")

    if apply_mode and not pre.ok:
        print(
            "[apply] ABORT: preflight found blocking plans. "
            "No files moved, no indexes rewritten, no backup created.",
            file=sys.stderr,
        )

    result = execute_plan(root, operations, plans, apply=apply_mode)
    log_path, report_path = write_apply_report(root, result, operations, apply=apply_mode)

    print(f"[apply] {'executed' if apply_mode else 'would-execute'} moves : {result.executed_moves}")
    print(f"[apply] skipped missing source  : {result.skipped_missing_source}")
    print(f"[apply] skipped dest exists     : {result.skipped_destination_exists}")
    print(f"[apply] safety-blocked          : {result.safety_blocked}")
    print(f"[apply] aborted                 : {result.aborted}")
    if apply_mode and result.preflight_ok and not result.aborted:
        print(f"[apply] backup dir              : {result.backup_dir}")
        print(f"[apply] main index rows removed : {result.index_rows_removed}")
        for session, removed in sorted(result.per_session_removed.items()):
            print(f"[apply]   {session}/slice_index.csv rows removed: {removed}")
    print(f"[apply] log csv                 : {log_path}")
    print(f"[apply] report md               : {report_path}")

    if not apply_mode:
        print("\n[dry-run] first 10 planned moves:")
        shown = 0
        for row in result.log_rows:
            if row.get("status") in ("would_move", "safety_blocked", "missing_source",
                                     "destination_exists"):
                print(
                    f"  {row['sample_id']} [{row['kind']}] {row['source_path']} -> "
                    f"{row['destination_path']} ({row['status']})"
                )
                shown += 1
                if shown >= 10:
                    break

    if apply_mode and not result.preflight_ok:
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
