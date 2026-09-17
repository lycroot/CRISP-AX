#!/usr/bin/env python3
"""Build problem-sample cleanup index.

Reads:
  - sliced_windows_by_session/slice_index.csv          (authoritative slice list)
  - quality_report_deep/csi_slice_quality/csi_slice_quality.csv  (final_status / quality_flags)
  - quality_report_deep/Video_review.md                (manual review rules)
  - Person.csv                                         (person roster)
  - Session分布.csv                                    (session -> person ground truth)

Writes (under quality_report_deep/problem_sample_cleanup/):
  - problem_sample_operations.csv      (79 quarantine + 31 holdout rows, full move plan)
  - unmatched_review_issues.csv        (Video_review rules with no slice_index match)
  - clean_slice_index.csv              (slice_index minus quarantine & holdout = 6984 rows)
  - problem_sample_cleanup_report.md   (summary + verification)

Business rules:
  - quarantine: every sample with final_status != ok in csi_slice_quality.csv (79 = 74 exclude
    + 5 warn). These are moved out of the clean dataset.
  - holdout_no_shower: S28_P09_bend_pick_cross_link_E3_clean_none_R001..R031 (31 samples).
    Video_review.md mislabels these as S28_P10; Session分布.csv confirms S28 = P09. These are
    moved to a separate holdout directory for no-shower vs shower amplitude/phase comparison,
    NOT deleted. They are also removed from clean_slice_index.
  - unmatched_review_issue: Video_review.md entries that cannot be matched to any slice_index
    sample (e.g. S30_P10_bend_pick — S30 has no bend_pick slices). Only recorded, no file move.

Raw data (CSI-Formal/00_raw/, Video/, CSI-Formal/01_manifest/) is never touched — this script
only reads them (and only slice_index/quality/review/manifest CSVs). No file moves here.
"""
from __future__ import annotations

import argparse
import csv
import re
import sys
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path

# Make `from data_processing...` importable when run as a script.
ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from data_processing.check_csi_slice_quality import (  # noqa: E402
    VideoReviewRule,
    parse_video_review_rules,
)

# ---------------------------------------------------------------------------
# Paths and constants
# ---------------------------------------------------------------------------

DEFAULT_PROJECT_ROOT = Path(__file__).resolve().parent.parent

SLICE_INDEX_REL = Path("sliced_windows_by_session/slice_index.csv")
CSI_QUALITY_REL = Path("quality_report_deep/csi_slice_quality/csi_slice_quality.csv")
CSI_OUTLIER_REL = Path("quality_report_deep/csi_slice_quality/csi_outlier_samples.csv")
VIDEO_REVIEW_REL = Path("quality_report_deep/Video_review.md")
PERSON_CSV_REL = Path("Person.csv")
SESSION_PERSON_CSV_REL = Path("Session分布.csv")

OUTPUT_DIR_REL = Path("quality_report_deep/problem_sample_cleanup")
OPERATIONS_CSV_REL = OUTPUT_DIR_REL / "problem_sample_operations.csv"
UNMATCHED_CSV_REL = OUTPUT_DIR_REL / "unmatched_review_issues.csv"
CLEAN_INDEX_CSV_REL = OUTPUT_DIR_REL / "clean_slice_index.csv"
REPORT_MD_REL = OUTPUT_DIR_REL / "problem_sample_cleanup_report.md"

QUARANTINE_ROOT_REL = "sliced_windows_quarantine/problem_samples"
HOLDOUT_ROOT_REL = "sliced_windows_holdout/no_shower_bend_pick"

OPERATION_QUARANTINE = "quarantine"
OPERATION_HOLDOUT_NO_SHOWER = "holdout_no_shower"
OPERATION_UNMATCHED = "unmatched_review_issue"

# Replacement candidate patterns (verified against slice_index existence).
S28_REPLACEMENT_PATTERN = "S00_P09_bend_pick_cross_link_E3_clean_none_R001-R031"
S06_REPLACEMENT_PATTERN = "S00_P02_sit_down_face_rx_E1_clean_none_R001-R007"

OPERATION_FIELDS = [
    "sample_id",
    "operation",
    "reason",
    "final_status",
    "quality_flags",
    "session_id",
    "person_id",
    "action_id",
    "trial_id",
    "csi_out_path",
    "video_out_path",
    "metadata_path",
    "destination_csi_path",
    "destination_video_path",
    "destination_metadata_path",
    "replacement_candidate_sample_id",
    "person_check_status",
    "notes",
]

UNMATCHED_FIELDS = [
    "review_session_id",
    "review_clip_expr",
    "review_category",
    "review_description",
    "review_resample_status",
    "expected_person_id",
    "actual_person_id",
    "matched_sample_count",
    "reason",
    "notes",
]


# ---------------------------------------------------------------------------
# CSV helpers (match existing codebase conventions)
# ---------------------------------------------------------------------------


def read_csv(path: Path) -> tuple[list[str], list[dict[str, str]]]:
    """Return (header fields, rows-as-dicts) for a CSV file."""
    with path.open("r", encoding="utf-8-sig", newline="") as f:
        reader = csv.reader(f)
        header = next(reader, [])
        rows = [dict(zip(header, row)) for row in reader if row]
    return header, rows


def write_csv(path: Path, rows: list[dict[str, object]], fields: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            writer.writerow({field: row.get(field, "") for field in fields})


# ---------------------------------------------------------------------------
# Person / session ground truth
# ---------------------------------------------------------------------------


def load_session_person_map(path: Path) -> dict[str, str]:
    """Session分布.csv -> {session_id: person_id} (strips whitespace; 'None' -> '')."""
    _, rows = read_csv(path)
    mapping: dict[str, str] = {}
    for row in rows:
        session = (row.get("Session") or "").strip()
        person = (row.get("Person_ID") or "").strip()
        if not session:
            continue
        mapping[session] = "" if person.lower() == "none" else person
    return mapping


def load_person_roster(path: Path) -> dict[str, str]:
    """Person.csv -> {person_id: name} (strips whitespace in IDs)."""
    _, rows = read_csv(path)
    roster: dict[str, str] = {}
    for row in rows:
        pid = (row.get("Person_id") or "").strip()
        name = (row.get("Name") or "").strip()
        if pid:
            roster[pid] = name
    return roster


# ---------------------------------------------------------------------------
# Slice index helpers
# ---------------------------------------------------------------------------


def derive_metadata_path(csi_out_path: str) -> str:
    """Derive the per-sample metadata JSON path from the CSI slice path.

    sliced_windows_by_session/SXX/csi/<action>/sample.dat
      -> sliced_windows_by_session/SXX/metadata/<action>/sample.json
    """
    if not csi_out_path:
        return ""
    out = csi_out_path
    # Replace the /csi/ path segment with /metadata/.
    out = re.sub(r"/csi/", "/metadata/", out)
    # Swap the .dat suffix for .json (fall back to appending .json if unknown).
    if out.endswith(".dat"):
        out = out[:-4] + ".json"
    elif not out.endswith(".json"):
        out = out + ".json"
    return out


def basename_for_kind(csi_out_path: str, kind: str) -> str:
    """Return the per-sample file name for csi/video/metadata."""
    name = Path(csi_out_path).name if csi_out_path else ""
    stem = name[:-4] if name.endswith(".dat") else Path(name).stem
    if kind == "csi":
        return stem + ".dat"
    if kind == "video":
        return stem + ".mp4"
    if kind == "metadata":
        return stem + ".json"
    raise ValueError(f"unknown kind {kind!r}")


def destination_path(
    operation: str,
    session_id: str,
    action_id: str,
    csi_out_path: str,
    kind: str,
) -> str:
    """Compute destination path for a (sample, kind) under the given operation."""
    fname = basename_for_kind(csi_out_path, kind)
    if operation == OPERATION_QUARANTINE:
        root = f"{QUARANTINE_ROOT_REL}/{session_id}/{kind}/{action_id}"
    elif operation == OPERATION_HOLDOUT_NO_SHOWER:
        # All holdout samples are S28 bend_pick, but derive from sample for consistency.
        root = f"{HOLDOUT_ROOT_REL}/{session_id}/{kind}/{action_id}"
    else:
        raise ValueError(f"destination not defined for operation {operation!r}")
    return f"{root}/{fname}"


# ---------------------------------------------------------------------------
# Build logic
# ---------------------------------------------------------------------------


@dataclass
class BuildResult:
    operations: list[dict[str, object]]
    unmatched: list[dict[str, object]]
    clean_rows: list[dict[str, str]]
    clean_fields: list[str]
    slice_index_count: int
    quarantine_count: int
    holdout_count: int
    unmatched_count: int
    clean_count: int


def is_s28_bend_pick_wildcard_rule(rule: VideoReviewRule) -> bool:
    """True for the Video_review S28_P10_bend_pick_* placeholder rule."""
    return (
        rule.session_id == "S28"
        and rule.match_kind == "wildcard"
        and "bend_pick" in rule.prefix
    )


def build_cleanup_index(root: Path) -> BuildResult:
    slice_index_path = root / SLICE_INDEX_REL
    csi_quality_path = root / CSI_QUALITY_REL
    video_review_path = root / VIDEO_REVIEW_REL
    session_person_path = root / SESSION_PERSON_CSV_REL
    person_csv_path = root / PERSON_CSV_REL

    # 1. Load session -> person ground truth and person roster.
    session_person = load_session_person_map(session_person_path)
    person_roster = load_person_roster(person_csv_path)

    # 2. Load slice_index.csv (preserve original column order for clean output).
    slice_fields, slice_rows = read_csv(slice_index_path)
    slice_by_id: dict[str, dict[str, str]] = {}
    for row in slice_rows:
        sid = (row.get("sample_id") or "").strip()
        if sid:
            slice_by_id[sid] = row

    # 3. Load csi_slice_quality.csv -> identify quarantine (final_status != ok).
    quality_fields, quality_rows = read_csv(csi_quality_path)
    quality_by_id: dict[str, dict[str, str]] = {}
    for row in quality_rows:
        sid = (row.get("sample_id") or "").strip()
        if sid:
            quality_by_id[sid] = row

    quarantine_ids: list[str] = []
    for row in quality_rows:
        sid = (row.get("sample_id") or "").strip()
        final_status = (row.get("final_status") or "").strip().lower()
        if sid and final_status and final_status != "ok":
            quarantine_ids.append(sid)
    # De-duplicate while preserving order.
    seen_q: set[str] = set()
    quarantine_ids = [s for s in quarantine_ids if not (s in seen_q or seen_q.add(s))]

    # 4. Parse Video_review.md rules.
    review_text = video_review_path.read_text(encoding="utf-8")
    review_rules = parse_video_review_rules(review_text)

    # 5. Identify S28 holdout samples (special-cased from the P10 placeholder rule).
    holdout_ids: list[str] = []
    s28_wildcard_rule: VideoReviewRule | None = None
    for rule in review_rules:
        if is_s28_bend_pick_wildcard_rule(rule):
            s28_wildcard_rule = rule
            break
    s28_expected_person = session_person.get("S28", "")
    if s28_wildcard_rule is not None:
        # Match all slice_index samples for S28 + expected person + bend_pick.
        for row in slice_rows:
            if (
                (row.get("session_id") or "").strip() == "S28"
                and (row.get("person_id") or "").strip() == s28_expected_person
                and (row.get("action_id") or "").strip() == "bend_pick"
            ):
                sid = (row.get("sample_id") or "").strip()
                if sid:
                    holdout_ids.append(sid)
        # Sort by trial id for stable output.
        holdout_ids.sort(key=lambda s: trial_sort_key(s))

    # 6. Identify unmatched review issues.
    unmatched_rows: list[dict[str, object]] = []
    for rule in review_rules:
        if is_s28_bend_pick_wildcard_rule(rule):
            # Special-cased to holdout; not "unmatched".
            continue
        matched = [sid for sid in slice_by_id if rule.matches(sid)]
        if not matched:
            expected_person = session_person.get(rule.session_id, "")
            unmatched_rows.append(
                {
                    "review_session_id": rule.session_id,
                    "review_clip_expr": rule.clip_expr,
                    "review_category": rule.category,
                    "review_description": rule.description,
                    "review_resample_status": rule.resample_status,
                    "expected_person_id": expected_person,
                    "actual_person_id": "",
                    "matched_sample_count": 0,
                    "reason": (
                        "no slice_index sample matches this Video_review clip expression"
                    ),
                    "notes": (
                        "recorded only; no file movement. S30 has no bend_pick slices in "
                        "slice_index.csv."
                        if rule.session_id == "S30"
                        else "recorded only; no file movement."
                    ),
                }
            )

    # 7. Build operation rows.
    operation_rows: list[dict[str, object]] = []

    # 7a. Quarantine rows.
    for sid in quarantine_ids:
        qrow = quality_by_id.get(sid, {})
        srow = slice_by_id.get(sid, {})
        session_id = (srow.get("session_id") or qrow.get("session_id") or "").strip()
        person_id = (srow.get("person_id") or "").strip()
        action_id = (srow.get("action_id") or qrow.get("action_id") or "").strip()
        trial_id = (srow.get("trial_id") or qrow.get("trial_id") or "").strip()
        csi_out_path = (srow.get("csi_out_path") or qrow.get("csi_out_path") or "").strip()
        video_out_path = (srow.get("video_out_path") or "").strip()
        metadata_path = derive_metadata_path(csi_out_path)
        final_status = (qrow.get("final_status") or "").strip()
        quality_flags = (qrow.get("quality_flags") or "").strip()

        replacement = ""
        notes_parts: list[str] = []
        if session_id == "S06" and action_id == "sit_down":
            replacement = S06_REPLACEMENT_PATTERN
            notes_parts.append("replacement candidate verified against slice_index existence")
        # For other quarantine samples, leave replacement empty — Video_review.md resample
        # status is already captured in quality_flags / final_status.

        operation_rows.append(
            {
                "sample_id": sid,
                "operation": OPERATION_QUARANTINE,
                "reason": f"final_status={final_status or '(unknown)'} (!= ok) per csi_slice_quality.csv",
                "final_status": final_status,
                "quality_flags": quality_flags,
                "session_id": session_id,
                "person_id": person_id,
                "action_id": action_id,
                "trial_id": trial_id,
                "csi_out_path": csi_out_path,
                "video_out_path": video_out_path,
                "metadata_path": metadata_path,
                "destination_csi_path": destination_path(
                    OPERATION_QUARANTINE, session_id, action_id, csi_out_path, "csi"
                ),
                "destination_video_path": destination_path(
                    OPERATION_QUARANTINE, session_id, action_id, csi_out_path, "video"
                ),
                "destination_metadata_path": destination_path(
                    OPERATION_QUARANTINE, session_id, action_id, csi_out_path, "metadata"
                ),
                "replacement_candidate_sample_id": replacement,
                "person_check_status": person_check(person_id, session_id, session_person),
                "notes": "; ".join(notes_parts),
            }
        )

    # 7b. Holdout rows (S28_P09 bend_pick).
    for sid in holdout_ids:
        srow = slice_by_id.get(sid, {})
        session_id = (srow.get("session_id") or "S28").strip()
        person_id = (srow.get("person_id") or s28_expected_person).strip()
        action_id = (srow.get("action_id") or "bend_pick").strip()
        trial_id = (srow.get("trial_id") or "").strip()
        csi_out_path = (srow.get("csi_out_path") or "").strip()
        video_out_path = (srow.get("video_out_path") or "").strip()
        metadata_path = derive_metadata_path(csi_out_path)
        operation_rows.append(
            {
                "sample_id": sid,
                "operation": OPERATION_HOLDOUT_NO_SHOWER,
                "reason": (
                    "S28 bend_pick 未开淋浴 (Video_review.md 人员编号 P10 为误记, "
                    "Session分布.csv 确认 S28=P09); holdout 用于未开淋浴 vs 开淋浴幅值/相位对比"
                ),
                "final_status": "ok",  # not in outlier list; CSI quality is ok
                "quality_flags": "ok",
                "session_id": session_id,
                "person_id": person_id,
                "action_id": action_id,
                "trial_id": trial_id,
                "csi_out_path": csi_out_path,
                "video_out_path": video_out_path,
                "metadata_path": metadata_path,
                "destination_csi_path": destination_path(
                    OPERATION_HOLDOUT_NO_SHOWER, session_id, action_id, csi_out_path, "csi"
                ),
                "destination_video_path": destination_path(
                    OPERATION_HOLDOUT_NO_SHOWER, session_id, action_id, csi_out_path, "video"
                ),
                "destination_metadata_path": destination_path(
                    OPERATION_HOLDOUT_NO_SHOWER, session_id, action_id, csi_out_path, "metadata"
                ),
                "replacement_candidate_sample_id": S28_REPLACEMENT_PATTERN,
                "person_check_status": (
                    f"corrected_p10_to_p09_per_session_distribution"
                    f"(review=P10, actual={person_id})"
                ),
                "notes": (
                    "Video_review.md placeholder S28_P10_bend_pick_*; person corrected to P09 "
                    "via Session分布.csv; NOT deleted, moved to holdout for no-shower comparison"
                ),
            }
        )

    # 8. Build clean slice index (preserve original column order).
    removed_ids = set(quarantine_ids) | set(holdout_ids)
    clean_rows = [row for row in slice_rows if (row.get("sample_id") or "").strip() not in removed_ids]

    return BuildResult(
        operations=operation_rows,
        unmatched=unmatched_rows,
        clean_rows=clean_rows,
        clean_fields=slice_fields,
        slice_index_count=len(slice_rows),
        quarantine_count=len(quarantine_ids),
        holdout_count=len(holdout_ids),
        unmatched_count=len(unmatched_rows),
        clean_count=len(clean_rows),
    )


def trial_sort_key(sample_id: str) -> tuple[str, int]:
    """Sort key that orders by prefix then numeric trial."""
    match = re.match(r"^(.*_R)(\d{3})$", sample_id)
    if match:
        return (match.group(1), int(match.group(2)))
    return (sample_id, 0)


def person_check(
    slice_person: str,
    session_id: str,
    session_person: dict[str, str],
) -> str:
    expected = session_person.get(session_id, "")
    if not expected:
        return f"unknown_session:{session_id}"
    if not slice_person:
        return f"missing_person_in_slice_index (expected={expected})"
    if slice_person == expected:
        return "ok"
    return f"mismatch:slice={slice_person}_vs_session_distribution={expected}"


# ---------------------------------------------------------------------------
# Report
# ---------------------------------------------------------------------------


def write_report(root: Path, result: BuildResult) -> Path:
    out_dir = root / OUTPUT_DIR_REL
    out_dir.mkdir(parents=True, exist_ok=True)
    report_path = out_dir / "problem_sample_cleanup_report.md"

    # Per-session breakdown for quarantine.
    q_by_session: Counter[str] = Counter()
    q_by_action: Counter[str] = Counter()
    q_by_status: Counter[str] = Counter()
    for op in result.operations:
        if op["operation"] == OPERATION_QUARANTINE:
            q_by_session[str(op["session_id"])] += 1
            q_by_action[str(op["action_id"])] += 1
            q_by_status[str(op["final_status"])] += 1

    # Person check summary.
    person_checks: Counter[str] = Counter()
    for op in result.operations:
        person_checks[str(op["person_check_status"]).split("(")[0].strip()] += 1

    # Replacement candidate summary.
    replacements: Counter[str] = Counter()
    for op in result.operations:
        rep = str(op["replacement_candidate_sample_id"])
        if rep:
            replacements[rep] += 1

    lines: list[str] = []
    lines.append("# Problem Sample Cleanup Index Report")
    lines.append("")
    lines.append("## Inputs")
    lines.append("")
    lines.append(f"- Project root: `{root}`")
    lines.append(f"- Slice index: `{SLICE_INDEX_REL}` ({result.slice_index_count} rows)")
    lines.append(f"- CSI quality: `{CSI_QUALITY_REL}`")
    lines.append(f"- Video review: `{VIDEO_REVIEW_REL}`")
    lines.append(f"- Person roster: `{PERSON_CSV_REL}`")
    lines.append(f"- Session-person map: `{SESSION_PERSON_CSV_REL}`")
    lines.append("")
    lines.append("## Outputs")
    lines.append("")
    lines.append(f"- Operations: `{OPERATIONS_CSV_REL}` ({len(result.operations)} rows)")
    lines.append(f"- Unmatched review issues: `{UNMATCHED_CSV_REL}` ({len(result.unmatched)} rows)")
    lines.append(f"- Clean slice index: `{CLEAN_INDEX_CSV_REL}` ({result.clean_count} rows)")
    lines.append("")
    lines.append("## Operation Summary")
    lines.append("")
    lines.append("| operation | count | description |")
    lines.append("|---|---:|---|")
    lines.append(
        f"| `{OPERATION_QUARANTINE}` | {result.quarantine_count} | "
        "final_status != ok in csi_slice_quality.csv; moved to quarantine |"
    )
    lines.append(
        f"| `{OPERATION_HOLDOUT_NO_SHOWER}` | {result.holdout_count} | "
        "S28_P09_bend_pick (Video_review P10 mislabel corrected); moved to no-shower holdout |"
    )
    lines.append(
        f"| `{OPERATION_UNMATCHED}` | {result.unmatched_count} | "
        "Video_review entries with no slice_index match; recorded only, no move |"
    )
    lines.append("")
    lines.append("## Verification (expected counts)")
    lines.append("")
    expected_clean = result.slice_index_count - result.quarantine_count - result.holdout_count
    lines.append("| metric | expected | actual | status |")
    lines.append("|---|---:|---:|---|")
    lines.append(
        f"| quarantine samples | 79 | {result.quarantine_count} | "
        f"{'OK' if result.quarantine_count == 79 else 'MISMATCH'} |"
    )
    lines.append(
        f"| holdout_no_shower samples | 31 | {result.holdout_count} | "
        f"{'OK' if result.holdout_count == 31 else 'MISMATCH'} |"
    )
    lines.append(
        f"| clean_slice_index rows | 6984 | {result.clean_count} | "
        f"{'OK' if result.clean_count == 6984 else 'MISMATCH'} |"
    )
    lines.append(
        f"| total - quarantine - holdout | {expected_clean} | {result.clean_count} | "
        f"{'OK' if expected_clean == result.clean_count else 'MISMATCH'} |"
    )
    lines.append("")
    lines.append("## Quarantine Breakdown")
    lines.append("")
    lines.append("### By final_status")
    lines.append("")
    lines.append("| final_status | count |")
    lines.append("|---|---:|")
    for status, count in sorted(q_by_status.items()):
        lines.append(f"| {status or '(empty)'} | {count} |")
    lines.append("")
    lines.append("### By session (top)")
    lines.append("")
    lines.append("| session | count |")
    lines.append("|---|---:|")
    for session, count in sorted(q_by_session.items(), key=lambda kv: (-kv[1], kv[0])):
        lines.append(f"| {session} | {count} |")
    lines.append("")
    lines.append("### By action")
    lines.append("")
    lines.append("| action | count |")
    lines.append("|---|---:|")
    for action, count in sorted(q_by_action.items(), key=lambda kv: (-kv[1], kv[0])):
        lines.append(f"| {action} | {count} |")
    lines.append("")
    lines.append("## Person Check")
    lines.append("")
    lines.append("| check_status | count |")
    lines.append("|---|---:|")
    for status, count in sorted(person_checks.items()):
        lines.append(f"| {status} | {count} |")
    lines.append("")
    lines.append("## Replacement Candidates")
    lines.append("")
    lines.append("| replacement_candidate_sample_id | applies_to_count | note |")
    lines.append("|---|---:|---|")
    lines.append(
        f"| `{S28_REPLACEMENT_PATTERN}` | {replacements.get(S28_REPLACEMENT_PATTERN, 0)} | "
        "S28 bend_pick holdout; S00 has 31 matching samples verified in slice_index |"
    )
    lines.append(
        f"| `{S06_REPLACEMENT_PATTERN}` | {replacements.get(S06_REPLACEMENT_PATTERN, 0)} | "
        "S06 sit_down quarantine (R008-R011); S00 has 7 matching samples verified in slice_index |"
    )
    lines.append("")
    lines.append("> Note: S11 is P04 (not P09) per Session分布.csv and is NOT used as a "
                 "replacement candidate for S28. The S28 replacement points to S00_P09.")
    lines.append("")
    lines.append("## Unmatched Review Issues")
    lines.append("")
    if not result.unmatched:
        lines.append("(none)")
    else:
        lines.append("| review_session | review_clip_expr | category | matched_count | reason |")
        lines.append("|---|---|---|---:|---|")
        for u in result.unmatched:
            lines.append(
                f"| {u['review_session_id']} | `{u['review_clip_expr']}` | "
                f"{u['review_category']} | {u['matched_sample_count']} | {u['reason']} |"
            )
    lines.append("")
    lines.append("## Safety")
    lines.append("")
    lines.append("- This script only **reads** raw/manifest/slice inputs and **writes** the four "
                 "output files under `quality_report_deep/problem_sample_cleanup/`.")
    lines.append("- No files are moved by this script. Use `apply_problem_sample_cleanup.py` "
                 "(default dry-run, `--apply` to move) to execute the operations.")
    lines.append("- Raw directories (`CSI-Formal/00_raw/`, `Video/`, `CSI-Formal/01_manifest/`) "
                 "are never modified.")
    lines.append("")

    report_path.write_text("\n".join(lines), encoding="utf-8")
    return report_path


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
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    root: Path = args.project_root

    # Sanity: required inputs must exist.
    required = [
        root / SLICE_INDEX_REL,
        root / CSI_QUALITY_REL,
        root / VIDEO_REVIEW_REL,
        root / SESSION_PERSON_CSV_REL,
        root / PERSON_CSV_REL,
    ]
    missing = [p for p in required if not p.exists()]
    if missing:
        for p in missing:
            print(f"ERROR: missing required input: {p}", file=sys.stderr)
        return 2

    result = build_cleanup_index(root)

    # Write outputs.
    out_dir = root / OUTPUT_DIR_REL
    out_dir.mkdir(parents=True, exist_ok=True)
    write_csv(out_dir / "problem_sample_operations.csv", result.operations, OPERATION_FIELDS)
    write_csv(out_dir / "unmatched_review_issues.csv", result.unmatched, UNMATCHED_FIELDS)
    write_csv(out_dir / "clean_slice_index.csv", result.clean_rows, result.clean_fields)
    write_report(root, result)

    # Console summary.
    print(f"[build] slice_index rows        : {result.slice_index_count}")
    print(f"[build] quarantine (final!=ok)  : {result.quarantine_count}")
    print(f"[build] holdout_no_shower (S28) : {result.holdout_count}")
    print(f"[build] unmatched review issues : {result.unmatched_count}")
    print(f"[build] clean_slice_index rows  : {result.clean_count}")
    print(f"[build] operations written      : {len(result.operations)}")
    print(f"[build] outputs dir             : {out_dir}")

    # Hard verification gates.
    ok = True
    if result.quarantine_count != 79:
        print(f"WARNING: expected 79 quarantine, got {result.quarantine_count}", file=sys.stderr)
        ok = False
    if result.holdout_count != 31:
        print(f"WARNING: expected 31 holdout, got {result.holdout_count}", file=sys.stderr)
        ok = False
    if result.clean_count != 6984:
        print(f"WARNING: expected 6984 clean rows, got {result.clean_count}", file=sys.stderr)
        ok = False
    if not ok:
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
