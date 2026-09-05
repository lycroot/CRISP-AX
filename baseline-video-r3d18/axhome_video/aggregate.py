from __future__ import annotations

import csv
import json
from collections.abc import Hashable
from pathlib import Path
from typing import Protocol, Sequence

import numpy as np

from .data import VideoSampleRecord
from .metrics import try_write_confusion_plot

SPLIT_ASSIGNMENT_FIELDS = (
    "sample_id",
    "person_id",
    "action_id",
    "archive_session",
    "split",
    "seed",
)
SUMMARY_METRICS = (
    "accuracy",
    "balanced_accuracy",
    "macro_f1",
    "weighted_f1",
)
RUN_FINITE_FIELDS = (*SUMMARY_METRICS, "test_loss", "best_val_macro_f1")


class ThreeWaySplit(Protocol):
    train: tuple[VideoSampleRecord, ...]
    val: tuple[VideoSampleRecord, ...]
    test: tuple[VideoSampleRecord, ...]


def split_summary(split: ThreeWaySplit, *, seed: int) -> dict[str, object]:
    counts = {
        "train": len(split.train),
        "validation": len(split.val),
        "test": len(split.test),
    }
    if min(counts.values()) <= 0:
        raise ValueError("split summary received an empty partition")
    total = sum(counts.values())
    return {
        "seed": int(seed),
        "total_samples": total,
        "counts": counts,
        "actual_ratios": {
            name: count / total for name, count in counts.items()
        },
    }


def write_split_assignments(
    path: str | Path,
    split: ThreeWaySplit,
    *,
    seed: int | None = None,
) -> None:
    resolved_seed = seed if seed is not None else getattr(split, "seed", None)
    if resolved_seed is None:
        raise ValueError("seed is required for split assignments")
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    rows = []
    for split_name, samples in (
        ("train", split.train),
        ("validation", split.val),
        ("test", split.test),
    ):
        for sample in samples:
            rows.append(
                {
                    "sample_id": sample.sample_id,
                    "person_id": sample.person_id,
                    "action_id": sample.action_id,
                    "archive_session": sample.archive_session_id,
                    "split": split_name,
                    "seed": int(resolved_seed),
                }
            )
    rows.sort(key=lambda row: str(row["sample_id"]))
    sample_ids = [str(row["sample_id"]) for row in rows]
    if len(sample_ids) != len(set(sample_ids)):
        raise ValueError("split assignments contain duplicate sample IDs")
    with destination.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=SPLIT_ASSIGNMENT_FIELDS)
        writer.writeheader()
        writer.writerows(rows)


def _read_assignment_map(path: str | Path) -> dict[str, tuple[str, int]]:
    source = Path(path)
    if not source.is_file():
        raise FileNotFoundError(f"split assignment file not found: {source}")
    assignments: dict[str, tuple[str, int]] = {}
    with source.open(newline="", encoding="utf-8-sig") as handle:
        reader = csv.DictReader(handle)
        required = {"sample_id", "split", "seed"}
        missing = required - set(reader.fieldnames or [])
        if missing:
            raise ValueError(
                f"{source} is missing assignment fields: "
                + ", ".join(sorted(missing))
            )
        for row_number, row in enumerate(reader, start=2):
            sample_id = row["sample_id"]
            if sample_id in assignments:
                raise ValueError(
                    f"{source}, row {row_number}: duplicate sample ID {sample_id}"
                )
            try:
                seed = int(row["seed"])
            except ValueError as exc:
                raise ValueError(
                    f"{source}, row {row_number}: invalid seed"
                ) from exc
            assignments[sample_id] = (row["split"], seed)
    if not assignments:
        raise ValueError(f"split assignment file is empty: {source}")
    return assignments


def validate_reference_assignments(
    video_path: str | Path,
    reference_path: str | Path,
    *,
    seed: int,
) -> None:
    video = _read_assignment_map(video_path)
    reference = _read_assignment_map(reference_path)
    if set(video) != set(reference):
        missing = sorted(set(reference) - set(video))
        extra = sorted(set(video) - set(reference))
        raise ValueError(
            "assignment sample IDs differ: "
            f"missing={missing[:5]}, extra={extra[:5]}"
        )
    wrong_seed = [
        sample_id
        for sample_id, (_, assigned_seed) in video.items()
        if assigned_seed != seed
    ]
    if wrong_seed:
        raise ValueError(
            f"video assignments use the wrong seed for {wrong_seed[0]}"
        )
    mismatched = [
        sample_id
        for sample_id in sorted(video)
        if video[sample_id] != reference[sample_id]
    ]
    if mismatched:
        sample_id = mismatched[0]
        raise ValueError(
            f"assignment mismatch for {sample_id}: "
            f"video={video[sample_id]}, reference={reference[sample_id]}"
        )


def summarize_runs(
    runs: Sequence[dict[str, object]],
    *,
    requested: Sequence[Hashable],
    failed: Sequence[dict[str, object]],
    class_names: Sequence[str],
    id_field: str = "seed",
    requested_key: str = "requested_seeds",
    completed_key: str = "completed_seeds",
    failed_key: str = "failed_seeds",
    protocol: str = "person_action_stratified_in_domain",
) -> dict[str, object]:
    requested_ids = list(requested)
    if not requested_ids or len(requested_ids) != len(set(requested_ids)):
        raise ValueError("requested run IDs must be non-empty and unique")
    classes = list(class_names)
    if not classes or len(classes) != len(set(classes)):
        raise ValueError("class_names must be non-empty and unique")

    run_by_id: dict[Hashable, dict[str, object]] = {}
    normalized_matrices: list[np.ndarray] = []
    for run in runs:
        run_id = run[id_field]
        if not isinstance(run_id, Hashable):
            raise ValueError(f"run {id_field} must be hashable")
        if run_id not in requested_ids:
            raise ValueError(f"completed run was not requested: {run_id}")
        if run_id in run_by_id:
            raise ValueError(f"duplicate completed run: {run_id}")
        for metric in RUN_FINITE_FIELDS:
            if not np.isfinite(float(run[metric])):
                raise ValueError(f"non-finite {metric} in video runs")
        matrix = np.asarray(run["confusion_matrix"], dtype=np.float64)
        expected_shape = (len(classes), len(classes))
        if matrix.shape != expected_shape:
            raise ValueError(
                f"run {run_id} confusion matrix must have shape "
                f"{expected_shape}, got {matrix.shape}"
            )
        if not np.isfinite(matrix).all() or np.any(matrix < 0):
            raise ValueError(f"invalid confusion matrix in run {run_id}")
        row_sums = matrix.sum(axis=1, keepdims=True)
        if np.any(row_sums <= 0):
            raise ValueError(f"empty true class in confusion matrix for {run_id}")
        run_by_id[run_id] = run
        normalized_matrices.append(matrix / row_sums)

    failures = [dict(item) for item in failed]
    failed_ids: list[Hashable] = []
    for item in failures:
        failure_id = item.get("id", item.get(id_field))
        if failure_id is None or "error" not in item:
            raise ValueError("failed run records require an ID and error")
        if not isinstance(failure_id, Hashable):
            raise ValueError("failed run ID must be hashable")
        if failure_id not in requested_ids:
            raise ValueError(f"failed run was not requested: {failure_id}")
        if failure_id in run_by_id:
            raise ValueError(f"run is both completed and failed: {failure_id}")
        item["id"] = failure_id
        item["error"] = str(item["error"])
        failed_ids.append(failure_id)
    if len(failed_ids) != len(set(failed_ids)):
        raise ValueError("failed runs contain duplicate IDs")

    completed = [run_id for run_id in requested_ids if run_id in run_by_id]
    architectures = sorted(
        {
            str(run_by_id[run_id]["arch"])
            for run_id in completed
            if run_by_id[run_id].get("arch")
        }
    )
    summary: dict[str, object] = {
        "protocol": protocol,
        "architecture": architectures[0] if len(architectures) == 1 else None,
        requested_key: requested_ids,
        completed_key: completed,
        failed_key: failures,
        "complete": completed == requested_ids and not failures,
        "class_names": classes,
        "num_completed": len(completed),
    }
    for metric in SUMMARY_METRICS:
        values = np.asarray(
            [float(run_by_id[run_id][metric]) for run_id in completed],
            dtype=np.float64,
        )
        summary[metric] = (
            {
                "mean": float(values.mean()),
                "std_population": float(values.std(ddof=0)),
                "min": float(values.min()),
                "max": float(values.max()),
            }
            if values.size
            else None
        )
    mean_matrix = (
        np.mean(np.stack(normalized_matrices, axis=0), axis=0)
        if normalized_matrices
        else None
    )
    summary["confusion_matrix_mean_normalized"] = (
        mean_matrix.tolist() if mean_matrix is not None else None
    )
    return summary


def _write_mean_matrix(
    path: Path, matrix: np.ndarray, class_names: Sequence[str]
) -> None:
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(["true\\pred", *class_names])
        for name, row in zip(class_names, matrix, strict=True):
            writer.writerow([name, *[f"{value:.10f}" for value in row]])


def write_protocol_summary(
    output_dir: str | Path,
    runs: Sequence[dict[str, object]],
    *,
    requested: Sequence[Hashable],
    failed: Sequence[dict[str, object]],
    class_names: Sequence[str],
    id_field: str,
    requested_key: str,
    completed_key: str,
    failed_key: str,
    protocol: str,
    runs_filename: str,
    summary_filename: str,
    run_fields: Sequence[str],
) -> dict[str, object]:
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    summary = summarize_runs(
        runs,
        requested=requested,
        failed=failed,
        class_names=class_names,
        id_field=id_field,
        requested_key=requested_key,
        completed_key=completed_key,
        failed_key=failed_key,
        protocol=protocol,
    )
    run_by_id = {run[id_field]: run for run in runs}
    with (output / runs_filename).open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(run_fields))
        writer.writeheader()
        for run_id in summary[completed_key]:
            run = run_by_id[run_id]
            writer.writerow({field: run.get(field, "") for field in run_fields})

    matrix_value = summary["confusion_matrix_mean_normalized"]
    matrix_csv = output / "confusion_matrix_mean_normalized.csv"
    matrix_png = output / "confusion_matrix_mean_normalized.png"
    if matrix_value is None:
        matrix_csv.unlink(missing_ok=True)
        matrix_png.unlink(missing_ok=True)
        plot_written = False
    else:
        matrix = np.asarray(matrix_value, dtype=np.float64)
        _write_mean_matrix(matrix_csv, matrix, class_names)
        plot_written = try_write_confusion_plot(
            matrix_png,
            matrix,
            class_names,
            title="Mean row-normalized confusion matrix",
            value_format=".2f",
        )
    summary["confusion_matrix_mean_normalized_png_written"] = plot_written
    (output / summary_filename).write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    return summary


def write_in_domain_summary(
    output_dir: str | Path,
    runs: Sequence[dict[str, object]],
    *,
    requested_seeds: Sequence[int],
    failed_seeds: Sequence[dict[str, object]],
    class_names: Sequence[str],
) -> dict[str, object]:
    return write_protocol_summary(
        output_dir,
        runs,
        requested=requested_seeds,
        failed=failed_seeds,
        class_names=class_names,
        id_field="seed",
        requested_key="requested_seeds",
        completed_key="completed_seeds",
        failed_key="failed_seeds",
        protocol="person_action_stratified_in_domain",
        runs_filename="in_domain_runs.csv",
        summary_filename="in_domain_summary.json",
        run_fields=(
            "seed",
            "arch",
            *SUMMARY_METRICS,
            "test_loss",
            "best_epoch",
            "best_val_macro_f1",
        ),
    )


def write_loso_summary(
    output_dir: str | Path,
    runs: Sequence[dict[str, object]],
    *,
    requested_people: Sequence[str],
    failed_folds: Sequence[dict[str, object]],
    class_names: Sequence[str],
) -> dict[str, object]:
    return write_protocol_summary(
        output_dir,
        runs,
        requested=requested_people,
        failed=failed_folds,
        class_names=class_names,
        id_field="test_person",
        requested_key="requested_test_people",
        completed_key="completed_test_people",
        failed_key="failed_folds",
        protocol="subject_wise_loso",
        runs_filename="loso_folds.csv",
        summary_filename="loso_summary.json",
        run_fields=(
            "arch",
            "test_person",
            "val_person",
            *SUMMARY_METRICS,
            "test_loss",
            "best_epoch",
            "best_val_macro_f1",
        ),
    )
