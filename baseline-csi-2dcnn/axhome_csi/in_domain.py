from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Sequence

import numpy as np

from .metrics import try_write_confusion_plot
from .splits import InDomainSplit, validate_in_domain_split

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
RUN_SUMMARY_FIELDS = (
    "seed",
    "arch",
    *SUMMARY_METRICS,
    "test_loss",
    "best_epoch",
    "best_val_macro_f1",
)
RUN_FINITE_FIELDS = (*SUMMARY_METRICS, "test_loss", "best_val_macro_f1")


def split_summary(split: InDomainSplit) -> dict[str, object]:
    """Return serializable counts and actual ratios for one split."""
    validate_in_domain_split(split)
    counts = {
        "train": len(split.train),
        "validation": len(split.val),
        "test": len(split.test),
    }
    total = sum(counts.values())
    return {
        "protocol": "person_action_stratified_in_domain",
        "stratification": ["person_id", "action_id"],
        "target_ratios": {
            "train": 0.65,
            "validation": 0.175,
            "test": 0.175,
        },
        "seed": split.seed,
        "total_samples": total,
        "counts": counts,
        "actual_ratios": {
            name: count / total for name, count in counts.items()
        },
    }


def write_split_assignments(path: str | Path, split: InDomainSplit) -> None:
    """Write one auditable row for every assigned sample."""
    validate_in_domain_split(split)
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
                    "seed": split.seed,
                }
            )
    rows.sort(key=lambda row: str(row["sample_id"]))
    with destination.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=SPLIT_ASSIGNMENT_FIELDS)
        writer.writeheader()
        writer.writerows(rows)


def _validated_requested_seeds(requested_seeds: Sequence[int]) -> list[int]:
    requested = [int(seed) for seed in requested_seeds]
    if not requested:
        raise ValueError("requested_seeds must not be empty")
    if len(requested) != len(set(requested)):
        raise ValueError("requested_seeds contains duplicates")
    return requested


def summarize_in_domain_runs(
    runs: Sequence[dict[str, object]],
    *,
    requested_seeds: Sequence[int],
    failed_seeds: Sequence[dict[str, object]],
    class_names: Sequence[str],
) -> dict[str, object]:
    """Summarize completed runs with finite population statistics."""
    requested = _validated_requested_seeds(requested_seeds)
    classes = list(class_names)
    if not classes or len(classes) != len(set(classes)):
        raise ValueError("class_names must be non-empty and unique")

    run_by_seed: dict[int, dict[str, object]] = {}
    matrices: list[np.ndarray] = []
    for run in runs:
        seed = int(run["seed"])
        if seed not in requested:
            raise ValueError(f"completed seed was not requested: {seed}")
        if seed in run_by_seed:
            raise ValueError(f"duplicate completed seed: {seed}")
        run_by_seed[seed] = run
        for metric in RUN_FINITE_FIELDS:
            value = float(run[metric])
            if not np.isfinite(value):
                raise ValueError(f"non-finite {metric} in in-domain runs")

        matrix = np.asarray(run["confusion_matrix"], dtype=np.float64)
        expected_shape = (len(classes), len(classes))
        if matrix.shape != expected_shape:
            raise ValueError(
                f"seed {seed} confusion matrix must have shape "
                f"{expected_shape}, got {matrix.shape}"
            )
        if not np.isfinite(matrix).all():
            raise ValueError(
                f"non-finite confusion matrix in in-domain seed {seed}"
            )
        if np.any(matrix < 0):
            raise ValueError(
                f"negative confusion matrix count in in-domain seed {seed}"
            )
        row_sums = matrix.sum(axis=1, keepdims=True)
        if np.any(row_sums <= 0):
            raise ValueError(
                f"empty true class in confusion matrix for seed {seed}"
            )
        matrices.append(matrix / row_sums)

    failures = [dict(failure) for failure in failed_seeds]
    failed_seed_values: list[int] = []
    for failure in failures:
        if "seed" not in failure or "error" not in failure:
            raise ValueError("failed seed records require seed and error")
        seed = int(failure["seed"])
        failure["seed"] = seed
        failure["error"] = str(failure["error"])
        if seed not in requested:
            raise ValueError(f"failed seed was not requested: {seed}")
        if seed in run_by_seed:
            raise ValueError(f"seed is both completed and failed: {seed}")
        failed_seed_values.append(seed)
    if len(failed_seed_values) != len(set(failed_seed_values)):
        raise ValueError("failed_seeds contains duplicates")

    completed = [seed for seed in requested if seed in run_by_seed]
    architectures = sorted(
        {
            str(run_by_seed[seed]["arch"])
            for seed in completed
            if run_by_seed[seed].get("arch")
        }
    )
    summary: dict[str, object] = {
        "protocol": "person_action_stratified_in_domain",
        "architecture": architectures[0] if len(architectures) == 1 else None,
        "requested_seeds": requested,
        "completed_seeds": completed,
        "failed_seeds": failures,
        "complete": completed == requested and not failures,
        "class_names": classes,
        "num_completed": len(completed),
    }
    for metric in SUMMARY_METRICS:
        values = np.asarray(
            [float(run_by_seed[seed][metric]) for seed in completed],
            dtype=np.float64,
        )
        if not np.isfinite(values).all():
            raise ValueError(f"non-finite {metric} in in-domain runs")
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
        np.mean(np.stack(matrices, axis=0), axis=0)
        if matrices
        else None
    )
    summary["confusion_matrix_mean_normalized"] = (
        mean_matrix.tolist() if mean_matrix is not None else None
    )
    return summary


def _write_mean_confusion_csv(
    path: Path, matrix: np.ndarray, class_names: Sequence[str]
) -> None:
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(["true\\pred", *class_names])
        for name, row in zip(class_names, matrix, strict=True):
            writer.writerow([name, *[f"{value:.10f}" for value in row]])


def write_in_domain_summary(
    output_dir: str | Path,
    runs: Sequence[dict[str, object]],
    *,
    requested_seeds: Sequence[int],
    failed_seeds: Sequence[dict[str, object]],
    class_names: Sequence[str],
) -> dict[str, object]:
    """Write partial or complete multi-seed CSV/JSON summaries."""
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    summary = summarize_in_domain_runs(
        runs,
        requested_seeds=requested_seeds,
        failed_seeds=failed_seeds,
        class_names=class_names,
    )
    completed_by_seed = {int(run["seed"]): run for run in runs}
    with (output / "in_domain_runs.csv").open(
        "w", newline="", encoding="utf-8"
    ) as handle:
        writer = csv.DictWriter(handle, fieldnames=RUN_SUMMARY_FIELDS)
        writer.writeheader()
        for seed in summary["completed_seeds"]:
            run = completed_by_seed[int(seed)]
            writer.writerow(
                {field: run.get(field, "") for field in RUN_SUMMARY_FIELDS}
            )

    matrix_value = summary["confusion_matrix_mean_normalized"]
    plot_path = output / "confusion_matrix_mean_normalized.png"
    csv_path = output / "confusion_matrix_mean_normalized.csv"
    if matrix_value is None:
        plot_path.unlink(missing_ok=True)
        csv_path.unlink(missing_ok=True)
        plot_written = False
    else:
        matrix = np.asarray(matrix_value, dtype=np.float64)
        _write_mean_confusion_csv(csv_path, matrix, class_names)
        plot_written = try_write_confusion_plot(
            plot_path,
            matrix,
            class_names,
            title="Mean row-normalized confusion matrix",
            value_format=".2f",
        )
    summary["confusion_matrix_mean_normalized_png_written"] = plot_written
    (output / "in_domain_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    return summary
