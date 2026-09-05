from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Sequence

import numpy as np


def balanced_class_weights(labels: np.ndarray, num_classes: int) -> np.ndarray:
    labels = np.asarray(labels, dtype=np.int64)
    if labels.ndim != 1 or labels.size == 0:
        raise ValueError("labels must be a non-empty one-dimensional array")
    if num_classes <= 0 or labels.min() < 0 or labels.max() >= num_classes:
        raise ValueError("labels are outside the configured class range")
    counts = np.bincount(labels, minlength=num_classes)
    if np.any(counts == 0):
        missing = np.flatnonzero(counts == 0).tolist()
        raise ValueError(f"training split is missing class indices: {missing}")
    weights = labels.size / (num_classes * counts.astype(np.float64))
    return weights.astype(np.float32)


def confusion_matrix(
    y_true: np.ndarray, y_pred: np.ndarray, num_classes: int
) -> np.ndarray:
    true = np.asarray(y_true, dtype=np.int64)
    pred = np.asarray(y_pred, dtype=np.int64)
    if true.ndim != 1 or pred.ndim != 1 or true.size != pred.size:
        raise ValueError("y_true and y_pred must be one-dimensional and equal length")
    if true.size == 0:
        raise ValueError("evaluation arrays must not be empty")
    if (
        true.min() < 0
        or pred.min() < 0
        or true.max() >= num_classes
        or pred.max() >= num_classes
    ):
        raise ValueError("evaluation labels are outside the class range")
    matrix = np.zeros((num_classes, num_classes), dtype=np.int64)
    np.add.at(matrix, (true, pred), 1)
    return matrix


def classification_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    *,
    class_names: Sequence[str],
) -> dict[str, object]:
    matrix = confusion_matrix(y_true, y_pred, len(class_names))
    support = matrix.sum(axis=1)
    predicted = matrix.sum(axis=0)
    true_positive = np.diag(matrix)
    precision = np.divide(
        true_positive,
        predicted,
        out=np.zeros_like(true_positive, dtype=np.float64),
        where=predicted > 0,
    )
    recall = np.divide(
        true_positive,
        support,
        out=np.zeros_like(true_positive, dtype=np.float64),
        where=support > 0,
    )
    f1 = np.divide(
        2 * precision * recall,
        precision + recall,
        out=np.zeros_like(precision),
        where=(precision + recall) > 0,
    )
    total = int(support.sum())
    present = support > 0
    accuracy = float(true_positive.sum() / total)
    balanced_accuracy = float(recall[present].mean())
    macro_f1 = float(f1.mean())
    weighted_f1 = float(np.sum(f1 * support) / total)
    per_class = [
        {
            "class_index": index,
            "class_name": class_name,
            "precision": float(precision[index]),
            "recall": float(recall[index]),
            "f1": float(f1[index]),
            "support": int(support[index]),
        }
        for index, class_name in enumerate(class_names)
    ]
    return {
        "accuracy": accuracy,
        "balanced_accuracy": balanced_accuracy,
        "macro_f1": macro_f1,
        "weighted_f1": weighted_f1,
        "num_samples": total,
        "per_class": per_class,
        "confusion_matrix": matrix.tolist(),
    }


def _write_confusion_csv(
    path: Path, matrix: np.ndarray, class_names: Sequence[str]
) -> None:
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(["true\\pred", *class_names])
        for name, row in zip(class_names, matrix, strict=True):
            writer.writerow([name, *row.tolist()])


def try_write_confusion_plot(
    path: str | Path,
    values: np.ndarray,
    class_names: Sequence[str],
    *,
    title: str,
    value_format: str,
) -> bool:
    matrix = np.asarray(values)
    expected_shape = (len(class_names), len(class_names))
    if matrix.shape != expected_shape:
        raise ValueError(
            f"confusion values must have shape {expected_shape}, got {matrix.shape}"
        )
    if not np.isfinite(matrix).all():
        raise ValueError("confusion values contain non-finite entries")
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        return False

    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    size = max(8.0, len(class_names) * 0.72)
    figure, axis = plt.subplots(figsize=(size, size))
    image = axis.imshow(matrix, cmap="Blues", aspect="auto")
    figure.colorbar(image, ax=axis, fraction=0.046, pad=0.04)
    axis.set_xticks(
        range(len(class_names)), labels=class_names, rotation=55, ha="right"
    )
    axis.set_yticks(range(len(class_names)), labels=class_names)
    axis.set_xlabel("Predicted label")
    axis.set_ylabel("True label")
    axis.set_title(title)
    if len(class_names) <= 15:
        threshold = float(matrix.max()) * 0.55 if matrix.size else 0
        for row in range(matrix.shape[0]):
            for column in range(matrix.shape[1]):
                axis.text(
                    column,
                    row,
                    format(matrix[row, column], value_format),
                    ha="center",
                    va="center",
                    fontsize=7,
                    color=(
                        "white" if matrix[row, column] > threshold else "black"
                    ),
                )
    figure.tight_layout()
    figure.savefig(destination, dpi=200)
    plt.close(figure)
    return True


def write_evaluation(
    output_dir: str | Path,
    *,
    metrics: dict[str, object],
    sample_ids: Sequence[str],
    y_true: np.ndarray,
    y_pred: np.ndarray,
    class_names: Sequence[str],
) -> None:
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    true = np.asarray(y_true, dtype=np.int64)
    pred = np.asarray(y_pred, dtype=np.int64)
    if len(sample_ids) != true.size or true.size != pred.size:
        raise ValueError("sample_ids, y_true and y_pred must have equal length")
    matrix = np.asarray(metrics["confusion_matrix"], dtype=np.int64)
    row_sums = matrix.sum(axis=1, keepdims=True)
    normalized = np.divide(
        matrix,
        row_sums,
        out=np.zeros_like(matrix, dtype=np.float64),
        where=row_sums > 0,
    )
    counts_written = try_write_confusion_plot(
        output / "confusion_matrix_counts.png",
        matrix,
        class_names,
        title="Confusion matrix (counts)",
        value_format="d",
    )
    normalized_written = try_write_confusion_plot(
        output / "confusion_matrix_normalized.png",
        normalized,
        class_names,
        title="Confusion matrix (row normalized)",
        value_format=".2f",
    )
    serialized = dict(metrics)
    serialized["confusion_matrix_png_written"] = (
        counts_written and normalized_written
    )
    (output / "metrics.json").write_text(
        json.dumps(serialized, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    with (output / "classification_report.csv").open(
        "w", newline="", encoding="utf-8"
    ) as handle:
        fields = [
            "class_index",
            "class_name",
            "precision",
            "recall",
            "f1",
            "support",
        ]
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(metrics["per_class"])
    with (output / "predictions.csv").open(
        "w", newline="", encoding="utf-8"
    ) as handle:
        writer = csv.writer(handle)
        writer.writerow(
            ["sample_id", "true_index", "true_label", "pred_index", "pred_label"]
        )
        for sample_id, true_index, pred_index in zip(
            sample_ids, true, pred, strict=True
        ):
            writer.writerow(
                [
                    sample_id,
                    int(true_index),
                    class_names[int(true_index)],
                    int(pred_index),
                    class_names[int(pred_index)],
                ]
            )
    _write_confusion_csv(output / "confusion_matrix_counts.csv", matrix, class_names)
