from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np

from axhome_csi.cli import add_training_arguments, configs_from_args
from axhome_csi.data import load_manifest
from axhome_csi.splits import rotating_loso_pairs


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run the CRISP-AX 10-fold subject-wise LOSO CSI-only baseline"
    )
    add_training_arguments(parser)
    parser.add_argument(
        "--test-people",
        nargs="*",
        default=None,
        help="Run only these test subjects (default: all of P01-P10)",
    )
    return parser.parse_args()


def _write_summary(output_dir: Path, rows: list[dict[str, object]]) -> None:
    fields = [
        "arch",
        "test_person",
        "val_person",
        "accuracy",
        "balanced_accuracy",
        "macro_f1",
        "weighted_f1",
        "test_loss",
        "best_epoch",
        "best_val_macro_f1",
        "num_samples",
    ]
    with (output_dir / "loso_folds.csv").open(
        "w", newline="", encoding="utf-8"
    ) as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            writer.writerow({field: row.get(field, "") for field in fields})
    architectures = sorted(
        {str(row["arch"]) for row in rows if row.get("arch")}
    )
    aggregate: dict[str, object] = {
        "protocol": "subject_wise_loso",
        "architecture": architectures[0] if len(architectures) == 1 else None,
        "num_folds": len(rows),
        "folds": rows,
    }
    for metric in ("accuracy", "balanced_accuracy", "macro_f1", "weighted_f1"):
        values = np.asarray([float(row[metric]) for row in rows])
        aggregate[metric] = {
            "mean": float(values.mean()),
            "std_population": float(values.std(ddof=0)),
            "min": float(values.min()),
            "max": float(values.max()),
        }
    (output_dir / "loso_summary.json").write_text(
        json.dumps(aggregate, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )


def main() -> None:
    args = parse_args()
    from axhome_csi.training import train_subject_fold

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    samples = load_manifest(args.dataset_root)
    pairs = rotating_loso_pairs(samples)
    if args.test_people:
        requested = set(args.test_people)
        available = {test for test, _ in pairs}
        unknown = requested - available
        if unknown:
            raise ValueError("unknown test people: " + ", ".join(sorted(unknown)))
        pairs = [pair for pair in pairs if pair[0] in requested]
    cache_config, training_config = configs_from_args(args)
    rows: list[dict[str, object]] = []
    for test_person, val_person in pairs:
        fold_dir = output_dir / f"test_{test_person}_val_{val_person}"
        print(
            f"start_fold test={test_person} val={val_person} output={fold_dir}",
            flush=True,
        )
        metrics = train_subject_fold(
            args.dataset_root,
            test_person=test_person,
            val_person=val_person,
            output_dir=fold_dir,
            cache_root=args.cache_dir,
            cache_config=cache_config,
            training_config=training_config,
        )
        rows.append(metrics)
        _write_summary(output_dir, rows)
    print((output_dir / "loso_summary.json").resolve())


if __name__ == "__main__":
    main()
