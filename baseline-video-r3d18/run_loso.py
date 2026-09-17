from __future__ import annotations

import argparse
import json
from pathlib import Path

from axhome_video.aggregate import (
    split_summary,
    write_loso_summary,
    write_split_assignments,
)
from axhome_video.cli import add_training_arguments, configs_from_args
from axhome_video.data import load_video_manifest
from axhome_video.splits import (
    HUMAN_ACTIONS,
    make_subject_fold,
    rotating_loso_pairs,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run the CRISP-AX Video-only 10-fold LOSO protocol"
    )
    add_training_arguments(parser)
    parser.add_argument(
        "--test-people",
        nargs="+",
        default=None,
        help="Optional: run only these test subjects (default: all subjects)",
    )
    return parser.parse_args()


def _error_record(
    test_person: str, val_person: str, error: Exception
) -> dict[str, object]:
    return {
        "id": test_person,
        "test_person": test_person,
        "val_person": val_person,
        "error": f"{type(error).__name__}: {error}",
    }


def main() -> None:
    args = parse_args()
    from axhome_video.training import train_sample_split

    samples = load_video_manifest(args.dataset_root, validate_paths=True)
    all_pairs = rotating_loso_pairs(samples)
    pair_map = dict(all_pairs)
    requested_people = (
        list(args.test_people)
        if args.test_people is not None
        else [test_person for test_person, _ in all_pairs]
    )
    if not requested_people or len(requested_people) != len(set(requested_people)):
        raise ValueError("--test-people must be non-empty and unique")
    unknown = set(requested_people) - set(pair_map)
    if unknown:
        raise ValueError("unknown test people: " + ", ".join(sorted(unknown)))

    output = Path(args.output_dir)
    output.mkdir(parents=True, exist_ok=True)
    cache_config, training_config = configs_from_args(args)
    completed: list[dict[str, object]] = []
    failed: list[dict[str, object]] = []
    summary: dict[str, object] = {}

    for test_person in requested_people:
        val_person = pair_map[test_person]
        run_output = output / f"test_{test_person}_val_{val_person}"
        try:
            fold = make_subject_fold(
                samples,
                test_person=test_person,
                val_person=val_person,
            )
            write_split_assignments(
                run_output / "split_assignments.csv",
                fold,
                seed=args.seed,
            )
            metrics = train_sample_split(
                args.dataset_root,
                train_samples=fold.train,
                val_samples=fold.val,
                test_samples=fold.test,
                output_dir=run_output,
                run_metadata={
                    "protocol": "subject_wise_loso",
                    "fold": {
                        **split_summary(fold, seed=args.seed),
                        "test_person": test_person,
                        "val_person": val_person,
                        "train_people": list(fold.train_people),
                    },
                },
                metric_metadata={
                    "seed": args.seed,
                    "protocol": "subject_wise_loso",
                    "test_person": test_person,
                    "val_person": val_person,
                },
                cache_root=args.cache_dir,
                cache_config=cache_config,
                training_config=training_config,
            )
            completed.append(metrics)
            print(
                f"fold_complete test={test_person} val={val_person}",
                flush=True,
            )
        except Exception as error:
            failed.append(_error_record(test_person, val_person, error))
            print(
                f"fold_failed test={test_person} val={val_person}: "
                f"{type(error).__name__}: {error}",
                flush=True,
            )
        summary = write_loso_summary(
            output,
            completed,
            requested_people=requested_people,
            failed_folds=failed,
            class_names=HUMAN_ACTIONS,
        )

    print(json.dumps(summary, ensure_ascii=False, indent=2), flush=True)
    if failed:
        raise RuntimeError(
            f"{len(failed)} of {len(requested_people)} LOSO folds failed; "
            f"see {output / 'loso_summary.json'}"
        )


if __name__ == "__main__":
    main()
