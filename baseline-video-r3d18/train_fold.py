from __future__ import annotations

import argparse
import json
from pathlib import Path

from axhome_video.aggregate import split_summary, write_split_assignments
from axhome_video.cli import add_training_arguments, configs_from_args
from axhome_video.data import load_video_manifest
from axhome_video.splits import make_subject_fold, rotating_loso_pairs


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train one CRISP-AX Video-only LOSO fold"
    )
    add_training_arguments(parser)
    parser.add_argument("--test-person", required=True)
    parser.add_argument("--val-person", default=None)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    from axhome_video.training import train_sample_split

    samples = load_video_manifest(args.dataset_root, validate_paths=True)
    pair_map = dict(rotating_loso_pairs(samples))
    if args.test_person not in pair_map:
        raise ValueError(f"unknown test person: {args.test_person}")
    val_person = args.val_person or pair_map[args.test_person]
    fold = make_subject_fold(
        samples, test_person=args.test_person, val_person=val_person
    )
    output = Path(args.output_dir)
    output.mkdir(parents=True, exist_ok=True)
    write_split_assignments(output / "split_assignments.csv", fold, seed=args.seed)
    cache_config, training_config = configs_from_args(args)
    metrics = train_sample_split(
        args.dataset_root,
        train_samples=fold.train,
        val_samples=fold.val,
        test_samples=fold.test,
        output_dir=output,
        run_metadata={
            "protocol": "subject_wise_loso",
            "fold": {
                **split_summary(fold, seed=args.seed),
                "test_person": args.test_person,
                "val_person": val_person,
                "train_people": list(fold.train_people),
            },
        },
        metric_metadata={
            "seed": args.seed,
            "protocol": "subject_wise_loso",
            "test_person": args.test_person,
            "val_person": val_person,
        },
        cache_root=args.cache_dir,
        cache_config=cache_config,
        training_config=training_config,
    )
    print(json.dumps(metrics, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()
