from __future__ import annotations

import argparse
import json
from pathlib import Path

from axhome_video.aggregate import (
    split_summary,
    validate_reference_assignments,
    write_split_assignments,
)
from axhome_video.cli import add_training_arguments, configs_from_args
from axhome_video.data import load_video_manifest
from axhome_video.splits import make_in_domain_split


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train one CRISP-AX Video-only in-domain seed"
    )
    add_training_arguments(parser)
    parser.add_argument(
        "--reference-assignment-root",
        default=None,
        help="Optional: root of the CSI in-domain results, used to verify the split per sample",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    from axhome_video.training import train_sample_split

    output = Path(args.output_dir)
    output.mkdir(parents=True, exist_ok=True)
    samples = load_video_manifest(args.dataset_root, validate_paths=True)
    split = make_in_domain_split(samples, seed=args.seed)
    assignments = output / "split_assignments.csv"
    write_split_assignments(assignments, split)
    reference_path = None
    if args.reference_assignment_root:
        reference_path = (
            Path(args.reference_assignment_root)
            / f"seed_{args.seed}"
            / "split_assignments.csv"
        )
        validate_reference_assignments(
            assignments, reference_path, seed=args.seed
        )
    cache_config, training_config = configs_from_args(args)
    metrics = train_sample_split(
        args.dataset_root,
        train_samples=split.train,
        val_samples=split.val,
        test_samples=split.test,
        output_dir=output,
        run_metadata={
            "protocol": "person_action_stratified_in_domain",
            "split": split_summary(split, seed=args.seed),
            "reference_assignment_path": (
                str(reference_path.resolve()) if reference_path else None
            ),
        },
        metric_metadata={
            "seed": args.seed,
            "protocol": "person_action_stratified_in_domain",
        },
        cache_root=args.cache_dir,
        cache_config=cache_config,
        training_config=training_config,
    )
    print(json.dumps(metrics, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()
