from __future__ import annotations

import argparse
import json
from pathlib import Path

from axhome_csi.cli import add_training_arguments, configs_from_args
from axhome_csi.data import load_manifest
from axhome_csi.splits import rotating_loso_pairs


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train one CRISP-AX subject-wise LOSO fold (CSI-only)"
    )
    add_training_arguments(parser)
    parser.add_argument("--test-person", required=True, help="Test subject, e.g. P01")
    parser.add_argument(
        "--val-person",
        default=None,
        help="Validation subject; defaults to the subject after the test subject",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    from axhome_csi.training import train_subject_fold

    samples = load_manifest(args.dataset_root)
    pair_map = dict(rotating_loso_pairs(samples))
    if args.test_person not in pair_map:
        raise ValueError(f"unknown test person: {args.test_person}")
    val_person = args.val_person or pair_map[args.test_person]
    cache_config, training_config = configs_from_args(args)
    metrics = train_subject_fold(
        args.dataset_root,
        test_person=args.test_person,
        val_person=val_person,
        output_dir=Path(args.output_dir),
        cache_root=args.cache_dir,
        cache_config=cache_config,
        training_config=training_config,
    )
    print(json.dumps(metrics, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
