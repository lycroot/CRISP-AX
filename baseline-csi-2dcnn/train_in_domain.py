from __future__ import annotations

import argparse
import json
from pathlib import Path

from axhome_csi.cli import add_training_arguments, configs_from_args
from axhome_csi.data import load_manifest
from axhome_csi.in_domain import split_summary, write_split_assignments
from axhome_csi.splits import make_in_domain_split


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "训练一个按被试与动作分层的 AXHome-MM-v1 in-domain CSI 2D CNN 实验"
        )
    )
    add_training_arguments(parser)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    from axhome_csi.training import train_sample_split

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    samples = load_manifest(args.dataset_root, validate_paths=True)
    split = make_in_domain_split(samples, seed=args.seed)
    write_split_assignments(output_dir / "split_assignments.csv", split)
    cache_config, training_config = configs_from_args(args)
    metrics = train_sample_split(
        args.dataset_root,
        train_samples=split.train,
        val_samples=split.val,
        test_samples=split.test,
        output_dir=output_dir,
        run_metadata={"in_domain_split": split_summary(split)},
        metric_metadata={
            "seed": args.seed,
            "protocol": "person_action_stratified_in_domain",
        },
        cache_root=args.cache_dir,
        cache_config=cache_config,
        training_config=training_config,
    )
    print(json.dumps(metrics, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
