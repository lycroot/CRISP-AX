from __future__ import annotations

import argparse
import json
from dataclasses import replace
from pathlib import Path

from axhome_csi.cli import add_training_arguments, configs_from_args
from axhome_csi.data import load_manifest
from axhome_csi.in_domain import (
    split_summary,
    write_in_domain_summary,
    write_split_assignments,
)
from axhome_csi.splits import HUMAN_ACTIONS, make_in_domain_split


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Run the CRISP-AX in-domain CSI experiment (subject x action stratified) for several seeds"
        )
    )
    add_training_arguments(parser)
    parser.add_argument(
        "--seeds",
        type=int,
        nargs="+",
        default=[2026, 2027, 2028, 2029, 2030],
        help="Random seeds to run in order (default: 2026 2027 2028 2029 2030)",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    from axhome_csi.training import train_sample_split

    requested_seeds = [int(seed) for seed in args.seeds]
    if len(requested_seeds) != len(set(requested_seeds)):
        raise ValueError("--seeds must not contain duplicates")
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    samples = load_manifest(args.dataset_root, validate_paths=True)
    cache_config, base_training_config = configs_from_args(args)
    completed_runs: list[dict[str, object]] = []
    failed_seeds: list[dict[str, object]] = []

    for seed in requested_seeds:
        seed_output = output_dir / f"seed_{seed}"
        print(f"in_domain_seed_start {seed}", flush=True)
        try:
            split = make_in_domain_split(samples, seed=seed)
            write_split_assignments(
                seed_output / "split_assignments.csv", split
            )
            metrics = train_sample_split(
                args.dataset_root,
                train_samples=split.train,
                val_samples=split.val,
                test_samples=split.test,
                output_dir=seed_output,
                run_metadata={"in_domain_split": split_summary(split)},
                metric_metadata={
                    "seed": seed,
                    "protocol": "person_action_stratified_in_domain",
                },
                cache_root=args.cache_dir,
                cache_config=cache_config,
                training_config=replace(base_training_config, seed=seed),
            )
            completed_runs.append(metrics)
            print(f"in_domain_seed_complete {seed}", flush=True)
        except Exception as exc:
            failure = {
                "seed": seed,
                "error": f"{type(exc).__name__}: {exc}",
            }
            failed_seeds.append(failure)
            print(
                f"in_domain_seed_failed {seed} {failure['error']}",
                flush=True,
            )
        write_in_domain_summary(
            output_dir,
            completed_runs,
            requested_seeds=requested_seeds,
            failed_seeds=failed_seeds,
            class_names=HUMAN_ACTIONS,
        )

    summary = write_in_domain_summary(
        output_dir,
        completed_runs,
        requested_seeds=requested_seeds,
        failed_seeds=failed_seeds,
        class_names=HUMAN_ACTIONS,
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2), flush=True)
    if failed_seeds:
        failed = ", ".join(str(item["seed"]) for item in failed_seeds)
        raise RuntimeError(f"in-domain seeds failed: {failed}")


if __name__ == "__main__":
    main()
