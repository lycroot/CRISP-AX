from __future__ import annotations

import argparse
import json
from dataclasses import replace
from pathlib import Path

from axhome_video.aggregate import (
    split_summary,
    validate_reference_assignments,
    write_in_domain_summary,
    write_split_assignments,
)
from axhome_video.cli import add_training_arguments, configs_from_args
from axhome_video.data import load_video_manifest
from axhome_video.splits import HUMAN_ACTIONS, make_in_domain_split

DEFAULT_SEEDS = (2026, 2027, 2028, 2029, 2030)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run the CRISP-AX Video-only in-domain protocol for several seeds"
    )
    add_training_arguments(parser)
    parser.add_argument(
        "--seeds",
        nargs="+",
        type=int,
        default=list(DEFAULT_SEEDS),
        help="Random seeds to run (default: 2026 2027 2028 2029 2030)",
    )
    parser.add_argument(
        "--reference-assignment-root",
        required=True,
        help="Root of the CSI in-domain results; per-sample split verification is mandatory for batch runs",
    )
    return parser.parse_args()


def _error_record(seed: int, error: Exception) -> dict[str, object]:
    return {
        "id": seed,
        "seed": seed,
        "error": f"{type(error).__name__}: {error}",
    }


def main() -> None:
    args = parse_args()
    from axhome_video.training import train_sample_split

    seeds = list(args.seeds)
    if not seeds or len(seeds) != len(set(seeds)):
        raise ValueError("--seeds must be non-empty and unique")

    output = Path(args.output_dir)
    output.mkdir(parents=True, exist_ok=True)
    samples = load_video_manifest(args.dataset_root, validate_paths=True)
    cache_config, base_training_config = configs_from_args(args)
    completed: list[dict[str, object]] = []
    failed: list[dict[str, object]] = []
    summary: dict[str, object] = {}

    for seed in seeds:
        run_output = output / f"seed_{seed}"
        reference_path: Path | None = None
        try:
            split = make_in_domain_split(samples, seed=seed)
            assignment_path = run_output / "split_assignments.csv"
            write_split_assignments(assignment_path, split)
            if args.reference_assignment_root:
                reference_path = (
                    Path(args.reference_assignment_root)
                    / f"seed_{seed}"
                    / "split_assignments.csv"
                )
                validate_reference_assignments(
                    assignment_path, reference_path, seed=seed
                )
            metrics = train_sample_split(
                args.dataset_root,
                train_samples=split.train,
                val_samples=split.val,
                test_samples=split.test,
                output_dir=run_output,
                run_metadata={
                    "protocol": "person_action_stratified_in_domain",
                    "split": split_summary(split, seed=seed),
                    "reference_assignment_path": (
                        str(reference_path.resolve())
                        if reference_path is not None
                        else None
                    ),
                },
                metric_metadata={
                    "seed": seed,
                    "protocol": "person_action_stratified_in_domain",
                },
                cache_root=args.cache_dir,
                cache_config=cache_config,
                training_config=replace(base_training_config, seed=seed),
            )
            completed.append(metrics)
            print(f"seed_complete {seed}", flush=True)
        except Exception as error:
            failed.append(_error_record(seed, error))
            print(
                f"seed_failed {seed}: {type(error).__name__}: {error}",
                flush=True,
            )
        summary = write_in_domain_summary(
            output,
            completed,
            requested_seeds=seeds,
            failed_seeds=failed,
            class_names=HUMAN_ACTIONS,
        )

    print(json.dumps(summary, ensure_ascii=False, indent=2), flush=True)
    if failed:
        raise RuntimeError(
            f"{len(failed)} of {len(seeds)} in-domain seeds failed; "
            f"see {output / 'in_domain_summary.json'}"
        )


if __name__ == "__main__":
    main()
