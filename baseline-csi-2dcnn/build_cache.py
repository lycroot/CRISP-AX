from __future__ import annotations

import argparse
import json
from concurrent.futures import ProcessPoolExecutor, as_completed
from datetime import datetime, timezone
from pathlib import Path

from axhome_csi.cache import CacheConfig, prepare_sample_cache
from axhome_csi.data import SampleRecord, load_manifest
from axhome_csi.splits import human_samples


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build the preprocessing cache for the CRISP-AX CSI-only baselines"
    )
    parser.add_argument("--dataset-root", required=True)
    parser.add_argument("--cache-dir", required=True)
    parser.add_argument("--target-packets", type=int, default=256)
    parser.add_argument("--target-subcarriers", type=int, default=128)
    parser.add_argument("--low-energy-ratio", type=float, default=0.05)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--limit", type=int, default=None, help="Process only the first N samples (for checking)")
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def _job(
    sample: SampleRecord,
    cache_dir: str,
    config: CacheConfig,
    overwrite: bool,
) -> str:
    return str(
        prepare_sample_cache(
            sample, cache_dir, config, overwrite=overwrite
        )
    )


def main() -> None:
    args = parse_args()
    if args.workers < 0:
        raise ValueError("workers must be non-negative")
    config = CacheConfig(
        target_packets=args.target_packets,
        target_subcarriers=args.target_subcarriers,
        low_energy_ratio=args.low_energy_ratio,
    )
    samples = list(human_samples(load_manifest(args.dataset_root, validate_paths=True)))
    samples.sort(key=lambda sample: sample.sample_id)
    if args.limit is not None:
        if args.limit <= 0:
            raise ValueError("limit must be positive")
        samples = samples[: args.limit]
    cache_dir = str(Path(args.cache_dir).resolve())
    failures: list[dict[str, str]] = []
    completed = 0

    def record_result(sample: SampleRecord, error: Exception | None) -> None:
        nonlocal completed
        if error is None:
            completed += 1
            if completed % 100 == 0 or completed == len(samples):
                print(f"cache_progress {completed}/{len(samples)}", flush=True)
        else:
            failures.append({"sample_id": sample.sample_id, "error": str(error)})
            print(f"cache_error sample={sample.sample_id} error={error}", flush=True)

    if args.workers in (0, 1):
        for sample in samples:
            try:
                _job(sample, cache_dir, config, args.overwrite)
            except Exception as exc:  # continue to produce a complete error report
                record_result(sample, exc)
            else:
                record_result(sample, None)
    else:
        with ProcessPoolExecutor(max_workers=args.workers) as executor:
            future_to_sample = {
                executor.submit(
                    _job, sample, cache_dir, config, args.overwrite
                ): sample
                for sample in samples
            }
            for future in as_completed(future_to_sample):
                sample = future_to_sample[future]
                try:
                    future.result()
                except Exception as exc:
                    record_result(sample, exc)
                else:
                    record_result(sample, None)

    report = {
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "dataset_root": str(Path(args.dataset_root).resolve()),
        "cache_root": cache_dir,
        "config": config.as_dict(),
        "requested_samples": len(samples),
        "completed_samples": completed,
        "failed_samples": failures,
    }
    report_path = Path(cache_dir) / config.signature / "cache_build_report.json"
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(report_path.resolve())
    if failures:
        raise SystemExit(1)


if __name__ == "__main__":
    main()

