from __future__ import annotations

import argparse
import json
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

from axhome_video.cache import VideoCacheConfig, cache_path, ensure_cache_entry
from axhome_video.data import VideoSampleRecord, load_video_manifest
from axhome_video.splits import human_samples


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build the CRISP-AX uniform 16-frame video cache"
    )
    parser.add_argument("--dataset-root", required=True)
    parser.add_argument("--cache-dir", required=True)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--frames", type=int, default=16)
    parser.add_argument("--height", type=int, default=128)
    parser.add_argument("--width", type=int, default=171)
    parser.add_argument("--rebuild", action="store_true")
    return parser.parse_args()


def _build_one(
    record: VideoSampleRecord,
    cache_root: str,
    config: VideoCacheConfig,
    rebuild: bool,
) -> dict[str, object]:
    if not record.video_path.is_file():
        raise FileNotFoundError(
            f"{record.sample_id}: video file not found: {record.video_path}"
        )
    clip = ensure_cache_entry(
        record, cache_root, config, rebuild=rebuild
    )
    path = cache_path(cache_root, record, config)
    return {
        "sample_id": record.sample_id,
        "cache_path": str(path),
        "bytes": int(path.stat().st_size),
        "shape": list(clip.shape),
    }


def main() -> None:
    args = parse_args()
    if args.workers <= 0:
        raise ValueError("--workers must be positive")
    if args.limit is not None and args.limit <= 0:
        raise ValueError("--limit must be positive")
    config = VideoCacheConfig(
        frames=args.frames, height=args.height, width=args.width
    )
    cache_root = Path(args.cache_dir)
    cache_root.mkdir(parents=True, exist_ok=True)
    try:
        samples = list(
            human_samples(
                load_video_manifest(args.dataset_root, validate_paths=False)
            )
        )
    except Exception as exc:
        report = {
            "config": config.as_dict(),
            "config_fingerprint": config.fingerprint(),
            "requested": 0,
            "completed": 0,
            "failed_count": 0,
            "failed": [],
            "cache_bytes": 0,
            "complete": False,
            "global_error": f"{type(exc).__name__}: {exc}",
        }
        (cache_root / "cache_build_report.json").write_text(
            json.dumps(report, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )
        print(json.dumps(report, ensure_ascii=False, indent=2), flush=True)
        raise RuntimeError("video cache manifest loading failed") from exc
    samples.sort(key=lambda sample: sample.sample_id)
    if args.limit is not None:
        samples = samples[: args.limit]
    completed: list[dict[str, object]] = []
    failed: list[dict[str, object]] = []

    if args.workers == 1:
        for index, sample in enumerate(samples, start=1):
            try:
                completed.append(
                    _build_one(sample, str(cache_root), config, args.rebuild)
                )
            except Exception as exc:
                failed.append(
                    {
                        "sample_id": sample.sample_id,
                        "error": f"{type(exc).__name__}: {exc}",
                    }
                )
            if index % 100 == 0 or index == len(samples):
                print(f"cache_progress {index}/{len(samples)}", flush=True)
    else:
        with ProcessPoolExecutor(max_workers=args.workers) as executor:
            futures = {
                executor.submit(
                    _build_one,
                    sample,
                    str(cache_root),
                    config,
                    args.rebuild,
                ): sample
                for sample in samples
            }
            for index, future in enumerate(as_completed(futures), start=1):
                sample = futures[future]
                try:
                    completed.append(future.result())
                except Exception as exc:
                    failed.append(
                        {
                            "sample_id": sample.sample_id,
                            "error": f"{type(exc).__name__}: {exc}",
                        }
                    )
                if index % 100 == 0 or index == len(samples):
                    print(f"cache_progress {index}/{len(samples)}", flush=True)

    completed.sort(key=lambda item: str(item["sample_id"]))
    failed.sort(key=lambda item: str(item["sample_id"]))
    report = {
        "config": config.as_dict(),
        "config_fingerprint": config.fingerprint(),
        "requested": len(samples),
        "completed": len(completed),
        "failed_count": len(failed),
        "failed": failed,
        "cache_bytes": sum(int(item["bytes"]) for item in completed),
        "complete": len(completed) == len(samples) and not failed,
        "global_error": None,
    }
    (cache_root / "cache_build_report.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(report, ensure_ascii=False, indent=2), flush=True)
    if failed:
        raise RuntimeError(f"video cache build failed for {len(failed)} samples")


if __name__ == "__main__":
    main()
