from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path
from typing import Sequence

import numpy as np

from axhome_video.data import VideoSampleRecord, load_video_manifest
from axhome_video.splits import human_samples


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="检查 AXHome-MM-v1 视频清单和 Video-only 基线样本分布"
    )
    parser.add_argument("--dataset-root", required=True)
    parser.add_argument("--output", default=None, help="可选 JSON 输出路径")
    parser.add_argument(
        "--validate-paths",
        action="store_true",
        help="若任一清单视频文件缺失则返回非零退出码",
    )
    return parser.parse_args()


def _sorted_counts(values: Sequence[str]) -> dict[str, int]:
    return dict(sorted(Counter(values).items()))


def _numeric_summary(values: Sequence[float | int]) -> dict[str, float]:
    array = np.asarray(values, dtype=np.float64)
    if array.size == 0 or not np.isfinite(array).all():
        raise ValueError("numeric profile values must be non-empty and finite")
    percentiles = np.percentile(array, [25, 50, 75])
    return {
        "min": float(array.min()),
        "q1": float(percentiles[0]),
        "median": float(percentiles[1]),
        "q3": float(percentiles[2]),
        "max": float(array.max()),
        "mean": float(array.mean()),
    }


def build_profile(
    dataset_root: str | Path,
) -> tuple[dict[str, object], list[VideoSampleRecord]]:
    root = Path(dataset_root).resolve()
    records = load_video_manifest(root, validate_paths=False)
    humans = human_samples(records)
    human_ids = {record.sample_id for record in humans}
    background = [
        record for record in records if record.sample_id not in human_ids
    ]
    missing = [record for record in records if not record.video_path.is_file()]
    profile: dict[str, object] = {
        "dataset_root": str(root),
        "manifest_path": str(root / "manifests" / "archive_index.csv"),
        "selection": "status=ok and sync_quality_flags=ok",
        "total_samples": len(records),
        "human_action_samples": len(humans),
        "background_samples": len(background),
        "action_counts": _sorted_counts(
            [record.action_id for record in records]
        ),
        "human_action_counts": _sorted_counts(
            [record.action_id for record in humans]
        ),
        "person_counts": _sorted_counts(
            [record.person_id for record in records]
        ),
        "environment_counts": _sorted_counts(
            [record.environment_id for record in records]
        ),
        "video_frames_written": _numeric_summary(
            [record.video_frames_written for record in records]
        ),
        "video_fps": _numeric_summary(
            [record.video_fps for record in records]
        ),
        "missing_video_files": len(missing),
        "missing_video_examples": [
            str(record.video_path) for record in missing[:10]
        ],
    }
    return profile, missing


def main() -> None:
    args = parse_args()
    profile, missing = build_profile(args.dataset_root)
    rendered = json.dumps(profile, ensure_ascii=False, indent=2) + "\n"
    print(rendered, end="", flush=True)
    if args.output:
        output = Path(args.output)
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(rendered, encoding="utf-8")
    if args.validate_paths and missing:
        raise FileNotFoundError(
            f"{len(missing)} video files listed by the manifest are missing"
        )


if __name__ == "__main__":
    main()
