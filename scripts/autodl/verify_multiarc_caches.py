#!/usr/bin/env python3
"""Read-only gate for the two frozen, architecture-independent data caches."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-root", required=True, type=Path)
    parser.add_argument("--csi-cache-root", required=True, type=Path)
    parser.add_argument("--video-cache-root", required=True, type=Path)
    args = parser.parse_args()
    project = Path(__file__).resolve().parents[2]
    sys.path.insert(0, str(project / "baseline-csi-2dcnn"))
    sys.path.insert(0, str(project / "baseline-video-r3d18"))
    import numpy as np
    from axhome_csi.cache import CacheConfig, cache_path as csi_path
    from axhome_csi.data import load_manifest
    from axhome_csi.splits import human_samples as csi_human_samples
    from axhome_video.cache import VideoCacheConfig, cache_path as video_path
    from axhome_video.data import load_video_manifest
    from axhome_video.splits import human_samples as video_human_samples

    csi_config = CacheConfig()
    video_config = VideoCacheConfig()
    csi_report = json.loads((args.csi_cache_root / csi_config.signature / "cache_build_report.json").read_text())
    video_report = json.loads((args.video_cache_root / "cache_build_report.json").read_text())
    if (csi_report.get("requested_samples") != 6960
            or csi_report.get("completed_samples") != 6960
            or csi_report.get("failed_samples") != []
            or csi_report.get("config") != csi_config.as_dict()):
        raise SystemExit("CSI cache report is incomplete or uses a different preprocessing config")
    if (video_report.get("requested") != 6960
            or video_report.get("completed") != 6960
            or video_report.get("failed") != []
            or video_report.get("complete") is not True
            or video_report.get("config") != video_config.as_dict()
            or video_report.get("config_fingerprint") != video_config.fingerprint()):
        raise SystemExit("Video cache report is incomplete or uses a different preprocessing config")
    csi_samples = tuple(csi_human_samples(load_manifest(args.dataset_root, validate_paths=True)))
    video_samples = tuple(video_human_samples(load_video_manifest(args.dataset_root, validate_paths=True)))
    if (len(csi_samples) != 6960 or len(video_samples) != 6960
            or {sample.sample_id for sample in csi_samples} != {sample.sample_id for sample in video_samples}):
        raise SystemExit("Expected 6960 paired human-action samples")
    for sample in csi_samples:
        array = np.load(csi_path(args.csi_cache_root, sample, csi_config), mmap_mode="r", allow_pickle=False)
        if array.shape != (2, 256, 128) or array.dtype != np.float32 or not np.isfinite(array).all():
            raise SystemExit(f"Invalid CSI cache: {sample.sample_id}")
        del array
    for sample in video_samples:
        array = np.load(video_path(args.video_cache_root, sample, video_config), mmap_mode="r", allow_pickle=False)
        if array.shape != (16, 128, 171, 3) or array.dtype != np.uint8:
            raise SystemExit(f"Invalid video cache: {sample.sample_id}")
        del array
    print("CACHE_ACCEPTANCE_PASSED csi=6960 video=6960", flush=True)


if __name__ == "__main__":
    main()
