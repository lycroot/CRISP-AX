from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Callable

import numpy as np

from .data import VideoSampleRecord

ClipDecoder = Callable[[VideoSampleRecord, "VideoCacheConfig"], np.ndarray]


@dataclass(frozen=True)
class VideoCacheConfig:
    frames: int = 16
    height: int = 128
    width: int = 171
    color_space: str = "RGB"
    implementation_version: int = 1

    def __post_init__(self) -> None:
        if min(self.frames, self.height, self.width) <= 0:
            raise ValueError("cache frames, height, and width must be positive")
        if self.color_space != "RGB":
            raise ValueError("only RGB video caches are supported")
        if self.implementation_version <= 0:
            raise ValueError("implementation_version must be positive")

    def as_dict(self) -> dict[str, object]:
        return asdict(self)

    def fingerprint(self) -> str:
        payload = json.dumps(
            self.as_dict(), sort_keys=True, separators=(",", ":")
        )
        return hashlib.sha256(payload.encode("utf-8")).hexdigest()[:16]


def uniform_frame_indices(total_frames: int, target_frames: int) -> np.ndarray:
    if target_frames <= 0:
        raise ValueError("target_frames must be positive")
    if total_frames < target_frames:
        raise ValueError(
            f"video has {total_frames} frames but {target_frames} are required"
        )
    indices = np.rint(
        np.linspace(0, total_frames - 1, target_frames)
    ).astype(np.int64)
    if np.unique(indices).size != target_frames:
        raise ValueError("uniform frame selection produced duplicate indices")
    return indices


def _safe_component(value: str, *, field: str) -> str:
    if not value or Path(value).name != value or value in {".", ".."}:
        raise ValueError(f"unsafe {field}: {value!r}")
    return value


def cache_path(
    cache_root: str | Path,
    record: VideoSampleRecord,
    config: VideoCacheConfig,
) -> Path:
    action = _safe_component(record.action_id, field="action_id")
    sample_id = _safe_component(record.sample_id, field="sample_id")
    return (
        Path(cache_root)
        / config.fingerprint()
        / action
        / f"{sample_id}.npy"
    )


def validate_cached_clip(
    clip: np.ndarray,
    config: VideoCacheConfig,
    *,
    sample_id: str = "clip",
) -> np.ndarray:
    array = np.asarray(clip)
    expected_shape = (config.frames, config.height, config.width, 3)
    if array.shape != expected_shape:
        raise ValueError(
            f"{sample_id}: cache shape {array.shape} does not match "
            f"{expected_shape}"
        )
    if array.dtype != np.uint8:
        raise ValueError(
            f"{sample_id}: cache dtype {array.dtype} must be uint8"
        )
    return array


def decode_selected_frames(
    record: VideoSampleRecord, config: VideoCacheConfig
) -> np.ndarray:
    """Decode a video sequentially and retain deterministic uniform frames."""
    if record.video_frames_written < config.frames:
        raise ValueError(
            f"{record.sample_id}: manifest declares fewer than "
            f"{config.frames} frames"
        )
    try:
        import av
        from PIL import Image
    except ImportError as exc:
        raise RuntimeError(
            "video caching requires PyAV and Pillow"
        ) from exc

    indices = uniform_frame_indices(
        record.video_frames_written, config.frames
    )
    index_set = {int(index) for index in indices}
    selected: dict[int, np.ndarray] = {}
    decoded_count = 0
    with av.open(str(record.video_path)) as container:
        if not container.streams.video:
            raise ValueError(f"{record.sample_id}: video stream is missing")
        for frame_index, frame in enumerate(container.decode(video=0)):
            if frame_index in index_set:
                image = frame.to_image().convert("RGB")
                image = image.resize(
                    (config.width, config.height),
                    resample=Image.Resampling.BILINEAR,
                )
                selected[frame_index] = np.asarray(
                    image, dtype=np.uint8
                ).copy()
            decoded_count += 1
    if decoded_count != record.video_frames_written:
        raise ValueError(
            f"{record.sample_id}: decoded {decoded_count} frames, manifest "
            f"declares {record.video_frames_written}"
        )
    missing = index_set - set(selected)
    if missing:
        raise ValueError(
            f"{record.sample_id}: failed to decode selected frame indices "
            + ", ".join(str(index) for index in sorted(missing))
        )
    clip = np.stack([selected[int(index)] for index in indices], axis=0)
    return validate_cached_clip(clip, config, sample_id=record.sample_id)


def load_cached_clip(
    cache_root: str | Path,
    record: VideoSampleRecord,
    config: VideoCacheConfig,
) -> np.ndarray:
    path = cache_path(cache_root, record, config)
    if not path.is_file():
        raise FileNotFoundError(f"video cache not found: {path}")
    try:
        clip = np.load(path, allow_pickle=False)
    except Exception as exc:
        raise ValueError(
            f"{record.sample_id}: failed to read video cache: {path}"
        ) from exc
    return validate_cached_clip(clip, config, sample_id=record.sample_id)


def ensure_cache_entry(
    record: VideoSampleRecord,
    cache_root: str | Path,
    config: VideoCacheConfig,
    *,
    decoder: ClipDecoder | None = None,
    rebuild: bool = False,
) -> np.ndarray:
    destination = cache_path(cache_root, record, config)
    if destination.is_file() and not rebuild:
        return load_cached_clip(cache_root, record, config)
    decode = decoder or decode_selected_frames
    clip = validate_cached_clip(
        decode(record, config), config, sample_id=record.sample_id
    )
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_suffix(".tmp.npy")
    try:
        np.save(temporary, clip, allow_pickle=False)
        temporary.replace(destination)
    finally:
        temporary.unlink(missing_ok=True)
    return clip
