from __future__ import annotations

from pathlib import Path
from typing import Sequence

import numpy as np
import torch
from torch.utils.data import Dataset

from .cache import VideoCacheConfig, load_cached_clip, validate_cached_clip
from .data import VideoSampleRecord

KINETICS_MEAN = (0.43216, 0.394666, 0.37645)
KINETICS_STD = (0.22803, 0.22145, 0.216989)
R3D_CROP_SIZE = 112


def transform_cached_clip(
    clip: np.ndarray,
    *,
    training: bool,
    crop_size: int = R3D_CROP_SIZE,
) -> torch.Tensor:
    """Convert cached THWC uint8 frames into normalized CTHW tensors."""
    array = np.asarray(clip)
    if array.ndim != 4 or array.shape[-1] != 3:
        raise ValueError("cached clip must have shape (time, height, width, 3)")
    if array.dtype != np.uint8:
        raise ValueError("cached clip must use uint8 RGB values")
    _, height, width, _ = array.shape
    if crop_size <= 0 or height < crop_size or width < crop_size:
        raise ValueError(
            f"crop size {crop_size} does not fit cached clip {array.shape}"
        )
    tensor = torch.from_numpy(array.copy()).permute(3, 0, 1, 2)
    if training:
        top = int(torch.randint(0, height - crop_size + 1, (1,)).item())
        left = int(torch.randint(0, width - crop_size + 1, (1,)).item())
        tensor = tensor[:, :, top : top + crop_size, left : left + crop_size]
        if bool(torch.rand(()) < 0.5):
            tensor = tensor.flip(-1)
    else:
        top = (height - crop_size) // 2
        left = (width - crop_size) // 2
        tensor = tensor[:, :, top : top + crop_size, left : left + crop_size]
    tensor = tensor.to(dtype=torch.float32).div_(255.0)
    mean = torch.tensor(KINETICS_MEAN, dtype=tensor.dtype).view(3, 1, 1, 1)
    std = torch.tensor(KINETICS_STD, dtype=tensor.dtype).view(3, 1, 1, 1)
    tensor = tensor.sub_(mean).div_(std)
    if not torch.isfinite(tensor).all():
        raise ValueError("video transform produced non-finite values")
    return tensor.contiguous()


class CachedVideoDataset(Dataset):
    def __init__(
        self,
        samples: Sequence[VideoSampleRecord],
        *,
        cache_root: str | Path,
        cache_config: VideoCacheConfig,
        action_to_index: dict[str, int],
        training: bool,
    ) -> None:
        if not samples:
            raise ValueError("video dataset samples must not be empty")
        self.samples = tuple(samples)
        self.cache_root = Path(cache_root)
        self.cache_config = cache_config
        self.action_to_index = dict(action_to_index)
        self.training = training
        unknown = sorted(
            {sample.action_id for sample in self.samples}
            - set(self.action_to_index)
        )
        if unknown:
            raise ValueError("unknown video actions: " + ", ".join(unknown))

    def __len__(self) -> int:
        return len(self.samples)

    @property
    def labels(self) -> np.ndarray:
        return np.asarray(
            [self.action_to_index[sample.action_id] for sample in self.samples],
            dtype=np.int64,
        )

    def __getitem__(self, index: int) -> tuple[torch.Tensor, int, str]:
        sample = self.samples[index]
        clip = load_cached_clip(
            self.cache_root, sample, self.cache_config
        )
        validate_cached_clip(
            clip, self.cache_config, sample_id=sample.sample_id
        )
        tensor = transform_cached_clip(clip, training=self.training)
        label = self.action_to_index[sample.action_id]
        return tensor, label, sample.sample_id
