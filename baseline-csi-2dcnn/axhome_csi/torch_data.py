from __future__ import annotations

from pathlib import Path
from typing import Sequence

import numpy as np
import torch
from torch.utils.data import Dataset

from .cache import CacheConfig, cache_path
from .data import SampleRecord, preprocess_amplitude
from .feitcsi import read_feitcsi


class CSIAmplitudeDataset(Dataset[tuple[torch.Tensor, int, str]]):
    def __init__(
        self,
        samples: Sequence[SampleRecord],
        *,
        class_to_index: dict[str, int],
        cache_config: CacheConfig,
        cache_root: str | Path | None = None,
    ) -> None:
        if not samples:
            raise ValueError("dataset split must not be empty")
        unknown = sorted(
            {sample.action_id for sample in samples} - set(class_to_index)
        )
        if unknown:
            raise ValueError("samples contain unknown actions: " + ", ".join(unknown))
        self.samples = tuple(samples)
        self.class_to_index = dict(class_to_index)
        self.cache_config = cache_config
        self.cache_root = Path(cache_root) if cache_root is not None else None

    def __len__(self) -> int:
        return len(self.samples)

    def _load_array(self, sample: SampleRecord) -> np.ndarray:
        if self.cache_root is not None:
            path = cache_path(self.cache_root, sample, self.cache_config)
            if path.is_file():
                array = np.load(path, allow_pickle=False)
                expected_shape = (
                    2,
                    self.cache_config.target_packets,
                    self.cache_config.target_subcarriers,
                )
                if array.shape != expected_shape or array.dtype != np.float32:
                    raise ValueError(
                        f"sample {sample.sample_id}: invalid cache {path}; "
                        f"expected float32 {expected_shape}, got {array.dtype} {array.shape}"
                    )
                return array

        csi, headers = read_feitcsi(sample.csi_path)
        if len(headers) != sample.csi_packets_written:
            raise ValueError(
                f"sample {sample.sample_id}: packet count mismatch; "
                f"manifest={sample.csi_packets_written}, parsed={len(headers)}"
            )
        return preprocess_amplitude(
            csi,
            target_packets=self.cache_config.target_packets,
            target_subcarriers=self.cache_config.target_subcarriers,
            low_energy_ratio=self.cache_config.low_energy_ratio,
        )

    def __getitem__(self, index: int) -> tuple[torch.Tensor, int, str]:
        sample = self.samples[index]
        array = self._load_array(sample)
        tensor = torch.from_numpy(np.ascontiguousarray(array))
        label = self.class_to_index[sample.action_id]
        return tensor, label, sample.sample_id

