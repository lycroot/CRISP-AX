from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np

from .data import SampleRecord, preprocess_amplitude
from .feitcsi import read_feitcsi


@dataclass(frozen=True)
class CacheConfig:
    target_packets: int = 256
    target_subcarriers: int = 128
    low_energy_ratio: float = 0.05

    def __post_init__(self) -> None:
        if self.target_packets <= 0 or self.target_subcarriers <= 0:
            raise ValueError("cache target dimensions must be positive")
        if not 0 <= self.low_energy_ratio < 1:
            raise ValueError("low_energy_ratio must be in [0, 1)")

    @property
    def signature(self) -> str:
        ratio = f"{self.low_energy_ratio:.4f}".rstrip("0").rstrip(".")
        return (
            f"amplitude_logz_t{self.target_packets}_f{self.target_subcarriers}"
            f"_low{ratio}_v1"
        )

    def as_dict(self) -> dict[str, object]:
        return {**asdict(self), "signature": self.signature}


def cache_path(
    cache_root: str | Path, sample: SampleRecord, config: CacheConfig
) -> Path:
    return (
        Path(cache_root)
        / config.signature
        / sample.archive_session_id
        / f"{sample.sample_id}.npy"
    )


def _valid_cached_array(path: Path, config: CacheConfig) -> bool:
    try:
        array = np.load(path, mmap_mode="r", allow_pickle=False)
    except (OSError, ValueError):
        return False
    return (
        array.dtype == np.float32
        and array.ndim == 3
        and array.shape[1:] == (
            config.target_packets,
            config.target_subcarriers,
        )
        and array.shape[0] > 0
    )


def prepare_sample_cache(
    sample: SampleRecord,
    cache_root: str | Path,
    config: CacheConfig,
    *,
    overwrite: bool = False,
) -> Path:
    """Parse, preprocess and atomically cache one sample."""
    output_path = cache_path(cache_root, sample, config)
    if output_path.is_file() and not overwrite and _valid_cached_array(
        output_path, config
    ):
        return output_path

    csi, headers = read_feitcsi(sample.csi_path)
    if len(headers) != sample.csi_packets_written:
        raise ValueError(
            f"sample {sample.sample_id}: packet count mismatch; "
            f"manifest={sample.csi_packets_written}, parsed={len(headers)}"
        )
    array = preprocess_amplitude(
        csi,
        target_packets=config.target_packets,
        target_subcarriers=config.target_subcarriers,
        low_energy_ratio=config.low_energy_ratio,
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path = output_path.with_suffix(".tmp.npy")
    try:
        np.save(temporary_path, array, allow_pickle=False)
        temporary_path.replace(output_path)
    finally:
        temporary_path.unlink(missing_ok=True)
    return output_path

