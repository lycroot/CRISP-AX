from __future__ import annotations

import hashlib
import importlib
import platform
from pathlib import Path
from typing import Any
from urllib.parse import urlparse


def sha256_file(path: str | Path) -> str:
    source = Path(path)
    digest = hashlib.sha256()
    with source.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def make_grad_scaler(torch_module: Any, *, enabled: bool) -> Any:
    modern = getattr(getattr(torch_module, "amp", None), "GradScaler", None)
    if modern is not None:
        return modern("cuda", enabled=enabled)
    return torch_module.cuda.amp.GradScaler(enabled=enabled)


def _module_version(module_name: str) -> str | None:
    try:
        module = importlib.import_module(module_name)
    except ImportError:
        return None
    value = getattr(module, "__version__", None)
    return str(value) if value is not None else None


def _accelerator_metadata() -> dict[str, object]:
    try:
        import torch
    except ImportError:
        return {
            "cuda_available": False,
            "cuda_runtime": None,
            "cudnn": None,
            "device_name": None,
        }
    available = bool(torch.cuda.is_available())
    return {
        "cuda_available": available,
        "cuda_runtime": torch.version.cuda,
        "cudnn": torch.backends.cudnn.version(),
        "device_name": torch.cuda.get_device_name(0) if available else None,
    }


def _pretrained_weight_metadata(arch: str) -> dict[str, object]:
    import torch

    from .model import ARCHITECTURE_WEIGHTS, _torchvision_entry

    _, weight = _torchvision_entry(arch)
    filename = Path(urlparse(weight.url).path).name
    checkpoint = Path(torch.hub.get_dir()) / "checkpoints" / filename
    if not checkpoint.is_file():
        raise FileNotFoundError(
            f"{arch} pretrained checkpoint was not cached after model "
            f"creation: {checkpoint}"
        )
    return {
        "arch": arch,
        "name": ARCHITECTURE_WEIGHTS[arch],
        "url": weight.url,
        "filename": filename,
        "local_path": str(checkpoint.resolve()),
        "sha256": sha256_file(checkpoint),
    }


def collect_runtime_metadata(
    *, include_r3d18_weight: bool, arch: str = "r3d_18"
) -> dict[str, object]:
    """Collect version, accelerator and pretrained-checkpoint provenance.

    ``include_r3d18_weight`` keeps its original name for backward
    compatibility; it now selects whether the checkpoint of ``arch`` is
    hashed and recorded.
    """
    return {
        "software_versions": {
            "python": platform.python_version(),
            "numpy": _module_version("numpy"),
            "torch": _module_version("torch"),
            "torchvision": _module_version("torchvision"),
            "pyav": _module_version("av"),
            "pillow": _module_version("PIL"),
        },
        "accelerator": _accelerator_metadata(),
        "pretrained_checkpoint": (
            _pretrained_weight_metadata(arch) if include_r3d18_weight else None
        ),
    }
