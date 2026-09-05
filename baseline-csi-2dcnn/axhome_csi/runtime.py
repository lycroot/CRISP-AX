from __future__ import annotations

import importlib
import platform


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


def collect_runtime_metadata() -> dict[str, object]:
    """Record the software and accelerator provenance of one training run.

    Cross-device bit-identical reproduction is not attainable: cuDNN kernel
    selection, floating-point reduction order and TF32 behaviour all differ
    between GPU architectures and driver versions. Recording the exact stack
    is therefore what makes a run auditable — reproduction is expected to
    fall within the reported seed/fold dispersion, not to match digit for
    digit.
    """
    return {
        "software_versions": {
            "python": platform.python_version(),
            "numpy": _module_version("numpy"),
            "torch": _module_version("torch"),
            "matplotlib": _module_version("matplotlib"),
        },
        "accelerator": _accelerator_metadata(),
    }
