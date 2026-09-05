from __future__ import annotations

VIDEO_ARCHITECTURES: tuple[str, ...] = ("r3d_18", "mc3_18", "r2plus1d_18")
DEFAULT_ARCHITECTURE = "r3d_18"

ARCHITECTURE_BUILDERS: dict[str, str] = {
    "r3d_18": "torchvision.models.video.r3d_18",
    "mc3_18": "torchvision.models.video.mc3_18",
    "r2plus1d_18": "torchvision.models.video.r2plus1d_18",
}
ARCHITECTURE_WEIGHTS: dict[str, str] = {
    "r3d_18": "R3D_18_Weights.KINETICS400_V1",
    "mc3_18": "MC3_18_Weights.KINETICS400_V1",
    "r2plus1d_18": "R2Plus1D_18_Weights.KINETICS400_V1",
}
ARCHITECTURE_DESCRIPTIONS: dict[str, str] = {
    "r3d_18": "18-layer network with full 3D convolutions",
    "mc3_18": "18-layer mixed convolution network: 3D early, 2D late",
    "r2plus1d_18": "18-layer network factorizing 3D into (2+1)D convolutions",
}


def validate_architecture(arch: str) -> str:
    if arch not in VIDEO_ARCHITECTURES:
        raise ValueError(
            f"unknown architecture: {arch}; "
            f"expected one of {', '.join(VIDEO_ARCHITECTURES)}"
        )
    return arch


def _torchvision_entry(arch: str):
    """Return the (constructor, weights enum) pair for one architecture."""
    validate_architecture(arch)
    try:
        from torchvision.models import video as tv_video
    except ImportError as exc:
        raise RuntimeError(
            "Video baselines require compatible PyTorch and Torchvision "
            "installations"
        ) from exc
    entries = {
        "r3d_18": (tv_video.r3d_18, tv_video.R3D_18_Weights),
        "mc3_18": (tv_video.mc3_18, tv_video.MC3_18_Weights),
        "r2plus1d_18": (tv_video.r2plus1d_18, tv_video.R2Plus1D_18_Weights),
    }
    constructor, weights_enum = entries[arch]
    return constructor, weights_enum.KINETICS400_V1


def build_video_model(
    *,
    num_classes: int,
    arch: str = DEFAULT_ARCHITECTURE,
    pretrained: bool = True,
):
    """Build one registered Torchvision video backbone with a new head.

    All architectures consume the same frozen 3 x 16 x 112 x 112 clip tensor,
    load Kinetics-400 weights of the same family and are trained under the
    same protocol and budget. The resulting numbers describe whether the
    released video windows carry action-discriminative information; they are
    not a claim that one architecture is superior.
    """
    if num_classes <= 0:
        raise ValueError("num_classes must be positive")
    constructor, kinetics_weights = _torchvision_entry(arch)
    try:
        from torch import nn
    except ImportError as exc:
        raise RuntimeError(
            "Video baselines require a compatible PyTorch installation"
        ) from exc
    weights = kinetics_weights if pretrained else None
    model = constructor(weights=weights)
    model.fc = nn.Linear(model.fc.in_features, num_classes)
    return model


def build_r3d18(*, num_classes: int, pretrained: bool = True):
    """Build a Torchvision R3D-18 with a task-specific classification head."""
    return build_video_model(
        num_classes=num_classes, arch="r3d_18", pretrained=pretrained
    )


def count_parameters(model) -> int:
    """Return the number of trainable parameters."""
    return int(sum(p.numel() for p in model.parameters() if p.requires_grad))
