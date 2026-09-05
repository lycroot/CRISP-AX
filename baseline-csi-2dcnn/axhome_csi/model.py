from __future__ import annotations

import torch
from torch import nn

CSI_ARCHITECTURES: tuple[str, ...] = (
    "cnn2d",
    "resnet18",
    "bilstm",
    "transformer",
)
DEFAULT_ARCHITECTURE = "cnn2d"

ARCHITECTURE_DESCRIPTIONS: dict[str, str] = {
    "cnn2d": (
        "Lightweight three-block 2D CNN over the two-RX CSI amplitude map"
    ),
    "resnet18": (
        "From-scratch ResNet-18 (BasicBlock, [2, 2, 2, 2]) without ImageNet "
        "initialization"
    ),
    "bilstm": (
        "Linear packet projection followed by a two-layer bidirectional LSTM"
    ),
    "transformer": (
        "Linear packet projection with learned positions and a Transformer "
        "encoder"
    ),
}


def _check_input(inputs: torch.Tensor) -> None:
    if inputs.ndim != 4:
        raise ValueError(
            "model input must have shape (batch, rx, time, frequency)"
        )


class ConvBlock(nn.Sequential):
    def __init__(
        self, in_channels: int, out_channels: int, kernel_size: int
    ) -> None:
        padding = kernel_size // 2
        super().__init__(
            nn.Conv2d(
                in_channels,
                out_channels,
                kernel_size=kernel_size,
                padding=padding,
                bias=False,
            ),
            nn.BatchNorm2d(out_channels),
            nn.Mish(),
            nn.MaxPool2d(kernel_size=2, stride=2),
        )


class CSI2DCNN(nn.Module):
    """Lightweight EHUNAM-inspired CNN for two-RX CSI amplitude maps."""

    def __init__(
        self, *, num_classes: int, in_channels: int = 2, dropout: float = 0.5
    ) -> None:
        super().__init__()
        if num_classes <= 1 or in_channels <= 0:
            raise ValueError("invalid model dimensions")
        if not 0 <= dropout < 1:
            raise ValueError("dropout must be in [0, 1)")
        self.features = nn.Sequential(
            ConvBlock(in_channels, 32, 5),
            ConvBlock(32, 64, 3),
            ConvBlock(64, 128, 3),
            nn.AdaptiveAvgPool2d((1, 1)),
        )
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(128, 256),
            nn.Mish(),
            nn.Dropout(dropout),
            nn.Linear(256, 128),
            nn.Mish(),
            nn.Dropout(dropout),
            nn.Linear(128, num_classes),
        )

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        _check_input(inputs)
        return self.classifier(self.features(inputs))


class BasicBlock(nn.Module):
    """Standard two-convolution residual block with optional downsampling."""

    expansion = 1

    def __init__(
        self, in_channels: int, out_channels: int, stride: int = 1
    ) -> None:
        super().__init__()
        self.conv1 = nn.Conv2d(
            in_channels,
            out_channels,
            kernel_size=3,
            stride=stride,
            padding=1,
            bias=False,
        )
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.conv2 = nn.Conv2d(
            out_channels, out_channels, kernel_size=3, padding=1, bias=False
        )
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.activation = nn.ReLU(inplace=True)
        if stride != 1 or in_channels != out_channels:
            self.downsample: nn.Module | None = nn.Sequential(
                nn.Conv2d(
                    in_channels,
                    out_channels,
                    kernel_size=1,
                    stride=stride,
                    bias=False,
                ),
                nn.BatchNorm2d(out_channels),
            )
        else:
            self.downsample = None

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        identity = (
            inputs if self.downsample is None else self.downsample(inputs)
        )
        out = self.activation(self.bn1(self.conv1(inputs)))
        out = self.bn2(self.conv2(out))
        return self.activation(out + identity)


class CSIResNet18(nn.Module):
    """ResNet-18 trained from scratch on two-RX CSI amplitude maps.

    The layout follows the standard BasicBlock [2, 2, 2, 2] configuration.
    Weights are randomly initialized rather than transferred from ImageNet,
    because a CSI amplitude map is not an RGB natural image; this model is a
    deeper convolutional reference point, not a transfer-learning result.
    """

    def __init__(
        self, *, num_classes: int, in_channels: int = 2, dropout: float = 0.5
    ) -> None:
        super().__init__()
        if num_classes <= 1 or in_channels <= 0:
            raise ValueError("invalid model dimensions")
        if not 0 <= dropout < 1:
            raise ValueError("dropout must be in [0, 1)")
        self.stem = nn.Sequential(
            nn.Conv2d(
                in_channels, 64, kernel_size=7, stride=2, padding=3, bias=False
            ),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1),
        )
        self.layer1 = self._make_layer(64, 64, stride=1)
        self.layer2 = self._make_layer(64, 128, stride=2)
        self.layer3 = self._make_layer(128, 256, stride=2)
        self.layer4 = self._make_layer(256, 512, stride=2)
        self.pool = nn.AdaptiveAvgPool2d((1, 1))
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Dropout(dropout),
            nn.Linear(512, num_classes),
        )
        for module in self.modules():
            if isinstance(module, nn.Conv2d):
                nn.init.kaiming_normal_(
                    module.weight, mode="fan_out", nonlinearity="relu"
                )
            elif isinstance(module, nn.BatchNorm2d):
                nn.init.ones_(module.weight)
                nn.init.zeros_(module.bias)

    @staticmethod
    def _make_layer(
        in_channels: int, out_channels: int, *, stride: int
    ) -> nn.Sequential:
        return nn.Sequential(
            BasicBlock(in_channels, out_channels, stride=stride),
            BasicBlock(out_channels, out_channels, stride=1),
        )

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        _check_input(inputs)
        features = self.stem(inputs)
        features = self.layer1(features)
        features = self.layer2(features)
        features = self.layer3(features)
        features = self.layer4(features)
        return self.classifier(self.pool(features))


class _PacketSequenceEncoder(nn.Module):
    """Flatten (batch, rx, time, frequency) into a per-packet feature sequence."""

    def __init__(
        self, *, in_channels: int, num_subcarriers: int, model_dim: int
    ) -> None:
        super().__init__()
        if num_subcarriers <= 0 or model_dim <= 0:
            raise ValueError("invalid sequence encoder dimensions")
        self.in_features = in_channels * num_subcarriers
        self.projection = nn.Linear(self.in_features, model_dim)
        self.norm = nn.LayerNorm(model_dim)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        _check_input(inputs)
        batch, channels, packets, subcarriers = inputs.shape
        features = channels * subcarriers
        if features != self.in_features:
            raise ValueError(
                f"per-packet feature size {features} does not match the "
                f"configured {self.in_features}; rebuild the model with a "
                "matching num_subcarriers"
            )
        sequence = inputs.permute(0, 2, 1, 3).reshape(batch, packets, features)
        return self.norm(self.projection(sequence))


class CSIBiLSTM(nn.Module):
    """Bidirectional LSTM over the packet axis of the CSI amplitude map."""

    def __init__(
        self,
        *,
        num_classes: int,
        in_channels: int = 2,
        num_subcarriers: int = 128,
        dropout: float = 0.5,
        model_dim: int = 128,
        hidden_size: int = 128,
        num_layers: int = 2,
    ) -> None:
        super().__init__()
        if num_classes <= 1 or in_channels <= 0:
            raise ValueError("invalid model dimensions")
        if not 0 <= dropout < 1:
            raise ValueError("dropout must be in [0, 1)")
        if num_layers <= 0 or hidden_size <= 0:
            raise ValueError("invalid recurrent dimensions")
        self.encoder = _PacketSequenceEncoder(
            in_channels=in_channels,
            num_subcarriers=num_subcarriers,
            model_dim=model_dim,
        )
        self.rnn = nn.LSTM(
            input_size=model_dim,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            bidirectional=True,
            dropout=dropout if num_layers > 1 else 0.0,
        )
        self.classifier = nn.Sequential(
            nn.LayerNorm(2 * hidden_size),
            nn.Dropout(dropout),
            nn.Linear(2 * hidden_size, num_classes),
        )

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        sequence = self.encoder(inputs)
        outputs, _ = self.rnn(sequence)
        return self.classifier(outputs.mean(dim=1))


class CSITransformer(nn.Module):
    """Transformer encoder over the packet axis of the CSI amplitude map."""

    def __init__(
        self,
        *,
        num_classes: int,
        in_channels: int = 2,
        num_subcarriers: int = 128,
        dropout: float = 0.5,
        model_dim: int = 128,
        num_heads: int = 4,
        num_layers: int = 4,
        feedforward_dim: int = 256,
        max_packets: int = 1024,
    ) -> None:
        super().__init__()
        if num_classes <= 1 or in_channels <= 0:
            raise ValueError("invalid model dimensions")
        if not 0 <= dropout < 1:
            raise ValueError("dropout must be in [0, 1)")
        if model_dim % num_heads != 0:
            raise ValueError("model_dim must be divisible by num_heads")
        if max_packets <= 0:
            raise ValueError("max_packets must be positive")
        self.encoder = _PacketSequenceEncoder(
            in_channels=in_channels,
            num_subcarriers=num_subcarriers,
            model_dim=model_dim,
        )
        self.max_packets = max_packets
        self.positions = nn.Parameter(torch.zeros(1, max_packets, model_dim))
        nn.init.trunc_normal_(self.positions, std=0.02)
        layer = nn.TransformerEncoderLayer(
            d_model=model_dim,
            nhead=num_heads,
            dim_feedforward=feedforward_dim,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.transformer = nn.TransformerEncoder(
            layer, num_layers=num_layers, enable_nested_tensor=False
        )
        self.classifier = nn.Sequential(
            nn.LayerNorm(model_dim),
            nn.Dropout(dropout),
            nn.Linear(model_dim, num_classes),
        )

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        sequence = self.encoder(inputs)
        packets = sequence.shape[1]
        if packets > self.max_packets:
            raise ValueError(
                f"sequence length {packets} exceeds max_packets "
                f"{self.max_packets}"
            )
        sequence = sequence + self.positions[:, :packets, :]
        encoded = self.transformer(sequence)
        return self.classifier(encoded.mean(dim=1))


def count_parameters(model: nn.Module) -> int:
    """Return the number of trainable parameters."""
    return int(sum(p.numel() for p in model.parameters() if p.requires_grad))


def build_model(
    *,
    num_classes: int,
    arch: str = DEFAULT_ARCHITECTURE,
    in_channels: int = 2,
    num_subcarriers: int = 128,
    dropout: float = 0.5,
) -> nn.Module:
    """Build one of the registered CSI-only baseline architectures.

    Every architecture consumes the same frozen ``(in_channels, packets,
    subcarriers)`` model input and is trained under the same protocol and
    budget, so the resulting numbers describe how much action-discriminative
    information the released CSI carries; they are not a claim that one
    architecture is superior.
    """
    if arch not in CSI_ARCHITECTURES:
        raise ValueError(
            f"unknown architecture: {arch}; "
            f"expected one of {', '.join(CSI_ARCHITECTURES)}"
        )
    if arch == "cnn2d":
        return CSI2DCNN(
            num_classes=num_classes,
            in_channels=in_channels,
            dropout=dropout,
        )
    if arch == "resnet18":
        return CSIResNet18(
            num_classes=num_classes,
            in_channels=in_channels,
            dropout=dropout,
        )
    if arch == "bilstm":
        return CSIBiLSTM(
            num_classes=num_classes,
            in_channels=in_channels,
            num_subcarriers=num_subcarriers,
            dropout=dropout,
        )
    return CSITransformer(
        num_classes=num_classes,
        in_channels=in_channels,
        num_subcarriers=num_subcarriers,
        dropout=dropout,
    )
