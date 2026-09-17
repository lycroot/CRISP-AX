from __future__ import annotations

import argparse
from typing import TYPE_CHECKING

from .cache import VideoCacheConfig
from .model import DEFAULT_ARCHITECTURE, VIDEO_ARCHITECTURES

if TYPE_CHECKING:
    from .training import TrainingConfig


def add_training_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--dataset-root", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--cache-dir", required=True)
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--learning-rate", type=float, default=1e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--patience", type=int, default=10)
    parser.add_argument("--num-workers", type=int, default=8)
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--device", default="auto")
    parser.add_argument(
        "--arch",
        default=DEFAULT_ARCHITECTURE,
        choices=list(VIDEO_ARCHITECTURES),
        help=(
            "Video-only baseline architecture; all architectures share the same frozen "
            "clip input, Kinetics-400 pretraining family and training budget"
        ),
    )
    parser.add_argument("--frames", type=int, default=16)
    parser.add_argument("--height", type=int, default=128)
    parser.add_argument("--width", type=int, default=171)
    parser.add_argument(
        "--amp", action=argparse.BooleanOptionalAction, default=True
    )
    parser.add_argument(
        "--pretrained", action=argparse.BooleanOptionalAction, default=True
    )


def configs_from_args(
    args: argparse.Namespace,
) -> tuple[VideoCacheConfig, "TrainingConfig"]:
    from .training import TrainingConfig

    cache_config = VideoCacheConfig(
        frames=args.frames, height=args.height, width=args.width
    )
    training_config = TrainingConfig(
        epochs=args.epochs,
        batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        weight_decay=args.weight_decay,
        patience=args.patience,
        num_workers=args.num_workers,
        seed=args.seed,
        device=args.device,
        amp=args.amp,
        pretrained=args.pretrained,
        arch=args.arch,
    )
    return cache_config, training_config
