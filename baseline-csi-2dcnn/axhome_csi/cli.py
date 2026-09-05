from __future__ import annotations

import argparse
from typing import TYPE_CHECKING

from .cache import CacheConfig
from .model import CSI_ARCHITECTURES, DEFAULT_ARCHITECTURE

if TYPE_CHECKING:
    from .training import TrainingConfig


def add_training_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--dataset-root", required=True, help="解压后的 AXHome-MM-v1 目录")
    parser.add_argument("--output-dir", required=True, help="结果输出目录")
    parser.add_argument("--cache-dir", default=None, help="可选的预处理缓存根目录")
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=0.0)
    parser.add_argument("--dropout", type=float, default=0.5)
    parser.add_argument("--patience", type=int, default=10)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument(
        "--device", default="auto", help="auto、cpu、cuda 或 cuda:0 等 PyTorch device"
    )
    parser.add_argument(
        "--arch",
        default=DEFAULT_ARCHITECTURE,
        choices=list(CSI_ARCHITECTURES),
        help=(
            "CSI-only 基线架构；全部架构共用同一冻结输入与同一训练预算，"
            "结果用于说明数据可分性而非模型优劣"
        ),
    )
    parser.add_argument("--target-packets", type=int, default=256)
    parser.add_argument("--target-subcarriers", type=int, default=128)
    parser.add_argument("--low-energy-ratio", type=float, default=0.05)


def configs_from_args(
    args: argparse.Namespace,
) -> tuple[CacheConfig, "TrainingConfig"]:
    from .training import TrainingConfig

    cache_config = CacheConfig(
        target_packets=args.target_packets,
        target_subcarriers=args.target_subcarriers,
        low_energy_ratio=args.low_energy_ratio,
    )
    training_config = TrainingConfig(
        epochs=args.epochs,
        batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        weight_decay=args.weight_decay,
        dropout=args.dropout,
        patience=args.patience,
        num_workers=args.num_workers,
        seed=args.seed,
        device=args.device,
        arch=args.arch,
    )
    return cache_config, training_config
