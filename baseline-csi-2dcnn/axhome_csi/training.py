from __future__ import annotations

import csv
import json
import random
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Sequence

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader

from .cache import CacheConfig
from .data import SampleRecord, load_manifest
from .metrics import (
    balanced_class_weights,
    classification_metrics,
    write_evaluation,
)
from .model import (
    ARCHITECTURE_DESCRIPTIONS,
    CSI_ARCHITECTURES,
    DEFAULT_ARCHITECTURE,
    build_model,
    count_parameters,
)
from .runtime import collect_runtime_metadata
from .splits import HUMAN_ACTIONS, make_subject_fold
from .torch_data import CSIAmplitudeDataset


@dataclass(frozen=True)
class TrainingConfig:
    epochs: int = 50
    batch_size: int = 32
    learning_rate: float = 1e-3
    weight_decay: float = 0.0
    dropout: float = 0.5
    patience: int = 10
    num_workers: int = 4
    seed: int = 2026
    device: str = "auto"
    arch: str = DEFAULT_ARCHITECTURE

    def __post_init__(self) -> None:
        if self.epochs <= 0 or self.batch_size <= 0:
            raise ValueError("epochs and batch_size must be positive")
        if self.learning_rate <= 0 or self.weight_decay < 0:
            raise ValueError("invalid optimizer configuration")
        if self.patience <= 0 or self.num_workers < 0:
            raise ValueError("invalid patience or num_workers")
        if not 0 <= self.dropout < 1:
            raise ValueError("dropout must be in [0, 1)")
        if self.arch not in CSI_ARCHITECTURES:
            raise ValueError(
                f"unknown architecture: {self.arch}; "
                f"expected one of {', '.join(CSI_ARCHITECTURES)}"
            )


def resolve_device(requested: str) -> torch.device:
    if requested == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    device = torch.device(requested)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is not available")
    return device


def set_reproducible_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


def _loader(
    dataset: CSIAmplitudeDataset,
    *,
    config: TrainingConfig,
    shuffle: bool,
    device: torch.device,
) -> DataLoader:
    generator = torch.Generator().manual_seed(config.seed)
    return DataLoader(
        dataset,
        batch_size=config.batch_size,
        shuffle=shuffle,
        num_workers=config.num_workers,
        pin_memory=device.type == "cuda",
        persistent_workers=config.num_workers > 0,
        generator=generator,
    )


def _run_epoch(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
    *,
    optimizer: torch.optim.Optimizer | None,
    class_names: Sequence[str],
) -> tuple[float, dict[str, object], list[str], np.ndarray, np.ndarray]:
    training = optimizer is not None
    model.train(training)
    total_loss = 0.0
    total_samples = 0
    true_labels: list[int] = []
    predictions: list[int] = []
    sample_ids: list[str] = []
    context = torch.enable_grad() if training else torch.inference_mode()
    with context:
        for inputs, labels, batch_sample_ids in loader:
            inputs = inputs.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)
            if training:
                optimizer.zero_grad(set_to_none=True)
            logits = model(inputs)
            loss = criterion(logits, labels)
            if training:
                loss.backward()
                optimizer.step()
            batch_size = labels.shape[0]
            total_loss += float(loss.detach().item()) * batch_size
            total_samples += batch_size
            true_labels.extend(labels.detach().cpu().tolist())
            predictions.extend(logits.argmax(dim=1).detach().cpu().tolist())
            sample_ids.extend(batch_sample_ids)
    y_true = np.asarray(true_labels, dtype=np.int64)
    y_pred = np.asarray(predictions, dtype=np.int64)
    metrics = classification_metrics(y_true, y_pred, class_names=class_names)
    return total_loss / total_samples, metrics, sample_ids, y_true, y_pred


def _write_history(path: Path, history: list[dict[str, object]]) -> None:
    if not history:
        return
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(history[0]))
        writer.writeheader()
        writer.writerows(history)


def train_sample_split(
    dataset_root: str | Path,
    *,
    train_samples: Sequence[SampleRecord],
    val_samples: Sequence[SampleRecord],
    test_samples: Sequence[SampleRecord],
    output_dir: str | Path,
    run_metadata: dict[str, object],
    metric_metadata: dict[str, object] | None = None,
    cache_root: str | Path | None = None,
    cache_config: CacheConfig | None = None,
    training_config: TrainingConfig | None = None,
    class_names: Sequence[str] = HUMAN_ACTIONS,
) -> dict[str, object]:
    """Train and evaluate explicit, non-overlapping sample partitions."""
    cache_config = cache_config or CacheConfig()
    training_config = training_config or TrainingConfig()
    class_names = tuple(class_names)
    class_to_index = {name: index for index, name in enumerate(class_names)}
    if len(class_to_index) != len(class_names):
        raise ValueError("class_names contains duplicates")
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    set_reproducible_seed(training_config.seed)
    device = resolve_device(training_config.device)

    selected_actions = set(class_names)
    train_samples = tuple(
        sample for sample in train_samples if sample.action_id in selected_actions
    )
    val_samples = tuple(
        sample for sample in val_samples if sample.action_id in selected_actions
    )
    test_samples = tuple(
        sample for sample in test_samples if sample.action_id in selected_actions
    )
    if not train_samples or not val_samples or not test_samples:
        raise ValueError("selected class set leaves an empty split")
    split_ids = []
    for split_name, split_samples in (
        ("train", train_samples),
        ("validation", val_samples),
        ("test", test_samples),
    ):
        sample_ids = [sample.sample_id for sample in split_samples]
        if len(sample_ids) != len(set(sample_ids)):
            raise ValueError(f"duplicate sample IDs in {split_name} split")
        split_ids.append(set(sample_ids))
    if any(
        left & right
        for index, left in enumerate(split_ids)
        for right in split_ids[index + 1 :]
    ):
        raise ValueError("train, validation, and test samples overlap")

    datasets = {
        "train": CSIAmplitudeDataset(
            train_samples,
            class_to_index=class_to_index,
            cache_config=cache_config,
            cache_root=cache_root,
        ),
        "val": CSIAmplitudeDataset(
            val_samples,
            class_to_index=class_to_index,
            cache_config=cache_config,
            cache_root=cache_root,
        ),
        "test": CSIAmplitudeDataset(
            test_samples,
            class_to_index=class_to_index,
            cache_config=cache_config,
            cache_root=cache_root,
        ),
    }
    loaders = {
        name: _loader(
            dataset,
            config=training_config,
            shuffle=name == "train",
            device=device,
        )
        for name, dataset in datasets.items()
    }

    train_labels = np.asarray(
        [class_to_index[sample.action_id] for sample in train_samples],
        dtype=np.int64,
    )
    weights = torch.from_numpy(
        balanced_class_weights(train_labels, len(class_names))
    ).to(device)
    model = build_model(
        num_classes=len(class_names),
        arch=training_config.arch,
        in_channels=2,
        num_subcarriers=cache_config.target_subcarriers,
        dropout=training_config.dropout,
    ).to(device)
    criterion = nn.CrossEntropyLoss(weight=weights)
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=training_config.learning_rate,
        weight_decay=training_config.weight_decay,
    )

    run_config = {
        "dataset_root": str(Path(dataset_root).resolve()),
        "cache_root": str(Path(cache_root).resolve()) if cache_root else None,
        "device": str(device),
        "class_names": list(class_names),
        "cache_config": cache_config.as_dict(),
        "training_config": asdict(training_config),
        "model": {
            "arch": training_config.arch,
            "description": ARCHITECTURE_DESCRIPTIONS[training_config.arch],
            "trainable_parameters": count_parameters(model),
        },
        "runtime": collect_runtime_metadata(),
    }
    reserved = set(run_config) & set(run_metadata)
    if reserved:
        raise ValueError(
            "run_metadata uses reserved keys: " + ", ".join(sorted(reserved))
        )
    run_config.update(run_metadata)
    (output / "run_config.json").write_text(
        json.dumps(run_config, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )

    history: list[dict[str, object]] = []
    best_score = float("-inf")
    best_epoch = 0
    stale_epochs = 0
    checkpoint_path = output / "best_model.pt"
    for epoch in range(1, training_config.epochs + 1):
        train_loss, train_metrics, _, _, _ = _run_epoch(
            model,
            loaders["train"],
            criterion,
            device,
            optimizer=optimizer,
            class_names=class_names,
        )
        val_loss, val_metrics, _, _, _ = _run_epoch(
            model,
            loaders["val"],
            criterion,
            device,
            optimizer=None,
            class_names=class_names,
        )
        row = {
            "epoch": epoch,
            "train_loss": train_loss,
            "train_accuracy": train_metrics["accuracy"],
            "train_macro_f1": train_metrics["macro_f1"],
            "val_loss": val_loss,
            "val_accuracy": val_metrics["accuracy"],
            "val_macro_f1": val_metrics["macro_f1"],
        }
        history.append(row)
        _write_history(output / "history.csv", history)
        print(
            f"epoch={epoch:03d} train_loss={train_loss:.4f} "
            f"train_macro_f1={float(train_metrics['macro_f1']):.4f} "
            f"val_loss={val_loss:.4f} "
            f"val_macro_f1={float(val_metrics['macro_f1']):.4f}",
            flush=True,
        )
        score = float(val_metrics["macro_f1"])
        if score > best_score:
            best_score = score
            best_epoch = epoch
            stale_epochs = 0
            torch.save(
                {
                    "model_state_dict": model.state_dict(),
                    "class_names": list(class_names),
                    "best_epoch": best_epoch,
                    "best_val_macro_f1": best_score,
                    "cache_config": cache_config.as_dict(),
                },
                checkpoint_path,
            )
        else:
            stale_epochs += 1
            if stale_epochs >= training_config.patience:
                print(f"early_stop epoch={epoch}", flush=True)
                break

    checkpoint = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(checkpoint["model_state_dict"])
    test_loss, test_metrics, sample_ids, y_true, y_pred = _run_epoch(
        model,
        loaders["test"],
        criterion,
        device,
        optimizer=None,
        class_names=class_names,
    )
    test_metrics.update(
        {
            "test_loss": test_loss,
            "best_epoch": best_epoch,
            "best_val_macro_f1": best_score,
            "arch": training_config.arch,
        }
    )
    if metric_metadata:
        reserved_metrics = set(test_metrics) & set(metric_metadata)
        if reserved_metrics:
            raise ValueError(
                "metric_metadata uses reserved keys: "
                + ", ".join(sorted(reserved_metrics))
            )
        test_metrics.update(metric_metadata)
    for metric_name in (
        "accuracy",
        "balanced_accuracy",
        "macro_f1",
        "weighted_f1",
        "test_loss",
        "best_val_macro_f1",
    ):
        if not np.isfinite(float(test_metrics[metric_name])):
            raise ValueError(f"non-finite evaluation metric: {metric_name}")
    write_evaluation(
        output,
        metrics=test_metrics,
        sample_ids=sample_ids,
        y_true=y_true,
        y_pred=y_pred,
        class_names=class_names,
    )
    return test_metrics


def train_subject_fold(
    dataset_root: str | Path,
    *,
    test_person: str,
    val_person: str,
    output_dir: str | Path,
    cache_root: str | Path | None = None,
    cache_config: CacheConfig | None = None,
    training_config: TrainingConfig | None = None,
    class_names: Sequence[str] = HUMAN_ACTIONS,
) -> dict[str, object]:
    """Train and evaluate one strictly subject-independent fold."""
    samples = load_manifest(dataset_root, validate_paths=True)
    fold = make_subject_fold(
        samples, test_person=test_person, val_person=val_person
    )
    selected_actions = set(class_names)
    train_samples = tuple(
        sample for sample in fold.train if sample.action_id in selected_actions
    )
    val_samples = tuple(
        sample for sample in fold.val if sample.action_id in selected_actions
    )
    test_samples = tuple(
        sample for sample in fold.test if sample.action_id in selected_actions
    )
    return train_sample_split(
        dataset_root,
        train_samples=train_samples,
        val_samples=val_samples,
        test_samples=test_samples,
        output_dir=output_dir,
        run_metadata={
            "fold": {
                "test_person": test_person,
                "val_person": val_person,
                "train_people": list(fold.train_people),
                "train_samples": len(train_samples),
                "val_samples": len(val_samples),
                "test_samples": len(test_samples),
            }
        },
        metric_metadata={
            "test_person": test_person,
            "val_person": val_person,
        },
        cache_root=cache_root,
        cache_config=cache_config,
        training_config=training_config,
        class_names=class_names,
    )
