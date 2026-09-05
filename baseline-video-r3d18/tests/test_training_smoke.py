from __future__ import annotations

import importlib.util
import json
import math
import tempfile
import unittest
from pathlib import Path

import numpy as np

from axhome_video.cache import VideoCacheConfig, ensure_cache_entry
from axhome_video.splits import make_subject_fold
from tests._fixtures import make_records


@unittest.skipUnless(
    importlib.util.find_spec("torch") is not None,
    "PyTorch is required",
)
class VideoTrainingSmokeTests(unittest.TestCase):
    def test_one_epoch_explicit_split_writes_outputs(self) -> None:
        import torch
        from torch import nn

        from axhome_video.training import TrainingConfig, train_sample_split

        class TinyVideoCNN(nn.Module):
            def __init__(self, classes: int) -> None:
                super().__init__()
                self.features = nn.Sequential(
                    nn.Conv3d(3, 4, kernel_size=1),
                    nn.ReLU(),
                    nn.AdaptiveAvgPool3d(1),
                )
                self.classifier = nn.Linear(4, classes)

            def forward(self, inputs: torch.Tensor) -> torch.Tensor:
                return self.classifier(self.features(inputs).flatten(1))

        records = make_records(per_stratum=1)
        fold = make_subject_fold(
            records, test_person="P01", val_person="P02"
        )
        config = VideoCacheConfig()
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            cache = root / "cache"
            output = root / "result"
            for index, record in enumerate(records):
                value = np.uint8((index * 17) % 255)
                ensure_cache_entry(
                    record,
                    cache,
                    config,
                    decoder=lambda *_, value=value: np.full(
                        (16, 128, 171, 3), value, dtype=np.uint8
                    ),
                )
            metrics = train_sample_split(
                root,
                train_samples=fold.train,
                val_samples=fold.val,
                test_samples=fold.test,
                output_dir=output,
                run_metadata={"protocol": "smoke"},
                cache_root=cache,
                cache_config=config,
                training_config=TrainingConfig(
                    epochs=1,
                    batch_size=2,
                    patience=1,
                    num_workers=0,
                    device="cpu",
                    amp=False,
                    pretrained=False,
                ),
                class_names=("walk", "sit_down"),
                model_factory=TinyVideoCNN,
            )

            for name in (
                "accuracy",
                "balanced_accuracy",
                "macro_f1",
                "weighted_f1",
                "test_loss",
            ):
                self.assertTrue(math.isfinite(float(metrics[name])), name)
            for name in (
                "best_model.pt",
                "run_config.json",
                "history.csv",
                "metrics.json",
                "predictions.csv",
            ):
                self.assertTrue((output / name).is_file(), name)
            run_config = json.loads(
                (output / "run_config.json").read_text(encoding="utf-8")
            )
            self.assertTrue(
                run_config["runtime"]["software_versions"]["python"]
            )
            self.assertTrue(
                run_config["runtime"]["software_versions"]["torch"]
            )
            self.assertIsNone(
                run_config["runtime"]["pretrained_checkpoint"]
            )


if __name__ == "__main__":
    unittest.main()
