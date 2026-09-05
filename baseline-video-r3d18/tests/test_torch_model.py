from __future__ import annotations

import importlib.util
import tempfile
import unittest
from pathlib import Path

import numpy as np

from axhome_video.cache import VideoCacheConfig, ensure_cache_entry
from tests._fixtures import make_records

TORCH_AVAILABLE = importlib.util.find_spec("torch") is not None
TORCHVISION_AVAILABLE = importlib.util.find_spec("torchvision") is not None


@unittest.skipUnless(
    TORCH_AVAILABLE and TORCHVISION_AVAILABLE,
    "PyTorch and Torchvision are required",
)
class VideoTorchAndModelTests(unittest.TestCase):
    def test_eval_transform_is_deterministic_and_finite(self) -> None:
        import torch

        from axhome_video.torch_data import transform_cached_clip

        clip = np.arange(16 * 128 * 171 * 3, dtype=np.uint8).reshape(
            16, 128, 171, 3
        )
        first = transform_cached_clip(clip, training=False)
        second = transform_cached_clip(clip, training=False)

        self.assertTrue(torch.equal(first, second))
        self.assertEqual(tuple(first.shape), (3, 16, 112, 112))
        self.assertTrue(bool(torch.isfinite(first).all()))

    def test_training_transform_has_expected_shape(self) -> None:
        import torch

        from axhome_video.torch_data import transform_cached_clip

        torch.manual_seed(2026)
        clip = np.zeros((16, 128, 171, 3), dtype=np.uint8)
        transformed = transform_cached_clip(clip, training=True)
        self.assertEqual(tuple(transformed.shape), (3, 16, 112, 112))

    def test_cached_video_dataset_returns_label_and_sample_id(self) -> None:
        from axhome_video.torch_data import CachedVideoDataset

        record = make_records(per_stratum=1)[0]
        config = VideoCacheConfig()
        with tempfile.TemporaryDirectory() as tmp:
            ensure_cache_entry(
                record,
                tmp,
                config,
                decoder=lambda *_: np.zeros(
                    (16, 128, 171, 3), dtype=np.uint8
                ),
            )
            dataset = CachedVideoDataset(
                (record,),
                cache_root=Path(tmp),
                cache_config=config,
                action_to_index={record.action_id: 0},
                training=False,
            )
            tensor, label, sample_id = dataset[0]

        self.assertEqual(tuple(tensor.shape), (3, 16, 112, 112))
        self.assertEqual(label, 0)
        self.assertEqual(sample_id, record.sample_id)

    def test_r3d18_maps_clip_to_fourteen_logits(self) -> None:
        import torch

        from axhome_video.model import build_r3d18

        model = build_r3d18(num_classes=14, pretrained=False)
        model.eval()
        with torch.no_grad():
            logits = model(torch.zeros(1, 3, 16, 112, 112))
        self.assertEqual(tuple(logits.shape), (1, 14))


if __name__ == "__main__":
    unittest.main()
