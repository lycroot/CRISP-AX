from __future__ import annotations

import hashlib
import platform
import tempfile
import unittest
from pathlib import Path

from axhome_video.runtime import (
    collect_runtime_metadata,
    make_grad_scaler,
    sha256_file,
)


class RuntimeMetadataTests(unittest.TestCase):
    def test_grad_scaler_falls_back_to_torch_22_cuda_namespace(self) -> None:
        class LegacyScaler:
            def __init__(self, *, enabled: bool) -> None:
                self.enabled = enabled

        class Namespace:
            pass

        fake_torch = Namespace()
        fake_torch.amp = Namespace()
        fake_torch.cuda = Namespace()
        fake_torch.cuda.amp = Namespace()
        fake_torch.cuda.amp.GradScaler = LegacyScaler

        scaler = make_grad_scaler(fake_torch, enabled=True)

        self.assertIsInstance(scaler, LegacyScaler)
        self.assertTrue(scaler.enabled)

    def test_collects_reproducibility_version_fields(self) -> None:
        metadata = collect_runtime_metadata(include_r3d18_weight=False)

        versions = metadata["software_versions"]
        self.assertEqual(versions["python"], platform.python_version())
        self.assertTrue(versions["numpy"])
        for field in ("torch", "torchvision", "pyav", "pillow"):
            self.assertIn(field, versions)
        for field in (
            "cuda_available",
            "cuda_runtime",
            "cudnn",
            "device_name",
        ):
            self.assertIn(field, metadata["accelerator"])
        self.assertIsNone(metadata["pretrained_checkpoint"])

    def test_sha256_file_records_exact_checkpoint_bytes(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "weights.pth"
            path.write_bytes(b"official-weights")
            actual = sha256_file(path)

        self.assertEqual(
            actual, hashlib.sha256(b"official-weights").hexdigest()
        )


if __name__ == "__main__":
    unittest.main()
