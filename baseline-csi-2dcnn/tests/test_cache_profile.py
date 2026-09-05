from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import numpy as np

from axhome_csi.cache import CacheConfig, cache_path, prepare_sample_cache
from axhome_csi.data import load_manifest
from axhome_csi.profile import summarize_samples
from tests._fixtures import write_mini_release


class CacheAndProfileTests(unittest.TestCase):
    def test_cache_is_created_with_configured_shape(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp) / "release"
            cache_root = Path(tmp) / "cache"
            write_mini_release(root)
            sample = load_manifest(root)[0]
            config = CacheConfig(target_packets=8, target_subcarriers=8)

            path = prepare_sample_cache(sample, cache_root, config)
            cached = np.load(path, allow_pickle=False)

            self.assertEqual(path, cache_path(cache_root, sample, config))
            self.assertEqual(cached.shape, (2, 8, 8))
            self.assertEqual(cached.dtype, np.float32)

    def test_cache_rejects_manifest_packet_mismatch(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp) / "release"
            cache_root = Path(tmp) / "cache"
            write_mini_release(root)
            sample = load_manifest(root)[0]
            bad_sample = sample.__class__(
                **{**sample.__dict__, "csi_packets_written": 99}
            )

            with self.assertRaisesRegex(ValueError, "packet count mismatch"):
                prepare_sample_cache(
                    bad_sample,
                    cache_root,
                    CacheConfig(target_packets=8, target_subcarriers=8),
                )

    def test_profile_counts_people_actions_and_packets(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp) / "release"
            write_mini_release(root)

            summary = summarize_samples(load_manifest(root))

        self.assertEqual(summary["samples_total"], 6)
        self.assertEqual(summary["samples_human"], 6)
        self.assertEqual(summary["person_counts"]["P01"], 2)
        self.assertEqual(summary["action_counts"]["walk"], 3)
        self.assertEqual(summary["packet_summary"]["median"], 12.0)


if __name__ == "__main__":
    unittest.main()

