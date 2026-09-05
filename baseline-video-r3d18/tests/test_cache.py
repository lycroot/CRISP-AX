from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import numpy as np

from axhome_video.cache import (
    VideoCacheConfig,
    cache_path,
    ensure_cache_entry,
    load_cached_clip,
    uniform_frame_indices,
)
from tests._fixtures import make_records


class VideoCacheTests(unittest.TestCase):
    def test_uniform_indices_cover_full_window(self) -> None:
        np.testing.assert_array_equal(
            uniform_frame_indices(151, 16),
            np.rint(np.linspace(0, 150, 16)).astype(np.int64),
        )

    def test_uniform_indices_reject_short_video(self) -> None:
        with self.assertRaisesRegex(ValueError, "required"):
            uniform_frame_indices(15, 16)

    def test_cache_entry_is_uint8_with_expected_shape(self) -> None:
        record = make_records(per_stratum=1)[0]
        config = VideoCacheConfig()
        expected = np.arange(
            config.frames * config.height * config.width * 3,
            dtype=np.uint8,
        ).reshape(config.frames, config.height, config.width, 3)
        with tempfile.TemporaryDirectory() as tmp:
            clip = ensure_cache_entry(
                record,
                tmp,
                config,
                decoder=lambda *_: expected,
            )
            loaded = load_cached_clip(tmp, record, config)
            path = cache_path(tmp, record, config)

        self.assertTrue(path.name.endswith(".npy"))
        self.assertEqual(clip.shape, (16, 128, 171, 3))
        self.assertEqual(clip.dtype, np.uint8)
        np.testing.assert_array_equal(loaded, expected)

    def test_different_configs_use_different_directories(self) -> None:
        self.assertNotEqual(
            VideoCacheConfig(frames=16).fingerprint(),
            VideoCacheConfig(frames=8).fingerprint(),
        )

    def test_rejects_decoder_with_wrong_shape_or_dtype(self) -> None:
        record = make_records(per_stratum=1)[0]
        config = VideoCacheConfig()
        with tempfile.TemporaryDirectory() as tmp:
            with self.assertRaisesRegex(ValueError, "cache shape"):
                ensure_cache_entry(
                    record,
                    tmp,
                    config,
                    decoder=lambda *_: np.zeros(
                        (15, 128, 171, 3), dtype=np.uint8
                    ),
                )
            with self.assertRaisesRegex(ValueError, "must be uint8"):
                ensure_cache_entry(
                    record,
                    tmp,
                    config,
                    decoder=lambda *_: np.zeros(
                        (16, 128, 171, 3), dtype=np.float32
                    ),
                )

    def test_corrupted_existing_cache_is_not_silently_rebuilt(self) -> None:
        record = make_records(per_stratum=1)[0]
        config = VideoCacheConfig()
        with tempfile.TemporaryDirectory() as tmp:
            path = cache_path(tmp, record, config)
            path.parent.mkdir(parents=True)
            path.write_bytes(b"not-a-numpy-file")
            with self.assertRaisesRegex(ValueError, "failed to read"):
                ensure_cache_entry(record, tmp, config)


if __name__ == "__main__":
    unittest.main()
