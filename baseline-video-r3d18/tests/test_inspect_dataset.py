from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from inspect_dataset import build_profile
from tests._fixtures import write_manifest_fixture


class VideoDatasetInspectionTests(unittest.TestCase):
    def test_profile_reports_counts_statistics_and_missing_files(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp) / "release"
            rows = write_manifest_fixture(root, trials=3)
            missing_path = root / rows[0]["archive_video_path"]
            missing_path.unlink()
            profile, missing = build_profile(root)

        self.assertEqual(profile["total_samples"], 18)
        self.assertEqual(profile["human_action_samples"], 18)
        self.assertEqual(profile["background_samples"], 0)
        self.assertEqual(profile["video_frames_written"]["median"], 151.0)
        self.assertEqual(profile["video_fps"]["mean"], 30.0)
        self.assertEqual(profile["missing_video_files"], 1)
        self.assertEqual(len(missing), 1)


if __name__ == "__main__":
    unittest.main()
