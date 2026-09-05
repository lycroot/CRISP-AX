from __future__ import annotations

import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

from tests._fixtures import write_manifest_fixture


class BuildCacheCliTests(unittest.TestCase):
    def test_manifest_error_is_recorded_in_cache_report(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp) / "release"
            root.mkdir()
            cache = Path(tmp) / "cache"
            script = Path(__file__).resolve().parents[1] / "build_cache.py"
            completed = subprocess.run(
                [
                    sys.executable,
                    str(script),
                    "--dataset-root",
                    str(root),
                    "--cache-dir",
                    str(cache),
                    "--workers",
                    "1",
                ],
                capture_output=True,
                text=True,
                check=False,
            )
            report_path = cache / "cache_build_report.json"

            self.assertNotEqual(completed.returncode, 0)
            self.assertTrue(report_path.is_file(), completed.stderr)
            report = json.loads(report_path.read_text(encoding="utf-8"))

        self.assertFalse(report["complete"])
        self.assertEqual(report["requested"], 0)
        self.assertEqual(report["failed_count"], 0)
        self.assertIn("FileNotFoundError", report["global_error"])

    def test_missing_video_is_recorded_in_cache_report(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp) / "release"
            cache = Path(tmp) / "cache"
            write_manifest_fixture(
                root, people=("P01",), actions=("walk",), trials=1
            )
            for video_path in root.glob("data/**/*.mp4"):
                video_path.unlink()
            script = Path(__file__).resolve().parents[1] / "build_cache.py"
            completed = subprocess.run(
                [
                    sys.executable,
                    str(script),
                    "--dataset-root",
                    str(root),
                    "--cache-dir",
                    str(cache),
                    "--workers",
                    "1",
                ],
                capture_output=True,
                text=True,
                check=False,
            )
            report_path = cache / "cache_build_report.json"

            self.assertNotEqual(completed.returncode, 0)
            self.assertTrue(report_path.is_file(), completed.stderr)
            report = json.loads(report_path.read_text(encoding="utf-8"))

        self.assertFalse(report["complete"])
        self.assertEqual(report["requested"], 1)
        self.assertEqual(report["completed"], 0)
        self.assertEqual(report["failed_count"], 1)
        self.assertIn("FileNotFoundError", report["failed"][0]["error"])


if __name__ == "__main__":
    unittest.main()
