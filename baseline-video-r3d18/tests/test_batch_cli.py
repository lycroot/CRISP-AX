from __future__ import annotations

import contextlib
import io
import sys
import unittest
from unittest.mock import patch

import run_in_domain


class BatchCliTests(unittest.TestCase):
    def test_formal_in_domain_requires_reference_assignments(self) -> None:
        stderr = io.StringIO()
        argv = [
            "run_in_domain.py",
            "--dataset-root",
            "dataset",
            "--cache-dir",
            "cache",
            "--output-dir",
            "results",
        ]
        with patch.object(sys, "argv", argv), contextlib.redirect_stderr(stderr):
            with self.assertRaises(SystemExit) as raised:
                run_in_domain.parse_args()

        self.assertEqual(raised.exception.code, 2)
        self.assertIn("--reference-assignment-root", stderr.getvalue())


if __name__ == "__main__":
    unittest.main()
