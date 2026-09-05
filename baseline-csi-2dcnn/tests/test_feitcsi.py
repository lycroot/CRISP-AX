from __future__ import annotations

import tempfile
import unittest
from io import BytesIO
from pathlib import Path

import numpy as np

from axhome_csi.feitcsi import (
    FeitCSIFormatError,
    read_feitcsi,
    read_feitcsi_stream,
)
from tests._fixtures import write_feitcsi


class FeitCSIReaderTests(unittest.TestCase):
    def test_reads_complex_packets_in_documented_order(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "sample.dat"
            expected = write_feitcsi(path, packets=3, subcarriers=6)

            csi, headers = read_feitcsi(path)

        self.assertEqual(csi.shape, (3, 2, 1, 6))
        self.assertTrue(np.array_equal(csi, expected))
        self.assertEqual(len(headers), 3)
        self.assertEqual(headers[0].csi_size, 48)
        self.assertEqual(headers[0].num_subcarriers, 6)
        self.assertEqual(headers[0].num_rx, 2)
        self.assertEqual(headers[0].num_tx, 1)

    def test_rejects_truncated_packet(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "truncated.dat"
            write_feitcsi(path, packets=1, subcarriers=6)
            with path.open("ab") as handle:
                handle.write(b"partial")

            with self.assertRaisesRegex(FeitCSIFormatError, "truncated header"):
                read_feitcsi(path)

    def test_reads_from_zip_compatible_binary_stream(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "stream.dat"
            expected = write_feitcsi(path, packets=2, subcarriers=5)
            stream = BytesIO(path.read_bytes())

            csi, headers = read_feitcsi_stream(stream, source="fixture.zip/sample.dat")

        self.assertTrue(np.array_equal(csi, expected))
        self.assertEqual(len(headers), 2)

    def test_rejects_header_payload_shape_mismatch(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "bad_shape.dat"
            write_feitcsi(path, packets=1, subcarriers=6)
            data = bytearray(path.read_bytes())
            data[46] = 1
            path.write_bytes(data)

            with self.assertRaisesRegex(FeitCSIFormatError, "payload size"):
                read_feitcsi(path)


if __name__ == "__main__":
    unittest.main()
