#!/usr/bin/env python3
"""Read-only integrity gate for the three official Kinetics-400 checkpoints."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import torch
from torchvision.models.video import MC3_18_Weights, R2Plus1D_18_Weights, R3D_18_Weights


# Sizes from the official HTTPS Content-Length responses; full hashes verified
# against the SHA-256 prefixes embedded in torchvision's official weight URLs.
WEIGHTS = (
    (
        R3D_18_Weights.KINETICS400_V1,
        "r3d_18-b3b3357e.pth",
        133_546_016,
        "b3b3357ead25631ec9c57362ff2128a92d0427e01e2cd184951a44380c3f2e9d",
    ),
    (
        MC3_18_Weights.KINETICS400_V1,
        "mc3_18-a90a0ba3.pth",
        46_841_888,
        "a90a0ba35ca1242d15b77511ff28bfb29cc596988b5ea36081042f8e2f54212b",
    ),
    (
        R2Plus1D_18_Weights.KINETICS400_V1,
        "r2plus1d_18-91a641e6.pth",
        126_162_996,
        "91a641e6c2ab531d1aca5f4321b4d802ec5c3babc15df855cdb6e39c6a1107c8",
    ),
)


def main() -> None:
    cache = Path(torch.hub.get_dir()) / "checkpoints"
    records = []
    for weights, filename, size, expected_hash in WEIGHTS:
        url = f"https://download.pytorch.org/models/{filename}"
        if weights.url != url:
            raise SystemExit(f"Unexpected torchvision weight URL: {weights.url}")
        path = cache / filename
        if not path.is_file() or path.stat().st_size != size:
            raise SystemExit(f"Missing or incomplete official checkpoint: {path}")
        digest = hashlib.sha256()
        with path.open("rb") as handle:
            for block in iter(lambda: handle.read(1024 * 1024), b""):
                digest.update(block)
        actual = digest.hexdigest()
        if actual != expected_hash:
            raise SystemExit(f"Checkpoint SHA-256 mismatch: {path}: {actual}")
        records.append({"path": str(path), "url": url, "bytes": size, "sha256": actual})
    print(json.dumps({"checkpoints": records, "complete": True}, indent=2), flush=True)
    print(f"CHECKPOINT_ACCEPTANCE_PASSED checkpoints={len(records)}", flush=True)


if __name__ == "__main__":
    main()
