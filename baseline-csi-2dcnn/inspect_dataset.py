from __future__ import annotations

import argparse
import json
from pathlib import Path

from axhome_csi.data import load_manifest
from axhome_csi.profile import summarize_samples


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="检查 AXHome-MM-v1 发布清单分布")
    parser.add_argument("--dataset-root", required=True)
    parser.add_argument("--output", default=None, help="可选 JSON 输出路径")
    parser.add_argument(
        "--validate-paths", action="store_true", help="同时检查每个 CSI 文件是否存在"
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    samples = load_manifest(args.dataset_root, validate_paths=args.validate_paths)
    summary = summarize_samples(samples)
    rendered = json.dumps(summary, ensure_ascii=False, indent=2) + "\n"
    print(rendered, end="")
    if args.output:
        output = Path(args.output)
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(rendered, encoding="utf-8")


if __name__ == "__main__":
    main()

