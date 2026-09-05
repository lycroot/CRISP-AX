#!/usr/bin/env python3
"""Verify the fixed AXHome multi-architecture result matrix and bundle evidence."""

from __future__ import annotations

import argparse
import csv
import json
import math
import tarfile
from pathlib import Path
from typing import Any


SEEDS = (2026, 2027, 2028, 2029, 2030)
LOSO_PAIRS = tuple(
    (f"P{index:02d}", f"P{(index % 10) + 1:02d}")
    for index in range(1, 11)
)
REQUIRED_RUN_FILES = (
    "run_config.json",
    "metrics.json",
    "history.csv",
    "classification_report.csv",
    "predictions.csv",
    "confusion_matrix_counts.csv",
)
EXPECTED_PARAMETERS = {
    "cnn2d": 161_934,
    "resnet18": 11_180_558,
    "bilstm": 696_718,
    "transformer": 696_206,
    "r3d_18": 33_173_454,
    "mc3_18": 11_497_422,
    "r2plus1d_18": 31_307_307,
}
FIRST_ROUND_GROUPS = (
    ("in_domain_resnet18", "resnet18", "csi", "in_domain"),
    ("loso_resnet18", "resnet18", "csi", "loso"),
    ("in_domain_bilstm", "bilstm", "csi", "in_domain"),
    ("loso_bilstm", "bilstm", "csi", "loso"),
    ("in_domain_transformer", "transformer", "csi", "in_domain"),
    ("loso_transformer", "transformer", "csi", "loso"),
    ("video_in_domain_mc3_18", "mc3_18", "video", "in_domain"),
    ("video_loso_mc3_18", "mc3_18", "video", "loso"),
    (
        "video_in_domain_r2plus1d_18",
        "r2plus1d_18",
        "video",
        "in_domain",
    ),
    ("video_loso_r2plus1d_18", "r2plus1d_18", "video", "loso"),
)
V2_GROUPS = (
    ("in_domain_cnn2d", "cnn2d", "csi", "in_domain"),
    ("loso_cnn2d", "cnn2d", "csi", "loso"),
    ("in_domain_resnet18", "resnet18", "csi", "in_domain"),
    ("loso_resnet18", "resnet18", "csi", "loso"),
    ("in_domain_bilstm", "bilstm", "csi", "in_domain"),
    ("loso_bilstm", "bilstm", "csi", "loso"),
    ("in_domain_transformer", "transformer", "csi", "in_domain"),
    ("loso_transformer", "transformer", "csi", "loso"),
    ("video_in_domain_r3d_18", "r3d_18", "video", "in_domain"),
    ("video_loso_r3d_18", "r3d_18", "video", "loso"),
    ("video_in_domain_mc3_18", "mc3_18", "video", "in_domain"),
    ("video_loso_mc3_18", "mc3_18", "video", "loso"),
    ("video_in_domain_r2plus1d_18", "r2plus1d_18", "video", "in_domain"),
    ("video_loso_r2plus1d_18", "r2plus1d_18", "video", "loso"),
)
SMOKE_GROUPS = tuple(
    (f"_smoke/{modality}_{arch}", arch, modality, "smoke")
    for arch, modality in (
        ("resnet18", "csi"), ("bilstm", "csi"),
        ("transformer", "csi"), ("mc3_18", "video"),
        ("r2plus1d_18", "video"),
    )
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Verify a fixed AXHome multi-architecture result matrix"
    )
    parser.add_argument("--results-root", required=True, type=Path)
    parser.add_argument(
        "--matrix",
        choices=("first-round", "v2"),
        default="first-round",
        help="Matrix to verify (default preserves the historical 75-run workflow)",
    )
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--group", action="append", help="Verify only the named group(s)")
    parser.add_argument(
        "--report",
        type=Path,
        default=None,
        help="JSON report path; defaults inside results-root",
    )
    parser.add_argument(
        "--bundle",
        action="store_true",
        help="Create a tar.gz containing all JSON/CSV evidence after success",
    )
    return parser.parse_args()


def load_json(path: Path) -> dict[str, Any]:
    with path.open(encoding="utf-8") as handle:
        value = json.load(handle)
    if not isinstance(value, dict):
        raise ValueError(f"JSON root must be an object: {path}")
    return value


def add_error(errors: list[str], condition: bool, message: str) -> None:
    if not condition:
        errors.append(message)


def finite_json_numbers(value: Any) -> bool:
    if isinstance(value, bool) or value is None or isinstance(value, str):
        return True
    if isinstance(value, (int, float)):
        return math.isfinite(float(value))
    if isinstance(value, list):
        return all(finite_json_numbers(item) for item in value)
    if isinstance(value, dict):
        return all(finite_json_numbers(item) for item in value.values())
    return True


def csv_has_column(path: Path, column: str) -> bool:
    with path.open(newline="", encoding="utf-8-sig") as handle:
        reader = csv.reader(handle)
        header = next(reader, [])
    return column in header


def nested_get(value: dict[str, Any], *keys: str) -> Any:
    current: Any = value
    for key in keys:
        if not isinstance(current, dict) or key not in current:
            return None
        current = current[key]
    return current


def run_directories(group: Path, protocol: str) -> tuple[Path, ...]:
    if protocol == "smoke":
        return (group,)
    if protocol == "in_domain":
        return tuple(group / f"seed_{seed}" for seed in SEEDS)
    return tuple(group / f"test_{test}_val_{val}" for test, val in LOSO_PAIRS)


def verify_group(
    results_root: Path,
    name: str,
    arch: str,
    modality: str,
    protocol: str,
    *,
    matrix: str,
) -> dict[str, Any]:
    group = results_root / name
    errors: list[str] = []
    add_error(errors, group.is_dir(), f"missing result directory: {group}")
    if not group.is_dir():
        return {"name": name, "architecture": arch, "ok": False, "errors": errors}

    summary_name = (
        "in_domain_summary.json" if protocol == "in_domain" else "loso_summary.json"
    )
    table_name = "in_domain_runs.csv" if protocol == "in_domain" else "loso_folds.csv"
    summary_path = group / summary_name
    table_path = group / table_name
    if protocol != "smoke":
        add_error(errors, summary_path.is_file(), f"missing summary: {summary_path}")
        add_error(errors, table_path.is_file(), f"missing table: {table_path}")

    if protocol != "smoke" and summary_path.is_file():
        try:
            summary = load_json(summary_path)
            add_error(
                errors,
                summary.get("architecture") == arch,
                f"{summary_path}: architecture must equal {arch}",
            )
            add_error(
                errors,
                finite_json_numbers(summary),
                f"{summary_path}: contains NaN or Inf",
            )
            if protocol == "in_domain":
                add_error(
                    errors,
                    summary.get("complete") is True,
                    f"{summary_path}: complete must be true",
                )
                add_error(
                    errors,
                    sorted(summary.get("completed_seeds", [])) == list(SEEDS),
                    f"{summary_path}: completed_seeds must be exactly {list(SEEDS)}",
                )
                add_error(
                    errors,
                    not summary.get("failed_seeds"),
                    f"{summary_path}: failed_seeds must be empty",
                )
            else:
                if modality == "video":
                    add_error(
                        errors,
                        summary.get("complete") is True,
                        f"{summary_path}: complete must be true",
                    )
                    add_error(
                        errors,
                        summary.get("completed_test_people")
                        == [test for test, _ in LOSO_PAIRS],
                        f"{summary_path}: completed_test_people must contain P01-P10 in order",
                    )
                    add_error(
                        errors,
                        summary.get("num_completed") == 10,
                        f"{summary_path}: num_completed must equal 10",
                    )
                else:
                    add_error(
                        errors,
                        summary.get("num_folds") == 10,
                        f"{summary_path}: num_folds must equal 10",
                    )
                add_error(
                    errors,
                    not summary.get("failed_folds"),
                    f"{summary_path}: failed_folds must be empty",
                )
            if matrix == "v2" and modality == "video":
                for metric in ("accuracy", "macro_f1"):
                    mean = nested_get(summary, metric, "mean")
                    add_error(
                        errors,
                        isinstance(mean, (int, float)) and float(mean) >= 0.95,
                        f"{summary_path}: {metric}.mean fell below the 0.95 stop line",
                    )
            if matrix == "v2" and arch == "cnn2d" and protocol == "loso":
                mean = nested_get(summary, "macro_f1", "mean")
                add_error(
                    errors,
                    isinstance(mean, (int, float)) and float(mean) >= 0.2,
                    f"{summary_path}: macro_f1.mean fell below the 0.2 stop line",
                )
        except (OSError, ValueError, json.JSONDecodeError) as error:
            errors.append(f"cannot read {summary_path}: {error}")

    if protocol != "smoke" and table_path.is_file():
        try:
            add_error(
                errors,
                csv_has_column(table_path, "arch"),
                f"{table_path}: missing arch column",
            )
        except OSError as error:
            errors.append(f"cannot read {table_path}: {error}")

    expected_parameters = EXPECTED_PARAMETERS[arch]
    for run_dir in run_directories(group, protocol):
        add_error(errors, run_dir.is_dir(), f"missing run directory: {run_dir}")
        if not run_dir.is_dir():
            continue
        for filename in REQUIRED_RUN_FILES:
            add_error(
                errors,
                (run_dir / filename).is_file(),
                f"missing run artifact: {run_dir / filename}",
            )

        config_path = run_dir / "run_config.json"
        if config_path.is_file():
            try:
                config = load_json(config_path)
                if modality == "csi":
                    recorded_arch = nested_get(config, "model", "arch")
                    parameters = nested_get(config, "model", "trainable_parameters")
                else:
                    recorded_arch = config.get("arch")
                    parameters = config.get("trainable_parameters")
                    add_error(errors, config.get("model") == f"torchvision.models.video.{arch}",
                              f"{config_path}: unexpected model builder")
                    checkpoint_hash = nested_get(
                        config, "runtime", "pretrained_checkpoint", "sha256"
                    )
                    add_error(
                        errors,
                        isinstance(checkpoint_hash, str) and len(checkpoint_hash) == 64
                        and all(char in "0123456789abcdef" for char in checkpoint_hash),
                        f"{config_path}: pretrained checkpoint sha256 is empty",
                    )
                add_error(
                    errors,
                    recorded_arch == arch,
                    f"{config_path}: recorded architecture must equal {arch}",
                )
                add_error(
                    errors,
                    parameters == expected_parameters,
                    f"{config_path}: trainable_parameters must equal {expected_parameters}",
                )
                if matrix == "v2":
                    device_name = nested_get(config, "runtime", "accelerator", "device_name")
                    add_error(
                        errors,
                        isinstance(device_name, str) and bool(device_name.strip()),
                        f"{config_path}: runtime.accelerator.device_name is empty",
                    )
                expected_training = {
                    "epochs": 1 if protocol == "smoke" else 50,
                    "patience": 1 if protocol == "smoke" else 10,
                    "batch_size": 32 if modality == "csi" else 16,
                    "learning_rate": (3e-4 if arch == "transformer" else 1e-3)
                    if modality == "csi" else 1e-4,
                    "weight_decay": 0.0 if modality == "csi" else 1e-4,
                    "num_workers": 8,
                    "arch": arch,
                }
                if modality == "csi":
                    expected_training["dropout"] = 0.1 if arch == "transformer" else 0.5
                    expected_cache = {"target_packets": 256, "target_subcarriers": 128, "low_energy_ratio": 0.05}
                else:
                    expected_training.update(amp=True, pretrained=True)
                    expected_cache = {"frames": 16, "height": 128, "width": 171, "color_space": "RGB"}
                    if protocol == "in_domain":
                        add_error(errors, bool(config.get("reference_assignment_path")),
                                  f"{config_path}: missing reference_assignment_path")
                for key, expected in expected_training.items():
                    add_error(errors, nested_get(config, "training_config", key) == expected,
                              f"{config_path}: training_config.{key} must equal {expected}")
                for key, expected in expected_cache.items():
                    add_error(errors, nested_get(config, "cache_config", key) == expected,
                              f"{config_path}: cache_config.{key} must equal {expected}")
            except (OSError, ValueError, json.JSONDecodeError) as error:
                errors.append(f"cannot read {config_path}: {error}")

        metrics_path = run_dir / "metrics.json"
        if metrics_path.is_file():
            try:
                metrics = load_json(metrics_path)
                add_error(
                    errors,
                    finite_json_numbers(metrics),
                    f"{metrics_path}: contains NaN or Inf",
                )
            except (OSError, ValueError, json.JSONDecodeError) as error:
                errors.append(f"cannot read {metrics_path}: {error}")

    return {
        "name": name,
        "architecture": arch,
        "modality": modality,
        "protocol": protocol,
        "ok": not errors,
        "errors": errors,
    }


def runtime_consistency(
    results_root: Path,
    groups: tuple[tuple[str, str, str, str], ...],
) -> dict[str, Any]:
    signatures: dict[str, int] = {}
    for name, _arch, _modality, protocol in groups:
        for run_dir in run_directories(results_root / name, protocol):
            config_path = run_dir / "run_config.json"
            if not config_path.is_file():
                continue
            config = load_json(config_path)
            signature = {
                "python": nested_get(config, "runtime", "software_versions", "python"),
                "numpy": nested_get(config, "runtime", "software_versions", "numpy"),
                "torch": nested_get(config, "runtime", "software_versions", "torch"),
                "cuda_runtime": nested_get(config, "runtime", "accelerator", "cuda_runtime"),
                "cudnn": nested_get(config, "runtime", "accelerator", "cudnn"),
                "device_name": nested_get(config, "runtime", "accelerator", "device_name"),
            }
            key = json.dumps(signature, sort_keys=True)
            signatures[key] = signatures.get(key, 0) + 1
    parsed = [dict(json.loads(key), run_count=count) for key, count in signatures.items()]
    return {
        "complete": len(parsed) == 1 and parsed[0]["run_count"] == sum(
            5 if group[3] == "in_domain" else 10 for group in groups
        ),
        "signatures": parsed,
    }


def write_bundle(
    results_root: Path,
    report_path: Path,
    groups: tuple[tuple[str, str, str, str], ...],
    *,
    matrix: str,
) -> Path:
    filename = (
        "axhome_multiarc_v2_json_csv_evidence.tar.gz"
        if matrix == "v2"
        else "axhome_multiarc_json_csv_evidence.tar.gz"
    )
    bundle = results_root / filename
    with tarfile.open(bundle, "w:gz") as archive:
        for group_name, _, _, _ in groups:
            group = results_root / group_name
            for path in sorted(group.rglob("*")):
                if path.is_file() and path.suffix.lower() in {".json", ".csv"}:
                    archive.add(path, arcname=path.relative_to(results_root))
        try:
            report_arcname = report_path.relative_to(results_root)
        except ValueError:
            report_arcname = Path(report_path.name)
        archive.add(report_path, arcname=report_arcname)
    return bundle


def main() -> None:
    args = parse_args()
    results_root = args.results_root.resolve()
    if args.bundle and (args.smoke or args.group):
        raise SystemExit("--bundle requires the full formal matrix")
    if args.smoke and args.matrix == "v2":
        raise SystemExit("the v2 workflow does not define smoke groups")
    formal_groups = V2_GROUPS if args.matrix == "v2" else FIRST_ROUND_GROUPS
    groups = SMOKE_GROUPS if args.smoke else formal_groups
    if args.group:
        unknown = set(args.group) - {group[0] for group in groups}
        if unknown:
            raise SystemExit(f"unknown group(s): {sorted(unknown)}")
        groups = tuple(group for group in groups if group[0] in args.group)
    report_name = "smoke_acceptance_report.json" if args.smoke else "acceptance_report.json"
    if args.group:
        report_name = "_".join(name.replace("/", "_") for name in args.group) + "_acceptance.json"
    report_path = (
        args.report.resolve()
        if args.report is not None
        else results_root / report_name
    )
    results = [
        verify_group(results_root, *group, matrix=args.matrix) for group in groups
    ]
    consistency = (
        runtime_consistency(results_root, groups)
        if args.matrix == "v2"
        else {"complete": True, "signatures": []}
    )
    report = {
        "results_root": str(results_root),
        "matrix": args.matrix,
        "expected_runs": sum(1 if group[3] == "smoke" else 5 if group[3] == "in_domain" else 10 for group in groups),
        "smoke": args.smoke,
        "groups": results,
        "runtime_consistency": consistency,
        "complete": all(result["ok"] for result in results) and consistency["complete"],
    }
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(report_path)
    if not report["complete"]:
        for result in results:
            for error in result["errors"]:
                print(f"ERROR [{result['name']}] {error}")
        if not consistency["complete"]:
            print(f"ERROR [runtime] inconsistent runtime signatures: {consistency['signatures']}")
        raise SystemExit(1)
    if args.bundle:
        print(write_bundle(results_root, report_path, groups, matrix=args.matrix))


if __name__ == "__main__":
    main()
