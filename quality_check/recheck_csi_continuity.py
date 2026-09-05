#!/usr/bin/env python3
"""独立复验 AXHome-MM-v1 发布切片的 CSI FTM 连续性指标。

目的
----
用一条独立的代码路径重算 `outputs/technical_validation_active7024_7d7ea605_full_20260801/`
中 `AXHome-MM-v1_full_audit.md` 第 3.1 节的全部数字，并逐项对拍：

- FTM 相对 packet rate（最小/Q1/中位数/均值/Q3/P95/最大，期望中位数 98.941 Hz）
- 每文件最大 FTM gap（同上分位数，期望中位数 10.216 ms）
- gap > 100 ms 的文件数（期望 9）与 > 200 ms 的文件数（期望 1）
- 文件数（7,024）与总 packet 数（3,824,573）
- 逐文件指标与参考 `csi_audit.csv` 的差异清单

计算口径（与原报告声明一致）
----------------------------
- FeitCSI header 272 字节小端：`csi_size` @0 (u32)，`ftm_clock` @8 (u32)，
  `timestamp_us` @12 (u64)，`num_rx` @46 (u8)，`num_tx` @47 (u8)，
  `num_subcarriers` @52 (u32)。payload = 4*rx*tx*sub 字节（int16 实/虚）。
- `ftm_clock` 为 uint32 设备相对计数器，3.125 ns/tick。相邻差值按模 2^32
  计算，天然处理回绕；差值 > 2^31 视为回退（不计入 gap/时长）；差值 0 单独计数。
- ftm_elapsed_sec = 全部正向差值之和 * 3.125e-9；
  ftm_packet_rate_hz = (packet_count - 1) / ftm_elapsed_sec。
- window_packet_rate_hz = packet_count / metadata `action_duration_sec`。
- gap 告警阈值 100 ms / 严重阈值 200 ms；rate 告警区间 [80, 130] Hz。
  阈值在扫描前固定，与本次复验结果无关。

仅使用 Python 标准库。Windows 直接运行，例如：

    python quality_check\\recheck_csi_continuity.py --workers 4

冒烟（只跑 16 个文件）：

    python quality_check\\recheck_csi_continuity.py --limit 16
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import statistics
import struct
import sys
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

HEADER_SIZE = 272
TICK_MS = 3.125e-6  # 3.125 ns -> ms
TICK_S = 3.125e-9  # 3.125 ns -> s
UINT32_MOD = 1 << 32
BACKWARD_THRESHOLD = 1 << 31  # 差值超过此值视为时钟回退

GAP_ALERT_MS = 100.0
GAP_SEVERE_MS = 200.0
RATE_ALERT_LO_HZ = 80.0
RATE_ALERT_HI_HZ = 130.0

# 原报告（AXHome-MM-v1_full_audit.md 2026-08-01 轮）第 3.1 节公布的数字。
# 复验的意义就在于和这些常量对拍；修改它们等于移动靶子，请勿随手改。
REPORTED = {
    "files": 7024,
    "total_packets": 3_824_573,
    "ftm_rate_hz": {
        "min": 90.233,
        "q1": 98.636,
        "median": 98.941,
        "mean": 98.876,
        "q3": 99.135,
        "p95": 99.253,
        "max": 129.900,
    },
    "max_ftm_gap_ms": {
        "min": 10.102,
        "q1": 10.184,
        "median": 10.216,
        "mean": 13.236,
        "q3": 20.115,
        "p95": 20.334,
        "max": 211.866,
    },
    "files_with_gap_gt_100ms": 9,
    "files_with_gap_gt_200ms": 1,
}

# 逐文件浮点对拍容差（参考 csi_audit.csv 保存的是完整精度，理论上应几乎一致）
FLOAT_ATOL = 1e-6


def percentile_linear(sorted_values: list[float], pct: float) -> float:
    """线性插值分位数（与 numpy.percentile 默认方法一致）。输入必须已排序。"""
    n = len(sorted_values)
    if n == 0:
        return math.nan
    if n == 1:
        return sorted_values[0]
    rank = (n - 1) * pct / 100.0
    low = int(math.floor(rank))
    high = min(low + 1, n - 1)
    frac = rank - low
    return sorted_values[low] + frac * (sorted_values[high] - sorted_values[low])


def summarize(values: list[float]) -> dict[str, float]:
    ordered = sorted(values)
    return {
        "min": ordered[0],
        "q1": percentile_linear(ordered, 25),
        "median": percentile_linear(ordered, 50),
        "mean": statistics.fmean(values),
        "q3": percentile_linear(ordered, 75),
        "p95": percentile_linear(ordered, 95),
        "max": ordered[-1],
    }


def analyze_one(
    dataset_root: str, sample_id: str, csi_rel: str, metadata_rel: str,
    manifest_packets: int, manifest_size: int,
) -> dict:
    """解析单个 .dat 的全部 header（跳过 payload 字节），返回连续性指标。"""
    result: dict = {
        "sample_id": sample_id,
        "csi_path": csi_rel,
        "status": "ok",
        "error": "",
    }
    csi_path = Path(dataset_root) / csi_rel
    try:
        blob = csi_path.read_bytes()
    except OSError as exc:
        result.update(status="error", error=f"read_failed: {exc}")
        return result

    file_size = len(blob)
    ftm_clocks: list[int] = []
    timestamp_nonzero = 0
    num_rx = num_tx = num_sub = 0
    shape_changes = 0
    size_mismatches = 0
    offset = 0
    packet_index = 0
    while offset + HEADER_SIZE <= file_size:
        csi_size = struct.unpack_from("<I", blob, offset)[0]
        ftm_clock = struct.unpack_from("<I", blob, offset + 8)[0]
        timestamp_us = struct.unpack_from("<Q", blob, offset + 12)[0]
        rx = blob[offset + 46]
        tx = blob[offset + 47]
        sub = struct.unpack_from("<I", blob, offset + 52)[0]
        if rx <= 0 or tx <= 0 or sub <= 0:
            result.update(
                status="error",
                error=f"packet {packet_index}: invalid dims rx={rx} tx={tx} sub={sub}",
            )
            return result
        expected_size = 4 * rx * tx * sub
        if csi_size != expected_size:
            size_mismatches += 1
        if packet_index == 0:
            num_rx, num_tx, num_sub = rx, tx, sub
        elif (rx, tx, sub) != (num_rx, num_tx, num_sub):
            shape_changes += 1
        if timestamp_us != 0:
            timestamp_nonzero += 1
        if offset + HEADER_SIZE + csi_size > file_size:
            break  # trailing partial payload，由 trailing_bytes 记录
        ftm_clocks.append(ftm_clock)
        offset += HEADER_SIZE + csi_size
        packet_index += 1

    packet_count = len(ftm_clocks)
    trailing_bytes = file_size - offset

    wrap_count = 0
    zero_delta_count = 0
    backward_count = 0
    gap_gt_100 = 0
    gap_gt_200 = 0
    max_gap_ms = 0.0
    forward_ticks_sum = 0
    for prev, cur in zip(ftm_clocks, ftm_clocks[1:]):
        delta = (cur - prev) % UINT32_MOD
        if delta == 0:
            zero_delta_count += 1
            continue
        if delta > BACKWARD_THRESHOLD:
            backward_count += 1
            continue
        if cur < prev:
            wrap_count += 1
        forward_ticks_sum += delta
        gap_ms = delta * TICK_MS
        if gap_ms > max_gap_ms:
            max_gap_ms = gap_ms
        if gap_ms > GAP_ALERT_MS:
            gap_gt_100 += 1
        if gap_ms > GAP_SEVERE_MS:
            gap_gt_200 += 1

    elapsed_sec = forward_ticks_sum * TICK_S
    ftm_rate = (packet_count - 1) / elapsed_sec if elapsed_sec > 0 else math.nan

    window_rate = math.nan
    action_duration = math.nan
    try:
        with (Path(dataset_root) / metadata_rel).open("r", encoding="utf-8") as fh:
            metadata = json.load(fh)
        action_duration = float(metadata["source_sync_row"]["action_duration_sec"])
        if action_duration > 0:
            window_rate = packet_count / action_duration
    except (OSError, KeyError, ValueError) as exc:
        result["error"] = f"metadata_issue: {exc}"

    result.update(
        packet_count=packet_count,
        packet_count_match_manifest=(packet_count == manifest_packets),
        file_size_bytes=file_size,
        file_size_match_manifest=(file_size == manifest_size),
        trailing_bytes=trailing_bytes,
        num_rx=num_rx,
        num_tx=num_tx,
        num_subcarriers=num_sub,
        shape_change_count=shape_changes,
        csi_size_mismatch_count=size_mismatches,
        timestamp_us_nonzero_count=timestamp_nonzero,
        ftm_wrap_count=wrap_count,
        ftm_zero_delta_count=zero_delta_count,
        ftm_backward_count=backward_count,
        ftm_gap_gt_100ms_count=gap_gt_100,
        ftm_gap_gt_200ms_count=gap_gt_200,
        max_ftm_gap_ms=max_gap_ms,
        ftm_elapsed_sec=elapsed_sec,
        ftm_packet_rate_hz=ftm_rate,
        action_duration_sec=action_duration,
        window_packet_rate_hz=window_rate,
    )
    return result


def compare_summary(name: str, computed: dict[str, float],
                    reported: dict[str, float]) -> list[dict]:
    rows = []
    for key in ("min", "q1", "median", "mean", "q3", "p95", "max"):
        got = computed[key]
        want = reported[key]
        # 报告值保留 3 位小数，容差取 0.0005（四舍五入半单位）加 epsilon
        ok = abs(got - want) <= 0.0005 + 1e-9
        rows.append({
            "check": f"{name}.{key}", "computed": round(got, 6),
            "reported": want, "pass": ok,
        })
    return rows


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    repo_root = Path(__file__).resolve().parent.parent
    parser.add_argument(
        "--dataset-root",
        default=str(repo_root / "dataset-release" / "AXHome-MM-v1"),
        help="发布包根目录（含 data/ 与 manifests/）",
    )
    parser.add_argument(
        "--reference-dir",
        default=str(
            repo_root / "outputs"
            / "technical_validation_active7024_7d7ea605_full_20260801"
        ),
        help="2026-08-01 全量审计输出目录（用于逐文件对拍，可选）",
    )
    parser.add_argument(
        "--output-dir",
        default="",
        help="复验结果输出目录，默认 outputs/recheck_csi_continuity_<时间戳>",
    )
    parser.add_argument("--workers", type=int, default=4, help="并行进程数")
    parser.add_argument("--limit", type=int, default=0, help="只跑前 N 个文件（冒烟用）")
    parser.add_argument(
        "--samples", nargs="*", default=[],
        help="只跑 sample_id 包含任一给定子串的文件（定向复验用）",
    )
    args = parser.parse_args()

    # Windows 控制台默认 GBK，避免中文输出乱码
    for stream in (sys.stdout, sys.stderr):
        try:
            stream.reconfigure(encoding="utf-8", errors="replace")
        except (AttributeError, OSError):
            pass

    dataset_root = Path(args.dataset_root)
    manifest_path = dataset_root / "manifests" / "archive_index.csv"
    if not manifest_path.is_file():
        print(f"[FATAL] 找不到清单: {manifest_path}", file=sys.stderr)
        return 2

    with manifest_path.open("r", encoding="utf-8", newline="") as fh:
        rows = list(csv.DictReader(fh))
    if args.samples:
        rows = [row for row in rows
                if any(token in row["sample_id"] for token in args.samples)]
    if args.limit > 0:
        rows = rows[: args.limit]
    full_run = args.limit == 0 and not args.samples
    print(f"待复验文件数: {len(rows)}（清单: {manifest_path}）")

    started = time.perf_counter()
    results: list[dict] = []
    if args.workers > 1 and len(rows) > 1:
        with ProcessPoolExecutor(max_workers=args.workers) as pool:
            futures = {
                pool.submit(
                    analyze_one,
                    str(dataset_root),
                    row["sample_id"],
                    row["archive_csi_path"],
                    row["archive_metadata_path"],
                    int(row["csi_packets_written"]),
                    int(row["csi_size_bytes"]),
                ): row["sample_id"]
                for row in rows
            }
            done = 0
            for future in as_completed(futures):
                results.append(future.result())
                done += 1
                if done % 500 == 0 or done == len(rows):
                    elapsed = time.perf_counter() - started
                    print(f"  进度 {done}/{len(rows)}，已用 {elapsed:.1f}s")
    else:
        for row in rows:
            results.append(analyze_one(
                str(dataset_root), row["sample_id"], row["archive_csi_path"],
                row["archive_metadata_path"], int(row["csi_packets_written"]),
                int(row["csi_size_bytes"]),
            ))
    wall = time.perf_counter() - started
    print(f"解析完成，用时 {wall:.1f}s")

    results.sort(key=lambda item: item["sample_id"])
    errors = [r for r in results if r["status"] != "ok"]
    ok_rows = [r for r in results if r["status"] == "ok"]

    # ---- 输出目录与逐文件 CSV ----
    if args.output_dir:
        output_dir = Path(args.output_dir)
    else:
        stamp = time.strftime("%Y%m%d_%H%M%S")
        output_dir = repo_root / "outputs" / f"recheck_csi_continuity_{stamp}"
    output_dir.mkdir(parents=True, exist_ok=True)
    csv_path = output_dir / "recheck_csi_continuity.csv"
    fieldnames = list(results[0].keys())
    with csv_path.open("w", encoding="utf-8", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(results)

    # ---- 汇总统计 ----
    checks: list[dict] = []

    def add_check(name: str, computed, reported, ok: bool) -> None:
        checks.append({"check": name, "computed": computed,
                       "reported": reported, "pass": ok})

    total_packets = sum(r["packet_count"] for r in ok_rows)
    if full_run:
        add_check("文件数", len(results), REPORTED["files"],
                  len(results) == REPORTED["files"])
        add_check("总 packet 数", total_packets, REPORTED["total_packets"],
                  total_packets == REPORTED["total_packets"])

    rates = [r["ftm_packet_rate_hz"] for r in ok_rows]
    gaps = [r["max_ftm_gap_ms"] for r in ok_rows]
    rate_stats = summarize(rates)
    gap_stats = summarize(gaps)
    if full_run:
        checks.extend(compare_summary("ftm_rate_hz", rate_stats,
                                      REPORTED["ftm_rate_hz"]))
        checks.extend(compare_summary("max_ftm_gap_ms", gap_stats,
                                      REPORTED["max_ftm_gap_ms"]))

    gap100_files = [r for r in ok_rows if r["ftm_gap_gt_100ms_count"] > 0]
    gap200_files = [r for r in ok_rows if r["ftm_gap_gt_200ms_count"] > 0]
    if full_run:
        add_check("gap>100ms 文件数", len(gap100_files),
                  REPORTED["files_with_gap_gt_100ms"],
                  len(gap100_files) == REPORTED["files_with_gap_gt_100ms"])
        add_check("gap>200ms 文件数", len(gap200_files),
                  REPORTED["files_with_gap_gt_200ms"],
                  len(gap200_files) == REPORTED["files_with_gap_gt_200ms"])

    rate_outliers = [r for r in ok_rows
                     if not (RATE_ALERT_LO_HZ <= r["ftm_packet_rate_hz"]
                             <= RATE_ALERT_HI_HZ)]
    packet_mismatch = [r for r in ok_rows if not r["packet_count_match_manifest"]]
    size_mismatch = [r for r in ok_rows if not r["file_size_match_manifest"]]
    ts_files = sum(1 for r in ok_rows if r["timestamp_us_nonzero_count"] > 0)

    # ---- 与参考 csi_audit.csv 逐文件对拍 ----
    ref_path = Path(args.reference_dir) / "csi_audit.csv"
    per_file_mismatches: list[dict] = []
    if ref_path.is_file():
        with ref_path.open("r", encoding="utf-8", newline="") as fh:
            reference = {row["sample_id"]: row for row in csv.DictReader(fh)}
        float_fields = ("max_ftm_gap_ms", "ftm_elapsed_sec",
                        "ftm_packet_rate_hz", "window_packet_rate_hz")
        int_fields = ("packet_count", "ftm_wrap_count", "ftm_zero_delta_count",
                      "ftm_gap_gt_100ms_count", "ftm_gap_gt_200ms_count",
                      "timestamp_us_nonzero_count")
        for r in ok_rows:
            ref = reference.get(r["sample_id"])
            if ref is None:
                per_file_mismatches.append(
                    {"sample_id": r["sample_id"], "field": "<row>",
                     "computed": "present", "reference": "missing"})
                continue
            for field in int_fields:
                if int(ref[field]) != r[field]:
                    per_file_mismatches.append(
                        {"sample_id": r["sample_id"], "field": field,
                         "computed": r[field], "reference": ref[field]})
            for field in float_fields:
                if abs(float(ref[field]) - r[field]) > FLOAT_ATOL:
                    per_file_mismatches.append(
                        {"sample_id": r["sample_id"], "field": field,
                         "computed": r[field], "reference": ref[field]})
        if full_run:
            add_check("逐文件对拍不一致条数", len(per_file_mismatches), 0,
                      len(per_file_mismatches) == 0)
    else:
        print(f"[WARN] 未找到参考 csi_audit.csv，跳过逐文件对拍: {ref_path}")

    # ---- 控制台报告 ----
    print("\n===== 复验汇总 =====")
    print(f"解析成功 {len(ok_rows)}/{len(results)}，解析失败 {len(errors)}")
    print(f"总 packet 数: {total_packets:,}")
    print("\nFTM 相对 packet rate (Hz)：")
    print(f"  本次计算: min={rate_stats['min']:.3f} Q1={rate_stats['q1']:.3f} "
          f"median={rate_stats['median']:.3f} mean={rate_stats['mean']:.3f} "
          f"Q3={rate_stats['q3']:.3f} P95={rate_stats['p95']:.3f} "
          f"max={rate_stats['max']:.3f}")
    print("  原报告  : min=90.233 Q1=98.636 median=98.941 mean=98.876 "
          "Q3=99.135 P95=99.253 max=129.900")
    print("每文件最大 FTM gap (ms)：")
    print(f"  本次计算: min={gap_stats['min']:.3f} Q1={gap_stats['q1']:.3f} "
          f"median={gap_stats['median']:.3f} mean={gap_stats['mean']:.3f} "
          f"Q3={gap_stats['q3']:.3f} P95={gap_stats['p95']:.3f} "
          f"max={gap_stats['max']:.3f}")
    print("  原报告  : min=10.102 Q1=10.184 median=10.216 mean=13.236 "
          "Q3=20.115 P95=20.334 max=211.866")
    print(f"\ngap>100ms 文件 {len(gap100_files)} 个，"
          f"gap>200ms 文件 {len(gap200_files)} 个")
    for r in gap100_files:
        print(f"  {r['sample_id']}: max_gap={r['max_ftm_gap_ms']:.3f} ms, "
              f"rate={r['ftm_packet_rate_hz']:.3f} Hz")
    print(f"rate 超出 [{RATE_ALERT_LO_HZ:.0f},{RATE_ALERT_HI_HZ:.0f}] Hz 的文件: "
          f"{len(rate_outliers)}")
    print(f"packet 数与清单不一致: {len(packet_mismatch)}；"
          f"文件大小与清单不一致: {len(size_mismatch)}；"
          f"timestamp_us 非零的文件: {ts_files}")
    if ref_path.is_file():
        print(f"逐文件对拍不一致条数: {len(per_file_mismatches)}")
        for item in per_file_mismatches[:20]:
            print(f"  {item['sample_id']} {item['field']}: "
                  f"computed={item['computed']} reference={item['reference']}")

    if full_run:
        print("\n===== 对拍结论 =====")
        failed = [c for c in checks if not c["pass"]]
        for c in checks:
            mark = "PASS" if c["pass"] else "FAIL"
            print(f"[{mark}] {c['check']}: computed={c['computed']} "
                  f"reported={c['reported']}")
        print(f"\n总体: {len(checks) - len(failed)}/{len(checks)} 项通过"
              + ("，全部吻合。" if not failed else f"，{len(failed)} 项不吻合！"))
    else:
        failed = []
        print("\n（冒烟模式：跳过与报告常量的全量对拍断言）")

    # ---- JSON 摘要 ----
    summary = {
        "dataset_root": str(dataset_root),
        "files_parsed": len(ok_rows),
        "files_error": len(errors),
        "total_packets": total_packets,
        "wall_time_sec": round(wall, 3),
        "ftm_rate_hz": rate_stats,
        "max_ftm_gap_ms": gap_stats,
        "files_with_gap_gt_100ms": [r["sample_id"] for r in gap100_files],
        "files_with_gap_gt_200ms": [r["sample_id"] for r in gap200_files],
        "rate_out_of_range_count": len(rate_outliers),
        "packet_count_mismatch_count": len(packet_mismatch),
        "file_size_mismatch_count": len(size_mismatch),
        "timestamp_us_nonzero_file_count": ts_files,
        "per_file_mismatches": per_file_mismatches,
        "checks": checks,
        "errors": [{"sample_id": r["sample_id"], "error": r["error"]}
                   for r in errors],
    }
    json_path = output_dir / "recheck_summary.json"
    with json_path.open("w", encoding="utf-8") as fh:
        json.dump(summary, fh, ensure_ascii=False, indent=2)
    print(f"\n逐文件明细: {csv_path}")
    print(f"汇总 JSON : {json_path}")

    if errors:
        return 1
    if full_run and any(not c["pass"] for c in checks):
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
