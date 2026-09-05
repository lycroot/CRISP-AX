#!/usr/bin/env python3
"""核验 FeitCSI .dat 的真实维度口径：996 个子载波到底是"每天线对"还是"合计"。

只依赖 numpy，不需要 axhome_csi 包，可以单独拿去任何机器上跑。

用法
----
    python check_csi_dims.py <文件或目录> [--files N] [--packets N]

例子
----
    # 抽查发布包里的无人 CSI
    python check_csi_dims.py "D:/Desktop/AXHmm/dataset-release/AXHome-MM-v1/data/S01/csi"

    # 只看一个文件、读前 200 包
    python check_csi_dims.py "D:/.../S01_NONE_background_idle_face_rx_E1_clean_none_R001.dat" --packets 200

输出会回答三件事
----------------
1. num_rx / num_tx / num_subcarriers 各是多少（子载波数是每个 rx-tx 对的）
2. 每个数据包一共多少个复数 CSI 值 = rx × tx × subcarriers
3. 实测包率（由 timestamp 差值推算），核对正文里的"约 100 Hz"
"""

from __future__ import annotations

import argparse
import struct
import sys
from pathlib import Path

import numpy as np

HEADER_SIZE = 272  # 与 axhome_csi/feitcsi.py 一致


def parse_header(raw: bytes) -> dict:
    """按 FeitCSI 的小端 272 字节头解析，字段偏移与 axhome_csi/feitcsi.py 相同。"""
    return {
        "csi_size": struct.unpack_from("<I", raw, 0)[0],
        "ftm_clock": struct.unpack_from("<I", raw, 8)[0],
        "timestamp": struct.unpack_from("<Q", raw, 12)[0],
        "num_rx": raw[46],
        "num_tx": raw[47],
        "num_subcarriers": struct.unpack_from("<I", raw, 52)[0],
        "rssi1": struct.unpack_from("<I", raw, 60)[0],
        "rssi2": struct.unpack_from("<I", raw, 64)[0],
    }


def inspect(path: Path, max_packets: int) -> dict | None:
    size = path.stat().st_size
    shapes: set[tuple[int, int, int]] = set()
    timestamps: list[int] = []

    with path.open("rb") as fh:
        first = fh.read(HEADER_SIZE)
        if len(first) < HEADER_SIZE:
            print(f"  ✗ {path.name}: 文件太小，读不到完整包头")
            return None
        h0 = parse_header(first)
        record = HEADER_SIZE + h0["csi_size"]
        fh.seek(0)

        n = 0
        while n < max_packets:
            raw = fh.read(HEADER_SIZE)
            if len(raw) < HEADER_SIZE:
                break
            h = parse_header(raw)
            payload = fh.read(h["csi_size"])
            if len(payload) < h["csi_size"]:
                print(f"  ✗ {path.name}: 第 {n} 包载荷被截断")
                break
            shapes.add((h["num_rx"], h["num_tx"], h["num_subcarriers"]))
            timestamps.append(h["timestamp"])
            n += 1

    rx, tx, sub = h0["num_rx"], h0["num_tx"], h0["num_subcarriers"]
    expected_payload = 4 * rx * tx * sub          # int16 实部 + int16 虚部
    divisible = (size % record == 0)
    total_packets = size / record

    print(f"  {path.name}")
    print(f"    包头声明        : num_rx={rx}  num_tx={tx}  num_subcarriers={sub}")
    print(f"    载荷字节        : csi_size={h0['csi_size']}  "
          f"（4×rx×tx×sub = {expected_payload}）"
          f"{'  ✅一致' if h0['csi_size'] == expected_payload else '  ❌不一致'}")
    print(f"    单包记录长度    : {HEADER_SIZE} + {h0['csi_size']} = {record} 字节")
    print(f"    文件大小        : {size:,} 字节 ÷ {record} = {total_packets:.3f}"
          f"{'  ✅整除' if divisible else '  ❌不整除（文件可能被截断）'}")
    print(f"    每包 CSI 值个数 : rx×tx×sub = {rx * tx * sub} 个复数")
    print(f"    读到的前 {n} 包维度: {sorted(shapes)}"
          f"{'  ✅全程一致' if len(shapes) == 1 else '  ❌中途变了'}")

    if len(timestamps) > 10:
        d = np.diff(np.array(timestamps, dtype=np.int64))
        d = d[d > 0]
        if d.size:
            med = float(np.median(d))
            # timestamp 单位未在格式中声明，按微秒/纳秒各推一次，取落在合理区间的
            guesses = []
            for unit, scale in (("µs", 1e6), ("ns", 1e9), ("ms", 1e3)):
                rate = scale / med
                if 1 <= rate <= 20000:
                    guesses.append(f"{rate:.2f} Hz（若 timestamp 单位为 {unit}）")
            print(f"    timestamp 中位间隔: {med:.0f}   推算包率: "
                  f"{'; '.join(guesses) if guesses else '无法判断单位'}")

    return {"rx": rx, "tx": tx, "sub": sub, "record": record,
            "divisible": divisible, "packets": total_packets}


def main() -> int:
    ap = argparse.ArgumentParser(description="核验 FeitCSI .dat 的维度口径")
    ap.add_argument("target", help=".dat 文件，或含 .dat 的目录")
    ap.add_argument("--files", type=int, default=5, help="目录模式下抽查几个文件（默认 5）")
    ap.add_argument("--packets", type=int, default=50, help="每个文件读前几包（默认 50）")
    args = ap.parse_args()

    target = Path(args.target)
    if target.is_dir():
        files = sorted(target.rglob("*.dat"))[: args.files]
        if not files:
            print(f"目录里没找到 .dat：{target}")
            return 1
        print(f"目录：{target}\n抽查 {len(files)} 个文件，每个读前 {args.packets} 包\n")
    elif target.is_file():
        files = [target]
        print()
    else:
        print(f"路径不存在：{target}")
        return 1

    results = [r for r in (inspect(f, args.packets) for f in files) if r]
    if not results:
        return 1

    print("\n" + "=" * 62)
    combos = {(r["rx"], r["tx"], r["sub"]) for r in results}
    if len(combos) == 1:
        rx, tx, sub = combos.pop()
        pts = rx * tx * sub
        print(f"结论：所有抽查文件一致 —— num_rx={rx}, num_tx={tx}, num_subcarriers={sub}")
        print()
        print(f"  ➜ {sub} 是【每个 rx–tx 天线对】的子载波数，")
        print(f"    与综述 'N sub-carriers per antenna per spatial stream' 同口径。")
        print(f"  ➜ 每个数据包共 {rx}×{tx}×{sub} = {pts} 个复数 CSI 值。")
        print()
        print(f"  正文与表 1、表 2 可写：{sub} 个子载波（每接收天线）；")
        print(f"  若要给总量，写：每包 {pts} 个信道估计值。")
    else:
        print(f"⚠️ 抽查到多种维度组合：{sorted(combos)}")
        print("   说明不同文件的采集配置不一致，正文不能用单一数字概括，需逐组说明。")

    if not all(r["divisible"] for r in results):
        print("\n⚠️ 有文件大小不能被单包长度整除，可能存在截断，建议单独检查。")
    return 0


if __name__ == "__main__":
    sys.exit(main())
