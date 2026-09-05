#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
extract_device_state_frames.py —— 为论文图 4（AXHome 无人家庭环境与设备状态示例）抽取候选帧。

按条件字符串在主索引中定位 background_idle 样本，从其发布视频里等间隔抽若干帧，
另外为每个条件生成一张联系表（contact sheet）便于快速挑图。

用法（在仓库根目录 D:\\Desktop\\AXHmm 下执行）：
    mamba activate pytorch
    python tools\\extract_device_state_frames.py
    python tools\\extract_device_state_frames.py --per-video 12 --trials all
    python tools\\extract_device_state_frames.py --conditions dry_pot_sim --per-video 20

依赖：opencv-python、Pillow、numpy
    pip install opencv-python Pillow
"""

from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

import cv2
import numpy as np
from PIL import Image, ImageDraw

# ---------------------------------------------------------------- 配置

# 图 4 四个子图对应的条件。key 为条件字符串在 sample_id 中的匹配片段，
# env 为限定环境（None 表示不限定）。顺序即图 4 的 (a)(b)(c)(d)。
TARGETS = [
    ("clean_none",          "E1", "a_E1客厅_清洁背景"),
    ("heater_cloth_front",  "E1", "b_E1客厅_暖风机前方置织物"),
    ("bucket_overflow_sim", "E3", "c_E3卫生间_容器溢水"),
    ("dry_pot_sim",         "E4", "d_E4厨房_模拟空锅加热"),
]

REPO_ROOT = Path(__file__).resolve().parent.parent
RELEASE_ROOT = REPO_ROOT / "dataset-release" / "AXHome-MM-v1"
INDEX_CSV = RELEASE_ROOT / "manifests" / "archive_index.csv"
OUT_ROOT = REPO_ROOT / "figures" / "figure_04_device_states" / "candidates"

# 窗口首尾各跳过的比例：边界帧容易赶上设备启停或镜头未稳
EDGE_SKIP = 0.08


# ---------------------------------------------------------------- 索引

def load_targets(conditions_filter=None, trials="first"):
    """从主索引里挑出目标样本。trials: 'first' 只取每个条件的第一条，'all' 取全部。"""
    if not INDEX_CSV.exists():
        sys.exit(f"[错误] 找不到主索引：{INDEX_CSV}")

    with INDEX_CSV.open(encoding="utf-8-sig", newline="") as fh:
        rows = [r for r in csv.DictReader(fh) if r["action_id"] == "background_idle"]

    picked = []
    for key, env, label in TARGETS:
        if conditions_filter and key not in conditions_filter:
            continue
        sel = [r for r in rows
               if key in r["sample_id"] and (env is None or r["environment_id"] == env)]
        sel.sort(key=lambda r: r["sample_id"])
        if not sel:
            print(f"[警告] 条件 {key}（{env}）在主索引中没有匹配样本，跳过")
            continue
        picked.append((key, label, sel if trials == "all" else sel[:1]))
        print(f"[索引] {label:28} 匹配 {len(sel)} 条，本次处理 {len(sel) if trials=='all' else 1} 条")
    return picked


# ---------------------------------------------------------------- 抽帧

def grab_frames(video_path: Path, n: int):
    """等间隔抽 n 帧，返回 [(frame_idx, seconds, BGR ndarray), ...]。"""
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        print(f"[警告] 打不开视频：{video_path}")
        return []

    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) or 0
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    if total <= 0:                       # 少数封装读不到总帧数，退化为顺序解码
        frames = []
        while True:
            ok, f = cap.read()
            if not ok:
                break
            frames.append(f)
        cap.release()
        if not frames:
            return []
        total = len(frames)
        lo, hi = int(total * EDGE_SKIP), int(total * (1 - EDGE_SKIP))
        idxs = np.linspace(lo, hi - 1, n, dtype=int)
        return [(int(i), i / fps, frames[int(i)]) for i in idxs]

    lo, hi = int(total * EDGE_SKIP), int(total * (1 - EDGE_SKIP))
    idxs = sorted(set(int(i) for i in np.linspace(lo, max(hi - 1, lo), n)))

    out, cursor = [], -1
    for want in idxs:
        frame = None
        # 先尝试 seek；部分编码 seek 不准，失败则从当前位置顺序推进
        if want > cursor and (cursor < 0 or want - cursor > 60):
            cap.set(cv2.CAP_PROP_POS_FRAMES, want)
            cursor = want - 1
        while cursor < want:
            ok, buf = cap.read()
            if not ok:
                break
            cursor += 1
            frame = buf
        if frame is None:
            print(f"[警告] 第 {want} 帧读取失败：{video_path.name}")
            continue
        out.append((want, want / fps, frame))
    cap.release()
    return out


def contact_sheet(items, out_path: Path, cols=4, thumb_w=420):
    """把候选帧拼成一张联系表，帧号和时间写在角上。"""
    if not items:
        return
    thumbs = []
    for idx, sec, bgr in items:
        rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
        im = Image.fromarray(rgb)
        im = im.resize((thumb_w, int(im.height * thumb_w / im.width)), Image.LANCZOS)
        d = ImageDraw.Draw(im)
        tag = f"#{idx}  {sec:.1f}s"
        d.rectangle([0, 0, 8 + 7 * len(tag), 22], fill=(0, 0, 0))
        d.text((5, 5), tag, fill=(255, 255, 0))
        thumbs.append(im)

    rows = (len(thumbs) + cols - 1) // cols
    w, h = thumbs[0].size
    sheet = Image.new("RGB", (cols * w, rows * h), (24, 24, 24))
    for i, im in enumerate(thumbs):
        sheet.paste(im, ((i % cols) * w, (i // cols) * h))
    out_path.parent.mkdir(parents=True, exist_ok=True)
    sheet.save(out_path)
    print(f"[联系表] {out_path.relative_to(REPO_ROOT)}")


# ---------------------------------------------------------------- 主流程

def main():
    ap = argparse.ArgumentParser(description="抽取图 4 候选帧")
    ap.add_argument("--per-video", type=int, default=8, help="每个视频抽多少帧（默认 8）")
    ap.add_argument("--trials", choices=["first", "all"], default="first",
                    help="first=每个条件只处理第一条试次，all=处理全部 5 条")
    ap.add_argument("--conditions", nargs="*", default=None,
                    help="只处理指定条件，如 --conditions dry_pot_sim bucket_overflow_sim")
    ap.add_argument("--cols", type=int, default=4, help="联系表列数")
    args = ap.parse_args()

    picked = load_targets(args.conditions, args.trials)
    if not picked:
        sys.exit("[错误] 没有匹配到任何样本")

    manifest = []
    for key, label, samples in picked:
        cond_dir = OUT_ROOT / label
        sheet_items = []
        for r in samples:
            sid = r["sample_id"]
            vid = RELEASE_ROOT.joinpath(*r["archive_video_path"].split("/"))
            if not vid.exists():
                print(f"[警告] 视频不存在：{vid}")
                continue
            print(f"[抽帧] {sid}")
            frames = grab_frames(vid, args.per_video)
            for idx, sec, bgr in frames:
                name = f"{sid}_f{idx:05d}_t{sec:05.1f}s.png"
                dst = cond_dir / name
                dst.parent.mkdir(parents=True, exist_ok=True)
                # 用 imencode 写盘，避免 imwrite 在非 ASCII 路径下静默失败
                ok, buf = cv2.imencode(".png", bgr)
                if not ok:
                    print(f"[警告] 编码失败：{name}")
                    continue
                buf.tofile(str(dst))
                manifest.append({
                    "condition": key, "panel": label, "sample_id": sid,
                    "environment_id": r["environment_id"], "frame_index": idx,
                    "seconds": f"{sec:.2f}", "png_path": str(dst.relative_to(REPO_ROOT)),
                })
            sheet_items.extend(frames)
        contact_sheet(sheet_items, OUT_ROOT / f"_sheet_{label}.png", cols=args.cols)

    mpath = OUT_ROOT / "candidates_manifest.csv"
    mpath.parent.mkdir(parents=True, exist_ok=True)
    with mpath.open("w", encoding="utf-8-sig", newline="") as fh:
        wr = csv.DictWriter(fh, fieldnames=list(manifest[0].keys()))
        wr.writeheader()
        wr.writerows(manifest)

    print(f"\n共导出 {len(manifest)} 帧 -> {OUT_ROOT.relative_to(REPO_ROOT)}")
    print(f"清单：{mpath.relative_to(REPO_ROOT)}")
    print("\n挑图提醒（M020 现场照片公开许可尚未取得）：")
    print("  · 避开窗外景观、门牌、屏幕内容和带标识的物品")
    print("  · 溢水等细节不明显时，可加放大框或箭头，但不要做会改变事件内容的图像处理")
    print("  · 四张最终图记得记回 figures/figure_04_device_states/ 并在主稿第 129 行替换占位符")


if __name__ == "__main__":
    main()
