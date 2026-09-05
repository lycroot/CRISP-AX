#!/usr/bin/env python3
"""Verify source coordinates and render B/facet review sheets without resampling data.

Run as a module from the repository root in the pytorch environment.
"""

from __future__ import annotations

import argparse
import hashlib
from io import BytesIO
import json
from pathlib import Path
import re
import subprocess
from unittest.mock import patch

import matplotlib
import matplotlib.pyplot as plt
from matplotlib.text import Text
import numpy as np
import pandas as pd
from PIL import Image, ImageChops, ImageDraw, ImageFont

from quality_check import unoccupied_environment_validation as generator
from quality_check.figure7_palette_comparison import simulate_image


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def capture_figure(frames: dict, layout: str):
    figures = []
    with patch.object(generator, "save_figure_triplet", side_effect=lambda fig, *_: figures.append(fig)):
        generator.plot_figure7_condition_distance(
            **frames, output_dir=Path("."), dpi=600,
            layout_variant=layout, palette_variant="p1",
        )
    assert len(figures) == 1
    return figures[0]


def verify_coordinates(frames: dict) -> dict:
    fig = capture_figure(frames, "facets")
    ordered = generator._figure7_variant_frame(
        frames["state_distance_frame"], frames["nearest_frame"]
    )
    assert len(fig.axes) == 4
    actual_widths = []
    plotted_codes = []
    expected_ratios = []
    for axis, room_id in zip(fig.axes[:3], generator.ENVIRONMENT_LABELS):
        room = ordered.loc[ordered["environment_id"] == room_id]
        actual_widths.append(axis.get_position().width * 7.2)
        assert axis.get_yscale() == "log" and axis.get_ylim() == (3.5, 55.0)
        for role, mean_column, sd_column in (
            ("within", "mean_intra_distance", "std_intra_distance"),
            ("between", "mean_inter_distance", "std_inter_distance"),
        ):
            mark = next(line for line in axis.lines if line.get_gid() == f"{role}-means")
            np.testing.assert_array_equal(mark.get_ydata(), room[mean_column])
            intervals = next(c for c in axis.collections if c.get_gid() == f"{role}-source-sd")
            endpoints = np.asarray(intervals.get_segments())[:, :, 1]
            np.testing.assert_array_equal(endpoints[:, 0], room[mean_column] - room[sd_column])
            np.testing.assert_array_equal(endpoints[:, 1], room[mean_column] + room[sd_column])
        nearest = next(c for c in axis.collections if c.get_gid() == "nearest-source-means")
        np.testing.assert_array_equal(nearest.get_offsets()[:, 1], room["mean_nearest_inter_distance"])
        plotted_codes.extend(label.get_text() for label in axis.get_xticklabels())
        expected_ratios.extend(f"{ratio:.2f}" for ratio in room["nearest_state_ratio"])
    np.testing.assert_allclose(actual_widths, [1.84, 2.76, 1.38], rtol=0, atol=1e-12)
    expected_codes = [generator.paper_state_code(str(s)) for s in ordered["state"]]
    assert plotted_codes == expected_codes
    actual_ratios = [t.get_text() for t in fig.texts if t.get_text() in expected_ratios]
    assert actual_ratios == expected_ratios
    within, between = generator._figure7_distance_arrays(frames["pair_frame"])
    pooled = fig.axes[3]
    assert pooled.get_xscale() == "linear" and pooled.get_xlim() == (0.0, 55.0)
    for role, values in (("within", within), ("between", between)):
        line = next(line for line in pooled.lines if line.get_gid() == f"{role}-ecdf")
        np.testing.assert_array_equal(line.get_xdata(), np.r_[0.0, np.sort(values)])
        np.testing.assert_array_equal(line.get_ydata(), np.r_[0.0, np.arange(1, len(values)+1)/len(values)])
        assert line.get_drawstyle() == "steps-post"
        assert line.get_linestyle() == "-" and line.get_linewidth() == 1.0
        assert line.get_solid_capstyle() == "butt" and line.get_solid_joinstyle() == "miter"
    assert pooled.get_legend() is None and not pooled.collections
    assert [t.get_text() for t in pooled.texts] == ["Within  (n = 126)", "Between  (n = 585)"]
    fig.canvas.draw()
    texts = [t for t in fig.findobj(Text) if t.get_visible() and t.get_text()]
    assert min(t.get_fontsize() for t in texts) >= 6.0
    renderer = fig.canvas.get_renderer()
    outside = []
    for text in texts:
        bounds = text.get_window_extent(renderer)
        if bounds.x0 < 0 or bounds.y0 < 0 or bounds.x1 > fig.bbox.width or bounds.y1 > fig.bbox.height:
            outside.append(text.get_text())
    assert not outside, outside
    plt.close(fig)
    return {"source_means_and_sd_endpoints_exact": True,
            "source_nearest_means_and_ratios_exact": True,
            "condition_order": expected_codes, "facet_widths_inches": actual_widths,
            "pair_counts": [len(within), len(between)], "ecdf_steps_and_style_exact": True,
            "minimum_visible_font_pt": min(t.get_fontsize() for t in texts),
            "text_outside_canvas": outside}


def verify_b_unchanged(frames: dict, baseline: Path) -> bool:
    fig = capture_figure(frames, "b")
    buffer = BytesIO()
    with matplotlib.rc_context({"savefig.bbox": None}):
        fig.savefig(buffer, format="png", dpi=600, facecolor="white", bbox_inches=None)
    plt.close(fig)
    buffer.seek(0)
    with Image.open(buffer) as candidate, Image.open(baseline) as reference:
        assert candidate.size == reference.size
        # RGB is intentional: RGBA.getbbox can hide RGB changes behind zero alpha differences.
        assert ImageChops.difference(candidate.convert("RGB"), reference.convert("RGB")).getbbox() is None
    return True


def write_comparison(
    baseline: Path, candidate: Path, output: Path,
    candidate_label: str = "Room facets / P1",
) -> None:
    font_path = generator.resolve_publication_font()[1]
    font = ImageFont.truetype(font_path, 23)
    small_font = ImageFont.truetype(font_path, 20)
    width = round(40 / 25.4 * 300)
    thumbnails = []
    for path in (baseline, candidate):
        with Image.open(path) as image:
            thumbnails.append(image.convert("RGB").resize(
                (width, round(width * image.height / image.width)), Image.Resampling.LANCZOS
            ))
    row_height = max(im.height for im in thumbnails)
    modes = ("original", "grayscale", "deuteranopia", "protanopia", "tritanopia")
    left, gap, top = 145, 35, 50
    sheet = Image.new("RGB", (left + 2 * width + 2 * gap, top + len(modes)*(row_height+25)), "white")
    draw = ImageDraw.Draw(sheet)
    for index, label in enumerate(("B / P1", candidate_label)):
        draw.text((left + index*(width+gap) + width/2, 15), label, fill="#333333", font=font, anchor="mt")
    for row, mode in enumerate(modes):
        y = top + row*(row_height+25)
        draw.text((left-12, y+row_height/2), mode.capitalize(), fill="#555555", font=small_font, anchor="rm")
        for column, im in enumerate(thumbnails):
            sheet.paste(simulate_image(im, mode), (left + column*(width+gap), y))
    sheet.save(output / "comparison_40mm_all_modes.png", dpi=(300, 300))
    # Larger, same-physical-width original comparison for desktop inspection.
    wide = 1080
    images = []
    for path in (baseline, candidate):
        with Image.open(path) as image:
            images.append(image.convert("RGB").resize(
                (wide, round(wide*image.height/image.width)), Image.Resampling.LANCZOS
            ))
    sheet = Image.new("RGB", (wide*2+100, max(im.height for im in images)+75), "white")
    draw = ImageDraw.Draw(sheet)
    for index, (label, im) in enumerate(zip(("B / P1", candidate_label), images)):
        x = 25+index*(wide+50)
        draw.text((x+wide/2, 12), label, fill="#333333", font=font, anchor="mt")
        sheet.paste(im, (x, 55))
    sheet.save(output / "comparison_with_B.png", dpi=(300, 300))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bundle", type=Path, required=True)
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[1]
    bundle = args.bundle.resolve()
    source = root / "paper-figures/figure9-condition-distance-20260904-r2/source_data"
    baseline = root / "paper-figures/figure7-palette-20260904/p1/figure7_condition_distance.png"
    review = bundle / "review"
    review.mkdir(exist_ok=True)
    generator.configure_plot_style()
    frames = {"pair_frame": pd.read_csv(source / "distance_pairs.csv"),
              "state_distance_frame": pd.read_csv(source / "state_distance_statistics.csv"),
              "nearest_frame": pd.read_csv(source / "nearest_state_separation.csv")}
    report = verify_coordinates(frames)
    report["existing_B_pixels_unchanged"] = verify_b_unchanged(frames, baseline)
    record = json.loads((bundle / "figure7_condition_distance_generation_record.json").read_text(encoding="utf-8"))
    assert record["generator"]["script_sha256"] == sha256(root / record["generator"]["script"])
    for artifact in record["artifacts"]:
        assert artifact["sha256"] == sha256(root / artifact["path"])
    for source_file in record["source_files"]:
        assert sha256(root / source_file["source_path"]) == source_file["sha256"] == sha256(root / source_file["copied_path"])
    report["artifact_and_source_hashes_valid"] = True
    png = bundle / "figure7_condition_distance.png"
    with Image.open(png) as image:
        assert image.size == (4320, 3390)
        assert image.info["dpi"][0] >= 599.9
        report["png_pixels"] = list(image.size)
    pdf_info = subprocess.check_output(
        ["pdfinfo", str(bundle / "figure7_condition_distance.pdf")], text=True,
    )
    assert re.search(r"Pages:\s+1\s", pdf_info)
    page_match = re.search(r"Page size:\s+([\d.]+) x ([\d.]+) pts", pdf_info)
    assert page_match, pdf_info
    size = [float(page_match[1]), float(page_match[2])]
    np.testing.assert_allclose(size, [518.4, 406.8], rtol=0, atol=0.001)
    report["pdf_page_points"] = size
    write_comparison(baseline, png, review)
    report["review_script_sha256"] = sha256(Path(__file__))
    report["review_sources"] = {str(p.relative_to(root)): sha256(p) for p in (baseline, png)}
    report["simulation_method"] = "Machado severity-100 matrices on linear-sRGB 40 mm thumbnails; grayscale from linear-light luminance"
    (review / "validation.json").write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    generator.write_generated_file_manifest(bundle)
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
