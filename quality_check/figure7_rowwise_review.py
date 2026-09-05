#!/usr/bin/env python3
"""Audit rowwise labels, exact source values, ratio geometry and retained ECDF."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
import re
import subprocess

import matplotlib.pyplot as plt
from matplotlib.text import Text
from matplotlib.patches import Rectangle
import numpy as np
import pandas as pd
from PIL import Image

from quality_check import unoccupied_environment_validation as generator
from quality_check.figure7_facet_review import (
    capture_figure, sha256, verify_b_unchanged, write_comparison,
)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bundle", type=Path, required=True)
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[1]
    bundle = args.bundle.resolve()
    source = root / "paper-figures/figure9-condition-distance-20260904-r2/source_data"
    frames = {
        "pair_frame": pd.read_csv(source / "distance_pairs.csv"),
        "state_distance_frame": pd.read_csv(source / "state_distance_statistics.csv"),
        "nearest_frame": pd.read_csv(source / "nearest_state_separation.csv"),
    }
    generator.configure_plot_style()
    fig = capture_figure(frames, "rowwise")
    ordered = generator._figure7_variant_frame(frames["state_distance_frame"], frames["nearest_frame"])
    assert len(fig.axes) == 3
    ratio_axis, ticks, ecdf = fig.axes
    fig.canvas.draw()
    row_renderer = fig.canvas.get_renderer()
    assert ratio_axis.get_xscale() == ticks.get_xscale() == "linear"
    assert ratio_axis.get_xlim() == (0.65, 3.4)
    marks = [line for line in ratio_axis.lines if line.get_marker() == "D"]
    assert len(marks) == 13
    ratios = ordered["nearest_state_ratio"].to_numpy(dtype=float)
    np.testing.assert_array_equal([m.get_xdata()[0] for m in marks], ratios)
    palette = generator.FIGURE7_PALETTES["p1"]
    for mark, row in zip(marks, ordered.itertuples(index=False)):
        code = generator.paper_state_code(str(row.state))
        label = next(t for t in fig.texts if t.get_text() == code)
        y = label.get_position()[1]
        row_texts = [t for t in fig.texts if math.isclose(t.get_position()[1], y, abs_tol=1e-12)]
        expected = [
            code, generator.paper_state_short_label(str(row.state)),
            f"{row.mean_intra_distance:.2f}", "±", f"{row.std_intra_distance:.2f}",
            f"{row.mean_nearest_inter_distance:.2f}",
            f"{row.mean_inter_distance:.2f}", "±", f"{row.std_inter_distance:.2f}",
            f"{row.nearest_state_ratio:.2f}",
        ]
        assert [t.get_text() for t in row_texts] == expected, code
        text_bounds = sorted((t.get_window_extent(row_renderer) for t in row_texts), key=lambda b: b.x0)
        assert all(a.x1 <= b.x0 for a, b in zip(text_bounds, text_bounds[1:])), code
        pixel_y = ratio_axis.transData.transform((float(row.nearest_state_ratio), mark.get_ydata()[0]))[1]
        assert math.isclose(pixel_y, y*fig.bbox.height, abs_tol=1e-8)
        row_color = palette["warning"] if row.nearest_state_ratio < 1 else palette["nearest"]
        assert mark.get_markeredgecolor() == row_color
        assert mark.get_markerfacecolor() == (row_color if row.nearest_state_ratio < 1 else "white")
        assert [t.get_color() for t in row_texts[2:5]] == [palette["within"]]*3
        assert row_texts[5].get_color() == palette["nearest"]
        assert [t.get_color() for t in row_texts[6:9]] == [palette["between"]]*3
    assert not any(isinstance(a, Rectangle) for a in fig.artists)
    assert not ratio_axis.patches
    within, between = generator._figure7_distance_arrays(frames["pair_frame"])
    curves = [line for line in ecdf.lines if line.get_drawstyle() == "steps-post"]
    assert len(curves) == 2 and ecdf.get_legend() is None and not ecdf.collections
    for line, values, role in zip(curves, (within, between), ("within", "between")):
        np.testing.assert_array_equal(line.get_xdata(), np.r_[0.0, np.sort(values)])
        np.testing.assert_array_equal(line.get_ydata(), np.r_[0.0, np.arange(1, len(values)+1)/len(values)])
        assert line.get_color() == palette[role]
        assert (line.get_linestyle(), line.get_linewidth(), line.get_solid_capstyle(), line.get_solid_joinstyle()) == ("-", 1.0, "butt", "miter")
    assert [t.get_text() for t in ecdf.texts] == ["Within  (n = 126)", "Between  (n = 585)"]
    assert [t.get_color() for t in ecdf.texts] == [palette["within"], palette["between"]]
    fig.canvas.draw()
    texts = [t for t in fig.findobj(Text) if t.get_visible() and t.get_text()]
    renderer = fig.canvas.get_renderer()
    outside = [t.get_text() for t in texts if not fig.bbox.contains(*t.get_window_extent(renderer).get_points()[0]) or not fig.bbox.contains(*t.get_window_extent(renderer).get_points()[1])]
    assert not outside, outside
    minimum_font = min(t.get_fontsize() for t in texts)
    assert minimum_font >= 6.0
    plt.close(fig)
    baseline = root / "paper-figures/figure7-palette-20260904/p1/figure7_condition_distance.png"
    assert verify_b_unchanged(frames, baseline)
    record = json.loads((bundle / "figure7_condition_distance_generation_record.json").read_text(encoding="utf-8"))
    assert record["generator"]["script_sha256"] == sha256(root / record["generator"]["script"])
    for item in record["source_files"]:
        assert item["sha256"] == sha256(root/item["source_path"]) == sha256(root/item["copied_path"])
    for item in record["artifacts"]:
        assert item["sha256"] == sha256(root/item["path"])
    png = bundle / "figure7_condition_distance.png"
    with Image.open(png) as image:
        assert image.size == (4320, 3330)
        assert image.info["dpi"][0] >= 599.9
    pdf_info = subprocess.check_output(["pdfinfo", str(bundle / "figure7_condition_distance.pdf")], text=True)
    assert re.search(r"Pages:\s+1\s", pdf_info)
    size = re.search(r"Page size:\s+([\d.]+) x ([\d.]+) pts", pdf_info)
    assert size
    np.testing.assert_allclose([float(size[1]), float(size[2])], [518.4, 399.6], atol=0.001)
    review = bundle / "review"
    review.mkdir(exist_ok=True)
    write_comparison(baseline, png, review, candidate_label="Rowwise / P1")
    report = {
        "source_values_and_full_labels_exact": True,
        "source_ratios_at_exact_x_positions": True,
        "all_rows_aligned_with_markers": True,
        "no_overlapping_row_text": True,
        "no_zebra_rectangles": True,
        "ecdf_exact_with_finalized_style": True,
        "all_role_colors_consistent": True,
        "pair_counts": [len(within), len(between)],
        "minimum_visible_font_pt": minimum_font,
        "text_outside_canvas": outside,
        "source_and_artifact_hashes_valid": True,
        "B_regression_RGB_pixels_identical": True,
        "figure_inches": [7.2, 5.55],
        "comparison_inputs": {str(p.relative_to(root)): sha256(p) for p in (baseline, png)},
        "review_script_sha256": sha256(Path(__file__)),
        "comparison_helper_sha256": sha256(Path(__file__).with_name("figure7_facet_review.py")),
        "simulation_method": "Machado severity-100 matrices in linear sRGB on 40 mm thumbnails; grayscale via linear-light luminance",
    }
    (review / "validation.json").write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    generator.write_generated_file_manifest(bundle)
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
