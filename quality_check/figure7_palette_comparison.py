#!/usr/bin/env python3
"""Audit Figure 7 palette systems and render CVD/grayscale comparisons."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image, ImageDraw, ImageFont

from quality_check.unoccupied_environment_validation import FIGURE7_PALETTES


PALETTE_NAMES = ("p1", "p2", "p3")
TEXT_ROLES = (
    "within",
    "nearest",
    "between",
    "warning",
    "text",
    "heading",
    "title",
    "axis",
)
DATA_ROLE_PAIRS = (
    ("within", "nearest"),
    ("within", "between"),
    ("nearest", "between"),
)
CVD_MATRICES = {
    # Machado, Oliveira, and Fernandes (2009), severity 100 matrices.
    # Coefficients verified against the Colorspacious reference implementation:
    # https://github.com/njsmith/colorspacious/blob/master/colorspacious/cvd.py
    "protanopia": np.array(
        (
            (0.152286, 1.052583, -0.204868),
            (0.114503, 0.786281, 0.099216),
            (-0.003882, -0.048116, 1.051998),
        ),
        dtype=float,
    ),
    "deuteranopia": np.array(
        (
            (0.367322, 0.860646, -0.227968),
            (0.280085, 0.672501, 0.047413),
            (-0.011820, 0.042940, 0.968881),
        ),
        dtype=float,
    ),
    "tritanopia": np.array(
        (
            (1.255528, -0.076749, -0.178779),
            (-0.078411, 0.930809, 0.147602),
            (0.004733, 0.691367, 0.303900),
        ),
        dtype=float,
    ),
}
MODE_LABELS = {
    "original": "Original",
    "deuteranopia": "Deuteranopia",
    "protanopia": "Protanopia",
    "tritanopia": "Tritanopia",
    "grayscale": "Grayscale",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--palette-root",
        type=Path,
        required=True,
        help="Directory containing p1/p2/p3 Figure 7 output bundles.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        required=True,
        help="Directory for metrics and comparison PNGs.",
    )
    return parser.parse_args()


def hex_rgb(color: str) -> np.ndarray:
    value = color.lstrip("#")
    return np.array(
        [int(value[index : index + 2], 16) for index in (0, 2, 4)],
        dtype=float,
    ) / 255.0


def srgb_to_linear(values: np.ndarray) -> np.ndarray:
    return np.where(
        values <= 0.04045,
        values / 12.92,
        ((values + 0.055) / 1.055) ** 2.4,
    )


def linear_to_srgb(values: np.ndarray) -> np.ndarray:
    clipped = np.clip(values, 0.0, 1.0)
    return np.where(
        clipped <= 0.0031308,
        clipped * 12.92,
        1.055 * clipped ** (1.0 / 2.4) - 0.055,
    )


def relative_luminance(color: str | np.ndarray) -> float:
    encoded = hex_rgb(color) if isinstance(color, str) else color
    linear = srgb_to_linear(np.asarray(encoded, dtype=float))
    return float(np.dot(linear, np.array((0.2126, 0.7152, 0.0722))))


def contrast_ratio(color_a: str | np.ndarray, color_b: str | np.ndarray) -> float:
    luminances = sorted(
        (relative_luminance(color_a), relative_luminance(color_b)),
        reverse=True,
    )
    return (luminances[0] + 0.05) / (luminances[1] + 0.05)


def simulated_rgb(color: str, matrix: np.ndarray) -> np.ndarray:
    linear = srgb_to_linear(hex_rgb(color))
    return linear_to_srgb(matrix @ linear)


def rgb_to_lab(encoded: np.ndarray) -> np.ndarray:
    red, green, blue = srgb_to_linear(np.asarray(encoded, dtype=float))
    xyz = np.array(
        (
            0.4124564 * red + 0.3575761 * green + 0.1804375 * blue,
            0.2126729 * red + 0.7151522 * green + 0.0721750 * blue,
            0.0193339 * red + 0.1191920 * green + 0.9503041 * blue,
        )
    )
    xyz /= np.array((0.95047, 1.0, 1.08883))
    threshold = 216.0 / 24389.0
    slope = 24389.0 / 27.0
    transformed = np.where(
        xyz > threshold,
        np.cbrt(xyz),
        (slope * xyz + 16.0) / 116.0,
    )
    return np.array(
        (
            116.0 * transformed[1] - 16.0,
            500.0 * (transformed[0] - transformed[1]),
            200.0 * (transformed[1] - transformed[2]),
        )
    )


def delta_e_76(color_a: np.ndarray, color_b: np.ndarray) -> float:
    return float(np.linalg.norm(rgb_to_lab(color_a) - rgb_to_lab(color_b)))


def simulate_image(image: Image.Image, mode: str) -> Image.Image:
    rgb = np.asarray(image.convert("RGB"), dtype=np.float32) / 255.0
    linear = srgb_to_linear(rgb)
    if mode == "grayscale":
        luminance = np.sum(
            linear * np.array((0.2126, 0.7152, 0.0722), dtype=np.float32),
            axis=2,
        )
        simulated_linear = np.repeat(luminance[:, :, None], 3, axis=2)
    elif mode in CVD_MATRICES:
        simulated_linear = linear @ CVD_MATRICES[mode].T
    elif mode == "original":
        simulated_linear = linear
    else:
        raise ValueError(f"Unsupported simulation mode {mode!r}")
    encoded = linear_to_srgb(simulated_linear)
    return Image.fromarray(np.round(encoded * 255.0).astype(np.uint8), "RGB")


def publication_font(size: int) -> ImageFont.FreeTypeFont | ImageFont.ImageFont:
    for candidate in (
        Path(r"C:\Windows\Fonts\arial.ttf"),
        Path(r"C:\Windows\Fonts\calibri.ttf"),
    ):
        if candidate.exists():
            return ImageFont.truetype(str(candidate), size=size)
    return ImageFont.load_default()


def thumbnail(image: Image.Image, width: int) -> Image.Image:
    height = round(image.height * width / image.width)
    return image.resize((width, height), Image.Resampling.LANCZOS)


def write_three_up(
    rendered_thumbs: dict[tuple[str, str], Image.Image],
    mode: str,
    output_path: Path,
) -> None:
    thumb_width = round(40.0 / 25.4 * 300.0)
    thumbs = {
        palette: rendered_thumbs[(mode, palette)]
        for palette in PALETTE_NAMES
    }
    gap = 22
    header = 62
    canvas = Image.new(
        "RGB",
        (
            thumb_width * len(PALETTE_NAMES) + gap * (len(PALETTE_NAMES) + 1),
            header + next(iter(thumbs.values())).height + gap,
        ),
        "white",
    )
    draw = ImageDraw.Draw(canvas)
    font = publication_font(28)
    title_font = publication_font(25)
    draw.text((gap, 10), MODE_LABELS[mode], fill="#303030", font=title_font)
    for index, palette in enumerate(PALETTE_NAMES):
        x_position = gap + index * (thumb_width + gap)
        label = palette.upper()
        bounds = draw.textbbox((0, 0), label, font=font)
        label_width = bounds[2] - bounds[0]
        draw.text(
            (x_position + (thumb_width - label_width) / 2, 10),
            label,
            fill="#303030",
            font=font,
        )
        canvas.paste(thumbs[palette], (x_position, header))
    canvas.save(output_path, dpi=(300, 300))


def write_grid(
    rendered_thumbs: dict[tuple[str, str], Image.Image],
    output_path: Path,
) -> None:
    modes = tuple(MODE_LABELS)
    thumb_width = round(40.0 / 25.4 * 300.0)
    sample_thumb = rendered_thumbs[("original", "p1")]
    gap = 18
    label_width = 190
    top_header = 50
    canvas = Image.new(
        "RGB",
        (
            label_width + len(PALETTE_NAMES) * (thumb_width + gap) + gap,
            top_header + len(modes) * (sample_thumb.height + gap),
        ),
        "white",
    )
    draw = ImageDraw.Draw(canvas)
    font = publication_font(25)
    small_font = publication_font(22)
    for index, palette in enumerate(PALETTE_NAMES):
        x_position = label_width + index * (thumb_width + gap)
        draw.text(
            (x_position + thumb_width / 2, 12),
            palette.upper(),
            anchor="ma",
            fill="#303030",
            font=font,
        )
    for row_index, mode in enumerate(modes):
        y_position = top_header + row_index * (sample_thumb.height + gap)
        draw.text(
            (label_width - 14, y_position + sample_thumb.height / 2),
            MODE_LABELS[mode],
            anchor="rm",
            fill="#303030",
            font=small_font,
        )
        for column_index, palette in enumerate(PALETTE_NAMES):
            x_position = label_width + column_index * (thumb_width + gap)
            rendered = rendered_thumbs[(mode, palette)]
            canvas.paste(rendered, (x_position, y_position))
    canvas.save(output_path, dpi=(300, 300))


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def palette_metrics() -> dict[str, Any]:
    report: dict[str, Any] = {
        "methods": {
            "contrast": "WCAG 2 relative luminance and contrast ratio",
            "grayscale": "linear-sRGB WCAG relative luminance encoded as neutral sRGB",
            "cvd": "Machado et al. (2009) severity-100 matrices applied in linear sRGB",
            "simulated_color_separation": "CIELAB D65 Delta E 1976",
        },
        "thresholds": {
            "minimum_text_contrast": 4.5,
            "minimum_data_role_luminance_ratio": 1.4,
            "minimum_warning_between_grayscale_ratio": 1.4,
            "minimum_warning_between_simulated_delta_e_76": 20.0,
        },
        "palettes": {},
    }
    white = "#FFFFFF"
    for palette_name in PALETTE_NAMES:
        palette = FIGURE7_PALETTES[palette_name]
        text_contrast = {
            role: {
                "on_white": contrast_ratio(palette[role], white),
                "on_zebra": contrast_ratio(
                    palette[role],
                    palette["zebra"],
                ),
            }
            for role in TEXT_ROLES
        }
        data_ratios = {
            f"{role_a}_vs_{role_b}": contrast_ratio(
                palette[role_a],
                palette[role_b],
            )
            for role_a, role_b in DATA_ROLE_PAIRS
        }
        warning = hex_rgb(palette["warning"])
        between = hex_rgb(palette["between"])
        warning_between = {
            "grayscale_relative_luminance_ratio": contrast_ratio(
                palette["warning"],
                palette["between"],
            ),
            "original_delta_e_76": delta_e_76(warning, between),
            "cvd": {},
        }
        for mode, matrix in CVD_MATRICES.items():
            simulated_warning = simulated_rgb(palette["warning"], matrix)
            simulated_between = simulated_rgb(palette["between"], matrix)
            warning_between["cvd"][mode] = {
                "delta_e_76": delta_e_76(
                    simulated_warning,
                    simulated_between,
                ),
                "relative_luminance_ratio": contrast_ratio(
                    simulated_warning,
                    simulated_between,
                ),
            }
        assert min(
            value[background]
            for value in text_contrast.values()
            for background in ("on_white", "on_zebra")
        ) >= 4.5
        assert min(data_ratios.values()) >= 1.4
        assert warning_between["grayscale_relative_luminance_ratio"] >= 1.4
        assert min(
            value["delta_e_76"]
            for value in warning_between["cvd"].values()
        ) >= 20.0
        report["palettes"][palette_name] = {
            "roles": palette,
            "text_contrast": text_contrast,
            "data_role_luminance_ratios": data_ratios,
            "warning_vs_between": warning_between,
            "checks_passed": True,
        }
    return report


def main() -> int:
    args = parse_args()
    palette_root = args.palette_root.resolve()
    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    image_paths = {
        palette: palette_root / palette / "figure7_condition_distance.png"
        for palette in PALETTE_NAMES
    }
    missing = [str(path) for path in image_paths.values() if not path.exists()]
    if missing:
        raise FileNotFoundError(f"Missing palette figures: {missing}")
    images = {palette: Image.open(path) for palette, path in image_paths.items()}
    sizes = {image.size for image in images.values()}
    if len(sizes) != 1:
        raise ValueError(f"Palette figure sizes differ: {sorted(sizes)}")
    thumb_width = round(40.0 / 25.4 * 300.0)
    base_thumbs = {
        palette: thumbnail(image, thumb_width)
        for palette, image in images.items()
    }
    rendered_thumbs = {
        (mode, palette): simulate_image(base_thumbs[palette], mode)
        for mode in MODE_LABELS
        for palette in PALETTE_NAMES
    }
    for mode in MODE_LABELS:
        write_three_up(
            rendered_thumbs,
            mode,
            output_dir / f"{mode}_three_up.png",
        )
    write_grid(rendered_thumbs, output_dir / "all_modes_grid.png")
    metrics = palette_metrics()
    metrics["source_figures"] = {
        palette: {
            "path": str(path),
            "sha256": sha256_file(path),
            "pixel_size": list(images[palette].size),
        }
        for palette, path in image_paths.items()
    }
    metrics["comparison_outputs"] = sorted(
        path.name for path in output_dir.glob("*.png")
    )
    (output_dir / "palette_metrics.json").write_text(
        json.dumps(metrics, indent=2),
        encoding="utf-8",
    )
    for image in images.values():
        image.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
