#!/usr/bin/env python3
"""Validate repeatability and separability of unoccupied CSI recordings.

The script discovers records from AXHome-MM-v1 metadata, decodes the released
FeitCSI ``.dat`` files with the parser already used by the CSI baseline, and
generates publication-oriented figures, tables, logs, and a Markdown report.

One metadata JSON / CSI file pair is treated as one independent recording.
Window counts are never used as repetition counts.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import logging
import math
import re
import shutil
import subprocess
import sys
import textwrap
from collections import Counter, defaultdict
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.lines import Line2D
from matplotlib.offsetbox import AnchoredOffsetbox, HPacker, TextArea
from matplotlib.patches import Patch, Rectangle
from scipy.spatial.distance import pdist, squareform


LOGGER = logging.getLogger("unoccupied_environment_validation")
EPSILON = 1e-12
UNSET_VALUES = {"", "n/a", "na", "null", "unknown", "<missing>"}
UNOCCUPIED_PERSON_VALUES = {
    "none",
    "no_person",
    "no-person",
    "nobody",
    "unoccupied",
    "vacant",
}
UNOCCUPIED_ACTIVITY_VALUES = {
    "background_idle",
    "empty_room",
    "environment_state",
    "no_person",
    "unoccupied",
    "vacant",
}
REQUESTED_FIELD_ALIASES: Mapping[str, tuple[str, ...]] = {
    "environment": (
        "environment",
        "environment_id",
        "environment_name",
        "room",
        "room_id",
    ),
    "condition": ("condition", "condition_id"),
    "activity": ("activity", "activity_id", "action", "action_id"),
    "scenario": ("scenario", "scenario_id"),
    "sample_id": ("sample_id",),
}
FIGURE_DPI = 600
FIGURE7_WIDTH_IN = 7.2
FIGURE7_HEIGHT_IN = 5.45
FIGURE7_CONTENT_LEFT_IN = 0.20
FIGURE7_LABEL_COLUMN_IN = 1.35
FIGURE7_PLOT_WIDTH_IN = 4.60
FIGURE7_PLOT_RATIO_GAP_IN = 0.05
FIGURE7_RATIO_WIDTH_IN = 0.75
FIGURE7_PANEL_A_HEIGHT_IN = 3.87
FIGURE7_PANEL_B_HEIGHT_IN = 0.58
FIGURE7_X_LIMITS = (0.0, 48.0)
FIGURE7_STEM = "figure7_condition_distance"
FIGURE7_LAYOUT_VARIANTS = ("r3", "a", "b", "c", "facets", "rowwise")
FIGURE7_FACET_HEIGHT_IN = 5.65
FIGURE7_PANEL_B_VARIANTS = ("default", "v1", "v2")
FIGURE7_PALETTE_VARIANTS = ("current", "p1", "p2", "p3")
FIGURE7_PALETTES: dict[str, dict[str, str]] = {
    "current": {
        "within": "#1A5FA8",
        "nearest": "#55585C",
        "between": "#C4611F",
        "warning": "#B03030",
        "text": "#333333",
        "heading": "#7C7C7C",
        "title": "#141414",
        "axis": "#4A4A4A",
        "grid": "#ECECEC",
        "zebra": "#F6F7F9",
        "rule": "#D8D8D8",
        "major_rule": "#B9B9B9",
        "reference": "#9A9A9A",
    },
    "p1": {
        "within": "#16365C",
        "nearest": "#4B747B",
        "between": "#5B477A",
        "warning": "#55151F",
        "text": "#28303A",
        "heading": "#626C78",
        "title": "#151B22",
        "axis": "#465463",
        "grid": "#E4EAF0",
        "zebra": "#F2F5F8",
        "rule": "#CCD4DC",
        "major_rule": "#AEB9C4",
        "reference": "#7B8792",
    },
    "p2": {
        "within": "#26313A",
        "nearest": "#666C71",
        "between": "#704609",
        "warning": "#12304A",
        "text": "#302E2A",
        "heading": "#69635B",
        "title": "#1E1C1A",
        "axis": "#514D47",
        "grid": "#ECE6DC",
        "zebra": "#F6F2EA",
        "rule": "#D9D1C4",
        "major_rule": "#BDB3A5",
        "reference": "#8A837A",
    },
    "p3": {
        "within": "#40205E",
        "nearest": "#2F7355",
        "between": "#75331F",
        "warning": "#102A43",
        "text": "#302B33",
        "heading": "#6D6570",
        "title": "#1E1722",
        "axis": "#554D58",
        "grid": "#ECE5ED",
        "zebra": "#F6F2F7",
        "rule": "#D8CDD9",
        "major_rule": "#BAAEBB",
        "reference": "#887D8B",
    },
}
FIGURE7_SOURCE_FILENAMES = (
    "distance_pairs.csv",
    "state_distance_statistics.csv",
    "nearest_state_separation.csv",
)
PUBLICATION_FONT_CANDIDATES = ("Arial", "DejaVu Sans")
PLOT_RANDOM_SEED = 20260728
V4_BOXPLOT_WIDTH = 0.45
V4_RATIO_LABEL_OFFSET_POINTS = 22.0
V5_BOXPLOT_WIDTH = 0.45
V5_RATIO_LABEL_OFFSET_POINTS = 6.0
V6_BOXPLOT_WIDTH = 0.45
V6_RATIO_LABEL_OFFSET_POINTS = 6.0
INTRA_COLOR = "#1A5FA8"
INTER_COLOR = "#C4611F"
INTRA_LIGHT_COLOR = "#D1DFEE"
INTER_LIGHT_COLOR = "#F3DFD2"
RATIO_OK_COLOR = "#595959"
RATIO_FLAG_COLOR = "#B03030"
EMPHASIZED_CONNECTOR_COLOR = "#8C8C8C"
MEAN_CURVE_COLOR = "#2F5D8C"
RECORDING_COLOR = "#9A9A9A"
BAND_COLOR = "#8FB3D1"
GRID_COLOR = "#EEEEEE"
CONNECTOR_COLOR = "#DCDCDC"
TEXT_COLOR = "#222222"

ENVIRONMENT_LABELS: Mapping[str, str] = {
    "E1": "Living room",
    "E3": "Bathroom",
    "E4": "Kitchen",
}
STATE_LABELS: Mapping[str, str] = {
    "E1__clean_none": "Clean",
    "E1__heater_on_heater_cloth_front": (
        "Warm air directed at clothing"
    ),
    "E1__heater_on_heater_stable": "Heater stable operation",
    "E1__heater_on_heater_transition": "Heater start-up transition",
    "E3__bucket_filling_sim_water_overflow_bucket": "Bucket filling",
    "E3__bucket_overflow_sim_water_overflow_bucket": "Bucket overflow",
    "E3__clean_none": "Clean",
    "E3__exhaust_fan_on_fan_vibration": "Exhaust fan",
    "E3__faucet_on_water_flow_sink": "Sink water flow",
    "E3__shower_on_water_flow_shower": "Shower water flow",
    "E4__clean_none": "Clean",
    "E4__stove_unattended_sim_dry_pot_sim": (
        "Unattended dry-pot heating"
    ),
    "E4__stove_unattended_sim_kettle_boiling_sim": "Kettle boiling",
}
STATE_CODE_LABELS: Mapping[str, tuple[str, str]] = {
    "E1__clean_none": ("L1", "Clean"),
    "E1__heater_on_heater_cloth_front": (
        "L2",
        "Fabric in front of heater",
    ),
    "E1__heater_on_heater_stable": ("L3", "Heater stable"),
    "E1__heater_on_heater_transition": ("L4", "Heater start-up"),
    "E3__bucket_filling_sim_water_overflow_bucket": (
        "B1",
        "Bucket filling",
    ),
    "E3__bucket_overflow_sim_water_overflow_bucket": (
        "B2",
        "Bucket overflow",
    ),
    "E3__clean_none": ("B3", "Clean"),
    "E3__exhaust_fan_on_fan_vibration": ("B4", "Exhaust fan"),
    "E3__faucet_on_water_flow_sink": ("B5", "Sink flow"),
    "E3__shower_on_water_flow_shower": ("B6", "Shower flow"),
    "E4__clean_none": ("K1", "Clean"),
    "E4__stove_unattended_sim_dry_pot_sim": (
        "K2",
        "Dry-pot heating",
    ),
    "E4__stove_unattended_sim_kettle_boiling_sim": (
        "K3",
        "Kettle boiling",
    ),
}
STATE_V5_SHORT_LABELS: Mapping[str, str] = {
    "E1__clean_none": "Clean",
    "E1__heater_on_heater_cloth_front": "Warm-air",
    "E1__heater_on_heater_stable": "Stable heater",
    "E1__heater_on_heater_transition": "Heater start-up",
    "E3__bucket_filling_sim_water_overflow_bucket": "Filling",
    "E3__bucket_overflow_sim_water_overflow_bucket": "Overflow",
    "E3__clean_none": "Clean",
    "E3__exhaust_fan_on_fan_vibration": "Exhaust fan",
    "E3__faucet_on_water_flow_sink": "Sink flow",
    "E3__shower_on_water_flow_shower": "Shower flow",
    "E4__clean_none": "Clean",
    "E4__stove_unattended_sim_dry_pot_sim": "Dry-pot",
    "E4__stove_unattended_sim_kettle_boiling_sim": "Boiling",
}
V5_PANEL_TITLES: Mapping[str, str] = {
    "(a)": "State-level repeatability and separation",
    "(b)": "Overall distance distribution",
    "(c)": "Nearest-state separation",
}
V5_FONT_SIZES: Mapping[str, float] = {
    "tick": 9.2,
    "state_a": 9.2,
    "state_c": 8.7,
    "environment": 9.7,
    "axis_label": 10.2,
    "panel_title": 10.2,
    "panel_label": 11.5,
    "ratio_value": 8.7,
    "legend": 9.0,
    "threshold": 8.2,
}
V6_FONT_SIZES: Mapping[str, float] = {
    "tick": 8.7,
    "state_a": 8.8,
    "state_c": 8.5,
    "environment_a": 9.2,
    "environment_c": 8.8,
    "axis_label": 9.3,
    "panel_label": 11.0,
    "ratio_value": 8.2,
    "legend": 8.5,
}
STATE_ORDER: tuple[str, ...] = tuple(STATE_LABELS)
SUPPLEMENTARY_ENVIRONMENT_FILES: Mapping[str, tuple[str, str]] = {
    "E1": (
        "supp_figure_S1a_living_room_mean_amplitude",
        "supp_figure_S2a_living_room_variability",
    ),
    "E3": (
        "supp_figure_S1b_bathroom_mean_amplitude",
        "supp_figure_S2b_bathroom_variability",
    ),
    "E4": (
        "supp_figure_S1c_kitchen_mean_amplitude",
        "supp_figure_S2c_kitchen_variability",
    ),
}


@dataclass(frozen=True)
class Recording:
    """One independently collected unoccupied recording."""

    sample_id: str
    session_id: str
    person_id: str
    environment_id: str
    action_id: str
    trial_id: str
    link_configuration: str
    state_token: str
    state: str
    state_source: str
    duration_sec: float
    csi_path: Path
    csi_path_relative: str
    metadata_path: Path
    metadata_path_relative: str


@dataclass(frozen=True)
class RecordingStatistics:
    """Amplitude statistics and features for one recording."""

    recording: Recording
    packet_count: int
    num_rx: int
    num_tx: int
    num_subcarriers: int
    low_energy_subcarrier_count: int
    amplitude_scale: float
    mean_curve: np.ndarray
    std_curve: np.ndarray
    normalized_mean_curve: np.ndarray
    normalized_std_curve: np.ndarray
    normalized_variance_curve: np.ndarray


def parse_args() -> argparse.Namespace:
    project_root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(
        description=(
            "Discover and validate repeatability/separability of unoccupied "
            "AXHome-MM-v1 CSI recordings."
        )
    )
    parser.add_argument(
        "--dataset-root",
        type=Path,
        default=project_root / "dataset-release" / "AXHome-MM-v1",
        help="AXHome-MM-v1 release root.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=project_root
        / "quality_check"
        / "results"
        / "unoccupied_validation",
        help="Directory for all generated outputs.",
    )
    parser.add_argument(
        "--source-results-dir",
        type=Path,
        default=project_root
        / "quality_check"
        / "results"
        / "unoccupied_validation",
        help=(
            "Existing result-CSV directory used by --redraw-figure7-only."
        ),
    )
    parser.add_argument(
        "--low-energy-ratio",
        type=float,
        default=0.05,
        help=(
            "Subcarriers below this fraction of the median recording energy "
            "are linearly interpolated (default: 0.05)."
        ),
    )
    parser.add_argument(
        "--dpi",
        type=int,
        default=FIGURE_DPI,
        help="Raster figure resolution (minimum and default: 600 DPI).",
    )
    parser.add_argument(
        "--distance-plot",
        choices=("both", "paired", "boxplot"),
        default="both",
        help=(
            "Standalone distance panel selection: state-level paired plot, "
            "pair-level boxplot, or both (default: both). The combined main "
            "figure always contains both; existing files are not deleted."
        ),
    )
    parser.add_argument(
        "--redraw-main-v2-only",
        action="store_true",
        help=(
            "Generate the v2 main figure, nearest-state CSV, and report "
            "section from existing result CSV files without scanning metadata "
            "or reading CSI."
        ),
    )
    parser.add_argument(
        "--redraw-main-v4-only",
        action="store_true",
        help=(
            "Generate the v4 main figure and report additions from existing "
            "result CSV files without scanning metadata or reading CSI."
        ),
    )
    parser.add_argument(
        "--redraw-main-v5-only",
        action="store_true",
        help=(
            "Generate the v5 main figure, caption, and report additions from "
            "existing result CSV files without scanning metadata, reading "
            "CSI, or rewriting distance CSV files."
        ),
    )
    parser.add_argument(
        "--redraw-main-v6-only",
        action="store_true",
        help=(
            "Generate the simplified v6 main figure and caption from "
            "existing result CSV files without scanning metadata, reading "
            "CSI, or rewriting distance CSV files."
        ),
    )
    parser.add_argument(
        "--redraw-main-v7-only",
        action="store_true",
        help=(
            "Generate the readability-focused v7 main figure and caption "
            "from existing result CSV files without scanning metadata, "
            "reading CSI, or rewriting distance CSV files."
        ),
    )
    parser.add_argument(
        "--redraw-main-v8-only",
        action="store_true",
        help=(
            "Generate the self-contained v8 main figure and caption from "
            "existing result CSV files without scanning metadata, reading "
            "CSI, or rewriting distance CSV files."
        ),
    )
    parser.add_argument(
        "--redraw-figure7-only",
        action="store_true",
        help=(
            "Generate the renumbered publication Figure 7 from existing "
            "distance-result CSV files, with copied source data and a "
            "generation record; metadata and CSI are not rescanned."
        ),
    )
    parser.add_argument(
        "--layout-variant",
        choices=FIGURE7_LAYOUT_VARIANTS,
        default="r3",
        help=(
            "Figure 7 layout used with --redraw-figure7-only: the retained "
            "r3 layout, design directions a/b/c, or the refined room-facet "
            "layout facets, or ratio-first rowwise layout (default: r3)."
        ),
    )
    parser.add_argument(
        "--panel-b-variant",
        choices=FIGURE7_PANEL_B_VARIANTS,
        default="default",
        help=(
            "Panel-b treatment for Figure 7 layout b: finalized solid-line "
            "ECDF with direct labels (default), historical ECDF with "
            "observation rugs (v1), or historical fixed-bin empirical "
            "histograms (v2). Other layouts require default."
        ),
    )
    parser.add_argument(
        "--palette-variant",
        choices=FIGURE7_PALETTE_VARIANTS,
        default="current",
        help=(
            "Complete color system for Figure 7 layouts b, facets, and rowwise. Palette variants "
            "p1/p2/p3 require the finalized default panel b (default: current)."
        ),
    )
    return parser.parse_args()


def configure_logging(output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    formatter = logging.Formatter(
        "%(asctime)s | %(levelname)-8s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )
    LOGGER.setLevel(logging.INFO)
    LOGGER.handlers.clear()

    console = logging.StreamHandler(sys.stdout)
    console.setFormatter(formatter)
    file_handler = logging.FileHandler(
        output_dir / "unoccupied_validation.log",
        mode="w",
        encoding="utf-8",
    )
    file_handler.setFormatter(formatter)
    LOGGER.addHandler(console)
    LOGGER.addHandler(file_handler)


def resolve_publication_font() -> tuple[str, str]:
    """Resolve the existing pipeline preference without adding a new font."""

    from matplotlib import font_manager

    for family in PUBLICATION_FONT_CANDIDATES:
        try:
            font_path = font_manager.findfont(
                family,
                fallback_to_default=False,
            )
        except ValueError:
            continue
        return family, str(Path(font_path).resolve())
    raise RuntimeError(
        "Neither Arial nor DejaVu Sans is available for publication figures"
    )


def configure_plot_style() -> None:
    font_family, _ = resolve_publication_font()
    plt.style.use("default")
    matplotlib.rcParams.update(
        {
            "font.family": font_family,
            "font.sans-serif": list(PUBLICATION_FONT_CANDIDATES),
            "font.size": 8.5,
            "axes.titlesize": 9,
            "axes.titleweight": "normal",
            "axes.labelsize": 9,
            "xtick.labelsize": 8.2,
            "ytick.labelsize": 8.2,
            "legend.fontsize": 8.2,
            "figure.titlesize": 10,
            "axes.linewidth": 0.7,
            "axes.facecolor": "white",
            "figure.facecolor": "white",
            "savefig.facecolor": "white",
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.grid": False,
            "grid.color": GRID_COLOR,
            "grid.linewidth": 0.5,
            "grid.alpha": 0.85,
            "lines.linewidth": 1.0,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
            "savefig.dpi": FIGURE_DPI,
            "savefig.bbox": "tight",
            "savefig.pad_inches": 0.03,
        }
    )


def validate_paper_labels(states: Iterable[str]) -> None:
    """Fail clearly rather than leaking raw metadata tokens into paper figures."""

    discovered = set(states)
    missing_states = discovered - set(STATE_LABELS)
    missing_state_codes = discovered - set(STATE_CODE_LABELS)
    missing_v5_labels = discovered - set(STATE_V5_SHORT_LABELS)
    missing_environments = {
        state.split("__", 1)[0] for state in discovered
    } - set(ENVIRONMENT_LABELS)
    if (
        missing_states
        or missing_state_codes
        or missing_v5_labels
        or missing_environments
    ):
        raise ValueError(
            "Paper label mapping is incomplete. Add explicit labels for "
            f"states={sorted(missing_states)} and "
            f"state_codes={sorted(missing_state_codes)} and "
            f"v5_labels={sorted(missing_v5_labels)} and "
            f"environments={sorted(missing_environments)}"
        )


def paper_state_label(state: str) -> str:
    validate_paper_labels((state,))
    return STATE_LABELS[state]


def paper_state_code(state: str) -> str:
    validate_paper_labels((state,))
    return STATE_CODE_LABELS[state][0]


def paper_state_short_label(state: str) -> str:
    validate_paper_labels((state,))
    return STATE_CODE_LABELS[state][1]


def paper_state_code_label(state: str) -> str:
    validate_paper_labels((state,))
    code, short_label = STATE_CODE_LABELS[state]
    return f"{code}  {short_label}"


def paper_state_v5_code_label(state: str) -> str:
    validate_paper_labels((state,))
    return f"{paper_state_code(state)}  {STATE_V5_SHORT_LABELS[state]}"


def paper_environment_label(environment_id: str) -> str:
    if environment_id not in ENVIRONMENT_LABELS:
        raise ValueError(
            f"No paper-readable label for environment {environment_id!r}"
        )
    return ENVIRONMENT_LABELS[environment_id]


def ordered_state_frame(frame: pd.DataFrame) -> pd.DataFrame:
    """Return a stable paper order without changing machine-readable labels."""

    validate_paper_labels(frame["state"].astype(str))
    order_lookup = {state: index for index, state in enumerate(STATE_ORDER)}
    ordered = frame.copy()
    ordered["_paper_order"] = ordered["state"].map(order_lookup)
    return (
        ordered.sort_values("_paper_order", kind="stable")
        .drop(columns="_paper_order")
        .reset_index(drop=True)
    )


def style_publication_axis(
    axis: matplotlib.axes.Axes,
    *,
    grid_axis: str | None = None,
) -> None:
    """Apply the shared Scientific Data-oriented axis treatment."""

    axis.set_facecolor("white")
    axis.spines["top"].set_visible(False)
    axis.spines["right"].set_visible(False)
    axis.spines["left"].set_linewidth(0.7)
    axis.spines["bottom"].set_linewidth(0.7)
    axis.tick_params(
        axis="both",
        direction="out",
        length=3.0,
        width=0.6,
        colors="#303030",
    )
    if grid_axis:
        axis.grid(
            axis=grid_axis,
            color=GRID_COLOR,
            linewidth=0.5,
            alpha=0.85,
            zorder=0,
        )


def add_panel_label(
    axis: matplotlib.axes.Axes,
    label: str,
    *,
    x: float = -0.12,
    y: float = 1.04,
) -> None:
    axis.text(
        x,
        y,
        label,
        transform=axis.transAxes,
        ha="left",
        va="bottom",
        fontsize=11,
        fontweight="bold",
        clip_on=False,
    )


def add_panel_heading(
    axis: matplotlib.axes.Axes,
    label: str,
    title: str,
    *,
    x: float,
    y: float = 1.025,
    label_fontsize: float | None = None,
    title_fontsize: float | None = None,
) -> None:
    """Place an adjacent bold panel label and concise explanatory title."""

    packed_heading = HPacker(
        children=[
            TextArea(
                label,
                textprops={
                    "fontsize": (
                        label_fontsize
                        if label_fontsize is not None
                        else V5_FONT_SIZES["panel_label"]
                    ),
                    "fontweight": "bold",
                    "color": TEXT_COLOR,
                },
            ),
            TextArea(
                title,
                textprops={
                    "fontsize": (
                        title_fontsize
                        if title_fontsize is not None
                        else V5_FONT_SIZES["panel_title"]
                    ),
                    "fontweight": "bold",
                    "color": TEXT_COLOR,
                },
            ),
        ],
        align="baseline",
        pad=0,
        sep=4,
    )
    heading = AnchoredOffsetbox(
        loc="lower left",
        child=packed_heading,
        frameon=False,
        pad=0,
        borderpad=0,
        bbox_to_anchor=(x, y),
        bbox_transform=axis.transAxes,
    )
    axis.add_artist(heading)


def ensure_baseline_parser(project_root: Path) -> tuple[Any, Any]:
    """Import the existing FeitCSI decoder and low-energy repair function."""

    baseline_root = project_root / "baseline-csi-2dcnn"
    if not baseline_root.is_dir():
        raise FileNotFoundError(
            "Existing CSI parser directory was not found: "
            f"{baseline_root}. The analysis intentionally reuses the "
            "baseline FeitCSI implementation."
        )
    sys.path.insert(0, str(baseline_root))
    try:
        from axhome_csi.data import interpolate_low_energy_subcarriers
        from axhome_csi.feitcsi import read_feitcsi
    except ImportError as exc:
        raise RuntimeError(
            "Unable to import the existing baseline FeitCSI parser from "
            f"{baseline_root}"
        ) from exc
    return read_feitcsi, interpolate_low_energy_subcarriers


def normalize_text(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, bool):
        return "true" if value else "false"
    return str(value).strip()


def leaf_name(field_path: str) -> str:
    return field_path.rsplit(".", 1)[-1].lower()


def iter_scalar_fields(
    value: Any, prefix: str = ""
) -> Iterable[tuple[str, Any]]:
    if isinstance(value, Mapping):
        for key, child in value.items():
            child_prefix = f"{prefix}.{key}" if prefix else str(key)
            yield from iter_scalar_fields(child, child_prefix)
    elif isinstance(value, list):
        for index, child in enumerate(value):
            child_prefix = f"{prefix}[{index}]"
            yield from iter_scalar_fields(child, child_prefix)
    else:
        yield prefix, value


def scalar_field_map(metadata: Mapping[str, Any]) -> dict[str, Any]:
    return {path: value for path, value in iter_scalar_fields(metadata)}


def values_for_leaf_names(
    fields: Mapping[str, Any], names: Sequence[str]
) -> list[str]:
    wanted = {name.lower() for name in names}
    values: list[str] = []
    for path, raw_value in fields.items():
        if leaf_name(path) not in wanted:
            continue
        value = normalize_text(raw_value)
        if value and value.lower() not in UNSET_VALUES and value not in values:
            values.append(value)
    return values


def first_field_value(
    fields: Mapping[str, Any],
    names: Sequence[str],
    *,
    default: str = "",
) -> str:
    values = values_for_leaf_names(fields, names)
    return values[0] if values else default


def metadata_files_from_release(dataset_root: Path) -> list[Path]:
    manifest = dataset_root / "manifests" / "metadata_files.txt"
    if manifest.is_file():
        paths = [
            dataset_root / line.strip()
            for line in manifest.read_text(encoding="utf-8").splitlines()
            if line.strip()
        ]
        LOGGER.info(
            "Discovered %d metadata files via %s",
            len(paths),
            manifest,
        )
    else:
        paths = sorted((dataset_root / "data").glob("**/metadata/**/*.json"))
        LOGGER.info(
            "Metadata manifest absent; discovered %d JSON files by directory scan",
            len(paths),
        )
    missing = [path for path in paths if not path.is_file()]
    if missing:
        preview = ", ".join(str(path) for path in missing[:3])
        raise FileNotFoundError(
            f"{len(missing)} metadata files listed by the release are missing; "
            f"examples: {preview}"
        )
    if not paths:
        raise FileNotFoundError(
            f"No metadata JSON files were discovered under {dataset_root}"
        )
    return paths


def load_archive_index(dataset_root: Path) -> dict[str, dict[str, str]]:
    index_path = dataset_root / "manifests" / "archive_index.csv"
    if not index_path.is_file():
        LOGGER.warning(
            "Archive index not found at %s; CSI paths will be inferred from "
            "metadata directory structure",
            index_path,
        )
        return {}
    frame = pd.read_csv(index_path, dtype=str, keep_default_na=False)
    required = {"sample_id", "archive_csi_path"}
    missing = required - set(frame.columns)
    if missing:
        raise ValueError(
            f"{index_path} is missing required columns: {sorted(missing)}"
        )
    if frame["sample_id"].duplicated().any():
        duplicates = frame.loc[
            frame["sample_id"].duplicated(keep=False), "sample_id"
        ].tolist()
        raise ValueError(
            "Archive index contains duplicate sample_id values: "
            f"{duplicates[:5]}"
        )
    LOGGER.info("Loaded archive index with %d rows", len(frame))
    return {
        str(row["sample_id"]): {
            str(column): normalize_text(row[column]) for column in frame.columns
        }
        for _, row in frame.iterrows()
    }


def is_unoccupied_record(fields: Mapping[str, Any]) -> tuple[bool, str]:
    """Use explicit metadata evidence rather than state-name assumptions."""

    people = values_for_leaf_names(
        fields, ("person_id", "subject_id", "participant_id", "person")
    )
    for person in people:
        if person.lower() in UNOCCUPIED_PERSON_VALUES:
            return True, f"person marker: {person}"

    occupancy_values = values_for_leaf_names(
        fields,
        (
            "occupancy",
            "occupancy_status",
            "presence",
            "presence_status",
            "is_occupied",
            "occupied",
        ),
    )
    for value in occupancy_values:
        if value.lower() in {
            "false",
            "0",
            "none",
            "no_person",
            "unoccupied",
            "vacant",
        }:
            return True, f"occupancy marker: {value}"

    activities = values_for_leaf_names(
        fields,
        (
            "action_id",
            "action",
            "activity_id",
            "activity",
            "scenario",
            "environment_state",
        ),
    )
    for activity in activities:
        if activity.lower() in UNOCCUPIED_ACTIVITY_VALUES:
            return True, f"activity marker: {activity}"
    return False, ""


def duration_from_metadata(fields: Mapping[str, Any]) -> float:
    duration = first_field_value(
        fields,
        (
            "action_duration_sec",
            "duration_sec",
            "recording_duration_sec",
            "csi_duration_sec",
        ),
    )
    if duration:
        try:
            return float(duration)
        except ValueError:
            pass

    start = first_field_value(
        fields, ("action_start_unix_ns", "start_unix_ns")
    )
    end = first_field_value(fields, ("action_end_unix_ns", "end_unix_ns"))
    if start and end:
        try:
            return (float(end) - float(start)) / 1_000_000_000.0
        except ValueError:
            pass
    return float("nan")


def parse_state_and_link(
    *,
    sample_id: str,
    session_id: str,
    person_id: str,
    action_id: str,
    environment_id: str,
    trial_id: str,
    fields: Mapping[str, Any],
) -> tuple[str, str, str]:
    """Discover state fields, falling back to metadata-anchored sample_id text."""

    environment_states = values_for_leaf_names(
        fields, ("environment_state", "state", "state_id")
    )
    if environment_states:
        state_token = "_".join(environment_states)
        state_source = "metadata:environment_state"
    else:
        conditions = values_for_leaf_names(
            fields, ("condition", "condition_id")
        )
        scenarios = values_for_leaf_names(fields, ("scenario", "scenario_id"))
        structured_values: list[str] = []
        for value in [*conditions, *scenarios]:
            if value not in structured_values:
                structured_values.append(value)
        if structured_values:
            state_token = "_".join(structured_values)
            state_source = "metadata:condition/scenario"
        else:
            if not environment_id or not trial_id:
                raise ValueError(
                    f"{sample_id}: cannot parse state without environment_id "
                    "and trial_id metadata anchors"
                )
            state_match = re.search(
                rf"_{re.escape(environment_id)}_(?P<state>.+)_"
                rf"{re.escape(trial_id)}$",
                sample_id,
            )
            if not state_match:
                raise ValueError(
                    f"{sample_id}: sample_id does not contain the metadata "
                    "environment/trial anchors needed to recover state"
                )
            state_token = state_match.group("state")
            state_source = "sample_id:between_environment_id_and_trial_id"

    prefix = f"{session_id}_{person_id}_{action_id}_"
    environment_marker = f"_{environment_id}_"
    link_configuration = "not_available"
    if sample_id.startswith(prefix) and environment_marker in sample_id:
        suffix_after_prefix = sample_id[len(prefix) :]
        link_configuration = suffix_after_prefix.split(
            environment_marker, 1
        )[0]
    return state_token, link_configuration, state_source


def resolve_csi_path(
    *,
    dataset_root: Path,
    metadata_path: Path,
    sample_id: str,
    archive_row: Mapping[str, str] | None,
    fields: Mapping[str, Any],
) -> tuple[Path, str]:
    candidates: list[tuple[Path, str]] = []
    if archive_row:
        archive_csi_path = archive_row.get("archive_csi_path", "")
        if archive_csi_path:
            candidates.append(
                (dataset_root / archive_csi_path, archive_csi_path)
            )

    for field_name in (
        "archive_csi_path",
        "csi_out_path",
        "csi_path",
        "csi_file",
    ):
        value = first_field_value(fields, (field_name,))
        if not value:
            continue
        candidate = Path(value)
        if candidate.is_absolute():
            candidates.append((candidate, str(candidate)))
        else:
            candidates.append((dataset_root / candidate, value))

    parts = list(metadata_path.parts)
    if "metadata" in parts:
        index = parts.index("metadata")
        parts[index] = "csi"
        sibling = Path(*parts).with_suffix(".dat")
        try:
            sibling_relative = sibling.relative_to(dataset_root).as_posix()
        except ValueError:
            sibling_relative = str(sibling)
        candidates.append((sibling, sibling_relative))

    seen: set[Path] = set()
    for candidate, display_path in candidates:
        candidate = candidate.resolve()
        if candidate in seen:
            continue
        seen.add(candidate)
        if candidate.is_file():
            try:
                relative = candidate.relative_to(dataset_root.resolve())
                return candidate, relative.as_posix()
            except ValueError:
                return candidate, display_path
    raise FileNotFoundError(
        f"{sample_id}: no released CSI file exists among discovered path "
        f"candidates: {[str(path) for path, _ in candidates]}"
    )


def discover_recordings(
    dataset_root: Path,
    metadata_paths: Sequence[Path],
    archive_index: Mapping[str, Mapping[str, str]],
) -> tuple[list[Recording], pd.DataFrame, pd.DataFrame]:
    """Scan metadata, inventory fields, and return unoccupied recordings."""

    field_counts: Counter[str] = Counter()
    field_values: dict[str, set[str]] = defaultdict(set)
    field_examples: dict[str, list[str]] = defaultdict(list)
    recordings: list[Recording] = []
    selection_reasons: Counter[str] = Counter()

    for metadata_path in metadata_paths:
        with metadata_path.open("r", encoding="utf-8") as handle:
            metadata = json.load(handle)
        fields = scalar_field_map(metadata)
        for field_path, raw_value in fields.items():
            value = normalize_text(raw_value)
            field_counts[field_path] += 1
            field_values[field_path].add(value)
            if value not in field_examples[field_path]:
                if len(field_examples[field_path]) < 3:
                    field_examples[field_path].append(value)

        is_unoccupied, reason = is_unoccupied_record(fields)
        if not is_unoccupied:
            continue
        selection_reasons[reason] += 1

        sample_id = first_field_value(fields, ("sample_id",))
        if not sample_id:
            raise ValueError(f"{metadata_path}: unoccupied record has no sample_id")
        session_id = first_field_value(fields, ("session_id",))
        person_id = first_field_value(
            fields,
            ("person_id", "subject_id", "participant_id"),
            default="not_available",
        )
        environment_id = first_field_value(
            fields,
            ("environment_id", "environment", "room_id", "room"),
            default="not_available",
        )
        action_id = first_field_value(
            fields,
            ("action_id", "activity_id", "action", "activity"),
            default="not_available",
        )
        trial_id = first_field_value(
            fields,
            ("trial_id", "recording_id", "repetition_id"),
            default="not_available",
        )
        state_token, link_configuration, state_source = parse_state_and_link(
            sample_id=sample_id,
            session_id=session_id,
            person_id=person_id,
            action_id=action_id,
            environment_id=environment_id,
            trial_id=trial_id,
            fields=fields,
        )
        state = f"{environment_id}__{state_token}"
        archive_row = archive_index.get(sample_id)
        csi_path, csi_relative = resolve_csi_path(
            dataset_root=dataset_root,
            metadata_path=metadata_path,
            sample_id=sample_id,
            archive_row=archive_row,
            fields=fields,
        )
        try:
            metadata_relative = metadata_path.resolve().relative_to(
                dataset_root.resolve()
            )
            metadata_relative_text = metadata_relative.as_posix()
        except ValueError:
            metadata_relative_text = str(metadata_path)
        recordings.append(
            Recording(
                sample_id=sample_id,
                session_id=session_id,
                person_id=person_id,
                environment_id=environment_id,
                action_id=action_id,
                trial_id=trial_id,
                link_configuration=link_configuration,
                state_token=state_token,
                state=state,
                state_source=state_source,
                duration_sec=duration_from_metadata(fields),
                csi_path=csi_path,
                csi_path_relative=csi_relative,
                metadata_path=metadata_path.resolve(),
                metadata_path_relative=metadata_relative_text,
            )
        )

    sample_ids = [recording.sample_id for recording in recordings]
    duplicates = [
        sample_id
        for sample_id, count in Counter(sample_ids).items()
        if count > 1
    ]
    if duplicates:
        raise ValueError(
            f"Duplicate unoccupied sample_id values: {duplicates[:5]}"
        )
    if not recordings:
        raise ValueError(
            "No unoccupied recordings were identified from explicit metadata "
            "person/occupancy/activity markers"
        )

    recordings.sort(
        key=lambda row: (
            row.environment_id,
            row.state_token,
            row.trial_id,
            row.sample_id,
        )
    )
    LOGGER.info(
        "Selected %d independent unoccupied recordings across %d states",
        len(recordings),
        len({recording.state for recording in recordings}),
    )
    for reason, count in sorted(selection_reasons.items()):
        LOGGER.info("Selection evidence '%s': %d records", reason, count)

    field_inventory = pd.DataFrame(
        [
            {
                "field_path": field_path,
                "field_name": leaf_name(field_path),
                "populated_metadata_files": count,
                "unique_value_count": len(field_values[field_path]),
                "example_values": " | ".join(field_examples[field_path]),
            }
            for field_path, count in sorted(field_counts.items())
        ]
    )

    requested_rows: list[dict[str, Any]] = []
    for requested_name, aliases in REQUESTED_FIELD_ALIASES.items():
        matches = sorted(
            field_path
            for field_path in field_counts
            if leaf_name(field_path) in {alias.lower() for alias in aliases}
        )
        requested_rows.append(
            {
                "requested_concept": requested_name,
                "status": "found" if matches else "not_present",
                "discovered_field_paths": " | ".join(matches),
                "example_values": " | ".join(
                    sorted(
                        {
                            value
                            for match in matches
                            for value in field_examples[match]
                        }
                    )[:6]
                ),
            }
        )
    requested_discovery = pd.DataFrame(requested_rows)
    for row in requested_rows:
        LOGGER.info(
            "Requested field concept %-11s -> %s%s",
            row["requested_concept"],
            row["status"],
            (
                f" ({row['discovered_field_paths']})"
                if row["discovered_field_paths"]
                else ""
            ),
        )
    return recordings, field_inventory, requested_discovery


def recording_inventory_frame(recordings: Sequence[Recording]) -> pd.DataFrame:
    group_counts = Counter(recording.state for recording in recordings)
    group_indices: Counter[str] = Counter()
    rows: list[dict[str, Any]] = []
    for recording in recordings:
        group_indices[recording.state] += 1
        rows.append(
            {
                "state": recording.state,
                "state_token": recording.state_token,
                "environment_id": recording.environment_id,
                "link_configuration": recording.link_configuration,
                "state_source": recording.state_source,
                "num_recordings_in_state": group_counts[recording.state],
                "recording_index_within_state": group_indices[recording.state],
                "sample_id": recording.sample_id,
                "trial_id": recording.trial_id,
                "duration_sec": recording.duration_sec,
                "csi_path": recording.csi_path_relative,
                "csi_path_absolute": str(recording.csi_path),
                "metadata_path": recording.metadata_path_relative,
            }
        )
    return pd.DataFrame(rows)


def extract_recording_statistics(
    recordings: Sequence[Recording],
    *,
    read_feitcsi: Any,
    interpolate_low_energy_subcarriers: Any,
    low_energy_ratio: float,
) -> list[RecordingStatistics]:
    results: list[RecordingStatistics] = []
    reference_dimensions: tuple[int, int, int] | None = None

    for index, recording in enumerate(recordings, start=1):
        LOGGER.info(
            "CSI %02d/%02d | %s | %s",
            index,
            len(recordings),
            recording.state,
            recording.sample_id,
        )
        csi, headers = read_feitcsi(recording.csi_path)
        if csi.ndim != 4:
            raise ValueError(
                f"{recording.sample_id}: expected CSI dimensions "
                f"(packet, rx, tx, subcarrier), got {csi.shape}"
            )
        packet_count, num_rx, num_tx, num_subcarriers = csi.shape
        if num_tx < 1:
            raise ValueError(
                f"{recording.sample_id}: no TX stream in CSI shape {csi.shape}"
            )
        dimensions = (num_rx, num_tx, num_subcarriers)
        if reference_dimensions is None:
            reference_dimensions = dimensions
            LOGGER.info(
                "Detected CSI layout: %d RX, %d TX, %d subcarriers",
                num_rx,
                num_tx,
                num_subcarriers,
            )
        elif dimensions != reference_dimensions:
            raise ValueError(
                f"{recording.sample_id}: CSI dimensions {dimensions} differ "
                f"from the first recording {reference_dimensions}"
            )

        amplitude = np.abs(csi[:, :, 0, :]).astype(np.float32, copy=False)
        repaired_amplitude, valid_mask = interpolate_low_energy_subcarriers(
            amplitude,
            low_energy_ratio=low_energy_ratio,
            epsilon=1e-6,
        )
        mean_curve = repaired_amplitude.mean(
            axis=(0, 1), dtype=np.float64
        )
        std_curve = repaired_amplitude.std(axis=(0, 1), dtype=np.float64)
        amplitude_scale = float(np.mean(mean_curve))
        if not np.isfinite(amplitude_scale) or amplitude_scale <= EPSILON:
            raise ValueError(
                f"{recording.sample_id}: invalid amplitude normalization scale "
                f"{amplitude_scale}"
            )
        normalized_mean = mean_curve / amplitude_scale
        normalized_std = std_curve / amplitude_scale
        normalized_variance = normalized_std**2
        arrays = (
            mean_curve,
            std_curve,
            normalized_mean,
            normalized_std,
            normalized_variance,
        )
        if not all(np.isfinite(array).all() for array in arrays):
            raise ValueError(
                f"{recording.sample_id}: amplitude statistics contain "
                "non-finite values"
            )

        results.append(
            RecordingStatistics(
                recording=recording,
                packet_count=packet_count,
                num_rx=num_rx,
                num_tx=num_tx,
                num_subcarriers=num_subcarriers,
                low_energy_subcarrier_count=int((~valid_mask).sum()),
                amplitude_scale=amplitude_scale,
                mean_curve=mean_curve,
                std_curve=std_curve,
                normalized_mean_curve=normalized_mean,
                normalized_std_curve=normalized_std,
                normalized_variance_curve=normalized_variance,
            )
        )
        del csi, amplitude, repaired_amplitude
    return results


def save_record_statistics(
    statistics: Sequence[RecordingStatistics], output_dir: Path
) -> None:
    curve_rows: list[pd.DataFrame] = []
    for item in statistics:
        curve_rows.append(
            pd.DataFrame(
                {
                    "state": item.recording.state,
                    "sample_id": item.recording.sample_id,
                    "trial_id": item.recording.trial_id,
                    "subcarrier_index": np.arange(item.num_subcarriers),
                    "mean_amplitude": item.mean_curve,
                    "std_amplitude": item.std_curve,
                    "normalized_mean_amplitude": item.normalized_mean_curve,
                    "normalized_std_amplitude": item.normalized_std_curve,
                    "normalized_variance": item.normalized_variance_curve,
                    "relative_mean_amplitude": item.normalized_mean_curve,
                    "temporal_sd_relative_amplitude": (
                        item.normalized_std_curve
                    ),
                    "relative_amplitude_variance": (
                        item.normalized_variance_curve
                    ),
                }
            )
        )
    pd.concat(curve_rows, ignore_index=True).to_csv(
        output_dir / "amplitude_curve_statistics.csv",
        index=False,
        float_format="%.10g",
    )

    np.savez_compressed(
        output_dir / "recording_feature_cache.npz",
        sample_id=np.asarray(
            [item.recording.sample_id for item in statistics]
        ),
        state=np.asarray([item.recording.state for item in statistics]),
        mean_amplitude=np.stack([item.mean_curve for item in statistics]),
        std_amplitude=np.stack([item.std_curve for item in statistics]),
        normalized_mean_amplitude=np.stack(
            [item.normalized_mean_curve for item in statistics]
        ),
        normalized_std_amplitude=np.stack(
            [item.normalized_std_curve for item in statistics]
        ),
        normalized_variance=np.stack(
            [item.normalized_variance_curve for item in statistics]
        ),
        relative_mean_amplitude=np.stack(
            [item.normalized_mean_curve for item in statistics]
        ),
        temporal_sd_relative_amplitude=np.stack(
            [item.normalized_std_curve for item in statistics]
        ),
        relative_amplitude_variance=np.stack(
            [item.normalized_variance_curve for item in statistics]
        ),
    )


def group_statistics(
    statistics: Sequence[RecordingStatistics],
) -> dict[str, list[RecordingStatistics]]:
    grouped: dict[str, list[RecordingStatistics]] = defaultdict(list)
    for item in statistics:
        grouped[item.recording.state].append(item)
    return dict(sorted(grouped.items()))


def save_figure(fig: matplotlib.figure.Figure, stem: Path, dpi: int) -> None:
    raster_dpi = max(int(dpi), FIGURE_DPI)
    fig.savefig(
        stem.with_suffix(".png"),
        dpi=raster_dpi,
        facecolor="white",
        bbox_inches="tight",
        pad_inches=0.03,
    )
    fig.savefig(
        stem.with_suffix(".pdf"),
        facecolor="white",
        bbox_inches="tight",
        pad_inches=0.03,
    )
    plt.close(fig)


def _environment_subplot_layout(num_states: int) -> tuple[int, int]:
    if num_states <= 0:
        raise ValueError("An environment figure needs at least one state")
    if num_states <= 3:
        return 1, num_states
    if num_states == 4:
        return 2, 2
    return 2, math.ceil(num_states / 2)


def _curves_for_kind(
    items: Sequence[RecordingStatistics],
    curve_kind: str,
) -> np.ndarray:
    if curve_kind == "mean":
        return np.stack([item.normalized_mean_curve for item in items])
    if curve_kind == "std":
        return np.stack([item.normalized_std_curve for item in items])
    raise ValueError(f"Unsupported curve kind: {curve_kind}")


def plot_environment_amplitude_supplements(
    grouped: Mapping[str, Sequence[RecordingStatistics]],
    *,
    output_dir: Path,
    dpi: int,
    curve_kind: str,
) -> None:
    """Create compact environment-specific supplementary CSI curve figures."""

    if curve_kind not in {"mean", "std"}:
        raise ValueError(f"Unsupported curve kind: {curve_kind}")
    validate_paper_labels(grouped)
    for environment_id in ENVIRONMENT_LABELS:
        states = [
            state
            for state in STATE_ORDER
            if state in grouped and state.startswith(f"{environment_id}__")
        ]
        if not states:
            continue
        nrows, ncols = _environment_subplot_layout(len(states))
        figure_width = max(6.8, 2.9 * ncols)
        figure_height = 2.45 * nrows + 0.55
        fig, axes = plt.subplots(
            nrows,
            ncols,
            figsize=(figure_width, figure_height),
            sharex=True,
            sharey=True,
            squeeze=False,
        )
        axes_array = axes.ravel()

        all_curve_values: list[np.ndarray] = []
        for state in states:
            state_curves = _curves_for_kind(grouped[state], curve_kind)
            all_curve_values.append(state_curves.reshape(-1))
            if curve_kind == "mean":
                across_mean = state_curves.mean(axis=0)
                across_sd = state_curves.std(axis=0, ddof=1)
                all_curve_values.extend(
                    [across_mean - across_sd, across_mean + across_sd]
                )
        flattened = np.concatenate(all_curve_values)
        y_min = float(np.nanmin(flattened))
        y_max = float(np.nanmax(flattened))
        y_padding = max((y_max - y_min) * 0.06, 0.01)
        shared_y_limits = (max(0.0, y_min - y_padding), y_max + y_padding)

        for state_index, (axis, state) in enumerate(
            zip(axes_array, states)
        ):
            items = grouped[state]
            curves = _curves_for_kind(items, curve_kind)
            subcarrier_index = np.arange(curves.shape[1])
            for curve in curves:
                axis.plot(
                    subcarrier_index,
                    curve,
                    color=RECORDING_COLOR,
                    alpha=0.52,
                    linewidth=0.7,
                    zorder=1,
                )
            across_mean = curves.mean(axis=0)
            if curve_kind == "mean":
                across_sd = curves.std(axis=0, ddof=1)
                axis.fill_between(
                    subcarrier_index,
                    across_mean - across_sd,
                    across_mean + across_sd,
                    color=BAND_COLOR,
                    alpha=0.18,
                    linewidth=0,
                    zorder=2,
                )
            axis.plot(
                subcarrier_index,
                across_mean,
                color=MEAN_CURVE_COLOR,
                linewidth=1.7,
                zorder=3,
            )
            axis.set_title(
                f"{paper_state_label(state)} (n={len(items)})",
                pad=4,
            )
            axis.set_xlim(0, curves.shape[1] - 1)
            axis.set_ylim(*shared_y_limits)
            row_index, column_index = divmod(state_index, ncols)
            if row_index == nrows - 1:
                axis.set_xlabel("Subcarrier index")
            style_publication_axis(axis, grid_axis="both")

        for axis in axes_array[len(states) :]:
            axis.set_visible(False)

        legend_handles: list[Any] = [
            Line2D(
                [0],
                [0],
                color=RECORDING_COLOR,
                alpha=0.52,
                linewidth=0.7,
                label="Independent recording",
            ),
            Line2D(
                [0],
                [0],
                color=MEAN_CURVE_COLOR,
                linewidth=1.7,
                label="Across-recording mean",
            ),
        ]
        if curve_kind == "mean":
            legend_handles.append(
                Patch(
                    facecolor=BAND_COLOR,
                    alpha=0.18,
                    edgecolor="none",
                    label="Across-recording ±1 SD",
                )
            )
        fig.legend(
            handles=legend_handles,
            loc="upper center",
            bbox_to_anchor=(0.5, 0.995),
            ncol=len(legend_handles),
            frameon=False,
            columnspacing=1.5,
            handlelength=2.2,
        )
        fig.supylabel(
            "Relative CSI amplitude"
            if curve_kind == "mean"
            else "Temporal SD of relative CSI amplitude",
            x=0.01,
            fontsize=9,
        )
        fig.subplots_adjust(
            left=0.09 if ncols >= 3 else 0.12,
            right=0.99,
            bottom=0.13,
            top=0.84,
            wspace=0.18,
            hspace=0.28,
        )
        output_stem = SUPPLEMENTARY_ENVIRONMENT_FILES[environment_id][
            0 if curve_kind == "mean" else 1
        ]
        save_figure(fig, output_dir / output_stem, dpi)


def slugify(value: str) -> str:
    slug = re.sub(r"[^A-Za-z0-9._-]+", "_", value).strip("_")
    return slug or "state"


def correlation_analysis(
    grouped: Mapping[str, Sequence[RecordingStatistics]],
    *,
    output_dir: Path,
    dpi: int,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    heatmap_dir = output_dir / "supplementary_correlation_heatmaps"
    matrix_dir = output_dir / "correlation_matrices"
    heatmap_dir.mkdir(parents=True, exist_ok=True)
    matrix_dir.mkdir(parents=True, exist_ok=True)

    state_rows: list[dict[str, Any]] = []
    pair_rows: list[dict[str, Any]] = []
    for state, items in grouped.items():
        curves = np.stack([item.normalized_mean_curve for item in items])
        matrix = np.corrcoef(curves)
        if np.ndim(matrix) == 0:
            matrix = np.asarray([[1.0]])
        labels = [item.recording.trial_id for item in items]
        sample_ids = [item.recording.sample_id for item in items]
        matrix_frame = pd.DataFrame(
            matrix, index=sample_ids, columns=sample_ids
        )
        matrix_frame.index.name = "sample_id"
        matrix_frame.to_csv(
            matrix_dir / f"{slugify(state)}_pearson_correlation.csv",
            float_format="%.8f",
        )

        upper = matrix[np.triu_indices(len(items), k=1)]
        mean_corr = float(np.mean(upper)) if upper.size else float("nan")
        std_corr = (
            float(np.std(upper, ddof=1)) if upper.size > 1 else float("nan")
        )
        state_rows.append(
            {
                "state": state,
                "environment_id": items[0].recording.environment_id,
                "state_token": items[0].recording.state_token,
                "environment_label": paper_environment_label(
                    items[0].recording.environment_id
                ),
                "paper_state_label": paper_state_label(state),
                "num_recordings": len(items),
                "num_recording_pairs": int(upper.size),
                "mean_intra_corr": mean_corr,
                "std_intra_corr": std_corr,
                "min_intra_corr": (
                    float(np.min(upper)) if upper.size else float("nan")
                ),
                "max_intra_corr": (
                    float(np.max(upper)) if upper.size else float("nan")
                ),
            }
        )
        for left in range(len(items)):
            for right in range(left + 1, len(items)):
                pair_rows.append(
                    {
                        "state": state,
                        "sample_id_a": sample_ids[left],
                        "sample_id_b": sample_ids[right],
                        "pearson_correlation": matrix[left, right],
                    }
                )

        fig, axis = plt.subplots(figsize=(4.1, 3.55))
        image = axis.imshow(
            matrix,
            cmap="coolwarm",
            vmin=-1,
            vmax=1,
            interpolation="nearest",
            aspect="equal",
        )
        axis.set_xticks(np.arange(len(labels)), labels=labels)
        axis.set_yticks(np.arange(len(labels)), labels=labels)
        for row_index in range(len(labels)):
            for column_index in range(len(labels)):
                value = float(matrix[row_index, column_index])
                axis.text(
                    column_index,
                    row_index,
                    f"{value:.3f}",
                    ha="center",
                    va="center",
                    fontsize=7.2,
                    color="white" if abs(value) >= 0.55 else "#202020",
                )
        colorbar = fig.colorbar(
            image,
            ax=axis,
            fraction=0.048,
            pad=0.04,
            ticks=[-1, -0.5, 0, 0.5, 1],
        )
        colorbar.set_label("Pearson correlation", fontsize=8)
        colorbar.ax.tick_params(labelsize=7, width=0.5, length=2.5)
        axis.set_title(
            f"{paper_environment_label(items[0].recording.environment_id)}"
            f" — {paper_state_label(state)} (n={len(items)})",
            pad=6,
        )
        axis.set_xlabel("Independent recording")
        axis.set_ylabel("Independent recording")
        axis.spines["top"].set_visible(True)
        axis.spines["right"].set_visible(True)
        axis.spines["left"].set_linewidth(0.6)
        axis.spines["bottom"].set_linewidth(0.6)
        axis.spines["top"].set_linewidth(0.6)
        axis.spines["right"].set_linewidth(0.6)
        fig.subplots_adjust(left=0.17, right=0.87, bottom=0.16, top=0.88)
        save_figure(
            fig,
            heatmap_dir / f"{slugify(state)}_correlation_heatmap",
            dpi,
        )

    state_frame = pd.DataFrame(state_rows)
    pair_frame = pd.DataFrame(pair_rows)
    state_frame.to_csv(
        output_dir / "state_correlation_statistics.csv",
        index=False,
        float_format="%.8f",
    )
    pair_frame.to_csv(
        output_dir / "within_state_correlation_pairs.csv",
        index=False,
        float_format="%.8f",
    )
    return state_frame, pair_frame


def _distance_plot_frame(pair_frame: pd.DataFrame) -> pd.DataFrame:
    plot_frame = pair_frame.loc[
        pair_frame["comparison_type"].isin(
            [
                "intra_state",
                "inter_state_matched_environment_link",
            ]
        ),
        ["comparison_type", "distance"],
    ].copy()
    plot_frame["distance_class"] = plot_frame["comparison_type"].map(
        {
            "intra_state": "Intra-state",
            "inter_state_matched_environment_link": "Inter-state",
        }
    )
    return plot_frame


def draw_distance_distribution(
    axis: matplotlib.axes.Axes,
    pair_frame: pd.DataFrame,
    *,
    compact_ylabel: bool = False,
    show_raw_scatter: bool = True,
    show_fliers: bool = False,
    box_width: float = 0.42,
    display_labels: tuple[str, str] = (
        "Intra-state",
        "Inter-state",
    ),
    ylabel: str | None = None,
    median_linewidth: float = 1.25,
    box_alpha: float = 0.72,
    show_counts: bool = False,
) -> None:
    """Draw pair-distance distributions with optional raw observations."""

    plot_frame = _distance_plot_frame(pair_frame)
    categories = ("Intra-state", "Inter-state")
    arrays = [
        plot_frame.loc[
            plot_frame["distance_class"] == category, "distance"
        ].to_numpy(dtype=float)
        for category in categories
    ]
    if show_raw_scatter:
        rng = np.random.default_rng(PLOT_RANDOM_SEED)
        for position, values in enumerate(arrays, start=1):
            jitter = rng.uniform(-0.06, 0.06, size=len(values))
            axis.scatter(
                np.full(len(values), position, dtype=float) + jitter,
                values,
                s=9,
                color="#3E3E3E",
                alpha=0.18,
                linewidths=0,
                zorder=1,
            )
    boxplot = axis.boxplot(
        arrays,
        positions=[1, 2],
        widths=box_width,
        patch_artist=True,
        showfliers=show_fliers,
        whis=1.5,
        medianprops={
            "color": "#111111",
            "linewidth": median_linewidth,
        },
        whiskerprops={"color": "#777777", "linewidth": 0.9},
        capprops={"color": "#777777", "linewidth": 0.9},
        boxprops={"edgecolor": "#555555", "linewidth": 0.9},
        flierprops={
            "marker": "o",
            "markersize": 2.5,
            "markerfacecolor": "none",
            "markeredgecolor": "#888888",
            "markeredgewidth": 0.65,
            "alpha": 0.55,
        },
        zorder=2,
    )
    for patch, color in zip(
        boxplot["boxes"], (INTRA_COLOR, INTER_COLOR)
    ):
        patch.set_facecolor(color)
        patch.set_alpha(box_alpha)
    tick_labels: tuple[str, ...] = display_labels
    if show_counts:
        tick_labels = tuple(
            f"{label}\nn = {len(values)}"
            for label, values in zip(display_labels, arrays)
        )
    axis.set_xticks([1, 2], tick_labels)
    axis.set_xlim(0.55, 2.45)
    axis.set_xlabel("")
    if ylabel is None:
        ylabel = (
            "Euclidean distance"
            if compact_ylabel
            else "Euclidean distance in standardized feature space"
        )
    axis.set_ylabel(ylabel)
    style_publication_axis(axis, grid_axis="y")


def plot_pair_level_distance_boxplot(
    pair_frame: pd.DataFrame,
    *,
    output_dir: Path,
    dpi: int,
) -> None:
    """Save the standalone main-text candidate for the distance distribution."""

    fig, axis = plt.subplots(figsize=(3.45, 3.15))
    draw_distance_distribution(axis, pair_frame)
    add_panel_label(axis, "(a)", x=-0.20, y=1.02)
    fig.subplots_adjust(left=0.22, right=0.98, bottom=0.16, top=0.96)
    save_figure(
        fig,
        output_dir / "main_panel_a_distance_distribution",
        dpi,
    )


def grouped_state_y_layout(
    frame: pd.DataFrame,
    *,
    heading_gap: float = 0.72,
    group_gap: float = 0.38,
) -> tuple[pd.DataFrame, np.ndarray, dict[str, float], float]:
    """Create state rows separated by non-data environment heading rows."""

    if heading_gap <= 0 or group_gap < 0:
        raise ValueError("Grouped-state spacing must be non-negative")
    ordered = ordered_state_frame(frame)
    positions: list[float] = []
    headings: dict[str, float] = {}
    cursor = 0.0
    for environment_id in ENVIRONMENT_LABELS:
        environment_mask = ordered["environment_id"] == environment_id
        environment_rows = ordered.loc[environment_mask]
        if environment_rows.empty:
            continue
        headings[environment_id] = cursor
        cursor += heading_gap
        for _ in range(len(environment_rows)):
            positions.append(cursor)
            cursor += 1.0
        cursor += group_gap
    if len(positions) != len(ordered):
        raise ValueError("Grouped plotting layout did not cover every state")
    return ordered, np.asarray(positions), headings, cursor


def _draw_environment_headings(
    axis: matplotlib.axes.Axes,
    headings: Mapping[str, float],
    *,
    fontsize: float = 8.0,
) -> None:
    for environment_id, position in headings.items():
        axis.text(
            -0.015,
            position,
            paper_environment_label(environment_id).upper(),
            transform=axis.get_yaxis_transform(),
            ha="right",
            va="center",
            fontsize=fontsize,
            fontweight="bold",
            color="#303030",
            clip_on=False,
        )


def draw_state_level_distances(
    axis: matplotlib.axes.Axes,
    state_frame: pd.DataFrame,
    *,
    show_legend: bool,
    state_label_style: str = "paper",
    heading_gap: float = 0.72,
    show_ratio_column: bool = True,
    legend_labels: tuple[str, str] = (
        "Intra-state mean ± SD",
        "Inter-state mean ± SD",
    ),
    x_label: str = (
        "Mean Euclidean distance in standardized feature space"
    ),
    marker_size: float = 4.8,
    errorbar_linewidth: float = 0.8,
    errorbar_alpha: float = 1.0,
    connector_linewidth: float = 0.85,
    connector_alpha: float = 1.0,
    connector_color: str = CONNECTOR_COLOR,
    nearest_state_frame: pd.DataFrame | None = None,
    state_label_fontsize: float | None = None,
    heading_fontsize: float = 8.0,
    legend_fontsize: float | None = None,
    legend_ncol: int = 2,
    legend_bottom_padding: float = 0.55,
) -> None:
    """Draw state means with descriptive mean ± SD error bars."""

    required_columns = {
        "state",
        "environment_id",
        "mean_intra_distance",
        "std_intra_distance",
        "mean_inter_distance",
        "std_inter_distance",
    }
    missing = required_columns - set(state_frame.columns)
    if missing:
        raise ValueError(
            "State-level distance plot is missing columns: "
            f"{sorted(missing)}"
        )
    if nearest_state_frame is not None:
        if show_ratio_column:
            raise ValueError(
                "nearest_state_frame cannot be combined with "
                "show_ratio_column; the inline ratio column and the "
                "nearest-state ratio are different quantities"
            )
        nearest_required = {"state", "nearest_state_ratio"}
        nearest_missing = nearest_required - set(
            nearest_state_frame.columns
        )
        if nearest_missing:
            raise ValueError(
                "Nearest-state frame is missing columns: "
                f"{sorted(nearest_missing)}"
            )
    plot_frame, positions, headings, layout_end = grouped_state_y_layout(
        state_frame,
        heading_gap=heading_gap,
    )
    distance_columns = [
        "mean_intra_distance",
        "std_intra_distance",
        "mean_inter_distance",
        "std_inter_distance",
    ]
    if plot_frame[distance_columns].isna().any().any():
        raise ValueError(
            "State-level distance plot cannot contain missing distances"
        )

    intra = plot_frame["mean_intra_distance"].to_numpy(dtype=float)
    intra_sd = plot_frame["std_intra_distance"].to_numpy(dtype=float)
    inter = plot_frame["mean_inter_distance"].to_numpy(dtype=float)
    inter_sd = plot_frame["std_inter_distance"].to_numpy(dtype=float)
    maximum_distance = float(
        np.max(
            np.concatenate(
                (intra + intra_sd, inter + inter_sd)
            )
        )
    )

    for position, intra_value, inter_value in zip(
        positions, intra, inter
    ):
        axis.plot(
            [intra_value, inter_value],
            [position, position],
            color=connector_color,
            linewidth=connector_linewidth,
            alpha=connector_alpha,
            zorder=1,
        )
    axis.errorbar(
        intra,
        positions,
        xerr=intra_sd,
        fmt="o",
        markersize=marker_size,
        color=INTRA_COLOR,
        ecolor=INTRA_COLOR,
        elinewidth=errorbar_linewidth,
        alpha=errorbar_alpha,
        capsize=2.0,
        capthick=0.7,
        markeredgecolor="white",
        markeredgewidth=0.45,
        label=legend_labels[0],
        zorder=3,
    )
    axis.errorbar(
        inter,
        positions,
        xerr=inter_sd,
        fmt="s",
        markersize=marker_size,
        color=INTER_COLOR,
        ecolor=INTER_COLOR,
        elinewidth=errorbar_linewidth,
        alpha=errorbar_alpha,
        capsize=2.0,
        capthick=0.7,
        markeredgecolor="white",
        markeredgewidth=0.45,
        label=legend_labels[1],
        zorder=3,
    )
    if errorbar_alpha < 1.0:
        marker_area = marker_size**2
        axis.scatter(
            intra,
            positions,
            s=marker_area,
            marker="o",
            color=INTRA_COLOR,
            edgecolor="white",
            linewidth=0.45,
            zorder=4,
        )
        axis.scatter(
            inter,
            positions,
            s=marker_area,
            marker="s",
            color=INTER_COLOR,
            edgecolor="white",
            linewidth=0.45,
            zorder=4,
        )
    if show_ratio_column:
        ratios = inter / np.maximum(intra, EPSILON)
        ratio_x = maximum_distance * 1.045
        for position, ratio in zip(positions, ratios):
            axis.text(
                ratio_x,
                position,
                f"{ratio:.2f}",
                ha="left",
                va="center",
                fontsize=7.4,
                color="#404040",
            )
        first_heading = min(headings.values())
        axis.text(
            ratio_x,
            first_heading,
            "Ratio",
            ha="left",
            va="center",
            fontsize=8,
            fontweight="bold",
            clip_on=False,
        )

    if nearest_state_frame is not None:
        ratio_lookup = dict(
            zip(
                nearest_state_frame["state"].astype(str),
                nearest_state_frame["nearest_state_ratio"].astype(float),
            )
        )
        ratio_x = maximum_distance * 1.045
        for position, state in zip(
            positions, plot_frame["state"].astype(str)
        ):
            if state not in ratio_lookup:
                raise ValueError(
                    f"Nearest-state frame has no ratio for state {state!r}"
                )
            ratio = ratio_lookup[state]
            below_threshold = ratio < 1.0
            axis.text(
                ratio_x,
                position,
                f"{ratio:.2f}",
                ha="left",
                va="center",
                fontsize=7.4,
                color=(
                    RATIO_FLAG_COLOR if below_threshold else "#404040"
                ),
                fontweight="bold" if below_threshold else "normal",
            )
        first_heading = min(headings.values())
        axis.text(
            ratio_x,
            first_heading,
            "Nearest-state\nratio",
            ha="left",
            va="center",
            fontsize=7.2,
            fontweight="bold",
            clip_on=False,
        )

    if state_label_style == "paper":
        state_labels = [
            paper_state_label(state) for state in plot_frame["state"]
        ]
    elif state_label_style == "code_short":
        state_labels = [
            paper_state_code_label(state) for state in plot_frame["state"]
        ]
    elif state_label_style == "code_v5_short":
        state_labels = [
            paper_state_v5_code_label(state)
            for state in plot_frame["state"]
        ]
    else:
        raise ValueError(
            f"Unsupported state-label style: {state_label_style!r}"
        )
    axis.set_yticks(positions, state_labels)
    if state_label_style in {"code_short", "code_v5_short"}:
        for label in axis.get_yticklabels():
            label.set_horizontalalignment("right")
            label.set_multialignment("right")
        axis.tick_params(axis="y", pad=6)
    if state_label_fontsize is not None:
        axis.tick_params(axis="y", labelsize=state_label_fontsize)
    axis.set_ylim(
        layout_end
        + (legend_bottom_padding if show_legend else 0.1),
        -0.38,
    )
    axis.set_xlim(
        0,
        maximum_distance
        * (
            1.28
            if nearest_state_frame is not None
            else (1.18 if show_ratio_column else 1.06)
        ),
    )
    axis.set_xlabel(x_label)
    axis.set_ylabel("")
    _draw_environment_headings(
        axis,
        headings,
        fontsize=heading_fontsize,
    )
    style_publication_axis(axis, grid_axis="x")
    if show_legend:
        if errorbar_alpha < 1.0:
            handles = [
                Line2D(
                    [0],
                    [0],
                    marker="o",
                    linestyle="none",
                    markersize=marker_size,
                    markerfacecolor=INTRA_COLOR,
                    markeredgecolor="white",
                    markeredgewidth=0.45,
                    label=legend_labels[0],
                ),
                Line2D(
                    [0],
                    [0],
                    marker="s",
                    linestyle="none",
                    markersize=marker_size,
                    markerfacecolor=INTER_COLOR,
                    markeredgecolor="white",
                    markeredgewidth=0.45,
                    label=legend_labels[1],
                ),
            ]
            axis.legend(
                handles=handles,
                loc="lower left",
                bbox_to_anchor=(0.0, 0.005),
                ncol=legend_ncol,
                frameon=False,
                handlelength=1.4,
                columnspacing=1.2,
                fontsize=legend_fontsize,
            )
        else:
            axis.legend(
                loc="lower left",
                bbox_to_anchor=(0.0, 0.005),
                ncol=legend_ncol,
                frameon=False,
                handlelength=1.4,
                columnspacing=1.2,
                fontsize=legend_fontsize,
            )


def plot_state_level_paired_distances(
    state_frame: pd.DataFrame,
    *,
    output_dir: Path,
    dpi: int,
) -> None:
    """Save the standalone state-level main-text candidate."""

    fig, axis = plt.subplots(figsize=(7.2, 6.65))
    draw_state_level_distances(axis, state_frame, show_legend=True)
    add_panel_label(axis, "(b)", x=-0.38, y=1.01)
    fig.subplots_adjust(left=0.35, right=0.98, bottom=0.09, top=0.99)
    save_figure(
        fig,
        output_dir / "main_panel_b_state_level_distances",
        dpi,
    )


def draw_intra_state_correlation(
    axis: matplotlib.axes.Axes,
    correlation_frame: pd.DataFrame,
    *,
    compact_labels: bool = False,
) -> None:
    """Draw state-level mean Pearson correlation with descriptive ±1 SD."""

    required_columns = {
        "state",
        "environment_id",
        "num_recordings",
        "mean_intra_corr",
        "std_intra_corr",
        "min_intra_corr",
    }
    missing = required_columns - set(correlation_frame.columns)
    if missing:
        raise ValueError(
            "Correlation summary plot is missing columns: "
            f"{sorted(missing)}"
        )
    plot_frame, positions, headings, layout_end = grouped_state_y_layout(
        correlation_frame
    )
    means = plot_frame["mean_intra_corr"].to_numpy(dtype=float)
    standard_deviations = (
        plot_frame["std_intra_corr"].fillna(0).to_numpy(dtype=float)
    )
    axis.errorbar(
        means,
        positions,
        xerr=standard_deviations,
        fmt="o",
        markersize=4.7,
        color=INTRA_COLOR,
        ecolor=INTRA_COLOR,
        elinewidth=0.8,
        capsize=2.0,
        capthick=0.7,
        markeredgecolor="white",
        markeredgewidth=0.45,
        zorder=3,
    )
    state_labels = [
        paper_state_label(state) for state in plot_frame["state"]
    ]
    if compact_labels:
        state_labels = [
            textwrap.fill(label, width=20) for label in state_labels
        ]
    axis.set_yticks(positions, state_labels)
    if compact_labels:
        axis.tick_params(axis="y", labelsize=6.8)
    for position, count in zip(
        positions, plot_frame["num_recordings"].astype(int)
    ):
        axis.text(
            1.015,
            position,
            f"n={count}",
            transform=axis.get_yaxis_transform(),
            ha="left",
            va="center",
            fontsize=7.3,
            color="#404040",
            clip_on=False,
        )
    axis.text(
        1.015,
        min(headings.values()),
        "Recordings",
        transform=axis.get_yaxis_transform(),
        ha="left",
        va="center",
        fontsize=8,
        fontweight="bold",
        clip_on=False,
    )
    minimum_observed = float(plot_frame["min_intra_corr"].min())
    x_min = 0.0
    if minimum_observed < 0:
        x_min = max(-1.0, minimum_observed - 0.05)
    axis.set_xlim(x_min, 1.0)
    axis.set_ylim(layout_end + 0.1, -0.38)
    axis.set_xlabel("Mean within-state Pearson correlation")
    axis.set_ylabel("")
    _draw_environment_headings(axis, headings)
    style_publication_axis(axis, grid_axis="x")


def plot_intra_state_correlation_summary(
    correlation_frame: pd.DataFrame,
    *,
    output_dir: Path,
    dpi: int,
) -> None:
    """Save the standalone state-level correlation main-text candidate."""

    fig, axis = plt.subplots(figsize=(6.4, 6.65))
    draw_intra_state_correlation(axis, correlation_frame)
    add_panel_label(axis, "(c)", x=-0.43, y=1.01)
    fig.subplots_adjust(left=0.42, right=0.88, bottom=0.09, top=0.99)
    save_figure(
        fig,
        output_dir / "main_panel_c_intra_state_correlation",
        dpi,
    )


def nearest_state_separation_analysis(
    *,
    pair_frame: pd.DataFrame,
    state_distance_frame: pd.DataFrame,
    output_dir: Path,
) -> pd.DataFrame:
    """Find each state's nearest alternative under matched context only."""

    pair_columns = {
        "comparison_type",
        "distance",
        "state_a",
        "state_b",
    }
    state_columns = {
        "state",
        "environment_id",
        "mean_intra_distance",
    }
    missing_pair_columns = pair_columns - set(pair_frame.columns)
    missing_state_columns = state_columns - set(
        state_distance_frame.columns
    )
    if missing_pair_columns or missing_state_columns:
        raise ValueError(
            "Nearest-state analysis is missing columns: "
            f"pair={sorted(missing_pair_columns)}, "
            f"state={sorted(missing_state_columns)}"
        )

    matched_pairs = pair_frame.loc[
        pair_frame["comparison_type"]
        == "inter_state_matched_environment_link"
    ].copy()
    if matched_pairs.empty:
        raise ValueError(
            "No matched environment/link inter-state pairs are available"
        )

    rows: list[dict[str, Any]] = []
    for state_row in ordered_state_frame(
        state_distance_frame
    ).itertuples(index=False):
        state = str(state_row.state)
        relevant = matched_pairs.loc[
            (matched_pairs["state_a"] == state)
            | (matched_pairs["state_b"] == state)
        ].copy()
        if relevant.empty:
            raise ValueError(
                f"{state}: no matched-context alternative state is available"
            )
        relevant["nearest_other_state"] = np.where(
            relevant["state_a"] == state,
            relevant["state_b"],
            relevant["state_a"],
        )
        alternative_means = (
            relevant.groupby("nearest_other_state", sort=True)["distance"]
            .mean()
            .sort_values(kind="stable")
        )
        nearest_other_state = str(alternative_means.index[0])
        mean_nearest_inter_distance = float(alternative_means.iloc[0])
        mean_intra_distance = float(state_row.mean_intra_distance)
        rows.append(
            {
                "state": state,
                "nearest_other_state": nearest_other_state,
                "mean_intra_distance": mean_intra_distance,
                "mean_nearest_inter_distance": (
                    mean_nearest_inter_distance
                ),
                "nearest_state_ratio": (
                    mean_nearest_inter_distance
                    / max(mean_intra_distance, EPSILON)
                ),
                "environment_id": str(state_row.environment_id),
                "environment_label": paper_environment_label(
                    str(state_row.environment_id)
                ),
                "paper_state_label": paper_state_label(state),
                "nearest_other_state_label": paper_state_label(
                    nearest_other_state
                ),
                "comparison_scope": (
                    "same environment_id and link_configuration"
                ),
            }
        )
    nearest_frame = pd.DataFrame(rows)
    nearest_frame.to_csv(
        output_dir / "nearest_state_separation.csv",
        index=False,
        float_format="%.8f",
    )
    return nearest_frame


def draw_nearest_state_separation(
    axis: matplotlib.axes.Axes,
    nearest_frame: pd.DataFrame,
    *,
    state_label_style: str = "paper",
    heading_gap: float = 0.72,
    fixed_value_offset: bool = False,
    value_offset_points: float = 7.0,
    value_offset_y_points: float = 0.0,
    state_label_fontsize: float = 7.0,
    heading_fontsize: float = 8.0,
    ratio_fontsize: float = 7.2,
    marker_area: float = 30.0,
    connector_linewidth: float = 0.9,
    connector_alpha: float = 1.0,
    show_threshold_annotation: bool = False,
    show_class_legend: bool = False,
    legend_fontsize: float = 8.2,
    x_padding: float | None = None,
    bottom_padding: float = 0.0,
    top_padding: float = 0.0,
    ok_color: str = INTRA_COLOR,
    flag_color: str = INTER_COLOR,
    x_label: str = "Nearest-state separation ratio",
    threshold_annotation_y: float = 0.985,
    threshold_annotation_va: str = "top",
) -> None:
    """Draw the nearest-alternative diagnostic as a compact lollipop plot."""

    required_columns = {
        "state",
        "environment_id",
        "nearest_state_ratio",
    }
    missing = required_columns - set(nearest_frame.columns)
    if missing:
        raise ValueError(
            "Nearest-state plot is missing columns: "
            f"{sorted(missing)}"
        )
    if value_offset_points <= 0:
        raise ValueError("Ratio-label offset must be positive")
    plot_frame, positions, headings, layout_end = grouped_state_y_layout(
        nearest_frame,
        heading_gap=heading_gap,
    )
    ratios = plot_frame["nearest_state_ratio"].to_numpy(dtype=float)
    colors = np.where(ratios >= 1.0, ok_color, flag_color)
    for position, ratio, color in zip(positions, ratios, colors):
        axis.plot(
            [1.0, ratio],
            [position, position],
            color=CONNECTOR_COLOR,
            linewidth=connector_linewidth,
            alpha=connector_alpha,
            zorder=1,
        )
        axis.scatter(
            ratio,
            position,
            s=marker_area,
            color=color,
            edgecolor="white",
            linewidth=0.5,
            zorder=3,
        )
    ratio_range = max(float(np.ptp(ratios)), 0.5)
    value_offset = ratio_range * 0.035
    for position, ratio in zip(positions, ratios):
        if fixed_value_offset:
            axis.annotate(
                f"{ratio:.2f}",
                xy=(ratio, position),
                xytext=(
                    value_offset_points,
                    value_offset_y_points,
                ),
                textcoords="offset points",
                ha="left",
                va="center",
                fontsize=ratio_fontsize,
                color=TEXT_COLOR,
                bbox={
                    "facecolor": "white",
                    "edgecolor": "none",
                    "alpha": 0.94,
                    "pad": 0.15,
                },
                zorder=4,
            )
        else:
            axis.text(
                ratio + value_offset,
                position,
                f"{ratio:.2f}",
                ha="left",
                va="center",
                fontsize=ratio_fontsize,
                color=TEXT_COLOR,
            )
    axis.axvline(
        1.0,
        color="#808080",
        linewidth=0.9,
        linestyle=(0, (3, 2)),
        zorder=0,
    )
    if state_label_style == "paper":
        state_labels = [
            textwrap.fill(paper_state_label(state), width=22)
            for state in plot_frame["state"]
        ]
    elif state_label_style == "code":
        state_labels = [
            paper_state_code(state) for state in plot_frame["state"]
        ]
    elif state_label_style == "code_v5_short":
        state_labels = [
            paper_state_v5_code_label(state)
            for state in plot_frame["state"]
        ]
    else:
        raise ValueError(
            f"Unsupported state-label style: {state_label_style!r}"
        )
    axis.set_yticks(positions, state_labels)
    axis.tick_params(axis="y", labelsize=state_label_fontsize)
    if state_label_style in {"code", "code_v5_short"}:
        axis.tick_params(axis="y", pad=6)
        for label in axis.get_yticklabels():
            label.set_horizontalalignment("right")
            label.set_multialignment("right")
    if x_padding is not None:
        x_min = max(
            0.0,
            min(0.65, float(ratios.min()) - 0.15),
        )
        label_padding = x_padding
    else:
        x_min = max(
            0.0,
            min(0.6, float(ratios.min()) - 0.15),
        )
        if fixed_value_offset:
            label_padding = max(0.50, ratio_range * 0.22)
        else:
            label_padding = max(0.24, value_offset * 3.5)
    x_max = float(ratios.max()) + label_padding
    if x_padding is not None:
        x_max = max(1.3, x_max)
    axis.set_xlim(x_min, x_max)
    axis.set_ylim(
        layout_end + 0.1 + bottom_padding,
        -0.38 - top_padding,
    )
    axis.set_xlabel(x_label)
    axis.set_ylabel("")
    _draw_environment_headings(
        axis,
        headings,
        fontsize=heading_fontsize,
    )
    style_publication_axis(axis, grid_axis="x")
    if show_threshold_annotation:
        axis.text(
            1.03,
            threshold_annotation_y,
            "Equal-distance threshold",
            transform=axis.get_xaxis_transform(),
            ha="left",
            va=threshold_annotation_va,
            fontsize=V5_FONT_SIZES["threshold"],
            color="#6F6F6F",
            bbox={
                "facecolor": "white",
                "edgecolor": "none",
                "alpha": 0.92,
                "pad": 0.15,
            },
            zorder=5,
        )
    if show_class_legend:
        handles = [
            Line2D(
                [0],
                [0],
                marker="o",
                linestyle="none",
                markersize=5.5,
                markerfacecolor=ok_color,
                markeredgecolor="white",
                markeredgewidth=0.5,
                label="Separated",
            ),
            Line2D(
                [0],
                [0],
                marker="o",
                linestyle="none",
                markersize=5.5,
                markerfacecolor=flag_color,
                markeredgecolor="white",
                markeredgewidth=0.5,
                label="Potential overlap",
            ),
        ]
        axis.legend(
            handles=handles,
            loc="lower center",
            bbox_to_anchor=(0.5, 0.005),
            ncol=2,
            frameon=False,
            fontsize=legend_fontsize,
            handletextpad=0.35,
            borderaxespad=0.2,
            labelspacing=0.25,
        )


def plot_main_combination_v2(
    *,
    pair_frame: pd.DataFrame,
    state_distance_frame: pd.DataFrame,
    nearest_state_frame: pd.DataFrame,
    output_dir: Path,
    dpi: int,
) -> None:
    """Create the balanced 65/35 main-text layout without a super-title."""

    fig = plt.figure(figsize=(13.0, 8.2))
    grid = fig.add_gridspec(
        2,
        2,
        width_ratios=(2.15, 1.0),
        height_ratios=(1.0, 1.0),
        left=0.20,
        right=0.985,
        bottom=0.08,
        top=0.97,
        hspace=0.38,
        wspace=0.42,
    )
    axis_a = fig.add_subplot(grid[:, 0])
    axis_b = fig.add_subplot(grid[0, 1])
    axis_c = fig.add_subplot(grid[1, 1])
    draw_state_level_distances(
        axis_a, state_distance_frame, show_legend=True
    )
    draw_distance_distribution(
        axis_b, pair_frame, compact_ylabel=True
    )
    draw_nearest_state_separation(axis_c, nearest_state_frame)
    add_panel_label(axis_a, "(a)", x=-0.16, y=1.01)
    add_panel_label(axis_b, "(b)", x=-0.14, y=1.02)
    add_panel_label(axis_c, "(c)", x=-0.14, y=1.02)
    save_figure(
        fig,
        output_dir / "figure_unoccupied_repeatability_main_v2",
        dpi,
    )


def plot_main_combination_v4(
    *,
    pair_frame: pd.DataFrame,
    state_distance_frame: pd.DataFrame,
    nearest_state_frame: pd.DataFrame,
    output_dir: Path,
    dpi: int,
) -> None:
    """Create the code-labelled v4 main figure without raw pair scatter."""

    fig = plt.figure(figsize=(13.0, 8.2))
    grid = fig.add_gridspec(
        2,
        2,
        width_ratios=(2.15, 1.0),
        height_ratios=(1.0, 1.0),
        left=0.15,
        right=0.985,
        bottom=0.08,
        top=0.97,
        hspace=0.38,
        wspace=0.40,
    )
    axis_a = fig.add_subplot(grid[:, 0])
    axis_b = fig.add_subplot(grid[0, 1])
    axis_c = fig.add_subplot(grid[1, 1])
    draw_state_level_distances(
        axis_a,
        state_distance_frame,
        show_legend=True,
        state_label_style="code_short",
        heading_gap=0.65,
    )
    draw_distance_distribution(
        axis_b,
        pair_frame,
        compact_ylabel=True,
        show_raw_scatter=False,
        show_fliers=True,
        box_width=V4_BOXPLOT_WIDTH,
    )
    draw_nearest_state_separation(
        axis_c,
        nearest_state_frame,
        state_label_style="code",
        heading_gap=0.65,
        fixed_value_offset=True,
        value_offset_points=V4_RATIO_LABEL_OFFSET_POINTS,
    )
    add_panel_label(axis_a, "(a)", x=-0.13, y=1.01)
    add_panel_label(axis_b, "(b)", x=-0.14, y=1.02)
    add_panel_label(axis_c, "(c)", x=-0.14, y=1.02)
    save_figure(
        fig,
        output_dir / "figure_unoccupied_repeatability_main_v4",
        dpi,
    )


def plot_main_combination_v5(
    *,
    pair_frame: pd.DataFrame,
    state_distance_frame: pd.DataFrame,
    nearest_state_frame: pd.DataFrame,
    output_dir: Path,
    dpi: int,
) -> None:
    """Create the self-explanatory v5 main figure from unchanged results."""

    # Render close to the intended ~180 mm double-column width so the
    # configured 9–10 pt text remains near its publication size.
    fig = plt.figure(figsize=(6.6, 6.3))
    grid = fig.add_gridspec(
        2,
        2,
        width_ratios=(1.35, 1.0),
        height_ratios=(1.0, 1.0),
        left=0.18,
        right=0.99,
        bottom=0.08,
        top=0.95,
        hspace=0.43,
        wspace=0.42,
    )
    axis_a = fig.add_subplot(grid[:, 0])
    axis_b = fig.add_subplot(grid[0, 1])
    axis_c = fig.add_subplot(grid[1, 1])

    draw_state_level_distances(
        axis_a,
        state_distance_frame,
        show_legend=True,
        state_label_style="code_short",
        heading_gap=0.65,
        show_ratio_column=False,
        legend_labels=(
            "Within-state distance",
            "Between-state distance",
        ),
        x_label="Mean pairwise distance",
        marker_size=5.5,
        errorbar_linewidth=0.9,
        connector_linewidth=0.8,
        connector_alpha=0.65,
        state_label_fontsize=V5_FONT_SIZES["state_a"],
        heading_fontsize=V5_FONT_SIZES["environment"],
        legend_fontsize=V5_FONT_SIZES["legend"],
        legend_ncol=1,
        legend_bottom_padding=1.05,
    )
    draw_distance_distribution(
        axis_b,
        pair_frame,
        show_raw_scatter=False,
        show_fliers=True,
        box_width=V5_BOXPLOT_WIDTH,
        display_labels=("Within state", "Between states"),
        ylabel="Pairwise distance",
        median_linewidth=1.4,
        box_alpha=0.75,
    )
    draw_nearest_state_separation(
        axis_c,
        nearest_state_frame,
        state_label_style="code_v5_short",
        heading_gap=0.65,
        fixed_value_offset=True,
        value_offset_points=V5_RATIO_LABEL_OFFSET_POINTS,
        value_offset_y_points=2.5,
        state_label_fontsize=V5_FONT_SIZES["state_c"],
        heading_fontsize=V5_FONT_SIZES["environment"],
        ratio_fontsize=V5_FONT_SIZES["ratio_value"],
        marker_area=34.0,
        connector_linewidth=0.8,
        connector_alpha=0.65,
        show_threshold_annotation=True,
        show_class_legend=True,
        legend_fontsize=V5_FONT_SIZES["legend"],
        x_padding=0.35,
        bottom_padding=1.05,
        top_padding=1.0,
    )

    axis_a.tick_params(
        axis="x",
        labelsize=V5_FONT_SIZES["tick"],
    )
    axis_b.tick_params(
        axis="both",
        labelsize=V5_FONT_SIZES["tick"],
    )
    axis_c.tick_params(
        axis="x",
        labelsize=V5_FONT_SIZES["tick"],
    )
    for axis in (axis_a, axis_b, axis_c):
        axis.xaxis.label.set_size(V5_FONT_SIZES["axis_label"])
        axis.yaxis.label.set_size(V5_FONT_SIZES["axis_label"])

    add_panel_heading(
        axis_a,
        "(a)",
        V5_PANEL_TITLES["(a)"],
        x=-0.18,
    )
    add_panel_heading(
        axis_b,
        "(b)",
        V5_PANEL_TITLES["(b)"],
        x=-0.15,
    )
    add_panel_heading(
        axis_c,
        "(c)",
        V5_PANEL_TITLES["(c)"],
        x=-0.15,
    )
    save_figure(
        fig,
        output_dir / "figure_unoccupied_repeatability_main_v5",
        dpi,
    )


def plot_main_combination_v6(
    *,
    pair_frame: pd.DataFrame,
    state_distance_frame: pd.DataFrame,
    nearest_state_frame: pd.DataFrame,
    output_dir: Path,
    dpi: int,
) -> None:
    """Create the compact v6 main figure from unchanged result tables."""

    fig = plt.figure(figsize=(7.1, 6.65))
    grid = fig.add_gridspec(
        2,
        2,
        width_ratios=(2.15, 1.0),
        height_ratios=(1.0, 1.0),
        left=0.26,
        right=0.99,
        bottom=0.09,
        top=0.97,
        hspace=0.36,
        wspace=0.36,
    )
    axis_a = fig.add_subplot(grid[:, 0])
    axis_b = fig.add_subplot(grid[0, 1])
    axis_c = fig.add_subplot(grid[1, 1])

    draw_state_level_distances(
        axis_a,
        state_distance_frame,
        show_legend=True,
        state_label_style="code_short",
        heading_gap=0.65,
        show_ratio_column=False,
        legend_labels=(
            "Within-state distance",
            "Between-state distance",
        ),
        x_label="Mean pairwise distance",
        marker_size=4.8,
        errorbar_linewidth=0.8,
        connector_linewidth=0.75,
        connector_alpha=0.65,
        state_label_fontsize=V6_FONT_SIZES["state_a"],
        heading_fontsize=V6_FONT_SIZES["environment_a"],
        legend_fontsize=V6_FONT_SIZES["legend"],
        legend_ncol=2,
        legend_bottom_padding=0.58,
    )
    draw_distance_distribution(
        axis_b,
        pair_frame,
        show_raw_scatter=False,
        show_fliers=True,
        box_width=V6_BOXPLOT_WIDTH,
        display_labels=("Within state", "Between states"),
        ylabel="Pairwise distance",
        median_linewidth=1.3,
        box_alpha=0.75,
    )
    axis_b.set_xlim(0.72, 2.28)
    draw_nearest_state_separation(
        axis_c,
        nearest_state_frame,
        state_label_style="code",
        heading_gap=0.65,
        fixed_value_offset=True,
        value_offset_points=V6_RATIO_LABEL_OFFSET_POINTS,
        value_offset_y_points=2.0,
        state_label_fontsize=V6_FONT_SIZES["state_c"],
        heading_fontsize=V6_FONT_SIZES["environment_c"],
        ratio_fontsize=V6_FONT_SIZES["ratio_value"],
        marker_area=28.0,
        connector_linewidth=0.75,
        connector_alpha=0.65,
        show_threshold_annotation=False,
        show_class_legend=False,
        x_padding=0.38,
        bottom_padding=0.05,
        top_padding=0.15,
    )

    axis_a.tick_params(
        axis="x",
        labelsize=V6_FONT_SIZES["tick"],
    )
    axis_b.tick_params(
        axis="both",
        labelsize=V6_FONT_SIZES["tick"],
    )
    axis_c.tick_params(
        axis="x",
        labelsize=V6_FONT_SIZES["tick"],
    )
    for axis in (axis_a, axis_b, axis_c):
        axis.xaxis.label.set_size(V6_FONT_SIZES["axis_label"])
        axis.yaxis.label.set_size(V6_FONT_SIZES["axis_label"])
    axis_c.xaxis.set_label_coords(0.44, -0.105)

    add_panel_label(axis_a, "(a)", x=-0.20, y=1.015)
    add_panel_label(axis_b, "(b)", x=-0.19, y=1.02)
    add_panel_label(axis_c, "(c)", x=-0.19, y=1.02)
    save_figure(
        fig,
        output_dir / "figure_unoccupied_repeatability_main_v6",
        dpi,
    )


def plot_main_combination_v7(
    *,
    pair_frame: pd.DataFrame,
    state_distance_frame: pd.DataFrame,
    nearest_state_frame: pd.DataFrame,
    output_dir: Path,
    dpi: int,
) -> None:
    """Create the readability-focused v7 main figure (minimal-change plan).

    Fixes versus v6: panel (c) uses self-contained code+name labels and a
    neutral/red threshold palette so its colors no longer collide with the
    within/between semantics of panels (a)-(b); the equal-distance threshold
    is annotated; panel (a) de-emphasizes the descriptive SD error bars and
    emphasizes the within-to-between mean gap; panel (b) annotates pair
    counts; axis labels state the standardized units.
    """

    fig = plt.figure(figsize=(7.1, 6.65))
    grid = fig.add_gridspec(
        2,
        2,
        width_ratios=(1.9, 1.0),
        height_ratios=(1.0, 1.0),
        left=0.26,
        right=0.99,
        bottom=0.12,
        top=0.97,
        hspace=0.40,
        wspace=0.50,
    )
    axis_a = fig.add_subplot(grid[:, 0])
    axis_b = fig.add_subplot(grid[0, 1])
    axis_c = fig.add_subplot(grid[1, 1])

    draw_state_level_distances(
        axis_a,
        state_distance_frame,
        show_legend=False,
        state_label_style="code_short",
        heading_gap=0.65,
        show_ratio_column=False,
        x_label="Mean pairwise distance (standardized units)",
        marker_size=4.8,
        errorbar_linewidth=0.7,
        errorbar_alpha=0.45,
        connector_linewidth=1.5,
        connector_alpha=0.95,
        connector_color=EMPHASIZED_CONNECTOR_COLOR,
        state_label_fontsize=V6_FONT_SIZES["state_a"],
        heading_fontsize=V6_FONT_SIZES["environment_a"],
        legend_bottom_padding=0.15,
    )
    draw_distance_distribution(
        axis_b,
        pair_frame,
        show_raw_scatter=False,
        show_fliers=True,
        box_width=V6_BOXPLOT_WIDTH,
        display_labels=("Within state", "Between states"),
        ylabel="Pairwise distance",
        median_linewidth=1.3,
        box_alpha=0.75,
        show_counts=True,
    )
    axis_b.set_xlim(0.72, 2.28)
    draw_nearest_state_separation(
        axis_c,
        nearest_state_frame,
        state_label_style="code_v5_short",
        heading_gap=0.65,
        fixed_value_offset=True,
        value_offset_points=V6_RATIO_LABEL_OFFSET_POINTS,
        value_offset_y_points=2.0,
        state_label_fontsize=V6_FONT_SIZES["state_c"],
        heading_fontsize=V6_FONT_SIZES["environment_c"],
        ratio_fontsize=V6_FONT_SIZES["ratio_value"],
        marker_area=28.0,
        connector_linewidth=0.75,
        connector_alpha=0.65,
        show_threshold_annotation=True,
        show_class_legend=False,
        x_padding=0.38,
        bottom_padding=1.0,
        top_padding=0.15,
        ok_color=RATIO_OK_COLOR,
        flag_color=RATIO_FLAG_COLOR,
        x_label="Separation ratio\n(nearest ÷ within-state)",
        threshold_annotation_y=0.02,
        threshold_annotation_va="bottom",
    )

    axis_a.tick_params(
        axis="x",
        labelsize=V6_FONT_SIZES["tick"],
    )
    axis_b.tick_params(
        axis="both",
        labelsize=V6_FONT_SIZES["tick"],
    )
    axis_c.tick_params(
        axis="x",
        labelsize=V6_FONT_SIZES["tick"],
    )
    for axis in (axis_a, axis_b, axis_c):
        axis.xaxis.label.set_size(V6_FONT_SIZES["axis_label"])
        axis.yaxis.label.set_size(V6_FONT_SIZES["axis_label"])

    legend_handles = [
        Line2D(
            [0],
            [0],
            marker="o",
            linestyle="none",
            markersize=4.8,
            markerfacecolor=INTRA_COLOR,
            markeredgecolor="white",
            markeredgewidth=0.45,
            label="Within-state distance",
        ),
        Line2D(
            [0],
            [0],
            marker="s",
            linestyle="none",
            markersize=4.8,
            markerfacecolor=INTER_COLOR,
            markeredgecolor="white",
            markeredgewidth=0.45,
            label="Between-state distance",
        ),
    ]
    fig.legend(
        handles=legend_handles,
        loc="lower left",
        bbox_to_anchor=(0.27, 0.004),
        ncol=2,
        frameon=False,
        handlelength=1.4,
        columnspacing=1.2,
        fontsize=V6_FONT_SIZES["legend"],
    )

    add_panel_label(axis_a, "(a)", x=-0.20, y=1.015)
    add_panel_label(axis_b, "(b)", x=-0.19, y=1.02)
    add_panel_label(axis_c, "(c)", x=-0.19, y=1.02)
    save_figure(
        fig,
        output_dir / "figure_unoccupied_repeatability_main_v7",
        dpi,
    )


def plot_main_combination_v8(
    *,
    pair_frame: pd.DataFrame,
    state_distance_frame: pd.DataFrame,
    nearest_state_frame: pd.DataFrame,
    output_dir: Path,
    dpi: int,
) -> None:
    """Create the structural v8 main figure (self-contained panel plan).

    Panel (c) of v6/v7 is folded into panel (a) as an annotated
    nearest-state separation ratio column, so every state row carries its
    distances, its short name, and its separation verdict in one place. The
    pooled distribution panel is retained at full height on the right. The
    standalone boxplot for a supplementary remains available via
    plot_pair_level_distance_boxplot.
    """

    fig = plt.figure(figsize=(7.1, 6.65))
    grid = fig.add_gridspec(
        1,
        2,
        width_ratios=(2.5, 1.0),
        left=0.235,
        right=0.985,
        bottom=0.10,
        top=0.92,
        wspace=0.30,
    )
    axis_a = fig.add_subplot(grid[0, 0])
    axis_b = fig.add_subplot(grid[0, 1])

    draw_state_level_distances(
        axis_a,
        state_distance_frame,
        show_legend=True,
        state_label_style="code_short",
        heading_gap=0.65,
        show_ratio_column=False,
        nearest_state_frame=nearest_state_frame,
        legend_labels=(
            "Within-state distance",
            "Between-state distance",
        ),
        x_label="Mean pairwise distance (standardized units)",
        marker_size=4.8,
        errorbar_linewidth=0.7,
        errorbar_alpha=0.45,
        connector_linewidth=1.5,
        connector_alpha=0.95,
        connector_color=EMPHASIZED_CONNECTOR_COLOR,
        state_label_fontsize=V6_FONT_SIZES["state_a"],
        heading_fontsize=V6_FONT_SIZES["environment_a"],
        legend_fontsize=V6_FONT_SIZES["legend"],
        legend_ncol=2,
        legend_bottom_padding=0.58,
    )
    draw_distance_distribution(
        axis_b,
        pair_frame,
        show_raw_scatter=False,
        show_fliers=True,
        box_width=V6_BOXPLOT_WIDTH,
        display_labels=("Within\nstate", "Between\nstates"),
        ylabel="Pairwise distance (standardized units)",
        median_linewidth=1.3,
        box_alpha=0.75,
        show_counts=True,
    )
    axis_b.set_xlim(0.72, 2.28)

    axis_a.tick_params(
        axis="x",
        labelsize=V6_FONT_SIZES["tick"],
    )
    axis_b.tick_params(
        axis="both",
        labelsize=V6_FONT_SIZES["tick"],
    )
    for axis in (axis_a, axis_b):
        axis.xaxis.label.set_size(V6_FONT_SIZES["axis_label"])
        axis.yaxis.label.set_size(V6_FONT_SIZES["axis_label"])

    add_panel_heading(
        axis_a,
        "(a)",
        "Per-state distances and nearest-state separation",
        x=-0.30,
        y=1.015,
        label_fontsize=10.5,
        title_fontsize=9.3,
    )
    add_panel_heading(
        axis_b,
        "(b)",
        "Pooled distributions",
        x=-0.42,
        y=1.015,
        label_fontsize=10.5,
        title_fontsize=9.3,
    )
    save_figure(
        fig,
        output_dir / "figure_unoccupied_repeatability_main_v8",
        dpi,
    )


def _figure7_distance_arrays(
    pair_frame: pd.DataFrame,
) -> tuple[np.ndarray, np.ndarray]:
    """Return the unchanged pooled within/between distance arrays."""

    plot_frame = _distance_plot_frame(pair_frame)
    within = plot_frame.loc[
        plot_frame["distance_class"] == "Intra-state",
        "distance",
    ].to_numpy(dtype=float)
    between = plot_frame.loc[
        plot_frame["distance_class"] == "Inter-state",
        "distance",
    ].to_numpy(dtype=float)
    if (len(within), len(between)) != (126, 585):
        raise ValueError(
            "Figure 7 requires the frozen pair counts n=126 and n=585; "
            f"found n={len(within)} and n={len(between)}"
        )
    return within, between


def draw_figure7_condition_panel(
    axis: matplotlib.axes.Axes,
    ratio_axis: matplotlib.axes.Axes,
    state_frame: pd.DataFrame,
    nearest_frame: pd.DataFrame,
) -> None:
    """Draw the quiet dumbbell panel and aligned ratio column."""

    required_state_columns = {
        "state",
        "environment_id",
        "mean_intra_distance",
        "std_intra_distance",
        "mean_inter_distance",
        "std_inter_distance",
    }
    missing_state_columns = required_state_columns - set(state_frame.columns)
    if missing_state_columns:
        raise ValueError(
            "Figure 7 state statistics are missing columns: "
            f"{sorted(missing_state_columns)}"
        )
    required_ratio_columns = {
        "state",
        "mean_nearest_inter_distance",
        "nearest_state_ratio",
    }
    missing_ratio_columns = required_ratio_columns - set(nearest_frame.columns)
    if missing_ratio_columns:
        raise ValueError(
            "Figure 7 nearest-condition results are missing columns: "
            f"{sorted(missing_ratio_columns)}"
        )

    plot_frame, positions, headings, _layout_end = grouped_state_y_layout(
        state_frame,
        heading_gap=0.64,
        group_gap=0.34,
    )
    label_column_x = -FIGURE7_LABEL_COLUMN_IN / FIGURE7_PLOT_WIDTH_IN
    item_label_x = label_column_x + 0.04 / FIGURE7_PLOT_WIDTH_IN
    if len(plot_frame) != 13:
        raise ValueError(
            f"Figure 7 requires 13 environment-condition rows, found {len(plot_frame)}"
        )
    intra = plot_frame["mean_intra_distance"].to_numpy(dtype=float)
    intra_sd = plot_frame["std_intra_distance"].to_numpy(dtype=float)
    inter = plot_frame["mean_inter_distance"].to_numpy(dtype=float)
    inter_sd = plot_frame["std_inter_distance"].to_numpy(dtype=float)
    ratio_lookup = dict(
        zip(
            nearest_frame["state"].astype(str),
            nearest_frame["nearest_state_ratio"].astype(float),
        )
    )
    nearest_distance_lookup = dict(
        zip(
            nearest_frame["state"].astype(str),
            nearest_frame["mean_nearest_inter_distance"].astype(float),
        )
    )
    states = plot_frame["state"].astype(str).tolist()
    missing_states = [
        state
        for state in states
        if state not in ratio_lookup or state not in nearest_distance_lookup
    ]
    if missing_states:
        raise ValueError(
            "Figure 7 has incomplete nearest-condition data for "
            f"{missing_states}"
        )
    nearest_distances = np.asarray(
        [nearest_distance_lookup[state] for state in states],
        dtype=float,
    )
    if not np.isfinite(
        np.concatenate(
            (intra, intra_sd, inter, inter_sd, nearest_distances)
        )
    ).all():
        raise ValueError("Figure 7 distance statistics must all be finite")

    bathroom_positions = positions[
        plot_frame["environment_id"].to_numpy() == "E3"
    ]
    bathroom_y_min = float(bathroom_positions[0]) - 0.50
    bathroom_y_max = float(bathroom_positions[-1]) + 0.50
    axis.add_patch(
        Rectangle(
            (label_column_x, bathroom_y_min),
            1.0 - label_column_x,
            bathroom_y_max - bathroom_y_min,
            transform=axis.get_yaxis_transform(),
            facecolor="#F5F5F5",
            edgecolor="none",
            linewidth=0,
            clip_on=False,
            zorder=-2,
        )
    )

    axis.errorbar(
        intra,
        positions,
        xerr=intra_sd,
        fmt="none",
        ecolor=INTRA_COLOR,
        elinewidth=1.0,
        capsize=0,
        zorder=2,
    )
    axis.errorbar(
        inter,
        positions,
        xerr=inter_sd,
        fmt="none",
        ecolor=INTER_COLOR,
        elinewidth=1.0,
        capsize=0,
        zorder=2,
    )
    axis.vlines(
        nearest_distances,
        positions - 0.275,
        positions + 0.275,
        color="#4A4A4A",
        linewidth=1.2,
        zorder=2.5,
    )
    marker_area = 4.8**2
    axis.scatter(
        intra,
        positions,
        s=marker_area,
        marker="o",
        facecolor=INTRA_COLOR,
        edgecolor=INTRA_COLOR,
        linewidth=0.7,
        zorder=3,
    )
    axis.scatter(
        inter,
        positions,
        s=marker_area,
        marker="o",
        facecolor="white",
        edgecolor=INTER_COLOR,
        linewidth=1.0,
        zorder=3,
    )

    axis.set_yticks(positions)
    axis.tick_params(axis="y", left=False, labelleft=False)
    for position, state in zip(
        positions,
        plot_frame["state"].astype(str),
    ):
        axis.text(
            item_label_x,
            position,
            paper_state_code_label(state),
            transform=axis.get_yaxis_transform(),
            ha="left",
            va="center",
            fontsize=7.0,
            color="#303030",
            clip_on=False,
        )

    for environment_id, heading_position in headings.items():
        axis.text(
            label_column_x,
            heading_position,
            paper_environment_label(environment_id),
            transform=axis.get_yaxis_transform(),
            ha="left",
            va="center",
            fontsize=7.2,
            color="#666666",
            clip_on=False,
        )
        separator_y = heading_position + 0.31
        axis.plot(
            [label_column_x, 1.0],
            [separator_y, separator_y],
            transform=axis.get_yaxis_transform(),
            color="#E2E2E2",
            linewidth=0.45,
            clip_on=False,
            zorder=0.5,
        )

    axis.set_xlim(*FIGURE7_X_LIMITS)
    axis.set_ylim(float(positions[-1]) + 0.45, -0.35)
    axis.set_ylabel("")
    axis.set_xlabel("")
    style_publication_axis(axis, grid_axis="x")
    axis.spines["bottom"].set_visible(False)
    axis.tick_params(axis="x", bottom=False, labelbottom=False)
    axis.text(
        label_column_x,
        1.018,
        "a  Per-condition distances",
        transform=axis.transAxes,
        ha="left",
        va="bottom",
        fontsize=8.2,
        fontweight="bold",
        color=TEXT_COLOR,
        clip_on=False,
    )

    legend_handles = [
        Line2D(
            [0],
            [0],
            marker="o",
            linestyle="none",
            markersize=4.8,
            markerfacecolor=INTRA_COLOR,
            markeredgecolor=INTRA_COLOR,
            markeredgewidth=0.7,
            label="Within-condition",
        ),
        Line2D(
            [0],
            [0],
            marker="o",
            linestyle="none",
            markersize=4.8,
            markerfacecolor="white",
            markeredgecolor=INTER_COLOR,
            markeredgewidth=1.0,
            label="Between-condition",
        ),
        Line2D(
            [0],
            [0],
            marker="|",
            linestyle="none",
            markersize=8.0,
            markeredgewidth=1.2,
            color="#4A4A4A",
            label="Nearest condition",
        ),
    ]
    axis.legend(
        handles=legend_handles,
        loc="lower left",
        bbox_to_anchor=(0.70, 0.025),
        ncol=1,
        frameon=False,
        handlelength=1.0,
        handletextpad=0.5,
        fontsize=7.0,
    )

    ratio_axis.set_xlim(0.0, 1.0)
    ratio_axis.set_ylim(axis.get_ylim())
    ratio_axis.patch.set_visible(False)
    ratio_axis.tick_params(
        axis="both",
        left=False,
        bottom=False,
        labelleft=False,
        labelbottom=False,
    )
    for spine in ratio_axis.spines.values():
        spine.set_visible(False)
    ratio_axis.axhspan(
        bathroom_y_min,
        bathroom_y_max,
        facecolor="#F5F5F5",
        edgecolor="none",
        linewidth=0,
        zorder=-2,
    )
    for position, state in zip(
        positions,
        plot_frame["state"].astype(str),
    ):
        if state not in ratio_lookup:
            raise ValueError(
                f"Figure 7 has no nearest-condition ratio for {state!r}"
            )
        ratio = ratio_lookup[state]
        ratio_axis.text(
            0.35,
            position,
            f"{ratio:.2f}",
            ha="right",
            va="center",
            fontsize=7.2,
            color=(RATIO_FLAG_COLOR if ratio < 1.0 else "#4D4D4D"),
            fontweight="bold" if ratio < 1.0 else "normal",
        )
    ratio_axis.text(
        0.50,
        -0.18,
        "Nearest-condition\nratio",
        ha="center",
        va="center",
        fontsize=6.8,
        color="#666666",
    )


def draw_figure7_distribution_band(
    axis: matplotlib.axes.Axes,
    pair_frame: pd.DataFrame,
) -> None:
    """Draw the pooled distributions horizontally on the shared x-axis."""

    within, between = _figure7_distance_arrays(pair_frame)
    label_column_x = -FIGURE7_LABEL_COLUMN_IN / FIGURE7_PLOT_WIDTH_IN
    arrays = (within, between)
    positions = (1.0, 0.0)
    boxplot = axis.boxplot(
        arrays,
        positions=positions,
        widths=0.62,
        vert=False,
        patch_artist=True,
        showfliers=False,
        whis=1.5,
        medianprops={"linewidth": 1.05},
        whiskerprops={"linewidth": 0.9},
        capprops={"linewidth": 0.9},
        boxprops={"linewidth": 0.85},
        zorder=2,
    )
    saturated_colors = (INTRA_COLOR, INTER_COLOR)
    light_colors = (INTRA_LIGHT_COLOR, INTER_LIGHT_COLOR)
    for index, (patch, edge_color, fill_color) in enumerate(
        zip(boxplot["boxes"], saturated_colors, light_colors)
    ):
        patch.set_facecolor(fill_color)
        patch.set_edgecolor(edge_color)
        for line in boxplot["whiskers"][2 * index : 2 * index + 2]:
            line.set_color(edge_color)
        for line in boxplot["caps"][2 * index : 2 * index + 2]:
            line.set_color(edge_color)
        boxplot["medians"][index].set_color(edge_color)
    axis.set_yticks(
        positions,
        (f"Within (n = {len(within)})", f"Between (n = {len(between)})"),
    )
    axis.set_ylim(-0.535, 1.535)
    axis.set_xlim(*FIGURE7_X_LIMITS)
    axis.set_xticks(np.arange(0.0, 41.0, 10.0))
    axis.set_xlabel("Pairwise distance (standardized units)", fontsize=8.0)
    axis.set_ylabel("")
    axis.tick_params(axis="x", labelsize=7.2, pad=2.5)
    axis.tick_params(axis="y", labelsize=7.0, pad=5.0)
    style_publication_axis(axis, grid_axis="x")
    axis.text(
        label_column_x,
        1.08,
        "b  Pooled",
        transform=axis.transAxes,
        ha="left",
        va="bottom",
        fontsize=8.2,
        fontweight="bold",
        color=TEXT_COLOR,
        clip_on=False,
    )


def save_figure_triplet(
    fig: matplotlib.figure.Figure,
    stem: Path,
    dpi: int,
) -> None:
    """Save fixed-canvas PNG, SVG, and PDF publication outputs."""

    width, _ = fig.get_size_inches()
    if not math.isclose(float(width), FIGURE7_WIDTH_IN, abs_tol=1e-9):
        raise ValueError(
            f"Figure 7 width must be {FIGURE7_WIDTH_IN:.1f} in, found {width:g}"
        )
    raster_dpi = max(int(dpi), FIGURE_DPI)
    with matplotlib.rc_context({"savefig.bbox": None}):
        fig.savefig(
            stem.with_suffix(".png"),
            dpi=raster_dpi,
            facecolor="white",
            bbox_inches=None,
        )
        fig.savefig(
            stem.with_suffix(".svg"),
            facecolor="white",
            bbox_inches=None,
        )
        fig.savefig(
            stem.with_suffix(".pdf"),
            facecolor="white",
            bbox_inches=None,
        )
    plt.close(fig)


def _plot_figure7_condition_distance_r3(
    *,
    pair_frame: pd.DataFrame,
    state_distance_frame: pd.DataFrame,
    nearest_frame: pd.DataFrame,
    output_dir: Path,
    dpi: int,
) -> None:
    """Create the retained r3 Figure 7 with a shared distance axis."""

    fig = plt.figure(figsize=(FIGURE7_WIDTH_IN, FIGURE7_HEIGHT_IN))
    axis_left = (
        FIGURE7_CONTENT_LEFT_IN + FIGURE7_LABEL_COLUMN_IN
    ) / FIGURE7_WIDTH_IN
    axis_width = FIGURE7_PLOT_WIDTH_IN / FIGURE7_WIDTH_IN
    ratio_left = (
        FIGURE7_CONTENT_LEFT_IN
        + FIGURE7_LABEL_COLUMN_IN
        + FIGURE7_PLOT_WIDTH_IN
        + FIGURE7_PLOT_RATIO_GAP_IN
    ) / FIGURE7_WIDTH_IN
    ratio_width = FIGURE7_RATIO_WIDTH_IN / FIGURE7_WIDTH_IN
    axis_a = fig.add_axes(
        [
            axis_left,
            1.30 / FIGURE7_HEIGHT_IN,
            axis_width,
            FIGURE7_PANEL_A_HEIGHT_IN / FIGURE7_HEIGHT_IN,
        ]
    )
    axis_b = fig.add_axes(
        [
            axis_left,
            0.43 / FIGURE7_HEIGHT_IN,
            axis_width,
            FIGURE7_PANEL_B_HEIGHT_IN / FIGURE7_HEIGHT_IN,
        ],
        sharex=axis_a,
    )
    ratio_axis_a = fig.add_axes(
        [
            ratio_left,
            1.30 / FIGURE7_HEIGHT_IN,
            ratio_width,
            FIGURE7_PANEL_A_HEIGHT_IN / FIGURE7_HEIGHT_IN,
        ],
        sharey=axis_a,
    )
    ratio_axis_b = fig.add_axes(
        [
            ratio_left,
            0.43 / FIGURE7_HEIGHT_IN,
            ratio_width,
            FIGURE7_PANEL_B_HEIGHT_IN / FIGURE7_HEIGHT_IN,
        ]
    )

    draw_figure7_condition_panel(
        axis_a,
        ratio_axis_a,
        state_distance_frame,
        nearest_frame,
    )
    draw_figure7_distribution_band(axis_b, pair_frame)

    ratio_axis_b.set_axis_off()
    separator_x = ratio_axis_a.get_position().x0
    fig.add_artist(
        Line2D(
            [separator_x, separator_x],
            [ratio_axis_b.get_position().y0, ratio_axis_a.get_position().y1],
            transform=fig.transFigure,
            color="#D9D9D9",
            linewidth=0.5,
            zorder=5,
        )
    )
    save_figure_triplet(fig, output_dir / FIGURE7_STEM, dpi)


def _figure7_variant_frame(
    state_distance_frame: pd.DataFrame,
    nearest_frame: pd.DataFrame,
) -> pd.DataFrame:
    """Return the 13 frozen rows with nearest-condition columns attached."""

    required_distance_columns = {
        "state",
        "environment_id",
        "mean_intra_distance",
        "std_intra_distance",
        "mean_inter_distance",
        "std_inter_distance",
    }
    required_nearest_columns = {
        "state",
        "mean_nearest_inter_distance",
        "nearest_state_ratio",
    }
    missing_distance = required_distance_columns - set(
        state_distance_frame.columns
    )
    missing_nearest = required_nearest_columns - set(nearest_frame.columns)
    if missing_distance or missing_nearest:
        raise ValueError(
            "Figure 7 variant inputs are incomplete: "
            f"distance={sorted(missing_distance)}, "
            f"nearest={sorted(missing_nearest)}"
        )
    ordered = ordered_state_frame(state_distance_frame)
    ordered_nearest = ordered_state_frame(nearest_frame)
    states = ordered["state"].astype(str).tolist()
    nearest_states = ordered_nearest["state"].astype(str).tolist()
    if len(ordered) != 13 or states != nearest_states:
        raise ValueError(
            "Figure 7 variants require the same 13 ordered condition rows "
            "in both frozen summary files"
        )
    result = ordered.copy()
    result["mean_nearest_inter_distance"] = ordered_nearest[
        "mean_nearest_inter_distance"
    ].to_numpy(dtype=float)
    result["nearest_state_ratio"] = ordered_nearest[
        "nearest_state_ratio"
    ].to_numpy(dtype=float)
    numeric_columns = (
        "mean_intra_distance",
        "std_intra_distance",
        "mean_nearest_inter_distance",
        "mean_inter_distance",
        "std_inter_distance",
        "nearest_state_ratio",
    )
    if not np.isfinite(result.loc[:, numeric_columns].to_numpy()).all():
        raise ValueError("Figure 7 variant statistics must all be finite")
    return result


def _figure7_grouped_positions(
    frame: pd.DataFrame,
) -> tuple[np.ndarray, dict[str, float]]:
    """Return compact row and heading positions for an already ordered frame."""

    layout_frame, positions, headings, _ = grouped_state_y_layout(
        frame,
        heading_gap=0.64,
        group_gap=0.32,
    )
    if layout_frame["state"].astype(str).tolist() != frame[
        "state"
    ].astype(str).tolist():
        raise ValueError("Figure 7 condition order changed during layout")
    return positions, headings


def _figure7_bathroom_span(
    frame: pd.DataFrame,
    positions: np.ndarray,
) -> tuple[float, float]:
    bathroom_positions = positions[
        frame["environment_id"].to_numpy() == "E3"
    ]
    return (
        float(bathroom_positions[0]) - 0.50,
        float(bathroom_positions[-1]) + 0.50,
    )


def _draw_figure7_summary_marks(
    axis: matplotlib.axes.Axes,
    frame: pd.DataFrame,
    positions: np.ndarray,
) -> None:
    """Draw all three frozen condition summaries on one distance axis."""

    within = frame["mean_intra_distance"].to_numpy(dtype=float)
    within_sd = frame["std_intra_distance"].to_numpy(dtype=float)
    nearest = frame["mean_nearest_inter_distance"].to_numpy(dtype=float)
    between = frame["mean_inter_distance"].to_numpy(dtype=float)
    between_sd = frame["std_inter_distance"].to_numpy(dtype=float)
    axis.errorbar(
        within,
        positions,
        xerr=within_sd,
        fmt="none",
        ecolor=INTRA_COLOR,
        elinewidth=0.95,
        capsize=0,
        zorder=2,
    )
    axis.errorbar(
        between,
        positions,
        xerr=between_sd,
        fmt="none",
        ecolor=INTER_COLOR,
        elinewidth=0.95,
        capsize=0,
        zorder=2,
    )
    axis.vlines(
        nearest,
        positions - 0.26,
        positions + 0.26,
        color="#4A4A4A",
        linewidth=1.2,
        zorder=2.5,
    )
    axis.scatter(
        within,
        positions,
        s=4.7**2,
        marker="o",
        facecolor=INTRA_COLOR,
        edgecolor=INTRA_COLOR,
        linewidth=0.7,
        zorder=3,
    )
    axis.scatter(
        between,
        positions,
        s=4.7**2,
        marker="o",
        facecolor="white",
        edgecolor=INTER_COLOR,
        linewidth=1.0,
        zorder=3,
    )


def _style_figure7_boxplot(
    axis: matplotlib.axes.Axes,
    within: np.ndarray,
    between: np.ndarray,
) -> None:
    boxplot = axis.boxplot(
        (within, between),
        positions=(1.0, 0.0),
        widths=0.58,
        vert=False,
        patch_artist=True,
        showfliers=False,
        whis=1.5,
        medianprops={"linewidth": 1.0},
        whiskerprops={"linewidth": 0.85},
        capprops={"linewidth": 0.85},
        boxprops={"linewidth": 0.85},
        zorder=2,
    )
    for index, (edge_color, fill_color) in enumerate(
        (
            (INTRA_COLOR, INTRA_LIGHT_COLOR),
            (INTER_COLOR, INTER_LIGHT_COLOR),
        )
    ):
        boxplot["boxes"][index].set_facecolor(fill_color)
        boxplot["boxes"][index].set_edgecolor(edge_color)
        boxplot["medians"][index].set_color(edge_color)
        for line in boxplot["whiskers"][2 * index : 2 * index + 2]:
            line.set_color(edge_color)
        for line in boxplot["caps"][2 * index : 2 * index + 2]:
            line.set_color(edge_color)


def _plot_figure7_condition_distance_variant_a(
    *,
    pair_frame: pd.DataFrame,
    state_distance_frame: pd.DataFrame,
    nearest_frame: pd.DataFrame,
    output_dir: Path,
    dpi: int,
) -> None:
    """Direction A: broken-axis interval forest with aligned distributions."""

    frame = _figure7_variant_frame(state_distance_frame, nearest_frame)
    positions, headings = _figure7_grouped_positions(frame)
    bath_low, bath_high = _figure7_bathroom_span(frame, positions)
    within, between = _figure7_distance_arrays(pair_frame)
    figure_height = 5.25
    fig = plt.figure(figsize=(FIGURE7_WIDTH_IN, figure_height))
    label_width = 1.48
    left_edge = 0.18 + label_width
    left_width = 3.55
    break_gap = 0.12
    right_width = 1.05
    ratio_gap = 0.08
    ratio_width = 0.65
    panel_a_bottom = 1.23
    panel_a_height = 3.62
    panel_b_bottom = 0.42
    panel_b_height = 0.54
    left_a = fig.add_axes(
        [
            left_edge / FIGURE7_WIDTH_IN,
            panel_a_bottom / figure_height,
            left_width / FIGURE7_WIDTH_IN,
            panel_a_height / figure_height,
        ]
    )
    right_a = fig.add_axes(
        [
            (left_edge + left_width + break_gap) / FIGURE7_WIDTH_IN,
            panel_a_bottom / figure_height,
            right_width / FIGURE7_WIDTH_IN,
            panel_a_height / figure_height,
        ],
        sharey=left_a,
    )
    left_b = fig.add_axes(
        [
            left_edge / FIGURE7_WIDTH_IN,
            panel_b_bottom / figure_height,
            left_width / FIGURE7_WIDTH_IN,
            panel_b_height / figure_height,
        ]
    )
    right_b = fig.add_axes(
        [
            (left_edge + left_width + break_gap) / FIGURE7_WIDTH_IN,
            panel_b_bottom / figure_height,
            right_width / FIGURE7_WIDTH_IN,
            panel_b_height / figure_height,
        ],
        sharey=left_b,
    )
    ratio_left = (
        left_edge + left_width + break_gap + right_width + ratio_gap
    )
    ratio_axis = fig.add_axes(
        [
            ratio_left / FIGURE7_WIDTH_IN,
            panel_a_bottom / figure_height,
            ratio_width / FIGURE7_WIDTH_IN,
            panel_a_height / figure_height,
        ],
        sharey=left_a,
    )

    for axis in (left_a, right_a):
        axis.axhspan(
            bath_low,
            bath_high,
            facecolor="#F5F5F5",
            edgecolor="none",
            zorder=-2,
        )
        _draw_figure7_summary_marks(axis, frame, positions)
        axis.set_ylim(float(positions[-1]) + 0.45, -0.35)
        style_publication_axis(axis, grid_axis="x")
        axis.spines["bottom"].set_visible(False)
        axis.tick_params(axis="x", bottom=False, labelbottom=False)
        axis.tick_params(axis="y", left=False, labelleft=False)
    left_a.set_xlim(0.0, 32.0)
    right_a.set_xlim(40.0, 55.0)
    left_a.spines["right"].set_visible(False)
    right_a.spines["left"].set_visible(False)

    label_x = -label_width / left_width
    item_x = label_x + 0.05 / left_width
    for position, state in zip(positions, frame["state"].astype(str)):
        left_a.text(
            item_x,
            position,
            paper_state_code_label(state),
            transform=left_a.get_yaxis_transform(),
            ha="left",
            va="center",
            fontsize=6.9,
            color="#303030",
            clip_on=False,
        )
    for environment_id, heading_y in headings.items():
        left_a.text(
            label_x,
            heading_y,
            paper_environment_label(environment_id),
            transform=left_a.get_yaxis_transform(),
            ha="left",
            va="center",
            fontsize=7.1,
            color="#666666",
            clip_on=False,
        )
        for axis in (left_a, right_a):
            axis.axhline(
                heading_y + 0.31,
                color="#E2E2E2",
                linewidth=0.45,
                zorder=0.5,
            )
    left_a.text(
        label_x,
        1.018,
        "a  Condition summaries on a broken distance axis",
        transform=left_a.transAxes,
        ha="left",
        va="bottom",
        fontsize=8.1,
        fontweight="bold",
        color=TEXT_COLOR,
        clip_on=False,
    )

    legend_handles = [
        Line2D(
            [0],
            [0],
            marker="o",
            linestyle="none",
            markersize=4.7,
            markerfacecolor=INTRA_COLOR,
            markeredgecolor=INTRA_COLOR,
            label="Within ± SD",
        ),
        Line2D(
            [0],
            [0],
            marker="|",
            linestyle="none",
            markersize=8.0,
            markeredgewidth=1.2,
            color="#4A4A4A",
            label="Nearest",
        ),
        Line2D(
            [0],
            [0],
            marker="o",
            linestyle="none",
            markersize=4.7,
            markerfacecolor="white",
            markeredgecolor=INTER_COLOR,
            label="All between ± SD",
        ),
    ]
    right_a.legend(
        handles=legend_handles,
        loc="lower center",
        bbox_to_anchor=(0.50, 0.02),
        frameon=False,
        fontsize=6.2,
        handlelength=1.0,
        handletextpad=0.4,
        borderaxespad=0,
    )

    ratio_axis.set_xlim(0.0, 1.0)
    ratio_axis.set_ylim(left_a.get_ylim())
    ratio_axis.set_axis_off()
    ratio_axis.axhspan(
        bath_low,
        bath_high,
        facecolor="#F5F5F5",
        edgecolor="none",
        zorder=-2,
    )
    for position, ratio in zip(
        positions,
        frame["nearest_state_ratio"].to_numpy(dtype=float),
    ):
        ratio_axis.text(
            0.52,
            position,
            f"{ratio:.2f}",
            ha="right",
            va="center",
            fontsize=7.0,
            color=RATIO_FLAG_COLOR if ratio < 1.0 else "#4D4D4D",
            fontweight="bold" if ratio < 1.0 else "normal",
        )
    ratio_axis.text(
        0.50,
        -0.18,
        "Nearest-condition\nratio",
        ha="center",
        va="center",
        fontsize=6.6,
        color="#666666",
    )

    for axis in (left_b, right_b):
        _style_figure7_boxplot(axis, within, between)
        axis.set_ylim(-0.53, 1.53)
        style_publication_axis(axis, grid_axis="x")
        axis.tick_params(axis="x", labelsize=6.8, pad=2)
    left_b.set_xlim(0.0, 32.0)
    left_b.set_xticks((0.0, 10.0, 20.0, 30.0))
    right_b.set_xlim(40.0, 55.0)
    right_b.set_xticks((40.0, 50.0))
    left_b.set_yticks(
        (1.0, 0.0),
        (f"Within (n = {len(within)})", f"Between (n = {len(between)})"),
    )
    left_b.tick_params(axis="y", labelsize=6.8, pad=5)
    right_b.tick_params(axis="y", left=False, labelleft=False)
    left_b.spines["right"].set_visible(False)
    right_b.spines["left"].set_visible(False)
    left_b.text(
        label_x,
        1.08,
        "b  Pooled distributions",
        transform=left_b.transAxes,
        ha="left",
        va="bottom",
        fontsize=8.1,
        fontweight="bold",
        color=TEXT_COLOR,
        clip_on=False,
    )

    slash_kwargs = {
        "color": "#444444",
        "clip_on": False,
        "linewidth": 0.75,
    }
    for left_axis, right_axis in ((left_a, right_a), (left_b, right_b)):
        left_axis.plot(
            (0.992, 1.008),
            (-0.010, 0.010),
            transform=left_axis.transAxes,
            **slash_kwargs,
        )
        right_axis.plot(
            (-0.008, 0.008),
            (-0.010, 0.010),
            transform=right_axis.transAxes,
            **slash_kwargs,
        )
    separator_x = ratio_left / FIGURE7_WIDTH_IN
    fig.add_artist(
        Line2D(
            [separator_x, separator_x],
            [panel_b_bottom / figure_height, (panel_a_bottom + panel_a_height) / figure_height],
            transform=fig.transFigure,
            color="#D9D9D9",
            linewidth=0.5,
        )
    )
    fig.text(
        (left_edge + left_width + break_gap / 2) / FIGURE7_WIDTH_IN,
        0.265 / figure_height,
        "32-40 omitted",
        ha="center",
        va="center",
        fontsize=6.0,
        color="#666666",
    )
    fig.text(
        (
            left_edge
            + (left_width + break_gap + right_width) / 2
        )
        / FIGURE7_WIDTH_IN,
        0.115 / figure_height,
        "Pairwise distance (standardized units; broken axis)",
        ha="center",
        va="center",
        fontsize=7.7,
        color=TEXT_COLOR,
    )
    save_figure_triplet(fig, output_dir / FIGURE7_STEM, dpi)


def _plot_figure7_condition_distance_variant_b(
    *,
    pair_frame: pd.DataFrame,
    state_distance_frame: pd.DataFrame,
    nearest_frame: pd.DataFrame,
    output_dir: Path,
    dpi: int,
    panel_b_variant: str = "default",
    palette_variant: str = "current",
    ratio_first: bool = False,
) -> None:
    """Direction B: aligned numeric table, ratio forest, and pooled ECDF."""

    frame = _figure7_variant_frame(state_distance_frame, nearest_frame)
    within, between = _figure7_distance_arrays(pair_frame)
    if panel_b_variant not in FIGURE7_PANEL_B_VARIANTS:
        raise ValueError(
            f"Unsupported Figure 7 panel-b variant {panel_b_variant!r}"
        )
    if palette_variant not in FIGURE7_PALETTE_VARIANTS:
        raise ValueError(
            f"Unsupported Figure 7 palette variant {palette_variant!r}"
        )
    if palette_variant != "current" and panel_b_variant != "default":
        raise ValueError(
            "Figure 7 palette variants p1/p2/p3 require the finalized "
            "default panel b"
        )
    if ratio_first and panel_b_variant != "default":
        raise ValueError("The rowwise layout requires the finalized default ECDF")
    figure_height = 5.55 if ratio_first else 5.22
    fig = plt.figure(figsize=(FIGURE7_WIDTH_IN, figure_height))

    palette = FIGURE7_PALETTES[palette_variant]
    within_color = palette["within"]
    neutral_color = palette["nearest"]
    between_color = palette["between"]
    warning_color = palette["warning"]
    text_color = palette["text"]
    heading_color = palette["heading"]
    title_color = palette["title"]
    axis_color = palette["axis"]
    grid_color = palette["grid"]
    zebra_color = palette["zebra"]
    rule_color = palette["rule"]
    major_rule_color = palette["major_rule"]
    reference_color = palette["reference"]

    left_margin = 0.06
    label_width = 1.55
    within_width = 1.02
    nearest_width = 0.75
    between_width = 1.10
    ratio_gap = 0.14
    lollipop_width = 2.00
    ratio_value_width = 0.50
    label_x = left_margin
    within_x = label_x + label_width
    nearest_x = within_x + within_width
    between_x = nearest_x + nearest_width
    separator_x = between_x + between_width + ratio_gap / 2
    lollipop_x = between_x + between_width + ratio_gap
    ratio_value_x = lollipop_x + lollipop_width
    right_edge = ratio_value_x + ratio_value_width

    row_height = 0.182
    group_header_height = 0.164
    table_padding = 0.045
    top_margin = 0.09
    panel_title_height = 0.215
    column_header_height = 0.210
    ratio_axis_height = 0.205
    panel_gap = 0.130
    panel_b_title_height = 0.190
    ecdf_height = 0.780

    label_fontsize = 7.2
    value_fontsize = 7.2
    group_fontsize = 7.2
    column_fontsize = 7.0
    tick_fontsize = 7.0
    x_label_fontsize = 7.8
    legend_fontsize = 6.9

    if ratio_first:
        # Identity -> graphical comparison -> exact source summaries.
        # Keep B's default geometry intact; this is an independent delivery.
        left_margin = label_x = 0.14
        label_width = 1.65
        lollipop_x = label_x + label_width
        lollipop_width = 2.00
        ratio_value_width = 0.34
        ratio_value_x = lollipop_x + lollipop_width
        within_x = ratio_value_x + ratio_value_width + 0.12
        within_width, nearest_width, between_width = 1.02, 0.66, 1.10
        nearest_x = within_x + within_width
        between_x = nearest_x + nearest_width
        right_edge = between_x + between_width
        separator_x = within_x - 0.06
        row_height = 0.190
        top_margin = 0.100
        panel_title_height = column_header_height = 0.240
        ratio_axis_height = 0.180
        panel_gap = 0.100
        panel_b_title_height = 0.200
        ecdf_height = 1.000
        label_fontsize = 7.3
        value_fontsize = 6.9
        group_fontsize = 7.2

    cursor = table_padding
    items: list[tuple[str, Any, float]] = []
    previous_environment: str | None = None
    for row in frame.itertuples(index=False):
        environment_id = str(row.environment_id)
        if environment_id != previous_environment:
            cursor += group_header_height
            items.append(
                (
                    "heading",
                    environment_id,
                    cursor - group_header_height * 0.34,
                )
            )
            previous_environment = environment_id
        items.append(("row", row, cursor + row_height / 2))
        cursor += row_height
    table_height = cursor + table_padding
    table_top = top_margin + panel_title_height + column_header_height
    ecdf_top = (
        table_top
        + table_height
        + ratio_axis_height
        + panel_gap
        + panel_b_title_height
    )

    def x_fraction(value: float) -> float:
        return value / FIGURE7_WIDTH_IN

    def y_from_top(value: float) -> float:
        return 1.0 - value / figure_height

    def add_axis_from_top(
        x: float,
        y_top: float,
        width: float,
        height: float,
    ) -> matplotlib.axes.Axes:
        return fig.add_axes(
            [
                x / FIGURE7_WIDTH_IN,
                1.0 - (y_top + height) / figure_height,
                width / FIGURE7_WIDTH_IN,
                height / figure_height,
            ]
        )

    ratio_axis = add_axis_from_top(
        lollipop_x,
        table_top,
        lollipop_width,
        table_height,
    )
    ratio_axis.patch.set_alpha(0.0)
    if ratio_first:
        ratio_axis.set_xlim(0.65, 3.4)
    else:
        ratio_axis.set_xscale("log", base=2)
        ratio_axis.set_xlim(0.70, 3.55)
    ratio_axis.set_ylim(table_height, 0.0)
    for spine_name in ("top", "right", "left", "bottom"):
        ratio_axis.spines[spine_name].set_visible(False)
    ratio_axis.tick_params(
        axis="both",
        length=0,
        labelbottom=False,
        labelleft=False,
    )

    tick_axis = add_axis_from_top(
        lollipop_x,
        table_top + table_height,
        lollipop_width,
        ratio_axis_height * 0.02,
    )
    tick_axis.patch.set_alpha(0.0)
    if ratio_first:
        tick_axis.set_xlim(0.65, 3.4)
    else:
        tick_axis.set_xscale("log", base=2)
        tick_axis.set_xlim(0.70, 3.55)
    tick_axis.set_yticks([])
    for spine_name in ("top", "right", "left", "bottom"):
        tick_axis.spines[spine_name].set_visible(False)
    tick_axis.set_xticks((1.0, 2.0, 3.0) if ratio_first else (0.75, 1.0, 1.5, 2.0, 3.0))
    tick_axis.set_xticklabels(("1\n(equal)", "2", "3") if ratio_first else ("0.75", "1", "1.5", "2", "3"))
    tick_axis.xaxis.set_minor_locator(matplotlib.ticker.NullLocator())
    tick_axis.tick_params(
        axis="x",
        labelsize=tick_fontsize,
        colors=axis_color,
        length=3.0,
        width=0.8,
        pad=2.0,
    )
    for grid_x in ((2.0, 3.0) if ratio_first else (0.75, 1.5, 2.0, 3.0)):
        ratio_axis.axvline(
            grid_x,
            color=grid_color,
            linewidth=0.6,
            zorder=0,
        )
    ratio_axis.axvline(
        1.0,
        color=reference_color,
        linewidth=0.8,
        linestyle=(0, (3.0, 2.0)),
        zorder=1,
    )

    def aligned_mean_sd(
        anchor_x: float,
        y: float,
        mean: float,
        sd: float,
        color: str,
    ) -> None:
        figure_y = y_from_top(table_top) - y / figure_height
        fig.text(
            x_fraction(anchor_x - 0.050),
            figure_y,
            f"{mean:.2f}",
            ha="right",
            va="center",
            fontsize=value_fontsize,
            color=color,
        )
        fig.text(
            x_fraction(anchor_x),
            figure_y,
            "±",
            ha="center",
            va="center",
            fontsize=value_fontsize,
            color=color,
        )
        fig.text(
            x_fraction(anchor_x + 0.050),
            figure_y,
            f"{sd:.2f}",
            ha="left",
            va="center",
            fontsize=value_fontsize,
            color=color,
        )

    zebra = True
    for kind, payload, y_position in items:
        figure_y = y_from_top(table_top) - y_position / figure_height
        if kind == "heading":
            environment_id = str(payload)
            fig.text(
                x_fraction(label_x),
                figure_y,
                paper_environment_label(environment_id),
                ha="left",
                va="center",
                fontsize=group_fontsize,
                color=heading_color,
                fontweight="normal",
            )
            rule_y = y_position + group_header_height * 0.42
            fig.add_artist(
                Line2D(
                    [x_fraction(label_x), x_fraction(right_edge)],
                    [
                        y_from_top(table_top) - rule_y / figure_height,
                        y_from_top(table_top) - rule_y / figure_height,
                    ],
                    color=rule_color,
                    linewidth=0.6,
                )
            )
            continue

        row = payload
        zebra = not zebra
        if zebra and not ratio_first:
            fig.add_artist(
                Rectangle(
                    (
                        x_fraction(label_x),
                        y_from_top(table_top)
                        - (y_position + row_height / 2) / figure_height,
                    ),
                    x_fraction(right_edge - label_x),
                    row_height / figure_height,
                    transform=fig.transFigure,
                    facecolor=zebra_color,
                    edgecolor="none",
                    zorder=0,
                )
            )
        state = str(row.state)
        fig.text(
            x_fraction(label_x + 0.055),
            figure_y,
            paper_state_code(state),
            ha="left",
            va="center",
            fontsize=label_fontsize,
            color=text_color,
        )
        fig.text(
            x_fraction(label_x + 0.28),
            figure_y,
            paper_state_short_label(state),
            ha="left",
            va="center",
            fontsize=label_fontsize,
            color=text_color,
            fontweight="bold" if ratio_first and float(row.nearest_state_ratio) < 1.0 else "normal",
        )
        aligned_mean_sd(
            within_x + within_width / 2,
            y_position,
            float(row.mean_intra_distance),
            float(row.std_intra_distance),
            within_color,
        )
        fig.text(
            x_fraction(nearest_x + nearest_width / 2 + 0.10),
            figure_y,
            f"{row.mean_nearest_inter_distance:.2f}",
            ha="right",
            va="center",
            fontsize=value_fontsize,
            color=neutral_color,
        )
        aligned_mean_sd(
            between_x + between_width / 2,
            y_position,
            float(row.mean_inter_distance),
            float(row.std_inter_distance),
            between_color,
        )

        ratio = float(row.nearest_state_ratio)
        ratio_color = warning_color if ratio < 1.0 else neutral_color
        ratio_axis.plot(
            (1.0, ratio),
            (y_position, y_position),
            color=ratio_color,
            linewidth=1.25 if ratio_first else 1.1,
            solid_capstyle="butt",
            zorder=3,
        )
        ratio_axis.plot(
            ratio,
            y_position,
            marker="D",
            markersize=4.7 if ratio_first else 4.4,
            markerfacecolor=ratio_color if ratio_first and ratio < 1.0 else "white",
            markeredgecolor=ratio_color,
            markeredgewidth=1.25,
            zorder=4,
        )
        fig.text(
            x_fraction(ratio_value_x + ratio_value_width - 0.055),
            figure_y,
            f"{ratio:.2f}",
            ha="right",
            va="center",
            fontsize=7.3 if ratio_first else value_fontsize,
            color=ratio_color,
            fontweight="bold" if ratio < 1.0 else "normal",
        )

    fig.add_artist(
        Line2D(
            [x_fraction(separator_x), x_fraction(separator_x)],
            [
                y_from_top(table_top + table_height),
                y_from_top(table_top - column_header_height + 0.03),
            ],
            color=rule_color,
            linewidth=0.7,
        )
    )
    fig.add_artist(
        Line2D(
            [x_fraction(label_x), x_fraction(right_edge)],
            [y_from_top(table_top - 0.015)] * 2,
            color=major_rule_color,
            linewidth=0.8,
        )
    )
    fig.add_artist(
        Line2D(
            [x_fraction(label_x), x_fraction(right_edge)],
            [y_from_top(table_top + table_height)] * 2,
            color=major_rule_color,
            linewidth=0.8,
        )
    )

    header_y = table_top - 0.075
    for x, width, label, color in (
        (within_x, within_width, "Within ± SD", within_color),
        (nearest_x, nearest_width, "Nearest", neutral_color),
        (between_x, between_width, "All between ± SD", between_color),
    ):
        fig.text(
            x_fraction(x + width / 2),
            y_from_top(header_y),
            label,
            ha="center",
            va="baseline",
            fontsize=column_fontsize,
            color=color,
            fontweight="bold",
        )
    fig.text(
        x_fraction(label_x + 0.055),
        y_from_top(header_y),
        "Condition",
        ha="left",
        va="baseline",
        fontsize=column_fontsize,
        color=text_color,
        fontweight="bold",
    )
    ratio_header_center = (
        (lollipop_x + ratio_value_x + ratio_value_width) / 2
        if ratio_first else (lollipop_x + right_edge) / 2
    )
    fig.text(
        x_fraction(ratio_header_center),
        y_from_top(header_y - 0.098),
        "Separation ratio",
        ha="center",
        va="baseline",
        fontsize=7.4 if ratio_first else column_fontsize,
        color=text_color,
        fontweight="bold",
    )
    fig.text(
        x_fraction(ratio_header_center),
        y_from_top(header_y),
        "nearest ÷ within",
        ha="center",
        va="baseline",
        fontsize=column_fontsize - 0.4,
        color=heading_color,
    )

    if ratio_first:
        fig.text(
            x_fraction((within_x + right_edge) / 2),
            y_from_top(header_y - 0.098),
            "Distances (standardized units)", ha="center", va="baseline",
            fontsize=6.6, color=heading_color,
        )

    distribution_left = label_x + 0.62
    distribution_width = right_edge - label_x - 0.62

    def style_panel_b_axis(axis: matplotlib.axes.Axes) -> None:
        style_publication_axis(axis)
        axis.spines["left"].set_color(axis_color)
        axis.spines["left"].set_linewidth(0.9)
        axis.spines["bottom"].set_color(axis_color)
        axis.spines["bottom"].set_linewidth(0.9)
        axis.tick_params(
            labelsize=tick_fontsize,
            colors=axis_color,
            length=3.0,
            width=0.8,
            pad=2.0,
        )

    if panel_b_variant in {"default", "v1"}:
        cdf_axis = add_axis_from_top(
            distribution_left,
            ecdf_top,
            distribution_width,
            ecdf_height,
        )
        cdf_axis.set_xlim(0.0, 55.0)
        cdf_axis.set_ylim(
            -0.145 if panel_b_variant == "v1" else -0.03,
            1.05,
        )
        style_panel_b_axis(cdf_axis)
        for grid_x in (10.0, 20.0, 30.0, 40.0, 50.0):
            cdf_axis.axvline(
                grid_x,
                color=grid_color,
                linewidth=0.6,
                zorder=0,
            )
        cdf_axis.axhline(
            0.5,
            color=grid_color,
            linewidth=0.6,
            zorder=0,
        )
        if panel_b_variant == "default":
            curve_specs = (
                (
                    within,
                    within_color,
                    30.6,
                    0.98,
                    "left",
                    f"Within  (n = {len(within)})",
                ),
                (
                    between,
                    between_color,
                    52.6,
                    0.60,
                    "right",
                    f"Between  (n = {len(between)})",
                ),
            )
            for values, color, curve_label_x, curve_label_y, alignment, label in curve_specs:
                ordered_values = np.sort(values)
                cumulative = (
                    np.arange(1, len(values) + 1, dtype=float) / len(values)
                )
                cdf_axis.step(
                    np.concatenate(([0.0], ordered_values)),
                    np.concatenate(([0.0], cumulative)),
                    where="post",
                    color=color,
                    linewidth=1.0,
                    solid_joinstyle="miter",
                    solid_capstyle="butt",
                    zorder=3,
                )
                cdf_axis.text(
                    curve_label_x,
                    curve_label_y,
                    label,
                    color=color,
                    fontsize=legend_fontsize,
                    ha=alignment,
                    va="center",
                    zorder=4,
                )
        else:
            for values, color, linestyle, label in (
                (
                    within,
                    within_color,
                    "-",
                    f"Within  (n = {len(within)})",
                ),
                (
                    between,
                    between_color,
                    (0, (4.0, 2.0)),
                    f"Between  (n = {len(between)})",
                ),
            ):
                ordered_values = np.sort(values)
                cumulative = (
                    np.arange(1, len(values) + 1, dtype=float) / len(values)
                )
                cdf_axis.step(
                    np.concatenate(([0.0], ordered_values)),
                    np.concatenate(([0.0], cumulative)),
                    where="post",
                    color=color,
                    linestyle=linestyle,
                    linewidth=1.5,
                    zorder=3,
                    label=label,
                )
            cdf_axis.axhline(
                0.0,
                color="#BEBEBE",
                linewidth=0.55,
                zorder=1,
            )
            cdf_axis.vlines(
                within,
                -0.132,
                -0.088,
                color=within_color,
                linewidth=0.38,
                alpha=0.62,
                zorder=2,
            )
            cdf_axis.vlines(
                between,
                -0.074,
                -0.030,
                color=between_color,
                linewidth=0.26,
                alpha=0.36,
                zorder=2,
            )
            cdf_axis.text(
                -0.45,
                -0.110,
                "Within",
                ha="right",
                va="center",
                fontsize=6.6,
                color=within_color,
                clip_on=False,
            )
            cdf_axis.text(
                -0.45,
                -0.052,
                "Between",
                ha="right",
                va="center",
                fontsize=6.6,
                color=between_color,
                clip_on=False,
            )
        cdf_axis.set_xticks((0.0, 10.0, 20.0, 30.0, 40.0, 50.0))
        cdf_axis.set_yticks((0.0, 0.5, 1.0))
        cdf_axis.set_yticklabels(("0", "0.5", "1.0"))
        cdf_axis.set_ylabel(
            "Cumulative\nfraction",
            fontsize=column_fontsize,
            color=text_color,
            labelpad=3.0,
            linespacing=1.3,
        )
        cdf_axis.set_xlabel(
            "Pairwise distance (standardized units)",
            fontsize=x_label_fontsize,
            color=text_color,
            labelpad=3.0,
        )
        if panel_b_variant == "v1":
            legend = cdf_axis.legend(
                loc="lower right",
                frameon=False,
                fontsize=legend_fontsize,
                handlelength=1.8,
                handletextpad=0.5,
                borderpad=0.2,
                bbox_to_anchor=(1.0, 0.02),
            )
            for legend_text in legend.get_texts():
                legend_text.set_color(text_color)
            panel_b_title = "Pooled ECDFs with observation rugs"
        else:
            panel_b_title = "Pooled distance distributions"
    else:
        bin_edges = np.arange(0.0, 58.0, 2.0)
        within_fraction = (
            np.histogram(within, bins=bin_edges)[0].astype(float)
            / len(within)
        )
        between_fraction = (
            np.histogram(between, bins=bin_edges)[0].astype(float)
            / len(between)
        )
        common_ymax = math.ceil(
            max(float(within_fraction.max()), float(between_fraction.max()))
            / 0.05
        ) * 0.05
        histogram_height = 0.33
        histogram_gap = 0.08
        histogram_axes: list[matplotlib.axes.Axes] = []
        for index, (fractions, color, linestyle, hatch, label) in enumerate(
            (
                (
                    within_fraction,
                    within_color,
                    "-",
                    "///",
                    f"Within  (n = {len(within)})",
                ),
                (
                    between_fraction,
                    between_color,
                    (0, (4.0, 2.0)),
                    "...",
                    f"Between  (n = {len(between)})",
                ),
            )
        ):
            histogram_axis = add_axis_from_top(
                distribution_left,
                ecdf_top + index * (histogram_height + histogram_gap),
                distribution_width,
                histogram_height,
            )
            histogram_axes.append(histogram_axis)
            histogram_axis.set_xlim(0.0, 55.0)
            histogram_axis.set_ylim(0.0, common_ymax)
            style_panel_b_axis(histogram_axis)
            step_patch = histogram_axis.stairs(
                fractions,
                bin_edges,
                baseline=0.0,
                fill=True,
                facecolor=matplotlib.colors.to_rgba(color, 0.12),
                edgecolor=color,
                linewidth=1.05,
                linestyle=linestyle,
                zorder=2,
            )
            step_patch.set_hatch(hatch)
            for grid_y in np.arange(0.1, common_ymax + 0.001, 0.1):
                histogram_axis.axhline(
                    grid_y,
                    color=grid_color,
                    linewidth=0.5,
                    zorder=0,
                )
            histogram_axis.set_yticks(
                np.arange(0.0, common_ymax + 0.001, 0.1)
            )
            histogram_axis.text(
                0.995,
                0.80,
                label,
                transform=histogram_axis.transAxes,
                ha="right",
                va="center",
                fontsize=6.6,
                color=text_color,
            )
            if index == 0:
                histogram_axis.tick_params(
                    axis="x",
                    bottom=False,
                    labelbottom=False,
                )
            else:
                histogram_axis.set_xticks(
                    (0.0, 10.0, 20.0, 30.0, 40.0, 50.0)
                )
                histogram_axis.set_xlabel(
                    "Pairwise distance (standardized units)",
                    fontsize=x_label_fontsize,
                    color=text_color,
                    labelpad=3.0,
                )
        fig.text(
            x_fraction(0.20),
            y_from_top(ecdf_top + ecdf_height / 2),
            "Fraction per\n2-unit bin",
            ha="center",
            va="center",
            rotation=90,
            fontsize=column_fontsize,
            color=text_color,
            linespacing=1.25,
        )
        panel_b_title = "Pooled distributions in fixed 2-unit bins"

    title_specs = (
        (
            "a",
            "Per-condition separation" if ratio_first else "Per-condition distance summary",
            None if ratio_first else "standardized units",
            top_margin,
        ),
        (
            "b",
            panel_b_title,
            None,
            ecdf_top - panel_b_title_height,
        ),
    )
    for panel_letter, title, subtitle, title_top in title_specs:
        title_y = y_from_top(title_top + 0.112)
        fig.text(
            x_fraction(left_margin),
            title_y,
            panel_letter,
            ha="left",
            va="baseline",
            fontsize=9.0,
            fontweight="bold",
            color=title_color,
        )
        title_artist = fig.text(
            x_fraction(left_margin + 0.14),
            title_y,
            title,
            ha="left",
            va="baseline",
            fontsize=8.4,
            fontweight="bold",
            color=title_color,
        )
        if subtitle:
            fig.canvas.draw()
            title_width = (
                title_artist.get_window_extent(fig.canvas.get_renderer()).width
                / fig.dpi
            )
            fig.text(
                x_fraction(left_margin + 0.14 + title_width + 0.11),
                title_y,
                subtitle,
                ha="left",
                va="baseline",
                fontsize=column_fontsize,
                color=heading_color,
            )

    save_figure_triplet(fig, output_dir / FIGURE7_STEM, dpi)


def _plot_figure7_condition_distance_variant_c(
    *,
    pair_frame: pd.DataFrame,
    state_distance_frame: pd.DataFrame,
    nearest_frame: pd.DataFrame,
    output_dir: Path,
    dpi: int,
) -> None:
    """Direction C: room facets with categorical tracks on a log distance axis."""

    frame = _figure7_variant_frame(state_distance_frame, nearest_frame)
    within_pairs, between_pairs = _figure7_distance_arrays(pair_frame)
    figure_height = 4.95
    fig = plt.figure(figsize=(FIGURE7_WIDTH_IN, figure_height))
    left = 0.58
    right = 0.16
    gap = 0.22
    facet_width = (FIGURE7_WIDTH_IN - left - right - 2 * gap) / 3
    facet_axes: list[matplotlib.axes.Axes] = []
    for index, environment_id in enumerate(ENVIRONMENT_LABELS):
        axis = fig.add_axes(
            [
                (left + index * (facet_width + gap)) / FIGURE7_WIDTH_IN,
                1.53 / figure_height,
                facet_width / FIGURE7_WIDTH_IN,
                2.82 / figure_height,
            ],
            sharey=facet_axes[0] if facet_axes else None,
        )
        facet_axes.append(axis)
        room = frame.loc[frame["environment_id"] == environment_id]
        x_positions = np.arange(len(room), dtype=float)
        within = room["mean_intra_distance"].to_numpy(dtype=float)
        within_sd = room["std_intra_distance"].to_numpy(dtype=float)
        nearest = room["mean_nearest_inter_distance"].to_numpy(dtype=float)
        between = room["mean_inter_distance"].to_numpy(dtype=float)
        between_sd = room["std_inter_distance"].to_numpy(dtype=float)
        ratios = room["nearest_state_ratio"].to_numpy(dtype=float)
        axis.errorbar(
            x_positions - 0.085,
            within,
            yerr=within_sd,
            fmt="o",
            markersize=4.3,
            markerfacecolor=INTRA_COLOR,
            markeredgecolor=INTRA_COLOR,
            ecolor=INTRA_COLOR,
            elinewidth=0.9,
            capsize=0,
            zorder=3,
        )
        axis.errorbar(
            x_positions + 0.085,
            between,
            yerr=between_sd,
            fmt="o",
            markersize=4.3,
            markerfacecolor="white",
            markeredgecolor=INTER_COLOR,
            markeredgewidth=0.95,
            ecolor=INTER_COLOR,
            elinewidth=0.9,
            capsize=0,
            zorder=3,
        )
        for x_position, within_value, nearest_value, ratio in zip(
            x_positions,
            within,
            nearest,
            ratios,
        ):
            connector_color = (
                RATIO_FLAG_COLOR if ratio < 1.0 else "#7A7A7A"
            )
            axis.plot(
                (x_position - 0.085, x_position),
                (within_value, nearest_value),
                color=connector_color,
                linewidth=0.8,
                zorder=2.5,
            )
            axis.scatter(
                [x_position],
                [nearest_value],
                s=4.1**2,
                marker="D",
                facecolor="white",
                edgecolor=connector_color,
                linewidth=0.9,
                zorder=4,
            )
            geometric_midpoint = math.sqrt(within_value * nearest_value)
            axis.annotate(
                f"{ratio:.2f}×",
                xy=(x_position, geometric_midpoint),
                xytext=(3, 0),
                textcoords="offset points",
                ha="left",
                va="center",
                fontsize=6.0,
                color=connector_color,
                fontweight="bold" if ratio < 1.0 else "normal",
                zorder=5,
            )
        axis.set_yscale("log")
        axis.set_ylim(4.0, 55.0)
        axis.set_yticks((5.0, 10.0, 20.0, 40.0))
        axis.set_yticklabels(("5", "10", "20", "40"))
        axis.minorticks_off()
        axis.set_xlim(-0.48, len(room) - 0.52)
        axis.set_xticks(
            x_positions,
            [paper_state_code(str(state)) for state in room["state"]],
        )
        axis.tick_params(axis="x", labelsize=6.8, pad=3, length=0)
        axis.tick_params(axis="y", labelsize=6.6, pad=2)
        style_publication_axis(axis, grid_axis="y")
        axis.spines["left"].set_visible(index == 0)
        if index > 0:
            axis.tick_params(axis="y", left=False, labelleft=False)
        axis.set_title(
            paper_environment_label(environment_id),
            fontsize=7.4,
            fontweight="bold",
            color="#555555",
            pad=5,
        )
    facet_axes[0].set_ylabel(
        "Distance (standardized units; log scale)",
        fontsize=7.4,
    )
    fig.text(
        left / FIGURE7_WIDTH_IN,
        4.73 / figure_height,
        "a  Room-faceted condition profiles",
        ha="left",
        va="center",
        fontsize=8.1,
        fontweight="bold",
        color=TEXT_COLOR,
    )
    legend_handles = [
        Line2D(
            [0],
            [0],
            marker="o",
            linestyle="none",
            markersize=4.3,
            markerfacecolor=INTRA_COLOR,
            markeredgecolor=INTRA_COLOR,
            label="Within ± SD",
        ),
        Line2D(
            [0],
            [0],
            marker="D",
            linestyle="none",
            markersize=4.1,
            markerfacecolor="white",
            markeredgecolor="#5A5A5A",
            label="Nearest (ratio label)",
        ),
        Line2D(
            [0],
            [0],
            marker="o",
            linestyle="none",
            markersize=4.3,
            markerfacecolor="white",
            markeredgecolor=INTER_COLOR,
            label="All between ± SD",
        ),
    ]
    fig.legend(
        handles=legend_handles,
        loc="upper right",
        bbox_to_anchor=(0.985, 0.978),
        ncol=3,
        frameon=False,
        fontsize=6.4,
        handlelength=1.0,
        columnspacing=0.9,
        handletextpad=0.4,
    )

    pooled_axis = fig.add_axes(
        [1.28 / 7.2, 0.37 / figure_height, 5.72 / 7.2, 0.62 / figure_height]
    )
    _style_figure7_boxplot(pooled_axis, within_pairs, between_pairs)
    pooled_axis.set_xscale("log")
    pooled_axis.set_xlim(2.0, 60.0)
    pooled_axis.set_xticks((2.0, 5.0, 10.0, 20.0, 40.0, 60.0))
    pooled_axis.set_xticklabels(("2", "5", "10", "20", "40", "60"))
    pooled_axis.minorticks_off()
    pooled_axis.set_yticks(
        (1.0, 0.0),
        (
            f"Within (n = {len(within_pairs)})",
            f"Between (n = {len(between_pairs)})",
        ),
    )
    pooled_axis.set_ylim(-0.53, 1.53)
    pooled_axis.set_xlabel(
        "Pairwise distance (standardized units; log scale)",
        fontsize=7.3,
    )
    pooled_axis.tick_params(axis="both", labelsize=6.4, pad=2)
    style_publication_axis(pooled_axis, grid_axis="x")
    pooled_axis.text(
        -0.19,
        1.07,
        "b  Pooled distributions",
        transform=pooled_axis.transAxes,
        ha="left",
        va="bottom",
        fontsize=8.1,
        fontweight="bold",
        color=TEXT_COLOR,
        clip_on=False,
    )
    save_figure_triplet(fig, output_dir / FIGURE7_STEM, dpi)


def _plot_figure7_condition_distance_facets(
    *,
    pair_frame: pd.DataFrame,
    state_distance_frame: pd.DataFrame,
    nearest_frame: pd.DataFrame,
    output_dir: Path,
    dpi: int,
    palette_variant: str = "current",
) -> None:
    """Refined facets: equal condition spacing, quiet SDs, and a full ECDF.

    Source means/SDs/ratios are unchanged. Log scaling is a coordinate
    transform, not a log transform followed by recomputation of summaries.
    """

    frame = _figure7_variant_frame(state_distance_frame, nearest_frame)
    within_pairs, between_pairs = _figure7_distance_arrays(pair_frame)
    palette = FIGURE7_PALETTES[palette_variant]
    height = FIGURE7_FACET_HEIGHT_IN
    fig = plt.figure(figsize=(FIGURE7_WIDTH_IN, height))
    left, right, gap, condition_pitch = 0.62, 0.16, 0.22, 0.46
    axes: list[matplotlib.axes.Axes] = []

    def figure_text(x: float, y: float, text: str, **kwargs: Any) -> Any:
        return fig.text(x / FIGURE7_WIDTH_IN, y / height, text, **kwargs)

    def sd_ink(color: str) -> tuple[float, float, float]:
        # Opaque light ink avoids color mixing where intervals cross a grid.
        rgb = matplotlib.colors.to_rgb(color)
        return tuple(0.52 * channel + 0.48 for channel in rgb)

    cursor = left
    for room_index, environment_id in enumerate(ENVIRONMENT_LABELS):
        room = frame.loc[frame["environment_id"] == environment_id]
        count = len(room)
        width = count * condition_pitch
        axis = fig.add_axes(
            [cursor / FIGURE7_WIDTH_IN, 2.35 / height,
             width / FIGURE7_WIDTH_IN, 2.35 / height],
            sharey=axes[0] if axes else None,
        )
        axes.append(axis)
        axis.set_gid(f"room-{environment_id}")
        positions = np.arange(count, dtype=float)
        axis.set_yscale("log")
        axis.set_ylim(3.5, 55.0)
        axis.set_xlim(-0.5, count - 0.5)
        axis.set_yticks((5.0, 10.0, 20.0, 40.0))
        axis.set_yticklabels(("5", "10", "20", "40"))
        axis.yaxis.set_minor_locator(matplotlib.ticker.NullLocator())
        for spine in axis.spines.values():
            spine.set_visible(False)
        axis.grid(axis="y", color=palette["grid"], linewidth=0.5, zorder=0)
        axis.tick_params(axis="y", length=0, pad=6, labelsize=7.2,
                         colors=palette["axis"], labelleft=room_index == 0)
        axis.set_xticks(positions, [paper_state_code(str(s)) for s in room["state"]])
        axis.tick_params(axis="x", length=0, pad=7, labelsize=7.4,
                         colors=palette["text"])

        for mean_column, sd_column, offset, role, fill in (
            ("mean_intra_distance", "std_intra_distance", -0.22, "within", True),
            ("mean_inter_distance", "std_inter_distance", 0.22, "between", False),
        ):
            means = room[mean_column].to_numpy(dtype=float)
            deviations = room[sd_column].to_numpy(dtype=float)
            if np.any(means - deviations <= 3.5) or np.any(means + deviations >= 55.0):
                raise ValueError("Facet range would clip a source mean +/- SD interval")
            container = axis.errorbar(
                positions + offset, means, yerr=deviations,
                fmt="o", linestyle="none", markersize=4.6,
                markerfacecolor=palette[role] if fill else "white",
                markeredgecolor=palette[role], markeredgewidth=1.05,
                ecolor=sd_ink(palette[role]), elinewidth=0.85,
                capsize=0, zorder=3,
            )
            container.lines[0].set_gid(f"{role}-means")
            container.lines[2][0].set_gid(f"{role}-source-sd")

        nearest_colors = [
            palette["warning"] if ratio < 1.0 else palette["nearest"]
            for ratio in room["nearest_state_ratio"]
        ]
        nearest_marks = axis.scatter(
            positions, room["mean_nearest_inter_distance"].to_numpy(dtype=float),
            marker="D", s=4.3**2, facecolors="white",
            edgecolors=nearest_colors, linewidths=1.1, zorder=4,
        )
        nearest_marks.set_gid("nearest-source-means")
        for position, ratio in zip(positions, room["nearest_state_ratio"]):
            ratio = float(ratio)
            figure_text(
                cursor + (position + 0.5) * condition_pitch, 2.035,
                f"{ratio:.2f}", ha="center", va="center", fontsize=7.1,
                color=palette["warning"] if ratio < 1 else palette["nearest"],
                fontweight="bold" if ratio < 1 else "normal",
            )
        figure_text(cursor, 4.905, paper_environment_label(environment_id),
                    ha="left", va="center", fontsize=8.2,
                    color=palette["heading"])
        fig.add_artist(Line2D(
            [cursor / FIGURE7_WIDTH_IN, (cursor + width) / FIGURE7_WIDTH_IN],
            [4.79 / height] * 2, transform=fig.transFigure,
            color=palette["rule"], linewidth=0.6,
        ))
        cursor += width + gap
    assert math.isclose(cursor - gap, FIGURE7_WIDTH_IN - right)
    axes[0].set_ylabel("Distance (standardized units; log scale)",
                       fontsize=7.7, labelpad=8, color=palette["text"])
    figure_text(left - 0.08, 2.035, "Nearest /\nwithin", ha="right", va="center",
                fontsize=6.5, color=palette["heading"], linespacing=1.2)

    # One restrained key for panel a; no condition-connecting segments.
    handles = [
        Line2D([], [], marker="o", linestyle="none", markersize=4.6,
               markerfacecolor=palette["within"], markeredgecolor=palette["within"],
               label="Within ± SD"),
        Line2D([], [], marker="D", linestyle="none", markersize=4.3,
               markerfacecolor="white", markeredgecolor=palette["nearest"],
               markeredgewidth=1.1, label="Nearest condition"),
        Line2D([], [], marker="o", linestyle="none", markersize=4.6,
               markerfacecolor="white", markeredgecolor=palette["between"],
               markeredgewidth=1.05, label="All between ± SD"),
    ]
    fig.legend(handles=handles, loc="center left",
               bbox_to_anchor=(left / FIGURE7_WIDTH_IN, 5.205 / height),
               ncol=3, frameon=False, fontsize=7.2, labelcolor=palette["text"],
               handlelength=1.2, handletextpad=0.4, columnspacing=1.7,
               borderaxespad=0)
    figure_text(0.18, 5.435, "a", fontsize=9.0, fontweight="bold",
                color=palette["title"], va="baseline")
    figure_text(0.36, 5.435, "Condition-level distances", fontsize=8.8,
                fontweight="bold", color=palette["title"], va="baseline")

    # The accepted pooled treatment is retained, with a taller plotting area.
    pooled = fig.add_axes([left / FIGURE7_WIDTH_IN, 0.42 / height,
                          (FIGURE7_WIDTH_IN - left - right) / FIGURE7_WIDTH_IN,
                          1.15 / height])
    pooled.set_gid("pooled-ecdf")
    pooled.set_xlim(0, 55)
    pooled.set_ylim(-0.03, 1.05)
    for spine_name in ("top", "right"):
        pooled.spines[spine_name].set_visible(False)
    for spine_name in ("left", "bottom"):
        pooled.spines[spine_name].set_color(palette["axis"])
        pooled.spines[spine_name].set_linewidth(0.7)
    for grid_x in (10, 20, 30, 40, 50):
        pooled.axvline(grid_x, color=palette["grid"], linewidth=0.5, zorder=0)
    pooled.axhline(0.5, color=palette["grid"], linewidth=0.5, zorder=0)
    for values, role, label, x, y, alignment in (
        (within_pairs, "within", f"Within  (n = {len(within_pairs)})", 30.6, 0.98, "left"),
        (between_pairs, "between", f"Between  (n = {len(between_pairs)})", 52.6, 0.60, "right"),
    ):
        ordered = np.sort(values)
        line, = pooled.step(
            np.r_[0.0, ordered], np.r_[0.0, np.arange(1, len(values) + 1) / len(values)],
            where="post", color=palette[role], linewidth=1.0, linestyle="-",
            solid_joinstyle="miter", solid_capstyle="butt", zorder=3,
        )
        line.set_gid(f"{role}-ecdf")
        pooled.text(x, y, label, ha=alignment, va="center", fontsize=7.1,
                    color=palette[role])
    pooled.set_xticks((0, 10, 20, 30, 40, 50))
    pooled.set_yticks((0.0, 0.5, 1.0), ("0", "0.5", "1.0"))
    pooled.tick_params(labelsize=7.2, colors=palette["axis"], length=3,
                       width=0.7, pad=3)
    pooled.set_ylabel("Cumulative\nfraction", fontsize=7.4, labelpad=7,
                       color=palette["text"], linespacing=1.3)
    pooled.set_xlabel("Pairwise distance (standardized units)", fontsize=7.8,
                       labelpad=4, color=palette["text"])
    figure_text(0.18, 1.775, "b", fontsize=9.0, fontweight="bold",
                color=palette["title"], va="baseline")
    figure_text(0.36, 1.775, "Pooled distance distributions", fontsize=8.8,
                fontweight="bold", color=palette["title"], va="baseline")
    save_figure_triplet(fig, output_dir / FIGURE7_STEM, dpi)


def figure7_layout_metadata(layout_variant: str) -> dict[str, Any]:
    """Return record-ready design metadata for one Figure 7 layout."""

    metadata: dict[str, dict[str, Any]] = {
        "r3": {
            "figure_size_inches": [FIGURE7_WIDTH_IN, FIGURE7_HEIGHT_IN],
            "layout": "r3 shared linear axis with a separate ratio column",
            "distance_scale": "linear 0–48",
            "minimum_font_size_pt": 6.8,
        },
        "a": {
            "figure_size_inches": [FIGURE7_WIDTH_IN, 5.25],
            "layout": "broken-axis interval forest with aligned pooled boxplots and ratio column",
            "distance_scale": "linear segments 0–32 and 40–55; 32–40 explicitly omitted",
            "minimum_font_size_pt": 6.0,
            "direction": "compress the empty tail while retaining positional comparison and SD intervals",
            "tradeoff": "absolute gaps cannot be judged across the explicit axis break",
        },
        "b": {
            "figure_size_inches": [FIGURE7_WIDTH_IN, 5.22],
            "layout": "aligned numeric condition table, log2 ratio forest with fixed value column, and pooled ECDF",
            "distance_scale": "absolute summaries printed numerically; ratio log2 0.70–3.55; pooled distribution linear 0–55",
            "minimum_font_size_pt": 6.6,
            "direction": "remove the sparse common position scale and prioritize exact reading",
            "tradeoff": "absolute distance patterns require table reading rather than immediate geometry",
            "row_background": "alternating palette-defined zebra rows across table and ratio region",
            "nearest_condition_color": "palette-defined",
            "ratio_value_alignment": "fixed right-aligned column",
            "mean_sd_alignment": "mean right; plus-minus centered; SD left",
        },
        "facets": {
            "figure_size_inches": [FIGURE7_WIDTH_IN, FIGURE7_FACET_HEIGHT_IN],
            "layout": "proportional room facets, quiet source SDs, aligned ratio labels, taller finalized ECDF",
            "distance_scale": "panel a log 3.5–55; panel b linear 0–55",
            "minimum_font_size_pt": 6.5,
            "facet_widths_inches": [1.84, 2.76, 1.38],
            "condition_pitch_inches": 0.46,
            "within_nearest_between_x_offsets": [-0.22, 0.0, 0.22],
            "facet_gap_inches": 0.22,
            "facet_height_inches": 2.35,
            "pooled_height_inches": 1.15,
            "sd_encoding": "source mean +/- SD endpoints, opaque 52-percent role ink plus 48-percent white; 0.85 pt; no caps",
            "summary_markers": "filled circle=within; hollow diamond=nearest; hollow circle=all between; no connectors",
            "ratio_encoding": "source ratios aligned below codes; nearest marks below within marks denote ratios < 1",
            "labels": "codes in plot; unchanged full English labels in caption",
            "tradeoff": "exact mean/SD lookup moves to frozen source tables; panel a uses a log scale",
        },
        "rowwise": {
            "figure_size_inches": [FIGURE7_WIDTH_IN, 5.55],
            "layout": "full condition labels, central linear ratio plot, exact distance columns, finalized ECDF",
            "distance_scale": "source summaries printed numerically; ratio linear 0.65–3.4; pooled distribution linear 0–55",
            "minimum_font_size_pt": 6.6,
            "reading_order": "condition -> nearest/within ratio and fixed value -> three exact distance summaries",
            "column_widths_inches": {"condition": 1.65, "ratio_plot": 2.0, "ratio_value": 0.34, "within": 1.02, "nearest": 0.66, "all_between": 1.10},
            "row_height_inches": 0.190,
            "row_background": "white; only room/header/bottom rules, no zebra texture",
            "ecdf_height_inches": 1.0,
            "ratio_definition": "nearest / within; dashed reference at 1; source ratios not recalculated",
            "subunity_emphasis": "filled diamond, left of 1, bold fixed ratio and condition name; palette warning color",
            "tradeoff": "not an absolute-distance position plot; linear ratio axis makes near-one deviations smaller than on B's log axis",
        },
        "c": {
            "figure_size_inches": [FIGURE7_WIDTH_IN, 4.95],
            "layout": "three room facets with categorical condition tracks and pooled log boxplots",
            "distance_scale": "log distance scale 4–55 in panel a and 2–60 in panel b",
            "minimum_font_size_pt": 6.0,
            "direction": "use room facets and log scaling to show the long tail without a break",
            "tradeoff": "log spacing and code-only facet ticks demand a more explanatory caption",
        },
    }
    if layout_variant not in metadata:
        raise ValueError(f"Unsupported Figure 7 layout variant {layout_variant!r}")
    return metadata[layout_variant]


def plot_figure7_condition_distance(
    *,
    pair_frame: pd.DataFrame,
    state_distance_frame: pd.DataFrame,
    nearest_frame: pd.DataFrame,
    output_dir: Path,
    dpi: int,
    layout_variant: str = "r3",
    panel_b_variant: str = "default",
    palette_variant: str = "current",
) -> None:
    """Dispatch to the requested provenance-preserving Figure 7 layout."""

    if layout_variant not in FIGURE7_LAYOUT_VARIANTS:
        raise ValueError(f"Unsupported Figure 7 layout variant {layout_variant!r}")
    if layout_variant in {"b", "rowwise"}:
        _plot_figure7_condition_distance_variant_b(
            pair_frame=pair_frame,
            state_distance_frame=state_distance_frame,
            nearest_frame=nearest_frame,
            output_dir=output_dir,
            dpi=dpi,
            panel_b_variant=panel_b_variant,
            palette_variant=palette_variant,
            ratio_first=layout_variant == "rowwise",
        )
        return
    if panel_b_variant != "default":
        raise ValueError(
            "--panel-b-variant v1/v2 is only valid with "
            "--layout-variant b"
        )
    if layout_variant == "facets":
        _plot_figure7_condition_distance_facets(
            pair_frame=pair_frame,
            state_distance_frame=state_distance_frame,
            nearest_frame=nearest_frame,
            output_dir=output_dir,
            dpi=dpi,
            palette_variant=palette_variant,
        )
        return
    if palette_variant != "current":
        raise ValueError(
            "--palette-variant p1/p2/p3 is only valid with "
            "--layout-variant b, facets, or rowwise"
        )
    plotters = {
        "r3": _plot_figure7_condition_distance_r3,
        "a": _plot_figure7_condition_distance_variant_a,
        "c": _plot_figure7_condition_distance_variant_c,
    }
    plotters[layout_variant](
        pair_frame=pair_frame,
        state_distance_frame=state_distance_frame,
        nearest_frame=nearest_frame,
        output_dir=output_dir,
        dpi=dpi,
    )


def cliffs_delta(
    reference_values: np.ndarray,
    comparison_values: np.ndarray,
) -> float:
    """Return Cliff's delta, positive when comparison values are larger."""

    differences = (
        comparison_values[:, np.newaxis]
        - reference_values[np.newaxis, :]
    )
    greater = int(np.count_nonzero(differences > 0))
    lower = int(np.count_nonzero(differences < 0))
    return (greater - lower) / float(differences.size)


def distance_analysis(
    statistics: Sequence[RecordingStatistics],
    *,
    output_dir: Path,
    dpi: int,
    distance_plot: str,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    raw_features = np.stack(
        [
            np.concatenate(
                (
                    item.normalized_mean_curve,
                    item.normalized_variance_curve,
                )
            )
            for item in statistics
        ]
    )
    # The comparison set is the complete discovered unoccupied-recording set.
    # One mean and one population SD are fitted per feature dimension across
    # all recording vectors. No recording is standardized independently.
    feature_mean = raw_features.mean(axis=0)
    feature_std = raw_features.std(axis=0, ddof=0)
    usable_feature_mask = feature_std > EPSILON
    if int(usable_feature_mask.sum()) < 2:
        raise ValueError("Fewer than two non-constant feature dimensions")
    standardized_features = (
        raw_features[:, usable_feature_mask]
        - feature_mean[usable_feature_mask]
    ) / feature_std[usable_feature_mask]
    distance_matrix = squareform(pdist(standardized_features, "euclidean"))

    pair_rows: list[dict[str, Any]] = []
    for left in range(len(statistics)):
        record_a = statistics[left].recording
        for right in range(left + 1, len(statistics)):
            record_b = statistics[right].recording
            same_state = record_a.state == record_b.state
            same_context = (
                record_a.environment_id == record_b.environment_id
                and record_a.link_configuration
                == record_b.link_configuration
            )
            if same_state:
                comparison_type = "intra_state"
            elif same_context:
                comparison_type = "inter_state_matched_environment_link"
            else:
                comparison_type = "inter_state_cross_environment_or_link"
            pair_rows.append(
                {
                    "comparison_type": comparison_type,
                    "distance": distance_matrix[left, right],
                    "state_a": record_a.state,
                    "state_b": record_b.state,
                    "sample_id_a": record_a.sample_id,
                    "sample_id_b": record_b.sample_id,
                    "environment_id_a": record_a.environment_id,
                    "environment_id_b": record_b.environment_id,
                    "link_configuration_a": record_a.link_configuration,
                    "link_configuration_b": record_b.link_configuration,
                }
            )
    pair_frame = pd.DataFrame(pair_rows)
    pair_frame.to_csv(
        output_dir / "distance_pairs.csv",
        index=False,
        float_format="%.8f",
    )

    state_rows: list[dict[str, Any]] = []
    states = sorted({item.recording.state for item in statistics})
    for state in states:
        intra = pair_frame.loc[
            (pair_frame["comparison_type"] == "intra_state")
            & (pair_frame["state_a"] == state),
            "distance",
        ]
        matched_inter = pair_frame.loc[
            (
                pair_frame["comparison_type"]
                == "inter_state_matched_environment_link"
            )
            & (
                (pair_frame["state_a"] == state)
                | (pair_frame["state_b"] == state)
            ),
            "distance",
        ]
        all_inter = pair_frame.loc[
            (pair_frame["comparison_type"] != "intra_state")
            & (
                (pair_frame["state_a"] == state)
                | (pair_frame["state_b"] == state)
            ),
            "distance",
        ]
        exemplar = next(
            item.recording for item in statistics if item.recording.state == state
        )
        state_rows.append(
            {
                "state": state,
                "environment_id": exemplar.environment_id,
                "state_token": exemplar.state_token,
                "environment_label": paper_environment_label(
                    exemplar.environment_id
                ),
                "paper_state_label": paper_state_label(state),
                "link_configuration": exemplar.link_configuration,
                "num_intra_pairs": len(intra),
                "mean_intra_distance": intra.mean(),
                "std_intra_distance": intra.std(ddof=1),
                "median_intra_distance": intra.median(),
                "q1_intra_distance": intra.quantile(0.25),
                "q3_intra_distance": intra.quantile(0.75),
                "iqr_intra_distance": (
                    intra.quantile(0.75) - intra.quantile(0.25)
                ),
                "num_matched_inter_pairs": len(matched_inter),
                "mean_inter_distance": matched_inter.mean(),
                "std_inter_distance": matched_inter.std(ddof=1),
                "median_inter_distance": matched_inter.median(),
                "q1_inter_distance": matched_inter.quantile(0.25),
                "q3_inter_distance": matched_inter.quantile(0.75),
                "iqr_inter_distance": (
                    matched_inter.quantile(0.75)
                    - matched_inter.quantile(0.25)
                ),
                "inter_intra_distance_ratio": (
                    matched_inter.mean()
                    / max(float(intra.mean()), EPSILON)
                ),
                "mean_all_context_inter_distance": all_inter.mean(),
            }
        )
    state_frame = pd.DataFrame(state_rows)
    state_frame.to_csv(
        output_dir / "state_distance_statistics.csv",
        index=False,
        float_format="%.8f",
    )

    overall_rows: list[dict[str, Any]] = []
    for comparison_type, values in pair_frame.groupby("comparison_type")[
        "distance"
    ]:
        overall_rows.append(
            {
                "comparison_type": comparison_type,
                "num_pairs": len(values),
                "mean_distance": values.mean(),
                "std_distance": values.std(ddof=1),
                "median_distance": values.median(),
                "q1_distance": values.quantile(0.25),
                "q3_distance": values.quantile(0.75),
                "iqr_distance": (
                    values.quantile(0.75) - values.quantile(0.25)
                ),
            }
        )
    overall_frame = pd.DataFrame(overall_rows)
    overall_frame.to_csv(
        output_dir / "overall_distance_statistics.csv",
        index=False,
        float_format="%.8f",
    )

    intra_values = pair_frame.loc[
        pair_frame["comparison_type"] == "intra_state", "distance"
    ].to_numpy(dtype=float)
    matched_inter_values = pair_frame.loc[
        pair_frame["comparison_type"]
        == "inter_state_matched_environment_link",
        "distance",
    ].to_numpy(dtype=float)
    intra_q1, intra_q3 = np.quantile(intra_values, [0.25, 0.75])
    inter_q1, inter_q3 = np.quantile(
        matched_inter_values, [0.25, 0.75]
    )
    effect_frame = pd.DataFrame(
        [
            {
                "reference_group": "intra_state",
                "comparison_group": (
                    "inter_state_matched_environment_link"
                ),
                "num_intra_pairs": len(intra_values),
                "num_inter_pairs": len(matched_inter_values),
                "intra_median": np.median(intra_values),
                "intra_q1": intra_q1,
                "intra_q3": intra_q3,
                "intra_iqr": intra_q3 - intra_q1,
                "inter_median": np.median(matched_inter_values),
                "inter_q1": inter_q1,
                "inter_q3": inter_q3,
                "inter_iqr": inter_q3 - inter_q1,
                "median_inter_intra_ratio": (
                    np.median(matched_inter_values)
                    / max(float(np.median(intra_values)), EPSILON)
                ),
                "cliffs_delta_inter_vs_intra": cliffs_delta(
                    intra_values, matched_inter_values
                ),
                "dependency_note": (
                    "Descriptive only: pairwise distances share recordings "
                    "and are not mutually independent."
                ),
            }
        ]
    )
    effect_frame.to_csv(
        output_dir / "distance_effect_size_statistics.csv",
        index=False,
        float_format="%.8f",
    )

    if distance_plot in {"both", "paired"}:
        plot_state_level_paired_distances(
            state_frame, output_dir=output_dir, dpi=dpi
        )
    if distance_plot in {"both", "boxplot"}:
        plot_pair_level_distance_boxplot(
            pair_frame, output_dir=output_dir, dpi=dpi
        )

    np.savez_compressed(
        output_dir / "standardized_feature_space.npz",
        sample_id=np.asarray(
            [item.recording.sample_id for item in statistics]
        ),
        state=np.asarray([item.recording.state for item in statistics]),
        feature_mask=usable_feature_mask,
        standardized_features=standardized_features,
        euclidean_distance_matrix=distance_matrix,
        feature_mean=feature_mean,
        feature_std=feature_std,
    )
    return state_frame, pair_frame, overall_frame, effect_frame


def state_inventory_frame(
    recordings: Sequence[Recording],
    statistics: Sequence[RecordingStatistics],
) -> pd.DataFrame:
    packet_lookup = {
        item.recording.sample_id: item.packet_count for item in statistics
    }
    rows: list[dict[str, Any]] = []
    for state, state_records in _group_recordings(recordings).items():
        durations = np.asarray(
            [record.duration_sec for record in state_records], dtype=float
        )
        rows.append(
            {
                "state": state,
                "environment_id": state_records[0].environment_id,
                "environment_label": paper_environment_label(
                    state_records[0].environment_id
                ),
                "state_token": state_records[0].state_token,
                "paper_state_label": paper_state_label(state),
                "link_configuration": state_records[0].link_configuration,
                "num_recordings": len(state_records),
                "sample_ids": " | ".join(
                    record.sample_id for record in state_records
                ),
                "trial_ids": " | ".join(
                    record.trial_id for record in state_records
                ),
                "mean_duration_sec": np.nanmean(durations),
                "min_duration_sec": np.nanmin(durations),
                "max_duration_sec": np.nanmax(durations),
                "total_csi_packets": sum(
                    packet_lookup[record.sample_id] for record in state_records
                ),
                "csi_paths": " | ".join(
                    record.csi_path_relative for record in state_records
                ),
            }
        )
    return pd.DataFrame(rows)


def _group_recordings(
    recordings: Sequence[Recording],
) -> dict[str, list[Recording]]:
    grouped: dict[str, list[Recording]] = defaultdict(list)
    for recording in recordings:
        grouped[recording.state].append(recording)
    return dict(sorted(grouped.items()))


def build_summary(
    *,
    state_inventory: pd.DataFrame,
    correlation_statistics: pd.DataFrame,
    distance_statistics: pd.DataFrame,
) -> pd.DataFrame:
    summary = state_inventory[
        [
            "state",
            "environment_id",
            "environment_label",
            "state_token",
            "paper_state_label",
            "link_configuration",
            "num_recordings",
            "mean_duration_sec",
            "min_duration_sec",
            "max_duration_sec",
        ]
    ].merge(
        correlation_statistics[
            [
                "state",
                "mean_intra_corr",
                "std_intra_corr",
                "num_recording_pairs",
            ]
        ],
        on="state",
        how="left",
        validate="one_to_one",
    )
    summary = summary.merge(
        distance_statistics[
            [
                "state",
                "mean_intra_distance",
                "std_intra_distance",
                "median_intra_distance",
                "iqr_intra_distance",
                "num_intra_pairs",
                "mean_inter_distance",
                "std_inter_distance",
                "median_inter_distance",
                "iqr_inter_distance",
                "num_matched_inter_pairs",
                "inter_intra_distance_ratio",
                "mean_all_context_inter_distance",
            ]
        ],
        on="state",
        how="left",
        validate="one_to_one",
    )
    summary["intra_distance"] = summary["mean_intra_distance"]
    summary["inter_distance"] = summary["mean_inter_distance"]
    required_first = [
        "state",
        "num_recordings",
        "mean_intra_corr",
        "intra_distance",
        "inter_distance",
    ]
    remaining = [
        column for column in summary.columns if column not in required_first
    ]
    return summary[required_first + remaining]


def markdown_table(
    frame: pd.DataFrame,
    columns: Sequence[tuple[str, str]],
    *,
    float_digits: int = 3,
) -> str:
    headers = [label for _, label in columns]
    lines = [
        "| " + " | ".join(headers) + " |",
        "| " + " | ".join("---" for _ in headers) + " |",
    ]
    for _, row in frame.iterrows():
        cells: list[str] = []
        for column, _ in columns:
            value = row[column]
            if pd.isna(value):
                text = "NA"
            elif isinstance(value, (float, np.floating)):
                text = f"{float(value):.{float_digits}f}"
            else:
                text = str(value)
            cells.append(text.replace("|", "\\|"))
        lines.append("| " + " | ".join(cells) + " |")
    return "\n".join(lines)


def load_exclusion_ledger(dataset_root: Path) -> pd.DataFrame:
    ledger_path = dataset_root / "manifests" / "excluded_samples.csv"
    if not ledger_path.is_file():
        return pd.DataFrame(
            columns=[
                "sample_id",
                "reason",
                "release_action",
                "trial_id",
            ]
        )
    return pd.read_csv(ledger_path, dtype=str, keep_default_na=False)


def clean_living_room_explanation(
    state_inventory: pd.DataFrame,
    exclusion_ledger: pd.DataFrame,
) -> str:
    target_state = "E1__clean_none"
    rows = state_inventory.loc[state_inventory["state"] == target_state]
    if rows.empty:
        return (
            "No Living room — Clean state was discovered in the active "
            "release."
        )
    row = rows.iloc[0]
    observed_trials = [
        value.strip()
        for value in str(row["trial_ids"]).split("|")
        if value.strip()
    ]
    sample_column = exclusion_ledger.get(
        "sample_id", pd.Series(dtype=str)
    ).astype(str)
    clean_exclusions = exclusion_ledger.loc[
        sample_column.str.contains("_E1_clean_none_", regex=False)
    ]
    exclusion_text = (
        "no Living room — Clean entry appears in the exclusion ledger"
        if clean_exclusions.empty
        else (
            "the exclusion ledger contains "
            f"{len(clean_exclusions)} Living room — Clean entry/entries"
        )
    )
    return (
        "The active release contains "
        + ", ".join(observed_trials)
        + f" for Living room — Clean ({exclusion_text}). "
        "There is no active R002 metadata/CSI pair, so the analysis reports "
        "the four released independent recordings and does not impute or "
        "duplicate a fifth recording."
    )


def write_normalization_audit(
    *,
    output_dir: Path,
    num_recordings: int,
    low_energy_ratio: float,
) -> None:
    audit = f"""# Normalization and feature-standardization audit

## Scope

This audit follows the implemented code path used for the
{num_recordings} active unoccupied recordings. It does not infer processing
from plot labels.

## Recording-level CSI amplitude scaling

For each decoded CSI file, amplitude is computed as `|H|`. After applying the
existing low-energy subcarrier interpolation rule (threshold:
{low_energy_ratio:.3f} times the recording median energy), the code calculates
the packet/RX mean curve and the packet/RX temporal standard-deviation curve.
One scalar scale factor is then fitted to that recording:

`amplitude_scale = mean(mean_amplitude_curve)`

Both curves are divided by this same recording-specific scalar. Therefore:

1. **Per-recording Min–Max normalization: no.** Neither a minimum nor a range
   is used.
2. **Per-recording Z-score normalization: no.** The curve is not centered and
   is not divided by its own subcarrier-wise standard deviation.
3. **Per-recording scale division: yes, by the recording's own mean
   amplitude.** The maximum and median are not used.

Paper figures describe the resulting dimensionless quantity as **Relative CSI
amplitude**. No additional smoothing, curve alignment, or outlier suppression
is applied for visualization.

## Distance-feature standardization

Each recording feature vector concatenates its relative mean-amplitude curve
and relative amplitude-variance curve. The code stacks all recording vectors
in the complete unoccupied-recording comparison set, then fits one mean and
one population standard deviation (`ddof=0`) for every feature dimension
across those {num_recordings} recordings. Constant dimensions are excluded.
Every recording is transformed with the same fitted feature-wise parameters
before Euclidean distance is calculated.

This is **comparison-set-level feature-wise standardization**, not a separate
Z-score fitted to each recording. The primary inter-state subset is then
restricted to pairs sharing both `environment_id` and `link_configuration`.

## Compliance decision

The implemented distance standardization already satisfies the requested
scope. No legacy result migration or distance-method change is required;
existing distance definitions and feature extraction are retained.
"""
    (output_dir / "normalization_audit.md").write_text(
        audit, encoding="utf-8"
    )


def state_code_frame() -> pd.DataFrame:
    """Return the fixed publication codebook in paper display order."""

    rows: list[dict[str, str]] = []
    for state in STATE_ORDER:
        environment_id = state.split("__", 1)[0]
        rows.append(
            {
                "Code": paper_state_code(state),
                "Environment": paper_environment_label(environment_id),
                "State": paper_state_short_label(state),
            }
        )
    return pd.DataFrame(rows)


def v4_caption_code_key() -> str:
    """Return the complete state-code key for the suggested figure caption."""

    entries = [
        (
            f"{row.Code} = {row.Environment} — {row.State}"
        )
        for row in state_code_frame().itertuples(index=False)
    ]
    return textwrap.fill(
        "State codes: " + "; ".join(entries) + ".",
        width=88,
    )


def v4_state_code_report_section() -> str:
    code_table = markdown_table(
        state_code_frame(),
        (
            ("Code", "Code"),
            ("Environment", "Environment"),
            ("State", "State"),
        ),
    )
    return f"""<!-- v4-state-code-mapping:start -->
## State codes used in the v4 main figure

{code_table}

### Suggested v4 figure caption

**Figure X | Repeatability and separability of unoccupied environmental
states.** (a) State-level intra-state and matched-context inter-state
distances. (b) Overall pairwise distance distributions shown as boxplots
without a raw scatter overlay. (c) Nearest-state separation ratios, with the
dashed reference line indicating a ratio of 1.

{v4_caption_code_key()}

Panel (a) displays each code together with its short state name, while panel
(c) uses codes alone because its plotting area is narrower. Environment
headings remain visible in both panels.

Panel (b) omits raw pairwise scatter points for visual clarity because
pairwise distances share recordings and should not be interpreted as fully
independent observations. The boxplot is calculated from all available
intra-state and matched-context inter-state distances, retains the unchanged
quartile and whisker definitions, and displays outliers as hollow circles.
<!-- v4-state-code-mapping:end -->
"""


def v5_caption_text(
    pair_frame: pd.DataFrame,
    nearest_frame: pd.DataFrame,
) -> str:
    """Return a paper-ready caption derived from existing result tables."""

    ordered_nearest = ordered_state_frame(nearest_frame)
    intra_pairs = int(
        (
            pair_frame["comparison_type"]
            == "intra_state"
        ).sum()
    )
    inter_pairs = int(
        (
            pair_frame["comparison_type"]
            == "inter_state_matched_environment_link"
        ).sum()
    )

    def ratio_description(state: str) -> str:
        match = ordered_nearest.loc[
            ordered_nearest["state"] == state
        ]
        if len(match) != 1:
            raise ValueError(
                f"Expected one nearest-state result for {state!r}"
            )
        row = match.iloc[0]
        nearest_state = str(row["nearest_other_state"])
        environment_id = state.split("__", 1)[0]
        return (
            f"{paper_state_code(state)} "
            f"({paper_environment_label(environment_id)} — "
            f"{STATE_V5_SHORT_LABELS[state]}; "
            f"nearest: {paper_state_code(nearest_state)} "
            f"{STATE_V5_SHORT_LABELS[nearest_state]}; "
            f"ratio = {float(row['nearest_state_ratio']):.2f})"
        )

    l3_description = ratio_description(
        "E1__heater_on_heater_stable"
    )
    k1_description = ratio_description("E4__clean_none")
    return f"""# Figure unoccupied repeatability main v5 — caption draft

**Figure X | Repeatability and separation of unoccupied environmental
states.** **(a)** State-level repeatability and separation. Each point shows
the mean pairwise Euclidean distance in the common standardized feature space;
blue circles represent within-state distances and orange squares represent
between-state distances restricted to recordings with the same environment
and WiFi link. Error bars show mean ± SD, and light-gray connectors join the
two state-level means. **(b)** Overall distance distributions for all
within-state pairs (n = {intra_pairs}) and matched-context between-state pairs
(n = {inter_pairs}). Raw pairwise scatter points are omitted for clarity;
these pairwise distances share recordings and are therefore not interpreted
as fully independent observations. **(c)** Nearest-state separation for each
state, defined as the mean distance to its nearest alternative state divided
by its mean within-state distance. The dashed line at ratio = 1 is the
equal-distance threshold. A ratio > 1 means that the nearest alternative state
is farther away than the within-state variation; a ratio < 1 means that the
nearest alternative state is closer than the average within-state variation.
Blue points (“Separated”) indicate ratio ≥ 1, whereas orange points
(“Potential overlap”) indicate ratio < 1.

The below-threshold results for {l3_description} and {k1_description} identify
conditions whose nearest alternative is comparable to or closer than their
average within-state variation. They should be interpreted conservatively as
potentially confusable conditions, not as direct classification error rates
or evidence that the underlying recordings are invalid.
"""


def write_v5_caption(
    *,
    output_dir: Path,
    pair_frame: pd.DataFrame,
    nearest_frame: pd.DataFrame,
) -> Path:
    caption_path = (
        output_dir
        / "figure_unoccupied_repeatability_main_v5_caption.md"
    )
    caption_path.write_text(
        v5_caption_text(pair_frame, nearest_frame),
        encoding="utf-8",
    )
    return caption_path


def v6_caption_text(
    pair_frame: pd.DataFrame,
) -> str:
    """Return the concise explanatory caption for the v6 main figure."""

    intra_pairs = int(
        (
            pair_frame["comparison_type"]
            == "intra_state"
        ).sum()
    )
    inter_pairs = int(
        (
            pair_frame["comparison_type"]
            == "inter_state_matched_environment_link"
        ).sum()
    )
    code_table = markdown_table(
        state_code_frame(),
        (
            ("Code", "Code"),
            ("Environment", "Environment"),
            ("State", "State"),
        ),
    )
    return f"""# Figure unoccupied repeatability main v6 — caption draft

**Figure X | Repeatability and separation of unoccupied environmental
states.** **(a)** State-level within-state and matched-context between-state
pairwise distances. Symbols show the mean and error bars show ±1 SD; blue
circles denote within-state distances, orange squares denote between-state
distances, and light-gray lines connect the two means for each state.
**(b)** Overall pairwise-distance distributions for within-state pairs
(n = {intra_pairs}) and between-state pairs matched by environment and WiFi
link (n = {inter_pairs}). All distances are retained in the boxplot
calculation; hollow circles indicate values beyond the whiskers. Raw
pairwise scatter is omitted because pairs can share recordings and should not
be interpreted as fully independent observations. **(c)** Nearest-state
separation ratio, defined as the mean distance to the nearest alternative
state divided by the mean within-state distance. A ratio > 1 indicates that
the nearest alternative is farther away than the average within-state
variation, whereas a ratio < 1 indicates potential overlap because the
nearest alternative is closer than that variation. The gray dashed line at
`x = 1` marks equal nearest-alternative and within-state distances. Blue
points indicate ratios ≥ 1; orange points indicate ratios < 1.

## State-code mapping

{code_table}
"""


def write_v6_caption(
    *,
    output_dir: Path,
    pair_frame: pd.DataFrame,
) -> Path:
    caption_path = (
        output_dir
        / "figure_unoccupied_repeatability_main_v6_caption.md"
    )
    caption_path.write_text(
        v6_caption_text(pair_frame),
        encoding="utf-8",
    )
    return caption_path


def _flagged_state_codes(nearest_state_frame: pd.DataFrame) -> str:
    """Return paper codes of states below the separation threshold."""

    flagged = ordered_state_frame(
        nearest_state_frame.loc[
            nearest_state_frame["nearest_state_ratio"] < 1.0
        ]
    )
    if flagged.empty:
        return "none"
    return ", ".join(
        paper_state_code(state) for state in flagged["state"].astype(str)
    )


def v7_caption_text(
    pair_frame: pd.DataFrame,
    nearest_state_frame: pd.DataFrame,
) -> str:
    """Return the concise explanatory caption for the v7 main figure."""

    intra_pairs = int(
        (
            pair_frame["comparison_type"]
            == "intra_state"
        ).sum()
    )
    inter_pairs = int(
        (
            pair_frame["comparison_type"]
            == "inter_state_matched_environment_link"
        ).sum()
    )
    code_table = markdown_table(
        state_code_frame(),
        (
            ("Code", "Code"),
            ("Environment", "Environment"),
            ("State", "State"),
        ),
    )
    flagged_codes = _flagged_state_codes(nearest_state_frame)
    return f"""# Figure unoccupied repeatability main v7 — caption draft

**Figure X | Repeatability and separation of unoccupied environmental
states.** **(a)** State-level within-state and matched-context between-state
pairwise distances, in standardized feature units. Symbols show the mean,
faint error bars show ±1 SD, and gray lines connect the two means for each
state; blue circles denote within-state distances and orange squares denote
between-state distances. **(b)** Overall pairwise-distance distributions for
within-state pairs (n = {intra_pairs}) and between-state pairs matched by
environment and WiFi link (n = {inter_pairs}). All distances are retained in
the boxplot calculation; hollow circles indicate values beyond the whiskers.
Raw pairwise scatter is omitted because pairs can share recordings and should
not be interpreted as fully independent observations. **(c)** Separation
ratio, defined as the mean distance to the nearest alternative state divided
by the mean within-state distance. A ratio > 1 indicates that the nearest
alternative is farther away than the average within-state variation, whereas
a ratio < 1 indicates potential overlap because the nearest alternative is
closer than that variation. The gray dashed line at `x = 1` marks equal
nearest-alternative and within-state distances. Gray points indicate ratios
≥ 1; red points indicate ratios < 1 ({flagged_codes}).

## State-code mapping

{code_table}
"""


def write_v7_caption(
    *,
    output_dir: Path,
    pair_frame: pd.DataFrame,
    nearest_state_frame: pd.DataFrame,
) -> Path:
    caption_path = (
        output_dir
        / "figure_unoccupied_repeatability_main_v7_caption.md"
    )
    caption_path.write_text(
        v7_caption_text(pair_frame, nearest_state_frame),
        encoding="utf-8",
    )
    return caption_path


def v8_caption_text(
    pair_frame: pd.DataFrame,
    nearest_state_frame: pd.DataFrame,
) -> str:
    """Return the concise explanatory caption for the v8 main figure."""

    intra_pairs = int(
        (
            pair_frame["comparison_type"]
            == "intra_state"
        ).sum()
    )
    inter_pairs = int(
        (
            pair_frame["comparison_type"]
            == "inter_state_matched_environment_link"
        ).sum()
    )
    code_table = markdown_table(
        state_code_frame(),
        (
            ("Code", "Code"),
            ("Environment", "Environment"),
            ("State", "State"),
        ),
    )
    flagged_codes = _flagged_state_codes(nearest_state_frame)
    return f"""# Figure unoccupied repeatability main v8 — caption draft

**Figure X | Repeatability and separation of unoccupied environmental
states.** All quantities are Euclidean distances between recording-level
feature vectors in the standardized feature space, computed over **distance
pairs**: within-state pairs (n = {intra_pairs}) connect two recordings of the
same environmental state, and between-state pairs (n = {inter_pairs}) connect
recordings of two different states measured under the same environment and
WiFi link configuration, so that environmental and hardware context is
matched. **(a)** For each state, symbols show the mean of the corresponding
distance pairs (blue circles: within-state; orange squares: between-state),
gray lines connect the two means, and error bars show ±1 SD of the
underlying pairwise distances; the error bars are descriptive spread of the
pair-level values, not confidence intervals on the mean. The right-hand
column reports the **nearest-state separation ratio**: for each state, the
mean distance to the nearest alternative state under matched environment and
link, divided by the mean within-state distance. A ratio > 1 indicates that
the nearest alternative is farther away than the average within-state
variation, whereas a ratio < 1 (bold red; {flagged_codes}) indicates
potential overlap with the nearest alternative. **(b)** Distributions of all
within-state and between-state distance pairs. All distances are retained in
the boxplot calculation; hollow circles indicate values beyond the whiskers.
Raw pairwise scatter is omitted because pairs can share recordings and
should not be interpreted as fully independent observations.

## State-code mapping

{code_table}
"""


def write_v8_caption(
    *,
    output_dir: Path,
    pair_frame: pd.DataFrame,
    nearest_state_frame: pd.DataFrame,
) -> Path:
    caption_path = (
        output_dir
        / "figure_unoccupied_repeatability_main_v8_caption.md"
    )
    caption_path.write_text(
        v8_caption_text(pair_frame, nearest_state_frame),
        encoding="utf-8",
    )
    return caption_path


def figure7_caption_text(
    pair_frame: pd.DataFrame,
    nearest_frame: pd.DataFrame,
    layout_variant: str = "r3",
    panel_b_variant: str = "default",
    palette_variant: str = "current",
) -> str:
    """Return a condition-consistent caption for the renumbered figure."""

    within, between = _figure7_distance_arrays(pair_frame)
    flagged_codes = _flagged_state_codes(nearest_frame)
    if layout_variant == "facets":
        mapping = "; ".join(
            paper_state_code_label(state).replace("  ", " ") for state in STATE_ORDER
        )
        return f"""# Figure 7 condition-distance caption draft — refined room facets

**Figure 7 | Repeatability and exploratory separation of unoccupied-environment
conditions.** All distances are Euclidean distances between recording-level
feature vectors in the standardized feature space. **a,** Room-faceted condition
summaries. Filled circles show mean within-condition distance, hollow diamonds
show mean distance to the nearest alternative condition, and hollow circles show
mean distance across all matched between-condition pairs, not only the nearest
alternative. Pale vertical lines show the original within/all-between means
±1 SD of pairwise distances; these are not confidence intervals. Nearest-condition
means have no SD. The common vertical axis is logarithmic: equal vertical
intervals indicate equal multiplicative changes. Original means and SD endpoints
are positioned on this axis without recomputing statistics in log space. Small
horizontal offsets separate the three markers; they encode no additional data.
Room widths are proportional to their condition counts (4:6:3), so condition
spacing is constant. The numbers below the condition codes give nearest-condition
distance ÷ within-condition distance. Ratios below 1 ({flagged_codes}) are emphasized;
their diamonds also fall below the corresponding filled circles. **b,** Exact
right-continuous empirical cumulative distributions for the pooled within-condition
(n = {len(within)}) and matched-context between-condition (n = {len(between)})
distances, on a linear distance axis. Both solid curves are identified by direct
labels. There is no smoothing, interpolation, binning, or observation rug.

Condition codes (in unchanged room and condition order): {mapping}.

Exact condition means and SDs remain available in the accompanying frozen
`source_data/state_distance_statistics.csv`; nearest-condition means and ratios
are in `source_data/nearest_state_separation.csv`.
"""
    if layout_variant == "a":
        return f"""# Figure 7 condition-distance caption draft — variant A

**Figure 7 | Descriptive repeatability and exploratory separation of AXHome
unoccupied-environment conditions.** All distances are Euclidean distances
between recording-level feature vectors in the standardized feature space.
**a,** Condition summaries on a broken distance axis. Filled blue circles
show mean within-condition distance, dark-gray vertical strokes show mean
distance to the nearest alternative condition, and hollow orange circles show
mean distance across all matched alternative conditions. Blue and orange
horizontal bars are ±1 SD of the underlying pairwise distances, not confidence
intervals on the mean; the nearest-condition summary has no SD. The right-hand
ratio is the gray-stroke value divided by the filled-circle value. Ratios below
1 ({flagged_codes}; L3 Heater stable and K1 Clean) are muted red. The axis
break explicitly omits 32–40 standardized units; values on either side retain
their linear scales, but distances must not be compared across the break by
visual length alone. **b,** Pooled within-condition (n = {len(within)}) and
matched-context between-condition (n = {len(between)}) distributions on the
same broken axis. Boxplots use every source distance in their calculations;
individual outlier symbols are omitted because pairwise distances share source
recordings.
"""
    if layout_variant in {"b", "rowwise"}:
        color_names = {
            "current": ("blue", "neutral-gray", "orange", "muted red"),
            "p1": ("navy", "smoky-teal", "violet", "dark wine"),
            "p2": ("charcoal", "slate", "ochre-brown", "deep navy"),
            "p3": ("aubergine", "forest-green", "brick", "dark navy"),
        }
        within_name, nearest_name, between_name, warning_name = color_names[palette_variant]
        if panel_b_variant == "v1":
            panel_b_description = f"""Empirical cumulative distributions of
all pooled within-condition (n = {len(within)}) and matched-context
between-condition (n = {len(between)}) distances. Curves are exact
right-continuous empirical steps without smoothing or interpolation. The two
rug bands place one tick at every observed distance, making the different
sample densities and therefore the different step textures explicit."""
        elif panel_b_variant == "v2":
            panel_b_description = f"""Separate empirical histograms for all
pooled within-condition (n = {len(within)}) and matched-context
between-condition (n = {len(between)}) distances. Both use the same fixed
2-standardized-unit bins and plot the within-group fraction per bin on a
common vertical scale. Every source distance enters exactly one bin; no
smoothing, interpolation, or density estimation is applied."""
        elif panel_b_variant == "default":
            panel_b_description = f"""Empirical cumulative distributions of
all pooled within-condition (n = {len(within)}) and matched-context
between-condition (n = {len(between)}) distances. Both curves are exact
right-continuous empirical steps, drawn as 1.0-pt solid lines and identified
by direct labels at their right ends; no smoothing or interpolation is
applied."""
        else:
            raise ValueError(
                f"Unsupported Figure 7 panel-b variant {panel_b_variant!r}"
            )
        layout_name = "ratio-first rowwise layout" if layout_variant == "rowwise" else "variant B"
        ratio_geometry = "central linear-scale plot" if layout_variant == "rowwise" else "adjacent log2 forest"
        ratio_cue = " Filled diamonds and bold condition names additionally identify ratios below 1." if layout_variant == "rowwise" else ""
        return f"""# Figure 7 condition-distance caption draft — {layout_name}

**Figure 7 | Descriptive repeatability and exploratory separation of AXHome
unoccupied-environment conditions.** All distances are Euclidean distances
between recording-level feature vectors in the standardized feature space.
**a,** Exact condition summaries presented as a compact table rather than on
a sparse common position scale. The {within_name} column gives mean within-condition
distance ±1 SD, the {nearest_name} column gives mean distance to the nearest
alternative condition, and the {between_name} column gives mean distance ±1 SD across
all matched between-condition pairs. Thus, “All between” is the mean over all
matched alternative-condition pairings, not the nearest alternative alone.
SDs describe the underlying pairwise distances and are not confidence
intervals on the mean. The {ratio_geometry} plots the separation ratio,
defined as nearest-condition distance ÷ within-condition distance; its values
are printed in the fixed right-aligned column, and the dashed line denotes
equality. Ratios below 1 are shown in {warning_name}: L3 Heater stable and K1
Clean.{ratio_cue} **b,** {panel_b_description}
"""
    if layout_variant == "c":
        code_mapping = "; ".join(
            paper_state_code_label(state).replace("  ", " ")
            for state in STATE_ORDER
        )
        return f"""# Figure 7 condition-distance caption draft — variant C

**Figure 7 | Descriptive repeatability and exploratory separation of AXHome
unoccupied-environment conditions.** All distances are Euclidean distances
between recording-level feature vectors in the standardized feature space.
**a,** Room-faceted condition profiles on a logarithmic distance axis. For
each condition code, the filled blue circle and blue vertical bar show mean
within-condition distance ±1 SD, the dark-gray diamond shows mean distance to
the nearest alternative condition, and the hollow orange circle and orange
vertical bar show mean distance ±1 SD across all matched alternative
conditions. SDs describe pairwise-distance dispersion, not confidence
intervals. The thin segment from the blue circle to the diamond encodes the
nearest-condition/within-condition ratio, which is printed beside it; downward
muted-red segments identify ratios below 1 ({flagged_codes}; L3 Heater stable
and K1 Clean). Small horizontal marker offsets prevent overlap and do not
encode another variable. **b,** Pooled within-condition (n = {len(within)})
and matched-context between-condition (n = {len(between)}) distributions on a
logarithmic axis. Boxplots use every source distance in their calculations;
individual outlier symbols are omitted because pairwise distances share source
recordings. Condition codes, in fixed plotting order, are: {code_mapping}.
"""
    if layout_variant != "r3":
        raise ValueError(f"Unsupported Figure 7 layout variant {layout_variant!r}")
    return f"""# Figure 7 condition-distance caption draft

**Figure 7 | Descriptive repeatability and exploratory separation of AXHome
unoccupied-environment conditions.** All distances are Euclidean distances
between recording-level feature vectors in the standardized feature space.
**a,** Mean within-condition and matched-context between-condition pairwise
distances for each environment-condition combination. Filled blue circles
denote mean within-condition distance and hollow orange circles denote mean
between-condition distance across all matched alternative conditions; their
error bars show ±1 SD of the underlying pairwise distances rather than
confidence intervals on the mean. A dark-gray vertical stroke without an
error bar marks the mean distance to the nearest alternative condition. The
right-hand ratio is the gray-stroke value divided by the filled-circle value,
not the hollow-circle value. Ratios below 1 ({flagged_codes}) are shown in
muted red; these are L3 Heater stable and K1 Clean. **b,** Pooled
within-condition (n = {len(within)}) and matched-context between-condition
(n = {len(between)}) distance distributions, drawn horizontally on the same
distance axis as panel a. Boxplots retain all distances in their calculations;
individual outlier symbols are omitted because pairwise distances share source
recordings.
"""


def write_figure7_caption(
    *,
    output_dir: Path,
    pair_frame: pd.DataFrame,
    nearest_frame: pd.DataFrame,
    layout_variant: str = "r3",
    panel_b_variant: str = "default",
    palette_variant: str = "current",
) -> Path:
    caption_path = output_dir / f"{FIGURE7_STEM}_caption.md"
    caption_path.write_text(
        figure7_caption_text(
            pair_frame,
            nearest_frame,
            layout_variant=layout_variant,
            panel_b_variant=panel_b_variant,
            palette_variant=palette_variant,
        ),
        encoding="utf-8",
    )
    return caption_path


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _portable_path(path: Path, project_root: Path) -> str:
    try:
        return path.resolve().relative_to(project_root.resolve()).as_posix()
    except ValueError:
        return str(path.resolve())


def _git_text(project_root: Path, *arguments: str) -> str:
    result = subprocess.run(
        ["git", "-C", str(project_root), *arguments],
        check=False,
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
    )
    return result.stdout.strip() if result.returncode == 0 else "unavailable"


def copy_figure7_source_data(
    *,
    source_results_dir: Path,
    output_dir: Path,
) -> list[dict[str, Any]]:
    """Copy the exact source CSV bytes into the figure delivery bundle."""

    source_data_dir = output_dir / "source_data"
    source_data_dir.mkdir(parents=True, exist_ok=True)
    records: list[dict[str, Any]] = []
    for filename in FIGURE7_SOURCE_FILENAMES:
        source_path = source_results_dir / filename
        copied_path = source_data_dir / filename
        shutil.copy2(source_path, copied_path)
        source_hash = _sha256_file(source_path)
        copied_hash = _sha256_file(copied_path)
        if source_hash != copied_hash:
            raise ValueError(f"Source-data copy hash mismatch for {filename}")
        records.append(
            {
                "source_path": source_path,
                "copied_path": copied_path,
                "sha256": source_hash,
                "byte_identical_copy": True,
            }
        )
    return records


def write_figure7_generation_record(
    *,
    project_root: Path,
    source_results_dir: Path,
    output_dir: Path,
    pair_frame: pd.DataFrame,
    state_distance_frame: pd.DataFrame,
    nearest_frame: pd.DataFrame,
    copied_sources: Sequence[Mapping[str, Any]],
    dpi: int,
    layout_variant: str = "r3",
    panel_b_variant: str = "default",
    palette_variant: str = "current",
) -> Path:
    """Write a reproducible generation record for the delivery bundle."""

    import scipy

    within, between = _figure7_distance_arrays(pair_frame)
    ordered_distances = ordered_state_frame(state_distance_frame)
    ordered_ratios = ordered_state_frame(nearest_frame)
    font_family, font_path = resolve_publication_font()
    layout_metadata = figure7_layout_metadata(layout_variant)
    panel_b_metadata = {
        "default": {
            "name": "finalized directly labelled exact ECDF",
            "empirical_distribution": "right-continuous ECDF; two solid 1.0-pt lines with miter joins, butt caps, and direct end labels; no legend, rug, smoothing, or interpolation",
        },
        "v1": {
            "name": "exact ECDF with observation rugs",
            "empirical_distribution": "right-continuous ECDF plus one rug tick per source distance; no smoothing or interpolation",
        },
        "v2": {
            "name": "stacked fixed-bin empirical histograms",
            "empirical_distribution": "within-group fractions in shared 2-unit bins; every source distance assigned once; no smoothing or interpolation",
        },
    }
    if panel_b_variant not in panel_b_metadata:
        raise ValueError(
            f"Unsupported Figure 7 panel-b variant {panel_b_variant!r}"
        )
    if palette_variant not in FIGURE7_PALETTES:
        raise ValueError(
            f"Unsupported Figure 7 palette variant {palette_variant!r}"
        )
    palette = FIGURE7_PALETTES[palette_variant]
    script_path = Path(__file__).resolve()
    artifact_paths = [
        output_dir / f"{FIGURE7_STEM}.{suffix}"
        for suffix in ("png", "svg", "pdf")
    ] + [output_dir / f"{FIGURE7_STEM}_caption.md"]
    record = {
        "schema_version": 1,
        "figure": {
            "current_number": "Figure 7",
            "former_manuscript_number": "Figure 9",
            "revision": (
                f"layout-variant-{layout_variant}-palette-{palette_variant}"
                if layout_variant in {"b", "facets", "rowwise"} and palette_variant != "current"
                else (
                    f"layout-variant-b-panel-b-{panel_b_variant}"
                    if layout_variant == "b" and panel_b_variant != "default"
                    else (
                        "r3"
                        if layout_variant == "r3"
                        else f"layout-variant-{layout_variant}"
                    )
                )
            ),
            "layout_variant": layout_variant,
            "panel_b_variant": panel_b_variant,
            "palette_variant": palette_variant,
            "stem": FIGURE7_STEM,
            "manuscript_modified": False,
        },
        "generated_at": datetime.now().astimezone().isoformat(
            timespec="seconds"
        ),
        "provenance_status": "sufficient",
        "generator": {
            "script": _portable_path(script_path, project_root),
            "script_sha256": _sha256_file(script_path),
            "argv": sys.argv,
            "python_executable": sys.executable,
            "python_version": sys.version.split()[0],
            "libraries": {
                "matplotlib": matplotlib.__version__,
                "numpy": np.__version__,
                "pandas": pd.__version__,
                "scipy": scipy.__version__,
            },
        },
        "git": {
            "branch": _git_text(project_root, "branch", "--show-current"),
            "head": _git_text(project_root, "rev-parse", "HEAD"),
            "dirty": bool(_git_text(project_root, "status", "--porcelain")),
        },
        "source_results_dir": _portable_path(
            source_results_dir,
            project_root,
        ),
        "source_files": [
            {
                "source_path": _portable_path(
                    Path(item["source_path"]),
                    project_root,
                ),
                "copied_path": _portable_path(
                    Path(item["copied_path"]),
                    project_root,
                ),
                "sha256": item["sha256"],
                "byte_identical_copy": item["byte_identical_copy"],
            }
            for item in copied_sources
        ],
        "numeric_integrity": {
            "environment_condition_rows": int(len(ordered_distances)),
            "within_condition_pair_count": int(len(within)),
            "between_condition_pair_count": int(len(between)),
            "nearest_condition_distances": {
                paper_state_code(str(row.state)): float(
                    row.mean_nearest_inter_distance
                )
                for row in ordered_ratios.itertuples(index=False)
            },
            "nearest_condition_ratios": {
                paper_state_code(str(row.state)): float(
                    row.nearest_state_ratio
                )
                for row in ordered_ratios.itertuples(index=False)
            },
            "values_recomputed": False,
            "source_csv_bytes_preserved": True,
        },
        "plotting_parameters": {
            **layout_metadata,
            "panel_b_treatment": (
                panel_b_metadata[panel_b_variant]
                if layout_variant in {"b", "facets", "rowwise"}
                else {"name": "pooled boxplots; all source values retained"}
            ),
            "png_dpi": max(int(dpi), FIGURE_DPI),
            "colors": (
                dict(palette)
                if layout_variant in {"b", "facets", "rowwise"}
                else {
                    "within_condition": INTRA_COLOR,
                    "between_condition": INTER_COLOR,
                    "within_condition_box_fill": INTRA_LIGHT_COLOR,
                    "between_condition_box_fill": INTER_LIGHT_COLOR,
                    "flagged_ratio": RATIO_FLAG_COLOR,
                }
            ),
            "three_required_condition_quantities": {
                "within_condition": "mean and source SD",
                "nearest_condition": "mean; no SD in source",
                "all_between": "mean and source SD",
            },
            "ratio_definition": "mean_nearest_inter_distance / mean_intra_distance",
            "pooled_pair_counts_asserted": [126, 585],
            "pooled_outlier_symbols": (
                "omitted for boxplots; all values retained in computation"
                if layout_variant in {"r3", "a", "c"}
                else "not applicable; ECDF retains and displays every value"
            ),
            "font_family": font_family,
            "font_path": font_path,
        },
        "labels": {
            paper_state_code(state): paper_state_code_label(state)
            for state in STATE_ORDER
        },
        "artifacts": [
            {
                "path": _portable_path(path, project_root),
                "sha256": _sha256_file(path),
            }
            for path in artifact_paths
        ],
    }
    record_path = output_dir / f"{FIGURE7_STEM}_generation_record.json"
    record_path.write_text(
        json.dumps(record, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    return record_path


def v5_report_section() -> str:
    return """<!-- v5-main-figure:start -->
## v5 main-figure presentation

The v5 main figure removes the mean inter/mean intra ratio column from panel
(a), because that quantity differs from the nearest-state separation ratio in
panel (c). Panel (a) now shows only within-state and between-state mean
distances; its error bars are descriptive mean ± SD in the common standardized
feature space.

Panel (b) uses the reader-facing labels “Within state” and “Between states.”
It retains every distance in the boxplot calculation but omits the raw
pairwise scatter overlay. Pairwise distances share recordings and are not
treated as fully independent observations.

For panel (c), the nearest-state separation ratio is:

`mean distance to the nearest alternative state / mean within-state distance`

- **ratio > 1:** the nearest alternative state is farther away than the
  within-state variation;
- **ratio < 1:** the nearest alternative state is closer than the average
  within-state variation.

The dashed line at `x = 1` is labelled **Equal-distance threshold**. Blue
points are labelled **Separated** (`ratio ≥ 1`), and orange points are labelled
**Potential overlap** (`ratio < 1`). L3 and K1 remain below the threshold and
are interpreted conservatively as potentially confusable conditions rather
than classification error rates.

The complete paper-ready draft is available in
[`figure_unoccupied_repeatability_main_v5_caption.md`](figure_unoccupied_repeatability_main_v5_caption.md).
<!-- v5-main-figure:end -->
"""


def update_report_for_v5(report_path: Path) -> None:
    """Insert the v5 interpretation and current figure links idempotently."""

    if report_path.is_file():
        report = report_path.read_text(encoding="utf-8")
    else:
        report = "# Unoccupied environment validation\n\n"
    section = v5_report_section()
    start_marker = "<!-- v5-main-figure:start -->"
    end_marker = "<!-- v5-main-figure:end -->"
    if start_marker in report and end_marker in report:
        start_index = report.index(start_marker)
        end_index = report.index(end_marker) + len(end_marker)
        report = report[:start_index] + section.rstrip() + report[end_index:]
    elif "<!-- v4-state-code-mapping:start -->" in report:
        report = report.replace(
            "<!-- v4-state-code-mapping:start -->",
            section + "\n<!-- v4-state-code-mapping:start -->",
            1,
        )
    elif "<!-- nearest-state-separation:start -->" in report:
        report = report.replace(
            "<!-- nearest-state-separation:start -->",
            section + "\n<!-- nearest-state-separation:start -->",
            1,
        )
    else:
        report = report.rstrip() + "\n\n" + section

    v5_link = (
        "- [Current v5 combined panels (a–c)]"
        "(figure_unoccupied_repeatability_main_v5.pdf)"
    )
    if v5_link not in report:
        v4_link = (
            "- [Current v4 combined panels (a–c)]"
            "(figure_unoccupied_repeatability_main_v4.pdf)"
        )
        previous_v4_link = (
            "- [Previous v4 combined panels (a–c)]"
            "(figure_unoccupied_repeatability_main_v4.pdf)"
        )
        if v4_link in report:
            report = report.replace(
                v4_link,
                v5_link + "\n" + previous_v4_link,
                1,
            )
        else:
            report = report.replace(
                "Main-text candidates:\n\n",
                "Main-text candidates:\n\n" + v5_link + "\n",
                1,
            )
    legacy_link = (
        "- [Legacy combined figure retained]"
        "(figure_unoccupied_repeatability_main.pdf)"
    )
    supplementary_heading = "\nComplete supplementary CSI curves:"
    if legacy_link in report and supplementary_heading in report:
        description_start = (
            report.index(legacy_link) + len(legacy_link)
        )
        description_end = report.index(
            supplementary_heading,
            description_start,
        )
        figure_description = (
            "\n\nThe current v5 main figure uses state-level within/between "
            "distances as\n"
            "panel (a), the scatter-free overall distance boxplot as panel "
            "(b), and\n"
            "nearest-state separation as panel (c). Panel (a) error bars "
            "show mean ± SD;\n"
            "panel (c) labels the equal-distance threshold and the two ratio "
            "classes.\n"
            "State-level Pearson correlation remains available in\n"
            "[`main_panel_c_intra_state_correlation.pdf`]"
            "(main_panel_c_intra_state_correlation.pdf)\n"
            "and the supplementary correlation outputs.\n"
        )
        report = (
            report[:description_start]
            + figure_description
            + report[description_end:]
        )
    report_path.write_text(report, encoding="utf-8")


def nearest_state_report_section(nearest_frame: pd.DataFrame) -> str:
    ordered = ordered_state_frame(nearest_frame)
    result_table = markdown_table(
        ordered,
        (
            ("environment_label", "Environment"),
            ("paper_state_label", "State"),
            ("nearest_other_state_label", "Nearest other state"),
            ("mean_intra_distance", "Mean intra distance"),
            (
                "mean_nearest_inter_distance",
                "Mean nearest-state distance",
            ),
            ("nearest_state_ratio", "Ratio"),
        ),
    )
    below_one = ordered.loc[ordered["nearest_state_ratio"] < 1.0]
    near_one = ordered.loc[
        ordered["nearest_state_ratio"].between(1.0, 1.2)
    ]

    def describe_rows(frame: pd.DataFrame) -> str:
        if frame.empty:
            return "- None."
        return "\n".join(
            (
                f"- {row.environment_label} — {row.paper_state_label} "
                f"→ nearest state: {row.nearest_other_state_label}; "
                f"ratio: {row.nearest_state_ratio:.3f}"
            )
            for row in frame.itertuples(index=False)
        )

    return f"""<!-- nearest-state-separation:start -->
## Nearest-state separation diagnostic

The revised main figure omits the state-level within-state Pearson correlation
panel because the correlations cluster close to 1, producing a mostly empty
0–1 axis while offering no direct comparison with between-state correlation.
The complete correlation CSV files, state summary, matrices, and heatmaps are
retained as supplementary results.

For each state `s`, the analysis first calculates its mean intra-state
distance `d_intra(s)`. Within the same `environment_id` and
`link_configuration`, it then averages recording-pair distances separately for
each alternative state `t`, selects the alternative with the smallest mean
distance `d_nearest(s)`, and reports:

`nearest-state separation ratio = d_nearest(s) / d_intra(s)`

A ratio above 1 means that even the nearest alternative state is farther away
than the state's mean internal variation. A ratio below 1 identifies a
potentially confusable condition; it is not by itself a classification error
rate.

{result_table}

### States with ratio below 1

{describe_rows(below_one)}

### States with ratio from 1.0 to 1.2

{describe_rows(near_one)}

Most states showed larger distances to their nearest alternative state than
their within-state variation. However, some state pairs, such as heater stable
operation and heater start-up transition, exhibited comparable or overlapping
distances and should be interpreted as potentially confusable conditions.
Kitchen — Clean also had a ratio below 1 relative to Kettle boiling. These
descriptive ratios use the existing standardized feature space and do not
alter the underlying distance calculation.

The machine-readable results are stored in
[`nearest_state_separation.csv`](nearest_state_separation.csv).
<!-- nearest-state-separation:end -->
"""


def update_report_with_nearest_section(
    report_path: Path,
    nearest_frame: pd.DataFrame,
) -> None:
    """Insert or replace the v2 main-figure rationale in an existing report."""

    section = nearest_state_report_section(nearest_frame)
    if report_path.is_file():
        report = report_path.read_text(encoding="utf-8")
    else:
        report = "# Unoccupied environment validation\n\n"
    start_marker = "<!-- nearest-state-separation:start -->"
    end_marker = "<!-- nearest-state-separation:end -->"
    if start_marker in report and end_marker in report:
        start_index = report.index(start_marker)
        end_index = report.index(end_marker) + len(end_marker)
        report = report[:start_index] + section.rstrip() + report[end_index:]
    elif "## State-level results" in report:
        report = report.replace(
            "## State-level results",
            section + "\n## State-level results",
            1,
        )
    else:
        report = report.rstrip() + "\n\n" + section

    old_main_link = (
        "- [Combined panels (a–c)]"
        "(figure_unoccupied_repeatability_main.pdf)"
    )
    new_main_links = (
        "- [Revised combined panels (a–c)]"
        "(figure_unoccupied_repeatability_main_v2.pdf)\n"
        "- [Legacy combined figure retained]"
        "(figure_unoccupied_repeatability_main.pdf)"
    )
    if old_main_link in report:
        report = report.replace(old_main_link, new_main_links, 1)
    elif "figure_unoccupied_repeatability_main_v2.pdf" not in report:
        report += (
            "\n\n- [Revised combined panels (a–c)]"
            "(figure_unoccupied_repeatability_main_v2.pdf)\n"
        )
    old_standalone_links = (
        "- [(a) Distance distribution]"
        "(main_panel_a_distance_distribution.pdf)\n"
        "- [(b) State-level distances]"
        "(main_panel_b_state_level_distances.pdf)\n"
        "- [(c) Within-state correlation]"
        "(main_panel_c_intra_state_correlation.pdf)\n"
    )
    report = report.replace(old_standalone_links, "")
    figure_note = (
        "The revised main figure uses state-level distances as panel (a), "
        "the overall\n"
        "distance distribution as panel (b), and nearest-state separation "
        "ratio as\n"
        "panel (c). State-level Pearson correlation remains available in\n"
        "[`main_panel_c_intra_state_correlation.pdf`]"
        "(main_panel_c_intra_state_correlation.pdf)\n"
        "and the supplementary correlation outputs."
    )
    if figure_note not in report:
        legacy_link = (
            "- [Legacy combined figure retained]"
            "(figure_unoccupied_repeatability_main.pdf)"
        )
        report = report.replace(
            legacy_link,
            legacy_link + "\n\n" + figure_note,
            1,
        )
    revised_correlation_text = (
        "The dedicated correlation summary displays state means ± "
        "descriptive\n"
        "SD and the actual independent-recording count; it is retained "
        "outside the\n"
        "revised main figure."
    )
    for previous_text in (
        "Panel (c) displays state means ± descriptive SD and\n"
        "the actual independent-recording count.",
        "The dedicated correlation summary displays state means ± "
        "descriptive SD and\n"
        "the actual independent-recording count; it is retained outside "
        "the revised main figure.",
    ):
        report = report.replace(previous_text, revised_correlation_text)
    report_path.write_text(report, encoding="utf-8")


def update_report_for_v4(report_path: Path) -> None:
    """Insert or replace the v4 codebook and figure-design explanation."""

    if report_path.is_file():
        report = report_path.read_text(encoding="utf-8")
    else:
        report = "# Unoccupied environment validation\n\n"
    section = v4_state_code_report_section()
    start_marker = "<!-- v4-state-code-mapping:start -->"
    end_marker = "<!-- v4-state-code-mapping:end -->"
    if start_marker in report and end_marker in report:
        start_index = report.index(start_marker)
        end_index = report.index(end_marker) + len(end_marker)
        report = report[:start_index] + section.rstrip() + report[end_index:]
    elif "<!-- nearest-state-separation:start -->" in report:
        report = report.replace(
            "<!-- nearest-state-separation:start -->",
            section + "\n<!-- nearest-state-separation:start -->",
            1,
        )
    elif "## State-level results" in report:
        report = report.replace(
            "## State-level results",
            section + "\n## State-level results",
            1,
        )
    else:
        report = report.rstrip() + "\n\n" + section

    v4_link = (
        "- [Current v4 combined panels (a–c)]"
        "(figure_unoccupied_repeatability_main_v4.pdf)"
    )
    if v4_link not in report:
        report = report.replace(
            "Main-text candidates:\n\n",
            "Main-text candidates:\n\n" + v4_link + "\n",
            1,
        )
    report = report.replace(
        "The revised main figure uses state-level distances as panel (a), "
        "the overall\n"
        "distance distribution as panel (b), and nearest-state separation "
        "ratio as\n"
        "panel (c).",
        "The v4 main figure uses code-labelled state-level distances as "
        "panel (a),\n"
        "the scatter-free overall distance boxplot as panel (b), and the "
        "code-only\n"
        "nearest-state separation ratio as panel (c).",
    )
    report_path.write_text(report, encoding="utf-8")


def _create_legacy_report(
    *,
    output_dir: Path,
    dataset_root: Path,
    field_discovery: pd.DataFrame,
    recording_inventory: pd.DataFrame,
    state_inventory: pd.DataFrame,
    summary: pd.DataFrame,
    correlation_pairs: pd.DataFrame,
    distance_pairs: pd.DataFrame,
    overall_distances: pd.DataFrame,
    low_energy_ratio: float,
    distance_plot: str,
) -> None:
    count_distribution = (
        state_inventory["num_recordings"].value_counts().sort_index()
    )
    count_text = ", ".join(
        f"{count} state(s) with {recordings} recording(s)"
        for recordings, count in count_distribution.items()
    )
    pooled_corr = float(
        correlation_pairs["pearson_correlation"].mean()
    )
    state_corr_min = float(summary["mean_intra_corr"].min())
    state_corr_max = float(summary["mean_intra_corr"].max())

    intra_distances = distance_pairs.loc[
        distance_pairs["comparison_type"] == "intra_state", "distance"
    ]
    matched_inter_distances = distance_pairs.loc[
        distance_pairs["comparison_type"]
        == "inter_state_matched_environment_link",
        "distance",
    ]
    mean_intra = float(intra_distances.mean())
    mean_inter = float(matched_inter_distances.mean())
    median_intra = float(intra_distances.median())
    median_inter = float(matched_inter_distances.median())
    distance_ratio = mean_inter / mean_intra
    all_durations = recording_inventory["duration_sec"].astype(float)
    state_distance_ratios = (
        summary["inter_distance"] / summary["intra_distance"]
    )
    num_states_inter_greater = int((state_distance_ratios > 1).sum())

    if distance_ratio > 1:
        interpretation = (
            "The matched-context inter-state distance exceeded the within-state "
            "distance, supporting observable separability after controlling for "
            "environment and link configuration."
        )
    else:
        interpretation = (
            "The matched-context inter-state distance did not exceed the "
            "within-state distance; the descriptive result does not support a "
            "separability claim without further analysis."
        )

    field_table = markdown_table(
        field_discovery,
        (
            ("requested_concept", "Requested concept"),
            ("status", "Status"),
            ("discovered_field_paths", "Discovered metadata fields"),
            ("example_values", "Examples"),
        ),
    )
    count_table = markdown_table(
        state_inventory,
        (
            ("state", "State"),
            ("num_recordings", "Independent recordings"),
            ("mean_duration_sec", "Mean duration (s)"),
            ("min_duration_sec", "Min duration (s)"),
            ("max_duration_sec", "Max duration (s)"),
        ),
    )
    result_table = markdown_table(
        summary,
        (
            ("state", "State"),
            ("num_recordings", "n"),
            ("mean_intra_corr", "Mean intra-state r"),
            ("intra_distance", "Mean intra distance"),
            ("inter_distance", "Mean matched inter distance"),
            ("inter_intra_distance_ratio", "Inter/intra ratio"),
        ),
    )
    overall_table = markdown_table(
        overall_distances,
        (
            ("comparison_type", "Comparison"),
            ("num_pairs", "Pairs"),
            ("mean_distance", "Mean"),
            ("std_distance", "SD"),
            ("median_distance", "Median"),
        ),
    )

    non_five = state_inventory.loc[
        state_inventory["num_recordings"] != 5,
        ["state", "num_recordings", "sample_ids"],
    ]
    if non_five.empty:
        count_note = (
            "Every discovered state contains exactly five independent records."
        )
    else:
        count_note = (
            "The released metadata is not exactly five-per-state for every "
            "label. The following deviations are retained rather than silently "
            "truncated or duplicated:\n\n"
            + markdown_table(
                non_five,
                (
                    ("state", "State"),
                    ("num_recordings", "n"),
                    ("sample_ids", "Released sample IDs"),
                ),
            )
        )

    distance_figure_links: list[str] = []
    if distance_plot in {"both", "paired"}:
        distance_figure_links.append(
            "- [Recommended main figure: state-level paired mean distances]"
            "(figure_3_state_level_paired_distances.png)"
        )
    if distance_plot in {"both", "boxplot"}:
        distance_figure_links.append(
            "- [Supplementary pair-level distance distribution]"
            "(figure_3_distance_boxplot.png)"
        )
    distance_figure_links_text = "\n".join(distance_figure_links)

    paper_paragraph = (
        f"AXHome-MM-v1 contains {len(recording_inventory)} unoccupied "
        f"environment recordings distributed across {len(state_inventory)} "
        "metadata-derived environment-state labels. Each metadata/CSI pair was "
        "treated as one independent recording, with a mean duration of "
        f"{all_durations.mean():.3f} s (range "
        f"{all_durations.min():.3f}-{all_durations.max():.3f} s). "
        "After low-energy subcarrier repair and recording-wise amplitude "
        "normalization, the pooled mean pairwise Pearson correlation between "
        f"repetitions of the same state was {pooled_corr:.3f}; state-level "
        f"means ranged from {state_corr_min:.3f} to {state_corr_max:.3f}. "
        "Using the concatenated mean-amplitude and amplitude-variance curves, "
        "the mean Euclidean distance was "
        f"{mean_intra:.3f} within states and {mean_inter:.3f} between states "
        "recorded in the same environment and link configuration "
        f"({distance_ratio:.2f}-fold inter/intra ratio). "
        + interpretation
    )

    report = f"""# Unoccupied environment repeatability and separability validation

## Scope and provenance

- Dataset root: `{dataset_root}`
- Independent unoccupied recordings: **{len(recording_inventory)}**
- Discovered environment-state labels: **{len(state_inventory)}**
- Repetition-count distribution: {count_text}
- Recording duration: mean **{all_durations.mean():.3f} s**, range **{all_durations.min():.3f}-{all_durations.max():.3f} s**

One released metadata JSON / CSI file pair is the independent acquisition unit.
No sliding-window or action-window subdivision count is used as a repetition
count.

## Metadata field discovery

{field_table}

The current release does not expose separate `condition`, `scenario`, or
`environment_state` fields. Consequently, the state token is recovered from
the section of `sample_id` between the metadata-provided `environment_id` and
`trial_id` anchors. This preserves the exact released labels; for example, the
living-room heater label is represented as
`E1__heater_on_heater_cloth_front`, rather than being renamed by the analysis.

## Data quantity

{count_table}

{count_note}

The complete sample-level inventory, including each `sample_id`, duration, and
released CSI path, is stored in
[`unoccupied_recordings.csv`](unoccupied_recordings.csv).

## CSI amplitude processing

The existing project FeitCSI decoder was reused. Each file was decoded as
`(packet, RX, TX, subcarrier)` complex CSI. Amplitude was computed as `|H|`.
The first TX stream was used, while all packets and both RX streams were
included. Subcarriers whose recording-level energy was below
{low_energy_ratio:.3f} times the median energy were linearly interpolated using
the existing baseline preprocessing rule. For each independent recording:

1. the amplitude mean curve was calculated over packet and RX dimensions;
2. the amplitude standard-deviation curve was calculated over the same
   dimensions; and
3. both curves were divided by the recording's mean amplitude, yielding
   dimensionless normalized curves.

- [Figure 1: normalized mean-amplitude curves](figure_1_amplitude_mean_curves.png)
- [Figure 2: normalized amplitude-SD curves](figure_2_amplitude_std_curves.png)

## Within-state correlation

Pearson correlation was computed between every pair of normalized mean-amplitude
curves within each state. The pooled pairwise mean correlation was
**{pooled_corr:.3f}**. State-level mean correlations ranged from
**{state_corr_min:.3f}** to **{state_corr_max:.3f}**.

Individual matrices and heatmaps are available under
[`correlation_matrices/`](correlation_matrices/) and
[`correlation_heatmaps/`](correlation_heatmaps/).

## Intra-state versus inter-state distance

Each recording feature concatenates its normalized mean-amplitude curve and
normalized amplitude-variance curve. Every feature dimension was standardized
across the complete unoccupied-recording set before ordinary Euclidean distance
was calculated.

The primary inter-state comparison is deliberately restricted to recordings
with the same `environment_id` and `link_configuration`. This prevents room
layout or link geometry from trivially creating the apparent class separation.
Cross-environment/link distances are retained as a supplementary category in
`distance_pairs.csv`.

{overall_table}

The pooled mean intra-state distance was **{mean_intra:.3f}** (median
**{median_intra:.3f}**), compared with a matched-context inter-state mean of
**{mean_inter:.3f}** (median **{median_inter:.3f}**), an inter/intra mean ratio
of **{distance_ratio:.2f}**.

{interpretation}

At the state level, **{num_states_inter_greater} of {len(summary)}** states had
a larger matched inter-state mean distance than their intra-state mean
distance. The paired figure treats the environment state, rather than each
overlapping recording pair, as the visual unit of analysis. The pair-level
boxplot is retained as a supplementary distribution view.

{distance_figure_links_text}

## State-level summary

{result_table}

The machine-readable version is
[`unoccupied_validation_summary.csv`](unoccupied_validation_summary.csv).

## Paper-ready Technical Validation text

> {paper_paragraph}

## Interpretation limits

- These results describe the released unoccupied recordings and do not imply
  population-level human-activity coverage.
- Pairwise observations share recordings and are therefore not independent;
  the module reports descriptive distances rather than treating every pair as
  an independent replicate in a significance test.
- The analysis reports the actual release counts (4-6 per state where
  applicable), rather than assuming that every state has exactly five files.
"""
    (output_dir / "unoccupied_validation_report.md").write_text(
        report, encoding="utf-8"
    )


def create_report(
    *,
    output_dir: Path,
    dataset_root: Path,
    field_discovery: pd.DataFrame,
    recording_inventory: pd.DataFrame,
    state_inventory: pd.DataFrame,
    summary: pd.DataFrame,
    correlation_pairs: pd.DataFrame,
    distance_pairs: pd.DataFrame,
    overall_distances: pd.DataFrame,
    effect_statistics: pd.DataFrame,
    nearest_state_frame: pd.DataFrame,
    exclusion_ledger: pd.DataFrame,
    low_energy_ratio: float,
    distance_plot: str,
) -> None:
    """Write the publication-oriented validation report."""

    count_distribution = (
        state_inventory["num_recordings"].value_counts().sort_index()
    )
    count_text = ", ".join(
        f"{count} state(s) with {recordings} recording(s)"
        for recordings, count in count_distribution.items()
    )
    pooled_corr = float(correlation_pairs["pearson_correlation"].mean())
    state_corr_min = float(summary["mean_intra_corr"].min())
    state_corr_max = float(summary["mean_intra_corr"].max())
    intra_distances = distance_pairs.loc[
        distance_pairs["comparison_type"] == "intra_state", "distance"
    ]
    matched_inter_distances = distance_pairs.loc[
        distance_pairs["comparison_type"]
        == "inter_state_matched_environment_link",
        "distance",
    ]
    mean_intra = float(intra_distances.mean())
    mean_inter = float(matched_inter_distances.mean())
    distance_ratio = mean_inter / mean_intra
    effect_row = effect_statistics.iloc[0]
    all_durations = recording_inventory["duration_sec"].astype(float)
    state_distance_ratios = (
        summary["inter_distance"] / summary["intra_distance"]
    )
    num_states_inter_greater = int((state_distance_ratios > 1).sum())
    clean_explanation = clean_living_room_explanation(
        state_inventory, exclusion_ledger
    )
    state_code_section = v4_state_code_report_section()
    nearest_section = nearest_state_report_section(nearest_state_frame)

    field_table = markdown_table(
        field_discovery,
        (
            ("requested_concept", "Requested concept"),
            ("status", "Status"),
            ("discovered_field_paths", "Discovered metadata fields"),
            ("example_values", "Examples"),
        ),
    )
    count_table = markdown_table(
        ordered_state_frame(state_inventory),
        (
            ("environment_label", "Environment"),
            ("paper_state_label", "State"),
            ("num_recordings", "Independent recordings"),
            ("mean_duration_sec", "Mean duration (s)"),
            ("min_duration_sec", "Min duration (s)"),
            ("max_duration_sec", "Max duration (s)"),
        ),
    )
    result_table = markdown_table(
        ordered_state_frame(summary),
        (
            ("environment_label", "Environment"),
            ("paper_state_label", "State"),
            ("num_recordings", "n"),
            ("mean_intra_corr", "Mean within-state r"),
            ("intra_distance", "Mean intra distance"),
            ("inter_distance", "Mean matched inter distance"),
            ("inter_intra_distance_ratio", "Inter/intra ratio"),
        ),
    )
    overall_table = markdown_table(
        overall_distances,
        (
            ("comparison_type", "Comparison"),
            ("num_pairs", "Pairs"),
            ("mean_distance", "Mean"),
            ("std_distance", "SD"),
            ("median_distance", "Median"),
            ("iqr_distance", "IQR"),
        ),
    )
    non_five = ordered_state_frame(
        state_inventory.loc[
            state_inventory["num_recordings"] != 5,
            [
                "state",
                "environment_id",
                "environment_label",
                "paper_state_label",
                "num_recordings",
                "sample_ids",
            ],
        ]
    )
    if non_five.empty:
        count_note = (
            "Every discovered state contains exactly five independent records."
        )
    else:
        count_note = (
            "The active release is not exactly five-per-state for every "
            "label. Deviations are retained rather than truncated, imputed, "
            "or duplicated:\n\n"
            + markdown_table(
                non_five,
                (
                    ("environment_label", "Environment"),
                    ("paper_state_label", "State"),
                    ("num_recordings", "n"),
                    ("sample_ids", "Released sample IDs"),
                ),
            )
        )
    if exclusion_ledger.empty:
        exclusion_text = "No release exclusion ledger entry was found."
    else:
        exclusion_text = markdown_table(
            exclusion_ledger,
            (
                ("sample_id", "Excluded sample"),
                ("reason", "Reason"),
                ("release_action", "Release action"),
            ),
        )

    paper_paragraph = (
        f"AXHome-MM-v1 contains {len(recording_inventory)} active unoccupied "
        f"environment recordings across {len(state_inventory)} states. "
        "After recording-wise mean-amplitude scaling, the pooled mean "
        "within-state Pearson correlation was "
        f"{pooled_corr:.3f}. The mean Euclidean distance was "
        f"{mean_intra:.3f} within states and {mean_inter:.3f} between states "
        "under matched environment and link configurations "
        f"({distance_ratio:.2f}-fold mean ratio). The independent repetitions "
        "exhibited consistent within-state CSI patterns, while between-state "
        "distances were generally larger than within-state variation under "
        "matched environment and link configurations. These results support "
        "the repeatability of the collected unoccupied-state recordings, "
        "while not implying that additional recordings would contain no new "
        "variation."
    )

    report = f"""# Unoccupied environment repeatability and separability validation

## Scope and provenance

- Dataset root: `{dataset_root}`
- Independent active unoccupied recordings: **{len(recording_inventory)}**
- Discovered environment-state labels: **{len(state_inventory)}**
- Repetition-count distribution: {count_text}
- Recording duration: mean **{all_durations.mean():.3f} s**, range
  **{all_durations.min():.3f}-{all_durations.max():.3f} s**
- Recorded release exclusions: **{len(exclusion_ledger)}**
- Requested standalone distance plot mode: `{distance_plot}`

One active metadata JSON / CSI file pair is the independent acquisition unit.
No window count is used as a repetition count.

## Metadata field discovery

{field_table}

The active release exposes environment and action identifiers, but does not
expose separate `condition`, `scenario`, or `environment_state` fields.
Consequently, the state token is recovered from `sample_id` using
metadata-provided environment and trial anchors. Machine-readable state tokens
remain unchanged; explicit readable labels are applied only to paper figures.

## Recording counts

{count_table}

{count_note}

### Why Living room — Clean is n=4

{clean_explanation}

The release exclusion ledger contains:

{exclusion_text}

The faucet exclusion is separate from the Living room — Clean count.
The complete sample inventory is
[`unoccupied_recordings.csv`](unoccupied_recordings.csv).

## Normalization method

CSI amplitude is computed as `|H|` after decoding the first TX stream and
retaining packets and both RX streams. The existing low-energy repair rule
interpolates subcarriers below {low_energy_ratio:.3f} times the recording
median energy. For each independent recording, the packet/RX mean-amplitude
curve and temporal amplitude-SD curve are divided by that recording's single
mean-amplitude scale.

This is **not** per-recording Min–Max normalization and **not** per-recording
Z-scoring. It is division by the recording's own mean amplitude; neither the
maximum nor median is used as the amplitude scale. Accordingly, figure axes
use **Relative CSI amplitude**. No additional smoothing, forced alignment,
record deletion, or outlier suppression is applied.

See [`normalization_audit.md`](normalization_audit.md) for the code-level
audit.

## Feature standardization

Each distance feature concatenates the relative mean-amplitude and relative
amplitude-variance curves. All {len(recording_inventory)} recording vectors in
the comparison set are stacked first. A single mean and population SD
(`ddof=0`) are fitted for each feature dimension across the complete set and
applied to every recording. Constant dimensions are excluded. A separate
Z-score is never fitted to each recording.

The primary inter-state comparison is restricted to pairs sharing both
`environment_id` and `link_configuration`. Cross-environment/link pairs remain
available in [`distance_pairs.csv`](distance_pairs.csv), but are not used as
the main inter-state contrast.

## Within-state correlation

Pearson correlation is calculated between every pair of relative
mean-amplitude curves within a state. The pooled pairwise mean was
**{pooled_corr:.3f}**; state means ranged from **{state_corr_min:.3f}** to
**{state_corr_max:.3f}**. The dedicated correlation summary displays state
means ± descriptive SD and the actual independent-recording count; it is
retained outside the revised main figure.

Complete matrices and heatmaps are retained under
[`correlation_matrices/`](correlation_matrices/) and
[`supplementary_correlation_heatmaps/`](supplementary_correlation_heatmaps/).

## Intra-state and inter-state distance

{overall_table}

The intra-state median was **{float(effect_row["intra_median"]):.3f}**
(IQR **{float(effect_row["intra_iqr"]):.3f}**). The matched-context inter-state
median was **{float(effect_row["inter_median"]):.3f}** (IQR
**{float(effect_row["inter_iqr"]):.3f}**), giving a median inter/intra ratio of
**{float(effect_row["median_inter_intra_ratio"]):.2f}**. Cliff's delta for
matched inter-state versus intra-state distance was
**{float(effect_row["cliffs_delta_inter_vs_intra"]):.3f}**; positive values
indicate larger inter-state distances.

At state level, **{num_states_inter_greater} of {len(summary)}** states had a
larger matched inter-state mean than intra-state mean. State-level error bars
are **mean ± SD**. A bootstrap confidence interval is not claimed because the
pairwise distances share recordings and do not form independent resampling
units.

The pairwise medians, IQRs, median ratio, and Cliff's delta are descriptive.
Pairwise distances share recordings and are not mutually independent; no
independent-pair significance claim or p-value is made.

{state_code_section}

{nearest_section}

## State-level results

{result_table}

The machine-readable summary is
[`unoccupied_validation_summary.csv`](unoccupied_validation_summary.csv), and
effect-size statistics are in
[`distance_effect_size_statistics.csv`](distance_effect_size_statistics.csv).

## Figure files

Main-text candidates:

- [Current v4 combined panels (a–c)](figure_unoccupied_repeatability_main_v4.pdf)
- [Revised combined panels (a–c)](figure_unoccupied_repeatability_main_v2.pdf)
- [Legacy combined figure retained](figure_unoccupied_repeatability_main.pdf)

The v4 main figure uses code-labelled state-level distances as panel (a), the
scatter-free overall distance boxplot as panel (b), and the code-only
nearest-state separation ratio as panel (c). State-level Pearson correlation
remains available in
[`main_panel_c_intra_state_correlation.pdf`](main_panel_c_intra_state_correlation.pdf)
and the supplementary correlation outputs.

Complete supplementary CSI curves:

- [S1a Living room mean amplitude](supp_figure_S1a_living_room_mean_amplitude.pdf)
- [S1b Bathroom mean amplitude](supp_figure_S1b_bathroom_mean_amplitude.pdf)
- [S1c Kitchen mean amplitude](supp_figure_S1c_kitchen_mean_amplitude.pdf)
- [S2a Living room variability](supp_figure_S2a_living_room_variability.pdf)
- [S2b Bathroom variability](supp_figure_S2b_bathroom_variability.pdf)
- [S2c Kitchen variability](supp_figure_S2c_kitchen_variability.pdf)

Mean-amplitude shading is the across-recording SD of independent
recording-level mean curves, not within-recording temporal SD. PNG counterparts
are written at 600 dpi; PDF text and lines remain vector elements. Existing
historical outputs are not deleted.

## Paper-ready Technical Validation text

> {paper_paragraph}

## Conservative interpretation

- The results describe the active released recordings and do not establish
  that five repetitions are definitively sufficient.
- Additional recordings may contain variation not represented here.
- Actual counts are retained: Living room — Clean is n=4; no fifth record is
  imputed or duplicated.
- The generally larger matched-context inter-state distances support
  observable separability, but do not make every pair an independent
  statistical replicate.
"""
    (output_dir / "unoccupied_validation_report.md").write_text(
        report, encoding="utf-8"
    )


def write_generated_file_manifest(output_dir: Path) -> None:
    paths = sorted(
        path.relative_to(output_dir).as_posix()
        for path in output_dir.rglob("*")
        if path.is_file() and path.name != "generated_files.txt"
    )
    (output_dir / "generated_files.txt").write_text(
        "\n".join(paths) + "\n", encoding="utf-8"
    )


def log_main_v2_summary(
    *,
    nearest_frame: pd.DataFrame,
    output_dir: Path,
    legacy_main_preserved: bool,
    dpi: int,
) -> None:
    ordered = ordered_state_frame(nearest_frame)
    LOGGER.info(
        "V2 | New main figure: %s",
        output_dir / "figure_unoccupied_repeatability_main_v2.pdf",
    )
    for row in ordered.itertuples(index=False):
        LOGGER.info(
            "V2 | %s / %s -> nearest: %s | ratio=%.3f",
            row.environment_label,
            row.paper_state_label,
            row.nearest_other_state_label,
            row.nearest_state_ratio,
        )
    below_one = ordered.loc[ordered["nearest_state_ratio"] < 1.0]
    near_one = ordered.loc[
        ordered["nearest_state_ratio"].between(1.0, 1.2)
    ]
    LOGGER.info(
        "V2 | Ratio < 1 states: %s",
        (
            "none"
            if below_one.empty
            else "; ".join(
                f"{row.environment_label} / {row.paper_state_label}"
                for row in below_one.itertuples(index=False)
            )
        ),
    )
    LOGGER.info(
        "V2 | Ratio 1.0-1.2 states: %s",
        (
            "none"
            if near_one.empty
            else "; ".join(
                f"{row.environment_label} / {row.paper_state_label}"
                for row in near_one.itertuples(index=False)
            )
        ),
    )
    pdf_path = output_dir / "figure_unoccupied_repeatability_main_v2.pdf"
    png_path = output_dir / "figure_unoccupied_repeatability_main_v2.png"
    LOGGER.info(
        "V2 | Legacy main figure preserved: %s",
        "yes" if legacy_main_preserved else "no legacy file was present",
    )
    LOGGER.info(
        "V2 | Output verification: PDF=%s; PNG=%s; PNG dpi=%d",
        "yes" if pdf_path.is_file() else "no",
        "yes" if png_path.is_file() else "no",
        max(int(dpi), FIGURE_DPI),
    )


def redraw_main_v2_from_existing(
    *,
    output_dir: Path,
    dpi: int,
) -> int:
    """Redraw v2 from result CSVs without accessing metadata or CSI."""

    pair_path = output_dir / "distance_pairs.csv"
    state_path = output_dir / "state_distance_statistics.csv"
    missing = [
        path for path in (pair_path, state_path) if not path.is_file()
    ]
    if missing:
        raise FileNotFoundError(
            "Cannot redraw main v2; missing existing result CSV files: "
            + ", ".join(str(path) for path in missing)
        )
    legacy_paths = (
        output_dir / "figure_unoccupied_repeatability_main.pdf",
        output_dir / "figure_unoccupied_repeatability_main.png",
    )
    legacy_main_preserved = all(path.is_file() for path in legacy_paths)
    pair_frame = pd.read_csv(pair_path)
    state_distance_frame = pd.read_csv(state_path)
    nearest_frame = nearest_state_separation_analysis(
        pair_frame=pair_frame,
        state_distance_frame=state_distance_frame,
        output_dir=output_dir,
    )
    plot_main_combination_v2(
        pair_frame=pair_frame,
        state_distance_frame=state_distance_frame,
        nearest_state_frame=nearest_frame,
        output_dir=output_dir,
        dpi=dpi,
    )
    update_report_with_nearest_section(
        output_dir / "unoccupied_validation_report.md",
        nearest_frame,
    )
    write_generated_file_manifest(output_dir)
    log_main_v2_summary(
        nearest_frame=nearest_frame,
        output_dir=output_dir,
        legacy_main_preserved=(
            legacy_main_preserved
            and all(path.is_file() for path in legacy_paths)
        ),
        dpi=dpi,
    )
    return 0


def log_main_v4_summary(*, output_dir: Path, dpi: int) -> None:
    """Print the requested v4 rendering choices and output locations."""

    LOGGER.info("V4 | Raw scatter layer removed: yes")
    LOGGER.info("V4 | Boxplot width: %.2f", V4_BOXPLOT_WIDTH)
    for state in STATE_ORDER:
        environment_id = state.split("__", 1)[0]
        LOGGER.info(
            "V4 | State code mapping: %s = %s / %s",
            paper_state_code(state),
            paper_environment_label(environment_id),
            paper_state_short_label(state),
        )
    pdf_path = (
        output_dir / "figure_unoccupied_repeatability_main_v4.pdf"
    )
    png_path = (
        output_dir / "figure_unoccupied_repeatability_main_v4.png"
    )
    LOGGER.info("V4 | PDF output: %s", pdf_path)
    LOGGER.info("V4 | PNG output: %s", png_path)
    LOGGER.info(
        "V4 | Output verification: PDF=%s; PNG=%s; PNG dpi=%d",
        "yes" if pdf_path.is_file() else "no",
        "yes" if png_path.is_file() else "no",
        max(int(dpi), FIGURE_DPI),
    )


def redraw_main_v4_from_existing(
    *,
    output_dir: Path,
    dpi: int,
) -> int:
    """Redraw v4 from existing result CSVs without metadata or CSI access."""

    pair_path = output_dir / "distance_pairs.csv"
    state_path = output_dir / "state_distance_statistics.csv"
    nearest_path = output_dir / "nearest_state_separation.csv"
    missing = [
        path for path in (pair_path, state_path) if not path.is_file()
    ]
    if missing:
        raise FileNotFoundError(
            "Cannot redraw main v4; missing existing result CSV files: "
            + ", ".join(str(path) for path in missing)
        )
    pair_frame = pd.read_csv(pair_path)
    state_distance_frame = pd.read_csv(state_path)
    if nearest_path.is_file():
        nearest_frame = pd.read_csv(nearest_path)
    else:
        nearest_frame = nearest_state_separation_analysis(
            pair_frame=pair_frame,
            state_distance_frame=state_distance_frame,
            output_dir=output_dir,
        )
    validate_paper_labels(state_distance_frame["state"].astype(str))
    validate_paper_labels(nearest_frame["state"].astype(str))
    plot_main_combination_v4(
        pair_frame=pair_frame,
        state_distance_frame=state_distance_frame,
        nearest_state_frame=nearest_frame,
        output_dir=output_dir,
        dpi=dpi,
    )
    report_path = output_dir / "unoccupied_validation_report.md"
    update_report_with_nearest_section(report_path, nearest_frame)
    update_report_for_v4(report_path)
    write_generated_file_manifest(output_dir)
    log_main_v4_summary(output_dir=output_dir, dpi=dpi)
    return 0


def log_main_v5_summary(*, output_dir: Path, dpi: int) -> None:
    """Print the requested v5 presentation and output summary."""

    LOGGER.info("V5 | Panel (a) Ratio column removed: yes")
    for state in STATE_ORDER:
        LOGGER.info(
            "V5 | Panel (c) short label: %s",
            paper_state_v5_code_label(state),
        )
    for panel_label, title in V5_PANEL_TITLES.items():
        LOGGER.info(
            "V5 | Panel title: %s %s",
            panel_label,
            title,
        )
    LOGGER.info(
        "V5 | Font sizes (pt): %s",
        ", ".join(
            f"{name}={size:g}"
            for name, size in V5_FONT_SIZES.items()
        ),
    )
    LOGGER.info("V5 | x=1 threshold explanation added: yes")
    LOGGER.info("V5 | Panel (c) class legend added: yes")
    pdf_path = (
        output_dir / "figure_unoccupied_repeatability_main_v5.pdf"
    )
    png_path = (
        output_dir / "figure_unoccupied_repeatability_main_v5.png"
    )
    caption_path = (
        output_dir
        / "figure_unoccupied_repeatability_main_v5_caption.md"
    )
    LOGGER.info("V5 | PDF output: %s", pdf_path)
    LOGGER.info("V5 | PNG output: %s", png_path)
    LOGGER.info("V5 | Caption output: %s", caption_path)
    LOGGER.info(
        "V5 | Output verification: PDF=%s; PNG=%s; caption=%s; "
        "PNG dpi=%d",
        "yes" if pdf_path.is_file() else "no",
        "yes" if png_path.is_file() else "no",
        "yes" if caption_path.is_file() else "no",
        max(int(dpi), FIGURE_DPI),
    )


def redraw_main_v5_from_existing(
    *,
    output_dir: Path,
    dpi: int,
) -> int:
    """Redraw v5 without scanning data or rewriting any result CSV."""

    pair_path = output_dir / "distance_pairs.csv"
    state_path = output_dir / "state_distance_statistics.csv"
    nearest_path = output_dir / "nearest_state_separation.csv"
    missing = [
        path
        for path in (pair_path, state_path, nearest_path)
        if not path.is_file()
    ]
    if missing:
        raise FileNotFoundError(
            "Cannot redraw main v5; missing existing result CSV files: "
            + ", ".join(str(path) for path in missing)
        )
    pair_frame = pd.read_csv(pair_path)
    state_distance_frame = pd.read_csv(state_path)
    nearest_frame = pd.read_csv(nearest_path)
    validate_paper_labels(state_distance_frame["state"].astype(str))
    validate_paper_labels(nearest_frame["state"].astype(str))
    plot_main_combination_v5(
        pair_frame=pair_frame,
        state_distance_frame=state_distance_frame,
        nearest_state_frame=nearest_frame,
        output_dir=output_dir,
        dpi=dpi,
    )
    write_v5_caption(
        output_dir=output_dir,
        pair_frame=pair_frame,
        nearest_frame=nearest_frame,
    )
    report_path = output_dir / "unoccupied_validation_report.md"
    update_report_with_nearest_section(report_path, nearest_frame)
    update_report_for_v5(report_path)
    write_generated_file_manifest(output_dir)
    log_main_v5_summary(output_dir=output_dir, dpi=dpi)
    return 0


def log_main_v6_summary(*, output_dir: Path, dpi: int) -> None:
    """Print the requested v6 layout and output summary."""

    LOGGER.info("V6 | Panel long titles removed: yes")
    LOGGER.info("V6 | Panel (c) displays state codes only: yes")
    LOGGER.info("V6 | Panel (c) class legend removed: yes")
    LOGGER.info("V6 | Equal-distance threshold text removed: yes")
    LOGGER.info(
        "V6 | Font sizes (pt): %s",
        ", ".join(
            f"{name}={size:g}"
            for name, size in V6_FONT_SIZES.items()
        ),
    )
    pdf_path = (
        output_dir / "figure_unoccupied_repeatability_main_v6.pdf"
    )
    png_path = (
        output_dir / "figure_unoccupied_repeatability_main_v6.png"
    )
    caption_path = (
        output_dir
        / "figure_unoccupied_repeatability_main_v6_caption.md"
    )
    LOGGER.info("V6 | PDF output: %s", pdf_path)
    LOGGER.info("V6 | PNG output: %s", png_path)
    LOGGER.info("V6 | Caption output: %s", caption_path)
    LOGGER.info(
        "V6 | Output verification: PDF=%s; PNG=%s; caption=%s; "
        "PNG dpi=%d",
        "yes" if pdf_path.is_file() else "no",
        "yes" if png_path.is_file() else "no",
        "yes" if caption_path.is_file() else "no",
        max(int(dpi), FIGURE_DPI),
    )


def redraw_main_v6_from_existing(
    *,
    output_dir: Path,
    dpi: int,
) -> int:
    """Redraw v6 without scanning data or rewriting result CSV files."""

    pair_path = output_dir / "distance_pairs.csv"
    state_path = output_dir / "state_distance_statistics.csv"
    nearest_path = output_dir / "nearest_state_separation.csv"
    missing = [
        path
        for path in (pair_path, state_path, nearest_path)
        if not path.is_file()
    ]
    if missing:
        raise FileNotFoundError(
            "Cannot redraw main v6; missing existing result CSV files: "
            + ", ".join(str(path) for path in missing)
        )
    pair_frame = pd.read_csv(pair_path)
    state_distance_frame = pd.read_csv(state_path)
    nearest_frame = pd.read_csv(nearest_path)
    validate_paper_labels(state_distance_frame["state"].astype(str))
    validate_paper_labels(nearest_frame["state"].astype(str))
    plot_main_combination_v6(
        pair_frame=pair_frame,
        state_distance_frame=state_distance_frame,
        nearest_state_frame=nearest_frame,
        output_dir=output_dir,
        dpi=dpi,
    )
    write_v6_caption(
        output_dir=output_dir,
        pair_frame=pair_frame,
    )
    write_generated_file_manifest(output_dir)
    log_main_v6_summary(output_dir=output_dir, dpi=dpi)
    return 0


def _load_existing_main_frames(
    output_dir: Path,
    *,
    variant: str,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    pair_path = output_dir / "distance_pairs.csv"
    state_path = output_dir / "state_distance_statistics.csv"
    nearest_path = output_dir / "nearest_state_separation.csv"
    missing = [
        path
        for path in (pair_path, state_path, nearest_path)
        if not path.is_file()
    ]
    if missing:
        raise FileNotFoundError(
            f"Cannot redraw main {variant}; missing existing result CSV "
            "files: " + ", ".join(str(path) for path in missing)
        )
    pair_frame = pd.read_csv(pair_path)
    state_distance_frame = pd.read_csv(state_path)
    nearest_frame = pd.read_csv(nearest_path)
    validate_paper_labels(state_distance_frame["state"].astype(str))
    validate_paper_labels(nearest_frame["state"].astype(str))
    return pair_frame, state_distance_frame, nearest_frame


def log_main_variant_summary(
    *,
    variant: str,
    output_dir: Path,
    dpi: int,
) -> None:
    stem = f"figure_unoccupied_repeatability_main_{variant}"
    pdf_path = output_dir / f"{stem}.pdf"
    png_path = output_dir / f"{stem}.png"
    caption_path = output_dir / f"{stem}_caption.md"
    LOGGER.info("%s | PDF output: %s", variant.upper(), pdf_path)
    LOGGER.info("%s | PNG output: %s", variant.upper(), png_path)
    LOGGER.info("%s | Caption output: %s", variant.upper(), caption_path)
    LOGGER.info(
        "%s | Output verification: PDF=%s; PNG=%s; caption=%s; PNG dpi=%d",
        variant.upper(),
        "yes" if pdf_path.is_file() else "no",
        "yes" if png_path.is_file() else "no",
        "yes" if caption_path.is_file() else "no",
        max(int(dpi), FIGURE_DPI),
    )


def redraw_main_v7_from_existing(
    *,
    output_dir: Path,
    dpi: int,
) -> int:
    """Redraw v7 without scanning data or rewriting result CSV files."""

    pair_frame, state_distance_frame, nearest_frame = (
        _load_existing_main_frames(output_dir, variant="v7")
    )
    plot_main_combination_v7(
        pair_frame=pair_frame,
        state_distance_frame=state_distance_frame,
        nearest_state_frame=nearest_frame,
        output_dir=output_dir,
        dpi=dpi,
    )
    write_v7_caption(
        output_dir=output_dir,
        pair_frame=pair_frame,
        nearest_state_frame=nearest_frame,
    )
    write_generated_file_manifest(output_dir)
    log_main_variant_summary(
        variant="v7",
        output_dir=output_dir,
        dpi=dpi,
    )
    return 0


def redraw_main_v8_from_existing(
    *,
    output_dir: Path,
    dpi: int,
) -> int:
    """Redraw v8 without scanning data or rewriting result CSV files."""

    pair_frame, state_distance_frame, nearest_frame = (
        _load_existing_main_frames(output_dir, variant="v8")
    )
    plot_main_combination_v8(
        pair_frame=pair_frame,
        state_distance_frame=state_distance_frame,
        nearest_state_frame=nearest_frame,
        output_dir=output_dir,
        dpi=dpi,
    )
    write_v8_caption(
        output_dir=output_dir,
        pair_frame=pair_frame,
        nearest_state_frame=nearest_frame,
    )
    write_generated_file_manifest(output_dir)
    log_main_variant_summary(
        variant="v8",
        output_dir=output_dir,
        dpi=dpi,
    )
    return 0


def redraw_figure7_from_existing(
    *,
    project_root: Path,
    source_results_dir: Path,
    output_dir: Path,
    dpi: int,
    layout_variant: str = "r3",
    panel_b_variant: str = "default",
    palette_variant: str = "current",
) -> int:
    """Generate the dated Figure 7 bundle from the frozen result CSVs."""

    if source_results_dir.resolve() == output_dir.resolve():
        raise ValueError(
            "Figure 7 source-results and delivery directories must differ"
        )
    pair_frame, state_distance_frame, nearest_frame = (
        _load_existing_main_frames(
            source_results_dir,
            variant="Figure 7",
        )
    )
    plot_figure7_condition_distance(
        pair_frame=pair_frame,
        state_distance_frame=state_distance_frame,
        nearest_frame=nearest_frame,
        output_dir=output_dir,
        dpi=dpi,
        layout_variant=layout_variant,
        panel_b_variant=panel_b_variant,
        palette_variant=palette_variant,
    )
    write_figure7_caption(
        output_dir=output_dir,
        pair_frame=pair_frame,
        nearest_frame=nearest_frame,
        layout_variant=layout_variant,
        panel_b_variant=panel_b_variant,
        palette_variant=palette_variant,
    )
    copied_sources = copy_figure7_source_data(
        source_results_dir=source_results_dir,
        output_dir=output_dir,
    )
    write_figure7_generation_record(
        project_root=project_root,
        source_results_dir=source_results_dir,
        output_dir=output_dir,
        pair_frame=pair_frame,
        state_distance_frame=state_distance_frame,
        nearest_frame=nearest_frame,
        copied_sources=copied_sources,
        dpi=dpi,
        layout_variant=layout_variant,
        panel_b_variant=panel_b_variant,
        palette_variant=palette_variant,
    )
    write_generated_file_manifest(output_dir)
    LOGGER.info(
        "Figure 7 | Layout variant=%s; panel-b variant=%s; palette=%s; "
        "PNG/SVG/PDF output stem: %s",
        layout_variant,
        panel_b_variant,
        palette_variant,
        output_dir / FIGURE7_STEM,
    )
    LOGGER.info(
        "Figure 7 | Frozen counts preserved: within-condition n=126; "
        "between-condition n=585; environment-condition rows=13"
    )
    layout_metadata = figure7_layout_metadata(layout_variant)
    LOGGER.info(
        "Figure 7 | Fixed width=%.1f in; height=%.2f in; PNG dpi=%d; %s",
        FIGURE7_WIDTH_IN,
        float(layout_metadata["figure_size_inches"][1]),
        max(int(dpi), FIGURE_DPI),
        layout_metadata["distance_scale"],
    )
    return 0


def main() -> int:
    args = parse_args()
    project_root = Path(__file__).resolve().parents[1]
    dataset_root = args.dataset_root.resolve()
    source_results_dir = args.source_results_dir.resolve()
    output_dir = args.output_dir.resolve()
    configure_logging(output_dir)
    configure_plot_style()
    LOGGER.info("Dataset root: %s", dataset_root)
    LOGGER.info("Output directory: %s", output_dir)
    LOGGER.info("Distance plot selection: %s", args.distance_plot)
    if args.dpi < FIGURE_DPI:
        raise ValueError(
            f"--dpi must be at least {FIGURE_DPI} for publication figures"
        )
    redraw_flags = (
        args.redraw_main_v2_only,
        args.redraw_main_v4_only,
        args.redraw_main_v5_only,
        args.redraw_main_v6_only,
        args.redraw_main_v7_only,
        args.redraw_main_v8_only,
        args.redraw_figure7_only,
    )
    if sum(bool(flag) for flag in redraw_flags) > 1:
        raise ValueError(
            "The --redraw-main-v*-only options are mutually exclusive"
        )
    if args.redraw_figure7_only:
        LOGGER.info(
            "Figure 7 redraw-only mode: frozen CSV results will be copied "
            "byte-for-byte; metadata, CSI, and manuscript will not be touched"
        )
        return redraw_figure7_from_existing(
            project_root=project_root,
            source_results_dir=source_results_dir,
            output_dir=output_dir,
            dpi=args.dpi,
            layout_variant=args.layout_variant,
            panel_b_variant=args.panel_b_variant,
            palette_variant=args.palette_variant,
        )
    if args.redraw_main_v7_only:
        LOGGER.info(
            "V7 redraw-only mode: existing CSV results will be used; "
            "metadata and CSI will not be scanned, and CSV files will not "
            "be rewritten"
        )
        return redraw_main_v7_from_existing(
            output_dir=output_dir,
            dpi=args.dpi,
        )
    if args.redraw_main_v8_only:
        LOGGER.info(
            "V8 redraw-only mode: existing CSV results will be used; "
            "metadata and CSI will not be scanned, and CSV files will not "
            "be rewritten"
        )
        return redraw_main_v8_from_existing(
            output_dir=output_dir,
            dpi=args.dpi,
        )
    if args.redraw_main_v6_only:
        LOGGER.info(
            "V6 redraw-only mode: existing CSV results will be used; "
            "metadata and CSI will not be scanned, and CSV files will not "
            "be rewritten"
        )
        return redraw_main_v6_from_existing(
            output_dir=output_dir,
            dpi=args.dpi,
        )
    if args.redraw_main_v5_only:
        LOGGER.info(
            "V5 redraw-only mode: existing CSV results will be used; "
            "metadata and CSI will not be scanned, and CSV files will not "
            "be rewritten"
        )
        return redraw_main_v5_from_existing(
            output_dir=output_dir,
            dpi=args.dpi,
        )
    if args.redraw_main_v4_only:
        LOGGER.info(
            "V4 redraw-only mode: existing CSV results will be used; "
            "metadata and CSI will not be scanned"
        )
        return redraw_main_v4_from_existing(
            output_dir=output_dir,
            dpi=args.dpi,
        )
    if args.redraw_main_v2_only:
        LOGGER.info(
            "V2 redraw-only mode: existing CSV results will be used; "
            "metadata and CSI will not be scanned"
        )
        return redraw_main_v2_from_existing(
            output_dir=output_dir,
            dpi=args.dpi,
        )
    if not dataset_root.is_dir():
        raise FileNotFoundError(f"Dataset root not found: {dataset_root}")
    if not 0 < args.low_energy_ratio < 1:
        raise ValueError("--low-energy-ratio must be between 0 and 1")

    read_feitcsi, interpolate_low_energy_subcarriers = (
        ensure_baseline_parser(project_root)
    )
    metadata_paths = metadata_files_from_release(dataset_root)
    archive_index = load_archive_index(dataset_root)
    recordings, field_inventory, field_discovery = discover_recordings(
        dataset_root, metadata_paths, archive_index
    )
    field_inventory.to_csv(
        output_dir / "metadata_field_inventory.csv", index=False
    )
    field_discovery.to_csv(
        output_dir / "requested_metadata_field_discovery.csv", index=False
    )

    recording_inventory = recording_inventory_frame(recordings)
    recording_inventory.to_csv(
        output_dir / "unoccupied_recordings.csv",
        index=False,
        float_format="%.6f",
    )
    for state, state_records in _group_recordings(recordings).items():
        LOGGER.info(
            "State %-55s | n=%d | trials=%s",
            state,
            len(state_records),
            ",".join(record.trial_id for record in state_records),
        )

    statistics = extract_recording_statistics(
        recordings,
        read_feitcsi=read_feitcsi,
        interpolate_low_energy_subcarriers=interpolate_low_energy_subcarriers,
        low_energy_ratio=args.low_energy_ratio,
    )
    save_record_statistics(statistics, output_dir)
    grouped = group_statistics(statistics)
    validate_paper_labels(grouped)
    plot_environment_amplitude_supplements(
        grouped,
        output_dir=output_dir,
        dpi=args.dpi,
        curve_kind="mean",
    )
    plot_environment_amplitude_supplements(
        grouped,
        output_dir=output_dir,
        dpi=args.dpi,
        curve_kind="std",
    )
    correlation_statistics, correlation_pairs = correlation_analysis(
        grouped, output_dir=output_dir, dpi=args.dpi
    )
    plot_intra_state_correlation_summary(
        correlation_statistics,
        output_dir=output_dir,
        dpi=args.dpi,
    )
    (
        distance_statistics,
        distance_pairs,
        overall_distances,
        effect_statistics,
    ) = (
        distance_analysis(
            statistics,
            output_dir=output_dir,
            dpi=args.dpi,
            distance_plot=args.distance_plot,
        )
    )
    nearest_state_frame = nearest_state_separation_analysis(
        pair_frame=distance_pairs,
        state_distance_frame=distance_statistics,
        output_dir=output_dir,
    )
    plot_main_combination_v6(
        pair_frame=distance_pairs,
        state_distance_frame=distance_statistics,
        nearest_state_frame=nearest_state_frame,
        output_dir=output_dir,
        dpi=args.dpi,
    )
    state_inventory = state_inventory_frame(recordings, statistics)
    state_inventory.to_csv(
        output_dir / "unoccupied_state_inventory.csv",
        index=False,
        float_format="%.8f",
    )
    summary = build_summary(
        state_inventory=state_inventory,
        correlation_statistics=correlation_statistics,
        distance_statistics=distance_statistics,
    )
    summary.to_csv(
        output_dir / "unoccupied_validation_summary.csv",
        index=False,
        float_format="%.8f",
    )
    exclusion_ledger = load_exclusion_ledger(dataset_root)
    write_normalization_audit(
        output_dir=output_dir,
        num_recordings=len(recordings),
        low_energy_ratio=args.low_energy_ratio,
    )
    create_report(
        output_dir=output_dir,
        dataset_root=dataset_root,
        field_discovery=field_discovery,
        recording_inventory=recording_inventory,
        state_inventory=state_inventory,
        summary=summary,
        correlation_pairs=correlation_pairs,
        distance_pairs=distance_pairs,
        overall_distances=overall_distances,
        effect_statistics=effect_statistics,
        nearest_state_frame=nearest_state_frame,
        exclusion_ledger=exclusion_ledger,
        low_energy_ratio=args.low_energy_ratio,
        distance_plot=args.distance_plot,
    )
    write_v6_caption(
        output_dir=output_dir,
        pair_frame=distance_pairs,
    )
    write_generated_file_manifest(output_dir)

    LOGGER.info(
        "Completed validation: %d recordings, %d states, %d generated files",
        len(recordings),
        len(grouped),
        sum(1 for path in output_dir.rglob("*") if path.is_file()),
    )
    LOGGER.info("FINAL | Discovered unoccupied states: %d", len(grouped))
    for state in STATE_ORDER:
        if state in grouped:
            LOGGER.info(
                "FINAL | %-42s | n=%d",
                (
                    f"{paper_environment_label(grouped[state][0].recording.environment_id)}"
                    f" — {paper_state_label(state)}"
                ),
                len(grouped[state]),
            )
    LOGGER.info(
        "FINAL | Curve normalization: each recording divided by its own "
        "mean amplitude (not Min-Max or per-recording Z-score)"
    )
    LOGGER.info(
        "FINAL | Feature standardization: feature-wise mean/SD fitted once "
        "across all %d unoccupied recording vectors",
        len(recordings),
    )
    LOGGER.info(
        "FINAL | Revised main figure: %s",
        output_dir / "figure_unoccupied_repeatability_main_v6.pdf",
    )
    log_main_v6_summary(
        output_dir=output_dir,
        dpi=args.dpi,
    )
    for environment_id, stems in SUPPLEMENTARY_ENVIRONMENT_FILES.items():
        for stem in stems:
            LOGGER.info(
                "FINAL | Supplementary figure: %s",
                output_dir / f"{stem}.pdf",
            )
    LOGGER.info(
        "FINAL | Active missing files: none; exclusion-ledger entries: %d",
        len(exclusion_ledger),
    )
    LOGGER.info(
        "FINAL | Living room — Clean n=4 reason: %s",
        clean_living_room_explanation(
            state_inventory, exclusion_ledger
        ),
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
