from __future__ import annotations

from functools import lru_cache
from html import escape
from pathlib import Path
from typing import Any
import math
import re

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import plotly.io as pio

from .gatherParticipantData import (
    CASE_SLICES,
    decode_slice_position,
    iter_case_data,
    iter_grid_datasets,
    parse_ipw3_zone_name,
)
from .heat_flux_computations import (
    CASE_SETTINGS as HEAT_FLUX_CASE_SETTINGS,
    _window_with_interpolated_edges,
    recovery_temperature,
)
from .participant_style import participant_color, participant_legend_rank, participant_marker
from .statistical_metrics import build_grid_level_box_plot, statistics_table_html
from .iceshape_builder import (
    EXPERIMENTAL_ICE_SHAPE_FILES,
    INCHES_TO_METRES,
    extract_roughness_key_from_zone_name as extract_ice_shape_roughness_key,
    find_submitted_ice_xz_columns,
    load_experimental_ice_shape_data,
    naca_clean_reference_points,
    onera_clean_reference_points,
    ordered_clean_reference_columns,
    parse_ipw3_ice_shape_zone_name,
    upper_horn_geometry,
    valid_submitted_ice_shape_rows,
)
from .plot_style import (
    DISTRIBUTION_NORMALIZATION,
    GRID_CONVERGENCE_NORMALIZATION,
    apply_individual_plot_overrides,
    apply_xy_style,
)

GRID_CONVERGENCE_PLOTS: list[dict[str, Any]] = [
    {"plot_key": "cl_vs_n", "title": "CL grid convergence", "x_candidates": ["N"], "y_candidates": ["CL"], "x_label": "h = N<sup>−1/3</sup> [-]", "y_label": "CL [-]", "filename_slug": "cl_vs_n", "group_by_roughness": True},
    {"plot_key": "cd_vs_n", "title": "CD grid convergence", "x_candidates": ["N"], "y_candidates": ["CD"], "x_label": "h = N<sup>−1/3</sup> [-]", "y_label": "CD [-]", "filename_slug": "cd_vs_n", "group_by_roughness": True},
    {"plot_key": "cmy_vs_n", "title": "Pitching moment grid convergence", "x_candidates": ["N"], "y_candidates": ["CMY", "CMZ"], "x_label": "h = N<sup>−1/3</sup> [-]", "y_label": "Pitching moment coefficient [-]", "filename_slug": "cmy_vs_n", "group_by_roughness": True},

    {"plot_key": "water_mass_vs_n", "title": "Water mass grid convergence", "x_candidates": ["N"], "y_candidates": ["WATER_MASS", "WaterMass"], "x_label": "h = N<sup>−1/3</sup> [-]", "y_label": "Water mass [g]", "filename_slug": "water_mass_vs_n", "combined_icing_plot": True},
    {"plot_key": "ice_mass_vs_n", "title": "Ice mass grid convergence", "x_candidates": ["N"], "y_candidates": ["ICE_MASS", "IceMass"], "x_label": "h = N<sup>−1/3</sup> [-]", "y_label": "Ice mass [g]", "filename_slug": "ice_mass_vs_n", "combined_icing_plot": True},
    {"plot_key": "water_evap_mass_vs_n", "title": "Water evaporation mass grid convergence", "x_candidates": ["N"], "y_candidates": ["WATER_EVAP_MASS", "WaterEvapMass"], "x_label": "h = N<sup>−1/3</sup> [-]", "y_label": "Water evaporation mass [g]", "filename_slug": "water_evap_mass_vs_n", "combined_icing_plot": True},
    {"plot_key": "qc_prime", "title": "Integrated convective heat transfer per unit span grid convergence", "x_label": "h = N<sup>−1/3</sup> [-]", "y_label": "Q<sub>c</sub>′ = ∫ HTC (T<sub>s</sub> − T<sub>rec</sub>) ds [W/m]", "filename_slug": "qc_prime_vs_n", "qc_prime_integration_plot": True},
]

WATER_MASS_ANALYSIS_PLOTS: list[dict[str, Any]] = [
    {"plot_key": "ice_to_water_ratio_vs_n", "title": "Ice-to-water mass ratio", "x_candidates": ["N"], "x_label": "h = N<sup>−1/3</sup> [-]", "y_label": "Ice mass / water mass [%]", "filename_slug": "ice_to_water_ratio_vs_n", "derived_icing_ratio": "ice_to_water"},
    {"plot_key": "ice_evap_to_water_ratio_vs_n", "title": "Ice-plus-evaporation to water mass ratio", "x_candidates": ["N"], "x_label": "h = N<sup>−1/3</sup> [-]", "y_label": "(Ice + evaporated water mass) / water mass [%]", "filename_slug": "ice_evap_to_water_ratio_vs_n", "derived_icing_ratio": "ice_plus_evap_to_water"},
]

CFD_GRID_CONVERGENCE_PLOTS = [
    plot_spec for plot_spec in GRID_CONVERGENCE_PLOTS
    if not plot_spec.get("combined_icing_plot", False) and not plot_spec.get("icing_plot", False)
]
ICING_GRID_CONVERGENCE_PLOTS = [
    plot_spec for plot_spec in GRID_CONVERGENCE_PLOTS
    if plot_spec.get("combined_icing_plot", False) or plot_spec.get("icing_plot", False)
]
OPTIONAL_ICING_DIAMETER_PLOTS: list[dict[str, Any]] = [
    {"plot_key": "water_mass_by_diameter_vs_n", "title": "Water mass by droplet diameter grid convergence", "x_candidates": ["N"], "y_candidates": ["WATER_MASS", "WaterMass"], "x_label": "h = N<sup>−1/3</sup> [-]", "y_label": "Water mass [g]", "filename_slug": "water_mass_by_diameter_vs_n", "diameter_plot": True},
    {"plot_key": "ice_mass_by_diameter_vs_n", "title": "Ice mass by droplet diameter grid convergence", "x_candidates": ["N"], "y_candidates": ["ICE_MASS", "IceMass"], "x_label": "h = N<sup>−1/3</sup> [-]", "y_label": "Ice mass [g]", "filename_slug": "ice_mass_by_diameter_vs_n", "diameter_plot": True},
    {"plot_key": "water_evap_mass_by_diameter_vs_n", "title": "Water evaporation mass by droplet diameter grid convergence", "x_candidates": ["N"], "y_candidates": ["WATER_EVAP_MASS", "WaterEvapMass"], "x_label": "h = N<sup>−1/3</sup> [-]", "y_label": "Water evaporation mass [g]", "filename_slug": "water_evap_mass_by_diameter_vs_n", "diameter_plot": True},
]

GRID_SPACING_COLUMN = "CHARACTERISTIC_GRID_SPACING"
GRID_SPACING_AXIS_TITLE = "h = N<sup>−1/3</sup> [-]"
VARIABLE_FILTER: set[str] | None = None
INCLUDE_QC = True


def set_variable_filter(variables: set[str] | None) -> None:
    global VARIABLE_FILTER
    VARIABLE_FILTER = variables


def set_include_qc(include: bool) -> None:
    global INCLUDE_QC
    INCLUDE_QC = include


def apply_participant_mass_conventions(participants) -> None:
    """Apply participant-specific integrated-mass definitions before plotting."""
    for participant in participants:
        if str(participant.participant_id).split(".", 1)[0].zfill(3) != "007":
            continue
        for case_id, case_data in participant.cases.items():
            if not case_id.startswith("TC_NACA0012_") or case_data.grid_convergence_data is None:
                continue
            for zone in case_data.grid_convergence_data.zones.values():
                water_column = find_column_case_insensitive(zone.data.columns, ["WATER_MASS", "WaterMass"])
                ice_column = find_column_case_insensitive(zone.data.columns, ["ICE_MASS", "IceMass"])
                evaporation_column = find_column_case_insensitive(
                    zone.data.columns, ["WATER_EVAP_MASS", "WaterEvapMass"],
                )
                if water_column is None or ice_column is None or evaporation_column is None:
                    continue
                water = pd.to_numeric(zone.data[water_column], errors="coerce")
                ice = pd.to_numeric(zone.data[ice_column], errors="coerce")
                valid = np.isfinite(water) & np.isfinite(ice) & (water > -998.0) & (ice > -998.0)
                zone.data.loc[valid, evaporation_column] = water[valid] - ice[valid]


def plot_matches_variable_filter(plot_spec: dict[str, Any]) -> bool:
    if VARIABLE_FILTER is None:
        return True
    plot_key = plot_spec["plot_key"].lower()
    aliases = {plot_key, plot_key.removesuffix("_vs_n"), plot_key.replace("_by_diameter_vs_n", "")}
    if plot_key == "qc_prime":
        aliases.update({"qc", "q_c", "q_c_prime", "integrated_convective_heat_transfer"})
    elif plot_key.startswith("water_evap_mass"):
        aliases.update({"evaporation", "water_evaporation", "water_evap_mass"})
    elif plot_key.startswith("water_mass"):
        aliases.update({"water", "water_mass"})
    elif plot_key.startswith("ice_mass"):
        aliases.update({"ice", "ice_mass"})
    elif plot_key in {"ice_to_water_ratio_vs_n", "ice_evap_to_water_ratio_vs_n"}:
        aliases.update({"ratio", "water_fate", "icing_ratio", "mass_ratio"})
    return bool(aliases & VARIABLE_FILTER)


def grid_cell_reference_path_for_case(case_id: str) -> Path | None:
    if "ONERAM6" in case_id.upper():
        return Path("R00_REFERENCE") / "ONERAM6_GRID.dat"
    if "NACA0012" in case_id.upper():
        return Path("R00_REFERENCE") / "NACA0012_GRID.dat"
    return None


@lru_cache(maxsize=None)
def load_grid_cell_counts(reference_path_text: str) -> dict[int, float]:
    grid_cell_counts: dict[int, float] = {}
    for line in Path(reference_path_text).read_text().splitlines():
        stripped = line.strip()
        if not stripped or stripped.upper().startswith("VARIABLES"):
            continue
        values = stripped.split()
        if len(values) < 2:
            continue
        try:
            grid_level = int(float(values[0]))
            num_cells = float(values[1])
        except ValueError:
            continue
        if grid_level > 0 and num_cells > 0.0:
            grid_cell_counts[grid_level] = num_cells
    return grid_cell_counts


def grid_cell_counts_for_case(case_id: str) -> dict[int, float]:
    reference_path = grid_cell_reference_path_for_case(case_id)
    if reference_path is None or not reference_path.exists():
        return {}
    return load_grid_cell_counts(str(reference_path))

def extract_icing_bin_set_from_zone_name(zone_name: str) -> str | None:
    match = re.search(r"_Icing_(?P<bin_set>BINS\d+)(?:_roughness_.+)?$", zone_name, re.IGNORECASE)
    if match is None:
        return None
    return match.group("bin_set").upper()


def extract_icing_roughness_key_from_zone_name(zone_name: str) -> str:
    match = re.search(r"_roughness_(?P<roughness>.+?)(?:_by_diameter)?$", zone_name, re.IGNORECASE)
    return match.group("roughness") if match is not None else "unspecified"


def bin_count_from_bin_set(bin_set: str) -> int | None:
    match = re.search(r"\d+", bin_set)
    if match is None:
        return None

    bin_count = int(match.group(0))
    return bin_count if bin_count > 0 else None


def display_bin_set(bin_set: str) -> str:
    """Return a compact user-facing distribution label (for example, 07)."""
    match = re.search(r"\d+", str(bin_set))
    return match.group(0).zfill(2) if match is not None else str(bin_set)


def is_required_grid_convergence_zone(zone) -> bool:
    requirement_column = find_column_case_insensitive(zone.data.columns, ["SOURCE_REQUIREMENT", "SourceRequirement"])
    if requirement_column is None:
        return True

    return any(str(value).strip().lower() == "required" for value in zone.data[requirement_column])


def grid_convergence_zone_matches_requirement(zone, requirement: str) -> bool:
    requirement_column = find_column_case_insensitive(zone.data.columns, ["SOURCE_REQUIREMENT", "SourceRequirement"])
    if requirement_column is None:
        return requirement == "required"
    values = {str(value).strip().lower() for value in zone.data[requirement_column]}
    return requirement.lower() in values

def slugify(text: str) -> str:
    text = text.strip().lower()
    text = re.sub(r"[^a-z0-9]+", "_", text)
    return text.strip("_") or "figure"

def extract_roughness_key_from_zone_name(zone_name: str) -> str | None:
    match = re.search(r"_CFD_roughness_(?P<roughness>.+)$", zone_name, re.IGNORECASE)
    if match is None:
        return None
    return match.group("roughness")


def format_roughness_title(roughness_key: str) -> str:
    if roughness_key == "smooth":
        return "No Roughness"

    if roughness_key == "variable_roughness":
        return "Variable roughness"

    if roughness_key.endswith("mm"):
        value = roughness_key[:-2]
        return f"Roughness height = {value} mm"

    return roughness_key


def format_icing_roughness_title(roughness_key: str) -> str | None:
    """Return a visible icing roughness label, omitting absent metadata."""
    if not roughness_key or roughness_key == "unspecified":
        return None
    return format_roughness_title(roughness_key)


def roughness_sort_key(roughness_key: str) -> tuple[int, float, str]:
    if roughness_key == "smooth":
        return (0, 0.0, roughness_key)

    if roughness_key == "variable_roughness":
        return (2, 0.0, roughness_key)

    if roughness_key.endswith("mm"):
        try:
            return (1, float(roughness_key[:-2]), roughness_key)
        except ValueError:
            pass

    return (3, 0.0, roughness_key)

def find_column_case_insensitive(columns, candidates: list[str]) -> str | None:
    lookup = {column.lower(): column for column in columns}
    for candidate in candidates:
        if candidate.lower() in lookup:
            return lookup[candidate.lower()]
    return None


def valid_numeric_rows(dataframe, *columns: str, positive_columns: set[str] | None = None):
    positive_columns = positive_columns or set()
    valid_mask = pd.Series(True, index=dataframe.index)

    for column in columns:
        values = pd.to_numeric(dataframe[column], errors="coerce")
        column_mask = values.notna() & (values > -998.0)
        if column in positive_columns:
            column_mask &= values > 0.0
        valid_mask &= column_mask

    return dataframe.loc[valid_mask]


def case_ordered_y_candidates(case_id: str, candidates: list[str]) -> list[str]:
    """Prefer the pitching-moment component used by each test case."""
    if "CMY" not in candidates or "CMZ" not in candidates:
        return candidates

    case_id_upper = case_id.upper()
    if "NACA0012" in case_id_upper:
        return ["CMZ", "CMY"]
    if "ONERAM6" in case_id_upper:
        return ["CMY", "CMZ"]
    return candidates


def participant_label(participant) -> str:
    """Return the legend label for a participant-level grid-convergence file.

    Grid convergence is stored under the test case, not under one grid level or
    dataset attempt. Therefore the legend uses only PID.
    """
    return f"{participant.participant_id}"


def format_x_hover_label(x_column: str) -> str:
    return "h = N^(-1/3)" if x_column == GRID_SPACING_COLUMN else x_column


def grid_convergence_coordinate(num_cells: float, l1_num_cells: float) -> float:
    """Return the characteristic spacing h = N^(-1/3) for a 3-D mesh."""
    del l1_num_cells  # Retained in the signature for existing callers.
    return num_cells ** (-1.0 / 3.0)


def grid_level_number_from_value(value: Any) -> int | None:
    match = re.search(r"\d+", str(value))
    if match is None:
        return None
    try:
        return int(match.group(0))
    except ValueError:
        return None


def add_grid_spacing_column(data, case_id: str, x_column: str, grid_column: str | None = None):
    grid_cell_counts = grid_cell_counts_for_case(case_id)
    l1_num_cells = grid_cell_counts.get(1)
    if not grid_cell_counts or l1_num_cells is None:
        return data.iloc[0:0].copy()

    working_data = data.copy()
    grid_levels: list[str | None] = []
    num_cells_values: list[float | None] = []
    grid_spacing_values: list[float | None] = []

    for _, row in working_data.iterrows():
        level_number = None
        if grid_column is not None and grid_column in working_data.columns:
            level_number = grid_level_number_from_value(row[grid_column])
        if level_number is None:
            level_number = grid_level_number_from_value(row[x_column])

        num_cells = grid_cell_counts.get(level_number) if level_number is not None else None
        if level_number is None or num_cells is None:
            grid_levels.append(None)
            num_cells_values.append(None)
            grid_spacing_values.append(None)
            continue

        grid_levels.append(f"L{level_number}")
        num_cells_values.append(num_cells)
        grid_spacing_values.append(grid_convergence_coordinate(num_cells, l1_num_cells))

    working_data["GRID_LEVEL_DISPLAY"] = grid_levels
    working_data["GRID_CELL_COUNT"] = num_cells_values
    working_data[GRID_SPACING_COLUMN] = grid_spacing_values
    return working_data.dropna(subset=[GRID_SPACING_COLUMN])


def plotly_config(filename: str) -> dict[str, Any]:
    return {
        "responsive": True,
        "displaylogo": False,
        "toImageButtonOptions": {"format": "png", "filename": filename, "height": 700, "width": 2000, "scale": 3},
    }


DEFER_PLOTLY_DIR: Path | None = None
PNG_EXPORT_DIR: Path | None = None
PNG_EXPORT_QUEUE: list[tuple[go.Figure, Path]] = []


def set_defer_plotly_html(output_dir: Path | None) -> None:
    global DEFER_PLOTLY_DIR
    DEFER_PLOTLY_DIR = output_dir


def set_png_export_dir(output_dir: Path | None) -> None:
    global PNG_EXPORT_DIR
    PNG_EXPORT_DIR = output_dir


def clear_png_export_queue() -> None:
    PNG_EXPORT_QUEUE.clear()


def flush_png_exports(scale: int = 3, width: int = 2000, height: int = 700) -> None:
    if not PNG_EXPORT_QUEUE:
        return
    figures, paths = zip(*PNG_EXPORT_QUEUE)
    pio.write_images(list(figures), list(paths), width=width, height=height, scale=scale)
    PNG_EXPORT_QUEUE.clear()


def figure_to_html_div(fig: go.Figure, filename: str, plot_title: str) -> str:
    if PNG_EXPORT_DIR is not None:
        # PNG deliverables contain submitted/absolute values only. Relative
        # panels and across-participant box plots remain available in HTML.
        if any(token in filename for token in ("_relative_to_l1", "_relative_to_bins15", "_statistics_boxplot")):
            return ""
        PNG_EXPORT_DIR.mkdir(parents=True, exist_ok=True)
        png_fig = go.Figure(fig)
        png_fig.update_layout(
            title=dict(text=plot_title, x=0.5, xanchor="center"),
            margin=dict(t=100),
        )
        PNG_EXPORT_QUEUE.append((png_fig, PNG_EXPORT_DIR / f"{filename}.png"))
        return ""
    figure_html = fig.to_html(full_html=False, include_plotlyjs="cdn", config=plotly_config(filename))
    # Plotly serializes legend names, hover templates, tick labels, and zone
    # metadata into this fragment. Normalize all visible distribution labels
    # at the output boundary without changing internal BINSxx lookup keys.
    figure_html = re.sub(r"BINS(01|03|07|15)", r"\1", figure_html, flags=re.IGNORECASE)
    plot_title = re.sub(r"BINS(01|03|07|15)", r"\1", plot_title, flags=re.IGNORECASE)
    if DEFER_PLOTLY_DIR is not None:
        DEFER_PLOTLY_DIR.mkdir(parents=True, exist_ok=True)
        fragment_path = DEFER_PLOTLY_DIR / f"{filename}.html"
        fragment_path.write_text(
            f'<!DOCTYPE html><html><head><meta charset="UTF-8"><meta name="viewport" content="width=device-width, initial-scale=1.0"><style>html,body{{margin:0;background:white}} .plotly-graph-div{{width:100%}}</style></head><body>{figure_html}</body></html>',
            encoding="utf-8",
        )
        return f'<iframe class="plotly-lazy-frame" data-plot-src="PLOTS/{escape(filename)}.html" title="{escape(filename)}"></iframe><div class="plot-loading">Plot queued…</div>'
    return f"""
    <div class="plot-download-shell" data-plot-filename="{escape(filename)}" data-plot-title="{escape(plot_title)}" data-download-width="2000" data-download-height="700">
      {figure_html}
      <div class="plot-download-actions">
        <button type="button" data-plot-download="with-legend">Download PNG with legend</button>
        <button type="button" data-plot-download="without-legend">Download PNG without legend</button>
      </div>
    </div>
    """


def style_xy_figure(fig: go.Figure, case_id: str, plot_key: str, x_label: str, y_label: str, height: int = 430) -> go.Figure:
    apply_xy_style(fig, case_id, x_label, y_label, plot_family="convergence", plot_key=plot_key, height=height)
    fig.update_xaxes(type="linear", tickformat=".4g", automargin=True)
    fig.update_yaxes(automargin=True)
    fig.update_layout(margin=dict(l=90, r=220, t=30, b=95))
    return fig


def style_grid_level_x_axis(fig: go.Figure, case_id: str) -> go.Figure:
    """Plot characteristic spacing h = N^(-1/3) on a logarithmic axis."""
    counts = grid_cell_counts_for_case(case_id)
    levels = sorted(counts)
    if not levels:
        return fig
    l1_num_cells = counts.get(1)
    if l1_num_cells is None:
        return fig
    tick_values = [grid_convergence_coordinate(counts[level], l1_num_cells) for level in levels]
    log_tick_values = [math.log10(value) for value in tick_values]
    log_range = [math.floor(min(log_tick_values)), math.ceil(max(log_tick_values))]
    if log_range[0] == log_range[1]:
        log_range[1] += 1
    fig.update_xaxes(
        type="log",
        tickmode="linear",
        dtick=1,
        tickformat=".0e",
        exponentformat="power",
        showexponent="all",
        range=log_range,
        showgrid=True,
        minor={"showgrid": True, "dtick": "D1"},
        title=dict(
            text=GRID_CONVERGENCE_NORMALIZATION.get(
                "grid_coordinate_axis_title",
                GRID_SPACING_AXIS_TITLE,
            ),
            font=dict(size=18),
        ),
    )
    return fig


def harmonize_comparison_figure(fig: go.Figure, x_order: list[str], x_title: str) -> go.Figure:
    """Apply the shared compact style used by paired convergence plots."""
    fig.update_layout(
        height=360,
        margin={"l": 110, "r": 20, "t": 80, "b": 75},
        legend={
            "orientation": "h", "x": 0.0, "xanchor": "left",
            "y": 1.02, "yanchor": "bottom", "font": {"size": 10},
        },
    )
    fig.update_xaxes(
        type="category", categoryorder="array", categoryarray=x_order,
        tickmode="array", tickvals=x_order, ticktext=x_order,
        range=None, autorange=True, automargin=False,
        title={"text": x_title},
    )
    fig.update_yaxes(automargin=False)
    return fig


def style_relative_difference_figure(fig: go.Figure) -> go.Figure:
    """Visually distinguish percentage-difference panels from absolute ones."""
    for trace in fig.data:
        if getattr(trace, "line", None) is not None:
            trace.line.dash = "solid"
            trace.line.width = 2
        if getattr(trace, "marker", None) is not None:
            trace.marker.symbol = "circle"
            trace.marker.line = {"width": 1.5}
    fig.update_layout(plot_bgcolor="#f7f9fc")
    return fig


def style_inverse_bin_figure(fig: go.Figure, case_id: str, plot_key: str, y_label: str, height: int = 430) -> go.Figure:
    apply_xy_style(fig, case_id, "Droplet distribution", y_label, plot_family="convergence", plot_key=plot_key, height=height)
    fig.update_xaxes(type="log", range=[-1.25, 0.05], tickmode="array", tickvals=[1.0 / 15.0, 1.0 / 7.0, 1.0 / 3.0, 1.0], ticktext=["15-bin", "7-bin", "3-bin", "1-bin"])
    fig.update_layout(margin=dict(l=90, r=220, t=30, b=95))
    return fig


def normalize_grid_convergence_to_l1(fig: go.Figure) -> list[str]:
    """Convert each trace to signed percent difference from its L1 value."""
    settings = GRID_CONVERGENCE_NORMALIZATION
    if not settings.get("enabled", True):
        return []

    reference_level = str(settings.get("reference_grid_level", "L1")).upper()
    grid_order = [str(level).upper() for level in settings.get("grid_order", ["L1", "L2", "L3", "L4"])]
    hover_label = str(settings.get("hover_label", "Relative to L1"))
    hover_format = str(settings.get("hover_format", ".4g"))
    missing_references: list[str] = []
    invalid_traces: list[Any] = []
    all_relative_values: list[float] = []

    for trace in fig.data:
        if trace.y is None or trace.customdata is None:
            continue

        original_y = list(trace.y)
        custom_rows = [list(row) for row in trace.customdata]
        original_value_index = len(custom_rows[0]) if custom_rows else 0
        reference_value: float | None = None
        for row, value in zip(custom_rows, original_y):
            if not row or str(row[0]).strip().upper() != reference_level:
                continue
            try:
                candidate = float(value)
            except (TypeError, ValueError):
                continue
            if math.isfinite(candidate) and candidate != 0.0:
                reference_value = candidate
                break

        if reference_value is None:
            missing_references.append(str(trace.name or "unnamed trace"))
            invalid_traces.append(trace)
            continue

        normalized_y: list[float | None] = []
        for row, value in zip(custom_rows, original_y):
            try:
                original_value = float(value)
            except (TypeError, ValueError):
                original_value = math.nan
            row.append(original_value if math.isfinite(original_value) else None)
            relative_value = (
                (original_value - reference_value) / reference_value * 100.0
                if math.isfinite(original_value)
                else None
            )
            normalized_y.append(relative_value)
            if relative_value is not None and math.isfinite(relative_value):
                all_relative_values.append(relative_value)

        rows_by_level = {
            str(row[0]).strip().upper(): (x_value, row, relative_value)
            for x_value, row, relative_value in zip(trace.x, custom_rows, normalized_y)
        }
        ordered_levels = [level for level in grid_order if level in rows_by_level]
        trace.x = [rows_by_level[level][0] for level in ordered_levels]
        trace.customdata = [rows_by_level[level][1] for level in ordered_levels]
        trace.y = [rows_by_level[level][2] for level in ordered_levels]
        trace.mode = "lines+markers"
        hover = str(trace.hovertemplate or "")
        hover = re.sub(
            r"%\{y(?::[^}]*)?\}",
            f"%{{customdata[{original_value_index}]}}",
            hover,
        )
        relative_hover = f"{escape(hover_label)}=%{{y:{hover_format}}}%<br>"
        trace.hovertemplate = hover.replace("<extra>", relative_hover + "<extra>")

    if invalid_traces:
        invalid_trace_ids = {id(trace) for trace in invalid_traces}
        fig.data = tuple(trace for trace in fig.data if id(trace) not in invalid_trace_ids)

    initial_range = settings.get("initial_y_range", [-5.0, 5.0])
    initial_bound = max(abs(float(initial_range[0])), abs(float(initial_range[1])))
    data_bound = max((abs(value) for value in all_relative_values), default=0.0)
    padding = 1.0 + float(settings.get("range_padding_fraction", 0.10))
    y_bound = max(initial_bound, data_bound * padding)
    fig.update_yaxes(range=[-y_bound, y_bound], autorange=False)
    reference_line = settings.get("reference_line", {})
    fig.add_hline(
        y=0.0,
        line_color=reference_line.get("color", "black"),
        line_width=reference_line.get("width", 1.5),
        line_dash=reference_line.get("dash", "dash"),
    )
    return []


def grid_statistics_html(
    fig: go.Figure,
    filename: str,
    plot_title: str,
    case_id: str | None = None,
    naca_no_roughness_only: bool = False,
    separate_parts: bool = False,
    compact_parts: bool = False,
) -> str | tuple[str, str]:
    """Render per-grid statistics and a box plot across visible traces."""
    statistics_source = go.Figure(fig)
    naca_smooth_only = bool(
        naca_no_roughness_only and case_id and "NACA0012" in case_id.upper()
    )
    if naca_smooth_only:
        # An absent roughness label is the normal no-roughness CFD/baseline
        # trace. Explicit roughness-height and variable-roughness traces remain
        # visible in the main plot but are excluded from NACA statistics.
        def is_naca_smooth_trace(trace) -> bool:
            name = str(getattr(trace, "name", "") or "").lower()
            meta = getattr(trace, "meta", None)
            stored_labels = meta.get("ipw3_roughness_labels", []) if isinstance(meta, dict) else []
            roughness_text = " ".join(str(value).lower() for value in stored_labels) or name
            return "roughness" not in roughness_text or "no roughness" in roughness_text

        statistics_source.data = tuple(trace for trace in statistics_source.data if is_naca_smooth_trace(trace))
    y_axis_title = str(statistics_source.layout.yaxis.title.text or "Value")
    box_fig, statistics = build_grid_level_box_plot(statistics_source, y_axis_title)
    if not box_fig.data:
        return ""
    if compact_parts:
        box_fig.update_layout(
            height=360,
            margin={"l": 110, "r": 20, "t": 80, "b": 75},
        )
    table_html = f"""
    {'<p class="plot-description">Statistical sample: NACA0012 no-roughness participants only.</p>' if naca_smooth_only else ''}
    {statistics_table_html(statistics, y_axis_title)}
    """
    box_plot_html = figure_to_html_div(
        box_fig,
        filename=f"{filename}_statistics_boxplot",
        plot_title=f"{plot_title} | Across-participant box and whisker",
    )
    if separate_parts:
        return table_html, box_plot_html
    return table_html + box_plot_html


NACA0012_MASS_PLOT_KEYS = {
    "water_mass_vs_n", "ice_mass_vs_n", "water_evap_mass_vs_n",
    "water_mass_by_diameter_vs_n", "ice_mass_by_diameter_vs_n",
    "water_evap_mass_by_diameter_vs_n",
}

NACA0012_EXPERIMENTAL_ICE_MASS_G = {
    "TC_NACA0012_AE3932": 101.0,
    "TC_NACA0012_AE3933": 85.9,
}


def scale_naca0012_mass_figure_to_grams(fig: go.Figure, case_id: str, plot_key: str) -> None:
    """Convert submitted kilogram trace values to grams at the output boundary."""
    if not case_id.startswith("TC_NACA0012_") or plot_key not in NACA0012_MASS_PLOT_KEYS:
        return
    for trace in fig.data:
        if trace.y is not None:
            trace.y = [float(value) * 1000.0 for value in trace.y]


def grid_convergence_figure_pair_html(
    fig: go.Figure,
    case_id: str,
    plot_key: str,
    filename: str,
    plot_title: str,
) -> str:
    """Render the original convergence plot followed by its L1-relative copy."""
    scale_naca0012_mass_figure_to_grams(fig, case_id, plot_key)
    grid_order = [
        str(level).upper()
        for level in GRID_CONVERGENCE_NORMALIZATION.get("grid_order", ["L1", "L2", "L3", "L4"])
    ]
    harmonize_comparison_figure(
        fig, grid_order,
        GRID_CONVERGENCE_NORMALIZATION.get("grid_axis_title", "Grid level"),
    )
    style_grid_level_x_axis(fig, case_id)
    # Keep physical reference lines on the absolute plot only; they are not
    # participant series and must not enter L1 normalization or statistics.
    participant_fig = go.Figure(fig)
    if plot_key == "ice_mass_vs_n" and case_id in NACA0012_EXPERIMENTAL_ICE_MASS_G:
        fig.add_hline(
            y=NACA0012_EXPERIMENTAL_ICE_MASS_G[case_id],
            line_color="black",
            line_width=1.5,
            line_dash="dash",
        )
    absolute_html = figure_to_html_div(fig, filename=filename, plot_title=plot_title)
    # Experimental horn angles are constant physical reference lines.  Keep
    # them on the raw plot, but do not treat them as participant grid series
    # in either across-participant statistics or L1-relative normalization.
    participant_fig.data = tuple(
        trace for trace in participant_fig.data
        if not str(getattr(trace, "legendgroup", "") or "").startswith("horn_reference_")
    )
    statistics_result = grid_statistics_html(
        participant_fig,
        filename,
        plot_title,
        case_id=case_id,
        naca_no_roughness_only=plot_key in {"cl_vs_n", "cd_vs_n", "cmy_vs_n", "qc_prime"},
        separate_parts=True,
        compact_parts=True,
    )
    if isinstance(statistics_result, tuple):
        statistics_table, box_plot_html = statistics_result
        box_html = statistics_table + box_plot_html
    else:
        statistics_table, box_plot_html = "", ""
        box_html = statistics_result
    if not GRID_CONVERGENCE_NORMALIZATION.get("enabled", True):
        return absolute_html + box_html

    relative_fig = go.Figure(participant_fig)
    missing_l1 = normalize_grid_convergence_to_l1(relative_fig)
    relative_key = f"{plot_key}_relative"
    apply_individual_plot_overrides(relative_fig, case_id, relative_key)
    harmonize_comparison_figure(
        relative_fig, grid_order,
        GRID_CONVERGENCE_NORMALIZATION.get("grid_axis_title", "Grid level"),
    )
    style_grid_level_x_axis(relative_fig, case_id)
    if plot_key in {"cl_vs_n", "cd_vs_n", "cmy_vs_n"}:
        style_relative_difference_figure(relative_fig)
        # Absolute coefficient plots use very small fixed tick intervals.
        # Those intervals are invalid for percentage differences and would
        # generate hundreds of overlapping labels, so let Plotly choose a
        # compact percentage scale independently for each relative plot.
        relative_fig.update_yaxes(
            dtick=None,
            tickmode="auto",
            nticks=7,
            tickformat=".3g",
            exponentformat="none",
        )
    relative_title = (
        "Grid Convergence of Integrated Lift Coefficient | Signed Relative Difference from L1"
        if plot_key == "cl_vs_n"
        else f"{plot_title} | Signed relative difference from L1"
    )
    relative_html = figure_to_html_div(
        relative_fig,
        filename=f"{filename}_relative_to_l1",
        plot_title=relative_title,
    )
    missing_html = ""
    if missing_l1:
        missing_html = '<ul class="plot-notes">' + "".join(
            f"<li>{escape(name)} was hidden because no valid L1 reference was available.</li>"
            for name in sorted(set(missing_l1))
        ) + "</ul>"

    # Shared CFD/icing convergence layout: raw and signed-relative plots in
    # the first row, then the statistics table and box plot in the second.
    linked_legend_attribute = ' data-linked-participant-legend="grid-convergence"'
    return f"""
        <div class="convergence-value-pair convergence-two-by-two-comparison"{linked_legend_attribute}>
          <div class="convergence-comparison-grid">
            <section class="convergence-comparison-panel">
              {absolute_html}
            </section>
            <section class="convergence-comparison-panel">
              {missing_html}
              {relative_html}
            </section>
          </div>
          <div class="convergence-statistics-grid">
            <section class="convergence-comparison-table">
              {statistics_table}
            </section>
            <section class="convergence-comparison-panel">
              {box_plot_html}
            </section>
          </div>
        </div>
        """


def normalize_distribution_to_bins15(fig: go.Figure) -> list[str]:
    """Convert distribution traces to signed percent difference from BINS15."""
    settings = DISTRIBUTION_NORMALIZATION
    if not settings.get("enabled", True):
        return []
    reference_bin = str(settings.get("reference_bin_set", "BINS15")).upper()
    bin_order = [str(value).upper() for value in settings.get("bin_order", ["BINS15", "BINS07", "BINS03", "BINS01"])]
    missing_references: list[str] = []
    invalid_traces: list[Any] = []
    all_values: list[float] = []

    for trace in fig.data:
        if trace.y is None or trace.customdata is None:
            continue
        original_y = list(trace.y)
        custom_rows = [list(row) for row in trace.customdata]
        original_index = len(custom_rows[0]) if custom_rows else 0
        reference_value: float | None = None
        for row, value in zip(custom_rows, original_y):
            if row and str(row[0]).upper() == reference_bin:
                try:
                    candidate = float(value)
                except (TypeError, ValueError):
                    continue
                if math.isfinite(candidate) and candidate != 0.0:
                    reference_value = candidate
                    break
        if reference_value is None:
            missing_references.append(str(trace.name or "unnamed trace"))
            invalid_traces.append(trace)
            continue

        rows_by_bin: dict[str, tuple[list[Any], float | None]] = {}
        for row, value in zip(custom_rows, original_y):
            try:
                original_value = float(value)
            except (TypeError, ValueError):
                original_value = math.nan
            row.append(original_value if math.isfinite(original_value) else None)
            relative_value = (
                (original_value - reference_value) / reference_value * 100.0
                if math.isfinite(original_value)
                else None
            )
            if relative_value is not None:
                all_values.append(relative_value)
            rows_by_bin[str(row[0]).upper()] = (row, relative_value)

        ordered_bins = [bin_set for bin_set in bin_order if bin_set in rows_by_bin]
        trace.x = ordered_bins
        trace.customdata = [rows_by_bin[bin_set][0] for bin_set in ordered_bins]
        trace.y = [rows_by_bin[bin_set][1] for bin_set in ordered_bins]
        trace.mode = "lines+markers"
        hover = str(trace.hovertemplate or "")
        hover = re.sub(r"%\{y(?::[^}]*)?\}", f"%{{customdata[{original_index}]}}", hover)
        hover_label = str(settings.get("hover_label", "Relative difference from BINS15"))
        hover_format = str(settings.get("hover_format", ".4g"))
        trace.hovertemplate = hover.replace(
            "<extra>", f"{escape(hover_label)}=%{{y:{hover_format}}}%<br><extra>"
        )

    if invalid_traces:
        invalid_trace_ids = {id(trace) for trace in invalid_traces}
        fig.data = tuple(trace for trace in fig.data if id(trace) not in invalid_trace_ids)

    initial_range = settings.get("initial_y_range", [-5.0, 5.0])
    initial_bound = max(abs(float(initial_range[0])), abs(float(initial_range[1])))
    data_bound = max((abs(value) for value in all_values), default=0.0)
    y_bound = max(initial_bound, data_bound * (1.0 + float(settings.get("range_padding_fraction", 0.10))))
    fig.update_xaxes(
        type="category", categoryorder="array", categoryarray=bin_order,
        tickmode="array", tickvals=bin_order, ticktext=bin_order, range=None,
        title={"text": settings.get("axis_title", "Droplet distribution")},
    )
    fig.update_yaxes(range=[-y_bound, y_bound], autorange=False)
    line = settings.get("reference_line", {})
    fig.add_hline(y=0.0, line_color=line.get("color", "black"), line_width=line.get("width", 1.5), line_dash=line.get("dash", "dash"))
    return []


def distribution_figure_pair_html(
    fig: go.Figure,
    case_id: str,
    plot_key: str,
    filename: str,
    plot_title: str,
) -> str:
    """Render original fixed-grid distribution values plus BINS15 differences."""
    scale_naca0012_mass_figure_to_grams(fig, case_id, plot_key)
    bin_order = [
        str(value).upper()
        for value in DISTRIBUTION_NORMALIZATION.get("bin_order", ["BINS15", "BINS07", "BINS03", "BINS01"])
    ]
    for trace in fig.data:
        if trace.customdata is None or trace.x is None:
            continue
        bins = [str(row[0]).strip().upper() for row in trace.customdata]
        if len(bins) == len(trace.x):
            trace.x = bins
    harmonize_comparison_figure(fig, bin_order, "Droplet distribution")
    absolute_html = figure_to_html_div(fig, filename=filename, plot_title=plot_title)
    if not DISTRIBUTION_NORMALIZATION.get("enabled", True):
        return absolute_html
    relative_fig = go.Figure(fig)
    # Experimental upper-horn references are fixed physical values, not
    # submitted distribution series, so they belong only on the raw plot.
    relative_fig.data = tuple(
        trace for trace in relative_fig.data
        if not str(getattr(trace, "legendgroup", "") or "").startswith("horn_reference_")
    )
    missing = normalize_distribution_to_bins15(relative_fig)
    apply_individual_plot_overrides(relative_fig, case_id, f"{plot_key}_relative_to_bins15")
    harmonize_comparison_figure(relative_fig, bin_order, "Droplet distribution")
    relative_html = figure_to_html_div(
        relative_fig,
        filename=f"{filename}_relative_to_bins15",
        plot_title=f"{plot_title} | Signed relative difference from 15",
    )
    notes = ""
    if missing:
        notes = '<ul class="plot-notes">' + "".join(
            f"<li>{escape(name)} was hidden because no valid BINS15 reference was available.</li>"
            for name in sorted(set(missing))
        ) + "</ul>"
    return f"""
    <div class="distribution-value-pair convergence-comparison-grid">
      <section class="convergence-comparison-panel">{absolute_html}</section>
      <section class="convergence-comparison-panel">{notes}{relative_html}</section>
    </div>
    """


def build_grid_convergence_figure(participants, case_id: str, plot_spec: dict[str, Any], roughness_filter: str | None = None, requirement: str = "required") -> tuple[go.Figure, int, list[str]]:
    seen_trace_keys: set[tuple[str, str, str]] = set()

    fig = go.Figure()
    trace_count = 0
    is_diameter_plot = plot_spec.get("diameter_plot", False)
    skipped_notes: list[str] = []

    for participant, case_data in iter_case_data(participants, case_id):
        if case_data.grid_convergence_data is None:
            continue
        label = participant_label(participant)
        color = participant_color(participant.participant_id)
        participant_trace_count = 0
        participant_had_matching_zone = False
        participant_had_variable = False

        for zone_name, zone in case_data.grid_convergence_data.zones.items():
            if not grid_convergence_zone_matches_requirement(zone, requirement):
                continue
            zone_is_diameter = "by_diameter" in zone_name.lower()
            roughness_key = extract_roughness_key_from_zone_name(zone_name)
            if roughness_filter is not None:
                if roughness_key != roughness_filter:
                    continue
            if is_diameter_plot != zone_is_diameter:
                continue

            participant_had_matching_zone = True
            x_column = find_column_case_insensitive(zone.data.columns, plot_spec["x_candidates"])
            y_column = find_column_case_insensitive(zone.data.columns, case_ordered_y_candidates(case_id, plot_spec["y_candidates"]))
            grid_column = find_column_case_insensitive(zone.data.columns, ["GRID_LEVEL", "GridLevel"])
            if x_column is None or y_column is None:
                continue
            participant_had_variable = True

            data = valid_numeric_rows(zone.data[[x_column, y_column]].copy(), x_column, y_column, positive_columns={x_column})
            data = add_grid_spacing_column(data, case_id, x_column, grid_column=grid_column)
            
            if data.empty:
                continue

            data = data.sort_values(GRID_SPACING_COLUMN)
            
            if is_diameter_plot:
                diameter_column = find_column_case_insensitive(zone.data.columns, ["DIAMETER", "Diameter"])
                bin_set_column = find_column_case_insensitive(zone.data.columns, ["BIN_SET", "BinSet"])
                if diameter_column is None:
                    continue

                working_data = zone.data.loc[data.index, [x_column, y_column, diameter_column] + ([bin_set_column] if bin_set_column is not None else [])].copy()
                working_data = add_grid_spacing_column(working_data, case_id, x_column, grid_column=grid_column)
                working_data = valid_numeric_rows(working_data, diameter_column)

                for diameter, diameter_data in working_data.groupby(diameter_column):
                    diameter_data = diameter_data.sort_values(GRID_SPACING_COLUMN)
                    if diameter_data.empty:
                        continue
                    bin_set = ""
                    if bin_set_column is not None:
                        bin_values = sorted(set(str(value) for value in diameter_data[bin_set_column]))
                        bin_set = f" {'/'.join(bin_values)}"
                    roughness_suffix = f" | {format_roughness_title(roughness_key)}" if roughness_key else ""
                    trace_name = f"{label}{bin_set} D={diameter:g} um{roughness_suffix}"
                    trace_key = (
                        participant.participant_id,
                        case_id,
                        plot_spec["plot_key"],
                        zone_name,
                        bin_set,
                        f"{float(diameter):.12g}",
                    )

                    if trace_key in seen_trace_keys:
                        continue

                    seen_trace_keys.add(trace_key)
                    fig.add_trace(
                        go.Scatter(
                            x=diameter_data[GRID_SPACING_COLUMN],
                            y=diameter_data[y_column],
                            mode="lines+markers",
                            name=trace_name,
                            legendgroup=trace_name,
                            legendrank=participant_legend_rank(participant.participant_id),
                            line=dict(color=color),
                            marker=participant_marker(participant.participant_id),
                            customdata=diameter_data[["GRID_LEVEL_DISPLAY", "GRID_CELL_COUNT"]],
                            hovertemplate=(
                                f"Participant: {escape(label)}<br>"
                                f"Case: {escape(case_id)}<br>"
                                f"Zone: {escape(zone_name)}<br>"
                                f"Diameter={diameter:g} um<br>"
                                f"{escape(format_x_hover_label(GRID_SPACING_COLUMN))}=%{{x:.6g}}<br>"
                                f"{escape(y_column)}=%{{y}}<br>"
                                "Grid level=%{customdata[0]}<br>"
                                "Num cells=%{customdata[1]:,.0f}<extra></extra>"
                            ),
                        )
                    )                    
                    trace_count += 1
                    participant_trace_count += 1

                continue

            trace_key = (
                participant.participant_id,
                case_id,
                plot_spec["plot_key"],
                zone_name,
                roughness_filter or "",
            )

            if trace_key in seen_trace_keys:
                continue

            seen_trace_keys.add(trace_key)
            fig.add_trace(
                go.Scatter(
                    x=data[GRID_SPACING_COLUMN],
                    y=data[y_column],
                    mode="lines+markers",
                    name=f"{label} | {format_roughness_title(roughness_key)}" if roughness_key else label,
                    legendgroup=f"{label}_{roughness_key or 'unspecified'}",
                    legendrank=participant_legend_rank(participant.participant_id),
                    line=dict(color=color),
                    marker=participant_marker(participant.participant_id),
                    customdata=data[["GRID_LEVEL_DISPLAY", "GRID_CELL_COUNT"]],
                    hovertemplate=(
                        f"Participant: {escape(label)}<br>"
                        f"Case: {escape(case_id)}<br>"
                        f"Zone: {escape(zone_name)}<br>"
                        f"{escape(format_x_hover_label(GRID_SPACING_COLUMN))}=%{{x:.6g}}<br>"
                        f"{escape(y_column)}=%{{y}}<br>"
                        "Grid level=%{customdata[0]}<br>"
                        "Num cells=%{customdata[1]:,.0f}<extra></extra>"
                    ),
                )
            )            
            trace_count += 1
            participant_trace_count += 1

        if participant_had_matching_zone and participant_trace_count == 0:
            variable_name = plot_spec["y_candidates"][0]
            if participant_had_variable:
                skipped_notes.append(f"Participant ID {label} did not provide valid {variable_name} values.")
            else:
                skipped_notes.append(f"Participant ID {label} did not provide {variable_name}.")

    style_xy_figure(fig, case_id, plot_spec["plot_key"], plot_spec["x_label"], plot_spec["y_label"])
    style_grid_level_x_axis(fig, case_id)
    apply_individual_plot_overrides(fig, case_id, plot_spec["plot_key"])
    return fig, trace_count, skipped_notes


def build_qc_prime_integration_figure(
    participants,
    case_id: str,
    slice_position: float | None = None,
) -> tuple[go.Figure, int, list[str]]:
    """Integrate submitted HTC/Ts cut data for one slice and plot grid convergence."""
    fig = go.Figure()
    trace_count = 0
    skipped_notes: list[str] = []
    settings = HEAT_FLUX_CASE_SETTINGS.get(case_id)
    cell_counts = grid_cell_counts_for_case(case_id)
    if settings is None or not cell_counts:
        return fig, trace_count, skipped_notes

    for participant, case_data in iter_case_data(participants, case_id):
        values: list[tuple[float, float, str, str]] = []
        for grid_level, grid_data in sorted(case_data.grid_levels.items()):
            level_number = grid_level_number_from_value(grid_level)
            num_cells = cell_counts.get(level_number) if level_number is not None else None
            if num_cells is None:
                continue
            for dataset_id, dataset_data in sorted(grid_data.datasets.items()):
                path = dataset_data.cut_data_file
                cut_data = dataset_data.cut_data
                if path is None or cut_data is None:
                    continue
                value = None
                reason = "no zone with valid s, Cp, HTC, and Ts values"
                t_inf = settings.t_inf
                # HTC and Ts are repeated for each droplet-bin zone. Integrate
                # the first usable zone at the requested slice so each dataset
                # contributes once.
                for zone in cut_data.zones.values():
                    if slice_position is not None:
                        zone_info = parse_ipw3_zone_name(zone.name)
                        zone_slice = (
                            decode_slice_position(zone_info["slice"])
                            if zone_info is not None and zone_info["type"] == "SLICE"
                            else None
                        )
                        if zone_slice is None or not math.isclose(
                            zone_slice, slice_position, rel_tol=0.0, abs_tol=1.0e-6
                        ):
                            continue
                    s_column = find_column_case_insensitive(zone.data.columns, ["s"])
                    htc_column = find_column_case_insensitive(zone.data.columns, ["HTC", "HeatTransferCoefficient"])
                    ts_column = find_column_case_insensitive(zone.data.columns, ["Ts", "WallTemperature", "SurfaceTemperature"])
                    cp_column = find_column_case_insensitive(zone.data.columns, ["Cp", "PressureCoefficient"])
                    if s_column is None or htc_column is None or ts_column is None or cp_column is None:
                        continue
                    data = valid_numeric_rows(zone.data, s_column, htc_column, ts_column, cp_column)
                    if len(data) < 2:
                        continue
                    s = pd.to_numeric(data[s_column], errors="coerce").to_numpy(dtype=float)
                    htc = pd.to_numeric(data[htc_column], errors="coerce").to_numpy(dtype=float)
                    ts = pd.to_numeric(data[ts_column], errors="coerce").to_numpy(dtype=float)
                    cp = pd.to_numeric(data[cp_column], errors="coerce").to_numpy(dtype=float)
                    t_rec = recovery_temperature(cp, t_inf, settings.mach_inf)
                    valid_recovery = np.isfinite(t_rec)
                    s, htc, ts, t_rec = s[valid_recovery], htc[valid_recovery], ts[valid_recovery], t_rec[valid_recovery]
                    q = htc * (ts - t_rec)
                    window_s, window_q = _window_with_interpolated_edges(s, q, settings.ds_min, settings.ds_max)
                    if len(window_s) >= 2:
                        value = float(np.trapezoid(window_q, window_s))
                        reason = None
                        break
                if value is None:
                    skipped_notes.append(
                        f"Participant ID {participant.participant_id}, {grid_level} {dataset_id}: {reason}."
                    )
                    continue
                values.append((grid_convergence_coordinate(num_cells, cell_counts[1]), value, grid_level, dataset_id))

        if not values:
            continue
        values.sort(key=lambda item: item[0])
        multiple_datasets = len({item[3] for item in values}) > 1
        for dataset_id in sorted({item[3] for item in values}):
            dataset_values = [item for item in values if item[3] == dataset_id]
            if not dataset_values:
                continue
            label = participant_label(participant)
            trace_name = f"{label} | {dataset_id}" if multiple_datasets else label
            fig.add_trace(go.Scatter(
                x=[item[0] for item in dataset_values],
                y=[item[1] for item in dataset_values],
                mode="lines+markers",
                name=trace_name,
                legendgroup=trace_name,
                legendrank=participant_legend_rank(participant.participant_id),
                line=dict(color=participant_color(participant.participant_id)),
                marker=participant_marker(participant.participant_id),
                customdata=[[item[2], item[3]] for item in dataset_values],
                hovertemplate=(
                    f"Participant: {escape(label)}<br>"
                    "Grid level=%{customdata[0]}<br>"
                    "Dataset=%{customdata[1]}<br>"
                    "h = N^(-1/3)=%{x:.6g}<br>"
                    "Q_c'=%{y:.6g} W/m<extra></extra>"
                ),
            ))
            trace_count += 1

    style_xy_figure(fig, case_id, "qc_prime", "h = N<sup>−1/3</sup> [-]", "Q<sub>c</sub>′ = ∫ HTC (T<sub>s</sub> − T<sub>rec</sub>) ds [W/m]")
    style_grid_level_x_axis(fig, case_id)
    fig.update_yaxes(autorange="reversed")
    apply_individual_plot_overrides(fig, case_id, "qc_prime")
    return fig, trace_count, skipped_notes


def build_derived_icing_ratio_figure(
    participants,
    case_id: str,
    plot_spec: dict[str, Any],
    requirement: str = "required",
    bin_set_filter: str | None = None,
) -> tuple[go.Figure, int, list[str]]:
    """Build icing mass-conversion ratios from submitted integrated masses."""
    fig = go.Figure()
    trace_count = 0
    skipped_notes: list[str] = []
    seen_trace_keys: set[tuple[str, str, str]] = set()

    for participant, case_data in iter_case_data(participants, case_id):
        if case_data.grid_convergence_data is None:
            continue
        label = participant_label(participant)
        color = participant_color(participant.participant_id)

        for zone_name, zone in case_data.grid_convergence_data.zones.items():
            if "by_diameter" in zone_name.lower() or not grid_convergence_zone_matches_requirement(zone, requirement):
                continue
            bin_set = extract_icing_bin_set_from_zone_name(zone_name)
            if bin_set is None:
                continue
            if bin_set_filter is not None and bin_set != bin_set_filter:
                continue

            x_column = find_column_case_insensitive(zone.data.columns, plot_spec["x_candidates"])
            grid_column = find_column_case_insensitive(zone.data.columns, ["GRID_LEVEL", "GridLevel"])
            water_column = find_column_case_insensitive(zone.data.columns, ["WATER_MASS", "WaterMass"])
            ice_column = find_column_case_insensitive(zone.data.columns, ["ICE_MASS", "IceMass"])
            evap_column = find_column_case_insensitive(zone.data.columns, ["WATER_EVAP_MASS", "WaterEvapMass"])
            required_columns = [x_column, water_column, ice_column]
            if plot_spec["derived_icing_ratio"] == "ice_plus_evap_to_water":
                required_columns.append(evap_column)
            if any(column is None for column in required_columns):
                continue

            selected_columns = [str(column) for column in required_columns]
            if grid_column is not None and grid_column not in selected_columns:
                selected_columns.append(grid_column)
            data = valid_numeric_rows(
                zone.data[selected_columns].copy(),
                *[str(column) for column in required_columns],
                positive_columns={str(x_column), str(water_column)},
            )
            data = add_grid_spacing_column(data, case_id, str(x_column), grid_column=grid_column)
            if data.empty:
                continue

            numerator = pd.to_numeric(data[str(ice_column)], errors="coerce")
            if plot_spec["derived_icing_ratio"] == "ice_plus_evap_to_water":
                numerator = numerator + pd.to_numeric(data[str(evap_column)], errors="coerce")
            data["DERIVED_MASS_RATIO_PERCENT"] = numerator / pd.to_numeric(data[str(water_column)], errors="coerce") * 100.0
            data["DERIVED_REMAINDER_PERCENT"] = 100.0 - data["DERIVED_MASS_RATIO_PERCENT"]
            data = data[np.isfinite(data["DERIVED_MASS_RATIO_PERCENT"])].sort_values(GRID_SPACING_COLUMN)
            if data.empty:
                continue

            roughness_key = extract_icing_roughness_key_from_zone_name(zone_name)
            # Water Mass Analysis uses only the required 1 mm baseline for
            # ONERA M6. NACA0012 retains every submitted roughness condition.
            if case_id == "TC_ONERAM6" and roughness_key.lower() != "1mm":
                continue
            trace_key = (participant.participant_id, zone_name, plot_spec["plot_key"])
            if trace_key in seen_trace_keys:
                continue
            seen_trace_keys.add(trace_key)
            detail_parts = [display_bin_set(bin_set)]
            if roughness_key != "unspecified":
                detail_parts.append(format_icing_roughness_title(roughness_key))
            trace_name = f"{label} | {' | '.join(detail_parts)}"
            fig.add_trace(go.Scatter(
                x=data[GRID_SPACING_COLUMN],
                y=data["DERIVED_MASS_RATIO_PERCENT"],
                mode="lines+markers",
                name=trace_name,
                legendgroup=trace_name,
                legendrank=participant_legend_rank(participant.participant_id),
                line={"color": color},
                marker=participant_marker(participant.participant_id),
                customdata=data[["GRID_LEVEL_DISPLAY", "GRID_CELL_COUNT", "DERIVED_REMAINDER_PERCENT"]],
                hovertemplate=(
                    f"Participant: {escape(label)}<br>Case: {escape(case_id)}<br>"
                    f"Distribution: {escape(display_bin_set(bin_set))}<br>"
                    + (f"Roughness: {escape(format_icing_roughness_title(roughness_key))}<br>" if roughness_key != "unspecified" else "")
                    + "Grid level=%{customdata[0]}<br>Num cells=%{customdata[1]:,.0f}<br>"
                    + f"{escape(plot_spec['y_label'])}=%{{y:.4g}}<br>"
                    + (
                        "Water not converted to ice=%{customdata[2]:.4g}%"
                        if plot_spec["derived_icing_ratio"] == "ice_to_water"
                        else "Water neither ice nor evaporated=%{customdata[2]:.4g}%"
                    )
                    + "<extra></extra>"
                ),
            ))
            trace_count += 1

    style_xy_figure(fig, case_id, plot_spec["plot_key"], plot_spec["x_label"], plot_spec["y_label"])
    style_grid_level_x_axis(fig, case_id)
    apply_individual_plot_overrides(fig, case_id, plot_spec["plot_key"])
    if trace_count == 0:
        skipped_notes.append("No grid-convergence zone contained the required integrated mass columns.")
    return fig, trace_count, skipped_notes


def collect_icing_ratio_bin_sets(participants, case_id: str, requirement: str = "required") -> list[str]:
    """Return submitted bin distributions that contain integrated icing masses."""
    bin_sets: set[str] = set()
    for _, case_data in iter_case_data(participants, case_id):
        if case_data.grid_convergence_data is None:
            continue
        for zone_name, zone in case_data.grid_convergence_data.zones.items():
            if "by_diameter" in zone_name.lower() or not grid_convergence_zone_matches_requirement(zone, requirement):
                continue
            bin_set = extract_icing_bin_set_from_zone_name(zone_name)
            roughness_key = extract_icing_roughness_key_from_zone_name(zone_name)
            if case_id == "TC_ONERAM6" and roughness_key.lower() != "1mm":
                continue
            if bin_set is not None:
                bin_sets.add(bin_set)
    return sorted(bin_sets, key=lambda value: (bin_count_from_bin_set(value) or 10**9, value))


def build_grid_convergence_plot_subsection(participants, case_id: str, plot_spec: dict[str, Any], requirement: str = "required") -> str:
    if plot_spec.get("derived_icing_ratio"):
        distribution_html = ""
        requirements = ("required", "optional") if requirement == "all" else (requirement,)
        for current_requirement in requirements:
            for bin_set in collect_icing_ratio_bin_sets(participants, case_id, requirement=current_requirement):
                fig, trace_count, _ = build_derived_icing_ratio_figure(
                    participants,
                    case_id,
                    plot_spec,
                    requirement=current_requirement,
                    bin_set_filter=bin_set,
                )
                if trace_count == 0:
                    continue
                bin_count = bin_count_from_bin_set(bin_set)
                distribution_label = f"{bin_count}-bin distribution" if bin_count != 1 else "Single-bin distribution"
                filename = f"{slugify(case_id)}_{plot_spec['filename_slug']}_{current_requirement}_{slugify(bin_set)}"
                title = f"{plot_spec['title']} | {distribution_label}"
                roughness_note = simplify_icing_participant_legend(fig)
                if case_id == "TC_ONERAM6":
                    roughness_note = ""
                figure_html = grid_convergence_figure_pair_html(
                    fig, case_id, plot_spec["plot_key"], filename, title,
                )
                distribution_html += f"""
                <section class="slice-plot-group">
                  <h5>{escape(distribution_label)}</h5>
                  {roughness_note}
                  <div class="plot-container">{figure_html}</div>
                </section>
                """
        if not distribution_html.strip():
            return ""
        interpretation = (
            "This is the percentage of submitted water mass that became ice. The remainder did not become ice."
            if plot_spec["derived_icing_ratio"] == "ice_to_water"
            else "This is the percentage of submitted water mass accounted for as ice or evaporated water. The remainder is water not represented by either outcome."
        )
        formula = (
            "R<sub>ice</sub> [%] = M<sub>ice</sub> / M<sub>water</sub> × 100."
            if plot_spec["derived_icing_ratio"] == "ice_to_water"
            else "R<sub>ice+evap</sub> [%] = (M<sub>ice</sub> + M<sub>evaporated</sub>) / M<sub>water</sub> × 100."
        )
        return f"""
        <details class="grid-section plot-subsection analysis-metric-section" data-variable-key="{escape(plot_spec['plot_key'])}" data-variable-label="{escape(plot_spec['title'])}">
          <summary>{escape(plot_spec['title'])}</summary>
          <p class="plot-description">Formula: {formula}</p>
          <p class="plot-description">{escape(interpretation)} Ratios are computed independently for every participant, droplet distribution, roughness condition, and grid level.</p>
          {distribution_html}
        </details>
        """

    if plot_spec.get("qc_prime_integration_plot", False):
        settings = HEAT_FLUX_CASE_SETTINGS[case_id]
        window = f"ds = {settings.ds_min if settings.ds_min is not None else 'data minimum'} to {settings.ds_max if settings.ds_max is not None else 'data maximum'} m"
        subsections: list[str] = []
        slice_positions = CASE_SLICES.get(case_id) or [None]
        for slice_position in slice_positions:
            fig, trace_count, skipped_notes = build_qc_prime_integration_figure(
                participants, case_id, slice_position=slice_position
            )
            if trace_count == 0:
                continue
            slice_label = f"Y = {slice_position:g} m" if slice_position is not None else "submitted slice"
            slice_slug = f"_y_{slice_position:g}" if slice_position is not None else ""
            filename = f"{slugify(case_id)}_{plot_spec['filename_slug']}{slice_slug}"
            title = f"{plot_spec['title']} | {slice_label}"
            figure_html = grid_convergence_figure_pair_html(fig, case_id, "qc_prime", filename, title)
            notes_html = ""
            if skipped_notes:
                notes_html = '<ul class="plot-notes">' + "".join(f"<li>{escape(note)}</li>" for note in sorted(set(skipped_notes))) + "</ul>"
            subsections.append(f"""
        <section class="plot-subsection" data-variable-key="qc_prime" data-variable-label="Integrated convective heat transfer per unit span">
          <h4>{escape(title)}</h4>
          <p class="plot-description">Slice {escape(slice_label)}. Signed trapezoidal integration of HTC (Ts − Trec) over {escape(window)}, with Trec approximated pointwise from Cp (γ = 1.4, r = 0.9, M∞ = {settings.mach_inf:g}); the Y-axis is reversed as for Cp. Legend: Participant ID.</p>
          {notes_html}
          <div class="plot-container">{figure_html}</div>
        </section>
            """)
        return "".join(subsections)

    if plot_spec.get("diameter_plot", False):
        return build_grid_convergence_diameter_subsection(participants, case_id, plot_spec, requirement=requirement)

    if plot_spec.get("combined_icing_plot", False):
        return build_combined_icing_subsection(participants, case_id, plot_spec, requirement=requirement)

    if plot_spec.get("group_by_roughness", False):
        return build_grid_convergence_roughness_subsection(participants, case_id, plot_spec, requirement=requirement)

    fig, trace_count, skipped_notes = build_grid_convergence_figure(participants, case_id, plot_spec, requirement=requirement)

    if trace_count == 0:
        return ""

    filename = f"{slugify(case_id)}_{plot_spec['filename_slug']}"
    figure_html = grid_convergence_figure_pair_html(fig, case_id, plot_spec["plot_key"], filename, plot_spec["title"])

    description = "Grid-convergence data from the case-level gridConvergence file. Missing values equal to -999 are ignored. Legend: Participant ID."

    notes_html = ""
    if skipped_notes:
        notes_html = '<ul class="plot-notes">' + "".join(f"<li>{escape(note)}</li>" for note in sorted(set(skipped_notes))) + "</ul>"

    return f"""
    <section class="plot-subsection" data-variable-key="{escape(plot_spec['plot_key'])}" data-variable-label="{escape(plot_spec['title'])}">
      <p class="plot-description">{escape(description)}</p>
      {notes_html}
      <div class="plot-container">
        {figure_html}
      </div>
    </section>
    """


def build_onera_upper_horn_reference_showcase(participants, case_id: str) -> str:
    """Show the M6 horn construction at each spanwise slice using participant 010."""
    showcase_shapes: dict[float, tuple[pd.Series, pd.Series]] = {}
    for participant, _, _, dataset_data in iter_grid_datasets(participants, case_id, "L1"):
        if participant.participant_id != "010":
            continue
        ice_data = getattr(dataset_data, "ice_shape_data", None)
        if ice_data is None:
            continue
        for zone_name, zone in ice_data.zones.items():
            zone_info = parse_ipw3_ice_shape_zone_name(zone_name)
            if (
                zone_info is None
                or zone_info["shape_role"] != "SINGLE_LAYER"
                or zone_info["bins"] != "BINS15"
                or extract_ice_shape_roughness_key(zone_name) != "1mm"
                or not zone_info["slice"]
            ):
                continue
            slice_position = decode_slice_position(zone_info["slice"])
            x_column, z_column = find_submitted_ice_xz_columns(zone.data.columns)
            if slice_position is None or x_column is None or z_column is None:
                continue
            shape = valid_submitted_ice_shape_rows(zone.data, x_column, z_column)
            if not shape.empty:
                showcase_shapes[slice_position] = (shape[x_column], shape[z_column])

    cards = ""
    for slice_position in CASE_SLICES.get(case_id, []):
        shape = showcase_shapes.get(slice_position)
        clean = onera_clean_reference_points(slice_position)
        if shape is None or clean is None:
            continue
        x, z = shape
        clean_x, clean_z, _ = clean
        geometry = upper_horn_geometry(x, z, case_id, slice_position)
        if geometry is None:
            continue
        leading, horn, angle_value = geometry
        construction_span = max(math.hypot(horn[0] - leading[0], horn[1] - leading[1]), 0.01)
        reference_length = 0.65 * construction_span
        arc_radius = 0.24 * construction_span
        arc_angles = np.linspace(0.0, math.radians(angle_value), 48)
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=clean_x, y=clean_z, mode="lines", name="Clean M6 section", line={"color": "#94a3b8", "width": 2}))
        fig.add_trace(go.Scatter(x=x, y=z, mode="lines", name="Participant 010", line={"color": "#111827", "width": 2}))
        fig.add_trace(go.Scatter(
            x=[leading[0], leading[0] + reference_length], y=[leading[1], leading[1]],
            mode="lines", name="Positive x-axis reference", line={"color": "#dc2626", "width": 2}, hoverinfo="skip",
        ))
        fig.add_trace(go.Scatter(
            x=[leading[0], horn[0]], y=[leading[1], horn[1]], mode="lines+markers",
            name="Upper-horn construction", line={"color": "#dc2626", "width": 3},
            marker={"color": ["#2563eb", "#dc2626"], "size": [8, 10], "symbol": ["circle", "diamond"]},
            customdata=[["Nearest clean-surface point"], ["Farthest-upstream submitted point"]],
            hovertemplate="%{customdata[0]}<br>X=%{x:.6g} m<br>Z=%{y:.6g} m<extra></extra>",
        ))
        fig.add_trace(go.Scatter(
            x=leading[0] + arc_radius * np.cos(arc_angles),
            y=leading[1] + arc_radius * np.sin(arc_angles),
            mode="lines", name="Upper-horn angle arc", line={"color": "#dc2626", "width": 3},
            hovertemplate=f"θupper = {angle_value:.2f}°<extra></extra>",
        ))
        fig.add_annotation(
            x=horn[0], y=horn[1], text=f"θupper = {angle_value:.2f}°", showarrow=True,
            arrowhead=2, ax=45, ay=-35, bgcolor="rgba(255,255,255,0.85)",
        )
        x_min, x_max = min(leading[0], horn[0]), max(leading[0], horn[0])
        z_min, z_max = min(leading[1], horn[1]), max(leading[1], horn[1])
        padding = 0.45 * construction_span
        fig.update_layout(
            height=340, showlegend=False, margin={"l": 50, "r": 15, "t": 15, "b": 48},
            plot_bgcolor="white", paper_bgcolor="white",
            xaxis={"title": "X [m]", "range": [x_min - padding, x_max + padding], "showgrid": True, "gridcolor": "#e2e8f0"},
            yaxis={"title": "Z [m]", "range": [z_min - padding, z_max + padding], "showgrid": True, "gridcolor": "#e2e8f0", "scaleanchor": "x", "scaleratio": 1.0},
        )
        slice_slug = str(slice_position).replace(".", "p")
        filename = f"{slugify(case_id)}_010_slice_{slice_slug}_upper_horn_angle_method"
        cards += f"""
        <article class="horn-reference-card">
          <h3>Y = {slice_position:g} m</h3>
          <div class="plot-container">{figure_to_html_div(fig, filename, f'Participant 010 upper-horn construction at Y = {slice_position:g} m')}</div>
        </article>
        """
    if not cards:
        return ""
    return f"""
    <section class="horn-reference-showcase">
      <h2>ONERA M6 Upper-Horn Angle Construction by Slice</h2>
      <p>Participant 010's L1, 15-bin, 1 mm roughness result illustrates the same calculation at each submitted M6 slice. The horn is the farthest-upstream submitted point (minimum X), measured from its nearest point on that slice's clean surface; no experimental M6 ice-shape reference is available.</p>
      <div class="horn-reference-grid">{cards}</div>
    </section>
    """


def build_upper_horn_reference_showcase(case_id: str) -> str:
    """Draw Max/Mean/Min CCS horn-angle constructions side by side."""
    reference_path = EXPERIMENTAL_ICE_SHAPE_FILES.get(case_id)
    clean_reference = naca_clean_reference_points()
    if reference_path is None or not reference_path.exists() or clean_reference is None:
        return ""
    _, _, clean_leading = clean_reference
    clean_data = load_experimental_ice_shape_data(str(Path("R00_REFERENCE") / "NACA0012_CLEAN_ROTATED.dat"))
    clean_candidates: list[tuple[np.ndarray, np.ndarray]] = []
    for clean_zone in clean_data.zones.values():
        clean_x_column = find_column_case_insensitive(clean_zone.data.columns, ["X", "CoordinateX"])
        clean_z_column = find_column_case_insensitive(clean_zone.data.columns, ["Z", "CoordinateZ"])
        if clean_x_column is None or clean_z_column is None:
            continue
        ordered_x, ordered_z = ordered_clean_reference_columns(case_id, clean_zone, clean_x_column, clean_z_column)
        clean_candidates.append((np.asarray(ordered_x, dtype=float), np.asarray(ordered_z, dtype=float)))
    if not clean_candidates:
        return ""
    clean_plot_x, clean_plot_z = max(clean_candidates, key=lambda candidate: len(candidate[0]))
    experimental = load_experimental_ice_shape_data(str(reference_path))
    zones = {name.replace("_", "").upper(): zone for name, zone in experimental.zones.items()}
    cards = ""
    for contour_key, contour_label in (("MAXCCS", "MaxCCS"), ("MEANCCS", "MeanCCS"), ("MINCCS", "MinCCS")):
        zone = zones.get(contour_key)
        if zone is None:
            continue
        x_column = find_column_case_insensitive(zone.data.columns, ["X", "CoordinateX"])
        z_column = find_column_case_insensitive(zone.data.columns, ["Y", "Z", "CoordinateZ"])
        if x_column is None or z_column is None:
            continue
        x = pd.to_numeric(zone.data[x_column], errors="coerce") * INCHES_TO_METRES
        z = pd.to_numeric(zone.data[z_column], errors="coerce") * INCHES_TO_METRES
        geometry = upper_horn_geometry(x, z)
        if geometry is None:
            continue
        leading, horn, angle_value = geometry
        construction_span = max(math.hypot(horn[0] - leading[0], horn[1] - leading[1]), 0.015)
        x_axis_angle = 0.0
        reference_length = 0.65 * construction_span
        arc_radius = 0.24 * construction_span
        arc_angles = np.linspace(x_axis_angle, math.radians(angle_value), 48)
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=clean_plot_x, y=clean_plot_z, mode="lines", name="Rotated clean airfoil", line={"color": "#94a3b8", "width": 2}))
        fig.add_trace(go.Scatter(x=x, y=z, mode="lines", name=f"Experimental {contour_label}", line={"color": "#111827", "width": 2}))
        # Solid global x-axis reference used as the zero direction.
        fig.add_trace(go.Scatter(
            x=[leading[0], leading[0] + reference_length],
            y=[leading[1], leading[1]],
            mode="lines", name="Positive x-axis reference",
            line={"color": "#dc2626", "width": 2},
            hoverinfo="skip",
        ))
        fig.add_trace(go.Scatter(
            x=[leading[0], horn[0]], y=[leading[1], horn[1]], mode="lines+markers",
            name="Upper-horn construction", line={"color": "#dc2626", "width": 3},
            marker={"color": ["#2563eb", "#dc2626"], "size": [8, 10], "symbol": ["circle", "diamond"]},
            customdata=[["Clean leading-edge reference"], ["Detected upper horn"]],
            hovertemplate="%{customdata[0]}<br>X=%{x:.6g} m<br>Z=%{y:.6g} m<extra></extra>",
        ))
        # Arc from the positive x-axis to the detected upper-horn ray.
        fig.add_trace(go.Scatter(
            x=leading[0] + arc_radius * np.cos(arc_angles),
            y=leading[1] + arc_radius * np.sin(arc_angles),
            mode="lines", name="Upper-horn angle arc",
            line={"color": "#dc2626", "width": 3},
            hovertemplate=f"θupper = {angle_value:.2f}°<extra></extra>",
        ))
        fig.add_annotation(
            x=horn[0], y=horn[1], text=f"θupper = {angle_value:.2f}°", showarrow=True,
            arrowhead=2, ax=45, ay=-35, bgcolor="rgba(255,255,255,0.85)",
        )
        x_min, x_max = min(leading[0], horn[0]), max(leading[0], horn[0])
        z_min, z_max = min(leading[1], horn[1]), max(leading[1], horn[1])
        padding = 0.35 * construction_span
        fig.update_layout(
            height=340, showlegend=False, margin={"l": 50, "r": 15, "t": 15, "b": 48},
            plot_bgcolor="white", paper_bgcolor="white",
            xaxis={"title": "X [m]", "range": [x_min - padding, x_max + padding], "showgrid": True, "gridcolor": "#e2e8f0"},
            yaxis={"title": "Z [m]", "range": [z_min - padding, z_max + padding], "showgrid": True, "gridcolor": "#e2e8f0", "scaleanchor": "x", "scaleratio": 1.0},
        )
        filename = f"{slugify(case_id)}_{contour_key.lower()}_upper_horn_angle_method"
        cards += f"""
        <article class="horn-reference-card">
          <h3>{contour_label}</h3>
          <div class="plot-container">{figure_to_html_div(fig, filename, f'{contour_label} upper-horn angle construction')}</div>
        </article>
        """
    if not cards:
        return ""
    return f"""
    <section class="horn-reference-showcase">
      <h2>Experimental Upper-Horn Angle Construction</h2>
      <p>The three drawings use the same method as participant data: the angle is measured from the positive global x-axis at the clean leading-edge reference to the detected upper-horn point.</p>
      <div class="horn-reference-grid">{cards}</div>
    </section>
    """


def build_upper_horn_angle_convergence_section(participants, case_id: str) -> str:
    """Build upper-horn convergence using case- and slice-specific clean sections."""
    if "NACA0012" not in case_id.upper() and "ONERAM6" not in case_id.upper():
        return ""
    if VARIABLE_FILTER is not None and not VARIABLE_FILTER.intersection({"horn", "horn_angle", "ice_horn_angle", "upper_horn_angle"}):
        return ""
    grid_order = ["L1", "L2", "L3", "L4"]
    angles: dict[tuple[str, str, str, str, str], tuple[int, float]] = {}
    for grid_level in grid_order:
        for participant, _, _, dataset_data in iter_grid_datasets(participants, case_id, grid_level):
            ice_data = getattr(dataset_data, "ice_shape_data", None)
            if ice_data is None:
                continue
            for zone_name, zone in ice_data.zones.items():
                zone_info = parse_ipw3_ice_shape_zone_name(zone_name)
                if zone_info is None or zone_info["shape_role"] not in {"SINGLE_LAYER", "FINAL_LAYER"}:
                    continue
                x_column, z_column = find_submitted_ice_xz_columns(zone.data.columns)
                if x_column is None or z_column is None:
                    continue
                shape = valid_submitted_ice_shape_rows(zone.data, x_column, z_column)
                slice_position = decode_slice_position(zone_info["slice"]) if zone_info["slice"] else None
                geometry = upper_horn_geometry(
                    shape[x_column], shape[z_column], case_id, slice_position,
                ) if not shape.empty else None
                if geometry is None:
                    continue
                slice_key = (
                    f"{slice_position:g}".replace(".", "p")
                    if slice_position is not None else "MCCS"
                )
                key = (participant.participant_id, zone_info["bins"], extract_ice_shape_roughness_key(zone_name), slice_key, grid_level)
                role_rank = 2 if zone_info["shape_role"] == "FINAL_LAYER" else 1
                if key not in angles or role_rank > angles[key][0]:
                    angles[key] = (role_rank, geometry[2])

    reference_angles: dict[str, float] = {}
    reference_path = EXPERIMENTAL_ICE_SHAPE_FILES.get(case_id)
    if reference_path is not None and reference_path.exists():
        experimental = load_experimental_ice_shape_data(str(reference_path))
        for zone_name, zone in experimental.zones.items():
            x_column = find_column_case_insensitive(zone.data.columns, ["X", "CoordinateX"])
            z_column = find_column_case_insensitive(zone.data.columns, ["Y", "Z", "CoordinateZ"])
            if x_column is None or z_column is None:
                continue
            geometry = upper_horn_geometry(
                pd.to_numeric(zone.data[x_column], errors="coerce") * INCHES_TO_METRES,
                pd.to_numeric(zone.data[z_column], errors="coerce") * INCHES_TO_METRES,
            )
            if geometry is not None:
                reference_angles[zone_name.replace("_", "").upper()] = geometry[2]

    html = ""
    cell_counts = grid_cell_counts_for_case(case_id)
    bin_sets = sorted({key[1] for key in angles}, key=lambda value: (bin_count_from_bin_set(value) or 10**9, value))
    is_onera = "ONERAM6" in case_id.upper()
    summary_slice_keys: list[str | None] = (
        sorted({key[3] for key in angles}, key=lambda value: decode_slice_position(value) or math.inf)
        if is_onera else [None]
    )
    reference_styles = {"MAXCCS": ("MaxCCS", "dash"), "MEANCCS": ("MeanCCS", "solid"), "MINCCS": ("MinCCS", "dot")}
    for bin_set, summary_slice_key in (
        (bin_set, slice_key) for bin_set in bin_sets for slice_key in summary_slice_keys
    ):
        fig = go.Figure()
        series_keys = sorted({
            (pid, roughness, slice_key)
            for pid, bins, roughness, slice_key, _ in angles
            if bins == bin_set and (summary_slice_key is None or slice_key == summary_slice_key)
        })
        for participant_id, roughness_key, slice_key in series_keys:
            levels = [level for level in grid_order if (participant_id, bin_set, roughness_key, slice_key, level) in angles]
            if not levels:
                continue
            roughness_label = format_icing_roughness_title(roughness_key)
            decoded_slice = decode_slice_position(slice_key)
            slice_label = f"Y = {decoded_slice:g} m" if decoded_slice is not None else slice_key
            name = participant_id + (f" | {roughness_label}" if roughness_label else "") + f" | {slice_label}"
            fig.add_trace(go.Scatter(
                x=[grid_convergence_coordinate(cell_counts[int(level[1:])], cell_counts[1]) for level in levels],
                y=[angles[(participant_id, bin_set, roughness_key, slice_key, level)][1] for level in levels],
                mode="lines+markers", name=name, legendgroup=name,
                legendrank=participant_legend_rank(participant_id), line={"color": participant_color(participant_id)},
                marker=participant_marker(participant_id), customdata=[[level] for level in levels],
                hovertemplate="Participant: %{fullData.name}<br>Grid=%{customdata[0]}<br>Upper horn angle=%{y:.4g}°<extra></extra>",
            ))
        for key, (label, dash) in reference_styles.items():
            if key not in reference_angles:
                continue
            fig.add_trace(go.Scatter(
                x=[grid_convergence_coordinate(cell_counts[index], cell_counts[1]) for index in (1, 2, 3, 4)],
                y=[reference_angles[key]] * 4, mode="lines", name=f"Exp. {label}",
                legendgroup=f"horn_reference_{key}", legendrank=1000,
                line={"color": "black", "width": 2, "dash": dash},
                hovertemplate=f"Experimental {label}<br>Upper horn angle=%{{y:.4g}}°<extra></extra>",
            ))
        if not fig.data:
            continue
        style_xy_figure(fig, case_id, "upper_horn_angle_vs_n", "Grid level", "Upper horn angle [deg]")
        style_grid_level_x_axis(fig, case_id)
        bin_count = bin_count_from_bin_set(bin_set)
        label = f"{bin_count}-bin distribution" if bin_count != 1 else "Single-bin distribution"
        summary_slice_position = decode_slice_position(summary_slice_key) if summary_slice_key is not None else None
        summary_slice_label = f"Y = {summary_slice_position:g} m" if summary_slice_position is not None else ""
        slice_suffix = f"_{slugify(summary_slice_label)}" if summary_slice_label else ""
        filename = f"{slugify(case_id)}_upper_horn_angle_{slugify(bin_set)}{slice_suffix}"
        title = f"Upper ice-horn angle grid convergence | {label}" + (f" | {summary_slice_label}" if summary_slice_label else "")
        roughness_note = simplify_icing_participant_legend(fig)
        html += f"""
        <section class="slice-plot-group">
          <h4>{escape(title)}</h4>
          <p class="plot-description">Formula: θ<sub>upper</sub> = atan2(z<sub>horn</sub> − z<sub>surface</sub>, x<sub>horn</sub> − x<sub>surface</sub>) relative to the positive global x-axis. For ONERA M6, the horn is each participant's farthest-upstream submitted point and the origin is its nearest point on the matching clean slice.</p>
          {roughness_note}
          <div class="plot-container">{grid_convergence_figure_pair_html(fig, case_id, "upper_horn_angle_vs_n", filename, title)}</div>
        </section>
        """

    # At each fixed grid level, compare the upper-horn angle across the
    # submitted droplet distributions (BINS01 -> BINS15).  This complements
    # the plots above, which instead hold the distribution fixed and vary the
    # grid level.
    for grid_level, summary_slice_key in (
        (grid_level, slice_key) for grid_level in grid_order for slice_key in summary_slice_keys
    ):
        fig = go.Figure()
        series_keys = sorted({
            (participant_id, roughness_key, slice_key)
            for participant_id, _, roughness_key, slice_key, level in angles
            if level == grid_level and (summary_slice_key is None or slice_key == summary_slice_key)
        })
        for participant_id, roughness_key, slice_key in series_keys:
            submitted_bins = [
                bin_set for bin_set in bin_sets
                if (participant_id, bin_set, roughness_key, slice_key, grid_level) in angles
            ]
            if not submitted_bins:
                continue
            roughness_label = format_icing_roughness_title(roughness_key)
            decoded_slice = decode_slice_position(slice_key)
            slice_label = f"Y = {decoded_slice:g} m" if decoded_slice is not None else slice_key
            name = participant_id + (f" | {roughness_label}" if roughness_label else "") + f" | {slice_label}"
            fig.add_trace(go.Scatter(
                x=submitted_bins,
                y=[angles[(participant_id, bin_set, roughness_key, slice_key, grid_level)][1] for bin_set in submitted_bins],
                mode="lines+markers", name=name, legendgroup=name,
                legendrank=participant_legend_rank(participant_id),
                line={"color": participant_color(participant_id)},
                marker=participant_marker(participant_id),
                customdata=[[bin_set] for bin_set in submitted_bins],
                hovertemplate=(
                    "Participant: %{fullData.name}<br>Grid=" + grid_level
                    + "<br>Distribution=%{x}<br>Upper horn angle=%{y:.4g}°<extra></extra>"
                ),
            ))
        for key, (label, dash) in reference_styles.items():
            if key not in reference_angles or not bin_sets:
                continue
            fig.add_trace(go.Scatter(
                x=bin_sets, y=[reference_angles[key]] * len(bin_sets),
                mode="lines", name=f"Exp. {label}",
                legendgroup=f"horn_reference_{key}", legendrank=1000,
                line={"color": "black", "width": 2, "dash": dash},
                hovertemplate=f"Experimental {label}<br>Upper horn angle=%{{y:.4g}}°<extra></extra>",
            ))
        if not fig.data:
            continue
        style_xy_figure(
            fig, case_id, "upper_horn_angle_vs_bins", "Droplet distribution",
            "Upper horn angle [deg]",
        )
        display_bin_sets = list(reversed(bin_sets))
        fig.update_xaxes(
            type="category", categoryorder="array", categoryarray=display_bin_sets,
            tickmode="array", tickvals=display_bin_sets, ticktext=[display_bin_set(value) for value in display_bin_sets],
            range=None, autorange=True,
            title={"text": "Droplet distribution (15 → 01)"},
        )
        summary_slice_position = decode_slice_position(summary_slice_key) if summary_slice_key is not None else None
        summary_slice_label = f"Y = {summary_slice_position:g} m" if summary_slice_position is not None else ""
        slice_suffix = f"_{slugify(summary_slice_label)}" if summary_slice_label else ""
        filename = f"{slugify(case_id)}_upper_horn_angle_distribution_convergence_{grid_level.lower()}{slice_suffix}"
        title = f"Upper ice-horn angle droplet-distribution convergence | {grid_level}" + (f" | {summary_slice_label}" if summary_slice_label else "")
        roughness_note = simplify_icing_participant_legend(fig)
        html += f"""
        <section class="slice-plot-group">
          <h4>{escape(title)}</h4>
          <p class="plot-description">Formula: θ<sub>upper</sub> = atan2(z<sub>horn</sub> − z<sub>LE</sub>, x<sub>horn</sub> − x<sub>LE</sub>) relative to the positive global x-axis. Grid level {escape(grid_level)} is held fixed while the submitted droplet distribution changes from 01 through 15.</p>
          {roughness_note}
          <div class="plot-container">{distribution_figure_pair_html(fig, case_id, "upper_horn_angle_vs_bins", filename, title)}</div>
        </section>
        """
    if not html:
        return ""
    showcase_html = (
        build_onera_upper_horn_reference_showcase(participants, case_id)
        if "ONERAM6" in case_id.upper()
        else build_upper_horn_reference_showcase(case_id)
    )
    return f"""
    {showcase_html}
    <section class="plot-subsection upper-horn-angle-analysis" data-variable-key="upper_horn_angle" data-variable-label="Upper ice-horn angle">
      <h3>Upper Ice-Horn Angle</h3>
      <div class="convergence-metric-content">{html}</div>
    </section>
    """


def build_grid_convergence_section(participants, case_id: str, category: str = "all", requirement: str = "required", metric: str | None = None) -> str:
    """Build CFD, icing, or all convergence plots for a case.

    Keeping ``all`` as the default preserves the public builder API used by PNG
    exports and slideshow mode while allowing the website to present CFD and
    icing convergence as distinct views.
    """
    plot_specs = {
        "all": GRID_CONVERGENCE_PLOTS,
        "cfd": CFD_GRID_CONVERGENCE_PLOTS,
        "icing": ICING_GRID_CONVERGENCE_PLOTS,
    }.get(category)
    if plot_specs is None:
        raise ValueError(f"Unknown grid-convergence category: {category}")
    if category != "cfd":
        plot_specs = [
            plot_spec for plot_spec in plot_specs
            if not plot_spec.get("qc_prime_integration_plot", False)
        ]
    if category == "icing" and requirement == "optional":
        plot_specs = [
            *[plot_spec for plot_spec in plot_specs if not plot_spec.get("qc_prime_integration_plot", False)],
            *OPTIONAL_ICING_DIAMETER_PLOTS,
        ]
    if metric is not None:
        metric_plot_keys = {
            "water": {"water_mass_vs_n", "water_mass_by_diameter_vs_n"},
            "ice": {"ice_mass_vs_n", "ice_mass_by_diameter_vs_n"},
            "evaporation": {"water_evap_mass_vs_n", "water_evap_mass_by_diameter_vs_n"},
        }
        if metric not in metric_plot_keys:
            raise ValueError(f"Unknown icing-convergence metric: {metric}")
        plot_specs = [plot_spec for plot_spec in plot_specs if plot_spec["plot_key"] in metric_plot_keys[metric]]
    plot_specs = [plot_spec for plot_spec in plot_specs if plot_matches_variable_filter(plot_spec)]

    html = ""

    for plot_spec in plot_specs:
        metric_html = build_grid_convergence_plot_subsection(participants, case_id, plot_spec, requirement=requirement)
        if not metric_html.strip():
            continue
        if metric is not None:
            # Dedicated water/ice/evaporation pages contain only one metric,
            # so an extra open/close control adds hierarchy without choice.
            html += f"""
            <div class="single-convergence-metric"
                 data-variable-key="{escape(plot_spec['plot_key'])}"
                 data-variable-label="{escape(plot_spec['title'])}">
              {metric_html}
            </div>
            """
            continue
        # One collapsible card per convergence variable. Original values,
        # L1-relative differences, statistics, roughness groups, slices, and
        # diameter distributions remain together inside the same card.
        html += f"""
        <details class="grid-section plot-subsection analysis-metric-section convergence-metric-section"
                 data-variable-key="{escape(plot_spec['plot_key'])}"
                 data-variable-label="{escape(plot_spec['title'])}">
          <summary>{escape(plot_spec['title'])}</summary>
          <div class="convergence-metric-content">
            {metric_html}
          </div>
        </details>
        """

    if not html.strip():
        label = "optional (O)" if requirement == "optional" else "required (R)"
        return f'<div class="warning-box">No {escape(label)} {escape(category.upper())} grid-convergence data were found for this selection.</div>'

    return f"""
    <section class="plot-subsection grid-convergence-section plot-filter-scope">
      <h3>{escape(('Optional ' if requirement == 'optional' else '') + {'all': 'Grid convergence', 'cfd': 'CFD grid convergence', 'icing': 'Icing grid convergence'}[category])}</h3>
      {html}
    </section>
    """


def build_water_mass_analysis_section(participants, case_id: str, requirement: str = "required") -> str:
    """Build the dedicated ratio-only water mass analysis section."""
    ratio_html = "".join(
        build_grid_convergence_plot_subsection(participants, case_id, plot_spec, requirement=requirement)
        for plot_spec in WATER_MASS_ANALYSIS_PLOTS
        if plot_matches_variable_filter(plot_spec)
    )
    if not ratio_html.strip():
        return ""
    section_title = "Water Mass Analysis | Roughness Height 1 mm" if case_id == "TC_ONERAM6" else "Water Mass Analysis"
    return f"""
    <section class="water-mass-analysis plot-filter-scope">
      <h3>{escape(section_title)}</h3>
      {ratio_html}
    </section>
    """


def _cutdata_roughness_key(zone_name: str) -> str:
    """Extract the roughness identifier needed by the beta-max analysis."""
    text = zone_name.strip()
    if re.search(r"(?:^|_)KS_(?:0|0p0|0\.0|smooth|none)(?:mm|m)?(?:_|$)", text, re.IGNORECASE):
        return "smooth"
    if re.search(r"(?:^|_)(?:VARIABLE_ROUGHNESS|VAR_ROUGHNESS|KS_VARIABLE|KS_VAR)(?:_|$)", text, re.IGNORECASE):
        return "variable_roughness"
    match = re.search(r"(?:^|_)KS_(?P<value>[0-9]+(?:p[0-9]+|\.[0-9]+)?)(?P<unit>mm|m)?(?:_|$)", text, re.IGNORECASE)
    if match is None:
        return "default_roughness"
    value = float(match.group("value").replace("p", "."))
    if (match.group("unit") or "mm").lower() == "m":
        value *= 1000.0
    return f"{value:g}mm"


def build_beta_max_analysis_section(participants, case_id: str, _slice_filter: str | None = None) -> str:
    """Plot beta maximum and impingement extent by participant/grid/distribution."""
    if VARIABLE_FILTER is not None and not VARIABLE_FILTER.intersection(
        {
            "beta_max", "betamax", "beta_peak_s", "peak_beta_s", "s_peak",
            "impingement", "impingement_limits", "impingement_width",
        }
    ):
        return ""
    grid_order = ["L1", "L2", "L3", "L4"]
    metrics: dict[tuple[str, str, str, str, str], dict[str, float]] = {}

    for grid_level in grid_order:
        for participant, _, _, dataset_data in iter_grid_datasets(participants, case_id, grid_level):
            cut_data = getattr(dataset_data, "cp_beta_cut_data", None)
            cut_variables = {str(variable).lower() for variable in getattr(cut_data, "variables", [])} if cut_data is not None else set()
            if cut_data is None or not cut_variables.intersection({"beta", "collectionefficiency"}):
                cut_data = getattr(dataset_data, "cut_data", None)
            if cut_data is None:
                continue
            for zone_name, zone in cut_data.zones.items():
                zone_info = parse_ipw3_zone_name(zone_name)
                if zone_info is None:
                    continue
                bin_set = zone_info["bins"]
                slice_key = zone_info["slice"]
                if case_id == "TC_ONERAM6":
                    slice_position = decode_slice_position(slice_key)
                    if slice_position is not None:
                        # Merge equivalent submitted spellings such as 0p1 /
                        # 0p10 and 1p4 / 1p40 into one physical slice.
                        slice_key = f"{slice_position:g}"
                roughness_key = _cutdata_roughness_key(zone_name)
                if case_id == "TC_ONERAM6" and roughness_key != "1mm":
                    continue
                beta_column = find_column_case_insensitive(zone.data.columns, ["Beta", "BETA", "CollectionEfficiency"])
                s_column = find_column_case_insensitive(zone.data.columns, ["s", "S"])
                if beta_column is None or s_column is None:
                    continue
                beta_values = pd.to_numeric(zone.data[beta_column], errors="coerce")
                s_values = pd.to_numeric(zone.data[s_column], errors="coerce")
                valid = np.isfinite(beta_values) & np.isfinite(s_values) & (beta_values > -998.0)
                beta_values = beta_values[valid]
                s_values = s_values[valid]
                if beta_values.empty:
                    continue
                impinging_s = s_values[beta_values > 0.0001]
                if impinging_s.empty:
                    continue
                key = (participant.participant_id, bin_set, roughness_key, slice_key, grid_level)
                zone_metrics = {
                    "beta_max": float(beta_values.max()),
                    "s_peak": float(s_values.loc[beta_values.idxmax()]),
                    "s_lower": float(impinging_s.min()),
                    "s_upper": float(impinging_s.max()),
                }
                zone_metrics["width"] = zone_metrics["s_upper"] - zone_metrics["s_lower"]
                previous = metrics.get(key)
                if previous is None:
                    metrics[key] = zone_metrics
                else:
                    if zone_metrics["beta_max"] > previous["beta_max"]:
                        previous["beta_max"] = zone_metrics["beta_max"]
                        previous["s_peak"] = zone_metrics["s_peak"]
                    previous["s_lower"] = min(previous["s_lower"], zone_metrics["s_lower"])
                    previous["s_upper"] = max(previous["s_upper"], zone_metrics["s_upper"])
                    previous["width"] = previous["s_upper"] - previous["s_lower"]

    # ONERA M6 has several spanwise slices.  Build a complete, independent
    # beta/impingement analysis for each one instead of combining slices as
    # separate traces in the same figures and statistics.
    if case_id == "TC_ONERAM6" and _slice_filter is None:
        slice_keys = sorted({key[3] for key in metrics}, key=lambda value: decode_slice_position(value))
        slice_sections = "".join(
            f"""
            <details class="grid-section analysis-metric-section onera-slice-analysis">
              <summary>Slice Y={escape(slice_key)}</summary>
              {build_beta_max_analysis_section(participants, case_id, _slice_filter=slice_key)}
            </details>
            """
            for slice_key in slice_keys
        )
        return f'<section class="beta-max-analysis plot-filter-scope">{slice_sections}</section>' if slice_sections else ""

    if _slice_filter is not None:
        metrics = {key: value for key, value in metrics.items() if key[3] == _slice_filter}

    slice_suffix = f"_{slugify(_slice_filter)}" if _slice_filter is not None else ""
    slice_title = f" | Slice Y={_slice_filter}" if _slice_filter is not None else ""

    bin_sets = sorted({key[1] for key in metrics}, key=lambda value: (bin_count_from_bin_set(value) or 10**9, value))
    metric_distribution_html = {"beta_max": "", "s_peak": "", "width": ""}
    cell_counts = grid_cell_counts_for_case(case_id)
    for bin_set in bin_sets:
        figures = {"beta_max": go.Figure(), "s_peak": go.Figure(), "limits": go.Figure(), "width": go.Figure()}
        series_keys = sorted({(pid, roughness, slice_key) for pid, bins, roughness, slice_key, _ in metrics if bins == bin_set})
        interval_count = len(series_keys)
        interval_span = 0.56
        for series_index, (participant_id, roughness_key, slice_key) in enumerate(series_keys):
            levels = [level for level in grid_order if (participant_id, bin_set, roughness_key, slice_key, level) in metrics]
            if not levels:
                continue
            x_values = [grid_convergence_coordinate(cell_counts[int(level[1:])], cell_counts[1]) for level in levels]
            roughness_label = format_icing_roughness_title(roughness_key)
            trace_name = participant_id + (f" | {roughness_label}" if roughness_key != "default_roughness" else "")
            trace_name += f" | Y={slice_key}"
            common = dict(
                x=x_values,
                mode="lines+markers", name=trace_name, legendgroup=trace_name,
                legendrank=participant_legend_rank(participant_id),
                line={"color": participant_color(participant_id)}, marker=participant_marker(participant_id),
                customdata=[[level, cell_counts[int(level[1:])]] for level in levels],
            )
            figures["beta_max"].add_trace(go.Scatter(
                **common, y=[metrics[(participant_id, bin_set, roughness_key, slice_key, level)]["beta_max"] for level in levels],
                hovertemplate="Participant: %{fullData.name}<br>Grid=%{customdata[0]}<br>Num cells=%{customdata[1]:,.0f}<br>βmax=%{y:.6g}<extra></extra>",
            ))
            figures["s_peak"].add_trace(go.Scatter(
                **common, y=[metrics[(participant_id, bin_set, roughness_key, slice_key, level)]["s_peak"] for level in levels],
                hovertemplate="Participant: %{fullData.name}<br>Grid=%{customdata[0]}<br>Num cells=%{customdata[1]:,.0f}<br>Peak β s position=%{y:.6g} m<extra></extra>",
            ))
            interval_offset = (
                0.0 if interval_count <= 1
                else -interval_span / 2.0 + interval_span * series_index / (interval_count - 1)
            )
            for level_index, level in enumerate(levels):
                level_metrics = metrics[(participant_id, bin_set, roughness_key, slice_key, level)]
                grid_index = grid_order.index(level)
                figures["limits"].add_trace(go.Scatter(
                    x=[grid_index + interval_offset, grid_index + interval_offset],
                    y=[level_metrics["s_lower"], level_metrics["s_upper"]],
                    mode="lines+markers", name=trace_name, legendgroup=trace_name,
                    showlegend=level_index == 0, legendrank=participant_legend_rank(participant_id),
                    line={"color": participant_color(participant_id), "width": 3, "dash": "solid"},
                    marker={**participant_marker(participant_id), "size": 8},
                    customdata=[[level, "lower"], [level, "upper"]],
                    hovertemplate="Participant: %{fullData.name}<br>Grid=%{customdata[0]}<br>%{customdata[1]} impingement limit=%{y:.6g} m<extra></extra>",
                ))
            figures["width"].add_trace(go.Scatter(
                **common, y=[metrics[(participant_id, bin_set, roughness_key, slice_key, level)]["width"] for level in levels],
                hovertemplate="Participant: %{fullData.name}<br>Grid=%{customdata[0]}<br>Impingement width=%{y:.6g} m<extra></extra>",
            ))
        if not figures["beta_max"].data:
            continue

        bin_count = bin_count_from_bin_set(bin_set)
        label = f"{bin_count}-bin distribution" if bin_count != 1 else "Single-bin distribution"
        plot_definitions = (
            ("beta_max", "beta_max_vs_n", "Peak collection efficiency [-]", "Maximum collection efficiency", "β<sub>max</sub> = max β(s) for β(s) &gt; 0."),
            ("s_peak", "peak_beta_s_vs_n", "Peak β s position [m]", "Peak β s position", "s<sub>βmax</sub> is the submitted surface-coordinate position where β(s) reaches its maximum."),
            ("width", "impingement_width_vs_n", "Surface impingement width [m]", "Impingement width", "W<sub>imp</sub> = s<sub>upper</sub> − s<sub>lower</sub> over submitted points where β(s) &gt; 0.0001, using s computed along the already rotated airfoil."),
        )
        for metric_key, style_key, y_label, title_prefix, formula in plot_definitions:
            fig = figures[metric_key]
            style_xy_figure(fig, case_id, style_key, "Grid level", y_label)
            style_grid_level_x_axis(fig, case_id)
            filename = f"{slugify(case_id)}_{metric_key}_{slugify(bin_set)}{slice_suffix}"
            title = f"{title_prefix} | {label}{slice_title}"
            roughness_note = simplify_icing_participant_legend(fig)
            if case_id == "TC_ONERAM6":
                roughness_note = ""
            # Show the submitted values first, followed by the signed change
            # from each participant's L1 result (L1 = 0%).
            figure_html = grid_convergence_figure_pair_html(
                fig, case_id, style_key, filename, title,
            )
            figure_html = re.sub(
                r'<p class="plot-description">Formula:.*?</p>',
                "", figure_html, flags=re.DOTALL,
            )
            metric_distribution_html[metric_key] += f"""
            <section class="slice-plot-group">
              <h4>{escape(label)}</h4>
              {roughness_note}
              <div class="plot-container">{figure_html}</div>
            </section>
            """

    # At each fixed grid level, show how βmax and impingement width change as
    # the submitted droplet distribution is refined from BINS01 to BINS15.
    distribution_metric_specs = (
        ("beta_max", "beta_max_vs_bins", "Peak collection efficiency [-]", "β<sub>max</sub> = max β(s) for the selected grid and distribution."),
        ("s_peak", "peak_beta_s_vs_bins", "Peak β s position [m]", "s<sub>βmax</sub> is the submitted surface-coordinate position where β(s) reaches its maximum for the selected grid and distribution."),
        ("width", "impingement_width_vs_bins", "Surface impingement width [m]", "W<sub>imp</sub> = s<sub>upper</sub> − s<sub>lower</sub> over submitted points where β(s) &gt; 0.0001 for the selected grid and distribution."),
    )
    for metric_key, style_key, y_label, formula in distribution_metric_specs:
        for grid_level in grid_order:
            fig = go.Figure()
            series_keys = sorted({(pid, roughness, slice_key) for pid, _, roughness, slice_key, level in metrics if level == grid_level})
            for participant_id, roughness_key, slice_key in series_keys:
                submitted_bins = [bin_set for bin_set in bin_sets if (participant_id, bin_set, roughness_key, slice_key, grid_level) in metrics]
                if not submitted_bins:
                    continue
                roughness_label = format_icing_roughness_title(roughness_key)
                trace_name = participant_id + (f" | {roughness_label}" if roughness_label else "") + f" | Y={slice_key}"
                fig.add_trace(go.Scatter(
                    x=submitted_bins,
                    y=[metrics[(participant_id, bin_set, roughness_key, slice_key, grid_level)][metric_key] for bin_set in submitted_bins],
                    mode="lines+markers", name=trace_name, legendgroup=trace_name,
                    legendrank=participant_legend_rank(participant_id),
                    line={"color": participant_color(participant_id)}, marker=participant_marker(participant_id),
                    customdata=[[bin_set] for bin_set in submitted_bins],
                    hovertemplate="Participant: %{fullData.name}<br>Grid=" + grid_level + "<br>Distribution=%{x}<br>Value=%{y:.6g}<extra></extra>",
                ))
            if not fig.data:
                continue
            style_xy_figure(fig, case_id, style_key, "Droplet distribution", y_label)
            display_bin_sets = list(reversed(bin_sets))
            fig.update_xaxes(
                type="category", categoryorder="array", categoryarray=display_bin_sets,
                tickmode="array", tickvals=display_bin_sets, ticktext=[display_bin_set(value) for value in display_bin_sets],
                range=None, autorange=True,
                title={"text": "Droplet distribution (15 → 01)"},
            )
            filename = f"{slugify(case_id)}_{metric_key}_distribution_convergence_{grid_level.lower()}{slice_suffix}"
            title_prefix = {
                "beta_max": "βmax",
                "s_peak": "Peak β s position",
                "width": "Impingement width",
            }[metric_key]
            title = f"{title_prefix} droplet-distribution convergence | {grid_level}{slice_title}"
            roughness_note = simplify_icing_participant_legend(fig)
            if case_id == "TC_ONERAM6":
                roughness_note = ""
            metric_distribution_html[metric_key] += f"""
            <section class="slice-plot-group">
              <h4>{escape(title)}</h4>
              {roughness_note}
              <div class="plot-container">{distribution_figure_pair_html(fig, case_id, style_key, filename, title)}</div>
            </section>
            """

    if not any(metric_distribution_html.values()):
        return ""
    metric_titles = {
        "beta_max": "Maximum Collection Efficiency (βmax)",
        "s_peak": "Peak β s Position",
        "width": "Impingement Width",
    }
    collapsible_metrics = "".join(
        f"""
        <details class="grid-section analysis-metric-section">
          <summary>{escape(metric_titles[metric_key])}</summary>
          {metric_distribution_html[metric_key]}
        </details>
        """
        for metric_key in ("beta_max", "s_peak", "width")
        if metric_distribution_html[metric_key]
    )
    return f"""
    <section class="beta-max-analysis plot-filter-scope">
      {collapsible_metrics}
    </section>
    """


def _participant_mass_values(participant, case_id: str, requirement: str, mass_kind: str) -> dict[tuple[str, str, str], float]:
    """Collect ice or water mass by (bin set, roughness, grid level)."""
    case_data = participant.cases.get(case_id)
    if case_data is None or case_data.grid_convergence_data is None:
        return {}
    values: dict[tuple[str, str, str], float] = {}
    for zone_name, zone in case_data.grid_convergence_data.zones.items():
        if "by_diameter" in zone_name.lower() or not grid_convergence_zone_matches_requirement(zone, requirement):
            continue
        bin_set = extract_icing_bin_set_from_zone_name(zone_name)
        if bin_set is None:
            continue
        mass_candidates = ["ICE_MASS", "IceMass"] if mass_kind == "ice" else ["WATER_MASS", "WaterMass"]
        mass_column = find_column_case_insensitive(zone.data.columns, mass_candidates)
        x_column = find_column_case_insensitive(zone.data.columns, ["N"])
        grid_column = find_column_case_insensitive(zone.data.columns, ["GRID_LEVEL", "GridLevel"])
        if mass_column is None or x_column is None:
            continue
        data = valid_numeric_rows(zone.data.copy(), x_column, mass_column, positive_columns={x_column})
        roughness_key = extract_icing_roughness_key_from_zone_name(zone_name)
        for _, row in data.iterrows():
            level_number = grid_level_number_from_value(row[grid_column]) if grid_column is not None else None
            if level_number is None:
                level_number = grid_level_number_from_value(row[x_column])
            if level_number is None:
                continue
            values.setdefault((bin_set, roughness_key, f"L{level_number}"), float(row[mass_column]))
    return values


def _participant_beta_max_values(participant, case_id: str) -> dict[tuple[str, str, str, str], float]:
    """Collect βmax by (bin set, roughness, slice, grid level)."""
    values: dict[tuple[str, str, str, str], float] = {}
    for grid_level in ("L1", "L2", "L3", "L4"):
        for _, _, _, dataset_data in iter_grid_datasets([participant], case_id, grid_level):
            cut_data = getattr(dataset_data, "cp_beta_cut_data", None)
            cut_variables = {str(variable).lower() for variable in getattr(cut_data, "variables", [])} if cut_data is not None else set()
            if cut_data is None or not cut_variables.intersection({"beta", "collectionefficiency"}):
                cut_data = getattr(dataset_data, "cut_data", None)
            if cut_data is None:
                continue
            for zone_name, zone in cut_data.zones.items():
                zone_info = parse_ipw3_zone_name(zone_name)
                if zone_info is None:
                    continue
                beta_column = find_column_case_insensitive(zone.data.columns, ["Beta", "BETA", "CollectionEfficiency"])
                if beta_column is None:
                    continue
                beta = pd.to_numeric(zone.data[beta_column], errors="coerce")
                beta = beta[np.isfinite(beta) & (beta > -998.0)]
                if beta.empty:
                    continue
                key = (zone_info["bins"], _cutdata_roughness_key(zone_name), zone_info["slice"], grid_level)
                values[key] = max(values.get(key, -math.inf), float(beta.max()))
    return values


def _build_ae3933_beta_max_comparison(participants) -> str:
    """Compare matched AE3933 and AE3932 βmax values at every grid level."""
    comparison_case = "TC_NACA0012_AE3933"
    reference_case = "TC_NACA0012_AE3932"
    grid_order = ["L1", "L2", "L3", "L4"]
    rows_by_distribution: dict[str, dict[tuple[str, str, str], list[dict[str, Any]]]] = {}
    for participant in participants:
        comparison_values = _participant_beta_max_values(participant, comparison_case)
        reference_values = _participant_beta_max_values(participant, reference_case)
        for key in sorted(set(comparison_values).intersection(reference_values)):
            bin_set, roughness_key, slice_key, grid_level = key
            reference_value = reference_values[key]
            comparison_value = comparison_values[key]
            difference = comparison_value - reference_value
            percent = difference / reference_value * 100.0 if reference_value != 0.0 else None
            rows_by_distribution.setdefault(bin_set, {}).setdefault(
                (participant.participant_id, roughness_key, slice_key), []
            ).append({"grid_level": grid_level, "ae3932": reference_value, "ae3933": comparison_value, "difference": difference, "percent": percent})

    output = ""
    cell_counts = grid_cell_counts_for_case(comparison_case)
    for bin_set in sorted(rows_by_distribution, key=lambda value: (bin_count_from_bin_set(value) or 10**9, value)):
        difference_fig, percent_fig = go.Figure(), go.Figure()
        for (participant_id, roughness_key, slice_key), rows in sorted(rows_by_distribution[bin_set].items()):
            rows = sorted(rows, key=lambda row: grid_order.index(row["grid_level"]))
            levels = [row["grid_level"] for row in rows]
            roughness_label = format_icing_roughness_title(roughness_key)
            trace_name = participant_id + (f" | {roughness_label}" if roughness_label else "") + f" | Y={slice_key}"
            common = dict(
                x=[grid_convergence_coordinate(cell_counts[int(level[1:])], cell_counts[1]) for level in levels],
                mode="lines+markers", name=trace_name, legendgroup=trace_name,
                legendrank=participant_legend_rank(participant_id), line={"color": participant_color(participant_id)},
                marker=participant_marker(participant_id),
                customdata=[[row["grid_level"], row["ae3932"], row["ae3933"]] for row in rows],
            )
            difference_fig.add_trace(go.Scatter(
                **common, y=[row["difference"] for row in rows],
                hovertemplate="Participant: %{fullData.name}<br>Grid=%{customdata[0]}<br>βmax,AE3932=%{customdata[1]:.6g}<br>βmax,AE3933=%{customdata[2]:.6g}<br>Δβmax=%{y:.6g}<extra></extra>",
            ))
            percent_fig.add_trace(go.Scatter(
                **common, y=[row["percent"] for row in rows],
                hovertemplate="Participant: %{fullData.name}<br>Grid=%{customdata[0]}<br>βmax,AE3932=%{customdata[1]:.6g}<br>βmax,AE3933=%{customdata[2]:.6g}<br>Δβmax=%{y:.6g}%<extra></extra>",
            ))
        if not difference_fig.data:
            continue
        for fig, style_key, y_label in (
            (difference_fig, "ae3933_minus_ae3932_beta_max", "βmax,AE3933 − βmax,AE3932 [-]"),
            (percent_fig, "ae3933_minus_ae3932_beta_max_percent", "βmax difference relative to AE3932 [%]"),
        ):
            style_xy_figure(fig, comparison_case, style_key, "Grid level", y_label)
            style_grid_level_x_axis(fig, comparison_case)
            fig.add_hline(y=0.0, line_dash="dash", line_color="black", line_width=1.5)
            for trace in fig.data:
                if trace.customdata is not None:
                    trace.x = [str(row[0]).upper() for row in trace.customdata]
            harmonize_comparison_figure(fig, grid_order, GRID_CONVERGENCE_NORMALIZATION.get("grid_axis_title", "Grid level"))
        roughness_note = simplify_icing_participant_legend(difference_fig)
        simplify_icing_participant_legend(percent_fig)
        style_relative_difference_figure(percent_fig)
        bin_count = bin_count_from_bin_set(bin_set)
        label = f"{bin_count}-bin distribution" if bin_count != 1 else "Single-bin distribution"
        slug = f"{slugify(comparison_case)}_comparison_with_3932_beta_max_{slugify(bin_set)}"
        statistics_result = grid_statistics_html(
            difference_fig, slug + "_difference",
            f"Comparison with 3932 | {label} | βmax difference",
            case_id=comparison_case, separate_parts=True, compact_parts=True,
        )
        statistics_table, box_plot_html = statistics_result if isinstance(statistics_result, tuple) else ("", "")
        output += f"""
        <section class="slice-plot-group">
          {roughness_note}
          <div class="convergence-comparison-grid">
            <section class="convergence-comparison-panel">
              <h5>βmax change: AE3933 − AE3932 | {escape(label)}</h5>
              {figure_to_html_div(difference_fig, slug + '_difference', f'Comparison with 3932 | {label} | βmax difference')}
            </section>
            <section class="convergence-comparison-panel">
              {figure_to_html_div(percent_fig, slug + '_percent', f'Comparison with 3932 | {label} | βmax percentage difference')}
            </section>
          </div>
          <div class="convergence-statistics-grid">
            <section class="convergence-comparison-table">{statistics_table}</section>
            <section class="convergence-comparison-panel">{box_plot_html}</section>
          </div>
        </section>
        """
    return output


def _build_ae3933_mass_comparison_metric(participants, requirement: str, mass_kind: str) -> str:
    """Compare one AE3933 integrated mass against matching AE3932 data."""
    mass_label = "Ice" if mass_kind == "ice" else "Water"
    comparison_case = "TC_NACA0012_AE3933"
    reference_case = "TC_NACA0012_AE3932"
    grid_order = ["L1", "L2", "L3", "L4"]
    rows_by_distribution: dict[str, dict[tuple[str, str], list[dict[str, Any]]]] = {}

    for participant in participants:
        comparison_values = _participant_mass_values(participant, comparison_case, requirement, mass_kind)
        reference_values = _participant_mass_values(participant, reference_case, requirement, mass_kind)
        for key in sorted(set(comparison_values).intersection(reference_values)):
            bin_set, roughness_key, grid_level = key
            # Submitted masses are kilograms; NACA0012 comparison plots use grams.
            reference_value = reference_values[key] * 1000.0
            comparison_value = comparison_values[key] * 1000.0
            difference = comparison_value - reference_value
            percent = difference / reference_value * 100.0 if reference_value != 0.0 else None
            rows_by_distribution.setdefault(bin_set, {}).setdefault(
                (participant.participant_id, roughness_key), []
            ).append({
                "grid_level": grid_level,
                "ae3932": reference_value,
                "ae3933": comparison_value,
                "difference": difference,
                "percent": percent,
            })

    distribution_html = ""
    cell_counts = grid_cell_counts_for_case(comparison_case)
    for bin_set in sorted(rows_by_distribution, key=lambda value: (bin_count_from_bin_set(value) or 10**9, value)):
        difference_fig = go.Figure()
        percent_fig = go.Figure()
        for (participant_id, roughness_key), rows in sorted(rows_by_distribution[bin_set].items()):
            rows = sorted(rows, key=lambda row: grid_order.index(row["grid_level"]))
            levels = [row["grid_level"] for row in rows]
            x_values = [grid_convergence_coordinate(cell_counts[int(level[1:])], cell_counts[1]) for level in levels]
            customdata = [[row["grid_level"], row["ae3932"], row["ae3933"]] for row in rows]
            roughness_label = format_icing_roughness_title(roughness_key)
            trace_name = participant_id + (f" | {roughness_label}" if roughness_key != "unspecified" else "")
            common = dict(
                x=x_values, mode="lines+markers", name=trace_name,
                legendgroup=trace_name, legendrank=participant_legend_rank(participant_id),
                line={"color": participant_color(participant_id)}, marker=participant_marker(participant_id),
                customdata=customdata,
            )
            difference_fig.add_trace(go.Scatter(
                **common,
                y=[row["difference"] for row in rows],
                hovertemplate="Participant: %{fullData.name}<br>Grid=%{customdata[0]}<br>AE3932=%{customdata[1]:.6g} g<br>AE3933=%{customdata[2]:.6g} g<br>AE3933 − AE3932=%{y:.6g} g<extra></extra>",
            ))
            percent_fig.add_trace(go.Scatter(
                **common,
                y=[row["percent"] for row in rows],
                hovertemplate="Participant: %{fullData.name}<br>Grid=%{customdata[0]}<br>AE3932=%{customdata[1]:.6g} g<br>AE3933=%{customdata[2]:.6g} g<br>Difference relative to AE3932=%{y:.6g}%<extra></extra>",
            ))

        if not difference_fig.data:
            continue
        style_xy_figure(difference_fig, comparison_case, f"ae3933_minus_ae3932_{mass_kind}_mass", "Grid level", f"AE3933 {mass_kind} mass − AE3932 {mass_kind} mass [g]")
        style_grid_level_x_axis(difference_fig, comparison_case)
        style_xy_figure(percent_fig, comparison_case, f"ae3933_minus_ae3932_{mass_kind}_mass_percent", "Grid level", f"{mass_label}-mass difference relative to AE3932 [%]")
        style_grid_level_x_axis(percent_fig, comparison_case)
        percent_fig.add_hline(y=0.0, line_dash="dash", line_color="black", line_width=1.5)
        for fig in (difference_fig, percent_fig):
            for trace in fig.data:
                if trace.customdata is not None:
                    trace.x = [str(row[0]).upper() for row in trace.customdata]
            harmonize_comparison_figure(fig, grid_order, GRID_CONVERGENCE_NORMALIZATION.get("grid_axis_title", "Grid level"))
        roughness_note = simplify_icing_participant_legend(difference_fig)
        simplify_icing_participant_legend(percent_fig)
        style_relative_difference_figure(percent_fig)
        bin_count = bin_count_from_bin_set(bin_set)
        distribution_label = f"{bin_count}-bin distribution" if bin_count != 1 else "Single-bin distribution"
        slug = f"{slugify(comparison_case)}_comparison_with_3932_{mass_kind}_mass_{slugify(bin_set)}"
        statistics_result = grid_statistics_html(
            difference_fig, slug + "_difference",
            f"Comparison with 3932 | {distribution_label} | {mass_label}-mass difference",
            case_id="TC_NACA0012_AE3933", separate_parts=True, compact_parts=True,
        )
        difference_statistics_table, difference_box_plot = statistics_result if isinstance(statistics_result, tuple) else ("", "")
        distribution_html += f"""
        <section class="slice-plot-group">
          {roughness_note}
          <div class="convergence-comparison-grid">
            <section class="convergence-comparison-panel">
              <h5>{mass_label}-mass difference: AE3933 − AE3932 | {escape(distribution_label)}</h5>
              {figure_to_html_div(difference_fig, slug + '_difference', f'Comparison with 3932 | {distribution_label} | {mass_label}-mass difference')}
            </section>
            <section class="convergence-comparison-panel">
              {figure_to_html_div(percent_fig, slug + '_percent', f'Comparison with 3932 | {distribution_label} | Percentage difference')}
            </section>
          </div>
          <div class="convergence-statistics-grid">
            <section class="convergence-comparison-table">{difference_statistics_table}</section>
            <section class="convergence-comparison-panel">{difference_box_plot}</section>
          </div>
        </section>
        """

    return distribution_html


def build_ae3933_ice_mass_comparison_section(participants, requirement: str = "required") -> str:
    """Compare AE3933 ice and water masses against matching AE3932 data."""
    ice_html = _build_ae3933_mass_comparison_metric(participants, requirement, "ice")
    water_html = _build_ae3933_mass_comparison_metric(participants, requirement, "water")
    beta_html = _build_ae3933_beta_max_comparison(participants)
    if not ice_html and not water_html and not beta_html:
        return ""
    metric_sections = "".join(
        f"""
        <details class="grid-section analysis-metric-section convergence-metric-section">
          <summary>{escape(label)}</summary>
          <div class="convergence-metric-content">{content}</div>
        </details>
        """
        for label, content in (
            ("Ice Mass Comparison", ice_html),
            ("Water Mass Comparison", water_html),
            ("Maximum Collection Efficiency (βmax) Comparison", beta_html),
        )
        if content
    )
    return f"""
    <section id="comparison-with-3932" class="comparison-with-3932 plot-filter-scope">
      <h3>Comparison with 3932</h3>
      <p class="plot-description">Matched by participant, grid level, droplet distribution, and roughness.</p>
      {metric_sections}
    </section>
    """

def collect_diameter_groups(participants, case_id: str, plot_spec: dict[str, Any], requirement: str = "required") -> list[tuple[str, str, float, float]]:
    groups: set[tuple[str, str, float, float]] = set()

    for participant, case_data in iter_case_data(participants, case_id):
        if case_data.grid_convergence_data is None:
            continue

        for zone_name, zone in case_data.grid_convergence_data.zones.items():
            if "by_diameter" not in zone_name.lower():
                continue
            if not grid_convergence_zone_matches_requirement(zone, requirement):
                continue

            diameter_column = find_column_case_insensitive(zone.data.columns, ["DIAMETER", "Diameter"])
            bin_set_column = find_column_case_insensitive(zone.data.columns, ["BIN_SET", "BinSet"])
            bin_column = find_column_case_insensitive(zone.data.columns, ["BIN", "Bin"])

            if diameter_column is None or bin_set_column is None or bin_column is None:
                continue

            for _, row in zone.data.iterrows():
                diameter = row[diameter_column]
                bin_set = str(row[bin_set_column])
                bin_number = row[bin_column]

                if diameter <= -998.0 or bin_number <= -998.0:
                    continue

                roughness_key = extract_icing_roughness_key_from_zone_name(zone_name)
                groups.add((roughness_key, bin_set, float(bin_number), float(diameter)))

    return sorted(groups, key=lambda item: (item[0], item[1], item[2]))

def build_grid_convergence_diameter_figure(participants, case_id: str, plot_spec: dict[str, Any], target_bin_set: str, target_bin_number: float, target_diameter: float, requirement: str = "required", roughness_filter: str | None = None) -> tuple[go.Figure, int, list[str]]:
    fig = go.Figure()
    trace_count = 0
    skipped_notes: list[str] = []
    seen_trace_keys: set[tuple[str, str, str, str, float]] = set()

    for participant, case_data in iter_case_data(participants, case_id):
        if case_data.grid_convergence_data is None:
            continue

        label = participant_label(participant)
        color = participant_color(participant.participant_id)

        for zone_name, zone in case_data.grid_convergence_data.zones.items():
            if "by_diameter" not in zone_name.lower():
                continue
            if not grid_convergence_zone_matches_requirement(zone, requirement):
                continue
            if roughness_filter is not None and extract_icing_roughness_key_from_zone_name(zone_name) != roughness_filter:
                continue

            x_column = find_column_case_insensitive(zone.data.columns, plot_spec["x_candidates"])
            y_column = find_column_case_insensitive(zone.data.columns, case_ordered_y_candidates(case_id, plot_spec["y_candidates"]))
            diameter_column = find_column_case_insensitive(zone.data.columns, ["DIAMETER", "Diameter"])
            bin_set_column = find_column_case_insensitive(zone.data.columns, ["BIN_SET", "BinSet"])
            bin_column = find_column_case_insensitive(zone.data.columns, ["BIN", "Bin"])
            grid_column = find_column_case_insensitive(zone.data.columns, ["GRID_LEVEL", "GridLevel"])

            if x_column is None or y_column is None or diameter_column is None or bin_set_column is None or bin_column is None:
                continue

            data = valid_numeric_rows(zone.data.copy(), x_column, y_column, positive_columns={x_column})
            data = data[data[bin_set_column].astype(str) == str(target_bin_set)]
            data = data[data[bin_column].astype(float) == float(target_bin_number)]
            data = data[data[diameter_column].astype(float) == float(target_diameter)]
            data = add_grid_spacing_column(data, case_id, x_column, grid_column=grid_column)

            if data.empty:
                continue

            data = data.sort_values(GRID_SPACING_COLUMN)

            trace_key = (participant.participant_id, plot_spec["plot_key"], zone_name, target_bin_set, float(target_diameter))

            if trace_key in seen_trace_keys:
                continue

            seen_trace_keys.add(trace_key)

            roughness_key = extract_icing_roughness_key_from_zone_name(zone_name)
            roughness_label = format_icing_roughness_title(roughness_key)
            # Roughness is stated in the plot heading/note, not repeated in
            # every participant legend entry.
            trace_label = label

            fig.add_trace(
                go.Scatter(
                    x=data[GRID_SPACING_COLUMN],
                    y=data[y_column],
                    mode="lines+markers",
                    name=trace_label,
                    legendgroup=f"{label}_{roughness_key}",
                    legendrank=participant_legend_rank(participant.participant_id),
                    line=dict(color=color),
                    marker=participant_marker(participant.participant_id),
                    customdata=data[["GRID_LEVEL_DISPLAY", "GRID_CELL_COUNT"]],
                    hovertemplate=(
                        f"Participant: {escape(label)}<br>"
                        f"Case: {escape(case_id)}<br>"
                        f"Distribution: {escape(display_bin_set(target_bin_set))}<br>"
                        f"Bin: {target_bin_number:g}<br>"
                        f"Diameter: {target_diameter:g} μm<br>"
                        + (f"Roughness: {escape(roughness_label)}<br>" if roughness_label else "")
                        + f"Zone: {escape(zone_name)}<br>"
                        f"{escape(format_x_hover_label(GRID_SPACING_COLUMN))}=%{{x:.6g}}<br>"
                        f"{escape(y_column)}=%{{y}}<br>"
                        "Grid level=%{customdata[0]}<br>"
                        "Num cells=%{customdata[1]:,.0f}<extra></extra>"
                    ),
                )
            )

            trace_count += 1

    style_xy_figure(fig, case_id, plot_spec["plot_key"], plot_spec["x_label"], plot_spec["y_label"])
    style_grid_level_x_axis(fig, case_id)
    apply_individual_plot_overrides(fig, case_id, plot_spec["plot_key"])
    return fig, trace_count, skipped_notes

def build_grid_convergence_diameter_subsection(
    participants,
    case_id: str,
    plot_spec: dict[str, Any],
    requirement: str = "required",
    target_bin_set: str | None = None,
    roughness_filter: str | None = None,
) -> str:
    groups = collect_diameter_groups(participants, case_id, plot_spec, requirement=requirement)
    if target_bin_set is not None:
        groups = [group for group in groups if group[1].upper() == target_bin_set.upper()]
    if roughness_filter is not None:
        groups = [group for group in groups if group[0].lower() == roughness_filter.lower()]

    if not groups:
        return ""

    html = ""

    for roughness_key, bin_set, bin_number, diameter in groups:
        fig, trace_count, skipped_notes = build_grid_convergence_diameter_figure(participants, case_id, plot_spec, bin_set, bin_number, diameter, requirement=requirement, roughness_filter=roughness_key)

        roughness_title = format_icing_roughness_title(roughness_key)
        title = " | ".join(part for part in [plot_spec['title'], roughness_title, display_bin_set(bin_set), f"Bin {bin_number:g}", f"D = {diameter:g} μm"] if part)

        if trace_count == 0:
            continue

        filename = f"{slugify(case_id)}_{plot_spec['filename_slug']}_{slugify(roughness_key)}_{slugify(bin_set)}_diameter_{bin_number:g}_{diameter:g}".replace(".", "p")
        figure_html = grid_convergence_figure_pair_html(fig, case_id, plot_spec["plot_key"], filename, title)

        html += f"""
        <section class="slice-plot-group">
          <h5>{escape(title)}</h5>
          <div class="plot-container">
            {figure_html}
          </div>
        </section>
        """

    if not html.strip():
        return ""

    return f"""
    <section class="plot-subsection" data-variable-key="{escape(plot_spec['plot_key'])}" data-variable-label="{escape(plot_spec['title'])}">
      <p class="plot-description">
        Diameter-resolved grid-convergence data. Each figure corresponds to one bin set and one droplet diameter. Legend: Participant ID.
      </p>
      {html}
    </section>
    """


def build_per_bin_analysis_section(
    participants,
    case_id: str,
    requirement: str = "optional",
    preferred_bin_set: str = "BINS15",
) -> str:
    """Build diameter-resolved convergence summaries from the largest O distribution."""
    available_groups = [
        group
        for plot_spec in OPTIONAL_ICING_DIAMETER_PLOTS
        for group in collect_diameter_groups(participants, case_id, plot_spec, requirement=requirement)
        if case_id != "TC_ONERAM6" or group[0].lower() == "1mm"
    ]
    available_bin_sets = sorted(
        {group[1] for group in available_groups},
        key=lambda value: (bin_count_from_bin_set(value) or -1, value),
        reverse=True,
    )
    if not available_bin_sets:
        return '<div class="warning-box">No optional diameter-resolved per-bin data were found for this case.</div>'
    selected_bin_set = next(
        (value for value in available_bin_sets if value.upper() == preferred_bin_set.upper()),
        available_bin_sets[0],
    )
    metric_sections = ""
    for plot_spec in OPTIONAL_ICING_DIAMETER_PLOTS:
        content = build_grid_convergence_diameter_subsection(
            participants,
            case_id,
            plot_spec,
            requirement=requirement,
            target_bin_set=selected_bin_set,
            roughness_filter="1mm" if case_id == "TC_ONERAM6" else None,
        )
        if not content:
            continue
        metric_sections += f"""
        <details class="grid-section analysis-metric-section convergence-metric-section" open>
          <summary>{escape(plot_spec['title'])}</summary>
          <div class="convergence-metric-content">{content}</div>
        </details>
        """
    if not metric_sections:
        return '<div class="warning-box">No usable optional diameter-resolved per-bin values were found for this case.</div>'
    selected_bin_count = bin_count_from_bin_set(selected_bin_set)
    selected_bin_label = (
        f"{selected_bin_count}-bin distribution" if selected_bin_count != 1
        else "Single-bin distribution"
    )
    roughness_note = " ONERA M6 is restricted to the 1 mm roughness condition." if case_id == "TC_ONERAM6" else ""
    return f"""
    <section class="per-bin-analysis plot-filter-scope">
      <h3>Per Bin Analysis — {escape(selected_bin_label)}</h3>
      <p class="plot-description">Source: optional (O) diameter-resolved submissions. Each summary holds one droplet bin and diameter fixed while comparing its L1–L4 convergence across participants. BINS15 is selected when available; otherwise the largest submitted distribution is used.{escape(roughness_note)}</p>
      {metric_sections}
    </section>
    """

def collect_cfd_roughness_keys(participants, case_id: str, plot_spec: dict[str, Any], requirement: str = "required") -> list[str]:
    roughness_keys: set[str] = set()

    for participant, case_data in iter_case_data(participants, case_id):
        if case_data.grid_convergence_data is None:
            continue

        for zone_name, zone in case_data.grid_convergence_data.zones.items():
            if not grid_convergence_zone_matches_requirement(zone, requirement):
                continue
            if "by_diameter" in zone_name.lower():
                continue

            roughness_key = extract_roughness_key_from_zone_name(zone_name)
            if roughness_key is None:
                continue

            x_column = find_column_case_insensitive(zone.data.columns, plot_spec["x_candidates"])
            y_column = find_column_case_insensitive(zone.data.columns, case_ordered_y_candidates(case_id, plot_spec["y_candidates"]))

            if x_column is not None and y_column is not None:
                roughness_keys.add(roughness_key)

    # The committee-required ONERA M6 CFD comparison is defined at KS = 1 mm.
    # Other ONERA roughness heights belong only on the optional CFD pages.
    if "ONERAM6" in case_id.upper() and requirement == "required":
        return ["1mm"] if "1mm" in roughness_keys else []

    preferred_order = ["smooth", "0.5mm", "1mm", "1.5mm", "variable_roughness"]
    ordered_keys = [key for key in preferred_order if key in roughness_keys]

    # Participants may submit valid fixed roughness heights that are not part of
    # the template's preferred set (for example, 1.0668 mm). Keep those keys so
    # their CFD coefficients are not silently omitted from the plots.
    additional_keys = sorted(
        roughness_keys.difference(preferred_order),
        key=roughness_sort_key,
    )
    return ordered_keys + additional_keys

def collect_combined_icing_grid_levels(participants, case_id: str, plot_spec: dict[str, Any], requirement: str = "required") -> list[str]:
    grid_levels: set[str] = set()

    for participant, case_data in iter_case_data(participants, case_id):
        if case_data.grid_convergence_data is None:
            continue

        for zone_name, zone in case_data.grid_convergence_data.zones.items():
            if "by_diameter" in zone_name.lower():
                continue
            if not grid_convergence_zone_matches_requirement(zone, requirement):
                continue

            bin_set = extract_icing_bin_set_from_zone_name(zone_name)
            if bin_set is None:
                continue

            x_column = find_column_case_insensitive(zone.data.columns, plot_spec["x_candidates"])
            y_column = find_column_case_insensitive(zone.data.columns, case_ordered_y_candidates(case_id, plot_spec["y_candidates"]))
            grid_column = find_column_case_insensitive(zone.data.columns, ["GRID_LEVEL", "GridLevel"])

            if x_column is None or y_column is None:
                continue

            for _, row in zone.data.iterrows():
                value = row[y_column]
                try:
                    if float(value) <= -998.0:
                        continue
                except (TypeError, ValueError):
                    continue
                level_number = None
                if grid_column is not None:
                    level_number = grid_level_number_from_value(row[grid_column])
                if level_number is None:
                    level_number = grid_level_number_from_value(row[x_column])
                if level_number is not None:
                    grid_levels.add(f"L{level_number}")

    return sorted(grid_levels, key=grid_level_number_from_value)


def collect_combined_icing_bin_sets(participants, case_id: str, plot_spec: dict[str, Any], requirement: str = "required") -> list[str]:
    """Return submitted required bin distributions that contain plottable values."""
    bin_sets: set[str] = set()

    for participant, case_data in iter_case_data(participants, case_id):
        if case_data.grid_convergence_data is None:
            continue

        for zone_name, zone in case_data.grid_convergence_data.zones.items():
            if "by_diameter" in zone_name.lower() or not grid_convergence_zone_matches_requirement(zone, requirement):
                continue

            bin_set = extract_icing_bin_set_from_zone_name(zone_name)
            if bin_set is None:
                continue

            x_column = find_column_case_insensitive(zone.data.columns, plot_spec["x_candidates"])
            y_column = find_column_case_insensitive(zone.data.columns, case_ordered_y_candidates(case_id, plot_spec["y_candidates"]))
            if x_column is None or y_column is None:
                continue

            data = valid_numeric_rows(zone.data, x_column, y_column, positive_columns={x_column})
            if not data.empty:
                bin_sets.add(bin_set)

    return sorted(bin_sets, key=lambda value: (bin_count_from_bin_set(value) or float("inf"), value))


def collect_combined_icing_roughness_keys(participants, case_id: str, plot_spec: dict[str, Any], requirement: str = "required") -> list[str]:
    roughness_keys: set[str] = set()
    for _, case_data in iter_case_data(participants, case_id):
        if case_data.grid_convergence_data is None:
            continue
        for zone_name, zone in case_data.grid_convergence_data.zones.items():
            if "by_diameter" in zone_name.lower() or not grid_convergence_zone_matches_requirement(zone, requirement):
                continue
            if extract_icing_bin_set_from_zone_name(zone_name) is None:
                continue
            y_column = find_column_case_insensitive(zone.data.columns, case_ordered_y_candidates(case_id, plot_spec["y_candidates"]))
            if y_column is not None and any(pd.to_numeric(zone.data[y_column], errors="coerce") > -998.0):
                roughness_keys.add(extract_icing_roughness_key_from_zone_name(zone_name))
    return sorted(roughness_keys, key=roughness_sort_key)


def build_distribution_icing_figure(participants, case_id: str, plot_spec: dict[str, Any], target_bin_set: str, requirement: str = "required", roughness_filter: str | None = None) -> tuple[go.Figure, int]:
    """Plot one droplet distribution across every available grid level."""
    fig = go.Figure()
    trace_count = 0

    for participant, case_data in iter_case_data(participants, case_id):
        if case_data.grid_convergence_data is None:
            continue

        label = participant_label(participant)
        color = participant_color(participant.participant_id)
        participant_rows: dict[str, list[pd.DataFrame]] = {}

        for zone_name, zone in case_data.grid_convergence_data.zones.items():
            if "by_diameter" in zone_name.lower() or not grid_convergence_zone_matches_requirement(zone, requirement):
                continue
            if extract_icing_bin_set_from_zone_name(zone_name) != target_bin_set:
                continue
            if roughness_filter is not None and extract_icing_roughness_key_from_zone_name(zone_name) != roughness_filter:
                continue

            x_column = find_column_case_insensitive(zone.data.columns, plot_spec["x_candidates"])
            y_column = find_column_case_insensitive(zone.data.columns, case_ordered_y_candidates(case_id, plot_spec["y_candidates"]))
            grid_column = find_column_case_insensitive(zone.data.columns, ["GRID_LEVEL", "GridLevel"])
            if x_column is None or y_column is None:
                continue

            selected_columns = [x_column, y_column]
            if grid_column is not None and grid_column not in selected_columns:
                selected_columns.append(grid_column)
            data = valid_numeric_rows(zone.data[selected_columns].copy(), x_column, y_column, positive_columns={x_column})
            data = add_grid_spacing_column(data, case_id, x_column, grid_column=grid_column)
            if not data.empty:
                roughness_key = extract_icing_roughness_key_from_zone_name(zone_name)
                participant_rows.setdefault(roughness_key, []).append(data)

        if not participant_rows:
            continue

        for roughness_key, roughness_rows in sorted(participant_rows.items()):
            data = pd.concat(roughness_rows).sort_values(GRID_SPACING_COLUMN)
            data = data.drop_duplicates(subset=["GRID_LEVEL_DISPLAY"], keep="first")
            y_column = find_column_case_insensitive(data.columns, case_ordered_y_candidates(case_id, plot_spec["y_candidates"]))
            if y_column is None:
                continue
            roughness_label = format_icing_roughness_title(roughness_key)
            # Roughness is stated in the plot heading/note, not repeated in
            # every participant legend entry.
            trace_label = label
            fig.add_trace(
                go.Scatter(
                    x=data[GRID_SPACING_COLUMN], y=data[y_column], mode="lines+markers",
                    name=trace_label,
                    legendgroup=f"{label}_{roughness_key}",
                    legendrank=participant_legend_rank(participant.participant_id),
                    line=dict(color=color), marker=participant_marker(participant.participant_id),
                    customdata=data[["GRID_LEVEL_DISPLAY", "GRID_CELL_COUNT"]],
                    hovertemplate=(f"Participant: {escape(label)}<br>Case: {escape(case_id)}<br>Distribution: {escape(display_bin_set(target_bin_set))}<br>" + (f"Roughness: {escape(roughness_label)}<br>" if roughness_label else "") +
                                   f"{escape(format_x_hover_label(GRID_SPACING_COLUMN))}=%{{x:.6g}}<br>{escape(y_column)}=%{{y}}<br>"
                                   "Grid level=%{customdata[0]}<br>Num cells=%{customdata[1]:,.0f}<extra></extra>"),
                )
            )
            trace_count += 1

    style_xy_figure(fig, case_id, plot_spec["plot_key"], plot_spec["x_label"], plot_spec["y_label"])
    style_grid_level_x_axis(fig, case_id)
    return fig, trace_count


def build_combined_icing_figure(participants, case_id: str, plot_spec: dict[str, Any], target_grid_level: str, requirement: str = "required", roughness_filter: str | None = None) -> tuple[go.Figure, int, list[str]]:
    fig = go.Figure()
    trace_count = 0
    skipped_notes: list[str] = []
    seen_trace_keys: set[tuple[str, str, str]] = set()

    for participant, case_data in iter_case_data(participants, case_id):
        if case_data.grid_convergence_data is None:
            continue

        label = participant_label(participant)
        color = participant_color(participant.participant_id)
        trace_rows_by_roughness: dict[str, list[dict[str, Any]]] = {}

        for zone_name, zone in case_data.grid_convergence_data.zones.items():
            if "by_diameter" in zone_name.lower():
                continue
            if not grid_convergence_zone_matches_requirement(zone, requirement):
                continue
            if roughness_filter is not None and extract_icing_roughness_key_from_zone_name(zone_name) != roughness_filter:
                continue

            bin_set = extract_icing_bin_set_from_zone_name(zone_name)
            if bin_set is None:
                continue

            bin_count = bin_count_from_bin_set(bin_set)
            if bin_count is None:
                continue

            x_column = find_column_case_insensitive(zone.data.columns, plot_spec["x_candidates"])
            y_column = find_column_case_insensitive(zone.data.columns, case_ordered_y_candidates(case_id, plot_spec["y_candidates"]))
            grid_column = find_column_case_insensitive(zone.data.columns, ["GRID_LEVEL", "GridLevel"])

            if x_column is None or y_column is None:
                continue

            for _, row in zone.data.iterrows():
                level_number = None
                if grid_column is not None:
                    level_number = grid_level_number_from_value(row[grid_column])
                if level_number is None:
                    level_number = grid_level_number_from_value(row[x_column])
                if level_number is None or f"L{level_number}" != target_grid_level:
                    continue

                y_value = row[y_column]
                try:
                    if float(y_value) <= -998.0:
                        continue
                except (TypeError, ValueError):
                    continue

                roughness_key = extract_icing_roughness_key_from_zone_name(zone_name)
                trace_rows_by_roughness.setdefault(roughness_key, []).append({
                    "inverse_bin_count": 1.0 / bin_count,
                    "bin_set": bin_set,
                    "bin_count": bin_count,
                    "y": y_value,
                    "zone_name": zone_name,
                })

        if not trace_rows_by_roughness:
            continue

        for roughness_key, trace_rows in sorted(trace_rows_by_roughness.items()):
            trace_key = (participant.participant_id, plot_spec["plot_key"], f"{target_grid_level}_{roughness_key}")
            if trace_key in seen_trace_keys:
                continue
            seen_trace_keys.add(trace_key)
            trace_rows = sorted(trace_rows, key=lambda item: item["inverse_bin_count"])
            customdata = [[row["bin_set"], row["bin_count"], row["zone_name"]] for row in trace_rows]
            roughness_label = format_icing_roughness_title(roughness_key)
            # Roughness is stated in the plot heading/note, not repeated in
            # every participant legend entry.
            trace_label = label
            fig.add_trace(
                go.Scatter(
                    x=[row["inverse_bin_count"] for row in trace_rows], y=[row["y"] for row in trace_rows],
                    mode="lines+markers", name=trace_label,
                    legendgroup=f"{label}_{roughness_key}", legendrank=participant_legend_rank(participant.participant_id),
                    line=dict(color=color), marker=participant_marker(participant.participant_id), customdata=customdata,
                    hovertemplate=(f"Participant: {escape(label)}<br>Case: {escape(case_id)}<br>Grid level: {escape(target_grid_level)}<br>" + (f"Roughness: {escape(roughness_label)}<br>" if roughness_label else "") +
                                   "Bin set: %{customdata[0]}<br>Number of bins: %{customdata[1]}<br>Zone: %{customdata[2]}<br>"
                                   f"1 / number of bins=%{{x:.6g}}<br>{escape(y_column)}=%{{y}}<extra></extra>"),
                )
            )
            trace_count += 1

    style_inverse_bin_figure(fig, case_id, plot_spec["plot_key"], plot_spec["y_label"])
    return fig, trace_count, skipped_notes


def build_combined_icing_subsection(participants, case_id: str, plot_spec: dict[str, Any], requirement: str = "required") -> str:
    grid_levels = collect_combined_icing_grid_levels(participants, case_id, plot_spec, requirement=requirement)
    bin_sets = collect_combined_icing_bin_sets(participants, case_id, plot_spec, requirement=requirement)
    roughness_keys = collect_combined_icing_roughness_keys(participants, case_id, plot_spec, requirement=requirement)

    if not grid_levels and not bin_sets:
        return ""

    distributions_html = ""
    figures_html = ""

    for bin_set in bin_sets:
      for roughness_key in roughness_keys:
        fig, trace_count = build_distribution_icing_figure(participants, case_id, plot_spec, bin_set, requirement=requirement, roughness_filter=roughness_key)
        if trace_count == 0:
            continue

        bin_count = bin_count_from_bin_set(bin_set)
        distribution_label = f"{bin_count}-bin distribution" if bin_count != 1 else "Single-bin distribution"
        roughness_title = format_icing_roughness_title(roughness_key)
        title = " | ".join(part for part in [plot_spec['title'], roughness_title, distribution_label, "All grid levels"] if part)
        filename = f"{slugify(case_id)}_{plot_spec['filename_slug']}_{slugify(roughness_key)}_{slugify(bin_set)}_all_grid_levels"
        figure_html = grid_convergence_figure_pair_html(fig, case_id, plot_spec["plot_key"], filename, title)
        distributions_html += f"""
        <section class="slice-plot-group">
          <h5>{escape(title)}</h5>
          <div class="plot-container">
            {figure_html}
          </div>
        </section>
        """

    for grid_level in grid_levels:
      for roughness_key in roughness_keys:
        fig, trace_count, skipped_notes = build_combined_icing_figure(participants, case_id, plot_spec, grid_level, requirement=requirement, roughness_filter=roughness_key)

        roughness_title = format_icing_roughness_title(roughness_key)
        title = " | ".join(part for part in [plot_spec['title'], roughness_title, grid_level] if part)

        if trace_count == 0:
            continue

        filename = f"{slugify(case_id)}_{plot_spec['filename_slug']}_{slugify(roughness_key)}_{slugify(grid_level)}_vs_inverse_bins"
        figure_html = distribution_figure_pair_html(
            fig, case_id, plot_spec["plot_key"], filename, title
        )

        figures_html += f"""
        <section class="slice-plot-group">
          <h5>{escape(title)}</h5>
          <div class="plot-container">
            {figure_html}
          </div>
        </section>
        """

    if not distributions_html.strip() and not figures_html.strip():
        return ""

    requirement_label = "optional" if requirement == "optional" else "required"
    distribution_group = ""
    if distributions_html.strip():
        distribution_group = f"""
        <div class="convergence-orientation-group">
          <p class="plot-description">
            One figure per submitted {requirement_label} droplet distribution, compared across every available grid level. Legend: Participant ID.
          </p>
          {distributions_html}
        </div>
        """

    level_group = ""
    if figures_html.strip():
        level_group = f"""
        <div class="convergence-orientation-group">
          <p class="plot-description">
            One figure per grid level, with {requirement_label} droplet distributions plotted against 1 / number of bins. Legend: Participant ID.
          </p>
          {figures_html}
        </div>
        """

    return f"""
    <section class="plot-subsection" data-variable-key="{escape(plot_spec['plot_key'])}" data-variable-label="{escape(plot_spec['title'])}">
      <p class="plot-description">
        Grid convergence for the {requirement_label} icing distributions in both comparison directions. Missing values equal to -999 are ignored.
      </p>
      {distribution_group}
      {level_group}
    </section>
    """

def simplify_roughness_legend(fig: go.Figure) -> str:
    """Remove roughness text from trace names and return an HTML mapping note."""
    roughness_by_participant: dict[str, set[str]] = {}
    for trace in fig.data:
        name = str(getattr(trace, "name", "") or "")
        parts = [part.strip() for part in name.split(" | ") if part.strip()]
        roughness_parts = [part for part in parts[1:] if "roughness" in part.lower()]
        if not roughness_parts:
            continue
        participant_id = parts[0]
        roughness_by_participant.setdefault(participant_id, set()).update(roughness_parts)
        trace.meta = {"ipw3_roughness_labels": roughness_parts}
        retained_parts = [parts[0], *[part for part in parts[1:] if "roughness" not in part.lower()]]
        trace.name = " | ".join(retained_parts)
    if not roughness_by_participant:
        return ""
    entries = [
        f"{escape(participant_id)}: {escape(', '.join(sorted(values)))}"
        for participant_id, values in sorted(roughness_by_participant.items())
    ]
    return '<div class="roughness-summary"><strong>Roughness height used by participant:</strong> ' + "; ".join(entries) + ".</div>"


def simplify_icing_participant_legend(fig: go.Figure) -> str:
    """Use participant-only legend labels and return the roughness mapping."""
    roughness_by_participant: dict[str, set[str]] = {}
    for trace in fig.data:
        legendgroup = str(getattr(trace, "legendgroup", "") or "")
        if legendgroup.startswith("horn_reference_"):
            continue
        name = str(getattr(trace, "name", "") or "")
        parts = [part.strip() for part in name.split(" | ") if part.strip()]
        if not parts:
            continue
        participant_id = parts[0]
        roughness_parts = [part for part in parts[1:] if "roughness" in part.lower()]
        if roughness_parts:
            roughness_by_participant.setdefault(participant_id, set()).update(roughness_parts)
            trace.meta = {"ipw3_roughness_labels": roughness_parts}
        trace.name = participant_id
    if not roughness_by_participant:
        return ""
    entries = [
        f"{escape(participant_id)}: {escape(', '.join(sorted(values)))}"
        for participant_id, values in sorted(roughness_by_participant.items())
    ]
    return '<div class="roughness-summary"><strong>Roughness height used by participant:</strong> ' + "; ".join(entries) + ".</div>"


def build_grid_convergence_roughness_subsection(participants, case_id: str, plot_spec: dict[str, Any], requirement: str = "required") -> str:
    roughness_keys = collect_cfd_roughness_keys(participants, case_id, plot_spec, requirement=requirement)

    if not roughness_keys:
        return ""

    figures_html = ""

    if "NACA0012" in case_id.upper():
        combined_fig = None
        combined_trace_count = 0
        for roughness_key in roughness_keys:
            roughness_fig, trace_count, _ = build_grid_convergence_figure(
                participants, case_id, plot_spec, roughness_filter=roughness_key, requirement=requirement
            )
            combined_trace_count += trace_count
            if combined_fig is None:
                combined_fig = go.Figure(roughness_fig)
            else:
                combined_fig.add_traces(list(roughness_fig.data))

        title = "All roughness heights"
        if combined_trace_count == 0 or combined_fig is None:
            return ""

        roughness_note = simplify_roughness_legend(combined_fig)
        filename = f"{slugify(case_id)}_{plot_spec['filename_slug']}_all_roughness"
        figure_html = grid_convergence_figure_pair_html(
            combined_fig,
            case_id,
            plot_spec["plot_key"],
            filename,
            f"{plot_spec['title']} | {title}",
        )
        return f"""
        <section class="plot-subsection" data-variable-key="{escape(plot_spec['plot_key'])}" data-variable-label="{escape(plot_spec['title'])}">
          <p class="plot-description">
            Grid-convergence data with all NACA0012 roughness heights overlaid. Missing values equal to -999 are ignored. Legend: Participant ID.
          </p>
          <section class="slice-plot-group">
            <h5>{escape(title)}</h5>
            {roughness_note}
            <div class="plot-container">{figure_html}</div>
          </section>
        </section>
        """

    for roughness_key in roughness_keys:
        fig, trace_count, skipped_notes = build_grid_convergence_figure(participants, case_id, plot_spec, roughness_filter=roughness_key, requirement=requirement)

        roughness_title = format_roughness_title(roughness_key)

        if trace_count == 0:
            continue

        # ONERA plots are already separated and titled by their common
        # roughness.  A participant mapping note is only needed for NACA0012,
        # where roughness differs between participants.
        roughness_note = simplify_roughness_legend(fig)
        if "NACA0012" not in case_id.upper():
            roughness_note = ""
        filename = f"{slugify(case_id)}_{plot_spec['filename_slug']}_{slugify(roughness_key)}"
        figure_html = grid_convergence_figure_pair_html(
            fig,
            case_id,
            plot_spec["plot_key"],
            filename,
            f"{plot_spec['title']} | {roughness_title}",
        )

        figures_html += f"""
        <section class="slice-plot-group">
          <h5>{escape(roughness_title)}</h5>
          {roughness_note}
          <div class="plot-container">
            {figure_html}
          </div>
        </section>
        """

    if not figures_html.strip():
        return ""

    return f"""
    <section class="plot-subsection" data-variable-key="{escape(plot_spec['plot_key'])}" data-variable-label="{escape(plot_spec['title'])}">
      <p class="plot-description">
        Grid-convergence data grouped by roughness condition. Missing values equal to -999 are ignored. Legend: Participant ID.
      </p>
      {figures_html}
    </section>
    """
