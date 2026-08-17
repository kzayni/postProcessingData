from __future__ import annotations

from pathlib import Path
from html import escape
from dataclasses import dataclass
from typing import Any
import math
import re

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import plotly.io as pio

from .gatherParticipantData import CASE_SLICES, decode_slice_position, iter_grid_datasets
from .heat_flux_computations import CASE_SETTINGS as HEAT_FLUX_CASE_SETTINGS, recovery_temperature
from .participant_style import participant_color, participant_legend_rank, participant_marker, participant_trace_mode

SAVE_IMAGE_PREVIEWS = False
IMAGE_PREVIEW_ROOT = Path("IMAGES_PREVIEW")
BETA_BINS = ["BINS01", "BINS03", "BINS07", "BINS15"]
ENABLE_COMBINED_BETA_BY_PARTICIPANT = False
INCLUDE_EXPERIMENTAL_DATA = True
VARIABLE_FILTER: set[str] | None = None

REFERENCE_DATA_SOURCES: list[dict[str, Any]] = [
    {
        "case_id": "TC_NACA0012_AE3932",
        "plot_key": "cp_vs_x",
        "path": Path("E00_Experimental-Data") / "EXP_NACA0012_CP.dat",
        "x_column": "X/C",
        "ordinate_column": "Y/C",
        "y_columns": ["CP_1", "CP_2"],
        "x_scale": 0.5334,
        "rotation_degrees": 4.0,
        "label": "Experimental Cp average",
    },
    {
        "case_id": "TC_NACA0012_AE3933",
        "plot_key": "cp_vs_x",
        "path": Path("E00_Experimental-Data") / "EXP_NACA0012_CP.dat",
        "x_column": "X/C",
        "ordinate_column": "Y/C",
        "y_columns": ["CP_1", "CP_2"],
        "x_scale": 0.5334,
        "rotation_degrees": 4.0,
        "label": "Experimental Cp average",
    },
    {
        "case_id": "TC_NACA0012_AE3932",
        "plot_key": "cp_vs_s",
        "path": Path("E00_Experimental-Data") / "EXP_NACA0012_CP.dat",
        "x_column": "X/C",
        "ordinate_column": "Y/C",
        "y_columns": ["CP_1", "CP_2"],
        "x_scale": 0.5334,
        "surface_distance_from_highlight": True,
        "label": "Experimental Cp average",
    },
    {
        "case_id": "TC_NACA0012_AE3933",
        "plot_key": "cp_vs_s",
        "path": Path("E00_Experimental-Data") / "EXP_NACA0012_CP.dat",
        "x_column": "X/C",
        "ordinate_column": "Y/C",
        "y_columns": ["CP_1", "CP_2"],
        "x_scale": 0.5334,
        "surface_distance_from_highlight": True,
        "label": "Experimental Cp average",
    },
]


def set_include_experimental_data(include: bool) -> None:
    global INCLUDE_EXPERIMENTAL_DATA
    INCLUDE_EXPERIMENTAL_DATA = include


def set_variable_filter(variables: set[str] | None) -> None:
    global VARIABLE_FILTER
    VARIABLE_FILTER = variables


def plot_matches_variable_filter(plot_spec: dict[str, Any]) -> bool:
    if VARIABLE_FILTER is None:
        return True
    plot_key = plot_spec["plot_key"].lower()
    aliases = {plot_key, plot_key.split("_vs_")[0]}
    if plot_key.startswith("beta_"):
        aliases.update({"beta", "collection_efficiency"})
    if plot_key.startswith("surface_temperature"):
        aliases.update({"temperature", "surface_temperature"})
    if plot_key.startswith("recovery_temperature"):
        aliases.update({"trec", "t_rec", "temperature", "recovery_temperature"})
    return bool(aliases & VARIABLE_FILTER)

CUTDATA_PLOTS: list[dict[str, Any]] = [
    {
        "plot_key": "cp_vs_x",
        "title": "Cp vs X",
        "description": "Pressure coefficient along the selected surface cut(s).",
        "x_candidates": ["X", "CoordinateX"],
        "y_candidates": ["Cp", "CP"],
        "x_label": "X [m]",
        "y_label": "Cp [-]",
        "filename_slug": "cp_vs_x",
        "bins_filter": None,
        "reverse_y_axis": True,
    },
    {
        "plot_key": "cp_vs_s",
        "title": "Cp vs s",
        "description": "Pressure coefficient along the selected surface cut(s), plotted against surface distance from the attachment line.",
        "x_candidates": ["s", "S"],
        "y_candidates": ["Cp", "CP"],
        "x_label": "Surface distance from attachment line [m]",
        "y_label": "Cp [-]",
        "filename_slug": "cp_vs_s",
        "bins_filter": None,
        "reverse_y_axis": True,
    },
    {
        "plot_key": "htc_vs_s",
        "title": "HTC vs s",
        "description": "Heat-transfer coefficient along the selected surface cut(s). The no-roughness HTC is plotted as the smooth roughness condition when available.",
        "x_candidates": ["s", "S"],
        "y_candidates": ["HTC", "HeatTransferCoefficient"],
        "clean_y_candidates": ["HTC_CLEAN", "HTC_Clean", "HTC_clean"],
        "x_label": "Surface distance from highlight [m]",
        "y_label": "Convective Heat Transfer [W/m2K]",
        "filename_slug": "htc_vs_s",
        "bins_filter": None,
        "y_range": [0, 2000],
    },
    {
        "plot_key": "beta_bins01_vs_s",
        "title": "Collection Efficiency vs s | BINS01",
        "description": "Collection efficiency for the single-bin droplet distribution.",
        "x_candidates": ["s", "S"],
        "y_candidates": ["Beta", "BETA", "CollectionEfficiency"],
        "x_label": "Surface distance from highlight [m]",
        "y_label": "Collection efficiency [-]",
        "filename_slug": "beta_bins01_vs_s",
        "bins_filter": "BINS01",
    },
    {
        "plot_key": "beta_bins03_vs_s",
        "title": "Collection Efficiency vs s | BINS03",
        "description": "Collection efficiency for the 3-bin droplet distribution.",
        "x_candidates": ["s", "S"],
        "y_candidates": ["Beta", "BETA", "CollectionEfficiency"],
        "x_label": "Surface distance from highlight [m]",
        "y_label": "Collection efficiency [-]",
        "filename_slug": "beta_bins03_vs_s",
        "bins_filter": "BINS03",
    },
    {
        "plot_key": "beta_bins07_vs_s",
        "title": "Collection Efficiency vs s | BINS07",
        "description": "Collection efficiency for the 7-bin droplet distribution.",
        "x_candidates": ["s", "S"],
        "y_candidates": ["Beta", "BETA", "CollectionEfficiency"],
        "x_label": "Surface distance from highlight [m]",
        "y_label": "Collection efficiency [-]",
        "filename_slug": "beta_bins07_vs_s",
        "bins_filter": "BINS07",
    },
    {
        "plot_key": "beta_bins15_vs_s",
        "title": "Collection Efficiency vs s | BINS15",
        "description": "Collection efficiency for the 15-bin droplet distribution.",
        "x_candidates": ["s", "S"],
        "y_candidates": ["Beta", "BETA", "CollectionEfficiency"],
        "x_label": "Surface distance from highlight [m]",
        "y_label": "Collection efficiency [-]",
        "filename_slug": "beta_bins15_vs_s",
        "bins_filter": "BINS15",
    },
    {
        "plot_key": "beta_cards_vs_s",
        "title": "Collection Efficiency vs s | BINS",
        "description": "Collection efficiency for all submitted bin distributions.",
        "x_candidates": ["s", "S"],
        "y_candidates": ["Beta", "BETA", "CollectionEfficiency"],
        "x_label": "Surface distance from highlight [m]",
        "y_label": "Collection efficiency [-]",
        "filename_slug": "beta_cards_vs_s",
        "bins_filter": None,
    },
    {
        "plot_key": "surface_temperature_vs_s",
        "title": "Surface temperature vs s",
        "description": "Surface temperature along the selected surface cut(s).",
        "x_candidates": ["s", "S"],
        "y_candidates": ["Ts", "TS", "WallTemperature", "SurfaceTemperature"],
        "x_label": "Surface distance from highlight [m]",
        "y_label": "Surface temperature [K]",
        "filename_slug": "surface_temperature_vs_s",
        "bins_filter": None,
    },
    {
        "plot_key": "recovery_temperature_vs_s",
        "title": "Recovery temperature vs s",
        "description": "Recovery temperature calculated pointwise from Cp using the case freestream temperature and Mach number (gamma = 1.4, recovery factor = 0.9).",
        "x_candidates": ["s", "S"],
        "y_candidates": ["Cp", "CP", "PressureCoefficient"],
        "x_label": "Surface distance from highlight [m]",
        "y_label": "Recovery temperature [K]",
        "filename_slug": "recovery_temperature_vs_s",
        "bins_filter": None,
        "derived_recovery_temperature": True,
    },
    {
        "plot_key": "freezing_fraction_vs_s",
        "title": "Freezing fraction vs s",
        "description": "Freezing fraction along the selected surface cut(s). Values below 1e-9 are shown as -1.0, following the adopted missing/negligible-value convention.",
        "x_candidates": ["s", "S"],
        "y_candidates": ["FF", "FreezingFraction"],
        "x_label": "Surface distance from highlight [m]",
        "y_label": "Freezing fraction [-]",
        "filename_slug": "freezing_fraction_vs_s",
        "bins_filter": None,
    },
]

BIN_LINE_DASHES = {
    "BINS01": "longdash",
    "BINS03": "solid",
    "BINS07": "dash",
    "BINS15": "dot",
}

GRID_LEVEL_LINE_DASHES = {
    "L1": "solid",
    "L2": "dash",
    "L3": "dot",
    "L4": "dashdot",
}

def slugify(text: str) -> str:
    text = text.strip().lower()
    text = re.sub(r"[^a-z0-9]+", "_", text)
    return text.strip("_") or "figure"

def extract_roughness_key_from_zone_name(zone_name: str) -> str:
    text = zone_name.strip()

    if re.search(r"(?:^|_)KS_(?:0|0p0|0\.0|smooth|none)(?:mm|m)?(?:_|$)", text, re.IGNORECASE):
        return "smooth"

    if re.search(r"(?:^|_)(?:KS_)?XX(?:mm|m)?(?:_|$)", text, re.IGNORECASE):
        return "default_roughness"

    if re.search(r"(?:^|_)(?:VARIABLE_ROUGHNESS|VAR_ROUGHNESS|KS_VARIABLE|KS_VAR)(?:_|$)", text, re.IGNORECASE):
        return "variable_roughness"

    match = re.search(r"(?:^|_)KS_(?P<value>[0-9]+(?:p[0-9]+|\.[0-9]+)?)(?P<unit>mm|m)?(?:_|$)", text, re.IGNORECASE)
    if match is not None:
        value = match.group("value").replace("p", ".")
        unit = (match.group("unit") or "mm").lower()

        if unit == "m":
            value_mm = float(value) * 1000.0
            return f"{value_mm:g}mm"

        return f"{float(value):g}mm"

    return "default_roughness"


def format_roughness_title(roughness_key: str) -> str:
    if roughness_key == "smooth":
        return "No Roughness"

    if roughness_key == "variable_roughness":
        return "Variable roughness"

    if roughness_key in {"default_roughness", "unspecified_roughness"}:
        return "Unspecified Roughness Height"

    if roughness_key.endswith("mm"):
        value = roughness_key[:-2]
        return f"Roughness height = {value} mm"

    return roughness_key


def roughness_sort_key(roughness_key: str) -> tuple[int, float, str]:
    if roughness_key == "smooth":
        return (0, 0.0, roughness_key)

    if roughness_key.endswith("mm"):
        try:
            return (1, float(roughness_key[:-2]), roughness_key)
        except ValueError:
            pass

    if roughness_key == "variable_roughness":
        return (2, 0.0, roughness_key)

    return (3, 0.0, roughness_key)


def format_roughness_list(roughness_keys: set[str]) -> str:
    if not roughness_keys:
        return "Unspecified Roughness Height"
    return ", ".join(format_roughness_title(key) for key in sorted(roughness_keys, key=roughness_sort_key))


def format_participant_roughness_summary(summary: dict[str, set[str]]) -> str:
    if not summary:
        return "Detected roughness by participant: none found."
    entries = [f"{participant_id}: {format_roughness_list(roughness_keys)}" for participant_id, roughness_keys in sorted(summary.items())]
    return "Detected roughness by participant: " + "; ".join(entries) + "."

def find_column_case_insensitive(columns, candidates: list[str]) -> str | None:
    lookup = {column.lower(): column for column in columns}
    for candidate in candidates:
        if candidate.lower() in lookup:
            return lookup[candidate.lower()]
    return None


def valid_xy_rows(dataframe, x_column: str, y_column: str):
    x_values = pd.to_numeric(dataframe[x_column], errors="coerce")
    y_values = pd.to_numeric(dataframe[y_column], errors="coerce")
    valid_mask = x_values.notna() & y_values.notna() & (x_values > -998.0) & (y_values > -998.0)
    return dataframe.loc[valid_mask]


def uses_surface_distance_axis(plot_spec: dict[str, Any]) -> bool:
    return any(candidate.lower() == "s" for candidate in plot_spec.get("x_candidates", []))


def highlight_point_description(case_id: str, slice_positions: list[float]) -> str:
    if case_id == "TC_ONERAM6":
        return "Surface distance s is measured from the highlight point X = 0 m, Y = selected slice location, Z = 0 m."

    return "Surface distance s is measured from the highlight point X = 0 m, Y = 0 m, Z = 0 m."


def participant_label(participant, dataset_data, grid_data=None) -> str:
    """Return the legend label for one participant/grid-level dataset.

    Dataset IDs now live under:
        participant.cases[case_id].grid_levels[grid_level].datasets[dataset_id]
    so DID is local to the case/grid level. Hide it when there is only one
    dataset for the participant in that grid level.
    """
    if grid_data is not None and len(grid_data.datasets) <= 1:
        return f"{participant.participant_id}"
    return f"{participant.participant_id}.{dataset_data.dataset_id}"


def plotly_config(filename: str) -> dict[str, Any]:
    return {
        "responsive": True,
        "displaylogo": False,
        "toImageButtonOptions": {"format": "png", "filename": filename, "height": 900, "width": 1200, "scale": 3},
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


def flush_png_exports(scale: int = 3) -> None:
    if not PNG_EXPORT_QUEUE:
        return
    figures, paths = zip(*PNG_EXPORT_QUEUE)
    pio.write_images(list(figures), list(paths), width=1350, height=900, scale=scale)
    PNG_EXPORT_QUEUE.clear()


def figure_to_html_div(fig: go.Figure, filename: str, plot_title: str) -> str:
    if PNG_EXPORT_DIR is not None:
        PNG_EXPORT_DIR.mkdir(parents=True, exist_ok=True)
        png_fig = go.Figure(fig)
        png_fig.update_layout(
            title=dict(text=plot_title, x=0.5, xanchor="center"),
            margin=dict(t=100),
        )
        PNG_EXPORT_QUEUE.append((png_fig, PNG_EXPORT_DIR / f"{filename}.png"))
        return ""
    figure_html = fig.to_html(full_html=False, include_plotlyjs="cdn", config=plotly_config(filename))
    if DEFER_PLOTLY_DIR is not None:
        DEFER_PLOTLY_DIR.mkdir(parents=True, exist_ok=True)
        fragment_path = DEFER_PLOTLY_DIR / f"{filename}.html"
        fragment_path.write_text(
            f'<!DOCTYPE html><html><head><meta charset="UTF-8"><meta name="viewport" content="width=device-width, initial-scale=1.0"><style>html,body{{margin:0;background:white}} .plotly-graph-div{{width:100%}}</style></head><body>{figure_html}</body></html>',
            encoding="utf-8",
        )
        return f'<iframe class="plotly-lazy-frame" data-plot-src="PLOTS/{escape(filename)}.html" title="{escape(filename)}"></iframe><div class="plot-loading">Plot queued…</div>'
    return f"""
    <div class="plot-download-shell" data-plot-filename="{escape(filename)}" data-plot-title="{escape(plot_title)}">
      {figure_html}
      <div class="plot-download-actions">
        <button type="button" data-plot-download="with-legend">Download PNG with legend</button>
        <button type="button" data-plot-download="without-legend">Download PNG without legend</button>
      </div>
    </div>
    """


def empty_placeholder(title: str, message: str) -> str:
    return ""


def parse_ipw3_zone_name(zone_name: str) -> dict[str, str] | None:
    pattern = re.compile(r"^SLICE_Y_(?P<slice>.+?)_(?P<bins>BINS\d+)(?:_(?P<tail>.*))?$", re.IGNORECASE)
    match = pattern.match(zone_name.strip())
    if match is None:
        return None

    tail = match.group("tail") or ""
    dataset_match = re.search(r"(?:^|_)(D\d+)(?:_|$)", tail, re.IGNORECASE)
    dataset_id = dataset_match.group(1).upper() if dataset_match is not None else "DXX"
    return {"slice": match.group("slice"), "bins": match.group("bins").upper(), "dataset": dataset_id}


def cutdata_zone_sort_key(zone_item: tuple[str, Any]) -> tuple[int, str]:
    """Order cut-data zones by bin count so shared fields use the first bin."""
    zone_name, _zone = zone_item
    zone_info = parse_ipw3_zone_name(zone_name)
    if zone_info is None:
        return (10**9, zone_name.lower())

    bins_match = re.search(r"\d+", zone_info["bins"])
    bin_count = int(bins_match.group(0)) if bins_match is not None else 10**9
    return (bin_count, zone_name.lower())


def format_slice_positions(slice_values: list[float]) -> str:
    if not slice_values:
        return "No slice location detected."
    unique_values = sorted(set(round(value, 8) for value in slice_values))
    return ", ".join(f"Y = {value:g} m" for value in unique_values)


def style_xy_figure(fig: go.Figure, x_label: str, y_label: str, height: int = 560, legend_right: bool = True, reverse_y_axis: bool = False, y_range: list[float] | None = None) -> go.Figure:
    if legend_right:
        legend = dict(orientation="v", x=1.02, xanchor="left", y=1.0, yanchor="top")
        margin = dict(l=90, r=220, t=30, b=80)
    else:
        legend = dict(orientation="h", x=0.0, xanchor="left", y=1.12, yanchor="bottom")
        margin = dict(l=90, r=40, t=70, b=80)

    yaxis = dict(title=dict(text=y_label, font=dict(size=18)), ticks="outside", showline=True, linecolor="black", linewidth=2, mirror=True, showgrid=True, gridcolor="lightgray", zeroline=False)
    if y_range is not None:
        yaxis["range"] = y_range
    elif reverse_y_axis:
        yaxis["autorange"] = "reversed"

    fig.update_layout(
        font=dict(family="Arial, Helvetica, sans-serif", size=16),
        autosize=True,
        height=height,
        title=None,
        showlegend=True,
        xaxis=dict(title=dict(text=x_label, font=dict(size=18)), ticks="outside", showline=True, linecolor="black", linewidth=2, mirror=True, showgrid=True, gridcolor="lightgray", zeroline=False),
        yaxis=yaxis,
        legend=legend,
        margin=margin,
        plot_bgcolor="white",
        paper_bgcolor="white",
    )
    return fig


def add_collection_efficiency_inset(
    fig: go.Figure,
    x_range: tuple[float, float] | None = None,
) -> go.Figure:
    """Overlay a leading-edge zoom on a collection-efficiency figure."""
    source_traces = list(fig.data)
    points: list[tuple[float, float]] = []
    trace_peaks: list[tuple[float, float]] = []
    for trace in source_traces:
        if trace.x is None or trace.y is None:
            continue
        trace_points: list[tuple[float, float]] = []
        for x_value, y_value in zip(trace.x, trace.y):
            try:
                x_number = float(x_value)
                y_number = float(y_value)
            except (TypeError, ValueError):
                continue
            if math.isfinite(x_number) and math.isfinite(y_number):
                points.append((x_number, y_number))
                trace_points.append((x_number, y_number))
        legend_group = str(getattr(trace, "legendgroup", "") or "")
        if trace_points and not legend_group.startswith("reference_"):
            trace_peak = max(trace_points, key=lambda point: point[1])
            trace_peaks.append(trace_peak)

    if len(points) < 2:
        return fig

    x_values = [point[0] for point in points]
    x_min, x_max = min(x_values), max(x_values)
    x_span = x_max - x_min
    if x_span <= 0:
        return fig

    # Include the peak location from every submitted trace. Centering only on
    # the single largest peak can clip participants whose peaks occur nearby
    # but at a different surface location.
    if not trace_peaks:
        return fig
    peak_x_values = [point[0] for point in trace_peaks]
    peak_x_min, peak_x_max = min(peak_x_values), max(peak_x_values)
    x_center = 0.5 * (peak_x_min + peak_x_max)
    x_half_width = max(0.006 * x_span, 0.5 * (peak_x_max - peak_x_min) + 0.002 * x_span)
    if x_range is None:
        zoom_x_min = max(x_min, x_center - x_half_width)
        zoom_x_max = min(x_max, x_center + x_half_width)
    else:
        zoom_x_min, zoom_x_max = x_range
    zoom_y_values = [
        y_value
        for x_value, y_value in points
        if zoom_x_min <= x_value <= zoom_x_max
    ]
    if not zoom_y_values:
        return fig

    # Zoom around the submitted peak values themselves, rather than the full
    # local curves, so the inset remains a true peak comparison.
    peak_y_values = [point[1] for point in trace_peaks]
    peak_band = max(0.20 * max(abs(value) for value in peak_y_values), 0.05)
    zoom_y_max = max(peak_y_values)
    zoom_y_min = min(peak_y_values) - peak_band
    # Leave enough headroom for the full crest and the explicit peak markers.
    # A merely numeric inclusion with a few thousandths of padding clips the
    # line/marker against the inset border and makes the peak look missing.
    y_padding = max(0.35 * peak_band, 0.06)
    zoom_y_range = [zoom_y_min - y_padding, zoom_y_max + y_padding]

    for trace in source_traces:
        inset_trace = go.Scatter(trace.to_plotly_json())
        inset_trace.update(xaxis="x2", yaxis="y2", showlegend=False, hoverinfo="skip")
        fig.add_trace(inset_trace)

    inset_axis_style = dict(
        ticks="outside",
        tickfont=dict(size=11),
        showline=True,
        linecolor="black",
        linewidth=2,
        mirror=True,
        showgrid=True,
        gridcolor="#dddddd",
        zeroline=False,
    )
    fig.update_layout(
        xaxis2=dict(
            **inset_axis_style,
            domain=[0.08, 0.30],
            anchor="y2",
            range=[zoom_x_min, zoom_x_max],
        ),
        yaxis2=dict(
            **inset_axis_style,
            domain=[0.55, 0.95],
            anchor="x2",
            range=zoom_y_range,
        ),
    )
    fig.add_shape(
        type="rect",
        xref="x",
        yref="y",
        x0=zoom_x_min,
        x1=zoom_x_max,
        y0=zoom_y_range[0],
        y1=zoom_y_range[1],
        line=dict(color="#444444", width=1.5, dash="dot"),
        fillcolor="rgba(0,0,0,0)",
    )
    return fig


def experimental_cp_peak_x(fig: go.Figure) -> float | None:
    """Return the plotted X coordinate of the highest experimental Cp point."""
    peak: tuple[float, float] | None = None
    for trace in fig.data:
        legend_group = str(getattr(trace, "legendgroup", "") or "")
        if not legend_group.startswith("reference_cp_vs_"):
            continue
        if trace.x is None or trace.y is None:
            continue
        for x_value, cp_value in zip(trace.x, trace.y):
            try:
                point = (float(cp_value), float(x_value))
            except (TypeError, ValueError):
                continue
            if all(math.isfinite(value) for value in point) and (peak is None or point[0] > peak[0]):
                peak = point
    return None if peak is None else peak[1]


def add_attachment_line(fig: go.Figure, x_location: float = 0.0) -> go.Figure:
    """Mark the attachment/highlight location on a Cp plot."""
    fig.add_shape(
        type="line",
        xref="x",
        yref="paper",
        x0=x_location,
        x1=x_location,
        y0=0.0,
        y1=1.0,
        line=dict(color="#333333", width=2, dash="dash"),
    )
    return fig


def add_cp_leading_edge_inset(
    fig: go.Figure,
    attachment_x: float | None = 0.0,
    position: str = "upper_right",
    y_high: float | None = None,
) -> go.Figure:
    """Overlay a tight, reversed-axis view of the maximum leading-edge Cp."""
    source_traces = list(fig.data)
    points: list[tuple[float, float]] = []
    trace_peaks: list[tuple[float, float]] = []
    experimental_peaks: list[tuple[float, float]] = []
    for trace in source_traces:
        if trace.x is None or trace.y is None:
            continue
        trace_points: list[tuple[float, float]] = []
        for x_value, y_value in zip(trace.x, trace.y):
            try:
                x_number = float(x_value)
                y_number = float(y_value)
            except (TypeError, ValueError):
                continue
            if math.isfinite(x_number) and math.isfinite(y_number):
                points.append((x_number, y_number))
                trace_points.append((x_number, y_number))
        if trace_points:
            trace_peak = max(trace_points, key=lambda point: point[1])
            legend_group = str(getattr(trace, "legendgroup", "") or "")
            if legend_group.startswith("reference_cp_vs_"):
                experimental_peaks.append(trace_peak)
            else:
                trace_peaks.append(trace_peak)

    if len(points) < 2:
        return fig

    x_values = [point[0] for point in points]
    y_values = [point[1] for point in points]
    x_min, x_max = min(x_values), max(x_values)
    x_span = x_max - x_min
    y_span = max(y_values) - min(y_values)
    if x_span <= 0 or y_span <= 0:
        return fig

    if not trace_peaks and not experimental_peaks:
        return fig
    participant_peak_x_values = [point[0] for point in trace_peaks]
    if attachment_x is not None:
        # Cp insets belong at the attachment line. For NACA0012 Cp-vs-X this
        # line is already located at the experimental Cp maximum; for Cp-vs-s
        # it is the s=0 attachment location.
        peak_x_center = attachment_x
        all_peak_x_values = participant_peak_x_values + [point[0] for point in experimental_peaks]
        peak_distance = max((abs(value - peak_x_center) for value in all_peak_x_values), default=0.0)
        x_half_width = max(0.006 * x_span, peak_distance + 0.002 * x_span)
    elif experimental_peaks:
        # The experimental maximum is the focal point. Expand symmetrically
        # around it until every participant maximum is also inside the inset.
        experimental_peak_x = max(experimental_peaks, key=lambda point: point[1])[0]
        peak_x_center = experimental_peak_x
        participant_distance = max(
            (abs(value - experimental_peak_x) for value in participant_peak_x_values),
            default=0.0,
        )
        x_half_width = max(0.006 * x_span, participant_distance + 0.002 * x_span)
    else:
        peak_x_min = min(participant_peak_x_values)
        peak_x_max = max(participant_peak_x_values)
        peak_x_center = 0.5 * (peak_x_min + peak_x_max)
        x_half_width = max(0.006 * x_span, 0.5 * (peak_x_max - peak_x_min) + 0.002 * x_span)
    # Keep all participant maxima visible without moving the inset focus away
    # from the experimental maximum when experimental Cp exists.
    zoom_x_min = peak_x_center - x_half_width
    zoom_x_max = peak_x_center + x_half_width
    local_y_values = [
        y_value
        for x_value, y_value in points
        if zoom_x_min <= x_value <= zoom_x_max
    ]
    if not local_y_values:
        return fig

    # Compare maxima across participants (and the experimental maximum when
    # present) without pulling the inset down over the rest of each curve.
    peak_y_values = [point[1] for point in trace_peaks + experimental_peaks]
    peak_band = max(0.20 * max(abs(value) for value in peak_y_values), 0.10)
    zoom_y_max = max(peak_y_values)
    zoom_y_min = min(peak_y_values) - peak_band
    y_padding = max(0.05 * peak_band, 0.01)
    zoom_y_low = zoom_y_min - y_padding
    zoom_y_high = y_high if y_high is not None else zoom_y_max + y_padding

    for trace in source_traces:
        inset_trace = go.Scatter(trace.to_plotly_json())
        inset_trace.update(xaxis="x2", yaxis="y2", showlegend=False, hoverinfo="skip")
        fig.add_trace(inset_trace)

    inset_axis_style = dict(
        ticks="outside",
        tickfont=dict(size=11),
        showline=True,
        linecolor="black",
        linewidth=2,
        mirror=True,
        showgrid=True,
        gridcolor="#dddddd",
        zeroline=False,
    )
    inset_domains = {
        "upper_right": ([0.74, 0.96], [0.55, 0.95]),
        "lower_middle": ([0.39, 0.61], [0.05, 0.45]),
        "lower_left": ([0.05, 0.27], [0.05, 0.45]),
    }
    x_domain, y_domain = inset_domains.get(position, inset_domains["upper_right"])
    fig.update_layout(
        xaxis2=dict(
            **inset_axis_style,
            domain=x_domain,
            anchor="y2",
            range=[zoom_x_min, zoom_x_max],
        ),
        yaxis2=dict(
            **inset_axis_style,
            domain=y_domain,
            anchor="x2",
            range=[zoom_y_high, zoom_y_low],
        ),
    )
    fig.add_shape(
        type="rect",
        xref="x",
        yref="y",
        x0=zoom_x_min,
        x1=zoom_x_max,
        y0=zoom_y_low,
        y1=zoom_y_high,
        line=dict(color="#444444", width=1.5, dash="dot"),
        fillcolor="rgba(0,0,0,0)",
    )
    if attachment_x is not None and zoom_x_min <= attachment_x <= zoom_x_max:
        fig.add_shape(
            type="line",
            xref="x2",
            yref="y2",
            x0=attachment_x,
            x1=attachment_x,
            y0=zoom_y_low,
            y1=zoom_y_high,
            line=dict(color="#333333", width=2, dash="dash"),
        )
    return fig


def iter_grid_data(participants, case_id: str, grid_level: str):
    """Iterate over all datasets for one case/grid level.

    This adapter keeps the rest of this builder readable while using the new
    gatherParticipantData hierarchy:
        participant -> case -> grid level -> dataset

    Yields:
        participant, case_data, grid_data, dataset_data
    """
    yield from iter_grid_datasets(participants, case_id, grid_level)


def cut_data_for_plot(dataset_data, plot_spec: dict[str, Any] | None = None):
    """Select supplemental Cp/Beta cut data when available and applicable."""
    plot_key = (plot_spec or {}).get("plot_key", "")
    supplemental_data = getattr(dataset_data, "cp_beta_cut_data", None)
    if supplemental_data is not None:
        supplemental_variables = {variable.lower() for variable in supplemental_data.variables}
        if plot_key.startswith("cp_vs_") and "cp" in supplemental_variables:
            return supplemental_data
        if plot_key.startswith("beta_") and "beta" in supplemental_variables:
            return supplemental_data
    return dataset_data.cut_data


def get_cutdata_zones_by_bins(dataset_data) -> dict[str, list[tuple[str, Any, float | None]]]:
    grouped: dict[str, list[tuple[str, Any, float | None]]] = {bins_id: [] for bins_id in BETA_BINS}
    cut_data = cut_data_for_plot(dataset_data, {"plot_key": "beta_cards_vs_s"})
    if cut_data is None:
        return grouped
    for zone_name, zone in cut_data.zones.items():
        zone_info = parse_ipw3_zone_name(zone_name)
        if zone_info is None:
            continue
        bins_id = zone_info["bins"]
        if bins_id not in grouped:
            continue
        slice_position = decode_slice_position(zone_info["slice"])
        grouped[bins_id].append((zone_name, zone, slice_position))
    return grouped


@dataclass
class ReferenceTrace:
    label: str
    x: list[float]
    y: list[float]
    mode: str = "lines"


def get_reference_sources(case_id: str, grid_level: str, plot_key: str) -> list[dict[str, Any]]:
    output = []
    for source in REFERENCE_DATA_SOURCES:
        if source.get("case_id") != case_id:
            continue
        if source.get("plot_key") != plot_key:
            continue
        source_grid = source.get("grid_level")
        if source_grid is not None and source_grid != grid_level:
            continue
        output.append(source)
    return output


def add_reference_traces(fig: go.Figure, case_id: str, grid_level: str, plot_key: str) -> int:
    if not INCLUDE_EXPERIMENTAL_DATA:
        return 0

    trace_count = 0
    sources = get_reference_sources(case_id, grid_level, plot_key)
    for source in sources:
        source_path = Path(source["path"])
        if not source_path.is_file():
            continue

        columns: list[str] = []
        rows: list[list[float]] = []
        for raw_line in source_path.read_text(encoding="utf-8").splitlines():
            line = raw_line.strip()
            if not line:
                continue
            if line.upper().startswith("VARIABLES"):
                columns = re.findall(r'"([^"]+)"', line)
                continue
            if line.upper().startswith("TITLE") or not columns:
                continue
            values = line.replace(",", " ").split()
            if len(values) < len(columns):
                continue
            try:
                rows.append([float(value) for value in values[:len(columns)]])
            except ValueError:
                continue

        x_column = source["x_column"]
        ordinate_column = source.get("ordinate_column")
        y_columns = list(source["y_columns"])
        if (
            not rows
            or x_column not in columns
            or (ordinate_column is not None and ordinate_column not in columns)
            or any(column not in columns for column in y_columns)
        ):
            continue

        x_index = columns.index(x_column)
        ordinate_index = columns.index(ordinate_column) if ordinate_column is not None else None
        y_indices = [columns.index(column) for column in y_columns]
        x_scale = float(source.get("x_scale", 1.0))
        rotation_angle = math.radians(float(source.get("rotation_degrees", 0.0)))
        cosine = math.cos(rotation_angle)
        sine = math.sin(rotation_angle)
        y_values = [sum(row[index] for index in y_indices) / len(y_indices) for row in rows]
        if source.get("surface_distance_from_highlight") and ordinate_index is not None:
            # Traverse the lower surface from the trailing edge to the leading
            # edge, then the upper surface back to the trailing edge. This
            # matches the signed-s convention used by the submitted cut data.
            lower = sorted(
                (index for index, row in enumerate(rows) if row[ordinate_index] <= 0.0),
                key=lambda index: rows[index][x_index],
                reverse=True,
            )
            upper = sorted(
                (index for index, row in enumerate(rows) if row[ordinate_index] > 0.0),
                key=lambda index: rows[index][x_index],
            )
            ordered_indices = lower + upper
            cumulative_by_index: dict[int, float] = {}
            cumulative_distance = 0.0
            previous_index: int | None = None
            for index in ordered_indices:
                if previous_index is not None:
                    dx = rows[index][x_index] - rows[previous_index][x_index]
                    dy = rows[index][ordinate_index] - rows[previous_index][ordinate_index]
                    cumulative_distance += x_scale * math.hypot(dx, dy)
                cumulative_by_index[index] = cumulative_distance
                previous_index = index
            # Participants use the projection of the configured NACA highlight
            # point (X, Z) = (0, 0) as s=0. Project that same point onto the
            # experimental polyline instead of anchoring s at the maximum-Cp
            # pressure tap, which is slightly downstream of the leading edge.
            highlight_distance = 0.0
            best_distance_squared = float("inf")
            for start_index, end_index in zip(ordered_indices, ordered_indices[1:]):
                start_x = x_scale * rows[start_index][x_index]
                start_z = x_scale * rows[start_index][ordinate_index]
                end_x = x_scale * rows[end_index][x_index]
                end_z = x_scale * rows[end_index][ordinate_index]
                dx = end_x - start_x
                dz = end_z - start_z
                segment_length_squared = dx * dx + dz * dz
                fraction = 0.0
                if segment_length_squared > 0.0:
                    fraction = max(
                        0.0,
                        min(1.0, -(start_x * dx + start_z * dz) / segment_length_squared),
                    )
                projected_x = start_x + fraction * dx
                projected_z = start_z + fraction * dz
                distance_squared = projected_x * projected_x + projected_z * projected_z
                if distance_squared < best_distance_squared:
                    best_distance_squared = distance_squared
                    highlight_distance = cumulative_by_index[start_index] + fraction * math.sqrt(segment_length_squared)
            # Match the submitted curves explicitly by the experimental Z
            # ordinate: Z < 0 is the lower surface and must have negative s;
            # Z > 0 is the upper surface and must have positive s.
            x_values = []
            for index, row in enumerate(rows):
                offset = cumulative_by_index[index] - highlight_distance
                ordinate = row[ordinate_index]
                if ordinate < 0.0:
                    x_values.append(-abs(offset))
                elif ordinate > 0.0:
                    x_values.append(abs(offset))
                else:
                    x_values.append(offset)
        else:
            x_values = [
                x_scale * (
                    row[x_index] * cosine
                    - (row[ordinate_index] if ordinate_index is not None else 0.0) * sine
                )
                for row in rows
            ]
        sorted_points = sorted(zip(x_values, y_values), key=lambda point: point[0])
        x_values = [point[0] for point in sorted_points]
        y_values = [point[1] for point in sorted_points]
        label = str(source.get("label", "Experimental"))

        fig.add_trace(
            go.Scatter(
                x=x_values,
                y=y_values,
                mode="markers",
                name=label,
                legendgroup=f"reference_{plot_key}",
                legendrank=10000,
                marker=dict(
                    color="#ff56be",
                    size=8,
                    symbol="square",
                    line=dict(color="black", width=1.5),
                ),
                hovertemplate=(
                    f"Source: {escape(source_path.name)}<br>"
                    f"Series: average of {escape(', '.join(y_columns))}<br>"
                    f"{'s' if source.get('surface_distance_from_highlight') else 'Rotated X'}=%{{x:.6g}} m<br>"
                    "Cp=%{y:.6g}<extra></extra>"
                ),
            )
        )
        trace_count += 1
    return trace_count


def slice_matches_filter(slice_position: float | None, slice_filter: float | None, tolerance: float = 1.0e-6) -> bool:
    if slice_filter is None:
        return True
    if slice_position is None:
        return False
    return abs(slice_position - slice_filter) <= tolerance


def collect_cutdata_slice_positions(participants, case_id: str, grid_level: str, bins_filter: str | None = None) -> list[float]:
    expected_slices = CASE_SLICES.get(case_id)
    if expected_slices:
        return sorted(set(round(value, 8) for value in expected_slices))

    slice_positions: list[float] = []

    for participant, case_data, grid_data, dataset_data in iter_grid_data(participants, case_id, grid_level):
        if dataset_data.cut_data is None:
            continue
        for zone_name in dataset_data.cut_data.zones:
            zone_info = parse_ipw3_zone_name(zone_name)
            if zone_info is None:
                continue
            if bins_filter is not None and zone_info["bins"] != bins_filter:
                continue
            slice_position = decode_slice_position(zone_info["slice"])
            if slice_position is not None:
                slice_positions.append(slice_position)

    return sorted(set(round(value, 8) for value in slice_positions))


def build_cutdata_figure(
    participants,
    case_id: str,
    grid_level: str,
    plot_spec: dict[str, Any],
    slice_filter: float | None = None,
    roughness_filter: str | None = None,
    show_cp_inset: bool = True,
) -> tuple[go.Figure, int, list[float], list[str]]:
    seen_trace_keys: set[tuple[Any, ...]] = set()

    fig = go.Figure()
    trace_count = 0
    slice_positions: list[float] = []
    skipped_notes: list[str] = []
    skipped_note_set: set[str] = set()
    bins_filter = plot_spec.get("bins_filter")
    is_beta_plot = any(
        candidate.lower() in {"beta", "collectionefficiency"}
        for candidate in plot_spec.get("y_candidates", [])
    )

    for participant, case_data, grid_data, dataset_data in iter_grid_data(participants, case_id, grid_level):
        cut_data = cut_data_for_plot(dataset_data, plot_spec)
        if cut_data is None:
            continue
        # For fields shared by all bin solutions (everything except Beta),
        # numeric bin ordering plus the bin-independent trace key below keeps
        # the first available solution for each slice and ks condition.
        zone_items = sorted(cut_data.zones.items(), key=cutdata_zone_sort_key)
        for zone_name, zone in zone_items:
            zone_info = parse_ipw3_zone_name(zone_name)
            bins_id = zone_info["bins"] if zone_info is not None else None
            if bins_filter is not None and bins_id != bins_filter:
                continue
            slice_position = None
            if zone_info is not None:
                slice_position = decode_slice_position(zone_info["slice"])

            if not slice_matches_filter(slice_position, slice_filter):
                continue

            roughness_key = extract_roughness_key_from_zone_name(zone_name)

            use_clean_htc = (
                plot_spec.get("plot_key") == "htc_vs_s"
                and roughness_filter == "smooth"
            )

            if use_clean_htc:
                roughness_key = "smooth"
            else:
                if roughness_filter is not None and roughness_key != roughness_filter:
                    continue
            if slice_position is not None:
                slice_positions.append(slice_position)
            x_column = find_column_case_insensitive(zone.data.columns, plot_spec["x_candidates"])
            if use_clean_htc:
                y_column = find_column_case_insensitive(zone.data.columns, plot_spec.get("clean_y_candidates", []))
            else:
                y_column = find_column_case_insensitive(zone.data.columns, plot_spec["y_candidates"])
            if x_column is None or y_column is None:
                skipped_note_set.add(f"Participant ID {participant.participant_id} did not provide {plot_spec['y_candidates'][0]}.")
                continue
            data = valid_xy_rows(zone.data[[x_column, y_column]].copy(), x_column, y_column)
            if data.empty:
                skipped_note_set.add(f"Participant ID {participant.participant_id} did not provide valid {plot_spec['y_candidates'][0]} values.")
                continue
            invert_participant_019_s = (
                case_id == "TC_ONERAM6"
                and str(participant.participant_id).zfill(3) == "019"
                and grid_level in {"L2", "L3", "L4"}
                and (plot_spec["plot_key"] == "htc_vs_s" or is_beta_plot)
                and x_column.lower() == "s"
            )
            if invert_participant_019_s:
                data[x_column] = -pd.to_numeric(data[x_column], errors="coerce")
            if x_column.lower() == "s":
                data = data.sort_values(x_column, kind="mergesort").reset_index(drop=True)
            if plot_spec.get("derived_recovery_temperature"):
                settings = HEAT_FLUX_CASE_SETTINGS.get(case_id)
                if settings is None:
                    skipped_note_set.add(f"No recovery-temperature settings are configured for {case_id}.")
                    continue
                cp_values = pd.to_numeric(data[y_column], errors="coerce").to_numpy(dtype=float)
                t_rec_values = recovery_temperature(cp_values, settings.t_inf, settings.mach_inf)
                valid_recovery = np.isfinite(t_rec_values)
                data = data.loc[valid_recovery].copy()
                if data.empty:
                    skipped_note_set.add(f"Participant ID {participant.participant_id} did not provide Cp values from which Trec could be calculated.")
                    continue
                y_column = "Trec"
                data[y_column] = t_rec_values[valid_recovery]
            if plot_spec["plot_key"] == "freezing_fraction_vs_s":
                data.loc[data[y_column] < 1e-9, y_column] = -1.0
            trace_name = participant_label(participant, dataset_data, grid_data)
            if "NACA0012" in case_id.upper():
                trace_name = f"{trace_name} | {format_roughness_title(roughness_key)}"
            slice_text = "unknown"
            if zone_info is not None:
                slice_value = decode_slice_position(zone_info["slice"])
                if slice_value is not None:
                    slice_text = f"Y = {slice_value:g} m"
            
            ##Adding traces color
            if is_beta_plot:
                zone_identity = zone_name
                trace_bins_id = bins_id or ""
            else:
                zone_identity = "HTC_CLEAN" if use_clean_htc else "SHARED_BIN_SOLUTION"
                trace_bins_id = ""

            trace_key = (
                participant.participant_id,
                dataset_data.dataset_id,
                case_id,
                grid_level,
                round(slice_position, 8) if slice_position is not None else None,
                plot_spec["plot_key"],
                trace_bins_id,
                roughness_key,
                zone_identity,
            )

            if trace_key in seen_trace_keys:
                continue

            seen_trace_keys.add(trace_key)
            color = participant_color(participant.participant_id)
            fig.add_trace(
                go.Scatter(
                    x=data[x_column],
                    y=data[y_column],
                    mode=participant_trace_mode(participant.participant_id),
                    name=trace_name,
                    legendgroup=trace_name,
                    legendrank=participant_legend_rank(participant.participant_id),
                    line=dict(color=color),
                    marker=participant_marker(participant.participant_id, len(data)),
                    hovertemplate=(
                        f"Participant: {escape(trace_name)}<br>"
                        f"Roughness: {escape(format_roughness_title(roughness_key))}<br>"
                        f"Case: {escape(case_id)}<br>"
                        f"Grid: {escape(grid_level)}<br>"
                        f"Bins: {escape(bins_id or 'not specified')}<br>"
                        f"Slice: {escape(slice_text)}<br>"
                        f"Zone: {escape(zone_name)}<br>"
                        f"{escape(x_column)}=%{{x}}<br>"
                        f"{escape(y_column)}=%{{y}}<extra></extra>"
                    ),
                )
            )
            trace_count += 1

    trace_count += add_reference_traces(fig, case_id, grid_level, plot_spec["plot_key"])
    style_xy_figure(
        fig,
        plot_spec["x_label"],
        plot_spec["y_label"],
        reverse_y_axis=plot_spec.get("reverse_y_axis", False),
        y_range=plot_spec.get("y_range"),
    )
    if case_id == "TC_ONERAM6" and plot_spec["plot_key"] == "htc_vs_s":
        fig.update_xaxes(range=[-0.4, 0.4])
    if case_id == "TC_ONERAM6" and is_beta_plot:
        fig.update_xaxes(range=[-0.15, 0.15])
    if case_id.startswith("TC_NACA0012_") and (
        plot_spec["plot_key"] == "htc_vs_s" or is_beta_plot
    ):
        fig.update_xaxes(range=[-0.4, 0.4])
    if is_beta_plot and show_cp_inset:
        add_collection_efficiency_inset(
            fig,
            x_range=(
                (-0.025, 0.025)
                if case_id == "TC_ONERAM6"
                else (-0.025, 0.025) if case_id.startswith("TC_NACA0012_") else None
            ),
        )
    if plot_spec["plot_key"] in {"cp_vs_x", "cp_vs_s"}:
        attachment_x = 0.0
        if case_id.startswith("TC_NACA0012_"):
            attachment_x = experimental_cp_peak_x(fig) or 0.0
        if case_id != "TC_ONERAM6":
            add_attachment_line(fig, attachment_x)
        inset_position = "upper_right"
        if case_id == "TC_ONERAM6":
            inset_position = "lower_middle" if plot_spec["plot_key"] == "cp_vs_x" else "lower_left"
        if show_cp_inset:
            add_cp_leading_edge_inset(
                fig,
                attachment_x=attachment_x if case_id != "TC_ONERAM6" else None,
                position=inset_position,
                y_high=(
                    1.17
                    if case_id == "TC_NACA0012_AE3933"
                    else 1.08 if case_id.startswith("TC_NACA0012_") else None
                ),
            )
    skipped_notes = sorted(skipped_note_set)
    return fig, trace_count, slice_positions, skipped_notes


def build_plot_description(plot_spec: dict[str, Any], slice_positions: list[float], case_id: str, roughness_summary: dict[str, set[str]] | None = None) -> str:
    slices_text = format_slice_positions(slice_positions)
    bins_filter = plot_spec.get("bins_filter")
    details = [plot_spec["description"]]
    if bins_filter is not None:
        details.append(f"Distribution: {bins_filter}.")
    if uses_surface_distance_axis(plot_spec):
        details.append(highlight_point_description(case_id, slice_positions))
    details.append(f"Slice location(s): {slices_text}.")
    if roughness_summary is not None:
        details.append(format_participant_roughness_summary(roughness_summary))
    if INCLUDE_EXPERIMENTAL_DATA and plot_spec.get("plot_key") == "cp_vs_x" and case_id.startswith("TC_NACA0012_"):
        details.append("Experimental Cp markers are the pointwise average of CP_1 and CP_2 from E00_Experimental-Data/EXP_NACA0012_CP.dat; X/C and Y/C are rotated +4° about the leading edge at (0, 0) and scaled by the 0.5334 m chord.")
    if INCLUDE_EXPERIMENTAL_DATA and plot_spec.get("plot_key") == "cp_vs_s" and case_id.startswith("TC_NACA0012_"):
        details.append("Experimental Cp markers use the same signed curvilinear-distance convention as participant data: s = 0 is the projection of (X, Z) = (0, 0) onto the surface, with negative s on the lower surface and positive s on the upper surface.")
    if case_id == "TC_ONERAM6" and (
        plot_spec.get("plot_key") == "htc_vs_s"
        or plot_spec.get("plot_key", "").startswith("beta_")
    ):
        details.append("For participant 019, the submitted L2-L4 HTC and Beta surface orientation is corrected by plotting against -s; Cp retains the original s orientation.")
    details.append("Legend: Participant ID.")
    return " ".join(details)


def build_participant_combined_beta_figure(participant, dataset_data, case_id: str, grid_level: str, grid_data, slice_filter: float | None = None) -> tuple[str, int, list[float]]:
    fig = go.Figure()
    trace_count = 0
    slice_positions: list[float] = []
    label = participant_label(participant, dataset_data, grid_data)
    color = participant_color(participant.participant_id)
    grouped_zones = get_cutdata_zones_by_bins(dataset_data)

    for bins_id in BETA_BINS:
        for zone_name, zone, slice_position in grouped_zones[bins_id]:
            if not slice_matches_filter(slice_position, slice_filter):
                continue
            x_column = find_column_case_insensitive(zone.data.columns, ["s", "S"])
            y_column = find_column_case_insensitive(zone.data.columns, ["Beta", "BETA", "CollectionEfficiency"])
            if x_column is None or y_column is None:
                continue
            data = valid_xy_rows(zone.data[[x_column, y_column]].copy(), x_column, y_column)
            if data.empty:
                continue
            data = data.sort_values(x_column, kind="mergesort").reset_index(drop=True)
            if slice_position is not None:
                slice_positions.append(slice_position)
            slice_text = f"Y = {slice_position:g} m" if slice_position is not None else "unknown"
            fig.add_trace(
                go.Scatter(
                    x=data[x_column],
                    y=data[y_column],
                    mode=participant_trace_mode(participant.participant_id),
                    name=bins_id,
                    legendgroup=bins_id,
                    line=dict(color=color, dash=BIN_LINE_DASHES.get(bins_id, "solid")),
                    marker=participant_marker(participant.participant_id, len(data)),
                    hovertemplate=(
                        f"Participant: {escape(label)}<br>"
                        f"Case: {escape(case_id)}<br>"
                        f"Grid: {escape(grid_level)}<br>"
                        f"Bins: {escape(bins_id)}<br>"
                        f"Slice: {escape(slice_text)}<br>"
                        f"Zone: {escape(zone_name)}<br>"
                        f"{escape(x_column)}=%{{x}}<br>"
                        f"{escape(y_column)}=%{{y}}<extra></extra>"
                    ),
                )
            )
            trace_count += 1

    if trace_count == 0:
        return "", 0, slice_positions
    style_xy_figure(fig, "s [m]", "Beta [-]", height=420, legend_right=False)
    fig.update_layout(margin=dict(l=70, r=25, t=20, b=60), legend=dict(orientation="h", x=0.0, y=1.12))
    slice_slug = f"_slice_{slice_filter:g}".replace(".", "p") if slice_filter is not None else ""
    filename = f"{slugify(case_id)}_{grid_level}_{participant.participant_id}_{dataset_data.dataset_id}{slice_slug}_combined_beta"
    plot_title = f"Collection Efficiency vs s | {grid_level} | {label}"
    if slice_filter is not None:
        plot_title += f" | Y = {slice_filter:g} m"
    return figure_to_html_div(fig, filename=filename, plot_title=plot_title), trace_count, slice_positions


def build_combined_beta_card(participant, dataset_data, case_id: str, grid_level: str, grid_data, slice_filter: float | None = None) -> str:
    participant_id = participant_label(participant, dataset_data, grid_data)
    figure_html, trace_count, slice_positions = build_participant_combined_beta_figure(participant, dataset_data, case_id, grid_level, grid_data, slice_filter=slice_filter)
    if trace_count == 0:
        return ""
    slice_title = f" | Y = {slice_filter:g} m" if slice_filter is not None else ""
    details = f"Legend: {participant_id}. Case: {case_id}. Grid level: {grid_level}. {highlight_point_description(case_id, slice_positions)} Slice location(s): {format_slice_positions(slice_positions)}. Curves included when available: BINS01, BINS03, BINS07, and BINS15."
    return f"""
    <article class="combined-beta-card">
      <h4>{escape(participant_id + slice_title)}</h4>
      <p class="plot-description">{escape(details)}</p>
      <div class="plot-container combined-beta-figure">
        {figure_html}
      </div>
    </article>
    """


def build_combined_beta_section(participants, case_id: str, grid_level: str) -> str:
    slice_positions = collect_cutdata_slice_positions(participants, case_id, grid_level)
    if not slice_positions:
        slice_positions = [None]
    cards_html = ""
    for slice_position in slice_positions:
        slice_cards_html = ""
        for participant, case_data, grid_data, dataset_data in iter_grid_data(participants, case_id, grid_level):
            if dataset_data.cut_data is None:
                continue
            slice_cards_html += build_combined_beta_card(participant, dataset_data, case_id, grid_level, grid_data, slice_filter=slice_position)
        if slice_cards_html:
            slice_title = f"Y = {slice_position:g} m" if slice_position is not None else "Slice unknown"
            cards_html += f"""
            <section class="slice-plot-group">
              <h5>{escape(slice_title)}</h5>
              {slice_cards_html}
            </section>
            """
    if not cards_html:
        return ""
    return f"""
    <section class="plot-subsection combined-beta-section" data-variable-key="combined_beta" data-variable-label="Combined Beta">
      <h4>Combined Beta by participant</h4>
      <p class="plot-description">
        Each card corresponds to one participant dataset. Inside each card, BINS01, BINS03, BINS07, and BINS15 are overlaid on the same Beta vs s figure. Participant cards are arranged three per row when space allows. Legends use only PID.DID.
      </p>
      <div class="combined-beta-gallery">
        {cards_html}
      </div>
    </section>
    """


def build_participant_combined_levels_cutdata_figure(
    participant,
    case_id: str,
    plot_spec: dict[str, Any],
    slice_filter: float | None,
) -> tuple[go.Figure, int]:
    """Overlay one participant's L1-L4 results for one CutData variable."""
    combined_fig: go.Figure | None = None
    participant_trace_count = 0
    reference_added = False

    for grid_level in ("L1", "L2", "L3", "L4"):
        level_fig, _, _, _ = build_cutdata_figure(
            [participant],
            case_id,
            grid_level,
            plot_spec,
            slice_filter=slice_filter,
            show_cp_inset=False,
        )
        if combined_fig is None:
            combined_fig = go.Figure(level_fig)
            combined_fig.data = ()

        for trace in level_fig.data:
            is_reference = str(trace.legendgroup).startswith("reference_")
            if is_reference:
                if reference_added:
                    continue
                reference_added = True
            else:
                participant_trace_count += 1
                original_name = str(trace.name)
                detail = original_name.split(" | ", 1)[1] if " | " in original_name else ""
                trace.name = grid_level + (f" | {detail}" if detail else "")
                trace.legendgroup = f"{grid_level}_{original_name}"
                trace.line.dash = GRID_LEVEL_LINE_DASHES[grid_level]
            combined_fig.add_trace(trace)

    if combined_fig is None or participant_trace_count == 0:
        return go.Figure(), 0

    style_xy_figure(
        combined_fig,
        plot_spec["x_label"],
        plot_spec["y_label"],
        reverse_y_axis=plot_spec.get("reverse_y_axis", False),
        height=390,
        legend_right=False,
    )
    combined_fig.update_layout(
        margin=dict(l=65, r=20, t=20, b=60),
        showlegend=False,
    )
    return combined_fig, participant_trace_count


def build_combined_levels_cutdata_section(participants, case_id: str) -> str:
    """Build three-column participant cards with L1-L4 CutData overlaid."""
    sections_html = ""
    plot_specs = [
        plot for plot in CUTDATA_PLOTS
        if plot["plot_key"] != "beta_cards_vs_s" and plot_matches_variable_filter(plot)
    ]

    for plot_spec in plot_specs:
        slice_positions = sorted(set(CASE_SLICES.get(case_id, []))) or [None]
        slice_groups_html = ""
        for slice_position in slice_positions:
            cards_html = ""
            for participant in participants:
                fig, trace_count = build_participant_combined_levels_cutdata_figure(
                    participant, case_id, plot_spec, slice_position
                )
                if trace_count == 0:
                    continue
                participant_name = participant.participant_id
                participant_case = participant.cases.get(case_id)
                if participant_case is not None:
                    dataset_ids = sorted({
                        dataset_id
                        for grid_data in participant_case.grid_levels.values()
                        for dataset_id in grid_data.datasets
                    })
                    if len(dataset_ids) == 1:
                        participant_name = f"{participant_name}.{dataset_ids[0]}"
                slice_slug = f"slice_{slice_position:g}".replace(".", "p") if slice_position is not None else "slice_unknown"
                filename = f"{slugify(case_id)}_{participant.participant_id}_{plot_spec['filename_slug']}_{slice_slug}_L1_L4"
                figure_html = figure_to_html_div(
                    fig,
                    filename=filename,
                    plot_title=f"{plot_spec['title']} | L1-L4 | Participant {participant_name}",
                )
                cards_html += f"""
                <article class="combined-grid-card">
                  <h5>Participant {escape(participant_name)}</h5>
                  <div class="plot-container combined-grid-figure">{figure_html}</div>
                </article>
                """
            if cards_html:
                slice_title = f"Y = {slice_position:g} m" if slice_position is not None else "Available slice"
                slice_groups_html += f"""
                <section class="combined-grid-slice-group">
                  <h4>{escape(slice_title)}</h4>
                  <div class="combined-grid-matrix">{cards_html}</div>
                </section>
                """
        if slice_groups_html:
            sections_html += f"""
            <section class="plot-subsection" data-variable-key="combined_{escape(plot_spec['plot_key'])}" data-variable-label="Combined {escape(plot_spec['title'])}">
              <h3>{escape(plot_spec['title'])} — L1–L4 by participant</h3>
              <p class="plot-description">Each card contains one participant with its available L1, L2, L3, and L4 curves overlaid. Grid levels are distinguished by line style; cards are arranged three per row.</p>
              {slice_groups_html}
            </section>
            """

    return f'<section class="plot-filter-scope combined-grid-filter-scope"><div class="variable-filter-controls" data-filter-title="Levels Combined CutData variables"></div>{sections_html}</section>' if sections_html else ""


def build_plot_subsection(
    participants,
    case_id: str,
    grid_level: str,
    plot_spec: dict[str, Any],
    roughness_filter_predicate=None,
) -> str:
    slice_positions = collect_cutdata_slice_positions(participants, case_id, grid_level, bins_filter=plot_spec.get("bins_filter"))

    if not slice_positions:
        slice_positions = [None]

    figures_html = ""
    no_roughness_figures_html = ""
    roughness_figures_html = ""
    group_onera_htc_by_roughness = case_id == "TC_ONERAM6" and plot_spec.get("plot_key") == "htc_vs_s"
    all_skipped_notes: list[str] = []

    for slice_position in slice_positions:
        roughness_keys = collect_cutdata_roughness_keys(participants, case_id, grid_level, plot_spec, slice_filter=slice_position)
        if roughness_filter_predicate is not None:
            roughness_keys = [key for key in roughness_keys if roughness_filter_predicate(key)]
            if not roughness_keys:
                continue

        if not roughness_keys:
            roughness_keys = [None]

        if "NACA0012" in case_id.upper():
            if plot_spec.get("plot_key") == "htc_vs_s" and "smooth" in roughness_keys:
                smooth_fig, smooth_trace_count, _, smooth_skipped_notes = build_cutdata_figure(
                    participants,
                    case_id,
                    grid_level,
                    plot_spec,
                    slice_filter=slice_position,
                    roughness_filter="smooth",
                )
                all_skipped_notes.extend(smooth_skipped_notes)
                if smooth_trace_count > 0:
                    slice_title = f"Y = {slice_position:g} m" if slice_position is not None else "Slice unknown"
                    smooth_title = f"{slice_title} | No roughness"
                    slice_slug = f"_slice_{slice_position:g}".replace(".", "p") if slice_position is not None else "_slice_unknown"
                    smooth_filename = f"{slugify(case_id)}_{grid_level}_{plot_spec['filename_slug']}{slice_slug}_no_roughness"
                    smooth_figure_html = figure_to_html_div(
                        smooth_fig,
                        filename=smooth_filename,
                        plot_title=f"{plot_spec['title']} | {grid_level} | {smooth_title}",
                    )
                    figures_html += f"""
                    <section class="slice-plot-group">
                      <h5>{escape(smooth_title)}</h5>
                      <div class="plot-container">
                        {smooth_figure_html}
                      </div>
                    </section>
                    """
                roughness_keys = [key for key in roughness_keys if key != "smooth"]
                if not roughness_keys:
                    continue

            combined_fig = None
            combined_trace_count = 0
            combined_skipped_notes: list[str] = []
            combined_reference_groups: set[str] = set()
            for roughness_key in roughness_keys:
                roughness_fig, roughness_trace_count, _, skipped_notes = build_cutdata_figure(
                    participants,
                    case_id,
                    grid_level,
                    plot_spec,
                    slice_filter=slice_position,
                    roughness_filter=roughness_key,
                )
                combined_skipped_notes.extend(skipped_notes)
                if combined_fig is None:
                    combined_fig = go.Figure(roughness_fig)
                    combined_reference_groups.update(
                        str(trace.legendgroup)
                        for trace in combined_fig.data
                        if str(trace.legendgroup).startswith("reference_")
                    )
                else:
                    for trace in roughness_fig.data:
                        legend_group = str(trace.legendgroup)
                        if legend_group.startswith("reference_"):
                            if legend_group in combined_reference_groups:
                                continue
                            combined_reference_groups.add(legend_group)
                        combined_fig.add_trace(trace)
                combined_trace_count = len(combined_fig.data)

            all_skipped_notes.extend(combined_skipped_notes)
            slice_title = f"Y = {slice_position:g} m" if slice_position is not None else "Slice unknown"
            roughness_group_title = "Roughness cases" if plot_spec.get("plot_key") == "htc_vs_s" else "All roughness heights"
            full_title = f"{slice_title} | {roughness_group_title}"
            if combined_trace_count == 0 or combined_fig is None:
                continue
            slice_slug = f"_slice_{slice_position:g}".replace(".", "p") if slice_position is not None else "_slice_unknown"
            filename = f"{slugify(case_id)}_{grid_level}_{plot_spec['filename_slug']}{slice_slug}_all_roughness"
            figure_html = figure_to_html_div(
                combined_fig,
                filename=filename,
                plot_title=f"{plot_spec['title']} | {grid_level} | {full_title}",
            )
            figures_html += f"""
            <section class="slice-plot-group">
              <h5>{escape(full_title)}</h5>
              <div class="plot-container">
                {figure_html}
              </div>
            </section>
            """
            continue

        for roughness_key in roughness_keys:
            fig, trace_count, figure_slice_positions, skipped_notes = build_cutdata_figure(
                participants,
                case_id,
                grid_level,
                plot_spec,
                slice_filter=slice_position,
                roughness_filter=roughness_key,
            )

            all_skipped_notes.extend(skipped_notes)

            slice_title = f"Y = {slice_position:g} m" if slice_position is not None else "Slice unknown"
            roughness_title = format_roughness_title(roughness_key) if roughness_key is not None else "Unspecified Roughness Height"
            full_title = f"{slice_title} | {roughness_title}"

            if trace_count == 0:
                continue
            slice_slug = f"_slice_{slice_position:g}".replace(".", "p") if slice_position is not None else "_slice_unknown"
            roughness_slug = f"_roughness_{slugify(roughness_key or 'unspecified')}"
            filename = f"{slugify(case_id)}_{grid_level}_{plot_spec['filename_slug']}{slice_slug}{roughness_slug}"
            figure_html = figure_to_html_div(
                fig,
                filename=filename,
                plot_title=f"{plot_spec['title']} | {grid_level} | {full_title}",
            )

            figure_card_html = f"""
            <section class="slice-plot-group">
              <h5>{escape(full_title)}</h5>
              <div class="plot-container">
                {figure_html}
              </div>
            </section>
            """
            if group_onera_htc_by_roughness:
                if roughness_key == "smooth":
                    no_roughness_figures_html += figure_card_html
                else:
                    roughness_figures_html += figure_card_html
            else:
                figures_html += figure_card_html

    if group_onera_htc_by_roughness:
        figures_html = no_roughness_figures_html + roughness_figures_html

    if not figures_html:
        return ""

    roughness_summary = collect_cutdata_participant_roughness_summary(participants, case_id, grid_level, plot_spec)
    if roughness_filter_predicate is not None:
        roughness_summary = {
            participant_id: {key for key in keys if roughness_filter_predicate(key)}
            for participant_id, keys in roughness_summary.items()
        }
        roughness_summary = {participant_id: keys for participant_id, keys in roughness_summary.items() if keys}
    description = build_plot_description(plot_spec, [value for value in slice_positions if value is not None], case_id, roughness_summary=roughness_summary)

    # A participant can have an invalid placeholder zone for one roughness/bin
    # and still contribute a valid trace from another zone. Once roughness
    # plots are combined, do not show a contradictory "did not provide"
    # warning for participants that are actually represented in the figure.
    contributing_participant_ids = set(roughness_summary)
    all_skipped_notes = [
        note
        for note in all_skipped_notes
        if not any(f"Participant ID {participant_id} " in note for participant_id in contributing_participant_ids)
    ]

    notes_html = ""
    if all_skipped_notes:
        notes_html = '<ul class="plot-notes">' + "".join(f"<li>{escape(note)}</li>" for note in sorted(set(all_skipped_notes))) + "</ul>"

    return f"""
    <section class="plot-subsection" data-variable-key="{escape(plot_spec['plot_key'])}" data-variable-label="{escape(plot_spec['title'])}">
      <h4>{escape(plot_spec["title"])}</h4>
      <p class="plot-description">{escape(description)}</p>
      {notes_html}
      {figures_html}
    </section>
    """


def build_grid_level_cutdata_plots(participants, case_id: str, grid_level: str, roughness_filter_predicate=None) -> str:
    html = ""
    for plot_spec in CUTDATA_PLOTS:
        if not plot_matches_variable_filter(plot_spec):
            continue
        if plot_spec.get("plot_key") == "beta_cards_vs_s":
            if not ENABLE_COMBINED_BETA_BY_PARTICIPANT:
                continue
            html += build_combined_beta_section(participants, case_id, grid_level)
        else:
            html += build_plot_subsection(
                participants,
                case_id,
                grid_level,
                plot_spec,
                roughness_filter_predicate=roughness_filter_predicate,
            )
    if not html:
        return ""
    return f"""
    <section class="plot-filter-scope cutdata-filter-scope">
      <h3>cutData</h3>
      <div class="variable-filter-controls" data-filter-title="cutData variables"></div>
      {html}
    </section>
    """

def collect_cutdata_participant_roughness_summary(participants, case_id: str, grid_level: str, plot_spec: dict[str, Any]) -> dict[str, set[str]]:
    summary: dict[str, set[str]] = {}
    bins_filter = plot_spec.get("bins_filter")

    for participant, case_data, grid_data, dataset_data in iter_grid_data(participants, case_id, grid_level):
        cut_data = cut_data_for_plot(dataset_data, plot_spec)
        if cut_data is None:
            continue

        for zone_name, zone in cut_data.zones.items():
            zone_info = parse_ipw3_zone_name(zone_name)
            bins_id = zone_info["bins"] if zone_info is not None else None
            if bins_filter is not None and bins_id != bins_filter:
                continue

            x_column = find_column_case_insensitive(zone.data.columns, plot_spec["x_candidates"])
            y_column = find_column_case_insensitive(zone.data.columns, plot_spec["y_candidates"])
            if x_column is None:
                continue

            participant_summary = summary.setdefault(participant.participant_id, set())

            clean_y_column = None
            if plot_spec.get("plot_key") == "htc_vs_s":
                clean_y_column = find_column_case_insensitive(zone.data.columns, plot_spec.get("clean_y_candidates", []))

            if clean_y_column is not None:
                clean_data = valid_xy_rows(zone.data[[x_column, clean_y_column]].copy(), x_column, clean_y_column)
                if not clean_data.empty:
                    participant_summary.add("smooth")

            if y_column is not None:
                data = valid_xy_rows(zone.data[[x_column, y_column]].copy(), x_column, y_column)
                if not data.empty:
                    participant_summary.add(extract_roughness_key_from_zone_name(zone_name))

    return {participant_id: roughness_keys for participant_id, roughness_keys in summary.items() if roughness_keys}


def collect_cutdata_roughness_keys(participants, case_id: str, grid_level: str, plot_spec: dict[str, Any], slice_filter: float | None = None) -> list[str]:
    roughness_keys: set[str] = set()
    bins_filter = plot_spec.get("bins_filter")

    for participant, case_data, grid_data, dataset_data in iter_grid_data(participants, case_id, grid_level):
        cut_data = cut_data_for_plot(dataset_data, plot_spec)
        if cut_data is None:
            continue

        for zone_name, zone in cut_data.zones.items():
            zone_info = parse_ipw3_zone_name(zone_name)
            bins_id = zone_info["bins"] if zone_info is not None else None

            if bins_filter is not None and bins_id != bins_filter:
                continue

            slice_position = None
            if zone_info is not None:
                slice_position = decode_slice_position(zone_info["slice"])

            if not slice_matches_filter(slice_position, slice_filter):
                continue

            x_column = find_column_case_insensitive(zone.data.columns, plot_spec["x_candidates"])
            y_column = find_column_case_insensitive(zone.data.columns, plot_spec["y_candidates"])

            if x_column is None:
                continue

            clean_y_column = None
            if plot_spec.get("plot_key") == "htc_vs_s":
                clean_y_column = find_column_case_insensitive(zone.data.columns, plot_spec.get("clean_y_candidates", []))

            if clean_y_column is not None:
                clean_data = valid_xy_rows(zone.data[[x_column, clean_y_column]].copy(), x_column, clean_y_column)
                if not clean_data.empty:
                    roughness_keys.add("smooth")

            if y_column is not None:
                data = valid_xy_rows(zone.data[[x_column, y_column]].copy(), x_column, y_column)
                if not data.empty:
                    roughness_keys.add(extract_roughness_key_from_zone_name(zone_name))

    preferred_order = ["smooth", "0.5mm", "1mm", "1.5mm", "variable_roughness", "default_roughness", "unspecified_roughness"]
    ordered = [key for key in preferred_order if key in roughness_keys]
    remaining = sorted(roughness_keys - set(ordered), key=roughness_sort_key)

    return ordered + remaining
