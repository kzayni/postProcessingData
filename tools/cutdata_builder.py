from __future__ import annotations

from pathlib import Path
from html import escape
from dataclasses import dataclass
from typing import Any
import re

import pandas as pd
import plotly.graph_objects as go
import plotly.io as pio

from .gatherParticipantData import CASE_SLICES, decode_slice_position, iter_grid_datasets
from .participant_style import participant_color, participant_legend_rank

SAVE_IMAGE_PREVIEWS = False
IMAGE_PREVIEW_ROOT = Path("IMAGES_PREVIEW")
BETA_BINS = ["BINS01", "BINS03", "BINS07", "BINS15"]
ENABLE_COMBINED_BETA_BY_PARTICIPANT = False

REFERENCE_DATA_SOURCES: list[dict[str, Any]] = []

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
        "plot_key": "freezing_fraction_vs_s",
        "title": "Freezing fraction vs s",
        "description": "Freezing fraction along the selected surface cut(s).",
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
    pio.write_images(list(figures), list(paths), width=1200, height=900, scale=scale)
    PNG_EXPORT_QUEUE.clear()


def figure_to_html_div(fig: go.Figure, filename: str) -> str:
    if PNG_EXPORT_DIR is not None:
        PNG_EXPORT_DIR.mkdir(parents=True, exist_ok=True)
        PNG_EXPORT_QUEUE.append((fig, PNG_EXPORT_DIR / f"{filename}.png"))
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
    return figure_html


def empty_placeholder(title: str, message: str) -> str:
    return f"""
    <div class="placeholder-card">
      <h4>{escape(title)}</h4>
      <p>{escape(message)}</p>
    </div>
    """


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


def style_xy_figure(fig: go.Figure, x_label: str, y_label: str, height: int = 560, legend_right: bool = True, reverse_y_axis: bool = False) -> go.Figure:
    if legend_right:
        legend = dict(orientation="v", x=1.02, xanchor="left", y=1.0, yanchor="top")
        margin = dict(l=90, r=220, t=30, b=80)
    else:
        legend = dict(orientation="h", x=0.0, xanchor="left", y=1.12, yanchor="bottom")
        margin = dict(l=90, r=40, t=70, b=80)

    yaxis = dict(title=dict(text=y_label, font=dict(size=18)), ticks="outside", showline=True, linecolor="black", linewidth=2, mirror=True, showgrid=True, gridcolor="lightgray", zeroline=False)
    if reverse_y_axis:
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


def iter_grid_data(participants, case_id: str, grid_level: str):
    """Iterate over all datasets for one case/grid level.

    This adapter keeps the rest of this builder readable while using the new
    gatherParticipantData hierarchy:
        participant -> case -> grid level -> dataset

    Yields:
        participant, case_data, grid_data, dataset_data
    """
    yield from iter_grid_datasets(participants, case_id, grid_level)


def get_cutdata_zones_by_bins(dataset_data) -> dict[str, list[tuple[str, Any, float | None]]]:
    grouped: dict[str, list[tuple[str, Any, float | None]]] = {bins_id: [] for bins_id in BETA_BINS}
    if dataset_data.cut_data is None:
        return grouped
    for zone_name, zone in dataset_data.cut_data.zones.items():
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
    trace_count = 0
    sources = get_reference_sources(case_id, grid_level, plot_key)
    for source in sources:
        print(f"Reference source configured but not loaded yet: {source}")
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


def build_cutdata_figure(participants, case_id: str, grid_level: str, plot_spec: dict[str, Any], slice_filter: float | None = None, roughness_filter: str | None = None) -> tuple[go.Figure, int, list[float], list[str]]:
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
        if dataset_data.cut_data is None:
            continue
        # For fields shared by all bin solutions (everything except Beta),
        # numeric bin ordering plus the bin-independent trace key below keeps
        # the first available solution for each slice and ks condition.
        zone_items = sorted(dataset_data.cut_data.zones.items(), key=cutdata_zone_sort_key)
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
                    mode="lines",
                    name=trace_name,
                    legendgroup=trace_name,
                    legendrank=participant_legend_rank(participant.participant_id),
                    line=dict(color=color),
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
    style_xy_figure(fig, plot_spec["x_label"], plot_spec["y_label"], reverse_y_axis=plot_spec.get("reverse_y_axis", False))
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
            if slice_position is not None:
                slice_positions.append(slice_position)
            slice_text = f"Y = {slice_position:g} m" if slice_position is not None else "unknown"
            fig.add_trace(
                go.Scatter(
                    x=data[x_column],
                    y=data[y_column],
                    mode="lines",
                    name=bins_id,
                    legendgroup=bins_id,
                    line=dict(color=color, dash=BIN_LINE_DASHES.get(bins_id, "solid")),
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
    return figure_to_html_div(fig, filename=filename), trace_count, slice_positions


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
        cards_html = empty_placeholder(title="Combined Beta", message="No matching cutData variables were found yet for this case/grid level.")
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


def build_plot_subsection(participants, case_id: str, grid_level: str, plot_spec: dict[str, Any]) -> str:
    slice_positions = collect_cutdata_slice_positions(participants, case_id, grid_level, bins_filter=plot_spec.get("bins_filter"))

    if not slice_positions:
        slice_positions = [None]

    figures_html = ""
    all_skipped_notes: list[str] = []

    for slice_position in slice_positions:
        roughness_keys = collect_cutdata_roughness_keys(participants, case_id, grid_level, plot_spec, slice_filter=slice_position)

        if not roughness_keys:
            roughness_keys = [None]

        if "NACA0012" in case_id.upper():
            combined_fig = None
            combined_trace_count = 0
            combined_skipped_notes: list[str] = []
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
                combined_trace_count += roughness_trace_count
                if combined_fig is None:
                    combined_fig = go.Figure(roughness_fig)
                else:
                    combined_fig.add_traces(list(roughness_fig.data))

            all_skipped_notes.extend(combined_skipped_notes)
            slice_title = f"Y = {slice_position:g} m" if slice_position is not None else "Slice unknown"
            full_title = f"{slice_title} | All roughness heights"
            if combined_trace_count == 0 or combined_fig is None:
                figure_html = empty_placeholder(
                    title=f"{plot_spec['title']} | {full_title}",
                    message="No matching cutData variables were found yet for this case/grid level and slice.",
                )
            else:
                slice_slug = f"_slice_{slice_position:g}".replace(".", "p") if slice_position is not None else "_slice_unknown"
                filename = f"{slugify(case_id)}_{grid_level}_{plot_spec['filename_slug']}{slice_slug}_all_roughness"
                figure_html = figure_to_html_div(combined_fig, filename=filename)
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
                figure_html = empty_placeholder(
                    title=f"{plot_spec['title']} | {full_title}",
                    message="No matching cutData variables were found yet for this case/grid level, slice, and roughness.",
                )
            else:
                slice_slug = f"_slice_{slice_position:g}".replace(".", "p") if slice_position is not None else "_slice_unknown"
                roughness_slug = f"_roughness_{slugify(roughness_key or 'unspecified')}"
                filename = f"{slugify(case_id)}_{grid_level}_{plot_spec['filename_slug']}{slice_slug}{roughness_slug}"
                figure_html = figure_to_html_div(fig, filename=filename)

            figures_html += f"""
            <section class="slice-plot-group">
              <h5>{escape(full_title)}</h5>
              <div class="plot-container">
                {figure_html}
              </div>
            </section>
            """

    roughness_summary = collect_cutdata_participant_roughness_summary(participants, case_id, grid_level, plot_spec)
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


def build_grid_level_cutdata_plots(participants, case_id: str, grid_level: str) -> str:
    html = ""
    for plot_spec in CUTDATA_PLOTS:
        if plot_spec.get("plot_key") == "beta_cards_vs_s":
            if not ENABLE_COMBINED_BETA_BY_PARTICIPANT:
                continue
            html += build_combined_beta_section(participants, case_id, grid_level)
        else:
            html += build_plot_subsection(participants, case_id, grid_level, plot_spec)
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
        if dataset_data.cut_data is None:
            continue

        for zone_name, zone in dataset_data.cut_data.zones.items():
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
        if dataset_data.cut_data is None:
            continue

        for zone_name, zone in dataset_data.cut_data.zones.items():
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
