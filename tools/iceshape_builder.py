from __future__ import annotations

import io
from contextlib import redirect_stdout
from functools import lru_cache
from html import escape
from pathlib import Path
from typing import Any
import re

import pandas as pd
import plotly.graph_objects as go

from .gatherParticipantData import CASE_SLICES, decode_slice_position, iter_grid_datasets, read_tecplot_dat
from .participant_style import participant_color


# Central place to tune ice-shape figure axes.
# Settings can be defined globally, per case, and per slice.
# Set any range to None to keep Plotly automatic scaling.
ICE_SHAPE_AXIS_SETTINGS = {
    "default": {
        "x_title": "X [m]",
        "y_title": "Z [m]",
        "x_range": None,
        "y_range": None,
    },
    "TC_ONERAM6": {
        "default": {
            "x_title": "X [m]",
            "y_title": "Z [m]",
            "x_range": None,
            "y_range": None,
        },
        "slices": {
            0.1: {
                "x_range": None,
                "y_range": None,
            },
            0.75: {
                "x_range": None,
                "y_range": None,
            },
            1.4: {
                "x_range": None,
                "y_range": None,
            },
        },
    },
    "TC_NACA0012_AE3932": {
        "default": {
            "x_title": "X [m]",
            "y_title": "Z [m]",
            "x_range": None,
            "y_range": None,
        },
        "slices": {
            0.9144: {
                "x_range": None,
                "y_range": None,
            },
        },
    },
    "TC_NACA0012_AE3933": {
        "default": {
            "x_title": "X [m]",
            "y_title": "Z [m]",
            "x_range": None,
            "y_range": None,
        },
        "slices": {
            0.9144: {
                "x_range": None,
                "y_range": None,
            },
        },
    },
}

def slugify(text: str) -> str:
    text = text.strip().lower()
    text = re.sub(r"[^a-z0-9]+", "_", text)
    return text.strip("_") or "figure"


def find_column_case_insensitive(columns, candidates: list[str]) -> str | None:
    lookup = {column.lower(): column for column in columns}
    for candidate in candidates:
        if candidate.lower() in lookup:
            return lookup[candidate.lower()]
    return None


def find_submitted_ice_xz_columns(columns) -> tuple[str | None, str | None]:
    """Return X/Z plotting columns for submitted ice-shape zones.

    Older submissions included both clean and iced coordinates:
        X, Y, Z, X_ICED, Y_ICED, Z_ICED

    Current submissions include only the iced coordinates. In that format, X/Z
    or CoordinateX/CoordinateZ are already the ice-shape coordinates; the clean
    trace is loaded separately from R00_REFERENCE.
    """
    x_column = find_column_case_insensitive(columns, ["X_ICED", "X_iced", "CoordinateX_Iced", "CoordinateX_iced"])
    z_column = find_column_case_insensitive(columns, ["Z_ICED", "Z_iced", "CoordinateZ_Iced", "CoordinateZ_iced"])

    if x_column is not None and z_column is not None:
        return x_column, z_column

    return (
        find_column_case_insensitive(columns, ["X", "CoordinateX"]),
        find_column_case_insensitive(columns, ["Z", "CoordinateZ"]),
    )


def valid_submitted_ice_shape_rows(dataframe: pd.DataFrame, x_column: str, z_column: str) -> pd.DataFrame:
    """Drop placeholder rows before plotting submitted ice-shape coordinates."""
    x_values = pd.to_numeric(dataframe[x_column], errors="coerce")
    z_values = pd.to_numeric(dataframe[z_column], errors="coerce")
    valid_mask = x_values.notna() & z_values.notna() & (x_values > -998.0) & (z_values > -998.0)
    return dataframe.loc[valid_mask]


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


def format_roughness_title(roughness_key: str | None) -> str:
    if roughness_key is None:
        return "Default roughness"

    if roughness_key == "smooth":
        return "No Roughness"

    if roughness_key == "variable_roughness":
        return "Variable roughness"

    if roughness_key in {"default_roughness", "unspecified_roughness"}:
        return "Default roughness"

    if roughness_key.endswith("mm"):
        return f"Roughness height = {roughness_key[:-2]} mm"

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
        return "Default roughness"
    return ", ".join(format_roughness_title(key) for key in sorted(roughness_keys, key=roughness_sort_key))


def format_participant_roughness_summary(summary: dict[str, set[str]]) -> str:
    if not summary:
        return "Detected roughness by participant: none found."
    entries = [f"{participant_id}: {format_roughness_list(roughness_keys)}" for participant_id, roughness_keys in sorted(summary.items())]
    return "Detected roughness by participant: " + "; ".join(entries) + "."


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


def set_defer_plotly_html(output_dir: Path | None) -> None:
    global DEFER_PLOTLY_DIR
    DEFER_PLOTLY_DIR = output_dir


def figure_to_html_div(fig: go.Figure, filename: str) -> str:
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


def format_slice_positions(slice_values: list[float]) -> str:
    if not slice_values:
        return "No slice location detected."
    unique_values = sorted(set(round(value, 8) for value in slice_values))
    return ", ".join(f"Y = {value:g} m" for value in unique_values)


def style_xy_figure(
    fig: go.Figure,
    x_label: str,
    y_label: str,
    height: int = 650,
    x_range: list[float] | tuple[float, float] | None = None,
    y_range: list[float] | tuple[float, float] | None = None,
) -> go.Figure:
    fig.update_layout(
        font=dict(family="Arial, Helvetica, sans-serif", size=16),
        autosize=True,
        height=height,
        title=None,
        showlegend=True,
        xaxis=dict(title=dict(text=x_label, font=dict(size=18)), range=x_range, ticks="outside", showline=True, linecolor="black", linewidth=2, mirror=True, showgrid=True, gridcolor="lightgray", zeroline=False),
        yaxis=dict(title=dict(text=y_label, font=dict(size=18)), range=y_range, ticks="outside", showline=True, linecolor="black", linewidth=2, mirror=True, showgrid=True, gridcolor="lightgray", zeroline=False),
        legend=dict(orientation="v", x=1.02, xanchor="left", y=1.0, yanchor="top"),
        margin=dict(l=90, r=260, t=30, b=80),
        plot_bgcolor="white",
        paper_bgcolor="white",
    )
    return fig


def ice_shape_axis_config(case_id: str, slice_filter: float | None) -> dict[str, Any]:
    config = dict(ICE_SHAPE_AXIS_SETTINGS["default"])
    case_config = ICE_SHAPE_AXIS_SETTINGS.get(case_id, {})

    if isinstance(case_config, dict):
        config.update(case_config.get("default", {}))

        if slice_filter is not None:
            rounded_slice = round(slice_filter, 8)
            for configured_slice, slice_config in case_config.get("slices", {}).items():
                if abs(float(configured_slice) - rounded_slice) <= 1.0e-8:
                    config.update(slice_config)
                    break

    return config


def iter_grid_data(participants, case_id: str, grid_level: str):
    """Iterate over all datasets for one case/grid level.

    This adapter uses the new gatherParticipantData hierarchy:
        participant -> case -> grid level -> dataset

    Yields:
        participant, case_data, grid_data, dataset_data
    """
    yield from iter_grid_datasets(participants, case_id, grid_level)


def parse_ipw3_ice_shape_zone_name(zone_name: str) -> dict[str, str] | None:
    zone_name = zone_name.strip()
    slice_pattern = re.compile(r"^SLICE_Y_(?P<slice>.+?)(?:_(?P<bins>BINS\d+))?(?:_(?P<tail>.*))?$", re.IGNORECASE)
    mccs_pattern = re.compile(r"^MCCS(?:_(?P<bins>BINS\d+))?(?:_(?P<tail>.*))?$", re.IGNORECASE)

    match = slice_pattern.match(zone_name)
    if match is not None:
        tail = match.group("tail") or ""
        dataset_match = re.search(r"(?:^|_)(D\d+)(?:_|$)", tail, re.IGNORECASE)
        layer_match = re.search(r"(?:^|_)(L\d+|LAYER\d+)(?:_|$)", tail, re.IGNORECASE)
        shape_role_match = re.search(r"(?:^|_)(SINGLE_LAYER|FINAL_LAYER)(?:_|$)", tail, re.IGNORECASE)
        return {
            "type": "SLICE",
            "slice": match.group("slice"),
            "bins": (match.group("bins") or "BINSXX").upper(),
            "dataset": dataset_match.group(1).upper() if dataset_match is not None else "DXX",
            "layer": layer_match.group(1).upper() if layer_match is not None else "",
            "shape_role": shape_role_match.group(1).upper() if shape_role_match is not None else "",
        }

    match = mccs_pattern.match(zone_name)
    if match is not None:
        tail = match.group("tail") or ""
        dataset_match = re.search(r"(?:^|_)(D\d+)(?:_|$)", tail, re.IGNORECASE)
        layer_match = re.search(r"(?:^|_)(L\d+|LAYER\d+)(?:_|$)", tail, re.IGNORECASE)
        shape_role_match = re.search(r"(?:^|_)(SINGLE_LAYER|FINAL_LAYER)(?:_|$)", tail, re.IGNORECASE)
        return {
            "type": "MCCS",
            "slice": "",
            "bins": (match.group("bins") or "BINSXX").upper(),
            "dataset": dataset_match.group(1).upper() if dataset_match is not None else "DXX",
            "layer": layer_match.group(1).upper() if layer_match is not None else "",
            "shape_role": shape_role_match.group(1).upper() if shape_role_match is not None else "",
        }

    return None


def slice_matches_filter(slice_position: float | None, slice_filter: float | None, tolerance: float = 1.0e-6) -> bool:
    if slice_filter is None:
        return True
    if slice_position is None:
        return False
    return abs(slice_position - slice_filter) <= tolerance


def expected_slice_positions(case_id: str) -> list[float | None]:
    slices = CASE_SLICES.get(case_id)
    if not slices:
        return [None]
    return sorted(set(round(value, 8) for value in slices))


def bin_sort_key(bins_id: str) -> tuple[int, str]:
    match = re.search(r"\d+", bins_id)
    if match is None:
        return (10**9, bins_id)
    return (int(match.group(0)), bins_id)


def detected_ice_shape_bins(participants, case_id: str, grid_level: str) -> list[str]:
    bins_ids: set[str] = set()

    for participant, case_data, grid_data, dataset_data in iter_grid_data(participants, case_id, grid_level):
        if dataset_data.ice_shape_data is None:
            continue

        for zone_name in dataset_data.ice_shape_data.zones:
            zone_info = parse_ipw3_ice_shape_zone_name(zone_name)
            if zone_info is not None and zone_info["bins"] != "BINSXX":
                bins_ids.add(zone_info["bins"])

    return sorted(bins_ids, key=bin_sort_key)


def detected_ice_shape_roughness_keys(participants, case_id: str, grid_level: str, slice_filter: float | None = None, bins_filter: str | None = None) -> list[str]:
    roughness_keys: set[str] = set()

    for participant, case_data, grid_data, dataset_data in iter_grid_data(participants, case_id, grid_level):
        if dataset_data.ice_shape_data is None:
            continue

        for zone_name, zone in dataset_data.ice_shape_data.zones.items():
            zone_info = parse_ipw3_ice_shape_zone_name(zone_name)
            if zone_info is None:
                continue
            if bins_filter is not None and zone_info["bins"] != bins_filter:
                continue
            slice_position = decode_slice_position(zone_info["slice"]) if zone_info["slice"] else None
            if not slice_matches_filter(slice_position, slice_filter):
                continue
            x_column, z_column = find_submitted_ice_xz_columns(zone.data.columns)
            if x_column is None or z_column is None:
                continue
            if valid_submitted_ice_shape_rows(zone.data, x_column, z_column).empty:
                continue
            roughness_keys.add(extract_roughness_key_from_zone_name(zone_name))

    preferred_order = ["smooth", "0.5mm", "1mm", "1.5mm", "variable_roughness", "default_roughness", "unspecified_roughness"]
    ordered = [key for key in preferred_order if key in roughness_keys]
    remaining = sorted(roughness_keys - set(ordered), key=roughness_sort_key)
    return ordered + remaining


def collect_ice_shape_participant_roughness_summary(participants, case_id: str, grid_level: str) -> dict[str, set[str]]:
    summary: dict[str, set[str]] = {}

    for participant, case_data, grid_data, dataset_data in iter_grid_data(participants, case_id, grid_level):
        if dataset_data.ice_shape_data is None:
            continue

        for zone_name, zone in dataset_data.ice_shape_data.zones.items():
            x_column, z_column = find_submitted_ice_xz_columns(zone.data.columns)
            if x_column is None or z_column is None:
                continue
            if valid_submitted_ice_shape_rows(zone.data, x_column, z_column).empty:
                continue
            summary.setdefault(participant.participant_id, set()).add(extract_roughness_key_from_zone_name(zone_name))

    return summary


def clean_reference_path_for_case(case_id: str) -> Path | None:
    if "ONERAM6" in case_id.upper():
        return Path("R00_REFERENCE") / "ONERAM6_CLEAN.dat"
    if "NACA0012" in case_id.upper():
        return Path("R00_REFERENCE") / "NACA0012_CLEAN.dat"
    return None


@lru_cache(maxsize=None)
def load_clean_reference_data(reference_path_text: str):
    # Reference geometry is reused by many figures; cache it and keep malformed
    # placeholder rows from flooding the build output.
    with redirect_stdout(io.StringIO()):
        return read_tecplot_dat(Path(reference_path_text), process_cutdata=False)


def ordered_clean_reference_columns(case_id: str, zone, x_column: str, z_column: str):
    """Return clean-reference points in a plotting-friendly order.

    NACA0012 reference rows can arrive in exported segment order, which makes a
    line trace jump between the leading and trailing edge. Sort that airfoil
    into a conventional upper-surface then lower-surface loop for display.
    """
    data = zone.data[[x_column, z_column]].dropna()
    if "NACA0012" not in case_id.upper() or data.empty:
        return data[x_column], data[z_column]

    segments = []
    segment_start = 0
    previous = None
    for row_number, row in enumerate(data.itertuples(index=False)):
        point = (float(getattr(row, x_column)), float(getattr(row, z_column)))
        if previous is not None:
            distance = ((point[0] - previous[0]) ** 2 + (point[1] - previous[1]) ** 2) ** 0.5
            if distance > 0.05:
                segment = data.iloc[segment_start:row_number].drop_duplicates()
                if len(segment) > 1:
                    segments.append(segment)
                segment_start = row_number
        previous = point

    segment = data.iloc[segment_start:].drop_duplicates()
    if len(segment) > 1:
        segments.append(segment)

    if not segments:
        return data[x_column], data[z_column]

    midline = float(data[z_column].mean())
    upper_segments = []
    lower_segments = []
    for segment in segments:
        if float(segment[z_column].mean()) >= midline:
            oriented = segment.sort_values(x_column, ascending=True)
            upper_segments.append(oriented)
        else:
            oriented = segment.sort_values(x_column, ascending=False)
            lower_segments.append(oriented)

    upper_segments.sort(key=lambda segment: float(segment[x_column].iloc[0]))
    lower_segments.sort(key=lambda segment: float(segment[x_column].iloc[0]), reverse=True)
    ordered = pd.concat(upper_segments + lower_segments, ignore_index=True).drop_duplicates()

    if ordered.empty:
        return data[x_column], data[z_column]

    first = ordered.iloc[0]
    last = ordered.iloc[-1]
    if abs(float(first[x_column]) - float(last[x_column])) > 1.0e-10 or abs(float(first[z_column]) - float(last[z_column])) > 1.0e-10:
        ordered = pd.concat([ordered, first.to_frame().T], ignore_index=True)

    return ordered[x_column], ordered[z_column]


def add_clean_reference_trace(fig: go.Figure, case_id: str, slice_filter: float | None = None) -> tuple[int, list[float]]:
    reference_path = clean_reference_path_for_case(case_id)
    if reference_path is None or not reference_path.exists():
        return 0, []

    reference_data = load_clean_reference_data(str(reference_path))
    trace_count = 0
    slice_positions: list[float] = []

    for zone_name, zone in reference_data.zones.items():
        zone_info = parse_ipw3_ice_shape_zone_name(zone_name)
        slice_position = None
        slice_text = "unknown"

        if zone_info is not None and zone_info["slice"]:
            slice_position = decode_slice_position(zone_info["slice"])
            slice_text = f"Y = {slice_position:g} m" if slice_position is not None else zone_info["slice"]

        if not slice_matches_filter(slice_position, slice_filter):
            continue

        x_column = find_column_case_insensitive(zone.data.columns, ["X", "CoordinateX"])
        z_column = find_column_case_insensitive(zone.data.columns, ["Z", "CoordinateZ"])
        if x_column is None or z_column is None:
            continue
        x_values, z_values = ordered_clean_reference_columns(case_id, zone, x_column, z_column)

        fig.add_trace(
            go.Scatter(
                x=x_values,
                y=z_values,
                mode="lines",
                name="Clean reference",
                legendgroup="clean_reference",
                line=dict(color="black", width=2),
                hovertemplate=(
                    f"Case: {escape(case_id)}<br>"
                    f"Source: {escape(str(reference_path))}<br>"
                    f"Shape: clean reference<br>"
                    f"Slice: {escape(slice_text)}<br>"
                    f"Zone: {escape(zone_name)}<br>"
                    f"{escape(x_column)}=%{{x}}<br>"
                    f"{escape(z_column)}=%{{y}}<extra></extra>"
                ),
            )
        )
        trace_count += 1
        if slice_position is not None:
            slice_positions.append(slice_position)

    return trace_count, slice_positions


def build_single_layer_ice_shape_figure(participants, case_id: str, grid_level: str, slice_filter: float | None = None, bins_filter: str | None = None, roughness_filter: str | None = None) -> tuple[go.Figure, int, list[float]]:
    """Build the single-layer ice-shape figure from iceShape zones.

    Important distinction:
        - SINGLE_LAYER zones from finalIceShape / iceShape files are drawn here.
        - FINAL_LAYER zones are drawn separately by the multi-layer/final plot.

    The plot is always X-Z:
        clean shape: reference clean shape from R00_REFERENCE
        iced shape : X_ICED vs Z_ICED from the SINGLE_LAYER zone
    """

    fig = go.Figure()
    _, reference_slice_positions = add_clean_reference_trace(fig, case_id, slice_filter=slice_filter)
    trace_count = 0
    slice_positions: list[float] = list(reference_slice_positions)

    for participant, case_data, grid_data, dataset_data in iter_grid_data(participants, case_id, grid_level):
        if dataset_data.ice_shape_data is None:
            continue

        label = participant_label(participant, dataset_data, grid_data)
        color = participant_color(participant.participant_id)

        for zone_index, (zone_name, zone) in enumerate(dataset_data.ice_shape_data.zones.items(), start=1):
            zone_info = parse_ipw3_ice_shape_zone_name(zone_name)

            if zone_info is not None:
                bins_id = zone_info["bins"]
                shape_type = zone_info["type"]
                shape_role = zone_info["shape_role"]
                roughness_key = extract_roughness_key_from_zone_name(zone_name)
                slice_position = None

                if zone_info["slice"]:
                    slice_position = decode_slice_position(zone_info["slice"])
                    if slice_position is not None:
                        slice_text = f"Y = {slice_position:g} m"
                    else:
                        slice_text = zone_info["slice"]
                else:
                    slice_text = "MCCS"
            else:
                bins_id = "not specified"
                shape_type = "unknown"
                shape_role = ""
                roughness_key = "default_roughness"
                slice_text = "unknown"
                slice_position = None

            if shape_role != "SINGLE_LAYER":
                continue
            if bins_filter is not None and bins_id != bins_filter:
                continue
            if roughness_filter is not None and roughness_key != roughness_filter:
                continue

            if not slice_matches_filter(slice_position, slice_filter):
                continue
            if slice_position is not None:
                slice_positions.append(slice_position)

            x_iced_column, z_iced_column = find_submitted_ice_xz_columns(zone.data.columns)
            if x_iced_column is None or z_iced_column is None:
                continue
            plot_data = valid_submitted_ice_shape_rows(zone.data, x_iced_column, z_iced_column)
            if plot_data.empty:
                continue

            trace_name = label

            fig.add_trace(
                go.Scatter(
                    x=plot_data[x_iced_column],
                    y=plot_data[z_iced_column],
                    mode="lines",
                    name=trace_name,
                    legendgroup=label,
                    line=dict(color=color),
                    hovertemplate=(
                        f"Participant: {escape(label)}<br>"
                        f"Case: {escape(case_id)}<br>"
                        f"Grid: {escape(grid_level)}<br>"
                        f"Source: finalIceShape / iceShape<br>"
                        f"Shape type: {escape(str(shape_type))}<br>"
                        f"Shape role: {escape(str(shape_role))}<br>"
                        f"Bins: {escape(str(bins_id))}<br>"
                        f"Roughness: {escape(format_roughness_title(roughness_key))}<br>"
                        f"Slice: {escape(str(slice_text))}<br>"
                        f"Zone: {escape(zone_name)}<br>"
                        f"{escape(x_iced_column)}=%{{x}}<br>"
                        f"{escape(z_iced_column)}=%{{y}}<extra></extra>"
                    ),
                )
            )

            trace_count += 1

    axis_config = ice_shape_axis_config(case_id, slice_filter)
    style_xy_figure(
        fig,
        axis_config["x_title"],
        axis_config["y_title"],
        x_range=axis_config["x_range"],
        y_range=axis_config["y_range"],
    )
    fig.update_yaxes(scaleanchor="x", scaleratio=1.0)

    return fig, trace_count, slice_positions


def build_multilayer_ice_shape_figure(participants, case_id: str, grid_level: str, slice_filter: float | None = None, bins_filter: str | None = None, roughness_filter: str | None = None) -> tuple[go.Figure, int, list[float]]:
    fig = go.Figure()
    _, reference_slice_positions = add_clean_reference_trace(fig, case_id, slice_filter=slice_filter)
    trace_count = 0
    slice_positions: list[float] = list(reference_slice_positions)

    for participant, case_data, grid_data, dataset_data in iter_grid_data(participants, case_id, grid_level):
        if dataset_data.ice_shape_data is None:
            continue

        label = participant_label(participant, dataset_data, grid_data)
        color = participant_color(participant.participant_id)
        num_layers = dataset_data.ice_shape_data.auxdata.get("NUM_LAYERS", "unknown")
        data_type = dataset_data.ice_shape_data.auxdata.get("DATA_TYPE", "unknown")

        for zone_index, (zone_name, zone) in enumerate(dataset_data.ice_shape_data.zones.items(), start=1):
            zone_info = parse_ipw3_ice_shape_zone_name(zone_name)

            if zone_info is not None:
                bins_id = zone_info["bins"]
                shape_type = zone_info["type"]
                shape_role = zone_info["shape_role"]
                layer_id = zone_info["layer"] or f"zone {zone_index}"
                roughness_key = extract_roughness_key_from_zone_name(zone_name)
                slice_position = None
                if zone_info["slice"]:
                    slice_position = decode_slice_position(zone_info["slice"])
                    if slice_position is not None:
                        slice_text = f"Y = {slice_position:g} m"
                    else:
                        slice_text = zone_info["slice"]
                else:
                    slice_text = "MCCS"
            else:
                bins_id = "not specified"
                shape_type = "unknown"
                shape_role = ""
                layer_id = f"zone {zone_index}"
                roughness_key = "default_roughness"
                slice_text = "unknown"
                slice_position = None

            if shape_role == "SINGLE_LAYER":
                continue
            if bins_filter is not None and bins_id != bins_filter:
                continue
            if roughness_filter is not None and roughness_key != roughness_filter:
                continue

            if not slice_matches_filter(slice_position, slice_filter):
                continue
            if slice_position is not None:
                slice_positions.append(slice_position)

            x_iced_column, z_iced_column = find_submitted_ice_xz_columns(zone.data.columns)

            if x_iced_column is None or z_iced_column is None:
                continue
            plot_data = valid_submitted_ice_shape_rows(zone.data, x_iced_column, z_iced_column)
            if plot_data.empty:
                continue

            trace_name = label if layer_id.startswith("zone ") else f"{label} {layer_id}"
            fig.add_trace(go.Scatter(x=plot_data[x_iced_column], y=plot_data[z_iced_column], mode="lines", name=trace_name, legendgroup=label, line=dict(color=color), hovertemplate=(f"Participant: {escape(label)}<br>" f"Case: {escape(case_id)}<br>" f"Grid: {escape(grid_level)}<br>" f"Shape type: {escape(str(shape_type))}<br>" f"Shape role: {escape(str(shape_role or 'FINAL_LAYER/legacy'))}<br>" f"Bins: {escape(str(bins_id))}<br>" f"Roughness: {escape(format_roughness_title(roughness_key))}<br>" f"Slice: {escape(str(slice_text))}<br>" f"Layer/zone: {escape(str(layer_id))}<br>" f"NUM_LAYERS: {escape(str(num_layers))}<br>" f"DATA_TYPE: {escape(str(data_type))}<br>" f"Zone: {escape(zone_name)}<br>" f"{escape(x_iced_column)}=%{{x}}<br>" f"{escape(z_iced_column)}=%{{y}}<extra></extra>")))
            trace_count += 1

    axis_config = ice_shape_axis_config(case_id, slice_filter)
    style_xy_figure(
        fig,
        axis_config["x_title"],
        axis_config["y_title"],
        x_range=axis_config["x_range"],
        y_range=axis_config["y_range"],
    )
    fig.update_yaxes(scaleanchor="x", scaleratio=1.0)
    return fig, trace_count, slice_positions


def build_ice_shape_section(participants, case_id: str, grid_level: str) -> str:
    """Build both ice-shape figures for one case/grid level.

    This section contains:
        1. SINGLE_LAYER zones from finalIceShape / iceShape.
        2. FINAL_LAYER or legacy final zones from finalIceShape / iceShape.
    """

    single_figures_html = ""
    multi_figures_html = ""
    configured_slices = expected_slice_positions(case_id)
    configured_bins = detected_ice_shape_bins(participants, case_id, grid_level)
    if not configured_bins:
        configured_bins = ["BINSXX"]
    configured_roughness = detected_ice_shape_roughness_keys(participants, case_id, grid_level)
    if not configured_roughness:
        configured_roughness = [None]

    for slice_position in configured_slices:
        slice_title = f"Y = {slice_position:g} m" if slice_position is not None else "Slice unknown"
        slice_slug = f"_slice_{slice_position:g}".replace(".", "p") if slice_position is not None else "_slice_unknown"

        for bins_id in configured_bins:
            bins_slug = f"_{bins_id.lower()}"
            figure_roughness = detected_ice_shape_roughness_keys(participants, case_id, grid_level, slice_filter=slice_position, bins_filter=bins_id)
            if not figure_roughness:
                figure_roughness = [None]

            for roughness_key in figure_roughness:
                roughness_slug = f"_roughness_{slugify(roughness_key or 'unspecified')}"
                roughness_title = format_roughness_title(roughness_key)
                bin_title = f"{slice_title} | {bins_id} | {roughness_title}"
                slice_key = slugify(slice_title)
                roughness_key_text = roughness_key or "unspecified"
                roughness_filter_key = slugify(roughness_key_text)
                roughness_filter_label = roughness_title

                single_fig, single_trace_count, single_slice_positions = build_single_layer_ice_shape_figure(participants, case_id, grid_level, slice_filter=slice_position, bins_filter=bins_id, roughness_filter=roughness_key)
                if single_trace_count == 0:
                    single_figure_html = empty_placeholder(
                        title=f"Single-layer ice shape | {bin_title}",
                        message="No matching SINGLE_LAYER iceShape zones were found yet for this case/grid level, slice, bin set, and roughness.",
                    )
                else:
                    single_filename = f"{slugify(case_id)}_{grid_level}_single_layer_ice_shape{slice_slug}{bins_slug}{roughness_slug}"
                    single_figure_html = figure_to_html_div(single_fig, filename=single_filename)

                single_figures_html += f"""
                <section class="slice-plot-group" data-slice-key="{escape(slice_key)}" data-slice-label="{escape(slice_title)}" data-roughness-key="{escape(roughness_filter_key)}" data-roughness-label="{escape(roughness_filter_label)}">
                  <h5>{escape(bin_title)}</h5>
                  <div class="plot-container">
                    {single_figure_html}
                  </div>
                </section>
                """

                multi_fig, multi_trace_count, multi_slice_positions = build_multilayer_ice_shape_figure(participants, case_id, grid_level, slice_filter=slice_position, bins_filter=bins_id, roughness_filter=roughness_key)
                if multi_trace_count == 0:
                    multi_figure_html = empty_placeholder(
                        title=f"Multi-layer final ice shape | {bin_title}",
                        message="No matching FINAL_LAYER iceShape zones were found yet for this case/grid level, slice, bin set, and roughness.",
                    )
                else:
                    multi_filename = f"{slugify(case_id)}_{grid_level}_multilayer_ice_shape{slice_slug}{bins_slug}{roughness_slug}"
                    multi_figure_html = figure_to_html_div(multi_fig, filename=multi_filename)

                multi_figures_html += f"""
                <section class="slice-plot-group" data-slice-key="{escape(slice_key)}" data-slice-label="{escape(slice_title)}" data-roughness-key="{escape(roughness_filter_key)}" data-roughness-label="{escape(roughness_filter_label)}">
                  <h5>{escape(bin_title)}</h5>
                  <div class="plot-container">
                    {multi_figure_html}
                  </div>
                </section>
                """

    configured_slice_text = format_slice_positions([value for value in configured_slices if value is not None])
    configured_bins_text = ", ".join(configured_bins)
    configured_roughness_text = ", ".join(format_roughness_title(value) for value in configured_roughness)
    participant_roughness_text = format_participant_roughness_summary(collect_ice_shape_participant_roughness_summary(participants, case_id, grid_level))
    single_description = (
        "Single-layer ice-shape comparison extracted from finalIceShape / iceShape zones whose names contain SINGLE_LAYER. "
        "The clean reference shape is drawn from R00_REFERENCE, and submitted ice-shape files are treated as iced coordinates. "
        f"Slice location(s): {configured_slice_text}. "
        f"Bin set(s): {configured_bins_text}. "
        f"Roughness condition(s): {configured_roughness_text}. "
        f"{participant_roughness_text} "
        "Legend: PID or PID.DID, bin set, and layer type."
    )

    multi_description = (
        "Multi-layer final ice-shape comparison extracted from finalIceShape / iceShape zones whose names contain FINAL_LAYER. "
        "Legacy zones without SINGLE_LAYER or FINAL_LAYER are kept in this final-layer plot. "
        "The clean reference shape is drawn from R00_REFERENCE. "
        f"Slice location(s): {configured_slice_text}. "
        f"Bin set(s): {configured_bins_text}. "
        f"Roughness condition(s): {configured_roughness_text}. "
        f"{participant_roughness_text} "
        "Legend: PID or PID.DID and layer/zone index."
    )

    return f"""
    <section class="plot-filter-scope ice-shape-filter-scope">
    <h3>Ice shape</h3>
    <div class="variable-filter-controls" data-filter-title="Ice-shape variables"></div>
    <div class="variable-filter-controls ice-shape-filter-controls"></div>

    <section class="plot-subsection ice-shape-subsection" data-variable-key="single_layer_ice_shape" data-variable-label="Single-layer ice shape">
      <h4>Single-layer ice shape</h4>
      <p class="plot-description">{escape(single_description)}</p>
      {single_figures_html}
    </section>

    <section class="plot-subsection ice-shape-subsection" data-variable-key="multi_layer_final_ice_shape" data-variable-label="Multi-layer final ice shape">
      <h4>Multi-layer final ice shape</h4>
      <p class="plot-description">{escape(multi_description)}</p>
      {multi_figures_html}
    </section>
    </section>
    """
