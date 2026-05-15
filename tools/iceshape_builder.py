from __future__ import annotations

import io
from contextlib import redirect_stdout
from functools import lru_cache
from html import escape
from pathlib import Path
from typing import Any
import re

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


def participant_label(participant, dataset_data) -> str:
    """Return the legend label for one participant/grid-level dataset.

    Dataset IDs now live under:
        participant.cases[case_id].grid_levels[grid_level].datasets[dataset_id]
    so the label is still PID.DID, but DID is local to the case/grid level.
    """
    return f"{participant.participant_id}.{dataset_data.dataset_id}"


def plotly_config(filename: str) -> dict[str, Any]:
    return {
        "responsive": True,
        "displaylogo": False,
        "toImageButtonOptions": {"format": "png", "filename": filename, "height": 900, "width": 1200, "scale": 3},
    }


def figure_to_html_div(fig: go.Figure, filename: str) -> str:
    return fig.to_html(full_html=False, include_plotlyjs="cdn", config=plotly_config(filename))


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

        fig.add_trace(
            go.Scatter(
                x=zone.data[x_column],
                y=zone.data[z_column],
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


def build_single_layer_ice_shape_figure(participants, case_id: str, grid_level: str, slice_filter: float | None = None, bins_filter: str | None = None) -> tuple[go.Figure, int, list[float]]:
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

        label = participant_label(participant, dataset_data)
        color = participant_color(participant.participant_id)

        for zone_index, (zone_name, zone) in enumerate(dataset_data.ice_shape_data.zones.items(), start=1):
            zone_info = parse_ipw3_ice_shape_zone_name(zone_name)

            if zone_info is not None:
                bins_id = zone_info["bins"]
                shape_type = zone_info["type"]
                shape_role = zone_info["shape_role"]
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
                slice_text = "unknown"
                slice_position = None

            if shape_role != "SINGLE_LAYER":
                continue
            if bins_filter is not None and bins_id != bins_filter:
                continue

            if not slice_matches_filter(slice_position, slice_filter):
                continue
            if slice_position is not None:
                slice_positions.append(slice_position)

            x_iced_column = find_column_case_insensitive(zone.data.columns, ["X_ICED", "X_iced", "CoordinateX_Iced", "CoordinateX_iced"])
            z_iced_column = find_column_case_insensitive(zone.data.columns, ["Z_ICED", "Z_iced", "CoordinateZ_Iced", "CoordinateZ_iced"])
            if x_iced_column is None or z_iced_column is None:
                continue

            trace_name = label

            fig.add_trace(
                go.Scatter(
                    x=zone.data[x_iced_column],
                    y=zone.data[z_iced_column],
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


def build_multilayer_ice_shape_figure(participants, case_id: str, grid_level: str, slice_filter: float | None = None, bins_filter: str | None = None) -> tuple[go.Figure, int, list[float]]:
    fig = go.Figure()
    _, reference_slice_positions = add_clean_reference_trace(fig, case_id, slice_filter=slice_filter)
    trace_count = 0
    slice_positions: list[float] = list(reference_slice_positions)

    for participant, case_data, grid_data, dataset_data in iter_grid_data(participants, case_id, grid_level):
        if dataset_data.ice_shape_data is None:
            continue

        label = participant_label(participant, dataset_data)
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
                slice_text = "unknown"
                slice_position = None

            if shape_role == "SINGLE_LAYER":
                continue
            if bins_filter is not None and bins_id != bins_filter:
                continue

            if not slice_matches_filter(slice_position, slice_filter):
                continue
            if slice_position is not None:
                slice_positions.append(slice_position)

            x_iced_column = find_column_case_insensitive(zone.data.columns, ["X_ICED", "X_iced", "CoordinateX_Iced", "CoordinateX_iced"])
            z_iced_column = find_column_case_insensitive(zone.data.columns, ["Z_ICED", "Z_iced", "CoordinateZ_Iced", "CoordinateZ_iced"])

            if x_iced_column is None or z_iced_column is None:
                continue

            trace_name = label if layer_id.startswith("zone ") else f"{label} {layer_id}"
            fig.add_trace(go.Scatter(x=zone.data[x_iced_column], y=zone.data[z_iced_column], mode="lines", name=trace_name, legendgroup=label, line=dict(color=color), hovertemplate=(f"Participant: {escape(label)}<br>" f"Case: {escape(case_id)}<br>" f"Grid: {escape(grid_level)}<br>" f"Shape type: {escape(str(shape_type))}<br>" f"Shape role: {escape(str(shape_role or 'FINAL_LAYER/legacy'))}<br>" f"Bins: {escape(str(bins_id))}<br>" f"Slice: {escape(str(slice_text))}<br>" f"Layer/zone: {escape(str(layer_id))}<br>" f"NUM_LAYERS: {escape(str(num_layers))}<br>" f"DATA_TYPE: {escape(str(data_type))}<br>" f"Zone: {escape(zone_name)}<br>" f"{escape(x_iced_column)}=%{{x}}<br>" f"{escape(z_iced_column)}=%{{y}}<extra></extra>")))
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

    for slice_position in configured_slices:
        slice_title = f"Y = {slice_position:g} m" if slice_position is not None else "Slice unknown"
        slice_slug = f"_slice_{slice_position:g}".replace(".", "p") if slice_position is not None else "_slice_unknown"

        for bins_id in configured_bins:
            bins_slug = f"_{bins_id.lower()}"
            bin_title = f"{slice_title} | {bins_id}"

            single_fig, single_trace_count, single_slice_positions = build_single_layer_ice_shape_figure(participants, case_id, grid_level, slice_filter=slice_position, bins_filter=bins_id)
            if single_trace_count == 0:
                single_figure_html = empty_placeholder(
                    title=f"Single-layer ice shape | {bin_title}",
                    message="No matching SINGLE_LAYER iceShape zones were found yet for this case/grid level, slice, and bin set.",
                )
            else:
                single_filename = f"{slugify(case_id)}_{grid_level}_single_layer_ice_shape{slice_slug}{bins_slug}"
                single_figure_html = figure_to_html_div(single_fig, filename=single_filename)

            single_figures_html += f"""
            <section class="slice-plot-group">
              <h5>{escape(bin_title)}</h5>
              <div class="plot-container">
                {single_figure_html}
              </div>
            </section>
            """

            multi_fig, multi_trace_count, multi_slice_positions = build_multilayer_ice_shape_figure(participants, case_id, grid_level, slice_filter=slice_position, bins_filter=bins_id)
            if multi_trace_count == 0:
                multi_figure_html = empty_placeholder(
                    title=f"Multi-layer final ice shape | {bin_title}",
                    message="No matching FINAL_LAYER iceShape zones were found yet for this case/grid level, slice, and bin set.",
                )
            else:
                multi_filename = f"{slugify(case_id)}_{grid_level}_multilayer_ice_shape{slice_slug}{bins_slug}"
                multi_figure_html = figure_to_html_div(multi_fig, filename=multi_filename)

            multi_figures_html += f"""
            <section class="slice-plot-group">
              <h5>{escape(bin_title)}</h5>
              <div class="plot-container">
                {multi_figure_html}
              </div>
            </section>
            """

    configured_slice_text = format_slice_positions([value for value in configured_slices if value is not None])
    configured_bins_text = ", ".join(configured_bins)
    single_description = (
        "Single-layer ice-shape comparison extracted from finalIceShape / iceShape zones whose names contain SINGLE_LAYER. "
        "The clean reference shape is drawn from R00_REFERENCE, and submitted ice shapes are drawn using X_ICED-Z_ICED. "
        f"Slice location(s): {configured_slice_text}. "
        f"Bin set(s): {configured_bins_text}. "
        "Legend: PID.DID, bin set, and layer type."
    )

    multi_description = (
        "Multi-layer final ice-shape comparison extracted from finalIceShape / iceShape zones whose names contain FINAL_LAYER. "
        "Legacy zones without SINGLE_LAYER or FINAL_LAYER are kept in this final-layer plot. "
        "The clean reference shape is drawn from R00_REFERENCE. "
        f"Slice location(s): {configured_slice_text}. "
        f"Bin set(s): {configured_bins_text}. "
        "Legend: PID.DID and layer/zone index."
    )

    return f"""
    <section class="plot-subsection ice-shape-subsection">
      <h4>Single-layer ice shape</h4>
      <p class="plot-description">{escape(single_description)}</p>
      {single_figures_html}
    </section>

    <section class="plot-subsection ice-shape-subsection">
      <h4>Multi-layer final ice shape</h4>
      <p class="plot-description">{escape(multi_description)}</p>
      {multi_figures_html}
    </section>
    """
