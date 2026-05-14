from __future__ import annotations

from html import escape
from typing import Any
import re

import plotly.graph_objects as go

from .gatherParticipantData import CASE_SLICES, decode_slice_position, iter_grid_datasets


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


def style_xy_figure(fig: go.Figure, x_label: str, y_label: str, height: int = 650) -> go.Figure:
    fig.update_layout(
        font=dict(family="Arial, Helvetica, sans-serif", size=16),
        autosize=True,
        height=height,
        title=None,
        showlegend=True,
        xaxis=dict(title=dict(text=x_label, font=dict(size=18)), ticks="outside", showline=True, linecolor="black", linewidth=2, mirror=True, showgrid=True, gridcolor="lightgray", zeroline=False),
        yaxis=dict(title=dict(text=y_label, font=dict(size=18)), ticks="outside", showline=True, linecolor="black", linewidth=2, mirror=True, showgrid=True, gridcolor="lightgray", zeroline=False),
        legend=dict(orientation="v", x=1.02, xanchor="left", y=1.0, yanchor="top"),
        margin=dict(l=90, r=260, t=30, b=80),
        plot_bgcolor="white",
        paper_bgcolor="white",
    )
    return fig


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
        return {
            "type": "SLICE",
            "slice": match.group("slice"),
            "bins": (match.group("bins") or "BINSXX").upper(),
            "dataset": dataset_match.group(1).upper() if dataset_match is not None else "DXX",
            "layer": layer_match.group(1).upper() if layer_match is not None else "",
        }

    match = mccs_pattern.match(zone_name)
    if match is not None:
        tail = match.group("tail") or ""
        dataset_match = re.search(r"(?:^|_)(D\d+)(?:_|$)", tail, re.IGNORECASE)
        layer_match = re.search(r"(?:^|_)(L\d+|LAYER\d+)(?:_|$)", tail, re.IGNORECASE)
        return {
            "type": "MCCS",
            "slice": "",
            "bins": (match.group("bins") or "BINSXX").upper(),
            "dataset": dataset_match.group(1).upper() if dataset_match is not None else "DXX",
            "layer": layer_match.group(1).upper() if layer_match is not None else "",
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


def build_single_layer_ice_shape_figure(participants, case_id: str, grid_level: str, slice_filter: float | None = None) -> tuple[go.Figure, int, list[float]]:
    """Build the single-layer/final ice-shape figure from cutData files.

    Important distinction:
        - cutData is used here only to draw the iced shape associated with the
          cutData variables.
        - finalIceShape / iceShape is used separately for the multi-layer shape
          history.

    The plot is always X-Z:
        clean shape: X vs Z
        iced shape : X_ICED vs Z_ICED
    """

    fig = go.Figure()
    trace_count = 0
    slice_positions: list[float] = []
    clean_legend_groups: set[str] = set()

    for participant, case_data, grid_data, dataset_data in iter_grid_data(participants, case_id, grid_level):
        # Single-layer ice shape is extracted from cutData.
        if dataset_data.cut_data is None:
            continue

        label = participant_label(participant, dataset_data)

        for zone_index, (zone_name, cut_zone) in enumerate(dataset_data.cut_data.zones.items(), start=1):
            # Reuse the cutData zone-name parser if the naming is compatible.
            # If it is not compatible, the code still plots the data and only
            # reports unknown metadata in the hover.
            zone_info = parse_ipw3_ice_shape_zone_name(zone_name)

            if zone_info is not None:
                bins_id = zone_info["bins"]
                shape_type = zone_info["type"]
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
                shape_type = "cutData"
                slice_text = "unknown"
                slice_position = None

            if not slice_matches_filter(slice_position, slice_filter):
                continue
            if slice_position is not None:
                slice_positions.append(slice_position)

            # Clean coordinates from cutData.
            x_clean_column = find_column_case_insensitive(cut_zone.data.columns, ["X", "CoordinateX"])
            z_clean_column = find_column_case_insensitive(cut_zone.data.columns, ["Z", "CoordinateZ"])

            # Iced coordinates from cutData.
            x_iced_column = find_column_case_insensitive(cut_zone.data.columns, ["X_ICED", "X_iced", "CoordinateX_Iced", "CoordinateX_iced"])
            z_iced_column = find_column_case_insensitive(cut_zone.data.columns, ["Z_ICED", "Z_iced", "CoordinateZ_Iced", "CoordinateZ_iced"])

            # If the cutData file does not contain iced coordinates, there is
            # no single-layer ice-shape curve to draw.
            if x_iced_column is None or z_iced_column is None:
                continue

            # Draw the clean shape once per participant/dataset.
            clean_group = f"{label}_clean_cutdata_{slice_filter}"

            if x_clean_column is not None and z_clean_column is not None and clean_group not in clean_legend_groups:
                fig.add_trace(
                    go.Scatter(
                        x=cut_zone.data[x_clean_column],
                        y=cut_zone.data[z_clean_column],
                        mode="lines",
                        name=f"{label} clean",
                        legendgroup=clean_group,
                        line=dict(dash="dash"),
                        hovertemplate=(
                            f"Participant: {escape(label)}<br>"
                            f"Case: {escape(case_id)}<br>"
                            f"Grid: {escape(grid_level)}<br>"
                            f"Source: cutData<br>"
                            f"Shape: clean<br>"
                            f"Zone: {escape(zone_name)}<br>"
                            f"{escape(x_clean_column)}=%{{x}}<br>"
                            f"{escape(z_clean_column)}=%{{y}}<extra></extra>"
                        ),
                    )
                )

                clean_legend_groups.add(clean_group)
                trace_count += 1

            # Draw the iced shape from cutData.
            trace_name = f"{label} single-layer"

            fig.add_trace(
                go.Scatter(
                    x=cut_zone.data[x_iced_column],
                    y=cut_zone.data[z_iced_column],
                    mode="lines",
                    name=trace_name,
                    legendgroup=label,
                    hovertemplate=(
                        f"Participant: {escape(label)}<br>"
                        f"Case: {escape(case_id)}<br>"
                        f"Grid: {escape(grid_level)}<br>"
                        f"Source: cutData<br>"
                        f"Shape type: {escape(str(shape_type))}<br>"
                        f"Bins: {escape(str(bins_id))}<br>"
                        f"Slice: {escape(str(slice_text))}<br>"
                        f"Zone: {escape(zone_name)}<br>"
                        f"{escape(x_iced_column)}=%{{x}}<br>"
                        f"{escape(z_iced_column)}=%{{y}}<extra></extra>"
                    ),
                )
            )

            trace_count += 1

    style_xy_figure(fig, "X [m]", "Z [m]")
    fig.update_yaxes(scaleanchor="x", scaleratio=1.0)

    return fig, trace_count, slice_positions


def build_multilayer_ice_shape_figure(participants, case_id: str, grid_level: str, slice_filter: float | None = None) -> tuple[go.Figure, int, list[float]]:
    fig = go.Figure()
    trace_count = 0
    slice_positions: list[float] = []
    clean_legend_groups: set[str] = set()

    for participant, case_data, grid_data, dataset_data in iter_grid_data(participants, case_id, grid_level):
        if dataset_data.ice_shape_data is None:
            continue

        label = participant_label(participant, dataset_data)
        num_layers = dataset_data.ice_shape_data.auxdata.get("NUM_LAYERS", "unknown")
        data_type = dataset_data.ice_shape_data.auxdata.get("DATA_TYPE", "unknown")

        for zone_index, (zone_name, zone) in enumerate(dataset_data.ice_shape_data.zones.items(), start=1):
            zone_info = parse_ipw3_ice_shape_zone_name(zone_name)

            if zone_info is not None:
                bins_id = zone_info["bins"]
                shape_type = zone_info["type"]
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
                layer_id = f"zone {zone_index}"
                slice_text = "unknown"
                slice_position = None

            if not slice_matches_filter(slice_position, slice_filter):
                continue
            if slice_position is not None:
                slice_positions.append(slice_position)

            x_clean_column = find_column_case_insensitive(zone.data.columns, ["X", "CoordinateX"])
            z_clean_column = find_column_case_insensitive(zone.data.columns, ["Z", "CoordinateZ"])
            x_iced_column = find_column_case_insensitive(zone.data.columns, ["X_ICED", "X_iced", "CoordinateX_Iced", "CoordinateX_iced"])
            z_iced_column = find_column_case_insensitive(zone.data.columns, ["Z_ICED", "Z_iced", "CoordinateZ_Iced", "CoordinateZ_iced"])

            if x_iced_column is None or z_iced_column is None:
                continue

            clean_group = f"{label}_clean_{slice_filter}"
            if x_clean_column is not None and z_clean_column is not None and clean_group not in clean_legend_groups:
                fig.add_trace(go.Scatter(x=zone.data[x_clean_column], y=zone.data[z_clean_column], mode="lines", name=f"{label} clean", legendgroup=clean_group, line=dict(dash="dash"), hovertemplate=(f"Participant: {escape(label)}<br>" f"Case: {escape(case_id)}<br>" f"Grid: {escape(grid_level)}<br>" f"Shape: clean<br>" f"Zone: {escape(zone_name)}<br>" f"{escape(x_clean_column)}=%{{x}}<br>" f"{escape(z_clean_column)}=%{{y}}<extra></extra>")))
                clean_legend_groups.add(clean_group)
                trace_count += 1

            trace_name = f"{label} {layer_id}"
            fig.add_trace(go.Scatter(x=zone.data[x_iced_column], y=zone.data[z_iced_column], mode="lines", name=trace_name, legendgroup=label, hovertemplate=(f"Participant: {escape(label)}<br>" f"Case: {escape(case_id)}<br>" f"Grid: {escape(grid_level)}<br>" f"Shape type: {escape(str(shape_type))}<br>" f"Bins: {escape(str(bins_id))}<br>" f"Slice: {escape(str(slice_text))}<br>" f"Layer/zone: {escape(str(layer_id))}<br>" f"NUM_LAYERS: {escape(str(num_layers))}<br>" f"DATA_TYPE: {escape(str(data_type))}<br>" f"Zone: {escape(zone_name)}<br>" f"{escape(x_iced_column)}=%{{x}}<br>" f"{escape(z_iced_column)}=%{{y}}<extra></extra>")))
            trace_count += 1

    style_xy_figure(fig, "X [m]", "Z [m]")
    fig.update_yaxes(scaleanchor="x", scaleratio=1.0)
    return fig, trace_count, slice_positions


def build_ice_shape_section(participants, case_id: str, grid_level: str) -> str:
    """Build both ice-shape figures for one case/grid level.

    This section contains:
        1. Single-layer ice shape from cutData.
        2. Multi-layer/final ice shape from finalIceShape / iceShape.
    """

    single_figures_html = ""
    multi_figures_html = ""
    configured_slices = expected_slice_positions(case_id)

    for slice_position in configured_slices:
        slice_title = f"Y = {slice_position:g} m" if slice_position is not None else "Slice unknown"
        slice_slug = f"_slice_{slice_position:g}".replace(".", "p") if slice_position is not None else "_slice_unknown"

        single_fig, single_trace_count, single_slice_positions = build_single_layer_ice_shape_figure(participants, case_id, grid_level, slice_filter=slice_position)
        if single_trace_count == 0:
            single_figure_html = empty_placeholder(
                title=f"Single-layer ice shape | {slice_title}",
                message="No matching cutData iced coordinates were found yet for this case/grid level and slice.",
            )
        else:
            single_filename = f"{slugify(case_id)}_{grid_level}_single_layer_ice_shape{slice_slug}"
            single_figure_html = figure_to_html_div(single_fig, filename=single_filename)

        single_figures_html += f"""
        <section class="slice-plot-group">
          <h5>{escape(slice_title)}</h5>
          <div class="plot-container">
            {single_figure_html}
          </div>
        </section>
        """

        multi_fig, multi_trace_count, multi_slice_positions = build_multilayer_ice_shape_figure(participants, case_id, grid_level, slice_filter=slice_position)
        if multi_trace_count == 0:
            multi_figure_html = empty_placeholder(
                title=f"Multi-layer final ice shape | {slice_title}",
                message="No matching finalIceShape variables were found yet for this case/grid level and slice.",
            )
        else:
            multi_filename = f"{slugify(case_id)}_{grid_level}_multilayer_ice_shape{slice_slug}"
            multi_figure_html = figure_to_html_div(multi_fig, filename=multi_filename)

        multi_figures_html += f"""
        <section class="slice-plot-group">
          <h5>{escape(slice_title)}</h5>
          <div class="plot-container">
            {multi_figure_html}
          </div>
        </section>
        """

    configured_slice_text = format_slice_positions([value for value in configured_slices if value is not None])
    single_description = (
        "Single-layer ice-shape comparison extracted from cutData. "
        "The clean shape is drawn using X-Z, and the iced shape is drawn using X_ICED-Z_ICED from the cutData file. "
        f"Slice location(s): {configured_slice_text}. "
        "Legend: PID.DID."
    )

    multi_description = (
        "Multi-layer or final ice-shape comparison extracted from finalIceShape / iceShape files. "
        "Each finalIceShape zone is drawn as one X-Z curve, so if one layer is stored per zone, "
        "the figure overlays all submitted layers. "
        f"Slice location(s): {configured_slice_text}. "
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
