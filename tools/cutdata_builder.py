from __future__ import annotations

from pathlib import Path
from html import escape
from dataclasses import dataclass
from typing import Any
import re

import plotly.graph_objects as go

from .gatherParticipantData import CASE_SLICES, decode_slice_position, iter_grid_datasets

SAVE_IMAGE_PREVIEWS = False
IMAGE_PREVIEW_ROOT = Path("IMAGES_PREVIEW")
BETA_BINS = ["BINS03", "BINS07", "BINS15"]

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
    },
    {
        "plot_key": "htc_vs_s",
        "title": "HTC vs s",
        "description": "Heat-transfer coefficient along the selected surface cut(s).",
        "x_candidates": ["s", "S"],
        "y_candidates": ["HTC", "HeatTransferCoefficient"],
        "x_label": "Surface distance from highlight [m]",
        "y_label": "Convective Heat Transfer [W/m2K]",
        "filename_slug": "htc_vs_s",
        "bins_filter": None,
    },
    {
        "plot_key": "htc_clean_vs_s",
        "title": "HTC vs s W/O Roughness",
        "description": "Heat-transfer coefficient along the selected surface cut(s) with no roughness applied.",
        "x_candidates": ["s", "S"],
        "y_candidates": ["HTC_CLEAN", "HTC_Clean", "HTC_clean"],
        "x_label": "Surface distance from highlight [m]",
        "y_label": "Convective Heat Transfer [W/m2K]",
        "filename_slug": "htc_clean_vs_s",
        "bins_filter": None,
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


def parse_ipw3_zone_name(zone_name: str) -> dict[str, str] | None:
    pattern = re.compile(r"^SLICE_Y_(?P<slice>.+?)_(?P<bins>BINS\d+)(?:_(?P<tail>.*))?$", re.IGNORECASE)
    match = pattern.match(zone_name.strip())
    if match is None:
        return None

    tail = match.group("tail") or ""
    dataset_match = re.search(r"(?:^|_)(D\d+)(?:_|$)", tail, re.IGNORECASE)
    dataset_id = dataset_match.group(1).upper() if dataset_match is not None else "DXX"
    return {"slice": match.group("slice"), "bins": match.group("bins").upper(), "dataset": dataset_id}


def format_slice_positions(slice_values: list[float]) -> str:
    if not slice_values:
        return "No slice location detected."
    unique_values = sorted(set(round(value, 8) for value in slice_values))
    return ", ".join(f"Y = {value:g} m" for value in unique_values)


def style_xy_figure(fig: go.Figure, x_label: str, y_label: str, height: int = 560, legend_right: bool = True) -> go.Figure:
    if legend_right:
        legend = dict(orientation="v", x=1.02, xanchor="left", y=1.0, yanchor="top")
        margin = dict(l=90, r=220, t=30, b=80)
    else:
        legend = dict(orientation="h", x=0.0, xanchor="left", y=1.12, yanchor="bottom")
        margin = dict(l=90, r=40, t=70, b=80)

    fig.update_layout(
        font=dict(family="Arial, Helvetica, sans-serif", size=16),
        autosize=True,
        height=height,
        title=None,
        showlegend=True,
        xaxis=dict(title=dict(text=x_label, font=dict(size=18)), ticks="outside", showline=True, linecolor="black", linewidth=2, mirror=True, showgrid=True, gridcolor="lightgray", zeroline=False),
        yaxis=dict(title=dict(text=y_label, font=dict(size=18)), ticks="outside", showline=True, linecolor="black", linewidth=2, mirror=True, showgrid=True, gridcolor="lightgray", zeroline=False),
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


def build_cutdata_figure(participants, case_id: str, grid_level: str, plot_spec: dict[str, Any], slice_filter: float | None = None) -> tuple[go.Figure, int, list[float], list[str]]:
    fig = go.Figure()
    trace_count = 0
    slice_positions: list[float] = []
    skipped_notes: list[str] = []
    skipped_note_set: set[str] = set()
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
            if slice_position is not None:
                slice_positions.append(slice_position)
            x_column = find_column_case_insensitive(zone.data.columns, plot_spec["x_candidates"])
            y_column = find_column_case_insensitive(zone.data.columns, plot_spec["y_candidates"])
            if x_column is None or y_column is None:
                skipped_note_set.add(f"Participant ID {participant.participant_id} did not provide {plot_spec['y_candidates'][0]}.")
                continue
            data = zone.data[[x_column, y_column]].copy()
            data = data[(data[x_column] != -999.0) & (data[y_column] != -999.0)]
            if data.empty:
                skipped_note_set.add(f"Participant ID {participant.participant_id} did not provide valid {plot_spec['y_candidates'][0]} values.")
                continue
            trace_name = participant_label(participant, dataset_data)
            slice_text = "unknown"
            if zone_info is not None:
                slice_value = decode_slice_position(zone_info["slice"])
                if slice_value is not None:
                    slice_text = f"Y = {slice_value:g} m"
            fig.add_trace(go.Scatter(x=data[x_column], y=data[y_column], mode="lines", name=trace_name, legendgroup=trace_name, hovertemplate=(f"Participant: {escape(trace_name)}<br>" f"Case: {escape(case_id)}<br>" f"Grid: {escape(grid_level)}<br>" f"Bins: {escape(bins_id or 'not specified')}<br>" f"Slice: {escape(slice_text)}<br>" f"Zone: {escape(zone_name)}<br>" f"{escape(x_column)}=%{{x}}<br>" f"{escape(y_column)}=%{{y}}<extra></extra>")))
            trace_count += 1

    trace_count += add_reference_traces(fig, case_id, grid_level, plot_spec["plot_key"])
    style_xy_figure(fig, plot_spec["x_label"], plot_spec["y_label"])
    skipped_notes = sorted(skipped_note_set)
    return fig, trace_count, slice_positions, skipped_notes


def build_plot_description(plot_spec: dict[str, Any], slice_positions: list[float]) -> str:
    slices_text = format_slice_positions(slice_positions)
    bins_filter = plot_spec.get("bins_filter")
    details = [plot_spec["description"]]
    if bins_filter is not None:
        details.append(f"Distribution: {bins_filter}.")
    details.append(f"Slice location(s): {slices_text}.")
    details.append("Legend: PID.DID.")
    return " ".join(details)


def build_participant_combined_beta_figure(participant, dataset_data, case_id: str, grid_level: str, grid_data, slice_filter: float | None = None) -> tuple[str, int, list[float]]:
    fig = go.Figure()
    trace_count = 0
    slice_positions: list[float] = []
    label = participant_label(participant, dataset_data)
    grouped_zones = get_cutdata_zones_by_bins(dataset_data)

    for bins_id in BETA_BINS:
        for zone_name, zone, slice_position in grouped_zones[bins_id]:
            if not slice_matches_filter(slice_position, slice_filter):
                continue
            x_column = find_column_case_insensitive(zone.data.columns, ["s", "S"])
            y_column = find_column_case_insensitive(zone.data.columns, ["Beta", "BETA", "CollectionEfficiency"])
            if x_column is None or y_column is None:
                continue
            data = zone.data[[x_column, y_column]].copy()
            data = data[(data[x_column] != -999.0) & (data[y_column] != -999.0)]
            if data.empty:
                continue
            if slice_position is not None:
                slice_positions.append(slice_position)
            slice_text = f"Y = {slice_position:g} m" if slice_position is not None else "unknown"
            fig.add_trace(go.Scatter(x=data[x_column], y=data[y_column], mode="lines", name=label, legendgroup=label, showlegend=(trace_count == 0), hovertemplate=(f"Participant: {escape(label)}<br>" f"Case: {escape(case_id)}<br>" f"Grid: {escape(grid_level)}<br>" f"Bins: {escape(bins_id)}<br>" f"Slice: {escape(slice_text)}<br>" f"Zone: {escape(zone_name)}<br>" f"{escape(x_column)}=%{{x}}<br>" f"{escape(y_column)}=%{{y}}<extra></extra>")))
            trace_count += 1

    if trace_count == 0:
        return "", 0, slice_positions
    style_xy_figure(fig, "s [m]", "Beta [-]", height=420, legend_right=False)
    fig.update_layout(margin=dict(l=70, r=25, t=20, b=60), legend=dict(orientation="h", x=0.0, y=1.12))
    slice_slug = f"_slice_{slice_filter:g}".replace(".", "p") if slice_filter is not None else ""
    filename = f"{slugify(case_id)}_{grid_level}_{participant.participant_id}_{dataset_data.dataset_id}{slice_slug}_combined_beta"
    return figure_to_html_div(fig, filename=filename), trace_count, slice_positions


def build_combined_beta_card(participant, dataset_data, case_id: str, grid_level: str, grid_data, slice_filter: float | None = None) -> str:
    participant_id = participant_label(participant, dataset_data)
    figure_html, trace_count, slice_positions = build_participant_combined_beta_figure(participant, dataset_data, case_id, grid_level, grid_data, slice_filter=slice_filter)
    if trace_count == 0:
        return ""
    slice_title = f" | Y = {slice_filter:g} m" if slice_filter is not None else ""
    details = f"Legend: {participant_id}. Case: {case_id}. Grid level: {grid_level}. Slice location(s): {format_slice_positions(slice_positions)}. Curves included when available: BINS03, BINS07, and BINS15."
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
    <section class="plot-subsection combined-beta-section">
      <h4>Combined Beta by participant</h4>
      <p class="plot-description">
        Each card corresponds to one participant dataset. Inside each card, BINS03, BINS07, and BINS15 are overlaid on the same Beta vs s figure. Participant cards are arranged three per row when space allows. Legends use only PID.DID.
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
        fig, trace_count, figure_slice_positions, skipped_notes = build_cutdata_figure(participants, case_id, grid_level, plot_spec, slice_filter=slice_position)
        all_skipped_notes.extend(skipped_notes)
        slice_title = f"Y = {slice_position:g} m" if slice_position is not None else "Slice unknown"
        if trace_count == 0:
            figure_html = empty_placeholder(title=f"{plot_spec['title']} | {slice_title}", message="No matching cutData variables were found yet for this case/grid level and slice.")
        else:
            slice_slug = f"_slice_{slice_position:g}".replace(".", "p") if slice_position is not None else "_slice_unknown"
            filename = f"{slugify(case_id)}_{grid_level}_{plot_spec['filename_slug']}{slice_slug}"
            figure_html = figure_to_html_div(fig, filename=filename)
        figures_html += f"""
        <section class="slice-plot-group">
          <h5>{escape(slice_title)}</h5>
          <div class="plot-container">
            {figure_html}
          </div>
        </section>
        """

    description = build_plot_description(plot_spec, [value for value in slice_positions if value is not None])
    notes_html = ""
    if all_skipped_notes:
        notes_html = '<ul class="plot-notes">' + "".join(f"<li>{escape(note)}</li>" for note in sorted(set(all_skipped_notes))) + "</ul>"
    return f"""
    <section class="plot-subsection">
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
            html += build_combined_beta_section(participants, case_id, grid_level)
        else:
            html += build_plot_subsection(participants, case_id, grid_level, plot_spec)
    return html
