from __future__ import annotations

from functools import lru_cache
from html import escape
from pathlib import Path
from typing import Any
import re

import pandas as pd
import plotly.graph_objects as go

from .gatherParticipantData import iter_case_data
from .participant_style import participant_color

GRID_CONVERGENCE_PLOTS: list[dict[str, Any]] = [
    {"plot_key": "cl_vs_n", "title": "CL grid convergence", "x_candidates": ["N"], "y_candidates": ["CL"], "x_label": "N<sup>-1/3</sup> [-]", "y_label": "CL [-]", "filename_slug": "cl_vs_n", "group_by_roughness": True},
    {"plot_key": "cd_vs_n", "title": "CD grid convergence", "x_candidates": ["N"], "y_candidates": ["CD"], "x_label": "N<sup>-1/3</sup> [-]", "y_label": "CD [-]", "filename_slug": "cd_vs_n", "group_by_roughness": True},
    {"plot_key": "cmy_vs_n", "title": "Pitching moment grid convergence", "x_candidates": ["N"], "y_candidates": ["CMY", "CMZ"], "x_label": "N<sup>-1/3</sup> [-]", "y_label": "Pitching moment coefficient [-]", "filename_slug": "cmy_vs_n", "group_by_roughness": True},

    {"plot_key": "water_mass_vs_n", "title": "Water mass grid convergence", "x_candidates": ["N"], "y_candidates": ["WATER_MASS", "WaterMass"], "x_label": "N<sup>-1/3</sup> [-]", "y_label": "Water mass [kg]", "filename_slug": "water_mass_vs_n", "combined_icing_plot": True},
    {"plot_key": "ice_mass_vs_n", "title": "Ice mass grid convergence", "x_candidates": ["N"], "y_candidates": ["ICE_MASS", "IceMass"], "x_label": "N<sup>-1/3</sup> [-]", "y_label": "Ice mass [kg]", "filename_slug": "ice_mass_vs_n", "combined_icing_plot": True},
    {"plot_key": "water_evap_mass_vs_n", "title": "Water evaporation mass grid convergence", "x_candidates": ["N"], "y_candidates": ["WATER_EVAP_MASS", "WaterEvapMass"], "x_label": "N<sup>-1/3</sup> [-]", "y_label": "Water evaporation mass [kg]", "filename_slug": "water_evap_mass_vs_n", "combined_icing_plot": True},
]

GRID_SPACING_COLUMN = "N_NEGATIVE_ONE_THIRD"


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


def bin_count_from_bin_set(bin_set: str) -> int | None:
    match = re.search(r"\d+", bin_set)
    if match is None:
        return None

    bin_count = int(match.group(0))
    return bin_count if bin_count > 0 else None


def is_required_grid_convergence_zone(zone) -> bool:
    requirement_column = find_column_case_insensitive(zone.data.columns, ["SOURCE_REQUIREMENT", "SourceRequirement"])
    if requirement_column is None:
        return True

    return any(str(value).strip().lower() == "required" for value in zone.data[requirement_column])

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
    return "N^(-1/3)" if x_column == GRID_SPACING_COLUMN else x_column


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
    if not grid_cell_counts:
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
        grid_spacing_values.append(num_cells ** (-1.0 / 3.0))

    working_data["GRID_LEVEL_DISPLAY"] = grid_levels
    working_data["GRID_CELL_COUNT"] = num_cells_values
    working_data[GRID_SPACING_COLUMN] = grid_spacing_values
    return working_data.dropna(subset=[GRID_SPACING_COLUMN])


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


def style_xy_figure(fig: go.Figure, x_label: str, y_label: str, height: int = 520) -> go.Figure:
    x_title = f"{x_label} <br><span style='font-size:14px'>&lt;- Finer (more cells)&nbsp;&nbsp;|&nbsp;&nbsp;Coarser (fewer cells) -&gt;</span>"
    fig.update_layout(
        font=dict(family="Arial, Helvetica, sans-serif", size=16),
        autosize=True,
        height=height,
        title=None,
        showlegend=True,
        xaxis=dict(title=dict(text=x_title, font=dict(size=18)), type="log", tickformat=".2e", exponentformat="power", showexponent="all", ticks="outside", showline=True, linecolor="black", linewidth=2, mirror=True, showgrid=True, gridcolor="lightgray", zeroline=False),
        yaxis=dict(title=dict(text=y_label, font=dict(size=18)), ticks="outside", showline=True, linecolor="black", linewidth=2, mirror=True, showgrid=True, gridcolor="lightgray", zeroline=False),
        legend=dict(orientation="v", x=1.02, xanchor="left", y=1.0, yanchor="top"),
        margin=dict(l=90, r=220, t=30, b=95),
        plot_bgcolor="white",
        paper_bgcolor="white",
    )
    return fig


def style_inverse_bin_figure(fig: go.Figure, y_label: str, height: int = 520) -> go.Figure:
    fig.update_layout(
        font=dict(family="Arial, Helvetica, sans-serif", size=16),
        autosize=True,
        height=height,
        title=None,
        showlegend=True,
        xaxis=dict(title=dict(text="1 / number of bins [-]", font=dict(size=18)), type="log", tickformat=".3g", ticks="outside", showline=True, linecolor="black", linewidth=2, mirror=True, showgrid=True, gridcolor="lightgray", zeroline=False),
        yaxis=dict(title=dict(text=y_label, font=dict(size=18)), ticks="outside", showline=True, linecolor="black", linewidth=2, mirror=True, showgrid=True, gridcolor="lightgray", zeroline=False),
        legend=dict(orientation="v", x=1.02, xanchor="left", y=1.0, yanchor="top"),
        margin=dict(l=90, r=220, t=30, b=95),
        plot_bgcolor="white",
        paper_bgcolor="white",
    )
    return fig


def build_grid_convergence_figure(participants, case_id: str, plot_spec: dict[str, Any], roughness_filter: str | None = None) -> tuple[go.Figure, int, list[str]]:
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
            zone_is_diameter = "by_diameter" in zone_name.lower()
            if roughness_filter is not None:
                roughness_key = extract_roughness_key_from_zone_name(zone_name)
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
                    trace_name = f"{label}{bin_set} D={diameter:g} um"
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
                            line=dict(color=color),
                            marker=dict(color=color),
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
                    name=label,
                    legendgroup=label,
                    line=dict(color=color),
                    marker=dict(color=color),
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

    style_xy_figure(fig, plot_spec["x_label"], plot_spec["y_label"])
    return fig, trace_count, skipped_notes


def build_grid_convergence_plot_subsection(participants, case_id: str, plot_spec: dict[str, Any]) -> str:
    if plot_spec.get("diameter_plot", False):
        return build_grid_convergence_diameter_subsection(participants, case_id, plot_spec)

    if plot_spec.get("combined_icing_plot", False):
        return build_combined_icing_subsection(participants, case_id, plot_spec)

    if plot_spec.get("group_by_roughness", False):
        return build_grid_convergence_roughness_subsection(participants, case_id, plot_spec)

    fig, trace_count, skipped_notes = build_grid_convergence_figure(participants, case_id, plot_spec)

    if trace_count == 0:
        figure_html = empty_placeholder(title=plot_spec["title"], message="No matching gridConvergence variables were found yet for this case.")
    else:
        filename = f"{slugify(case_id)}_{plot_spec['filename_slug']}"
        figure_html = figure_to_html_div(fig, filename=filename)

    description = "Grid-convergence data from the case-level gridConvergence file. Missing values equal to -999 are ignored. Legend: Participant ID."

    notes_html = ""
    if skipped_notes:
        notes_html = '<ul class="plot-notes">' + "".join(f"<li>{escape(note)}</li>" for note in sorted(set(skipped_notes))) + "</ul>"

    return f"""
    <section class="plot-subsection" data-variable-key="{escape(plot_spec['plot_key'])}" data-variable-label="{escape(plot_spec['title'])}">
      <h4>{escape(plot_spec["title"])}</h4>
      <p class="plot-description">{escape(description)}</p>
      {notes_html}
      <div class="plot-container">
        {figure_html}
      </div>
    </section>
    """


def build_grid_convergence_section(participants, case_id: str) -> str:
    html = ""

    for plot_spec in GRID_CONVERGENCE_PLOTS:
        html += build_grid_convergence_plot_subsection(participants, case_id, plot_spec)

    return f"""
    <section class="plot-subsection grid-convergence-section plot-filter-scope">
      <h3>Grid convergence</h3>
      <div class="variable-filter-controls" data-filter-title="Grid-convergence variables"></div>
      {html}
    </section>
    """

def collect_diameter_groups(participants, case_id: str, plot_spec: dict[str, Any]) -> list[tuple[str, float, float]]:
    groups: set[tuple[str, float, float]] = set()

    for participant, case_data in iter_case_data(participants, case_id):
        if case_data.grid_convergence_data is None:
            continue

        for zone_name, zone in case_data.grid_convergence_data.zones.items():
            if "by_diameter" not in zone_name.lower():
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

                groups.add((bin_set, float(bin_number), float(diameter)))

    return sorted(groups, key=lambda item: (item[0], item[1], item[2]))

def build_grid_convergence_diameter_figure(participants, case_id: str, plot_spec: dict[str, Any], target_bin_set: str, target_bin_number: float, target_diameter: float) -> tuple[go.Figure, int, list[str]]:
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

            fig.add_trace(
                go.Scatter(
                    x=data[GRID_SPACING_COLUMN],
                    y=data[y_column],
                    mode="lines+markers",
                    name=label,
                    legendgroup=label,
                    line=dict(color=color),
                    marker=dict(color=color),
                    customdata=data[["GRID_LEVEL_DISPLAY", "GRID_CELL_COUNT"]],
                    hovertemplate=(
                        f"Participant: {escape(label)}<br>"
                        f"Case: {escape(case_id)}<br>"
                        f"Bin set: {escape(target_bin_set)}<br>"
                        f"Bin: {target_bin_number:g}<br>"
                        f"Diameter: {target_diameter:g} μm<br>"
                        f"Zone: {escape(zone_name)}<br>"
                        f"{escape(format_x_hover_label(GRID_SPACING_COLUMN))}=%{{x:.6g}}<br>"
                        f"{escape(y_column)}=%{{y}}<br>"
                        "Grid level=%{customdata[0]}<br>"
                        "Num cells=%{customdata[1]:,.0f}<extra></extra>"
                    ),
                )
            )

            trace_count += 1

    style_xy_figure(fig, plot_spec["x_label"], plot_spec["y_label"])
    return fig, trace_count, skipped_notes

def build_grid_convergence_diameter_subsection(participants, case_id: str, plot_spec: dict[str, Any]) -> str:
    groups = collect_diameter_groups(participants, case_id, plot_spec)

    if not groups:
        return f"""
        <section class="plot-subsection" data-variable-key="{escape(plot_spec['plot_key'])}" data-variable-label="{escape(plot_spec['title'])}">
          <h4>{escape(plot_spec["title"])}</h4>
          {empty_placeholder(title=plot_spec["title"], message="No diameter-resolved grid-convergence data were found yet for this case.")}
        </section>
        """

    html = ""

    for bin_set, bin_number, diameter in groups:
        fig, trace_count, skipped_notes = build_grid_convergence_diameter_figure(participants, case_id, plot_spec, bin_set, bin_number, diameter)

        title = f"{plot_spec['title']} | {bin_set} | Diameter {bin_number:g} | D = {diameter:g} μm"

        if trace_count == 0:
            figure_html = empty_placeholder(title=title, message="No matching values were found for this diameter.")
        else:
            filename = f"{slugify(case_id)}_{plot_spec['filename_slug']}_{slugify(bin_set)}_diameter_{bin_number:g}_{diameter:g}".replace(".", "p")
            figure_html = figure_to_html_div(fig, filename=filename)

        html += f"""
        <section class="slice-plot-group">
          <h5>{escape(title)}</h5>
          <div class="plot-container">
            {figure_html}
          </div>
        </section>
        """

    return f"""
    <section class="plot-subsection" data-variable-key="{escape(plot_spec['plot_key'])}" data-variable-label="{escape(plot_spec['title'])}">
      <h4>{escape(plot_spec["title"])}</h4>
      <p class="plot-description">
        Diameter-resolved grid-convergence data. Each figure corresponds to one bin set and one droplet diameter. Legend: Participant ID.
      </p>
      {html}
    </section>
    """

def collect_cfd_roughness_keys(participants, case_id: str, plot_spec: dict[str, Any]) -> list[str]:
    roughness_keys: set[str] = set()

    for participant, case_data in iter_case_data(participants, case_id):
        if case_data.grid_convergence_data is None:
            continue

        for zone_name, zone in case_data.grid_convergence_data.zones.items():
            if "by_diameter" in zone_name.lower():
                continue

            roughness_key = extract_roughness_key_from_zone_name(zone_name)
            if roughness_key is None:
                continue

            x_column = find_column_case_insensitive(zone.data.columns, plot_spec["x_candidates"])
            y_column = find_column_case_insensitive(zone.data.columns, case_ordered_y_candidates(case_id, plot_spec["y_candidates"]))

            if x_column is not None and y_column is not None:
                roughness_keys.add(roughness_key)

    preferred_order = ["smooth", "0.5mm", "1mm", "1.5mm", "variable_roughness"]
    return [key for key in preferred_order if key in roughness_keys]

def collect_combined_icing_grid_levels(participants, case_id: str, plot_spec: dict[str, Any]) -> list[str]:
    grid_levels: set[str] = set()

    for participant, case_data in iter_case_data(participants, case_id):
        if case_data.grid_convergence_data is None:
            continue

        for zone_name, zone in case_data.grid_convergence_data.zones.items():
            if "by_diameter" in zone_name.lower():
                continue
            if not is_required_grid_convergence_zone(zone):
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


def build_combined_icing_figure(participants, case_id: str, plot_spec: dict[str, Any], target_grid_level: str) -> tuple[go.Figure, int, list[str]]:
    fig = go.Figure()
    trace_count = 0
    skipped_notes: list[str] = []
    seen_trace_keys: set[tuple[str, str, str]] = set()

    for participant, case_data in iter_case_data(participants, case_id):
        if case_data.grid_convergence_data is None:
            continue

        label = participant_label(participant)
        color = participant_color(participant.participant_id)
        trace_rows: list[dict[str, Any]] = []

        for zone_name, zone in case_data.grid_convergence_data.zones.items():
            if "by_diameter" in zone_name.lower():
                continue
            if not is_required_grid_convergence_zone(zone):
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

                trace_rows.append({
                    "inverse_bin_count": 1.0 / bin_count,
                    "bin_set": bin_set,
                    "bin_count": bin_count,
                    "y": y_value,
                    "zone_name": zone_name,
                })

        if not trace_rows:
            continue

        trace_key = (participant.participant_id, plot_spec["plot_key"], target_grid_level)
        if trace_key in seen_trace_keys:
            continue

        seen_trace_keys.add(trace_key)
        trace_rows = sorted(trace_rows, key=lambda item: item["inverse_bin_count"])
        customdata = [[row["bin_set"], row["bin_count"], row["zone_name"]] for row in trace_rows]

        fig.add_trace(
            go.Scatter(
                x=[row["inverse_bin_count"] for row in trace_rows],
                y=[row["y"] for row in trace_rows],
                mode="lines+markers",
                name=label,
                legendgroup=label,
                line=dict(color=color),
                marker=dict(color=color),
                customdata=customdata,
                hovertemplate=(
                    f"Participant: {escape(label)}<br>"
                    f"Case: {escape(case_id)}<br>"
                    f"Grid level: {escape(target_grid_level)}<br>"
                    "Bin set: %{customdata[0]}<br>"
                    "Number of bins: %{customdata[1]}<br>"
                    "Zone: %{customdata[2]}<br>"
                    "1 / number of bins=%{x:.6g}<br>"
                    f"{escape(y_column)}=%{{y}}<extra></extra>"
                ),
            )
        )

        trace_count += 1

    style_inverse_bin_figure(fig, plot_spec["y_label"])
    return fig, trace_count, skipped_notes


def build_combined_icing_subsection(participants, case_id: str, plot_spec: dict[str, Any]) -> str:
    grid_levels = collect_combined_icing_grid_levels(participants, case_id, plot_spec)

    if not grid_levels:
        return f"""
        <section class="plot-subsection" data-variable-key="{escape(plot_spec['plot_key'])}" data-variable-label="{escape(plot_spec['title'])}">
          <h4>{escape(plot_spec["title"])}</h4>
          {empty_placeholder(title=plot_spec["title"], message="No combined icing grid-convergence data were found yet for this case.")}
        </section>
        """

    figures_html = ""

    for grid_level in grid_levels:
        fig, trace_count, skipped_notes = build_combined_icing_figure(participants, case_id, plot_spec, grid_level)

        title = f"{plot_spec['title']} | {grid_level}"

        if trace_count == 0:
            figure_html = empty_placeholder(title=title, message="No matching combined values were found for this grid level.")
        else:
            filename = f"{slugify(case_id)}_{plot_spec['filename_slug']}_{slugify(grid_level)}_vs_inverse_bins"
            figure_html = figure_to_html_div(fig, filename=filename)

        figures_html += f"""
        <section class="slice-plot-group">
          <h5>{escape(title)}</h5>
          <div class="plot-container">
            {figure_html}
          </div>
        </section>
        """

    return f"""
    <section class="plot-subsection" data-variable-key="{escape(plot_spec['plot_key'])}" data-variable-label="{escape(plot_spec['title'])}">
      <h4>{escape(plot_spec["title"])}</h4>
      <p class="plot-description">
        Combined icing data from required sheets plotted against 1 / number of bins. Each figure corresponds to one grid level. Missing values equal to -999 are ignored. Legend: Participant ID.
      </p>
      {figures_html}
    </section>
    """

def build_grid_convergence_roughness_subsection(participants, case_id: str, plot_spec: dict[str, Any]) -> str:
    roughness_keys = collect_cfd_roughness_keys(participants, case_id, plot_spec)

    if not roughness_keys:
        return f"""
        <section class="plot-subsection" data-variable-key="{escape(plot_spec['plot_key'])}" data-variable-label="{escape(plot_spec['title'])}">
          <h4>{escape(plot_spec["title"])}</h4>
          {empty_placeholder(title=plot_spec["title"], message="No matching CFD gridConvergence variables were found yet for this case.")}
        </section>
        """

    figures_html = ""

    for roughness_key in roughness_keys:
        fig, trace_count, skipped_notes = build_grid_convergence_figure(participants, case_id, plot_spec, roughness_filter=roughness_key)

        roughness_title = format_roughness_title(roughness_key)

        if trace_count == 0:
            figure_html = empty_placeholder(title=roughness_title, message="No matching values were found for this roughness.")
        else:
            filename = f"{slugify(case_id)}_{plot_spec['filename_slug']}_{slugify(roughness_key)}"
            figure_html = figure_to_html_div(fig, filename=filename)

        figures_html += f"""
        <section class="slice-plot-group">
          <h5>{escape(roughness_title)}</h5>
          <div class="plot-container">
            {figure_html}
          </div>
        </section>
        """

    return f"""
    <section class="plot-subsection" data-variable-key="{escape(plot_spec['plot_key'])}" data-variable-label="{escape(plot_spec['title'])}">
      <h4>{escape(plot_spec["title"])}</h4>
      <p class="plot-description">
        Grid-convergence data grouped by roughness condition. Missing values equal to -999 are ignored. Legend: Participant ID.
      </p>
      {figures_html}
    </section>
    """
