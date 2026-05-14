from __future__ import annotations

from html import escape
from typing import Any
import re

import plotly.graph_objects as go

from .gatherParticipantData import iter_case_data
from .participant_style import participant_color

GRID_CONVERGENCE_PLOTS: list[dict[str, Any]] = [
    {"plot_key": "cl_vs_n", "title": "CL grid convergence", "x_candidates": ["N"], "y_candidates": ["CL"], "x_label": "N [-]", "y_label": "CL [-]", "filename_slug": "cl_vs_n"},
    {"plot_key": "cd_vs_n", "title": "CD grid convergence", "x_candidates": ["N"], "y_candidates": ["CD"], "x_label": "N [-]", "y_label": "CD [-]", "filename_slug": "cd_vs_n"},
    {"plot_key": "cmy_vs_n", "title": "CMY grid convergence", "x_candidates": ["N"], "y_candidates": ["CMY"], "x_label": "N [-]", "y_label": "CMY [-]", "filename_slug": "cmy_vs_n"},
    {"plot_key": "water_mass_vs_n", "title": "Water mass grid convergence", "x_candidates": ["N"], "y_candidates": ["WATER_MASS", "WaterMass"], "x_label": "N [-]", "y_label": "Water mass [kg]", "filename_slug": "water_mass_vs_n"},
    {"plot_key": "ice_mass_vs_n", "title": "Ice mass grid convergence", "x_candidates": ["N"], "y_candidates": ["ICE_MASS", "IceMass"], "x_label": "N [-]", "y_label": "Ice mass [kg]", "filename_slug": "ice_mass_vs_n"},
    {"plot_key": "water_evap_mass_vs_n", "title": "Water evaporation mass grid convergence", "x_candidates": ["N"], "y_candidates": ["WATER_EVAP_MASS", "WaterEvapMass"], "x_label": "N [-]", "y_label": "Water evaporation mass [kg]", "filename_slug": "water_evap_mass_vs_n"},
    {"plot_key": "water_mass_by_diameter_vs_n", "title": "Water mass grid convergence by diameter", "x_candidates": ["N"], "y_candidates": ["WATER_MASS", "WaterMass"], "x_label": "N [-]", "y_label": "Water mass [kg]", "filename_slug": "water_mass_by_diameter_vs_n", "diameter_plot": True},
    {"plot_key": "ice_mass_by_diameter_vs_n", "title": "Ice mass grid convergence by diameter", "x_candidates": ["N"], "y_candidates": ["ICE_MASS", "IceMass"], "x_label": "N [-]", "y_label": "Ice mass [kg]", "filename_slug": "ice_mass_by_diameter_vs_n", "diameter_plot": True},
    {"plot_key": "water_evap_mass_by_diameter_vs_n", "title": "Water evaporation mass grid convergence by diameter", "x_candidates": ["N"], "y_candidates": ["WATER_EVAP_MASS", "WaterEvapMass"], "x_label": "N [-]", "y_label": "Water evaporation mass [kg]", "filename_slug": "water_evap_mass_by_diameter_vs_n", "diameter_plot": True},
]

seen_trace_keys: set[tuple[str, str, str]] = set()

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


def participant_label(participant) -> str:
    """Return the legend label for a participant-level grid-convergence file.

    Grid convergence is stored under the test case, not under one grid level or
    dataset attempt. Therefore the legend uses only PID.
    """
    return f"{participant.participant_id}"


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


def style_xy_figure(fig: go.Figure, x_label: str, y_label: str, height: int = 520) -> go.Figure:
    fig.update_layout(
        font=dict(family="Arial, Helvetica, sans-serif", size=16),
        autosize=True,
        height=height,
        title=None,
        showlegend=True,
        xaxis=dict(title=dict(text=x_label, font=dict(size=18)), type="log", autorange="reversed", ticks="outside", showline=True, linecolor="black", linewidth=2, mirror=True, showgrid=True, gridcolor="lightgray", zeroline=False),
        yaxis=dict(title=dict(text=y_label, font=dict(size=18)), ticks="outside", showline=True, linecolor="black", linewidth=2, mirror=True, showgrid=True, gridcolor="lightgray", zeroline=False),
        legend=dict(orientation="v", x=1.02, xanchor="left", y=1.0, yanchor="top"),
        margin=dict(l=90, r=220, t=30, b=80),
        plot_bgcolor="white",
        paper_bgcolor="white",
    )
    return fig


def build_grid_convergence_figure(participants, case_id: str, plot_spec: dict[str, Any], roughness_filter: str | None = None) -> tuple[go.Figure, int, list[str]]:
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
            y_column = find_column_case_insensitive(zone.data.columns, plot_spec["y_candidates"])
            if x_column is None or y_column is None:
                continue
            participant_had_variable = True

            data = zone.data[[x_column, y_column]].copy()
            data = data[(data[x_column] > 0.0) & (data[x_column] != -999.0) & (data[y_column] != -999.0)]
            
            if data.empty:
                continue

            data = data.sort_values(x_column)
            
            if is_diameter_plot:
                diameter_column = find_column_case_insensitive(zone.data.columns, ["DIAMETER", "Diameter"])
                bin_set_column = find_column_case_insensitive(zone.data.columns, ["BIN_SET", "BinSet"])
                if diameter_column is None:
                    continue

                working_data = zone.data.loc[data.index, [x_column, y_column, diameter_column] + ([bin_set_column] if bin_set_column is not None else [])].copy()
                working_data = working_data[working_data[diameter_column] != -999.0]

                for diameter, diameter_data in working_data.groupby(diameter_column):
                    diameter_data = diameter_data.sort_values(x_column)
                    if diameter_data.empty:
                        continue
                    bin_set = ""
                    if bin_set_column is not None:
                        bin_values = sorted(set(str(value) for value in diameter_data[bin_set_column]))
                        bin_set = f" {'/'.join(bin_values)}"
                    trace_name = f"{label}{bin_set} D={diameter:g} um"
                    trace_key = (
                        participant.participant_id,
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
                            x=diameter_data[x_column],
                            y=diameter_data[y_column],
                            mode="lines+markers",
                            name=trace_name,
                            legendgroup=trace_name,
                            line=dict(color=color),
                            marker=dict(color=color),
                            hovertemplate=(
                                f"Participant: {escape(label)}<br>"
                                f"Case: {escape(case_id)}<br>"
                                f"Zone: {escape(zone_name)}<br>"
                                f"Diameter={diameter:g} um<br>"
                                f"{escape(x_column)}=%{{x}}<br>"
                                f"{escape(y_column)}=%{{y}}<extra></extra>"
                            ),
                        )
                    )                    
                    trace_count += 1
                    participant_trace_count += 1

                continue

            grid_column = find_column_case_insensitive(zone.data.columns, ["GRID_LEVEL", "GridLevel"])
            custom_data = None
            if grid_column is not None:
                custom_data = zone.data.loc[data.index, grid_column]
                trace_key = (
                    participant.participant_id,
                    plot_spec["plot_key"],
                    zone_name,
                )

                if trace_key in seen_trace_keys:
                    continue

                seen_trace_keys.add(trace_key)
                fig.add_trace(
                    go.Scatter(
                        x=data[x_column],
                        y=data[y_column],
                        mode="lines+markers",
                        name=label,
                        legendgroup=label,
                        line=dict(color=color),
                        marker=dict(color=color),
                        customdata=custom_data,
                        hovertemplate=(
                            f"Participant: {escape(label)}<br>"
                            f"Case: {escape(case_id)}<br>"
                            f"Zone: {escape(zone_name)}<br>"
                            f"{escape(x_column)}=%{{x}}<br>"
                            f"{escape(y_column)}=%{{y}}<br>"
                            "Grid level=%{customdata}<extra></extra>"
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
    is_diameter_plot = plot_spec.get("diameter_plot", False)

    if is_diameter_plot:
        fig, trace_count, skipped_notes = build_grid_convergence_figure(participants, case_id, plot_spec)

        if trace_count == 0:
            figure_html = empty_placeholder(title=plot_spec["title"], message="No matching gridConvergence variables were found yet for this case.")
        else:
            filename = f"{slugify(case_id)}_{plot_spec['filename_slug']}"
            figure_html = figure_to_html_div(fig, filename=filename)

        return f"""
        <section class="plot-subsection">
          <h4>{escape(plot_spec["title"])}</h4>
          <p class="plot-description">
            Diameter-resolved grid-convergence data. Missing values equal to -999 are ignored. Legend: PID.
          </p>
          <div class="plot-container">
            {figure_html}
          </div>
        </section>
        """

    roughness_keys = collect_cfd_roughness_keys(participants, case_id, plot_spec)

    if not roughness_keys:
        return f"""
        <section class="plot-subsection">
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
    <section class="plot-subsection">
      <h4>{escape(plot_spec["title"])}</h4>
      <p class="plot-description">
        Grid-convergence data grouped by roughness condition. Missing values equal to -999 are ignored. Legend: PID.
      </p>
      {figures_html}
    </section>
    """


def build_grid_convergence_section(participants, case_id: str) -> str:
    html = ""

    for plot_spec in GRID_CONVERGENCE_PLOTS:
        if plot_spec.get("diameter_plot", False):
            html += build_grid_convergence_diameter_subsection(participants, case_id, plot_spec)
        else:
            html += build_grid_convergence_plot_subsection(participants, case_id, plot_spec)

    return f"""
    <section class="plot-subsection grid-convergence-section">
      <h3>Grid convergence</h3>
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

                if diameter == -999.0 or bin_number == -999.0:
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
            y_column = find_column_case_insensitive(zone.data.columns, plot_spec["y_candidates"])
            diameter_column = find_column_case_insensitive(zone.data.columns, ["DIAMETER", "Diameter"])
            bin_set_column = find_column_case_insensitive(zone.data.columns, ["BIN_SET", "BinSet"])
            bin_column = find_column_case_insensitive(zone.data.columns, ["BIN", "Bin"])
            grid_column = find_column_case_insensitive(zone.data.columns, ["GRID_LEVEL", "GridLevel"])

            if x_column is None or y_column is None or diameter_column is None or bin_set_column is None or bin_column is None:
                continue

            data = zone.data.copy()
            data = data[(data[x_column] > 0.0) & (data[x_column] != -999.0) & (data[y_column] != -999.0)]
            data = data[data[bin_set_column].astype(str) == str(target_bin_set)]
            data = data[data[bin_column].astype(float) == float(target_bin_number)]
            data = data[data[diameter_column].astype(float) == float(target_diameter)]

            if data.empty:
                continue

            data = data.sort_values(x_column)
            custom_data = data[grid_column] if grid_column is not None else None

            trace_key = (participant.participant_id, plot_spec["plot_key"], zone_name, target_bin_set, float(target_diameter))

            if trace_key in seen_trace_keys:
                continue

            seen_trace_keys.add(trace_key)

            fig.add_trace(
                go.Scatter(
                    x=data[x_column],
                    y=data[y_column],
                    mode="lines+markers",
                    name=label,
                    legendgroup=label,
                    line=dict(color=color),
                    marker=dict(color=color),
                    customdata=custom_data,
                    hovertemplate=(
                        f"Participant: {escape(label)}<br>"
                        f"Case: {escape(case_id)}<br>"
                        f"Bin set: {escape(target_bin_set)}<br>"
                        f"Bin: {target_bin_number:g}<br>"
                        f"Diameter: {target_diameter:g} μm<br>"
                        f"Zone: {escape(zone_name)}<br>"
                        f"{escape(x_column)}=%{{x}}<br>"
                        f"{escape(y_column)}=%{{y}}<br>"
                        "Grid level=%{customdata}<extra></extra>"
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
        <section class="plot-subsection">
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
    <section class="plot-subsection">
      <h4>{escape(plot_spec["title"])}</h4>
      <p class="plot-description">
        Diameter-resolved grid-convergence data. Each figure corresponds to one bin set and one droplet diameter. Legend: PID.
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
            y_column = find_column_case_insensitive(zone.data.columns, plot_spec["y_candidates"])

            if x_column is not None and y_column is not None:
                roughness_keys.add(roughness_key)

    preferred_order = ["smooth", "0.5mm", "1mm", "1.5mm", "variable_roughness"]
    return [key for key in preferred_order if key in roughness_keys]