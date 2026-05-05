from __future__ import annotations

from html import escape
from typing import Any
import re

import plotly.graph_objects as go

from .gatherParticipantData import iter_case_data

GRID_CONVERGENCE_PLOTS: list[dict[str, Any]] = [
    {"plot_key": "cl_vs_n", "title": "CL grid convergence", "x_candidates": ["N"], "y_candidates": ["CL"], "x_label": "N [-]", "y_label": "CL [-]", "filename_slug": "cl_vs_n"},
    {"plot_key": "cd_vs_n", "title": "CD grid convergence", "x_candidates": ["N"], "y_candidates": ["CD"], "x_label": "N [-]", "y_label": "CD [-]", "filename_slug": "cd_vs_n"},
    {"plot_key": "cmx_vs_n", "title": "CMX grid convergence", "x_candidates": ["N"], "y_candidates": ["CMX"], "x_label": "N [-]", "y_label": "CMX [-]", "filename_slug": "cmx_vs_n"},
    {"plot_key": "cmy_vs_n", "title": "CMY grid convergence", "x_candidates": ["N"], "y_candidates": ["CMY"], "x_label": "N [-]", "y_label": "CMY [-]", "filename_slug": "cmy_vs_n"},
    {"plot_key": "cmz_vs_n", "title": "CMZ grid convergence", "x_candidates": ["N"], "y_candidates": ["CMZ"], "x_label": "N [-]", "y_label": "CMZ [-]", "filename_slug": "cmz_vs_n"},
    {"plot_key": "water_mass_vs_n", "title": "Water mass grid convergence", "x_candidates": ["N"], "y_candidates": ["WATER_MASS", "WaterMass"], "x_label": "N [-]", "y_label": "Water mass [kg]", "filename_slug": "water_mass_vs_n"},
    {"plot_key": "ice_mass_vs_n", "title": "Ice mass grid convergence", "x_candidates": ["N"], "y_candidates": ["ICE_MASS", "IceMass"], "x_label": "N [-]", "y_label": "Ice mass [kg]", "filename_slug": "ice_mass_vs_n"},
    {"plot_key": "water_evap_mass_vs_n", "title": "Water evaporation mass grid convergence", "x_candidates": ["N"], "y_candidates": ["WATER_EVAP_MASS", "WaterEvapMass"], "x_label": "N [-]", "y_label": "Water evaporation mass [kg]", "filename_slug": "water_evap_mass_vs_n"},
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


def build_grid_convergence_figure(participants, case_id: str, plot_spec: dict[str, Any]) -> tuple[go.Figure, int]:
    fig = go.Figure()
    trace_count = 0

    for participant, case_data in iter_case_data(participants, case_id):
        if case_data.grid_convergence_data is None:
            continue
        label = participant_label(participant)
        for zone_name, zone in case_data.grid_convergence_data.zones.items():
            x_column = find_column_case_insensitive(zone.data.columns, plot_spec["x_candidates"])
            y_column = find_column_case_insensitive(zone.data.columns, plot_spec["y_candidates"])
            if x_column is None or y_column is None:
                continue
            data = zone.data[[x_column, y_column]].copy()
            data = data[(data[x_column] > 0.0) & (data[y_column] != -999.0)]
            if data.empty:
                continue
            grid_column = find_column_case_insensitive(zone.data.columns, ["GRID_LEVEL", "GridLevel"])
            custom_data = None
            if grid_column is not None:
                custom_data = zone.data.loc[data.index, grid_column]
            fig.add_trace(go.Scatter(x=data[x_column], y=data[y_column], mode="lines+markers", name=label, legendgroup=label, customdata=custom_data, hovertemplate=(f"Participant: {escape(label)}<br>" f"Case: {escape(case_id)}<br>" f"Zone: {escape(zone_name)}<br>" f"{escape(x_column)}=%{{x}}<br>" f"{escape(y_column)}=%{{y}}<br>" "Grid level=%{customdata}<extra></extra>")))
            trace_count += 1

    style_xy_figure(fig, plot_spec["x_label"], plot_spec["y_label"])
    return fig, trace_count


def build_grid_convergence_plot_subsection(participants, case_id: str, plot_spec: dict[str, Any]) -> str:
    fig, trace_count = build_grid_convergence_figure(participants, case_id, plot_spec)
    if trace_count == 0:
        figure_html = empty_placeholder(title=plot_spec["title"], message="No matching gridConvergence variables were found yet for this case.")
    else:
        filename = f"{slugify(case_id)}_{plot_spec['filename_slug']}"
        figure_html = figure_to_html_div(fig, filename=filename)
    description = "Grid-convergence data from the case-level gridConvergence file. Missing values equal to -999 are ignored. Legend: PID."
    return f"""
    <section class="plot-subsection">
      <h4>{escape(plot_spec["title"])}</h4>
      <p class="plot-description">{escape(description)}</p>
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
    <section class="plot-subsection grid-convergence-section">
      <h3>Grid convergence</h3>
      {html}
    </section>
    """
