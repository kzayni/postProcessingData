"""Statistical summaries and box-and-whisker plots for grid convergence.

Each grid level is summarized across visible participant traces that contain
valid data at all four grids (L1-L4). The statistics are computed from the
original physical values, using the same complete sample at every grid.
"""

from __future__ import annotations

from typing import Any, Iterable
import math
import re
from html import escape

import numpy as np
import plotly.graph_objects as go

from .participant_style import participant_color


# =============================================================================
# EDITABLE BOX-PLOT STYLE
# =============================================================================
STATISTICAL_PLOT_STYLE: dict[str, Any] = {
    "grid_order": ["L1", "L2", "L3", "L4"],
    # Only traces with a valid value at every listed grid level are included.
    "require_all_grid_levels": True,
    "height": 520,
    "font": {"family": "Arial, Helvetica, sans-serif", "size": 16},
    "axis_title_size": 18,
    "box_fill_color": "rgba(0, 0, 0, 0)",
    "box_line_color": "#000000",
    "box_line_width": 2,
    "marker_color": "#1f77b4",
    "marker_size": 7,
    "show_all_points": False,
    "jitter": 0.0,
    "point_position": 0.0,
    "plot_bgcolor": "white",
    "paper_bgcolor": "white",
    "gridcolor": "lightgray",
    "box_width": 0.2,
    "margin": {"l": 90, "r": 40, "t": 30, "b": 95},
}


def values_by_grid_level(
    fig: go.Figure,
    grid_order: Iterable[str] | None = None,
) -> dict[str, list[float]]:
    """Extract finite Y values by grid level from convergence customdata."""
    order = [str(level).upper() for level in (grid_order or STATISTICAL_PLOT_STYLE["grid_order"])]
    values: dict[str, list[float]] = {level: [] for level in order}
    for trace in fig.data:
        if trace.visible is False or trace.y is None or trace.customdata is None:
            continue
        trace_values: dict[str, list[float]] = {level: [] for level in order}
        for row, value in zip(trace.customdata, trace.y):
            if row is None or len(row) == 0:
                continue
            level = str(row[0]).strip().upper()
            if level not in values:
                continue
            try:
                number = float(value)
            except (TypeError, ValueError):
                continue
            if math.isfinite(number):
                trace_values[level].append(number)
        if STATISTICAL_PLOT_STYLE.get("require_all_grid_levels", True) and any(
            not trace_values[level] for level in order
        ):
            continue
        for level in order:
            values[level].extend(trace_values[level])
    return values


def complete_grid_participant_ids(
    fig: go.Figure,
    grid_order: Iterable[str] | None = None,
) -> list[str]:
    """Return participant IDs from the same complete traces used in statistics."""
    order = [str(level).upper() for level in (grid_order or STATISTICAL_PLOT_STYLE["grid_order"])]
    participant_ids: set[str] = set()
    for trace in fig.data:
        if trace.visible is False or trace.y is None or trace.customdata is None:
            continue
        valid_levels: set[str] = set()
        for row, value in zip(trace.customdata, trace.y):
            if row is None or len(row) == 0:
                continue
            level = str(row[0]).strip().upper()
            try:
                number = float(value)
            except (TypeError, ValueError):
                continue
            if level in order and math.isfinite(number):
                valid_levels.add(level)
        if STATISTICAL_PLOT_STYLE.get("require_all_grid_levels", True) and any(level not in valid_levels for level in order):
            continue
        trace_name = str(getattr(trace, "name", "") or "").strip()
        id_match = re.search(r"(?<!\d)(\d{3}(?:\.\d+)?)(?!\d)", trace_name)
        participant_id = id_match.group(1) if id_match else trace_name.split(" | ", 1)[0]
        if participant_id:
            participant_ids.add(participant_id)
    return sorted(participant_ids)


def complete_grid_points(
    fig: go.Figure,
    grid_order: Iterable[str] | None = None,
) -> dict[str, list[tuple[float, str]]]:
    """Return value/participant pairs for traces valid at every grid level."""
    order = [str(level).upper() for level in (grid_order or STATISTICAL_PLOT_STYLE["grid_order"])]
    points: dict[str, list[tuple[float, str]]] = {level: [] for level in order}
    for trace in fig.data:
        if trace.visible is False or trace.y is None or trace.customdata is None:
            continue
        trace_values: dict[str, list[float]] = {level: [] for level in order}
        for row, value in zip(trace.customdata, trace.y):
            if row is None or len(row) == 0:
                continue
            level = str(row[0]).strip().upper()
            try:
                number = float(value)
            except (TypeError, ValueError):
                continue
            if level in trace_values and math.isfinite(number):
                trace_values[level].append(number)
        if STATISTICAL_PLOT_STYLE.get("require_all_grid_levels", True) and any(not trace_values[level] for level in order):
            continue
        trace_name = str(getattr(trace, "name", "") or "").strip()
        id_match = re.search(r"(?<!\d)(\d{3}(?:\.\d+)?)(?!\d)", trace_name)
        participant_id = id_match.group(1) if id_match else trace_name.split(" | ", 1)[0]
        for level in order:
            points[level].extend((value, participant_id) for value in trace_values[level])
    return points


def compute_grid_level_statistics(
    values: dict[str, list[float]],
) -> dict[str, dict[str, float | int | None]]:
    """Compute mean, median, sample standard deviation, CV, quartiles, and IQR."""
    statistics: dict[str, dict[str, float | int | None]] = {}
    for level, level_values in values.items():
        data = np.asarray(level_values, dtype=float)
        data = data[np.isfinite(data)]
        if data.size == 0:
            statistics[level] = {
                "count": 0, "mean": None, "median": None, "standard_deviation": None,
                "coefficient_of_variation": None,
                "q1": None, "q3": None, "iqr": None,
            }
            continue
        q1, median, q3 = np.percentile(data, [25.0, 50.0, 75.0])
        mean = float(np.mean(data))
        standard_deviation = float(np.std(data, ddof=1)) if data.size > 1 else None
        statistics[level] = {
            "count": int(data.size),
            "mean": mean,
            "median": float(median),
            # Sample standard deviation; undefined for a single observation.
            "standard_deviation": standard_deviation,
            # Magnitude-based sample CV; undefined for a zero mean or when
            # the sample standard deviation itself is undefined.
            "coefficient_of_variation": (
                standard_deviation / abs(mean) * 100.0
                if standard_deviation is not None and not math.isclose(mean, 0.0, abs_tol=1e-15)
                else None
            ),
            "q1": float(q1),
            "q3": float(q3),
            "iqr": float(q3 - q1),
        }
    return statistics


def build_grid_level_box_plot(
    source_fig: go.Figure,
    y_axis_title: str,
) -> tuple[go.Figure, dict[str, dict[str, float | int | None]]]:
    """Build one box per grid level and return it with its statistics."""
    style = STATISTICAL_PLOT_STYLE
    order = [str(level) for level in style["grid_order"]]
    values = values_by_grid_level(source_fig, order)
    statistics = compute_grid_level_statistics(values)
    participant_ids = complete_grid_participant_ids(source_fig, order)
    participant_points = complete_grid_points(source_fig, order)
    for level_statistics in statistics.values():
        level_statistics["participant_ids"] = participant_ids
    box_fig = go.Figure()

    for level in order:
        level_values = values[level]
        if not level_values:
            continue
        stats = statistics[level]
        std_text = "n/a" if stats["standard_deviation"] is None else f"{stats['standard_deviation']:.6g}"
        cv_text = "n/a" if stats["coefficient_of_variation"] is None else f"{stats['coefficient_of_variation']:.6g}%"
        box_fig.add_trace(go.Box(
            y=level_values,
            name=level,
            showlegend=False,
            boxpoints="all" if style["show_all_points"] else False,
            jitter=style["jitter"],
            pointpos=style["point_position"],
            fillcolor=style["box_fill_color"],
            width=style["box_width"],
            line={"color": style["box_line_color"], "width": style["box_line_width"]},
            marker={"color": style["marker_color"], "size": style["marker_size"]},
            meta=[stats["count"], stats["mean"], stats["median"], std_text, cv_text, stats["q1"], stats["q3"], stats["iqr"]],
            hovertemplate=(
                "Grid: " + level + "<br>Value=%{y:.6g}<br>"
                "n=%{meta[0]}<br>Mean=%{meta[1]:.6g}<br>Median=%{meta[2]:.6g}<br>"
                "Sample standard deviation=%{meta[3]}<br>"
                "Coefficient of variation=%{meta[4]}<br>"
                "Q1=%{meta[5]:.6g}<br>Q3=%{meta[6]:.6g}<br>IQR=%{meta[7]:.6g}"
                "<extra></extra>"
            ),
        ))
        level_points = participant_points[level]
        # Plotly Box accepts only one marker color per trace, so use one
        # transparent one-point trace per participant. Points are always
        # visible and retain their participant colors; no toggle is needed.
        for point_index, (point_value, participant_id) in enumerate(level_points):
            box_fig.add_trace(go.Box(
                x=[level],
                y=[point_value],
                customdata=[[participant_id]],
                name=f"{level} participant data",
                legendgroup=f"participant_points_{level}",
                showlegend=False,
                visible=True,
                boxpoints="all",
                jitter=0.0,
                pointpos=style["point_position"],
                fillcolor="rgba(0,0,0,0)",
                line={"color": "rgba(0,0,0,0)", "width": 0},
                marker={"color": participant_color(participant_id), "size": style["marker_size"]},
                hoveron="points",
                hovertemplate="Participant: %{customdata[0]}<br>Grid: " + level + "<br>Value=%{y:.6g}<extra></extra>",
            ))
    box_fig.update_layout(
        font=style["font"],
        height=style["height"],
        showlegend=False,
        boxmode="overlay",
        xaxis={
            "title": {"text": "Grid level<br><span style='font-size:14px'>← Finer&nbsp;&nbsp;|&nbsp;&nbsp;Coarser →</span>", "font": {"size": style["axis_title_size"]}},
            "categoryorder": "array", "categoryarray": order,
            "showline": True, "linecolor": "black", "linewidth": 2,
            "mirror": True, "showgrid": True, "gridcolor": style["gridcolor"],
        },
        yaxis={
            "title": {"text": y_axis_title, "font": {"size": style["axis_title_size"]}},
            "showline": True, "linecolor": "black", "linewidth": 2,
            "mirror": True, "showgrid": True, "gridcolor": style["gridcolor"],
            "zeroline": False,
        },
        margin=style["margin"],
        plot_bgcolor=style["plot_bgcolor"],
        paper_bgcolor=style["paper_bgcolor"],
    )
    return box_fig, statistics


def statistics_table_html(
    statistics: dict[str, dict[str, Any]],
    y_axis_title: str = "",
) -> str:
    """Render the computed values so dispersion statistics are explicit."""
    rows = ""
    participant_ids: list[str] = []
    for level in STATISTICAL_PLOT_STYLE["grid_order"]:
        stats = statistics.get(level)
        if not stats or stats["count"] == 0:
            continue
        if not participant_ids:
            participant_ids = [str(value) for value in stats.get("participant_ids", [])]
        def formatted(value: float | int | None) -> str:
            return "n/a" if value is None else f"{float(value):.6g}"
        def formatted_percent(value: float | int | None) -> str:
            return "n/a" if value is None else f"{float(value):.6g}%"
        rows += (
            f"<tr><td>{escape(level)}</td>"
            f"<td>{formatted(stats['mean'])}</td>"
            f"<td>{formatted(stats['median'])}</td>"
            f"<td>{formatted(stats['standard_deviation'])}</td>"
            f"<td>{formatted_percent(stats['coefficient_of_variation'])}</td>"
            f"<td>{formatted(stats['q1'])}</td><td>{formatted(stats['q3'])}</td>"
            f"<td>{formatted(stats['iqr'])}</td></tr>"
        )
    if not rows:
        return ""
    participant_summary = (
        f"Number of participants considered = {len(participant_ids)} | "
        f"IDs: {escape(', '.join(participant_ids))}"
    )
    unit_match = re.search(r"(\[[^\[\]]+\])\s*$", y_axis_title)
    unit_suffix = f" {escape(unit_match.group(1))}" if unit_match is not None else ""
    return f"""
    <p class="plot-description statistical-participants">{participant_summary}</p>
    <div class="readme-table-wrapper">
      <table class="participant-table statistical-table">
        <thead><tr><th>Grid</th><th>Mean{unit_suffix}</th><th>Median{unit_suffix}</th><th>Sample standard deviation{unit_suffix}</th><th>Coefficient of variation [%]</th><th>Q1{unit_suffix}</th><th>Q3{unit_suffix}</th><th>IQR{unit_suffix}</th></tr></thead>
        <tbody>{rows}</tbody>
      </table>
    </div>
    """
