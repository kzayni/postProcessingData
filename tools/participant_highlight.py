"""Highlight one participant while retaining muted comparison traces."""

from __future__ import annotations

import re

MUTED_PARTICIPANT_COLOR = "#b8b8b8"
MUTED_PARTICIPANT_LABEL = "IPW3 participants"


def trace_participant_id(trace) -> str:
    meta = trace.meta if isinstance(trace.meta, dict) else {}
    participant_id = meta.get("ipw3_participant_id")
    if participant_id:
        return str(participant_id).zfill(3)
    group_match = re.fullmatch(r"participant_symbol_(\d{1,3})", str(trace.legendgroup or ""))
    if group_match:
        return group_match.group(1).zfill(3)
    name_match = re.match(r"^(\d{1,3})(?=\D|$)", str(trace.name or "").strip())
    return name_match.group(1).zfill(3) if name_match else ""


def highlight_participant(figure, participant_id: str) -> None:
    """Mute non-selected participant traces and collapse their legend IDs."""
    selected = str(participant_id).zfill(3)
    muted_legend_shown: set[str] = set()
    for trace in figure.data:
        pid = trace_participant_id(trace)
        if not pid or pid == selected:
            continue
        if getattr(trace, "line", None) is not None:
            trace.line.color = MUTED_PARTICIPANT_COLOR
        if getattr(trace, "marker", None) is not None:
            trace.marker.color = MUTED_PARTICIPANT_COLOR
            if trace.marker.line is not None:
                trace.marker.line.color = MUTED_PARTICIPANT_COLOR
        legend_id = str(trace.legend or "legend")
        trace.name = MUTED_PARTICIPANT_LABEL
        trace.legendgroup = f"muted_participants_{legend_id}"
        trace.showlegend = legend_id not in muted_legend_shown
        muted_legend_shown.add(legend_id)


def bring_participant_to_front(figure, participant_id: str) -> None:
    """Draw the selected participant after every comparison/reference trace."""
    selected = str(participant_id).zfill(3)
    background = tuple(
        trace for trace in figure.data if trace_participant_id(trace) != selected
    )
    foreground = tuple(
        trace for trace in figure.data if trace_participant_id(trace) == selected
    )
    figure.data = background + foreground


def order_comparison_traces(figure, foreground_participant_id: str | None = None) -> None:
    """Layer participant IDs, experiments, then the clean reference."""
    foreground = str(foreground_participant_id).zfill(3) if foreground_participant_id else ""
    ordinary = []
    participants = []
    experimental = []
    mccs_background = []
    clean = []
    for position, trace in enumerate(figure.data):
        group = str(trace.legendgroup or "").lower()
        name = str(trace.name or "").lower()
        pid = trace_participant_id(trace)
        if group == "clean_reference":
            clean.append((position, trace))
        elif group.startswith((
            "experimental_meanccs",
            "experimental_maxccs",
            "experimental_minccs",
        )):
            mccs_background.append((position, trace))
        elif group.startswith(("experimental_", "reference_")) or name.startswith("exp"):
            experimental.append((position, trace))
        elif pid:
            # Keep legend entries ordered numerically even when one participant
            # is promoted in the SVG drawing order.
            meta = trace.meta if isinstance(trace.meta, dict) else {}
            grid_level = str(meta.get("ipw3_grid_level", "")).upper()
            if not re.fullmatch(r"L[1-4]", grid_level):
                trace_name = str(trace.name or "").strip().upper()
                grid_level = trace_name if re.fullmatch(r"L[1-4]", trace_name) else ""
            trace.legendrank = int(grid_level[1:]) if grid_level else int(pid)
            participants.append((pid == foreground, int(pid), position, trace))
        else:
            ordinary.append((position, trace))
    participants.sort(key=lambda item: (item[0], item[1], item[2]))
    figure.data = (
        tuple(trace for _, trace in ordinary)
        + tuple(trace for _, trace in mccs_background)
        + tuple(trace for _, _, _, trace in participants)
        + tuple(trace for _, trace in experimental)
        + tuple(trace for _, trace in clean)
    )
