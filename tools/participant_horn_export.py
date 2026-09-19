"""Upper-horn construction figures for submitted participant ice contours."""

from __future__ import annotations

import math
import re

import numpy as np
import plotly.graph_objects as go

from . import convergence_data_builder as convergence
from .gatherParticipantData import decode_slice_position, iter_grid_datasets
from .iceshape_builder import (
    find_submitted_ice_xz_columns,
    naca_clean_reference_points,
    onera_clean_reference_points,
    parse_ipw3_ice_shape_zone_name,
    upper_horn_geometry,
    valid_submitted_ice_shape_rows,
)


def participant_horn_method_figures(participants, case_id: str):
    """Yield (participant ID, filename, figure) for each available condition.

    One contour is retained per participant, slice, and roughness. Prefer L1,
    then the largest bin count and a single-layer contour.
    """
    selected = {}
    for level_number in range(1, 5):
        level = f"L{level_number}"
        for participant, _, _, dataset in iter_grid_datasets(participants, case_id, level):
            ice_data = getattr(dataset, "ice_shape_data", None)
            if ice_data is None:
                continue
            pid = str(participant.participant_id).zfill(3)
            for zone_name, zone in ice_data.zones.items():
                info = parse_ipw3_ice_shape_zone_name(zone_name)
                if info is None or info["shape_role"] not in {"SINGLE_LAYER", "FINAL_LAYER"}:
                    continue
                slice_position = decode_slice_position(info["slice"]) if info["slice"] else None
                if case_id == "TC_ONERAM6" and slice_position is None:
                    continue
                x_column, z_column = find_submitted_ice_xz_columns(zone.data.columns)
                if x_column is None or z_column is None:
                    continue
                shape = valid_submitted_ice_shape_rows(zone.data, x_column, z_column)
                if shape.empty:
                    continue
                roughness = convergence.extract_ice_shape_roughness_key(zone_name)
                bins = convergence.bin_count_from_bin_set(info["bins"]) or 0
                key = (pid, slice_position, roughness)
                rank = (-level_number, bins, int(info["shape_role"] == "SINGLE_LAYER"))
                if key not in selected or rank > selected[key][0]:
                    selected[key] = (rank, level, bins, shape[x_column].to_numpy(), shape[z_column].to_numpy())

    for (pid, slice_position, roughness), (_, level, bins, x, z) in sorted(
        selected.items(), key=lambda item: (item[0][0], item[0][1] or 0, item[0][2])
    ):
        clean = (onera_clean_reference_points(slice_position)
                 if case_id == "TC_ONERAM6" else naca_clean_reference_points())
        if clean is None:
            continue
        geometry = upper_horn_geometry(x, z, case_id, slice_position)
        if geometry is None:
            continue
        reference, horn, angle = geometry
        clean_x, clean_z, _ = clean
        span = max(math.hypot(horn[0] - reference[0], horn[1] - reference[1]), 0.01)
        ray_length = 0.65 * span
        arc_radius = 0.24 * span
        arc_angles = np.linspace(0.0, math.radians(angle), 48)

        figure = go.Figure()
        figure.add_trace(go.Scatter(x=clean_x, y=clean_z, mode="lines", name="Clean surface",
                                    line={"color": "#94a3b8", "width": 2}))
        figure.add_trace(go.Scatter(x=x, y=z, mode="lines", name=f"Participant {pid}",
                                    meta={"ipw3_participant_id": pid},
                                    line={"color": "#111827", "width": 2}))
        figure.add_trace(go.Scatter(
            x=[reference[0], reference[0] + ray_length], y=[reference[1], reference[1]],
            mode="lines", name="Positive x-axis", line={"color": "#dc2626", "width": 2},
        ))
        figure.add_trace(go.Scatter(
            x=[reference[0], horn[0]], y=[reference[1], horn[1]], mode="lines+markers",
            name="Upper-horn construction", line={"color": "#dc2626", "width": 3},
            marker={"color": ["#2563eb", "#dc2626"], "size": [8, 10],
                    "symbol": ["circle", "diamond"]},
        ))
        figure.add_trace(go.Scatter(
            x=reference[0] + arc_radius * np.cos(arc_angles),
            y=reference[1] + arc_radius * np.sin(arc_angles),
            mode="lines", name="Upper-horn angle", line={"color": "#dc2626", "width": 3},
        ))
        figure.add_annotation(x=horn[0], y=horn[1], text=f"θupper = {angle:.2f}°",
                              showarrow=True, arrowhead=2, ax=45, ay=-35,
                              bgcolor="rgba(255,255,255,0.85)")
        pad = (0.45 if case_id == "TC_ONERAM6" else 0.35) * span
        figure.update_layout(
            showlegend=False, paper_bgcolor="white", plot_bgcolor="white",
            xaxis={"title": "X [m]", "range": [min(reference[0], horn[0]) - pad,
                                               max(reference[0], horn[0]) + pad],
                   "showline": True, "mirror": True,
                   "linecolor": "black", "linewidth": 2},
            yaxis={"title": "Z [m]", "range": [min(reference[1], horn[1]) - pad,
                                               max(reference[1], horn[1]) + pad],
                   "scaleanchor": "x", "scaleratio": 1.0,
                   "showline": True, "mirror": True,
                   "linecolor": "black", "linewidth": 2},
        )
        figure.add_shape(
            type="rect", xref="x domain", yref="y domain",
            x0=0, x1=1, y0=0, y1=1,
            line={"color": "black", "width": 2},
            fillcolor="rgba(0,0,0,0)", layer="above",
        )
        slice_slug = f"slice_{str(slice_position).replace('.', 'p')}" if slice_position is not None else "slice_0p9144"
        roughness_slug = re.sub(r"[^a-z0-9]+", "_", roughness.lower()).strip("_")
        filename = (f"{case_id.lower()}_p{pid}_{level}_{slice_slug}_bins{bins:02d}_"
                    f"roughness_{roughness_slug}_upper_horn_angle_method.png")
        yield pid, filename, figure
