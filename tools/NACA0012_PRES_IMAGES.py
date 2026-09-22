"""Curated NACA0012 presentation image configuration and exporter."""

from __future__ import annotations

import copy
from pathlib import Path
import re
import shutil
import tempfile

from . import convergence_data_builder, cutdata_builder, iceshape_builder
from .plot_style import NACA0012_ROUGHNESS_GROUP_STYLE

from tools.roughness_legend import apply_roughness_participant_legend, grouped_legend_positions
from tools.participant_highlight import (
    bring_participant_to_front,
    highlight_participant,
    order_comparison_traces,
    trace_participant_id as get_trace_participant_id,
)

# Editable participant symbols for grouped-roughness presentation plots.
# Participant symbol size in the legend (pixels).
GROUPED_ROUGHNESS_PARTICIPANT_SYMBOL_SIZE = 16
# Participant symbol size on the plotted curves (pixels).
GROUPED_ROUGHNESS_PLOT_SYMBOL_SIZE = 9
# Larger markers used by the L1 and L2 grouped-turbulence ice-shape figures.
TURBULENCE_MODEL_ICE_PARTICIPANT_SYMBOL_SIZE = 24
TURBULENCE_MODEL_ICE_PLOT_SYMBOL_SIZE = 18
# Participant marker sizes for integrated Qc' convergence figures.
QC_PARTICIPANT_SYMBOL_SIZE = 24
QC_PLOT_SYMBOL_SIZE = 18
# Editable paper-coordinate positions for the roughness-line, participant-symbol,
# and experimental legend rows in grouped-roughness plots.
# The experimental row is the anchor. Each available row above it is raised
# by this amount; with no experimental data the symbols stay at the anchor.
GROUPED_LEGEND_ROW_SHIFT = 0.05
GROUPED_ROUGHNESS_LEGEND_X = 0.0
GROUPED_PARTICIPANT_LEGEND_X = 0.0
GROUPED_EXPERIMENTAL_LEGEND_POSITION = {"x": 0.0, "y": 1.02}
GROUPED_ROUGHNESS_PARTICIPANT_SYMBOLS = {
    "001": "circle", "002": "square", "003": "diamond",
    "004": "triangle-up", "006": "triangle-down", "007": "cross",
    "008": "x", "009": "star", "010": "pentagon",
    "013": "hexagon", "014": "triangle-left", "015": "triangle-right",
    "019": "hourglass", "020": "bowtie",
}

# Turbulence-model families reported in tools/submissions.md. Variants such as
# k-omega TNT/SST and SA-neg/QCR stay within their parent model family.
NACA0012_TURBULENCE_MODEL_STYLES = {
    "kw": {
        "participant_ids": {"001", "002", "003", "004", "008", "009", "019"},
        "label": "k-ω",
        "color": "#1f77b4",
        "rank": 0,
    },
    "sa": {
        "participant_ids": {"007", "010", "013", "014"},
        "label": "SA",
        "color": "#d62728",
        "rank": 1,
    },
}

NACA0012_TURBULENCE_ROUGHNESS_COLORS = {
    "sa": {
        "smooth": "#7F7F7F",    # gray
        "half_mm": "#ff7f0e",   # orange
        "one_mm": "#d62728",    # red
        "variable": "#800020",  # burgundy
    },
    "kw": {
        "smooth": "#17BECF",    # cyan
        "half_mm": "#56B4E9",   # light blue
        "one_mm": "#1F4E9E",    # dark blue
        "variable": "#7B2CBF",  # purple
    },
}

NACA0012_DROPLET_MODEL_STYLES = {
    "lagrangian": {
        "participant_ids": {"002", "010", "013", "014"},
        "label": "Lagrangian",
        "color": "#d62728",
        "rank": 0,
    },
    "eulerian": {
        "participant_ids": {"001", "004", "007", "008", "009", "019"},
        "label": "Eulerian",
        "color": "#1f77b4",
        "rank": 1,
    },
}

NACA0012_THERMODYNAMICS_MODEL_STYLES = {
    "messinger": {
        "participant_ids": {"001", "002", "003", "004", "007", "010", "013", "014"},
        "label": "Messinger",
        "color": "#d62728",
        "rank": 0,
    },
    "swim_other": {
        "participant_ids": {"008", "009", "019"},
        "label": "SWIM / Other",
        "color": "#1f77b4",
        "rank": 1,
    },
}

OUTPUT_DIR = Path("FIGURES_NACA0012")

# Standard NACA0012 presentation canvas in pixels.
WIDTH = 1350
HEIGHT = 1000

MedianLineConfig = dict[str, object]
FigureSpec = tuple[str, str, int, int, list[str], bool] | tuple[
    str, str, int, int, list[str], bool, MedianLineConfig
]

# POLIMO L1-L4 BINS15 ice-shape overlay stroke widths, in pixels.
# Edit these values to change only the two participant-007 presentation plots.
POLIMO_COMBINED_ICE_LINE_WIDTH = {"L1": 3, "L2": 5, "L3": 5, "L4": 5}


def _read_polimo_normals_comparison(path: Path):
    """Read and rotate only the supplemental BINS15 single-layer contour."""
    import plotly.graph_objects as go
    from .gatherParticipantData import read_tecplot_dat, rotate_naca0012_ice_shape

    if not path.exists():
        return None
    data = read_tecplot_dat(path, process_cutdata=False)
    rotate_naca0012_ice_shape(data)
    zone = next((
        zone for name, zone in data.zones.items()
        if "BINS15" in name.upper() and "SINGLE_LAYER" in name.upper()
    ), None)
    if zone is None:
        return None
    x_column = next((name for name in zone.data if name.upper() == "X_ICED"), None)
    z_column = next((name for name in zone.data if name.upper() == "Z_ICED"), None)
    if x_column is None or z_column is None:
        return None
    valid = zone.data[[x_column, z_column]].apply(lambda column: column > -998.0).all(axis=1)
    return go.Scatter(
        x=zone.data.loc[valid, x_column], y=zone.data.loc[valid, z_column], mode="lines",
        name="007 (Updated Surface Normals)",
        legendgroup="007_no_surface_normals_update",
        line={"color": "#d62728", "dash": "solid", "width": 4},
        meta={"ipw3_participant_id": "007", "ipw3_normals_comparison": True},
    )


def _read_polimo_level_set_comparison(path: Path):
    """Read and rotate the supplemental BINS07 level-set contour."""
    import plotly.graph_objects as go
    from .gatherParticipantData import read_tecplot_dat, rotate_naca0012_ice_shape

    if not path.exists():
        return None
    data = read_tecplot_dat(path, process_cutdata=False)
    rotate_naca0012_ice_shape(data)
    zone = next((
        zone for name, zone in data.zones.items()
        if "BINS07" in name.upper() and "SINGLE_LAYER" in name.upper()
    ), None)
    if zone is None:
        return None
    x_column = next((name for name in zone.data if name.upper() == "X_ICED"), None)
    z_column = next((name for name in zone.data if name.upper() == "Z_ICED"), None)
    if x_column is None or z_column is None:
        return None
    valid = zone.data[[x_column, z_column]].apply(lambda column: column > -998.0).all(axis=1)
    return go.Scatter(
        x=zone.data.loc[valid, x_column], y=zone.data.loc[valid, z_column], mode="lines",
        name="Level Set", legendgroup="007_level_set", legendrank=2,
        line={"color": "#d62728", "dash": "solid", "width": 5},
        meta={"ipw3_participant_id": "007", "ipw3_geometry_method": "level_set"},
    )


def _read_polimo_multilayer_comparison(path: Path):
    """Return the requested blue/red contours from a rotated multilayer file."""
    import plotly.graph_objects as go
    from .gatherParticipantData import read_tecplot_dat, rotate_naca0012_ice_shape

    if not path.exists():
        return []
    data = read_tecplot_dat(path, process_cutdata=False)
    rotate_naca0012_ice_shape(data)
    traces = []
    styles = (
    (("KS_0p5334mm", "KS_0p5mm"), "007 | k<sub>s</sub> = 0.5334 mm", "#1f77b4"),
    (("KS_0p2667mm",), "007 | k<sub>s</sub> = 0.2667 mm", "#d62728"),
    )
    for zone_keys, label, color in styles:
        zone = next((zone for name, zone in data.zones.items()
                     if any(zone_key in name for zone_key in zone_keys)), None)
        if zone is None:
            continue
        x_column = next((name for name in zone.data if name.upper() == "X_ICED"), None)
        z_column = next((name for name in zone.data if name.upper() == "Z_ICED"), None)
        if x_column is None or z_column is None:
            continue
        valid = zone.data[[x_column, z_column]].apply(lambda column: column > -998.0).all(axis=1)
        traces.append(go.Scatter(
            x=zone.data.loc[valid, x_column], y=zone.data.loc[valid, z_column],
            mode="lines", name=label, legendgroup=label,
            line={"color": color, "dash": "solid", "width": 4},
            meta={"ipw3_participant_id": "007", "ipw3_multilayer_comparison": True},
        ))
    return traces


def _read_polimo_maxccs(path: Path):
    """Read each roughness-labelled Tecplot BLOCK MaxCCS outer edge."""
    import math
    import plotly.graph_objects as go
    from .gatherParticipantData import NACA0012_ROTATION_CENTER_X, NACA0012_ROTATION_DEGREES

    if not path.exists():
        return None
    lines = path.read_text(encoding="utf-8").splitlines()
    zone_matches = [
        (i, match) for i, line in enumerate(lines)
        if (match := re.search(
            r'ZONE\s+T\s*=\s*"(?P<name>(?:007_)?MAXCCS_BINS\d+_KS_[^"]+)"',
            line, re.IGNORECASE,
        ))
    ]
    traces = []
    colors = {"0p2667": "#d62728", "0p5334": "#1f77b4"}
    for match_index, (zone_index, zone_match) in enumerate(zone_matches):
        zone_end = zone_matches[match_index + 1][0] if match_index + 1 < len(zone_matches) else len(lines)
        zone_name = zone_match.group("name")
        roughness_match = re.search(r"_KS_(?P<value>[0-9]+(?:p[0-9]+)?)mm$", zone_name, re.IGNORECASE)
        if roughness_match is None or roughness_match.group("value") not in colors:
            continue
        roughness_key = roughness_match.group("value")
        dt_index = next((i for i in range(zone_index, min(zone_end, zone_index + 12))
                         if "DT=" in lines[i].upper()), None)
        if dt_index is None:
            continue
        connectivity_index = next((
            i for i in range(dt_index + 1, zone_end)
            if len(lines[i].split()) == 2
            and all(token.lstrip("+-").isdigit() for token in lines[i].split())
        ), None)
        if connectivity_index is None:
            continue
        values = [float(token) for line in lines[dt_index + 1:connectivity_index] for token in line.split()]
        edges = [
            tuple(int(token) - 1 for token in line.split())
            for line in lines[connectivity_index:zone_end]
            if len(line.split()) == 2 and all(token.lstrip("+-").isdigit() for token in line.split())
        ]
        if not edges:
            continue
        node_count = max(max(edge) for edge in edges) + 1
        if len(values) < 2 * node_count:
            continue
        adjacency = {index: [] for index in range(node_count)}
        for first, second in edges:
            adjacency[first].append(second)
            adjacency[second].append(first)
        endpoints = [index for index, neighbors in adjacency.items() if len(neighbors) == 1]
        current, previous, ordered = (endpoints[0] if endpoints else edges[0][0]), None, []
        while current is not None and len(ordered) <= node_count:
            ordered.append(current)
            candidates = [neighbor for neighbor in adjacency[current] if neighbor != previous]
            following = next((neighbor for neighbor in candidates if neighbor not in ordered), None)
            previous, current = current, following
        x_block, y_block = values[:node_count], values[node_count:2 * node_count]
        angle = math.radians(NACA0012_ROTATION_DEGREES)
        cosine, sine = math.cos(angle), math.sin(angle)
        center_x = NACA0012_ROTATION_CENTER_X
        rotated_x = []
        rotated_y = []
        for index in ordered:
            centered_x = x_block[index] - center_x
            rotated_x.append(center_x + centered_x * cosine - y_block[index] * sine)
            rotated_y.append(centered_x * sine + y_block[index] * cosine)
        roughness = roughness_key.replace("p", ".")
        traces.append(go.Scatter(
            x=rotated_x, y=rotated_y,
            mode="lines", fill="none",
            name=f"007 (MaxCCS, k<sub>s</sub> = {roughness} mm)",
            legendgroup=f"007_maxccs_{roughness_key}",
            line={"color": colors[roughness_key], "dash": "solid", "width": 4},
            marker={"size": 0}, meta={"ipw3_participant_id": "007", "ipw3_maxccs": True},
        ))
    return traces

# Independent dimensions for the three-panel ice-shape figures.
PANEL_WIDTH = 2400
PANEL_HEIGHT = 700
# Space reserved above the three-panel axes for the two AE3932 legend rows.
# These are pixel distances, so their separation stays readable when the
# aspect-fitted panel height changes.
PANEL_LEGEND_CLEARANCE_PX = 64
PANEL_LEGEND_ROW_GAP_PX = 52

# Add curated presentation figures using this format:
#
#   "CATEGORY/output_filename.png": (
#       "CASE_ID",
#       "generated_source_filename.png",
#   ),
#
# - CATEGORY is the destination folder under FIGURES, for example AERODYNAMIC,
#   HTC, ICE_SHAPES, ICE_ACCRETION, IMPINGEMENT, or SURF_TEMP_FF.
# - CASE_ID identifies the staging folder that creates the source plot, for
#   example TC_NACA0012_AE3932 or TC_NACA0012_AE3933.
# - generated_source_filename.png must exactly match the PNG filename produced
#   by the normal plot builder before it is copied into the curated list.
# - The final list contains participant IDs to omit from that figure. Use an
#   empty list to keep everyone. Example: ["001", "014"] excludes those two.
# - The boolean after excluded IDs controls legend visibility. Set it
#   to False for a legend-free presentation figure.
# - Grid-convergence plots may add a final dictionary with enabled, value,
#   color, dash, and width to draw a full-width horizontal median line.
# - The dictionary key is the final relative path under FIGURES. Legend
#   visibility does not change the filename.
#
# Example:
#   "ICE_ACCRETION/tc_naca0012_ae3933_ice_to_water_ratio_vs_n_required_bins15.png": (
#       "TC_NACA0012_AE3933",
#       "tc_naca0012_ae3933_ice_to_water_ratio_vs_n_required_bins15.png",
#       2000,
#       800,
#       ["001", "014"],
#       True,
#   )
NACA0012_PRESENTATION_FIGURES: dict[str, FigureSpec] = {
    # AERODYNAMIC
    # Surface pressure
    "AERODYNAMIC/tc_naca0012_ae3932_L1_cp_vs_x_slice_0p9144_all_roughness.png": ("TC_NACA0012_AE3932", "tc_naca0012_ae3932_L1_cp_vs_x_slice_0p9144_all_roughness.png", WIDTH, HEIGHT, [], True),
    "AERODYNAMIC/tc_naca0012_ae3932_L1_cp_vs_s_slice_0p9144_all_roughness.png": ("TC_NACA0012_AE3932", "tc_naca0012_ae3932_L1_cp_vs_s_slice_0p9144_all_roughness.png", WIDTH, HEIGHT, [], True),
    "AERODYNAMIC/tc_naca0012_ae3933_L1_cp_vs_x_slice_0p9144_all_roughness.png": ("TC_NACA0012_AE3933", "tc_naca0012_ae3933_L1_cp_vs_x_slice_0p9144_all_roughness.png", WIDTH, HEIGHT, [], True),

    # Grid sensitivity — absolute values
    "AERODYNAMIC/tc_naca0012_ae3932_cd_vs_n_all_roughness.png": ("TC_NACA0012_AE3932", "tc_naca0012_ae3932_cd_vs_n_all_roughness.png", WIDTH, HEIGHT, [], True),
    "AERODYNAMIC/tc_naca0012_ae3932_cl_vs_n_all_roughness.png": ("TC_NACA0012_AE3932", "tc_naca0012_ae3932_cl_vs_n_all_roughness.png", WIDTH, HEIGHT, ["006"], True,),
    "AERODYNAMIC/tc_naca0012_ae3932_cmy_vs_n_all_roughness.png": ("TC_NACA0012_AE3932", "tc_naca0012_ae3932_cmy_vs_n_all_roughness.png", WIDTH, HEIGHT, ["006"], True),

    # Grid sensitivity — difference from L1
    "AERODYNAMIC/tc_naca0012_ae3932_cd_vs_n_all_roughness_relative_to_l1.png": ("TC_NACA0012_AE3932", "tc_naca0012_ae3932_cd_vs_n_all_roughness_relative_to_l1.png", WIDTH, HEIGHT, [], True),
    "AERODYNAMIC/tc_naca0012_ae3932_cl_vs_n_all_roughness_relative_to_l1.png": ("TC_NACA0012_AE3932", "tc_naca0012_ae3932_cl_vs_n_all_roughness_relative_to_l1.png", WIDTH, HEIGHT, ["006"], True),

    # HTC - 3932
    # Surface distributions
    "HTC/tc_naca0012_ae3932_L1_htc_vs_s_slice_0p9144_all_roughness.png": ("TC_NACA0012_AE3932", "tc_naca0012_ae3932_L1_htc_vs_s_slice_0p9144_all_roughness.png", WIDTH, HEIGHT, [], True),
    "HTC/tc_naca0012_ae3932_L1_htc_vs_s_slice_0p9144_grouped_roughness.png": ("TC_NACA0012_AE3932", "tc_naca0012_ae3932_L1_htc_vs_s_slice_0p9144_grouped_roughness.png", WIDTH, HEIGHT, [], True),
    "HTC/tc_naca0012_ae3932_L1_htc_vs_s_slice_0p9144_grouped_turbulence_models.png": ("TC_NACA0012_AE3932", "tc_naca0012_ae3932_L1_htc_vs_s_slice_0p9144_grouped_turbulence_models.png", WIDTH, HEIGHT, [], True),
    "HTC/tc_naca0012_ae3932_L1_recovery_temperature_vs_s_slice_0p9144_all_roughness.png": ("TC_NACA0012_AE3932", "tc_naca0012_ae3932_L1_recovery_temperature_vs_s_slice_0p9144_all_roughness.png", WIDTH, HEIGHT, [], True),

    # Integrated heat-transfer grid sensitivity - 3932
    "HTC/tc_naca0012_ae3932_qc_prime_vs_n_y_0.9144.png": ("TC_NACA0012_AE3932", "tc_naca0012_ae3932_qc_prime_vs_n_y_0.9144.png", WIDTH, HEIGHT, ["001","006"], True),
    "HTC/tc_naca0012_ae3932_qc_prime_vs_n_y_0.9144_relative_to_l1.png": ("TC_NACA0012_AE3932", "tc_naca0012_ae3932_qc_prime_vs_n_y_0.9144_relative_to_l1.png", WIDTH, HEIGHT, ["001","006"], True),
    "HTC/tc_naca0012_ae3932_qc_prime_vs_n_y_0.9144_grouped_roughness.png": ("TC_NACA0012_AE3932", "tc_naca0012_ae3932_qc_prime_vs_n_y_0.9144_grouped_roughness.png", WIDTH, HEIGHT, ["001","006"], True),
    "HTC/tc_naca0012_ae3932_qc_prime_vs_n_y_0.9144_grouped_turbulence_models.png": ("TC_NACA0012_AE3932", "tc_naca0012_ae3932_qc_prime_vs_n_y_0.9144_grouped_turbulence_models.png", WIDTH, HEIGHT, ["001", "006"], True),

    # Integrated heat-transfer grid sensitivity - 3933
    "HTC/tc_naca0012_ae3933_L1_htc_vs_s_slice_0p9144_all_roughness.png": ("TC_NACA0012_AE3933", "tc_naca0012_ae3933_L1_htc_vs_s_slice_0p9144_all_roughness.png", WIDTH, HEIGHT, [], True),
    "HTC/tc_naca0012_ae3933_L1_htc_vs_s_slice_0p9144_grouped_roughness.png": ("TC_NACA0012_AE3933", "tc_naca0012_ae3933_L1_htc_vs_s_slice_0p9144_grouped_roughness.png", WIDTH, HEIGHT, [], True),
    "HTC/tc_naca0012_ae3933_L1_htc_vs_s_slice_0p9144_grouped_turbulence_models.png": ("TC_NACA0012_AE3933", "tc_naca0012_ae3933_L1_htc_vs_s_slice_0p9144_grouped_turbulence_models.png", WIDTH, HEIGHT, [], True),
    "HTC/tc_naca0012_ae3933_qc_prime_vs_n_y_0.9144.png": ("TC_NACA0012_AE3933", "tc_naca0012_ae3933_qc_prime_vs_n_y_0.9144.png", WIDTH, HEIGHT, ["001","006"], True),
    "HTC/tc_naca0012_ae3933_qc_prime_vs_n_y_0.9144_relative_to_l1.png": ("TC_NACA0012_AE3933", "tc_naca0012_ae3933_qc_prime_vs_n_y_0.9144_relative_to_l1.png", WIDTH, HEIGHT, ["001","006"], True),
    "HTC/tc_naca0012_ae3933_qc_prime_vs_n_y_0.9144_grouped_roughness.png": ("TC_NACA0012_AE3933", "tc_naca0012_ae3933_qc_prime_vs_n_y_0.9144_grouped_roughness.png", WIDTH, HEIGHT, ["001","006"], True),
    "HTC/tc_naca0012_ae3933_qc_prime_vs_n_y_0.9144_grouped_turbulence_models.png": ("TC_NACA0012_AE3933", "tc_naca0012_ae3933_qc_prime_vs_n_y_0.9144_grouped_turbulence_models.png", WIDTH, HEIGHT, ["001", "006"], True),

    # ICE ACCRETION — AE3932
    # Ice shape
    "ICE_SHAPES/tc_naca0012_ae3932_experimental_ice_shape.png": ("TC_NACA0012_AE3932", "tc_naca0012_ae3932_experimental_ice_shape.png", WIDTH, HEIGHT, [], True),
    "ICE_SHAPES/tc_naca0012_ae3932_finest_grid_multilayer_ice_shape_highest_bins.png": ("TC_NACA0012_AE3932", "tc_naca0012_ae3932_finest_grid_multilayer_ice_shape_highest_bins.png", WIDTH, HEIGHT, [], True),
    "ICE_SHAPES/tc_naca0012_ae3932_L1_multilayer_ice_shape_001_slice_0p9144_bins01.png": ("TC_NACA0012_AE3932", "tc_naca0012_ae3932_L1_multilayer_ice_shape_001_slice_0p9144_bins01.png", WIDTH, HEIGHT, [], True),
    "ICE_SHAPES/tc_naca0012_ae3932_L1_single_layer_ice_shape_slice_0p9144_bins15_roughness_unspecified.png": ("TC_NACA0012_AE3932", "tc_naca0012_ae3932_L1_single_layer_ice_shape_slice_0p9144_bins15_roughness_unspecified.png", WIDTH, HEIGHT, [], True),
    "ICE_SHAPES/tc_naca0012_ae3932_L1_single_layer_ice_shape_slice_0p9144_bins15_grouped_roughness.png": ("TC_NACA0012_AE3932", "tc_naca0012_ae3932_L1_single_layer_ice_shape_slice_0p9144_bins15_grouped_roughness.png", WIDTH, HEIGHT, [], True),
    "ICE_SHAPES/tc_naca0012_ae3932_L1_single_layer_ice_shape_slice_0p9144_bins15_grouped_turbulence_models.png": ("TC_NACA0012_AE3932", "tc_naca0012_ae3932_L1_single_layer_ice_shape_slice_0p9144_bins15_grouped_turbulence_models.png", WIDTH, HEIGHT, [], True),
    "ICE_SHAPES/tc_naca0012_ae3932_L2_single_layer_ice_shape_slice_0p9144_bins15_roughness_unspecified.png": ("TC_NACA0012_AE3932", "tc_naca0012_ae3932_L2_single_layer_ice_shape_slice_0p9144_bins15_roughness_unspecified.png", WIDTH, HEIGHT, [], True),
    "ICE_SHAPES/tc_naca0012_ae3932_L2_single_layer_ice_shape_slice_0p9144_bins15_grouped_roughness.png": ("TC_NACA0012_AE3932", "tc_naca0012_ae3932_L2_single_layer_ice_shape_slice_0p9144_bins15_grouped_roughness.png", WIDTH, HEIGHT, [], True),
    "ICE_SHAPES/tc_naca0012_ae3932_L2_single_layer_ice_shape_slice_0p9144_bins15_grouped_turbulence_models.png": ("TC_NACA0012_AE3932", "tc_naca0012_ae3932_L2_single_layer_ice_shape_slice_0p9144_bins15_grouped_turbulence_models.png", WIDTH, HEIGHT, [], True),
    "ICE_SHAPES/tc_naca0012_ae3932_L2_single_layer_ice_shape_slice_0p9144_bins15_grouped_roughness_panels.png": ("TC_NACA0012_AE3932", "tc_naca0012_ae3932_L2_single_layer_ice_shape_slice_0p9144_bins15_grouped_roughness_panels.png", PANEL_WIDTH, PANEL_HEIGHT, [], True),
    "ICE_SHAPES/tc_naca0012_ae3932_L2_single_layer_ice_shape_slice_0p9144_bins15_grouped_turbulence_models_panels.png": ("TC_NACA0012_AE3932", "tc_naca0012_ae3932_L2_single_layer_ice_shape_slice_0p9144_bins15_grouped_turbulence_models_panels.png", PANEL_WIDTH, PANEL_HEIGHT, [], True),

    # Water-fate ratios
    "ICE_ACCRETION/tc_naca0012_ae3932_ice_evap_to_water_ratio_vs_n_required_bins15.png": ("TC_NACA0012_AE3932", "tc_naca0012_ae3932_ice_evap_to_water_ratio_vs_n_required_bins15.png", WIDTH, HEIGHT, [], True),

    # Ice-mass grid and droplet-bin sensitivity
    "ICE_ACCRETION/tc_naca0012_ae3932_ice_mass_vs_n_unspecified_bins15_all_grid_levels.png": ("TC_NACA0012_AE3932", "tc_naca0012_ae3932_ice_mass_vs_n_unspecified_bins15_all_grid_levels.png", WIDTH, HEIGHT, [], True),
    "ICE_ACCRETION/tc_naca0012_ae3932_water_evap_mass_vs_n_unspecified_bins15_all_grid_levels.png": ("TC_NACA0012_AE3932", "tc_naca0012_ae3932_water_evap_mass_vs_n_unspecified_bins15_all_grid_levels.png", WIDTH, HEIGHT, [], True),
    "ICE_ACCRETION/tc_naca0012_ae3932_water_evap_mass_vs_n_unspecified_l1_vs_inverse_bins.png": ("TC_NACA0012_AE3932", "tc_naca0012_ae3932_water_evap_mass_vs_n_unspecified_l1_vs_inverse_bins.png", WIDTH, HEIGHT, [], True),
    "ICE_ACCRETION/tc_naca0012_ae3932_ice_mass_vs_n_unspecified_l1_vs_inverse_bins.png": ("TC_NACA0012_AE3932", "tc_naca0012_ae3932_ice_mass_vs_n_unspecified_l1_vs_inverse_bins.png", WIDTH, HEIGHT, ["001"], True),
    "ICE_ACCRETION/tc_naca0012_ae3932_ice_to_water_ratio_vs_n_required_bins15.png": ("TC_NACA0012_AE3932", "tc_naca0012_ae3932_ice_to_water_ratio_vs_n_required_bins15.png", WIDTH, HEIGHT, ["013"], True),

    # Upper-horn grid and droplet-bin sensitivity
    "ICE_ACCRETION/tc_naca0012_ae3932_upper_horn_angle_bins15.png": ("TC_NACA0012_AE3932", "tc_naca0012_ae3932_upper_horn_angle_bins15.png", WIDTH, HEIGHT, [], True),
    "ICE_ACCRETION/tc_naca0012_ae3932_upper_horn_angle_distribution_convergence_l1.png": ("TC_NACA0012_AE3932", "tc_naca0012_ae3932_upper_horn_angle_distribution_convergence_l1.png", WIDTH, HEIGHT, [], True),

    # ICE ACCRETION — AE3933
    "ICE_SHAPES/tc_naca0012_ae3933_experimental_ice_shape.png": ("TC_NACA0012_AE3933", "tc_naca0012_ae3933_experimental_ice_shape.png", WIDTH, HEIGHT, [], True),
    "ICE_SHAPES/tc_naca0012_ae3933_finest_grid_multilayer_ice_shape_highest_bins.png": ("TC_NACA0012_AE3933", "tc_naca0012_ae3933_finest_grid_multilayer_ice_shape_highest_bins.png", WIDTH, HEIGHT, [], True),
    "ICE_SHAPES/tc_naca0012_ae3933_L1_single_layer_ice_shape_slice_0p9144_bins15_grouped_roughness.png": ("TC_NACA0012_AE3933", "tc_naca0012_ae3933_L1_single_layer_ice_shape_slice_0p9144_bins15_grouped_roughness.png", WIDTH, HEIGHT, [], True),
    "ICE_SHAPES/tc_naca0012_ae3933_L1_single_layer_ice_shape_slice_0p9144_bins15_grouped_turbulence_models.png": ("TC_NACA0012_AE3933", "tc_naca0012_ae3933_L1_single_layer_ice_shape_slice_0p9144_bins15_grouped_turbulence_models.png", WIDTH, HEIGHT, [], True),
    "ICE_SHAPES/tc_naca0012_ae3933_L2_single_layer_ice_shape_slice_0p9144_bins15_grouped_roughness.png": ("TC_NACA0012_AE3933", "tc_naca0012_ae3933_L2_single_layer_ice_shape_slice_0p9144_bins15_grouped_roughness.png", WIDTH, HEIGHT, [], True),
    "ICE_SHAPES/tc_naca0012_ae3933_L2_single_layer_ice_shape_slice_0p9144_bins15_grouped_turbulence_models.png": ("TC_NACA0012_AE3933", "tc_naca0012_ae3933_L2_single_layer_ice_shape_slice_0p9144_bins15_grouped_turbulence_models.png", WIDTH, HEIGHT, [], True),
    "ICE_SHAPES/tc_naca0012_ae3933_L2_single_layer_ice_shape_slice_0p9144_bins15_grouped_roughness_panels.png": ("TC_NACA0012_AE3933", "tc_naca0012_ae3933_L2_single_layer_ice_shape_slice_0p9144_bins15_grouped_roughness_panels.png", PANEL_WIDTH, PANEL_HEIGHT, [], True),
    "ICE_SHAPES/tc_naca0012_ae3933_L2_single_layer_ice_shape_slice_0p9144_bins15_grouped_turbulence_models_panels.png": ("TC_NACA0012_AE3933", "tc_naca0012_ae3933_L2_single_layer_ice_shape_slice_0p9144_bins15_grouped_turbulence_models_panels.png", PANEL_WIDTH, PANEL_HEIGHT, [], True),

    # Ice shapes
    "ICE_SHAPES/tc_naca0012_ae3933_L2_ice_shape_001_bins01_008_bins07.png": ("TC_NACA0012_AE3933", "tc_naca0012_ae3933_L2_ice_shape_001_bins01_008_bins07.png", WIDTH, HEIGHT, [], True),
    "ICE_SHAPES/tc_naca0012_ae3933_L1_single_layer_ice_shape_slice_0p9144_bins15_roughness_unspecified.png": ("TC_NACA0012_AE3933", "tc_naca0012_ae3933_L1_single_layer_ice_shape_slice_0p9144_bins15_roughness_unspecified.png", WIDTH, HEIGHT, [], True),
    "ICE_SHAPES/tc_naca0012_ae3933_L2_single_layer_ice_shape_slice_0p9144_bins15_roughness_unspecified.png": ("TC_NACA0012_AE3933", "tc_naca0012_ae3933_L2_single_layer_ice_shape_slice_0p9144_bins15_roughness_unspecified.png", WIDTH, HEIGHT, [], True),

    # Comparison with AE3932
    "ICE_ACCRETION/tc_naca0012_ae3933_comparison_with_3932_ice_mass_bins15_difference.png": ("TC_NACA0012_AE3933", "tc_naca0012_ae3933_comparison_with_3932_ice_mass_bins15_difference.png", WIDTH, HEIGHT, [], True),
    "ICE_ACCRETION/tc_naca0012_ae3933_comparison_with_3932_ice_mass_bins15_percent.png": ("TC_NACA0012_AE3933", "tc_naca0012_ae3933_comparison_with_3932_ice_mass_bins15_percent.png", WIDTH, HEIGHT, [], True),

    # Water-fate ratios
    "ICE_ACCRETION/tc_naca0012_ae3933_ice_evap_to_water_ratio_vs_n_required_bins15.png": ("TC_NACA0012_AE3933", "tc_naca0012_ae3933_ice_evap_to_water_ratio_vs_n_required_bins15.png", WIDTH, HEIGHT, [], True),

    # Ice-mass grid and droplet-bin sensitivity
    "ICE_ACCRETION/tc_naca0012_ae3933_ice_mass_vs_n_unspecified_bins15_all_grid_levels.png": ("TC_NACA0012_AE3933", "tc_naca0012_ae3933_ice_mass_vs_n_unspecified_bins15_all_grid_levels.png", WIDTH, HEIGHT, [], True),
    "ICE_ACCRETION/tc_naca0012_ae3933_water_evap_mass_vs_n_unspecified_bins15_all_grid_levels.png": ("TC_NACA0012_AE3933", "tc_naca0012_ae3933_water_evap_mass_vs_n_unspecified_bins15_all_grid_levels.png", WIDTH, HEIGHT, [], True),
    "ICE_ACCRETION/tc_naca0012_ae3933_water_evap_mass_vs_n_unspecified_l1_vs_inverse_bins.png": ("TC_NACA0012_AE3933", "tc_naca0012_ae3933_water_evap_mass_vs_n_unspecified_l1_vs_inverse_bins.png", WIDTH, HEIGHT, [], True),
    "ICE_ACCRETION/tc_naca0012_ae3933_ice_to_water_ratio_vs_n_required_bins15.png": ("TC_NACA0012_AE3933", "tc_naca0012_ae3933_ice_to_water_ratio_vs_n_required_bins15.png", WIDTH, HEIGHT, ["001"], True),

    # Upper-horn grid and droplet-bin sensitivity
    "ICE_ACCRETION/tc_naca0012_ae3933_upper_horn_angle_bins15.png": ("TC_NACA0012_AE3933", "tc_naca0012_ae3933_upper_horn_angle_bins15.png", WIDTH, HEIGHT, [], True),
    "ICE_ACCRETION/tc_naca0012_ae3933_upper_horn_angle_distribution_convergence_l1.png": ("TC_NACA0012_AE3933", "tc_naca0012_ae3933_upper_horn_angle_distribution_convergence_l1.png", WIDTH, HEIGHT, ["013"], True),
    "ICE_ACCRETION/tc_naca0012_ae3933_ice_mass_vs_n_unspecified_l1_vs_inverse_bins.png": ("TC_NACA0012_AE3933", "tc_naca0012_ae3933_ice_mass_vs_n_unspecified_l1_vs_inverse_bins.png", WIDTH, HEIGHT, [], True),

    # IMPINGEMENT
    # Collection-efficiency distribution
    "IMPINGEMENT/tc_naca0012_ae3932_L1_beta_bins15_vs_s_slice_0p9144_all_roughness.png": ("TC_NACA0012_AE3932", "tc_naca0012_ae3932_L1_beta_bins15_vs_s_slice_0p9144_all_roughness.png", WIDTH, HEIGHT, [], True),

    # Peak collection efficiency — absolute, grid difference, and bin difference
    "IMPINGEMENT/tc_naca0012_ae3932_beta_max_bins15.png": ("TC_NACA0012_AE3932", "tc_naca0012_ae3932_beta_max_bins15.png", WIDTH, HEIGHT, [], True),
    "IMPINGEMENT/tc_naca0012_ae3932_beta_max_bins15_relative_to_l1.png": ("TC_NACA0012_AE3932", "tc_naca0012_ae3932_beta_max_bins15_relative_to_l1.png", WIDTH, HEIGHT, [], True),
    "IMPINGEMENT/tc_naca0012_ae3932_beta_max_distribution_convergence_l1_relative_to_bins15.png": ("TC_NACA0012_AE3932", "tc_naca0012_ae3932_beta_max_distribution_convergence_l1_relative_to_bins15.png", WIDTH, HEIGHT, [], True),
    "IMPINGEMENT/tc_naca0012_ae3932_beta_max_distribution_convergence_l1.png": ("TC_NACA0012_AE3932", "tc_naca0012_ae3932_beta_max_distribution_convergence_l1.png", WIDTH, HEIGHT, [], True),

    # Peak position — absolute, grid difference, and bin difference
    "IMPINGEMENT/tc_naca0012_ae3932_s_peak_bins15.png": ("TC_NACA0012_AE3932", "tc_naca0012_ae3932_s_peak_bins15.png", WIDTH, HEIGHT, [], True),
    "IMPINGEMENT/tc_naca0012_ae3932_s_peak_bins15_relative_to_l1.png": ("TC_NACA0012_AE3932", "tc_naca0012_ae3932_s_peak_bins15_relative_to_l1.png", WIDTH, HEIGHT, [], True),
    "IMPINGEMENT/tc_naca0012_ae3932_s_peak_distribution_convergence_l1_relative_to_bins15.png": ("TC_NACA0012_AE3932", "tc_naca0012_ae3932_s_peak_distribution_convergence_l1_relative_to_bins15.png", WIDTH, HEIGHT, [], True),

    # Impingement width — absolute, grid difference, and bin difference
    "IMPINGEMENT/tc_naca0012_ae3932_width_bins15.png": ("TC_NACA0012_AE3932", "tc_naca0012_ae3932_width_bins15.png", WIDTH, HEIGHT, ["003", "006"], True),
    "IMPINGEMENT/tc_naca0012_ae3932_width_bins15_relative_to_l1.png": ("TC_NACA0012_AE3932", "tc_naca0012_ae3932_width_bins15_relative_to_l1.png", WIDTH, HEIGHT, ["006"], True),
    "IMPINGEMENT/tc_naca0012_ae3932_width_distribution_convergence_l1_relative_to_bins15.png": ("TC_NACA0012_AE3932", "tc_naca0012_ae3932_width_distribution_convergence_l1_relative_to_bins15.png", WIDTH, HEIGHT, [], True),
    "IMPINGEMENT/tc_naca0012_ae3932_width_distribution_convergence_l1.png": ("TC_NACA0012_AE3932", "tc_naca0012_ae3932_width_distribution_convergence_l1.png", WIDTH, HEIGHT, [], True),

    # Total water mass — grid and droplet-bin sensitivity
    "IMPINGEMENT/tc_naca0012_ae3932_water_mass_vs_n_unspecified_bins15_all_grid_levels.png": ("TC_NACA0012_AE3932", "tc_naca0012_ae3932_water_mass_vs_n_unspecified_bins15_all_grid_levels.png", WIDTH, HEIGHT, ["013"], True),
    "IMPINGEMENT/tc_naca0012_ae3932_water_mass_vs_n_unspecified_bins15_all_grid_levels_relative_to_l1.png": ("TC_NACA0012_AE3932", "tc_naca0012_ae3932_water_mass_vs_n_unspecified_bins15_all_grid_levels_relative_to_l1.png", WIDTH, HEIGHT, ["001", "013"], True),
    "IMPINGEMENT/tc_naca0012_ae3932_water_mass_vs_n_unspecified_l1_vs_inverse_bins.png": ("TC_NACA0012_AE3932", "tc_naca0012_ae3932_water_mass_vs_n_unspecified_l1_vs_inverse_bins.png", WIDTH, HEIGHT, ["001", "015", "013"], True),
    "IMPINGEMENT/tc_naca0012_ae3932_water_mass_vs_n_unspecified_l2_vs_inverse_bins.png": ("TC_NACA0012_AE3932", "tc_naca0012_ae3932_water_mass_vs_n_unspecified_l2_vs_inverse_bins.png", WIDTH, HEIGHT, ["001", "015"], True),
    "IMPINGEMENT/tc_naca0012_ae3932_water_mass_vs_n_unspecified_l3_vs_inverse_bins.png": ("TC_NACA0012_AE3932", "tc_naca0012_ae3932_water_mass_vs_n_unspecified_l3_vs_inverse_bins.png", WIDTH, HEIGHT, ["001", "015"], True),
    "IMPINGEMENT/tc_naca0012_ae3932_water_mass_vs_n_unspecified_l4_vs_inverse_bins.png": ("TC_NACA0012_AE3932", "tc_naca0012_ae3932_water_mass_vs_n_unspecified_l4_vs_inverse_bins.png", WIDTH, HEIGHT, ["001", "015"], True),
    "IMPINGEMENT/tc_naca0012_ae3932_water_mass_vs_n_unspecified_l1_vs_inverse_bins_relative_to_bins15.png": ("TC_NACA0012_AE3932", "tc_naca0012_ae3932_water_mass_vs_n_unspecified_l1_vs_inverse_bins_relative_to_bins15.png", WIDTH, HEIGHT, ["013"], True),
    "IMPINGEMENT/tc_naca0012_ae3932_water_mass_vs_n_unspecified_all_grid_levels_vs_inverse_bins.png": ("TC_NACA0012_AE3932", "tc_naca0012_ae3932_water_mass_vs_n_unspecified_all_grid_levels_vs_inverse_bins.png", WIDTH, HEIGHT, ["001", "015"], True),
    "IMPINGEMENT/tc_naca0012_ae3933_water_mass_vs_n_unspecified_all_grid_levels_vs_inverse_bins.png": ("TC_NACA0012_AE3933", "tc_naca0012_ae3933_water_mass_vs_n_unspecified_all_grid_levels_vs_inverse_bins.png", WIDTH, HEIGHT, ["001", "015"], True),

    # Diameter-resolved across-participant variation
    "IMPINGEMENT/tc_naca0012_ae3932_water_mass_relative_iqr_vs_droplet_diameter_bins15_all_grid_levels.png": ("TC_NACA0012_AE3932", "tc_naca0012_ae3932_water_mass_relative_iqr_vs_droplet_diameter_bins15_all_grid_levels.png", WIDTH, HEIGHT, [], True),
    "IMPINGEMENT/tc_naca0012_ae3932_water_mass_coefficient_of_variation_vs_droplet_diameter_bins15_all_grid_levels.png": ("TC_NACA0012_AE3932", "tc_naca0012_ae3932_water_mass_coefficient_of_variation_vs_droplet_diameter_bins15_all_grid_levels.png", WIDTH, HEIGHT, [], True),
    "IMPINGEMENT/tc_naca0012_ae3933_water_mass_relative_iqr_vs_droplet_diameter_bins15_all_grid_levels.png": ("TC_NACA0012_AE3933", "tc_naca0012_ae3933_water_mass_relative_iqr_vs_droplet_diameter_bins15_all_grid_levels.png", WIDTH, HEIGHT, [], True),
    "IMPINGEMENT/tc_naca0012_ae3933_water_mass_coefficient_of_variation_vs_droplet_diameter_bins15_all_grid_levels.png": ("TC_NACA0012_AE3933", "tc_naca0012_ae3933_water_mass_coefficient_of_variation_vs_droplet_diameter_bins15_all_grid_levels.png", WIDTH, HEIGHT, [], True),

    # SURFACE TEMPERATURE / FREEZING FRACTION
    # AE3932 surface distributions
    "SURF_TEMP_FF/tc_naca0012_ae3932_L1_freezing_fraction_vs_s_slice_0p9144_all_roughness.png": ("TC_NACA0012_AE3932", "tc_naca0012_ae3932_L1_freezing_fraction_vs_s_slice_0p9144_all_roughness.png", WIDTH, HEIGHT, [], True),
    "SURF_TEMP_FF/tc_naca0012_ae3932_L1_freezing_fraction_vs_s_slice_0p9144_grouped_roughness.png": ("TC_NACA0012_AE3932", "tc_naca0012_ae3932_L1_freezing_fraction_vs_s_slice_0p9144_grouped_roughness.png", WIDTH, HEIGHT, [], True),
    "SURF_TEMP_FF/tc_naca0012_ae3932_L1_freezing_fraction_vs_s_slice_0p9144_grouped_thermodynamics_models.png": ("TC_NACA0012_AE3932", "tc_naca0012_ae3932_L1_freezing_fraction_vs_s_slice_0p9144_grouped_thermodynamics_models.png", WIDTH, HEIGHT, [], True),
    "SURF_TEMP_FF/tc_naca0012_ae3932_L1_surface_temperature_vs_s_slice_0p9144_all_roughness.png": ("TC_NACA0012_AE3932", "tc_naca0012_ae3932_L1_surface_temperature_vs_s_slice_0p9144_all_roughness.png", WIDTH, HEIGHT, ["006"], True),
    "SURF_TEMP_FF/tc_naca0012_ae3932_L1_surface_temperature_vs_s_slice_0p9144_grouped_roughness.png": ("TC_NACA0012_AE3932", "tc_naca0012_ae3932_L1_surface_temperature_vs_s_slice_0p9144_grouped_roughness.png", WIDTH, HEIGHT, [], True),
    "SURF_TEMP_FF/tc_naca0012_ae3932_L1_surface_temperature_vs_s_slice_0p9144_grouped_thermodynamics_models.png": ("TC_NACA0012_AE3932", "tc_naca0012_ae3932_L1_surface_temperature_vs_s_slice_0p9144_grouped_thermodynamics_models.png", WIDTH, HEIGHT, [], True),
    "SURF_TEMP_FF/tc_naca0012_ae3932_mean_surface_temperature_vs_n_slice_0p9144_all_roughness.png": ("TC_NACA0012_AE3932", "tc_naca0012_ae3932_mean_surface_temperature_vs_n_slice_0p9144_all_roughness.png", WIDTH, HEIGHT, [], True),
    "SURF_TEMP_FF/tc_naca0012_ae3932_mean_freezing_fraction_vs_n_slice_0p9144_all_roughness.png": ("TC_NACA0012_AE3932", "tc_naca0012_ae3932_mean_freezing_fraction_vs_n_slice_0p9144_all_roughness.png", WIDTH, HEIGHT, [], True),

    # AE3933 surface distributions
    "SURF_TEMP_FF/tc_naca0012_ae3933_L1_freezing_fraction_vs_s_slice_0p9144_all_roughness.png": ("TC_NACA0012_AE3933", "tc_naca0012_ae3933_L1_freezing_fraction_vs_s_slice_0p9144_all_roughness.png", WIDTH, HEIGHT, [], True),
    "SURF_TEMP_FF/tc_naca0012_ae3933_L1_freezing_fraction_vs_s_slice_0p9144_grouped_roughness.png": ("TC_NACA0012_AE3933", "tc_naca0012_ae3933_L1_freezing_fraction_vs_s_slice_0p9144_grouped_roughness.png", WIDTH, HEIGHT, [], True),
    "SURF_TEMP_FF/tc_naca0012_ae3933_L1_freezing_fraction_vs_s_slice_0p9144_grouped_thermodynamics_models.png": ("TC_NACA0012_AE3933", "tc_naca0012_ae3933_L1_freezing_fraction_vs_s_slice_0p9144_grouped_thermodynamics_models.png", WIDTH, HEIGHT, [], True),
    "SURF_TEMP_FF/""tc_naca0012_ae3933_L1_surface_temperature_vs_s_slice_0p9144_all_roughness.png": ("TC_NACA0012_AE3933", "tc_naca0012_ae3933_L1_surface_temperature_vs_s_slice_0p9144_all_roughness.png", WIDTH, HEIGHT, ["006"], True),
    "SURF_TEMP_FF/tc_naca0012_ae3933_L1_surface_temperature_vs_s_slice_0p9144_grouped_roughness.png": ("TC_NACA0012_AE3933", "tc_naca0012_ae3933_L1_surface_temperature_vs_s_slice_0p9144_grouped_roughness.png", WIDTH, HEIGHT, [], True),
    "SURF_TEMP_FF/tc_naca0012_ae3933_L1_surface_temperature_vs_s_slice_0p9144_grouped_thermodynamics_models.png": ("TC_NACA0012_AE3933", "tc_naca0012_ae3933_L1_surface_temperature_vs_s_slice_0p9144_grouped_thermodynamics_models.png", WIDTH, HEIGHT, [], True),
    "SURF_TEMP_FF/tc_naca0012_ae3933_mean_surface_temperature_vs_n_slice_0p9144_all_roughness.png": ("TC_NACA0012_AE3933", "tc_naca0012_ae3933_mean_surface_temperature_vs_n_slice_0p9144_all_roughness.png", WIDTH, HEIGHT, [], True),
    "SURF_TEMP_FF/tc_naca0012_ae3933_mean_freezing_fraction_vs_n_slice_0p9144_all_roughness.png": ("TC_NACA0012_AE3933", "tc_naca0012_ae3933_mean_freezing_fraction_vs_n_slice_0p9144_all_roughness.png", WIDTH, HEIGHT, [], True),

    # Ice Horns
    # AE3932 horn-construction methods
    "ICE_HORNS/tc_naca0012_ae3932_maxccs_upper_horn_angle_method.png": ("TC_NACA0012_AE3932", "tc_naca0012_ae3932_maxccs_upper_horn_angle_method.png", WIDTH, HEIGHT, [], False),
    "ICE_HORNS/tc_naca0012_ae3932_meanccs_upper_horn_angle_method.png": ("TC_NACA0012_AE3932", "tc_naca0012_ae3932_meanccs_upper_horn_angle_method.png", WIDTH, HEIGHT, [], False),
    "ICE_HORNS/tc_naca0012_ae3932_minccs_upper_horn_angle_method.png": ("TC_NACA0012_AE3932", "tc_naca0012_ae3932_minccs_upper_horn_angle_method.png", WIDTH, HEIGHT, [], False),

    # AE3933 horn-construction methods
    "ICE_HORNS/tc_naca0012_ae3933_upper_horn_angle_by_participant_bins15_all_grid_levels.png": ("TC_NACA0012_AE3933", "tc_naca0012_ae3933_upper_horn_angle_by_participant_bins15_all_grid_levels.png", WIDTH, HEIGHT, [], True),
    "ICE_HORNS/tc_naca0012_ae3933_upper_horn_angle_by_participant_bins15_all_grid_levels_grouped_roughness.png": ("TC_NACA0012_AE3933", "tc_naca0012_ae3933_upper_horn_angle_by_participant_bins15_all_grid_levels_grouped_roughness.png", WIDTH, HEIGHT, [], True),
    "ICE_HORNS/tc_naca0012_ae3933_maxccs_upper_horn_angle_method.png": ("TC_NACA0012_AE3933", "tc_naca0012_ae3933_maxccs_upper_horn_angle_method.png", WIDTH, HEIGHT, [], False),
    "ICE_HORNS/tc_naca0012_ae3933_meanccs_upper_horn_angle_method.png": ("TC_NACA0012_AE3933", "tc_naca0012_ae3933_meanccs_upper_horn_angle_method.png", WIDTH, HEIGHT, [], False),
    "ICE_HORNS/tc_naca0012_ae3933_minccs_upper_horn_angle_method.png": ("TC_NACA0012_AE3933", "tc_naca0012_ae3933_minccs_upper_horn_angle_method.png", WIDTH, HEIGHT, [], False),
}

for _grid_level in ("L1", "L2", "L3", "L4"):
    _filename = (
        f"tc_naca0012_ae3932_{_grid_level}_beta_bins15_vs_s_slice_0p9144_"
        "grouped_droplet_models.png"
    )
    NACA0012_PRESENTATION_FIGURES[f"IMPINGEMENT/{_filename}"] = (
        "TC_NACA0012_AE3932", _filename, WIDTH, HEIGHT, [], True,
    )


# Figures intentionally omitted from the curated NACA0012 presentation export.
# Keep this list centralized so dynamically assembled queues cannot silently
# restore a retired image on a later rebuild.
NACA0012_EXCLUDED_PRESENTATION_STEMS = {
    "tc_naca0012_ae3933_L1_cp_vs_x_slice_0p9144_all_roughness",
    "tc_naca0012_ae3932_L1_htc_vs_s_slice_0p9144_grouped_roughness",
    "tc_naca0012_ae3932_qc_prime_vs_n_y_0.9144_grouped_roughness",
    "tc_naca0012_ae3932_qc_prime_vs_n_y_0.9144_relative_to_l1",
    "tc_naca0012_ae3933_L1_htc_vs_s_slice_0p9144_grouped_roughness",
    "tc_naca0012_ae3933_qc_prime_vs_n_y_0.9144_grouped_roughness",
    "tc_naca0012_ae3933_qc_prime_vs_n_y_0.9144_relative_to_l1",
    "tc_naca0012_ae3932_ice_evap_to_water_ratio_vs_n_required_bins15",
    "tc_naca0012_ae3932_water_evap_mass_vs_n_unspecified_bins15_all_grid_levels",
    "tc_naca0012_ae3932_water_evap_mass_vs_n_unspecified_l1_vs_inverse_bins",
    "tc_naca0012_ae3933_ice_evap_to_water_ratio_vs_n_required_bins15",
    "tc_naca0012_ae3933_water_evap_mass_vs_n_unspecified_bins15_all_grid_levels",
    "tc_naca0012_ae3933_water_evap_mass_vs_n_unspecified_l1_vs_inverse_bins",
    "tc_naca0012_ae3933_upper_horn_angle_by_participant_bins15_all_grid_levels_grouped_roughness",
    "tc_naca0012_ae3933_upper_horn_angle_by_participant_bins15_all_grid_levels",
    "tc_naca0012_ae3932_L1_single_layer_ice_shape_slice_0p9144_bins15_grouped_roughness",
    "tc_naca0012_ae3932_L2_single_layer_ice_shape_slice_0p9144_bins15_grouped_roughness_panels",
    "tc_naca0012_ae3932_L2_single_layer_ice_shape_slice_0p9144_bins15_grouped_roughness",
    "tc_naca0012_ae3933_L1_single_layer_ice_shape_slice_0p9144_bins15_grouped_roughness",
    "tc_naca0012_ae3933_L2_single_layer_ice_shape_slice_0p9144_bins15_grouped_roughness_panels",
    "tc_naca0012_ae3933_L2_single_layer_ice_shape_slice_0p9144_bins15_grouped_roughness",
    "tc_naca0012_ae3932_L1_beta_bins15_vs_s_slice_0p9144_grouped_droplet_models",
    "tc_naca0012_ae3932_L2_beta_bins15_vs_s_slice_0p9144_grouped_droplet_models",
    "tc_naca0012_ae3932_L3_beta_bins15_vs_s_slice_0p9144_grouped_droplet_models",
    "tc_naca0012_ae3932_L4_beta_bins15_vs_s_slice_0p9144_grouped_droplet_models",
    "tc_naca0012_ae3932_s_peak_bins15_relative_to_l1",
    "tc_naca0012_ae3932_s_peak_bins15",
    "tc_naca0012_ae3932_s_peak_distribution_convergence_l1_relative_to_bins15",
    "tc_naca0012_ae3932_water_mass_vs_n_unspecified_all_grid_levels_vs_inverse_bins",
    "tc_naca0012_ae3932_water_mass_vs_n_unspecified_l2_vs_inverse_bins",
    "tc_naca0012_ae3932_water_mass_vs_n_unspecified_l3_vs_inverse_bins",
    "tc_naca0012_ae3932_water_mass_vs_n_unspecified_l4_vs_inverse_bins",
    "tc_naca0012_ae3933_water_mass_coefficient_of_variation_vs_droplet_diameter_bins15_all_grid_levels",
    "tc_naca0012_ae3933_water_mass_relative_iqr_vs_droplet_diameter_bins15_all_grid_levels",
    "tc_naca0012_ae3933_water_mass_vs_n_unspecified_all_grid_levels_vs_inverse_bins",
    "tc_naca0012_ae3932_L1_freezing_fraction_vs_s_slice_0p9144_grouped_roughness",
    "tc_naca0012_ae3932_L1_surface_temperature_vs_s_slice_0p9144_grouped_roughness",
    "tc_naca0012_ae3932_mean_freezing_fraction_vs_n_slice_0p9144_all_roughness",
    "tc_naca0012_ae3933_L1_freezing_fraction_vs_s_slice_0p9144_grouped_roughness",
    "tc_naca0012_ae3933_L1_surface_temperature_vs_s_slice_0p9144_grouped_roughness",
    "tc_naca0012_ae3933_mean_freezing_fraction_vs_n_slice_0p9144_all_roughness",
}

NACA0012_PRESENTATION_FIGURES = {
    destination: spec
    for destination, spec in NACA0012_PRESENTATION_FIGURES.items()
    if Path(destination).stem not in NACA0012_EXCLUDED_PRESENTATION_STEMS
}


def queue_finest_multilayer_ice_shapes(participants, staging_dir: Path) -> None:
    """Queue one highest-resolution/highest-bin multilayer comparison per case."""
    import plotly.graph_objects as go

    for case_id in ("TC_NACA0012_AE3932", "TC_NACA0012_AE3933"):
        cell_counts = convergence_data_builder.grid_cell_counts_for_case(case_id)
        finest_first = [f"L{level}" for level in sorted(cell_counts, key=cell_counts.get, reverse=True)]
        selected_participants: set[str] = set()
        combined: go.Figure | None = None
        reference_traces: dict[str, object] = {}

        for grid_level in finest_first:
            level_figure, _, _ = iceshape_builder.build_multilayer_ice_shape_figure(
                participants, case_id, grid_level,
                slice_filter=0.9144, bins_filter=None, roughness_filter=None,
            )
            if combined is None:
                combined = go.Figure(layout=level_figure.layout)

            candidates: dict[str, list[tuple[int, object, str, str]]] = {}
            for trace in level_figure.data:
                group = str(trace.legendgroup or "")
                if group == "clean_reference" or group.startswith(("experimental_", "reference_")):
                    reference_traces.setdefault(group, trace)
                    continue
                participant_match = re.match(r"^(\d{1,3})(?=\D|$)", str(trace.name or ""))
                distribution_match = re.search(r"Distribution: (\d+)", str(trace.hovertemplate or ""))
                if participant_match is None or distribution_match is None:
                    continue
                participant_id = participant_match.group(1).zfill(3)
                if participant_id in selected_participants:
                    continue
                roughness_match = re.search(r"Roughness: ([^<]+)", str(trace.hovertemplate or ""))
                data_type_match = re.search(r"DATA_TYPE: ([^<]+)", str(trace.hovertemplate or ""))
                condition = " | ".join(
                    value for value in (
                        roughness_match.group(1) if roughness_match else "",
                        data_type_match.group(1) if data_type_match else "",
                    ) if value and value.lower() != "unknown"
                )
                candidates.setdefault(participant_id, []).append(
                    (int(distribution_match.group(1)), trace, condition, str(trace.name or ""))
                )

            for participant_id, participant_candidates in sorted(candidates.items()):
                highest_bins = max(item[0] for item in participant_candidates)
                highest_bin_candidates = [item for item in participant_candidates if item[0] == highest_bins]
                variable = [item for item in highest_bin_candidates if "variable" in (item[2] + item[3]).lower()]
                standard = [item for item in highest_bin_candidates if item not in variable]
                keep = ([standard[0]] if standard else []) + variable if variable else [highest_bin_candidates[0]]
                for bins, trace, condition, _ in keep:
                    trace.name = f"{participant_id} | {grid_level} | {bins:02d} bins" + (
                        f" | {condition}" if condition else ""
                    )
                    trace.legendgroup = f"multilayer_{participant_id}_{grid_level}_{bins}_{condition}"
                    combined.add_trace(trace)
                selected_participants.add(participant_id)

        if combined is None or not selected_participants:
            raise RuntimeError(f"No multilayer ice shapes found for {case_id}")
        for trace in reference_traces.values():
            combined.add_trace(trace)
        # Plotly draws later traces on top. Keep the experimental envelope below
        # every submitted contour (including 001) for both AE3932 and AE3933,
        # while retaining the clean airfoil as the topmost reference outline.
        experimental = tuple(
            trace for trace in combined.data
            if str(trace.legendgroup or "").startswith(("experimental_", "reference_"))
        )
        submitted = tuple(
            trace for trace in combined.data
            if not str(trace.legendgroup or "").startswith(("experimental_", "reference_"))
            and trace.legendgroup != "clean_reference"
        )
        clean_reference = tuple(
            trace for trace in combined.data if trace.legendgroup == "clean_reference"
        )
        combined.data = experimental + submitted + clean_reference
        filename = f"{case_id.lower()}_finest_grid_multilayer_ice_shape_highest_bins"
        iceshape_builder.set_png_export_dir(staging_dir / case_id)
        iceshape_builder.figure_to_html_div(
            combined, filename,
            f"{case_id.removeprefix('TC_NACA0012_')} | Final multilayer ice shapes",
        )


def _filter_queue_for_participant(queue, participant_id: str) -> None:
    """Keep figures containing this participant and remove other participants."""
    selected = str(participant_id).zfill(3)
    kept = []
    for figure, path in queue:
        participant_traces = []
        selected_traces = []
        for trace in figure.data:
            meta = trace.meta if isinstance(trace.meta, dict) else {}
            pid = str(meta.get("ipw3_participant_id", "")).zfill(3) if meta.get("ipw3_participant_id") else ""
            if not pid:
                match = re.match(r"^(\d{1,3})(?=\D|$)", str(trace.name or "").strip())
                pid = match.group(1).zfill(3) if match else ""
            if pid:
                participant_traces.append(trace)
                if pid == selected:
                    selected_traces.append(trace)
        if not selected_traces:
            continue
        participant_trace_ids = {id(trace) for trace in participant_traces}
        selected_trace_ids = {id(trace) for trace in selected_traces}
        figure.data = tuple(
            trace for trace in figure.data
            if id(trace) not in participant_trace_ids or id(trace) in selected_trace_ids
        )
        kept.append((figure, path))
    queue[:] = kept


def _add_main_and_inset_borders(figure) -> None:
    """Draw explicit plot frames that survive Plotly static export."""
    figure.update_xaxes(showline=True, mirror=True, linecolor="black", linewidth=2)
    figure.update_yaxes(showline=True, mirror=True, linecolor="black", linewidth=2)
    figure.add_shape(
        type="rect", xref="x domain", yref="y domain",
        x0=0, x1=1, y0=0, y1=1,
        line={"color": "black", "width": 2},
        fillcolor="rgba(0,0,0,0)", layer="above",
    )
    if (
        getattr(figure.layout, "xaxis2", None) is not None
        and getattr(figure.layout, "yaxis2", None) is not None
    ):
        figure.add_shape(
            type="rect", xref="x2 domain", yref="y2 domain",
            x0=0, x1=1, y0=0, y1=1,
            line={"color": "black", "width": 2},
            fillcolor="rgba(0,0,0,0)", layer="above",
        )


def _add_configured_grid_median_line(figure, settings: MedianLineConfig | None) -> None:
    """Add the optional full-width median line to a grid-refinement plot."""
    if not settings or not settings.get("enabled") or settings.get("value") is None:
        return
    figure.add_hline(
        y=float(settings["value"]),
        line_color=str(settings["color"]),
        line_dash=str(settings["dash"]),
        line_width=float(settings["width"]),
        layer="above",
    )


def _replace_cp_with_champs_grid_levels(queue, target_names: set[str]) -> None:
    """Replace participant Cp traces with CHAMPS L1-L4 curves."""
    colors = {
        "L1": "#1f77b4", "L2": "#2ca02c",
        "L3": "#ff7f0e", "L4": "#d62728",
    }
    figures = {path.name: figure for figure, path in queue}
    for target_name in target_names:
        target = figures.get(target_name)
        if target is None:
            raise RuntimeError(f"Missing CHAMPS Cp presentation plot: {target_name}")
        sources = {}
        for level in colors:
            source_name = target_name.replace("_L1_", f"_{level}_", 1)
            source = figures.get(source_name)
            if source is None:
                raise RuntimeError(f"Missing CHAMPS Cp grid source: {source_name}")
            sources[level] = [
                copy.deepcopy(trace) for trace in source.data
                if get_trace_participant_id(trace) == "007"
            ]
        target.data = tuple(
            trace for trace in target.data if not get_trace_participant_id(trace)
        )
        for trace in target.data:
            is_experimental = (
                str(trace.legendgroup or "").startswith(("experimental_", "reference_"))
                or str(trace.name or "").lower().startswith("exp")
            )
            if is_experimental:
                trace.zorder = 10
        for level in ("L4", "L3", "L2", "L1"):
            color = colors[level]
            source_traces = sources[level]
            if not source_traces:
                raise RuntimeError(f"Missing CHAMPS {level} Cp trace for {target_name}")
            legend_shown = False
            for trace in source_traces:
                trace.name = level
                trace.legendgroup = f"007_cp_{level.lower()}"
                trace.legendrank = int(level[1:])
                trace.zorder = 5 - int(level[1:])
                if getattr(trace, "line", None) is not None:
                    trace.line.color = color
                    trace.line.dash = "solid"
                    trace.line.width = 5
                if getattr(trace, "marker", None) is not None:
                    trace.marker.color = color
                is_main_axis = (trace.xaxis or "x") == "x" and (trace.yaxis or "y") == "y"
                trace.showlegend = is_main_axis and not legend_shown
                legend_shown = legend_shown or is_main_axis
                target.add_trace(trace)
        target.update_xaxes(showline=True, mirror=True, linecolor="black", linewidth=2)
        target.update_yaxes(showline=True, mirror=True, linecolor="black", linewidth=2)
        _add_main_and_inset_borders(target)


def _queue_champs_eulerian_beta_distributions(queue, case_dir: Path) -> str:
    """Add a CHAMPS-only L1 Eulerian beta distribution comparison."""
    colors = {
        "15": "#1f77b4", "07": "#2ca02c",
        "03": "#ff7f0e", "01": "#d62728",
    }
    legend_ranks = {"15": 1, "07": 2, "03": 3, "01": 4}
    source_template = (
        "tc_naca0012_ae3932_L1_beta_bins{bins}_vs_s_"
        "slice_0p9144_all_roughness.png"
    )
    figures = {path.name: figure for figure, path in queue}
    bins07_name = source_template.format(bins="07")
    if bins07_name not in figures:
        raise RuntimeError(f"Missing CHAMPS Eulerian beta source: {bins07_name}")
    comparison = copy.deepcopy(figures[bins07_name])
    comparison.data = ()
    comparison.layout.shapes = ()
    comparison.layout.xaxis2 = None
    comparison.layout.yaxis2 = None
    zorders = {"15": 4, "07": 3, "03": 2, "01": 1}
    for bins in ("15", "07", "03", "01"):
        color = colors[bins]
        source_name = source_template.format(bins=bins)
        source = figures.get(source_name)
        if source is None:
            raise RuntimeError(f"Missing CHAMPS Eulerian beta source: {source_name}")
        source_traces = [
            copy.deepcopy(trace) for trace in source.data
            if get_trace_participant_id(trace) == "007"
            and (trace.xaxis or "x") == "x"
            and (trace.yaxis or "y") == "y"
        ]
        if not source_traces:
            raise RuntimeError(f"Missing CHAMPS BINS{bins} Eulerian beta trace")
        legend_shown = False
        for trace in source_traces:
            count = int(bins)
            meta = dict(trace.meta) if isinstance(trace.meta, dict) else {}
            meta["ipw3_participant_id"] = "007"
            meta["ipw3_bins"] = f"BINS{bins}"
            trace.meta = meta
            trace.name = f"{count} Bin" if count == 1 else f"{count} Bins"
            trace.legendgroup = f"007_beta_eulerian_bins{bins}"
            trace.legendrank = legend_ranks[bins]
            trace.zorder = zorders[bins]
            trace.line.color = color
            trace.line.dash = "solid"
            trace.line.width = 5
            if getattr(trace, "marker", None) is not None:
                trace.marker.color = color
            is_main_axis = (trace.xaxis or "x") == "x" and (trace.yaxis or "y") == "y"
            trace.showlegend = is_main_axis and not legend_shown
            legend_shown = legend_shown or is_main_axis
            comparison.add_trace(trace)
    cutdata_builder.add_collection_efficiency_inset(
        comparison, x_range=(-0.025, 0.025), y_range=None,
    )
    comparison.data = tuple(
        trace for trace in comparison.data
        if get_trace_participant_id(trace) == "007"
    )
    comparison.update_xaxes(showline=True, mirror=True, linecolor="black", linewidth=2)
    comparison.update_yaxes(showline=True, mirror=True, linecolor="black", linewidth=2)
    _add_main_and_inset_borders(comparison)
    filename = (
        "tc_naca0012_ae3932_beta_bins07_vs_s_slice_0p9144_"
        "all_roughness_eulerian_grid_levels.png"
    )
    queue.append((comparison, case_dir / filename))
    return filename


def _queue_polimo_bins01_beta_grid_panels(participant, queue, case_dir: Path) -> str:
    """Queue POLIMO's AE3932 BINS01 Eulerian/Lagrangian L1-L4 panel plot."""
    import numpy
    import pandas as pd
    import plotly.graph_objects as go
    from plotly.subplots import make_subplots
    from .gatherParticipantData import ZoneData, add_curvilinear_distance_to_zone, read_tecplot_dat

    case_id = "TC_NACA0012_AE3932"
    case_path = participant.path / "TC_NACA0012_AE3932_D01"
    figure = make_subplots(
        rows=2, cols=2, horizontal_spacing=0.09, vertical_spacing=0.14,
        subplot_titles=("L1", "L2", "L3", "L4"),
    )
    method_styles = {
        "Eulerian": ("#1f77b4", "circle"),
        "Lagrangian": ("#d62728", "circle"),
    }
    shown_methods = set()

    def read_lagrangian(path: Path, zone_name: str) -> ZoneData:
        rows = []
        selected = False
        lines = path.read_text().splitlines()
        submitted_zones = [
            line.split('"')[1] for line in lines
            if line.lstrip().upper().startswith("ZONE") and '"' in line
        ]
        selected_zone = zone_name
        if zone_name not in submitted_zones and len(submitted_zones) == 1:
            # The newly completed L1 BINS01 file initially retained its old
            # BINS07 zone label. Its sole submitted zone is still authoritative.
            selected_zone = submitted_zones[0]
        for line in lines:
            if line.lstrip().upper().startswith("ZONE"):
                selected = selected_zone == line.split('"')[1]
                continue
            if not selected or not line.strip() or line.lstrip().startswith(("#", "VARIABLES")):
                continue
            values = [float(value) for value in line.split()]
            if len(values) == 4 and numpy.isfinite(values).all():
                rows.append(values)
        if not rows:
            raise RuntimeError(f"No BINS01 Lagrangian Beta rows in {path.name}")
        return ZoneData(zone_name, pd.DataFrame(rows, columns=["X", "Y", "Z", "Beta"]))

    for panel_index, level in enumerate(("L1", "L2", "L3", "L4"), start=1):
        row, col = ((panel_index - 1) // 2 + 1, (panel_index - 1) % 2 + 1)
        zone_name = "SLICE_Y_0p9144_BINS01_KS_0p5334mm_CUTDATA"
        eulerian_path = case_path / f"TC_NACA0012_3932_{level}_cutData_V1.dat"
        eulerian_data = read_tecplot_dat(eulerian_path, process_cutdata=False)
        eulerian_zone = eulerian_data.zones.get(zone_name)
        if eulerian_zone is None:
            raise RuntimeError(f"Missing Eulerian {level} BINS01 zone in {eulerian_path.name}")
        methods = [("Eulerian", eulerian_path, eulerian_zone)]
        lagrangian_path = case_path / f"TC_NACA0012_3932_{level}_LAG_BETA_V1.dat"
        methods.append(("Lagrangian", lagrangian_path, read_lagrangian(lagrangian_path, zone_name)))
        for method, path, zone in methods:
            add_curvilinear_distance_to_zone(path, zone, case_id=case_id)
            data = zone.data.loc[
                numpy.isfinite(zone.data["s"])
                & numpy.isfinite(zone.data["Beta"])
                & (zone.data["Beta"] >= 0)
            ].sort_values("s")
            color, symbol = method_styles[method]
            trace = go.Scatter(
                x=data["s"], y=data["Beta"],
                mode="lines" if method == "Eulerian" else "lines+markers",
                name=method, legendgroup=method.lower(),
                showlegend=method not in shown_methods,
                line={"color": color, "width": 5, "dash": "solid"},
                marker={"color": color, "symbol": symbol, "size": 9, "maxdisplayed": 35,
                        "line": {"color": "#000000", "width": 1}},
                meta={"ipw3_participant_id": "007", "ipw3_droplet_model": method.lower()},
            )
            shown_methods.add(method)
            figure.add_trace(trace, row=row, col=col)

    figure.update_xaxes(
        range=[-0.1, 0.1], title_text="s [m]",
        showline=True, mirror=True, linecolor="black", linewidth=2,
    )
    figure.update_yaxes(
        title_text="β [-]", rangemode="tozero",
        showline=True, mirror=True, linecolor="black", linewidth=2,
    )
    figure.update_layout(width=2000, height=1400, paper_bgcolor="white", plot_bgcolor="white")

    filename = "tc_naca0012_ae3932_beta_bins01_lagrangian_vs_eulerian_grid_panels.png"
    queue.append((figure, case_dir / filename))
    return filename


def _queue_polimo_water_mass_method_comparisons(participant, queue, case_dir: Path) -> list[str]:
    """Queue POLIMO BINS01/BINS07 water-mass Eulerian/Lagrangian comparisons."""
    import plotly.graph_objects as go

    source = (
        participant.path / "TC_NACA0012_AE3932_D01"
        / "TC_NACA0012_3932_BETA_LAG_VS_EUL.dat"
    )
    blocks: dict[str, list[tuple[int, float, float]]] = {}
    current_bins: str | None = None
    for raw_line in source.read_text().splitlines():
        line = raw_line.strip()
        bins_match = re.fullmatch(r"#\s*(\d+)\s+Bins", line, re.IGNORECASE)
        if bins_match:
            current_bins = bins_match.group(1).zfill(2)
            blocks.setdefault(current_bins, [])
            continue
        if not line or line.upper().startswith("VARIABLES") or current_bins is None:
            continue
        values = line.split()
        if len(values) != 3:
            continue
        level, lagrangian, eulerian = int(float(values[0])), float(values[1]), float(values[2])
        blocks[current_bins].append((level, lagrangian * 1000.0, eulerian * 1000.0))

    cell_counts = convergence_data_builder.grid_cell_counts_for_case("TC_NACA0012_AE3932")
    figure = go.Figure()
    for bins in ("07", "01"):
        rows = blocks.get(bins, [])
        if not rows:
            raise RuntimeError(f"Missing POLIMO BINS{bins} water-mass comparison data in {source.name}")
        rows = sorted(rows, key=lambda row: cell_counts[row[0]] ** (-1.0 / 3.0))
        x_values = [cell_counts[level] ** (-1.0 / 3.0) for level, _, _ in rows]
        for label, color, value_index in (
            ("Lagrangian", "#d62728", 1),
            ("Eulerian", "#1f77b4", 2),
        ):
            figure.add_trace(go.Scatter(
                x=x_values,
                y=[row[value_index] for row in rows],
                mode="lines+markers",
                name=f"{label} - {int(bins)} bin" + ("s" if bins != "01" else ""),
                legendgroup=f"{label.lower()}_bins{bins}",
                line={
                    "color": color, "width": 5,
                    "dash": "solid" if bins == "07" else "dash",
                },
                marker={
                    "color": color,
                    "symbol": "square" if bins == "01" else "circle",
                    "size": 14,
                },
                customdata=[[f"L{row[0]}"] for row in rows],
                meta={"ipw3_participant_id": "007"},
                hovertemplate=(
                    "%{fullData.name}<br>Grid=%{customdata[0]}<br>"
                    "h=%{x:.6g}<br>m<sub>water</sub>=%{y:.6g} g<extra></extra>"
                ),
            ))
    convergence_data_builder.style_xy_figure(
        figure, "TC_NACA0012_AE3932", "water_mass_vs_n",
        "h = N<sub>cells</sub><sup>−1/3</sup> [-]", "m<sub>water</sub> [g]",
    )
    convergence_data_builder.style_grid_level_x_axis(
        figure, "TC_NACA0012_AE3932",
    )
    # Match the grid-convergence axis used by the main-presentation water-mass
    # figures.  The generic helper expands to the full 10^-3 decade, which
    # leaves these four NACA0012 grid levels unnecessarily compressed.
    figure.update_xaxes(
        type="log",
        range=[-2.65, -2.20],
        autorange=False,
        tickmode="array",
        dtick=None,
        tickvals=[2.556e-3, 3.334e-3, 4.486e-3, 5.546e-3],
        ticktext=[
            "2×10<sup>−3</sup>",
            "3×10<sup>−3</sup>",
            "4×10<sup>−3</sup>",
            "5×10<sup>−3</sup>",
        ],
        ticks="outside",
        showticklabels=True,
        showgrid=True,
        minor={
            "tickmode": "array",
            "tickvals": [2.9192e-3, 3.8673e-3, 4.9879e-3],
            "showgrid": True,
            "ticks": "outside",
            "ticklen": 4,
            "tickcolor": "black",
            "gridcolor": "#b0b0b0",
        },
        automargin=True,
        title_text="N<sub>cells</sub><sup>−1/3</sup> [-]",
    )
    figure.update_layout(
        width=WIDTH,
        height=HEIGHT,
        font={"family": "Arial, Helvetica, sans-serif", "size": 32},
        legend={
            "orientation": "h", "x": 0.0, "xanchor": "left",
            "y": 1.02, "yanchor": "bottom", "font": {"size": 24},
        },
        margin={"l": 100, "r": 50, "t": 125, "b": 85},
        paper_bgcolor="white",
        plot_bgcolor="white",
    )
    figure.update_xaxes(title_font={"size": 36}, tickfont={"size": 28})
    figure.update_yaxes(
        title_font={"size": 36}, tickfont={"size": 28},
        title_standoff=50, automargin=True,
    )
    filename = (
        "tc_naca0012_ae3932_water_mass_vs_n_"
        "lagrangian_vs_eulerian_bins01_bins07.png"
    )
    queue.append((figure, case_dir / filename))
    return [filename]


def _queue_beta_droplet_model_groups(queue, case_dir: Path) -> None:
    """Group AE3932 15-bin collection efficiency by droplet formulation."""
    figures = {path.name: figure for figure, path in queue}
    for grid_level in ("L1", "L2", "L3", "L4"):
        source_name = (
            f"tc_naca0012_ae3932_{grid_level}_beta_bins15_vs_s_"
            "slice_0p9144_all_roughness.png"
        )
        source = figures.get(source_name)
        if source is None:
            raise RuntimeError(f"Missing AE3932 BINS15 beta source: {source_name}")
        grouped = type(source)(source)
        grouped_traces = []
        shown_models = set()
        for trace in grouped.data:
            participant_id_value = get_trace_participant_id(trace)
            model_key = next((
                key for key, style in NACA0012_DROPLET_MODEL_STYLES.items()
                if participant_id_value in style["participant_ids"]
            ), None)
            if model_key is None:
                continue
            style = NACA0012_DROPLET_MODEL_STYLES[model_key]
            meta = dict(trace.meta) if isinstance(trace.meta, dict) else {}
            meta["ipw3_participant_id"] = participant_id_value
            meta["ipw3_droplet_model"] = model_key
            trace.meta = meta
            trace.line.update(color=style["color"], width=5, dash="solid")
            trace.marker.update(
                color=style["color"],
                line={"color": "#000000", "width": 1},
            )
            trace.mode = "lines+markers"
            trace.name = style["label"]
            trace.legendgroup = f"beta_droplet_model_{model_key}"
            trace.legendrank = style["rank"]
            is_main_axis = (trace.xaxis or "x") == "x" and (trace.yaxis or "y") == "y"
            trace.showlegend = is_main_axis and model_key not in shown_models
            if is_main_axis:
                shown_models.add(model_key)
            grouped_traces.append(trace)
        if not grouped_traces:
            raise RuntimeError(f"No documented droplet-model traces in {source_name}")
        grouped.data = tuple(sorted(grouped_traces, key=lambda trace: trace.legendrank))
        output_name = source_name.replace(
            "all_roughness.png", "grouped_droplet_models.png",
        )
        queue.append((grouped, case_dir / output_name))


def _queue_champs_ice_shape_distributions(queue, case_id: str, case_dir: Path) -> str:
    """Add a CHAMPS-only L1 single-layer comparison across bin counts."""
    colors = {
        "15": "#1f77b4", "07": "#2ca02c",
        "03": "#ff7f0e", "01": "#d62728",
    }
    template = (
        f"{case_id.lower()}_L1_single_layer_ice_shape_slice_0p9144_"
        "bins{bins}_roughness_unspecified.png"
    )
    figures = {path.name: figure for figure, path in queue}
    bins15_name = template.format(bins="15")
    if bins15_name not in figures:
        raise RuntimeError(f"Missing CHAMPS ice-shape source: {bins15_name}")
    comparison = copy.deepcopy(figures[bins15_name])
    comparison.data = tuple(
        trace for trace in comparison.data if not get_trace_participant_id(trace)
    )
    for bins in ("15", "07", "03", "01"):
        source_name = template.format(bins=bins)
        source = figures.get(source_name)
        if source is None:
            continue
        traces = [
            copy.deepcopy(trace) for trace in source.data
            if get_trace_participant_id(trace) == "007"
        ]
        if not traces:
            continue
        legend_shown = False
        for trace in traces:
            count = int(bins)
            meta = dict(trace.meta) if isinstance(trace.meta, dict) else {}
            meta["ipw3_participant_id"] = "007"
            meta["ipw3_bins"] = f"BINS{bins}"
            trace.meta = meta
            trace.name = f"{count} Bin" if count == 1 else f"{count} Bins"
            trace.legendgroup = f"007_ice_bins{bins}"
            trace.legendrank = {"15": 1, "07": 2, "03": 3, "01": 4}[bins]
            trace.zorder = {"15": 4, "07": 3, "03": 2, "01": 1}[bins]
            trace.line.color = colors[bins]
            trace.line.dash = "solid"
            trace.line.width = 5
            trace.mode = "lines"
            trace.showlegend = not legend_shown
            legend_shown = True
            comparison.add_trace(trace)
    axis = iceshape_builder.ice_shape_axis_config(case_id, 0.9144)
    x_range, y_range = iceshape_builder.leading_edge_axis_ranges(
        comparison, axis["leading_edge_fraction"],
    )
    if x_range is not None and axis.get("x_min") is not None:
        x_range[0] = axis["x_min"]
    comparison.update_xaxes(range=axis["x_range"] or x_range)
    comparison.update_yaxes(range=axis["y_range"] or y_range)
    _add_main_and_inset_borders(comparison)
    filename = (
        f"{case_id.lower()}_L1_single_layer_ice_shape_"
        "slice_0p9144_all_bins.png"
    )
    queue.append((comparison, case_dir / filename))
    return filename


def write_naca0012_presentation_figures(
    participants, output_dir: Path, participant_id: str | None = None,
    highlight: bool = False,
) -> None:
    """Generate and route the curated NACA0012 plot set for the presentation."""
    import build_site_ipw3 as site

    build_grid_convergence_section = site.build_grid_convergence_section
    build_water_mass_analysis_section = site.build_water_mass_analysis_section
    build_beta_max_analysis_section = site.build_beta_max_analysis_section
    build_grid_page_content = site.build_grid_page_content
    build_ae3933_ice_mass_comparison_section = site.build_ae3933_ice_mass_comparison_section
    case_ids = ("TC_NACA0012_AE3932", "TC_NACA0012_AE3933")
    presentation_figures = dict(NACA0012_PRESENTATION_FIGURES)
    if participant_id is not None and str(participant_id).zfill(3) == "007":
        beta_name = "tc_naca0012_ae3932_L1_beta_bins07_vs_s_slice_0p9144_all_roughness.png"
        presentation_figures[f"IMPINGEMENT/{beta_name}"] = (
            "TC_NACA0012_AE3932", beta_name, WIDTH, HEIGHT, [], True,
        )
    with tempfile.TemporaryDirectory(prefix="ipw3_naca0012_pres_") as staging_text:
        staging_dir = Path(staging_text)
        cutdata_builder.clear_png_export_queue()
        iceshape_builder.clear_png_export_queue()
        convergence_data_builder.clear_png_export_queue()

        for case_id in case_ids:
            case_dir = staging_dir / case_id
            cutdata_builder.set_png_export_dir(case_dir)
            iceshape_builder.set_png_export_dir(case_dir)
            convergence_data_builder.set_png_export_dir(case_dir, include_relative=True)
            build_grid_convergence_section(participants, case_id, category="cfd")
            build_grid_convergence_section(participants, case_id, category="icing", requirement="required")
            if case_id in {"TC_NACA0012_AE3932", "TC_NACA0012_AE3933"}:
                convergence_data_builder.queue_all_grid_levels_inverse_bin_figure(participants, case_id)
            build_water_mass_analysis_section(participants, case_id)
            build_beta_max_analysis_section(participants, case_id)
            convergence_data_builder.build_water_mass_diameter_dispersion_figures(participants, case_id)
            convergence_data_builder.build_upper_horn_angle_convergence_section(participants, case_id)
            convergence_data_builder.build_upper_horn_angle_participant_summary(participants, case_id)
            convergence_data_builder.build_upper_horn_angle_participant_summary(participants, case_id, group_by_roughness=True)
            # Participant-specific horn-method and ice-limit image collections
            # are intentionally excluded from the curated presentation export.
            requested_levels = {"L1"}
            if participant_id is not None and str(participant_id).zfill(3) == "007":
                requested_levels.update({"L2", "L3", "L4"})
            for figure_case, source_name, *_ in NACA0012_PRESENTATION_FIGURES.values():
                if figure_case != case_id:
                    continue
                level_match = re.search(r"_(L[1-4])_", source_name)
                if level_match:
                    requested_levels.add(level_match.group(1))
            for grid_level in sorted(requested_levels):
                build_grid_page_content(participants, case_id, grid_level)

        # Include identified single-bin submissions when a participant has no
        # contour in the selected 15-bin comparison.
        for figure, path in tuple(iceshape_builder.PNG_EXPORT_QUEUE):
            match = re.fullmatch(
                r"tc_naca0012_ae393[23]_(L[12])_single_layer_ice_shape_slice_0p9144_bins15_roughness_unspecified.png",
                path.name,
            )
            if not match:
                continue
            for pid in ("001", "009"):
                if any(isinstance(t.meta, dict) and t.meta.get("ipw3_participant_id") == pid for t in figure.data):
                    continue
                selected = [p for p in participants if str(p.participant_id).zfill(3) == pid]
                source, _, _ = iceshape_builder.build_single_layer_ice_shape_figure(
                    selected, path.parent.name, match.group(1),
                    slice_filter=0.9144, bins_filter="BINS01",
                )
                for trace in source.data:
                    meta = dict(trace.meta) if isinstance(trace.meta, dict) else {}
                    if meta.get("ipw3_participant_id") != pid:
                        continue
                    meta["ipw3_single_bin_comparison"] = True
                    trace.meta = meta
                    trace.name = f"{pid} (1 bin) | " + str(trace.name).split(" | ", 1)[-1]
                    if pid == "009":
                        trace.name = "009 (1 bin)"
                    figure.add_trace(trace)

        queue_finest_multilayer_ice_shapes(participants, staging_dir)

        # Participant 001 supplied the valid AE3932 multilayer contour at L1/BINS01.
        multilayer_case = "TC_NACA0012_AE3932"
        multilayer_fig, multilayer_count, _ = iceshape_builder.build_multilayer_ice_shape_figure(
            [p for p in participants if str(p.participant_id).zfill(3) == "001"],
            multilayer_case, "L1", slice_filter=0.9144, bins_filter="BINS01",
        )
        if not multilayer_count:
            raise RuntimeError("Missing requested participant 001 AE3932 L1 BINS01 multilayer ice shape")
        iceshape_builder.set_png_export_dir(staging_dir / multilayer_case)
        iceshape_builder.figure_to_html_div(
            multilayer_fig,
            "tc_naca0012_ae3932_L1_multilayer_ice_shape_001_slice_0p9144_bins01",
            "AE3932 | Participant 001 | L1 | 1 bin | Final multilayer ice shape",
        )

        # Mixed-distribution L2 comparison explicitly requested for AE3933.
        mixed_case = "TC_NACA0012_AE3933"
        single_fig, single_count, _ = iceshape_builder.build_single_layer_ice_shape_figure(
            [p for p in participants if str(p.participant_id).zfill(3) == "001"],
            mixed_case, "L2", slice_filter=0.9144, bins_filter="BINS01",
        )
        mixed_fig, multi_count, _ = iceshape_builder.build_multilayer_ice_shape_figure(
            [p for p in participants if str(p.participant_id).zfill(3) == "008"],
            mixed_case, "L2", slice_filter=0.9144, bins_filter="BINS07",
        )
        if not single_count or not multi_count:
            raise RuntimeError("Missing requested AE3933 L2 shapes: 001 BINS01 or 008 BINS07")
        for trace in mixed_fig.data:
            if str(trace.name or "").startswith("008"):
                trace.name = "008 | 7 bins | Final layer"
        for trace in single_fig.data:
            if str(trace.name or "").startswith("001"):
                trace.name = "001 | 1 bin | Single layer"
                mixed_fig.add_trace(trace)
        # Keep the clean-airfoil mask and outline above the submitted shapes.
        mixed_fig.data = tuple(t for t in mixed_fig.data if t.legendgroup != "clean_reference") + tuple(
            t for t in mixed_fig.data if t.legendgroup == "clean_reference"
        )
        iceshape_builder.set_png_export_dir(staging_dir / mixed_case)
        iceshape_builder.figure_to_html_div(
            mixed_fig, "tc_naca0012_ae3933_L2_ice_shape_001_bins01_008_bins07",
            "AE3933 | L2 | 001: 1 bin, single layer | 008: 7 bins, final layer",
        )

        # Reuse the comparison axes and experimental styling for standalone plots.
        for case_id in case_ids:
            source_name = f"{case_id.lower()}_L1_single_layer_ice_shape_slice_0p9144_bins15_roughness_unspecified.png"
            for source_figure, source_path in tuple(iceshape_builder.PNG_EXPORT_QUEUE):
                if source_path.parent.name != case_id or source_path.name != source_name:
                    continue
                experimental_figure = type(source_figure)(source_figure)
                experimental_figure.data = tuple(
                    trace for trace in experimental_figure.data
                    if str(trace.legendgroup or "").startswith("experimental_")
                    or trace.legendgroup == "clean_reference"
                )
                if not any(
                    str(trace.legendgroup or "").startswith("experimental_")
                    for trace in experimental_figure.data
                ):
                    raise RuntimeError(f"No experimental ice shapes available for {case_id}")
                experimental_path = source_path.with_name(f"{case_id.lower()}_experimental_ice_shape.png")
                iceshape_builder.PNG_EXPORT_QUEUE.append((experimental_figure, experimental_path))
                break

        # Add HTC and surface-temperature copies styled by roughness treatment.
        htc_roughness_order = {"one_mm": 0, "half_mm": 1, "variable": 2}
        for htc_case_id, variable, title in (
            (case_id, variable, title)
            for case_id in case_ids
            for variable, title in (
                ("htc", "Heat-transfer coefficient"),
                ("surface_temperature", "Surface temperature"),
                ("freezing_fraction", "Freezing fraction"),
            )
        ):
            case_slug = htc_case_id.lower()
            htc_source_name = f"{case_slug}_L1_{variable}_vs_s_slice_0p9144_all_roughness.png"
            htc_special_name = f"{case_slug}_L1_{variable}_vs_s_slice_0p9144_grouped_roughness.png"
            for source_figure, source_path in tuple(cutdata_builder.PNG_EXPORT_QUEUE):
                if source_path.parent.name != htc_case_id or source_path.name != htc_source_name:
                    continue
                grouped_figure = type(source_figure)(source_figure)
                grouped_traces = []
                shown_groups: set[str] = set()
                for trace in grouped_figure.data:
                    trace_name = str(trace.name or "")
                    trace_name_lower = trace_name.lower()
                    trace_meta = dict(trace.meta) if isinstance(trace.meta, dict) else {}
                    roughness_key = str(trace_meta.get("ipw3_roughness_key", "")).lower()
                    if roughness_key == "variable_roughness" or "variable roughness" in trace_name_lower:
                        group_key = "variable"
                    elif roughness_key == "1mm" or re.search(r"roughness height\s*=\s*1(?:\.0+)?\s*mm", trace_name_lower):
                        group_key = "one_mm"
                    elif roughness_key in {"0.5mm", "0.5334mm"} or re.search(r"roughness height\s*=\s*(?:0\.5(?:0+)?|0\.5334)\s*mm", trace_name_lower):
                        group_key = "half_mm"
                    else:
                        continue
                    group_style = NACA0012_ROUGHNESS_GROUP_STYLE[group_key]
                    participant_match = re.match(r"^(\d{1,3})(?=\D|$)", trace_name.strip())
                    if participant_match:
                        trace_meta["ipw3_participant_id"] = f"{int(participant_match.group(1)):03d}"
                    trace.meta = trace_meta
                    trace.line.update(group_style["line"])
                    trace.marker.update(group_style["marker"])
                    trace.mode = "lines+markers" if group_style.get("show_markers", False) else "lines"
                    trace.name = str(group_style["label"])
                    trace.legendgroup = f"{variable}_roughness_{group_key}"
                    trace.legendrank = htc_roughness_order[group_key]
                    trace.showlegend = group_key not in shown_groups
                    shown_groups.add(group_key)
                    grouped_traces.append(trace)
                grouped_figure.data = tuple(
                    sorted(grouped_traces, key=lambda item: item.legendrank)
                )
                grouped_figure.update_layout(
                    title={"text": f"{title} | L1 | Roughness groups", "x": 0.5, "xanchor": "center"},
                )
                cutdata_builder.PNG_EXPORT_QUEUE.append((grouped_figure, source_path.with_name(htc_special_name)))

                if variable in {"surface_temperature", "freezing_fraction"}:
                    thermodynamics_figure = type(source_figure)(source_figure)
                    thermodynamics_traces = []
                    shown_thermodynamics_groups: set[str] = set()
                    for trace in thermodynamics_figure.data:
                        trace_participant_id = get_trace_participant_id(trace)
                        thermodynamics_key = next((
                            key for key, style in NACA0012_THERMODYNAMICS_MODEL_STYLES.items()
                            if trace_participant_id in style["participant_ids"]
                        ), None)
                        if thermodynamics_key is None:
                            continue
                        thermodynamics_style = NACA0012_THERMODYNAMICS_MODEL_STYLES[thermodynamics_key]
                        trace_meta = dict(trace.meta) if isinstance(trace.meta, dict) else {}
                        trace_meta["ipw3_participant_id"] = trace_participant_id
                        trace.meta = trace_meta
                        trace.line.update(
                            color=thermodynamics_style["color"], width=5, dash="solid",
                        )
                        trace.marker.update(
                            color=thermodynamics_style["color"], maxdisplayed=10,
                            line={"color": "#000000", "width": 1},
                        )
                        trace.mode = "lines+markers"
                        group = f"{variable}_thermodynamics_model_{thermodynamics_key}"
                        trace.name = str(thermodynamics_style["label"])
                        trace.legendgroup = group
                        trace.legendrank = int(thermodynamics_style["rank"])
                        trace.showlegend = group not in shown_thermodynamics_groups
                        shown_thermodynamics_groups.add(group)
                        thermodynamics_traces.append(trace)
                    if thermodynamics_traces:
                        thermodynamics_figure.data = tuple(sorted(
                            thermodynamics_traces, key=lambda item: item.legendrank,
                        ))
                        thermodynamics_figure.update_layout(
                            title={"text": f"{title} | L1 | Thermodynamics models", "x": 0.5, "xanchor": "center"},
                        )
                        thermodynamics_name = (
                            f"{case_slug}_L1_{variable}_vs_s_slice_0p9144_"
                            "grouped_thermodynamics_models.png"
                        )
                        cutdata_builder.PNG_EXPORT_QUEUE.append((
                            thermodynamics_figure,
                            source_path.with_name(thermodynamics_name),
                        ))

                # Ts and FF combine both model families and all roughnesses,
                # matching the HTC model/roughness palette.
                if variable != "htc":
                    model_figure = type(source_figure)(source_figure)
                    model_traces = []
                    shown_combinations: set[str] = set()
                    roughness_labels = {
                        "smooth": "Smooth",
                        "half_mm": "0.5 mm", "one_mm": "1 mm",
                        "variable": "Variable",
                    }
                    roughness_ranks = {"smooth": 0, "half_mm": 1, "one_mm": 2, "variable": 3}
                    for trace in model_figure.data:
                        trace_name_lower = str(trace.name or "").lower()
                        trace_meta = dict(trace.meta) if isinstance(trace.meta, dict) else {}
                        roughness_key = str(trace_meta.get("ipw3_roughness_key", "")).lower()
                        if roughness_key == "smooth" or "no roughness" in trace_name_lower:
                            trace_roughness = "smooth"
                        elif roughness_key == "variable_roughness" or "variable roughness" in trace_name_lower:
                            trace_roughness = "variable"
                        elif roughness_key == "1mm" or re.search(
                            r"roughness height\s*=\s*1(?:\.0+)?\s*mm", trace_name_lower,
                        ):
                            trace_roughness = "one_mm"
                        elif roughness_key in {"0.5mm", "0.5334mm"} or re.search(
                            r"roughness height\s*=\s*(?:0\.5(?:0+)?|0\.5334)\s*mm",
                            trace_name_lower,
                        ):
                            trace_roughness = "half_mm"
                        else:
                            continue
                        participant_id_value = get_trace_participant_id(trace)
                        model_key = next((
                            key for key, style in NACA0012_TURBULENCE_MODEL_STYLES.items()
                            if participant_id_value in style["participant_ids"]
                        ), None)
                        if model_key is None:
                            continue
                        model_style = NACA0012_TURBULENCE_MODEL_STYLES[model_key]
                        color = NACA0012_TURBULENCE_ROUGHNESS_COLORS[model_key][trace_roughness]
                        group = f"{variable}_turbulence_model_{model_key}_{trace_roughness}"
                        trace_meta["ipw3_participant_id"] = participant_id_value
                        trace.meta = trace_meta
                        trace.line.update(color=color, width=5, dash="solid")
                        trace.marker.update(
                            color=color, maxdisplayed=10,
                            line={"color": "#000000", "width": 1},
                        )
                        trace.mode = "lines+markers"
                        trace.name = f'{model_style["label"]} - {roughness_labels[trace_roughness]}'
                        trace.legendgroup = group
                        trace.legendrank = int(model_style["rank"]) * 4 + roughness_ranks[trace_roughness]
                        trace.showlegend = group not in shown_combinations
                        shown_combinations.add(group)
                        model_traces.append(trace)
                    if model_traces:
                        model_figure.data = tuple(sorted(model_traces, key=lambda item: item.legendrank))
                        model_name = (
                            f"{case_slug}_L1_{variable}_vs_s_slice_0p9144_"
                            "grouped_turbulence_models.png"
                        )
                        cutdata_builder.PNG_EXPORT_QUEUE.append((model_figure, source_path.with_name(model_name)))
                        presentation_figures[f"SURF_TEMP_FF/{model_name}"] = (
                            htc_case_id, model_name, WIDTH, HEIGHT, [], True,
                        )
                break

        # Add two HTC copies grouped by the reported turbulence-model family.
        # The original all-roughness and grouped-roughness plots remain queued.
        for htc_case_id in case_ids:
            case_slug = htc_case_id.lower()
            source_name = f"{case_slug}_L1_htc_vs_s_slice_0p9144_all_roughness.png"
            grouped_name = f"{case_slug}_L1_htc_vs_s_slice_0p9144_grouped_turbulence_models.png"
            for source_figure, source_path in tuple(cutdata_builder.PNG_EXPORT_QUEUE):
                if source_path.parent.name != htc_case_id or source_path.name != source_name:
                    continue
                grouped_figure = type(source_figure)(source_figure)
                grouped_traces = []
                shown_model_roughness_groups: set[str] = set()
                for trace in grouped_figure.data:
                    source_trace_name = str(trace.name or "").lower()
                    source_trace_meta = trace.meta if isinstance(trace.meta, dict) else {}
                    roughness_key = str(source_trace_meta.get("ipw3_roughness_key", "")).lower()
                    trace_participant_id = get_trace_participant_id(trace)
                    model_key = next((
                        key for key, style in NACA0012_TURBULENCE_MODEL_STYLES.items()
                        if trace_participant_id in style["participant_ids"]
                    ), None)
                    if model_key is None:
                        continue
                    style = NACA0012_TURBULENCE_MODEL_STYLES[model_key]
                    trace_meta = dict(trace.meta) if isinstance(trace.meta, dict) else {}
                    trace_meta["ipw3_participant_id"] = trace_participant_id
                    trace.meta = trace_meta
                    trace.line.width = 5
                    if roughness_key == "smooth" or "no roughness" in source_trace_name:
                        roughness_group = "smooth"
                        roughness_label = "Smooth"
                        roughness_rank = 0
                    elif roughness_key == "1mm" or re.search(
                        r"roughness height\s*=\s*1(?:\.0+)?\s*mm", source_trace_name,
                    ):
                        roughness_group = "one_mm"
                        roughness_label = "1 mm"
                        roughness_rank = 2
                    elif roughness_key == "variable_roughness" or "variable roughness" in source_trace_name:
                        roughness_group = "variable"
                        roughness_label = "Variable"
                        roughness_rank = 3
                    else:
                        roughness_group = "half_mm"
                        roughness_label = "0.5 mm"
                        roughness_rank = 1
                    combination_color = NACA0012_TURBULENCE_ROUGHNESS_COLORS[model_key][roughness_group]
                    trace.line.color = combination_color
                    trace.line.dash = "solid"
                    trace.marker.color = combination_color
                    trace.marker.maxdisplayed = 10
                    trace.marker.line = {"color": "#000000", "width": 1}
                    trace.mode = "lines+markers"
                    model_roughness_group = f"htc_turbulence_model_{model_key}_{roughness_group}"
                    trace.name = f'{style["label"]} - {roughness_label}'
                    trace.legendgroup = model_roughness_group
                    trace.legendrank = int(style["rank"]) * 4 + roughness_rank
                    trace.showlegend = model_roughness_group not in shown_model_roughness_groups
                    shown_model_roughness_groups.add(model_roughness_group)
                    grouped_traces.append(trace)
                grouped_figure.data = tuple(sorted(
                    grouped_traces, key=lambda item: item.legendrank,
                ))
                grouped_figure.update_layout(
                    title={"text": "Heat-transfer coefficient | L1 | Turbulence models", "x": 0.5, "xanchor": "center"},
                )
                cutdata_builder.PNG_EXPORT_QUEUE.append((
                    grouped_figure, source_path.with_name(grouped_name),
                ))
                break

        for source_figure, source_path in tuple(iceshape_builder.PNG_EXPORT_QUEUE):
            ice_case_id = source_path.parent.name
            level_match = re.search(r"_(L[1-4])_", source_path.name)
            if level_match is None:
                continue
            ice_grid_level = level_match.group(1)
            ice_source_name = f"{ice_case_id.lower()}_{ice_grid_level}_single_layer_ice_shape_slice_0p9144_bins15_roughness_unspecified.png"
            ice_special_name = f"{ice_case_id.lower()}_{ice_grid_level}_single_layer_ice_shape_slice_0p9144_bins15_grouped_roughness.png"
            if ice_case_id not in case_ids or source_path.name != ice_source_name:
                continue
            grouped_figure = type(source_figure)(source_figure)
            grouped_traces = []
            shown_groups: set[str] = set()
            for trace in grouped_figure.data:
                trace_name = str(trace.name or "")
                trace_name_lower = trace_name.lower()
                trace_meta = dict(trace.meta) if isinstance(trace.meta, dict) else {}
                roughness_key = str(trace_meta.get("ipw3_roughness_key", "")).lower()
                # Preserve experimental contours and the clean-airfoil trace.
                if "exp" in trace_name_lower or str(trace.legendgroup or "").startswith("experimental_") or trace.legendgroup == "clean_reference":
                    grouped_traces.append(trace)
                    continue
                if trace_meta.get("ipw3_participant_id") == "009" and trace_meta.get("ipw3_single_bin_comparison"):
                    group_key = "half_mm"
                elif roughness_key == "variable_roughness" or "variable roughness" in trace_name_lower:
                    group_key = "variable"
                elif roughness_key == "1mm" or re.search(r"roughness height\s*=\s*1(?:\.0+)?\s*mm", trace_name_lower):
                    group_key = "one_mm"
                elif roughness_key in {"0.5mm", "0.5334mm"} or re.search(r"roughness height\s*=\s*(?:0\.5(?:0+)?|0\.5334)\s*mm", trace_name_lower):
                    group_key = "half_mm"
                else:
                    continue
                group_style = NACA0012_ROUGHNESS_GROUP_STYLE[group_key]
                participant_match = re.match(r"^(\d{1,3})(?=\D|$)", trace_name.strip())
                if participant_match:
                    trace_meta["ipw3_participant_id"] = f"{int(participant_match.group(1)):03d}"
                trace.meta = trace_meta
                trace.line.update(group_style["line"])
                trace.marker.update(group_style["marker"])
                trace.mode = "lines+markers" if group_style.get("show_markers", False) else "lines"
                trace.name = str(group_style["label"])
                trace.legendgroup = f"ice_roughness_{group_key}"
                trace.showlegend = group_key not in shown_groups
                shown_groups.add(group_key)
                grouped_traces.append(trace)
            grouped_figure.data = tuple(grouped_traces)
            grouped_figure.update_layout(
                title={"text": f"Single-layer ice shape | {ice_grid_level} | 15 bins | Roughness groups", "x": 0.5, "xanchor": "center"},
            )
            iceshape_builder.PNG_EXPORT_QUEUE.append((grouped_figure, source_path.with_name(ice_special_name)))

            if ice_grid_level in {"L1", "L2"}:
                model_figure = type(grouped_figure)(grouped_figure)
                model_traces = []
                shown_combinations: set[str] = set()
                roughness_labels = {
                    "half_mm": "0.5 mm", "one_mm": "1 mm",
                    "variable": "Variable",
                }
                roughness_ranks = {"half_mm": 0, "one_mm": 1, "variable": 2}
                for trace in model_figure.data:
                    group = str(trace.legendgroup or "")
                    if (
                        group == "clean_reference"
                        or group.startswith("experimental_")
                    ):
                        model_traces.append(trace)
                        continue
                    group_match = re.fullmatch(
                        r"ice_roughness_(half_mm|one_mm|variable)", group,
                    )
                    if group_match is None:
                        continue
                    roughness_group = group_match.group(1)
                    participant_id_value = get_trace_participant_id(trace)
                    model_key = next((
                        key for key, style in NACA0012_TURBULENCE_MODEL_STYLES.items()
                        if participant_id_value in style["participant_ids"]
                    ), None)
                    if model_key is None:
                        continue
                    model_style = NACA0012_TURBULENCE_MODEL_STYLES[model_key]
                    color = NACA0012_TURBULENCE_ROUGHNESS_COLORS[model_key][roughness_group]
                    combination = f"ice_turbulence_model_{model_key}_{roughness_group}"
                    trace.line.update(color=color, width=5, dash="solid")
                    trace.marker.update(color=color)
                    trace.mode = "lines+markers"
                    trace.name = f'{model_style["label"]} - {roughness_labels[roughness_group]}'
                    trace.legendgroup = combination
                    trace.legendrank = int(model_style["rank"]) * 3 + roughness_ranks[roughness_group]
                    trace.showlegend = combination not in shown_combinations
                    shown_combinations.add(combination)
                    model_traces.append(trace)
                model_figure.data = tuple(sorted(
                    model_traces,
                    key=lambda trace: (
                        100 if str(trace.legendgroup or "").startswith(("experimental_", "clean_reference"))
                        else trace.legendrank or 0
                    ),
                ))
                model_name = ice_special_name.replace(
                    "grouped_roughness.png", "grouped_turbulence_models.png",
                )
                iceshape_builder.PNG_EXPORT_QUEUE.append((
                    model_figure, source_path.with_name(model_name),
                ))

        for qc_case_id in case_ids:
            case_slug = qc_case_id.lower()
            qc_source_name = f"{case_slug}_qc_prime_vs_n_y_0.9144.png"
            qc_special_name = f"{case_slug}_qc_prime_vs_n_y_0.9144_grouped_roughness.png"
            for source_figure, source_path in tuple(convergence_data_builder.PNG_EXPORT_QUEUE):
                if source_path.parent.name != qc_case_id or source_path.name != qc_source_name:
                    continue
                grouped_figure = type(source_figure)(source_figure)
                grouped_traces = []
                shown_groups: set[str] = set()
                for trace in grouped_figure.data:
                    trace_meta = trace.meta if isinstance(trace.meta, dict) else {}
                    roughness_key = str(trace_meta.get("ipw3_roughness_key", "")).lower()
                    if roughness_key == "variable_roughness":
                        group_key = "variable"
                    elif roughness_key == "0.2mm":
                        group_key = "point_two_mm"
                    elif roughness_key == "1mm":
                        group_key = "one_mm"
                    elif roughness_key in {"0.5mm", "0.5334mm"}:
                        group_key = "half_mm"
                    else:
                        continue
                    group_style = NACA0012_ROUGHNESS_GROUP_STYLE[group_key]
                    participant_match = re.match(r"^(\d{1,3})(?=\D|$)", str(trace.name or "").strip())
                    trace_meta = dict(trace.meta) if isinstance(trace.meta, dict) else {}
                    if participant_match:
                        trace_meta["ipw3_participant_id"] = f"{int(participant_match.group(1)):03d}"
                    trace.meta = trace_meta
                    trace.line.update(group_style["line"])
                    trace.marker.update(group_style["marker"])
                    trace.mode = "lines+markers" if group_style.get("show_markers", False) else "lines"
                    trace.name = str(group_style["label"])
                    trace.legendgroup = f"qc_roughness_{group_key}"
                    trace.showlegend = group_key not in shown_groups
                    shown_groups.add(group_key)
                    grouped_traces.append(trace)
                grouped_figure.data = tuple(grouped_traces)
                grouped_figure.update_layout(
                    title={"text": "Integrated convective heat transfer | Roughness groups", "x": 0.5, "xanchor": "center"},
                )
                convergence_data_builder.PNG_EXPORT_QUEUE.append((grouped_figure, source_path.with_name(qc_special_name)))

                model_figure = type(source_figure)(source_figure)
                model_traces = []
                shown_model_roughness_groups: set[str] = set()
                roughness_labels = {
                    "smooth": "Smooth", "half_mm": "0.5 mm",
                    "one_mm": "1 mm", "variable": "Variable",
                }
                roughness_ranks = {
                    "smooth": 0, "half_mm": 1, "one_mm": 2, "variable": 3,
                }
                for trace in model_figure.data:
                    trace_meta = dict(trace.meta) if isinstance(trace.meta, dict) else {}
                    roughness_key = str(trace_meta.get("ipw3_roughness_key", "")).lower()
                    trace_participant_id = get_trace_participant_id(trace)
                    model_key = next((
                        key for key, style in NACA0012_TURBULENCE_MODEL_STYLES.items()
                        if trace_participant_id in style["participant_ids"]
                    ), None)
                    if model_key is None:
                        continue
                    if roughness_key == "smooth":
                        roughness_group = "smooth"
                    elif roughness_key == "1mm":
                        roughness_group = "one_mm"
                    elif roughness_key == "variable_roughness":
                        roughness_group = "variable"
                    elif roughness_key in {"0.5mm", "0.5334mm"}:
                        roughness_group = "half_mm"
                    else:
                        continue
                    model_style = NACA0012_TURBULENCE_MODEL_STYLES[model_key]
                    combination_color = NACA0012_TURBULENCE_ROUGHNESS_COLORS[model_key][roughness_group]
                    group = f"qc_turbulence_model_{model_key}_{roughness_group}"
                    trace_meta["ipw3_participant_id"] = trace_participant_id
                    trace.meta = trace_meta
                    trace.line.update(color=combination_color, width=5, dash="solid")
                    trace.marker.update(
                        color=combination_color,
                        line={"color": "#000000", "width": 1},
                    )
                    trace.mode = "lines+markers"
                    trace.name = f'{model_style["label"]} - {roughness_labels[roughness_group]}'
                    trace.legendgroup = group
                    trace.legendrank = int(model_style["rank"]) * 4 + roughness_ranks[roughness_group]
                    trace.showlegend = group not in shown_model_roughness_groups
                    shown_model_roughness_groups.add(group)
                    model_traces.append(trace)
                if model_traces:
                    model_figure.data = tuple(sorted(model_traces, key=lambda item: item.legendrank))
                    model_figure.update_layout(
                        title={"text": "Integrated convective heat transfer | Turbulence models", "x": 0.5, "xanchor": "center"},
                    )
                    model_name = f"{case_slug}_qc_prime_vs_n_y_0.9144_grouped_turbulence_models.png"
                    convergence_data_builder.PNG_EXPORT_QUEUE.append((
                        model_figure, source_path.with_name(model_name),
                    ))
                break

        convergence_data_builder.set_png_export_dir(staging_dir / "TC_NACA0012_AE3933", include_relative=True)
        build_ae3933_ice_mass_comparison_section(participants, requirement="required")

        # Apply one presentation-ready canvas and legend treatment across all
        # plot families before rasterization.
        roughness_legend_part = re.compile(
            r"(?:roughness|^no roughness$|^default roughness$|^variable roughness$|^unspecified roughness height$)",
            re.IGNORECASE,
        )
        for module in (convergence_data_builder, cutdata_builder, iceshape_builder):
            for figure, export_path in module.PNG_EXPORT_QUEUE:
                case_id = export_path.parent.name

                width = 2000
                height = 800
                excluded_participants: set[str] = set()
                show_legend = True
                median_line: MedianLineConfig | None = None

                for presentation_spec in presentation_figures.values():
                    presentation_case_id, source_name, image_width, image_height, excluded_ids, figure_show_legend = presentation_spec[:6]
                    if presentation_case_id == case_id and source_name == export_path.name:
                        width = image_width
                        height = image_height
                        show_legend = figure_show_legend
                        excluded_participants = {
                            f"{int(participant_id):03d}" if str(participant_id).strip().isdigit() else str(participant_id).strip()
                            for participant_id in excluded_ids
                        }
                        median_line = presentation_spec[6] if len(presentation_spec) > 6 else None
                        break

                if excluded_participants:
                    figure.data = tuple(
                        trace
                        for trace in figure.data
                        if not (
                            (trace_participant_id := (
                                str((trace.meta or {}).get("ipw3_participant_id", ""))
                                if isinstance(trace.meta, dict)
                                else ""
                            )) in excluded_participants
                            or (
                                not trace_participant_id
                                and
                                (participant_match := re.match(r"^(\d{1,3})(?=\D|$)", str(trace.name or "").strip()))
                                and f"{int(participant_match.group(1)):03d}" in excluded_participants
                            )
                        )
                    )

                if "_grouped_roughness" in export_path.stem.lower() and "_limit_" not in export_path.stem and "ice_width" not in export_path.stem:
                    shown_groups: set[str] = set()
                    for trace in figure.data:
                        legend_group = str(trace.legendgroup or "")
                        if "_roughness_" not in legend_group:
                            continue
                        trace.showlegend = legend_group not in shown_groups
                        shown_groups.add(legend_group)

                if "_cd_vs_n_all_roughness" in export_path.name:
                    figure.data = tuple(
                        trace
                        for trace in figure.data
                        if "No Roughness" in (
                            (getattr(trace, "meta", None) or {}).get("ipw3_roughness_labels", [])
                            if isinstance(getattr(trace, "meta", None), dict)
                            else []
                        )
                    )

                ice_mass_y_range_by_case = {"TC_NACA0012_AE3932": (80.0, 120.0), "TC_NACA0012_AE3933": (80.0, 120.0)}

                if "_ice_mass_vs_n_" in export_path.name and case_id in ice_mass_y_range_by_case:
                    y_min, y_max = ice_mass_y_range_by_case[case_id]
                    figure.update_yaxes(range=[y_min, y_max], autorange=False)

                if export_path.stem in {
                    "tc_naca0012_ae3932_ice_mass_vs_n_unspecified_l1_vs_inverse_bins",
                    "tc_naca0012_ae3933_ice_mass_vs_n_unspecified_l1_vs_inverse_bins",
                }:
                    figure.update_yaxes(range=[80.0, 130.0], autorange=False)

                qc_prime_y_range = (0.0, 400.0)
                if "_qc_prime_" in export_path.name.lower():
                    y_min, y_max = qc_prime_y_range
                    figure.update_yaxes(range=[y_min, y_max], autorange=False)

                plot_name = export_path.stem.lower()
                if plot_name == "tc_naca0012_ae3932_width_bins15_relative_to_l1":
                    figure.update_yaxes(range=[-4.0, 8.0], autorange=False)
                if plot_name == "tc_naca0012_ae3933_comparison_with_3932_ice_mass_bins15_percent":
                    figure.update_yaxes(title_text="Δm<sub>ice</sub> [%]")
                    figure.layout.shapes = tuple(
                        shape for shape in figure.layout.shapes
                        if not (
                            shape.type == "line" and shape.y0 == 0 and shape.y1 == 0
                            and shape.line.dash == "dash"
                        )
                    )
                is_diameter_statistics_plot = "_vs_droplet_diameter_" in plot_name
                is_grouped_roughness_plot = any(
                    group_name in plot_name
                    for group_name in (
                        "_grouped_roughness", "_grouped_turbulence_models",
                        "_grouped_thermodynamics_models",
                    )
                )
                for trace in figure.data:
                    if trace.name:
                        name_parts = [part.strip() for part in str(trace.name).split(" | ")]
                        filtered_parts = [part for part in name_parts if not roughness_legend_part.search(part)]
                        trace.name = " | ".join(filtered_parts) or name_parts[0]

                        if "exp" in trace.name.lower():
                            if module is iceshape_builder:
                                continue  # Respect the editable experimental envelope styling.
                            trace.line.width = 2
                            trace.marker.size = 10
                            continue  # Leave experimental traces at their default style.

                    if not (is_diameter_statistics_plot or is_grouped_roughness_plot) and hasattr(trace, "line") and trace.line is not None:
                        trace.line.width = 5

                    if not (is_diameter_statistics_plot or is_grouped_roughness_plot) and hasattr(trace, "marker") and trace.marker is not None:
                        trace.marker.size = 14

                figure.update_layout(
                    title=None,
                    width=width,
                    height=height,
                    showlegend=show_legend,
                    font={"family": "Arial, Helvetica, sans-serif", "size": 32},
                    legend={
                        "orientation": "h",
                        "x": 0.0,
                        "xanchor": "left",
                        "y": 1.02,
                        "yanchor": "bottom",
                        "font": {"size": 24},
                    },
                    margin={"l": 100, "r": 50, "t": 125 if show_legend else 30, "b": 85},
                    paper_bgcolor="white",
                    plot_bgcolor="white",
                )

                figure.update_xaxes(title_font={"size": 36}, tickfont={"size": 28})
                figure.update_yaxes(title_font={"size": 36}, tickfont={"size": 28}, title_standoff=50, automargin=True)

                is_ice_horn_method_plot = "_upper_horn_angle_method" in plot_name
                is_participant_horn_summary = "_upper_horn_angle_by_participant_" in plot_name
                is_ice_limits_roughness_effect = "_roughness_effect" in plot_name
                is_convergence_plot = (
                    module is convergence_data_builder
                    and not is_ice_horn_method_plot
                    and not is_diameter_statistics_plot
                    and not is_participant_horn_summary
                    and not is_ice_limits_roughness_effect
                )
                is_bin_convergence_plot = (
                    is_convergence_plot
                    and (
                        "_vs_inverse_bins" in plot_name
                        or "_distribution_convergence_" in plot_name
                    )
                )

                if is_convergence_plot and not is_bin_convergence_plot:
                    # Comparison panels may arrive with categorical L1–L4
                    # coordinates. Restore physical spacing for the log axis.
                    cell_counts = convergence_data_builder.grid_cell_counts_for_case(case_id)
                    for trace in figure.data:
                        if trace.x is None:
                            continue
                        spacing_x = []
                        for value in trace.x:
                            level_match = re.fullmatch(r"L(\d+)", str(value).strip(), re.IGNORECASE)
                            level = int(level_match.group(1)) if level_match else None
                            spacing_x.append(
                                cell_counts[level] ** (-1.0 / 3.0)
                                if level in cell_counts else value
                            )
                        trace.x = spacing_x
                    figure.update_layout(margin={"l": 100, "r": 50, "t": 125 if show_legend else 30, "b": 145})
                    figure.update_xaxes(
                        type="log",
                        range=[-2.65, -2.20],
                        autorange=False,
                        categoryorder=None,
                        categoryarray=None,
                        tickmode="array",
                        dtick=None,
                        tickvals=[2.556e-3, 3.334e-3, 4.486e-3, 5.546e-3],
                        ticktext=[
                            "2×10<sup>−3</sup>",
                            "3×10<sup>−3</sup>",
                            "4×10<sup>−3</sup>",
                            "5×10<sup>−3</sup>",
                        ],
                        ticks="outside",
                        showticklabels=True,
                        showgrid=True,
                        minor={
                            "tickmode": "array",
                            "tickvals": [2.9192e-3, 3.8673e-3, 4.9879e-3],
                            "showgrid": True,
                            "ticks": "outside",
                            "ticklen": 4,
                            "tickcolor": "black",
                            "gridcolor": "#b0b0b0",
                        },
                        automargin=True,
                        title_text=(
                            "N<sub>cells</sub><sup>−1/3</sup> [-]"
                        ),
                    )
                    _add_configured_grid_median_line(figure, median_line)

                if is_bin_convergence_plot:
                    # distribution_figure_pair_html() uses categorical BINSxx
                    # coordinates for the HTML view. Convert them back to the
                    # numeric inverse-bin coordinate required by this log axis.
                    for trace in figure.data:
                        if trace.x is None:
                            continue
                        inverse_bin_x = []
                        for x_value in trace.x:
                            bin_match = re.fullmatch(r"BINS(\d+)", str(x_value).strip(), re.IGNORECASE)
                            inverse_bin_x.append(
                                1.0 / int(bin_match.group(1))
                                if bin_match and int(bin_match.group(1)) > 0
                                else x_value
                            )
                        trace.x = inverse_bin_x

                    figure.update_layout(margin={"l": 100, "r": 50, "t": 125 if show_legend else 30, "b": 145})
                    figure.update_xaxes(
                        type="log",
                        range=[-1.25, 0.05],
                        autorange=False,
                        categoryorder=None,
                        categoryarray=None,
                        tickmode="array",
                        dtick=None,
                        tickvals=[1/15, 1/7, 1/3, 1],
                        ticktext=["1/15", "1/7", "1/3", "1"],
                        ticks="outside",
                        showticklabels=True,
                        showgrid=True,
                        minor={"showgrid": True, "dtick": "D1", "ticks": "outside"},
                        automargin=True,
                        title_text=(
                            "1 / N<sub>bins</sub> [-]"
                            "<br><span style='font-size:24px'>"
                            "</span>"
                        )
                    )
        if any(
            spec[1].endswith("_grouped_droplet_models.png")
            for spec in presentation_figures.values()
        ):
            _queue_beta_droplet_model_groups(
                cutdata_builder.PNG_EXPORT_QUEUE,
                staging_dir / "TC_NACA0012_AE3932",
            )

        if participant_id is not None and not highlight:
            for queue in (
                convergence_data_builder.PNG_EXPORT_QUEUE,
                cutdata_builder.PNG_EXPORT_QUEUE,
                iceshape_builder.PNG_EXPORT_QUEUE,
            ):
                _filter_queue_for_participant(queue, participant_id)

        if participant_id is not None and str(participant_id).zfill(3) == "007":
            polimo = next(p for p in participants if str(p.participant_id).zfill(3) == "007")
            bins01_panel_name = _queue_polimo_bins01_beta_grid_panels(
                polimo,
                cutdata_builder.PNG_EXPORT_QUEUE,
                staging_dir / "TC_NACA0012_AE3932",
            )
            presentation_figures[f"IMPINGEMENT/{bins01_panel_name}"] = (
                "TC_NACA0012_AE3932", bins01_panel_name,
                PANEL_WIDTH, 1400, [], True,
            )
            water_mass_comparison_names = _queue_polimo_water_mass_method_comparisons(
                polimo,
                convergence_data_builder.PNG_EXPORT_QUEUE,
                staging_dir / "TC_NACA0012_AE3932",
            )
            for water_mass_name in water_mass_comparison_names:
                presentation_figures[f"IMPINGEMENT/{water_mass_name}"] = (
                    "TC_NACA0012_AE3932", water_mass_name,
                    WIDTH, HEIGHT, [], True,
                )
            _replace_cp_with_champs_grid_levels(
                cutdata_builder.PNG_EXPORT_QUEUE,
                {
                    "tc_naca0012_ae3932_L1_cp_vs_x_slice_0p9144_all_roughness.png",
                    "tc_naca0012_ae3932_L1_cp_vs_s_slice_0p9144_all_roughness.png",
                },
            )
            eulerian_beta_grid_name = _queue_champs_eulerian_beta_distributions(
                cutdata_builder.PNG_EXPORT_QUEUE,
                staging_dir / "TC_NACA0012_AE3932",
            )
            presentation_figures[f"IMPINGEMENT/{eulerian_beta_grid_name}"] = (
                "TC_NACA0012_AE3932", eulerian_beta_grid_name,
                WIDTH, HEIGHT, [], True,
            )

        if participant_id is not None and str(participant_id).zfill(3) == "007":
            # Start with each existing L1 comparison to retain its MCCS
            # envelope, clean outline, axes, and presentation layout.
            polimo = next(p for p in participants if str(p.participant_id).zfill(3) == "007")
            shape_names = set()
            for case_id in case_ids:
                case_dir = staging_dir / case_id
                source_name = (
                    f"{case_id.lower()}_L1_single_layer_ice_shape"
                    "_slice_0p9144_bins15_roughness_unspecified.png"
                )
                source = next((queued for queued, path in iceshape_builder.PNG_EXPORT_QUEUE
                               if path == case_dir / source_name), None)
                if source is None:
                    raise RuntimeError(f"Missing L1 ice-shape presentation plot: {source_name}")
                figure = type(source)(source)
                figure.data = tuple(
                    trace for trace in figure.data
                    if (isinstance(trace.meta, dict) and trace.meta.get("ipw3_participant_id") == "007")
                    or str(trace.legendgroup or "").startswith(("experimental_", "reference_"))
                    or trace.legendgroup == "clean_reference"
                )
                l1 = [trace for trace in figure.data
                      if isinstance(trace.meta, dict) and trace.meta.get("ipw3_participant_id") == "007"]
                if len(l1) != 1:
                    raise RuntimeError(f"Expected one POLIMO L1 contour for {case_id}; found {len(l1)}")
                l1[0].name = "L1"
                l1[0].legendgroup = "007_L1"
                l1[0].legendrank = 1
                l1[0].zorder = 4
                l1[0].line.color = "#1f77b4"
                l1[0].line.dash = "solid"
                l1[0].line.width = POLIMO_COMBINED_ICE_LINE_WIDTH["L1"]
                l1[0].mode = "lines"
                grid_colors = {
                    "L2": "#2ca02c", "L3": "#ff7f0e", "L4": "#d62728",
                }
                for level in ("L4", "L3", "L2"):
                    level_figure, count, _ = iceshape_builder.build_single_layer_ice_shape_figure(
                        [polimo], case_id, level, slice_filter=0.9144,
                        bins_filter="BINS15", roughness_filter=None,
                    )
                    submitted = [trace for trace in level_figure.data
                                 if isinstance(trace.meta, dict) and trace.meta.get("ipw3_participant_id") == "007"]
                    if count != 1 or len(submitted) != 1:
                        raise RuntimeError(f"Expected one POLIMO {level} contour for {case_id}; found {count}")
                    trace = submitted[0]
                    trace.name = level
                    trace.legendgroup = f"007_{level}"
                    trace.legendrank = int(level[1:])
                    trace.zorder = 5 - int(level[1:])
                    trace.line.color = grid_colors[level]
                    trace.line.dash = "solid"
                    trace.line.width = POLIMO_COMBINED_ICE_LINE_WIDTH[level]
                    trace.mode = "lines"
                    figure.add_trace(trace)
                figure.update_layout(showlegend=True)
                _add_main_and_inset_borders(figure)
                filename = f"{case_id.lower()}_single_layer_ice_shape_slice_0p9144_bins15.png"
                iceshape_builder.set_png_export_dir(case_dir)
                iceshape_builder.figure_to_html_div(
                    figure, filename.removesuffix(".png"),
                    "",
                )
                shape_names.add(f"ICE_SHAPES/{filename}")
                presentation_figures[f"ICE_SHAPES/{filename}"] = (
                    case_id, filename, WIDTH, HEIGHT, [], True,
                )

                bins_filename = _queue_champs_ice_shape_distributions(
                    iceshape_builder.PNG_EXPORT_QUEUE, case_id, case_dir,
                )
                shape_names.add(f"ICE_SHAPES/{bins_filename}")
                presentation_figures[f"ICE_SHAPES/{bins_filename}"] = (
                    case_id, bins_filename, WIDTH, HEIGHT, [], True,
                )

                case_number = case_id.rsplit("AE", 1)[-1]
                level_set_path = (
                    polimo.path / f"TC_NACA0012_AE{case_number}_D01"
                    / f"TC_NACA0012_{case_number}_L1_finalIceShape_LEVEL_SET_V1.dat"
                )
                level_set_trace = _read_polimo_level_set_comparison(level_set_path)
                if level_set_trace is not None:
                    algebraic_source, algebraic_count, _ = iceshape_builder.build_single_layer_ice_shape_figure(
                        [polimo], case_id, "L1", slice_filter=0.9144,
                        bins_filter="BINS07", roughness_filter=None,
                    )
                    algebraic_traces = [
                        trace for trace in algebraic_source.data
                        if get_trace_participant_id(trace) == "007"
                    ]
                    if algebraic_count != 1 or len(algebraic_traces) != 1:
                        raise RuntimeError(
                            f"Expected one POLIMO L1 BINS07 algebraic contour for {case_id}; "
                            f"found {algebraic_count}"
                        )
                    algebraic_trace = algebraic_traces[0]
                    algebraic_trace.name = "Algebraic"
                    algebraic_trace.legendgroup = "007_algebraic"
                    algebraic_trace.legendrank = 1
                    algebraic_trace.line.update(color="#1f77b4", dash="solid", width=5)
                    algebraic_trace.mode = "lines"
                    comparison_figure = type(source)(source)
                    comparison_figure.data = tuple(
                        trace for trace in comparison_figure.data
                        if not get_trace_participant_id(trace)
                    )
                    comparison_figure.add_trace(algebraic_trace)
                    comparison_figure.add_trace(level_set_trace)
                    comparison_figure.update_layout(showlegend=True)
                    _add_main_and_inset_borders(comparison_figure)
                    comparison_name = (
                        f"{case_id.lower()}_L1_single_layer_ice_shape_slice_0p9144_"
                        "bins07_level_set_vs_algebraic.png"
                    )
                    iceshape_builder.set_png_export_dir(case_dir)
                    iceshape_builder.figure_to_html_div(
                        comparison_figure, comparison_name.removesuffix(".png"), "",
                    )
                    shape_names.add(f"ICE_SHAPES/{comparison_name}")
                    presentation_figures[f"ICE_SHAPES/{comparison_name}"] = (
                        case_id, comparison_name, WIDTH, HEIGHT, [], True,
                    )

                # Add a separate L1 comparison when the supplemental contour
                # is available. It retains POLIMO, MCCS, and the clean outline.
                normals_path = (
                    polimo.path / f"TC_NACA0012_AE{case_number}_D01"
                    / f"TC_NACA0012_{case_number}_L1_finalIceShape_UPDATE_NORMALS_V1.dat"
                )
                normals_trace = _read_polimo_normals_comparison(normals_path)
                if normals_trace is not None:
                    comparison = type(source)(source)
                    comparison.data = tuple(
                        trace for trace in comparison.data
                        if (isinstance(trace.meta, dict) and trace.meta.get("ipw3_participant_id") == "007")
                        or str(trace.legendgroup or "").startswith(("experimental_", "reference_"))
                        or trace.legendgroup == "clean_reference"
                    )
                    for trace in comparison.data:
                        if isinstance(trace.meta, dict) and trace.meta.get("ipw3_participant_id") == "007":
                            trace.name = "007 (Initial Surface Normals)"
                            trace.legendgroup = "007_initial_surface_normals"
                    comparison.add_trace(normals_trace)
                    comparison_name = (
                        f"{case_id.lower()}_L1_single_layer_ice_shape"
                        "_slice_0p9144_bins15_normals_comparison.png"
                    )
                    iceshape_builder.figure_to_html_div(
                        comparison, comparison_name.removesuffix(".png"), "",
                    )
                    shape_names.add(f"ICE_SHAPES/{comparison_name}")
                    presentation_figures[f"ICE_SHAPES/{comparison_name}"] = (
                        case_id, comparison_name, WIDTH, HEIGHT, [], True,
                    )

                maxccs_path = (
                    polimo.path / f"TC_NACA0012_AE{case_number}_D01"
                    / f"TC_NACA0012_{case_number}_L1_finalIceShape_MCCS_V1.dat"
                )
                maxccs_traces = _read_polimo_maxccs(maxccs_path)
                if maxccs_traces:
                    multilayer = type(source)(source)
                    multilayer.data = tuple(
                        trace for trace in multilayer.data
                        if str(trace.legendgroup or "").startswith(("experimental_", "reference_"))
                        or trace.legendgroup == "clean_reference"
                    )
                    for maxccs_trace in maxccs_traces:
                        multilayer.add_trace(maxccs_trace)
                    multilayer_name = (
                        f"{case_id.lower()}_L1_multilayer_ice_shape"
                        "_slice_0p9144_bins07_roughness_comparisons_MCCS.png"
                    )
                    iceshape_builder.figure_to_html_div(
                        multilayer, multilayer_name.removesuffix(".png"), "",
                    )
                    shape_names.add(f"ICE_SHAPES/{multilayer_name}")
                    presentation_figures[f"ICE_SHAPES/{multilayer_name}"] = (
                        case_id, multilayer_name, WIDTH, HEIGHT, [], True,
                    )

                # Separate multilayer-only comparison: retain the experimental
                # MCCS and clean reference, but do not add participant MCCS.
                multilayer_path = (
                    polimo.path / f"TC_NACA0012_AE{case_number}_D01"
                    / f"TC_NACA0012_{case_number}_L1_finalIceShape_MULTILAYER_V1.dat"
                )
                submitted_multilayer = _read_polimo_multilayer_comparison(multilayer_path)
                if submitted_multilayer:
                    multilayer_only = type(source)(source)
                    multilayer_only.data = tuple(
                        trace for trace in multilayer_only.data
                        if str(trace.legendgroup or "").startswith(("experimental_", "reference_"))
                        or trace.legendgroup == "clean_reference"
                    )
                    for trace in submitted_multilayer:
                        multilayer_only.add_trace(trace)
                    multilayer_only_name = (
                        f"{case_id.lower()}_L1_multilayer_ice_shape"
                        "_slice_0p9144_bins07_roughness_comparison.png"
                    )
                    iceshape_builder.figure_to_html_div(
                        multilayer_only, multilayer_only_name.removesuffix(".png"), "",
                    )
                    shape_names.add(f"ICE_SHAPES/{multilayer_only_name}")
                    presentation_figures[f"ICE_SHAPES/{multilayer_only_name}"] = (
                        case_id, multilayer_only_name, WIDTH, HEIGHT, [], True,
                    )
            presentation_figures = {
                destination: spec for destination, spec in presentation_figures.items()
                if not destination.startswith("ICE_SHAPES/") or destination in shape_names
            }

        for module in (convergence_data_builder, cutdata_builder, iceshape_builder):
            for figure, export_path in module.PNG_EXPORT_QUEUE:
                for trace in figure.data:
                    if trace.legendgroup == "clean_reference":
                        trace.showlegend = False
                    if module is iceshape_builder and trace.name:
                        trace.name = re.sub(r"^Exp\.\s*", "", trace.name)
                if "_grouped_roughness" in export_path.stem and "_limit_" not in export_path.stem and "ice_width" not in export_path.stem:
                    apply_roughness_participant_legend(figure, GROUPED_ROUGHNESS_PARTICIPANT_SYMBOLS,
                        participant_symbol_size=GROUPED_ROUGHNESS_PARTICIPANT_SYMBOL_SIZE,
                        plot_symbol_size=GROUPED_ROUGHNESS_PLOT_SYMBOL_SIZE,
                        roughness_position={"x": GROUPED_ROUGHNESS_LEGEND_X},
                        participant_position={"x": GROUPED_PARTICIPANT_LEGEND_X},
                        experimental_position=GROUPED_EXPERIMENTAL_LEGEND_POSITION,
                        row_shift=GROUPED_LEGEND_ROW_SHIFT)

                if "_grouped_turbulence_models" in export_path.stem:
                    is_ice_turbulence_figure = export_path.stem in {
                        "tc_naca0012_ae3932_L1_single_layer_ice_shape_slice_0p9144_bins15_grouped_turbulence_models",
                        "tc_naca0012_ae3932_L2_single_layer_ice_shape_slice_0p9144_bins15_grouped_turbulence_models",
                        "tc_naca0012_ae3933_L1_single_layer_ice_shape_slice_0p9144_bins15_grouped_turbulence_models",
                        "tc_naca0012_ae3933_L2_single_layer_ice_shape_slice_0p9144_bins15_grouped_turbulence_models",
                    }
                    apply_roughness_participant_legend(figure, GROUPED_ROUGHNESS_PARTICIPANT_SYMBOLS,
                        participant_symbol_size=(
                            TURBULENCE_MODEL_ICE_PARTICIPANT_SYMBOL_SIZE
                            if is_ice_turbulence_figure
                            else GROUPED_ROUGHNESS_PARTICIPANT_SYMBOL_SIZE
                        ),
                        plot_symbol_size=(
                            TURBULENCE_MODEL_ICE_PLOT_SYMBOL_SIZE
                            if is_ice_turbulence_figure
                            else GROUPED_ROUGHNESS_PLOT_SYMBOL_SIZE
                        ),
                        roughness_position={"x": GROUPED_ROUGHNESS_LEGEND_X},
                        participant_position={"x": GROUPED_PARTICIPANT_LEGEND_X},
                        experimental_position=GROUPED_EXPERIMENTAL_LEGEND_POSITION,
                        row_shift=GROUPED_LEGEND_ROW_SHIFT,
                        group_token="_turbulence_model_")

                    if export_path.stem in {
                        "tc_naca0012_ae3932_L1_single_layer_ice_shape_slice_0p9144_bins15_grouped_turbulence_models",
                        "tc_naca0012_ae3932_L2_single_layer_ice_shape_slice_0p9144_bins15_grouped_turbulence_models",
                        "tc_naca0012_ae3933_L2_single_layer_ice_shape_slice_0p9144_bins15_grouped_turbulence_models",
                    }:
                        for trace in figure.data:
                            trace_meta = trace.meta if isinstance(trace.meta, dict) else {}
                            participant_id_value = str(
                                trace_meta.get("ipw3_participant_id", "")
                            ).zfill(3)
                            if participant_id_value not in GROUPED_ROUGHNESS_PARTICIPANT_SYMBOLS:
                                continue
                            trace.mode = "lines+markers"
                            trace.marker.update(
                                symbol=GROUPED_ROUGHNESS_PARTICIPANT_SYMBOLS[participant_id_value],
                                size=TURBULENCE_MODEL_ICE_PLOT_SYMBOL_SIZE,
                            )
                            trace.line.update(
                                dash="solid",
                                width=6 if participant_id_value == "003" else 5,
                            )

                if "_grouped_thermodynamics_models" in export_path.stem:
                    apply_roughness_participant_legend(figure, GROUPED_ROUGHNESS_PARTICIPANT_SYMBOLS,
                        participant_symbol_size=GROUPED_ROUGHNESS_PARTICIPANT_SYMBOL_SIZE,
                        plot_symbol_size=GROUPED_ROUGHNESS_PLOT_SYMBOL_SIZE,
                        roughness_position={"x": GROUPED_ROUGHNESS_LEGEND_X},
                        participant_position={"x": GROUPED_PARTICIPANT_LEGEND_X},
                        experimental_position=GROUPED_EXPERIMENTAL_LEGEND_POSITION,
                        row_shift=GROUPED_LEGEND_ROW_SHIFT,
                        group_token="_thermodynamics_model_")

                if "_grouped_droplet_models" in export_path.stem:
                    apply_roughness_participant_legend(figure, GROUPED_ROUGHNESS_PARTICIPANT_SYMBOLS,
                        participant_symbol_size=GROUPED_ROUGHNESS_PARTICIPANT_SYMBOL_SIZE,
                        plot_symbol_size=GROUPED_ROUGHNESS_PLOT_SYMBOL_SIZE,
                        roughness_position={"x": GROUPED_ROUGHNESS_LEGEND_X},
                        participant_position={"x": GROUPED_PARTICIPANT_LEGEND_X},
                        experimental_position=GROUPED_EXPERIMENTAL_LEGEND_POSITION,
                        row_shift=GROUPED_LEGEND_ROW_SHIFT,
                        group_token="_droplet_model_")

                if (
                    "_qc_prime_" in export_path.stem.lower()
                    and any(group_name in export_path.stem for group_name in (
                        "_grouped_roughness", "_grouped_turbulence_models",
                    ))
                ):
                    for trace in figure.data:
                        trace_meta = trace.meta if isinstance(trace.meta, dict) else {}
                        participant_id_value = str(
                            trace_meta.get("ipw3_participant_id", "")
                        ).zfill(3)
                        if participant_id_value in GROUPED_ROUGHNESS_PARTICIPANT_SYMBOLS:
                            trace.mode = "lines+markers"
                            trace.marker.update(
                                symbol=GROUPED_ROUGHNESS_PARTICIPANT_SYMBOLS[participant_id_value],
                                size=QC_PLOT_SYMBOL_SIZE,
                            )
                        participant_symbol_match = re.fullmatch(
                            r"participant_symbol_(\d{3})",
                            str(trace.legendgroup or ""),
                        )
                        if participant_symbol_match:
                            trace.marker.update(size=QC_PARTICIPANT_SYMBOL_SIZE)

                if re.fullmatch(
                    r"tc_naca0012_ae393[23]_qc_prime_vs_n_y_0\.9144",
                    export_path.stem.lower(),
                ):
                    for trace in figure.data:
                        trace_meta = trace.meta if isinstance(trace.meta, dict) else {}
                        if trace_meta.get("ipw3_participant_id") is None:
                            continue
                        trace.mode = "lines+markers"
                        trace.marker.update(symbol="circle", size=QC_PLOT_SYMBOL_SIZE)

                if ((any(key in export_path.stem for key in (
                    "_htc_vs_s_", "_surface_temperature_vs_s_", "_freezing_fraction_vs_s_",
                ))) and any(
                    group_name in export_path.stem
                    for group_name in (
                        "_grouped_roughness", "_grouped_turbulence_models",
                        "_grouped_thermodynamics_models",
                    )
                )) or any(
                    key in export_path.stem for key in ("_mean_surface_temperature_", "_mean_freezing_fraction_")
                ):
                    figure.update_layout(legend={"title": {"text": ""}}, legend2={"title": {"text": ""}})

                if module is iceshape_builder:
                    figure.update_layout(
                        legend={"title": {"text": ""}, "x": 0.20, "xanchor": "left"},
                        legend2={"title": {"text": ""}, "x": 0.20, "xanchor": "left"},
                    )

                # Keep participant 019's density distinction after presentation styling.
                for trace in figure.data:
                    meta = trace.meta if isinstance(trace.meta, dict) else {}
                    if module is iceshape_builder and meta.get("ipw3_participant_id") == "019" and "ipw3_density_model" in meta:
                        trace.line.dash = "dashdot" if meta["ipw3_density_model"] == "variable" else "solid"
                        trace.mode = "lines"

                experimental_traces = [
                    trace for trace in figure.data
                    if str(trace.legendgroup or "").startswith(("experimental_", "reference_"))
                    or str(trace.name or "").lower().startswith("exp")
                ]
                is_cp_plot = module is cutdata_builder and (
                    "_cp_vs_s_" in plot_name or "_cp_vs_x_" in plot_name
                )
                if experimental_traces and not is_cp_plot:
                    for trace in experimental_traces:
                        trace.legend = "legend3"
                    has_participant_row = any(trace.legend == "legend2" for trace in figure.data)
                    figure.update_layout(
                        legend={"y": 1.30 if has_participant_row else 1.16},
                        legend3={
                            "orientation": "h", "x": 0.20 if module is iceshape_builder else 0.0, "xanchor": "left",
                            "y": 1.19 if has_participant_row else 1.05,
                            "yanchor": "bottom", "font": {"size": 24},
                            "traceorder": "normal", "title": {"text": ""},
                        },
                    )
                    if has_participant_row:
                        figure.update_layout(legend2={"y": 1.02})
                    if figure.layout.showlegend is not False:
                        figure.update_layout(margin={"t": max(
                            300 if has_participant_row else 220, figure.layout.margin.t or 0
                        )})

                if is_cp_plot:
                    # Cp uses a single horizontal legend for participants and
                    # the experimental series; inset copies remain hidden.
                    shown_exp = False
                    for trace in experimental_traces:
                        trace.legend = "legend"
                        trace.name = "Exp."
                        if trace.showlegend is not False:
                            trace.showlegend = not shown_exp
                            shown_exp = True
                    exp_y = GROUPED_EXPERIMENTAL_LEGEND_POSITION["y"]
                    figure.update_layout(
                        legend={"y": exp_y + GROUPED_LEGEND_ROW_SHIFT,
                                "bgcolor": "rgba(0,0,0,0)"},
                    )

                if ("_upper_horn_angle_bins" in plot_name or
                        "_upper_horn_angle_distribution_convergence_" in plot_name):
                    # These convergence plots have participant lines and
                    # experimental reference lines, but no symbol-only row.
                    # Stack the two existing rows at the bottom positions.
                    exp_y = GROUPED_EXPERIMENTAL_LEGEND_POSITION["y"]
                    has_exp = bool(experimental_traces)
                    figure.update_layout(
                        legend={"y": exp_y + GROUPED_LEGEND_ROW_SHIFT * int(has_exp),
                                "bgcolor": "rgba(0,0,0,0)"},
                        legend3={"y": exp_y, "bgcolor": "rgba(0,0,0,0)"},
                    )

                if "horn" in plot_name:
                    _add_main_and_inset_borders(figure)

        # Add three-panel L2 roughness comparisons while retaining the combined plots.
        from plotly.subplots import make_subplots
        import plotly.graph_objects as go

        for source, path in tuple(iceshape_builder.PNG_EXPORT_QUEUE):
            if path.name not in {
                f"{case_id.lower()}_L2_single_layer_ice_shape_slice_0p9144_bins15_grouped_roughness.png"
                for case_id in case_ids
            }:
                continue
            panels = make_subplots(rows=1, cols=3, horizontal_spacing=0.035,
                                   subplot_titles=("0.5 mm", "1 mm", "Variable roughness"))
            for col, group_key in enumerate(("half_mm", "one_mm", "variable"), start=1):
                for original in source.data:
                    group = str(original.legendgroup or "")
                    is_reference = group == "clean_reference" or group.startswith("experimental_")
                    if group != f"ice_roughness_{group_key}" and not is_reference:
                        continue
                    if original.x is None or all(value is None for value in original.x):
                        continue
                    trace = go.Scatter(original.to_plotly_json())
                    trace.update(xaxis=None, yaxis=None)
                    if is_reference and col > 1:
                        trace.showlegend = False
                    panels.add_trace(trace, row=1, col=col)
                for axis in ("xaxis", "yaxis"):
                    settings = getattr(source.layout, axis).to_plotly_json()
                    for key in ("domain", "anchor", "scaleanchor", "matches"):
                        settings.pop(key, None)
                    settings.update(constrain="domain", autorange=False)
                    if axis == "yaxis":
                        settings.update(scaleanchor="x" if col == 1 else f"x{col}", scaleratio=1)
                    update = panels.update_xaxes if axis == "xaxis" else panels.update_yaxes
                    update(**settings, row=1, col=col)
            # Keep the participant-symbol legend once for the entire figure.
            for original in source.data:
                if str(original.legendgroup or "").startswith("participant_symbol_"):
                    panels.add_trace(go.Scatter(original.to_plotly_json()), row=1, col=1)
            panels.update_annotations(font=dict(size=40), yshift=12)
            panels.update_xaxes(title_font=dict(size=38), tickfont=dict(size=30), title_standoff=12)
            panels.update_yaxes(title_font=dict(size=38), tickfont=dict(size=30), title_standoff=12, automargin=True)
            for col in (2, 3):
                panels.update_yaxes(title_text="", title_standoff=0, row=1, col=col)
            panels.update_layout(width=2400, height=700, font=source.layout.font,
                                 paper_bgcolor="white", plot_bgcolor="white",
                                 margin=dict(l=85, r=30, t=165, b=75),
                                 showlegend=source.layout.showlegend,
                                 legend2=source.layout.legend2.to_plotly_json(),
                                 legend3=source.layout.legend3.to_plotly_json())
            panels.update_layout(legend2=dict(font=dict(size=30)),
                                 legend3=dict(font=dict(size=30)))
            if path.parent.name == "TC_NACA0012_AE3933":
                panels.layout.annotations = ()
                panels.update_layout(showlegend=False)
            else:
                panels.update_annotations(y=1.0, yanchor="bottom", yshift=8)
            panel_path = path.with_name(path.stem + "_panels.png")
            iceshape_builder.PNG_EXPORT_QUEUE.append((panels, panel_path))

            if path.parent.name in {"TC_NACA0012_AE3932", "TC_NACA0012_AE3933"}:
                model_panels = make_subplots(
                    rows=1, cols=2, horizontal_spacing=0.045,
                    subplot_titles=("k-ω", "SA"),
                )
                roughness_labels = {
                    "half_mm": "0.5 mm",
                    "one_mm": "1 mm",
                    "variable": "Variable",
                }
                shown_combinations = set()
                included_participants = set()
                for col, model_key in enumerate(("kw", "sa"), start=1):
                    model_style = NACA0012_TURBULENCE_MODEL_STYLES[model_key]
                    for original in source.data:
                        group = str(original.legendgroup or "")
                        is_reference = (
                            group == "clean_reference"
                            or group.startswith("experimental_")
                        )
                        if is_reference:
                            if original.x is None or all(value is None for value in original.x):
                                continue
                            trace = go.Scatter(original.to_plotly_json())
                            trace.update(xaxis=None, yaxis=None)
                            if col > 1:
                                trace.showlegend = False
                            model_panels.add_trace(trace, row=1, col=col)
                            continue
                        group_match = re.fullmatch(
                            r"ice_roughness_(half_mm|one_mm|variable)", group,
                        )
                        if group_match is None:
                            continue
                        participant_id_value = get_trace_participant_id(original)
                        if participant_id_value not in model_style["participant_ids"]:
                            continue
                        if original.x is None or all(value is None for value in original.x):
                            continue
                        roughness_group = group_match.group(1)
                        color = NACA0012_TURBULENCE_ROUGHNESS_COLORS[model_key][roughness_group]
                        trace = go.Scatter(original.to_plotly_json())
                        trace.update(xaxis=None, yaxis=None)
                        trace.line.update(color=color, width=5, dash="solid")
                        trace.marker.update(
                            color=color,
                            size=TURBULENCE_MODEL_ICE_PLOT_SYMBOL_SIZE,
                        )
                        trace.name = f'{model_style["label"]} - {roughness_labels[roughness_group]}'
                        combination = f"ice_turbulence_model_{model_key}_{roughness_group}"
                        trace.legendgroup = combination
                        trace.showlegend = False
                        if combination not in shown_combinations:
                            model_panels.add_trace(
                                go.Scatter(
                                    x=[None], y=[None], mode="lines",
                                    name=trace.name, legendgroup=combination,
                                    line={"color": color, "width": 5, "dash": "solid"},
                                    showlegend=True, hoverinfo="skip",
                                ),
                                row=1, col=1,
                            )
                            shown_combinations.add(combination)
                        included_participants.add(participant_id_value)
                        model_panels.add_trace(trace, row=1, col=col)
                    for axis in ("xaxis", "yaxis"):
                        settings = getattr(source.layout, axis).to_plotly_json()
                        for key in ("domain", "anchor", "scaleanchor", "matches"):
                            settings.pop(key, None)
                        settings.update(constrain="domain", autorange=False)
                        if axis == "yaxis":
                            settings.update(
                                scaleanchor="x" if col == 1 else f"x{col}",
                                scaleratio=1,
                            )
                        update = model_panels.update_xaxes if axis == "xaxis" else model_panels.update_yaxes
                        update(**settings, row=1, col=col)
                for original in source.data:
                    symbol_match = re.fullmatch(
                        r"participant_symbol_(\d{3})",
                        str(original.legendgroup or ""),
                    )
                    if symbol_match and symbol_match.group(1) in included_participants:
                        symbol_trace = go.Scatter(original.to_plotly_json())
                        symbol_trace.marker.update(
                            size=TURBULENCE_MODEL_ICE_PARTICIPANT_SYMBOL_SIZE,
                        )
                        model_panels.add_trace(
                            symbol_trace, row=1, col=1,
                        )
                model_panels.update_annotations(font=dict(size=40), yshift=12)
                model_panels.update_xaxes(
                    title_font=dict(size=38), tickfont=dict(size=30), title_standoff=12,
                )
                model_panels.update_yaxes(
                    title_font=dict(size=38), tickfont=dict(size=30),
                    title_standoff=12, automargin=True,
                )
                model_panels.update_yaxes(title_text="", title_standoff=0, row=1, col=2)
                model_panels.update_layout(
                    width=2400, height=700, font=source.layout.font,
                    paper_bgcolor="white", plot_bgcolor="white",
                    margin=dict(l=85, r=30, t=165, b=75), showlegend=True,
                    legend={
                        **source.layout.legend.to_plotly_json(),
                        "x": 0.0,
                        "xanchor": "left",
                    },
                    legend2=source.layout.legend2.to_plotly_json(),
                    legend3=source.layout.legend3.to_plotly_json(),
                )
                turbulence_panel_path = path.with_name(
                    path.name.replace(
                        "grouped_roughness.png",
                        "grouped_turbulence_models_panels.png",
                    )
                )
                iceshape_builder.PNG_EXPORT_QUEUE.append(
                    (model_panels, turbulence_panel_path)
                )

        # Fit the canvas to the fixed ice-shape aspect ratio rather than leaving
        # unused space when Plotly compresses an axis domain.
        for figure, path in iceshape_builder.PNG_EXPORT_QUEUE:
            x_range = figure.layout.xaxis.range
            y_range = figure.layout.yaxis.range
            if not x_range or not y_range:
                continue
            x_span = abs(x_range[1] - x_range[0])
            y_span = abs(y_range[1] - y_range[0])
            if not x_span or not y_span:
                continue
            is_panels = path.stem.endswith("_panels")
            width = figure.layout.width or 1350
            left, right, bottom = (125 if is_panels else 85), 25, 75
            legend_ids = {
                trace.legend or "legend" for trace in figure.data
                if trace.showlegend is not False
            } if figure.layout.showlegend is not False else set()
            top = 35 + 48 * len(legend_ids)
            if is_panels and figure.layout.annotations:
                top += 45
            plot_width = width - left - right
            if is_panels:
                if "_grouped_turbulence_models_panels" in path.stem:
                    plot_width *= (1 - 0.045) / 2
                else:
                    plot_width *= (1 - 2 * 0.035) / 3
            plot_height = plot_width * y_span / x_span
            panel_row_gap_px = max(PANEL_LEGEND_ROW_GAP_PX, GROUPED_LEGEND_ROW_SHIFT * plot_height)
            use_panel_legend_layout = is_panels and (
                path.parent.name == "TC_NACA0012_AE3932"
                or "_grouped_turbulence_models_panels" in path.stem
            )
            if use_panel_legend_layout:
                top = max(top, PANEL_LEGEND_CLEARANCE_PX + panel_row_gap_px + 50)
            figure.update_layout(
                height=round(plot_height + top + bottom),
                margin=dict(l=left, r=right, t=top, b=bottom),
            )
            if not is_panels:
                # The aspect-fitted plotting area now starts at paper x=0.
                for legend_id in legend_ids:
                    figure.update_layout(**{legend_id: dict(x=0.0, xanchor="left")})
            if use_panel_legend_layout:
                # Keep the participant and experimental rows aligned across
                # the AE3932 and AE3933 turbulence-model panels.
                has_experimental_legend = any(
                    (trace.legend or "legend") == "legend3" and trace.showlegend is not False
                    for trace in figure.data
                )
                experimental_y = 1 + PANEL_LEGEND_CLEARANCE_PX / plot_height
                participant_y = experimental_y + (
                    panel_row_gap_px / plot_height if has_experimental_legend else 0
                )
                figure.update_layout(
                    legend2={"x": GROUPED_PARTICIPANT_LEGEND_X, "y": participant_y,
                             "yanchor": "bottom", "bgcolor": "rgba(0,0,0,0)"},
                    legend3={**GROUPED_EXPERIMENTAL_LEGEND_POSITION,
                             "y": experimental_y, "yanchor": "bottom",
                             "bgcolor": "rgba(0,0,0,0)"},
                )
            elif "_grouped_roughness" in path.stem and "_limit_" not in path.stem and "ice_width" not in path.stem:
                # Keep the editable positions authoritative after the
                # ice-shape aspect-fit and panel layout adjustments above.
                has_experimental_legend = any(
                    (trace.legend or "legend") == "legend3" for trace in figure.data
                )
                roughness_position, participant_position = grouped_legend_positions(
                    GROUPED_EXPERIMENTAL_LEGEND_POSITION,
                    GROUPED_LEGEND_ROW_SHIFT,
                    has_experimental_legend,
                    roughness_x=GROUPED_ROUGHNESS_LEGEND_X,
                    participant_x=GROUPED_PARTICIPANT_LEGEND_X,
                )
                figure.update_layout(
                    legend=roughness_position,
                    legend2=participant_position,
                    legend3=GROUPED_EXPERIMENTAL_LEGEND_POSITION,
                )

        if participant_id is not None and str(participant_id).zfill(3) == "007":
            # Keep the IPW3 comparison and add POLIMO's Lagrangian curve.
            # Both POLIMO curves use the submitted +4° coordinate correction
            # before signed s is calculated.
            from .POLIMO_NACA0012_PLOTS import BETA_COMPARISONS, comparison_figure
            target = "tc_naca0012_ae3932_L1_beta_bins07_vs_s_slice_0p9144_all_roughness.png"
            polimo = next(p for p in participants if str(p.participant_id).zfill(3) == "007")
            target_path = staging_dir / "TC_NACA0012_AE3932" / target
            figure = next((queued for queued, path in cutdata_builder.PNG_EXPORT_QUEUE
                           if path == target_path), None)
            if figure is None:
                raise RuntimeError(f"Missing BINS07 presentation plot: {target}")
            eulerian = [trace for trace in figure.data
                        if str(trace.name or "").strip().startswith("007")]
            if not eulerian:
                raise RuntimeError("Missing POLIMO Eulerian BINS07 trace")
            for index, trace in enumerate(eulerian):
                trace.name = "007 (Eulerian)"
                trace.legendgroup = "007_eulerian"
                trace.showlegend = index == 0
                trace.line.color = "#1f77b4"
                trace.line.dash = "solid"
                trace.mode = "lines"
            lagrangian = comparison_figure(polimo.path, BETA_COMPARISONS[0]).data[0]
            lagrangian.name = "007 (Lagrangian)"
            lagrangian.legendgroup = "007_lagrangian"
            lagrangian.line.color = "#d62728"
            lagrangian.line.dash = "solid"
            lagrangian.mode = "lines+markers"
            lagrangian.marker = dict(color="#d62728", size=9, maxdisplayed=35)
            lagrangian.meta = {"ipw3_participant_id": "007"}
            figure.add_trace(lagrangian)
            # The queued collection-efficiency figure already contains an
            # x2/y2 inset. Add the same Lagrangian profile to that zoom.
            if figure.layout.xaxis2 is not None and figure.layout.yaxis2 is not None:
                import plotly.graph_objects as go
                zoom_trace = go.Scatter(lagrangian.to_plotly_json())
                zoom_trace.update(xaxis="x2", yaxis="y2", showlegend=False,
                                  hoverinfo="skip")
                figure.add_trace(zoom_trace)
                x_limits = figure.layout.xaxis2.range
                y_limits = figure.layout.yaxis2.range
                if x_limits and y_limits:
                    zoom_values = [float(y) for x, y in zip(lagrangian.x, lagrangian.y)
                                   if x_limits[0] <= float(x) <= x_limits[1]]
                    if zoom_values:
                        low = min(y_limits[0], min(zoom_values) - 0.01)
                        high = max(y_limits[1], max(zoom_values) + 0.01)
                        figure.layout.yaxis2.range = [low, high]
            figure.data = tuple(
                trace for trace in figure.data
                if get_trace_participant_id(trace) == "007"
            )
            presentation_figures = {
                destination: spec for destination, spec in presentation_figures.items()
                if (
                    (
                        not destination.startswith("IMPINGEMENT/")
                        or destination.endswith(target)
                        or destination.endswith(eulerian_beta_grid_name)
                        or destination.endswith(bins01_panel_name)
                        or Path(destination).name in water_mass_comparison_names
                    )
                    and (
                        not destination.startswith("HTC/")
                        or destination in {
                            "HTC/tc_naca0012_ae3932_L1_htc_vs_s_slice_0p9144_all_roughness.png",
                            "HTC/tc_naca0012_ae3933_L1_htc_vs_s_slice_0p9144_all_roughness.png",
                        }
                    )
                    and (
                        not destination.startswith("ICE_ACCRETION/")
                        or destination in {
                            "ICE_ACCRETION/tc_naca0012_ae3932_upper_horn_angle_bins15.png",
                            "ICE_ACCRETION/tc_naca0012_ae3932_upper_horn_angle_distribution_convergence_l1.png",
                            "ICE_ACCRETION/tc_naca0012_ae3933_upper_horn_angle_bins15.png",
                            "ICE_ACCRETION/tc_naca0012_ae3933_upper_horn_angle_distribution_convergence_l1.png",
                            "ICE_ACCRETION/tc_naca0012_ae3932_ice_mass_vs_n_unspecified_bins15_all_grid_levels.png",
                            "ICE_ACCRETION/tc_naca0012_ae3932_ice_mass_vs_n_unspecified_l1_vs_inverse_bins.png",
                            "ICE_ACCRETION/tc_naca0012_ae3933_ice_mass_vs_n_unspecified_bins15_all_grid_levels.png",
                            "ICE_ACCRETION/tc_naca0012_ae3933_ice_mass_vs_n_unspecified_l1_vs_inverse_bins.png",
                        }
                    )
                    and not destination.startswith(("ICE_LIMITS/", "ICE_HORNS/"))
                    and (
                        not destination.startswith("ICE_HORN_PARTICIPANT/")
                        or destination.startswith("ICE_HORN_PARTICIPANT/p007/")
                    )
                    and (
                        not destination.startswith("SURF_TEMP_FF/")
                        or destination in {
                            "SURF_TEMP_FF/tc_naca0012_ae3932_L1_freezing_fraction_vs_s_slice_0p9144_all_roughness.png",
                            "SURF_TEMP_FF/tc_naca0012_ae3932_L1_surface_temperature_vs_s_slice_0p9144_all_roughness.png",
                            "SURF_TEMP_FF/tc_naca0012_ae3933_L1_freezing_fraction_vs_s_slice_0p9144_all_roughness.png",
                            "SURF_TEMP_FF/tc_naca0012_ae3933_L1_surface_temperature_vs_s_slice_0p9144_all_roughness.png",
                        }
                    )
                    and destination not in {
                        "AERODYNAMIC/tc_naca0012_ae3932_cd_vs_n_all_roughness_relative_to_l1.png",
                        "AERODYNAMIC/tc_naca0012_ae3932_cl_vs_n_all_roughness_relative_to_l1.png",
                        "AERODYNAMIC/tc_naca0012_ae3932_cmy_vs_n_all_roughness_relative_to_l1.png",
                        "AERODYNAMIC/tc_naca0012_ae3933_L1_cp_vs_x_slice_0p9144_all_roughness.png",
                    }
                )
            }

        if participant_id is not None:
            for queue in (
                convergence_data_builder.PNG_EXPORT_QUEUE,
                cutdata_builder.PNG_EXPORT_QUEUE,
                iceshape_builder.PNG_EXPORT_QUEUE,
            ):
                for figure, _ in queue:
                    if highlight:
                        highlight_participant(figure, participant_id)
                    bring_participant_to_front(figure, participant_id)

        if participant_id is not None and str(participant_id).zfill(3) == "007":
            cp_legend_plots = {
                "tc_naca0012_ae3932_L1_cp_vs_x_slice_0p9144_all_roughness",
                "tc_naca0012_ae3932_L1_cp_vs_s_slice_0p9144_all_roughness",
            }
            for module in (convergence_data_builder, cutdata_builder, iceshape_builder):
                for figure, export_path in module.PNG_EXPORT_QUEUE:
                    for trace in figure.data:
                        if get_trace_participant_id(trace) == "007":
                            trace.name = re.sub(r"^007(?=\b|\s|[|(])", "CHAMPS", str(trace.name or ""))
                    if export_path.stem in cp_legend_plots:
                        shown_muted = False
                        shown_champs = False
                        shown_experimental = False
                        for trace in figure.data:
                            pid = get_trace_participant_id(trace)
                            is_experimental = (
                                str(trace.legendgroup or "").startswith(("experimental_", "reference_"))
                                or str(trace.name or "").startswith("Exp")
                            )
                            if pid and pid != "007":
                                trace.legend = "legend"
                                trace.name = "IPW3 participants"
                                trace.legendgroup = "ipw3_participants"
                                trace.legendrank = 1
                                trace.showlegend = not shown_muted
                                shown_muted = True
                            elif pid == "007":
                                trace.legend = "legend"
                                if str(trace.legendgroup or "").startswith("007_cp_"):
                                    level_match = re.search(r"L([1-4])$", str(trace.name or ""))
                                    trace.legendrank = (
                                        int(level_match.group(1)) if level_match else 7
                                    )
                                else:
                                    trace.name = "CHAMPS"
                                    trace.legendgroup = "champs"
                                    trace.legendrank = 7
                                    trace.showlegend = not shown_champs
                                    shown_champs = True
                            elif is_experimental:
                                trace.legend = "legend"
                                trace.name = "Exp."
                                trace.legendrank = 10000
                                trace.showlegend = not shown_experimental
                                shown_experimental = True
                        figure.update_layout(legend={
                            "orientation": "h", "x": 0.0, "xanchor": "left",
                            "y": 1.05, "yanchor": "bottom",
                            "traceorder": "normal", "title": {"text": ""},
                            "bgcolor": "rgba(0,0,0,0)",
                        })

        presentation_figures = {
            destination: spec
            for destination, spec in presentation_figures.items()
            if Path(destination).stem not in NACA0012_EXCLUDED_PRESENTATION_STEMS
            and not destination.startswith(("ICE_HORN_PARTICIPANT/", "ICE_LIMITS/"))
        }

        for queue in (
            convergence_data_builder.PNG_EXPORT_QUEUE,
            cutdata_builder.PNG_EXPORT_QUEUE,
            iceshape_builder.PNG_EXPORT_QUEUE,
        ):
            queue[:] = [
                (figure, export_path)
                for figure, export_path in queue
                if export_path.stem not in NACA0012_EXCLUDED_PRESENTATION_STEMS
            ]
            for figure, _ in queue:
                order_comparison_traces(figure, participant_id)

        # Preserve each curated figure's configured presentation canvas instead
        # of replacing it with the plot family's normal PNG export dimensions.
        convergence_data_builder.flush_png_exports(scale=1, width=None, height=None)
        cutdata_builder.flush_png_exports(scale=1, width=None, height=None)
        iceshape_builder.flush_png_exports(scale=1, width=None, height=None)

        missing: list[str] = []
        for destination_text, presentation_spec in presentation_figures.items():
            case_id, source_name, width, height, excluded_ids, show_legend = presentation_spec[:6]
            source = staging_dir / case_id / source_name
            destination = output_dir / destination_text
            if not source.exists():
                missing.append(f"{destination_text} <- {case_id}/{source_name}")
                continue
            destination.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(source, destination)

        if missing and participant_id is None:
            raise RuntimeError("Missing presentation plot exports:\n  " + "\n  ".join(missing))



def generate(
    participants, output_dir: Path = OUTPUT_DIR, participant_id: str | None = None,
    highlight: bool = False,
) -> int:
    if output_dir.exists():
        shutil.rmtree(output_dir)
    write_naca0012_presentation_figures(
        participants, output_dir, participant_id=participant_id, highlight=highlight,
    )
    return sum(1 for path in output_dir.rglob("*.png"))


def main() -> None:
    import build_site_ipw3 as site

    highlight_points = {
        "TC_NACA0012_AE3932": (0.0, None, 0.0),
        "TC_NACA0012_AE3933": (0.0, None, 0.0),
        "TC_ONERAM6": (0.0, None, 0.0),
    }
    participants = site.load_participants(Path("."), highlight_points_by_case=highlight_points)
    convergence_data_builder.apply_participant_mass_conventions(participants)
    figure_count = generate(participants)
    print(f"Wrote {figure_count} NACA0012 presentation figures in {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
