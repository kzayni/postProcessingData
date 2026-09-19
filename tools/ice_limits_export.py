"""Presentation plots of geometric ice limits from submitted ice contours."""

from __future__ import annotations

from collections import defaultdict
from functools import lru_cache
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.graph_objects as go

from .gatherParticipantData import decode_slice_position, iter_grid_datasets
from .iceshape_builder import (
    extract_roughness_key_from_zone_name,
    find_submitted_ice_xz_columns,
    load_clean_reference_data,
    onera_clean_reference_points,
    ordered_clean_reference_columns,
    parse_ipw3_ice_shape_zone_name,
    valid_submitted_ice_shape_rows,
)
from .participant_style import participant_marker
from .plot_style import NACA0012_ROUGHNESS_GROUP_STYLE, apply_xy_style


# A point belongs to the iced region when its shortest distance to the clean
# section is at least this value. The two crossings are the reported ice limits.
ICE_THICKNESS_THRESHOLD_M = 0.001
GRID_LEVELS = ("L1", "L2", "L3", "L4")
BIN_SETS = ("BINS01", "BINS03", "BINS07", "BINS15")
M6_ROUGHNESS_COLORS = {
    "0.5mm": "#2CA02C", "1mm": "#D62728",
    "1.5mm": "#1F77B4", "variable_roughness": "#000000",
}
ROUGHNESS_LABELS = {
    "0.5mm": "0.5 mm", "0.5334mm": "0.5334 mm",
    "1mm": "1 mm", "1.5mm": "1.5 mm",
    "variable_roughness": "Variable", "default_roughness": "Unspecified",
}


@lru_cache(maxsize=None)
def clean_section(case_id: str, slice_position: float) -> tuple[np.ndarray, np.ndarray]:
    if case_id == "TC_ONERAM6":
        reference = onera_clean_reference_points(slice_position)
        if reference is None:
            raise ValueError(f"No clean M6 section at Y={slice_position:g} m")
        return np.asarray(reference[0], dtype=float), np.asarray(reference[1], dtype=float)
    path = Path("R00_REFERENCE/NACA0012_CLEAN_ROTATED.dat")
    reference = load_clean_reference_data(str(path))
    zone = next(iter(reference.zones.values()))
    x, z = ordered_clean_reference_columns(case_id, zone, "CoordinateX", "CoordinateZ")
    return np.asarray(x, dtype=float), np.asarray(z, dtype=float)


def ice_limits_from_contour(
    ice_x: np.ndarray, ice_z: np.ndarray, clean_x: np.ndarray, clean_z: np.ndarray,
    threshold_m: float = ICE_THICKNESS_THRESHOLD_M,
) -> tuple[float, float, float] | None:
    """Project ice points onto clean segments and return lower/upper s and width."""
    clean = np.column_stack((clean_x, clean_z))
    ice = np.column_stack((ice_x, ice_z))
    valid = np.isfinite(ice).all(axis=1) & (ice > -998).all(axis=1)
    ice = ice[valid]
    if len(ice) < 3 or len(clean) < 3:
        return None
    starts = clean[:-1]
    segments = np.diff(clean, axis=0)
    lengths = np.linalg.norm(segments, axis=1)
    usable = lengths > 1e-12
    starts, segments, lengths = starts[usable], segments[usable], lengths[usable]
    if not len(starts):
        return None
    cumulative = np.r_[0.0, np.cumsum(np.linalg.norm(np.diff(clean, axis=0), axis=1))]
    segment_starts_s = cumulative[:-1][usable]
    leading_index = int(np.argmin(clean_x))
    leading_s = cumulative[leading_index]
    # NACA reference runs TE-upper-LE-lower-TE; M6 runs the opposite way.
    before_is_upper = float(np.median(clean_z[:leading_index])) > float(np.median(clean_z[leading_index + 1:]))
    projected_positions = []
    thicknesses = []
    for chunk_start in range(0, len(ice), 256):
        points = ice[chunk_start:chunk_start + 256]
        fractions = np.clip(
            np.einsum("ijk,jk->ij", points[:, None, :] - starts[None, :, :], segments)
            / (lengths * lengths), 0.0, 1.0,
        )
        closest = starts[None, :, :] + fractions[:, :, None] * segments[None, :, :]
        distances = np.linalg.norm(points[:, None, :] - closest, axis=2)
        nearest = np.argmin(distances, axis=1)
        rows = np.arange(len(points))
        projected_positions.extend((segment_starts_s[nearest] + fractions[rows, nearest] * lengths[nearest]).tolist())
        thicknesses.extend(distances[rows, nearest].tolist())
    projected = np.asarray(projected_positions)
    thickness = np.asarray(thicknesses)
    signed_s = np.abs(projected - leading_s)
    before_leading = projected < leading_s
    signed_s[before_leading] *= 1 if before_is_upper else -1
    signed_s[~before_leading] *= -1 if before_is_upper else 1
    iced_s = signed_s[thickness >= threshold_m]
    if len(iced_s) < 3:
        return None
    lower, upper = float(np.min(iced_s)), float(np.max(iced_s))
    return lower, upper, upper - lower


def collect_ice_limits(participants, case_id: str, slice_filter: float | None = None) -> dict[tuple[str, str, str, str, float], tuple[float, float, float]]:
    """Use one standard-density single-layer contour per submitted condition."""
    selected: dict[tuple[str, str, str, str, float], tuple[int, tuple[float, float, float]]] = {}
    for level in GRID_LEVELS:
        for participant, _, _, dataset in iter_grid_datasets(participants, case_id, level):
            ice_data = getattr(dataset, "ice_shape_data", None)
            if ice_data is None:
                continue
            for zone_name, zone in ice_data.zones.items():
                info = parse_ipw3_ice_shape_zone_name(zone_name)
                if info is None or info["type"] != "SLICE" or info["bins"] not in BIN_SETS:
                    continue
                if info.get("density_model") == "variable":
                    continue
                slice_position = decode_slice_position(info["slice"])
                if slice_position is None or (slice_filter is not None and not np.isclose(slice_position, slice_filter)):
                    continue
                role = info["shape_role"]
                if role not in {"SINGLE_LAYER", "FINAL_LAYER"}:
                    continue
                x_column, z_column = find_submitted_ice_xz_columns(zone.data.columns)
                if x_column is None or z_column is None:
                    continue
                rows = valid_submitted_ice_shape_rows(zone.data, x_column, z_column)
                if rows.empty:
                    continue
                ice_x = pd.to_numeric(rows[x_column], errors="coerce").to_numpy(dtype=float)
                ice_z = pd.to_numeric(rows[z_column], errors="coerce").to_numpy(dtype=float)
                clean_x, clean_z = clean_section(case_id, slice_position)
                limits = ice_limits_from_contour(ice_x, ice_z, clean_x, clean_z)
                if limits is None:
                    continue
                key = (str(participant.participant_id).zfill(3), level, info["bins"],
                       extract_roughness_key_from_zone_name(zone_name), slice_position)
                rank = 0 if role == "SINGLE_LAYER" else 1
                if key not in selected or rank < selected[key][0]:
                    selected[key] = (rank, limits)
    return {key: value for key, (_, value) in selected.items()}


def _roughness_color(case_id: str, roughness: str) -> str:
    if case_id == "TC_ONERAM6":
        return M6_ROUGHNESS_COLORS.get(roughness, "#777777")
    if roughness not in {"0.5mm", "0.5334mm", "1mm", "variable_roughness"}:
        return "#777777"
    group = "half_mm" if roughness in {"0.5mm", "0.5334mm"} else "one_mm" if roughness == "1mm" else "variable"
    return NACA0012_ROUGHNESS_GROUP_STYLE[group]["line"]["color"]


def _style(fig: go.Figure, case_id: str, y_label: str, x_label: str) -> None:
    apply_xy_style(fig, case_id, x_label, y_label, plot_family="convergence", plot_key="ice_limits")
    fig.update_layout(margin=dict(l=105, r=45, t=105, b=90), legend=dict(orientation="h", x=0, y=1.03),
                      paper_bgcolor="white", plot_bgcolor="white")
    fig.update_xaxes(showgrid=True)
    fig.update_yaxes(showgrid=True)


def ice_limit_figures(participants, case_id: str, slice_filter: float | None = None):
    """Yield named grid and roughness comparisons for every available slice."""
    values = collect_ice_limits(participants, case_id, slice_filter=slice_filter)
    if not values:
        return
    metric_specs = ((2, "ice_width", "Width<sub>ice</sub> [m]"),)
    slices = sorted({key[4] for key in values})
    for slice_position in slices:
        slug = str(slice_position).replace(".", "p")
        # BINS15 grid convergence, with a separate line for each roughness.
        series = defaultdict(dict)
        for (pid, level, bins, roughness, y), limits in values.items():
            if y == slice_position and bins == "BINS15":
                series[(pid, roughness)][level] = limits
        for metric_index, metric_name, y_label in metric_specs:
            fig = go.Figure()
            for (pid, roughness), levels in sorted(series.items()):
                ordered = [level for level in GRID_LEVELS if level in levels]
                if not ordered:
                    continue
                fig.add_trace(go.Scatter(
                    x=ordered, y=[levels[level][metric_index] for level in ordered],
                    name=ROUGHNESS_LABELS.get(roughness, roughness),
                    legendgroup=f"ice_width_roughness_{roughness}", mode="lines+markers",
                    line=dict(color=_roughness_color(case_id, roughness), width=5),
                    marker={**participant_marker(pid), "size": 9,
                            "color": _roughness_color(case_id, roughness),
                            "line": {"color": "#000000", "width": 1}},
                    meta={"ipw3_participant_id": pid},
                    hovertemplate="%{fullData.name}<br>Grid=%{x}<br>Value=%{y:.5g} m<extra></extra>",
                ))
            if fig.data:
                _style(fig, case_id, y_label, "Grid level")
                fig.update_xaxes(categoryorder="array", categoryarray=list(GRID_LEVELS))
                yield f"{case_id.lower()}_{metric_name}_bins15_slice_{slug}_grouped_roughness.png", fig

        relative_fig = go.Figure()
        for (pid, roughness), levels in sorted(series.items()):
            baseline = levels.get("L1")
            if baseline is None or baseline[2] == 0:
                continue
            ordered = [level for level in GRID_LEVELS if level in levels]
            relative_fig.add_trace(go.Scatter(
                x=ordered,
                y=[100 * (levels[level][2] - baseline[2]) / baseline[2] for level in ordered],
                name=ROUGHNESS_LABELS.get(roughness, roughness),
                legendgroup=f"ice_width_roughness_{roughness}", mode="lines+markers",
                line=dict(color=_roughness_color(case_id, roughness), width=5),
                marker={**participant_marker(pid), "size": 9,
                        "color": _roughness_color(case_id, roughness),
                        "line": {"color": "#000000", "width": 1}},
                meta={"ipw3_participant_id": pid},
                hovertemplate="%{fullData.name}<br>Grid=%{x}<br>Width change=%{y:.5g}%<extra></extra>",
            ))
        if relative_fig.data:
            _style(relative_fig, case_id, "ΔWidth<sub>ice</sub> from L1 [%]", "Grid level")
            relative_fig.update_xaxes(categoryorder="array", categoryarray=list(GRID_LEVELS))
            yield f"{case_id.lower()}_ice_width_bins15_slice_{slug}_grouped_roughness_relative_to_l1.png", relative_fig

        bin_series = defaultdict(dict)
        for (pid, level, bins, roughness, y), limits in values.items():
            if y == slice_position and level == "L1":
                bin_series[(pid, roughness)][bins] = limits[2]
        for relative in (False, True):
            fig = go.Figure()
            for (pid, roughness), bins_values in sorted(bin_series.items()):
                ordered = [bins for bins in BIN_SETS if bins in bins_values]
                baseline = bins_values.get("BINS15")
                if len(ordered) < 2 or (relative and (baseline is None or baseline == 0)):
                    continue
                fig.add_trace(go.Scatter(
                    x=[1 / int(bins[4:]) for bins in ordered],
                    y=[100 * (bins_values[bins] - baseline) / baseline if relative else bins_values[bins] for bins in ordered],
                    name=f"{pid} | {ROUGHNESS_LABELS.get(roughness, roughness)}",
                    legendgroup=f"{pid}_{roughness}", mode="lines+markers",
                    line=dict(color=_roughness_color(case_id, roughness), width=3),
                    marker={**participant_marker(pid), "size": 9},
                    meta={"ipw3_participant_id": pid},
                    hovertemplate="%{fullData.name}<br>1/Nbins=%{x:.4g}<br>Value=%{y:.5g}<extra></extra>",
                ))
            if fig.data:
                _style(fig, case_id,
                       "ΔWidth<sub>ice</sub> from 15 bins [%]" if relative else "Width<sub>ice</sub> [m]",
                       "1 / N<sub>bins</sub> [-]")
                fig.update_xaxes(type="log", range=[-1.25, 0.05], tickmode="array",
                                 tickvals=[1 / 15, 1 / 7, 1 / 3, 1],
                                 ticktext=["1/15", "1/7", "1/3", "1"])
                suffix = "_relative_to_bins15" if relative else ""
                yield f"{case_id.lower()}_ice_width_L1_slice_{slug}_vs_inverse_bins{suffix}.png", fig

        # At L1, compare roughness heights for participants that submitted them.
        roughness_series = defaultdict(dict)
        comparison_bins = "BINS07" if case_id == "TC_ONERAM6" else "BINS15"
        for (pid, level, bins, roughness, y), limits in values.items():
            if y == slice_position and level == "L1" and bins == comparison_bins:
                roughness_series[pid][roughness] = limits
        for metric_index, metric_name, y_label in metric_specs:
            fig = go.Figure()
            for pid, conditions in sorted(roughness_series.items()):
                ordered = sorted(conditions, key=lambda key: (key == "variable_roughness", key))
                fig.add_trace(go.Scatter(
                    x=[ROUGHNESS_LABELS.get(key, key) for key in ordered],
                    y=[conditions[key][metric_index] for key in ordered],
                    name=pid, legendgroup=pid,
                    mode="lines+markers" if len(ordered) > 1 else "markers",
                    marker={**participant_marker(pid), "size": 11},
                    meta={"ipw3_participant_id": pid},
                    hovertemplate="Participant %{fullData.name}<br>Roughness=%{x}<br>Value=%{y:.5g} m<extra></extra>",
                ))
            if fig.data:
                _style(fig, case_id, y_label, "Roughness height")
                yield f"{case_id.lower()}_{metric_name}_L1_{comparison_bins.lower()}_slice_{slug}_roughness_effect.png", fig
