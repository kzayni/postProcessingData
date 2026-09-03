"""Compute and plot integrated convective heat transfer per unit span, Q_c'.

Edit ``CASE_SETTINGS`` to set a permanent integration window for each case, or
use ``--ds-window CASE:MIN:MAX`` for a one-off override.  ``None`` means the
lowest/highest submitted surface coordinate.
"""

from __future__ import annotations

import argparse
import csv
from dataclasses import dataclass, replace
import math
from pathlib import Path
import re

import numpy as np
import plotly.graph_objects as go

try:
    from .participant_style import (
        participant_color,
        participant_legend_rank,
        participant_marker,
        participant_trace_mode,
    )
    from .plot_style import GRID_CONVERGENCE_NORMALIZATION, apply_xy_style
except ImportError:  # Allow ``python3 tools/heat_flux_computations.py``.
    from participant_style import (
        participant_color,
        participant_legend_rank,
        participant_marker,
        participant_trace_mode,
    )
    from plot_style import GRID_CONVERGENCE_NORMALIZATION, apply_xy_style


ROOT = Path(__file__).resolve().parents[1]
LEVELS = ("L1", "L2", "L3", "L4")


def grid_cell_counts_for_case(case_id: str) -> dict[str, float]:
    reference_name = "ONERAM6_GRID.dat" if "ONERAM6" in case_id.upper() else "NACA0012_GRID.dat"
    reference_path = ROOT / "R00_REFERENCE" / reference_name
    counts: dict[str, float] = {}
    if not reference_path.exists():
        return counts
    for line in reference_path.read_text().splitlines():
        values = line.strip().split()
        if len(values) < 2:
            continue
        try:
            level, num_cells = int(float(values[0])), float(values[1])
        except ValueError:
            continue
        if level > 0 and num_cells > 0.0:
            counts[f"L{level}"] = num_cells
    return counts


def grid_convergence_coordinate(num_cells: float, l1_num_cells: float) -> float:
    return math.log(l1_num_cells / num_cells) / 3.0


@dataclass(frozen=True)
class CaseSettings:
    directory_names: tuple[str, ...]
    ds_min: float | None = None
    ds_max: float | None = None
    # Case static temperature used as T_inf in the Q_c' integral.
    t_inf: float = 262.305
    mach_inf: float = 0.0


# Change ds_min and ds_max here to use a different window for each case.
CASE_SETTINGS: dict[str, CaseSettings] = {
    "TC_NACA0012_AE3932": CaseSettings(
        ("TC_NACA0012_AE3932_D01", "NACA0012_AE3932_D01"),
        ds_min=-0.28,
        ds_max=0.27,
        t_inf=262.305,
        mach_inf=0.3151,
    ),
    "TC_NACA0012_AE3933": CaseSettings(
        ("TC_NACA0012_AE3933_D01", "NACA0012_AE3933_D01"),
        ds_min=-0.28,
        ds_max=0.27,
        t_inf=265.8,
        mach_inf=0.3127,
    ),
    "TC_ONERAM6": CaseSettings(
        ("TC_ONERAM6_D01",),
        ds_min=-0.50,
        ds_max=0.50,
        t_inf=267.16,
        mach_inf=0.4,
    ),
}


def parse_variables_and_first_zone(path: Path) -> tuple[list[str], np.ndarray]:
    variables: list[str] = []
    rows: list[list[float]] = []
    in_zone = False
    for line in path.read_text(encoding="utf-8", errors="ignore").splitlines():
        stripped = line.strip()
        if stripped.upper().startswith("VARIABLES"):
            variables = re.findall(r'"([^"]+)"', stripped)
        elif stripped.upper().startswith("ZONE"):
            if in_zone:
                break
            in_zone = True
        elif in_zone and stripped and not stripped.startswith("#"):
            try:
                row = [float(value) for value in stripped.replace(",", " ").split()]
            except ValueError:
                continue
            if not variables:
                rows.append(row)
            elif len(row) <= len(variables):
                rows.append(row + [-999.0] * (len(variables) - len(row)))
    return variables, np.asarray(rows, dtype=float)


def find_column(variables: list[str], candidates: tuple[str, ...]) -> int | None:
    lookup = {name.strip().lower(): index for index, name in enumerate(variables)}
    return next((lookup[name.lower()] for name in candidates if name.lower() in lookup), None)


def read_freestream_temperature(path: Path, fallback: float) -> float:
    header = path.read_text(encoding="utf-8", errors="ignore")[:8000]
    match = re.search(r"Freestream\s+Temp(?:erature)?\s*:\s*([-+\d.eE]+)", header, re.I)
    return float(match.group(1)) if match else fallback


def recovery_temperature(
    cp: np.ndarray,
    t_inf: float,
    mach_inf: float,
    gamma: float = 1.4,
    recovery_factor: float = 0.9,
) -> np.ndarray:
    """Approximate local recovery temperature from Cp and freestream state."""
    pressure_term = 1.0 + 0.5 * gamma * mach_inf**2 * cp
    total_pressure_ratio = (1.0 + 0.5 * (gamma - 1.0) * mach_inf**2) ** (
        gamma / (gamma - 1.0)
    )
    with np.errstate(invalid="ignore", divide="ignore"):
        mach_e_squared = 2.0 / (gamma - 1.0) * (
            (total_pressure_ratio / pressure_term) ** ((gamma - 1.0) / gamma) - 1.0
        )
        mach_e = np.sqrt(mach_e_squared)
        t_zero = t_inf * (1.0 + 0.5 * (gamma - 1.0) * mach_inf**2)
        t_e = t_zero / (1.0 + 0.5 * (gamma - 1.0) * mach_e**2)
        return t_e * (1.0 + recovery_factor * 0.5 * (gamma - 1.0) * mach_e**2)


def _window_with_interpolated_edges(
    s: np.ndarray, q: np.ndarray, ds_min: float | None, ds_max: float | None
) -> tuple[np.ndarray, np.ndarray]:
    order = np.argsort(s)
    s, q = s[order], q[order]
    # Duplicate s values make interpolation ambiguous; average their q values.
    unique_s, inverse = np.unique(s, return_inverse=True)
    sums = np.bincount(inverse, weights=q)
    q = sums / np.bincount(inverse)
    s = unique_s
    lower = s[0] if ds_min is None else ds_min
    upper = s[-1] if ds_max is None else ds_max
    if lower >= upper or upper < s[0] or lower > s[-1]:
        return np.array([]), np.array([])
    lower, upper = max(lower, s[0]), min(upper, s[-1])
    inside = (s > lower) & (s < upper)
    clipped_s = np.concatenate(([lower], s[inside], [upper]))
    clipped_q = np.interp(clipped_s, s, q)
    return clipped_s, clipped_q


def integrate_file(
    path: Path,
    ds_min: float | None = None,
    ds_max: float | None = None,
    t_inf: float | None = None,
    mach_inf: float = 0.0,
    fallback_t_inf: float = 262.305,
) -> tuple[float | None, str | None]:
    variables, cut = parse_variables_and_first_zone(path)
    htc_col = find_column(variables, ("HTC", "HeatTransferCoefficient"))
    ts_col = find_column(variables, ("Ts", "WallTemperature", "SurfaceTemperature"))
    cp_col = find_column(variables, ("Cp", "PressureCoefficient"))
    if htc_col is None or ts_col is None or cp_col is None:
        return None, "missing Cp, HTC, or Ts column"
    mapping_path = path.with_name(path.stem + "_sMap.dat")
    if not mapping_path.exists():
        return None, "missing sMap file"
    map_variables, mapping = parse_variables_and_first_zone(mapping_path)
    s_col = find_column(map_variables, ("s",))
    index_col = find_column(map_variables, ("variable Index", "VARIABLE_INDEX"))
    if s_col is None or index_col is None or len(mapping) != len(cut):
        return None, "invalid sMap file"
    indices = np.rint(mapping[:, index_col]).astype(int)
    if not len(indices) or indices.min() < 0 or indices.max() >= len(cut):
        return None, "sMap index outside data"
    s, htc, ts, cp = mapping[:, s_col], cut[indices, htc_col], cut[indices, ts_col], cut[indices, cp_col]
    valid = np.isfinite(s) & np.isfinite(htc) & np.isfinite(ts) & np.isfinite(cp) & (htc > -998) & (ts > -998) & (cp > -998)
    if valid.sum() < 2:
        return None, "no valid HTC/Ts rows"
    actual_t_inf = read_freestream_temperature(path, fallback_t_inf) if t_inf is None else t_inf
    t_rec = recovery_temperature(cp[valid], actual_t_inf, mach_inf)
    valid_recovery = np.isfinite(t_rec)
    s, q = _window_with_interpolated_edges(
        s[valid][valid_recovery],
        htc[valid][valid_recovery] * (ts[valid][valid_recovery] - t_rec[valid_recovery]),
        ds_min,
        ds_max,
    )
    if len(s) < 2:
        return None, "ds window does not overlap at least two surface locations"
    return float(np.trapezoid(q, s)), None


def iter_case_files(settings: CaseSettings):
    for directory_name in settings.directory_names:
        yield from ROOT.glob(f"*/{directory_name}/*_cutData_V1.dat")


def compute_case(case_id: str, settings: CaseSettings):
    values: dict[str, dict[str, float]] = {}
    skipped: list[tuple[str, str, str]] = []
    for path in sorted(set(iter_case_files(settings))):
        level_match = re.search(r"_(L[1-4])_cutData", path.name, re.I)
        if level_match is None:
            continue
        participant, level = path.parent.parent.name, level_match.group(1).upper()
        value, reason = integrate_file(
            path, settings.ds_min, settings.ds_max, t_inf=settings.t_inf, mach_inf=settings.mach_inf
        )
        if value is None:
            skipped.append((participant, level, reason or "unknown error"))
        else:
            values.setdefault(participant, {})[level] = value
    return values, skipped


def write_case_outputs(
    case_id: str,
    settings: CaseSettings,
    values: dict[str, dict[str, float]],
    output_dir: Path = ROOT,
    image_scale: int = 1,
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    slug = case_id.removeprefix("TC_")
    csv_path = output_dir / f"ALL_PARTICIPANTS_{slug}_Qc_prime_values.csv"
    with csv_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(["participant", "ds_min_m", "ds_max_m", *[f"{x}_W_per_m" for x in LEVELS]])
        for participant, by_level in sorted(values.items()):
            writer.writerow([participant, settings.ds_min, settings.ds_max,
                             *[f"{by_level[x]:.9g}" if x in by_level else "" for x in LEVELS]])

    cell_counts = grid_cell_counts_for_case(case_id)
    fig = go.Figure()
    for participant, by_level in sorted(values.items()):
        available = [level for level in LEVELS if level in by_level]
        participant_id = participant.split("_", 1)[0]
        fig.add_scatter(
            x=[grid_convergence_coordinate(cell_counts[level], cell_counts["L1"]) for level in available],
            y=[by_level[level] for level in available],
            mode=participant_trace_mode(participant_id),
            name=participant_id,
            legendgroup=participant_id,
            legendrank=participant_legend_rank(participant_id),
            line={"color": participant_color(participant_id)},
            marker=participant_marker(participant_id),
            customdata=available,
            hovertemplate="Grid level=%{customdata}<br>log(h/h_L1)=%{x:.6g}<br>Q_c'=%{y:.6g} W/m<extra></extra>",
        )
    window = f"ds = {settings.ds_min if settings.ds_min is not None else 'data min'} to " \
             f"{settings.ds_max if settings.ds_max is not None else 'data max'} m"
    apply_xy_style(
        fig,
        case_id,
        "log(h/h<sub>L1</sub>) [-]",
        "Q_c' = ∫ HTC (Ts − Trec) ds [W/m]",
        plot_family="convergence",
        plot_key="qc_prime",
        height=700,
    )
    fig.update_layout(
        title={"text": f"Integrated Convective Heat Transfer per Unit Span (Q_c') Grid Convergence | All Participants | {case_id}",
               "x": 0.5, "xanchor": "center"},
        width=2000,
        margin={"l": 90, "r": 220, "t": 100, "b": 95},
        annotations=[{"text": window, "xref": "paper", "yref": "paper", "x": 0,
                      "y": -0.12, "showarrow": False}],
    )
    fig.update_xaxes(
        type="linear",
        tickmode="array",
        tickvals=[grid_convergence_coordinate(cell_counts[level], cell_counts["L1"]) for level in LEVELS],
        ticktext=[f"{grid_convergence_coordinate(cell_counts[level], cell_counts['L1']):.3f}" for level in LEVELS],
        title={"text": "log(h/h<sub>L1</sub>) [-]"},
    )
    fig.write_image(
        str(output_dir / f"ALL_PARTICIPANTS_{slug}_Qc_prime_grid_convergence.png"),
        width=2000,
        height=700,
        scale=image_scale,
    )
    if GRID_CONVERGENCE_NORMALIZATION.get("enabled", True):
        relative_fig = go.Figure(fig)
        reference_level = str(GRID_CONVERGENCE_NORMALIZATION.get("reference_grid_level", "L1"))
        for trace in relative_fig.data:
            levels = [str(level) for level in trace.customdata]
            if reference_level not in levels:
                trace.visible = "legendonly"
                continue
            reference_value = float(trace.y[levels.index(reference_level)])
            if not np.isfinite(reference_value) or reference_value == 0.0:
                trace.visible = "legendonly"
                continue
            trace.y = [(float(value) - reference_value) / reference_value * 100.0 for value in trace.y]
        relative_fig.update_layout(
            title={"text": f"Integrated Convective Heat Transfer per Unit Span (Q_c') Grid Convergence | Signed Relative Difference from L1 | {case_id}"},
        )
        relative_fig.update_yaxes(title={"text": "Q_c' difference from L1 [%]"})
        relative_fig.write_image(
            str(output_dir / f"ALL_PARTICIPANTS_{slug}_Qc_prime_grid_convergence_relative_to_L1.png"),
            width=2000,
            height=700,
            scale=image_scale,
        )


def parse_window_override(value: str) -> tuple[str, float | None, float | None]:
    try:
        case_id, lower, upper = value.split(":")
        return case_id, None if lower.lower() == "none" else float(lower), \
            None if upper.lower() == "none" else float(upper)
    except ValueError as exc:
        raise argparse.ArgumentTypeError("use CASE:MIN:MAX (use 'none' for an open end)") from exc


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--var", choices=("qc_prime", "qc"), default="qc_prime",
                        help="Generate integrated convective heat transfer per unit span (Q_c'); 'qc' remains an alias.")
    parser.add_argument("--case", action="append", choices=tuple(CASE_SETTINGS),
                        help="Generate only this case; repeat for multiple cases.")
    parser.add_argument("--ds-window", action="append", default=[], type=parse_window_override,
                        metavar="CASE:MIN:MAX", help="Override a case's ds window; 'none' opens an end.")
    args = parser.parse_args()

    selected = set(args.case or CASE_SETTINGS)
    settings = {case_id: value for case_id, value in CASE_SETTINGS.items() if case_id in selected}
    for case_id, lower, upper in args.ds_window:
        if case_id not in settings:
            parser.error(f"ds-window case is not selected or known: {case_id}")
        settings[case_id] = replace(settings[case_id], ds_min=lower, ds_max=upper)

    for case_id, case_settings in settings.items():
        values, skipped = compute_case(case_id, case_settings)
        write_case_outputs(case_id, case_settings, values)
        print(f"{case_id}: {len(values)} participants; {len(skipped)} skipped files")
        for participant, level, reason in skipped:
            print("SKIPPED", case_id, participant, level, reason)


if __name__ == "__main__":
    main()
