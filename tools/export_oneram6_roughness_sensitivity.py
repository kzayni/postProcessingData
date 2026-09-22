"""Export same-participant, same-grid ONERA M6 roughness differences."""

from collections import defaultdict
from pathlib import Path
import math
import re
from statistics import median

import numpy as np
import pandas as pd

import build_site_ipw3 as site
from tools import convergence_data_builder as convergence
from tools import cutdata_builder, iceshape_builder
from tools.heat_flux_computations import CASE_SETTINGS, recovery_temperature
from tools.gatherParticipantData import (
    CASE_SLICES, decode_slice_position, iter_case_data, iter_grid_datasets,
    parse_ipw3_zone_name,
)


CASE_ID = "TC_ONERAM6"
OUTPUT_DIR = Path("FIGURES_ONERAM6") / "ROUGHNESS"
LEVELS = ("L4", "L3", "L2", "L1")
MASS_COLUMNS = ("WATER_MASS", "ICE_MASS", "WATER_EVAP_MASS")
METADATA_COLUMNS = {
    "N", "GRID_LEVEL", "SOURCE_SHEET", "SOURCE_REQUIREMENT", "BIN_SET",
    "BIN", "DIAMETER", "ROUGHNESS_KEY", "ROUGHNESS_HEIGHT",
}


def height(key):
    if key == "smooth":
        return 0.0
    match = re.fullmatch(r"(\d+(?:\.\d+)?)mm", str(key), re.I)
    return float(match.group(1)) if match else None


def label(key):
    return "Smooth" if key == "smooth" else f"{height(key):g} mm"


def numeric(value):
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return number if math.isfinite(number) and number > -998.0 else None


def add(store, variable, slice_key, participant, level, roughness, value):
    if height(roughness) is None:
        return
    value = numeric(value)
    if value is not None:
        store[variable][slice_key][(str(participant).zfill(3), level, roughness)] = value


def level_from_row(row):
    raw = row.get("GRID_LEVEL", row.get("N"))
    number = convergence.grid_level_number_from_value(raw)
    return f"L{number}" if number in (1, 2, 3, 4) else None


def collect_grid_convergence(participants, store):
    """Discover numeric workbook quantities; compare icing at matching BINS07."""
    for participant, case_data in iter_case_data(participants, CASE_ID):
        workbook = case_data.grid_convergence_data
        if workbook is None:
            continue
        pid = participant.participant_id
        for zone_name, zone in workbook.zones.items():
            is_icing = convergence.extract_icing_bin_set_from_zone_name(zone_name) is not None
            is_diameter = "by_diameter" in zone_name.lower()
            if is_icing:
                if convergence.extract_icing_bin_set_from_zone_name(zone_name) != "BINS07":
                    continue
                roughness = convergence.extract_icing_roughness_key_from_zone_name(zone_name)
                variables = [column for column in MASS_COLUMNS if column in zone.data]
            else:
                roughness = convergence.extract_roughness_key_from_zone_name(zone_name)
                variables = [column for column in zone.data if column.upper() not in METADATA_COLUMNS]
            if height(roughness) is None:
                continue
            if is_diameter:
                # Diameter rows partition the same integrated mass. Sum only
                # valid bins at each grid level; prefer a submitted total below.
                for level_value, group in zone.data.groupby("GRID_LEVEL"):
                    level = level_from_row(group.iloc[0])
                    if level is None:
                        continue
                    for variable in variables:
                        values = pd.to_numeric(group[variable], errors="coerce")
                        values = values[np.isfinite(values) & (values > -998)]
                        if not values.empty:
                            key = (str(pid).zfill(3), level, roughness)
                            store[variable][""].setdefault(key, float(values.sum()))
                continue
            for _, row in zone.data.iterrows():
                level = level_from_row(row)
                if level is None:
                    continue
                for variable in variables:
                    # A direct integrated total takes precedence over bins.
                    add(store, variable, "", pid, level, roughness, row[variable])


def collect_derived_mass_ratios(store):
    water = store.get("WATER_MASS", {}).get("", {})
    ice = store.get("ICE_MASS", {}).get("", {})
    evaporation = store.get("WATER_EVAP_MASS", {}).get("", {})
    for key, mass in water.items():
        if mass == 0:
            continue
        pid, level, roughness = key
        if key in ice:
            add(store, "ICE_TO_WATER_RATIO", "", pid, level, roughness, 100 * ice[key] / mass)
        if key in ice and key in evaporation:
            add(store, "ICE_EVAP_TO_WATER_RATIO", "", pid, level, roughness,
                100 * (ice[key] + evaporation[key]) / mass)


def collect_scalar_plot_traces(participants, store):
    roughnesses = set()
    for level in LEVELS:
        for _, _, _, dataset in iter_grid_datasets(participants, CASE_ID, level):
            cut_data = getattr(dataset, "cut_data", None)
            if cut_data is None:
                continue
            for zone_name in cut_data.zones:
                roughness = convergence._cutdata_roughness_key(zone_name)
                if height(roughness) is not None:
                    roughnesses.add(roughness)
    for slice_position in CASE_SLICES[CASE_ID]:
        slice_key = f"Y = {slice_position:g} m"
        for roughness in sorted(roughnesses, key=height):
            fig, _, _ = convergence.build_qc_prime_integration_figure(
                participants, CASE_ID, slice_position=slice_position,
                roughness_filter=roughness,
            )
            for trace in fig.data:
                pid = (trace.meta or {}).get("ipw3_participant_id")
                if not pid or trace.customdata is None:
                    continue
                for value, custom in zip(trace.y, trace.customdata):
                    add(store, "QC_PRIME", slice_key, pid, str(custom[0]), roughness, value)
        for variable in ("mean_surface_temperature_vs_n", "mean_freezing_fraction_vs_n"):
            spec = next(item for item in convergence.GRID_CONVERGENCE_PLOTS
                        if item["plot_key"] == variable)
            fig, _, _ = convergence.build_cutdata_mean_convergence_figure(
                participants, CASE_ID, spec, slice_position=slice_position,
            )
            target = "MEAN_SURFACE_TEMPERATURE" if "temperature" in variable else "MEAN_FREEZING_FRACTION"
            for trace in fig.data:
                meta = trace.meta if isinstance(trace.meta, dict) else {}
                pid, roughness = meta.get("ipw3_participant_id"), meta.get("ipw3_roughness_key")
                if not pid or height(roughness) is None or trace.customdata is None:
                    continue
                for value, custom in zip(trace.y, trace.customdata):
                    add(store, target, slice_key, pid, str(custom[0]), roughness, value)


def collect_horn_angles(participants, store):
    for level in LEVELS:
        for participant, _, _, dataset in iter_grid_datasets(participants, CASE_ID, level):
            ice_data = getattr(dataset, "ice_shape_data", None)
            if ice_data is None:
                continue
            selected = {}
            for zone_name, zone in ice_data.zones.items():
                info = iceshape_builder.parse_ipw3_ice_shape_zone_name(zone_name)
                if info is None or info["shape_role"] not in {"SINGLE_LAYER", "FINAL_LAYER"}:
                    continue
                if info["bins"] != "BINS07":
                    continue
                roughness = iceshape_builder.extract_roughness_key_from_zone_name(zone_name)
                slice_position = decode_slice_position(info["slice"]) if info["slice"] else None
                if height(roughness) is None or slice_position is None:
                    continue
                x_column, z_column = iceshape_builder.find_submitted_ice_xz_columns(zone.data.columns)
                if x_column is None or z_column is None:
                    continue
                shape = iceshape_builder.valid_submitted_ice_shape_rows(zone.data, x_column, z_column)
                if shape.empty:
                    continue
                geometry = iceshape_builder.upper_horn_geometry(
                    shape[x_column], shape[z_column], CASE_ID, slice_position,
                )
                if geometry is None:
                    continue
                key = (slice_position, roughness)
                role_rank = 2 if info["shape_role"] == "FINAL_LAYER" else 1
                if key not in selected or role_rank > selected[key][0]:
                    selected[key] = (role_rank, geometry[2])
            for (slice_position, roughness), (_, angle) in selected.items():
                add(store, "UPPER_HORN_ANGLE", f"Y = {slice_position:g} m",
                    participant.participant_id, level, roughness, angle)


def profile_mean(frame, x_column, y_values):
    x = pd.to_numeric(frame[x_column], errors="coerce").to_numpy(dtype=float)
    y = np.asarray(y_values, dtype=float)
    valid = np.isfinite(x) & np.isfinite(y) & (y > -998.0)
    if valid.sum() < 2:
        return None
    grouped = pd.DataFrame({"x": x[valid], "y": y[valid]}).groupby("x", as_index=False)["y"].mean()
    grouped = grouped.sort_values("x")
    if len(grouped) < 2:
        return None
    span = float(grouped["x"].iloc[-1] - grouped["x"].iloc[0])
    if span <= 0:
        return None
    return float(np.trapezoid(grouped["y"], grouped["x"]) / span)


def collect_cutdata_profiles(participants, store):
    """Reduce matching 7-bin surface profiles to signed s-weighted means."""
    columns = {
        "Cp": "CP", "HTC": "HTC", "Beta": "BETA", "Ts": "TS",
        "FF": "FF", "RhoIce": "RHO_ICE", "HTC_CLEAN": "HTC_CLEAN",
    }
    settings = CASE_SETTINGS[CASE_ID]
    for level in LEVELS:
        for participant, _, _, dataset in iter_grid_datasets(participants, CASE_ID, level):
            cut_data = getattr(dataset, "cut_data", None)
            if cut_data is None:
                continue
            for zone_name, zone in cut_data.zones.items():
                info = parse_ipw3_zone_name(zone_name)
                if info is None or info["bins"] != "BINS07" or not info["slice"]:
                    continue
                slice_position = decode_slice_position(info["slice"])
                roughness = convergence._cutdata_roughness_key(zone_name)
                if slice_position is None or height(roughness) is None:
                    continue
                frame = zone.data
                x_column = convergence.find_column_case_insensitive(frame.columns, ["s"])
                if x_column is None:
                    continue
                slice_key = f"Y = {slice_position:g} m"
                for source_column, variable in columns.items():
                    y_column = convergence.find_column_case_insensitive(frame.columns, [source_column])
                    if y_column is None:
                        continue
                    y = pd.to_numeric(frame[y_column], errors="coerce").to_numpy(dtype=float)
                    value = profile_mean(frame, x_column, y)
                    add(store, variable, slice_key, participant.participant_id, level, roughness, value)
                    if source_column == "Cp":
                        valid = np.isfinite(y) & (y > -998.0)
                        trec = np.full(len(y), np.nan)
                        trec[valid] = recovery_temperature(y[valid], settings.t_inf, settings.mach_inf)
                        value = profile_mean(frame, x_column, trec)
                        add(store, "TREC", slice_key, participant.participant_id, level, roughness, value)


def transition_rows(values):
    participants = sorted({pid for pid, _, _ in values})
    rows = []
    for pid in participants:
        states = sorted({roughness for p, _, roughness in values if p == pid}, key=height)
        for initial, final in zip(states, states[1:]):
            changes = {}
            for level in LEVELS:
                before = values.get((pid, level, initial))
                after = values.get((pid, level, final))
                if before is not None and after is not None and before != 0:
                    changes[level] = 100 * (after - before) / abs(before)
            if changes:
                rows.append((pid, initial, final, changes, median(changes.values())))
    return rows


def format_number(value):
    return f"{value:.2f}" if value is not None else "--"


def raw_data_table(values):
    lines = [
        "| Participant | Roughness | L4 | L3 | L2 | L1 |",
        "|-------------|-----------|----|----|----|----|",
    ]
    participants = sorted({pid for pid, _, _ in values})
    for pid in participants:
        roughnesses = sorted({roughness for p, _, roughness in values if p == pid}, key=height)
        for roughness in roughnesses:
            cells = [pid, label(roughness)]
            cells += [
                f"{values[(pid, level, roughness)]:.10g}"
                if (pid, level, roughness) in values else "--"
                for level in LEVELS
            ]
            lines.append("| " + " | ".join(cells) + " |")
    return "\n".join(lines)


def table_for_slice(values):
    rows = transition_rows(values)
    if not rows:
        return None
    lines = [
        "| Participant | Roughness change | L4 [%] | L3 [%] | L2 [%] | L1 [%] | Median [%] |",
        "|-------------|------------------|--------|--------|--------|--------|------------|",
    ]
    for pid, initial, final, changes, participant_median in rows:
        cells = [pid, f"{label(initial)} -> {label(final)}"]
        cells += [format_number(changes.get(level)) for level in LEVELS]
        cells.append(format_number(participant_median))
        lines.append("| " + " | ".join(cells) + " |")
    lines += ["", "## Cross-Participant Summary", "",
              "| Roughness change | N | Median [%] | Min [%] | Max [%] |",
              "|------------------|---|------------|---------|---------|"]
    grouped = defaultdict(list)
    for _, initial, final, _, value in rows:
        grouped[(initial, final)].append(value)
    for (initial, final), values in sorted(grouped.items(), key=lambda item: (height(item[0][0]), height(item[0][1]))):
        if len(values) < 2:
            continue
        lines.append(f"| {label(initial)} -> {label(final)} | {len(values)} | "
                     f"{median(values):.2f} | {min(values):.2f} | {max(values):.2f} |")
    if all(len(values) < 2 for values in grouped.values()):
        lines += ["", "No exact transition is shared by multiple participants."]
    return "\n".join(lines)


def main():
    participants = site.load_participants(
        Path("."), highlight_points_by_case={CASE_ID: (0.0, None, 0.0)},
    )
    convergence.apply_participant_mass_conventions(participants)
    store = defaultdict(lambda: defaultdict(dict))
    collect_grid_convergence(participants, store)
    collect_derived_mass_ratios(store)
    collect_scalar_plot_traces(participants, store)
    collect_horn_angles(participants, store)
    collect_cutdata_profiles(participants, store)
    OUTPUT_DIR.mkdir(exist_ok=True)
    generated = []
    for variable, slices in sorted(store.items()):
        sections = []
        raw_sections = []
        has_nonzero_effect = False
        for slice_key, values in sorted(slices.items()):
            rows = transition_rows(values)
            has_nonzero_effect |= any(
                any(change != 0 for change in changes.values())
                for _, _, _, changes, _ in rows
            )
            table = table_for_slice(values)
            if table is None:
                continue
            heading = f"### {slice_key}\n\n" if slice_key else ""
            sections.append(heading + table)
            raw_sections.append(heading + raw_data_table(values))
        if not sections or not has_nonzero_effect:
            continue
        display_variable = "CM" if variable == "CMY" else variable
        description = (
            "Icing mass values use the submitted BINS07 distribution, the bin set shared by optional roughness submissions. "
            "Diameter-resolved water masses are summed when no integrated total is submitted.\n\n"
            if variable in MASS_COLUMNS or variable in {"ICE_TO_WATER_RATIO", "ICE_EVAP_TO_WATER_RATIO"} else ""
        )
        description += (
            "CM uses the workbook's CMY pitching-moment coefficient.\n\n"
            if variable == "CMY" else ""
        )
        description += (
            "Surface-profile values are signed, distance-weighted means over the submitted 7-bin cut at each spanwise slice. "
            "No roughness state or spatial point is interpolated.\n\n"
            if variable in {"CP", "HTC", "BETA", "TS", "FF", "RHO_ICE", "HTC_CLEAN", "TREC"} else ""
        )
        description += (
            "Variable roughness has no single numeric height and is excluded from ordered height transitions. "
            "A transition is reported only where both states have values at the same grid level. "
            "Undefined percentages with a zero initial value are omitted."
        )
        path = OUTPUT_DIR / f"{display_variable}.md"
        path.write_text(
            f"# {display_variable} — Roughness Sensitivity\n\n{description}\n\n"
            "## Percentage Change Between Roughness Heights\n\n"
            + "\n\n".join(sections)
            + "\n\n## Raw Data Used\n\n"
            + "\n\n".join(raw_sections) + "\n",
            encoding="utf-8",
        )
        generated.append(path)
    for path in generated:
        print(path)


if __name__ == "__main__":
    main()
