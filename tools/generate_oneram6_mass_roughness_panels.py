"""Export ONERA M6 7-bin water and ice mass roughness participant panels."""

from pathlib import Path
import shutil

import pandas as pd
import plotly.graph_objects as go

import build_site_ipw3 as site
from tools import convergence_data_builder as convergence
from tools.ONERAM6_PRES_IMAGES import (
    CASE_ID, COEFFICIENT_ROUGHNESS_STYLES,
    ONERAM6_PARTICIPANT_TURBULENCE_LABELS, OUTPUT_DIR,
)
from tools.gatherParticipantData import iter_case_data
from tools.roughness_panels import build_roughness_participant_panels


def build_mass_panel(participants, plot_key: str) -> go.Figure:
    spec = next(item for item in convergence.GRID_CONVERGENCE_PLOTS
                if item["plot_key"] == plot_key)
    mass_column = "WATER_MASS" if plot_key == "water_mass_vs_n" else "ICE_MASS"
    combined = go.Figure()
    for participant, case_data in iter_case_data(participants, CASE_ID):
        data = case_data.grid_convergence_data
        if data is None:
            continue
        pid = str(participant.participant_id).zfill(3)
        for rank, (roughness, (label, color)) in enumerate(COEFFICIENT_ROUGHNESS_STYLES.items()):
            if roughness == "smooth":
                continue
            matching = []
            for zone_name, zone in data.zones.items():
                if convergence.extract_icing_bin_set_from_zone_name(zone_name) != "BINS07":
                    continue
                if convergence.extract_icing_roughness_key_from_zone_name(zone_name) != roughness:
                    continue
                frame = zone.data
                if mass_column not in frame or "N" not in frame:
                    continue
                values = frame[["N", mass_column]].copy()
                values = values.apply(pd.to_numeric, errors="coerce")
                values = values[(values["N"] > 0) & (values[mass_column] > -998)]
                if values.empty:
                    continue
                if "by_diameter" in zone_name.lower():
                    values = values.groupby("N", as_index=False)[mass_column].sum()
                matching.append(("by_diameter" in zone_name.lower(), values))
            if not matching:
                continue
            # Prefer submitted totals when the same values also appear by diameter.
            values = sorted(matching, key=lambda item: item[0])[0][1]
            values = convergence.add_grid_spacing_column(values, CASE_ID, "N")
            values = values.sort_values(convergence.GRID_SPACING_COLUMN)
            if values.empty:
                continue
            combined.add_trace(go.Scatter(
                x=values[convergence.GRID_SPACING_COLUMN], y=values[mass_column],
                mode="lines+markers", name=label,
                legendgroup=f"{plot_key}_roughness_{roughness}", legendrank=rank,
                line={"color": color, "width": 5},
                marker={"color": color, "size": 9, "line": {"color": "#000000", "width": 1}},
                meta={"ipw3_participant_id": pid, "ipw3_roughness_key": roughness},
            ))
    if not combined.data:
        raise RuntimeError(f"No ONERA M6 7-bin data for {plot_key}")
    roughness_by_participant = {}
    for trace in combined.data:
        meta = trace.meta
        roughness_by_participant.setdefault(meta["ipw3_participant_id"], set()).add(
            meta["ipw3_roughness_key"]
        )
    combined.data = tuple(
        trace for trace in combined.data
        if len(roughness_by_participant[trace.meta["ipw3_participant_id"]]) > 1
    )
    if not combined.data:
        raise RuntimeError(f"No participants submitted multiple roughness values for {plot_key}")
    convergence.style_xy_figure(combined, CASE_ID, plot_key, spec["x_label"], spec["y_label"])
    convergence.style_grid_level_x_axis(combined, CASE_ID)
    if plot_key == "water_mass_vs_n":
        combined.update_yaxes(range=[3, 4], autorange=False)
    else:
        mass_values = [float(value) for trace in combined.data for value in trace.y]
        span = max(mass_values) - min(mass_values)
        padding = max(span * 0.1, 0.05)
        combined.update_yaxes(
            range=[min(mass_values) - padding, max(mass_values) + padding],
            autorange=False,
        )
    panel = build_roughness_participant_panels(
        combined, columns=2, participant_title_size=30,
        participant_model_labels=ONERAM6_PARTICIPANT_TURBULENCE_LABELS,
    )
    if plot_key == "water_mass_vs_n":
        panel.update_layout(legend={"y": 1.16})
    return panel


def main() -> None:
    participants = site.load_participants(
        Path("."), highlight_points_by_case={CASE_ID: (0.0, None, 0.0)},
    )
    convergence.apply_participant_mass_conventions(participants)
    for plot_key, category in (("water_mass_vs_n", "IMPINGEMENT"),
                               ("ice_mass_vs_n", "ICE_ACCRETION")):
        figure = build_mass_panel(participants, plot_key)
        destination = OUTPUT_DIR / category / f"tc_oneram6_{plot_key}_grouped_roughness_participants.png"
        destination.parent.mkdir(parents=True, exist_ok=True)
        figure.write_image(destination)
        print(destination)
        roughness_destination = OUTPUT_DIR / "ROUGHNESS" / destination.name
        roughness_destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(destination, roughness_destination)
        print(roughness_destination)


if __name__ == "__main__":
    main()
