"""Additional POLIMO NACA presentation plots.

BETA_COMPARISONS accepts explicit paths and zones for additional comparisons.
Run separately from the curated presentation exporter with
``python3 -m tools.POLIMO_NACA0012_PLOTS``.
"""
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.graph_objects as go

from .gatherParticipantData import (
    ZoneData, add_curvilinear_distance_to_zone, read_tecplot_dat,
    normalize_participant_id,
)
from .plot_style import apply_xy_style

BETA_COMPARISONS = [
    dict(
        case_id="TC_NACA0012_AE3932",
        title="POLIMO NACA0012 AE3932 · L1 · BINS07",
        lagrangian="TC_NACA0012_AE3932_D01/TC_NACA0012_3932_L1_LAG_BETA_V1.dat",
        lagrangian_zone="SLICE_Y_0p9144_BINS07_KS_0p5334mm_CUTDATA",
        eulerian="TC_NACA0012_AE3932_D01/TC_NACA0012_3932_L1_cutData_V1.dat",
        eulerian_zone="SLICE_Y_0p9144_BINS07_KS_0p5334mm_CUTDATA",
        filename="tc_naca0012_ae3932_L1_beta_bins07_lagrangian_vs_eulerian.png",
    ),
]


def comparison_figure(root: Path, spec: dict) -> go.Figure:
    """Build a comparison using the site's signed surface-distance mapping."""
    lag_path = root / spec["lagrangian"]
    # The supplemental format has only a ZONE header and X Y Z Beta rows.
    rows = []
    selected = False
    for line in lag_path.read_text().splitlines():
        if line.lstrip().upper().startswith('ZONE'):
            selected = spec["lagrangian_zone"] == line.split('"')[1]
            continue
        if not selected or not line.strip() or line.lstrip().startswith('#'):
            continue
        values = [float(value) for value in line.split()]
        if len(values) != 4 or not np.isfinite(values).all():
            raise ValueError(f"Expected finite X Y Z Beta values in {lag_path}")
        rows.append(values)
    if not rows:
        raise ValueError(f"No Lagrangian data in {lag_path}")
    lag = ZoneData("Lagrangian", pd.DataFrame(rows, columns=["X", "Y", "Z", "Beta"]))
    eul_path = root / spec["eulerian"]
    eul = read_tecplot_dat(eul_path, process_cutdata=False).zones[spec["eulerian_zone"]]
    fig = go.Figure()
    for zone, path, label, color, dash in (
        (lag, lag_path, "Lagrangian", "#d62728", "solid"),
        (eul, eul_path, "Eulerian", "#1f77b4", "solid"),
    ):
        add_curvilinear_distance_to_zone(path, zone, case_id=spec["case_id"])
        data = zone.data.sort_values("s")
        data = data.loc[np.isfinite(data["Beta"]) & (data["Beta"] >= 0)]
        if data.empty:
            raise ValueError(f"No valid beta values in {path}")
        fig.add_trace(go.Scatter(x=data.s, y=data.Beta, name=label, mode="lines",
                                 line=dict(color=color, dash=dash, width=3)))
    apply_xy_style(fig, spec["case_id"], "s [m]", "β [-]",
                   plot_family="cutdata", plot_key="beta", x_range=[-0.1, 0.1])
    fig.update_layout(title=dict(text=spec["title"], y=0.97, yanchor="top"),
                      margin=dict(t=85), width=1350, height=1000)
    fig.update_xaxes(title_text="s [m]")
    return fig


def export_comparisons(participants, output_dir, comparisons, participant_id=None):
    if participant_id is not None and normalize_participant_id(participant_id) != "007":
        return 0
    polimo = next((p for p in participants if normalize_participant_id(p.participant_id) == "007"), None)
    if polimo is None:
        return 0
    for spec in comparisons:
        figure = comparison_figure(polimo.path, spec)
        destination = Path(output_dir) / "IMPINGEMENT" / spec["filename"]
        destination.parent.mkdir(parents=True, exist_ok=True)
        figure.write_image(destination, scale=1)
    return len(comparisons)


def generate(participants, output_dir, participant_id=None, highlight=False):
    """Export method comparisons in either participant display mode."""
    return export_comparisons(participants, output_dir, BETA_COMPARISONS, participant_id)


def main():
    """Export supplemental POLIMO NACA plots without building a presentation."""
    import argparse
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=Path("FIGURES_POLIMO_NACA0012"))
    args = parser.parse_args()
    from types import SimpleNamespace
    root = Path(__file__).resolve().parents[1] / "007_POLIMO_CHAMPS"
    count = generate([SimpleNamespace(participant_id="007", path=root)], args.output)
    print(f"Wrote {count} plot(s) to {args.output / 'IMPINGEMENT'}")


if __name__ == "__main__":
    main()
