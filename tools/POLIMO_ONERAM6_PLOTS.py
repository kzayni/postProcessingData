"""Additional POLIMO ONERA M6 Lagrangian/Eulerian beta comparisons."""
from pathlib import Path
from .POLIMO_NACA0012_PLOTS import comparison_figure, export_comparisons

BETA_COMPARISONS = [
    dict(
        case_id="TC_ONERAM6",
        title=f"POLIMO ONERA M6 · L1 · slice {slice_label} m",
        lagrangian="TC_ONERAM6_D01/TC_ONERAM6_L1_LAG_BETA_V1.dat",
        lagrangian_zone=f"SLICE_Y_{slice_slug}_BINS15_KS_1mm_CUTDATA",
        eulerian="TC_ONERAM6_D01/TC_ONERAM6_L1_cutData_V1.dat",
        eulerian_zone=f"SLICE_Y_{slice_slug}_BINS01_KS_1mm_CUTDATA",
        filename=(
            f"tc_oneram6_L1_beta_bins01_vs_s_slice_{slice_slug}_roughness_1mm.png"
        ),
    )
    for slice_slug, slice_label in (("0p1", "0.1"), ("0p75", "0.75"), ("1p4", "1.4"))
]


def generate(participants, output_dir, participant_id=None, highlight=False):
    return export_comparisons(participants, output_dir, BETA_COMPARISONS, participant_id)


def main():
    import argparse
    from types import SimpleNamespace
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=Path("FIGURES_POLIMO_ONERAM6"))
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[1] / "007_POLIMO_CHAMPS"
    count = generate([SimpleNamespace(participant_id="007", path=root)], args.output)
    print(f"Wrote {count} plot(s) to {args.output / 'IMPINGEMENT'}")


if __name__ == "__main__":
    main()
