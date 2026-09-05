"""Generate presentation-ready ONERA M6 images for all configured slices."""

from __future__ import annotations

import math
from pathlib import Path
import re
import shutil
import tempfile

from tools import convergence_data_builder, cutdata_builder, iceshape_builder
from tools.gatherParticipantData import CASE_SLICES, VALID_GRID_LEVELS


CASE_ID = "TC_ONERAM6"
OUTPUT_DIR = Path("FIGURES_ONERAM6")
WIDTH = 1350
HEIGHT = 700

FigureSpec = tuple[str, str, int, int, list[str], bool]


def _spec(category: str, filename: str, excluded: list[str] | None = None, show_legend: bool = True) -> tuple[str, FigureSpec]:
    return f"{category}/{filename}", (CASE_ID, filename, WIDTH, HEIGHT, excluded or [], show_legend)


# Curated ONERA M6 presentation list. Add/remove entries here exactly as in
# NACA0012_PRESENTATION_FIGURES. Slice-specific entries are expanded for every
# configured ONERA station: Y = 0.1, 0.75, and 1.4 m.
ONERAM6_PRESENTATION_FIGURES: dict[str, FigureSpec] = dict([
    _spec("AERODYNAMIC", "tc_oneram6_cl_vs_n_1mm.png"),
    _spec("AERODYNAMIC", "tc_oneram6_cd_vs_n_1mm.png"),
    _spec("AERODYNAMIC", "tc_oneram6_cmy_vs_n_1mm.png"),
    _spec("IMPINGEMENT", "tc_oneram6_water_mass_vs_n_1mm_bins15_all_grid_levels.png"),
    _spec("ICE_ACCRETION", "tc_oneram6_ice_mass_vs_n_1mm_bins15_all_grid_levels.png"),
    _spec("ICE_ACCRETION", "tc_oneram6_ice_to_water_ratio_vs_n_required_bins15.png"),
    _spec("ICE_ACCRETION", "tc_oneram6_ice_evap_to_water_ratio_vs_n_required_bins15.png"),
])

for _slice_value in CASE_SLICES[CASE_ID]:
    _slice_slug = str(_slice_value).replace(".", "p")
    _slice_decimal = str(_slice_value)
    ONERAM6_PRESENTATION_FIGURES.update(dict([
        _spec("AERODYNAMIC", f"tc_oneram6_L1_cp_vs_s_slice_{_slice_slug}_roughness_1mm.png"),
        _spec("AERODYNAMIC", f"tc_oneram6_L1_cp_vs_x_slice_{_slice_slug}_roughness_1mm.png"),
        _spec("HTC", f"tc_oneram6_L1_htc_vs_s_slice_{_slice_slug}_roughness_1mm.png"),
        _spec("HTC", f"tc_oneram6_L1_recovery_temperature_vs_s_slice_{_slice_slug}_roughness_1mm.png"),
        _spec("HTC", f"tc_oneram6_qc_prime_vs_n_y_{_slice_decimal}.png"),
        _spec("IMPINGEMENT", f"tc_oneram6_L1_beta_bins15_vs_s_slice_{_slice_slug}_roughness_1mm.png"),
        _spec("SURF_TEMP_FF", f"tc_oneram6_L1_surface_temperature_vs_s_slice_{_slice_slug}_roughness_1mm.png"),
        _spec("SURF_TEMP_FF", f"tc_oneram6_L1_freezing_fraction_vs_s_slice_{_slice_slug}_roughness_1mm.png"),
        _spec("ICE_SHAPES", f"tc_oneram6_L1_single_layer_ice_shape_slice_{_slice_slug}_bins15_roughness_1mm.png"),
        _spec("ICE_SHAPES", f"tc_oneram6_L1_multilayer_ice_shape_slice_{_slice_slug}_bins15_roughness_1mm.png"),
    ]))


def _style_figure(figure, module, export_path: Path, spec: FigureSpec) -> None:
    _, _, width, height, excluded_ids, show_legend = spec
    name = export_path.stem.lower()
    is_convergence = module is convergence_data_builder and "_vs_n" in name
    is_bin_convergence = "_vs_inverse_bins" in name or "_distribution_convergence_" in name

    excluded = {str(value).zfill(3) for value in excluded_ids}
    if excluded:
        figure.data = tuple(
            trace for trace in figure.data
            if not (
                isinstance(trace.meta, dict)
                and str(trace.meta.get("ipw3_participant_id", "")).zfill(3) in excluded
            )
            and not (
                (match := re.match(r"^(\d{1,3})(?=\D|$)", str(trace.name or "").strip()))
                and match.group(1).zfill(3) in excluded
            )
        )

    for trace in figure.data:
        if getattr(trace, "line", None) is not None:
            trace.line.width = 5
        if getattr(trace, "marker", None) is not None:
            trace.marker.size = 14

    figure.update_layout(
        title=None,
        width=width,
        height=height,
        showlegend=show_legend,
        font={"family": "Arial, Helvetica, sans-serif", "size": 32},
        legend={
            "orientation": "h", "x": 0.0, "xanchor": "left",
            "y": 1.02, "yanchor": "bottom", "font": {"size": 24},
        },
        margin={"l": 100, "r": 50, "t": 125 if show_legend else 30, "b": 85},
        paper_bgcolor="white",
        plot_bgcolor="white",
    )
    figure.update_xaxes(title_font={"size": 36}, tickfont={"size": 28})
    figure.update_yaxes(
        title_font={"size": 36}, tickfont={"size": 28},
        title_standoff=50, automargin=True,
    )

    if is_convergence and not is_bin_convergence:
        cell_counts = convergence_data_builder.grid_cell_counts_for_case(CASE_ID)
        tick_values = [cell_counts[level] ** (-1.0 / 3.0) for level in sorted(cell_counts)]
        for trace in figure.data:
            if trace.x is None:
                continue
            converted = []
            for value in trace.x:
                match = re.fullmatch(r"L(\d+)", str(value).strip(), re.IGNORECASE)
                level = int(match.group(1)) if match else None
                converted.append(cell_counts[level] ** (-1.0 / 3.0) if level in cell_counts else value)
            trace.x = converted
        if tick_values:
            log_values = [math.log10(value) for value in tick_values]
            padding = max((max(log_values) - min(log_values)) * 0.08, 0.02)
            figure.update_xaxes(
                type="log", autorange=False,
                range=[min(log_values) - padding, max(log_values) + padding],
                tickmode="array", tickvals=tick_values,
                ticktext=[f"{value * 1.0e3:.3f}" for value in tick_values],
                ticks="outside", showticklabels=True, showgrid=True,
                title_text="N<sup>−1/3</sup> [×10<sup>−3</sup>]",
            )
            figure.update_layout(margin={"l": 100, "r": 50, "t": 125, "b": 125})


def generate(participants, output_dir: Path = OUTPUT_DIR) -> int:
    """Export ONERA M6 figures, including Y=0.1, 0.75, and 1.4 m cuts."""
    import build_site_ipw3 as site

    configured_slices = tuple(CASE_SLICES[CASE_ID])
    if output_dir.exists():
        shutil.rmtree(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    convergence_data_builder.clear_png_export_queue()
    cutdata_builder.clear_png_export_queue()
    iceshape_builder.clear_png_export_queue()

    with tempfile.TemporaryDirectory(prefix="ipw3_oneram6_pres_") as staging_text:
        case_dir = Path(staging_text) / CASE_ID
        convergence_data_builder.set_png_export_dir(case_dir, include_relative=True)
        cutdata_builder.set_png_export_dir(case_dir)
        iceshape_builder.set_png_export_dir(case_dir)

        site.build_grid_convergence_section(participants, CASE_ID, category="cfd")
        site.build_grid_convergence_section(participants, CASE_ID, category="icing", requirement="required")
        site.build_water_mass_analysis_section(participants, CASE_ID)
        site.build_beta_max_analysis_section(participants, CASE_ID)
        convergence_data_builder.build_upper_horn_angle_convergence_section(participants, CASE_ID)
        for grid_level in sorted(VALID_GRID_LEVELS):
            site.build_grid_page_content(participants, CASE_ID, grid_level)

        queues = (
            (convergence_data_builder, convergence_data_builder.PNG_EXPORT_QUEUE),
            (cutdata_builder, cutdata_builder.PNG_EXPORT_QUEUE),
            (iceshape_builder, iceshape_builder.PNG_EXPORT_QUEUE),
        )
        requested_names = {spec[1] for spec in ONERAM6_PRESENTATION_FIGURES.values()}
        for _, queue in queues:
            queue[:] = [(figure, path) for figure, path in queue if path.name in requested_names]
        queued_names = [path.name for _, queue in queues for _, path in queue]
        missing_slices = [
            value for value in configured_slices
            if not any(f"slice_{str(value).replace('.', 'p')}" in name for name in queued_names)
        ]
        if missing_slices:
            raise RuntimeError(f"Missing ONERA M6 presentation slices: {missing_slices}")

        specs_by_name = {spec[1]: spec for spec in ONERAM6_PRESENTATION_FIGURES.values()}
        for module, queue in queues:
            for figure, export_path in queue:
                _style_figure(figure, module, export_path, specs_by_name[export_path.name])

        convergence_data_builder.flush_png_exports(scale=1, width=None, height=None)
        cutdata_builder.flush_png_exports(scale=1, width=None, height=None)
        iceshape_builder.flush_png_exports(scale=1, width=None, height=None)

        missing: list[str] = []
        for destination_text, (_, source_name, _, _, _, _) in ONERAM6_PRESENTATION_FIGURES.items():
            source = case_dir / source_name
            destination = output_dir / destination_text
            if not source.exists():
                missing.append(f"{destination_text} <- {source_name}")
                continue
            destination.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(source, destination)
        if missing:
            raise RuntimeError("Missing ONERA M6 presentation plot exports:\n  " + "\n  ".join(missing))
    return len(ONERAM6_PRESENTATION_FIGURES)


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
    print(f"Wrote {figure_count} ONERA M6 presentation figures in {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
