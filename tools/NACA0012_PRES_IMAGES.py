"""Curated NACA0012 presentation image configuration and exporter."""

from __future__ import annotations

from pathlib import Path
import re
import shutil
import tempfile

from . import convergence_data_builder, cutdata_builder, iceshape_builder
from .plot_style import NACA0012_ROUGHNESS_GROUP_STYLE

OUTPUT_DIR = Path("FIGURES_NACA0012")

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
# - The final boolean controls legend visibility and defaults to True. Set it
#   to False for a legend-free presentation figure.
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
NACA0012_PRESENTATION_FIGURES: dict[str, tuple[str, str, int, int, list[str], bool]] = {
    # AERODYNAMIC
    # Surface pressure
    "AERODYNAMIC/tc_naca0012_ae3932_L1_cp_vs_s_slice_0p9144_all_roughness.png": ("TC_NACA0012_AE3932", "tc_naca0012_ae3932_L1_cp_vs_s_slice_0p9144_all_roughness.png", 1350, 700, [], True),

    # Grid sensitivity — absolute values
    "AERODYNAMIC/tc_naca0012_ae3932_cd_vs_n_all_roughness.png": ("TC_NACA0012_AE3932", "tc_naca0012_ae3932_cd_vs_n_all_roughness.png", 1350, 700, [], True),
    "AERODYNAMIC/tc_naca0012_ae3932_cl_vs_n_all_roughness.png": ("TC_NACA0012_AE3932", "tc_naca0012_ae3932_cl_vs_n_all_roughness.png", 1350, 700, [], True),
    "AERODYNAMIC/tc_naca0012_ae3932_cmy_vs_n_all_roughness.png": ("TC_NACA0012_AE3932", "tc_naca0012_ae3932_cmy_vs_n_all_roughness.png", 1350, 700, [], True),

    # Grid sensitivity — difference from L1
    "AERODYNAMIC/tc_naca0012_ae3932_cd_vs_n_all_roughness_relative_to_l1.png": ("TC_NACA0012_AE3932", "tc_naca0012_ae3932_cd_vs_n_all_roughness_relative_to_l1.png", 1350, 700, [], True),
    "AERODYNAMIC/tc_naca0012_ae3932_cl_vs_n_all_roughness_relative_to_l1.png": ("TC_NACA0012_AE3932", "tc_naca0012_ae3932_cl_vs_n_all_roughness_relative_to_l1.png", 1350, 700, [], True),

    # HTC - 3932
    # Surface distributions
    "HTC/tc_naca0012_ae3932_L1_htc_vs_s_slice_0p9144_all_roughness.png": ("TC_NACA0012_AE3932", "tc_naca0012_ae3932_L1_htc_vs_s_slice_0p9144_all_roughness.png", 1350, 700, [], True),
    "HTC/tc_naca0012_ae3932_L1_htc_vs_s_slice_0p9144_grouped_roughness.png": ("TC_NACA0012_AE3932", "tc_naca0012_ae3932_L1_htc_vs_s_slice_0p9144_grouped_roughness.png", 1350, 700, [], True),
    "HTC/tc_naca0012_ae3932_L1_recovery_temperature_vs_s_slice_0p9144_all_roughness.png": ("TC_NACA0012_AE3932", "tc_naca0012_ae3932_L1_recovery_temperature_vs_s_slice_0p9144_all_roughness.png", 1350, 700, [], True),

    # Integrated heat-transfer grid sensitivity - 3932
    "HTC/tc_naca0012_ae3932_qc_prime_vs_n_y_0.9144.png": ("TC_NACA0012_AE3932", "tc_naca0012_ae3932_qc_prime_vs_n_y_0.9144.png", 1350, 700, ["001","006"], True),
    "HTC/tc_naca0012_ae3932_qc_prime_vs_n_y_0.9144_grouped_roughness.png": ("TC_NACA0012_AE3932", "tc_naca0012_ae3932_qc_prime_vs_n_y_0.9144_grouped_roughness.png", 1350, 700, ["001","006"], True),

    # Integrated heat-transfer grid sensitivity - 3933
    "HTC/tc_naca0012_ae3933_L1_htc_vs_s_slice_0p9144_all_roughness.png": ("TC_NACA0012_AE3933", "tc_naca0012_ae3933_L1_htc_vs_s_slice_0p9144_all_roughness.png", 1350, 700, [], True),
    "HTC/tc_naca0012_ae3933_L1_htc_vs_s_slice_0p9144_grouped_roughness.png": ("TC_NACA0012_AE3933", "tc_naca0012_ae3933_L1_htc_vs_s_slice_0p9144_grouped_roughness.png", 1350, 700, [], True),
    "HTC/tc_naca0012_ae3933_qc_prime_vs_n_y_0.9144.png": ("TC_NACA0012_AE3933", "tc_naca0012_ae3933_qc_prime_vs_n_y_0.9144.png", 1350, 700, ["001","006"], True),
    "HTC/tc_naca0012_ae3933_qc_prime_vs_n_y_0.9144_grouped_roughness.png": ("TC_NACA0012_AE3933", "tc_naca0012_ae3933_qc_prime_vs_n_y_0.9144_grouped_roughness.png", 1350, 700, ["001","006"], True),

    # ICE ACCRETION — AE3932
    # Ice shape
    "ICE_SHAPES/tc_naca0012_ae3932_L1_multilayer_ice_shape_001_slice_0p9144_bins01.png": ("TC_NACA0012_AE3932", "tc_naca0012_ae3932_L1_multilayer_ice_shape_001_slice_0p9144_bins01.png", 1350, 700, [], True),
    "ICE_SHAPES/tc_naca0012_ae3932_L2_single_layer_ice_shape_slice_0p9144_bins15_roughness_unspecified.png": ("TC_NACA0012_AE3932", "tc_naca0012_ae3932_L2_single_layer_ice_shape_slice_0p9144_bins15_roughness_unspecified.png", 1350, 700, [], True),
    "ICE_SHAPES/tc_naca0012_ae3932_L2_single_layer_ice_shape_slice_0p9144_bins15_grouped_roughness.png": ("TC_NACA0012_AE3932", "tc_naca0012_ae3932_L2_single_layer_ice_shape_slice_0p9144_bins15_grouped_roughness.png", 1350, 700, [], True),

    # Water-fate ratios
    "ICE_ACCRETION/tc_naca0012_ae3932_ice_evap_to_water_ratio_vs_n_required_bins15.png": ("TC_NACA0012_AE3932", "tc_naca0012_ae3932_ice_evap_to_water_ratio_vs_n_required_bins15.png", 1350, 700, [], True),

    # Ice-mass grid and droplet-bin sensitivity
    "ICE_ACCRETION/tc_naca0012_ae3932_ice_mass_vs_n_unspecified_bins15_all_grid_levels.png": ("TC_NACA0012_AE3932", "tc_naca0012_ae3932_ice_mass_vs_n_unspecified_bins15_all_grid_levels.png", 1350, 700, [], True),
    "ICE_ACCRETION/tc_naca0012_ae3932_ice_mass_vs_n_unspecified_l1_vs_inverse_bins.png": ("TC_NACA0012_AE3932", "tc_naca0012_ae3932_ice_mass_vs_n_unspecified_l1_vs_inverse_bins.png", 1350, 700, [], True),
    "ICE_ACCRETION/tc_naca0012_ae3932_ice_to_water_ratio_vs_n_required_bins15.png": ("TC_NACA0012_AE3932", "tc_naca0012_ae3932_ice_to_water_ratio_vs_n_required_bins15.png", 1350, 700, [], True),

    # Upper-horn grid and droplet-bin sensitivity
    "ICE_ACCRETION/tc_naca0012_ae3932_upper_horn_angle_bins15.png": ("TC_NACA0012_AE3932", "tc_naca0012_ae3932_upper_horn_angle_bins15.png", 1350, 700, [], True),
    "ICE_ACCRETION/tc_naca0012_ae3932_upper_horn_angle_distribution_convergence_l1.png": ("TC_NACA0012_AE3932", "tc_naca0012_ae3932_upper_horn_angle_distribution_convergence_l1.png", 1350, 700, [], True),

    # ICE ACCRETION — AE3933
    "ICE_SHAPES/tc_naca0012_ae3933_L2_single_layer_ice_shape_slice_0p9144_bins15_grouped_roughness.png": ("TC_NACA0012_AE3933", "tc_naca0012_ae3933_L2_single_layer_ice_shape_slice_0p9144_bins15_grouped_roughness.png", 1350, 700, [], True),

    # Ice shapes
    "ICE_SHAPES/tc_naca0012_ae3933_L2_ice_shape_001_bins01_008_bins07.png": ("TC_NACA0012_AE3933", "tc_naca0012_ae3933_L2_ice_shape_001_bins01_008_bins07.png", 1350, 700, [], True),
    "ICE_SHAPES/tc_naca0012_ae3933_L1_single_layer_ice_shape_slice_0p9144_bins15_roughness_unspecified.png": ("TC_NACA0012_AE3933", "tc_naca0012_ae3933_L1_single_layer_ice_shape_slice_0p9144_bins15_roughness_unspecified.png", 1350, 700, [], True),

    # Comparison with AE3932
    "ICE_ACCRETION/tc_naca0012_ae3933_comparison_with_3932_ice_mass_bins15_difference.png": ("TC_NACA0012_AE3933", "tc_naca0012_ae3933_comparison_with_3932_ice_mass_bins15_difference.png", 1350, 700, [], True),

    # Water-fate ratios
    "ICE_ACCRETION/tc_naca0012_ae3933_ice_evap_to_water_ratio_vs_n_required_bins15.png": ("TC_NACA0012_AE3933", "tc_naca0012_ae3933_ice_evap_to_water_ratio_vs_n_required_bins15.png", 1350, 700, [], True),

    # Ice-mass grid and droplet-bin sensitivity
    "ICE_ACCRETION/tc_naca0012_ae3933_ice_mass_vs_n_unspecified_bins15_all_grid_levels.png": ("TC_NACA0012_AE3933", "tc_naca0012_ae3933_ice_mass_vs_n_unspecified_bins15_all_grid_levels.png", 1350, 700, [], True),
    "ICE_ACCRETION/tc_naca0012_ae3933_ice_to_water_ratio_vs_n_required_bins15.png": ("TC_NACA0012_AE3933", "tc_naca0012_ae3933_ice_to_water_ratio_vs_n_required_bins15.png", 1350, 700, ["001"], True),

    # Upper-horn grid and droplet-bin sensitivity
    "ICE_ACCRETION/tc_naca0012_ae3933_upper_horn_angle_bins15.png": ("TC_NACA0012_AE3933", "tc_naca0012_ae3933_upper_horn_angle_bins15.png", 1350, 700, [], True),
    "ICE_ACCRETION/tc_naca0012_ae3933_upper_horn_angle_distribution_convergence_l1.png": ("TC_NACA0012_AE3933", "tc_naca0012_ae3933_upper_horn_angle_distribution_convergence_l1.png", 1350, 700, [], True),
    "ICE_ACCRETION/tc_naca0012_ae3933_ice_mass_vs_n_unspecified_l1_vs_inverse_bins.png": ("TC_NACA0012_AE3933", "tc_naca0012_ae3933_ice_mass_vs_n_unspecified_l1_vs_inverse_bins.png", 1350, 700, [], True),

    # IMPINGEMENT
    # Collection-efficiency distribution
    "IMPINGEMENT/tc_naca0012_ae3932_L1_beta_bins15_vs_s_slice_0p9144_all_roughness.png": ("TC_NACA0012_AE3932", "tc_naca0012_ae3932_L1_beta_bins15_vs_s_slice_0p9144_all_roughness.png", 1350, 700, [], True),

    # Peak collection efficiency — absolute, grid difference, and bin difference
    "IMPINGEMENT/tc_naca0012_ae3932_beta_max_bins15.png": ("TC_NACA0012_AE3932", "tc_naca0012_ae3932_beta_max_bins15.png", 1350, 700, [], True),
    "IMPINGEMENT/tc_naca0012_ae3932_beta_max_bins15_relative_to_l1.png": ("TC_NACA0012_AE3932", "tc_naca0012_ae3932_beta_max_bins15_relative_to_l1.png", 1350, 700, [], True),
    "IMPINGEMENT/tc_naca0012_ae3932_beta_max_distribution_convergence_l1_relative_to_bins15.png": ("TC_NACA0012_AE3932", "tc_naca0012_ae3932_beta_max_distribution_convergence_l1_relative_to_bins15.png", 1350, 700, [], True),

    # Peak position — absolute, grid difference, and bin difference
    "IMPINGEMENT/tc_naca0012_ae3932_s_peak_bins15.png": ("TC_NACA0012_AE3932", "tc_naca0012_ae3932_s_peak_bins15.png", 1350, 700, [], True),
    "IMPINGEMENT/tc_naca0012_ae3932_s_peak_bins15_relative_to_l1.png": ("TC_NACA0012_AE3932", "tc_naca0012_ae3932_s_peak_bins15_relative_to_l1.png", 1350, 700, [], True),
    "IMPINGEMENT/tc_naca0012_ae3932_s_peak_distribution_convergence_l1_relative_to_bins15.png": ("TC_NACA0012_AE3932", "tc_naca0012_ae3932_s_peak_distribution_convergence_l1_relative_to_bins15.png", 1350, 700, [], True),

    # Impingement width — absolute, grid difference, and bin difference
    "IMPINGEMENT/tc_naca0012_ae3932_width_bins15.png": ("TC_NACA0012_AE3932", "tc_naca0012_ae3932_width_bins15.png", 1350, 700, ["003", "006"], True),
    "IMPINGEMENT/tc_naca0012_ae3932_width_bins15_relative_to_l1.png": ("TC_NACA0012_AE3932", "tc_naca0012_ae3932_width_bins15_relative_to_l1.png", 1350, 700, ["006"], True),
    "IMPINGEMENT/tc_naca0012_ae3932_width_distribution_convergence_l1_relative_to_bins15.png": ("TC_NACA0012_AE3932", "tc_naca0012_ae3932_width_distribution_convergence_l1_relative_to_bins15.png", 1350, 700, [], True),

    # Total water mass — grid and droplet-bin sensitivity
    "IMPINGEMENT/tc_naca0012_ae3932_water_mass_vs_n_unspecified_bins15_all_grid_levels.png": ("TC_NACA0012_AE3932", "tc_naca0012_ae3932_water_mass_vs_n_unspecified_bins15_all_grid_levels.png", 1350, 700, [], True),
    "IMPINGEMENT/tc_naca0012_ae3932_water_mass_vs_n_unspecified_bins15_all_grid_levels_relative_to_l1.png": ("TC_NACA0012_AE3932", "tc_naca0012_ae3932_water_mass_vs_n_unspecified_bins15_all_grid_levels_relative_to_l1.png", 1350, 700, ["001"], True),
    "IMPINGEMENT/tc_naca0012_ae3932_water_mass_vs_n_unspecified_l1_vs_inverse_bins.png": ("TC_NACA0012_AE3932", "tc_naca0012_ae3932_water_mass_vs_n_unspecified_l1_vs_inverse_bins.png", 1350, 700, [], True),

    # Diameter-resolved across-participant variation
    "IMPINGEMENT/tc_naca0012_ae3932_water_mass_relative_iqr_vs_droplet_diameter_bins15_all_grid_levels.png": ("TC_NACA0012_AE3932", "tc_naca0012_ae3932_water_mass_relative_iqr_vs_droplet_diameter_bins15_all_grid_levels.png", 1350, 700, [], True),
    "IMPINGEMENT/tc_naca0012_ae3932_water_mass_coefficient_of_variation_vs_droplet_diameter_bins15_all_grid_levels.png": ("TC_NACA0012_AE3932", "tc_naca0012_ae3932_water_mass_coefficient_of_variation_vs_droplet_diameter_bins15_all_grid_levels.png", 1350, 700, [], True),
    "IMPINGEMENT/tc_naca0012_ae3933_water_mass_relative_iqr_vs_droplet_diameter_bins15_all_grid_levels.png": ("TC_NACA0012_AE3933", "tc_naca0012_ae3933_water_mass_relative_iqr_vs_droplet_diameter_bins15_all_grid_levels.png", 1350, 700, [], True),
    "IMPINGEMENT/tc_naca0012_ae3933_water_mass_coefficient_of_variation_vs_droplet_diameter_bins15_all_grid_levels.png": ("TC_NACA0012_AE3933", "tc_naca0012_ae3933_water_mass_coefficient_of_variation_vs_droplet_diameter_bins15_all_grid_levels.png", 1350, 700, [], True),

    # SURFACE TEMPERATURE / FREEZING FRACTION
    # AE3932 surface distributions
    "SURF_TEMP_FF/tc_naca0012_ae3932_L1_freezing_fraction_vs_s_slice_0p9144_all_roughness.png": ("TC_NACA0012_AE3932", "tc_naca0012_ae3932_L1_freezing_fraction_vs_s_slice_0p9144_all_roughness.png", 1350, 700, [], True),
    "SURF_TEMP_FF/tc_naca0012_ae3932_L1_surface_temperature_vs_s_slice_0p9144_all_roughness.png": ("TC_NACA0012_AE3932", "tc_naca0012_ae3932_L1_surface_temperature_vs_s_slice_0p9144_all_roughness.png", 1350, 700, [], True),
    "SURF_TEMP_FF/tc_naca0012_ae3932_mean_surface_temperature_vs_n_slice_0p9144_all_roughness.png": ("TC_NACA0012_AE3932", "tc_naca0012_ae3932_mean_surface_temperature_vs_n_slice_0p9144_all_roughness.png", 1350, 700, [], True),
    "SURF_TEMP_FF/tc_naca0012_ae3932_mean_freezing_fraction_vs_n_slice_0p9144_all_roughness.png": ("TC_NACA0012_AE3932", "tc_naca0012_ae3932_mean_freezing_fraction_vs_n_slice_0p9144_all_roughness.png", 1350, 700, [], True),

    # AE3933 surface distributions
    "SURF_TEMP_FF/tc_naca0012_ae3933_L1_freezing_fraction_vs_s_slice_0p9144_all_roughness.png": ("TC_NACA0012_AE3933", "tc_naca0012_ae3933_L1_freezing_fraction_vs_s_slice_0p9144_all_roughness.png", 1350, 700, [], True),
    "SURF_TEMP_FF/tc_naca0012_ae3933_L1_surface_temperature_vs_s_slice_0p9144_all_roughness.png": ("TC_NACA0012_AE3933", "tc_naca0012_ae3933_L1_surface_temperature_vs_s_slice_0p9144_all_roughness.png", 1350, 700, [], True),
    "SURF_TEMP_FF/tc_naca0012_ae3933_mean_surface_temperature_vs_n_slice_0p9144_all_roughness.png": ("TC_NACA0012_AE3933", "tc_naca0012_ae3933_mean_surface_temperature_vs_n_slice_0p9144_all_roughness.png", 1350, 700, [], True),
    "SURF_TEMP_FF/tc_naca0012_ae3933_mean_freezing_fraction_vs_n_slice_0p9144_all_roughness.png": ("TC_NACA0012_AE3933", "tc_naca0012_ae3933_mean_freezing_fraction_vs_n_slice_0p9144_all_roughness.png", 1350, 700, [], True),

    # Ice Horns
    # AE3932 horn-construction methods
    "ICE_HORNS/tc_naca0012_ae3932_upper_horn_angle_by_participant_bins15_all_grid_levels.png": ("TC_NACA0012_AE3932", "tc_naca0012_ae3932_upper_horn_angle_by_participant_bins15_all_grid_levels.png", 1350, 700, [], True),
    "ICE_HORNS/tc_naca0012_ae3932_upper_horn_angle_by_participant_bins15_all_grid_levels_grouped_roughness.png": ("TC_NACA0012_AE3932", "tc_naca0012_ae3932_upper_horn_angle_by_participant_bins15_all_grid_levels_grouped_roughness.png", 1350, 700, [], True),
    "ICE_HORNS/tc_naca0012_ae3932_maxccs_upper_horn_angle_method.png": ("TC_NACA0012_AE3932", "tc_naca0012_ae3932_maxccs_upper_horn_angle_method.png", 1350, 700, [], False),
    "ICE_HORNS/tc_naca0012_ae3932_meanccs_upper_horn_angle_method.png": ("TC_NACA0012_AE3932", "tc_naca0012_ae3932_meanccs_upper_horn_angle_method.png", 1350, 700, [], False),
    "ICE_HORNS/tc_naca0012_ae3932_minccs_upper_horn_angle_method.png": ("TC_NACA0012_AE3932", "tc_naca0012_ae3932_minccs_upper_horn_angle_method.png", 1350, 700, [], False),

    # AE3933 horn-construction methods
    "ICE_HORNS/tc_naca0012_ae3933_upper_horn_angle_by_participant_bins15_all_grid_levels.png": ("TC_NACA0012_AE3933", "tc_naca0012_ae3933_upper_horn_angle_by_participant_bins15_all_grid_levels.png", 1350, 700, [], True),
    "ICE_HORNS/tc_naca0012_ae3933_upper_horn_angle_by_participant_bins15_all_grid_levels_grouped_roughness.png": ("TC_NACA0012_AE3933", "tc_naca0012_ae3933_upper_horn_angle_by_participant_bins15_all_grid_levels_grouped_roughness.png", 1350, 700, [], True),
    "ICE_HORNS/tc_naca0012_ae3933_maxccs_upper_horn_angle_method.png": ("TC_NACA0012_AE3933", "tc_naca0012_ae3933_maxccs_upper_horn_angle_method.png", 1350, 700, [], False),
    "ICE_HORNS/tc_naca0012_ae3933_meanccs_upper_horn_angle_method.png": ("TC_NACA0012_AE3933", "tc_naca0012_ae3933_meanccs_upper_horn_angle_method.png", 1350, 700, [], False),
    "ICE_HORNS/tc_naca0012_ae3933_minccs_upper_horn_angle_method.png": ("TC_NACA0012_AE3933", "tc_naca0012_ae3933_minccs_upper_horn_angle_method.png", 1350, 700, [], False),
}


def write_naca0012_presentation_figures(participants, output_dir: Path) -> None:
    """Generate and route the curated NACA0012 plot set for the presentation."""
    import build_site_ipw3 as site

    build_grid_convergence_section = site.build_grid_convergence_section
    build_water_mass_analysis_section = site.build_water_mass_analysis_section
    build_beta_max_analysis_section = site.build_beta_max_analysis_section
    build_grid_page_content = site.build_grid_page_content
    build_ae3933_ice_mass_comparison_section = site.build_ae3933_ice_mass_comparison_section
    case_ids = ("TC_NACA0012_AE3932", "TC_NACA0012_AE3933")
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
            build_water_mass_analysis_section(participants, case_id)
            build_beta_max_analysis_section(participants, case_id)
            convergence_data_builder.build_water_mass_diameter_dispersion_figures(participants, case_id)
            convergence_data_builder.build_upper_horn_angle_convergence_section(participants, case_id)
            convergence_data_builder.build_upper_horn_angle_participant_summary(participants, case_id)
            convergence_data_builder.build_upper_horn_angle_participant_summary(participants, case_id, group_by_roughness=True)
            requested_levels = {"L1"}
            for figure_case, source_name, *_ in NACA0012_PRESENTATION_FIGURES.values():
                if figure_case != case_id:
                    continue
                level_match = re.search(r"_(L[1-4])_", source_name)
                if level_match:
                    requested_levels.add(level_match.group(1))
            for grid_level in sorted(requested_levels):
                build_grid_page_content(participants, case_id, grid_level)

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

        # Create a presentation-only HTC copy whose visual identity is the
        # roughness treatment rather than the participant. The original HTC
        # export remains in the queue unchanged.
        for htc_case_id in case_ids:
            case_slug = htc_case_id.lower()
            htc_source_name = f"{case_slug}_L1_htc_vs_s_slice_0p9144_all_roughness.png"
            htc_special_name = f"{case_slug}_L1_htc_vs_s_slice_0p9144_grouped_roughness.png"
            for source_figure, source_path in tuple(cutdata_builder.PNG_EXPORT_QUEUE):
                if source_path.parent.name != htc_case_id or source_path.name != htc_source_name:
                    continue
                grouped_figure = type(source_figure)(source_figure)
                grouped_traces = []
                shown_groups: set[str] = set()
                for trace in grouped_figure.data:
                    trace_name = str(trace.name or "")
                    trace_name_lower = trace_name.lower()
                    if "variable roughness" in trace_name_lower:
                        group_key = "variable"
                    elif re.search(r"roughness height\s*=\s*1(?:\.0+)?\s*mm", trace_name_lower):
                        group_key = "one_mm"
                    elif re.search(r"roughness height\s*=\s*(?:0\.5(?:0+)?|0\.5334)\s*mm", trace_name_lower):
                        group_key = "half_mm"
                    else:
                        continue
                    group_style = NACA0012_ROUGHNESS_GROUP_STYLE[group_key]
                    participant_match = re.match(r"^(\d{1,3})(?=\D|$)", trace_name.strip())
                    trace_meta = dict(trace.meta) if isinstance(trace.meta, dict) else {}
                    if participant_match:
                        trace_meta["ipw3_participant_id"] = f"{int(participant_match.group(1)):03d}"
                    trace.meta = trace_meta
                    trace.line.update(group_style["line"])
                    trace.marker.update(group_style["marker"])
                    trace.mode = "lines+markers" if group_style.get("show_markers", False) else "lines"
                    trace.name = str(group_style["label"])
                    trace.legendgroup = f"htc_roughness_{group_key}"
                    trace.showlegend = group_key not in shown_groups
                    shown_groups.add(group_key)
                    grouped_traces.append(trace)
                grouped_figure.data = tuple(grouped_traces)
                grouped_figure.update_layout(
                    title={"text": "Heat-transfer coefficient | L1 | Roughness groups", "x": 0.5, "xanchor": "center"},
                )
                cutdata_builder.PNG_EXPORT_QUEUE.append((grouped_figure, source_path.with_name(htc_special_name)))
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
                # Preserve experimental contours and the clean-airfoil trace.
                if "exp" in trace_name_lower or str(trace.legendgroup or "").startswith("experimental_") or trace.legendgroup == "clean_reference":
                    grouped_traces.append(trace)
                    continue
                if "variable roughness" in trace_name_lower:
                    group_key = "variable"
                elif re.search(r"roughness height\s*=\s*1(?:\.0+)?\s*mm", trace_name_lower):
                    group_key = "one_mm"
                elif re.search(r"roughness height\s*=\s*(?:0\.5(?:0+)?|0\.5334)\s*mm", trace_name_lower):
                    group_key = "half_mm"
                else:
                    continue
                group_style = NACA0012_ROUGHNESS_GROUP_STYLE[group_key]
                participant_match = re.match(r"^(\d{1,3})(?=\D|$)", trace_name.strip())
                trace_meta = dict(trace.meta) if isinstance(trace.meta, dict) else {}
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

                for _, (presentation_case_id, source_name, image_width, image_height, excluded_ids, figure_show_legend) in NACA0012_PRESENTATION_FIGURES.items():
                    if presentation_case_id == case_id and source_name == export_path.name:
                        width = image_width
                        height = image_height
                        show_legend = figure_show_legend
                        excluded_participants = {
                            f"{int(participant_id):03d}" if str(participant_id).strip().isdigit() else str(participant_id).strip()
                            for participant_id in excluded_ids
                        }
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

                if "_grouped_roughness" in export_path.stem.lower():
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

                qc_prime_y_range = (0.0, 400.0)
                if "_qc_prime_" in export_path.name.lower():
                    y_min, y_max = qc_prime_y_range
                    figure.update_yaxes(range=[y_min, y_max], autorange=False)

                plot_name = export_path.stem.lower()
                is_diameter_statistics_plot = "_vs_droplet_diameter_" in plot_name
                is_grouped_roughness_plot = "_grouped_roughness" in plot_name
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
                is_convergence_plot = (
                    module is convergence_data_builder
                    and not is_ice_horn_method_plot
                    and not is_diameter_statistics_plot
                    and not is_participant_horn_summary
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
                        ticktext=["2.556", "3.334", "4.486", "5.546"],
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
                            "N<sup>−1/3</sup> [×10<sup>−3</sup>]"
                            "<br><span style='font-size:24px'>"
                            "← Finest grid&nbsp;&nbsp;&nbsp;|&nbsp;&nbsp;&nbsp;Coarsest grid →"
                            "</span>"
                        ),
                    )

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
                            "1 / N<sub>bins</sub>"
                            "<br><span style='font-size:24px'>"
                            "</span>"
                        )
                    )



        # Preserve each curated figure's configured presentation canvas instead
        # of replacing it with the plot family's normal PNG export dimensions.
        convergence_data_builder.flush_png_exports(scale=1, width=None, height=None)
        cutdata_builder.flush_png_exports(scale=1, width=None, height=None)
        iceshape_builder.flush_png_exports(scale=1, width=None, height=None)

        missing: list[str] = []
        for destination_text, (case_id, source_name, width, height, excluded_ids, show_legend) in NACA0012_PRESENTATION_FIGURES.items():
            source = staging_dir / case_id / source_name
            destination = output_dir / destination_text
            if not source.exists():
                missing.append(f"{destination_text} <- {case_id}/{source_name}")
                continue
            destination.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(source, destination)

        if missing:
            raise RuntimeError("Missing presentation plot exports:\n  " + "\n  ".join(missing))



def generate(participants, output_dir: Path = OUTPUT_DIR) -> int:
    if output_dir.exists():
        shutil.rmtree(output_dir)
    write_naca0012_presentation_figures(participants, output_dir)
    return len(NACA0012_PRESENTATION_FIGURES)


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
