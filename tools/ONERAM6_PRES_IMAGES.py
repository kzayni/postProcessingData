"""Generate presentation-ready ONERA M6 images for all configured slices."""

from __future__ import annotations

import copy
import math
from pathlib import Path
import re
import shutil
import tempfile

import numpy

from tools import convergence_data_builder, cutdata_builder, iceshape_builder
from tools.participant_horn_export import participant_horn_method_figures
from tools.ice_limits_export import ice_limit_figures
from tools.gatherParticipantData import CASE_SLICES, VALID_GRID_LEVELS


CASE_ID = "TC_ONERAM6"
from tools.roughness_legend import apply_roughness_participant_legend
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
GROUPED_ROUGHNESS_PLOT_SYMBOL_SIZE = 12
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

ONERAM6_TURBULENCE_MODEL_STYLES = {
    "kw": {
        "participant_ids": {"001", "002", "004", "008", "009", "019"},
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

ONERAM6_TURBULENCE_ROUGHNESS_COLORS = {
    "kw": {
        "0.5mm": "#56B4E9", "1mm": "#1F4E9E",
        "1.5mm": "#0072B2", "variable_roughness": "#7B2CBF",
    },
    "sa": {
        "0.5mm": "#ff7f0e", "1mm": "#d62728",
        "1.5mm": "#A50F15", "variable_roughness": "#800020",
    },
}

OUTPUT_DIR = Path("FIGURES_ONERAM6")

# Standard ONERA M6 presentation canvas in pixels.
WIDTH = 1350
HEIGHT = 1000

MedianLineConfig = dict[str, object]
FigureSpec = tuple[str, str, int, int, list[str], bool] | tuple[
    str, str, int, int, list[str], bool, MedianLineConfig
]


# ONERA M6 presentation figures. Each plot can be configured independently.
# Key: output path under FIGURES_ONERAM6.
# Value: (case ID, source PNG, width, height, excluded IDs, show legend,
#         optional grid-median configuration).
# Use [] to include everyone; for example, ["001", "014"] excludes those participants.
ONERAM6_PRESENTATION_FIGURES: dict[str, FigureSpec] = {
    "ICE_ACCRETION/tc_oneram6_ice_mass_vs_n_1mm_l1_vs_inverse_bins.png": (CASE_ID, "tc_oneram6_ice_mass_vs_n_1mm_l1_vs_inverse_bins.png", WIDTH, HEIGHT, [], True),
    "ICE_ACCRETION/tc_oneram6_water_evap_mass_vs_n_1mm_l1_vs_inverse_bins.png": (CASE_ID, "tc_oneram6_water_evap_mass_vs_n_1mm_l1_vs_inverse_bins.png", WIDTH, HEIGHT, ["001"], True),
    "ICE_ACCRETION/tc_oneram6_water_evap_mass_vs_n_1mm_bins15_all_grid_levels.png": (CASE_ID, "tc_oneram6_water_evap_mass_vs_n_1mm_bins15_all_grid_levels.png", WIDTH, HEIGHT, ["001"], True),
    "ICE_ACCRETION/tc_oneram6_ice_mass_vs_n_1mm_l1_vs_inverse_bins_relative_to_bins15.png": (CASE_ID, "tc_oneram6_ice_mass_vs_n_1mm_l1_vs_inverse_bins_relative_to_bins15.png", WIDTH, HEIGHT, [], True),
    "HTC/tc_oneram6_qc_prime_vs_n_y_0.1_grouped_roughness.png": (CASE_ID, "tc_oneram6_qc_prime_vs_n_y_0.1_grouped_roughness.png", WIDTH, HEIGHT, ["001", "002"], True),
    "HTC/tc_oneram6_qc_prime_vs_n_y_0.1_grouped_roughness_relative_to_l1.png": (CASE_ID, "tc_oneram6_qc_prime_vs_n_y_0.1_grouped_roughness_relative_to_l1.png", WIDTH, HEIGHT, ["001", "002"], True),
    "HTC/tc_oneram6_qc_prime_vs_n_y_0.75_grouped_roughness.png": (CASE_ID, "tc_oneram6_qc_prime_vs_n_y_0.75_grouped_roughness.png", WIDTH, HEIGHT, ["001","002"], True),
    "HTC/tc_oneram6_qc_prime_vs_n_y_0.75_grouped_roughness_relative_to_l1.png": (CASE_ID, "tc_oneram6_qc_prime_vs_n_y_0.75_grouped_roughness_relative_to_l1.png", WIDTH, HEIGHT, ["001","002"], True),
    "HTC/tc_oneram6_qc_prime_vs_n_y_1.4_grouped_roughness.png": (CASE_ID, "tc_oneram6_qc_prime_vs_n_y_1.4_grouped_roughness.png", WIDTH, HEIGHT, ["001", "002"], True),
    "HTC/tc_oneram6_qc_prime_vs_n_y_1.4_grouped_roughness_relative_to_l1.png": (CASE_ID, "tc_oneram6_qc_prime_vs_n_y_1.4_grouped_roughness_relative_to_l1.png", WIDTH, HEIGHT, ["001", "002"], True),
    "SURF_TEMP_FF/tc_oneram6_mean_surface_temperature_vs_n_slice_0p1_roughness_1mm.png": (CASE_ID, "tc_oneram6_mean_surface_temperature_vs_n_slice_0p1_roughness_1mm.png", WIDTH, HEIGHT, ["001"], True),
    "SURF_TEMP_FF/tc_oneram6_mean_surface_temperature_vs_n_slice_0p1_roughness_1mm_relative_to_l1.png": (CASE_ID, "tc_oneram6_mean_surface_temperature_vs_n_slice_0p1_roughness_1mm_relative_to_l1.png", WIDTH, HEIGHT, ["001"], True),
    "SURF_TEMP_FF/tc_oneram6_mean_surface_temperature_vs_n_slice_0p1_grouped_roughness.png": (CASE_ID, "tc_oneram6_mean_surface_temperature_vs_n_slice_0p1_grouped_roughness.png", WIDTH, HEIGHT, ["001"], True),
    "SURF_TEMP_FF/tc_oneram6_mean_surface_temperature_vs_n_slice_0p1_grouped_roughness_relative_to_l1.png": (CASE_ID, "tc_oneram6_mean_surface_temperature_vs_n_slice_0p1_grouped_roughness_relative_to_l1.png", WIDTH, HEIGHT, ["001"], True),
    "SURF_TEMP_FF/tc_oneram6_mean_freezing_fraction_vs_n_slice_0p1_roughness_1mm.png": (CASE_ID, "tc_oneram6_mean_freezing_fraction_vs_n_slice_0p1_roughness_1mm.png", WIDTH, HEIGHT, ["001"], True),
    "SURF_TEMP_FF/tc_oneram6_mean_freezing_fraction_vs_n_slice_0p1_roughness_1mm_relative_to_l1.png": (CASE_ID, "tc_oneram6_mean_freezing_fraction_vs_n_slice_0p1_roughness_1mm_relative_to_l1.png", WIDTH, HEIGHT, ["001"], True),
    "SURF_TEMP_FF/tc_oneram6_mean_freezing_fraction_vs_n_slice_0p1_grouped_roughness.png": (CASE_ID, "tc_oneram6_mean_freezing_fraction_vs_n_slice_0p1_grouped_roughness.png", WIDTH, HEIGHT, ["001"], True),
    "SURF_TEMP_FF/tc_oneram6_mean_freezing_fraction_vs_n_slice_0p1_grouped_roughness_relative_to_l1.png": (CASE_ID, "tc_oneram6_mean_freezing_fraction_vs_n_slice_0p1_grouped_roughness_relative_to_l1.png", WIDTH, HEIGHT, ["001"], True),
    "SURF_TEMP_FF/tc_oneram6_mean_surface_temperature_vs_n_slice_0p75_roughness_1mm.png": (CASE_ID, "tc_oneram6_mean_surface_temperature_vs_n_slice_0p75_roughness_1mm.png", WIDTH, HEIGHT, ["001"], True),
    "SURF_TEMP_FF/tc_oneram6_mean_surface_temperature_vs_n_slice_0p75_roughness_1mm_relative_to_l1.png": (CASE_ID, "tc_oneram6_mean_surface_temperature_vs_n_slice_0p75_roughness_1mm_relative_to_l1.png", WIDTH, HEIGHT, ["001"], True),
    "SURF_TEMP_FF/tc_oneram6_mean_surface_temperature_vs_n_slice_0p75_grouped_roughness.png": (CASE_ID, "tc_oneram6_mean_surface_temperature_vs_n_slice_0p75_grouped_roughness.png", WIDTH, HEIGHT, ["001"], True),
    "SURF_TEMP_FF/tc_oneram6_mean_surface_temperature_vs_n_slice_0p75_grouped_roughness_relative_to_l1.png": (CASE_ID, "tc_oneram6_mean_surface_temperature_vs_n_slice_0p75_grouped_roughness_relative_to_l1.png", WIDTH, HEIGHT, ["001"], True),
    "SURF_TEMP_FF/tc_oneram6_mean_freezing_fraction_vs_n_slice_0p75_roughness_1mm.png": (CASE_ID, "tc_oneram6_mean_freezing_fraction_vs_n_slice_0p75_roughness_1mm.png", WIDTH, HEIGHT, ["001"], True),
    "SURF_TEMP_FF/tc_oneram6_mean_freezing_fraction_vs_n_slice_0p75_roughness_1mm_relative_to_l1.png": (CASE_ID, "tc_oneram6_mean_freezing_fraction_vs_n_slice_0p75_roughness_1mm_relative_to_l1.png", WIDTH, HEIGHT, ["001"], True),
    "SURF_TEMP_FF/tc_oneram6_mean_freezing_fraction_vs_n_slice_0p75_grouped_roughness.png": (CASE_ID, "tc_oneram6_mean_freezing_fraction_vs_n_slice_0p75_grouped_roughness.png", WIDTH, HEIGHT, ["001"], True),
    "SURF_TEMP_FF/tc_oneram6_mean_freezing_fraction_vs_n_slice_0p75_grouped_roughness_relative_to_l1.png": (CASE_ID, "tc_oneram6_mean_freezing_fraction_vs_n_slice_0p75_grouped_roughness_relative_to_l1.png", WIDTH, HEIGHT, ["001"], True),
    "SURF_TEMP_FF/tc_oneram6_mean_surface_temperature_vs_n_slice_1p4_roughness_1mm.png": (CASE_ID, "tc_oneram6_mean_surface_temperature_vs_n_slice_1p4_roughness_1mm.png", WIDTH, HEIGHT, ["001"], True),
    "SURF_TEMP_FF/tc_oneram6_mean_surface_temperature_vs_n_slice_1p4_roughness_1mm_relative_to_l1.png": (CASE_ID, "tc_oneram6_mean_surface_temperature_vs_n_slice_1p4_roughness_1mm_relative_to_l1.png", WIDTH, HEIGHT, ["001"], True),
    "SURF_TEMP_FF/tc_oneram6_mean_surface_temperature_vs_n_slice_1p4_grouped_roughness.png": (CASE_ID, "tc_oneram6_mean_surface_temperature_vs_n_slice_1p4_grouped_roughness.png", WIDTH, HEIGHT, ["001"], True),
    "SURF_TEMP_FF/tc_oneram6_mean_surface_temperature_vs_n_slice_1p4_grouped_roughness_relative_to_l1.png": (CASE_ID, "tc_oneram6_mean_surface_temperature_vs_n_slice_1p4_grouped_roughness_relative_to_l1.png", WIDTH, HEIGHT, ["001"], True),
    "SURF_TEMP_FF/tc_oneram6_mean_freezing_fraction_vs_n_slice_1p4_roughness_1mm.png": (CASE_ID, "tc_oneram6_mean_freezing_fraction_vs_n_slice_1p4_roughness_1mm.png", WIDTH, HEIGHT, ["001"], True),
    "SURF_TEMP_FF/tc_oneram6_mean_freezing_fraction_vs_n_slice_1p4_roughness_1mm_relative_to_l1.png": (CASE_ID, "tc_oneram6_mean_freezing_fraction_vs_n_slice_1p4_roughness_1mm_relative_to_l1.png", WIDTH, HEIGHT, ["001"], True),
    "SURF_TEMP_FF/tc_oneram6_mean_freezing_fraction_vs_n_slice_1p4_grouped_roughness.png": (CASE_ID, "tc_oneram6_mean_freezing_fraction_vs_n_slice_1p4_grouped_roughness.png", WIDTH, HEIGHT, ["001"], True),
    "SURF_TEMP_FF/tc_oneram6_mean_freezing_fraction_vs_n_slice_1p4_grouped_roughness_relative_to_l1.png": (CASE_ID, "tc_oneram6_mean_freezing_fraction_vs_n_slice_1p4_grouped_roughness_relative_to_l1.png", WIDTH, HEIGHT, ["001"], True),
    "HTC/tc_oneram6_L1_htc_vs_s_slice_0p1_all_roughness.png": (CASE_ID, "tc_oneram6_L1_htc_vs_s_slice_0p1_all_roughness.png", WIDTH, HEIGHT, [], True),
    "HTC/tc_oneram6_L1_htc_vs_s_slice_0p75_all_roughness.png": (CASE_ID, "tc_oneram6_L1_htc_vs_s_slice_0p75_all_roughness.png", WIDTH, HEIGHT, [], True),
    "HTC/tc_oneram6_L1_htc_vs_s_slice_1p4_all_roughness.png": (CASE_ID, "tc_oneram6_L1_htc_vs_s_slice_1p4_all_roughness.png", WIDTH, HEIGHT, [], True),
    "IMPINGEMENT/tc_oneram6_water_mass_coefficient_of_variation_vs_droplet_diameter_bins15_all_grid_levels.png": (CASE_ID, "tc_oneram6_water_mass_coefficient_of_variation_vs_droplet_diameter_bins15_all_grid_levels.png", WIDTH, HEIGHT, [], True),
    "IMPINGEMENT/tc_oneram6_water_mass_relative_iqr_vs_droplet_diameter_bins15_all_grid_levels.png": (CASE_ID, "tc_oneram6_water_mass_relative_iqr_vs_droplet_diameter_bins15_all_grid_levels.png", WIDTH, HEIGHT, [], True),

    # Integrated quantities
    "AERODYNAMIC/tc_oneram6_cl_vs_n_1mm.png": (
        CASE_ID, "tc_oneram6_cl_vs_n_1mm.png", WIDTH, HEIGHT, [], True,
        {"enabled": False, "value": 0.5, "color": "#ff7f0e", "dash": "dash", "width": 3},
    ),
    "AERODYNAMIC/tc_oneram6_cd_vs_n_1mm.png": (CASE_ID, "tc_oneram6_cd_vs_n_1mm.png", WIDTH, HEIGHT, [], True),
    "AERODYNAMIC/tc_oneram6_cmy_vs_n_1mm.png": (CASE_ID, "tc_oneram6_cmy_vs_n_1mm.png", WIDTH, HEIGHT, [], True),
    "AERODYNAMIC/tc_oneram6_cl_vs_n_1mm_relative_to_l1.png": (CASE_ID, "tc_oneram6_cl_vs_n_1mm_relative_to_l1.png", WIDTH, HEIGHT, [], True),
    "AERODYNAMIC/tc_oneram6_cd_vs_n_1mm_relative_to_l1.png": (CASE_ID, "tc_oneram6_cd_vs_n_1mm_relative_to_l1.png", WIDTH, HEIGHT, [], True),
    "AERODYNAMIC/tc_oneram6_cmy_vs_n_1mm_relative_to_l1.png": (CASE_ID, "tc_oneram6_cmy_vs_n_1mm_relative_to_l1.png", WIDTH, HEIGHT, [], True),
    "IMPINGEMENT/tc_oneram6_water_mass_vs_n_1mm_bins15_all_grid_levels.png": (CASE_ID, "tc_oneram6_water_mass_vs_n_1mm_bins15_all_grid_levels.png", WIDTH, HEIGHT, ["015"], True),
    "IMPINGEMENT/tc_oneram6_water_mass_vs_n_1mm_bins15_all_grid_levels_relative_to_l1.png": (CASE_ID, "tc_oneram6_water_mass_vs_n_1mm_bins15_all_grid_levels_relative_to_l1.png", WIDTH, HEIGHT, ["015"], True),
    "IMPINGEMENT/tc_oneram6_water_mass_vs_n_1mm_l1_vs_inverse_bins.png": (CASE_ID, "tc_oneram6_water_mass_vs_n_1mm_l1_vs_inverse_bins.png", WIDTH, HEIGHT, ["001", "015"], True),
    "IMPINGEMENT/tc_oneram6_water_mass_vs_n_1mm_l2_vs_inverse_bins.png": (CASE_ID, "tc_oneram6_water_mass_vs_n_1mm_l2_vs_inverse_bins.png", WIDTH, HEIGHT, ["001", "015"], True),
    "IMPINGEMENT/tc_oneram6_water_mass_vs_n_1mm_l3_vs_inverse_bins.png": (CASE_ID, "tc_oneram6_water_mass_vs_n_1mm_l3_vs_inverse_bins.png", WIDTH, HEIGHT, ["001", "015"], True),
    "IMPINGEMENT/tc_oneram6_water_mass_vs_n_1mm_l4_vs_inverse_bins.png": (CASE_ID, "tc_oneram6_water_mass_vs_n_1mm_l4_vs_inverse_bins.png", WIDTH, HEIGHT, ["001", "015"], True),
    "IMPINGEMENT/tc_oneram6_water_mass_vs_n_1mm_l1_vs_inverse_bins_relative_to_bins15.png": (CASE_ID, "tc_oneram6_water_mass_vs_n_1mm_l1_vs_inverse_bins_relative_to_bins15.png", WIDTH, HEIGHT, ["001", "015"], True),
    "IMPINGEMENT/tc_oneram6_water_mass_vs_n_1mm_all_grid_levels_vs_inverse_bins.png": (CASE_ID, "tc_oneram6_water_mass_vs_n_1mm_all_grid_levels_vs_inverse_bins.png", WIDTH, HEIGHT, ["001", "015"], True),
    "ICE_ACCRETION/tc_oneram6_ice_mass_vs_n_1mm_bins15_all_grid_levels.png": (CASE_ID, "tc_oneram6_ice_mass_vs_n_1mm_bins15_all_grid_levels.png", WIDTH, HEIGHT, [], True),
    "ICE_ACCRETION/tc_oneram6_ice_mass_vs_n_1mm_bins15_all_grid_levels_relative_to_l1.png": (CASE_ID, "tc_oneram6_ice_mass_vs_n_1mm_bins15_all_grid_levels_relative_to_l1.png", WIDTH, HEIGHT, [], True),
    "ICE_ACCRETION/tc_oneram6_ice_to_water_ratio_vs_n_required_bins15.png": (CASE_ID, "tc_oneram6_ice_to_water_ratio_vs_n_required_bins15.png", WIDTH, HEIGHT, [], True),
    "ICE_ACCRETION/tc_oneram6_ice_evap_to_water_ratio_vs_n_required_bins15.png": (CASE_ID, "tc_oneram6_ice_evap_to_water_ratio_vs_n_required_bins15.png", WIDTH, HEIGHT, [], True),

    # Y = 0.1 m
    "IMPINGEMENT/tc_oneram6_width_bins15_0_1.png": (CASE_ID, "tc_oneram6_width_bins15_0_1.png", WIDTH, HEIGHT, ["015"], True),
    "IMPINGEMENT/tc_oneram6_width_bins15_0_1_relative_to_l1.png": (CASE_ID, "tc_oneram6_width_bins15_0_1_relative_to_l1.png", WIDTH, HEIGHT, ["015"], True),
    "IMPINGEMENT/tc_oneram6_width_distribution_convergence_l1_0_1.png": (CASE_ID, "tc_oneram6_width_distribution_convergence_l1_0_1.png", WIDTH, HEIGHT, ["015"], True),
    "IMPINGEMENT/tc_oneram6_width_distribution_convergence_l1_0_1_relative_to_bins15.png": (CASE_ID, "tc_oneram6_width_distribution_convergence_l1_0_1_relative_to_bins15.png", WIDTH, HEIGHT, ["015"], True),
    "AERODYNAMIC/tc_oneram6_L1_cp_vs_s_slice_0p1_roughness_1mm.png": (CASE_ID, "tc_oneram6_L1_cp_vs_s_slice_0p1_roughness_1mm.png", WIDTH, HEIGHT, [], True),
    "AERODYNAMIC/tc_oneram6_L1_cp_vs_x_slice_0p1_roughness_1mm.png": (CASE_ID, "tc_oneram6_L1_cp_vs_x_slice_0p1_roughness_1mm.png", WIDTH, HEIGHT, [], True),
    "HTC/tc_oneram6_L1_htc_vs_s_slice_0p1_roughness_1mm.png": (CASE_ID, "tc_oneram6_L1_htc_vs_s_slice_0p1_roughness_1mm.png", WIDTH, HEIGHT, [], True),
    "HTC/tc_oneram6_L1_recovery_temperature_vs_s_slice_0p1_roughness_1mm.png": (CASE_ID, "tc_oneram6_L1_recovery_temperature_vs_s_slice_0p1_roughness_1mm.png", WIDTH, HEIGHT, [], True),
    "HTC/tc_oneram6_qc_prime_vs_n_y_0.1.png": (CASE_ID, "tc_oneram6_qc_prime_vs_n_y_0.1.png", WIDTH, HEIGHT, ["001", "002"], True),
    "HTC/tc_oneram6_qc_prime_vs_n_y_0.1_relative_to_l1.png": (CASE_ID, "tc_oneram6_qc_prime_vs_n_y_0.1_relative_to_l1.png", WIDTH, HEIGHT, ["001", "002"], True),
    "IMPINGEMENT/tc_oneram6_L1_beta_bins15_vs_s_slice_0p1_roughness_1mm.png": (CASE_ID, "tc_oneram6_L1_beta_bins15_vs_s_slice_0p1_roughness_1mm.png", WIDTH, HEIGHT, ["015"], True),
    "SURF_TEMP_FF/tc_oneram6_L1_surface_temperature_vs_s_slice_0p1_roughness_1mm.png": (CASE_ID, "tc_oneram6_L1_surface_temperature_vs_s_slice_0p1_roughness_1mm.png", WIDTH, HEIGHT, [], True),
    "SURF_TEMP_FF/tc_oneram6_L1_freezing_fraction_vs_s_slice_0p1_roughness_1mm.png": (CASE_ID, "tc_oneram6_L1_freezing_fraction_vs_s_slice_0p1_roughness_1mm.png", WIDTH, HEIGHT, [], True),
    "ICE_SHAPES/tc_oneram6_L1_single_layer_ice_shape_slice_0p1_bins15_roughness_1mm.png": (CASE_ID, "tc_oneram6_L1_single_layer_ice_shape_slice_0p1_bins15_roughness_1mm.png", WIDTH, HEIGHT, [], True),
    "ICE_SHAPES/tc_oneram6_finest_grid_multilayer_ice_shape_slice_0p1_all_conditions.png": (CASE_ID, "tc_oneram6_finest_grid_multilayer_ice_shape_slice_0p1_all_conditions.png", WIDTH, HEIGHT, [], True),

    # Y = 0.75 m
    "IMPINGEMENT/tc_oneram6_width_bins15_0_75.png": (CASE_ID, "tc_oneram6_width_bins15_0_75.png", WIDTH, HEIGHT, ["015"], True),
    "IMPINGEMENT/tc_oneram6_width_bins15_0_75_relative_to_l1.png": (CASE_ID, "tc_oneram6_width_bins15_0_75_relative_to_l1.png", WIDTH, HEIGHT, ["015"], True),
    "IMPINGEMENT/tc_oneram6_width_distribution_convergence_l1_0_75.png": (CASE_ID, "tc_oneram6_width_distribution_convergence_l1_0_75.png", WIDTH, HEIGHT, ["001", "015"], True),
    "IMPINGEMENT/tc_oneram6_width_distribution_convergence_l1_0_75_relative_to_bins15.png": (CASE_ID, "tc_oneram6_width_distribution_convergence_l1_0_75_relative_to_bins15.png", WIDTH, HEIGHT, ["001", "015"], True),
    "AERODYNAMIC/tc_oneram6_L1_cp_vs_s_slice_0p75_roughness_1mm.png": (CASE_ID, "tc_oneram6_L1_cp_vs_s_slice_0p75_roughness_1mm.png", WIDTH, HEIGHT, [], True),
    "AERODYNAMIC/tc_oneram6_L1_cp_vs_x_slice_0p75_roughness_1mm.png": (CASE_ID, "tc_oneram6_L1_cp_vs_x_slice_0p75_roughness_1mm.png", WIDTH, HEIGHT, [], True),
    "HTC/tc_oneram6_L1_htc_vs_s_slice_0p75_roughness_1mm.png": (CASE_ID, "tc_oneram6_L1_htc_vs_s_slice_0p75_roughness_1mm.png", WIDTH, HEIGHT, [], True),
    "HTC/tc_oneram6_L1_recovery_temperature_vs_s_slice_0p75_roughness_1mm.png": (CASE_ID, "tc_oneram6_L1_recovery_temperature_vs_s_slice_0p75_roughness_1mm.png", WIDTH, HEIGHT, [], True),
    "HTC/tc_oneram6_qc_prime_vs_n_y_0.75.png": (CASE_ID, "tc_oneram6_qc_prime_vs_n_y_0.75.png", WIDTH, HEIGHT, ["001", "002"], True),
    "HTC/tc_oneram6_qc_prime_vs_n_y_0.75_relative_to_l1.png": (CASE_ID, "tc_oneram6_qc_prime_vs_n_y_0.75_relative_to_l1.png", WIDTH, HEIGHT, ["001", "002"], True),
    "IMPINGEMENT/tc_oneram6_L1_beta_bins15_vs_s_slice_0p75_roughness_1mm.png": (CASE_ID, "tc_oneram6_L1_beta_bins15_vs_s_slice_0p75_roughness_1mm.png", WIDTH, HEIGHT, ["015"], True),
    "SURF_TEMP_FF/tc_oneram6_L1_surface_temperature_vs_s_slice_0p75_roughness_1mm.png": (CASE_ID, "tc_oneram6_L1_surface_temperature_vs_s_slice_0p75_roughness_1mm.png", WIDTH, HEIGHT, [], True),
    "SURF_TEMP_FF/tc_oneram6_L1_freezing_fraction_vs_s_slice_0p75_roughness_1mm.png": (CASE_ID, "tc_oneram6_L1_freezing_fraction_vs_s_slice_0p75_roughness_1mm.png", WIDTH, HEIGHT, [], True),
    "ICE_SHAPES/tc_oneram6_L1_single_layer_ice_shape_slice_0p75_bins15_roughness_1mm.png": (CASE_ID, "tc_oneram6_L1_single_layer_ice_shape_slice_0p75_bins15_roughness_1mm.png", WIDTH, HEIGHT, [], True),
    "ICE_SHAPES/tc_oneram6_finest_grid_multilayer_ice_shape_slice_0p75_all_conditions.png": (CASE_ID, "tc_oneram6_finest_grid_multilayer_ice_shape_slice_0p75_all_conditions.png", WIDTH, HEIGHT, [], True),

    # Y = 1.4 m
    "IMPINGEMENT/tc_oneram6_width_bins15_1_4.png": (CASE_ID, "tc_oneram6_width_bins15_1_4.png", WIDTH, HEIGHT, ["015"], True),
    "IMPINGEMENT/tc_oneram6_width_bins15_1_4_relative_to_l1.png": (CASE_ID, "tc_oneram6_width_bins15_1_4_relative_to_l1.png", WIDTH, HEIGHT, ["015"], True),
    "IMPINGEMENT/tc_oneram6_width_distribution_convergence_l1_1_4.png": (CASE_ID, "tc_oneram6_width_distribution_convergence_l1_1_4.png", WIDTH, HEIGHT, ["015"], True),
    "IMPINGEMENT/tc_oneram6_width_distribution_convergence_l1_1_4_relative_to_bins15.png": (CASE_ID, "tc_oneram6_width_distribution_convergence_l1_1_4_relative_to_bins15.png", WIDTH, HEIGHT, ["015"], True),
    "AERODYNAMIC/tc_oneram6_L1_cp_vs_s_slice_1p4_roughness_1mm.png": (CASE_ID, "tc_oneram6_L1_cp_vs_s_slice_1p4_roughness_1mm.png", WIDTH, HEIGHT, [], True),
    "AERODYNAMIC/tc_oneram6_L1_cp_vs_x_slice_1p4_roughness_1mm.png": (CASE_ID, "tc_oneram6_L1_cp_vs_x_slice_1p4_roughness_1mm.png", WIDTH, HEIGHT, [], True),
    "HTC/tc_oneram6_L1_htc_vs_s_slice_1p4_roughness_1mm.png": (CASE_ID, "tc_oneram6_L1_htc_vs_s_slice_1p4_roughness_1mm.png", WIDTH, HEIGHT, [], True),
    "HTC/tc_oneram6_L1_recovery_temperature_vs_s_slice_1p4_roughness_1mm.png": (CASE_ID, "tc_oneram6_L1_recovery_temperature_vs_s_slice_1p4_roughness_1mm.png", WIDTH, HEIGHT, [], True),
    "HTC/tc_oneram6_qc_prime_vs_n_y_1.4.png": (CASE_ID, "tc_oneram6_qc_prime_vs_n_y_1.4.png", WIDTH, HEIGHT, ["001", "002"], True),
    "HTC/tc_oneram6_qc_prime_vs_n_y_1.4_relative_to_l1.png": (CASE_ID, "tc_oneram6_qc_prime_vs_n_y_1.4_relative_to_l1.png", WIDTH, HEIGHT, ["001", "002"], True),
    "IMPINGEMENT/tc_oneram6_L1_beta_bins15_vs_s_slice_1p4_roughness_1mm.png": (CASE_ID, "tc_oneram6_L1_beta_bins15_vs_s_slice_1p4_roughness_1mm.png", WIDTH, HEIGHT, ["015"], True),
    "SURF_TEMP_FF/tc_oneram6_L1_surface_temperature_vs_s_slice_1p4_roughness_1mm.png": (CASE_ID, "tc_oneram6_L1_surface_temperature_vs_s_slice_1p4_roughness_1mm.png", WIDTH, HEIGHT, [], True),
    "SURF_TEMP_FF/tc_oneram6_L1_freezing_fraction_vs_s_slice_1p4_roughness_1mm.png": (CASE_ID, "tc_oneram6_L1_freezing_fraction_vs_s_slice_1p4_roughness_1mm.png", WIDTH, HEIGHT, [], True),
    "ICE_SHAPES/tc_oneram6_L1_single_layer_ice_shape_slice_1p4_bins15_roughness_1mm.png": (CASE_ID, "tc_oneram6_L1_single_layer_ice_shape_slice_1p4_bins15_roughness_1mm.png", WIDTH, HEIGHT, [], True),
    "ICE_SHAPES/tc_oneram6_finest_grid_multilayer_ice_shape_slice_1p4_all_conditions.png": (CASE_ID, "tc_oneram6_finest_grid_multilayer_ice_shape_slice_1p4_all_conditions.png", WIDTH, HEIGHT, [], True),
}

# L1 surface distributions grouped by roughness. The participant-panel
# companion PNGs are created automatically from each grouped figure.
for _slice_value in CASE_SLICES[CASE_ID]:
    _slice_slug = str(_slice_value).replace(".", "p")
    for _roughness_slug in ("0p5mm", "1mm", "1p5mm", "variable"):
        _filename = (
            f"tc_oneram6_L1_htc_vs_s_slice_{_slice_slug}_"
            f"roughness_{_roughness_slug}_grouped_turbulence_models.png"
        )
        ONERAM6_PRESENTATION_FIGURES[f"HTC/{_filename}"] = (
            CASE_ID, _filename, WIDTH, HEIGHT, ["015"], True,
        )
    _filename = (
        f"tc_oneram6_L1_htc_vs_s_slice_{_slice_slug}_"
        "grouped_turbulence_models.png"
    )
    ONERAM6_PRESENTATION_FIGURES[f"HTC/{_filename}"] = (
        CASE_ID, _filename, WIDTH, HEIGHT, ["015"], True,
    )
    for _variable in ("surface_temperature", "freezing_fraction"):
        _filename = (
            f"tc_oneram6_L1_{_variable}_vs_s_slice_{_slice_slug}_"
            "grouped_turbulence_models.png"
        )
        ONERAM6_PRESENTATION_FIGURES[f"SURF_TEMP_FF/{_filename}"] = (
            CASE_ID, _filename, WIDTH, HEIGHT, ["015"], True,
        )
    for _variable in ("surface_temperature", "freezing_fraction"):
        _filename = f"tc_oneram6_L1_{_variable}_vs_s_slice_{_slice_slug}_grouped_roughness.png"
        ONERAM6_PRESENTATION_FIGURES[f"SURF_TEMP_FF/{_filename}"] = (
            CASE_ID, _filename, WIDTH, HEIGHT, [], True,
        )

for _slice_value in CASE_SLICES[CASE_ID]:
    _slice_slug = str(_slice_value).replace(".", "p")
    _filename = f"tc_oneram6_L1_single_layer_ice_shape_slice_{_slice_slug}_bins07_grouped_roughness.png"
    ONERAM6_PRESENTATION_FIGURES[f"ICE_SHAPES/{_filename}"] = (CASE_ID, _filename, WIDTH, HEIGHT, [], True)
    for _roughness_slug in ("0p5mm", "1mm", "1p5mm", "variable"):
        _filename = (
            f"tc_oneram6_L1_single_layer_ice_shape_slice_{_slice_slug}_bins07_"
            f"roughness_{_roughness_slug}_grouped_turbulence_models.png"
        )
        ONERAM6_PRESENTATION_FIGURES[f"ICE_SHAPES/{_filename}"] = (
            CASE_ID, _filename, WIDTH, HEIGHT, ["015"], True,
        )
    _filename = (
        f"tc_oneram6_L1_single_layer_ice_shape_slice_{_slice_slug}_bins07_"
        "grouped_turbulence_models.png"
    )
    ONERAM6_PRESENTATION_FIGURES[f"ICE_SHAPES/{_filename}"] = (
        CASE_ID, _filename, WIDTH, HEIGHT, ["015"], True,
    )


# Horn exports use the same single-layer selections as the ice-shape plots.
for _slice_value in CASE_SLICES[CASE_ID]:
    _slug = str(_slice_value).replace(".", "p")
    _example = f"tc_oneram6_010_slice_{_slug}_upper_horn_angle_method.png"
    ONERAM6_PRESENTATION_FIGURES[f"ICE_HORNS/{_example}"] = (CASE_ID, _example, WIDTH, HEIGHT, [], False)
    for _selection in ("bins15_roughness_1mm", "bins07_grouped_roughness"):
        for _relative in ("", "_relative_to_l1"):
            _name = f"tc_oneram6_upper_horn_angle_vs_n_slice_{_slug}_{_selection}{_relative}.png"
            ONERAM6_PRESENTATION_FIGURES[f"ICE_HORNS/{_name}"] = (CASE_ID, _name, WIDTH, HEIGHT, [], True)
    for _relative in ("", "_relative_to_bins15"):
        _name = f"tc_oneram6_upper_horn_angle_distribution_convergence_l1_slice_{_slug}_roughness_1mm{_relative}.png"
        ONERAM6_PRESENTATION_FIGURES[f"ICE_HORNS/{_name}"] = (CASE_ID, _name, WIDTH, HEIGHT, [], True)


# Independent M6 HTC styles: all participants share the roughness style.
HTC_ROUGHNESS_STYLES = {
    "0.5mm": ("0.5 mm", "#2CA02C", "circle"),
    "1mm": ("1 mm", "#D62728", "square"),
    "1.5mm": ("1.5 mm", "#1F77B4", "diamond"),
    "variable_roughness": ("Variable", "#000000", "triangle-up"),
}


# CFD (R) and (O) sheets can each contain several roughness heights.
# Keep these coefficient colors separate from the HTC palette for easy editing.
COEFFICIENT_ROUGHNESS_STYLES = {
    "smooth": ("Smooth", "#000000"),
    "0.5mm": ("0.5 mm", "#2CA02C"),
    "1mm": ("1 mm", "#D62728"),
    "1.5mm": ("1.5 mm", "#1F77B4"),
    "variable_roughness": ("Variable", "#FF7F0E"),
}

COEFFICIENT_MATRIX_PARTICIPANTS = {"004", "010", "019"}


for _coefficient in ("cl", "cd", "cmy"):
    _filename = f"tc_oneram6_{_coefficient}_vs_n_grouped_roughness_participants.png"
    ONERAM6_PRESENTATION_FIGURES[f"AERODYNAMIC/{_filename}"] = (
        CASE_ID, _filename, WIDTH, HEIGHT, [], True,
    )


def _queue_coefficients_by_roughness(participants, case_dir: Path) -> None:
    """Build one participant matrix per M6 CFD coefficient."""
    import plotly.graph_objects as go
    from tools.roughness_panels import build_roughness_participant_panels

    selected = [
        participant for participant in participants
        if str(participant.participant_id).zfill(3) in COEFFICIENT_MATRIX_PARTICIPANTS
    ]

    for plot_key in ("cl_vs_n", "cd_vs_n", "cmy_vs_n"):
        spec = next(p for p in convergence_data_builder.GRID_CONVERGENCE_PLOTS if p["plot_key"] == plot_key)
        combined = None
        for rank, (roughness, (label, color)) in enumerate(COEFFICIENT_ROUGHNESS_STYLES.items()):
            seen_participants = set()
            for requirement in ("required", "optional"):
                figure, count, _ = convergence_data_builder.build_grid_convergence_figure(
                    selected, CASE_ID, spec, roughness_filter=roughness, requirement=requirement,
                )
                if not count:
                    continue
                if combined is None:
                    combined = go.Figure(layout=figure.layout)
                for trace in figure.data:
                    match = re.match(r"^(\d{1,3})(?=\D|$)", str(trace.name or ""))
                    if not match:
                        continue
                    participant_id = match.group(1).zfill(3)
                    if participant_id in seen_participants:
                        continue
                    seen_participants.add(participant_id)
                    trace.name = label
                    trace.legendgroup = f"{plot_key}_roughness_{roughness}"
                    trace.legendrank = rank
                    trace.meta = {"ipw3_participant_id": participant_id, "ipw3_roughness_key": roughness}
                    trace.line.update(color=color, width=5, dash="solid")
                    trace.marker.update(color=color, size=9, maxdisplayed=20,
                                        line={"color": "#000000", "width": 1})
                    trace.mode = "lines+markers"
                    combined.add_trace(trace)
        if combined is None or not combined.data:
            continue
        combined.update_layout(legend_traceorder="normal")
        matrix_name = f"tc_oneram6_{plot_key}_grouped_roughness_participants.png"
        _style_figure(combined, convergence_data_builder, case_dir / matrix_name,
                      (CASE_ID, matrix_name, WIDTH, HEIGHT, [], True))
        # The standard coefficient limits are tuned for the 1 mm plots and
        # clip 010's smooth CL (about 0.1301–0.1308). Fit the matrix to all
        # submitted roughness curves instead.
        values = [
            float(value) for trace in combined.data if trace.y is not None
            for value in trace.y if value is not None and math.isfinite(float(value))
        ]
        if values:
            span = max(values) - min(values)
            padding = max(span * 0.07, 1e-6)
            combined.update_yaxes(range=[min(values) - padding, max(values) + padding],
                                  autorange=False, dtick=None)
        matrix = build_roughness_participant_panels(
            combined, columns=2, participant_title_size=30,
        )
        convergence_data_builder.PNG_EXPORT_QUEUE.append((matrix, case_dir / matrix_name))


def _queue_all_roughness_htc(participants, case_dir: Path) -> None:
    import plotly.graph_objects as go

    spec = next(p for p in cutdata_builder.CUTDATA_PLOTS if p["plot_key"] == "htc_vs_s")
    for slice_value in CASE_SLICES[CASE_ID]:
        combined = None
        for rank, (roughness, (label, color, symbol)) in enumerate(HTC_ROUGHNESS_STYLES.items()):
            figure, _, _, _ = cutdata_builder.build_cutdata_figure(
                participants, CASE_ID, "L1", spec,
                slice_filter=slice_value, roughness_filter=roughness,
            )
            if combined is None:
                combined = go.Figure(layout=figure.layout)
            for index, trace in enumerate(figure.data):
                trace.line.update(color=color, width=5, dash="solid")
                trace.marker.update(
                    color=color, size=9, symbol=symbol, maxdisplayed=20,
                    line={"color": "#000000", "width": 1},
                )
                trace.mode = "lines+markers"
                meta = dict(trace.meta) if isinstance(trace.meta, dict) else {}
                match = re.match(r"^(\d{1,3})(?=\D|$)", str(trace.name or ""))
                if match:
                    meta["ipw3_participant_id"] = match.group(1).zfill(3)
                meta["ipw3_roughness_key"] = roughness
                trace.meta = meta
                trace.legendgroup = f"htc_roughness_{roughness}"
                trace.legendrank = rank
                trace.name = label
                trace.showlegend = index == 0
                combined.add_trace(trace)
        if combined is None or not combined.data:
            raise RuntimeError(f"No M6 HTC roughness data for slice {slice_value}")
        combined.update_layout(legend_traceorder="normal")
        slug = str(slice_value).replace(".", "p")
        path = case_dir / f"tc_oneram6_L1_htc_vs_s_slice_{slug}_all_roughness.png"
        cutdata_builder.PNG_EXPORT_QUEUE.append((combined, path))


def _queue_turbulence_roughness_htc(case_dir: Path) -> None:
    """Make per-roughness and all-roughness M6 HTC model comparisons."""
    roughness_slugs = {
        "0.5mm": "0p5mm",
        "1mm": "1mm",
        "1.5mm": "1p5mm",
        "variable_roughness": "variable",
    }
    for slice_value in CASE_SLICES[CASE_ID]:
        slug = str(slice_value).replace(".", "p")
        source_name = f"tc_oneram6_L1_htc_vs_s_slice_{slug}_all_roughness.png"
        source = next((
            figure for figure, path in cutdata_builder.PNG_EXPORT_QUEUE
            if path.name == source_name
        ), None)
        if source is None:
            raise RuntimeError(f"Missing M6 HTC source plot: {source_name}")
        all_grouped = type(source)(layout=source.layout)
        all_grouped_traces = []
        shown_combinations = set()
        for roughness_key, roughness_slug in roughness_slugs.items():
            grouped = type(source)(source)
            grouped_traces = []
            shown_groups = set()
            for trace in grouped.data:
                participant_id = get_trace_participant_id(trace)
                meta = dict(trace.meta) if isinstance(trace.meta, dict) else {}
                if (
                    participant_id == "015"
                    or meta.get("ipw3_roughness_key") != roughness_key
                ):
                    continue
                model_key = next((
                    key for key, style in ONERAM6_TURBULENCE_MODEL_STYLES.items()
                    if participant_id in style["participant_ids"]
                ), None)
                if model_key is None:
                    continue
                model_style = ONERAM6_TURBULENCE_MODEL_STYLES[model_key]
                color = ONERAM6_TURBULENCE_ROUGHNESS_COLORS[model_key][roughness_key]
                group = f"htc_turbulence_model_{model_key}"
                meta["ipw3_participant_id"] = participant_id
                trace.meta = meta
                trace.line.update(
                    color=color, width=5, dash="solid"
                )
                trace.marker.update(
                    color=color, maxdisplayed=20,
                    line={"color": "#000000", "width": 1},
                )
                trace.mode = "lines+markers"
                trace.name = model_style["label"]
                trace.legendgroup = group
                trace.legendrank = model_style["rank"]
                trace.showlegend = group not in shown_groups
                shown_groups.add(group)
                grouped_traces.append(trace)
            if not grouped_traces:
                raise RuntimeError(
                    f"No M6 HTC data for slice {slice_value}, roughness {roughness_key}"
                )
            grouped.data = tuple(
                sorted(grouped_traces, key=lambda trace: trace.legendrank)
            )
            output_name = (
                f"tc_oneram6_L1_htc_vs_s_slice_{slug}_roughness_{roughness_slug}"
                "_grouped_turbulence_models.png"
            )
            cutdata_builder.PNG_EXPORT_QUEUE.append(
                (grouped, case_dir / output_name)
            )
            roughness_label = HTC_ROUGHNESS_STYLES[roughness_key][0]
            roughness_rank = tuple(roughness_slugs).index(roughness_key)
            for source_trace in source.data:
                trace = type(source_trace)(source_trace)
                participant_id = get_trace_participant_id(trace)
                meta = dict(trace.meta) if isinstance(trace.meta, dict) else {}
                if (
                    participant_id == "015"
                    or meta.get("ipw3_roughness_key") != roughness_key
                ):
                    continue
                model_key = next((
                    key for key, style in ONERAM6_TURBULENCE_MODEL_STYLES.items()
                    if participant_id in style["participant_ids"]
                ), None)
                if model_key is None:
                    continue
                model_style = ONERAM6_TURBULENCE_MODEL_STYLES[model_key]
                color = ONERAM6_TURBULENCE_ROUGHNESS_COLORS[model_key][roughness_key]
                group = f"htc_turbulence_model_{model_key}_{roughness_key}"
                meta["ipw3_participant_id"] = participant_id
                trace.meta = meta
                trace.line.update(color=color, width=5, dash="solid")
                trace.marker.update(
                    color=color, maxdisplayed=20,
                    line={"color": "#000000", "width": 1},
                )
                trace.mode = "lines+markers"
                trace.name = f'{model_style["label"]} - {roughness_label}'
                trace.legendgroup = group
                trace.legendrank = int(model_style["rank"]) * len(roughness_slugs) + roughness_rank
                trace.showlegend = group not in shown_combinations
                shown_combinations.add(group)
                all_grouped_traces.append(trace)
        if not all_grouped_traces:
            raise RuntimeError(f"No combined M6 HTC turbulence-model data for slice {slice_value}")
        all_grouped.add_traces(sorted(
            all_grouped_traces, key=lambda trace: trace.legendrank,
        ))
        combined_name = (
            f"tc_oneram6_L1_htc_vs_s_slice_{slug}_"
            "grouped_turbulence_models.png"
        )
        cutdata_builder.PNG_EXPORT_QUEUE.append((all_grouped, case_dir / combined_name))


def _queue_surface_fields_by_roughness(participants, case_dir: Path) -> None:
    """Queue M6 Ts and FF distributions colored by roughness at every slice."""
    import plotly.graph_objects as go

    plot_keys = ("surface_temperature_vs_s", "freezing_fraction_vs_s")
    specs = {
        key: next(item for item in cutdata_builder.CUTDATA_PLOTS if item["plot_key"] == key)
        for key in plot_keys
    }
    for slice_value in CASE_SLICES[CASE_ID]:
        slice_slug = str(slice_value).replace(".", "p")
        for plot_key, spec in specs.items():
            combined = None
            model_combined = None
            shown_model_roughness = set()
            for rank, (roughness, (label, color, _)) in enumerate(HTC_ROUGHNESS_STYLES.items()):
                figure, _, _, _ = cutdata_builder.build_cutdata_figure(
                    participants, CASE_ID, "L1", spec,
                    slice_filter=slice_value, roughness_filter=roughness,
                )
                if combined is None:
                    combined = go.Figure(layout=figure.layout)
                    model_combined = go.Figure(layout=figure.layout)
                for trace in figure.data:
                    participant_match = re.match(r"^(\d{1,3})(?=\D|$)", str(trace.name or ""))
                    if participant_match is None:
                        continue
                    participant_id = participant_match.group(1).zfill(3)
                    meta = dict(trace.meta) if isinstance(trace.meta, dict) else {}
                    meta.update(
                        ipw3_participant_id=participant_id,
                        ipw3_roughness_key=roughness,
                    )
                    trace.meta = meta
                    trace.line.update(color=color, width=5, dash="solid")
                    trace.marker.update(
                        color=color, size=GROUPED_ROUGHNESS_PLOT_SYMBOL_SIZE,
                        line={"color": "#000000", "width": 1},
                    )
                    if plot_key == "freezing_fraction_vs_s":
                        trace.marker.maxdisplayed = 12
                    trace.mode = "lines+markers"
                    trace.name = label
                    trace.legendgroup = f"{plot_key}_roughness_{roughness}"
                    trace.legendrank = rank
                    combined.add_trace(trace)
                    if participant_id == "015":
                        continue
                    model_key = next((
                        key for key, style in ONERAM6_TURBULENCE_MODEL_STYLES.items()
                        if participant_id in style["participant_ids"]
                    ), None)
                    if model_key is None:
                        continue
                    model_trace = type(trace)(trace)
                    model_style = ONERAM6_TURBULENCE_MODEL_STYLES[model_key]
                    group = f"{plot_key}_turbulence_model_{model_key}_{roughness}"
                    model_color = ONERAM6_TURBULENCE_ROUGHNESS_COLORS[model_key][roughness]
                    model_trace.line.update(
                        color=model_color, width=5, dash="solid",
                    )
                    model_trace.marker.update(
                        color=model_color,
                        size=GROUPED_ROUGHNESS_PLOT_SYMBOL_SIZE,
                        line={"color": "#000000", "width": 1},
                    )
                    model_trace.name = f'{model_style["label"]} - {label}'
                    model_trace.legendgroup = group
                    model_trace.legendrank = int(model_style["rank"]) * len(HTC_ROUGHNESS_STYLES) + rank
                    model_trace.showlegend = group not in shown_model_roughness
                    shown_model_roughness.add(group)
                    model_combined.add_trace(model_trace)
            if combined is None or not combined.data:
                raise RuntimeError(f"No M6 {plot_key} roughness data for slice {slice_value}")
            if plot_key == "freezing_fraction_vs_s":
                combined.update_xaxes(
                    range=[-0.125, 0.125],
                    autorange=False,
                )
            combined.update_layout(legend_traceorder="normal")
            filename = f"tc_oneram6_L1_{spec['filename_slug']}_slice_{slice_slug}_grouped_roughness.png"
            cutdata_builder.PNG_EXPORT_QUEUE.append((combined, case_dir / filename))
            if model_combined is None or not model_combined.data:
                raise RuntimeError(
                    f"No M6 {plot_key} turbulence-model data for slice {slice_value}"
                )
            model_combined.update_layout(legend_traceorder="normal")
            model_name = (
                f"tc_oneram6_L1_{spec['filename_slug']}_slice_{slice_slug}_"
                "grouped_turbulence_models.png"
            )
            cutdata_builder.PNG_EXPORT_QUEUE.append((model_combined, case_dir / model_name))


def _queue_ice_shapes_by_roughness(participants, case_dir: Path) -> None:
    """Queue 7-bin roughness and turbulence-model M6 ice shapes."""
    import plotly.graph_objects as go

    styles = HTC_ROUGHNESS_STYLES
    roughness_slugs = {
        "0.5mm": "0p5mm", "1mm": "1mm", "1.5mm": "1p5mm",
        "variable_roughness": "variable",
    }
    for slice_value in CASE_SLICES[CASE_ID]:
        combined = None
        model_combined = None
        shown_combinations = set()
        slug = str(slice_value).replace(".", "p")
        for rank, (roughness, (label, color, _)) in enumerate(styles.items()):
            source, _, _ = iceshape_builder.build_single_layer_ice_shape_figure(
                participants, CASE_ID, "L1", slice_filter=slice_value,
                bins_filter="BINS07", roughness_filter=roughness,
            )
            model_source, _, _ = iceshape_builder.build_single_layer_ice_shape_figure(
                participants, CASE_ID, "L1", slice_filter=slice_value,
                bins_filter="BINS07", roughness_filter=roughness,
            )
            if combined is None:
                combined = go.Figure(layout=source.layout)
                model_combined = go.Figure(layout=model_source.layout)
                for trace in source.data:
                    if trace.legendgroup == "clean_reference":
                        trace.showlegend = False
                        combined.add_trace(trace)
                for trace in model_source.data:
                    if trace.legendgroup == "clean_reference":
                        reference = go.Scatter(trace.to_plotly_json())
                        reference.showlegend = False
                        model_combined.add_trace(reference)
            per_roughness = go.Figure(layout=model_source.layout)
            shown_models = set()
            for trace in model_source.data:
                if trace.legendgroup == "clean_reference":
                    reference = go.Scatter(trace.to_plotly_json())
                    reference.showlegend = False
                    per_roughness.add_trace(reference)
            for trace in source.data:
                meta = dict(trace.meta) if isinstance(trace.meta, dict) else {}
                participant_value = meta.get("ipw3_participant_id")
                if not participant_value:
                    continue
                participant_id = str(participant_value).zfill(3)
                meta["ipw3_roughness_key"] = roughness
                trace.meta = meta
                trace.name = label
                trace.legendgroup = f"ice_roughness_{roughness}"
                trace.legendrank = rank
                trace.line.update(color=color, width=5, dash="solid")
                trace.marker.update(color=color, size=9, maxdisplayed=20)
                trace.mode = "lines+markers"
                combined.add_trace(trace)
            for trace in model_source.data:
                meta = dict(trace.meta) if isinstance(trace.meta, dict) else {}
                participant_value = meta.get("ipw3_participant_id")
                if not participant_value:
                    continue
                participant_id = str(participant_value).zfill(3)
                meta["ipw3_roughness_key"] = roughness
                trace.meta = meta
                if participant_id == "015":
                    continue
                model_key = next((
                    key for key, style in ONERAM6_TURBULENCE_MODEL_STYLES.items()
                    if participant_id in style["participant_ids"]
                ), None)
                if model_key is None:
                    continue
                model_style = ONERAM6_TURBULENCE_MODEL_STYLES[model_key]
                model_color = ONERAM6_TURBULENCE_ROUGHNESS_COLORS[model_key][roughness]
                for target, group, name, show_set in (
                    (
                        per_roughness,
                        f"ice_turbulence_model_{model_key}",
                        model_style["label"],
                        shown_models,
                    ),
                    (
                        model_combined,
                        f"ice_turbulence_model_{model_key}_{roughness}",
                        f'{model_style["label"]} - {label}',
                        shown_combinations,
                    ),
                ):
                    model_trace = go.Scatter(trace.to_plotly_json())
                    model_trace.line.update(color=model_color, width=5, dash="solid")
                    model_trace.marker.update(color=model_color, size=9, maxdisplayed=20)
                    model_trace.name = name
                    model_trace.legendgroup = group
                    model_trace.legendrank = int(model_style["rank"]) * len(styles) + rank
                    model_trace.showlegend = group not in show_set
                    show_set.add(group)
                    target.add_trace(model_trace)
            if len(per_roughness.data) > 1:
                per_name = (
                    f"tc_oneram6_L1_single_layer_ice_shape_slice_{slug}_bins07_"
                    f"roughness_{roughness_slugs[roughness]}_grouped_turbulence_models.png"
                )
                iceshape_builder.PNG_EXPORT_QUEUE.append(
                    (per_roughness, case_dir / per_name)
                )
        roughness_by_participant = {}
        for trace in combined.data:
            meta = trace.meta if isinstance(trace.meta, dict) else {}
            pid = meta.get("ipw3_participant_id")
            if pid:
                roughness_by_participant.setdefault(pid, set()).add(meta["ipw3_roughness_key"])
        eligible = {pid for pid, settings in roughness_by_participant.items() if len(settings) > 1}
        combined.data = tuple(
            trace for trace in combined.data
            if trace.legendgroup == "clean_reference"
            or (isinstance(trace.meta, dict) and trace.meta.get("ipw3_participant_id") in eligible)
        )
        # The initial layout is inherited from the first roughness figure, whose
        # fixed limits may clip contours added from later roughness settings.
        # Recompute shared leading-edge limits from every retained contour so
        # the combined plot and its participant-panel matrix fit all shapes.
        axis = iceshape_builder.ice_shape_axis_config(CASE_ID, slice_value)
        x_range, y_range = iceshape_builder.leading_edge_axis_ranges(
            combined, axis["leading_edge_fraction"]
        )
        if x_range is not None and axis.get("x_min") is not None:
            x_range[0] = axis["x_min"]
        combined.update_xaxes(range=axis["x_range"] or x_range)
        combined.update_yaxes(range=axis["y_range"] or y_range)
        filename = f"tc_oneram6_L1_single_layer_ice_shape_slice_{slug}_bins07_grouped_roughness.png"
        iceshape_builder.PNG_EXPORT_QUEUE.append((combined, case_dir / filename))
        if model_combined is None or len(model_combined.data) <= 1:
            raise RuntimeError(f"No M6 turbulence-model ice shapes for slice {slice_value}")
        model_x_range, model_y_range = iceshape_builder.leading_edge_axis_ranges(
            model_combined, axis["leading_edge_fraction"]
        )
        if model_x_range is not None and axis.get("x_min") is not None:
            model_x_range[0] = axis["x_min"]
        model_combined.update_xaxes(range=axis["x_range"] or model_x_range)
        model_combined.update_yaxes(range=axis["y_range"] or model_y_range)
        model_name = (
            f"tc_oneram6_L1_single_layer_ice_shape_slice_{slug}_bins07_"
            "grouped_turbulence_models.png"
        )
        iceshape_builder.PNG_EXPORT_QUEUE.append((model_combined, case_dir / model_name))


def _queue_horn_comparisons(participants, case_dir: Path) -> None:
    """Horn convergence from the same single-layer contours as the comparisons."""
    import plotly.graph_objects as go

    styles = {**HTC_ROUGHNESS_STYLES, "variable_roughness": ("Variable", "#4C78A8", "triangle-up")}
    for slice_value in CASE_SLICES[CASE_ID]:
        for grouped in (False, True):
            series = {}
            for level in ("L1", "L2", "L3", "L4"):
                source, _, _ = iceshape_builder.build_single_layer_ice_shape_figure(
                    participants, CASE_ID, level, slice_filter=slice_value,
                    bins_filter="BINS07" if grouped else "BINS15",
                    roughness_filter=None if grouped else "1mm",
                )
                if not grouped:
                    cira, _, _ = iceshape_builder.build_single_layer_ice_shape_figure(
                        [p for p in participants if str(p.participant_id).zfill(3) == "001"],
                        CASE_ID, level, slice_filter=slice_value, bins_filter="BINS01", roughness_filter="1mm",
                    )
                    existing = {t.meta.get("ipw3_participant_id") for t in source.data if isinstance(t.meta, dict)}
                    if "001" not in existing:
                        source.add_traces([t for t in cira.data if isinstance(t.meta, dict) and t.meta.get("ipw3_participant_id") == "001"])
                for trace in source.data:
                    meta = trace.meta if isinstance(trace.meta, dict) else {}
                    pid = meta.get("ipw3_participant_id")
                    if not pid:
                        continue
                    zone = re.search(r"Zone: ([^<]+)", str(trace.hovertemplate or ""))
                    if zone is None:
                        continue
                    roughness = iceshape_builder.extract_roughness_key_from_zone_name(zone.group(1))
                    if roughness not in styles:
                        continue
                    geometry = iceshape_builder.upper_horn_geometry(trace.x, trace.y, CASE_ID, slice_value)
                    if geometry is None:
                        continue
                    key = (pid, roughness, str(trace.legendgroup), meta.get("ipw3_density_model", "standard"))
                    series.setdefault(key, {})[level] = geometry[2]
            roughness_sets = {}
            for pid, roughness, _, _ in series:
                roughness_sets.setdefault(pid, set()).add(roughness)
            figure = go.Figure()
            for (pid, roughness, dataset, density), values in sorted(series.items()):
                if grouped and len(roughness_sets[pid]) < 2:
                    continue
                levels = sorted(values)
                label, color, _ = styles[roughness]
                name = label if grouped else ("001 (1 bin)" if pid == "001" else dataset)
                figure.add_trace(go.Scatter(
                    x=levels, y=[values[level] for level in levels],
                    customdata=[[level] for level in levels], name=name,
                    mode="lines+markers", legendgroup=f"horn_roughness_{roughness}" if grouped else name,
                    meta={"ipw3_participant_id": pid, "ipw3_roughness_key": roughness},
                    line=dict(color=color if grouped else iceshape_builder.participant_color(pid), width=5),
                    marker=dict(size=9),
                    hovertemplate="%{fullData.name}<br>Grid=%{customdata[0]}<br>α=%{y:.4g}°<extra></extra>",
                ))
            if not figure.data:
                raise RuntimeError(f"No M6 horn comparison data at Y={slice_value}, grouped={grouped}")
            figure.update_layout(xaxis_title="N<sub>cells</sub><sup>−1/3</sup> [-]", yaxis_title="α<sub>upper, horn</sub> [deg]")
            slug = str(slice_value).replace(".", "p")
            selection = "bins07_grouped_roughness" if grouped else "bins15_roughness_1mm"
            path = case_dir / f"tc_oneram6_upper_horn_angle_vs_n_slice_{slug}_{selection}.png"
            convergence_data_builder.PNG_EXPORT_QUEUE.append((figure, path))
            relative = go.Figure(figure)
            convergence_data_builder.normalize_grid_convergence_to_l1(relative)
            relative.update_yaxes(title_text="Δα<sub>upper, horn</sub> / α<sub>upper, horn,L1</sub> [%]")
            convergence_data_builder.PNG_EXPORT_QUEUE.append((relative, path.with_name(path.stem + "_relative_to_l1.png")))


def _queue_horn_bin_comparisons(participants, case_dir: Path) -> None:
    """L1 single-layer horn-angle sensitivity to bin count at fixed 1 mm roughness."""
    import plotly.graph_objects as go

    for slice_value in CASE_SLICES[CASE_ID]:
        series = {}
        for bins in ("BINS01", "BINS03", "BINS07", "BINS15"):
            source, _, _ = iceshape_builder.build_single_layer_ice_shape_figure(
                participants, CASE_ID, "L1", slice_filter=slice_value,
                bins_filter=bins, roughness_filter="1mm",
            )
            for trace in source.data:
                meta = trace.meta if isinstance(trace.meta, dict) else {}
                pid = meta.get("ipw3_participant_id")
                if not pid:
                    continue
                geometry = iceshape_builder.upper_horn_geometry(trace.x, trace.y, CASE_ID, slice_value)
                if geometry is None:
                    continue
                key = (pid, str(trace.legendgroup), meta.get("ipw3_density_model", "standard"))
                series.setdefault(key, {})[bins] = geometry[2]
        figure = go.Figure()
        for (pid, dataset, density), values in sorted(series.items()):
            bins = sorted(values, reverse=True)
            label = dataset + (" | ρ<sub>ice</sub>(s)" if density == "variable" else "")
            figure.add_trace(go.Scatter(
                x=bins, y=[values[b] for b in bins], customdata=[[b] for b in bins],
                name=label, legendgroup=label, meta={"ipw3_participant_id": pid},
                mode="lines+markers", line=dict(color=iceshape_builder.participant_color(pid)),
                marker=iceshape_builder.participant_marker(pid),
                hovertemplate="%{fullData.name}<br>%{customdata[0]}<br>α=%{y:.4g}°<extra></extra>",
            ))
        if not figure.data:
            raise RuntimeError(f"No M6 L1 horn bin-sensitivity data at Y={slice_value}")
        figure.update_layout(yaxis_title="α<sub>upper, horn</sub> [deg]")
        slug = str(slice_value).replace(".", "p")
        path = case_dir / f"tc_oneram6_upper_horn_angle_distribution_convergence_l1_slice_{slug}_roughness_1mm.png"
        convergence_data_builder.PNG_EXPORT_QUEUE.append((figure, path))
        relative = go.Figure(figure)
        convergence_data_builder.normalize_distribution_to_bins15(relative)
        relative.update_yaxes(title_text="Δα<sub>upper, horn</sub> / α<sub>upper, horn,15 bins</sub> [%]")
        convergence_data_builder.PNG_EXPORT_QUEUE.append((relative, path.with_name(path.stem + "_relative_to_bins15.png")))


def _queue_combined_multilayer_ice(participants, case_dir: Path) -> None:
    import plotly.graph_objects as go

    for slice_value in CASE_SLICES[CASE_ID]:
        combined = None
        reference_groups = set()
        shape_count = 0
        selected_levels: dict[str, str] = {}
        cell_counts = convergence_data_builder.grid_cell_counts_for_case(CASE_ID)
        levels = sorted(cell_counts, key=cell_counts.get, reverse=True)
        for level_number in levels:
            level = f"L{level_number}"
            figure, count, _ = iceshape_builder.build_multilayer_ice_shape_figure(
                participants, CASE_ID, level, slice_filter=slice_value,
                bins_filter=None, roughness_filter=None,
            )
            if combined is None:
                combined = go.Figure(layout=figure.layout)
            shape_count += count
            for trace in figure.data:
                group = str(trace.legendgroup or "")
                if group == "clean_reference" or group.startswith(("experimental_", "reference_")):
                    if group in reference_groups:
                        continue
                    reference_groups.add(group)
                else:
                    participant = re.match(r"^(\d{1,3})(?=\D|$)", str(trace.name or ""))
                    if participant:
                        participant_id = participant.group(1).zfill(3)
                        selected_levels.setdefault(participant_id, level)
                        if selected_levels[participant_id] != level:
                            continue
                    details = [str(trace.name), level]
                    for field in ("Distribution", "Roughness"):
                        match = re.search(rf"{field}: ([^<]+)", str(trace.hovertemplate or ""))
                        if match:
                            details.append(match.group(1))
                    trace.name = " | ".join(details)
                    trace.legendgroup = f"{group}_{level}"
                    if trace.line is not None:
                        trace.line.dash = iceshape_builder.ice_shape_grid_line_dash(
                            CASE_ID, participant.group(1) if participant else "", level,
                        )
                combined.add_trace(trace)
        if combined is None or shape_count == 0:
            raise RuntimeError(f"No M6 multilayer ice shapes for slice {slice_value}")
        # At the selected finest grid, retain the largest bin count per participant.
        largest_bins: dict[str, int] = {}
        trace_bins = {}
        for index, trace in enumerate(combined.data):
            participant = re.match(r"^(\d{1,3})(?=\D|$)", str(trace.name or ""))
            distribution = re.search(r"Distribution: (\d+)", str(trace.hovertemplate or ""))
            if participant and distribution:
                participant_id = participant.group(1).zfill(3)
                bins = int(distribution.group(1))
                trace_bins[index] = (participant_id, bins)
                largest_bins[participant_id] = max(largest_bins.get(participant_id, 0), bins)
        combined.data = tuple(
            trace for index, trace in enumerate(combined.data)
            if index not in trace_bins or trace_bins[index][1] == largest_bins[trace_bins[index][0]]
        )
        # Keep the clean wing outline on top of the submitted contours.
        combined.data = tuple(t for t in combined.data if t.legendgroup != "clean_reference") + tuple(
            t for t in combined.data if t.legendgroup == "clean_reference"
        )
        axis = iceshape_builder.ice_shape_axis_config(CASE_ID, slice_value)
        x_range, y_range = iceshape_builder.leading_edge_axis_ranges(combined, axis["leading_edge_fraction"])
        combined.update_xaxes(range=axis["x_range"] or x_range)
        combined.update_yaxes(range=axis["y_range"] or y_range)
        slug = str(slice_value).replace(".", "p")
        path = case_dir / f"tc_oneram6_finest_grid_multilayer_ice_shape_slice_{slug}_all_conditions.png"
        iceshape_builder.PNG_EXPORT_QUEUE.append((combined, path))


def _queue_champs_multilayer_vs_grey_single_layer(participants, case_dir: Path) -> None:
    """Overlay CHAMPS multilayer data on other participants' grey single-layer shapes."""
    import plotly.graph_objects as go
    from .gatherParticipantData import read_tecplot_dat

    polimo = next(
        participant for participant in participants
        if str(participant.participant_id).zfill(3) == "007"
    )
    multilayer_path = (
        polimo.path / "TC_ONERAM6_D01"
        / "TC_ONERAM6_L1_finalIceShape_MULTILAYER_V1.dat"
    )
    multilayer_data = read_tecplot_dat(multilayer_path, process_cutdata=False)

    for slice_value in CASE_SLICES[CASE_ID]:
        single, single_count, _ = iceshape_builder.build_single_layer_ice_shape_figure(
            participants, CASE_ID, "L1", slice_filter=slice_value,
            bins_filter="BINS07", roughness_filter="1mm",
        )
        if single_count == 0:
            raise RuntimeError(f"Missing M6 single-layer shapes at Y={slice_value:g} m")

        combined = go.Figure(layout=single.layout)
        for trace in single.data[:single_count]:
            if get_trace_participant_id(trace) == "007":
                continue
            participant_name = str(trace.name or "")
            trace.name = f"{participant_name} | Single layer"
            trace.legendgroup = f"single_layer_{trace.legendgroup}"
            trace.line.color = "#9E9E9E"
            trace.line.width = 3
            if trace.marker is not None:
                trace.marker.color = "#9E9E9E"
            trace.showlegend = True
            trace.zorder = 1
            combined.add_trace(trace)

        slug = str(slice_value).replace(".", "p")
        zone_name = f"SLICE_Y_{slug}_BINS07_KS_1mm_FINAL_LAYER"
        if zone_name not in multilayer_data.zones:
            raise RuntimeError(f"Missing {zone_name} in {multilayer_path}")
        zone = multilayer_data.zones[zone_name]
        x_column = next((column for column in zone.data if column.upper() == "X_ICED"), None)
        z_column = next((column for column in zone.data if column.upper() == "Z_ICED"), None)
        if x_column is None or z_column is None:
            raise RuntimeError(f"Missing iced coordinates in {multilayer_path}")
        valid = (
            numpy.isfinite(zone.data[x_column]) & numpy.isfinite(zone.data[z_column])
            & (zone.data[x_column] > -998.0) & (zone.data[z_column] > -998.0)
        )
        combined.add_trace(go.Scatter(
            x=zone.data.loc[valid, x_column], y=zone.data.loc[valid, z_column],
            mode="lines", name="007 | CHAMPS multilayer",
            legendgroup="007_champs_multilayer", legendrank=0, zorder=2,
            line={"color": "#1f77b4", "width": 5, "dash": "solid"},
            meta={"ipw3_participant_id": "007", "ipw3_grid_level": "L1"},
        ))

        # Reuse the single-layer figure's reference geometry once and keep it on top.
        for trace in single.data[single_count:]:
            trace.zorder = 3
            combined.add_trace(trace)

        combined.update_layout(
            title={
                "text": (
                    "CHAMPS multi-layer final ice shape vs single-layer participants | "
                    f"L1 | Y = {slice_value:g} m | 07 | Roughness height = 1 mm"
                ),
                "x": 0.5,
                "xanchor": "center",
            },
            width=WIDTH,
            height=HEIGHT,
        )
        filename = (
            "tc_oneram6_L1_champs_multilayer_vs_single_layer_participants_grey_"
            f"slice_{slug}_bins07_roughness_1mm.png"
        )
        iceshape_builder.PNG_EXPORT_QUEUE.append((combined, case_dir / filename))


def _queue_qc_roughness(participants, case_dir: Path) -> None:
    import plotly.graph_objects as go

    for slice_value in CASE_SLICES[CASE_ID]:
        combined = None
        for rank, (roughness, (label, color, symbol)) in enumerate(HTC_ROUGHNESS_STYLES.items()):
            figure, _, _ = convergence_data_builder.build_qc_prime_integration_figure(
                participants, CASE_ID, slice_position=slice_value, roughness_filter=roughness,
            )
            if combined is None:
                combined = go.Figure(layout=figure.layout)
            for index, trace in enumerate(figure.data):
                trace.line.update(color=color, width=5, dash="solid")
                trace.marker.update(color=color, size=9, symbol=symbol, maxdisplayed=20,
                                    line={"color": "#000000", "width": 1})
                trace.mode = "lines+markers"
                trace.name = label
                trace.legendgroup = f"qc_roughness_{roughness}"
                trace.legendrank = rank
                trace.showlegend = index == 0
                combined.add_trace(trace)
        if combined is None or not combined.data:
            raise RuntimeError(f"No M6 Qc roughness data for slice {slice_value}")
        combined.update_layout(legend_traceorder="normal")
        path = case_dir / f"tc_oneram6_qc_prime_vs_n_y_{slice_value}_grouped_roughness.png"
        convergence_data_builder.PNG_EXPORT_QUEUE.append((combined, path))
        relative = go.Figure(combined)
        convergence_data_builder.normalize_grid_convergence_to_l1(relative)
        convergence_data_builder.apply_individual_plot_overrides(relative, CASE_ID, "qc_prime_relative")
        relative_path = path.with_name(path.stem + "_relative_to_l1.png")
        convergence_data_builder.PNG_EXPORT_QUEUE.append((relative, relative_path))


def _queue_mean_roughness_variants() -> None:
    import plotly.graph_objects as go

    for source, path in tuple(convergence_data_builder.PNG_EXPORT_QUEUE):
        if not path.name.startswith("tc_oneram6_mean_") or not (path.name.endswith("_all_roughness.png") or path.name.endswith("_all_roughness_relative_to_l1.png")):
            continue
        one_mm = go.Figure(source)
        one_mm.data = tuple(
            trace for trace in one_mm.data
            if isinstance(trace.meta, dict) and trace.meta.get("ipw3_roughness_key") == "1mm"
        )
        for trace in one_mm.data:
            trace.name = str(trace.meta["ipw3_participant_id"]).zfill(3)
        grouped = go.Figure(layout=source.layout)
        for rank, (roughness, (label, color, symbol)) in enumerate(HTC_ROUGHNESS_STYLES.items()):
            shown = False
            for original in source.data:
                if not isinstance(original.meta, dict) or original.meta.get("ipw3_roughness_key") != roughness:
                    continue
                trace = go.Scatter(original)
                trace.line.update(color=color, width=5, dash="solid")
                trace.marker.update(color=color, size=9, symbol=symbol, maxdisplayed=20,
                                    line={"color": "#000000", "width": 1})
                trace.mode = "lines+markers"
                trace.name = label
                trace.legendgroup = f"mean_roughness_{roughness}"
                trace.legendrank = rank
                trace.showlegend = not shown
                shown = True
                grouped.add_trace(trace)
        grouped.update_layout(legend_traceorder="normal")
        if not one_mm.data or not grouped.data:
            raise RuntimeError(f"Missing mean roughness data for {path.name}")
        for figure, suffix in ((one_mm, "roughness_1mm"), (grouped, "grouped_roughness")):
            destination = path.with_name(path.name.replace("all_roughness", suffix))
            convergence_data_builder.PNG_EXPORT_QUEUE.append((figure, destination))


def _add_surface_temperature_zoom(figure) -> None:
    import plotly.graph_objects as go

    x_range = [-0.1, 0.1]
    sources = [trace for trace in figure.data if trace.x is not None and trace.y is not None
               and trace.xaxis in (None, "x")]
    temperatures = [float(value) for trace in sources for value in trace.y
                    if value is not None and math.isfinite(float(value))]
    if not temperatures or max(temperatures) <= 270:
        return
    maximum = max(temperatures)
    for source in sources:
        trace = go.Scatter(source.to_plotly_json())
        trace.update(xaxis="x2", yaxis="y2", showlegend=False, hoverinfo="skip")
        figure.add_trace(trace)
    axis_style = dict(ticks="outside", tickfont={"size": 20}, showline=True,
                      linecolor="black", linewidth=2, mirror=True,
                      showgrid=True, gridcolor="#dddddd", zeroline=False)
    figure.update_layout(
        xaxis2={**axis_style, "domain": [0.65, 0.96], "anchor": "y2",
                "range": list(x_range), "autorange": False},
        yaxis2={**axis_style, "domain": [0.08, 0.48], "anchor": "x2",
                "range": [270, maximum], "autorange": False},
    )
    figure.add_shape(type="rect", xref="x2 domain", yref="y2 domain",
                     x0=0, x1=1, y0=0, y1=1, fillcolor="white",
                     line={"width": 0}, layer="below")


def _style_figure(figure, module, export_path: Path, spec: FigureSpec) -> None:
    _, _, width, height, excluded_ids, show_legend = spec[:6]
    median_line = spec[6] if len(spec) > 6 else None
    name = export_path.stem.lower()
    if re.fullmatch(r"tc_oneram6_(cl|cd|cmy)_vs_n_grouped_roughness_participants", name):
        return  # The subplot matrix already carries its own axis and legend layout.
    is_convergence = module is convergence_data_builder and (
        "_vs_n" in name or any(f"_{metric}_bins" in name for metric in ("beta_max", "s_peak", "width"))
        or "_ice_width_bins15_slice_" in name
    )
    is_bin_convergence = "_vs_inverse_bins" in name or "_distribution_convergence_" in name

    excluded = {str(value).zfill(3) for value in excluded_ids}
    if "freezing_fraction" in name:
        excluded.add("015")
    if excluded:
        figure.data = tuple(
            trace for trace in figure.data
            if not (
                isinstance(trace.meta, dict)
                and str(trace.meta.get("ipw3_participant_id", "")).zfill(3) in excluded
            )
            and not (
                not (isinstance(trace.meta, dict) and trace.meta.get("ipw3_participant_id"))
                and (match := re.match(r"^(\d{1,3})(?=\D|$)", str(trace.name or "").strip()))
                and match.group(1).zfill(3) in excluded
            )
        )

    # Reassign group legend entries after exclusions, including when the
    # removed participant supplied the original visible legend entry.
    if "_grouped_roughness" in name or "_grouped_turbulence_models" in name:
        shown_groups = set()
        for trace in figure.data:
            trace.showlegend = trace.legendgroup not in shown_groups
            shown_groups.add(trace.legendgroup)

    is_diameter_statistics = "_vs_droplet_diameter_" in name
    if is_diameter_statistics:
        figure.update_xaxes(range=[5.0, None], autorange="max", rangemode="normal")
    is_roughness_htc = (
        ("_htc_vs_s_" in name and name.endswith("_all_roughness"))
        or "_grouped_roughness" in name
        or "_grouped_turbulence_models" in name
    )
    for trace in figure.data:
        if not (is_diameter_statistics or is_roughness_htc) and getattr(trace, "line", None) is not None:
            trace.line.width = 5
        if not (is_diameter_statistics or is_roughness_htc) and getattr(trace, "marker", None) is not None:
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
    if "_cp_vs_s_" in name or "_cp_vs_x_" in name:
        # Cp has only the participant row for M6, so it sits at the same
        # anchor used when grouped plots have no experimental row.
        figure.update_layout(legend={
            "y": GROUPED_EXPERIMENTAL_LEGEND_POSITION["y"],
            "bgcolor": "rgba(0,0,0,0)",
        })

    if is_bin_convergence:
        # Restore inverse-bin coordinates from the HTML categorical labels.
        for trace in figure.data:
            if trace.x is None:
                continue
            inverse_bin_x = []
            for value in trace.x:
                match = re.fullmatch(r"BINS(\d+)", str(value).strip(), re.IGNORECASE)
                inverse_bin_x.append(
                    1.0 / int(match.group(1))
                    if match and int(match.group(1)) > 0 else value
                )
            trace.x = inverse_bin_x
        figure.update_layout(margin={"l": 100, "r": 50, "t": 125 if show_legend else 30, "b": 145})
        figure.update_xaxes(
            type="log", range=[-1.25, 0.05], autorange=False,
            categoryorder=None, categoryarray=None,
            tickmode="array", dtick=None,
            tickvals=[1/15, 1/7, 1/3, 1],
            ticktext=["1/15", "1/7", "1/3", "1"],
            ticks="outside", showticklabels=True, showgrid=True,
            minor={"showgrid": True, "dtick": "D1", "ticks": "outside"},
            automargin=True,
            title_text="1 / N<sub>bins</sub> [-]<br><span style='font-size:24px'></span>",
        )

    if is_convergence and not is_bin_convergence:
        cell_counts = convergence_data_builder.grid_cell_counts_for_case(CASE_ID)

        for trace in figure.data:
            if trace.x is None:
                continue
            converted = []
            for value in trace.x:
                match = re.fullmatch(r"L(\d+)", str(value).strip(), re.IGNORECASE)
                level = int(match.group(1)) if match else None
                converted.append(cell_counts[level] ** (-1.0 / 3.0) if level in cell_counts else value)
            trace.x = converted

        figure.update_xaxes(
            type="log",
            autorange=False,
            range=[numpy.log10(2e-3), numpy.log10(3e-2)],
            tickmode="array",
            tickvals=[0.003, 0.01, 0.03],
            ticktext=["3×10<sup>−3</sup>", "10<sup>−2</sup>", "3×10<sup>−2</sup>"],
            tickformat=None,
            tick0=None,
            dtick=None,
            ticks="outside",
            showticklabels=True,
            showgrid=True,
            title_text="N<sub>cells</sub><sup>−1/3</sup> [-]",
            minor=dict(
                ticks="outside",
                showgrid=True,
            ),
        )

        figure.update_layout(margin={"l": 100, "r": 50, "t": 125, "b": 125})
        _add_configured_grid_median_line(figure, median_line)

    if is_roughness_htc:
        apply_roughness_participant_legend(figure, GROUPED_ROUGHNESS_PARTICIPANT_SYMBOLS,
                        participant_symbol_size=GROUPED_ROUGHNESS_PARTICIPANT_SYMBOL_SIZE,
                        plot_symbol_size=GROUPED_ROUGHNESS_PLOT_SYMBOL_SIZE,
                        roughness_position={"x": GROUPED_ROUGHNESS_LEGEND_X},
                        participant_position={"x": GROUPED_PARTICIPANT_LEGEND_X},
                        experimental_position=GROUPED_EXPERIMENTAL_LEGEND_POSITION,
                        row_shift=GROUPED_LEGEND_ROW_SHIFT,
                        group_token=(
                            "_turbulence_model_"
                            if "_grouped_turbulence_models" in name else "_roughness_"
                        ))

    if "_upper_horn_angle_vs_n_" in name or "_upper_horn_angle_distribution_convergence_" in name:
        figure.update_xaxes(showline=True, linecolor="#555555", gridcolor="#d9d9d9",
                            minor=dict(showgrid=False, ticks=""))
        figure.update_yaxes(type="linear", showline=True, linecolor="#555555",
                            showgrid=True, gridcolor="#d9d9d9", zeroline=False)
        if "_relative_to_" not in name:
            values = [float(y) for t in figure.data if t.y is not None for y in t.y
                      if y is not None and numpy.isfinite(float(y))]
            if values:
                padding = max(2.0, 0.08 * (max(values) - min(values)))
                figure.update_yaxes(range=[min(values) - padding, max(values) + padding], autorange=False)

    if "horn" in name:
        _add_main_and_inset_borders(figure)

    if (is_roughness_htc and "_htc_vs_s_" in name) or any(
        key in name for key in ("_mean_surface_temperature_", "_mean_freezing_fraction_")
    ):
        figure.update_layout(legend={"title": {"text": ""}}, legend2={"title": {"text": ""}})

    if "_surface_temperature_vs_s_" in name:
        figure.update_yaxes(range=[270, 275], autorange=False)
    if "_freezing_fraction_vs_s_" in name:
        figure.update_yaxes(range=[-0.05, 1.05], autorange=False)

    if "_surface_temperature_vs_s_" in name:
        _add_surface_temperature_zoom(figure)


def _add_001_single_bin_ice_shapes(participants):
    """Include CIRA's identified single-bin result in the L1 15-bin comparisons."""
    selected = [p for p in participants if str(p.participant_id).zfill(3) == "001"]
    if not selected:
        return
    for figure, path in iceshape_builder.PNG_EXPORT_QUEUE:
        match = re.fullmatch(
            r"tc_oneram6_L1_single_layer_ice_shape_slice_([0-9p]+)_bins15_roughness_1mm.png",
            path.name,
        )
        if not match:
            continue
        source, _, _ = iceshape_builder.build_single_layer_ice_shape_figure(
            selected, CASE_ID, "L1", slice_filter=float(match.group(1).replace("p", ".")),
            bins_filter="BINS01", roughness_filter="1mm",
        )
        for trace in source.data:
            if not re.match(r"^001(?=\D|$)", str(trace.name or "")):
                continue
            trace.name = "001 (1 bin)"
            figure.add_trace(trace)


def _queue_roughness_participant_panels(queues):
    from tools.roughness_panels import build_roughness_participant_panels

    exports = []
    destinations = {spec[1]: Path(destination) for destination, spec in ONERAM6_PRESENTATION_FIGURES.items()}
    for module, queue in queues:
        for figure, path in tuple(queue):
            if path.stem.endswith("_participants"):
                continue
            if "_limit_" in path.stem or "ice_width" in path.stem:
                continue
            if "_upper_horn_angle_" in path.stem and "_relative_to_l1" in path.stem:
                continue
            if "_grouped_roughness" not in path.stem and not ("_htc_vs_s_" in path.stem and path.stem.endswith("_all_roughness")):
                continue
            roughness_by_participant = {}
            for trace in figure.data:
                meta = trace.meta if isinstance(trace.meta, dict) else {}
                pid = meta.get("ipw3_participant_id")
                group = str(trace.legendgroup or "")
                if pid and "_roughness_" in group:
                    roughness_by_participant.setdefault(str(pid).zfill(3), set()).add(group)
            eligible_participants = {
                pid for pid, groups in roughness_by_participant.items() if len(groups) > 1
            }
            if not eligible_participants:
                continue
            panel_source = type(figure)(figure)
            panel_source.data = tuple(
                trace for trace in panel_source.data
                if trace.legendgroup == "clean_reference"
                or (isinstance(trace.meta, dict)
                    and str(trace.meta.get("ipw3_participant_id", "")).zfill(3) in eligible_participants)
            )
            use_two_columns = any(key in path.stem for key in (
                "_htc_vs_s_",
                "_upper_horn_angle_",
                "_mean_surface_temperature_",
                "_mean_freezing_fraction_",
                "_freezing_fraction_vs_s_",
                "_qc_prime_",
            ))
            is_ice_shape = "_single_layer_ice_shape_" in path.stem
            panels = build_roughness_participant_panels(
                panel_source,
                ice_shapes=is_ice_shape,
                columns=2 if use_two_columns else None,
                participant_title_size=30 if use_two_columns or is_ice_shape else 26,
            )
            if "_single_layer_ice_shape_" in path.stem:
                for axis_name in panels.layout:
                    if not axis_name.startswith("yaxis"):
                        continue
                    axis = panels.layout[axis_name]
                    if axis.visible is False:
                        continue
                    suffix = axis_name.removeprefix("yaxis")
                    axis.update(scaleanchor=f"x{suffix}", scaleratio=1, constrain="domain")
                    panels.layout[f"xaxis{suffix}"].constrain = "domain"
            if any(key in path.stem for key in ("_htc_vs_s_", "_mean_surface_temperature_", "_mean_freezing_fraction_")):
                panels.update_layout(legend_title_text="")
            panel_path = path.with_name(path.stem + "_participants.png")
            if panel_path.name == "tc_oneram6_mean_surface_temperature_vs_n_slice_1p4_grouped_roughness_participants.png":
                panels.update_yaxes(range=[269, 275], autorange=False)
            if panel_path.name == "tc_oneram6_mean_surface_temperature_vs_n_slice_0p75_grouped_roughness_participants.png":
                panels.update_yaxes(range=[272, 275], autorange=False)
            if panel_path.name == "tc_oneram6_mean_freezing_fraction_vs_n_slice_0p1_grouped_roughness_participants.png":
                panels.update_yaxes(range=[0, 1], autorange=False)
            queue.append((panels, panel_path))
            exports.append((panel_path, destinations[path.name].with_name(panel_path.name)))
    return exports


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
        l4_layout = figures[target_name.replace("_L1_", "_L4_", 1)].layout
        l4_inset_xaxis = getattr(l4_layout, "xaxis2", None)
        l4_inset_yaxis = getattr(l4_layout, "yaxis2", None)
        if (
            "_slice_0p1_" in target_name
            and l4_inset_xaxis is not None
            and l4_inset_yaxis is not None
            and l4_inset_xaxis.range
            and l4_inset_yaxis.range
        ):
            target.layout.xaxis2.range = list(l4_inset_xaxis.range)
            target.layout.yaxis2.range = list(l4_inset_yaxis.range)
            l4_zoom_box = next((
                shape for shape in (l4_layout.shapes or ())
                if shape.type == "rect"
                and shape.xref == "x"
                and shape.yref == "y"
                and getattr(shape.line, "dash", None) == "dot"
            ), None)
            target_zoom_box = next((
                shape for shape in (target.layout.shapes or ())
                if shape.type == "rect"
                and shape.xref == "x"
                and shape.yref == "y"
                and getattr(shape.line, "dash", None) == "dot"
            ), None)
            if l4_zoom_box is not None and target_zoom_box is not None:
                target_zoom_box.x0 = l4_zoom_box.x0
                target_zoom_box.x1 = l4_zoom_box.x1
                target_zoom_box.y0 = l4_zoom_box.y0
                target_zoom_box.y1 = l4_zoom_box.y1
        target.update_xaxes(showline=True, mirror=True, linecolor="black", linewidth=2)
        target.update_yaxes(showline=True, mirror=True, linecolor="black", linewidth=2)
        _add_main_and_inset_borders(target)


def _queue_champs_beta_distributions(queue, target_names: set[str]) -> set[str]:
    """Keep BINS01 Eulerian-only targets and add L1 all-bin comparisons."""
    colors = {
        "15": "#1f77b4", "07": "#2ca02c",
        "03": "#ff7f0e", "01": "#d62728",
    }
    figures = {path.name: figure for figure, path in queue}
    paths = {path.name: path for _, path in queue}
    added_names = set()
    for target_name in target_names:
        target = figures.get(target_name)
        if target is None:
            raise RuntimeError(f"Missing CHAMPS beta presentation plot: {target_name}")
        sources = {}
        for bins in colors:
            source_name = target_name.replace("beta_bins01", f"beta_bins{bins}", 1)
            source = figures.get(source_name)
            if source is None:
                raise RuntimeError(f"Missing CHAMPS beta source: {source_name}")
            sources[bins] = [
                copy.deepcopy(trace) for trace in source.data
                if get_trace_participant_id(trace) == "007"
                and (trace.xaxis or "x") == "x"
                and (trace.yaxis or "y") == "y"
            ]
        target.data = tuple(
            trace for trace in target.data
            if get_trace_participant_id(trace) == "007"
        )
        eulerian_legend_shown = False
        for trace in target.data:
            trace.name = "007 (Eulerian)"
            trace.legendgroup = "007_eulerian"
            trace.line.color = "#1f77b4"
            trace.line.dash = "solid"
            trace.line.width = 5
            trace.mode = "lines"
            is_main_axis = (trace.xaxis or "x") == "x" and (trace.yaxis or "y") == "y"
            trace.showlegend = is_main_axis and not eulerian_legend_shown
            eulerian_legend_shown = eulerian_legend_shown or is_main_axis
        if (
            getattr(target.layout, "yaxis2", None) is not None
            and target.layout.yaxis2.range
        ):
            target.layout.yaxis2.range = [target.layout.yaxis2.range[0], 0.9]
        target.update_xaxes(showline=True, mirror=True, linecolor="black", linewidth=2)
        target.update_yaxes(showline=True, mirror=True, linecolor="black", linewidth=2)
        _add_main_and_inset_borders(target)

        distribution = copy.deepcopy(target)
        distribution.data = ()
        distribution.layout.shapes = ()
        distribution.layout.xaxis2 = None
        distribution.layout.yaxis2 = None
        zorders = {"15": 4, "07": 3, "03": 2, "01": 1}
        for bins in ("15", "07", "03", "01"):
            source_traces = sources[bins]
            if not source_traces:
                raise RuntimeError(f"Missing CHAMPS BINS{bins} beta trace for {target_name}")
            legend_shown = False
            for trace in source_traces:
                count = int(bins)
                meta = dict(trace.meta) if isinstance(trace.meta, dict) else {}
                meta["ipw3_participant_id"] = "007"
                meta["ipw3_bins"] = f"BINS{bins}"
                trace.meta = meta
                trace.name = (
                    f"{count} Bin (Eulerian)" if count == 1
                    else f"{count} Bins (Eulerian)"
                )
                trace.legendgroup = f"007_beta_eulerian_bins{bins}"
                trace.legendrank = {"15": 1, "07": 2, "03": 3, "01": 4}[bins]
                trace.zorder = zorders[bins]
                trace.line.color = colors[bins]
                trace.line.dash = "solid"
                trace.line.width = 5
                trace.mode = "lines"
                trace.showlegend = not legend_shown
                legend_shown = True
                distribution.add_trace(trace)
        cutdata_builder.add_collection_efficiency_inset(
            distribution, x_range=(-0.025, 0.025), y_range=None,
        )
        inset_y_range = distribution.layout.yaxis2.range
        if inset_y_range:
            distribution.layout.yaxis2.range = [inset_y_range[0], 0.9]
        distribution.update_xaxes(
            showline=True, mirror=True, linecolor="black", linewidth=2,
        )
        distribution.update_yaxes(
            showline=True, mirror=True, linecolor="black", linewidth=2,
        )
        _add_main_and_inset_borders(distribution)
        distribution_name = target_name.replace("beta_bins01", "beta_all_bins", 1)
        queue.append((distribution, paths[target_name].with_name(distribution_name)))
        added_names.add(distribution_name)
    return added_names


def _queue_champs_htc_grid_levels(queue, target_names: set[str]) -> set[str]:
    """Add CHAMPS-only L1-L4 HTC comparisons without changing source plots."""
    colors = {
        "L1": "#1f77b4", "L2": "#2ca02c",
        "L3": "#ff7f0e", "L4": "#d62728",
    }
    figures = {path.name: figure for figure, path in queue}
    paths = {path.name: path for _, path in queue}
    added_names = set()
    for target_name in target_names:
        target = figures.get(target_name)
        if target is None:
            raise RuntimeError(f"Missing CHAMPS HTC source plot: {target_name}")
        sources = {}
        for level in colors:
            source_name = target_name.replace("_L1_", f"_{level}_", 1)
            source = figures.get(source_name)
            if source is None:
                raise RuntimeError(f"Missing CHAMPS HTC grid source: {source_name}")
            sources[level] = [
                copy.deepcopy(trace) for trace in source.data
                if get_trace_participant_id(trace) == "007"
            ]
        comparison = copy.deepcopy(target)
        comparison.data = ()
        for level in ("L4", "L3", "L2", "L1"):
            source_traces = sources[level]
            if not source_traces:
                raise RuntimeError(f"Missing CHAMPS {level} HTC trace for {target_name}")
            legend_shown = False
            for trace in source_traces:
                meta = dict(trace.meta) if isinstance(trace.meta, dict) else {}
                meta["ipw3_participant_id"] = "007"
                meta["ipw3_grid_level"] = level
                trace.meta = meta
                trace.name = level
                trace.legendgroup = f"007_htc_{level.lower()}"
                trace.legendrank = int(level[1:])
                trace.zorder = 5 - int(level[1:])
                trace.line.color = colors[level]
                trace.line.dash = "solid"
                trace.line.width = 5
                trace.mode = "lines"
                trace.showlegend = not legend_shown
                legend_shown = True
                comparison.add_trace(trace)
        comparison.update_xaxes(
            showline=True, mirror=True, linecolor="black", linewidth=2,
        )
        comparison.update_yaxes(
            showline=True, mirror=True, linecolor="black", linewidth=2,
        )
        _add_main_and_inset_borders(comparison)
        comparison_name = target_name.replace("_L1_htc_", "_htc_", 1)
        comparison_name = comparison_name.replace(
            "_roughness_1mm.png", "_roughness_1mm_grid_levels.png",
        )
        queue.append((comparison, paths[target_name].with_name(comparison_name)))
        added_names.add(comparison_name)
    return added_names


def _queue_champs_ice_shape_distributions(queue, target_names: set[str]) -> set[str]:
    """Add CHAMPS-only L1 single-layer comparisons across bin counts."""
    colors = {
        "15": "#1f77b4", "07": "#2ca02c",
        "03": "#ff7f0e", "01": "#d62728",
    }
    figures = {path.name: figure for figure, path in queue}
    paths = {path.name: path for _, path in queue}
    added_names = set()
    for target_name in target_names:
        target = figures.get(target_name)
        if target is None:
            raise RuntimeError(f"Missing CHAMPS ice-shape source: {target_name}")
        comparison = copy.deepcopy(target)
        comparison.data = tuple(
            trace for trace in comparison.data if not get_trace_participant_id(trace)
        )
        for bins in ("15", "07", "03", "01"):
            source_name = target_name.replace("bins07", f"bins{bins}", 1)
            source = figures.get(source_name)
            if source is None:
                raise RuntimeError(f"Missing CHAMPS ice-shape source: {source_name}")
            traces = [
                copy.deepcopy(trace) for trace in source.data
                if get_trace_participant_id(trace) == "007"
            ]
            if not traces:
                raise RuntimeError(f"Missing CHAMPS BINS{bins} contour for {target_name}")
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
        slice_match = re.search(r"_slice_(\d+p\d+)_", target_name)
        if slice_match:
            slice_value = float(slice_match.group(1).replace("p", "."))
            axis = iceshape_builder.ice_shape_axis_config(CASE_ID, slice_value)
            x_range, y_range = iceshape_builder.leading_edge_axis_ranges(
                comparison, axis["leading_edge_fraction"],
            )
            comparison.update_xaxes(range=axis["x_range"] or x_range)
            comparison.update_yaxes(range=axis["y_range"] or y_range)
        _add_main_and_inset_borders(comparison)
        comparison_name = target_name.replace(
            "bins07_roughness_1mm.png", "all_bins_roughness_1mm.png",
        )
        queue.append((comparison, paths[target_name].with_name(comparison_name)))
        added_names.add(comparison_name)
    return added_names


def _queue_champs_normals_comparisons(queue, participant_path: Path) -> set[str]:
    """Add L1 BINS15 initial/updated surface-normal comparisons for each slice."""
    import plotly.graph_objects as go
    from .gatherParticipantData import read_tecplot_dat

    normals_path = (
        participant_path / "TC_ONERAM6_D01"
        / "TC_ONERAM6_L1_finalIceShape_UPDATE_NORMALS_V1.dat"
    )
    if not normals_path.exists():
        return set()
    normals_data = read_tecplot_dat(normals_path, process_cutdata=False)
    figures = {path.name: figure for figure, path in queue}
    paths = {path.name: path for _, path in queue}
    added_names = set()
    for slice_value in CASE_SLICES[CASE_ID]:
        slice_slug = str(slice_value).replace(".", "p")
        source_name = (
            f"tc_oneram6_L1_single_layer_ice_shape_slice_{slice_slug}"
            "_bins15_roughness_1mm.png"
        )
        source = figures.get(source_name)
        if source is None:
            raise RuntimeError(f"Missing CHAMPS normals-comparison source: {source_name}")
        zone_name = f"SLICE_Y_{slice_slug}_BINS15_KS_1mm_SINGLE_LAYER"
        zone = normals_data.zones.get(zone_name)
        if zone is None:
            raise RuntimeError(f"Missing {zone_name} in {normals_path}")
        x_column = next((name for name in zone.data if name.upper() == "X_ICED"), None)
        z_column = next((name for name in zone.data if name.upper() == "Z_ICED"), None)
        if x_column is None or z_column is None:
            raise RuntimeError(f"Missing iced coordinates in {zone_name}")
        valid = (
            numpy.isfinite(zone.data[x_column])
            & numpy.isfinite(zone.data[z_column])
            & (zone.data[x_column] > -998.0)
            & (zone.data[z_column] > -998.0)
        )

        comparison = copy.deepcopy(source)
        comparison.data = tuple(
            trace for trace in comparison.data
            if get_trace_participant_id(trace) in (None, "007")
        )
        for trace in comparison.data:
            if get_trace_participant_id(trace) == "007":
                trace.name = "007 (Initial Surface Normals)"
                trace.legendgroup = "007_initial_surface_normals"
                trace.line.color = "#1f77b4"
                trace.line.dash = "solid"
                trace.line.width = 5
                trace.mode = "lines"
        comparison.add_trace(go.Scatter(
            x=zone.data.loc[valid, x_column],
            y=zone.data.loc[valid, z_column],
            mode="lines",
            name="007 (Updated Surface Normals)",
            legendgroup="007_updated_surface_normals",
            line={"color": "#d62728", "dash": "solid", "width": 5},
            meta={"ipw3_participant_id": "007", "ipw3_normals_comparison": True},
        ))
        axis = iceshape_builder.ice_shape_axis_config(CASE_ID, slice_value)
        x_range, y_range = iceshape_builder.leading_edge_axis_ranges(
            comparison, axis["leading_edge_fraction"],
        )
        comparison.update_xaxes(range=axis["x_range"] or x_range)
        comparison.update_yaxes(range=axis["y_range"] or y_range)
        _add_main_and_inset_borders(comparison)
        comparison_name = (
            f"tc_oneram6_L1_single_layer_ice_shape_slice_{slice_slug}"
            "_bins15_normals_comparison.png"
        )
        queue.append((comparison, paths[source_name].with_name(comparison_name)))
        added_names.add(comparison_name)
    return added_names


def _replace_ice_shapes_with_champs_grid_levels(queue, target_names: set[str]) -> None:
    """Keep only CHAMPS L1-L4 contours and reference geometry."""
    colors = {
        "L1": "#1f77b4", "L2": "#2ca02c",
        "L3": "#ff7f0e", "L4": "#d62728",
    }
    figures = {path.name: figure for figure, path in queue}
    for target_name in target_names:
        is_multilayer = "_multilayer_ice_shape_" in target_name
        target = figures.get(target_name)
        if target is None:
            raise RuntimeError(f"Missing CHAMPS ice-shape plot: {target_name}")
        sources = {}
        for level in colors:
            source_name = target_name.replace("_L1_", f"_{level}_", 1)
            source = figures.get(source_name)
            if source is None:
                raise RuntimeError(f"Missing CHAMPS ice-shape source: {source_name}")
            sources[level] = [
                copy.deepcopy(trace) for trace in source.data
                if get_trace_participant_id(trace) == "007"
            ]
        target.data = tuple(
            trace for trace in target.data if not get_trace_participant_id(trace)
        )
        for level in ("L4", "L3", "L2", "L1"):
            source_traces = sources[level]
            if not source_traces:
                if is_multilayer:
                    continue
                raise RuntimeError(f"Missing CHAMPS {level} contour for {target_name}")
            legend_shown = False
            for trace in source_traces:
                meta = dict(trace.meta) if isinstance(trace.meta, dict) else {}
                meta["ipw3_participant_id"] = "007"
                meta["ipw3_grid_level"] = level
                trace.meta = meta
                trace.name = level
                trace.legendgroup = f"007_ice_{level.lower()}"
                trace.legendrank = int(level[1:])
                trace.zorder = 5 - int(level[1:])
                trace.line.color = colors[level]
                trace.line.dash = "solid"
                trace.line.width = 5
                trace.mode = "lines"
                trace.showlegend = not legend_shown
                legend_shown = True
                target.add_trace(trace)
        slice_match = re.search(r"_slice_(\d+p\d+)_", target_name)
        if slice_match:
            slice_value = float(slice_match.group(1).replace("p", "."))
            axis = iceshape_builder.ice_shape_axis_config(CASE_ID, slice_value)
            x_range, y_range = iceshape_builder.leading_edge_axis_ranges(
                target, axis["leading_edge_fraction"],
            )
            target.update_xaxes(range=axis["x_range"] or x_range)
            target.update_yaxes(range=axis["y_range"] or y_range)
        _add_main_and_inset_borders(target)


def generate(
    participants, output_dir: Path = OUTPUT_DIR, participant_id: str | None = None,
    highlight: bool = False,
) -> int:
    """Export ONERA M6 figures, including Y=0.1, 0.75, and 1.4 m cuts."""
    import build_site_ipw3 as site

    configured_slices = tuple(CASE_SLICES[CASE_ID])
    presentation_figures = dict(ONERAM6_PRESENTATION_FIGURES)
    if participant_id is not None and str(participant_id).zfill(3) == "007":
        for slice_value in configured_slices:
            slice_slug = str(slice_value).replace(".", "p")
            filename = (
                f"tc_oneram6_L1_beta_bins01_vs_s_slice_{slice_slug}_roughness_1mm.png"
            )
            presentation_figures[f"IMPINGEMENT/{filename}"] = (
                CASE_ID, filename, WIDTH, HEIGHT, ["015"], True,
            )
            for layer in ("single_layer", "multilayer"):
                filename = (
                    f"tc_oneram6_L1_{layer}_ice_shape_slice_{slice_slug}"
                    "_bins07_roughness_1mm.png"
                )
                presentation_figures[f"ICE_SHAPES/{filename}"] = (
                    CASE_ID, filename, WIDTH, HEIGHT, [], True,
                )
        champs_htc = {
            "HTC/tc_oneram6_L1_htc_vs_s_slice_0p1_roughness_1mm.png",
            "HTC/tc_oneram6_L1_htc_vs_s_slice_0p75_roughness_1mm.png",
            "HTC/tc_oneram6_L1_htc_vs_s_slice_1p4_roughness_1mm.png",
        }
        champs_impingement = {
            "IMPINGEMENT/tc_oneram6_L1_beta_bins01_vs_s_slice_0p1_roughness_1mm.png",
            "IMPINGEMENT/tc_oneram6_L1_beta_bins01_vs_s_slice_0p75_roughness_1mm.png",
            "IMPINGEMENT/tc_oneram6_L1_beta_bins01_vs_s_slice_1p4_roughness_1mm.png",
        }
        champs_surface_fields = {
            "SURF_TEMP_FF/tc_oneram6_L1_freezing_fraction_vs_s_slice_0p75_roughness_1mm.png",
            "SURF_TEMP_FF/tc_oneram6_L1_surface_temperature_vs_s_slice_0p75_roughness_1mm.png",
        }
        champs_ice_shapes = {
            f"ICE_SHAPES/tc_oneram6_L1_{layer}_ice_shape_slice_{slice_slug}"
            "_bins07_roughness_1mm.png"
            for layer in ("single_layer", "multilayer")
            for slice_slug in ("0p1", "0p75", "1p4")
        }
        champs_horn_distribution = {
            f"ICE_HORNS/tc_oneram6_upper_horn_angle_distribution_convergence_l1_"
            f"slice_{slice_slug}_roughness_1mm.png"
            for slice_slug in ("0p1", "0p75", "1p4")
        }
        excluded_champs_aerodynamic = {
            f"AERODYNAMIC/tc_oneram6_{coefficient}_vs_n_{suffix}.png"
            for coefficient in ("cl", "cd", "cmy")
            for suffix in (
                "grouped_roughness_participants",
                "1mm_relative_to_l1",
            )
        }
        presentation_figures = {
            destination: spec for destination, spec in presentation_figures.items()
            if (not destination.startswith("HTC/") or destination in champs_htc)
            and (
                not destination.startswith("IMPINGEMENT/")
                or destination in champs_impingement
            )
            and (
                not destination.startswith("SURF_TEMP_FF/")
                or destination in champs_surface_fields
            )
            and (
                not destination.startswith("ICE_SHAPES/")
                or destination in champs_ice_shapes
            )
            and (
                not destination.startswith("ICE_HORNS/")
                or destination in champs_horn_distribution
            )
            and not destination.startswith("ICE_HORN_PARTICIPANT/")
            and not destination.startswith("ICE_LIMITS/")
            and destination not in excluded_champs_aerodynamic
        }
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
        convergence_data_builder.queue_all_grid_levels_inverse_bin_figure(participants, CASE_ID)
        site.build_water_mass_analysis_section(participants, CASE_ID)
        convergence_data_builder.build_water_mass_diameter_dispersion_figures(participants, CASE_ID)
        site.build_beta_max_analysis_section(participants, CASE_ID)
        convergence_data_builder.build_upper_horn_angle_convergence_section(participants, CASE_ID)
        for pid, filename, figure in participant_horn_method_figures(participants, CASE_ID):
            if participant_id is not None and str(participant_id).zfill(3) == "007":
                continue
            convergence_data_builder.PNG_EXPORT_QUEUE.append((figure, case_dir / filename))
            presentation_figures[f"ICE_HORN_PARTICIPANT/p{pid}/{filename}"] = (
                CASE_ID, filename, WIDTH, HEIGHT, [], False,
            )
        for filename, figure in ice_limit_figures(participants, CASE_ID, slice_filter=0.75):
            if participant_id is not None and str(participant_id).zfill(3) == "007":
                continue
            convergence_data_builder.PNG_EXPORT_QUEUE.append((figure, case_dir / filename))
            presentation_figures[f"ICE_LIMITS/{filename}"] = (
                CASE_ID, filename, WIDTH, HEIGHT, [], True,
            )
        for grid_level in sorted(VALID_GRID_LEVELS):
            site.build_grid_page_content(participants, CASE_ID, grid_level)

        _queue_coefficients_by_roughness(participants, case_dir)
        _queue_all_roughness_htc(participants, case_dir)
        _queue_turbulence_roughness_htc(case_dir)
        _queue_surface_fields_by_roughness(participants, case_dir)
        _queue_mean_roughness_variants()
        _queue_qc_roughness(participants, case_dir)
        _queue_combined_multilayer_ice(participants, case_dir)
        _queue_ice_shapes_by_roughness(participants, case_dir)
        _queue_horn_comparisons(participants, case_dir)
        _queue_horn_bin_comparisons(participants, case_dir)
        _add_001_single_bin_ice_shapes(participants)

        queues = (
            (convergence_data_builder, convergence_data_builder.PNG_EXPORT_QUEUE),
            (cutdata_builder, cutdata_builder.PNG_EXPORT_QUEUE),
            (iceshape_builder, iceshape_builder.PNG_EXPORT_QUEUE),
        )
        if participant_id is not None and not highlight:
            for _, queue in queues:
                _filter_queue_for_participant(queue, participant_id)
        if participant_id is not None and str(participant_id).zfill(3) == "007":
            polimo = next(
                p for p in participants if str(p.participant_id).zfill(3) == "007"
            )
            _replace_cp_with_champs_grid_levels(
                cutdata_builder.PNG_EXPORT_QUEUE,
                {
                    f"tc_oneram6_L1_cp_vs_{coordinate}_slice_{slice_slug}"
                    "_roughness_1mm.png"
                    for coordinate in ("x", "s")
                    for slice_slug in ("0p1", "0p75", "1p4")
                },
            )
            beta_distribution_names = _queue_champs_beta_distributions(
                cutdata_builder.PNG_EXPORT_QUEUE,
                {
                    f"tc_oneram6_L1_beta_bins01_vs_s_slice_{slice_slug}"
                    "_roughness_1mm.png"
                    for slice_slug in ("0p1", "0p75", "1p4")
                },
            )
            for filename in beta_distribution_names:
                presentation_figures[f"IMPINGEMENT/{filename}"] = (
                    CASE_ID, filename, WIDTH, HEIGHT, [], True,
                )
            htc_grid_names = _queue_champs_htc_grid_levels(
                cutdata_builder.PNG_EXPORT_QUEUE,
                {
                    f"tc_oneram6_L1_htc_vs_s_slice_{slice_slug}_roughness_1mm.png"
                    for slice_slug in ("0p1", "0p75", "1p4")
                },
            )
            for filename in htc_grid_names:
                presentation_figures[f"HTC/{filename}"] = (
                    CASE_ID, filename, WIDTH, HEIGHT, [], True,
                )
            ice_bins_names = _queue_champs_ice_shape_distributions(
                iceshape_builder.PNG_EXPORT_QUEUE,
                {
                    f"tc_oneram6_L1_single_layer_ice_shape_slice_{slice_slug}"
                    "_bins07_roughness_1mm.png"
                    for slice_slug in ("0p1", "0p75", "1p4")
                },
            )
            for filename in ice_bins_names:
                presentation_figures[f"ICE_SHAPES/{filename}"] = (
                    CASE_ID, filename, WIDTH, HEIGHT, [], True,
                )
            _replace_ice_shapes_with_champs_grid_levels(
                iceshape_builder.PNG_EXPORT_QUEUE,
                {
                    f"tc_oneram6_L1_{layer}_ice_shape_slice_{slice_slug}"
                    "_bins07_roughness_1mm.png"
                    for layer in ("single_layer", "multilayer")
                    for slice_slug in ("0p1", "0p75", "1p4")
                },
            )
            normals_names = _queue_champs_normals_comparisons(
                iceshape_builder.PNG_EXPORT_QUEUE, polimo.path,
            )
            for filename in normals_names:
                presentation_figures[f"ICE_SHAPES/{filename}"] = (
                    CASE_ID, filename, WIDTH, HEIGHT, [], True,
                )
        requested_names = {spec[1] for spec in presentation_figures.values()}
        for _, queue in queues:
            queue[:] = [(figure, path) for figure, path in queue if path.name in requested_names]
        queued_names = [path.name for _, queue in queues for _, path in queue]
        missing_slices = [
            value for value in configured_slices
            if not any(f"slice_{str(value).replace('.', 'p')}" in name for name in queued_names)
        ]
        if missing_slices:
            raise RuntimeError(f"Missing ONERA M6 presentation slices: {missing_slices}")

        specs_by_name = {spec[1]: spec for spec in presentation_figures.values()}
        for module, queue in queues:
            for figure, export_path in queue:
                _style_figure(figure, module, export_path, specs_by_name[export_path.name])

        if participant_id is not None and str(participant_id).zfill(3) == "007":
            from .POLIMO_ONERAM6_PLOTS import BETA_COMPARISONS, comparison_figure
            from .gatherParticipantData import read_tecplot_dat
            import plotly.graph_objects as go

            multilayer_path = (
                polimo.path / "TC_ONERAM6_D01"
                / "TC_ONERAM6_L1_finalIceShape_MULTILAYER_V1.dat"
            )
            multilayer_data = read_tecplot_dat(multilayer_path, process_cutdata=False)
            multilayer_figures = {
                path.name: figure for figure, path in iceshape_builder.PNG_EXPORT_QUEUE
            }
            for slice_value in configured_slices:
                slice_slug = str(slice_value).replace(".", "p")
                multilayer_target = (
                    f"tc_oneram6_L1_multilayer_ice_shape_slice_{slice_slug}"
                    "_bins07_roughness_1mm.png"
                )
                multilayer_figure = multilayer_figures.get(multilayer_target)
                if multilayer_figure is None:
                    raise RuntimeError(
                        f"Missing CHAMPS multilayer presentation plot: {multilayer_target}"
                    )
                zone_name = f"SLICE_Y_{slice_slug}_BINS07_KS_1mm_FINAL_LAYER"
                if zone_name not in multilayer_data.zones:
                    raise RuntimeError(f"Missing {zone_name} in {multilayer_path}")
                zone = multilayer_data.zones[zone_name]
                x_column = next(
                    (column for column in zone.data if column.upper() == "X_ICED"), None,
                )
                z_column = next(
                    (column for column in zone.data if column.upper() == "Z_ICED"), None,
                )
                if x_column is None or z_column is None:
                    raise RuntimeError(f"Missing iced coordinates in {multilayer_path}")
                valid = (
                    numpy.isfinite(zone.data[x_column])
                    & numpy.isfinite(zone.data[z_column])
                    & (zone.data[x_column] > -998.0)
                    & (zone.data[z_column] > -998.0)
                )
                multilayer_figure.data = tuple(
                    trace for trace in multilayer_figure.data
                    if not get_trace_participant_id(trace)
                    or (
                        isinstance(trace.meta, dict)
                        and trace.meta.get("ipw3_grid_level") != "L1"
                    )
                )
                multilayer_figure.add_trace(go.Scatter(
                    x=zone.data.loc[valid, x_column],
                    y=zone.data.loc[valid, z_column],
                    mode="lines",
                    name="L1",
                    legendgroup="007_ice_l1",
                    legendrank=1,
                    zorder=4,
                    line={"color": "#1f77b4", "dash": "solid", "width": 5},
                    meta={"ipw3_participant_id": "007", "ipw3_grid_level": "L1"},
                ))
                axis = iceshape_builder.ice_shape_axis_config(CASE_ID, slice_value)
                x_range, y_range = iceshape_builder.leading_edge_axis_ranges(
                    multilayer_figure, axis["leading_edge_fraction"],
                )
                multilayer_figure.update_xaxes(range=axis["x_range"] or x_range)
                multilayer_figure.update_yaxes(range=axis["y_range"] or y_range)

            beta_figures = {
                path.name: figure for figure, path in cutdata_builder.PNG_EXPORT_QUEUE
            }
            for comparison in BETA_COMPARISONS:
                target = comparison["filename"]
                figure = beta_figures.get(target)
                if figure is None:
                    raise RuntimeError(f"Missing single-bin presentation plot: {target}")
                eulerian = [
                    trace for trace in figure.data
                    if get_trace_participant_id(trace) == "007"
                ]
                if not eulerian:
                    raise RuntimeError(f"Missing POLIMO Eulerian trace in {target}")

                lagrangian = comparison_figure(polimo.path, comparison).data[0]
                lagrangian.name = "007 (Lagrangian)"
                lagrangian.legendgroup = "007_lagrangian"
                lagrangian.line.color = "#d62728"
                lagrangian.line.dash = "solid"
                lagrangian.mode = "lines+markers"
                lagrangian.marker = dict(
                    color="#d62728", size=9, maxdisplayed=35,
                )
                lagrangian.meta = {"ipw3_participant_id": "007"}
                figure.add_trace(lagrangian)
                if figure.layout.xaxis2 is not None and figure.layout.yaxis2 is not None:
                    zoom_trace = go.Scatter(lagrangian.to_plotly_json())
                    zoom_trace.update(
                        xaxis="x2", yaxis="y2", showlegend=False, hoverinfo="skip",
                    )
                    figure.add_trace(zoom_trace)
                    x_limits = figure.layout.xaxis2.range
                    y_limits = figure.layout.yaxis2.range
                    if x_limits and y_limits:
                        zoom_values = [
                            float(y) for x, y in zip(lagrangian.x, lagrangian.y)
                            if x_limits[0] <= float(x) <= x_limits[1]
                        ]
                        if zoom_values:
                            figure.layout.yaxis2.range = [
                                min(y_limits[0], min(zoom_values) - 0.01),
                                max(y_limits[1], max(zoom_values) + 0.01),
                            ]

        panel_exports = _queue_roughness_participant_panels(queues)

        if participant_id is not None:
            for _, queue in queues:
                for figure, _ in queue:
                    if highlight:
                        highlight_participant(figure, participant_id)
                    bring_participant_to_front(figure, participant_id)

        if participant_id is not None and str(participant_id).zfill(3) == "007":
            for _, queue in queues:
                for figure, _ in queue:
                    for trace in figure.data:
                        if get_trace_participant_id(trace) == "007":
                            trace.name = re.sub(
                                r"^007(?=\b|\s|[|(])", "CHAMPS", str(trace.name or "")
                            )

        for _, queue in queues:
            for figure, _ in queue:
                order_comparison_traces(figure, participant_id)

        convergence_data_builder.flush_png_exports(scale=1, width=None, height=None)
        cutdata_builder.flush_png_exports(scale=1, width=None, height=None)
        iceshape_builder.flush_png_exports(scale=1, width=None, height=None)

        for source, relative_destination in panel_exports:
            destination = output_dir / relative_destination
            destination.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(source, destination)

        missing: list[str] = []
        for destination_text, presentation_spec in presentation_figures.items():
            _, source_name, _, _, _, _ = presentation_spec[:6]
            source = case_dir / source_name
            destination = output_dir / destination_text
            if not source.exists():
                missing.append(f"{destination_text} <- {source_name}")
                continue
            destination.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(source, destination)
        if missing and participant_id is None:
            raise RuntimeError("Missing ONERA M6 presentation plot exports:\n  " + "\n  ".join(missing))
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
    print(f"Wrote {figure_count} ONERA M6 presentation figures in {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
