"""Central Plotly styling for every IPW3 plot.

Edit this file to change plot appearance.  Both NACA0012 test cases use the
``NACA0012`` profile; ONERA M6 uses the independent ``ONERAM6`` profile.
Values in a case profile override ``COMMON_PLOT_STYLE``.
"""

from __future__ import annotations

from copy import deepcopy
from typing import Any

import plotly.graph_objects as go


# Shared experimental envelopes for both NACA0012 cases and every output.
# RGBA's fourth value controls fill transparency: 0=invisible, 1=opaque.
NACA0012_EXPERIMENTAL_ENVELOPE_STYLE = {
    "MAXCCS": {"fill_color": "rgba(198, 219, 239,0.8)", "show_markers": False,
               "line_color": "rgba(0,0,0,0.5)", "line_width": 2, "line_dash": "solid"},
    "MINCCS": {"fill_color": "rgba(253, 208, 162,0.8)", "show_markers": False,
               "line_color": "rgba(0,0,0,0.5)", "line_width": 2, "line_dash": "solid"},
    "MEANCCS": {"line_dash": "dash", "line_width": 4, "show_markers": False},
}


# =============================================================================
# SECTION 1 — GLOBAL STYLE (applies to every plot for every case)
# =============================================================================
# Change fonts, axis lines/grids, backgrounds, legend placement, margins, and
# the default height of each plot family here.
COMMON_PLOT_STYLE: dict[str, Any] = {
    # Text used for tick labels, legends, and annotations.
    "font": {"family": "Arial, Helvetica, sans-serif", "size": 16},
    "axis_title_size": 18,

    # Base appearance of both X and Y axes.
    "axis": {
        "ticks": "outside",
        "showline": True,
        "linecolor": "black",
        "linewidth": 2,
        "mirror": True,
        "showgrid": True,
        "gridcolor": "lightgray",
        "zeroline": False,
    },
    "plot_bgcolor": "white",
    "paper_bgcolor": "white",

    # Default legend location. Individual horizontal legends are handled by
    # apply_xy_style() below.
    "legend": {"orientation": "v", "x": 1.02, "xanchor": "left", "y": 1.0, "yanchor": "top"},
    "margin": {"l": 90, "r": 220, "t": 30, "b": 80},

    # Heights in pixels for the three major categories of plot.
    "height": {"convergence": 520, "cutdata": 560, "ice_shape": 650},
}


# =============================================================================
# SECTION 1B — GRID-CONVERGENCE NORMALIZATION
# =============================================================================
# Every original L1-L4 grid-convergence plot is followed by a second copy using
# signed (value - L1) / L1 * 100 differences (L1 = 0%). This
# applies to CL, CD, CM, water mass, ice mass, evaporation mass,
# diameter-resolved mass, and Qc-prime plots. Fixed-grid plots comparing
# droplet distributions are not normalized.
GRID_CONVERGENCE_NORMALIZATION: dict[str, Any] = {
    "enabled": True,
    "reference_grid_level": "L1",
    # Finest grid first, progressing toward the coarsest grid.
    "grid_order": ["L1", "L2", "L3", "L4"],
    "grid_axis_title": "Grid level",
    "grid_coordinate_axis_title": "h = N<sub>cells</sub><sup>−1/3</sup> [-]",
    "axis_title_suffix": "relative difference from L1 [%]",
    "hover_label": "Relative difference from L1",
    "hover_format": ".4g",
    # Start at ±5%; expand symmetrically with 10% padding when data exceed it.
    "initial_y_range": [-5.0, 5.0],
    "range_padding_fraction": 0.10,
    "reference_line": {"color": "black", "width": 1.5, "dash": "dash"},
}


# =============================================================================
# SECTION 1C — DROPLET-DISTRIBUTION NORMALIZATION
# =============================================================================
# Every original fixed-grid distribution plot is followed by a signed
# difference plot using the 15-bin distribution as the numerical reference.
# BINS15 is therefore 0%; positive/negative signs are retained.
DISTRIBUTION_NORMALIZATION: dict[str, Any] = {
    "enabled": True,
    "reference_bin_set": "BINS15",
    "bin_order": ["BINS15", "BINS07", "BINS03", "BINS01"],
    "axis_title": "Droplet distribution<br><span style='font-size:14px'>← Finer&nbsp;&nbsp;|&nbsp;&nbsp;Coarser →</span>",
    "hover_label": "Relative difference from 15",
    "hover_format": ".4g",
    "initial_y_range": [-5.0, 5.0],
    "range_padding_fraction": 0.10,
    "reference_line": {"color": "black", "width": 1.5, "dash": "dash"},
}


# =============================================================================
# SECTION 1D — DIAMETER-VARIATION CURVES (IQR AND COEFFICIENT OF VARIATION)
# =============================================================================
# Each grid level can have an independent line and marker appearance. Plotly
# dash values include "solid", "dash", "dot", and "dashdot"; marker symbols
# include "circle", "square", "diamond", and "triangle-up".
DIAMETER_VARIATION_STYLE: dict[str, dict[str, Any]] = {
    "L1": {
        "line": {"color": "black", "width": 3, "dash": "solid"},
        "marker": {"color": "black", "size": 9, "symbol": "circle", "line": {"color": "black", "width": 1}},
    },
    "L2": {
        "line": {"color": "#4D4D4D", "width": 3, "dash": "dash"},
        "marker": {"color": "#4D4D4D", "size": 9, "symbol": "square", "line": {"color": "#4D4D4D", "width": 1}},
    },
    "L3": {
        "line": {"color": "#808080", "width": 3, "dash": "dashdot"},
        "marker": {"color": "#808080", "size": 9, "symbol": "diamond", "line": {"color": "#808080", "width": 1}},
    },
    "L4": {
        "line": {"color": "#B3B3B3", "width": 3, "dash": "dot"},
        "marker": {"color": "#B3B3B3", "size": 9, "symbol": "triangle-up", "line": {"color": "#B3B3B3", "width": 1}},
    },
}


# Special NACA0012 HTC and ice-shape presentation plots grouped by roughness
# treatment. Edit these dictionaries to change either plot family.
NACA0012_ROUGHNESS_GROUP_STYLE: dict[str, dict[str, Any]] = {
    "point_two_mm": {
        "label": "0.2 mm",
        "show_markers": True,
        "line": {"color": "#FF7F0E", "width": 5, "dash": "solid"},
        "marker": {"color": "#FF7F0E", "size": 12, "symbol": "triangle-up", "maxdisplayed": 10, "line": {"color": "#000000", "width": 1}},
    },
    "half_mm": {
        "label": "0.5 / 0.5334 mm",
        "show_markers": True,
        "line": {"color": "#2CA02C", "width": 5, "dash": "solid"},
        "marker": {"color": "#2CA02C", "size": 12, "symbol": "circle", "maxdisplayed": 10, "line": {"color": "#000000", "width": 1}},
    },
    "one_mm": {
        "label": "1 mm",
        "show_markers": True,
        "line": {"color": "#D62728", "width": 5, "dash": "solid"},
        "marker": {"color": "#D62728", "size": 12, "symbol": "square", "maxdisplayed": 10, "line": {"color": "#000000", "width": 1}},
    },
    "variable": {
        "label": "Variable",
        "show_markers": True,
        "line": {"color": "#1F77B4", "width": 5, "dash": "solid"},
        "marker": {"color": "#1F77B4", "size": 12, "symbol": "diamond", "maxdisplayed": 10, "line": {"color": "#000000", "width": 1}},
    },
}


# =============================================================================
# SECTION 2 — CASE-SPECIFIC STYLE
# =============================================================================
# The AE3932 and AE3933 cases intentionally share the single NACA0012 section.
# ONERA M6 is independent, so edits in its section do not affect NACA0012.
#
# Available keys in each case section:
#   layout                 Plotly layout overrides applied to all plot families
#   axis                   Plotly X/Y axis overrides applied to all plot families
#   cutdata_x_ranges       X limits for named cut-data plots
#   beta_inset_x_range     X limits of the collection-efficiency zoom inset
#   beta_inset_y_range     Y limits of the collection-efficiency zoom inset
#   ice_shape              Titles, limits, leading-edge zoom, and slice overrides
#   plots                  Individual customization for every plot variable
#
# Keys available inside each individual plot block:
#   height: 600
#   xaxis: {"range": [xmin, xmax], "title": {"text": "Custom X"}}
#   yaxis: {"range": [ymin, ymax], "title": {"text": "Custom Y"}}
#   layout: {"showlegend": False, "plot_bgcolor": "#fafafa"}
#   traces: {"line": {"width": 3}}  # Applied to all traces in that plot
CASE_PLOT_STYLES: dict[str, dict[str, Any]] = {
    # -------------------------------------------------------------------------
    # SECTION 2A — NACA0012 (shared by AE3932 and AE3933)
    # -------------------------------------------------------------------------
    "NACA0012": {
        # Put any case-wide Plotly layout/axis overrides in these dictionaries.
        # Example: "layout": {"paper_bgcolor": "#fafafa"}
        "layout": {},
        "axis": {},

        # Cut-data X-axis limits [minimum, maximum], in metres.
        "cutdata_x_ranges": {
            "cp_vs_s": [-0.2, 0.2],
            "recovery_temperature_vs_s": [-0.2, 0.2],
            "freezing_fraction_vs_s": [-0.1, 0.1],
            "htc_vs_s": [-0.4, 0.4],
            "beta": [-0.2, 0.1],
        },
        # Collection-efficiency inset X limits [minimum, maximum], in metres.
        "beta_inset_x_range": [-0.025, 0.025],
        # Collection-efficiency inset Y limits [minimum, maximum]. None = automatic.
        "beta_inset_y_range": None,

        # Ice-shape plot axes. Use None for Plotly/leading-edge automatic limits.
        # Add values inside a slice entry to override only that slice, e.g.
        # 0.9144: {"x_range": [-0.01, 0.12], "y_range": [-0.04, 0.04]}.
        "ice_shape": {
            "x_title": "X [m]",
            "y_title": "Z [m]",
            "x_range": None,
            "y_range": [-0.04, 0.03],
            "leading_edge_fraction": 0.25,
            "x_min": -0.04,
            "slices": {0.9144: {}},
        },

        # INDIVIDUAL NACA0012 PLOTS (shared by AE3932 and AE3933).
        # Add Plotly values to any block. Examples:
        # "cl_vs_n": {"yaxis": {"range": [0.1, 0.5]}, "height": 600,
        #              "traces": {"line": {"width": 3}}}
        # "cp_vs_x": {"xaxis": {"range": [0, 0.55]}, "layout": {"showlegend": False}}
        # Beta has ONE block shared by BINS01/BINS03/BINS07/BINS15/CARDS.
        "plots": {
            # Grid convergence: aerodynamic coefficients
            "cl_vs_n": {"height": 520, "yaxis": {"title": {"text": "CL [-]"}, "range": [0.40, 0.50], "dtick": 0.02}},
            "cd_vs_n": {"height": 520, "yaxis": {"title": {"text": "CD [-]"}, "range": [0.005, 0.02], "dtick": 0.0025}},
            "cmy_vs_n": {"height": 520, "yaxis": {"title": {"text": "CM [-]"}, "range": [-0.025, 0.01], "dtick": 0.005}},
            # Grid convergence: total icing masses
            "water_mass_vs_n": {"height": 520, "yaxis": {"title": {"text": "m<sub>water</sub> [g]"}}},
            "ice_mass_vs_n": {"height": 520, "yaxis": {"title": {"text": "m<sub>ice</sub> [g]"}}},
            "water_evap_mass_vs_n": {"height": 520, "yaxis": {"title": {"text": "m<sub>evap</sub> [g]"}, "range": [0.0, 15.0]}},
            # Grid convergence: icing masses by droplet diameter
            "water_mass_by_diameter_vs_n": {"height": 520, "yaxis": {"title": {"text": "m<sub>water</sub> [g]"}}},
            "ice_mass_by_diameter_vs_n": {"height": 520, "yaxis": {"title": {"text": "m<sub>ice</sub> [g]"}}},
            "water_evap_mass_by_diameter_vs_n": {"height": 520, "yaxis": {"title": {"text": "Water evaporation mass [g]"}}},
            # Across-participant diameter variation (optional BINS15 data)
            "water_mass_relative_iqr_vs_droplet_diameter": {"height": 520, "yaxis": {"title": {"text": "m<sub>water</sub> relative IQR [%]"}}},
            "water_mass_cv_vs_droplet_diameter": {"height": 520, "yaxis": {"title": {"text": "m<sub>water</sub> coefficient of variation [%]"}}},
            "qc_prime": {"height": 520, "yaxis": {"title": {"text": "Q<sub>c</sub>′ [W/m]"}, "range": [-100.0, 400.0]}},
            # L1-relative copies displayed after each original convergence plot
            "cl_vs_n_relative": {"height": 520, "yaxis": {"title": {"text": "ΔCL from L1 [%]"}}},
            "cd_vs_n_relative": {"height": 520, "yaxis": {"title": {"text": "ΔCD from L1 [%]"}, "dtick": 25.0}},
            "cmy_vs_n_relative": {"height": 520, "yaxis": {"title": {"text": "ΔCM from L1 [%]"}}},
            "water_mass_vs_n_relative": {"height": 520, "yaxis": {"title": {"text": "Δm<sub>water</sub> from L1 [%]"}, "range": [-6.0, 6.0]}},
            "ice_mass_vs_n_relative": {"height": 520, "yaxis": {"title": {"text": "ΔIce-mass from L1 [%]"}}},
            "water_evap_mass_vs_n_relative": {"height": 520, "yaxis": {"title": {"text": "ΔEvaporation-mass from L1 [%]"}}},
            "water_mass_by_diameter_vs_n_relative": {"height": 520, "yaxis": {"title": {"text": "Δm<sub>water</sub> from L1 [%]"}}},
            "ice_mass_by_diameter_vs_n_relative": {"height": 520, "yaxis": {"title": {"text": "ΔIce-mass from L1 [%]"}}},
            "water_evap_mass_by_diameter_vs_n_relative": {"height": 520, "yaxis": {"title": {"text": "ΔEvaporation-mass from L1 [%]"}}},
            "qc_prime_relative": {"height": 520, "yaxis": {"title": {"text": "Q<sub>c</sub>′ difference from L1 [%]"}}},
            # Fixed-grid droplet-distribution differences from BINS15
            "water_mass_vs_n_relative_to_bins15": {"height": 520, "yaxis": {"title": {"text": "Δm<sub>water</sub> from 15 bins [%]"}}},
            "ice_mass_vs_n_relative_to_bins15": {"height": 520, "yaxis": {"title": {"text": "Ice-mass difference from 15 bins [%]"}}},
            "water_evap_mass_vs_n_relative_to_bins15": {"height": 520, "yaxis": {"title": {"text": "Evaporation-mass difference from 15 [%]"}}},
            # Derived icing water-fate ratios
            "ice_to_water_ratio_vs_n": {"height": 520, "yaxis": {"title": {"text": "m<sub>ice</sub> / m<sub>water</sub> [%]"},"range": [85.0, 100.0]}},
            "ice_evap_to_water_ratio_vs_n": {"height": 520, "yaxis": {"title": {"text": "(m<sub>ice</sub> + m<sub>evap.</sub>) / m<sub>water</sub> [%]"}}},
            # Water Mass Analysis: collection efficiency and impingement extent
            "beta_max_vs_n": {"height": 520, "yaxis": {"title": {"text": "β<sub>max</sub> [-]"}, "range": [0.6, 0.8]}},
            "beta_max_vs_n_relative": {"height": 520, "yaxis": {"title": {"text": "Δβ<sub>max</sub> from L1 [%]"}}},
            "peak_beta_s_vs_n": {"height": 520, "yaxis": {"title": {"text": "s<sub>βmax</sub> [m]"}}},
            "peak_beta_s_vs_n_relative": {"height": 520, "yaxis": {"title": {"text": "Δs<sub>βmax</sub> from L1 [%]"}}},
            "impingement_width_vs_n": {"height": 520, "yaxis": {"title": {"text": "Width<sub>imp</sub> [m]"}, "range": [0.25, 0.35]}},
            "impingement_width_vs_n_relative": {"height": 520, "yaxis": {"title": {"text": "ΔWidth<sub>imp</sub> from L1 [%]"}, "range": [-5.0, 5.0]}},
            "beta_max_vs_bins": {"height": 520, "yaxis": {"title": {"text": "β<sub>max</sub> [-]"}}},
            "peak_beta_s_vs_bins": {"height": 520, "yaxis": {"title": {"text": "s<sub>βmax</sub> [m]"}}},
            "peak_beta_s_vs_bins_relative_to_bins15": {"height": 520, "yaxis": {"title": {"text": "Δs<sub>βmax</sub> from 15 bins [%]"}}},
            "impingement_width_vs_bins": {"height": 520, "yaxis": {"title": {"text": "Width<sub>imp</sub> [m]"}}},
            "beta_max_vs_bins_relative_to_bins15": {"height": 520, "yaxis": {"title": {"text": "Δβ<sub>max</sub> from 15 bins [%]"}}},
            "impingement_width_vs_bins_relative_to_bins15": {"height": 520, "yaxis": {"title": {"text": "ΔWidth<sub>imp</sub> from 15 bins [%]"}}},
            "upper_horn_angle_vs_n": {"height": 520, "yaxis": {"title": {"text": "α<sub>upper, horn</sub> [deg]"}, "range": [90.0, 185.0]}},
            "upper_horn_angle_vs_bins": {"height": 520, "yaxis": {"title": {"text": "α<sub>upper, horn</sub> [deg]"}, "range": [90.0, 185.0]}},
            "upper_horn_angle_by_participant": {"height": 520, "xaxis": {"title": {"text": "α<sub>upper, horn</sub> [deg]"}}, "yaxis": {"title": {"text": "Participant ID"}}},
            # AE3933 ice- and water-mass comparisons against matching AE3932 data
            "ae3933_minus_ae3932_ice_mass": {"height": 520, "yaxis": {"title": {"text": "m<sub>ice, AE3933</sub> − m<sub>ice, AE3932</sub> [g]"}}},
            "ae3933_minus_ae3932_ice_mass_percent": {"height": 520, "yaxis": {"title": {"text": "Ice-mass difference relative to AE3932 [%]"}}},
            "ae3933_minus_ae3932_water_mass": {"height": 520, "yaxis": {"title": {"text": "AE3933 − AE3932 water mass [g]"}}},
            "ae3933_minus_ae3932_water_mass_percent": {"height": 520, "yaxis": {"title": {"text": "Water-mass difference relative to AE3932 [%]"}}},
            "ae3933_minus_ae3932_beta_max": {"height": 520, "yaxis": {"title": {"text": "AE3933 − AE3932 βmax [-]"}}},
            "ae3933_minus_ae3932_beta_max_percent": {"height": 520, "yaxis": {"title": {"text": "βmax difference relative to AE3932 [%]"}}},
            # All cut-data plots
            "cp_vs_x": {"height": 560, "xaxis": {"title": {"text": "X/C [-]"}, "range": [-0.05, 1.05]}, "yaxis": {"title": {"text": "Cp [-]"}, "autorange": "reversed"}},
            "cp_vs_s": {"height": 560, "xaxis": {"title": {"text": "Surface distance from highlight [m]"}}, "yaxis": {"title": {"text": "Cp [-]"}, "autorange": "reversed"}},
            "htc_vs_s": {"height": 560, "xaxis": {"title": {"text": "Surface distance from highlight [m]"}, "range": [-0.1, 0.1]}, "yaxis": {"title": {"text": "HTC [W/m²K]"}, "range": [0, 2000]}},
            "beta": {"height": 560, "xaxis": {"title": {"text": "Surface distance from highlight [m]"}}, "yaxis": {"title": {"text": "β [-]"}}},  # X range comes from cutdata_x_ranges above.
            "surface_temperature_vs_s": {"height": 560, "xaxis": {"title": {"text": "Surface distance from highlight [m]"}, "range": [-0.1, 0.1]}, "yaxis": {"title": {"text": "T<sub>s</sub> [K]"}, "range": [254.0, 274.0]}},
            "recovery_temperature_vs_s": {"height": 560, "xaxis": {"title": {"text": "Surface distance from highlight [m]"}}, "yaxis": {"title": {"text": "T<sub>rec</sub> [K]"}}},
            "freezing_fraction_vs_s": {"height": 560, "xaxis": {"title": {"text": "Surface distance from highlight [m]"}, "range": [-0.1, 0.1]}, "yaxis": {"title": {"text": "FF [-]"}}},
            "mean_surface_temperature_vs_n": {"height": 520, "yaxis": {"title": {"text": "Mean surface temperature [K]"}, "range": [260.0, 273.15]}},
            "mean_freezing_fraction_vs_n": {"height": 520, "yaxis": {"title": {"text": "Mean freezing fraction [-]"}}},
            # Ice shapes
            "ice_shape_single": {"height": 650, "xaxis": {"constrain": "domain", "autorange": False, "title": {"text": "X [m]"}}, "yaxis": {"constrain": "domain", "autorange": False, "title": {"text": "Z [m]"}, "scaleanchor": "x", "scaleratio": 1.0}},
            "ice_shape_final": {"height": 650, "xaxis": {"constrain": "domain", "autorange": False, "title": {"text": "X [m]"}}, "yaxis": {"constrain": "domain", "autorange": False, "title": {"text": "Z [m]"}, "scaleanchor": "x", "scaleratio": 1.0}},
        },
    },

    # -------------------------------------------------------------------------
    # SECTION 2B — ONERA M6 (independent customization)
    # -------------------------------------------------------------------------
    "ONERAM6": {
        # Put any ONERA-M6-wide Plotly layout/axis overrides here.
        "layout": {},
        "axis": {},

        # Cut-data X-axis limits [minimum, maximum], in metres.
        "cutdata_x_ranges": {
            "cp_vs_s": [-0.2, 0.2],
            "recovery_temperature_vs_s": [-0.2, 0.2],
            "freezing_fraction_vs_s": [-0.125, 0.125],
            "htc_vs_s": [-0.4, 0.4],
            "beta": [-0.15, 0.15],
        },
        # Collection-efficiency inset X limits [minimum, maximum], in metres.
        "beta_inset_x_range": [-0.025, 0.025],
        # Collection-efficiency inset Y limits [minimum, maximum]. None = automatic.
        "beta_inset_y_range": None,

        # Ice-shape defaults plus optional overrides for each spanwise slice.
        # Example slice override: 0.75: {"x_range": [...], "y_range": [...]}.
        "ice_shape": {
            "x_title": "X [m]",
            "y_title": "Z [m]",
            "x_range": None,
            "y_range": None,
            "leading_edge_fraction": 0.25,
            "slices": {0.1: {}, 0.75: {}, 1.4: {}},
        },

        # INDIVIDUAL ONERA M6 PLOTS.
        # Each block applies to ALL roughness heights; no roughness-specific
        # style is needed. Beta is also shared by every bin/card plot.
        "plots": {
            "mean_surface_temperature_vs_n_relative": {"height": 520, "yaxis": {"title": {"text": "ΔT̅<sub>s</sub> from L1 [%]"}}},
            "mean_freezing_fraction_vs_n_relative": {"height": 520, "yaxis": {"title": {"text": "ΔF̅F̅ from L1 [%]"}}},
            "mean_surface_temperature_vs_n": {"height": 520, "yaxis": {"title": {"text": "T̅<sub>s</sub> [K]"}}},
            "mean_freezing_fraction_vs_n": {"height": 520, "yaxis": {"title": {"text": "F̅F̅ [-]"}}},
            # Grid convergence: aerodynamic coefficients
            "cl_vs_n": {"height": 520, "yaxis": {"title": {"text": "CL [-]"}, "range": [0.115, 0.13], "dtick": 0.005}},
            "cd_vs_n": {"height": 520, "yaxis": {"title": {"text": "CD [-]"}, "range": [0.005, 0.025], "dtick": 0.0025}},
            "cmy_vs_n": {"height": 520, "yaxis": {"title": {"text": "CM [-]"}, "range": [-0.055, -0.035], "dtick": 0.0025}},
            # Grid convergence: total icing masses
            "water_mass_vs_n": {"height": 520, "yaxis": {"title": {"text": "m<sub>water</sub> [kg]"}}},
            "ice_mass_vs_n": {"height": 520, "yaxis": {"title": {"text": "m<sub>ice</sub> [kg]"}}},
            "water_evap_mass_vs_n": {"height": 520, "yaxis": {"title": {"text": "m<sub>evap</sub> [kg]"}}},
            # Grid convergence: icing masses by droplet diameter
            "water_mass_by_diameter_vs_n": {"height": 520, "yaxis": {"title": {"text": "m<sub>water</sub> [kg]"}}},
            "ice_mass_by_diameter_vs_n": {"height": 520, "yaxis": {"title": {"text": "m<sub>ice</sub> [kg]"}}},
            "water_evap_mass_by_diameter_vs_n": {"height": 520, "yaxis": {"title": {"text": "Water evaporation mass [kg]"}}},
            "water_mass_relative_iqr_vs_droplet_diameter": {"height": 520, "yaxis": {"title": {"text": "m<sub>water</sub> relative IQR [%]"}}},
            "water_mass_cv_vs_droplet_diameter": {"height": 520, "yaxis": {"title": {"text": "m<sub>water</sub> coefficient of variation [%]"}}},
            "qc_prime": {"height": 520, "yaxis": {"title": {"text": "Q<sub>c</sub>′ [W/m]"}}},
            # L1-relative copies displayed after each original convergence plot
            "cl_vs_n_relative": {"height": 520, "yaxis": {"title": {"text": "ΔCL from L1 [%]"}}},
            "cd_vs_n_relative": {"height": 520, "yaxis": {"title": {"text": "ΔCD from L1 [%]"}}},
            "cmy_vs_n_relative": {"height": 520, "yaxis": {"title": {"text": "ΔCM from L1 [%]"}}},
            "water_mass_vs_n_relative": {"height": 520, "yaxis": {"title": {"text": "Δm<sub>water</sub> from L1 [%]"}}},
            "ice_mass_vs_n_relative": {"height": 520, "yaxis": {"title": {"text": "ΔIce-mass from L1 [%]"}}},
            "water_evap_mass_vs_n_relative": {"height": 520, "yaxis": {"title": {"text": "ΔEvaporation-mass from L1 [%]"}}},
            "water_mass_by_diameter_vs_n_relative": {"height": 520, "yaxis": {"title": {"text": "Δm<sub>water</sub> from L1 [%]"}}},
            "ice_mass_by_diameter_vs_n_relative": {"height": 520, "yaxis": {"title": {"text": "ΔIce-mass from L1 [%]"}}},
            "water_evap_mass_by_diameter_vs_n_relative": {"height": 520, "yaxis": {"title": {"text": "ΔEvaporation-mass from L1 [%]"}}},
            "qc_prime_relative": {"height": 520, "yaxis": {"title": {"text": "Q<sub>c</sub>′ difference from L1 [%]"}}},
            # Fixed-grid droplet-distribution differences from BINS15
            "water_mass_vs_n_relative_to_bins15": {"height": 520, "yaxis": {"title": {"text": "Δm<sub>water</sub> from 15 bins [%]"}}},
            "ice_mass_vs_n_relative_to_bins15": {"height": 520, "yaxis": {"title": {"text": "Ice-mass difference from 15 bins [%]"}}},
            "water_evap_mass_vs_n_relative_to_bins15": {"height": 520, "yaxis": {"title": {"text": "Evaporation-mass difference from 15 [%]"}}},
            # Derived icing water-fate ratios (shared across roughness heights)
            "ice_to_water_ratio_vs_n": {"height": 520, "yaxis": {"title": {"text": "m<sub>ice</sub> / m<sub>water</sub> [%]"}, "range": [40, 105], "autorange": False, "dtick": 10}},
            "ice_evap_to_water_ratio_vs_n": {"height": 520, "yaxis": {"title": {"text": "(m<sub>ice</sub> + m<sub>evap.</sub>) / m<sub>water</sub> [%]"}, "range": [40, 105], "autorange": False, "dtick": 10}},
            # Water Mass Analysis: collection efficiency and impingement extent
            "beta_max_vs_n": {"height": 520, "yaxis": {"title": {"text": "β<sub>max</sub> [-]"}}},
            "beta_max_vs_n_relative": {"height": 520, "yaxis": {"title": {"text": "Δβ<sub>max</sub> from L1 [%]"}}},
            "peak_beta_s_vs_n": {"height": 520, "yaxis": {"title": {"text": "s<sub>βmax</sub> [m]"}}},
            "peak_beta_s_vs_n_relative": {"height": 520, "yaxis": {"title": {"text": "Δs<sub>βmax</sub> from L1 [%]"}}},
            "impingement_width_vs_n": {"height": 520, "yaxis": {"title": {"text": "Width<sub>imp</sub> [m]"}}},
            "impingement_width_vs_n_relative": {"height": 520, "yaxis": {"title": {"text": "ΔWidth<sub>imp</sub> from L1 [%]"}}},
            "beta_max_vs_bins": {"height": 520, "yaxis": {"title": {"text": "β<sub>max</sub> [-]"}}},
            "peak_beta_s_vs_bins": {"height": 520, "yaxis": {"title": {"text": "s<sub>βmax</sub> [m]"}}},
            "peak_beta_s_vs_bins_relative_to_bins15": {"height": 520, "yaxis": {"title": {"text": "Δs<sub>βmax</sub> from 15 bins [%]"}}},
            "impingement_width_vs_bins": {"height": 520, "yaxis": {"title": {"text": "Width<sub>imp</sub> [m]"}}},
            "beta_max_vs_bins_relative_to_bins15": {"height": 520, "yaxis": {"title": {"text": "Δβ<sub>max</sub> from 15 bins [%]"}}},
            "impingement_width_vs_bins_relative_to_bins15": {"height": 520, "yaxis": {"title": {"text": "ΔWidth<sub>imp</sub> from 15 bins [%]"}}},
            "upper_horn_angle_vs_n": {"height": 520, "yaxis": {"title": {"text": "α<sub>upper, horn</sub> [deg]"}}},
            "upper_horn_angle_vs_bins": {"height": 520, "yaxis": {"title": {"text": "α<sub>upper, horn</sub> [deg]"}}},
            # All cut-data plots (one style across all roughness heights)
            "cp_vs_x": {"height": 560, "xaxis": {"title": {"text": "X / C [-]"}, "range": [-0.05, 1.05]}, "yaxis": {"title": {"text": "Cp [-]"}, "range": [1.0, -1.0], "autorange": False}},
            "cp_vs_s": {"height": 560, "xaxis": {"title": {"text": "Surface distance from highlight [m]"}}, "yaxis": {"title": {"text": "Cp [-]"}, "autorange": "reversed"}},
            "htc_vs_s": {"height": 560, "xaxis": {"title": {"text": "Surface distance from highlight [m]"}, "range": [-0.05, 0.05]}, "yaxis": {"title": {"text": "HTC [W/m²K]"}, "range": [0, 1500]}},
            "beta": {"height": 560, "xaxis": {"title": {"text": "Surface distance from highlight [m]"}, "range": [-0.15, 0.15]}, "yaxis": {"title": {"text": "β [-]"}}},  # Shared across every bin and roughness height.
            "surface_temperature_vs_s": {"height": 560, "xaxis": {"title": {"text": "Surface distance from highlight [m]"}}, "yaxis": {"title": {"text": "T<sub>s</sub> [K]"}}},
            "recovery_temperature_vs_s": {"height": 560, "xaxis": {"title": {"text": "Surface distance from highlight [m]"}}, "yaxis": {"title": {"text": "T<sub>rec</sub> [K]"}}},
            "freezing_fraction_vs_s": {"height": 560, "xaxis": {"title": {"text": "Surface distance from highlight [m]"}, "range": [-0.125, 0.125]}, "yaxis": {"title": {"text": "FF [-]"}}},
            # Ice shapes
            "ice_shape_single": {"height": 650, "xaxis": {"title": {"text": "X [m]"}}, "yaxis": {"title": {"text": "Z [m]"}, "scaleanchor": "x", "scaleratio": 1.0}},
            "ice_shape_final": {"height": 650, "xaxis": {"title": {"text": "X [m]"}}, "yaxis": {"title": {"text": "Z [m]"}, "scaleanchor": "x", "scaleratio": 1.0}},
        },
    },
}


# =============================================================================
# SECTION 3 — STYLE APPLICATION HELPERS (normally no editing needed)
# =============================================================================
# Everything below maps case IDs to the sections above and applies those values
# consistently in the convergence, cut-data, heat-flux, and ice-shape builders.
def case_style_name(case_id: str | None) -> str | None:
    if not case_id:
        return None
    if case_id.startswith("TC_NACA0012_"):
        return "NACA0012"
    if case_id == "TC_ONERAM6":
        return "ONERAM6"
    return None


def case_plot_style(case_id: str | None) -> dict[str, Any]:
    return CASE_PLOT_STYLES.get(case_style_name(case_id) or "", {})


def canonical_plot_key(plot_key: str | None) -> str | None:
    """Map every beta bin/card plot to the one shared ``beta`` style block."""
    if not plot_key:
        return None
    is_surface_beta_plot = (
        plot_key == "beta"
        or plot_key.startswith("beta_bins")
        or plot_key.startswith("beta_cards")
    )
    return "beta" if is_surface_beta_plot else plot_key


def individual_plot_style(case_id: str | None, plot_key: str | None) -> dict[str, Any]:
    key = canonical_plot_key(plot_key)
    return case_plot_style(case_id).get("plots", {}).get(key, {}) if key else {}


def _merge_axis_style(base: dict[str, Any], override: dict[str, Any]) -> None:
    """Merge an axis override while retaining unspecified title font values."""
    override = deepcopy(override)
    title_override = override.pop("title", None)
    base.update(override)
    if title_override:
        base.setdefault("title", {}).update(title_override)


def apply_individual_plot_overrides(fig: go.Figure, case_id: str, plot_key: str) -> go.Figure:
    """Reapply user overrides after a builder's required axis configuration."""
    plot = individual_plot_style(case_id, plot_key)
    if plot.get("xaxis"):
        fig.update_xaxes(**plot["xaxis"])
    if plot.get("yaxis"):
        fig.update_yaxes(**plot["yaxis"])
    if plot.get("layout"):
        fig.update_layout(**plot["layout"])
    if "height" in plot:
        fig.update_layout(height=plot["height"])
    if plot.get("traces"):
        fig.update_traces(**plot["traces"])
    return fig


def apply_xy_style(
    fig: go.Figure,
    case_id: str | None,
    x_label: str,
    y_label: str,
    *,
    plot_family: str,
    plot_key: str | None = None,
    height: int | None = None,
    legend_right: bool = True,
    reverse_y_axis: bool = False,
    x_range: list[float] | tuple[float, float] | None = None,
    y_range: list[float] | tuple[float, float] | None = None,
) -> go.Figure:
    """Apply common style plus the selected NACA0012/ONERAM6 overrides."""
    common = COMMON_PLOT_STYLE
    case = case_plot_style(case_id)
    plot = individual_plot_style(case_id, plot_key)
    axis_base = deepcopy(common["axis"])
    axis_base.update(case.get("axis", {}))
    xaxis = {**axis_base, "title": {"text": x_label, "font": {"size": common["axis_title_size"]}}}
    yaxis = {**axis_base, "title": {"text": y_label, "font": {"size": common["axis_title_size"]}}}
    if x_range is not None:
        xaxis["range"] = list(x_range)
    if y_range is not None:
        yaxis["range"] = list(y_range)
    elif reverse_y_axis:
        yaxis["autorange"] = "reversed"
    # Individual plot values are applied last so the editable plot block wins.
    _merge_axis_style(xaxis, plot.get("xaxis", {}))
    _merge_axis_style(yaxis, plot.get("yaxis", {}))

    if legend_right:
        legend = deepcopy(common["legend"])
        margin = deepcopy(common["margin"])
    else:
        legend = {"orientation": "h", "x": 0.0, "xanchor": "left", "y": 1.12, "yanchor": "bottom"}
        margin = {"l": 90, "r": 40, "t": 70, "b": 80}

    layout = {
        "font": deepcopy(common["font"]), "autosize": True,
        "height": plot.get("height", height or common["height"][plot_family]), "title": None,
        "showlegend": True, "xaxis": xaxis, "yaxis": yaxis,
        "legend": legend, "margin": margin,
        "plot_bgcolor": common["plot_bgcolor"], "paper_bgcolor": common["paper_bgcolor"],
    }
    layout.update(case.get("layout", {}))
    layout.update(plot.get("layout", {}))
    fig.update_layout(**layout)
    if plot.get("traces"):
        fig.update_traces(**plot["traces"])
    return fig


def cutdata_x_range(case_id: str, plot_key: str, is_beta_plot: bool = False) -> list[float] | None:
    key = "beta" if is_beta_plot else plot_key
    value = case_plot_style(case_id).get("cutdata_x_ranges", {}).get(key)
    return list(value) if value is not None else None


def beta_inset_x_range(case_id: str) -> tuple[float, float] | None:
    value = case_plot_style(case_id).get("beta_inset_x_range")
    return tuple(value) if value is not None else None


def beta_inset_y_range(case_id: str) -> tuple[float, float] | None:
    value = case_plot_style(case_id).get("beta_inset_y_range")
    return tuple(value) if value is not None else None


def ice_shape_axis_config(case_id: str, slice_filter: float | None) -> dict[str, Any]:
    config = deepcopy(case_plot_style(case_id).get("ice_shape", {}))
    slices = config.pop("slices", {})
    if slice_filter is not None:
        for configured_slice, override in slices.items():
            if abs(float(configured_slice) - round(slice_filter, 8)) <= 1.0e-8:
                config.update(override)
                break
    return config
