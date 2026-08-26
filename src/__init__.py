"""
FOGGIE Dynamics Analysis Package
=================================

Tools for analyzing galaxy density profiles from FOGGIE simulations.

This package provides:
- Parametric density models (Hernquist, NFW, exponential disk, etc.)
- MCMC-based fitting routines using emcee
- Statistical analysis and model comparison
- Visualization tools for 1D and 2D density profiles
- Workflow orchestration for single snapshots and time-series

Submodules
----------
models : Density profile models (Hernquist, NFW, exponential disk, two-power, etc.)
fitting : MCMC fitting functions for various models
plotting : Visualization tools for density profiles and fitting results
analysis : Statistical analysis, model comparison, mass calculations
filters_and_utils : Utility functions for data processing and field creation
density_profiles : Profile calculation and mass integration routines
workflow : High-level workflow orchestration for fitting pipelines
config : Configuration and logging utilities

Quick Start
-----------
>>> from src import workflow, config
>>> config_dict = config.load_config()
>>> logger = config.setup_logging('halo_008508', 'star')
>>> result = workflow.process_snapshot('halo_008508', 'star', config_dict)

Notes
-----
See TODO.md and REFACTORING.md in project root for planned improvements
and refactoring opportunities.
"""

__version__ = "0.1.0"

from .analysis import (
    calc_masses_with_uncertainties,
    calculate_component_statistics,
    calculate_fit_statistics,
    calculate_model_masses,
    compare_masses,
    compare_models,
    gelman_rubin_diagnostic,
    generate_component_summary,
    get_shape_tensor,
    save_evolution_data,
    save_results,
    spherical_average,
)
from .config import ProjectPaths, load_config, setup_logging
from .density_profiles import (
    check_extended_mass,
    find_virial_radius,
    integrate_disk_mass,
    integrate_spherical_mass,
    interpolate_r200_from_metadata,
    iterative_center,
)
from .filters_and_utils import (
    add_gaussian_weight,
    add_inverse_radius_weight,
    add_rotated_fields,
    apply_data_range_mask,
    create_profiles,
    extract_density,
    profile_eq_str,
    rotation_matrix_from_vectors,
)
from .fitting import (
    fit_bulge_emcee,
    fit_cored_bulge_emcee,
    fit_disk_emcee,
    fit_double_disk_flaring_emcee,
    fit_double_exponential_disk_emcee,
    fit_exp_cutoff_emcee,
    fit_exp_emcee,
    fit_general_emcee,
    fit_halo_emcee,
    fit_single_disk_flaring_emcee,
    fit_three_power_emcee,
    fit_two_power_emcee,
)
from .io import ensure_directory, load_hdf5, save_figure, save_hdf5
from .models import (
    ModelRegistry,
    cored_power_model,
    disk_density,
    double_exp_model,
    double_exponential_disk,
    hernquist_density,
    nfw_density,
    sech_squared,
    three_power_model,
    two_power_model,
)
from .plotting import (
    apply_axis_style,
    get_component_colors,
    get_plot_config,
    plot_1d_density,
    plot_2d_density_with_marginals,
    plot_2d_model_comparison,
    plot_2d_total_model_comparison,
    plot_combined_components,
    plot_comp_projection_with_r200,
    plot_component_mass_evolution,
    plot_component_mass_ratio_evolution,
    plot_corner,
    plot_data_mass_evolution,
    plot_disk_slices,
    plot_disk_subtraction,
    plot_fitted_mass_evolution,
    plot_galaxy_density_profile,
    plot_halo_shape_evolution,
    plot_profiles,
    plot_reddish_2022_figure,
    plot_results,
    plot_shape_analysis,
    plot_mass_error
)
from .workflow import (
    DensityProfile,
    FittingResult,
    find_snapshots,
    fit_component_models,
    fit_dm_components,
    fit_gas_components,
    fit_stellar_components,
    generate_snapshot_profiles,
    process_snapshot,
    process_time_series,
)

__all__ = [
    # Models
    "ModelRegistry",
    "hernquist_density",
    "nfw_density",
    "disk_density",
    "two_power_model",
    "double_exponential_disk",
    "cored_power_model",
    "double_exp_model",
    "sech_squared",
    "three_power_model",
    # Fitting
    "fit_bulge_emcee",
    "fit_disk_emcee",
    "fit_halo_emcee",
    "fit_two_power_emcee",
    "fit_three_power_emcee",
    "fit_double_exponential_disk_emcee",
    "fit_cored_bulge_emcee",
    "fit_exp_emcee",
    "fit_single_disk_flaring_emcee",
    "fit_double_disk_flaring_emcee",
    "fit_exp_cutoff_emcee",
    "fit_general_emcee",
    # Plotting
    "plot_1d_density",
    "plot_2d_density_with_marginals",
    "plot_disk_slices",
    "plot_results",
    "plot_corner",
    "plot_combined_components",
    "plot_galaxy_density_profile",
    "plot_2d_model_comparison",
    "plot_2d_total_model_comparison",
    "plot_disk_subtraction",
    "get_plot_config",
    "get_component_colors",
    "apply_axis_style",
    "plot_profiles",
    "plot_fitted_mass_evolution",
    "plot_data_mass_evolution",
    "plot_component_mass_evolution",
    "plot_component_mass_ratio_evolution",
    "plot_halo_shape_evolution",
    "plot_comp_projection_with_r200",
    "plot_reddish_2022_figure",
    "plot_shape_analysis",
    "plot_mass_error",
    # Analysis
    "spherical_average",
    "calculate_fit_statistics",
    "gelman_rubin_diagnostic",
    "compare_models",
    "save_results",
    "get_shape_tensor",
    "calculate_component_statistics",
    "compare_masses",
    "calculate_model_masses",
    "save_evolution_data",
    "generate_component_summary",
    "calc_masses_with_uncertainties",
    # Utilities
    "add_rotated_fields",
    "rotation_matrix_from_vectors",
    "create_profiles",
    "extract_density",
    "profile_eq_str",
    "add_gaussian_weight",
    "add_inverse_radius_weight",
    "apply_data_range_mask",
    # Density Profiles
    "find_virial_radius",
    "check_extended_mass",
    "iterative_center",
    "integrate_spherical_mass",
    "integrate_disk_mass",
    "interpolate_r200_from_metadata",
    # Workflow
    "DensityProfile",
    "FittingResult",
    "fit_component_models",
    "fit_stellar_components",
    "fit_gas_components",
    "fit_dm_components",
    "find_snapshots",
    "process_snapshot",
    "process_time_series",
    "generate_snapshot_profiles",
    # Config
    "ProjectPaths",
    "load_config",
    "setup_logging",
    # io
    "ensure_directory",
    "load_hdf5",
    "save_hdf5",
    "save_figure",
]
