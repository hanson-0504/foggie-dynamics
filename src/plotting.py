from pathlib import Path
from typing import TYPE_CHECKING, Dict, Optional, Any

import h5py
import matplotlib
import matplotlib.cm as cm
import matplotlib.colors as mcolors
import matplotlib.pyplot as plt
import numpy as np
import yt
import pandas as pd
from matplotlib.colors import LogNorm, SymLogNorm
from matplotlib.gridspec import GridSpec
from matplotlib.patches import Circle
from scipy.interpolate import interp1d

from .config import ProjectPaths
from .filters_and_utils import apply_data_range_mask
from .io import load_hdf5, save_figure
from .models import ModelRegistry
from .workflow import DensityProfile

matplotlib.use("Agg")

plt.style.use('/home/shanson/foggie-dynamics/config/paper.mplstyle')


if TYPE_CHECKING:
    from .workflow import FittingResult


def get_plot_config(
    config: Dict,
    plot_type: Optional[str] = 'default'
) -> Dict:
    """
    Get plotting configuration for a specific plot type.

    Merges default settings with plot-type-specific settings.
    Plot-type-specific settings override defaults.

    Parameters
    ----------
    config : dict
        Full configuration dictionary
    plot_type : str
        Type of plot ('comparison_2d', 'profile_1d', 'corner')

    Returns
    -------
    dict
        Merged configuration with defaults and plot-type-specific settings
    """
    plotting_config = config.get("plotting", {})

    # Start with default settings
    plot_cfg = plotting_config.get("default", {}).copy()

    # Override with plot-type-specific settings
    if plot_type in plotting_config:
        plot_cfg.update(plotting_config[plot_type])

    return plot_cfg


def get_component_colors(config: Dict, color_type: str = "structural") -> Dict[str, str]:
    """Get component color pallette from config."""
    plotting_config = config.get("plotting", {})
    colors_config = plotting_config.get("colors", {})
    defaults = {
        "structural": {'bulge': '#4C72B0', 'disk': '#DD8452', 'halo': '#55A868', 'total': '#C44E52'},
        "component_type": {'star': '#1f77b4', 'gas': '#2ca02c', 'dm': '#ff7f0e', 'total': '#000000'}
    }
    return colors_config.get(color_type, defaults.get(color_type, {}))


def apply_axis_style(ax, config: Dict, plot_type: str = "default"):
    """Apply consistent styling to axes (fontsize, grid)."""
    plt_cfg = get_plot_config(config, plot_type)
    ax.xaxis.label.set_fontsize(plt_cfg.get("fontsize_label", 10))
    ax.yaxis.label.set_fontsize(plt_cfg.get("fontsize_label", 10))
    ax.tick_params(axis='both', labelsize=plt_cfg.get("fontsize_tick", 9))


def plot_2d_density_with_marginals(
    R_bins,
    z_bins,
    rho,
    figsize=(12, 7),
    cmap="viridis",
    vmin=None,
    vmax=None,
    percentiles=[1, 99],
    log_R=True,
    log_z=True,
    log_colorbar=True,
    title=None,
    savefig=None,
    **kwargs,
):
    """
    Plot 2D density map with marginal histograms.

    Parameters:
    -----------
    R_bins : array
        Bin edges for radial coordinate (length N+1)
    z_bins : array
        Bin edges for vertical coordinate (length M+1)
    rho : array (N, M)
        2D density array
    figsize : tuple
        Figure size (width, height)
    cmap : str
        Colormap name
    vmin, vmax : float, optional
        Colorbar limits. If None, computed from percentiles
    percentiles : list
        [min, max] percentiles for automatic vmin/vmax
    log_R : bool
        Use log scale for R axis
    log_z : bool
        Use log scale for z axis
    log_colorbar : bool
        Use log scale for colorbar
    title : str, optional
        Figure title
    savefig : str, optional
        Path to save figure
    **kwargs : dict
        Additional arguments passed to pcolormesh

    Returns:
    --------
    fig, (ax_main, ax_top, ax_right, ax_cbar)
    """
    # Compute marginal distributions
    rho_R = np.nansum(rho, axis=1)  # sum over z
    rho_z = np.nansum(rho, axis=0)  # sum over R

    # compute colorbar limits if not provided
    if vmin is None or vmax is None:
        rho_positive = rho[rho > 0]
        if len(rho_positive) > 0:
            if hasattr(rho_positive, "value"):  # Handle unyt arrays
                vmin_calc, vmax_calc = np.percentile(rho_positive.value, percentiles)
                vmin = vmin or (vmin_calc * rho.units)
                vmax = vmax or (vmax_calc * rho.units)
            else:
                vmin_calc, vmax_calc = np.percentile(rho_positive, percentiles)
                vmin = vmin or vmin_calc
                vmax = vmax or vmax_calc
        else:
            vmin = vmin or rho.min()
            vmax = vmax or rho.max()

    # create figure with GridSpec
    fig = plt.figure(figsize=figsize)
    gs = GridSpec(
        2, 2, width_ratios=[4, 1], height_ratios=[1, 4], hspace=0.05, wspace=0.05
    )

    # Main density plot
    ax_main = fig.add_subplot(gs[1, 0])
    norm = LogNorm(vmin=vmin, vmax=vmax) if log_colorbar else None
    mesh = ax_main.pcolormesh(
        R_bins, z_bins, rho.T, cmap=cmap, norm=norm, shading="auto", **kwargs
    )
    ax_main.set_ylabel("Height z [kpc]")
    ax_main.set_xlabel("Cylindrical Radius R [kpc]")
    if log_R:
        ax_main.set_xscale("log")
    if log_z:
        ax_main.set_yscale("log")

    # Top histogram (radial)
    ax_top = fig.add_subplot(gs[0, 0], sharex=ax_main)
    ax_top.stairs(rho_R, R_bins, fill=True, alpha=0.7, edgecolor="black", linewidth=1.5)
    if log_R:
        ax_top.set_xscale("log")
    ax_top.set_yscale("log")
    if title:
        ax_top.set_title(title, fontsize=14)

    ax_top.tick_params(labelbottom=False)
    ax_top.set_ylabel(r"Total $\rho(R)$")

    # Right histogram (height)
    ax_right = fig.add_subplot(gs[1, 1], sharey=ax_main)
    ax_right.stairs(
        rho_z,
        z_bins,
        orientation="horizontal",
        fill=True,
        alpha=0.7,
        edgecolor="black",
        linewidth=1.5,
    )
    ax_right.set_xscale("log")
    if log_z:
        ax_right.set_yscale("log")

    ax_right.tick_params(labelleft=False)
    ax_right.set_xlabel(r"Total $\rho(z)$", fontsize=12)

    # Colorbar in empty corner
    ax_cbar = fig.add_subplot(gs[0, 1])
    cbar = plt.colorbar(mesh, cax=ax_cbar, label="Density [Msun/kpc**3]")
    cbar.ax.tick_params(labelsize=9)

    if savefig:
        save_figure(fig, savefig)
    return fig, (ax_main, ax_top, ax_right, ax_cbar)


def plot_2d_model_comparison(
    R_bins,
    z_bins,
    rho_data,
    model_evaluator,
    model_summary,
    model_name,
    config: Optional[Dict] = None,
    figsize=(6.27, 4.7),
    cmap="viridis",
    vmin=None,
    vmax=None,
    percentiles=[1, 99],
    log_R=True,
    log_z=True,
    log_colorbar=True,
    title=None,
    savefig=None,
    max_residual=None,
    dpi=300,
    **kwargs,
):
    """
    Plot 2D density data, best-fit model, and residuals side-by-side with marginals.

    Parameters:
    -----------
    R_bins : array
        Bin edges for radial coordinate (length N+1)
    z_bins : array
        Bin edges for vertical coordinate (length M+1)
    rho_data : array (N, M)
        2D observed density array
    model_evaluator : callable
        Function that evaluates model: evaluator((R, z), summary) -> rho_model
    model_summary : dict
        Model parameters
    model_name : str
        Name of the best model
    config : dict, optional
        Full configuration dictionary. If provided, plotting settings are extracted automatically.
        Individual parameters below can override config settings.
    figsize : tuple
        Figure size (width, height)
    cmap : str
        Colormap name
    vmin, vmax : float, optional
        Colorbar limits for data and model. If None, computed from data percentiles
    percentiles : list
        [min, max] percentiles for automatic vmin/vmax
    log_R : bool
        Use log scale for R axis
    log_z : bool
        Use log scale for z axis
    log_colorbar : bool
        Use log scale for colorbar
    title : str, optional
        Figure suptitle
    savefig : str, optional
        Path to save figure
    max_residual : float, optional
        Maximum absolute residual value for colorbar. Values beyond this will be clipped.
        If None, uses 95th percentile of absolute residuals.
    dpi : int, optional
        Resolution for saved figures. Default: 300
    **kwargs : dict
        Additional arguments passed to pcolormesh

    Returns:
    --------
    fig
    """
    # Extract plotting settings from config if provided
    if config is not None:
        plot_cfg = get_plot_config(config, "comparison_2d")
        # Use config values as defaults, but allow explicit parameters to override
        figsize = kwargs.pop("figsize", None) or tuple(plot_cfg.get("figsize", figsize))
        cmap = kwargs.pop("cmap", None) or plot_cfg.get("cmap", cmap)
        percentiles = kwargs.pop("percentiles", None) or plot_cfg.get(
            "percentiles", percentiles
        )
        log_R = (
            kwargs.pop("log_R", None)
            if "log_R" in kwargs
            else plot_cfg.get("log_R", log_R)
        )
        log_z = (
            kwargs.pop("log_z", None)
            if "log_z" in kwargs
            else plot_cfg.get("log_z", log_z)
        )
        log_colorbar = (
            kwargs.pop("log_colorbar", None)
            if "log_colorbar" in kwargs
            else plot_cfg.get("log_colorbar", log_colorbar)
        )
        max_residual = kwargs.pop("max_residual", None) or plot_cfg.get(
            "max_residual", max_residual
        )
        dpi = kwargs.pop("dpi", None) or plot_cfg.get("dpi", dpi)

    # Evaluate model on grid
    R_centers = 0.5 * (R_bins[:-1] + R_bins[1:])
    z_centers = 0.5 * (z_bins[:-1] + z_bins[1:])
    R_grid, z_grid = np.meshgrid(R_centers, z_centers, indexing="ij")
    R_flat = R_grid.flatten()
    z_flat = z_grid.flatten()

    # Get model geometry and convert coordinates if needed
    geometry = ModelRegistry.get_geometry(model_name)
    if geometry == "spherical":
        # Convert cylindrical (R, z) to spherical radius r
        r_flat = np.sqrt(R_flat**2 + z_flat**2)
        rho_model_flat = model_evaluator((r_flat,), model_summary)
    else:  # cylindrical
        rho_model_flat = model_evaluator((R_flat, z_flat), model_summary)
    rho_model = rho_model_flat.reshape(R_grid.shape)

    # Compute residuals (fractional)
    rho_residual = (rho_data - rho_model) / np.where(rho_data > 0, rho_data, 1)

    # Compute colorbar limits if not provided (use data for reference)
    if vmin is None or vmax is None:
        rho_positive = rho_data[rho_data > 0]
        if len(rho_positive) > 0:
            if hasattr(rho_positive, "value"):
                vmin_calc, vmax_calc = np.percentile(rho_positive.value, percentiles)
                vmin = vmin or (vmin_calc * rho_data.units)
                vmax = vmax or (vmax_calc * rho_data.units)
            else:
                vmin_calc, vmax_calc = np.percentile(rho_positive, percentiles)
                vmin = vmin or vmin_calc
                vmax = vmax or vmax_calc
        else:
            vmin = vmin or rho_data.min()
            vmax = vmax or rho_data.max()

    # Create figure with 3 columns (data, model, residual)
    fig = plt.figure(figsize=figsize)

    # Panel titles
    panel_titles = ["Data", f"Best Model: {model_name}", "Frac. Residual"]
    data_arrays = [rho_data, rho_model, None]  # residuals handled separately

    for i, (panel_title, data_array) in enumerate(zip(panel_titles, data_arrays)):
        # Create GridSpec for this panel
        left = 0.05 + i * 0.32
        gs = GridSpec(
            2,
            2,
            left=left,
            right=left + 0.28,
            bottom=0.18,
            top=0.87,
            width_ratios=[4, 1],
            height_ratios=[1, 4],
            hspace=0.05,
            wspace=0.05,
        )

        # Main density plot
        ax_main = fig.add_subplot(gs[1, 0])

        if i < 2:  # Data or Model
            norm = LogNorm(vmin=vmin, vmax=vmax) if log_colorbar else None
            mesh = ax_main.pcolormesh(
                R_bins,
                z_bins,
                data_array.T,
                cmap=cmap,
                norm=norm,
                shading="auto",
                **kwargs,
            )
        else:  # Residuals
            # Use symmetric log colorbar for residuals
            if max_residual is not None:
                res_max = max_residual
            else:
                res_max = np.nanpercentile(np.abs(rho_residual), 95)

            # Clip residuals to prevent extreme values from swamping colorbar
            rho_residual_clipped = np.clip(rho_residual, -res_max, res_max)

            # Linear threshold: values smaller than this are scaled linearly
            linthresh = res_max / 100.0  # 1% of max for linear region near zero
            norm_residual = SymLogNorm(
                linthresh=linthresh, vmin=-res_max, vmax=res_max, base=10
            )
            mesh = ax_main.pcolormesh(
                R_bins,
                z_bins,
                rho_residual_clipped.T,
                cmap="RdBu_r",
                norm=norm_residual,
                shading="auto",
                **kwargs,
            )

        ax_main.set_ylabel("Height z [kpc]" if i == 0 else "")
        ax_main.set_xlabel("Cylindrical Radius R [kpc]")
        if log_R:
            ax_main.set_xscale("log")
        if log_z:
            ax_main.set_yscale("log")
        if i > 0:
            ax_main.tick_params(labelleft=False)
        apply_axis_style(ax_main, config or {}, 'comparison_2d')

        # Compute marginals
        if i < 2:
            rho_R = np.nansum(data_array, axis=1)
            rho_z = np.nansum(data_array, axis=0)
        else:
            rho_R = np.nansum(rho_residual, axis=1)
            rho_z = np.nansum(rho_residual, axis=0)

        # Top histogram (radial)
        ax_top = fig.add_subplot(gs[0, 0], sharex=ax_main)
        ax_top.stairs(
            rho_R, R_bins, fill=True, alpha=0.7, edgecolor="black", linewidth=1.5
        )
        if log_R:
            ax_top.set_xscale("log")
        if i < 2:
            ax_top.set_yscale("log")
        ax_top.tick_params(labelbottom=False)
        ax_top.set_title(panel_title, fontsize=12, fontweight="bold", pad=4)
        if i == 0:
            ax_top.set_ylabel(r"$\rho(R)$", fontsize=12)

        # Right histogram (height)
        ax_right = fig.add_subplot(gs[1, 1], sharey=ax_main)
        ax_right.stairs(
            rho_z,
            z_bins,
            orientation="horizontal",
            fill=True,
            alpha=0.7,
            edgecolor="black",
            linewidth=1.5,
        )
        if i < 2:
            ax_right.set_xscale("log")
        if log_z:
            ax_right.set_yscale("log")
        ax_right.tick_params(labelleft=False)
        if i == 0:
            ax_right.xaxis.set_label_position("top")
            ax_right.set_xlabel(r"$\rho(z)$", fontsize=10, labelpad=3)

        # Colorbar
        ax_cbar = fig.add_subplot(gs[0, 1])
        cbar = plt.colorbar(mesh, cax=ax_cbar)
        cbar.ax.tick_params(labelsize=9)

    if title:
        fig.suptitle(title, fontsize=14, fontweight="bold", y=0.98)

    if savefig:
        save_figure(fig, savefig, dpi=dpi)

    return fig


def plot_2d_total_model_comparison(
    R_bins,
    z_bins,
    rho_data,
    components_dict,  # {'bulge': {'evaluator': func, 'summary': dict, 'model_name': str}, ...}
    config: Optional[Dict] = None,  # Full config dict - recommended approach
    figsize=(18, 7),
    cmap="viridis",
    vmin=None,
    vmax=None,
    percentiles=[1, 99],
    log_R=True,
    log_z=True,
    log_colorbar=True,
    title=None,
    savefig=None,
    max_residual=None,
    dpi=300,
    **kwargs,
):
    """
    Plot 2D density data with total model (sum of bulge+disk+halo) and residuals.

    Shows three panels side-by-side: Data | Total Model | Residual

    Parameters:
    -----------
    R_bins : array
        Bin edges for radial coordinate (length N+1)
    z_bins : array
        Bin edges for vertical coordinate (length M+1)
    rho_data : array (N, M)
        2D observed density array
    components_dict : dict
        Dictionary with component names as keys ('bulge', 'disk', 'halo') and values as dicts containing:
            - 'evaluator': callable model evaluator function
            - 'summary': dict of model parameters
            - 'model_name': str name of the model
    config : dict, optional
        Full configuration dictionary. If provided, plotting settings are extracted automatically.
        Individual parameters below can override config settings.
    figsize : tuple
        Figure size (width, height)
    cmap : str
        Colormap name for density plots
    vmin, vmax : float, optional
        Colorbar limits. If None, computed from data percentiles
    percentiles : list
        [min, max] percentiles for automatic vmin/vmax
    log_R : bool
        Use log scale for R axis
    log_z : bool
        Use log scale for z axis
    log_colorbar : bool
        Use log scale for colorbar
    title : str, optional
        Figure suptitle
    savefig : str, optional
        Path to save figure
    max_residual : float, optional
        Maximum absolute residual value for colorbar. Values beyond this will be clipped.
        If None, uses 95th percentile of absolute residuals.
    dpi : int, optional
        Resolution for saved figures. Default: 300
    **kwargs : dict
        Additional arguments passed to pcolormesh

    Returns:
    --------
    fig
    """
    # Extract plotting settings from config if provided
    if config is not None:
        plot_cfg = get_plot_config(config, "comparison_2d")
        # Use config values as defaults, but allow explicit parameters to override
        figsize = kwargs.pop("figsize", None) or tuple(plot_cfg.get("figsize", figsize))
        cmap = kwargs.pop("cmap", None) or plot_cfg.get("cmap", cmap)
        percentiles = kwargs.pop("percentiles", None) or plot_cfg.get(
            "percentiles", percentiles
        )
        log_R = (
            kwargs.pop("log_R", None)
            if "log_R" in kwargs
            else plot_cfg.get("log_R", log_R)
        )
        log_z = (
            kwargs.pop("log_z", None)
            if "log_z" in kwargs
            else plot_cfg.get("log_z", log_z)
        )
        log_colorbar = (
            kwargs.pop("log_colorbar", None)
            if "log_colorbar" in kwargs
            else plot_cfg.get("log_colorbar", log_colorbar)
        )
        max_residual = kwargs.pop("max_residual", None) or plot_cfg.get(
            "max_residual", max_residual
        )
        dpi = kwargs.pop("dpi", None) or plot_cfg.get("dpi", dpi)

    # Filter out None components
    available_components = {
        k: v
        for k, v in components_dict.items()
        if v is not None and v.get("summary") is not None
    }

    if len(available_components) == 0:
        raise ValueError("No valid components provided")

    # Prepare grid for evaluation
    R_centers = 0.5 * (R_bins[:-1] + R_bins[1:])
    z_centers = 0.5 * (z_bins[:-1] + z_bins[1:])
    R_grid, z_grid = np.meshgrid(R_centers, z_centers, indexing="ij")
    R_flat = R_grid.flatten()
    z_flat = z_grid.flatten()

    # Evaluate each component and sum to get total model
    rho_total = np.zeros_like(rho_data)
    component_names = []

    for comp_name, comp_info in available_components.items():
        evaluator = comp_info["evaluator"]
        summary = comp_info["summary"]
        model_name = comp_info.get("model_name", comp_name)
        try:
            # Get model geometry and convert coordinates if needed
            geometry = ModelRegistry.get_geometry(model_name)
            if geometry == "spherical":
                # Convert cylindrical (R, z) to spherical radius r
                r_flat = np.sqrt(R_flat**2 + z_flat**2)
                rho_flat = evaluator((r_flat,), summary)
            else:  # cylindrical
                rho_flat = evaluator((R_flat, z_flat), summary)
            rho_component = rho_flat.reshape(R_grid.shape)
            rho_total += rho_component
            component_names.append(f"{comp_name}({model_name})")
        except Exception as e:
            print(f"Warning: Could not evaluate {comp_name} model: {e}")

    # Compute residuals
    rho_residual = (rho_data - rho_total) / np.where(rho_data > 0, rho_data, 1)

    # Compute colorbar limits from data if not provided
    if vmin is None or vmax is None:
        rho_positive = rho_data[rho_data > 0]
        if len(rho_positive) > 0:
            if hasattr(rho_positive, "value"):
                vmin_calc, vmax_calc = np.percentile(rho_positive.value, percentiles)
                vmin = vmin or (vmin_calc * rho_data.units)
                vmax = vmax or (vmax_calc * rho_data.units)
            else:
                vmin_calc, vmax_calc = np.percentile(rho_positive, percentiles)
                vmin = vmin or vmin_calc
                vmax = vmax or vmax_calc
        else:
            vmin = vmin or rho_data.min()
            vmax = vmax or rho_data.max()

    # Create figure with 3 columns
    fig = plt.figure(figsize=figsize)

    # Build model name string
    model_str = " + ".join(component_names) if component_names else "Total Model"

    # Panel titles
    panel_titles = ["Data", f"Total Model\n{model_str}", "Frac. Residual"]
    data_arrays = [rho_data, rho_total, None]  # residuals handled separately

    for i, (panel_title, data_array) in enumerate(zip(panel_titles, data_arrays)):
        # Create GridSpec for this panel
        left = 0.05 + i * 0.32
        gs = GridSpec(
            2,
            2,
            left=left,
            right=left + 0.28,
            bottom=0.18,
            top=0.87,
            width_ratios=[4, 1],
            height_ratios=[1, 4],
            hspace=0.05,
            wspace=0.05,
        )

        # Main density plot
        ax_main = fig.add_subplot(gs[1, 0])

        if i < 2:  # Data or Model
            norm = LogNorm(vmin=vmin, vmax=vmax) if log_colorbar else None
            mesh = ax_main.pcolormesh(
                R_bins,
                z_bins,
                data_array.T,
                cmap=cmap,
                norm=norm,
                shading="auto",
                **kwargs,
            )
        else:  # Residuals
            # Use symmetric log colorbar for residuals
            if max_residual is not None:
                res_max = max_residual
            else:
                res_max = np.nanpercentile(np.abs(rho_residual), 95)

            # Clip residuals to prevent extreme values from swamping colorbar
            rho_residual_clipped = np.clip(rho_residual, -res_max, res_max)

            # Linear threshold: values smaller than this are scaled linearly
            linthresh = res_max / 100.0  # 1% of max for linear region near zero
            norm_residual = SymLogNorm(
                linthresh=linthresh, vmin=-res_max, vmax=res_max, base=10
            )
            mesh = ax_main.pcolormesh(
                R_bins,
                z_bins,
                rho_residual_clipped.T,
                cmap="RdBu_r",
                norm=norm_residual,
                shading="auto",
                **kwargs,
            )

        ax_main.set_ylabel("Height z [kpc]" if i == 0 else "")
        ax_main.set_xlabel("Cylindrical Radius R [kpc]")
        if log_R:
            ax_main.set_xscale("log")
        if log_z:
            ax_main.set_yscale("log")
        if i > 0:
            ax_main.tick_params(labelleft=False)
        apply_axis_style(ax_main, config or {}, 'comparison_2d')

        # Compute marginals
        if i < 2:
            rho_R = np.nansum(data_array, axis=1)
            rho_z = np.nansum(data_array, axis=0)
        else:
            rho_R = np.nansum(rho_residual, axis=1)
            rho_z = np.nansum(rho_residual, axis=0)

        # Top histogram (radial)
        ax_top = fig.add_subplot(gs[0, 0], sharex=ax_main)
        ax_top.stairs(
            rho_R, R_bins, fill=True, alpha=0.7, edgecolor="black", linewidth=1.5
        )
        if log_R:
            ax_top.set_xscale("log")
        if i < 2:
            ax_top.set_yscale("log")
        ax_top.tick_params(labelbottom=False)
        ax_top.set_title(panel_title, fontsize=12, fontweight="bold", pad=4)
        if i == 0:
            ax_top.set_ylabel(r"$\Sigma\,\rho(R)$", fontsize=12)

        # Right histogram (height)
        ax_right = fig.add_subplot(gs[1, 1], sharey=ax_main)
        ax_right.stairs(
            rho_z,
            z_bins,
            orientation="horizontal",
            fill=True,
            alpha=0.7,
            edgecolor="black",
            linewidth=1.5,
        )
        if i < 2:
            ax_right.set_xscale("log")
        if log_z:
            ax_right.set_yscale("log")
        ax_right.tick_params(labelleft=False)
        if i == 0:
            ax_right.xaxis.set_label_position("top")
            ax_right.set_xlabel(r"$\Sigma\,\rho(z)$", fontsize=10, labelpad=3)

        # Colorbar
        ax_cbar = fig.add_subplot(gs[0, 1])
        cbar = plt.colorbar(mesh, cax=ax_cbar)
        cbar.ax.tick_params(labelsize=9)
        if i < 2:
            cbar.set_label(r"$\rho\ [\mathrm{M}_\odot/\mathrm{kpc}^3]$", fontsize=10)
        else:
            cbar.set_label("Frac. Residual", fontsize=10)

    if title:
        fig.suptitle(title, fontsize=14, fontweight="bold", y=0.98)

    if savefig:
        save_figure(fig, savefig, dpi=dpi)

    return fig


def plot_1d_density(
    r_bins,
    rho,
    figsize=(6.27,5.0),
    title=None,
    savefig=None,
    log_rho=True,
    log_r=True,
    label=None,
    **kwargs,
):
    """
    Plot 1D radial density profile.

    Parameters:
    -----------
    r_bins : array
        Bin edges for radial coordinate (length N+1)
    rho : array
        Density values (length N)
    figsize : tuple
        Figure size (width, height)
    title : str, optional
        Plot title
    savefig : str, optional
        Path to save figure
    log_rho : bool
        Use log scale for density axis
    log_r : bool
        Use log scale for radius axis
    **kwargs : dict
        Additional arguments passed to plot()

    Returns:
    --------
    fig, ax
    """
    # Compute bin centers from edges
    r_centers = 0.5 * (r_bins[:-1] + r_bins[1:])
    if hasattr(r_centers, "to"):
        r_kpc = r_centers.to("kpc").value
    else:
        r_kpc = np.asarray(r_centers)
    if hasattr(rho, "to"):
        rho_values = rho.to("Msun/kpc**3").value
    else:
        rho_values = np.asarray(rho)

    fig = plt.figure(figsize=figsize)
    ax = fig.add_subplot()
    ax.plot(r_kpc, rho_values, label=label, **kwargs)
    if log_r:
        ax.set_xscale("log")
    if log_rho:
        ax.set_yscale("log")
    ax.set_ylabel("Density [Msun/kpc**3]", fontsize=12)
    ax.set_xlabel("Spherical Radius r [kpc]", fontsize=12)
    if title:
        ax.set_title(title, fontsize=14)
    if savefig:
        save_figure(fig, savefig)
    return fig, ax


def plot_results(
    coords,
    data,
    data_range,
    model_summaries,
    model_evaluators,
    best_model_name,
    component_name,
    coord_label,
    config,
    log_scale=True,
    save_path=None,
    title=None,
):
    """
    Plot 1D density profile with all fitted models, highlighting the best.

    Parameters
    ----------
    coords : tuple of arrays
        (r,) for 1D spherical profiles
    data : array
        Observed density
    model_summaries : dict
        {model_name: summary_dict}
    model_evaluators : dict
        {model_name: callable(coords, summary) -> model_prediction}
    best_model_name : str
        Name of best-fitting model
    data_range : dict
        Coordinate ranges for masking
    component_name : str
        'Bulge', 'Disk', or 'Halo'
    coord_label : str
        Label for x-axis (e.g., 'r [kpc]')
    log_scale : bool, optional
        Use log-log scale. Default: True
    save_path : str or Path, optional
        If provided, save figure to this path

    Returns
    -------
    fig : matplotlib.figure.Figure
        The generated figure
    """
    plt_cfg = get_plot_config(config, 'results_1d')
    fig, axes = plt.subplots(
        2, 1, figsize=plt_cfg.get('figsize', (7, 6)), sharex=True, gridspec_kw={"height_ratios": [3, 1]}
    )
    # Apply mask
    mask = apply_data_range_mask(coords, data, data_range)
    coords_plot = tuple(c[mask] for c in coords)
    data_plot = data[mask]

    # Check if any valid data points remain for log scaling
    if len(data_plot) == 0:
        print(f"Warning: No data points for {component_name} after masking. Skipping plot.")
        plt.close(fig)
        return None

    if log_scale and (not np.any(data_plot > 0) or not np.any(coords_plot[0] > 0)):
        print(f"Warning: No positive values for {component_name} (required for log scale). Skipping plot.")
        plt.close(fig)
        return None

    # Top panel: data and models
    ax = axes[0]
    apply_axis_style(ax, config, 'results_1d')
    ax.scatter(
        coords_plot[0], data_plot, s=20, c="k", alpha=0.5, label="Data", zorder=1
    )

    colors = plt.cm.tab10(range(len(model_summaries)))
    for i, (name, summary) in enumerate(model_summaries.items()):
        if summary is None:
            continue
        elif name == 'best_model':
            continue
        try:
            model_pred = model_evaluators[name](coords_plot, summary)
            if name == best_model_name:
                ax.plot(
                    coords_plot[0],
                    model_pred,
                    c=colors[i],
                    lw=3,
                    label=f"{name} (best)",
                    zorder=10,
                )

                # Add uncertainty bands for best model if samples are available
                if "samples" in summary and summary["samples"] is not None:
                    samples = summary["samples"]
                    n_samples = len(samples)

                    # Get parameter names from ModelRegistry
                    from .models import ModelRegistry

                    param_names = ModelRegistry.get_param_names(name)

                    # Check that samples match expected parameters
                    # Samples include an extra column for log_sigma at the end
                    if samples.shape[1] < len(param_names):
                        # Not enough samples to map to parameters, skip uncertainty band
                        pass
                    else:
                        # Randomly select subset of posterior samples
                        n_draws = min(200, n_samples)
                        indices = np.random.choice(
                            n_samples, size=n_draws, replace=False
                        )

                        # Evaluate model for each sample
                        model_draws = []

                        # IMPORTANT: MCMC samples are stored in log10 space for density/scale parameters
                        # but ModelRegistry param_names may use linear names (e.g., 'rho0' instead of 'log10_rho0')
                        # We need to convert samples from log10 to linear space

                        # Parameters that are sampled in log10 space
                        log_params = {
                            "rho0",
                            "rs",
                            "rc",
                            "rd",
                            "R_d",
                            "R_d1",
                            "R_d2",
                            "z_0",
                            "rho01",
                            "rho02",
                            "rho_0",
                            "r_s",
                            "r_c",
                            "rho_01",
                            "rho_02",
                        }

                        for draw_idx, idx in enumerate(indices):
                            temp_summary = summary.copy()

                            # Extract parameters from this sample
                            for param_idx, param_name in enumerate(param_names):
                                sample_val = samples[idx, param_idx]

                                # Check if this parameter was sampled in log10 space
                                # (most density and scale parameters are)
                                param_base = param_name.replace(
                                    "_", ""
                                )  # Handle variations like r_s vs rs

                                if param_name in log_params or param_base in log_params:
                                    # Sample is in log10 space, convert to linear
                                    linear_val = 10**sample_val
                                    temp_summary[param_name] = linear_val

                                    # Also store the log10 version
                                    log_name = f"log10_{param_name}"
                                    temp_summary[log_name] = sample_val

                                    # Handle underscore variations
                                    if "_" in param_name:
                                        no_underscore = param_name.replace("_", "")
                                        temp_summary[no_underscore] = linear_val
                                        temp_summary[f"log10_{no_underscore}"] = (
                                            sample_val
                                        )
                                else:
                                    # Parameter sampled in linear space (e.g., alpha, beta, A, B)
                                    temp_summary[param_name] = sample_val

                            if draw_idx == 0:  # Debug first sample
                                for pname in param_names:
                                    print(f"  {pname} = {temp_summary[pname]:.3e}")

                            try:
                                model_draw = model_evaluators[name](
                                    coords_plot, temp_summary
                                )
                                model_draws.append(model_draw)
                                if draw_idx == 0:
                                    print(
                                        f"  Model eval range: [{model_draw.min():.3e}, {model_draw.max():.3e}]"
                                    )
                            except Exception as e:
                                # Skip failed evaluations
                                if draw_idx == 0:
                                    print(f"  Model eval FAILED: {e}")
                                    import traceback

                                    traceback.print_exc()
                                continue

                        if len(model_draws) > 10:  # Need enough samples
                            model_draws = np.array(model_draws)

                            # Calculate percentiles (16th and 84th = 1-sigma for Gaussian)
                            lower = np.percentile(model_draws, 16, axis=0)
                            upper = np.percentile(model_draws, 84, axis=0)

                            # Plot 1-sigma band with increased alpha for better visibility
                            ax.fill_between(
                                coords_plot[0],
                                lower,
                                upper,
                                color=colors[i],
                                alpha=0.25,
                                zorder=9,
                                label=f"{name} $1\\sigma$",
                            )
            else:
                ax.plot(
                    coords_plot[0],
                    model_pred,
                    lw=1.5,
                    ls="--",
                    c=colors[i],
                    label=name,
                    alpha=0.7,
                    zorder=5,
                )
        except Exception as e:
            print(f"Error evaluating model {name} for plotting: {e}")

    ax.set_ylabel(r"$\rho$ [M$_\odot$ kpc$^{-3}$]", fontsize=12)
    if title is None:
        ax.set_title(f"{component_name.capitalize()} Density Profile Fitting", fontsize=14)
    else:
        ax.set_title(title, fontsize=14)
    ax.legend(loc="best", framealpha=0.9)
    if log_scale:
        ax.set_yscale("log")
        ax.set_xscale("log")

    # Bottom panel: residuals
    ax = axes[1]
    apply_axis_style(ax, config, 'results_1d')
    for i, (name, summary) in enumerate(model_summaries.items()):
        if summary is None:
            continue
        elif name == 'best_model':
            continue
        try:
            model_pred = model_evaluators[name](coords_plot, summary)
            residual = (data_plot - model_pred) / data_plot  # fractional

            if name == best_model_name:
                ax.scatter(
                    coords_plot[0],
                    residual,
                    s=15,
                    color=colors[i],
                    marker="o",
                    zorder=10,
                )
            else:
                ax.scatter(
                    coords_plot[0],
                    residual,
                    s=10,
                    color=colors[i],
                    marker="x",
                    alpha=0.5,
                    zorder=5,
                )
        except Exception as e:
            print(f"Error evaluating model {name} for residual plotting: {e}")

    ax.axhline(0, c="k", ls="--", lw=1)
    ax.set_ylabel("Residual\n(data-model)/data", fontsize=10)
    ax.set_xlabel(coord_label, fontsize=12)  # residual sub-panel: label intentionally smaller
    if log_scale:
        ax.set_xscale("log")
    ax.set_ylim(-1, 1)

    plt.tight_layout()

    if save_path:
        save_figure(fig, save_path, dpi=plt_cfg.get('dpi', 200))
    return fig


def plot_disk_slices(
    R_grid,
    z_grid,
    rho_2d,
    model_summaries,
    model_evaluators,
    best_model_name,
    z_slices=[0.0, 1.0, 3.0],
    R_range=(3, 30),
    component_name="Disk",
    save_path=None,
):
    """
    Plot radial density profiles at different vertical heights for disk models.

    Shows rho(R) at fixed z values to visualize disk breaks and model fits.
    """
    n_slices = len(z_slices)
    fig, axes = plt.subplots(n_slices, 1, figsize=(12, 3.5 * n_slices), sharex=True)
    if n_slices == 1:
        axes = [axes]

    colors = plt.cm.tab10(range(len(model_summaries)))

    for i, z_slice in enumerate(z_slices):
        ax = axes[i]

        # Find closest z index in data
        z_idx = np.argmin(np.abs(z_grid[0, :] - z_slice))
        z_actual = z_grid[0, z_idx]

        # Extract radial profile at this height
        R_data = R_grid[:, z_idx]
        rho_data = rho_2d[:, z_idx]

        # Apply R range mask
        mask = apply_data_range_mask((R_data,), rho_data, {0: R_range})
        R_plot = R_data[mask]
        rho_plot = rho_data[mask]

        # Plot data
        ax.scatter(R_plot, rho_plot, s=30, color="k", alpha=0.6, label="Data", zorder=1)
        # Plot models
        R_smooth = np.linspace(R_range[0], R_range[1], 200)
        z_smooth = np.full_like(R_smooth, z_actual)

        for j, (name, summary) in enumerate(model_summaries.items()):
            if summary is None:
                continue
            try:
                # Get model geometry and convert coordinates if needed
                geometry = ModelRegistry.get_geometry(name)
                if geometry == "spherical":
                    # Convert cylindrical (R, z) to spherical radius r
                    r_smooth = np.sqrt(R_smooth**2 + z_smooth**2)
                    rho_model = model_evaluators[name]((r_smooth,), summary)
                else:  # cylindrical
                    rho_model = model_evaluators[name]((R_smooth, z_smooth), summary)

                if name == best_model_name:
                    ax.plot(
                        R_smooth,
                        rho_model,
                        color=colors[j],
                        lw=3,
                        label=f"{name} (best)",
                        zorder=10,
                    )
                else:
                    ax.plot(
                        R_smooth,
                        rho_model,
                        color=colors[j],
                        lw=1.5,
                        ls="--",
                        alpha=0.7,
                        label=name,
                        zorder=5,
                    )
            except Exception as e:
                print(f"Error plotting {name} at z={z_slice}: {e}")

        # Formatting
        ax.set_ylabel(r"$\rho$ [M$_\odot$ kpc$^{-3}$]", fontsize=12)
        ax.set_yscale("log")
        ax.legend(loc="upper right", framealpha=0.9, fontsize=10)
        ax.text(
            0.02,
            0.95,
            f"z = {z_actual} kpc",
            transform=ax.transAxes,
            fontsize=12,
            verticalalignment="top",
            bbox=dict(boxstyle="round", facecolor="white", alpha=0.5),
        )

        if i == 0:
            ax.set_title(
                f"{component_name} Radial Profiles at different heights",
                fontsize=14,
                fontweight="bold",
            )

    axes[-1].set_xlabel("R [kpc]", fontsize=12)
    if save_path:
        save_figure(fig, save_path)

    return fig


def plot_combined_components(
    results: Dict[str, "FittingResult"],  # dict with 'bulge', 'disk', 'halo' keys
    profile: DensityProfile,
    component_type="star",  # 'star', 'gas', 'dm'
    r_range=(1, 200),  # radius range to plot
    save_path=None,
    config: Optional[Dict] = None,
):
    """
    Plot combined density profile showing all components + total fit.

    Shows:
    - Data (spherical profile)
    - Bulge model (spherical)
    - Disk model (evaluated at z=0)
    - Halo model (spherical)
    - Total = bulge + disk + halo
    - Residual panel (data - total) / data
    """
    plt_cfg = get_plot_config(config or {}, 'combined_1d')
    fig, axes = plt.subplots(
        2, 1, sharex=True, gridspec_kw={"height_ratios": [3, 1]},
        figsize=plt_cfg.get('figsize', [7, 6])
    )
    ax_main = axes[0]
    ax_resid = axes[1]
    apply_axis_style(ax_main, config or {}, 'combined_1d')
    apply_axis_style(ax_resid, config or {}, 'combined_1d')

    r_data = profile.r_1d
    rho_data = profile.rho_1d

    # Mask data to r_range
    mask = (
        (r_data >= r_range[0])
        & (r_data <= r_range[1])
        & (rho_data >= 0)
        & (np.isfinite(rho_data))
    )
    r_plot = r_data[mask]
    rho_plot = rho_data[mask]

    r_smooth = np.logspace(np.log10(r_range[0]), np.log10(r_range[1]), 200)

    # Initialize arrays to store model densities
    rho_bulge = np.zeros_like(r_smooth)
    rho_disk = np.zeros_like(r_smooth)
    rho_halo = np.zeros_like(r_smooth)

    # Initialize model names
    bulge_model = None
    disk_model = None
    halo_model = None

    # Evaluate bulge
    if "bulge" in results and results["bulge"] is not None:
        bulge_model = results["bulge"].best_model
        bulge_summary = results["bulge"].summary
        if bulge_model and bulge_summary:
            bulge_evaluator = ModelRegistry.get_evaluator(bulge_model)
            rho_bulge = bulge_evaluator((r_smooth,), bulge_summary)

    # Evaluate disk (spherically averaged from 2D)
    if "disk" in results and results["disk"] is not None:
        disk_model = results["disk"].best_model
        disk_summary = results["disk"].summary
        if disk_model and disk_summary:
            from .analysis import spherically_average_model

            disk_evaluator = ModelRegistry.get_evaluator(disk_model)
            rho_disk = spherically_average_model(disk_evaluator, r_smooth, disk_summary)

    # Evaluate halo
    if "halo" in results and results["halo"] is not None:
        halo_model = results["halo"].best_model
        halo_summary = results["halo"].summary
        if halo_model and halo_summary:
            halo_evaluator = ModelRegistry.get_evaluator(halo_model)
            rho_halo = halo_evaluator((r_smooth,), halo_summary)

    rho_total = rho_bulge + rho_disk + rho_halo

    ax_main.scatter(
        r_plot, rho_plot, s=30, c="k", alpha=0.6, label="Data (spherical)", zorder=1
    )

    # Plot individual components with different styles
    ax_main.plot(
        r_smooth,
        rho_bulge,
        ls="--",
        lw=2,
        alpha=0.7,
        label=f"Bulge ({bulge_model or 'N/A'})",
        color="C0",
        zorder=5,
    )
    ax_main.plot(
        r_smooth,
        rho_disk,
        ls="--",
        lw=2,
        alpha=0.7,
        label=f"Disc ({disk_model or 'N/A'})",
        color="C1",
        zorder=5,
    )
    ax_main.plot(
        r_smooth,
        rho_halo,
        ls="--",
        lw=2,
        alpha=0.7,
        label=f"Halo ({halo_model or 'N/A'})",
        color="C2",
        zorder=5,
    )

    # Plot total with thick solid line
    ax_main.plot(
        r_smooth, rho_total, ls="-", lw=3, label="Total Model", color="red", zorder=10
    )

    ax_main.set_ylabel(r"$\rho$ [M$_\odot$ kpc$^{-3}$]", fontsize=12)
    ax_main.set_yscale("log")
    ax_main.set_xscale("log")
    ax_main.set_ylim(max(min(rho_plot), 1), max(rho_plot) * 5)
    ax_main.legend(loc="best", framealpha=0.9, fontsize=10)
    ax_main.set_title(
        f"{component_type.capitalize()} Component Decomposition",
        fontsize=14,
        fontweight="bold",
    )

    # Interpolate total model onto data points for residuals
    total_interp = interp1d(
        r_smooth, rho_total, kind="cubic", bounds_error=False, fill_value="extrapolate"
    )
    rho_total_at_data = total_interp(r_plot)

    # Fractional residuals (handle divide by zero)
    residual = np.divide(
        rho_plot - rho_total_at_data,
        rho_plot,
        where=rho_plot != 0,
        out=np.zeros_like(rho_plot),
    )

    ax_resid.scatter(r_plot, residual, s=20, c="k", alpha=0.6)
    ax_resid.axhline(0, c="red", ls="--", lw=2, zorder=5)
    ax_resid.set_ylabel("Residual\n(data-model)/data", fontsize=10)
    ax_resid.set_xlabel("r [kpc]", fontsize=12)
    ax_resid.set_ylim(-1, 1)
    ax_resid.grid(True, alpha=0.3)

    plt.tight_layout()

    if save_path:
        save_figure(fig, save_path)

    return fig


def plot_disk_subtraction(r_1d, rho_1d, rho_disk, rho_bulge, save_path=None):
    """Plot total, disk, and residual profiles."""
    fig, ax = plt.subplots(figsize=(12, 6))

    ax.loglog(r_1d, rho_1d, "o", label="Total (data)", alpha=0.6)
    ax.loglog(r_1d, rho_disk, "-", label="Disk model", lw=2)
    ax.loglog(r_1d, rho_bulge, "s", label="Bulge model", alpha=0.6)

    ax.set_xlabel("r [kpc]")
    ax.set_ylabel(r"$\rho$ [M$_\odot$ kpc$^{-3}$]")
    ax.legend()
    ax.grid(alpha=0.3)

    if save_path:
        save_figure(fig, save_path)

    return fig


def plot_corner(model_name, summary, save_path=None, **corner_kwargs):
    """
    Create a corner plot showing MCMC posterior distributions.

    Parameters
    ----------
    model_name : str
        Name of the fitted model (used to get parameter names from registry)
    summary : dict
        Fitting summary containing 'samples' array (shape: n_samples x n_params)
    save_path : str or Path, optional
        If provided, save figure to this path
    **corner_kwargs : dict
        Additional keyword arguments passed to corner.corner()

    Returns
    -------
    fig : matplotlib.figure.Figure
        The generated corner plot figure

    Notes
    -----
    - Parameter names are retrieved from ModelRegistry
    - Samples include the log_sigma parameter which is handled separately
    - LaTeX formatting is applied to parameter labels for better readability
    """
    import corner

    # Check if samples exist
    if "samples" not in summary or summary["samples"] is None:
        print(f"Warning: No samples found in summary for {model_name}")
        return None

    samples = summary["samples"]

    if len(samples) == 0:
        print(f"Warning: Empty samples array for {model_name}")
        return None

    # Get parameter names from ModelRegistry
    param_names = ModelRegistry.get_param_names(model_name)

    # MCMC samples include log_sigma as the last parameter
    # We need to add it to our labels
    n_params_model = len(param_names)
    n_params_total = samples.shape[1]

    if n_params_total == n_params_model + 1:
        labels = param_names + ["log_sigma"]
    else:
        # Fallback: use generic labels if dimensions don't match
        labels = [f"param_{i}" for i in range(n_params_total)]

    # Create LaTeX-formatted labels for better visualization
    latex_labels = []
    for label in labels:
        # Convert common parameter names to LaTeX
        if label.startswith("log10_rho"):
            # Extract subscript if present (e.g., log10_rho01 -> \log_{10}\rho_{01})
            suffix = label.replace("log10_rho", "")
            if suffix:
                latex_labels.append(f"$\\log_{{10}} \\rho_{{{suffix}}}$")
            else:
                latex_labels.append("$\\log_{{10}} \\rho_0$")
        elif label.startswith("log10_"):
            param = label.replace("log10_", "")
            latex_labels.append(f"$\\log_{{10}}$ {param}")
        elif label == "rho_0":
            latex_labels.append("$\\rho_0$")
        elif label == "r_s":
            latex_labels.append("$r_s$ [kpc]")
        elif label == "R_d" or label == "R_d1" or label == "R_d2":
            latex_labels.append(f"${label}$ [kpc]")
        elif label == "z_0":
            latex_labels.append("$z_0$ [kpc]")
        elif label == "alpha":
            latex_labels.append("$\\alpha$")
        elif label == "beta":
            latex_labels.append("$\\beta$")
        elif label == "log_sigma":
            latex_labels.append("$\\log \\sigma$")
        elif label == "A" or label == "B":
            latex_labels.append(f"${label}$")
        elif label == "r_core" or label == "rcore" or label == "rc" or label == "r_c":
            latex_labels.append("$r_c$")
        else:
            # Default: wrap in $ for math mode
            latex_labels.append(f"${label}$")

    # Set default corner plot options
    corner_defaults = {
        "quantiles": [0.16, 0.5, 0.84],
        "show_titles": True,
        "title_fmt": ".3f",
        "title_kwargs": {"fontsize": 12},
        "label_kwargs": {"fontsize": 14},
        "smooth": 1.0,
        "bins": 30,
    }

    # Update with user-provided kwargs
    corner_defaults.update(corner_kwargs)

    # Create the corner plot
    fig = corner.corner(samples, labels=latex_labels, **corner_defaults)

    # Add title to the figure
    fig.suptitle(f"{model_name} Posterior Distributions", fontsize=16, y=1.02)

    if save_path:
        save_figure(fig, save_path)

    return fig


def plot_galaxy_density_profile(
    profiles, save_path=None, r_range=(0.3, 250), title=None, figsize=(12, 6)
):
    """
    Plot density profiles for all components (stars, gas, dark matter).

    Creates a plot similar to the standard galaxy density profile visualization,
    showing stars, gas, dark matter, and total density on a log-log scale.

    Parameters
    ----------
    profiles : dict
        Dictionary of DensityProfile objects with keys 'star', 'gas', 'dm'
        e.g., {'star': DensityProfile, 'gas': DensityProfile, 'dm': DensityProfile}
    save_path : str or Path, optional
        Path to save the figure. If None, figure is not saved.
    r_range : tuple, optional
        (min, max) radius range to plot in kpc. Default: (0.3, 250)
    title : str, optional
        Custom title. If None, uses default "Galaxy Density Profile"
    figsize : tuple, optional
        Figure size (width, height). Default: (12, 8)

    Returns
    -------
    fig : matplotlib.figure.Figure
        The created figure

    Examples
    --------
    >>> profiles = {
    ...     'star': DensityProfile(...),
    ...     'gas': DensityProfile(...),
    ...     'dm': DensityProfile(...)
    ... }
    >>> fig = plot_galaxy_density_profile(profiles,
    ...                                    save_path='plots/density_profile.png')
    """
    component_labels = {"star": "Stars", "gas": "Gas", "dm": "Dark Matter"}
    component_colors = {"star": "#1f77b4", "gas": "#2ca02c", "dm": "#ff7f0e"}

    if not profiles:
        raise ValueError("No density profiles provided")

    # Extract r and rho from DensityProfile objects
    profile_data = {}
    for comp, profile in profiles.items():
        if profile is not None:
            profile_data[comp] = {"r": profile.r_1d, "rho": profile.rho_1d}

    # Create figure
    fig, ax = plt.subplots(figsize=figsize)

    # Plot each component
    for comp in ["star", "dm", "gas"]:  # Plot in specific order
        if comp in profile_data:
            data = profile_data[comp]
            mask = (data["r"] >= r_range[0]) & (data["r"] <= r_range[1])
            ax.loglog(
                data["r"][mask],
                data["rho"][mask],
                label=component_labels[comp],
                color=component_colors[comp],
                linewidth=2.5,
            )

    # Calculate and plot total density
    if len(profile_data) > 1:
        # Find common radial grid (use the finest grid available)
        all_r = [profile_data[c]["r"] for c in profile_data]
        r_common = all_r[0]  # Use first component's grid as reference

        # Interpolate all components to common grid
        total_rho = np.zeros_like(r_common)
        for comp in profile_data:
            data = profile_data[comp]
            if len(data["r"]) == len(r_common) and np.allclose(data["r"], r_common):
                total_rho += data["rho"]
            else:
                # Interpolate to common grid
                f = interp1d(
                    data["r"],
                    data["rho"],
                    kind="linear",
                    bounds_error=False,
                    fill_value=0,
                )
                total_rho += f(r_common)

        # Plot total
        mask = (r_common >= r_range[0]) & (r_common <= r_range[1])
        ax.loglog(
            r_common[mask],
            total_rho[mask],
            label="Total",
            color="black",
            linewidth=3,
            linestyle="-",
        )

    # Formatting
    ax.set_xlabel("Radius [kpc]", fontsize=12)
    ax.set_ylabel(r"Density [M$_\odot$/kpc$^{-3}$]", fontsize=12)
    ax.set_xlim(r_range)
    ax.legend(fontsize=12, frameon=True, loc="upper right")
    ax.grid(True, alpha=0.3, which="both")

    if title is None:
        title = "Galaxy Density Profile"

    ax.set_title(title, fontsize=14, pad=15)

    fig.tight_layout()

    if save_path:
        save_figure(fig, save_path)

    return fig


def plot_profiles(
    halo: str,
    name: str,
    redshift: float,
    PATHS: ProjectPaths,
    config: Dict,
    save_fig: Optional[bool] = True,
    experiment: Optional[str] = None,
    show_fig: bool = False
):
    """
    Plot density profiles for stars, gas, and dark matter for halo at redshift.
    """
    data_path = PATHS.snapshot_data(halo, redshift, experiment=experiment)
    plot_path = PATHS.snapshot_plots(halo, redshift, experiment=experiment)
    # Plot density profiles
    StarProfile = DensityProfile.from_hdf5(f"{data_path}/3d_profile_star.h5", 'star', False)
    DMProfile = DensityProfile.from_hdf5(f"{data_path}/3d_profile_dm.h5", 'dm', False)
    GasProfile = DensityProfile.from_hdf5(f"{data_path}/3d_profile_gas.h5", 'gas', False)
    # Total density
    total_density = StarProfile.rho_1d + DMProfile.rho_1d + GasProfile.rho_1d
    r_centers = StarProfile.r_1d

    plt_cfg = get_plot_config(config, plot_type='profile_1d')
    fig, ax = plt.subplots(figsize=plt_cfg.get('figsize', [10, 8]))

    # Filter out invalid values for plotting (mask applied to BOTH x and y)
    star_mask = np.isfinite(StarProfile.rho_1d) & (StarProfile.rho_1d > 0) & (r_centers > 0.28)
    dm_mask = np.isfinite(DMProfile.rho_1d) & (DMProfile.rho_1d > 0) & (r_centers > 0.28)
    gas_mask = np.isfinite(GasProfile.rho_1d) & (GasProfile.rho_1d > 0) & (r_centers > 0.28)
    total_mask = np.isfinite(total_density) & (total_density > 0) & (r_centers > 0.28)

    # print(min(GasProfile.r_1d[gas_mask]))

    ax.set_title(f"Galaxy Density Profile -- {halo} / {name} / z={redshift:.3f}")

    linew = plt_cfg.get('linewidth', 2.0)
    linew_b = plt_cfg.get('linewidth_bold', 3.0)

    ax.loglog(StarProfile.r_1d[star_mask], StarProfile.rho_1d[star_mask], label='Stars', linewidth=linew)
    ax.loglog(DMProfile.r_1d[dm_mask], DMProfile.rho_1d[dm_mask], label='Dark Matter', linewidth=linew)
    ax.loglog(GasProfile.r_1d[gas_mask], GasProfile.rho_1d[gas_mask], label='Gas', linewidth=linew)
    ax.loglog(r_centers[total_mask], total_density[total_mask], 'k-', linewidth=linew_b, label='Total', zorder=5)

    ax.set_xlabel("Radius [kpc]")
    ax.set_ylabel("Density [Msun/kpc³]")
    ax.set_xlim(0.2, 1.5 * max(r_centers))
    fig.tight_layout()
    ax.legend()
    if save_fig:
        save_figure(fig, f"{plot_path}/total_data.png", plt_cfg.get('dpi', 200))

    if show_fig:
        return fig
    else:
        plt.close(fig)


def plot_fitted_mass_evolution(
    halo: str,
    name: str,
    comp: str,
    config: Dict,
    PATHS: ProjectPaths,
    save_fig: Optional[bool] = False,
    show_uncertainties: bool = True,
    experiment: Optional[str] = None,
):
    plt_cfg = get_plot_config(config, 'mass_evolution')

    plot_path = (
        PATHS.plots / experiment / halo
        if experiment else PATHS.plots / halo
    )
    base = (
        PATHS.results / experiment / halo
        if experiment else PATHS.results / halo
    )
    results = load_hdf5(f"{base}/evolution_time_series.h5")
    redshifts = results['time']['redshifts']

    # Determine which raw components we need
    if comp == 'total':
        raw_comps = ['dm', 'star', 'gas']
    elif comp == 'baryon':
        raw_comps = ['star', 'gas']
    else:
        raw_comps = [comp]

    mass_data = {c: results[c]['mass_evolution'] for c in raw_comps}
    radii_to_extract = [5, 10, 15, 20, 30, 50, 75, 100, 150, 200]  # kpc
    masses: dict[int, np.ndarray] = {}
    masses_hi: dict[int, np.ndarray] = {}
    masses_lo: dict[int, np.ndarray] = {}

    # Get snapshot keys present in all required components
    snapshot_keys = sorted(
        set.intersection(*(set(k for k in mass_data[c] if k.startswith("snapshot_")) for c in raw_comps))
    )
    for snapshot in snapshot_keys:
        i = int(snapshot.split("_")[1])

        # Interpolate and sum masses for all required components
        total = np.zeros(len(radii_to_extract))
        total_lo_sq = np.zeros(len(radii_to_extract))
        total_hi_sq = np.zeros(len(radii_to_extract))

        for c in raw_comps:
            r_arr = np.array(mass_data[c][snapshot]['radii'])
            m_arr = np.array(mass_data[c][snapshot]['total'])
            total += np.interp(radii_to_extract, r_arr, m_arr)

            # Interpolate uncertainties if available
            if show_uncertainties and 'total_lo' in mass_data[c][snapshot]:
                lo_arr = np.array(mass_data[c][snapshot]['total_lo'])
                hi_arr = np.array(mass_data[c][snapshot]['total_hi'])
                total_lo_sq += np.interp(radii_to_extract, r_arr, lo_arr)**2
                total_hi_sq += np.interp(radii_to_extract, r_arr, hi_arr)**2

        masses[i] = total
        if show_uncertainties:
            masses_lo[i] = np.sqrt(total_lo_sq) if np.any(total_lo_sq > 0) else np.zeros_like(total)
            masses_hi[i] = np.sqrt(total_hi_sq) if np.any(total_hi_sq > 0) else np.zeros_like(total)

    # Filter to valid snapshots only (skip index 0 if present)
    valid_snapshots = sorted(i for i in masses if i > 0)
    if not valid_snapshots:
        return
    mass_array = np.array([masses[i] for i in valid_snapshots]).T
    mass_lo_array = None
    mass_hi_array = None

    if show_uncertainties and masses_lo:
        mass_lo_array = np.array([masses_lo.get(i, np.zeros_like(masses[i])) for i in valid_snapshots]).T
        mass_hi_array = np.array([masses_hi.get(i, np.zeros_like(masses[i])) for i in valid_snapshots]).T
    redshifts_plot = np.array([redshifts[i] for i in valid_snapshots])

    redshifts_plot = np.log(1 + redshifts_plot)

    # Plot
    fig, ax = plt.subplots(figsize=plt_cfg.get('figsize', [10, 6]))

    norm = mcolors.Normalize(vmin=redshifts_plot.min(), vmax=redshifts_plot.max())
    cmap = cm.get_cmap(plt_cfg.get('cmap', 'turbo_r'))
    sm = cm.ScalarMappable(cmap=cmap, norm=norm)
    ax.set_yscale('log')

    for j, z in enumerate(redshifts_plot):
        color = cmap(norm(z))

        # Plot error band
        if mass_lo_array is not None and mass_hi_array is not None and np.any(mass_lo_array[:, j] > 0):
            y_lo = (mass_array[:, j] - mass_lo_array[:, j]).astype(float)
            y_hi = (mass_array[:, j] + mass_hi_array[:, j]).astype(float)
            ax.fill_between(radii_to_extract, y_lo, y_hi, color=color, alpha=0.2)

        # Plot line
        ax.plot(
            radii_to_extract, mass_array[:, j], '-o',
            color=color,
            markersize=plt_cfg.get('markersize', 4)
        )

    cbar = fig.colorbar(sm, ax=ax)
    cbar.set_label(r"$\ln(1+z)$")

    ax.set_xlabel('Radius [kpc]')
    ax.set_ylabel(r'M(<r) [$M_\odot$]')
    ax.set_title(f'{halo} - {name} - {comp.capitalize()} Mass Evolution (Fitted Profile)')
    ax.set_xscale('log')
    plt.tight_layout()
    if save_fig:
        save_figure(
            fig,
            f"{plot_path}/fitted_{comp}_mass_evolution.png",
            plt_cfg.get('dpi', 200)
        )
    plt.close(fig)


def plot_data_mass_evolution(
    halo: str,
    name: str,
    comp: str,
    config: Dict,
    PATHS: ProjectPaths,
    save_fig: Optional[bool] = False,
    experiment: Optional[str] = None,
):
    plt_cfg = get_plot_config(config, 'mass_evolution')
    base = (
        PATHS.results / experiment / halo
        if experiment else PATHS.results / halo
    )
    results = load_hdf5(f"{base}/evolution_time_series.h5")
    redshifts = results['time']['redshifts']

    fig, ax = plt.subplots(figsize=plt_cfg.get('figsize', (10, 6)))

    cmap = cm.get_cmap(plt_cfg.get('cmap', 'turbo_r'))
    trans_redshifts = np.log(1 + np.array(redshifts))
    if len(trans_redshifts) == 0:
        return
    norm = mcolors.Normalize(vmin=trans_redshifts.min(), vmax=trans_redshifts.max())

    data_masses = {}
    for z in redshifts:
        path = PATHS.snapshot_data(halo, z, experiment=experiment) / "metadata.h5"
        metadata = load_hdf5(path)
        data_masses = metadata['enclosed_masses']

        radii = data_masses['radii']
        if comp != 'baryon':
            comp_mass = data_masses[f'M_{comp}']
        else:
            comp_mass = data_masses['M_star'] + data_masses['M_gas']
        ax.loglog(
            radii, comp_mass, '-o',
            color=cmap(norm(np.log(1 + z))),
            markersize=plt_cfg.get('markersize', 4),
            linewidth=plt_cfg.get('linewidth', 1)
        )

    ax.set_ylabel(r'M(<r) $[M_\odot]$')
    ax.set_xlabel('Radius [kpc]')
    ax.set_title(f'{halo} - {name} - {comp.capitalize()} Mass Evolution (Simulation Data)')

    sm = cm.ScalarMappable(cmap=cmap, norm=norm)
    cbar = fig.colorbar(sm, ax=ax)
    cbar.set_label(r'$\ln(1+z)$')

    if save_fig:
        path = (
            PATHS.plots / experiment / halo
            if experiment else PATHS.plots / halo
        )
        save_figure(
            fig,
            f'{path}/data_{comp}_mass_evolution.png',
            plt_cfg.get('dpi', 200)
        )
    plt.close(fig)


def plot_component_mass_ratio_evolution(
    halo: str,
    name: str,
    config: Dict,
    PATHS: ProjectPaths,
    save_fig: Optional[bool] = False,
    experiment: Optional[str] = None,
):
    plt_cfg = get_plot_config(config, 'ratio_evolution')
    base = PATHS.results / experiment if experiment else PATHS.results
    results = load_hdf5(f"{base}/{halo}/evolution_time_series.h5")
    redshifts = results['time']['redshifts']

    masses_gas = results['gas']['mass_evolution']
    masses_dm = results['dm']['mass_evolution']
    masses_star = results['star']['mass_evolution']

    M_bulge = []
    M_disk = []
    M_halo = []
    valid_indices = []  # Track which snapshot indices have data
    snapshot_keys = sorted(k for k in masses_dm if k.startswith("snapshot_"))
    for snapshot in snapshot_keys:
        i = int(snapshot.split("_")[1])

        valid_indices.append(i)
        M_bulge.append(masses_star[snapshot].get('bulge', 0)[-1])
        M_disk.append(masses_gas[snapshot].get('disk', 0)[-1] + masses_star[snapshot].get('disk', 0)[-1])
        M_halo.append(
            masses_star[snapshot].get('halo', 0)[-1] +
            masses_gas[snapshot].get('halo', 0)[-1] +
            masses_dm[snapshot].get('halo', 0)[-1]
        )

    M_bulge = np.array(M_bulge)
    M_disk = np.array(M_disk)
    M_halo = np.array(M_halo)

    log_p = np.log10(M_bulge / (M_bulge + M_disk))  # Bulge-to-Disk ratio
    log_q = np.log10(M_disk / (M_disk + M_halo))   # Disk-to-Halo ratio

    # Use redshifts only for valid snapshots
    z_orig = np.array([redshifts[i] for i in valid_indices])
    z_plot = np.log(1 + z_orig)

    fig, ax = plt.subplots(figsize=plt_cfg.get('figsize', (8, 6)))

    # Color by redshift
    cmap = cm.get_cmap(plt_cfg.get('cmap', 'turbo_r'))
    norm = mcolors.Normalize(vmin=min(z_plot), vmax=max(z_plot))

    # Plot points colored by redshift
    sc = ax.scatter(
        log_p, log_q, c=z_plot,
        cmap=cmap, norm=norm,
        s=plt_cfg.get('scatter_size', 60),
        edgecolors='white',
        linewidth=0.5,
        zorder=3
    )

    # Connect points with a line to show evolution track
    ax.plot(
        log_p, log_q, '-',
        color='gray', alpha=0.4,
        linewidth=plt_cfg.get('linewidth', 1),
        zorder=1
    )

    # Annotate first and last points (show original z values)
    ax.annotate(f'z={z_orig[0]:.1f}', (log_p[0], log_q[0]),
                xytext=(8, 8), textcoords='offset points', fontsize=9)
    ax.annotate(f'z={z_orig[-1]:.1f}', (log_p[-1], log_q[-1]),
                xytext=(8, -12), textcoords='offset points', fontsize=9)

    # Add colorbar
    cbar = fig.colorbar(sc, ax=ax, pad=0.02)
    cbar.set_label(r'$\ln(1+z)$')

    ax.set_xlabel(r'$\log_{10}(M_{\rm bulge} / M_{\rm disk})$')
    ax.set_ylabel(r'$\log_{10}(M_{\rm disk} / M_{\rm halo})$')
    ax.set_title(f'{halo} / {name}\nMass Ratio Evolution')
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    if save_fig:
        if experiment:
            path = PATHS.plots / experiment / halo / "component_mass_ratio_evolution.png"
        else:
            path = PATHS.plots / halo / "component_mass_ratio_evolution.png"
        save_figure(
            fig,
            path,
            plt_cfg.get('dpi', 200)
        )
    plt.close(fig)


def plot_reddish_2022_figure(
    halo: str,
    name: str,
    config: Dict,
    PATHS: ProjectPaths,
    background_data_file: Optional[Path | str] = None,
    save_fig: bool = False,
    experiment: Optional[str] = None,
    show_fig: bool = False,
    exclude_low_p: float = -2.5,
):
    plt_cfg = get_plot_config(config, 'ratio_evolution')
    base = PATHS.results / experiment if experiment else PATHS.results
    results = load_hdf5(f"{base}/{halo}/evolution_time_series.h5")
    redshifts = results['time']['redshifts']

    masses_gas = results['gas']['mass_evolution']
    masses_dm = results['dm']['mass_evolution']
    masses_star = results['star']['mass_evolution']

    M_bulge = []
    M_disk = []
    M_halo = []
    valid_indices = []  # Track which snapshot indices have data
    snapshot_keys = sorted(k for k in masses_dm if k.startswith("snapshot_"))
    for snapshot in snapshot_keys:
        i = int(snapshot.split("_")[1])

        valid_indices.append(i)
        M_bulge.append(masses_star[snapshot].get('bulge', 0)[-1])
        M_disk.append(
            masses_gas[snapshot].get('disk', 0)[-1] +
            masses_star[snapshot].get('disk', 0)[-1]
        )
        M_halo.append(
            masses_star[snapshot].get('halo', 0)[-1] +
            masses_gas[snapshot].get('halo', 0)[-1] +
            masses_dm[snapshot].get('halo', 0)[-1]
        )

    M_bulge = np.array(M_bulge)
    M_disk = np.array(M_disk)
    M_halo = np.array(M_halo)

    with np.errstate(divide='ignore', invalid='ignore'):
        log_p = np.log10(M_bulge / (M_bulge + M_disk))  # Bulge-to-Disk ratio
        log_q = np.log10(M_disk / (M_disk + M_halo))   # Disk-to-Halo ratio

    # Use redshifts only for valid snapshots
    z_orig = np.array([redshifts[i] for i in valid_indices])
    z_plot = np.log(1 + z_orig)

    mask = (log_p > exclude_low_p)
    z_plot, log_p, log_q, z_orig = z_plot[mask], log_p[mask], log_q[mask], z_orig[mask]

    if len(z_plot) == 0:
        return

    fig, ax = plt.subplots(figsize=plt_cfg.get('figsize', (8, 6)))

    # Plot Reddish+2022 background growth-rate field
    if background_data_file is not None:
        gridsize = 200
        with h5py.File(background_data_file, 'r') as bg:
            grvals = bg['tabGR'][:].reshape(gridsize, gridsize)
        bg_q, bg_p = np.meshgrid(
            np.linspace(0., 1.0, gridsize),
            np.concatenate([np.linspace(0., 0.4, gridsize - 1), [1.0]]),
            indexing='ij'
        )
        # Convert linear p, q to log10 scale to match plot axes
        log_bg_p = np.log10(np.clip(bg_p, bg_p[bg_p > 0].min(), 1))
        log_bg_q = np.log10(np.clip(bg_q, bg_q[bg_q > 0].min(), 1))
        cf = ax.contourf(log_bg_p, log_bg_q, grvals, levels=20,
                         cmap='inferno_r', alpha=0.6, zorder=0)

    # Color by redshift
    cmap = plt.get_cmap(plt_cfg.get('cmap', 'turbo_r'))
    norm = mcolors.Normalize(vmin=min(z_plot), vmax=max(z_plot))

    # Plot points colored by redshift
    sc = ax.scatter(
        log_p, log_q, c=z_plot,
        cmap=cmap, norm=norm,
        s=plt_cfg.get('scatter_size', 60),
        edgecolors='white',
        linewidth=0.5,
        zorder=3
    )

    # Connect points with a line to show evolution track
    ax.plot(
        log_p, log_q, '-',
        color='gray', alpha=0.4,
        linewidth=plt_cfg.get('linewidth', 1),
        zorder=1
    )
    ax.set_xlim(-1.75, 0)
    ax.set_ylim(-1.75, 0)

    # Annotate first and last points (show original z values)
    ax.annotate(f'z={z_orig[0]:.1f}', (log_p[0], log_q[0]),
                xytext=(8, 8), textcoords='offset points', fontsize=9)
    ax.annotate(f'z={z_orig[-1]:.1f}', (log_p[-1], log_q[-1]),
                xytext=(8, -12), textcoords='offset points', fontsize=9)

    # Arrow showing direction of time evolution (midpoint of track)
    if len(log_p) >= 3:
        mid = len(log_p) // 2
        ax.annotate(
            '', xy=(log_p[mid + 1], log_q[mid + 1]),
            xytext=(log_p[mid], log_q[mid]),
            arrowprops=dict(arrowstyle='->', color='gray', lw=1.5),
            zorder=2
        )

    # Add colorbar for redshift (right side) — must be created before top colorbar
    # so both are sized against the same final axes width
    cbar = fig.colorbar(sc, ax=ax, pad=0.02)
    cbar.set_label(r'$\ln(1+z)$')

    # Add top colorbar for growth rate, now that axes width is finalised
    if background_data_file is not None:
        gr_cbar = fig.colorbar(cf, ax=ax, location='top', pad=0.02, aspect=20)
        gr_cbar.set_label('Growth rate (Reddish+2022)', fontsize=8)
        gr_cbar.ax.tick_params(labelsize=7)

    ax.set_xlabel(r'$\log_{10}(p)=\log_{10}(M_{\rm bulge} / (M_{\rm bulge} + M_{\rm disc}))$')
    ax.set_ylabel(r'$\log_{10}(q)=\log_{10}(M_{\rm disc} / (M_{\rm disc} + M_{\rm halo}))$')

    # suptitle avoids clashing with the top colorbar
    fig.suptitle(f'{halo} / {name}  —  Galactic Bar Growth', fontsize=11)

    plt.tight_layout()
    if save_fig:
        if experiment:
            path = PATHS.plots / experiment / halo / "component_mass_ratio_evolution.png"
        else:
            path = PATHS.plots / halo / "component_mass_ratio_evolution.png"
        save_figure(
            fig,
            path,
            plt_cfg.get('dpi', 200)
        )
    if show_fig:
        return fig
    else:
        plt.close(fig)


def plot_component_mass_evolution(
    halo: str,
    name: str,
    config: Dict,
    PATHS: ProjectPaths,
    save_fig: Optional[bool] = False,
    experiment: Optional[str] = None,
):
    plt_cfg = get_plot_config(config, 'component_evolution')
    base = PATHS.results / experiment if experiment else PATHS.results
    results = load_hdf5(f"{base}/{halo}/evolution_time_series.h5")
    redshifts = results['time']['redshifts']

    masses_gas = results['gas']['mass_evolution']
    masses_dm = results['dm']['mass_evolution']
    masses_star = results['star']['mass_evolution']

    M_bulge = []
    M_disk = []
    M_halo = []
    valid_indices = []  # Track which snapshot indices have data
    snapshot_keys = sorted(k for k in masses_dm if k.startswith("snapshot_"))
    for snapshot in snapshot_keys:
        i = int(snapshot.split("_")[1])

        valid_indices.append(i)
        M_bulge.append(masses_star[snapshot].get('bulge', 0)[-1])
        M_disk.append(masses_gas[snapshot].get('disk', 0)[-1] + masses_star[snapshot].get('disk', 0)[-1])
        M_halo.append(masses_star[snapshot].get('halo', 0)[-1] + masses_gas[snapshot].get('halo', 0)[-1] + masses_dm[snapshot].get('halo', 0)[-1])

    M_bulge = np.array(M_bulge)
    M_disk = np.array(M_disk)
    M_halo = np.array(M_halo)

    masses = {
        'bulge': M_bulge,
        'disk': M_disk,
        'halo': M_halo,
        'total': (M_bulge + M_disk + M_halo)
    }

    components = ['bulge', 'disk', 'halo', 'total']

    # Get redshifts and times only for valid snapshots
    valid_redshifts = np.array([redshifts[i] for i in valid_indices])

    fig, (ax, ax_r200) = plt.subplots(
        2, 1, figsize=plt_cfg.get('figsize', (9, 7)),
        height_ratios=[3, 1],
        sharex=True,
        gridspec_kw={'hspace': 0.05})

    colors = get_component_colors(config, 'structural')

    trans_z = np.log(1 + valid_redshifts)

    for comp in components:
        label = "Disc" if comp == "disk" else comp.capitalize()
        ax.semilogy(
            trans_z, masses[comp], '-o',
            label=label, color=colors[comp],
            linewidth=plt_cfg.get('linewidth', 2),
            markersize=plt_cfg.get('markersize', 5),
            markeredgecolor='white',
            markeredgewidth=0.5)

    ax.set_title(r"Galaxy Component Mass within $R_{200}$ vs redshift" + f"\n{halo} / {name}")
    ax.set_ylabel(r"Total Mass $[M_\odot]$")

    ax2 = ax.twiny()
    z_ticks = [0, 0.5, 1, 2, 3, 4, 5, 6]
    trans_z_ticks = np.log(1 + np.array(z_ticks))
    ax2.set_xlim(ax.get_xlim())
    ax2.set_xticks(trans_z_ticks)
    ax2.set_xticklabels([f'{z}' for z in z_ticks])
    ax2.set_xlabel('Redshift $z$')
    ax2.invert_xaxis()

    # get R200 values only for valid redshifts
    R200_values = []
    for z in valid_redshifts:
        metadata = load_hdf5(
            PATHS.snapshot_data(halo, z, experiment=experiment) / "metadata.h5"
        )
        R200_values.append(metadata['R200_kpc'])

    # R200 panel below
    ax_r200.plot(
        trans_z, R200_values, 'k-o',
        markersize=plt_cfg.get('markersize', 4)
    )
    ax_r200.set_ylabel(r'$R_{200}$ [kpc]')
    ax_r200.set_xlabel('ln(1 + z)')
    ax_r200.set_yscale('log')
    ax_r200.invert_xaxis()

    ax.legend(frameon=True)
    if save_fig:
        save_figure(
            fig,
            f"{PATHS.plots}/{experiment}/{halo}/component_mass_evolution.png" if experiment
            else f"{PATHS.plots}/{halo}/component_mass_evolution.png",
            plt_cfg.get('dpi', 200)
        )
    plt.close(fig)


def plot_halo_shape_evolution(
    halo: str,
    name: str,
    config: Dict,
    PATHS: ProjectPaths,
    save_fig: bool = False,
    show_fig: bool = False,
    experiment: Optional[str] = None,
):
    """
    Plot the evolution of two-power density profile parameters (alpha, beta, rs)
    across redshift for all components (star, gas, dm).

    Parameters
    ----------
    halo : str
        Halo identifier (e.g., 'halo_008508')
    name : str
        Human-readable halo name (e.g., 'Tempest')
    config : Dict
        Configuration dictionary
    PATHS : ProjectPaths
        Project paths object
    save_fig : bool, optional
        Whether to save the figure
    """
    from .workflow import FittingResult

    plt_cfg = get_plot_config(config, 'halo_shape')
    components = ["star", "gas", "dm"]
    params = ['alpha', 'beta', 'rs']

    # Load evolution data to get redshifts
    if experiment is not None:
        halo_ev_path = PATHS.results / experiment / halo / "evolution_time_series.h5"
    else:
        halo_ev_path = PATHS.results / halo / "evolution_time_series.h5"
    halo_evolution_data = load_hdf5(halo_ev_path)
    redshifts = halo_evolution_data['time']['redshifts']

    # Initialize results structure
    results: Dict[str, Dict] = {comp: {} for comp in components}
    for comp in components:
        for param in params:
            results[comp][param] = []
            results[comp][f'{param}_high'] = []
            results[comp][f'{param}_low'] = []
        results[comp]['redshifts_plt'] = []

    # Collect data for each component and redshift
    for comp in components:
        for redshift in redshifts:
            plot_redshift = np.log(1 + redshift)
            results_path = PATHS.snapshot_results(halo, redshift, experiment)
            results_file = results_path / f"{comp}_halo_results.h5"

            if not results_file.exists():
                continue

            halo_res = FittingResult.from_hdf5(results_file)
            summary = halo_res.summary

            if summary is None:
                continue

            # Handle NFW model (fixed alpha=1, beta=3)
            if halo_res.best_model == "NFW":
                results[comp]['alpha'].append(1)
                results[comp]['beta'].append(3)
                results[comp]['alpha_high'].append(1)
                results[comp]['alpha_low'].append(1)
                results[comp]['beta_high'].append(3)
                results[comp]['beta_low'].append(3)
                rs = summary['rs']
                rs_err = summary['rs_err']
                results[comp]['rs'].append(rs)
                results[comp]['rs_low'].append(rs_err[0])
                results[comp]['rs_high'].append(rs_err[1])
            else:
                for param in params:
                    results[comp][param].append(summary[param])
                    err_bounds = summary[f"{param}_err"]
                    results[comp][f'{param}_low'].append(err_bounds[0])
                    results[comp][f'{param}_high'].append(err_bounds[1])

            results[comp]['redshifts_plt'].append(plot_redshift)

    # Create the plot
    fig, axes = plt.subplots(
        3, 1,
        figsize=plt_cfg.get('figsize', (8, 12)),
        sharex=True
    )
    comp_colors = get_component_colors(config, 'component_type')

    param_labels = {
        'alpha': r'$\alpha$ (inner slope)',
        'beta': r'$\beta$ (outer slope)',
        'rs': r'$r_s$ [kpc]'
    }

    for ax, param in zip(axes, params):
        apply_axis_style(ax, config, 'halo_shape')
        ax.set_ylabel(param_labels.get(param, param))

        for comp in components:
            x = np.array(results[comp]['redshifts_plt'])
            y = np.array(results[comp][param])
            y_low = np.array(results[comp][f'{param}_low'])
            y_high = np.array(results[comp][f'{param}_high'])

            if len(x) == 0:
                continue

            # Sort by x before plotting
            sort_idx = np.argsort(x)
            x = x[sort_idx]
            y = y[sort_idx]
            y_low = y_low[sort_idx]
            y_high = y_high[sort_idx]

            color = comp_colors.get(comp, 'gray')
            ax.plot(x, y, '-o', label=comp, color=color,
                    markersize=plt_cfg.get('markersize', 5))
            ax.fill_between(x, y_low, y_high, color=color, alpha=0.2)
            ax.axhline(
                np.mean(y), color=color, linestyle='--',
                linewidth=2, alpha=0.7, label=f'Mean {param}'
            )

        if param == 'rs':
            ax.set_yscale('log')
        if param == 'alpha':
            ax.axhline(1, color='black', linestyle='--', linewidth=1.5,
                       label='NFW', alpha=0.8)
        if param == 'beta':
            ax.axhline(3, color='black', linestyle='--', linewidth=1.5,
                       label='NFW', alpha=0.8)

    # Add linear redshift scale on top axis
    def ln1pz_to_z(ln1pz):
        """Convert ln(1+z) to z"""
        return np.exp(ln1pz) - 1

    def z_to_ln1pz(z):
        """Convert z to ln(1+z)"""
        return np.log(1 + z)

    for i, ax in enumerate(axes):
        # create secondary axis
        secax = ax.secondary_xaxis('top', functions=(ln1pz_to_z, z_to_ln1pz))

        # y-axis ticks
        ax.tick_params(axis='y', which='both', left=True, right=False,
                       labelleft=True, direction='in')
        # x-axis
        if i == 0:
            ax.legend(loc='best')
            secax.set_xlabel(r'Redshift $z$')
            ax.tick_params(axis='x', which='both', bottom=True, top=False,
                           labelbottom=False, direction='in')
            secax.tick_params(axis='x', which='both', bottom=False, top=True,
                              labeltop=True, direction='in')
        elif i == len(axes) - 1:
            ax.set_xlabel(r'$\ln(1+z)$')
            ax.invert_xaxis()
            ax.tick_params(axis='x', which='both', bottom=True, top=False,
                           labelbottom=True, direction='in')
            secax.tick_params(axis='x', which='both', bottom=False, top=True,
                              labeltop=False, direction='in')
        else:
            ax.tick_params(axis='x', which='both', bottom=True, top=False,
                           labelbottom=False)
            secax.tick_params(axis='x', which='both', bottom=False, top=True,
                              labeltop=False, direction='in')

    fig.suptitle(f"{halo} / {name}\nHalo Shape Parameter Evolution")
    plt.tight_layout()

    if save_fig:
        save_figure(
            fig,
            (f"{PATHS.plots}/{experiment}/{halo}/halo_shape_evolution.png")
            if experiment else f"{PATHS.plots}/{halo}/halo_shape_evolution.png",
            plt_cfg.get('dpi', 200)
        )
    if show_fig:
        return fig
    else:
        plt.close(fig)


def plot_comp_projection_with_r200(
    ds,
    center,
    r200_kpc: float,
    redshift: float,
    output_path: Path | str,
    component: str,
    config: Optional[Dict] = None,
    width_factor: Optional[float] = None,
):
    """
    Create Component projection plot with R200 circle overlay.

    Parameters
    ----------
    ds : yt dataset
        Dataset to plot
    center : array-like
        Center coordinates
    r200_kpc : float
        Virial radius in kpc
    redshift : float
        Redshift of snapshot
    output_path : Path or str
        Where to save the plot
    component : str
        Component type ('star', 'gas', 'dm')
    config : dict, optional
        Configuration dictionary. If None, uses defaults.
    width_factor : float, optional
        Override for plot width as multiple of R200. If None, uses config value.
    """
    # Load projection config with defaults
    proj_config = get_plot_config(config or {}, 'projection')

    # Get width factor from parameter, config, or default
    if width_factor is None:
        width_factor = proj_config.get('width_factor', 3.0)
    width_kpc = width_factor * r200_kpc

    # Get colormap - try component-specific first, then default
    cmaps = proj_config.get('cmaps', {})
    cmap = cmaps.get(component, proj_config.get('cmap', 'viridis'))

    # Create projection based on component type
    if component in ["dm", "star"]:
        proj = yt.ParticleProjectionPlot(
            ds,
            "z",  # Project along z-axis
            (component, "particle_mass"),
            center=center,
            width=(width_kpc, "kpc"),
            depth=(width_kpc, "kpc"),
        )
        proj.set_unit((component, "particle_mass"), "Msun")
        proj.set_cmap((component, "particle_mass"), cmap)
        field = (component, "particle_mass")
    else:  # gas
        proj = yt.ProjectionPlot(
            ds,
            "z",
            ("gas", "density"),
            center=center,
            width=(width_kpc, "kpc"),
            weight_field=("gas", "density"),
        )
        proj.set_cmap(("gas", "density"), cmap)
        field = ("gas", "density")

    proj.set_axes_unit("kpc")

    # Get the matplotlib figure and axis
    plot = proj.plots[field]
    ax = plot.axes

    # Add R200 circle with config styling
    r200_style = proj_config.get('r200_circle', {})
    circle = Circle(
        (0, 0),  # Center at origin (since we centered the plot)
        r200_kpc,
        fill=False,
        edgecolor=r200_style.get('color', 'red'),
        linewidth=r200_style.get('linewidth', 2),
        linestyle=r200_style.get('linestyle', '--'),
        label=f"R$_{{200}}$ = {r200_kpc:.0f} kpc",
    )
    ax.add_patch(circle)

    # Add additional radius markers from config
    markers = proj_config.get('radius_markers', [
        {'factor': 0.5, 'color': 'orange', 'linestyle': ':', 'linewidth': 1, 'alpha': 0.6},
        {'factor': 1.5, 'color': 'cyan', 'linestyle': ':', 'linewidth': 1, 'alpha': 0.6},
    ])
    for marker in markers:
        circle = Circle(
            (0, 0),
            marker['factor'] * r200_kpc,
            fill=False,
            edgecolor=marker.get('color', 'orange'),
            linewidth=marker.get('linewidth', 1),
            linestyle=marker.get('linestyle', ':'),
            alpha=marker.get('alpha', 0.6),
        )
        ax.add_patch(circle)

    # Add title with redshift
    title_fontsize = proj_config.get('title_fontsize', 14)
    ax.set_title(f"{component.upper()} Projection (z={redshift:.2f})", fontsize=title_fontsize)

    # Add legend with config styling
    legend_config = proj_config.get('legend', {})
    ax.legend(
        loc=legend_config.get('loc', 'upper right'),
        fontsize=legend_config.get('fontsize', 10),
        framealpha=legend_config.get('framealpha', 0.8)
    )

    # Save the plot
    proj.save(str(output_path))

    return proj


def plot_shape_analysis(
    halo: str,
    name: str,
    redshift: float,
    config: Dict,
    PATHS: ProjectPaths,
    save_fig: bool = False,
    experiment: Optional[str] = None,
    show_fig: bool = False,
    metadata: Optional[Dict] = None,
):
    plt_cfg = get_plot_config(config, 'shape_analysis')

    if metadata is None:
        metadata = load_hdf5(PATHS.snapshot_data(halo, redshift, experiment=experiment) / "metadata.h5")

    shape_tensors = metadata['shape_tensors']
    r = shape_tensors['radii']  # (N,) array of radii
    tensors = shape_tensors['tensors']  # (N, 3, 3) array of shape tensors

    # get eigenvectors and axes
    eigenvalues, eigenvectors = np.linalg.eigh(tensors)  # (N, 3) and (N, 3, 3)
    # eigh returns ascending order - reverse (a > b > c)
    idx = np.argsort(eigenvalues, axis=-1)[:, ::-1]          # (N, 3)
    eigenvalues = np.take_along_axis(eigenvalues, idx, axis=-1)
    eigenvectors = np.take_along_axis(eigenvectors, idx[:, np.newaxis, :], axis=-1)

    a, b, c = np.sqrt(eigenvalues).T  # each shape (N,)
    # eigenvectors[:, :, 0] → major axis at each radius
    # eigenvectors[:, :, 2] → minor axis (disk normal) at each radius

    ba = b / a
    ca = c / a
    T = (a**2 - b**2) / (a**2 - c**2)  # Triaxiality

    # Minor axis tilt relative to angular momentum vector (warp diagnostic)
    minor_vecs = eigenvectors[:, :, 2]  # (N, 3) — disk normal at each radius
    l_hat = metadata['angular_momentum_unit_vector']  # (3,)
    cos_theta = np.abs(np.einsum('ni,i->n', minor_vecs, l_hat))
    tilt_deg = np.degrees(np.arccos(np.clip(cos_theta, 0, 1)))

    fig, axes = plt.subplots(1, 3, figsize=plt_cfg.get('figsize', [6.27, 4.7]))

    # --- Panel 1: Axis ratios ---
    ax = axes[0]
    ax.plot(r, ba, 'o-', color='steelblue', label='b/a (intermediate/major)')
    ax.plot(r, ca, 's-', color='tomato', label='c/a (minor/major)')

    # Threshold regions
    ax.axhspan(0.0, 0.3, alpha=0.08, color='tomato', label='Disk-dominated (c/a < 0.3)')
    ax.axhspan(0.5, 1.0, alpha=0.08, color='steelblue', label='Spheroidal (c/a > 0.5)')
    ax.axhline(0.3, ls='--', lw=0.8, color='tomato', alpha=0.6)
    ax.axhline(0.5, ls='--', lw=0.8, color='steelblue', alpha=0.6)

    ax.set_xscale('log')
    ax.set_xlabel('Radius [kpc]')
    ax.set_ylabel('Axis ratio')
    ax.set_title('Stellar shape vs radius')
    ax.set_ylim(0, 1.05)
    ax.legend(fontsize=9)
    apply_axis_style(ax, config, 'shape_analysis')

    # --- Panel 2: Triaxiality ---
    ax = axes[1]
    ax.plot(r, T, 'D-', color='mediumpurple')
    ax.axhspan(0.0, 1 / 3, alpha=0.08, color='steelblue', label='Oblate (T < 1/3)')
    ax.axhspan(2 / 3, 1.0, alpha=0.08, color='tomato', label='Prolate (T > 2/3)')
    ax.axhline(1 / 3, ls='--', lw=0.8, color='steelblue', alpha=0.6)
    ax.axhline(2 / 3, ls='--', lw=0.8, color='tomato', alpha=0.6)
    ax.text(r[-1] * 1.02, 1 / 6, 'Oblate', fontsize=8, color='steelblue', va='center')
    ax.text(r[-1] * 1.02, 5 / 6, 'Prolate', fontsize=8, color='tomato', va='center')
    ax.text(r[-1] * 1.02, 0.5, 'Triaxial', fontsize=8, color='grey', va='center')

    ax.set_xscale('log')
    ax.set_xlabel('Radius [kpc]')
    ax.set_ylabel(r'Triaxiality $T = (a^2-b^2)/(a^2-c^2)$')
    ax.set_title('Triaxiality vs radius')
    ax.set_ylim(0, 1)
    ax.legend(fontsize=9, loc='upper left')
    apply_axis_style(ax, config, 'shape_analysis')

    # --- Panel 3: Minor axis tilt (warp diagnostic) ---
    ax = axes[2]
    ax.plot(r, tilt_deg, '^-', color='darkorange')
    ax.axhline(0, ls='--', lw=0.8, color='grey', alpha=0.6)
    ax.axhspan(0, 10, alpha=0.08, color='steelblue', label='Aligned (<10°)')
    ax.set_xscale('log')
    ax.set_xlabel('Radius [kpc]')
    ax.set_ylabel('Minor axis tilt from L [degrees]')
    ax.set_title('Disk normal orientation vs radius\n(non-zero = warp)')
    ax.legend(fontsize=9)
    apply_axis_style(ax, config, 'shape_analysis')

    plt.suptitle(f'{halo}-{name} (z={redshift:.3f}) — Stellar shape tensor analysis', fontsize=14)
    plt.tight_layout()
    if save_fig:
        save_figure(
            fig,
            PATHS.snapshot_plots(
                halo, redshift, experiment=experiment
            ) / "shape_analysis.png",
            plt_cfg.get('dpi', 200)
        )
    if show_fig:
        plt.show()
    else:
        plt.close(fig)


def plot_mass_error(
    csv_path: Path | str,
    component: str, halo: str, name: str,
    config: Dict, save_fig: bool, show_fig: bool,
    experiment: Optional[str],
    PATHS: ProjectPaths
)-> Any | None:
    df = pd.read_csv(csv_path)
    plt_cfg = get_plot_config(config, 'mass_evolution')
    mask = (
        (df['sim_mass'] > 0) & (df['model_mass'] > 0)
    )
    df['rel_error'] = np.abs(
        (df['model_mass'] - df['sim_mass']) / df['sim_mass']
    )
    df.loc[~mask, 'rel_error'] = np.nan
    redshifts = np.array(sorted(df['redshift'].unique()))
    plot_redshifts = np.log(1+redshifts)

    # colour map
    cmap = plt.get_cmap(plt_cfg.get('cmap', 'turbo_r'))
    norm = plt.Normalize(vmin=min(plot_redshifts), vmax=np.log(4))
    
    sub = df[
        (df['halo'] == halo)           &
        (df['component'] == component) &
        (df['redshift'] <= 3)          &
        (df['sim_mass'] > 0)           &
        (df['model_mass'] > 0)
    ]
    
    fig, ax = plt.subplots(figsize=plt_cfg.get('figsize', (8, 6)))
    
    for z in redshifts:
        sub_z = sub[(sub['redshift'] == z)]
        if sub_z.empty:
            continue
        z_plot = np.log(1+z)
        color = cmap(norm(z_plot))
        ax.plot(sub_z['radius_kpc'], sub_z['rel_error'], color=color)
    
    ax.axhline(0.1, linestyle='-', color='grey', alpha=0.4, lw=0.8, label='10%')
    ax.axhline(0.5, linestyle='-', color='red', alpha=0.4, lw=0.8, label='50%')
    ax.axhline(sub['rel_error'].mean(), linestyle='--', color='black', label='mean')
    ax.axhline(sub['rel_error'].median(), linestyle=':', color='black', label='median')
    ax.legend()

    # Colorbar for redshift
    sm = cm.ScalarMappable(cmap=cmap, norm=norm)
    sm.set_array([])
    fig.colorbar(sm, ax=ax, label='ln(1+z)')

    ax.set(
        xscale = "log",
        yscale = "log",
        xlabel = "Radius [kpc]",
        ylabel = "Rel. Residual - (|model - data| / data)"
    )

    fig.suptitle(f'Mass profiles — {component} - {halo} / {name}', fontsize=14)
    plt.tight_layout()
    if save_fig:
        if experiment:
            path = PATHS.plots / experiment / halo / f"{component}_mass_error_evol.png"
        else:
            path = PATHS.plots / halo / f"{component}_mass_error_evol.png"
        save_figure(fig, path, plt_cfg.get('dpi', 200))
    if show_fig:
        return fig
    else:
        plt.close(fig)
