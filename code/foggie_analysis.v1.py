# Comparison Script for FOGGIE Dynamics Analysis

import argparse
import logging
import sys
from datetime import datetime
from pathlib import Path
from time import time

import h5py
import matplotlib.pyplot as plt
import numpy as np
import yaml
import yt

# Add parent directory to path so we can import src
DIR = Path(__file__).parent.resolve()
PROJECT_ROOT = DIR.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src import density_profiles as dp
from src import filters_and_utils as utils
from src import fitting, models, plotting
from src.analysis import compare_models, save_results
from src.filters_and_utils import extract_density

DATA_DIR = PROJECT_ROOT / "datasets"
PLOT_DIR = PROJECT_ROOT / "plots"
RESULTS_DIR = PROJECT_ROOT / "results"


def setup_logging(halo_id, component):
    log_file = (
        RESULTS_DIR
        / halo_id
        / f"{component}_analysis_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"
    )
    log_file.parent.mkdir(parents=True, exist_ok=True)

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",  # Added %(name)s
        handlers=[logging.FileHandler(log_file), logging.StreamHandler()],
    )
    return logging.getLogger(__name__)


def load_config(config_file=None):
    """Load fitting configuration from YAML file"""
    if config_file is None:
        config_file = PROJECT_ROOT / "config" / "fitting_config.yaml"

    with open(config_file, "r") as f:
        config = yaml.safe_load(f)

    # Handle special values like null for max(r_1d)
    return config


def disk_flaring_log(R, z, log10_rho01, log10_rho02, R_d1, R_d2, A, B):
    return models.double_disk_flaring(
        R, z, 10**log10_rho01, 10**log10_rho02, R_d1, R_d2, A, B
    )


def fit_stellar_components(
    r_1d, rho_1d, R, z, rho_2d, counts, config, logger, verbose, save_data, halo_id
):
    """Fit bulge/disk/halo for stellar component.

    Args:
        r_1d: 1D radial bins centers
        rho_1d: 1D spherical density profile
        R: Cylindrical R bin centers
        z: Cylindrical z bin centers
        rho_2d: 2D cylindrical density profile
        counts: Particle counts per bin (for spherical profile)
        config: Full configuration dict
        logger: Logger instance
        verbose: If True, log detailed fitting reports
        save_data: If True, save results to HDF5
        halo_id: Halo identifier for file paths

    Returns:
        dict: Results dictionary with keys 'bulge', 'disk', 'halo' (if enabled).
              Each contains 'best_model', 'best_summary', 'table', 'figure'.
    """
    results = {}
    star_config = config["star"]

    # R and z are 1D bin centers, need to create 2D grid
    R_grid, z_grid = np.meshgrid(R, z, indexing="ij")
    rho_2d_flat = rho_2d.flatten()
    R_flat = R_grid.flatten()
    z_flat = z_grid.flatten()

    # bulge fitting
    if star_config["bulge"].get("enabled", True):
        logger.info("FITTING BULGE COMPONENT")

        r_range = tuple(star_config["bulge"].get("r_range"))
        bounds = star_config["bulge"].get("bounds")
        p0 = star_config["bulge"].get("p0")
        fitting_params = star_config["bulge"].get("fitting", {})
        nwalkers = fitting_params.get("nwalkers", 64)
        nsteps = fitting_params.get("nsteps", 9000)

        weights = None
        if "weights" in star_config["bulge"]:
            w_cfg = star_config["bulge"]["weights"]
            if w_cfg.get("type") == "gaussian":
                weights = utils.add_gaussian_weight(
                    r_1d,
                    r_range,
                    mean=w_cfg.get("mean", None),
                    sigma_frac=w_cfg.get("sigma_frac", 0.1),
                )

        summaries = {}

        for model in star_config["bulge"]["models"]:
            logger.info(f"Fitting {model} model...")
            try:
                if model == "Hernquist":
                    _, summary = fitting.fit_bulge_emcee(
                        r_1d,
                        rho_1d,
                        r_range=r_range,
                        bounds=bounds,
                        p0=p0,
                        nwalkers=nwalkers,
                        nsteps=nsteps,
                        weights=weights,
                        logger=logger,
                    )
                elif model == "Two-Power":
                    _, summary = fitting.fit_two_power_emcee(
                        r_1d,
                        rho_1d,
                        r_range=r_range,
                        bounds=bounds,
                        p0=p0,
                        nwalkers=nwalkers,
                        nsteps=nsteps,
                        weights=weights,
                        logger=logger,
                    )
                elif model == "Exponential":
                    _, summary = fitting.fit_exp_emcee(
                        r_1d,
                        rho_1d,
                        r_range=r_range,
                        bounds=bounds,
                        p0=p0,
                        nsteps=nsteps,
                        nwalkers=nwalkers,
                        weights=weights,
                        logger=logger,
                    )
                elif model == "Exp-Cutoff":
                    _, summary = fitting.fit_general_emcee(
                        coords=(r_1d,),
                        data=rho_1d,
                        model_func=models.exp_cutoff,
                        param_names=["rho0", "r_core", "r_cutoff", "alpha"],
                        bounds=bounds,
                        p0=p0,
                        data_range={"r": r_range},
                        weights=weights,
                        nwalkers=nwalkers,
                        nsteps=nsteps,
                        logger=logger,
                    )
                else:
                    logger.error(f"Model {model} unknown!")
                    summary = None
            except Exception as e:
                logger.error(f"Error fitting {model} model: {e}", exc_info=True)
                summary = None
            summaries[model] = summary

        # Build evaluators for successfully fitted models
        evaluators = {}
        if "Hernquist" in summaries and summaries["Hernquist"] is not None:
            evaluators["Hernquist"] = lambda coords, s: models.hernquist_density(
                coords[0], s["rho_0"], s["r_s"]
            )
        if "Two-Power" in summaries and summaries["Two-Power"] is not None:
            evaluators["Two-Power"] = lambda coords, s: models.two_power_model(
                coords[0], s["rho_0"], s["r_s"], s["alpha"], s["beta"]
            )
        if "Exponential" in summaries and summaries["Exponential"] is not None:
            evaluators["Exponential"] = lambda coords, s: s["rho_0"] * np.exp(
                -coords[0] / s["r_s"]
            )
        if "Exp-Cutoff" in summaries and summaries["Exp-Cutoff"] is not None:
            evaluators["Exp-Cutoff"] = lambda coords, s: models.exp_cutoff(
                coords[0], s["rho0"], s["r_core"], s["r_cutoff"], s["alpha"]
            )

        summaries_clean = {k: v for k, v in summaries.items() if k is not None}
        evaluators_clean = {k: v for k, v in evaluators.items() if k is not None}

        best_bulge, best_bulge_summary, bulge_table, bulge_report = compare_models(
            model_summaries=summaries_clean,
            coords=(r_1d,),
            data=rho_1d,
            logger=logger,
            data_range={0: r_range},
            model_evaluators=evaluators_clean,
            counts=counts,
        )
        if verbose:
            logger.info(bulge_report)
        results["bulge"] = {
            "best_model": best_bulge,
            "best_summary": best_bulge_summary,
            "table": bulge_table,
        }

        bulge_fig = plotting.plot_results(
            coords=(r_1d,),
            data=rho_1d,
            data_range={0: r_range},
            model_summaries=summaries_clean,
            model_evaluators=evaluators_clean,
            best_model_name=best_bulge,
            component_name="Stellar Bulge",
            coord_label="r [kpc]",
            save_path=f"{PLOT_DIR}/{halo_id}/star_bulge_fit.png",
        )
        results["bulge"]["figure"] = bulge_fig

        if save_data:
            save_results(
                f"{RESULTS_DIR}/{halo_id}/star_bulge_results.h5",
                summaries_clean,
                best_bulge,
                bulge_table,
            )
        logger.info(f"Bulge fitting completed")
    else:
        logger.info("Skipping bulge fitting (disabled in config)")

    if star_config["disk"].get("enabled", True):
        logger.info("FITTING DISK COMPONENT")

        R_range = tuple(star_config["disk"].get("R_range"))
        z_max = star_config["disk"].get("z_max")
        bounds = star_config["disk"].get("bounds")
        p0 = star_config["disk"].get("p0")
        fitting_params = star_config["disk"].get("fitting", {})
        nwalkers = fitting_params.get("nwalkers", 64)
        nsteps = fitting_params.get("nsteps", 12000)

        weights = None
        if "weights" in star_config["disk"]:
            w_cfg = star_config["disk"]["weights"]
            if w_cfg.get("type") == "gaussian":
                weights = utils.add_gaussian_weight(
                    R_flat,
                    R_range,
                    mean=w_cfg.get("mean", None),
                    sigma_frac=w_cfg.get("sigma_frac", 0.1),
                )

        summaries = {}
        for model in star_config["disk"]["models"]:
            logger.info(f"Fitting {model} model...")
            try:
                if model == "Single-Exponential":
                    _, summary = fitting.fit_disk_emcee(
                        R_flat,
                        z_flat,
                        rho_2d_flat,
                        R_range,
                        z_max,
                        p0=p0,
                        bounds=bounds,
                        weights=weights,
                        nwalkers=nwalkers,
                        nsteps=nsteps,
                        logger=logger,
                    )
                elif model == "Double-Exponential":
                    _, summary = fitting.fit_double_exponential_disk_emcee(
                        R_flat,
                        z_flat,
                        rho_2d_flat,
                        R_range,
                        z_max,
                        bounds=bounds,
                        p0=p0,
                        weights=weights,
                        nwalkers=nwalkers,
                        nsteps=nsteps,
                        logger=logger,
                    )
                elif model == "Disk-Flaring":
                    _, summary = fitting.fit_general_emcee(
                        coords=(R_flat, z_flat),
                        data=rho_2d_flat,
                        model_func=models.double_disk_flaring,
                        param_names=[
                            "log10_rho01",
                            "log10_rho02",
                            "R_d1",
                            "R_d2",
                            "A",
                            "B",
                        ],
                        bounds=bounds,
                        p0=p0,
                        use_log_space=True,
                        nwalkers=nwalkers,
                        nsteps=nsteps,
                        weights=weights,
                        logger=logger,
                    )
                else:
                    logger.error(f"Model {model} unknown!")
                    summary = None
            except Exception as e:
                logger.error(f"Error fitting {model} model: {e}", exc_info=True)
                summary = None
            summaries[model] = summary

        # Build Evaluators for successfully fitted models
        evaluators = {}
        if (
            "Single-Exponential" in summaries
            and summaries["Single-Exponential"] is not None
        ):
            evaluators["Single-Exponential"] = lambda coords, s: models.disk_density(
                coords[0], coords[1], s["rho_0"], s["R_d"], s["z_0"]
            )
        if (
            "Double-Exponential" in summaries
            and summaries["Double-Exponential"] is not None
        ):
            evaluators["Double-Exponential"] = (
                lambda coords, s: models.double_exp_model(
                    coords[0],
                    coords[1],
                    s["rho_01"],
                    s["rho_02"],
                    s["R_d1"],
                    s["R_d2"],
                    s["z_0"],
                )
            )
        if "Disk-Flaring" in summaries and summaries["Disk-Flaring"] is not None:
            evaluators["Disk-Flaring"] = lambda coords, s: disk_flaring_log(
                coords[0],
                coords[1],
                s["log10_rho01"],
                s["log10_rho02"],  # No conversion needed in lambda
                s["R_d1"],
                s["R_d2"],
                s["A"],
                s["B"],
            )

        summaries_clean = {k: v for k, v in summaries.items() if k is not None}
        evaluators_clean = {k: v for k, v in evaluators.items() if k is not None}

        best_disk, best_disk_summary, disk_table, disk_report = compare_models(
            model_summaries=summaries_clean,
            coords=(R_flat, z_flat),
            data=rho_2d_flat,
            logger=logger,
            data_range={0: R_range, 1: (0, z_max)},
            model_evaluators=evaluators_clean,
            counts=None,
        )
        if verbose:
            logger.info(disk_report)

        results["disk"] = {
            "best_model": best_disk,
            "best_summary": best_disk_summary,
            "table": disk_table,
        }
        disk_fig = plotting.plot_disk_slices(
            R_grid,
            z_grid,
            rho_2d,
            summaries_clean,
            evaluators_clean,
            best_disk,
            z_slices=[0.5, 1.0, 3.0],
            R_range=R_range,
            component_name="Stellar Disk",
            save_path=f"{PLOT_DIR}/{halo_id}/star_disk_fit.png",
        )
        results["disk"]["figure"] = disk_fig
        if save_data:
            save_results(
                f"{RESULTS_DIR}/{halo_id}/star_disk_results.h5",
                summaries_clean,
                best_disk,
                disk_table,
            )
        logger.info(f"Disk fitting completed")
    else:
        logger.info("Skipping disk fitting (disabled in config)")

    if star_config["halo"].get("enabled", True):
        logger.info("FITTING HALO COMPONENT")

        r_range = tuple(star_config["halo"].get("r_range"))
        bounds = star_config["halo"].get("bounds")
        p0 = star_config["halo"].get("p0")
        fitting_params = star_config["halo"].get("fitting", {})
        nwalkers = fitting_params.get("nwalkers", 64)
        nsteps = fitting_params.get("nsteps", 10000)

        weights = None
        if "weights" in star_config["halo"]:
            w_cfg = star_config["halo"]["weights"]
            if w_cfg.get("type") == "gaussian":
                weights = utils.add_gaussian_weight(
                    r_1d,
                    r_range,
                    mean=w_cfg.get("mean", None),
                    sigma_frac=w_cfg.get("sigma_frac", 0.1),
                )
        summaries = {}

        for model in star_config["halo"]["models"]:
            logger.info(f"Fitting {model} model...")
            try:
                if model == "NFW":
                    _, summary = fitting.fit_halo_emcee(
                        r=r_1d,
                        rho=rho_1d,
                        r_range=r_range,
                        bounds=bounds,
                        p0=p0,
                        nwalkers=nwalkers,
                        nsteps=nsteps,
                        weights=weights,
                        logger=logger,
                    )
                elif model == "Two-Power":
                    _, summary = fitting.fit_two_power_emcee(
                        r=r_1d,
                        rho=rho_1d,
                        r_range=r_range,
                        bounds=bounds,
                        p0=p0,
                        weights=weights,
                        nwalkers=nwalkers,
                        nsteps=nsteps,
                        logger=logger,
                    )
                else:
                    logger.error(f"Model {model} unknown!")
                    summary = None
            except Exception as e:
                logger.error(f"Error fitting {model}: {e}", exc_info=True)
                summary = None
            summaries[model] = summary

        # Build evaluators for successful models
        evaluators = {}
        if "NFW" in summaries and summaries["NFW"] is not None:
            evaluators["NFW"] = lambda coords, s: models.nfw_density(
                coords[0], s["rho_0"], s["r_s"]
            )
        if "Two-Power" in summaries and summaries["Two-Power"] is not None:
            evaluators["Two-Power"] = lambda coords, s: models.two_power_model(
                coords[0], s["rho_0"], s["r_s"], s["alpha"], s["beta"]
            )

        summaries_clean = {k: v for k, v in summaries.items() if k is not None}
        evaluators_clean = {k: v for k, v in evaluators.items() if k is not None}

        best_halo, best_halo_summary, halo_table, halo_report = compare_models(
            model_summaries=summaries_clean,
            coords=(r_1d,),
            data=rho_1d,
            logger=logger,
            data_range={0: r_range},
            model_evaluators=evaluators_clean,
            counts=counts,
        )
        if verbose:
            logger.info(halo_report)

        results["halo"] = {
            "best_model": best_halo,
            "best_summary": best_halo_summary,
            "table": halo_table,
        }

        halo_fig = plotting.plot_results(
            coords=(r_1d,),
            data=rho_1d,
            data_range={0: r_range},
            model_summaries=summaries_clean,
            model_evaluators=evaluators_clean,
            best_model_name=best_halo,
            component_name="Stellar Halo",
            coord_label="r [kpc]",
            save_path=f"{PLOT_DIR}/{halo_id}/star_halo_fit.png",
        )
        results["halo"]["figure"] = halo_fig

        if save_data:
            save_results(
                f"{RESULTS_DIR}/{halo_id}/star_halo_results.h5",
                summaries_clean,
                best_halo,
                halo_table,
            )
        logger.info("Halo fitting completed")
    else:
        logger.info("Skipping halo fitting (disabled in config)")

    return results


def fit_gas_components(
    r_1d, rho_1d, R, z, rho_2d, config, logger, verbose, save_data, halo_id
):
    """Fit disk/halo for gas component.

    Args:
        r_1d: 1D radial bin centers
        rho_1d: 1D spherical density profile
        R: Cylindrical R bin centers
        z: Cylindrical z bin centers
        rho_2d: 2D cylindrical density profile
        config: full configuration dict
        logger: logger instance
        verbose: If True, log detailed fitting reports
        save_data: If True, save results to HDF5
        halo_id: Halo identifier for file paths

    Returns:
        dict: Results dictionary with keys 'disk', 'halo' (if enabled).
              Each contains 'best_model', 'best_summary', 'table', 'figure'.
    """
    results = {}
    gas_config = config["gas"]

    # Get 2d grid for R and z
    R_grid, z_grid = np.meshgrid(R, z, indexing="ij")
    rho_2d_flat = rho_2d.flatten()
    R_flat, z_flat = R_grid.flatten(), z_grid.flatten()

    # Disk Fitting
    if gas_config["disk"].get("enabled", True):
        logger.info("FITTING DISK COMPONENT")
        disk_config = gas_config["disk"]

        R_range = tuple(disk_config.get("R_range"))
        z_max = disk_config.get("z_max")
        bounds = disk_config.get("bounds")
        p0 = disk_config.get("p0")
        fitting_params = disk_config.get("fitting", {})
        nwalkers = fitting_params.get("nwalkers", 64)
        nsteps = fitting_params.get("nsteps", 12000)

        weights = None
        if "weights" in disk_config:
            w_cfg = disk_config["weights"]
            if w_cfg.get("type") == "gaussian":
                weights = utils.add_gaussian_weight(
                    R_flat,
                    R_range,
                    mean=w_cfg.get("mean", None),
                    sigma_frac=w_cfg.get("sigma_frac", 0.1),
                )

        summaries = {}
        for model in disk_config["models"]:
            logger.info(f"Fitting {model} model...")
            try:
                if model == "Single-Exponential":
                    _, summary = fitting.fit_disk_emcee(
                        R=R_flat,
                        z=z_flat,
                        rho=rho_2d_flat,
                        R_range=R_range,
                        z_max=z_max,
                        bounds=bounds,
                        weights=weights,
                        p0=p0,
                        nwalkers=nwalkers,
                        nsteps=nsteps,
                        logger=logger,
                    )
                elif model == "Double-Exponential":
                    _, summary = fitting.fit_double_exponential_disk_emcee(
                        R=R_flat,
                        z=z_flat,
                        rho=rho_2d_flat,
                        R_range=R_range,
                        z_max=z_max,
                        bounds=bounds,
                        weights=weights,
                        p0=p0,
                        nwalkers=nwalkers,
                        nsteps=nsteps,
                        logger=logger,
                    )
                else:
                    logger.error(f"Model {model} unknown!")
                    summary = None
            except Exception as e:
                logger.error(f"Error fitting {model}: {e}")
                summary = None
            summaries[model] = summary

        # Build evaluators for successful models
        evaluators = {}
        if (
            "Single-Exponential" in summaries
            and summaries["Single-Exponential"] is not None
        ):
            evaluators["Single-Exponential"] = lambda coords, s: models.disk_density(
                coords[0], coords[1], s["rho_0"], s["R_d"], s["z_0"]
            )
        if (
            "Double-Exponential" in summaries
            and summaries["Double-Exponential"] is not None
        ):
            evaluators["Double-Exponential"] = (
                lambda coords, s: models.double_exp_model(
                    coords[0],
                    coords[1],
                    s["rho_01"],
                    s["rho_02"],
                    s["R_d1"],
                    s["R_d2"],
                    s["z_0"],
                )
            )

        summaries_clean = {k: v for k, v in summaries.items() if k is not None}
        evaluators_clean = {k: v for k, v in evaluators.items() if k is not None}

        best_disk, best_disk_summary, disk_table, disk_report = compare_models(
            model_summaries=summaries_clean,
            coords=(R_flat, z_flat),
            data=rho_2d_flat,
            logger=logger,
            data_range={0: R_range, 1: (0, z_max)},
            model_evaluators=evaluators_clean,
        )
        if verbose:
            logger.info(disk_report)
        results["disk"] = {
            "best_model": best_disk,
            "best_summary": best_disk_summary,
            "table": disk_table,
        }
        disk_fig = plotting.plot_disk_slices(
            R_grid,
            z_grid,
            rho_2d,
            summaries_clean,
            evaluators_clean,
            best_disk,
            z_slices=[0.5, 1.0, 3.0],
            R_range=R_range,
            component_name="Gas Disk",
            save_path=f"{PLOT_DIR}/{halo_id}/gas_disk_fit.png",
        )
        results["disk"]["figure"] = disk_fig
        if save_data:
            save_results(
                f"{RESULTS_DIR}/{halo_id}/gas_disk_results.h5",
                summaries_clean,
                best_disk,
                disk_table,
            )
            logger.info(f"Disk fitting completed")
    else:
        logger.info("Skipping disk fitting (disabled in config)")

    if gas_config["halo"].get("enabled", True):
        halo_config = gas_config["halo"]
        logger.info("FITTING HALO COMPONENT")

        r_range = tuple(halo_config.get("r_range"))
        bounds = halo_config.get("bounds")
        p0 = halo_config.get("p0")
        fitting_params = halo_config.get("fitting", {})
        nwalkers = fitting_params.get("nwalkers", 64)
        nsteps = fitting_params.get("nsteps", 8000)

        weights = None
        if "weights" in halo_config:
            w_cfg = halo_config["weights"]
            if w_cfg.get("type") == "gaussian":
                weights = utils.add_gaussian_weight(
                    r_1d,
                    r_range,
                    mean=w_cfg.get("mean", None),
                    sigma_frac=w_cfg.get("sigma_frac", 0.1),
                )
        summaries = {}

        for model in halo_config["models"]:
            logger.info(f"Fitting {model} model....")
            try:
                if model == "NFW":
                    _, summary = fitting.fit_halo_emcee(
                        r_1d,
                        rho_1d,
                        logger,
                        r_range,
                        bounds,
                        p0,
                        nwalkers,
                        nsteps,
                        weights=weights,
                    )
                elif model == "Two-Power":
                    _, summary = fitting.fit_two_power_emcee(
                        r_1d,
                        rho_1d,
                        logger,
                        r_range,
                        bounds,
                        p0,
                        weights,
                        nwalkers,
                        nsteps,
                    )
                else:
                    logger.error(f"Model {model} unknown!")
                    summary = None
            except Exception as e:
                logger.error(f"Error fitting {model}: {e}", exc_info=True)
                summary = None
            summaries[model] = summary

        # build evaluators for successful models
        evaluators = {}
        if "NFW" in summaries and summaries["NFW"] is not None:
            evaluators["NFW"] = lambda coords, s: models.nfw_density(
                coords[0], s["rho_0"], s["r_s"]
            )
        if "Two-Power" in summaries and summaries["Two-Power"] is not None:
            evaluators["Two-Power"] = lambda coords, s: models.two_power_model(
                coords[0], s["rho_0"], s["r_s"], s["alpha"], s["beta"]
            )

        summaries_clean = {k: v for k, v in summaries.items() if k is not None}
        evaluators_clean = {k: v for k, v in evaluators.items() if k is not None}

        best_halo, best_halo_summary, halo_table, halo_report = compare_models(
            model_summaries=summaries_clean,
            coords=(r_1d,),
            data=rho_1d,
            logger=logger,
            data_range={0: r_range},
            model_evaluators=evaluators_clean,
        )
        if verbose:
            logger.info(halo_report)
        results["halo"] = {
            "best_model": best_halo,
            "best_summary": best_halo_summary,
            "table": halo_table,
        }

        halo_fig = plotting.plot_results(
            coords=(r_1d,),
            data=rho_1d,
            data_range={0: r_range},
            model_summaries=summaries_clean,
            model_evaluators=evaluators_clean,
            best_model_name=best_halo,
            component_name="Gas Halo",
            coord_label="r [kpc]",
            save_path=f"{PLOT_DIR}/{halo_id}/gas_halo_fit.png",
        )
        results["halo"]["figure"] = halo_fig

        if save_data:
            save_results(
                f"{RESULTS_DIR}/{halo_id}/gas_halo_results.h5",
                summaries_clean,
                best_halo,
                halo_table,
            )
        logger.info("Halo fitting completed")
    else:
        logger.info("Skipping halo fitting (disabled in config)")

    return results


def fit_dm_components(
    r_1d, rho_1d, counts, config, logger, verbose, save_data, halo_id
):
    """Fit Halo for dark matter component.

    Args:
        r_1d: 1D radial bins centers
        rho_1d: 1D spherical density profile
        counts: Particle counts per bin (for spherical profile)
        config: Full configuration dict
        logger: Logger instance
        verbose: If True, log detailed fitting reports
        save_data: If True, save results to HDF5
        halo_id: Halo identifier for file paths

    Returns:
        dict: Results dictionary with keys 'halo' (if enabled).
              Each contains 'best_model', 'best_summary', 'table', 'figure'.
    """
    results = {}
    dm_config = config["dm"]

    if not dm_config["halo"].get("enabled", True):
        logger.warning("Dark matter halo fitting is disabled in config")
        return results

    logger.info("FITTING HALO COMPONENT")
    halo_config = dm_config["halo"]
    r_range = tuple(halo_config.get("r_range"))
    bounds = halo_config.get("bounds")
    p0 = halo_config.get("p0")
    fitting_params = halo_config.get("fitting", {})
    nwalkers = fitting_params.get("nwalkers", 64)
    nsteps = fitting_params.get("nsteps", 10000)

    weights = None
    if "weights" in halo_config:
        w_cfg = halo_config["weights"]
        if w_cfg.get("type") == "gaussian":
            weights = utils.add_gaussian_weight(
                r_1d,
                r_range,
                mean=w_cfg.get("mean", None),
                sigma_frac=w_cfg.get("sigma_frac", 0.1),
            )

    summaries = {}

    for model in halo_config["models"]:
        logger.info(f"Fitting {model} model...")
        try:
            if model == "NFW":
                _, summary = fitting.fit_halo_emcee(
                    r=r_1d,
                    rho=rho_1d,
                    r_range=r_range,
                    bounds=bounds,
                    p0=p0,
                    nwalkers=nwalkers,
                    nsteps=nsteps,
                    weights=weights,
                    logger=logger,
                )
            elif model == "Two-Power":
                _, summary = fitting.fit_two_power_emcee(
                    r=r_1d,
                    rho=rho_1d,
                    r_range=r_range,
                    bounds=bounds,
                    p0=p0,
                    weights=weights,
                    nwalkers=nwalkers,
                    nsteps=nsteps,
                    logger=logger,
                )
            else:
                logger.error(f"Model {model} unknown!")
                summary = None
        except Exception as e:
            logger.error(f"Error fitting {model}; {e}", exc_info=True)
            summary = None
        summaries[model] = summary

    # build evaluators
    evaluators = {}
    if "NFW" in summaries and summaries["NFW"] is not None:
        evaluators["NFW"] = lambda coords, s: models.nfw_density(
            coords[0], s["rho_0"], s["r_s"]
        )
    if "Two-Power" in summaries and summaries["Two-Power"] is not None:
        evaluators["Two-Power"] = lambda coords, s: models.two_power_model(
            coords[0], s["rho_0"], s["r_s"], s["alpha"], s["beta"]
        )

    summaries_clean = {k: v for k, v in summaries.items() if k is not None}
    evaluators_clean = {k: v for k, v in evaluators.items() if k is not None}

    best_halo, best_halo_summary, halo_table, halo_report = compare_models(
        model_summaries=summaries_clean,
        coords=(r_1d,),
        data=rho_1d,
        logger=logger,
        data_range={0: r_range},
        model_evaluators=evaluators_clean,
        counts=counts,
    )
    if verbose:
        logger.info(halo_report)

    results["halo"] = {
        "best_model": best_halo,
        "best_summary": best_halo_summary,
        "table": halo_table,
    }

    halo_fig = plotting.plot_results(
        coords=(r_1d,),
        data=rho_1d,
        data_range={0: r_range},
        model_summaries=summaries_clean,
        model_evaluators=evaluators_clean,
        best_model_name=best_halo,
        component_name="DM Halo",
        coord_label="r [kpc]",
        save_path=f"{PLOT_DIR}/{halo_id}/dm_halo_fit.png",
    )
    results["halo"]["figure"] = halo_fig

    if save_data:
        save_results(
            f"{RESULTS_DIR}/{halo_id}/dm_halo_results.h5",
            summaries_clean,
            best_halo,
            halo_table,
        )
    return results


def calculate_model_masses(results, radii, component_type, logger):
    """
    Calculate mass estimates from fitted models at specified radii.

    Args:
        results: Dictionary containing fitting results for each component
        radii: Array of radii to calculate masses at [kpc]
        component_type: 'star', 'gas', or 'dm'
        logger: Logger instance

    Returns:
        dict: Masses from each component model at each radius
    """
    masses = {"radii": radii, "bulge": [], "disk": [], "halo": [], "total": []}

    for r in radii:
        M_bulge = 0.0
        M_disk = 0.0
        M_halo = 0.0

        # Calculate bulge mass if available
        if "bulge" in results:
            best_model = results["bulge"]["best_model"]
            summary = results["bulge"]["best_summary"]

            if best_model == "Hernquist":
                M_bulge, _ = dp.integrate_spherical_mass(
                    models.hernquist_density, r, summary["rho_0"], summary["r_s"]
                )
            elif best_model == "Exponential":
                M_bulge, _ = dp.integrate_spherical_mass(
                    lambda r, rho0, rs: rho0 * np.exp(-r / rs),
                    r,
                    summary["rho_0"],
                    summary["r_s"],
                )
            elif best_model == "Two-Power":
                M_bulge, _ = dp.integrate_spherical_mass(
                    models.two_power_model,
                    r,
                    summary["rho_0"],
                    summary["r_s"],
                    summary["alpha"],
                    summary["beta"],
                )
            elif best_model == "Exp-Cutoff":
                M_bulge, _ = dp.integrate_spherical_mass(
                    models.exp_cutoff,
                    r,
                    summary["rho0"],
                    summary["r_core"],
                    summary["r_cutoff"],
                    summary["alpha"],
                )

        # Calculate disk mass if available
        if "disk" in results:
            best_model = results["disk"]["best_model"]
            summary = results["disk"]["best_summary"]

            # Use 2*r as z_max for disk integration (reasonable approximation)
            z_max = 2 * r

            if best_model == "Single-Exponential":
                M_disk, _ = dp.integrate_disk_mass(
                    r, z_max, summary["rho_0"], summary["R_d"], summary["z_0"]
                )
            elif best_model == "Double-Exponential":
                # Calculate thin + thick disk
                M_thin, _ = dp.integrate_disk_mass(
                    r, z_max, summary["rho_01"], summary["R_d1"], summary["z_0"]
                )
                M_thick, _ = dp.integrate_disk_mass(
                    r, z_max, summary["rho_02"], summary["R_d2"], summary["z_0"]
                )
                M_disk = M_thin + M_thick

        # Calculate halo mass if available
        if "halo" in results:
            best_model = results["halo"]["best_model"]
            summary = results["halo"]["best_summary"]

            if best_model == "NFW":
                M_halo, _ = dp.integrate_spherical_mass(
                    models.nfw_density, r, summary["rho_0"], summary["r_s"]
                )
            elif best_model == "Two-Power":
                M_halo, _ = dp.integrate_spherical_mass(
                    models.two_power_model,
                    r,
                    summary["rho_0"],
                    summary["r_s"],
                    summary["alpha"],
                    summary["beta"],
                )

        masses["bulge"].append(M_bulge)
        masses["disk"].append(M_disk)
        masses["halo"].append(M_halo)
        masses["total"].append(M_bulge + M_disk + M_halo)

    # Convert to arrays
    for key in ["bulge", "disk", "halo", "total"]:
        masses[key] = np.array(masses[key])

    return masses


def compare_masses(model_masses, sim_masses, component_type, logger):
    """
    Compare model mass estimates with actual simulation masses.

    Args:
        model_masses: Dictionary from calculate_model_masses()
        sim_masses: Dictionary from metadata['enclosed_masses']
        component_type: 'star', 'gas', or 'dm'
        logger: Logger instance
    """
    logger.info("\n" + "=" * 70)
    logger.info("MASS COMPARISON: MODEL vs SIMULATION")
    logger.info("=" * 70)

    # Map component types to simulation mass keys
    sim_key_map = {"star": "M_star", "gas": "M_gas", "dm": "M_dm"}

    if component_type not in sim_key_map:
        logger.warning(f"Unknown component type: {component_type}")
        return

    sim_key = sim_key_map[component_type]
    radii = model_masses["radii"]

    logger.info(f"\n{component_type.upper()} Component Mass Comparison:")
    logger.info("-" * 70)
    logger.info(
        f"{'Radius':>8} {'Sim Mass':>12} {'Model Mass':>12} {'Difference':>12} {'Rel. Error':>12}"
    )
    logger.info(
        f"{'[kpc]':>8} {'[Msun]':>12} {'[Msun]':>12} {'[Msun]':>12} {'[%]':>12}"
    )
    logger.info("-" * 70)

    for i, r in enumerate(radii):
        # Find corresponding simulation mass (radii might not match exactly)
        sim_idx = np.argmin(np.abs(sim_masses["radii"] - r))
        sim_mass = sim_masses[sim_key][sim_idx]
        model_mass = model_masses["total"][i]

        diff = model_mass - sim_mass
        rel_error = (diff / sim_mass) * 100 if sim_mass > 0 else np.inf

        logger.info(
            f"{r:8.1f} {sim_mass:12.3e} {model_mass:12.3e} {diff:12.3e} {rel_error:12.2f}"
        )

    # Calculate statistics
    sim_masses_interp = np.interp(radii, sim_masses["radii"], sim_masses[sim_key])
    rel_errors = ((model_masses["total"] - sim_masses_interp) / sim_masses_interp) * 100

    logger.info("-" * 70)
    logger.info(f"Mean relative error: {np.mean(rel_errors):.2f}%")
    logger.info(f"Median relative error: {np.median(rel_errors):.2f}%")
    logger.info(f"RMS relative error: {np.sqrt(np.mean(rel_errors**2)):.2f}%")
    logger.info("=" * 70 + "\n")


def generate_component_summary(results, logger):
    """generate and log summary of fitting results."""
    summary_lines = []
    summary_lines.append("=" * 70)
    summary_lines.append("SUMMARY OF BEST-FIT MODELS")
    summary_lines.append("=" * 70)

    for component_name in ["bulge", "disk", "halo"]:
        if component_name in results:
            comp = results[component_name]
            summary_lines.append("")
            summary_lines.append(f"{component_name.upper()} COMPONENT:")
            summary_lines.append(f"Model: {comp['best_model']}")
            summary_lines.append(
                utils.profile_eq_str(comp["best_model"], comp["best_summary"])
            )

            summary_lines.append(
                f"\nReduced χ² = {comp['table'][comp['best_model']]['chi2_reduced']:.3f}"
            )
            summary_lines.append(
                f"BIC = {comp['table'][comp['best_model']]['bic']:.1f}"
            )

    summary_lines.append("=" * 70)
    logger.info("\n".join(summary_lines))


def main():
    t0 = time()
    parser = argparse.ArgumentParser(
        description="FOGGIE Dynamics Analysis: Fit density profiles to simulation data",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python foggie_analysis.py --halo 8508 --component star --show_plots --save_data
  python foggie_analysis.py --halo 8508 --component dm --save_data
  python foggie_analysis.py --halo 8508 --component gas
  python foggie_analysis.py --halo 8508 --component all --save_data
  python foggie_analysis.py --halo 8508 --component star --config config_path
        """,
    )

    parser.add_argument(
        "--halo", type=str, required=True, help="Halo ID (dataset directory name)"
    )
    parser.add_argument(
        "--component",
        type=str,
        required=True,
        default="all",
        choices=["star", "dm", "gas", "all"],
        help="Component to analyze: star, dm, gas, or all",
    )
    parser.add_argument(
        "--show_plots",
        action="store_true",
        help="Display matplotlib plots after fitting",
    )
    parser.add_argument(
        "--save_data", action="store_true", help="Save fitting results to HDF5 files"
    )
    parser.add_argument(
        "--config",
        type=str,
        help="Path to config YAML (default: config/fitting_config.yaml)",
    )
    parser.add_argument(
        "--nwalkers", type=int, default=None, help="Override number of MCMC walkers"
    )
    parser.add_argument(
        "--nsteps", type=int, default=None, help="Override number of MCMC steps"
    )

    args = parser.parse_args()

    # load config first
    try:
        config = load_config(args.config)
    except FileNotFoundError:
        print("Config file not found. Please create config/fitting_config.yaml")
        sys.exit(1)

    # Format halo_id: if just a number is provided, prepend 'halo_00'
    halo_id = args.halo if args.halo.startswith("halo_") else f"halo_00{args.halo}"
    component = args.component
    show_plots = args.show_plots
    save_data = args.save_data

    # Handle 'all' components case
    components_to_run = ["star", "gas", "dm"] if component == "all" else [component]

    logger = setup_logging(halo_id, component)
    logger.info("Starting FOGGIE Dynamics Analysis...")

    for current_component in components_to_run:
        logger.info(f"\n{'='*70}")
        logger.info(f"PROCESSING COMPONENT: {current_component.upper()}")
        logger.info(f"{'='*70}\n")

        component_config = config.get(current_component)
        if component_config is None:
            logger.error(f"No configuration found for component '{current_component}'")
            continue

        # Apply CLI overrides if provided
        if args.nwalkers is not None:
            # Override nwalkers in all fitting sections
            for subcomponent in component_config.values():
                if "fitting" in subcomponent:
                    subcomponent["fitting"]["nwalkers"] = args.nwalkers
        if args.nsteps is not None:
            for subcomponent in component_config.values():
                if "fitting" in subcomponent:
                    subcomponent["fitting"]["nsteps"] = args.nsteps

        logger.info(f"Loaded configuration for {current_component}")

        # Load Simulation Data
        ds = yt.load(str(DATA_DIR / halo_id / f"3d_profile_{current_component}.h5"))  # type: ignore
        metadata_file = DATA_DIR / halo_id / "metadata.h5"
        metadata = {}
        with h5py.File(metadata_file, "r") as f:
            metadata.update({key: f[key][()] for key in f.keys() if isinstance(f[key], h5py.Dataset)})  # type: ignore
            metadata.update(dict(f.attrs.items()))

        # get density profile data
        rho_1d, r_1d, r_bins, counts = extract_density(
            ds, "spherical", current_component, True
        )
        rho_2d, R, z, R_bins, z_bins = extract_density(
            ds, "cylindrical", current_component
        )

        if current_component == "star":
            # looking at fitting bulge/disk/halo for stars
            logger.info("Fitting stellar component models...")
            t1 = time()

            results = fit_stellar_components(
                r_1d,
                rho_1d,
                R,
                z,
                rho_2d,
                counts,
                config,
                logger,
                True,
                save_data,
                halo_id,
            )
            # DEBUG:
            logger.info(f"Summary keys: {list(results['disk']['best_summary'].keys())}")

            generate_component_summary(results, logger)

            # Calculate and compare masses
            if "enclosed_masses" in metadata:
                sim_masses = metadata["enclosed_masses"]
                model_masses = calculate_model_masses(
                    results, sim_masses["radii"], "star", logger
                )
                compare_masses(model_masses, sim_masses, "star", logger)
            else:
                logger.warning(
                    "No enclosed mass data found in metadata. Skipping mass comparison."
                )

            t2 = time()
            logger.info(f"Stellar fitting complete in {t2 - t1:.1f} seconds")

        elif current_component == "gas":
            # Gas fitting = disk + halo
            # Fit single and double exp model
            t1 = time()
            logger.info("Fitting Disk models for Gas...")

            results = fit_gas_components(
                r_1d, rho_1d, R, z, rho_2d, config, logger, True, save_data, halo_id
            )

            generate_component_summary(results, logger)

            # Calculate and compare masses
            if "enclosed_masses" in metadata:
                sim_masses = metadata["enclosed_masses"]
                model_masses = calculate_model_masses(
                    results, sim_masses["radii"], "gas", logger
                )
                compare_masses(model_masses, sim_masses, "gas", logger)
            else:
                logger.warning(
                    "No enclosed mass data found in metadata. Skipping mass comparison."
                )

            t2 = time()
            logger.info(f"Gas fitting complete in {t2 - t1:.1f} seconds")

        elif current_component == "dm":
            # DM fitting = halo only
            t1 = time()
            logger.info("\nFitting Halo models for Dark Matter...")

            results = fit_dm_components(
                r_1d, rho_1d, counts, config, logger, True, save_data, halo_id
            )

            generate_component_summary(results, logger)

            # Calculate and compare masses
            if "enclosed_masses" in metadata:
                sim_masses = metadata["enclosed_masses"]
                model_masses = calculate_model_masses(
                    results, sim_masses["radii"], "dm", logger
                )
                compare_masses(model_masses, sim_masses, "dm", logger)
            else:
                logger.warning(
                    "No enclosed mass data found in metadata. Skipping mass comparison."
                )

            t2 = time()
            logger.info(f"Dark Matter Halo fitting completed in {t2 - t1:.1f} seconds.")

        else:
            logger.warning(f"Component '{current_component}' not in star, gas, dm.")
            continue

    logger.info(f"Total analysis time: {time() - t0:.1f} seconds.")

    if show_plots:
        plt.show()


if __name__ == "__main__":
    main()
