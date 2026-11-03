"""
Workflow orchestration for FOGGIE dynamics analysis.

Takes a halo's snapshots from raw simulation output to fitted density
models and enclosed masses for the stellar, gas and dark matter components.

Pipeline (top to bottom):

    process_time_series        loop over snapshots; generate or load profiles,
      |                        record results, save evolution data incrementally
      +- generate_snapshot_profiles   build 3D profiles + metadata for one snapshot
      |                               (steps live in snapshot_profiles.py)
      +- process_snapshot      fit one component and compute its masses/statistics
           +- fit_components          fit each subcomponent listed in SUBCOMPONENTS
                +- fit_component_models   fitting range, weights, model comparison, saving
                     +- fit_models_parallel   run each candidate model (serial or parallel)

Key pieces:
    DensityProfile   1D spherical and 2D cylindrical density data for a component
    FittingResult    best model, posterior samples, summary and comparison table
    SUBCOMPONENTS    component -> [(subcomponent, geometry)], e.g. star -> bulge/disk/halo
    ModelRegistry    (models.py) maps model names to fitters and evaluators

Fitting ranges come from each subcomponent's config and are converted to kpc
per snapshot, either as R200 fractions or redshift-scaled z=0 ranges
(see _data_range).

TODO:
    - Unit tests beyond tests/integration
    - Move inline defaults (MCMC nwalkers/nsteps, fallback mass radii) to config
    - Reduce side effects in process_snapshot (it loads from and saves to disk)
"""
import functools
import gc
import logging
import multiprocessing
import os
import re
from glob import glob
from time import time
import tracemalloc
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any, Dict, List, Optional, Tuple, Union

if TYPE_CHECKING:
    from yt.data_objects.profiles import ProfileND  # type: ignore

import numpy as np
import psutil  # type: ignore
import yt  # type: ignore

# Import local modules
from .analysis import (
    calc_masses_with_uncertainties,
    calculate_component_statistics,
    compare_masses,
    compare_models,
    save_evolution_data,
    save_results,
)
from .config import ProjectPaths, load_config, setup_logging
from .filters_and_utils import (
    add_gaussian_weight,
    add_inverse_radius_weight,
    convert_r200_range_to_kpc,
    scale_range_with_redshift,
)
from .io import load_hdf5
from .io import save_hdf5 as save_metadata
from .models import ModelRegistry
from .snapshot_profiles import (
    available_components,
    load_dataset,
    register_particle_filters,
    find_center,
    compute_r200,
    make_r200_projections,
    shape_analysis,
    angular_momentum,
    disk_frame,
    build_profiles,
    mass_radii,
    enclosed_masses,
    build_metadata,
    snapshot_data_dir,
    write_profiles
)

PATHS = ProjectPaths.create()
DATA_DIR = PATHS.data
RESULTS_DIR = PATHS.results
PLOT_DIR = PATHS.plots


@dataclass
class DensityProfile:
    """Container for density profile data."""

    r_1d: np.ndarray
    rho_1d: np.ndarray
    r_bins: np.ndarray
    R: np.ndarray
    z: np.ndarray
    rho_2d: np.ndarray
    R_bins: np.ndarray
    z_bins: np.ndarray

    # Quality metrics - 1D radial (particles only, None for gas)
    counts: Optional[np.ndarray] = None  # Number of occupied bins per radial shell

    # Statistics - 3D bin arrays (all components)
    mean_mass: Optional[np.ndarray] = None  # mean mass in each 3D bin
    std: Optional[np.ndarray] = None  # standard deviation
    frac: Optional[np.ndarray] = None  # fractional variance
    N: Optional[np.ndarray] = None  # bin counts
    sem: Optional[np.ndarray] = None  # standard error of the mean

    @classmethod
    def from_profile_dataset(
        cls,
        ds,
        component: str,
        include_stats: bool = False
    ):
        """
        Create DensityProfile from a yt profile dataset.

        This is the recommended way to create a DensityProfile from saved
        profile data. Extracts both spherical (1D) and cylindrical (2D)
        density profiles, optionally including statistical information.

        Args:
            ds: yt profile dataset (loaded from .h5 file)
            component: Component name - 'star', 'dm', or 'gas'
            include_stats: If True, extract profile statistics (mean, std, sem, etc.)

        Returns:
            DensityProfile: Instance with all density and (optionally) statistical data

        Examples:
            >>> ds = yt.load('3d_profile_star.h5')
            >>> profile = DensityProfile.from_profile_dataset(ds, 'star')
            >>> print(f"Radial range: {profile.r_1d.min():.2f} - {profile.r_1d.max():.2f} kpc")
        """
        from .filters_and_utils import extract_density

        spherical = extract_density(
            ds, "spherical", component, include_stats=include_stats
        )
        cylindrical = extract_density(ds, "cylindrical", component)

        # Extract stats from nested dict if present
        stats = spherical.get("stats", {}) if include_stats else {}

        return cls(
            r_1d=spherical["r_1d"],
            rho_1d=spherical["rho_1d"],
            r_bins=spherical["r_bins"],
            R=cylindrical["R"],
            z=cylindrical["z"],
            rho_2d=cylindrical["rho_2d"],
            R_bins=cylindrical["R_bins"],
            z_bins=cylindrical["z_bins"],
            counts=spherical.get("counts"),
            mean_mass=stats.get("mean"),
            std=stats.get("std"),
            frac=stats.get("frac"),
            N=stats.get("N"),
            sem=stats.get("sem"),
        )

    @classmethod
    def from_hdf5(
        cls,
        filepath: Path | str,
        component: str,
        include_stats: bool = False
    ):
        """
        Load DensityProfile from HDF5 file.
        """
        ds = yt.load(str(filepath))
        return cls.from_profile_dataset(ds, component, include_stats)
    
    @classmethod
    def from_profile(cls, profile, component, include_stats=False):
        """Create DensityProfile from an in-memory yt profile
        (same result as saving and reloading it)."""
        return cls.from_profile_dataset(_InMemoryProfile(profile), component, include_stats)


class _InMemoryProfile:
    """Exposes an in-memory yt profile through the ds.data[...] lookups extract_density uses on saved profiles."""

    def __init__(self, profile):
        self.data = {f"{ax}_bins": getattr(profile, f"{ax}_bins") for ax in "xyz"}
        for (_, fname), arr in profile.field_data.items():  # e.g. ("star", "particle_mass") -> "particle_mass"
            self.data[fname] = arr


@dataclass
class FittingResult:
    """Container for model fitting results."""

    best_model: Optional[str]
    samples: Optional[np.ndarray]
    summary: Optional[Dict[str, Any]]
    model_comparison: Dict[str, Any]

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dict for backward compatibility."""
        return {
            "best_model": self.best_model,
            "best_summary": self.summary,
            "table": self.model_comparison.get("table"),
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "FittingResult":
        """Create FittingResult from dict (for backward compatibility)."""
        return cls(
            best_model=data.get("best_model"),
            samples=None,
            summary=data.get(data["best_model"]) if data.get("best_model") else None,
            model_comparison={"table": data.get("table")},
        )

    @classmethod
    def from_hdf5(cls, path: Path | str) -> "FittingResult":
        """Create FittingResult from hdf5 file"""
        path = Path(path)
        data = load_hdf5(path)
        return FittingResult.from_dict(data)

    @classmethod
    def empty(cls) -> "FittingResult":
        """Result for a skipped or failed fit."""
        return cls(best_model=None, samples=None, summary=None,
                   model_comparison={"table": None})


def log_memory_usage(logger, stage: str):
    """Log current and peak memory usage."""
    process = psutil.Process(os.getpid())
    mem_gb = process.memory_info().rss / 1024**3
    logger.info(f"Memory at {stage}: {mem_gb:.2f} GB")

    if tracemalloc.is_tracing():
        current, peak = tracemalloc.get_traced_memory()
        logger.info(f"Peak traced memory: {peak / 1024**3} GB")


def _fit_single_model_worker(
    model_name: str,
    geometry: str,
    coords: Tuple[np.ndarray, ...],
    data: np.ndarray,
    data_range: Dict,
    bounds: Dict,
    p0: Dict,
    weights: Optional[np.ndarray],
    nwalkers: int,
    nsteps: int,
    uncertainties: Optional[np.ndarray],
    fixed_uncertainty: Optional[float],
) -> Tuple[str, Optional[Dict]]:
    """
    Worker function for parallel model fitting.

    Must be at module level (not nested) to be picklable by ProcessPoolExecutor.

    Args:
        model_name: Name of model to fit
        geometry: 'spherical' or 'cylindrical'
        coords: Coordinate arrays
        data: Density data
        data_range: Fitting range constraints
        bounds: Parameter bounds
        p0: Initial parameter guesses
        weights: Optional weights for data points
        nwalkers: Number of MCMC walkers
        nsteps: Number of MCMC steps
        uncertainties: Optional measured uncertainties on data points.
                      If provided, used in chi-squared calculation.
        fixed_uncertainty: Fractional uncertainty used when uncertainties is None (default 0.1).

    Returns:
        Tuple of (model_name, summary_dict) where summary_dict is None if fitting failed
    """
    # Create a simple logger for this worker process
    logger = logging.getLogger(__name__)

    logger.info(f"Fitting {model_name} model...")
    try:
        args, kwargs = ModelRegistry.construct_fitter_args(
            geometry=geometry,
            coords=coords,
            data=data,
            data_range=data_range,
            bounds=bounds,
            p0=p0,
            weights=weights,
            logger=logger,
            nwalkers=nwalkers,
            nsteps=nsteps,
            uncertainties=uncertainties,
            fixed_uncertainty=fixed_uncertainty,
        )
        _, summary = ModelRegistry.fit_model(model_name, *args, **kwargs)

        # Log fitted parameters
        if summary is not None:
            param_names = ModelRegistry.get_param_names(model_name)
            fitted_params = {name: summary.get(name) for name in param_names}
            logger.info(f"Fitted parameters for {model_name}: {fitted_params}")
        else:
            logger.warning(f"No summary returned for {model_name}")

        return model_name, summary

    except Exception as e:
        logger.error(f"Error fitting {model_name}: {e}", exc_info=True)
        return model_name, None


def fit_models_parallel(
    models: List[str],
    geometry: str,
    coords: Tuple[np.ndarray, ...],
    data: np.ndarray,
    data_range: Dict,
    bounds: Dict,
    p0: Dict,
    weights: Optional[np.ndarray],
    logger: logging.Logger,
    nwalkers: int,
    nsteps: int,
    n_jobs: int,
    uncertainties: Optional[np.ndarray],
    fixed_uncertainty: Optional[float] = 0.1,
) -> Dict[str, Optional[Dict]]:
    """
    Fit multiple models in serial or parallel.

    Args:
        models: List of model names to fit
        geometry: 'spherical' or 'cylindrical'
        coords: Coordinate tuple
        data: Density data
        data_range: Fitting range
        bounds: Parameter bounds
        p0: Initial guesses
        weights: Optional weights
        logger: Logger instance
        nwalkers: Number of MCMC walkers
        nsteps: Number of MCMC steps
        n_jobs: Number of parallel jobs (-1 = all cores, 1 = serial)
        uncertainties: Optional measured uncertainties on data points.
                      If provided, used in chi-squared calculation during MCMC fitting.
                      Should be extracted from DensityProfile.sem or similar.
        fixed_uncertainty: Fractional uncertainty used when uncertainties is None (default 0.1).

    Returns:
        Dict mapping model names to fit summaries
    """
    fit = functools.partial(
        _fit_single_model_worker,
        geometry=geometry, coords=coords, data=data, data_range=data_range,
        bounds=bounds, p0=p0, weights=weights, nwalkers=nwalkers, nsteps=nsteps,
        uncertainties=uncertainties, fixed_uncertainty=fixed_uncertainty,
    )

    if n_jobs == -1:
        n_jobs = multiprocessing.cpu_count()

    n_jobs = min(n_jobs, len(models))

    # Serial or parallel execution based on n_jobs
    if n_jobs <= 1:
        # Serial execution (for debugging)
        logger.info(f"Fitting {len(models)} models in serial mode")
        return dict(fit(model_name=m) for m in models)

    # Parallel execution
    logger.info(f"Fitting {len(models)} models in parallel using {n_jobs} workers")
    summaries = {}

    with ProcessPoolExecutor(max_workers=n_jobs) as executor:
        futures = {executor.submit(fit, model_name=m): m for m in models}
        # Collect results as they complete
        for future in as_completed(futures):
            model = futures[future]
            try:
                _, summaries[model] = future.result()
                logger.info(f"Completed {model}")
            except Exception as e:
                logger.error(f"Failed to retrieve result for {model}: {e}", exc_info=True)
                summaries[model] = None

    return summaries


COORD_NAMES = {"spherical": ("r",), "cylindrical": ("R", "z")}


def _log_data_summary(coords, data, geometry, logger):
    valid = np.isfinite(data) & (data > 0)
    msg = f"  Data summary: {valid.sum()}/{len(data)} valid points"
    if not valid.any():
        logger.info(msg + " - NO VALID DATA")
        return
    for name, c in zip(COORD_NAMES[geometry], coords):
        msg += f", {name}=[{c.min():.3f}, {c.max():.3f}] kpc"
    logger.info(msg + f", rho=[{data[valid].min():.2e}, {data[valid].max():.2e}] Msun/kpc^3")


def _data_range(cfg, geometry, metadata, name, logger):
    """Fitting range in kpc as {axis: (lo, hi)}, or None if the config is incomplete."""
    metadata = metadata or {}
    redshift = metadata.get("redshift", 0.0)
    R200_kpc = metadata.get("R200_kpc")
    scaling = cfg.get("range_scaling", {})
    units = scaling.get("units", "kpc")
    if units == "r200" and R200_kpc is None:
        logger.warning(f"R200 scaling requested but R200_kpc not in metadata. "
                       f"Falling back to kpc mode for {name}")
        units = "kpc"
    key = "r_range" if geometry == "spherical" else "R_range"
    if cfg.get(key) is None or (geometry == "cylindrical" and cfg.get("z_max") is None):
        logger.error(f"Missing required range ('{key}'/'z_max') in {name} config")
        return None

    data_range = {0: _resolve_range(cfg[key], units, R200_kpc, redshift, scaling)}
    if geometry == "cylindrical":
        z_min, z_max = cfg.get("z_min", 0.0), cfg["z_max"]
        if units == "r200":
            z_min, z_max = z_min * R200_kpc, max(0.3, z_max * R200_kpc)
        elif scaling.get("mode", "none") != "none":
            z_min, z_max = z_min / (1 + redshift), max(0.3, z_max / (1 + redshift))
        data_range[1] = (z_min, z_max)
    logger.info(f"{name} fitting range (kpc): {data_range}")
    return data_range


def _make_weights(cfg, x, fit_range):
    w = cfg.get("weights")
    if not w:
        return None
    if w.get("type") == "gaussian":
        return add_gaussian_weight(x, fit_range, mean=w.get("mean"),
                                   sigma_frac=w.get("sigma_frac", 0.1))
    if w.get("type") == "inverse_radius":
        return add_inverse_radius_weight(x, fit_range, power=w.get("power", 1.0),
                                         r_scale=w.get("r_scale"))
    return None


def _resolve_range(rng, units, R200_kpc, redshift, scaling):
    """Convert a config (lo, hi) range to physical kpc for this snapshot."""
    rng = tuple(rng)
    if units == "r200":
        return convert_r200_range_to_kpc(rng, R200_kpc)
    mode = scaling.get("mode", "none")
    if mode == "none":
        return rng
    return scale_range_with_redshift(
        rng, redshift, min_scale=scaling.get("min_scale", 0.1),
        max_scale=scaling.get("max_scale", 1.0),
        scaling_mode=mode
    )


def fit_component_models(
    component_config: Dict,  # config['star']['bulge']
    component_name: str,  # 'bulge', 'disk', 'halo'
    component_type: str,  # 'star', 'gas', 'dm'
    geometry: str,  # 'spherical' or 'cylindrical'
    coords: Tuple[np.ndarray, ...],  # Already prepared: (r,) or (R_flat, z_flat)
    data: np.ndarray,  # Prepared: rho_1d, rho_2d_flat
    logger: logging.Logger,
    halo_id: str,
    n_jobs: int,
    save_data: bool = True,
    verbose: bool = True,
    counts: Optional[np.ndarray] = None,
    uncertainties: Optional[np.ndarray] = None,
    fixed_uncertainty: Optional[float] = 0.1,
    metadata: Optional[Dict] = None,
    experiment: Optional[str] = None,
) -> FittingResult:
    """
    Universal fitting function for any component and geometry.

    Handles:
    - Spherical profiles (bulge, halo)
    - Cylindrical profiles (disk)
    - All model types through registry lookup
    - Weight calculation
    - Model comparison
    - Result saving

    Args:
        component_config: Component-specific config (e.g., config['star']['bulge'])
        component_name: Name of component ('bulge', 'disk', 'halo')
        component_type: Type of component ('star', 'gas', 'dm')
        geometry: 'spherical' or 'cylindrical'
        coords: Coordinate tuple - (r,) for spherical, (R_flat, z_flat) for cylindrical
        data: Flattened density data
        logger: Logger instance
        halo_id: Halo identifier
        n_jobs: Number of parallel jobs
        save_data: Whether to save results to disk
        verbose: Whether to log detailed reports
        counts: Optional particle counts (for spherical profiles)
        uncertainties: Optional measured uncertainties (sigma) on data points.
                      If provided, used in chi-squared: χ² = Σ[(data-model)²/σ²].
                      Should be extracted from DensityProfile.sem (standard error of mean).
                      Must match shape of coords/data arrays

    Returns:
        FittingResult: Container with best_model, samples, summary, and model_comparison
    """
    # Check if enabled
    if not component_config.get("enabled", True):
        logger.info(f"Skipping {component_name} fitting (disabled in config)")
        return FittingResult.empty()

    if geometry not in COORD_NAMES:
        logger.error(f"Unknown geometry {geometry}")
        return FittingResult.empty()
    
    logger.info(f"FITTING {component_name.upper()} COMPONENT")
    _log_data_summary(coords, data, geometry, logger)
    
    data_range = _data_range(component_config, geometry, metadata, component_name, logger)
    if data_range is None:
        return FittingResult.empty()
    
    fitting = component_config.get("fitting", {})
    summaries = fit_models_parallel(
        models=component_config["models"],
        geometry=geometry,
        coords=coords,
        data=data,
        data_range=data_range,
        bounds=component_config["bounds"],
        p0=component_config["p0"],
        weights=_make_weights(component_config, coords[0], data_range[0]),
        logger=logger,
        nwalkers=fitting.get("nwalkers", 64),
        nsteps=fitting.get("nsteps", 10000),
        n_jobs=n_jobs,
        uncertainties=uncertainties,
        fixed_uncertainty=fixed_uncertainty,
    )
    summaries = {m: s for m, s in summaries.items() if s is not None}
    evaluators = {m: ModelRegistry.get_evaluator(m) for m in summaries}

    # compare models
    best_model, best_summary, table, report = compare_models(
        model_summaries=summaries,
        coords=coords,
        data=data,
        logger=logger,
        data_range=data_range,
        model_evaluators=evaluators,
        counts=counts,
    )
    if verbose:
        logger.info(report)
    log_memory_usage(logger, "Compare models")

    # Save results
    if save_data:
        if metadata and 'redshift' in metadata:
            results_dir = PATHS.snapshot_results(halo_id, metadata['redshift'], experiment=experiment)
        else:
            logger.warning("No redshift in metadata, using halo-level results")
            results_dir = PATHS.halo_results(halo_id, experiment=experiment)

        save_results(
            results_dir / f"{component_type}_{component_name}_results.h5",
            summaries,
            best_model,
            logger,
            table,
        )

    logger.info(f"{component_name.capitalize()} fitting completed")

    # Return results as FittingResult dataclass
    return FittingResult(
        best_model=best_model,
        samples=best_summary.get("samples") if best_summary else np.array([]),
        summary=best_summary if best_summary else {},
        model_comparison={"table": table},
    )


def _uncertainty_extractor(sem, N, geometry, logger=None):
    """
    Extract uncertainties from 3D profile statistics.

    Args:
        sem: 3D standard error of mean array (shape: 64x64x64, axes: r, R, z)
        N: 3D bin count array (shape: 64x64x64)
        geometry: 'spherical' or 'cylindrical'
        logger: Optional logger for info messages

    Returns:
        uncertainties array matching geometry:
            - spherical: 1D array (64,) for r
            - cylindrical: 2D array (64, 64) for (R, z)
    """
    if geometry == "spherical":
        # sum over R, z axes to get radial profile uncertainties
        # weight by sqrt(N) when combining (error propagation)
        N_1d = np.sum(N, axis=(1, 2))
        sem_sum = np.sum(sem * np.sqrt(N), axis=(1, 2))

        # combined sem for radial bins
        uncertainties_1d = sem_sum / np.sqrt(N_1d)
        # filter out bins with too few particles
        mask_reliable = N_1d >= 10
        uncertainties_1d = np.where(mask_reliable, uncertainties_1d, np.inf)

        if logger:
            logger.info(
                f"Spherical uncertainties: median = {np.median(uncertainties_1d[mask_reliable]):.3e}"
            )

        return uncertainties_1d

    elif geometry == "cylindrical":
        # sum over r axis to get cylindrical profile uncertainties
        # weight by sqrt(N) when combining (error propagation)
        N_2d = np.sum(N, axis=0)  # Shape: (64, 64) for (R, z)
        sem_sum = np.sum(sem * np.sqrt(N), axis=0)  # Sum over r axis

        # Combined SEM for (R, z) bins
        uncertainties_2d = sem_sum / np.sqrt(N_2d)

        # Filter out bins with too few particles/cells
        mask_reliable = N_2d >= 10
        uncertainties_2d = np.where(mask_reliable, uncertainties_2d, np.inf)

        if logger:
            n_reliable = np.sum(mask_reliable)
            logger.info(
                f"Cylindrical uncertainties: {n_reliable}/{mask_reliable.size} bins with N >= 10"
            )
            logger.info(
                f"Median uncertainty: {np.median(uncertainties_2d[mask_reliable]):.3e}"
            )

        return uncertainties_2d
    else:
        raise ValueError(f"Unknown geometry: {geometry}")


def _profile_uncertainties(profile, label, logger):
    """Return (sigma_1d, sigma_2d_flat); either may be None."""
    if profile.sem is not None and profile.N is not None:
        logger.info(f"Using measured SEM uncertainties for {label} fitting.")
        return (_uncertainty_extractor(profile.sem, profile.N, "spherical", logger),
                _uncertainty_extractor(profile.sem, profile.N, "cylindrical", logger).ravel())
    if profile.counts is not None:
        ok = profile.counts >= 1
        logger.info(f"Using Poisson uncertainties for {label} fitting.")
        mean_rel_err = float(np.mean(100.0 / np.sqrt(profile.counts[ok])))
        logger.info(f"Mean Poisson uncertainty: {mean_rel_err:.2f}%")
        sigma_1d = np.full_like(profile.rho_1d, np.inf)
        sigma_1d[ok] = profile.rho_1d[ok] / np.sqrt(profile.counts[ok])
        return sigma_1d, None
    logger.warning(f"No statistics available for {label}, using default 10% uncertainty")
    return None, None


SUBCOMPONENTS = {
    "star": [("bulge", "spherical"), ("disk", "cylindrical"), ("halo", "spherical")],
    "gas":  [("disk", "cylindrical"), ("halo", "spherical")],
    "dm":   [("halo", "spherical")],
}


def fit_components(
    component: str,
    profile: DensityProfile,
    config: Dict,
    logger: logging.Logger,
    **common
) -> Dict[str, FittingResult]:
    """Fit every subcomponent of a component (e.g. bulge/disk/halo for stars).

    Subcomponents and their geometries come from SUBCOMPONENTS. Spherical fits
    use the 1D radial profile; cylindrical fits use the flattened (R, z) grid.
    Uncertainties come from the profile's SEM statistics if present, otherwise
    Poisson errors from particle counts, otherwise the fitter's default.

    Args:
        component: 'star', 'gas' or 'dm'
        profile: DensityProfile for this component
        config: Full configuration dict; reads config[component][subcomponent]
        logger: Logger instance
        **common: Passed through to fit_component_models (halo_id, n_jobs,
                  save_data, verbose, metadata, experiment)

    Returns:
        Dict[str, FittingResult]: One FittingResult per subcomponent, keyed by
              name (e.g. 'bulge', 'disk', 'halo').
    """
    sig_1d, sig_2d = _profile_uncertainties(profile, component, logger)
    R, z = (g.ravel() for g in np.meshgrid(profile.R, profile.z, indexing="ij"))
    inputs = {
        "spherical": dict(coords=(profile.r_1d,), data=profile.rho_1d,
                          uncertainties=sig_1d, counts=profile.counts),
        "cylindrical": dict(coords=(R, z), data=profile.rho_2d.ravel(),
                            uncertainties=sig_2d),
    }
    results = {}
    for name, geom in SUBCOMPONENTS[component]:
        results[name] = fit_component_models(
            component_config=config[component][name], component_name=name,
            component_type=component, geometry=geom, logger=logger,
            **inputs[geom], **common)
        log_memory_usage(logger, f"fit {component} {name}")
    return results


def _snapshot_info(path, stype, num):
    ds = yt.load(path)
    return {"dd_num": int(num), "path": path, "type": stype,
            "redshift": float(ds.current_redshift),
            "time_Gyr": float(ds.current_time.to("Gyr").value)}


def find_snapshots(
    halo_path: str | Path,
    snapshot_type: str | list = "DD",
    logger: Optional[logging.Logger] = None,
) -> List[Dict[str, Union[int, str, float]]]:
    """
    Find all available snapshots for a halo.

    Args:
        halo_path: Path to halo directory (e.g., '/path/to/halo_008508/nref11c_nref9f')
        snapshot_type: 'DD' or 'RD'
        logger: Optional logger instance

    Returns:
        list: Sorted list of snapshot directories with metadata
    """
    # Allow snapshot_type to be a list (e.g., ['DD','RD']) or 'BOTH'
    if isinstance(snapshot_type, str) and snapshot_type.upper() == "BOTH":
        types = ["DD", "RD"]
    elif isinstance(snapshot_type, (list, tuple)):
        types = [str(t).upper() for t in snapshot_type]
    else:
        types = [str(snapshot_type).upper()]

    snapshots: List[Dict[str, Union[int, str, float]]] = []
    failed_loads: List[tuple] = []
    n_dirs = 0

    for stype in types:
        pattern = os.path.join(halo_path, f"{stype}*")
        snapshot_dirs = glob(pattern)
        n_dirs += len(snapshot_dirs)

        if not snapshot_dirs:
            msg = f"No {stype} snapshots found in {halo_path}"
            if logger:
                logger.warning(msg)
            else:
                print(f"Warning: {msg}")
            continue

        for snap_dir in snapshot_dirs:
            # Extract snapshot number
            match = re.search(rf"{stype}(\d+)", os.path.basename(snap_dir))
            if match:
                dd_num = int(match.group(1))

                # Check if nested DD#### exists
                nested_path = os.path.join(snap_dir, os.path.basename(snap_dir))
                if os.path.exists(nested_path):
                    # load to get redshift
                    try:
                        snapshots.append(_snapshot_info(nested_path, stype, dd_num))
                    except Exception as e:
                        failed_loads.append((nested_path, str(e)))

    if failed_loads:
        msg = f"Failed to load {len(failed_loads)}/{n_dirs} snapshots"
        if logger:
            logger.warning(msg)
            for path, error in failed_loads:
                logger.debug(f"{path}: {error}")
        else:
            print(f"Warning: {msg}")

    # sort by dd number chronological
    snapshots.sort(key=lambda x: x["dd_num"])
    return snapshots


def _data_experiment(experiment, sim_type):
    """Experiment name used for profile directories; non-'full' sims default to their sim_type."""
    return experiment or (sim_type if sim_type != "full" else None)


def process_snapshot(
    halo_id: str,
    component: str,
    config: Dict[str, Any],
    save_data: bool = True,
    verbose: bool = True,
    logger: Optional[logging.Logger] = None,
    profiles_dict: Optional[Dict[str, Union[DensityProfile, Any]]] = None,
    metadata_dict: Optional[Dict[str, Any]] = None,
    n_jobs: int = 1,
    redshift: Optional[float] = None,
    dd_num: Optional[int] = None,
    experiment: Optional[str] = None,
    sim_type: str = "full",
):
    """
    Process snapshot for given component
    refactored core of original main()

    Args:
        halo_id: Halo identifier (e.g., 'halo_008508')
        component: 'star', 'gas', 'dm'
        config: configuration dictionary
        save_data: Save results or not
        show_plots: If True, keep plots open; if False, close them after saving
        verbose: logging verbosity
        logger: logger instance (created if none)
        profiles_dict: Optional dict of DensityProfile objects or yt datasets
                      (from generate_snapshot_profiles or loaded externally)
        metadata_dict: Optional metadata dict (from generate_snapshot_profiles)
        n_jobs: Number of parallel jobs (-1 = all cores, 1 = serial)

    Returns:
        dict: Results containing fitted parameters, masses, and metadata
    """
    if logger is None:
        logger = setup_logging(component)

    assert logger is not None  # Type narrowing: logger is guaranteed non-None after setup

    if component not in SUBCOMPONENTS:
        logger.error(f"Unknown component: {component}")
        return None

    # Use provided profiles or load from disk
    if profiles_dict is not None and metadata_dict is not None:
        logger.debug(f"Using in-memory profiles for {component}")
        obj = profiles_dict.get(component)
        if obj is None:
            raise ValueError(f"Component '{component}' not found in profiles_dict")
        
        profile = obj if isinstance(obj, DensityProfile) else DensityProfile.from_profile_dataset(obj, component)
        metadata = metadata_dict
    else:
        logger.debug(f"Loading profiles from disk for {component}")
        # If redshift not provided, try to find it from old-format metadata
        if redshift is None:
            # Try old format first
            old_path = DATA_DIR / halo_id / "metadata.h5"
            if old_path.exists():
                temp_metadata = load_hdf5(old_path)
                redshift = temp_metadata.get('redshift')
                dd_num = temp_metadata.get('dd_num')
            if redshift is None:
                raise ValueError(f"Redshift required for {halo_id} but not found in metadata")

        # Use new path construction
        # Use sim_type as experiment if not "full"
        profile_dir = PATHS.snapshot_data(
            halo_id, redshift, dd_num,
            experiment=_data_experiment(experiment, sim_type)
        )
        # Extract density profiles
        profile = DensityProfile.from_hdf5(
            profile_dir / f"3d_profile_{component}.h5", component
        )
        metadata = load_hdf5(profile_dir / "metadata.h5")
        log_memory_usage(logger, "load profiles")

    results = fit_components(
        component, profile, config, logger,
        halo_id=halo_id, n_jobs=n_jobs, save_data=save_data, verbose=verbose,
        metadata=metadata, experiment=experiment
    )

    # Calculate Masses
    # Build fallback radii: R200-scaled if config says so, else fixed kpc
    range_units = config.get("range_scaling", {}).get("units", "kpc") if config else "kpc"
    R200_kpc = metadata.get("R200_kpc")
    if range_units == "r200" and R200_kpc is not None:
        fallback_radii = np.array([0.02, 0.05, 0.1, 0.25, 0.5]) * R200_kpc
    else:
        fallback_radii = np.array([5, 10, 20, 50, 100], dtype=float)

    sim_masses = metadata.get("enclosed_masses", {})
    sim_radii = sim_masses.get("radii")
    if sim_radii is not None and len(sim_radii) > 0:
        model_masses = calc_masses_with_uncertainties(results, sim_radii, n_samples=100)
        compare_masses(model_masses, sim_masses, component)
    else:
        if sim_masses:
            logger.warning(f"Simulation masses for {component} have no radii - using standard radii")
        model_masses = calc_masses_with_uncertainties(results, fallback_radii, n_samples=100)
    log_memory_usage(logger, "calculate masses")
    # Calculate derived statistics
    component_stats = calculate_component_statistics(
        results, model_masses, component, metadata
    )

    output = {
        "component": component,
        "halo_id": halo_id,
        "results": results,
        "model_masses": model_masses,
        "statistics": component_stats,
        "metadata": metadata,
    }
    return output


def _new_evolution_data(halo_id, components):
    return {
        "halo_id": halo_id, "snapshots": [], "redshifts": [], "times_Gyr": [],
        "components": {c: {"fitted_params": [], "best_models": [], "masses": []}
                       for c in components}
    }


def _record(evo, snap, comp_results):
    """Append one snapshot; comp_results maps component -> process_snapshot output or None."""
    evo["snapshots"].append(snap["dd_num"])
    evo["redshifts"].append(snap["redshift"])
    evo["times_Gyr"].append(snap["time_Gyr"])
    for comp, res in comp_results.items():
        evo["components"][comp]["fitted_params"].append(res["results"] if res else None)
        evo["components"][comp]["masses"].append(res["model_masses"] if res else None)


def process_time_series(
    raw_data_root: Path | str,
    halo_id: str,
    logger: logging.Logger,
    components: List[str] = ["star", "gas", "dm"],
    snapshot_type: str = "DD",
    snapshot_list: Optional[List[str]] = None,
    config: Optional[Dict] = None,
    save_data: bool = True,
    n_jobs: int = 1,
    generate_profile: bool = False,
    experiment: Optional[str] = None,
    sim_type: str = "full",
    sim_subdir: Optional[str] = None,
    make_projections: bool = True,
    center_find_method: str = "average",
    start_snapshot: Optional[int] = None,
) -> Dict[str, Any]:
    """
    Process multiple snapshots to build time-series of galaxy evolution.

    Args:
        base_path: Base FOGGIE path (e.g., '/cephfs/brs/nobackup/FOGGIE')
        halo_id: Halo identifier (e.g., 'halo_008508')
        components: List of components to process
        snapshot_type: 'DD' or 'RD'
        snapshot_list: Optional list of specific DD/RD numbers to process
        config: Configuration dict (loaded if None)
        save_data: Save results to HDF5
        n_jobs: Number of parallel jobs for model fitting (-1 = all cores, 1 = serial)
        generate_profile: If True, regenerate profiles even if they already exist (default: False)

    Returns:
        dict: Time-series dataset with all snapshots
    """
    if config is None:
        config = load_config()

    assert config is not None

    logger.info("=" * 70)
    logger.info("FOGGIE TIME-SERIES ANALYSIS")
    logger.info("=" * 70)

    # Discover or use provided snapshots
    halo_path = os.path.join(raw_data_root, halo_id, sim_subdir or "nref11c_nref9f")
    if snapshot_list is not None:
        # User provided explicit list; respect snapshot_type (DD or RD)
        snapshots = []
        # Normalize snapshot_type to a single string
        stype = (snapshot_type.upper() if isinstance(snapshot_type, str) else "DD")
        stype = "DD" if stype == "BOTH" else stype

        for num in snapshot_list:
            name = f"{stype}{int(num):04d}"
            snapshots.append(
                _snapshot_info(os.path.join(halo_path, name, name), stype, num)
            )
    else:
        # Auto-discover
        logger.info(f"Discovering {snapshot_type} snapshots in {halo_path}...")
        snapshots = find_snapshots(halo_path, snapshot_type)

    # Filter by start_snapshot if provided
    if start_snapshot is not None:
        before = len(snapshots)
        snapshots = [s for s in snapshots if int(s["dd_num"]) >= start_snapshot]
        logger.info(f"Skipping snapshots before DD{start_snapshot:04d} ({before - len(snapshots)} skipped)")

    logger.info(f"Found {len(snapshots)} snapshots")
    for snap in snapshots:
        logger.info(
            f"   {snap['type']}{snap['dd_num']:04d}: z={snap['redshift']}, t={snap['time_Gyr']:.2f} Gyr"
        )

    # Initialize time-series data structure
    evolution_data = _new_evolution_data(halo_id, components)

    exp = _data_experiment(experiment, sim_type)

    # Process each snapshot
    for i, snap in enumerate(snapshots):
        start_time = time()
        label = f"{snap['type']}{snap['dd_num']:04d}"
        logger.info("=" * 70)
        logger.info(f"PROCESSING SNAPSHOT {i + 1}/{len(snapshots)}: {label}")
        logger.info(f"Redshift z={snap['redshift']:.3f}, Time={snap['time_Gyr']:.2f} Gyr")
        logger.info("=" * 70)

        # check for existing profiles
        profile_dir = PATHS.snapshot_data(halo_id, float(snap["redshift"]), experiment=exp)
        profiles_exist = all(
            (profile_dir / f"3d_profile_{comp}.h5").exists()
            and (profile_dir / "metadata.h5").exists()
            for comp in components
        )

        # Track whether we have in-memory profiles
        profiles_dict = None
        metadata_dict = None

        if generate_profile or not profiles_exist:
            logger.info("Generating density profiles...")
            try:
                profiles_dict, metadata_dict = generate_snapshot_profiles(
                    snap_path=str(snap["path"]),
                    halo_id=halo_id,
                    logger=logger,
                    skip_shape_analysis=False,
                    center_find_method=center_find_method,
                    save_profiles=True,
                    component="all" if len(components) > 1 else components[0],
                    sim_type=sim_type,
                    make_projections=make_projections,
                    experiment=experiment,
                    redshift=float(snap["redshift"]),
                    config=config,
                )
                logger.info(f"Successfully generated profiles for {halo_id}/{label}")
                log_memory_usage(logger, "generating snapshot profiles")
            except Exception as e:
                logger.error(f"Failed to generate profiles for {label}: {e}", exc_info=True)
                _record(evolution_data, snap, dict.fromkeys(components))
                continue
        else:
            logger.info(f"Profiles already exist for {halo_id} at z={snap['redshift']}, "
                        "skipping generation (set generate_profile=True to regenerate)")

        # Process each component
        comp_results = {}
        for comp in components:
            logger.info(f"Processing {comp} component...")

            try:
                comp_results[comp] = process_snapshot(
                    halo_id=halo_id,
                    component=comp,
                    config=config,
                    save_data=save_data,
                    verbose=True,
                    logger=logger,
                    profiles_dict=profiles_dict,
                    metadata_dict=metadata_dict,
                    n_jobs=n_jobs,
                    redshift=float(snap['redshift']),
                    dd_num=int(snap['dd_num']),
                    experiment=experiment,
                    sim_type=sim_type
                )
                log_memory_usage(logger, "Process snapshot")
            except Exception as e:
                logger.error(f"Error processing {comp} at {label}: {e}", exc_info=True)
                comp_results[comp] = None
        _record(evolution_data, snap, comp_results)


        # ---- Incremental save to keep memory bounded ----
        if save_data:
            save_evolution_data(
                evolution_data,
                halo_id,
                logger,
                append=True,  # Append to existing file
                experiment=experiment,
            )
            evolution_data = _new_evolution_data(halo_id, components)

        # ---- Snapshot memory cleanup ----
        del profiles_dict, metadata_dict
        gc.collect()
        log_memory_usage(logger, "after snapshot cleanup")
        logger.info(f"Completed snapshot in {(time() - start_time)/60:.2f} minutes")

    # Note: Data already saved incrementally above, no final save needed
    # Return empty structure (data is in HDF5 file)
    return evolution_data


def generate_snapshot_profiles(
    snap_path: str | Path,
    halo_id: str,
    logger: logging.Logger,
    output_dir: Optional[Path] = None,
    skip_shape_analysis: bool = True,
    center_find_method: str = "average",
    save_profiles: bool = True,
    component: str = "all",
    sim_type: str = "full",
    make_projections: bool = True,
    experiment: Optional[str] = None,
    redshift: Optional[float] = None,
    config: Optional[Dict[str, Any]] = None,
):
    avail_comps = available_components(sim_type, component)
    ds, snapshot_info = load_dataset(snap_path, logger, redshift, halo_id)
    redshift = float(snapshot_info["redshift"])
    log_memory_usage(logger, "Load Dataset")

    register_particle_filters(ds, sim_type, logger)
    log_memory_usage(logger, "Register particle filters")

    center = find_center(ds, sim_type, center_find_method, logger)
    logger.info(f"Galaxy Center (Mpc): {center.to('Mpc')}")
    log_memory_usage(logger, "Find Center")

    r200 = compute_r200(ds, center, halo_id, redshift, sim_type, PATHS, logger)
    sphere = ds.sphere(center.to("kpc"), ds.quan(r200, "kpc"))

    if make_projections:
        make_r200_projections(
            ds, center, r200, redshift, avail_comps, snap_path, 
            halo_id, PATHS, logger, experiment, config
        )

    # shape analysis uses star particles
    shape_tensors = None
    if not skip_shape_analysis and "star" in avail_comps:
        shape_tensors = shape_analysis(sphere, center, r200, logger)
    
    # calculate angular momentum within a smaller sphere
    inner_sphere = ds.sphere(center.to("kpc"), ds.quan(5, "kpccm"))
    bulk_vel = inner_sphere.quantities.bulk_velocity(use_gas=True, use_particles=True)
    inner_sphere.set_field_parameter("bulk_velocity", bulk_vel)
    logger.info(f"Inner sphere bulk velocity = {bulk_vel.to('km/s')}")
    if "star" in avail_comps:
        n_star = len(inner_sphere["star", "particle_mass"])
        logger.info(f"Star particles within inner sphere = {n_star}")
        if n_star < 100:
            logger.warning(f"Only {n_star} star particles in inner sphere - disk frame may be unreliable")

    l_hats = angular_momentum(inner_sphere, avail_comps, logger)
    R_mats = disk_frame(ds, center, l_hats, avail_comps, logger)
    
    profiles = build_profiles(ds, sphere, r200, avail_comps, logger)
    log_memory_usage(logger, "Create Profiles")
        
    radii = mass_radii(r200, config, logger)
    masses = enclosed_masses(sphere, radii, avail_comps, logger)
    log_memory_usage(logger, "Calculate enclosed masses")

    metadata = build_metadata(
        halo_id, snapshot_info, center, r200, R_mats, l_hats,
        snap_path, masses, shape_tensors
    )
    
    if save_profiles:
        data_path = snapshot_data_dir(PATHS, halo_id, redshift, sim_type, experiment, output_dir)
        write_profiles(profiles, data_path, logger)
        filepath = data_path/ "metadata.h5"
        save_metadata(filepath, metadata)
        logger.info(f"Saved metadata to {filepath}")
        
    logger.info("Converting profiles to DensityProfile objects...")
    density_profiles = {
        comp: DensityProfile.from_profile(prof, comp)
        for comp, prof in profiles.items()
    }
    logger.info("Profile generation completed successfully")
    return density_profiles, metadata
