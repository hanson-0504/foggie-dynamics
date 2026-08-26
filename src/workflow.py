"""
Workflow orchestration functions for FOGGIE dynamics analysis.
Handles fitting workflows for stellar, gas, and dark matter components.

ARCHITECTURE:
=============

This module uses a registry-based pattern to eliminate code duplication:

1. **ModelRegistry** (models.py): Central registry of all density models
   - Maps model names to fitting functions and evaluators
   - Constructs appropriate arguments based on geometry
   - Eliminates all if/elif chains for model selection

2. **fit_component_models()**: Universal fitting function
   - Handles both spherical (bulge, halo) and cylindrical (disk) geometries
   - Automatically selects correct fitter via ModelRegistry
   - Manages weights, model comparison, plotting, and saving
   - Single source of truth for all fitting workflows

3. **Component-specific wrappers**: fit_stellar/gas/dm_components()
   - Prepare coordinate arrays (flattening, meshgrid)
   - Delegate to fit_component_models() for each subcomponent
   - Minimal boilerplate (~50-75 lines each vs ~200+ lines before)

REFACTORING ACHIEVEMENTS:
========================
✅ Consolidated three 200+ line functions into one 150-line generic function
✅ Eliminated ~500 lines of duplicated code (63% reduction)
✅ Created ModelRegistry pattern (eliminates all if/elif chains)
✅ Standardized fitting interface across all components
✅ Created DensityProfile dataclass for structured density data (Dec 2024)
✅ Consolidated I/O operations into src/io.py module (Dec 2024):
   - ensure_directory() - unified directory creation (~8 duplicates eliminated)
   - save_hdf5()/load_hdf5() - generic HDF5 operations (~55 lines saved)
   - save_figure() - matplotlib figure saving (~7 duplicates eliminated)
✅ Updated generate_snapshot_profiles() to return DensityProfile objects
✅ Created plot_galaxy_density_profile() for unified component visualization

FUTURE IMPROVEMENTS:
===================
- Add type hints throughout
- Extend ProjectPaths class with additional path construction methods
- Consider SnapshotMetadata dataclass for metadata consistency

TODO: HIGH PRIORITY IMPROVEMENTS
==================================
1. ADD TESTING INFRASTRUCTURE - Create tests/ directory with pytest
   - Unit tests for I/O operations (save_hdf5, load_hdf5, ensure_directory)
   - Integration tests for density profile loading/saving
   - Test ModelRegistry pattern with mock models

2. ✅ COMPLETED: Create DensityProfile dataclass (see lines 99-110)
   - Successfully implemented with r_1d, rho_1d, R, z, rho_2d, counts fields
   - Used throughout workflow.py and plotting.py

3. PARTIAL: Separate I/O from computation
   ✅ Created src/io.py with generic I/O utilities
   ✅ generate_snapshot_profiles() returns DensityProfile objects
   ⚠️  Still TODO: Make process_snapshot() fully testable (reduce side effects)

4. Extract magic numbers to src/constants.py
   - Profile bin counts (64x64x64)
   - Default radii for mass calculations ([5, 10, 15, 20, 30, 50, 75, 100, 150, 200])
   - MCMC defaults (nwalkers=64, nsteps=10000)

5. Add custom exception hierarchy (ModelFitError, DataError, IOError, etc.)
"""

import datetime
import gc
import logging
import multiprocessing
import os
import re
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
    get_shape_tensor,
    save_evolution_data,
    save_results,
)
from .config import ProjectPaths, load_config, setup_logging
from .density_profiles import find_virial_radius, iterative_center
from .filters_and_utils import (
    add_gaussian_weight,
    add_inverse_radius_weight,
    add_rotated_fields,
    convert_r200_range_to_kpc,
    create_profiles,
    filter_highres_particles,
    rotation_matrix_from_vectors,
)
from .io import ensure_directory, load_hdf5
from .io import save_hdf5 as save_metadata
from .models import ModelRegistry

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
        include_stats: bool = True
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

    Returns:
        Tuple of (model_name, summary_dict) where summary_dict is None if fitting failed
    """
    # Import here to avoid circular imports
    import logging

    from .models import ModelRegistry

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

    Returns:
        Dict mapping model names to fit summaries
    """
    if n_jobs == -1:
        n_jobs = multiprocessing.cpu_count()

    n_jobs = min(n_jobs, len(models))

    # Serial or parallel execution based on n_jobs
    if n_jobs <= 1:
        # Serial execution (for debugging)
        logger.info(f"Fitting {len(models)} models in serial mode")
        summaries = {}
        for model in models:
            model_name, summary = _fit_single_model_worker(
                model_name=model,
                geometry=geometry,
                coords=coords,
                data=data,
                data_range=data_range,
                bounds=bounds,
                p0=p0,
                weights=weights,
                nwalkers=nwalkers,
                nsteps=nsteps,
                uncertainties=uncertainties,
                fixed_uncertainty=fixed_uncertainty,
            )
            summaries[model_name] = summary
        return summaries

    else:
        # Parallel execution
        logger.info(f"Fitting {len(models)} models in parallel using {n_jobs} workers")
        summaries = {}

        with ProcessPoolExecutor(max_workers=n_jobs) as executor:
            # Submit all jobs with explicit arguments
            futures = {
                executor.submit(
                    _fit_single_model_worker,
                    model_name=model,
                    geometry=geometry,
                    coords=coords,
                    data=data,
                    data_range=data_range,
                    bounds=bounds,
                    p0=p0,
                    weights=weights,
                    nwalkers=nwalkers,
                    nsteps=nsteps,
                    uncertainties=uncertainties,
                    fixed_uncertainty=fixed_uncertainty,
                ): model
                for model in models
            }

            # Collect results as they complete
            for future in as_completed(futures):
                model = futures[future]
                try:
                    model_name, summary = future.result()
                    summaries[model_name] = summary
                    logger.info(f"Completed {model_name}")
                except Exception as e:
                    logger.error(
                        f"Failed to retrieve result for {model}: {e}", exc_info=True
                    )
                    summaries[model] = None

        return summaries


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
        return FittingResult(
            best_model=None,
            samples=None,
            summary=None,
            model_comparison={"table": None, "figure": None},
        )
    logger.info(f"FITTING {component_name.upper()} COMPONENT")

    # Log data summary for diagnostics
    valid_mask = np.isfinite(data) & (data > 0)
    n_valid = np.sum(valid_mask)
    if geometry == "spherical":
        r = coords[0]
        logger.info(
            f"  Data summary: {n_valid}/{len(data)} valid points, "
            f"r=[{np.min(r):.3f}, {np.max(r):.3f}] kpc, "
            f"rho=[{np.min(data[valid_mask]):.2e}, {np.max(data[valid_mask]):.2e}] Msun/kpc³"
            if n_valid > 0 else f"  Data summary: {n_valid}/{len(data)} valid points - NO VALID DATA"
        )
    else:  # cylindrical
        R, z = coords[0], coords[1]
        logger.info(
            f"  Data summary: {n_valid}/{len(data)} valid points, "
            f"R=[{np.min(R):.3f}, {np.max(R):.3f}] kpc, "
            f"z=[{np.min(z):.3f}, {np.max(z):.3f}] kpc, "
            f"rho=[{np.min(data[valid_mask]):.2e}, {np.max(data[valid_mask]):.2e}] Msun/kpc³"
            if n_valid > 0 else f"  Data summary: {n_valid}/{len(data)} valid points - NO VALID DATA"
        )

    # Extract fitting parameters
    bounds = component_config.get("bounds")
    p0 = component_config.get("p0")
    fitting_params = component_config.get("fitting", {})
    nwalkers = fitting_params.get("nwalkers", 64)
    nsteps = fitting_params.get("nsteps", 10000)

    r_range = None
    R_range = None

    # Get metadata values for scaling
    redshift = metadata.get("redshift", 0.0) if metadata else 0.0
    R200_kpc = metadata.get("R200_kpc") if metadata else None

    # Get scaling configuration (defaults to no scaling for backward compatibility)
    scaling_config = component_config.get("range_scaling", {})
    range_units = scaling_config.get("units", "kpc")  # "kpc" or "r200"
    scaling_mode = scaling_config.get("mode", "none")  # "linear", "sqrt", "log", "none"
    min_scale = scaling_config.get("min_scale", 0.1)
    max_scale = scaling_config.get("max_scale", 1.0)

    # Validate R200 availability if needed
    if range_units == "r200" and R200_kpc is None:
        logger.warning(
            f"R200 scaling requested but R200_kpc not in metadata. "
            f"Falling back to kpc mode for {component_name}."
        )
        range_units = "kpc"

    # get data range based on geometry
    if geometry == "spherical":
        r_range_config = component_config.get("r_range")
        if r_range_config is None:
            logger.error(f"Missing required 'r_range' in {component_name} config")
            return FittingResult(
                best_model=None,
                samples=None,
                summary=None,
                model_comparison={"table": None},
            )

        if range_units == "r200":
            # r_range is in R200 fractions, convert to kpc
            r_range_r200 = tuple(r_range_config)
            r_range = convert_r200_range_to_kpc(r_range_r200, R200_kpc)
            logger.info(
                f"R200-scaled r_range: {r_range_r200} R200 → "
                f"({r_range[0]:.3f}, {r_range[1]:.3f}) kpc "
                f"(R200={R200_kpc:.1f} kpc)"
            )
        else:
            # r_range is in kpc, optionally apply redshift scaling
            r_range_z0 = tuple(r_range_config)
            if scaling_mode != "none":
                from .filters_and_utils import scale_range_with_redshift
                r_range = scale_range_with_redshift(
                    r_range_z0,
                    redshift,
                    min_scale=min_scale,
                    max_scale=max_scale,
                    scaling_mode=scaling_mode
                )
                logger.info(
                    f"Scaled r_range for z={redshift:.3f}: "
                    f"{r_range_z0} → {r_range} kpc (mode={scaling_mode}, scale={1 / (1 + redshift):.3f})"
                )
            else:
                r_range = r_range_z0

        data_range = {0: r_range}
    elif geometry == "cylindrical":
        R_range_config = component_config.get("R_range")
        z_max_config = component_config.get("z_max")
        if R_range_config is None or z_max_config is None:
            logger.error(
                f"Missing required 'R_range' or 'z_max' in {component_name} config"
            )
            return FittingResult(
                best_model=None,
                samples=None,
                summary=None,
                model_comparison={"table": None},
            )

        if range_units == "r200":
            # R_range and z_max are in R200 fractions
            R_range_r200 = tuple(R_range_config)
            z_max_r200 = z_max_config
            R_range = convert_r200_range_to_kpc(R_range_r200, R200_kpc)
            z_max = max(0.3, z_max_r200 * R200_kpc)  # Convert z_max, ensure minimum
            logger.info(
                f"R200-scaled R_range: {R_range_r200} R200 → "
                f"({R_range[0]:.3f}, {R_range[1]:.3f}) kpc"
            )
            logger.info(f"R200-scaled z_max: {z_max_r200} R200 → {z_max:.2f} kpc")
        else:
            # Ranges in kpc, optionally apply redshift scaling
            R_range_z0 = tuple(R_range_config)
            z_max_z0 = z_max_config
            if scaling_mode != "none":
                from .filters_and_utils import scale_range_with_redshift
                R_range = scale_range_with_redshift(
                    R_range_z0,
                    redshift,
                    min_scale=min_scale,
                    max_scale=max_scale,
                    scaling_mode=scaling_mode
                )
                z_max = z_max_z0 * (1.0 / (1.0 + redshift))  # Scale z_max with scale factor
                z_max = max(0.3, z_max)  # Ensure minimum z resolution
                logger.info(
                    f"Scaled R_range for z={redshift:.3f}: "
                    f"{R_range_z0} → {R_range} kpc (mode={scaling_mode})"
                )
                logger.info(f"Scaled z_max: {z_max_z0} → {z_max:.2f} kpc")
            else:
                R_range = R_range_z0
                z_max = z_max_z0

        data_range = {0: R_range, 1: (0.3, z_max)}
    else:
        logger.error(f"Unknown geometry {geometry}")
        return FittingResult(
            best_model=None,
            samples=None,
            summary=None,
            model_comparison={"table": None},
        )

    # calculate weights if configured
    weights = None
    if "weights" in component_config:
        w_cfg = component_config["weights"]
        if w_cfg.get("type") == "gaussian":
            # for spherical: use coords[0] (r)
            # for cylindrical: use coords[0] (R_flat)
            target_range = r_range if geometry == "spherical" else R_range
            weights = add_gaussian_weight(
                coords[0],
                target_range,
                mean=w_cfg.get("mean", None),
                sigma_frac=w_cfg.get("sigma_frac", 0.1),
            )
        elif w_cfg.get("type") == "inverse_radius":
            # Inverse radius weighting: w ∝ 1/r^power
            target_range = r_range if geometry == "spherical" else R_range
            weights = add_inverse_radius_weight(
                coords[0],
                target_range,
                power=w_cfg.get("power", 1.0),
                r_scale=w_cfg.get("r_scale", None),
            )
    # Fit all models using ModelRegistry
    summaries = {}
    models = component_config["models"]

    assert bounds is not None
    assert p0 is not None

    summaries = fit_models_parallel(
        models=models,
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
        n_jobs=n_jobs,
        uncertainties=uncertainties,
        fixed_uncertainty=fixed_uncertainty,
    )

    # Build evaluators using ModelRegistry
    evaluators = {}
    for model, summary in summaries.items():
        if summary is not None:
            evaluators[model] = ModelRegistry.get_evaluator(model)

    # Clean up None values
    summaries_clean = {k: v for k, v in summaries.items() if v is not None}
    evaluators_clean = {k: v for k, v in evaluators.items() if v is not None}

    # compare models
    best_model, best_summary, table, report = compare_models(
        model_summaries=summaries_clean,
        coords=coords,
        data=data,
        logger=logger,
        data_range=data_range,
        model_evaluators=evaluators_clean,
        counts=counts,
    )
    if verbose:
        logger.info(report)
    log_memory_usage(logger, "Compare models")

    # Save results
    if save_data:
        if metadata and 'redshift' in metadata:
            redshift = metadata['redshift']
            results_dir = PATHS.snapshot_results(halo_id, redshift, experiment=experiment)
            results_path = results_dir / f"{component_type}_{component_name}_results.h5"
        else:
            logger.warning("No redshift in metadata, using halo-level results")
            results_dir = PATHS.halo_results(halo_id, experiment=experiment)
            results_path = results_dir / f"{component_type}_{component_name}_results.h5"

        save_results(
            results_path,
            summaries_clean,
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


def subtract_disk_from_spherical(
    r_1d: np.ndarray, rho_1d, disk_evaluator, disk_summary, z_max=5.0
):
    """
    Subtract integrated disk contribution from spherical profile.

    For each radial bin, integrate disk density along z-direction.
    """
    rho_disk_integrated = np.zeros_like(r_1d)

    for i, r in enumerate(r_1d):
        # Define z integration points
        z_points = np.linspace(0, z_max, 50)

        # Evaluate disk at (R=r, z)
        R_coords = np.full_like(z_points, r)
        rho_disk_z = disk_evaluator((R_coords, z_points), disk_summary)

        # Integrate (simple trapezoidal rule)
        # Factor of 2 for symmetry above/below midplane
        rho_disk_integrated[i] = 2 * np.trapezoid(rho_disk_z, z_points) / (2 * z_max)

    rho_residual = rho_1d - rho_disk_integrated
    return np.maximum(rho_residual, 1e-30)


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


def fit_stellar_components(
    profile: DensityProfile,
    config: Dict,
    logger: logging.Logger,
    verbose: bool,
    save_data: bool,
    halo_id: str,
    n_jobs: int,
    metadata: Optional[Dict] = None,
    experiment: Optional[str] = None,
) -> Dict[str, Optional[FittingResult]]:
    """Fit bulge/disk/halo for stellar component.

    This function prepares coordinate arrays and delegates to fit_component_models
    for each subcomponent (bulge, disk, halo).

    Args:
        profile: DensityProfile
        config: Full configuration dict
        logger: Logger instance
        verbose: If True, log detailed fitting reports
        save_data: If True, save results to HDF5
        halo_id: Halo identifier for file paths
        show_plots: If True, keep plots open; if False, close them after saving

    Returns:
        Dict[str, Optional[FittingResult]]: Results dictionary with keys 'bulge', 'disk', 'halo'.
              Each contains a FittingResult object with best_model, summary, samples, and model_comparison.
    """
    results: Dict[str, Optional[FittingResult]] = {}
    star_config = config["star"]

    r_1d = profile.r_1d
    rho_1d = profile.rho_1d
    R = profile.R
    z = profile.z
    rho_2d = profile.rho_2d
    counts = profile.counts

    # Extract uncertainties from profile statistics
    uncertainties_1d = None
    uncertainties_2d = None
    if profile.sem is not None and profile.N is not None:
        logger.info("Using measured SEM uncertainties for stellar fitting")
        uncertainties_1d = _uncertainty_extractor(
            profile.sem, profile.N, "spherical", logger
        )
        uncertainties_2d = _uncertainty_extractor(
            profile.sem, profile.N, "cylindrical", logger
        )
    elif profile.counts is not None:
        # Use Poisson uncertainties: σ_ρ = ρ / sqrt(N_particles)
        logger.info(
            "Using Poisson uncertainties from particle counts for stellar fitting"
        )
        uncertainties_1d = np.divide(
            profile.rho_1d, np.sqrt(profile.counts), where=profile.counts >= 1
        )
        mean_rel_err = np.mean(
            np.divide(100.0, np.sqrt(profile.counts), where=profile.counts >= 1)
        )
        logger.info(f"Mean Poisson uncertainty: {mean_rel_err:.2f}%")
    else:
        logger.warning("No statistics available, using default 10% uncertainty")

    # R and z are 1D bin centers, need to create 2D grid
    R_grid, z_grid = np.meshgrid(R, z, indexing="ij")
    rho_2d_flat = rho_2d.flatten()
    R_flat = R_grid.flatten()
    z_flat = z_grid.flatten()

    # Only fit bulge to original profile if NOT doing disk-subtracted fitting
    if not star_config.get("bulge_subtracted_disk", {}).get("enabled", False):
        results["bulge"] = fit_component_models(
            component_config=star_config["bulge"],
            component_name="bulge",
            component_type="star",
            geometry="spherical",
            coords=(r_1d,),
            data=rho_1d,
            logger=logger,
            halo_id=halo_id,
            save_data=save_data,
            verbose=verbose,
            counts=counts,
            n_jobs=n_jobs,
            uncertainties=uncertainties_1d,
            metadata=metadata,
            experiment=experiment,
        )
        log_memory_usage(logger, "fit stellar bulge")
    else:
        results["bulge"] = None  # Will be fitted after disk subtraction

    # disk fitting
    results["disk"] = fit_component_models(
        component_config=star_config["disk"],
        component_name="disk",
        component_type="star",
        geometry="cylindrical",
        coords=(R_flat, z_flat),
        data=rho_2d_flat,
        logger=logger,
        halo_id=halo_id,
        save_data=save_data,
        verbose=verbose,
        n_jobs=n_jobs,
        uncertainties=(
            uncertainties_2d.flatten() if uncertainties_2d is not None else None
        ),
        metadata=metadata,
        experiment=experiment,
    )
    log_memory_usage(logger, "fit stellar disk")

    # If disk-subtracted mode is enabled, fit bulge to residual after subtracting disk
    if results["disk"] and star_config.get("bulge_subtracted_disk", {}).get(
        "enabled", False
    ):
        logger.info("Computing disk-subtracted spherical profile for bulge fitting")
        # get best disk model and evaluator
        best_disk_model = results["disk"].best_model
        best_disk_summary = results["disk"].summary
        disk_evaluator = ModelRegistry.get_evaluator(best_disk_model)

        rho_bulge = subtract_disk_from_spherical(
            r_1d, rho_1d, disk_evaluator, best_disk_summary, z_max=5
        )

        logger.info(
            f"Residual at r=1 kpc: {rho_bulge[np.argmin(np.abs(r_1d - 1.0))]:.3e}"
        )

        # Fit bulge to disk-subtracted residual
        results["bulge"] = fit_component_models(
            component_config=star_config["bulge_subtracted_disk"],
            component_name="bulge",
            component_type="star",
            geometry="spherical",
            coords=(r_1d,),
            data=rho_bulge,
            logger=logger,
            halo_id=halo_id,
            save_data=save_data,
            verbose=verbose,
            counts=counts,
            n_jobs=n_jobs,
            uncertainties=uncertainties_1d,
            metadata=metadata,
            experiment=experiment,
        )
        log_memory_usage(logger, "fit stellar disk subtracted bulge")

    # halo fitting
    results["halo"] = fit_component_models(
        component_config=star_config["halo"],
        component_name="halo",
        component_type="star",
        geometry="spherical",
        coords=(r_1d,),
        data=rho_1d,
        logger=logger,
        halo_id=halo_id,
        save_data=save_data,
        verbose=verbose,
        counts=counts,
        n_jobs=n_jobs,
        uncertainties=uncertainties_1d,
        metadata=metadata,
        experiment=experiment,
    )
    log_memory_usage(logger, "fit stellar halo")

    return results


def fit_gas_components(
    profile: DensityProfile,
    config: Dict,
    logger: logging.Logger,
    verbose: bool,
    save_data: bool,
    halo_id: str,
    n_jobs: int,
    metadata: Optional[Dict] = None,
    experiment: Optional[str] = None,
) -> Dict[str, Optional[FittingResult]]:
    """Fit disk/halo for gas component.

    Extracts uncertainties from profile.sem and profile.N for proper chi-squared fitting.

    Args:
        profile: DensityProfile object
        config: full configuration dict
        logger: logger instance
        verbose: If True, log detailed fitting reports
        save_data: If True, save results to HDF5
        halo_id: Halo identifier for file paths
        n_jobs: Number of parallel jobs
        show_plots: If True, keep plots open; if False, close them after saving

    Returns:
        Dict[str, Optional[FittingResult]]: Results dictionary with keys 'disk', 'halo'.
              Each contains a FittingResult object.
    """
    results: Dict[str, Optional[FittingResult]] = {}
    gas_config = config["gas"]

    r_1d = profile.r_1d
    rho_1d = profile.rho_1d
    R = profile.R
    z = profile.z
    rho_2d = profile.rho_2d

    # Get 2d grid for R and z
    R_grid, z_grid = np.meshgrid(R, z, indexing="ij")
    rho_2d_flat = rho_2d.flatten()
    R_flat, z_flat = R_grid.flatten(), z_grid.flatten()

    # Extract uncertainties from profile statistics
    uncertainties_1d = None
    uncertainties_2d = None
    if profile.sem is not None and profile.N is not None:
        logger.info("Using measured SEM uncertainties for gas fitting")
        uncertainties_1d = _uncertainty_extractor(
            profile.sem, profile.N, "spherical", logger
        )
        uncertainties_2d = _uncertainty_extractor(
            profile.sem, profile.N, "cylindrical", logger
        )
    elif profile.counts is not None:
        # Use Poisson uncertainties: sigma_rho = rho / sqrt(N_particles)
        logger.info("Using Poisson uncertainties from particle counts for gas fitting")
        uncertainties_1d = np.divide(
            profile.rho_1d, np.sqrt(profile.counts), where=profile.counts >= 1
        )
        mean_rel_err = np.mean(
            np.divide(100.0, np.sqrt(profile.counts), where=profile.counts >= 1)
        )
        logger.info(f"Mean Poisson uncertainty: {mean_rel_err:.2f}%")
    else:
        logger.warning("No statistics available for gas, using default 10% uncertainty")

    # Disk Fitting
    results["disk"] = fit_component_models(
        component_config=gas_config["disk"],
        component_name="disk",
        component_type="gas",
        geometry="cylindrical",
        coords=(R_flat, z_flat),
        data=rho_2d_flat,
        logger=logger,
        halo_id=halo_id,
        save_data=save_data,
        verbose=verbose,
        n_jobs=n_jobs,
        uncertainties=(
            uncertainties_2d.flatten() if uncertainties_2d is not None else None
        ),
        metadata=metadata,
        experiment=experiment,
    )
    log_memory_usage(logger, "fit gas disk")
    results["halo"] = fit_component_models(
        component_config=gas_config["halo"],
        component_name="halo",
        component_type="gas",
        geometry="spherical",
        coords=(r_1d,),
        data=rho_1d,
        logger=logger,
        halo_id=halo_id,
        save_data=save_data,
        n_jobs=n_jobs,
        uncertainties=uncertainties_1d,
        metadata=metadata,
        experiment=experiment,
    )
    log_memory_usage(logger, "fit gas halo")

    return results


def fit_dm_components(
    profile: DensityProfile,
    config: Dict,
    logger: logging.Logger,
    verbose: bool,
    save_data: bool,
    halo_id: str,
    n_jobs: int,
    metadata: Optional[Dict] = None,
    experiment: Optional[str] = None,
) -> Dict[str, Optional[FittingResult]]:
    """Fit Halo for dark matter component.

    Extracts uncertainties from profile.sem and profile.N for proper chi-squared fitting.

    Args:
        profile: DensityProfile object
        config: Full configuration dict
        logger: Logger instance
        verbose: If True, log detailed fitting reports
        save_data: If True, save results to HDF5
        halo_id: Halo identifier for file paths
        n_jobs: Number of parallel jobs
        show_plots: If True, keep plots open; if False, close them after saving

    Returns:
        Dict[str, Optional[FittingResult]]: Results dictionary with key 'halo'.
              Contains a FittingResult object.
    """
    results: Dict[str, Optional[FittingResult]] = {}

    # Extract uncertainties from profile statistics
    uncertainties_1d = None
    if profile.sem is not None and profile.N is not None:
        logger.info("Using measured SEM uncertainties for DM fitting")
        uncertainties_1d = _uncertainty_extractor(
            profile.sem, profile.N, "spherical", logger
        )
    elif profile.counts is not None:
        # Use Poisson uncertainties: σ_ρ = ρ / sqrt(N_particles)
        logger.info("Using Poisson uncertainties from particle counts for DM fitting")
        uncertainties_1d = np.divide(
            profile.rho_1d, np.sqrt(profile.counts), where=profile.counts >= 1
        )
        mean_rel_err = np.mean(
            np.divide(100.0, np.sqrt(profile.counts), where=profile.counts >= 1)
        )
        logger.info(f"Mean Poisson uncertainty: {mean_rel_err:.2f}%")
    else:
        logger.warning("No statistics available for DM, using default 10% uncertainty")

    results["halo"] = fit_component_models(
        component_config=config["dm"]["halo"],
        component_name="halo",
        component_type="dm",
        geometry="spherical",
        coords=(profile.r_1d,),
        data=profile.rho_1d,
        logger=logger,
        halo_id=halo_id,
        save_data=save_data,
        verbose=verbose,
        counts=profile.counts,
        n_jobs=n_jobs,
        uncertainties=uncertainties_1d,
        metadata=metadata,
        experiment=experiment,
    )
    log_memory_usage(logger, "fit dark matter halo")

    return results


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
    import re
    from glob import glob

    # Allow snapshot_type to be a list (e.g., ['DD','RD']) or 'BOTH'
    if isinstance(snapshot_type, str) and snapshot_type.upper() == "BOTH":
        types = ["DD", "RD"]
    elif isinstance(snapshot_type, (list, tuple)):
        types = [str(t).upper() for t in snapshot_type]
    else:
        types = [str(snapshot_type).upper()]

    snapshots: List[Dict[str, Union[int, str, float]]] = []
    failed_loads: List[tuple] = []

    for stype in types:
        pattern = os.path.join(halo_path, f"{stype}*")
        snapshot_dirs = glob(pattern)

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
                        ds = yt.load(nested_path)  # type: ignore
                        snapshots.append(
                            {
                                "dd_num": dd_num,
                                "path": nested_path,
                                "redshift": float(ds.current_redshift),
                                "time_Gyr": float(ds.current_time.to("Gyr").value),
                                "type": stype,
                            }
                        )
                    except Exception as e:
                        failed_loads.append((nested_path, str(e)))

    if failed_loads:
        msg = f"Failed to load {len(failed_loads)}/{len(snapshot_dirs)} snapshots"
        if logger:
            logger.warning(msg)
            for path, error in failed_loads:
                logger.debug(f"{path}: {error}")
        else:
            print(f"Warning: {msg}")

    # sort by dd number chronological
    snapshots.sort(key=lambda x: x["dd_num"])
    return snapshots


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

    # Use provided profiles or load from disk
    if profiles_dict is not None and metadata_dict is not None:
        logger.debug(f"Using in-memory profiles for {component}")
        obj = profiles_dict.get(component)
        if obj is None:
            raise ValueError(f"Component '{component}' not found in profiles_dict")

        # Check if it's already a DensityProfile or a yt dataset
        if isinstance(obj, DensityProfile):
            logger.debug(f"Using DensityProfile object directly for {component}")
            profile = obj
        else:
            logger.debug(f"Converting yt dataset to DensityProfile for {component}")
            profile = DensityProfile.from_profile_dataset(obj, component)

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
        exp = experiment if experiment else (sim_type if sim_type != "full" else None)
        profile_dir = PATHS.snapshot_data(halo_id, redshift, dd_num, experiment=exp)
        ds = yt.load(str(profile_dir / f"3d_profile_{component}.h5"))
        metadata = load_hdf5(profile_dir / "metadata.h5")

        # Extract density profiles
        profile = DensityProfile.from_profile_dataset(ds, component)
        log_memory_usage(logger, "load profiles")

    # Fit models based on component type
    results = None
    if component == "star":
        results = fit_stellar_components(
            profile=profile,
            config=config,
            logger=logger,
            verbose=verbose,
            save_data=save_data,
            halo_id=halo_id,
            n_jobs=n_jobs,
            metadata=metadata,
            experiment=experiment,
        )
    elif component == "gas":
        results = fit_gas_components(
            profile=profile,
            config=config,
            logger=logger,
            verbose=verbose,
            save_data=save_data,
            halo_id=halo_id,
            n_jobs=n_jobs,
            metadata=metadata,
            experiment=experiment,
        )
    elif component == "dm":
        results = fit_dm_components(
            profile=profile,
            config=config,
            logger=logger,
            verbose=verbose,
            save_data=save_data,
            halo_id=halo_id,
            n_jobs=n_jobs,
            metadata=metadata,
            experiment=experiment,
        )
    else:
        logger.error(f"Unknown component: {component}")
        return None

    # Clean results
    clean_results: dict[str, FittingResult] = {
        k: v for k, v in results.items() if v is not None
    }

    # Calculate Masses
    # Build fallback radii: R200-scaled if config says so, else fixed kpc
    range_units = config.get("range_scaling", {}).get("units", "kpc") if config else "kpc"
    R200_kpc = metadata.get("R200_kpc")
    if range_units == "r200" and R200_kpc is not None:
        fallback_radii = np.array([0.02, 0.05, 0.1, 0.25, 0.5]) * R200_kpc
    else:
        fallback_radii = np.array([5, 10, 20, 50, 100], dtype=float)

    if "enclosed_masses" in metadata:
        sim_masses = metadata["enclosed_masses"]

        # Guard: Check if sim_masses has valid data
        if "radii" not in sim_masses or len(sim_masses["radii"]) == 0:
            logger.warning(
                f"Simulation masses exist but radii array is empty for {component} - "
                f"using standard radii instead"
            )
            model_masses = calc_masses_with_uncertainties(
                clean_results, fallback_radii, n_samples=100
            )
        else:
            model_masses = calc_masses_with_uncertainties(
                clean_results,
                sim_masses["radii"],
                n_samples=100
            )
            compare_masses(model_masses, sim_masses, component)
    else:
        # Calculate at standard radii if no sim masses
        model_masses = calc_masses_with_uncertainties(
            clean_results, fallback_radii, n_samples=100
        )
    log_memory_usage(logger, "calculate masses")
    # Calculate derived statistics
    component_stats = calculate_component_statistics(
        clean_results, model_masses, component, metadata
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
        stype = (
            snapshot_type.upper() if isinstance(snapshot_type, str) else "DD"
        )
        if stype == "BOTH":
            # Default to DD when ambiguous
            stype = "DD"

        for dd_num in snapshot_list:
            snap_dirname = f"{stype}{int(dd_num):04d}"
            snap_path = os.path.join(halo_path, snap_dirname, snap_dirname)
            ds = yt.load(snap_path)  # type: ignore
            snapshots.append({
                "dd_num": int(dd_num),
                "path": snap_path,
                "redshift": float(ds.current_redshift),
                "time_Gyr": float(ds.current_time.to("Gyr").value),
                "type": stype,
            })
            del ds
            gc.collect()
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
        stype = snap.get("type", "DD")
        logger.info(
            f"   {stype}{snap['dd_num']:04d}: z={snap['redshift']}, t={snap['time_Gyr']:.2f} Gyr"
        )

    # Initialize time-series data structure
    evolution_data = {
        "halo_id": halo_id,
        "snapshots": [],
        "redshifts": [],
        "times_Gyr": [],
        "components": {},
    }
    for comp in components:
        evolution_data["components"][comp] = {
            "fitted_params": [],
            "best_models": [],
            "masses": [],
        }
    # Process each snapshot
    for i, snap in enumerate(snapshots):
        start_time = time()
        logger.info("=" * 70)
        stype = snap.get("type", "DD")
        logger.info(
            f"PROCESSING SNAPSHOT {i + 1}/{len(snapshots)}: {stype}{snap['dd_num']:04d}"
        )
        logger.info(
            f"Redshift z={snap['redshift']:.3f}, Time={snap['time_Gyr']:.2f} Gyr"
        )
        logger.info("=" * 70)

        # check for existing profiles
        # Use sim_type as experiment if not "full"
        exp = experiment if experiment else (sim_type if sim_type != "full" else None)
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
            # Generate density profiles for this snapshot
            logger.info("Generating density profiles...")
            component_arg = "all" if len(components) > 1 else None
            try:
                if component_arg == "all":
                    profiles_dict, metadata_dict = generate_snapshot_profiles(
                        snap_path=str(snap["path"]),
                        halo_id=halo_id,
                        logger=logger,
                        skip_shape_analysis=False,
                        center_find_method=center_find_method,
                        save_profiles=True,
                        component="all",
                        sim_type=sim_type,
                        make_projections=make_projections,
                        experiment=experiment,
                        redshift=float(snap['redshift']),
                        config=config,
                    )
                else:
                    profiles_dict = {}
                    metadata_dict = None
                    for comp in components:
                        p, m = generate_snapshot_profiles(
                            snap_path=str(snap["path"]),
                            halo_id=halo_id,
                            logger=logger,
                            center_find_method=center_find_method,
                            skip_shape_analysis=False,
                            save_profiles=True,
                            component=comp,
                            sim_type=sim_type,
                            make_projections=make_projections,
                            experiment=experiment,
                            redshift=float(snap['redshift']),
                            config=config,
                        )
                        profiles_dict.update(p)
                        if metadata_dict is None:
                            metadata_dict = m

                stype = snap.get("type", "DD")
                logger.info(f"Successfully generated profiles for {halo_id}/{stype}{snap['dd_num']:04d}")
                log_memory_usage(logger, "generating snapshot profiles")
            except Exception as e:
                stype = snap.get("type", "DD")
                logger.error(
                    f"Failed to generate profiles for {stype}{snap['dd_num']:04d}: {e}",
                    exc_info=True,
                )
                # Skip this snapshot
                for comp in components:
                    evolution_data["components"][comp]["fitted_params"].append(None)
                    evolution_data["components"][comp]["masses"].append(None)
                evolution_data["snapshots"].append(snap["dd_num"])
                evolution_data["redshifts"].append(snap["redshift"])
                evolution_data["times_Gyr"].append(snap["time_Gyr"])
                continue
        else:
            logger.info(f"Profiles already exist for {halo_id} @" +
                        f"redshift {snap['redshift']}, " +
                        "skipping generation " +
                        "(set generate_profile=True to regenerate)")

        # Process each component
        for comp in components:
            logger.info(f"Processing {comp} component...")

            try:
                result = process_snapshot(
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
                )
                log_memory_usage(logger, "Process snapshot")
                if result is not None:
                    # Store results
                    evolution_data["components"][comp]["fitted_params"].append(
                        result["results"]
                    )
                    evolution_data["components"][comp]["masses"].append(
                        result["model_masses"]
                    )

            except Exception as e:
                stype = snap.get("type", "DD")
                logger.error(f"Error processing {comp} at {stype}{snap['dd_num']:04d}: {e}", exc_info=True)
                evolution_data["components"][comp]["fitted_params"].append(None)
                evolution_data["components"][comp]["masses"].append(None)
        evolution_data["snapshots"].append(snap["dd_num"])
        evolution_data["redshifts"].append(snap["redshift"])
        evolution_data["times_Gyr"].append(snap["time_Gyr"])

        # ---- Incremental save to keep memory bounded ----
        if save_data:
            save_evolution_data(
                evolution_data,
                halo_id,
                logger,
                append=True,  # Append to existing file
                experiment=experiment,
            )
            # Clear accumulated data after saving (keep structure intact)
            for comp in components:
                evolution_data["components"][comp]["fitted_params"].clear()
                evolution_data["components"][comp]["masses"].clear()
            evolution_data["snapshots"].clear()
            evolution_data["redshifts"].clear()
            evolution_data["times_Gyr"].clear()

        # ---- Snapshot memory cleanup ----
        if profiles_dict is not None:
            del profiles_dict
            profiles_dict = None
        if metadata_dict is not None:
            del metadata_dict
            metadata_dict = None

        gc.collect()
        log_memory_usage(logger, "after snapshot cleanup")
        logging.info(f"Completed snapshot in {(time() - start_time)/60:.2f} minutes")

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
    """
    Generate density profiles for a single snapshot.

    Replicates main.py workflow for time-series processing.

    Args:
        snap_path: Path to snapshot (e.g., '.../DD2427/DD2427')
        halo_id: Halo identifier for output (e.g., 'halo_008508')
        output_dir: Output directory (defaults to DATA_DIR/halo_id)
        logger: Logger instance
        skip_shape_analysis: Skip diagnostic shape analysis (faster)
        center_find_method: How to determine galaxy center. Options:
            - "initial_only": Use find_max (gas sims) or histogram (DMO) only
            - "shrinking_sphere": Refine initial guess with shrinking sphere, use result directly
            - "shrinking_sphere_global": Shrinking sphere from whole box (no initial guess needed)
            - "average": Refine with shrinking sphere, average both results (default)
            The shrinking sphere uses star particles (full), gas cells (gas_only),
            or DM particles (dmo) automatically based on sim_type.
        save_profiles: If True, save profiles to disk; if False, only return them
        component: Component(s) to generate profiles for. Options: 'all', 'star', 'gas',
            'dm'. When 'all', generates profiles for all available components based on
            sim_type (star+gas+dm for 'full', gas+dm for 'gas_only', dm only for 'dmo').
        sim_type: Simulation type. Options: 'full' (hydro+stars+dm), 'gas_only' (hydro+dm,
            no stars), 'dmo' (dark matter only). Determines particle filters, center-finding
            method, and which components are available.
        make_projections: If True, create projection plots with R200 overlay for visualization.
        experiment: Optional experiment name (e.g., 'r200_scaling'). Use to
            organize outputs into experiment-specific subdirectories.
        redshift: Optional redshift override (if not provided will use dataset current_redshift)

    Returns:
        tuple: (profiles_dict, metadata_dict)
            profiles_dict: Dict of yt profile objects {component: profile}
            metadata_dict: Dict with center, R200, rotation matrix, masses
    """
    has_gas = sim_type != "dmo"
    has_stars = sim_type == "full"

    if component == "all":
        avail_components = (
            ["star", "gas", "dm"] if has_stars
            else ['gas', 'dm'] if has_gas
            else ['dm']
        )
    else:
        avail_components = [component]

    # Force-skip shape analysis when no stars available (requires star particles)
    if not has_stars:
        skip_shape_analysis = True

    logger.info(f"Loading dataset: {halo_id}")
    ds = yt.load(snap_path)  # type: ignore
    log_memory_usage(logger, "Load Datset")

    # Extract snapshot metadata
    if redshift is None:
        redshift = float(ds.current_redshift)
        logger.debug(f"Using redshift from dataset: {redshift}")
    else:
        logger.debug(f"Using provided redshift: {redshift} (dataset reports {ds.current_redshift})")
    time_Gyr = float(ds.current_time.to("Gyr").value)

    # Extract DD/RD number from path
    snap_match = re.search(r"(DD|RD)(\d+)", str(snap_path))
    dd_num = int(snap_match.group(2)) if snap_match else None

    if has_stars:
        ds.add_particle_filter("star")
        ds.add_particle_filter("dm")
    elif sim_type == "gas_only":
        # gas_only sims have no star particles, so creation_time is never
        # defined. Re-register the "dm" filter using particle mass to select
        # high-res DM particles and exclude low-res boundary particles.
        ad_temp = ds.all_data()
        all_masses_temp = ad_temp["all", "particle_mass"].to("Msun").value
        _, hr_mass_threshold = filter_highres_particles(all_masses_temp)
        del ad_temp, all_masses_temp
        gc.collect()

        logger.info(f"gas_only: high-res particle mass = {hr_mass_threshold:.2e} Msun")

        hr_mass_min = hr_mass_threshold * 0.9
        hr_mass_max = hr_mass_threshold * 1.1

        def _dm_highres(pfilter, data):
            mass = data["all", "particle_mass"].to("Msun").value
            return (mass >= hr_mass_min) & (mass <= hr_mass_max)

        yt.add_particle_filter(
            "dm", function=_dm_highres, filtered_type="all",
        )
        ds.add_particle_filter("dm")
    else:  # dmo
        # Determine high-res particle mass to exclude boundary particles
        ad_temp = ds.all_data()
        all_masses_temp = ad_temp["all", "particle_mass"].to("Msun").value
        _, hr_mass_threshold = filter_highres_particles(all_masses_temp)
        del ad_temp, all_masses_temp
        gc.collect()

        logger.info(f"DMO: high-res particle mass = {hr_mass_threshold:.2e} Msun")

        hr_mass_min = hr_mass_threshold * 0.9
        hr_mass_max = hr_mass_threshold * 1.1

        def _dm_highres(pfilter, data):
            mass = data["all", "particle_mass"].to("Msun").value
            return (mass >= hr_mass_min) & (mass <= hr_mass_max)

        yt.add_particle_filter(
            "dm", function=_dm_highres, filtered_type="all",
        )
        ds.add_particle_filter("dm")

    # --- Initial center guess (skip for shrinking_sphere_global) ---
    center_find_max = None
    if center_find_method != "shrinking_sphere_global":
        if sim_type == "dmo":
            ad = ds.all_data()
            positions = np.column_stack([
                ad["all", "particle_position_x"].to('kpc').value,
                ad["all", "particle_position_y"].to('kpc').value,
                ad["all", "particle_position_z"].to('kpc').value,
            ])
            masses_all = ad["all", "particle_mass"].to("Msun").value

            # Filter to high-res particles to avoid bias from
            # massive low-resolution boundary particles
            hr_mask, hr_mass = filter_highres_particles(masses_all)
            pos_hr = positions[hr_mask]
            logger.info(
                f"DMO histogram center: using {hr_mask.sum():,} / {len(hr_mask):,} "
                f"high-res particles (mass ~ {hr_mass:.2e} Msun)"
            )

            # Unweighted histogram (particle count) on high-res particles
            H, edges = np.histogramdd(pos_hr, bins=128)
            idx = np.unravel_index(H.argmax(), H.shape)
            center_find_max = ds.arr(
                [(edges[d][idx[d]] + edges[d][idx[d] + 1]) / 2.0 for d in range(3)],
                'kpc'
            )
            del ad, positions, masses_all, pos_hr, H, edges
            gc.collect()
        else:
            _, center_find_max = ds.find_max(("gas", "dark_matter_density"))
        logger.info(f"Galaxy center (find_max method): {center_find_max}")

    # --- Shrinking sphere refinement ---
    if center_find_method != "initial_only":
        try:
            ad = ds.all_data()

            # Choose particle/cell type and parameters based on sim_type
            if sim_type == "dmo":
                all_pos = np.column_stack([
                    ad["all", "particle_position_x"].to("kpc").value,
                    ad["all", "particle_position_y"].to("kpc").value,
                    ad["all", "particle_position_z"].to("kpc").value,
                ])
                all_mass = ad["all", "particle_mass"].to("Msun").value

                # Filter to high-res particles to prevent boundary particles
                # from biasing the mass-weighted shrinking sphere
                hr_mask, hr_mass = filter_highres_particles(all_mass)
                pos = all_pos[hr_mask]
                mass = all_mass[hr_mask]
                logger.info(
                    f"Shrinking sphere: using {hr_mask.sum():,} high-res particles "
                    f"(mass ~ {hr_mass:.2e} Msun)"
                )
                del all_pos, all_mass

                R_init, min_radius_iter = 200.0, 1.0
                center_type = "high-res DM particles"
            elif sim_type == "gas_only":
                all_pos = np.column_stack([
                    ad["all", "particle_position_x"].to("kpc").value,
                    ad["all", "particle_position_y"].to("kpc").value,
                    ad["all", "particle_position_z"].to("kpc").value,
                ])
                all_mass = ad["all", "particle_mass"].to("Msun").value
                hr_mask, hr_mass = filter_highres_particles(all_mass)
                pos = all_pos[hr_mask]
                mass = all_mass[hr_mask]
                logger.info(
                    f"Shrinking sphere: using {hr_mask.sum():,} high-res DM particles "
                    f"(mass ~ {hr_mass:.2e} Msun)"
                )
                del all_pos, all_mass
                R_init, min_radius_iter = 200.0, 1.0
                center_type = "high-res DM particles"
            else:  # full
                pos = np.column_stack([
                    ad["star", "particle_position_x"].to("kpc").value,
                    ad["star", "particle_position_y"].to("kpc").value,
                    ad["star", "particle_position_z"].to("kpc").value,
                ])
                mass = ad["star", "particle_mass"].to("Msun").value
                R_init, min_radius_iter = 50.0, 0.1
                center_type = "star particles"

            # Global mode: start from whole box (no seed, no R_init)
            # Seeded modes: refine from the initial guess
            if center_find_method == "shrinking_sphere_global":
                logger.info(f"Finding center using global shrinking sphere on {center_type}...")
                center_init_arg = None
                R_init_arg = None
            else:
                logger.info(f"Refining center using shrinking sphere on {center_type}...")
                center_init_arg = np.asarray(center_find_max.to("kpc").value)
                R_init_arg = R_init

            center_iter, _ = iterative_center(
                pos,
                mass,
                center_init=center_init_arg,
                R_init=R_init_arg,
                min_radius=min_radius_iter,
            )
            center_iter = ds.arr(center_iter, "kpc")
            logger.info(f"Galaxy center (iterative method): {center_iter}")

            # Log comparison against initial guess if available
            if center_find_max is not None:
                delta = (center_find_max - center_iter).to("kpc")
                abs_diff = np.linalg.norm(delta)
                rel_diff = abs_diff / np.linalg.norm(center_find_max)
                logger.info(
                    f"Comparison: change in method values (initial - iterative) = {delta}"
                )
                logger.info(f"Absolute difference = {abs_diff:.3f}")
                logger.info(
                    f"Relative difference (initial - iterative / initial) = {rel_diff:.3%}"
                )

                # Check difference against simulation resolution
                min_resolution_code = ds.domain_width[0].to("code_length").v / (2 ** ds.parameters.get("MaximumRefinementLevel", 14))
                min_resolution_kpc = float(ds.quan(min_resolution_code, "code_length").to("kpc").v)
                logger.info(f"Simulation resolution (finest cell size) = {min_resolution_kpc:.3f} kpc")

                if abs_diff < min_resolution_kpc:
                    logger.info("Center difference is below resolution limit - methods agree within numerical precision")
                elif abs_diff < 3 * min_resolution_kpc:
                    logger.warning(f"Center difference ({abs_diff:.3f} kpc) is comparable to resolution ({min_resolution_kpc:.3f} kpc)")
                else:
                    logger.info(f"Center difference ({abs_diff:.3f} kpc) is {abs_diff / min_resolution_kpc:.1f}x resolution")

            # Apply center_find_method
            if center_find_method == "average":
                logger.info("Using mean of initial and iterative centers!")
                center = np.mean([center_find_max.to('kpc').value, center_iter.to('kpc').value], axis=0)
            else:  # "shrinking_sphere" or "shrinking_sphere_global"
                logger.info(f"Using {center_find_method} center!")
                center = center_iter.to('kpc').value

            log_memory_usage(logger, "calculating iterative center")

            del ad, pos, mass
            gc.collect()
        except Exception as e:
            logger.warning(f"Iterative center refinement failed: {e}")
            if center_find_max is not None:
                logger.warning("Falling back to initial center")
                center = center_find_max.to("kpc").value
            else:
                raise RuntimeError(f"Global shrinking sphere failed with no fallback: {e}")
    else:
        logger.info("Using initial center only (find_max/histogram)!")
        center = center_find_max.to("kpc").value

    # Ensure center is a proper yt unyt_array with units
    center = ds.arr(center, 'kpc')
    logger.info(f"Galaxy Center (Mpc): {center.to('Mpc')}")

    logger.info("Calculating Virial Radius...")

    min_rad = 0.5   # kpc
    max_rad = 300.0  # kpc

    # Calculate R200 directly from the dataset
    try:
        R200 = find_virial_radius(
            ds,
            center,
            tol=1e-3,
            max_iter=500,
            min_rad=min_rad,
            max_rad=max_rad,
            sim_type=sim_type,
        )
        logger.info(f"Calculated R200 = {R200:.2f} kpc from dataset.")
    except Exception as e:
        logger.warning(f"Direct R200 calculation failed: {e}", exc_info=True)
        R200 = None

    # Fallback to full-physics metadata for non-full sims
    if R200 is None and sim_type != 'full':
        full_meta_path = PATHS.snapshot_data(halo_id, redshift) / "metadata.h5"

        if full_meta_path.exists():
            logger.info("Falling back to R200 from full-physics run")
            full_meta = load_hdf5(full_meta_path)
            R200 = full_meta.get("R200_kpc")
            if R200 is not None:
                logger.info(f"R200 from full_physics metadata: {R200:.2f} kpc")
        else:
            from .density_profiles import interpolate_r200_from_metadata
            R200 = interpolate_r200_from_metadata(PATHS.data / halo_id, redshift)
            if R200 is not None:
                logger.info(f"Interpolated R200 from metadata: {R200:.2f} kpc from full-physics metadata")
    if R200 is None:
        logger.error("Failed to determine R200 from dataset and full-physics metadata. Using R200 = 300 kpc as fallback.")
        R200 = 300.0

    radius = ds.quan(R200, "kpc")
    logger.info(f"R200 = {radius.to('kpc'):.2f} kpc, center = {center.to('kpc')} kpc")

    sphere = ds.sphere(center.to("kpc"), radius)

    if make_projections:
        logger.info("Creating projection plot with R200 overlay...")

        # Determine projection output directory
        proj_dir = PATHS.snapshot_plots(
            halo_id, redshift, experiment=experiment
        ) / "projections"
        proj_dir.mkdir(parents=True, exist_ok=True)

        # Extract snapshot name (e.g., DD2427, RD0030)
        snap_name = Path(snap_path).name

        for comp in avail_components:
            proj_path = proj_dir / f"{snap_name}_{comp}_projection_r200.png"

            try:
                # Lazy import to avoid circular dependency
                from .plotting import plot_comp_projection_with_r200

                plot_comp_projection_with_r200(
                    ds=ds,
                    center=center,
                    r200_kpc=R200,
                    redshift=redshift,
                    output_path=proj_path,
                    component=comp,
                    config=config,
                )
                logger.info(f"Saved projection to {proj_path}")
            except Exception as e:
                logger.error(f"Failed to create projection plot: {e}", exc_info=True)

    shape_tensors: List[Tuple[float, np.ndarray]] = []
    if not skip_shape_analysis:
        logger.info("Shape Analysis at different radii")

        radii_fractions = [0.01, 0.02, 0.05, 0.1, 0.2, 0.3, 0.5]  # fractions of R200
        radii_to_test = [f * R200 for f in radii_fractions]  # kpc
        for r in radii_to_test:
            try:
                test_sphere = ds.sphere(center.to("kpc"), ds.quan(r, "kpc"))

                x = (
                    test_sphere['star', 'particle_position_x'] - center[0]
                ).to('kpc').value
                y = (
                    test_sphere['star', 'particle_position_y'] - center[1]
                ).to('kpc').value
                z = (
                    test_sphere['star', 'particle_position_z'] - center[2]
                ).to('kpc').value

                star_positions = np.vstack((x, y, z))

                star_weights = np.array(
                    test_sphere["star", "particle_mass"].to("Msun").value
                )
                logger.info(f"Radius = {r} kpc:")

                # Try to compute shape tensor with error handling
                shape_tensor, (a, b, c) = get_shape_tensor(star_weights, star_positions)
                shape_tensors.append((r, shape_tensor))
                axis_ratio_ba = b / a
                axis_ratio_ca = c / a
                logger.info(f"Major axis (a): {a:.2f}")
                logger.info(f"Intermediate axis (b): {b:.2f}")
                logger.info(f"Minor axis (c): {c:.2f}")

                if axis_ratio_ca < 0.3 and axis_ratio_ba > 0.7:
                    logger.info("Likely disk-dominated (flattened, circular in-plane)")
                elif axis_ratio_ca < 0.5:
                    logger.info("Possible disk + bulge system")
                else:
                    logger.info("Likely spheroidal/elliptical (not disk-dominated)")
            except ValueError as e:
                logger.warning(f"Could not compute shape tensor at r={r} kpc: {e}", exc_info=True)

            finally:
                try:
                    del x, y, z
                    del star_positions
                    del star_weights
                except NameError:
                    pass
                if test_sphere is not None:
                    del test_sphere

                gc.collect()

    logger.info("Calculating Angular Momentum...")
    l = sphere.quantities.angular_momentum_vector(use_gas=has_gas, use_particles=True)
    l_hat = l / np.linalg.norm(l)
    del l
    gc.collect()

    logger.info(f"Angular Momentum unit vector = {l_hat}")
    Rmat = rotation_matrix_from_vectors(np.array([0, 0, 1]), l_hat.d)

    logger.debug(f"Rmat:\n{Rmat}")
    logger.debug(f"Rmat contains NaN: {np.any(np.isnan(Rmat))}")
    logger.debug(f"Rmat contains Inf: {np.any(np.isinf(Rmat))}")
    logger.debug(f"Rmat determinant: {np.linalg.det(Rmat):.6f} (should be ~1.0)")
    logger.debug(f"Rmat is orthogonal: {np.allclose(Rmat @ Rmat.T, np.eye(3))}")

    for comp in avail_components:
        add_rotated_fields(ds, Rmat, center.to("kpc"), comp=comp)

    logger.debug("\n=== DEBUG: Testing derived fields ===")

    try:
        for comp in avail_components:
            R_disk = sphere[(comp, "R_disk")]
            if len(R_disk) > 0:
                logger.debug(
                    f"{comp.upper()} R_disk range: {R_disk.min():.3e} to {R_disk.max():.3e}"
                )
            else:
                logger.debug(f"{comp.upper()} R_disk: no particles in sphere")
    except Exception as e:
        logger.error(e, exc_info=True)

    logger.info("Getting density profiles...")

    # Scale with R200, but use redshift-dependent floors
    min_rad_kpc = max(0.001 * R200, 0.01)  # High-z: tighter minimum

    min_radius = ds.quan(min_rad_kpc, "kpc")
    logger.info(f"Using min_radius = {min_rad_kpc:.3f} kpc (0.1% of R200={R200:.1f} kpc)")

    profiles = {
        comp: create_profiles(sphere, min_radius, radius, comp=comp)
        for comp in avail_components
    }

    if not profiles:
        raise RuntimeError("Profile creation failed completely")

    valid_profiles: Dict[str, "ProfileND"] = {
        comp: prof for comp, prof in profiles.items()
        if prof is not None
    }

    if not valid_profiles:
        raise RuntimeError("No profiles were successfully created")

    logger.info(
        f"Successfully created {len(valid_profiles)} profile(s): {list(valid_profiles.keys())}"
    )

    log_memory_usage(logger, "Create profiles")

    if output_dir is None:
        # Use sim_type as experiment if not "full"
        exp = experiment if experiment else (sim_type if sim_type != "full" else None)
        data_dir = PATHS.snapshot_data(halo_id, redshift, experiment=exp)
    else:
        data_dir = Path(output_dir)
    ensure_directory(data_dir)

    if save_profiles:
        for comp, profile in valid_profiles.items():
            filename = data_dir / f"3d_profile_{comp}.h5"
            profile.save_as_dataset(str(filename))
            logger.info(f"Saved {comp} profile to {filename}")

    # Calculate enclosed masses at various radii
    logger.info("Calculating enclosed masses...")
    # Use R200-scaled fractions when config says units="r200", else fixed kpc
    range_units = config.get("range_scaling", {}).get("units", "kpc") if config else "kpc"
    if range_units == "r200" and R200 is not None:
        r200_fractions = np.array([0.01, 0.02, 0.05, 0.1, 0.15, 0.25, 0.5, 0.75, 1.0])
        radii_for_mass = r200_fractions * R200
        logger.info(f"Using R200-scaled mass radii: {r200_fractions} × R200={R200:.1f} kpc")
    else:
        default_radii = [5, 10, 15, 20, 30, 50, 75, 100, 150, 200]
        if config is not None:
            radii_list = config.get("constants", {}).get("radii_list", default_radii)
        else:
            radii_list = default_radii
        radii_for_mass = np.array(radii_list, dtype=float)
        radii_for_mass = radii_for_mass[radii_for_mass <= R200]

    # Filter out radii smaller than yt's minimum sphere size
    min_radius_code = ds.index.get_smallest_dx().to("kpc").value
    valid_mask = radii_for_mass > min_radius_code
    if not valid_mask.all():
        n_skipped = (~valid_mask).sum()
        logger.warning(
            f"Skipping {n_skipped} radii below yt minimum sphere size "
            f"({min_radius_code:.4f} kpc): {radii_for_mass[~valid_mask]}"
        )
        radii_for_mass = radii_for_mass[valid_mask]

    masses = {"radii": radii_for_mass}

    for comp in avail_components:
        masses[f"M_{comp}"] = np.zeros_like(radii_for_mass, dtype=float)
    if component == "all" and len(avail_components) > 1:
        masses["M_total"] = np.zeros_like(radii_for_mass, dtype=float)

    for i, r in enumerate(radii_for_mass):
        test_sphere = ds.sphere(center.to("kpc"), ds.quan(r, "kpc"))

        if component == "star":
            M_star = test_sphere[("star", "particle_mass")].sum().to("Msun").value
            masses["M_star"][i] = M_star
        elif component == "gas":
            M_gas = test_sphere[("gas", "cell_mass")].sum().to("Msun").value
            masses["M_gas"][i] = M_gas
        elif component == "dm":
            M_dm = test_sphere[("dm", "particle_mass")].sum().to("Msun").value
            masses["M_dm"][i] = M_dm
        elif component == "all":
            total = 0.0
            if "star" in avail_components:
                M_star = test_sphere[("star", "particle_mass")].sum().to("Msun").value
                masses["M_star"][i] = M_star
                total += M_star
            if "gas" in avail_components:
                M_gas = test_sphere[("gas", "cell_mass")].sum().to("Msun").value
                masses["M_gas"][i] = M_gas
                total += M_gas
            if "dm" in avail_components:
                M_dm = test_sphere[("dm", "particle_mass")].sum().to("Msun").value
                masses["M_dm"][i] = M_dm
                total += M_dm
            if len(avail_components) > 1:
                masses["M_total"][i] = total
        else:
            logger.error(f"Unknown component for mass calculation: {component}")

    log_memory_usage(logger, "Calculate enclosed masses")

    # Convert lists to arrays
    for key in masses.keys():
        if key != "radii":
            masses[key] = np.array(masses[key])

    # Create metadata dictionary
    metadata = {
        "halo_id": halo_id,
        "redshift": redshift,
        "time_Gyr": time_Gyr,
        "dd_num": dd_num,
        "center_kpc": center.to("kpc").value,
        "R200_kpc": R200,
        "rotation_matrix": Rmat,
        "angular_momentum_unit_vector": l_hat.d,
        "sim_path": str(snap_path),
        "analysis_timestamp": datetime.datetime.now().isoformat(),
        "enclosed_masses": masses,
    }
    if shape_tensors:
        metadata["shape_tensors"] = {
            "radii": np.array([r for r, _ in shape_tensors]),
            "tensors": np.array([tensor for _, tensor in shape_tensors])
        }

    # Optionally save metadata
    if save_profiles:
        filepath = str(data_dir / "metadata.h5")
        save_metadata(data_dict=metadata, filepath=filepath)
        logger.info(f"Saved metadata to {filepath}")

    # Convert yt profiles to DensityProfile objects
    logger.info("Converting profiles to DensityProfile objects...")
    density_profiles: Dict[str, DensityProfile] = {}
    for comp in valid_profiles.keys():
        profile_file = data_dir / f"3d_profile_{comp}.h5"
        if profile_file.exists():
            ds_profile = yt.load(str(profile_file))  # type: ignore
            density_profiles[comp] = DensityProfile.from_profile_dataset(
                ds_profile, comp
            )
            logger.info(f"Created DensityProfile for {comp}")
        else:
            logger.warning(
                f"Profile file not found for {comp}, skipping DensityProfile creation"
            )

    logger.info("Profile generation completed successfully")
    # --- Memory Cleanup ---
    try:
        del ds
        del sphere
        del valid_profiles
        del ad
    except NameError:
        pass

    gc.collect()
    logger.debug("Freed yt dataset and forced garbage collection")

    return density_profiles, metadata
