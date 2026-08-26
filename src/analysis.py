import gc
import logging
import warnings
from pathlib import Path
from typing import TYPE_CHECKING, Any, Dict, List, Optional, Tuple

import h5py
import numpy as np
from scipy.integrate import IntegrationWarning

from .config import ProjectPaths
from .io import ensure_directory
from .models import ModelRegistry

if TYPE_CHECKING:
    from .workflow import FittingResult

logger = logging.getLogger(__name__)


def spherical_average(data_2d, r_2d, r_bins, weights=None):
    """
    Average data_2d into radial bins.

    Parameters
    ----------
    data_2d : 2D array
        Data to average (e.g., density)
    r_2d : 2D array
        Spherical radius at each grid point
    r_bins : 1D array
        Radial bin edges
    weights : 2D array, optional
        Weights for averaging (e.g., cell volumes). If None, uses simple mean.

    Returns
    -------
    averaged : 1D array
        Averaged data in each radial bin
    """
    averaged = np.zeros(len(r_bins) - 1)
    for i in range(len(r_bins) - 1):
        mask = (r_2d >= r_bins[i]) & (r_2d < r_bins[i + 1])
        if mask.any():
            if weights is not None:
                # Volume-weighted average
                averaged[i] = np.average(data_2d[mask], weights=weights[mask])
            else:
                # Simple mean
                averaged[i] = np.mean(data_2d[mask])
        else:
            averaged[i] = np.nan
    return averaged


def spherically_average_model(model_evaluator, r_array, model_summary, n_theta=50):
    """
    Convert cylindrical model rho(R, z) to spherical rho(r) by direct sampling.

    Parameters
    ----------
    model_evaluator: callable
        Function that takes (R, z) coordinates and returns density
    r_array : array
        Spherical radii at which to evaluate
    model_summary : dict
        Model parameters
    n_theta : int
        Number of polar angle samples per radius

    Returns
    -------
    rho_spherical: array
        Spherically-averaged density at each r
    """
    rho_spherical = np.zeros_like(r_array)

    for i, r in enumerate(r_array):
        theta = np.linspace(0, np.pi, n_theta)
        R_samples = r * np.sin(theta)
        z_samples = r * np.cos(theta)

        rho_samples = model_evaluator((R_samples, z_samples), model_summary)
        weights = np.sin(theta)

        valid_mask = np.isfinite(rho_samples)
        if np.any(valid_mask):
            rho_spherical[i] = np.average(
                rho_samples[valid_mask], weights=weights[valid_mask]
            )
    return rho_spherical


def calculate_fit_statistics(
    data,
    model,
    particle_counts=None,
    n_parameters=7,
    systematic_floor=0.1,
    min_counts=None,
    verbose=False,
):
    """
    Calculate Fit statistics including uncertainties (poisson and systematic),
    residuals [(data - model) / data], chi-squared and the root mean squared of
    the residuals.

    Args:
        data (array): Density Data points (units = Msun/kpc**3)
        model (array): Density Model points evaluated at the data points (same units)
        particle_counts (array, optional): Number of particle counts in each bin.
                                           Defaults to None.
        n_parameters (int, optional): Number of model parameters. Defaults to 7.
        systematic_floor (float, optional): Minimum uncertainty on data.
                                            Defaults to 0.1.
        min_counts (int, optional): Minimum counts in a bin to be useable for
                                    statistics. Defaults to None.
        verbose (bool, optional): Print summaries of analysis. Defaults to True.

    Returns:
        dict: summary of statistics includes: chi-squared, reduced chi-squared,
              dof, root-mean-squared, resdiuals, uncertainties, mask, and number
              of data points.
    """
    data = np.asarray(data)
    model = np.asarray(model)

    # Apply minimum count mask if requested
    if min_counts is not None and particle_counts is not None:
        mask = particle_counts >= min_counts
        if verbose:
            print(f"Excluded {np.sum(~mask)} bins with counts < {min_counts}")
    else:
        mask = np.ones(len(data), dtype=bool)

    # Calculate Uncertainties
    if particle_counts is not None:
        counts_masked = np.asarray(particle_counts)[mask]
        poisson_unc = data[mask] / np.sqrt(counts_masked)
        systematic_unc = systematic_floor * data[mask]
        unc = np.sqrt(poisson_unc**2 + systematic_unc**2)
    else:
        # Pure systematic unceratinty
        unc = systematic_floor * data[mask]

    # Calculate residuals
    residuals = (data[mask] - model[mask]) / data[mask]

    # Calculate chi-squared
    chi_squared = np.sum(((data[mask] - model[mask]) / unc) ** 2)

    # Degrees of Freedom
    n_data = np.sum(mask)
    dof = n_data - (n_parameters if n_parameters else 0)

    # Reduced chi-squared
    reduced_chi_squared = chi_squared / dof if dof > 0 else np.nan

    # Sumary stats
    rms_residual = np.sqrt(np.mean(residuals**2))
    median_residual = np.median(residuals)
    mean_frac_unc = np.mean(unc / data[mask])

    # Print Diagnostics
    if verbose:
        print(f"Fit Statistics:")
        print(f"  chi-squared = {chi_squared:.2f}")
        if n_parameters:
            print(f"  Reduced chi-squared = {reduced_chi_squared:.2f} (dof={dof})")
        print(
            f"  RMS fractional residual: {rms_residual:.3f} ({rms_residual * 100:.1f}%)"
        )
        print(f"  Median residual: {median_residual:.3f} ({median_residual * 100:.1f}%)")
        print(
            f"  Mean fractional uncertainty: {mean_frac_unc:.3f} ({mean_frac_unc * 100:.1f}%)"
        )
        print(f"  Residual range: [{np.min(residuals):.3f}, {np.max(residuals):.3f}]")

        # Check for outliers
        n_outliers = np.sum(np.abs(residuals) > 1)
        if n_outliers > 0:
            print(f"  WARNING: {n_outliers} bins with |residual| > 1")

    return {
        "chi_squared": chi_squared,
        "reduced_chi_squared": reduced_chi_squared,
        "dof": dof,
        "rms_residual": rms_residual,
        "median_residual": median_residual,
        "residuals": residuals,
        "uncertainties": unc,
        "mask": mask,
        "n_data": n_data,
    }


def gelman_rubin_diagnostic(sampler, nburn, param_names, verbose=False):
    """
    Calculate the Gelman-Rubin convergence diagnostic (R-hat) for MCMC chains.

    The Gelman-Rubin statistic assesses MCMC convergence by comparing the
    between-chain variance to the within-chain variance. Values close to 1.0
    indicate convergence; values > 1.1 typically suggest chains have not yet
    converged. This implementation uses the split-chain approach where each
    chain is split in half to increase the number of chains for comparison.

    Parameters
    ----------
    sampler : emcee.EnsembleSampler
        The MCMC sampler object containing the chain history.
    nburn : int
        Number of burn-in steps to discard from the beginning of each chain.
    param_names : list of str
        Names of the model parameters corresponding to each dimension of the
        parameter space. Used for printing diagnostic output.
    verbose : bool, optional
        If True, prints R-hat values for each parameter and the maximum R-hat
        across all parameters. Defaults to True.

    Returns
    -------
    rhat : numpy.ndarray
        Array of R-hat values, one for each parameter. Shape: (ndim,)
        Values close to 1.0 indicate good convergence.

    Notes
    -----
    The R-hat statistic is calculated as:

        R-hat = sqrt(V_total / W)

    where V_total is the estimated marginal posterior variance and W is the
    within-chain variance. See Gelman & Rubin (1992) and Brooks & Gelman (1998)
    for details on the diagnostic.

    References
    ----------
    Gelman, A., & Rubin, D. B. (1992). Inference from iterative simulation
    using multiple sequences. Statistical Science, 7(4), 457-472.

    Brooks, S. P., & Gelman, A. (1998). General methods for monitoring
    convergence of iterative simulations. Journal of Computational and
    Graphical Statistics, 7(4), 434-455.
    """
    chains = sampler.get_chain(discard=nburn)
    nwalkers, nsteps, ndim = chains.shape

    split_point = nsteps // 2
    chain_1 = chains[:split_point]
    chain_2 = chains[split_point: 2 * split_point]

    rhat = []
    for param_idx in range(ndim):
        all_chains = np.concatenate(
            [
                chain_1[:, :, param_idx].T,  # shape = (nwalkers, split_point)
                chain_2[:, :, param_idx].T,  # shape = (nwalkers, split_point)
            ],
            axis=0,
        )  # shape = (2*nwalkers, split_point)

        chain_means = np.mean(all_chains, axis=1)
        B = split_point * np.var(chain_means, ddof=1)

        chain_vars = np.var(all_chains, axis=1, ddof=1)
        W = np.mean(chain_vars)

        var_total = ((split_point - 1) / split_point) * W + B / split_point
        rhat.append(np.sqrt(var_total / W))

    rhat = np.array(rhat)
    if verbose:
        logger.info("Gelman-Rubin R-hat Diagnostic:")
        for name, r in zip(param_names, rhat):
            logger.info(f"   {name:12s}: {r:.4f}")
        logger.info(f"    Max R-hat: {np.max(rhat):.4f}")
    return rhat


def compare_models(
    model_summaries,
    coords,
    data,
    logger,
    data_range=None,
    model_evaluators=None,
    use_log_space=True,
    counts=None,
    sigma_sys=0.10,
):
    """
    Compare models using chi-squared statistic.

    Parameters
    ----------
    model_summaries : dict
        Dictionary of {model_name: summary_dict} from fitting functions.
        If summary is None, model is skipped.
    coords : tuple of arrays
        Coordinate arrays (e.g., (r,) for 1D, (R, z) for 2D).
    data : array
        Observed data to compare against.
    data_range : dict, optional
        Coordinate ranges: {coord_name: (min, max), ...} or
        {coord_idx: (min, max), ...}. Default: use all data.
    model_evaluators : dict, optional
        Dictionary of {model_name: callable} where each callable is
        model_func(coords, summary) -> predicted_data.
        If None, attempts to infer from summary keys.
    use_log_space : bool, optional
        If True, compare in log10 space. Default: True.
    counts : array, optional
        Particle counts per bin for Poisson uncertainties.
        If provided, uncertainties are σ_log = 0.434/√N.
        If None, uses fitted sigma or fixed uncertainty.
    sigma_sys : float, optional
        Systematic uncertainty floor in log-space (dex).
        Added in quadrature: σ_total = √(σ_Poisson² + σ_sys²).
        Accounts for substructure, binning, centering, etc.
        Default: 0.0 (Poisson only).
        Typical values: 0.05-0.15 dex.

    Returns
    -------
    best_model_name : str or None
        Name of best-fitting model (lowest reduced chi-squared).
    best_summary : dict or None
        Summary dict of best model.
    comparison_table : dict
        Dictionary with comparison metrics for each model.
    report : str
        Report of the comparison method

    Examples
    --------
    # 1D bulge comparison
    model_evals = {
        'Hernquist': lambda coords, s: models.hernquist_density(
            coords[0], s['rho_0'], s['r_s']),
        'Exponential': lambda coords, s: s['rho_0'] * np.exp(-coords[0] / s['r_s'])
    }

    best, summary, table = compare_models(
        {'Hernquist': hern_sum, 'Exponential': exp_sum},
        coords=(r,),
        data=rho,
        data_range={0: (0.3, 5)},  # coord index 0 = r
        model_evaluators=model_evals
    )

    # 2D disk comparison
    model_evals = {
        'Single': lambda coords, s: models.disk_density(
            coords[0], coords[1], s['rho_0'], s['R_d'], s['z_0']),
        'Double': lambda coords, s: models.double_exp_model(
            coords[0], coords[1], s['rho_01'], s['rho_02'],
            s['R_d1'], s['R_d2'], s['z_0'])
    }

    best, summary, table = compare_models(
        {'Single': single_sum, 'Double': double_sum},
        coords=(R, z),
        data=rho,
        data_range={0: (3, 30), 1: (0, 5)},  # R and z ranges
        model_evaluators=model_evals
    )
    """
    output_lines = []
    # Apply data range mask
    mask = np.ones(len(data), dtype=bool)
    mask &= np.isfinite(data) & (data > 0)

    if data_range is not None:
        for coord_key, (coord_min, coord_max) in data_range.items():
            # Support both integer indices and string names
            if isinstance(coord_key, int):
                coord_idx = coord_key
            else:
                logger.error(
                    f"Invalid coordinate key type: {type(coord_key)}. "
                    f"Expected int, got {coord_key}"
                )
                raise ValueError(
                    "String coordinate names not yet supported. Use integer indices."
                )

            mask &= (coords[coord_idx] >= coord_min) & (coords[coord_idx] < coord_max)

    coords_fit = tuple(c[mask] for c in coords)
    data_fit = data[mask]
    counts_fit = counts[mask] if counts is not None else None
    n_data = len(data_fit)

    if n_data < 10:
        logger.warning("Very few data points in fitting region!")

    # Storage for comparison
    comparison_table = {}

    output_lines.append(
        f"{'Model':<20s} {'χ²':>10s} {'χ²_red':>10s} {'R_hat_max':>10s} {'Status':>10s}"
    )
    output_lines.append("-" * 70)

    for model_name, summary in model_summaries.items():
        if summary is None:
            output_lines.append(f"{model_name:<20s} {'FAILED':>58s}")
            continue

        # Get number of parameters (excluding 'samples' and 'sampler')
        samples = summary.get("samples")
        if samples is None:
            logger.warning(f"{model_name} has no MCMC samples")
            output_lines.append(f"{model_name:<20s} {'NO SAMPLES':>58s}")
            continue

        n_params = samples.shape[1]

        # Evaluate model
        if model_evaluators is not None and model_name in model_evaluators:
            try:
                model_pred = model_evaluators[model_name](coords_fit, summary)
            except Exception as e:
                logger.error(
                    f"Model evaluation error for {model_name}: {e}", exc_info=True
                )
                output_lines.append(f"{model_name:<20s} {'EVAL ERROR':>58s}")
                output_lines.append(f"  Error: {e}")
                continue
        else:
            output_lines.append(f"{model_name:<20s} {'NO EVALUATOR':>58s}")
            continue

        # Check for valid model predictions
        if np.any(~np.isfinite(model_pred)) or np.any(model_pred <= 0):
            logger.warning(
                f"Invalid model predictions for {model_name}: "
                f"non-finite={np.sum(~np.isfinite(model_pred))}, "
                f"non-positive={np.sum(model_pred <= 0)}"
            )
            output_lines.append(f"{model_name:<20s} {'INVALID MODEL':>58s}")
            continue

        # Calculate log-likelihood
        if use_log_space:
            log_data = np.log10(data_fit)
            log_model = np.log10(model_pred)

            # Determine uncertainties
            if counts_fit is not None:
                # Poisson uncertainty in log-space: σ_log = 0.434 / √N
                # Add minimum count floor to avoid division by zero
                counts_safe = np.maximum(counts_fit, 1.0)
                sigma_poisson = 0.434 / np.sqrt(counts_safe)

                # Add systematic floor if specified
                if sigma_sys > 0:
                    sigma_total = np.sqrt(sigma_poisson**2 + sigma_sys**2)
                else:
                    sigma_total = sigma_poisson

                # Note: We use Poisson (+optional systematic) uncertainties.
                # The fitted MCMC 'sigma' is not used to avoid double-counting.
            else:
                # Fall back to fixed uncertainty if no counts available
                if sigma_sys > 0:
                    sigma_total = np.maximum(sigma_sys, 0.05)
                else:
                    sigma_total = 0.05  # 0.05 dex ≈ 12% fractional uncertainty

            chi2 = np.sum(((log_data - log_model) / sigma_total) ** 2)
        else:
            # Linear space (for completeness, but typically use log-space)
            if counts_fit is not None:
                # Poisson uncertainty: σ = ρ / √N
                counts_safe = np.maximum(counts_fit, 1.0)
                sigma_poisson = data_fit / np.sqrt(counts_safe)

                # Add systematic floor
                if sigma_sys > 0:
                    sigma_sys_linear = sigma_sys * data_fit  # Convert dex to linear
                    sigma_total = np.sqrt(sigma_poisson**2 + sigma_sys_linear**2)
                else:
                    sigma_total = sigma_poisson
            else:
                # Fall back to fractional uncertainty
                if sigma_sys > 0:
                    frac_err = np.maximum(sigma_sys, 0.05)
                else:
                    frac_err = 0.05
                sigma_total = frac_err * data_fit

            chi2 = np.sum(((data_fit - model_pred) / sigma_total) ** 2)

        reduced_chi2 = chi2 / (n_data - n_params)

        # Get convergence diagnostic
        rhat_values = summary.get("rhat", [1.0])
        if isinstance(rhat_values, np.ndarray):
            rhat_max = np.max(rhat_values)
        else:
            rhat_max = 1.0

        # Store results
        comparison_table[model_name] = {
            "chi2": chi2,
            "chi2_reduced": reduced_chi2,
            "rhat_max": rhat_max,
            "n_params": n_params,
            "summary": summary,
        }

        # Print row
        status = "OK" if rhat_max < 1.1 else "CHECK"
        output_lines.append(
            f"{model_name:<20s} {chi2:10.2f} {reduced_chi2:10.3f} "
            f"{rhat_max:10.3f} {status:>10s}"
        )

    # Find best model
    if not comparison_table:
        logger.warning("No valid models to compare!")
        return None, None, {}, "No valid models to compare"

    best_model_name = min(
        comparison_table, key=lambda k: comparison_table[k]["chi2_reduced"]
    )
    best_summary = comparison_table[best_model_name]["summary"]
    logger.info(
        f"Best model selected: {best_model_name} (χ²_red={comparison_table[best_model_name]['chi2_reduced']:.3f})"
    )

    output_lines.append("-" * 70)
    output_lines.append(
        f"*** BEST MODEL: {best_model_name} (χ²_red = {comparison_table[best_model_name]['chi2_reduced']:.3f}) ***"
    )

    return best_model_name, best_summary, comparison_table, "\n".join(output_lines)


def save_results(
    file_path, model_summaries, best_model_name, logger, comparison_table=None
):
    """
    Save fitted model parameters, statistics, and diagnostics to HDF5 file.

    Parameters
    ----------
    file_path : str or Path
        Output HDF5 file path (e.g., 'results/halo/star_bulge_results.h5')
    model_summaries : dict
        Dictionary of {model_name: summary_dict} from fitting functions
    best_model_name : str
        Name of best fitting model
    comparison_table : dict, optional
        Output from compare_models() with chi-squared metrics
    """
    file_path = Path(file_path)
    ensure_directory(file_path.parent)

    with h5py.File(file_path, "w") as f:
        # store best model metadata (convert None to empty string for HDF5 compatibility)
        f.attrs["best_model"] = best_model_name if best_model_name is not None else ""

        # save each model's results
        for model_name, summary in model_summaries.items():
            if summary is None:
                continue

            grp = f.create_group(model_name)
            grp.attrs["is_best"] = model_name == best_model_name

            # Save MCMC samples
            if "samples" in summary:
                grp.create_dataset("samples", data=summary["samples"])

            # save fitted parameters (median + uncertainties)
            for param_name, value in summary.items():
                if param_name in ["samples", "sampler", "rhat", "n_eff"]:
                    continue  # Skip these - handled separately below
                if isinstance(value, (int, float, np.number)):
                    grp.attrs[param_name] = value
                elif isinstance(value, np.ndarray):
                    grp.create_dataset(param_name, data=value)

            # save convergence diagnostics
            if "rhat" in summary:
                grp.create_dataset("rhat", data=summary["rhat"])
            if "n_eff" in summary and summary["n_eff"] is not None:
                grp.create_dataset("n_eff", data=summary["n_eff"])

            if comparison_table and model_name in comparison_table:
                metrics = comparison_table[model_name]
                grp.attrs["chi2"] = metrics["chi2"]
                grp.attrs["chi2_reduced"] = metrics["chi2_reduced"]
                grp.attrs["rhat_max"] = metrics["rhat_max"]
                grp.attrs["n_params"] = metrics["n_params"]

    logger.info(f"Saved results to {file_path}")


def get_shape_tensor(
    weights: np.ndarray,
    positions: np.ndarray,
) -> Tuple[np.ndarray, Tuple[float, float, float]]:
    """
    Get the shape of the galactic disk.

    Parameters
    ----------
    weights : array
        Particle weights (e.g., masses)
    positions : array (3, N)
        Particle positions in 3D

    Returns
    -------
    shape_tensor : array (3, 3)
        The shape tensor matrix
    axis_ratios : tuple (a, b, c)
        The three axis ratios derived from eigenvalues

    Raises
    ------
    ValueError
        If inputs contain invalid data or insufficient particles
    """
    # Validate inputs
    n = weights.size
    if n == 0:
        raise ValueError("No particles found - cannot compute shape tensor")

    if n < 10:
        raise ValueError(f"Too few particles ({n}) - need at least 10 for reliable shape analysis")

    # Check for NaNs or infs in inputs
    if not np.isfinite(weights).all():
        raise ValueError("Weights contain NaNs or infs")

    if not np.isfinite(positions).all():
        raise ValueError("Positions contain NaNs or infs")

    # Check if weights sum is valid
    weight_sum = weights.sum()
    if weight_sum <= 0 or not np.isfinite(weight_sum):
        raise ValueError(f"Invalid weight sum: {weight_sum}")

    # ---- Shape tensor (memory-efficient) ----
    # shape_tensor_ij = sum_k w_k x_i,k x_j,k / sum_k w_k
    shape_tensor = np.einsum(
        "k,ik,jk->ij",
        weights,
        positions,
        positions,
        optimize=True,
    ) / weight_sum

    # Check if shape tensor has invalid values
    if not np.isfinite(shape_tensor).all():
        raise ValueError("Shape tensor contains NaNs or infs")

    # get eigenvalues and vectors
    try:
        eigenvalues, eigenvectors = np.linalg.eigh(shape_tensor)
    except np.linalg.LinAlgError as e:
        raise ValueError(f"Failed to compute eigenvalues: {e}")

    # Sort descending
    eigenvalues = eigenvalues[::-1]
    eigenvectors = eigenvectors[:, ::-1]

    # Check for negative eigenvalues (shouldn't happen for a proper shape tensor)
    if np.any(eigenvalues < 0):
        raise ValueError(f"Negative eigenvalues detected: {eigenvalues}")

    # axis ratios (for a disk: a >> b = c)
    a, b, c = np.sqrt(eigenvalues)

    return shape_tensor, (a, b, c)


def calculate_model_masses(
    results: Dict[str, "FittingResult"],
    radii: np.ndarray,
    logger: logging.Logger
):
    """
    Calculate mass estimates from fitted models at specified radii.

    Uses ModelRegistry to automatically select the correct mass integrator
    for each model, eliminating hardcoded model-specific logic.

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

        # Calculate bulge mass if available (using ModelRegistry)
        if "bulge" in results:
            best_model = results["bulge"].best_model
            summary = results["bulge"].summary

            if summary is not None and best_model is not None:
                try:
                    mass_integrator = ModelRegistry.get_mass_integrator(best_model)
                    M_bulge = mass_integrator(r, summary)
                except Exception as e:
                    logger.warning(
                        f"Could not calculate bulge mass for {best_model}: {e}"
                    )

        # Calculate disk mass if available (using ModelRegistry)
        if "disk" in results:
            best_model = results["disk"].best_model
            summary = results["disk"].summary

            if summary is not None and best_model is not None:
                try:
                    z_max = 2 * r  # Use 2*r as z_max for disk integration
                    mass_integrator = ModelRegistry.get_mass_integrator(best_model)
                    # For cylindrical models, mass_integrator expects (R, z, summary)
                    M_disk = mass_integrator(r, z_max, summary)
                except Exception as e:
                    logger.warning(
                        f"Could not calculate disk mass for {best_model}: {e}"
                    )

        # Calculate halo mass if available (using ModelRegistry)
        if "halo" in results:
            best_model = results["halo"].best_model
            summary = results["halo"].summary

            if summary is not None and best_model is not None:
                try:
                    mass_integrator = ModelRegistry.get_mass_integrator(best_model)
                    M_halo = mass_integrator(r, summary)
                except Exception as e:
                    logger.warning(
                        f"Could not calculate halo mass for {best_model}: {e}"
                    )

        masses["bulge"].append(M_bulge)
        masses["disk"].append(M_disk)
        masses["halo"].append(M_halo)
        masses["total"].append(M_bulge + M_disk + M_halo)

    # Convert to arrays
    for key in ["bulge", "disk", "halo", "total"]:
        masses[key] = np.array(masses[key])

    return masses


def calc_masses_with_uncertainties(
    results: Dict[str, "FittingResult"],
    radii: np.ndarray,
    n_samples: int = 1000,
):
    # Guard: Check if radii is empty
    if len(radii) == 0:
        logger.warning("Cannot calculate masses: radii array is empty")
        # Return arrays of zeros instead of empty lists to maintain consistency
        return {
            'radii': np.array([]),
            'bulge': np.array([]), 'bulge_lo': np.array([]), 'bulge_hi': np.array([]),
            'disk': np.array([]), 'disk_lo': np.array([]), 'disk_hi': np.array([]),
            'halo': np.array([]), 'halo_lo': np.array([]), 'halo_hi': np.array([]),
            'total': np.array([]), 'total_lo': np.array([]), 'total_hi': np.array([])
        }

    # Guard: Check if results is empty
    if not results:
        logger.warning("Cannot calculate masses: no fitting results provided")
        # Return numpy arrays (not lists) to maintain consistency with normal return path
        zeros = np.zeros(len(radii))
        return {
            'radii': radii,
            'bulge': zeros.copy(), 'bulge_lo': zeros.copy(), 'bulge_hi': zeros.copy(),
            'disk': zeros.copy(), 'disk_lo': zeros.copy(), 'disk_hi': zeros.copy(),
            'halo': zeros.copy(), 'halo_lo': zeros.copy(), 'halo_hi': zeros.copy(),
            'total': zeros.copy(), 'total_lo': zeros.copy(), 'total_hi': zeros.copy()
        }

    masses: Dict[str, Any] = {'radii': radii}

    # Count active components for logging
    active_components = [c for c in ["bulge", "disk", "halo"] if c in results]
    total_integrations = len(active_components) * len(radii) * n_samples
    logger.info(f"Starting mass calculation: {len(radii)} radii x {n_samples} samples x {len(active_components)} components = {total_integrations:,} integrations")

    for component in ["bulge", "disk", "halo"]:
        masses[component] = []
        masses[f"{component}_lo"] = []
        masses[f"{component}_hi"] = []

        if component not in results:
            for _ in radii:
                masses[component].append(0.0)
                masses[f"{component}_lo"].append(0.0)
                masses[f"{component}_hi"].append(0.0)
            continue

        best_model = results[component].best_model
        summary = results[component].summary

        if summary is None or best_model is None:
            # Component exists but has no valid fit - fill with zeros
            for _ in radii:
                masses[component].append(0.0)
                masses[f"{component}_lo"].append(0.0)
                masses[f"{component}_hi"].append(0.0)
            continue

        logger.info(f"  Calculating {component} masses using {best_model} model...")

        mass_integrator = ModelRegistry.get_mass_integrator(best_model)
        flat_samples = summary['samples']
        # Use stored param names if available, otherwise fallback to model-specific mapping
        param_names = summary.get('_param_names')
        if param_names is None:
            # Fallback for older result files without _param_names
            # Map model names to their expected sample column order
            model_param_map = {
                "NFW": ["log10_rho0", "log10_rs", "log_sigma"],
                "Hernquist": ["log10_rho0", "log10_rs", "log_sigma"],
                "Exponential": ["log10_rho0", "log10_rs", "log_sigma"],
                "Two-Power": ["log10_rho0", "log10_rs", "alpha", "beta", "log_sigma"],
                "Three-Power": ["log10_rho0", "log10_rs", "alpha", "beta", "gamma", "log_sigma"],
                "Single-Exponential": ["log10_rho01", "R_d1", "z_0", "log_sigma"],
                "Double-Exponential": ["log10_rho01", "log10_rho02", "R_d1", "R_d2", "log10_z0", "log_sigma"],
                "Single-Disk-Flaring": ["log10_rho01", "R_d1", "A", "B", "log_sigma"],
                "Double-Disk-Flaring": ["log10_rho01", "log10_rho02", "R_d1", "R_d2", "A", "B", "log_sigma"],
                "Exp-Cutoff": ["log10_rho0", "r_core", "r_cutoff", "alpha", "log_sigma"],
                "Cored-bulge": ["log10_rho0", "log10_rs", "log10_rc", "alpha", "beta", "log_sigma"],
                "Core-Only": ["log10_rho0", "log10_rs", "log10_rc", "alpha", "log_sigma"],
            }
            param_names = model_param_map.get(best_model)
            if param_names is None:
                # Last resort: filter keys and truncate
                param_names = [k for k in summary if not k.endswith('_err') and k not in
                               ["samples", "acceptance_fraction", "rhat", "n_eff", "sigma"]]
                param_names = param_names[:flat_samples.shape[1]]
        # subsample
        indices = np.random.choice(
            len(flat_samples),
            min(n_samples, len(flat_samples)),
            replace=False
        )

        # Track progress and failures for this component
        n_radii = len(radii)
        log_interval = max(1, n_radii // 10)  # Log every 10%
        failed_integrations = 0

        # Suppress integration warnings during mass calculation (they're expected for some samples)
        with warnings.catch_warnings():
            warnings.filterwarnings('ignore', category=IntegrationWarning)

            for i_r, r in enumerate(radii):
                # Progress logging
                if i_r % log_interval == 0 or i_r == n_radii - 1:
                    progress = (i_r + 1) / n_radii * 100
                    logger.debug(f"    {component}: {i_r + 1}/{n_radii} radii ({progress:.0f}%)")

                mass_samples = []
                for idx in indices:
                    sample_dict = {name: flat_samples[idx, i]
                                   for i, name in enumerate(param_names)}
                    # Add linear-scale aliases for log params (mass integrators expect linear scale)
                    for key in list(sample_dict.keys()):
                        if key.startswith("log10_"):
                            base = key.replace("log10_", "")
                            sample_dict[base] = 10 ** sample_dict[key]
                        if "_" in key and not key.endswith("_err"):
                            no_underscore = key.replace("_", "")
                            if no_underscore not in sample_dict:
                                sample_dict[no_underscore] = sample_dict[key]
                    try:
                        if component == "disk":
                            mass_samples.append(mass_integrator(r, 2 * r, sample_dict))
                        else:
                            mass_samples.append(mass_integrator(r, sample_dict))
                    except Exception:
                        failed_integrations += 1

                if len(mass_samples) == 0:
                    masses[component].append(np.nan)
                    masses[f"{component}_lo"].append(np.nan)
                    masses[f"{component}_hi"].append(np.nan)
                    continue
                lo, med, hi = np.percentile(mass_samples, [16, 50, 84])
                masses[component].append(med)
                masses[f"{component}_lo"].append(med - lo)
                masses[f"{component}_hi"].append(hi - med)

        # Summary for this component
        total_attempted = n_radii * len(indices)
        if failed_integrations > 0:
            logger.warning(f"    {component}: {failed_integrations}/{total_attempted} integrations failed ({100 * failed_integrations / total_attempted:.1f}%)")
        else:
            logger.info(f"    {component}: completed {total_attempted:,} integrations successfully")

    # Debug: Log list lengths before conversion
    logger.debug(f"Pre-conversion list lengths: bulge={len(masses.get('bulge', []))}, "
                 f"disk={len(masses.get('disk', []))}, halo={len(masses.get('halo', []))}, "
                 f"expected={len(radii)}")

    # Convert to arrays
    for key in masses:
        if key != "radii":
            masses[key] = np.array(masses[key])

    # Guard: Ensure all component arrays have consistent shapes before summing
    # This handles edge cases where radii might be empty or components are missing
    component_shapes = [masses["bulge"].shape, masses["disk"].shape, masses["halo"].shape]
    if len(set(component_shapes)) > 1:
        logger.warning(
            f"Component arrays have inconsistent shapes: "
            f"bulge={masses['bulge'].shape}, disk={masses['disk'].shape}, halo={masses['halo'].shape}. "
            f"This indicates an issue with radii or component initialization."
        )
        # Use the maximum shape and pad others with zeros
        max_len = max(len(masses["bulge"]), len(masses["disk"]), len(masses["halo"]))
        for comp in ["bulge", "disk", "halo"]:
            if len(masses[comp]) < max_len:
                masses[comp] = np.pad(masses[comp], (0, max_len - len(masses[comp])), constant_values=0)
            for suffix in ["_lo", "_hi"]:
                key = f"{comp}{suffix}"
                if len(masses[key]) < max_len:
                    masses[key] = np.pad(masses[key], (0, max_len - len(masses[key])), constant_values=0)

    masses["total"] = masses["bulge"] + masses["disk"] + masses["halo"]
    # Quadrature sum for independent errors (approximation)
    masses["total_lo"] = np.sqrt(masses["bulge_lo"]**2 + masses["disk_lo"]**2 + masses["halo_lo"]**2)
    masses["total_hi"] = np.sqrt(masses["bulge_hi"]**2 + masses["disk_hi"]**2 + masses["halo_hi"]**2)

    return masses


def compare_masses(
    model_masses: Dict[str, np.ndarray],
    sim_masses: Dict[str, np.ndarray],
    component_type: str,
):
    """
    Compare model mass estimates with actual simulation masses.

    Args:
        model_masses: Dictionary from calculate_model_masses()
        sim_masses: Dictionary from metadata['enclosed_masses']
        component_type: 'star', 'gas', or 'dm'
        logger: Logger instance
    """
    logger.info("=" * 70)
    logger.info("MASS COMPARISON: MODEL vs SIMULATION")
    logger.info("=" * 70)

    has_uncertainties = "total_lo" in model_masses

    # Map component types to simulation mass keys
    sim_key_map = {"star": "M_star", "gas": "M_gas", "dm": "M_dm"}

    if component_type not in sim_key_map:
        logger.warning(f"Unknown component type: {component_type}")
        return

    sim_key = sim_key_map[component_type]

    # Guard: Check if simulation masses are available
    if "radii" not in sim_masses or len(sim_masses["radii"]) == 0:
        logger.warning(
            f"No simulation mass data available for {component_type} - "
            f"skipping mass comparison (empty radii array)"
        )
        return

    if sim_key not in sim_masses or len(sim_masses[sim_key]) == 0:
        logger.warning(
            f"No {sim_key} data available in simulation masses for {component_type} - "
            f"skipping mass comparison"
        )
        return

    # Guard: Check if model masses are available
    if "radii" not in model_masses or len(model_masses["radii"]) == 0:
        logger.warning(
            f"No model mass data available for {component_type} - "
            f"skipping mass comparison (empty radii array)"
        )
        return

    if "total" not in model_masses or len(model_masses["total"]) == 0:
        logger.warning(
            f"No total mass data in model masses for {component_type} - "
            f"skipping mass comparison"
        )
        return

    radii = model_masses["radii"]

    logger.info(f"{component_type.upper()} Component Mass Comparison:")
    logger.info("-" * 70)
    if has_uncertainties:
        logger.info(
            f"{'Radius':>8} {'Sim Mass':>12} {'Model Mass':>12} {'Uncertainty':>14} {'Rel. Error':>12}"
        )
    else:
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

        if has_uncertainties:
            err_lo = model_masses['total_lo'][i]
            err_hi = model_masses['total_hi'][i]
            logger.info(
                f"{r:8.1f} {sim_mass:12.3e} {model_mass:12.3e} +{err_hi:.2e}/-{err_lo:.2e} {rel_error:12.2f}"
            )
        else:
            logger.info(
                f"{r:8.1f} {sim_mass:12.3e} {model_mass:12.3e} {diff:12.3e} {rel_error:12.2f}"
            )
    # Calculate statistics
    sim_masses_interp = np.interp(radii, sim_masses["radii"], sim_masses[sim_key])
    rel_errors = np.divide(
        100 * (model_masses['total'] - sim_masses_interp),
        sim_masses_interp,
        out=np.zeros_like(model_masses['total']),
        where=sim_masses_interp != 0,
    )

    logger.info("-" * 70)
    logger.info(f"Mean relative error: {np.mean(rel_errors):.2f}%")
    logger.info(f"Median relative error: {np.median(rel_errors):.2f}%")
    logger.info(f"RMS relative error: {np.sqrt(np.mean(rel_errors**2)):.2f}%")
    logger.info("=" * 70)


def calculate_component_statistics(
    results: Dict[str, "FittingResult"],
    model_masses: Dict,
    component_type: str,
    metadata_dict: Optional[Dict],
):
    """
    Calculate derived statistics from fitting results.

    Args:
        results: Dictionary with fitting results
        model_masses: Dictionary with calculated masses
        component_type: 'star', 'gas', or 'dm'

    Returns:
        dict: Derived statistics
    """
    stats = {}

    # Mass fractions
    if component_type == "star" and "bulge" in results and "disk" in results:
        M_bulge_50 = model_masses["bulge"][
            np.argmin(np.abs(model_masses["radii"] - 50))
        ]
        M_disk_50 = model_masses["disk"][np.argmin(np.abs(model_masses["radii"] - 50))]
        M_total_50 = M_bulge_50 + M_disk_50

        if M_total_50 > 0:
            stats["bulge_fraction"] = M_bulge_50 / M_total_50
            stats["disk_fraction"] = M_disk_50 / M_total_50
            stats["disk_to_total"] = M_disk_50 / M_total_50  # D/T ratio

    # Scale lengths
    if "disk" in results:
        summary = results["disk"].summary
        if summary is not None:
            if "R_d" in summary:
                stats["disk_scale_length"] = summary["R_d"]
            elif "R_d1" in summary:
                stats["disk_scale_length"] = summary["R_d1"]
                if "R_d2" in summary:
                    stats["thick_disk_scale_length"] = summary["R_d2"]

            if "z_0" in summary:
                stats["disk_scale_height"] = summary["z_0"]

    if "bulge" in results:
        summary = results["bulge"].summary
        if summary is not None and "r_s" in summary:
            stats["bulge_scale_radius"] = summary["r_s"]

    # Halo concentration (c = r_vir / r_s)
    if "halo" in results:
        summary = results["halo"].summary
        if summary is not None and "r_s" in summary and metadata_dict is not None:
            stats["halo_scale_radius"] = summary["r_s"]
            R200 = metadata_dict["R200_kpc"]  # No fallback needed
            stats["halo_concentration"] = (
                R200 / summary["r_s"] if summary["r_s"] > 0 else np.nan
            )

    # Goodness of fit summary
    chi2_values = []
    for comp_name in ["bulge", "disk", "halo"]:
        if comp_name in results:
            best_model = results[comp_name].best_model
            table = results[comp_name].model_comparison.get("table", {})
            if best_model and best_model in table:
                chi2_values.append(table[best_model]["chi2_reduced"])

    if chi2_values:
        stats["mean_chi2_reduced"] = np.mean(chi2_values)

    return stats


def generate_component_summary(
    results: Dict[str, "FittingResult"],
    component_type=None,
    model_masses=None,
    metadata_dict: Optional[Dict] = None,
):
    """Generate and log summary of fitting results with statistics."""
    from .filters_and_utils import profile_eq_str

    summary_lines = []
    summary_lines.append("=" * 70)
    summary_lines.append("SUMMARY OF BEST-FIT MODELS")
    summary_lines.append("=" * 70)

    for component_name in ["bulge", "disk", "disk_subtracted_bulge", "halo"]:
        if component_name in results:
            comp = results[component_name]
            best_model = comp.best_model
            best_summary = comp.summary
            table = comp.model_comparison.get("table", {})

            # Skip if component is disabled or has no valid fit
            if not best_model or not best_summary:
                continue

            summary_lines.append("")
            # Use a cleaner display name for disk_subtracted_bulge
            display_name = (
                "DISK-SUBTRACTED BULGE"
                if component_name == "disk_subtracted_bulge"
                else component_name.upper()
            )
            summary_lines.append(f"{display_name} COMPONENT:")
            summary_lines.append(f"Model: {best_model}")
            summary_lines.append(profile_eq_str(best_model, best_summary))

            if best_model in table:
                summary_lines.append(f"χ² = {table[best_model]['chi2']:.2f}")
                summary_lines.append(
                    f"Reduced χ² = {table[best_model]['chi2_reduced']:.3f}"
                )

    # Add derived statistics if available
    if component_type and model_masses:
        stats = calculate_component_statistics(
            results, model_masses, component_type, metadata_dict
        )
        if stats:
            summary_lines.append("")
            summary_lines.append("DERIVED STATISTICS:")
            summary_lines.append("-" * 70)

            if "disk_to_total" in stats:
                summary_lines.append(
                    f"Disk-to-Total (D/T) ratio: {stats['disk_to_total']:.3f}"
                )
            if "bulge_fraction" in stats:
                summary_lines.append(f"Bulge fraction: {stats['bulge_fraction']:.3f}")
                summary_lines.append(f"Disk fraction: {stats['disk_fraction']:.3f}")
            if "disk_scale_length" in stats:
                summary_lines.append(
                    f"Disk scale length: {stats['disk_scale_length']:.2f} kpc"
                )
            if "disk_scale_height" in stats:
                summary_lines.append(
                    f"Disk scale height: {stats['disk_scale_height']:.2f} kpc"
                )
            if "halo_concentration" in stats:
                summary_lines.append(
                    f"Halo concentration (c): {stats['halo_concentration']:.2f}"
                )
            if "mean_chi2_reduced" in stats:
                summary_lines.append(
                    f"Mean reduced χ²: {stats['mean_chi2_reduced']:.3f}"
                )

    summary_lines.append("=" * 70)
    logger.info("\n".join(summary_lines))


def save_evolution_data(
    evolution_data: Dict,
    halo_id: str,
    logger: logging.Logger,
    filename: Optional[str] = None,
    append: bool = True,
    experiment: Optional[str] = None,
):
    """
    Save (or append) time-series evolution data to HDF5.

    If append=True, this function appends new snapshots to an existing file
    without overwriting previous data. Intended to be called once per snapshot
    to keep memory bounded.

    Memory management features:
        - Gzip compression on all array datasets
        - Chunked storage for resizable datasets
        - Periodic flushing after each component
        - Explicit garbage collection between components

    Args:
        evolution_data: Dictionary with time-series results
        halo_id: Halo identifier
        logger: Logger instance
        filename: Optional filename to save as (default: "evolution_time_series.h5")
        append: Whether to append to existing file (default: True)
        experiment: Optional experiment name for organizing results
    """
    PATHS = ProjectPaths.create()

    if filename is None:
        filename = "evolution_time_series.h5"
    elif not filename.endswith(".h5"):
        filename = f"{filename}.h5"

    output_dir = PATHS.halo_results(halo_id, experiment=experiment)
    output_file = output_dir / filename

    # Validate required keys
    required_keys = ["halo_id", "snapshots", "redshifts", "times_Gyr", "components"]
    missing_keys = [key for key in required_keys if key not in evolution_data]
    if missing_keys:
        raise ValueError(f"Missing required keys in evolution_data: {missing_keys}")

    mode = "a" if append and output_file.exists() else "w"
    logger.info(f"Saving evolution data to {output_file} (mode='{mode}')")

    with h5py.File(output_file, mode) as f:
       # ------------------------------------------------------------------
        # Global metadata (safe to overwrite / update)
        # ------------------------------------------------------------------
        f.attrs["halo_id"] = evolution_data["halo_id"]
        f.attrs["n_snapshots"] = len(evolution_data["snapshots"])

        # ------------------------------------------------------------------
        # TIME AXIS (append-safe, resizable)
        # ------------------------------------------------------------------
        # HDF5 compression settings for memory efficiency
        compression_opts = {"compression": "gzip", "compression_opts": 4}

        if "time" not in f:
            time_grp = f.create_group("time")
            time_grp.create_dataset(
                "snapshots",
                data=np.array(evolution_data['snapshots']),
                maxshape=(None,),
                chunks=True,
            )
            time_grp.create_dataset(
                "redshifts",
                data=np.array(evolution_data['redshifts']),
                maxshape=(None,),
                chunks=True,
            )
            time_grp.create_dataset(
                "times_Gyr",
                data=np.array(evolution_data["times_Gyr"]),
                maxshape=(None,),
                chunks=True,
            )
            start_idx = 0
        else:
            time_grp = f["time"]
            snapshots_ds = time_grp["snapshots"]
            redshifts_ds = time_grp["redshifts"]
            times_ds = time_grp["times_Gyr"]

            # Check if datasets are chunked (required for resizing)
            # Legacy files may not have chunked datasets
            if snapshots_ds.chunks is None:
                logger.warning("Existing time datasets are not chunked. Recreating with chunking enabled.")
                # Read existing data
                old_snapshots = snapshots_ds[:]
                old_redshifts = redshifts_ds[:]
                old_times = times_ds[:]

                # Delete old datasets
                del time_grp["snapshots"]
                del time_grp["redshifts"]
                del time_grp["times_Gyr"]

                # Recreate with chunking
                time_grp.create_dataset(
                    "snapshots",
                    data=old_snapshots,
                    maxshape=(None,),
                    chunks=True,
                )
                time_grp.create_dataset(
                    "redshifts",
                    data=old_redshifts,
                    maxshape=(None,),
                    chunks=True,
                )
                time_grp.create_dataset(
                    "times_Gyr",
                    data=old_times,
                    maxshape=(None,),
                    chunks=True,
                )

                # Get fresh references
                snapshots_ds = time_grp["snapshots"]
                redshifts_ds = time_grp["redshifts"]
                times_ds = time_grp["times_Gyr"]

            start_idx = snapshots_ds.shape[0]

            # Append new entries
            n_new = len(evolution_data["snapshots"])
            if n_new > 0:
                snapshots_ds.resize(start_idx + n_new, axis=0)
                redshifts_ds.resize(start_idx + n_new, axis=0)
                times_ds.resize(start_idx + n_new, axis=0)

                snapshots_ds[start_idx:] = evolution_data["snapshots"]
                redshifts_ds[start_idx:] = evolution_data["redshifts"]
                times_ds[start_idx:] = evolution_data["times_Gyr"]

        # Update total snapshot count
        f.attrs['n_snapshots'] = f['time']['snapshots'].shape[0]

        # ------------------------------------------------------------------
        # COMPONENT DATA
        # ------------------------------------------------------------------
        for comp_name, comp_data in evolution_data["components"].items():
            comp_grp = f.require_group(comp_name)

            # -----------------------------
            # FITTED PARAMETERS
            # -----------------------------
            for i, result in enumerate(comp_data["fitted_params"]):
                if result is None:
                    logger.debug(f"Skipping snapshot {i} for {comp_name} (no fitting results)")
                    continue

                snap_idx = start_idx + i
                snap_grp = comp_grp.require_group(f"snapshot_{snap_idx:04d}")

                # Save each subcomponent (bulge, disk, halo)
                for subcomp_name, subcomp_result in result.items():
                    if subcomp_name not in ["bulge", "disk", "halo"]:
                        continue

                    # Support both dict-like results and FittingResult objects
                    if hasattr(subcomp_result, "get"):
                        # dict-like interface (legacy)
                        best_model = subcomp_result.get("best_model")
                        best_summary = subcomp_result.get("best_summary")
                        table = subcomp_result.get("table", {})
                    else:
                        # dataclass/object interface
                        best_model = getattr(subcomp_result, "best_model", None)
                        # `FittingResult.summary` holds the parameter dict
                        best_summary = getattr(subcomp_result, "summary", None)
                        mcomp = getattr(subcomp_result, "model_comparison", {})
                        table = mcomp.get("table", {}) if isinstance(mcomp, dict) else {}

                    if not best_model or not best_summary:
                        continue

                    sub_grp = snap_grp.require_group(subcomp_name)
                    sub_grp.attrs["best_model"] = best_model

                    # Save fitted parameters (handle both scalars and arrays)
                    for param_name, param_value in best_summary.items():
                        # Skip non-serializable items
                        if param_name in ["samples", "sampler"]:
                            continue
                        try:
                            # Scalars and small arrays go in attrs
                            if np.isscalar(param_value) or isinstance(
                                param_value, (int, float, str, np.number)
                            ):
                                sub_grp.attrs[param_name] = param_value
                            # Arrays go in datasets
                            elif isinstance(param_value, np.ndarray):
                                sub_grp.create_dataset(
                                    param_name, data=param_value, **compression_opts
                                )
                            else:
                                logger.warning(
                                    f"Cannot save parameter {param_name} of type {type(param_value)}"
                                )
                        except (TypeError, ValueError) as e:
                            logger.warning(
                                f"Failed to save parameter {param_name}: {e}"
                            )

                    # Save chi2 statistics (table already computed above)
                    if best_model in table:
                        sub_grp.attrs["chi2"] = table[best_model]["chi2"]
                        sub_grp.attrs["chi2_reduced"] = table[best_model][
                            "chi2_reduced"
                        ]

            # -----------------------------
            # MASS EVOLUTION
            # -----------------------------
            mass_grp = comp_grp.require_group("mass_evolution")
            masses = comp_data.get("masses", [])
            for i, mass_data in enumerate(masses):
                if mass_data is None:
                    logger.debug(
                        f"Skipping snapshot {i} for {comp_name} masses (no mass data)"
                    )
                    continue
                snap_idx = start_idx + i
                snap_grp = mass_grp.require_group(f"snapshot_{snap_idx:04d}")

                for key in ("radii", "bulge", "disk", "halo", "total"):
                    if key in snap_grp:
                        del snap_grp[key]
                    snap_grp.create_dataset(
                        key, data=mass_data[key], **compression_opts
                    )

            # Flush to disk and free memory after each component
            f.flush()
            del masses
            gc.collect()

        # Final cleanup
        f.flush()

    gc.collect()

    # Log summary
    n_snapshots = len(evolution_data["snapshots"])
    components = list(evolution_data["components"].keys())
    logger.info(
        f"Evolution data saved successfully: {n_snapshots} snapshots, "
        f"{len(components)} components ({', '.join(components)})"
    )
