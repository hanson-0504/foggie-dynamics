import logging
from multiprocessing import Pool
from typing import Callable, Dict, List, Optional, Tuple

import emcee
import numpy as np

from .analysis import gelman_rubin_diagnostic
from .models import (
    cored_power_model,
    disk_density,
    double_disk_flaring,
    double_exp_model,
    exp_cutoff,
    hernquist_density,
    nfw_density,
    only_core,
    single_disk_flaring,
    three_power_model,
    two_power_model,
)


def fit_general_emcee(
    coords: Tuple[np.ndarray, ...],
    data: np.ndarray,
    model_func: Callable,
    param_names: List[str],
    bounds: Dict[str, Tuple[float, float]],
    logger: logging.Logger,
    p0: Optional[Dict[str, float]] = None,
    data_range: Optional[Dict[str, Tuple[float, float]]] = None,
    weights: Optional[np.ndarray] = None,
    use_log_space: bool = True,
    soft_priors: Optional[Callable] = None,
    nwalkers: int = 64,
    nsteps: int = 4000,
    nburn: int = 1000,
    thin: int | str = "auto",
    verbose: bool = True,
    uncertainties: Optional[np.ndarray] = None,
    fixed_uncertainty: float = 0.1,
    min_points: int = 10,
    auto_expand_range: bool = True,
    max_range_expansions: int = 3,
):
    """
    General MCMC fitting function using emcee.

    Parameters
    ----------
    coords : tuple of arrays
        Coordinate arrays (e.g., (r,) for 1D, (R, z) for 2D).
    data : array
        Observed data values to fit.
    model_func : callable
        Model function: model_func(*coords, *params) -> model_data.
        Should accept coordinates followed by parameter values.
    param_names : list of str
        Names of parameters in order expected by model_func.
    bounds : dict
        Parameter bounds: {param_name: (min, max), ...}.
        Must include 'log_sigma' for scatter parameter.
    p0 : dict, optional
        Initial parameter guesses: {param_name: value, ...}.
        If None, uses midpoint of bounds.
    data_range : dict, optional
        Coordinate ranges for fitting: {coord_name: (min, max), ...}.
        If None, uses all data.
    weights : array, optional
        Weights for each data point. Default: uniform weights.
    use_log_space : bool, optional
        If True, fit in log10 space. Default: True.
    soft_priors : callable, optional
        Function soft_priors(param_dict) -> log_prior_contrib.
        Adds Gaussian penalties to prior. Default: None.
    nwalkers : int, optional
        Number of MCMC walkers. Default: 64.
    nsteps : int, optional
        Total MCMC steps. Default: 4000.
    nburn : int, optional
        Burn-in steps to discard. Default: 1000.
    thin : int or 'auto', optional
        Thinning factor for samples. If 'auto' or None, automatically
        determined from autocorrelation time (~0.5 * tau_max).
        Default: 'auto'.
    verbose : bool, optional
        Print diagnostics. Default: True.
    uncertainties : array, optional
        Measured uncertainties (sigma) for each data point.
        If None, uses fixed fractional uncertainty.
        If provided, uses: sigma_total = sqrt(uncertainties^2 + sigma_extra^2).
        Must be in the same units as data (not log-space).
        Example: For density profiles, use standard error of the mean (SEM) or
        standard deviation from profile statistics
    fixed_uncertainty : float, optional
        Fallback incase uncertainties is none, default = 0.1

    Returns
    -------
    flat_samples : ndarray
        Posterior samples after burn-in and thinning.
    summary : dict
        Contains median values, uncertainties, and diagnostics.

    Examples
    --------
    # 1D spherical fit
    def nfw_model(r, rho0, rs):
        return rho0 / ((r/rs) * (1 + r/rs)**2)

    samples, summary = fit_general_emcee(
        coords=(r,),
        data=rho,
        model_func=nfw_model,
        param_names=['rho0', 'rs'],
        bounds={'rho0': (1e6, 1e10), 'rs': (10, 100), 'log_sigma': (-10, 1)},
        data_range={'r': (10, 200)}
    )

    # 2D disk fit
    def disk_model(R, z, rho0, Rd, z0):
        return rho0 * np.exp(-R/Rd) * np.exp(-np.abs(z)/z0)

    samples, summary = fit_general_emcee(
        coords=(R, z),
        data=rho,
        model_func=disk_model,
        param_names=['rho0', 'Rd', 'z0'],
        bounds={'rho0': (1e6, 1e10), 'Rd': (2, 15), 'z0': (0.1, 2), 'log_sigma': (-10, 1)},
        data_range={'R': (2, 10), 'z': (0, 1)}
    )
    """
    # -------------------------------------------------
    # Validate inputs
    # -------------------------------------------------
    if "log_sigma" not in bounds:
        logger.error("bounds must include 'log_sigma' for scatter parameter")
        return None, None

    model_param_names = [p for p in param_names if p != "log_sigma"]
    ndim = len(param_names) + 1  # +1 for log_sigma

    # -------------------------------------------------
    # Apply data range mask
    # -------------------------------------------------
    # Start with all valid, finite, positive data points
    def make_mask(range_dict: Optional[Dict[str, Tuple[float, float]]]) -> np.ndarray:
        m = np.isfinite(data) & (data > 0)
        if range_dict is not None:
            coord_names_local = list(range_dict.keys())
            for i, coord_name in enumerate(coord_names_local):
                coord_min, coord_max = range_dict[coord_name]
                m &= (coords[i] >= coord_min) & (coords[i] < coord_max)
        return m

    # Initial mask and selection
    mask = make_mask(data_range)
    coords_fit = tuple(c[mask] for c in coords)
    data_fit = data[mask]

    # If too few points, optionally try expanding the requested data_range
    if len(data_fit) < min_points and data_range is not None and auto_expand_range:
        # Work on a copy of the ranges so we don't mutate caller state
        expanded_range: Dict[str, Tuple[float, float]] = {
            k: (float(v[0]), float(v[1])) for k, v in data_range.items()
        }
        expansions = 0
        while len(data_fit) < min_points and expansions < max_range_expansions:
            # Expand each coordinate range around its center by a factor of ~2
            for k in list(expanded_range.keys()):
                mn, mx = expanded_range[k]
                center = 0.5 * (mn + mx)
                half = 0.5 * (mx - mn)
                # Increase half-width; cap at actual data min/max
                new_half = half * 2.0
                coord_idx = list(data_range.keys()).index(k)
                data_min = float(np.min(coords[coord_idx]))
                data_max = float(np.max(coords[coord_idx]))
                new_mn = max(data_min, center - new_half)
                new_mx = min(data_max, center + new_half)
                expanded_range[k] = (float(new_mn), float(new_mx))

            # Recompute mask/selection with expanded ranges
            mask = make_mask(expanded_range)
            coords_fit = tuple(c[mask] for c in coords)
            data_fit = data[mask]
            expansions += 1

        if len(data_fit) >= min_points:
            logger.info(
                f"Expanded data_range {expansions} time(s) to include {len(data_fit)} points."
            )
            # Replace data_range locally so downstream diagnostics report the actual used range
            data_range = expanded_range

    # Normalize weights to maintain effective sample size
    if weights is None:
        weights_fit = np.ones_like(data_fit)
    else:
        weights_fit = weights[mask]
        # Normalize so weights sum to N (preserves chi-squared statistics)
        weights_fit = weights_fit / np.sum(weights_fit) * len(weights_fit)

    # Apply mask to uncertainties
    if uncertainties is not None:
        uncertainties_fit = uncertainties[mask]
        # check for invalid uncertainties
        if np.any(uncertainties_fit <= 0) or np.any(~np.isfinite(uncertainties_fit)):
            logger.warning(
                "Some uncertainties are non-positive or non-finite, using fallback 10%"
            )
            uncertainties_fit = None
    else:
        uncertainties_fit = None

    if len(data_fit) < min_points:
        total_points = len(data)
        msg = (
            f"Not enough valid points in fitting region: {len(data_fit)} "
            f"(total available={total_points})."
        )
        if data_range is not None:
            msg += f" data_range={data_range}."

        # Add a short preview of the coordinates and data to help debugging
        try:
            coord_names = list(data_range.keys()) if data_range is not None else [f"coord{i}" for i in range(len(coords_fit))]
            previews = []
            for i, name in enumerate(coord_names):
                vals = coords_fit[i][:10]
                previews.append(f"{name}={np.array2string(vals, threshold=5)}")
            msg += " coords_preview=[" + ", ".join(previews) + "]"
        except Exception:
            # If previewing fails for any reason, don't block the error
            pass
        logger.error(msg)
        return None, None

    # -------------------------------------------------
    # Set initial guesses
    # -------------------------------------------------
    if p0 is None:
        p0 = {}
        for pname in model_param_names:
            if pname in bounds:
                p0[pname] = 0.5 * (bounds[pname][0] + bounds[pname][1])

    initial_params = [p0[pname] for pname in model_param_names] + [np.log(0.05)]
    initial = np.array(initial_params)

    # -------------------------------------------------
    # Define log_prior, log_likelihood, log_posterior
    # -------------------------------------------------
    def log_prior(theta):
        # Hard bounds check
        for i, pname in enumerate(model_param_names):
            if not (bounds[pname][0] < theta[i] < bounds[pname][1]):
                return -np.inf

        log_sigma = theta[-1]
        if not (bounds["log_sigma"][0] < log_sigma < bounds["log_sigma"][1]):
            return -np.inf

        # Soft priors (optional)
        if soft_priors is not None:
            param_dict = {pname: theta[i] for i, pname in enumerate(model_param_names)}
            param_dict["log_sigma"] = log_sigma
            return soft_priors(param_dict)

        return 0.0

    def log_likelihood(theta):
        params = theta[:-1]  # Model parameters
        log_sigma = theta[-1]  # Intrinsic scatter parameter
        sigma_extra = np.exp(np.clip(log_sigma, -20, 20))

        # Evaluate model at data coordinates
        try:
            model = model_func(*coords_fit, *params)
        except (ValueError, TypeError, ArithmeticError) as e:
            logger.debug(f"Model Evaluation failed: {e}")
            return -np.inf

        # Reject unphysical models (negative or non-finite densities)
        if np.any(model <= 0) or np.any(~np.isfinite(model)):
            return -np.inf

        # Calculate chi-squared in appropriate space
        # Use measured uncertainties if available
        if uncertainties_fit is not None:
            # use measured uncertainties from profile statistics
            if use_log_space:
                # Fit in log10 space (better for density profiles spanning many orders)
                # For y = log10(x), sigma_y ~ sigma_x / (x*ln(10))
                log_data = np.log10(data_fit)
                log_model = np.log10(model)
                sigma_log_measured = uncertainties_fit / (data_fit * np.log(10))
                # Combine measured uncertainty with intrinsic scatter
                sigma_log = np.sqrt(sigma_log_measured**2 + sigma_extra**2)
                chi2 = ((log_data - log_model) / sigma_log) ** 2
            else:
                # Fit in linear space
                # Combine measured uncertainty with intrinsic scatter
                sigma_total = np.sqrt(uncertainties_fit**2 + sigma_extra**2)
                chi2 = ((data_fit - model) / sigma_total) ** 2
        else:
            # FALLBACK: use old fixed 10% if no uncertainties provided
            if use_log_space:
                log_data = np.log10(data_fit)
                log_model = np.log10(model)
                sigma_log = np.sqrt((fixed_uncertainty**2 + sigma_extra**2))
                chi2 = ((log_data - log_model) / sigma_log) ** 2
            else:
                sigma_total = np.sqrt((0.05 * data_fit) ** 2 + sigma_extra**2)
                chi2 = ((data_fit - model) / sigma_total) ** 2

        # Return weighted log-likelihood
        return -0.5 * np.sum(weights_fit * chi2)

    def log_posterior(theta):
        lp = log_prior(theta)
        if not np.isfinite(lp):
            return -np.inf
        ll = log_likelihood(theta)
        if not np.isfinite(ll):
            return -np.inf
        return lp + ll

    # -------------------------------------------------
    # Initialize walkers
    # -------------------------------------------------
    # Create small perturbations around initial guess (1% scale)
    scales = np.array(
        [0.01 * abs(initial[i]) if initial[i] != 0 else 0.01 for i in range(ndim)]
    )

    # Try to find valid starting positions (up to 100 attempts)
    # Valid means all walkers have finite log-posterior
    pos = None
    for _ in range(100):
        pos = initial + scales * np.random.randn(nwalkers, ndim)
        if all(np.isfinite(log_posterior(p)) for p in pos):
            break
    else:
        logger.error("Could not initialize valid walker positions")
        return None, None

    if pos is None:
        logger.error("Could not initialize valid walker positions")
        return None, None

    if verbose:
        logger.info(f"Initial log-posterior: {log_posterior(initial):.2f}")

    # -------------------------------------------------
    # Run MCMC
    # -------------------------------------------------
    sampler = emcee.EnsembleSampler(nwalkers, ndim, log_posterior)

    try:
        sampler.run_mcmc(pos, nsteps, progress=False)
    except Exception:
        logger.exception("MCMC sampling failed unexpectedly")
        return None, None

    # -------------------------------------------------
    # Diagnostics
    # -------------------------------------------------
    tau_max = None
    n_eff = None
    if verbose:
        try:
            tau = sampler.get_autocorr_time()
            tau_max = np.max(tau)
            logger.info(f"Autocorrelation time: {tau}")
            # effective sample size n_eff
            n_eff = sampler.get_chain(flat=True).shape[0] / tau
            if nsteps < 50 * tau_max:
                logger.warning(
                    f"Chain may not be converged. Need ~{50 * tau_max:.0f} steps"
                )
        except emcee.autocorr.AutocorrError as e:
            logger.warning(
                f"Chain too short to estimate autocorrelation time: {e}", exc_info=True
            )
            tau_max = None
            n_eff = None

        acceptance = np.mean(sampler.acceptance_fraction)
        logger.info(f"Mean acceptance fraction: {acceptance:.3f}")

        if acceptance < 0.15:
            logger.warning("Low acceptance fraction.")
        elif acceptance > 0.6:
            logger.warning("High acceptance fraction.")

    # Gelman-Rubin diagnostic (R-hat: should be < 1.1 for convergence)
    all_param_names = model_param_names + ["log_sigma"]
    rhat = gelman_rubin_diagnostic(
        sampler, nburn, param_names=all_param_names, verbose=verbose
    )

    # -------------------------------------------------
    # Determine optimal thinning
    # -------------------------------------------------
    # Thinning removes correlated samples to get independent posterior draws
    if thin is None or thin == "auto":
        if tau_max is not None and np.isfinite(tau_max):
            # Thin by about half the autocorrelation time (conservative choice)
            # This ensures samples are approximately independent
            thin = max(1, int(0.5 * tau_max))
            if verbose:
                logger.info(
                    f"Auto-thinning: using thin={thin} (based on tau_max={tau_max:.1f})"
                )
        else:
            # Fallback if autocorrelation couldn't be computed
            thin = 10
            if verbose:
                logger.info(
                    f"Auto-thinning: using default thin={thin} (tau unavailable)"
                )

    # -------------------------------------------------
    # Extract and summarize samples
    # -------------------------------------------------
    flat_samples = sampler.get_chain(discard=nburn, thin=thin, flat=True)

    if flat_samples is not None:
        if len(flat_samples) < 100 and verbose:
            logger.warning(
                f"Only {len(flat_samples)} samples after burn-in and thinning."
            )

        medians = np.median(flat_samples, axis=0)

        summary = {
            "samples": flat_samples,
            "acceptance_fraction": np.mean(sampler.acceptance_fraction),
            # 'sampler': sampler,
            "rhat": rhat,
            "n_eff": n_eff,
            "_param_names": model_param_names + ["log_sigma"],
        }

        # Add individual parameter summaries
        for i, pname in enumerate(model_param_names):
            summary[pname] = medians[i]
            summary[f"{pname}_err"] = np.percentile(flat_samples[:, i], [16, 84])

        summary["sigma"] = np.exp(medians[-1])

        # Standardize parameter names for backward compatibility
        # This ensures models can access params with or without underscores
        # and automatically converts log10_X parameters to linear scale
        for pname in list(summary.keys()):
            if pname.startswith("log10_"):
                # Add linear version: log10_rho0 → rho0 = 10^(log10_rho0)
                base_name = pname.replace("log10_", "")
                summary[base_name] = 10 ** summary[pname]
                # Also convert error bars to linear scale
                if f"{pname}_err" in summary:
                    summary[f"{base_name}_err"] = 10 ** summary[f"{pname}_err"]

            # Handle underscore variations for flexible parameter access
            # Examples: rho_0 → rho0, r_s → rs, R_d1 → Rd1
            if "_" in pname and not pname.endswith("_err"):
                no_underscore = pname.replace("_", "")
                if no_underscore not in summary:
                    summary[no_underscore] = summary[pname]
                    if f"{pname}_err" in summary:
                        summary[f"{no_underscore}_err"] = summary[f"{pname}_err"]

        return flat_samples, summary
    else:
        logger.error("MCMC sampling failed")
        return None, None


def fit_bulge_emcee(
    r: np.ndarray,
    rho: np.ndarray,
    logger: logging.Logger,
    r_range: Tuple[float, float] = (0.3, 3.0),
    bounds: Optional[Dict] = None,
    p0: Optional[Dict] = None,
    weights: Optional[np.ndarray] = None,
    uncertainties: Optional[np.ndarray] = None,
    soft_priors: Optional[Callable] = None,
    **kwargs,
):
    """
    Fit a Hernquist bulge profile using emcee MCMC sampler.

    Parameters
    ----------
    r : array_like
        Radial positions in kpc.
    rho : array_like
        Density values in Msun/kpc^3.
    r_range : tuple of float, optional
        (min, max) radial range to fit within, in kpc. Default: (0.1, 2.0).
    bounds : dict, optional
        Parameter bounds: {'log10_rho0': (min, max), 'log10_rs': (min, max),
                          'log_sigma': (min, max)}.
        Default: {'log10_rho0': (6, 12), 'log10_rs': (-1.3, 0.7),
                  'log_sigma': (-10, 1)}.
    p0 : dict, optional
        Parameter guesses: {'log10_rho0': log10(rho0), 'log10_rs': log10(a)}
        Default: {'log10_rho0': log10(median(rho)), 'log10_rs': log10(0.5)}
    weights : array_like, optional
        Weights for data points.
    uncertainties : array_like, optional
        Uncertainties on data points.
    soft_priors : callable, optional
        Function soft_priors(param_dict) -> log_prior_contrib.
        Adds Gaussian penalties to prior. Default: None.
    **kwargs
        Additional arguments passed to fit_general_emcee (nwalkers, nsteps,
        nburn, thin, verbose, etc.).

    Returns
    -------
    flat_samples : ndarray
        Posterior samples with columns [log10_rho0, log10_rs, log_sigma].
    summary : dict
        Dictionary containing fitted parameters and samples.
    """
    param_names = ["log10_rho0", "log10_rs"]
    # -------------------------------------------------
    # Set default bounds if not provided
    # -------------------------------------------------
    if bounds is None:
        bounds = {
            "log10_rho0": (6, 12),
            "log10_rs": (-1.3, 0.7),  # 0.05 to 5.0 kpc in log10 space
            "log_sigma": (-10, 1),
        }

    data_range = {"r": r_range}

    def log_hernquist(r, log10_rho0, log10_rs):
        return hernquist_density(r, 10**log10_rho0, 10**log10_rs)

    return fit_general_emcee(
        coords=(r,),
        data=rho,
        model_func=log_hernquist,
        param_names=param_names,
        bounds=bounds,
        data_range=data_range,
        logger=logger,
        p0=p0,
        weights=weights,
        uncertainties=uncertainties,
        soft_priors=soft_priors,
        **kwargs,
    )


def fit_disk_emcee(
    R: np.ndarray,
    z: np.ndarray,
    rho: np.ndarray,
    R_range: Tuple[float, float],
    z_max: float,
    logger: logging.Logger,
    p0: Optional[Dict] = None,
    bounds: Optional[Dict] = None,
    weights: Optional[np.ndarray] = None,
    uncertainties: Optional[np.ndarray] = None,
    soft_priors: Optional[Callable] = None,
    **kwargs,
):
    """
    Fit an exponential disk using emcee MCMC sampler.

    Parameters
    ----------
    R, z : array (kpc)
        Cylindrical coordinates of data points.
    rho : array (Msun/kpc^3)
        Corresponding density values.
    R_range : tuple
        (R_min, R_max) radial limits for fitting region.
    z_max : float
        Vertical limit for fitting region (default 1.0 kpc).
    bounds : dict, optional
        Parameter bounds: {'log10_rho0': (min, max), 'R_d': (min, max),
                          'z_0': (min, max), 'log_sigma': (min, max)}.
        Default: {'log10_rho0': (6, 12), 'R_d': (2, 15),
                  'z_0': (0.01, 2.0), 'log_sigma': (-10, 1)}.
    p0 : dict, optional
        Parameter guesses: {'log10_rho0': log10_rho0, 'R_d': R_d,
                            'z_0': z_0}
        Default: {'log10_rho0': log10(median(rho)), 'R_d': 7,
                  'z_0': 0.5}
    weights : array_like, optional
        Weights for each data point. Higher weight = more influence.
        Default: None (uniform weights).
    uncertainties : array, optional

    soft_priors : callable, optional
        function soft_priors(param_dict) -> log_prior_contrib.
        Adds Gaussian penalties to prior.
    **kwargs
        Additional arguments passed to fit_general_emcee (nwalkers, nsteps
        nburn, thin, verbose, etc.).

    Returns
    -------
    samples : ndarray
        Posterior samples [log10(rho_0), R_d, z_0, log_sigma].
    summary : dict
        Contains median and percentile values for each parameter.
    """
    # -------------------------------------------------
    # Set default bounds if not provided
    # -------------------------------------------------
    param_names = ["log10_rho01", "R_d1", "z_0"]

    if bounds is None:
        bounds = {
            "log10_rho01": (6, 12),
            "R_d1": (2, 15),
            "z_0": (0.01, 2.0),
            "log_sigma": (-10, 1),
        }

    data_range = {"R": R_range, "z": (0.3, z_max)}

    def log_disk_density(R, z, log10_rho0, Rd, z0):
        return disk_density(R, z, 10**log10_rho0, Rd, z0)

    return fit_general_emcee(
        coords=(R, z),
        data=rho,
        model_func=log_disk_density,
        param_names=param_names,
        bounds=bounds,
        weights=weights,
        uncertainties=uncertainties,
        p0=p0,
        soft_priors=soft_priors,
        data_range=data_range,
        logger=logger,
        **kwargs,
    )


def fit_halo_emcee(
    r: np.ndarray,
    rho: np.ndarray,
    logger: logging.Logger,
    r_range: Tuple[float, float] = (10, 200),
    bounds: Optional[Dict] = None,
    p0: Optional[Dict] = None,
    weights: Optional[np.ndarray] = None,
    uncertainties: Optional[np.ndarray] = None,
    soft_priors: Optional[Callable] = None,
    **kwargs,
):
    """
    Fit a NFW Halo profile using emcee MCMC sampler.

    Parameters
    ----------
    r : array_like
        Radial positions in kpc.
    rho : array_like
        Density values in Msun/kpc^3.
    r_range : tuple of float, optional
        (min, max) radial range to fit within, in kpc. Default: (10, 200).
    bounds : dict, optional
        Parameter bounds: {'log10_rho0': (min, max), 'log10_rs': (min, max),
                          'log_sigma': (min, max)}.
        Default: {'log10_rho0': (4, 10), 'log10_rs': (0.7, 2),
                  'log_sigma': (-10, 1)}.
    p0 : dict, optional
        Initial guesses for log10_rho0 and log10_r_s
        Default: {'log10_rho0': log10(r_mid * 4), 'rs': log10(r_mid)}.
    weights : array_like, optional
        Weights for data points.
    uncertainties : array_like, optional
        Uncertainties on data points.
    soft_priors : callable, optional
        Function soft_priors(param_dict) -> log_prior_contrib.
        Adds Gaussian penalties to prior. Default: None.
    **kwargs
        Additional arguments passed to fit_general_emcee (nwalkers, nsteps,
        nburn, thin, verbose, etc.).

    Returns
    -------
    flat_samples : ndarray, shape (n_samples, 3)
        Posterior samples with columns [log10_rho0, rs, log_sigma],
        after burn-in removal and thinning by factor of 10.
    summary : dict
        Dictionary containing:
        - 'rho_0': median rho_0 in Msun/kpc^3
        - 'r_s': median scale radius in kpc
        - 'sigma': median extra scatter parameter
        - 'samples': reference to flat_samples array
        - 'acceptance_fraction': mean acceptance fraction
        - 'sampler': reference to the emcee sampler object
    """
    # -------------------------------------------------
    # Set default bounds if not provided
    # -------------------------------------------------
    if bounds is None:
        bounds = {
            "log10_rho0": (1, 10),
            "log10_rs": (1.0, 1.9),
            "log_sigma": (-10, 1),
        }

    param_names = ["log10_rho0", "log10_rs"]

    data_range = {"r": r_range}

    def nfw_density_r(r, log10_rho0, log10_rs):
        rho0 = 10**log10_rho0
        rs = 10**log10_rs
        return nfw_density(r, rho0, rs)

    return fit_general_emcee(
        coords=(r,),
        data=rho,
        model_func=nfw_density_r,
        param_names=param_names,
        bounds=bounds,
        data_range=data_range,
        weights=weights,
        uncertainties=uncertainties,
        soft_priors=soft_priors,
        logger=logger,
        p0=p0,
        **kwargs,
    )


def fit_two_power_emcee(
    r: np.ndarray,
    rho: np.ndarray,
    logger: logging.Logger,
    r_range: Tuple[float, float] = (10, 200),
    bounds: Optional[Dict] = None,
    p0: Optional[Dict] = None,
    weights: Optional[np.ndarray] = None,
    uncertainties: Optional[np.ndarray] = None,
    soft_priors: Optional[Callable] = None,
    **kwargs,
):
    """
    Fit a 2 power profile using emcee MCMC sampler.

    Parameters
    ----------
    r : array_like
        Radial positions in kpc.
    rho : array_like
        Density values in Msun/kpc^3.
    r_range : tuple of float
        (min, max) radial range to fit within, in kpc.
    bounds : dict, optional
        Parameter bounds: {'log10_rho0': (min, max), 'log10_rs': (min, max),
                          'alpha': (min, max), 'beta': (min, max),
                          'log_sigma': (min, max)}.
        Default: {'log10_rho0': (1, 10), 'log10_rs': (1.0, 1.9),
                  'alpha': (0, 4), 'beta': (0, 5), 'log_sigma': (-10, 1)}.
    p0 : dict, optional
        Initial guesses: {'log10_rho0': log10_rho0, 'log10_rs': log10_rs,
                         'alpha': alpha, 'beta': beta}
        Default: {'log10_rho0': log10(r_mid * 4), 'log10_rs': log10(r_mid),
                  'alpha': 1, 'beta': 3}.
    weights : array_like, optional
        Weights for each data point. Higher weight = more influence.
        Default: None (uniform weights).
    uncertainties : array_like, optional
        Uncertainties on data points.
    soft_priors : callable, optional
        Function soft_priors(param_dict) -> log_prior_contrib.
        Adds Gaussian penalties to prior. Default: None.
    **kwargs
        Additional arguments passed to fit_general_emcee (nwalkers, nsteps,
        nburn, thin, verbose, etc.).

    Returns
    -------
    flat_samples : ndarray, shape (n_samples, 5)
        Posterior samples with columns [log10_rho0, log10_rs, alpha, beta, log_sigma],
        after burn-in removal and thinning by factor of 10.
    summary : dict
        Dictionary containing:
        - 'rho_0': median rho_0 in Msun/kpc^3
        - 'r_s': median scale radius in kpc
        - 'alpha': inner power-law slope when r << rs model behaves like r^-alpha
        - 'beta': outer power-law slope when r >> rs model behaves like r^-beta
        - 'sigma': median extra scatter parameter
        - 'samples': reference to flat_samples array
        - 'acceptance_fraction': mean acceptance fraction
        - 'sampler': reference to the emcee sampler object
    """
    # -------------------------------------------------
    # Set default bounds if not provided
    # -------------------------------------------------
    if bounds is None:
        bounds = {
            "log10_rho0": (1, 10),
            "log10_rs": (1.0, 1.9),
            "alpha": (0, 4),
            "beta": (0, 5),
            "log_sigma": (-10, 1),
        }

    param_names = ["log10_rho0", "log10_rs", "alpha", "beta"]

    data_range = {"r": r_range}

    def two_power(r, log10_rho0, log10_rs, alpha, beta):
        rho0 = 10**log10_rho0
        rs = 10**log10_rs
        return two_power_model(r, rho0, rs, alpha, beta)

    return fit_general_emcee(
        coords=(r,),
        data=rho,
        model_func=two_power,
        bounds=bounds,
        p0=p0,
        weights=weights,
        uncertainties=uncertainties,
        data_range=data_range,
        param_names=param_names,
        soft_priors=soft_priors,
        logger=logger,
        **kwargs,
    )


def fit_three_power_emcee(
    r: np.ndarray,
    rho: np.ndarray,
    logger: logging.Logger,
    r_range: Tuple[float, float] = (0.3, 5),
    bounds: Optional[Dict] = None,
    p0: Optional[Dict] = None,
    weights: Optional[np.ndarray] = None,
    uncertainties: Optional[np.ndarray] = None,
    soft_priors: Optional[Callable] = None,
    **kwargs,
):
    """
    Fit a 3 power profile using emcee MCMC sampler.

    Parameters
    ----------
    r : array_like
        Radial positions in kpc.
    rho : array_like
        Density values in Msun/kpc^3.
    r_range : tuple of float, optional
        (min, max) radial range to fit within, in kpc. Default: (10, 200).
    bounds : dict, optional
        Parameter bounds: {'log10_rho0': (min, max), 'log10_rs': (min, max),
                          'alpha': (min, max), 'beta': (min, max),
                          'gamma': (min, max), 'log_sigma': (min, max)}.
        Default: {'log10_rho0': (1, 10), 'log10_rs': (1.0, 1.9),
                  'alpha': (0, 2), 'beta': (3, 6), 'gamma': (0.1, 3)
                  'log_sigma': (-10, 1)}.
    p0 : dict, optional
        Initial guesses: {'log10_rho0': log10_rho0, 'log10_rs': log10_rs,
                         'alpha': alpha, 'beta': beta, 'gamma': gamma}
        Default: {'log10_rho0': log10(r_mid * 4), 'log10_rs': log10(r_mid),
                  'alpha': 1, 'beta': 4, 'gamma': 1}.
    weights : array_like, optional
        Weights for each data point. Higher weight = more influence.
        Default: None (uniform weights).
    uncertainties : array_like, optional
        Uncertainties on data points.
    soft_priors : callable, optional
        Function soft_priors(param_dict) -> log_prior_contrib.
        Adds Gaussian penalties to prior. Default: None.
    **kwargs
        Additional arguments passed to fit_general_emcee (nwalkers, nsteps,
        nburn, thin, verbose, etc.).

    Returns
    -------
    flat_samples : ndarray, shape (n_samples, 5)
        Posterior samples with columns [log10_rho0, log10_rs, alpha, beta, log_sigma],
        after burn-in removal and thinning by factor of 10.
    summary : dict
        Dictionary containing:
        - 'rho_0': median rho_0 in Msun/kpc^3
        - 'r_s': median scale radius in kpc
        - 'alpha': inner power-law slope when r << rs model behaves like r^-alpha
        - 'beta': outer power-law slope when r >> rs model behaves like r^-beta
        - 'gamma': transition sharpness when r ~ rs - controls how smooth the transition is.
        - 'sigma': median extra scatter parameter
        - 'samples': reference to flat_samples array
        - 'acceptance_fraction': mean acceptance fraction
        - 'sampler': reference to the emcee sampler object
    """
    # -------------------------------------------------
    # Set default bounds if not provided
    # -------------------------------------------------
    if bounds is None:
        bounds = {
            "log10_rho0": (1, 10),
            "log10_rs": (1.0, 1.9),
            "alpha": (0, 2),
            "beta": (3, 6),
            "gamma": (0.1, 3),
            "log_sigma": (-10, 1),
        }

    param_names = ["log10_rho0", "log10_rs", "alpha", "beta", "gamma"]

    data_range = {"r": r_range}

    def three_power(r, log10_rho0, log10_rs, alpha, beta, gamma):
        return three_power_model(r, 10**log10_rho0, 10**log10_rs, alpha, beta, gamma)

    return fit_general_emcee(
        coords=(r,),
        data=rho,
        model_func=three_power,
        param_names=param_names,
        bounds=bounds,
        p0=p0,
        data_range=data_range,
        weights=weights,
        uncertainties=uncertainties,
        soft_priors=soft_priors,
        logger=logger,
        **kwargs,
    )


def fit_double_exponential_disk_emcee(
    R: np.ndarray,
    z: np.ndarray,
    rho: np.ndarray,
    R_range: Tuple[float, float],
    z_max: float,
    logger: logging.Logger,
    bounds: Optional[Dict] = None,
    p0: Optional[Dict] = None,
    weights: Optional[np.ndarray] = None,
    uncertainties: Optional[np.ndarray] = None,
    soft_priors: Optional[Callable] = None,
    **kwargs,
):
    """
    Fit a double exponential radial disk profile using emcee MCMC sampler.

    rho(R, z) = rho_01 * exp(-R/R_d1) + rho_02 * exp(-R/R_d2) * sech^2(z / z0)

    Parameters
    ----------
    R : array (kpc)
        Cylindrical radii of data points
    rho : array (Msun/kpc^3)
        Corresponding density values (typically midplane densities)
    R_range : tuple
        (R_min, R_max) radial limits for fitting region
    bounds : dict, optional
        Parameter bounds: {
            'log10_rho01': (min, max),
            'log10_rho02': (min, max),
            'R_d1': (min, max),
            'R_d2': (min, max),
            'log10_z0': (min, max),
            'log_sigma': (min, max)
        }
        Default: {
            'log10_rho01': (5, 11),
            'log10_rho02': (5, 11),
            'R_d1': (0.5, 15),
            'R_d2': (0.5, 20),
            'log10_z0': (0.01, 3),
            'log_sigma': (-10, 1)
        }
    p0 : dict, optional
        Parameter guesses: {
            'log10_rho01': log10_rho01,
            'log10_rho02': log10_rho02,
            'R_d1': R_d1,
            'R_d2': R_d2,
            'log10_z0': log10_z0
        }
        Default: Estimated from data
    weights : array, optional
        Weights for each data point. Higher weight = more influence.
        Default: None (uniform weights)
    uncertainties : array_like, optional
        Uncertainties on data points.
    soft_priors : callable, optional
        Function soft_priors(param_dict) -> log_prior_contrib.
        Adds Gaussian penalties to prior. Default: None.
    **kwargs
        Additional arguments passed to fit_general_emcee (nwalkers, nsteps,
        nburn, thin, verbose, etc.).

    Returns
    -------
    samples : ndarray
        Posterior samples [log10_rho01, log10_rho02, R_d1, R_d2, z_0, log_sigma]
    summary : dict
        Contains median and percentile values for each parameter
    """
    # -------------------------------------------------
    # Set default bounds if not provided
    # -------------------------------------------------
    if bounds is None:
        bounds = {
            "log10_rho01": (0, 13),  # wide but finite
            "log10_rho02": (0, 13),
            "R_d1": (0.2, 12.0),
            "R_d2": (0.5, 40.0),  # allow outer disk but penalize in prior below
            "log10_z0": (
                -2.5,
                0.7,
            ),  # log10(z0/kpc) roughly [-2.5, 0.7] => z0 ~ [0.003, 5] kpc
            "log_sigma": (-10, 1),
        }

    param_names = ["log10_rho01", "log10_rho02", "R_d1", "R_d2", "log10_z0"]

    data_range = {"R": R_range, "z": (0.3, z_max)}

    def double_exp(R, z, log10_rho01, log10_rho02, Rd1, Rd2, log10_z0):
        return double_exp_model(
            R, z, 10**log10_rho01, 10**log10_rho02, Rd1, Rd2, 10**log10_z0
        )

    return fit_general_emcee(
        coords=(R, z),
        data=rho,
        model_func=double_exp,
        param_names=param_names,
        bounds=bounds,
        p0=p0,
        data_range=data_range,
        weights=weights,
        uncertainties=uncertainties,
        soft_priors=soft_priors,
        logger=logger,
        **kwargs,
    )


def fit_cored_bulge_emcee(
    r: np.ndarray,
    rho: np.ndarray,
    logger: logging.Logger,
    r_range: Tuple[float, float] = (0.3, 5),
    bounds: Optional[Dict] = None,
    p0: Optional[Dict] = None,
    weights: Optional[np.ndarray] = None,
    uncertainties: Optional[np.ndarray] = None,
    soft_priors=None,
    **kwargs,
):
    """
    Fit the bulge with a cored component

    Parameters
    ----------
    r : array_like
        Radial positions in kpc.
    rho : array_like
        Density values in Msun/kpc^3.
    r_range : tuple of float, optional
        (min, max) radial range to fit within, in kpc. Default: (0.3, 5).
    bounds : dict, optional
        Parameter bounds: {'log10_rho0': (min, max), 'log10_rs': (min, max),
                           'log10_rc': (min, max),
                           'alpha': (min, max), 'beta': (min, max),
                           'log_sigma': (min, max)}.
        Default: {'log10_rho0': (1, 10), 'log10_rs': (1.0, 1.9),
                  'log10_rc': (log10(0.3), 0), 'alpha': (0, 2),
                  'beta': (3, 6), 'log_sigma': (-10, 1)}.
    p0 : dict, optional
        Initial guesses: {'log10_rho0': log10_rho0, 'log10_rs': log10_rs,
                          'log10_rc': log10_rc, 'alpha': alpha, 'beta': beta}
        Default: {'log10_rho0': log10(r_mid * 4), 'log10_rs': log10(r_mid),
                  'log10_rc': 0.5, 'alpha': 1, 'beta': 4}.
    weights : array_like, optional
        Weights for each data point. Higher weight = more influence.
        Default: None (uniform weights).
    uncertainties : array_like, optional
        Uncertainties on data points.
    soft_priors : callable, optional
        Function soft_priors(param_dict) -> log_prior_contrib.
        Adds Gaussian penalties to prior. Default: None.
    **kwargs
        Additional arguments passed to fit_general_emcee (nwalkers, nsteps,
        nburn, thin, verbose, etc.).

    Returns
    -------
    flat_samples : ndarray, shape (n_samples, 5)
        Posterior samples with columns [log10_rho0, log10_rs, alpha, beta, log_sigma],
        after burn-in removal and thinning by factor of 10.
    summary : dict
        Dictionary containing:
        - 'rho_0': median rho_0 in Msun/kpc^3
        - 'r_s': median scale radius in kpc
        - 'r_c': core scale radius in kpc
        - 'alpha': inner power-law slope when r << rs model behaves like r^-alpha
        - 'beta': outer power-law slope when r >> rs model behaves like r^-beta
        - 'sigma': median extra scatter parameter
        - 'samples': reference to flat_samples array
        - 'acceptance_fraction': mean acceptance fraction
        - 'sampler': reference to the emcee sampler object
    """
    # -------------------------------------------------
    # Set default bounds if not provided
    # -------------------------------------------------
    if bounds is None:
        bounds = {
            "log10_rho0": (1, 10),
            "log10_rs": (np.log10(0.1), np.log10(5)),
            "log10_rc": (np.log10(0.1), np.log10(1)),
            "alpha": (0, 2),
            "beta": (3, 6),
            "log_sigma": (-10, 1),
        }

    def core_model(r, log10_rho0, log10_rs, log10_rc, alpha, beta):
        rho0 = 10**log10_rho0
        rs = 10**log10_rs
        rc = 10**log10_rc
        return cored_power_model(r, rho0, rs, rc, alpha, beta)

    param_names = ["log10_rho0", "log10_rs", "log10_rc", "alpha", "beta"]

    data_range = {"r": r_range}

    return fit_general_emcee(
        coords=(r,),
        data=rho,
        model_func=core_model,
        param_names=param_names,
        bounds=bounds,
        p0=p0,
        data_range=data_range,
        weights=weights,
        uncertainties=uncertainties,
        soft_priors=soft_priors,
        logger=logger,
        **kwargs,
    )


def fit_exp_emcee(
    r: np.ndarray,
    rho: np.ndarray,
    logger: logging.Logger,
    r_range: Tuple[float, float] = (0.3, 5),
    bounds: Optional[Dict] = None,
    p0: Optional[Dict] = None,
    weights: Optional[np.ndarray] = None,
    uncertainties: Optional[np.ndarray] = None,
    soft_priors: Optional[Callable] = None,
    **kwargs,
):
    """
    Fit an exponential model to bulge component but should be general using MCMC sampling.

    rho(r) = rho0 * exp(-r / r_s)

    Parameters
    ----------
    r : array_like
        Radial positions in kpc.
    rho : array_like
        Density values in Msun/kpc^3.
    r_range : tuple of float, optional
        (min, max) radial range to fit within, in kpc. Default: (10, 200).
    bounds : dict, optional
        Parameter bounds: {'log10_rho0': (min, max), 'log10_rs': (min, max),
                          'log_sigma': (min, max)}.
        Default: {'log10_rho0': (1, 10), 'log10_rs': (1.0, 1.9),
                  'log_sigma': (-10, 1)}.
    p0 : dict, optional
        Initial guesses: {'log10_rho0': log10_rho0, 'log10_rs': log10_rs}
        Default: {'log10_rho0': log10(r_mid * 4), 'log10_rs': log10(r_mid)}.
    weights : array_like, optional
        Weights for each data point. Higher weight = more influence.
        Default: None (uniform weights).
    uncertainties : array_like, optional
        Uncertainties on data points.
    soft_priors : callable, optional
        Function soft_priors(param_dict) -> log_prior_contrib.
        Adds Gaussian penalties to prior. Default: None.
    **kwargs
        Additional arguments passed to fit_general_emcee (nwalkers, nsteps,
        nburn, thin, verbose, etc.).

    Returns
    -------
    flat_samples : ndarray, shape (n_samples, 5)
        Posterior samples with columns [log10_rho0, log10_rs, log_sigma],
        after burn-in removal and thinning by factor of 10.
    summary : dict
        Dictionary containing:
        - 'rho_0': median rho_0 in Msun/kpc^3
        - 'r_s': median scale radius in kpc
        - 'sigma': median extra scatter parameter
        - 'samples': reference to flat_samples array
        - 'acceptance_fraction': mean acceptance fraction
        - 'sampler': reference to the emcee sampler object
    """
    # -------------------------------------------------
    # Set default bounds if none are provided
    # -------------------------------------------------
    if bounds is None:
        bounds = {
            "log10_rho0": (1, 10),
            "log10_rs": np.log10((0.1, 3)),
            "log_sigma": (-10, 1),
        }

    def exp_model(r, log10_rho0, log10_rs):
        rho0 = 10**log10_rho0
        rs = 10**log10_rs
        return rho0 * np.exp(-r / rs)

    data_range = {"r": r_range}

    param_names = ["log10_rho0", "log10_rs"]

    return fit_general_emcee(
        coords=(r,),
        data=rho,
        model_func=exp_model,
        param_names=param_names,
        bounds=bounds,
        p0=p0,
        data_range=data_range,
        weights=weights,
        uncertainties=uncertainties,
        soft_priors=soft_priors,
        logger=logger,
        **kwargs,
    )


def fit_exp_cutoff_emcee(
    r: np.ndarray,
    rho: np.ndarray,
    logger: logging.Logger,
    r_range: Tuple[float, float],
    bounds: Optional[Dict] = None,
    p0: Optional[Dict] = None,
    weights: Optional[np.ndarray] = None,
    uncertainties: Optional[np.ndarray] = None,
    soft_priors: Optional[Callable] = None,
    **kwargs,
):
    """
    Fit an exponential cutoff profile using emcee MCMC sampler.

    This model is useful for components with a central core and exponential
    decay at large radii:

        rho(r) = rho_0 * exp(-r / r_cutoff) / (r_core + r)^alpha

    Parameters
    ----------
    r : array_like
        Radial positions in kpc.
    rho : array_like
        Density values in Msun/kpc^3.
    r_range : tuple of float
        (min, max) radial range to fit within, in kpc.
    bounds : dict, optional
        Parameter bounds: {
            'log10_rho0': (min, max),
            'r_core': (min, max),
            'r_cutoff': (min, max),
            'alpha': (min, max),
            'log_sigma': (min, max)
        }.
        Default: {
            'log10_rho0': (6, 11),
            'r_core': (0.1, 1.5),
            'r_cutoff': (0.5, 2.0),
            'alpha': (0, 4),
            'log_sigma': (-10, 1.1)
        }
    p0 : dict, optional
        Initial parameter guesses: {
            'log10_rho0': log10_rho0,
            'r_core': r_core,
            'r_cutoff': r_cutoff,
            'alpha': alpha
        }.
        If None, uses midpoint of bounds.
    weights : array_like, optional
        Weights for each data point. Higher weight = more influence.
        Default: None (uniform weights).
    uncertainties : array_like, optional
        Uncertainties on data points.
    soft_priors : callable, optional
        Function soft_priors(param_dict) -> log_prior_contrib.
        Adds Gaussian penalties to prior. Default: None.
    **kwargs
        Additional arguments passed to fit_general_emcee (nwalkers, nsteps,
        nburn, thin, verbose, etc.).

    Returns
    -------
    samples : ndarray
        Posterior samples [log10_rho0, r_core, r_cutoff, alpha, log_sigma].
    summary : dict
        Contains median values, uncertainties, and diagnostics for each parameter.
    """
    param_names = ["log10_rho0", "r_core", "r_cutoff", "alpha"]

    if bounds is None:
        bounds = {
            "log10_rho0": (6, 11),
            "r_core": (0.1, 1.5),
            "r_cutoff": (0.5, 2.0),
            "alpha": (0, 4),
            "log_sigma": (-10, 1.1),
        }

    data_range = {"r": r_range}

    def log_exp_cutoff(r, log10_rho0, r_core, r_cutoff, alpha):
        rho0 = 10**log10_rho0
        return exp_cutoff(r, rho0, r_core, r_cutoff, alpha)

    return fit_general_emcee(
        coords=(r,),
        data=rho,
        model_func=log_exp_cutoff,
        param_names=param_names,
        bounds=bounds,
        p0=p0,
        weights=weights,
        uncertainties=uncertainties,
        data_range=data_range,
        soft_priors=soft_priors,
        logger=logger,
        **kwargs,
    )


def fit_double_disk_flaring_emcee(
    R: np.ndarray,
    z: np.ndarray,
    rho: np.ndarray,
    R_range: Tuple[float, float],
    z_max: float,
    logger: logging.Logger,
    bounds: Optional[Dict] = None,
    p0: Optional[Dict] = None,
    weights: Optional[np.ndarray] = None,
    uncertainties: Optional[np.ndarray] = None,
    soft_priors: Optional[Callable] = None,
    **kwargs,
):
    """
    Fit a flaring double disk profile using emcee MCMC sampler.

    This model describes a disk with vertically varying scale height,
    useful for representing stellar disks with flaring structure:

        rho(R, z) = disk_flaring(R, z, rho01, rho02, R_d1, R_d2, A, B)

    where the scale height increases with radius as h(R) = h0 * (R/R0)^A.

    Parameters
    ----------
    R : array_like
        Cylindrical radii in kpc.
    z : array_like
        Vertical heights in kpc.
    rho : array_like
        Density values in Msun/kpc^3.
    R_range : tuple of float
        (min, max) radial range for fitting region, in kpc.
    z_max : float
        Maximum vertical height for fitting region, in kpc.
    bounds : dict, optional
        Parameter bounds: {
            'log10_rho01': (min, max),
            'log10_rho02': (min, max),
            'R_d1': (min, max),
            'R_d2': (min, max),
            'A': (min, max),     # flaring exponent
            'B': (min, max),     # additional shape parameter
            'log_sigma': (min, max)
        }.
        Default: {
            'log10_rho01': (3, 10),
            'log10_rho02': (1, 8),
            'R_d1': (0.1, 12.0),
            'R_d2': (5, 50.0),
            'A': (0, 1),
            'B': (0.1, 5),
            'log_sigma': (-10, 1.1)
        }
    p0 : dict, optional
        Initial parameter guesses: {
            'log10_rho01': log10_rho01,
            'log10_rho02': log10_rho02,
            'R_d1': R_d1,
            'R_d2': R_d2,
            'A': A,
            'B': B
        }.
        If None, uses midpoint of bounds.
    weights : array_like, optional
        Weights for each data point. Higher weight = more influence.
        Default: None (uniform weights).
    uncertainties : array_like, optional
        Uncertainties on data points.
    soft_priors : callable, optional
        Function soft_priors(param_dict) -> log_prior_contrib.
        Adds Gaussian penalties to prior. Default: None.
    **kwargs
        Additional arguments passed to fit_general_emcee (nwalkers, nsteps,
        nburn, thin, verbose, etc.).

    Returns
    -------
    samples : ndarray
        Posterior samples [log10_rho01, log10_rho02, R_d1, R_d2, A, B, log_sigma].
    summary : dict
        Contains median values, uncertainties, and diagnostics for each parameter.
    """
    param_names = ["log10_rho01", "log10_rho02", "R_d1", "R_d2", "A", "B"]
    if bounds is None:
        bounds = {
            "log10_rho01": (3, 10),
            "log10_rho02": (1, 8),
            "R_d1": (0.1, 12.0),
            "R_d2": (5, 50.0),
            "A": (0, 1),
            "B": (0.1, 5),
            "log_sigma": (-10, 1.1),
        }

    data_range = {"R": R_range, "z": (0.3, z_max)}

    def disk_flaring_log(R, z, log10_rho01, log10_rho02, R_d1, R_d2, A, B):
        """Wrapper for disk_flaring model that handles log-space parameters."""
        return double_disk_flaring(
            R, z, 10**log10_rho01, 10**log10_rho02, R_d1, R_d2, A, B
        )

    return fit_general_emcee(
        coords=(R, z),
        data=rho,
        model_func=disk_flaring_log,
        param_names=param_names,
        bounds=bounds,
        p0=p0,
        data_range=data_range,
        weights=weights,
        uncertainties=uncertainties,
        soft_priors=soft_priors,
        logger=logger,
        **kwargs,
    )


def fit_single_disk_flaring_emcee(
    R: np.ndarray,
    z: np.ndarray,
    rho: np.ndarray,
    R_range: Tuple[float, float],
    z_max: float,
    logger: logging.Logger,
    bounds: Optional[Dict] = None,
    p0: Optional[Dict] = None,
    weights: Optional[np.ndarray] = None,
    uncertainties: Optional[np.ndarray] = None,
    soft_priors: Optional[Callable] = None,
    **kwargs,
):
    """
    Fit a flaring single disk profile using emcee MCMC sampler.

    This model describes a disk with vertically varying scale height,
    useful for representing stellar disks with flaring structure:

        rho(R, z) = single_disk_flaring(R, z, rho01, R_d1, A, B)

    where the scale height increases with radius as h(R) = A*R + B.

    Parameters
    ----------
    R : array_like
        Cylindrical radii in kpc.
    z : array_like
        Vertical heights in kpc.
    rho : array_like
        Density values in Msun/kpc^3.
    R_range : tuple of float
        (min, max) radial range for fitting region, in kpc.
    z_max : float
        Maximum vertical height for fitting region, in kpc.
    bounds : dict, optional
        Parameter bounds: {
            'log10_rho01': (min, max),
            'R_d1': (min, max),
            'A': (min, max),     # flaring exponent
            'B': (min, max),     # additional shape parameter
            'log_sigma': (min, max)
        }.
        Default: {
            'log10_rho01': (3, 10),
            'R_d1': (0.1, 12.0),
            'A': (0, 1),
            'B': (0.1, 5),
            'log_sigma': (-10, 1.1)
        }
    p0 : dict, optional
        Initial parameter guesses: {
            'log10_rho01': log10_rho01,
            'R_d1': R_d1,
            'A': A,
            'B': B
        }.
        If None, uses midpoint of bounds.
    weights : array_like, optional
        Weights for each data point. Higher weight = more influence.
        Default: None (uniform weights).
    uncertainties : array_like, optional
        Uncertainties on data points.
    soft_priors : callable, optional
        Function soft_priors(param_dict) -> log_prior_contrib.
        Adds Gaussian penalties to prior. Default: None.
    **kwargs
        Additional arguments passed to fit_general_emcee (nwalkers, nsteps,
        nburn, thin, verbose, etc.).

    Returns
    -------
    samples : ndarray
        Posterior samples [log10_rho01, R_d1, A, B, log_sigma].
    summary : dict
        Contains median values, uncertainties, and diagnostics for each parameter.
    """
    param_names = ["log10_rho01", "R_d1", "A", "B"]
    if bounds is None:
        bounds = {
            "log10_rho01": (3, 10),
            "R_d1": (0.1, 12.0),
            "A": (0, 1),
            "B": (0.1, 5),
            "log_sigma": (-10, 1.1),
        }

    data_range = {"R": R_range, "z": (0.3, z_max)}

    def disk_flaring_log(R, z, log10_rho01, R_d1, A, B):
        """Wrapper for disk_flaring model that handles log-space parameters."""
        return single_disk_flaring(R, z, 10**log10_rho01, R_d1, A, B)

    return fit_general_emcee(
        coords=(R, z),
        data=rho,
        model_func=disk_flaring_log,
        param_names=param_names,
        bounds=bounds,
        p0=p0,
        data_range=data_range,
        weights=weights,
        soft_priors=soft_priors,
        logger=logger,
        uncertainties=uncertainties,
        **kwargs,
    )


def fit_core_emcee(
    r: np.ndarray,
    rho: np.ndarray,
    logger: logging.Logger,
    r_range: Tuple[float, float] = (0.3, 5),
    bounds: Optional[Dict] = None,
    p0: Optional[Dict] = None,
    weights: Optional[np.ndarray] = None,
    uncertainties: Optional[np.ndarray] = None,
    soft_priors: Optional[Callable] = None,
    **kwargs,
):
    """Fitter wrapper for core only bulge profile."""
    param_names = ["log10_rho0", "log10_rs", "log10_rc", "alpha"]
    if bounds is None:
        bounds = {
            "log10_rho0": (1, 10),
            "log10_rs": (np.log10(0.1), np.log10(5)),
            "log10_rc": (np.log10(0.1), np.log10(1)),
            "alpha": (0, 5),
            "log_sigma": (-10, 1),
        }

    data_range = {"r": r_range}

    def model(r, log10_rho0, log10_rs, log10_rc, alpha):
        return only_core(r, 10**log10_rho0, 10**log10_rs, 10**log10_rc, alpha)

    return fit_general_emcee(
        coords=(r,),
        data=rho,
        model_func=model,
        param_names=param_names,
        bounds=bounds,
        p0=p0,
        data_range=data_range,
        weights=weights,
        soft_priors=soft_priors,
        logger=logger,
        uncertainties=uncertainties,
        **kwargs,
    )
