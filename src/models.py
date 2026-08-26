from functools import lru_cache
from typing import Any, Callable, Dict, Optional, Tuple, cast

import numpy as np

from .density_profiles import (
    integrate_disk_mass,
    integrate_flaring_double_disk_mass,
    integrate_flaring_single_disk_mass,
    integrate_spherical_mass,
)

ModelEvaluator = Callable[[Tuple[np.ndarray, ...], Dict[str, Any]], np.ndarray]
FitterFunc = Callable[..., Tuple[Any, Dict[str, Any]]]
MassIntegrator = Callable[..., float]
ModelEquation = Callable[..., str]


class ModelRegistry:
    """Registry of available density models and their fitting functions."""

    _models: Dict[str, Dict[str, Any]] = {}

    @classmethod
    def register(
        cls,
        name: str,
        geometry: str,
        fitter: str,
        param_names: list,
        evaluator: ModelEvaluator,
        mass_integrator: MassIntegrator,
        profile_eq: ModelEquation,
    ):
        """Register a new model."""
        cls._models[name] = {
            "geometry": geometry,
            "fitter": fitter,
            "param_names": param_names,
            "evaluator": evaluator,
            "mass_integrator": mass_integrator,
            "profile_eq": profile_eq,
        }

    @classmethod
    def get_fitter(cls, model_name: str) -> FitterFunc:
        """Get fitting function for a model."""
        # TODO: Refactor to avoid circular import - consider moving ModelRegistry
        # to a separate registry.py file so models.py and fitting.py can both import it
        from . import fitting

        model_info = cls._models[model_name]
        return getattr(fitting, model_info["fitter"])

    @classmethod
    @lru_cache(maxsize=32)
    def get_evaluator(cls, model_name: str) -> ModelEvaluator:
        """Get evaluator function for a model."""
        return cls._models[model_name]["evaluator"]

    @classmethod
    def get_geometry(cls, model_name: str) -> str:
        """Get geometry for a model."""
        return cls._models[model_name]["geometry"]

    @classmethod
    def fit_model(cls, model_name: str, *args, **kwargs):
        """Fit a model by name."""
        fitter = cls.get_fitter(model_name)
        return fitter(*args, **kwargs)

    @classmethod
    @lru_cache(maxsize=32)
    def get_mass_integrator(cls, model_name: str) -> MassIntegrator:
        return cls._models[model_name]["mass_integrator"]

    @classmethod
    def get_profile_eq(cls, model_name: str) -> ModelEquation:
        return cls._models[model_name]["profile_eq"]

    @classmethod
    @lru_cache(maxsize=32)
    def get_param_names(cls, model_name: str) -> list:
        """Get parameter names for a model."""
        return cls._models[model_name]["param_names"]

    @classmethod
    def list_models(cls) -> list:
        """Return a list of all registered model names."""
        return list(cls._models.keys())

    @classmethod
    def construct_fitter_args(
        cls,
        geometry: str,
        coords: Tuple[np.ndarray, ...],
        data: np.ndarray,
        data_range: Dict,
        bounds: Dict,
        p0: Dict,
        weights: Optional[np.ndarray],
        logger,
        nwalkers: int,
        nsteps: int,
        uncertainties: Optional[np.ndarray],
        fixed_uncertainty: Optional[float],
    ) -> Tuple[Tuple, Dict]:
        """
        Construct positional and keyword arguments for a model's fitter function.

        Args:
            model_name: Name of the model
            geometry: 'spherical' or 'cylindrical'
            coords: Coordinate arrays (r,) or (R, z)
            data: Density data
            data_range: Fitting range constraints
            bounds: Parameter bounds dict
            p0: Initial parameter guesses dict
            weights: Optional weights for data points
            logger: Logger instance
            nwalkers: Number of MCMC walkers
            nsteps: Number of MCMC steps
            uncertainties: Optional measured uncertainties (sigma) on data.
                          Passed to fitting function for use in chi-squared calculation.

        Returns:
            (args, kwargs) tuple ready to pass to the fitter
        """
        if geometry == "spherical":
            # Spherical: fitter(r, rho, logger, r_range, ...)
            r = coords[0]
            rho = data
            r_range = data_range[0]  # (min, max)

            args = (r, rho, logger, r_range)
            kwargs = {
                "bounds": bounds,
                "p0": p0,
                "weights": weights,
                "nwalkers": nwalkers,
                "nsteps": nsteps,
                "uncertainties": uncertainties,
                "fixed_uncertainty": fixed_uncertainty,
            }

        elif geometry == "cylindrical":
            # Cylindrical: fitter(R, z, rho, R_range, z_max, logger, ...)
            R = coords[0]
            z = coords[1]
            rho = data
            R_range = data_range[0]  # (min, max)
            z_max = data_range[1][1]  # upper limit from (0, z_max)

            args = (R, z, rho, R_range, z_max, logger)
            kwargs = {
                "bounds": bounds,
                "p0": p0,
                "weights": weights,
                "nwalkers": nwalkers,
                "nsteps": nsteps,
                "uncertainties": uncertainties,
                "fixed_uncertainty": fixed_uncertainty,
            }
        else:
            raise ValueError(f"Unknown geometry: {geometry}")

        return args, kwargs


# Helper functions for complex profile equation formatting
def _format_double_exp_eq(params):
    """Format Double-Exponential profile equation with parameter name variations."""
    # Handle both old naming (sig1, rho_01) and new naming (log10_rho01)
    rho01 = params.get("sig1") or params.get("rho_01")
    if rho01 is None and "log10_rho01" in params:
        rho01 = 10 ** params["log10_rho01"]
    rho01_str = f"{rho01:.3e}" if rho01 is not None else "N/A"

    rho02 = params.get("sig2") or params.get("rho_02")
    if rho02 is None and "log10_rho02" in params:
        rho02 = 10 ** params["log10_rho02"]
    rho02_str = f"{rho02:.3e}" if rho02 is not None else "N/A"

    Rd1 = params.get("Rd1") or params.get("R_d1")
    Rd1_str = f"{Rd1:.3f}" if Rd1 is not None else "N/A"

    Rd2 = params.get("Rd2") or params.get("R_d2")
    Rd2_str = f"{Rd2:.3f}" if Rd2 is not None else "N/A"

    z0 = params.get("z0") or params.get("z_0") or params.get("B")
    z0_str = f"{z0:.3f}" if z0 is not None else "N/A"

    return (
        f"rho(R, z) = [rho_inner_0 * exp(-R/R_inner_d) + rho_outer_0 * exp(-R/R_outer_d)] * sech^2(z/z_0)\n"
        f"  rho_inner_0 = {rho01_str} Msun/kpc^3\n"
        f"  R_inner_d   = {Rd1_str} kpc\n"
        f"  rho_outer_0 = {rho02_str} Msun/kpc^3\n"
        f"  R_outer_d   = {Rd2_str} kpc\n"
        f"  z_0         = {z0_str} kpc"
    )


def _format_disk_flaring_eq(params):
    """Format Disk-Flaring profile equation for both single and double disk models."""
    # Check if this is a single-disk or double-disk model
    # Single disk has log10_rho01 and R_d1 (not log10_rho02 and R_d2)
    is_single_disk = (
        ("log10_rho01" in params and "log10_rho02" not in params)
        or ("rho0" in params or "rho_0" in params)
        and "Rd" in params
    )

    # Get flaring parameters (common to both models)
    A = params.get("A")
    A_str = f"{A:.3f}" if A is not None else "N/A"

    B = params.get("B")
    B_str = f"{B:.3f}" if B is not None else "N/A"

    if is_single_disk:
        # Single disk model: log10_rho01, R_d1, A, B
        rho0 = params.get("rho0") or params.get("rho_0")
        if rho0 is None and "log10_rho01" in params:
            rho0 = 10 ** params["log10_rho01"]
        elif rho0 is None and "log10_rho0" in params:  # Legacy support
            rho0 = 10 ** params["log10_rho0"]
        rho0_str = f"{rho0:.3e}" if rho0 is not None else "N/A"

        Rd = params.get("R_d1") or params.get("Rd1") or params.get("Rd")
        Rd_str = f"{Rd:.3f}" if Rd is not None else "N/A"

        return (
            f"rho(R,z) = rho_0 * exp(-R/R_d) * sech^2(z/[A*R + B])\n"
            f"  rho_0 = {rho0_str} Msun/kpc^3\n"
            f"  R_d   = {Rd_str} kpc\n"
            f"  A     = {A_str}\n"
            f"  B     = {B_str} kpc"
        )
    else:
        # Double disk model: rho01, rho02, Rd1, Rd2, A, B
        rho01 = params.get("rho_01")
        if rho01 is None and "log10_rho01" in params:
            rho01 = 10 ** params["log10_rho01"]
        rho01_str = f"{rho01:.3e}" if rho01 is not None else "N/A"

        rho02 = params.get("rho_02")
        if rho02 is None and "log10_rho02" in params:
            rho02 = 10 ** params["log10_rho02"]
        rho02_str = f"{rho02:.3e}" if rho02 is not None else "N/A"

        Rd1 = params.get("R_d1") or params.get("Rd1")
        Rd1_str = f"{Rd1:.3f}" if Rd1 is not None else "N/A"

        Rd2 = params.get("R_d2") or params.get("Rd2")
        Rd2_str = f"{Rd2:.3f}" if Rd2 is not None else "N/A"

        return (
            f"rho(R,z) = [rho_inner_0 * exp(-R/R_inner_d) + rho_outer_0 * exp(-R/R_outer_d)] * sech^2(z/[A*R + B])\n"
            f"  rho_inner_0 = {rho01_str} Msun/kpc^3\n"
            f"  R_inner_d   = {Rd1_str} kpc\n"
            f"  rho_outer_0 = {rho02_str} Msun/kpc^3\n"
            f"  R_outer_d   = {Rd2_str} kpc\n"
            f"  A           = {A_str}\n"
            f"  B           = {B_str} kpc"
        )


# Register all models
ModelRegistry.register(
    "Hernquist",
    geometry="spherical",
    fitter="fit_bulge_emcee",
    param_names=["rho0", "rs"],
    evaluator=lambda coords, s: np.asarray(
        hernquist_density(coords[0], s["rho0"], s["rs"])
    ),
    mass_integrator=lambda r, summary: integrate_spherical_mass(
        hernquist_density, r, summary["rho0"], summary["rs"]
    )[0],
    profile_eq=lambda params: (
        f"rho(r) = rho_0 / [(r/r_s) * (1 + r/r_s)^3]\n"
        f"  rho_0 = {params.get('rho0', params.get('rho0', 'N/A')):.3e} Msun/kpc^3\n"
        f"  r_s   = {params.get('rs', params.get('rs', 'N/A')):.3f} kpc"
    ),
)
ModelRegistry.register(
    "NFW",
    geometry="spherical",
    fitter="fit_halo_emcee",
    param_names=["rho0", "rs"],
    evaluator=lambda coords, s: np.asarray(nfw_density(coords[0], s["rho0"], s["rs"])),
    mass_integrator=lambda r, summary: integrate_spherical_mass(
        nfw_density, r, summary["rho0"], summary["rs"]
    )[0],
    profile_eq=lambda params: (
        f"rho(r) = rho_0 / [(r/r_s) * (1 + r/r_s)^2]\n"
        f"  rho_0 = {params.get('rho0', params.get('rho0', 'N/A')):.3e} Msun/kpc^3\n"
        f"  r_s   = {params.get('rs', params.get('rs', 'N/A')):.3f} kpc"
    ),
)
ModelRegistry.register(
    "Two-Power",
    geometry="spherical",
    fitter="fit_two_power_emcee",
    param_names=["log10_rho0", "log10_rs", "alpha", "beta"],
    evaluator=lambda coords, s: np.asarray(
        two_power_model(
            coords[0], 10 ** s["log10_rho0"], 10 ** s["log10_rs"], s["alpha"], s["beta"]
        )
    ),
    mass_integrator=lambda r, summary: integrate_spherical_mass(
        two_power_model,
        r,
        10 ** summary["log10_rho0"],
        10 ** summary["log10_rs"],
        summary["alpha"],
        summary["beta"],
    )[0],
    profile_eq=lambda params: (
        f"rho(r) = rho_0 / [(r/r_s)^alpha * (1 + r/r_s)^(beta-alpha)]\n"
        f"  rho_0 = {10**params.get('log10_rho0', 0):.3e} Msun/kpc^3\n"
        f"  r_s   = {10**params.get('log10_rs', 0):.3f} kpc\n"
        f"  alpha = {params.get('alpha', 'N/A'):.3f}\n"
        f"  beta  = {params.get('beta', 'N/A'):.3f}"
    ),
)
ModelRegistry.register(
    "Single-Exponential",
    geometry="cylindrical",
    fitter="fit_disk_emcee",
    param_names=["rho01", "Rd1", "z0"],
    evaluator=lambda coords, s: np.asarray(
        disk_density(coords[0], coords[1], s["rho01"], s["Rd1"], s["z0"])
    ),
    mass_integrator=lambda R, z, summary: integrate_disk_mass(
        R, z, summary["rho01"], summary["Rd1"], summary["z0"]
    )[0],
    profile_eq=lambda params: (
        f"rho(R, z) = rho_0 * exp(-R/R_d) * sech^2(z/z_0)\n"
        f"  rho_0 = {params.get('rho01', 'N/A'):.3e} Msun/kpc^3\n"
        f"  R_d   = {params.get('Rd1', 'N/A'):.3f} kpc\n"
        f"  z_0   = {params.get('z0', 'N/A'):.3f} kpc"
    ),
)
ModelRegistry.register(
    "Double-Exponential",
    geometry="cylindrical",
    fitter="fit_double_exponential_disk_emcee",
    param_names=["rho01", "rho02", "Rd1", "Rd2", "z0"],
    evaluator=lambda coords, s: cast(
        np.ndarray,
        np.asarray(
            double_exp_model(
                coords[0],
                coords[1],
                s["rho01"],
                s["rho02"],
                s["Rd1"],
                s["Rd2"],
                s["z0"],
            )
        ),
    ),
    mass_integrator=lambda R, z, summary: (
        integrate_disk_mass(R, z, summary["rho01"], summary["Rd1"], summary["z0"])[0]
        + integrate_disk_mass(R, z, summary["rho02"], summary["Rd2"], summary["z0"])[0]
    ),
    profile_eq=lambda params: _format_double_exp_eq(params),
)
ModelRegistry.register(
    "Exponential",
    geometry="spherical",
    fitter="fit_exp_emcee",
    param_names=["rho0", "rs"],
    evaluator=lambda coords, s: np.asarray(exp_model(coords[0], s["rho0"], s["rs"])),
    mass_integrator=lambda r, summary: integrate_spherical_mass(
        exp_model, r, summary["rho0"], summary["rs"]
    )[0],
    profile_eq=lambda params: (
        f"rho(r) = rho_0 * exp(-r/r_s)\n"
        f"  rho_0 = {params.get('rho0', params.get('rho_0', 'N/A')):.3e} Msun/kpc^3\n"
        f"  r_s   = {params.get('rs', params.get('rs', 'N/A')):.3f} kpc"
    ),
)
ModelRegistry.register(
    "Exp-Cutoff",
    geometry="spherical",
    fitter="fit_exp_cutoff_emcee",
    param_names=["rho0", "r_core", "r_cutoff", "alpha"],
    evaluator=lambda coords, s: np.asarray(
        exp_cutoff(coords[0], s["rho0"], s["r_core"], s["r_cutoff"], s["alpha"])
    ),
    mass_integrator=lambda r, summary: integrate_spherical_mass(
        exp_cutoff,
        r,
        summary["rho0"],
        summary["r_core"],
        summary["r_cutoff"],
        summary["alpha"],
    )[0],
    profile_eq=lambda params: (
        f"rho(r) = rho_0 * exp(-r / r_cutoff) / (r_core + r)^alpha\n"
        f"  rho_0 = {params.get('rho0', 'N/A'):.3e} Msun/kpc^3\n"
        f"  r_core = {params.get('r_core', 'N/A'):.3f} kpc\n"
        f"  r_cutoff = {params.get('r_cutoff', 'N/A'):.3f} kpc\n"
        f"  alpha = {params.get('alpha', 'N/A'):.3f}"
    ),
)
ModelRegistry.register(
    "Single-Disk-Flaring",
    geometry="cylindrical",
    fitter="fit_single_disk_flaring_emcee",
    param_names=["log10_rho01", "R_d1", "A", "B"],
    evaluator=lambda coords, s: cast(
        np.ndarray,
        np.asarray(
            single_disk_flaring(
                coords[0], coords[1], 10 ** s["log10_rho01"], s["R_d1"], s["A"], s["B"]
            )
        ),
    ),
    mass_integrator=lambda R, z, summary: integrate_flaring_single_disk_mass(
        R, z, 10 ** summary["log10_rho01"], summary["R_d1"], summary["A"], summary["B"]
    )[0],
    profile_eq=lambda params: _format_disk_flaring_eq(params),
)
ModelRegistry.register(
    "Double-Disk-Flaring",
    geometry="cylindrical",
    fitter="fit_double_disk_flaring_emcee",
    param_names=["log10_rho01", "log10_rho02", "Rd1", "Rd2", "A", "B"],
    evaluator=lambda coords, s: cast(
        np.ndarray,
        np.asarray(
            double_disk_flaring(
                coords[0],
                coords[1],
                10 ** s["log10_rho01"],
                10 ** s["log10_rho02"],
                s["Rd1"],
                s["Rd2"],
                s["A"],
                s["B"],
            )
        ),
    ),
    mass_integrator=lambda R, z, summary: integrate_flaring_double_disk_mass(
        R,
        z,
        10 ** summary["log10_rho01"],
        10 ** summary["log10_rho02"],
        summary["Rd1"],
        summary["Rd2"],
        summary["A"],
        summary["B"],
    )[0],
    profile_eq=lambda params: _format_disk_flaring_eq(params),
)
ModelRegistry.register(
    "Cored-bulge",
    geometry="spherical",
    fitter="fit_cored_bulge_emcee",
    param_names=["rho0", "rs", "rc", "alpha", "beta"],
    evaluator=lambda coords, s: np.asarray(
        cored_power_model(coords[0], s["rho0"], s["rs"], s["rc"], s["alpha"], s["beta"])
    ),
    mass_integrator=lambda r, summary: integrate_spherical_mass(
        cored_power_model,
        r,
        summary["rho0"],
        summary["rs"],
        summary["rc"],
        summary["alpha"],
        summary["beta"],
    )[0],
    profile_eq=lambda params: (
        f"rho(r) = rho_0 / [(r_c + r/r_s)^alpha * (1 + r/r_s)^(beta-alpha)]\n"
        f"  rho_0 = {params.get('rho0', 'N/A'):.3e} Msun/kpc^3\n"
        f"  r_s   = {params.get('rs', 'N/A'):.3f} kpc\n"
        f"  r_c   = {params.get('rc', 'N/A'):.3f} kpc\n"
        f"  alpha = {params.get('alpha', 'N/A'):.3f}\n"
        f"  beta  = {params.get('beta', 'N/A'):.3f}"
    ),
)
ModelRegistry.register(
    "Three-Power",
    geometry="spherical",
    fitter="fit_three_power_emcee",
    param_names=["log10_rho0", "log10_rs", "alpha", "beta", "gamma"],
    evaluator=lambda coords, s: np.asarray(
        three_power_model(
            coords[0],
            10 ** s["log10_rho0"],
            10 ** s["log10_rs"],
            s["alpha"],
            s["beta"],
            s["gamma"],
        )
    ),
    mass_integrator=lambda r, summary: integrate_spherical_mass(
        three_power_model,
        r,
        10 ** summary["log10_rho0"],
        10 ** summary["log10_rs"],
        summary["alpha"],
        summary["beta"],
        summary["gamma"],
    )[0],
    profile_eq=lambda params: (
        f"rho(r) = rho_0 / [(r/r_s)^alpha * (1 + (r/r_s)^gamma)^((beta-alpha)/gamma)]\n"
        f"  rho_0 = {10**params.get('log10_rho0', 0):.3e} Msun/kpc^3\n"
        f"  r_s   = {10**params.get('log10_rs', 0):.3f} kpc\n"
        f"  alpha = {params.get('alpha', 'N/A'):.3f}\n"
        f"  beta  = {params.get('beta', 'N/A'):.3f}\n"
        f"  gamma = {params.get('gamma', 'N/A'):.3f}"
    ),
)
ModelRegistry.register(
    "Core-Only",
    geometry="spherical",
    fitter="fit_core_emcee",
    param_names=["rho0", "rs", "rc", "alpha"],
    evaluator=lambda coords, s: np.asarray(
        s["rho0"] * (s["rc"] + coords[0] / s["rs"]) ** -s["alpha"]
    ),
    mass_integrator=lambda r, summary: integrate_spherical_mass(
        only_core, r, summary["rho0"], summary["rs"], summary["rc"], summary["alpha"]
    )[0],
    profile_eq=lambda params: (
        f"rho(r) = rho_0 / (r_core + r / r_s)^alpha\n"
        f"  rho_0  = {params.get('rho0', params.get('rho_0', 'N/A')):.3e} Msun/kpc^3\n"
        f"  r_s    = {params.get('rs', params.get('rs', 'N/A')):.3f} kpc\n"
        f"  r_core = {params.get('rc', params.get('r_c', 'N/A')):.3f} kpc\n"
        f"  alpha  = {params.get('alpha', 'N/A'):.3f}"
    ),
)


# ---- density models -----
def hernquist_density(r, rho_0, a):
    """
    Spheroid models for bulge stellar components of the galaxy

    Parameters:
    -----------
    r     : spherical radius [kpc]
    rho_0 : central density [Msun/kpc^3]
    a     : scale radius [kpc]
    """
    return two_power_model(r, rho_0, a, 1, 4)


def nfw_density(r, rho_0, r_s):
    """
    NFW profile for Dark Matter halo

    Parameters:
    ----------
    r     : spherical radius [kpc]
    rho_0 : central density [Msun/kpc^3]
    r_s   : scale radius [kpc]
    """
    return two_power_model(r, rho_0, r_s, 1, 3)


def disk_density(R, z, rho_0, R_d, z_0):
    """
    Exponential disk density with sech^2 vertical profile.
    Numerically stable version.

    Parameters:
    -----------
    R : float or array
        Cylindrical radius
    z : float or array
        Height above disk plane
    rho_0 : float
        Central density
    R_d : float
        Radial scale length
    z_0 : float
        Vertical scale height

    Returns:
    --------
    rho : float or array
        Density at (R, z)
    """
    z_scaled = np.abs(z) / (2 * z_0)

    # Use numerically stable form for sech^2
    # For large |z_scaled|, cosh^-2 ≈ 4*exp(-2*|z_scaled|)
    # Use direct formula for small values, asymptotic form for large

    # Threshold where we switch to asymptotic form (avoids overflow)
    threshold = 20  # cosh(20) ≈ 2.4e8, well below overflow

    # Handle scalar case (for numerical integration)
    if np.ndim(z_scaled) == 0:
        if z_scaled < threshold:
            result = 1.0 / np.cosh(z_scaled) ** 2
        else:
            result = 4 * np.exp(-2 * z_scaled)
        return rho_0 * np.exp(-R / R_d) * result

    # Handle array case (for fitting)
    result = np.zeros_like(z_scaled, dtype=float)

    # Small z: use exact formula
    small_mask = z_scaled < threshold
    if np.any(small_mask):
        result[small_mask] = 1.0 / np.cosh(z_scaled[small_mask]) ** 2

    # Large z: use asymptotic approximation
    large_mask = ~small_mask
    if np.any(large_mask):
        result[large_mask] = 4 * np.exp(-2 * z_scaled[large_mask])

    return rho_0 * np.exp(-R / R_d) * result


def two_power_model(r, rho0, rs, alpha, beta):
    """
    2 Power law profile for Bulge/Disk/Spherical regions

    Args:
        r (array): Spherical radius [kpc]
        rho0 (float): Central density [Msun/kpc**3]
        rs (float): Scale radius [kpc]
        alpha (float): Inner slope power
        beta (float): Outer slope power

    Returns:
        array: density model values
    """
    r = np.asarray(r, dtype=float)

    # Handle scalar case (for numerical integration)
    if np.ndim(r) == 0:
        rr = max(float(r), 1e-10)
        return rho0 / ((rr / rs) ** alpha * (1 + rr / rs) ** (beta - alpha))

    # Handle array case (for fitting)
    small = r <= 0
    rr = r.copy()
    rr[small] = 1e-10
    return rho0 / ((rr / rs) ** alpha * (1 + rr / rs) ** (beta - alpha))


def cored_power_model(r, rho0, rs, rc, alpha, beta):
    """
    Cored Power model for irregular bulge profile

    Args:
        r (array): Spherical radius [kpc]
        rho0 (float): Central density [Msun/kpc**3]
        rs (float): Scale radius [kpc]
        rc (float): Core radius [kpc]
        alpha (float): Inner slope power
        beta (float): Outer slope power

    Returns:
        array: density model values
    """
    r = np.asarray(r, dtype=float)

    # Handle scalar case (for numerical integration)
    if np.ndim(r) == 0:
        rr = max(float(r), 1e-10)
        return rho0 / ((rc + rr / rs) ** alpha * (1 + rr / rs) ** (beta - alpha))

    # Handle array case (for fitting)
    small = r <= 0
    rr = r.copy()
    rr[small] = 1e-10
    return rho0 / ((rc + rr / rs) ** alpha * (1 + rr / rs) ** (beta - alpha))


def three_power_model(r, rho0, rs, alpha, beta, gamma):
    """
    Same as 2 power model with a transition factor gamma
    """
    r = np.asarray(r, dtype=float)

    # Handle scalar case (for numerical integration)
    if np.ndim(r) == 0:
        rr = max(float(r), 1e-10)
        return rho0 / (
            (rr / rs) ** alpha * (1 + (rr / rs) ** gamma) ** ((beta - alpha) / gamma)
        )

    # Handle array case (for fitting)
    small = r <= 0
    rr = r.copy()
    rr[small] = 1e-10
    return rho0 / (
        (rr / rs) ** alpha * (1 + (rr / rs) ** gamma) ** ((beta - alpha) / gamma)
    )


def double_exponential_disk(R, Sigma_1, R_d1, Sigma_2, R_d2):
    """Double exponential radial disk profile (sum of two exponentials)."""
    R = np.asarray(R, dtype=float)
    # Clamp tiny scale lengths to avoid overflow
    R_d1 = np.maximum(R_d1, 1e-3)
    R_d2 = np.maximum(R_d2, 1e-3)
    rho = Sigma_1 * np.exp(-R / R_d1) + Sigma_2 * np.exp(-R / R_d2)
    # Ensure strictly positive
    rho = np.clip(rho, 1e-300, np.inf)
    return rho


def sech_squared(z, const, z0):
    """Numerically stable sech^2(z/(2z0)) profile."""
    z = np.asarray(z, dtype=float)
    z0 = np.maximum(z0, 1e-3)  # prevent divide-by-zero
    z_scaled = np.abs(z) / z0

    # Use asymptotic form beyond threshold to avoid overflow
    threshold = 20.0

    # Handle scalar case (for numerical integration)
    if np.ndim(z_scaled) == 0:
        if z_scaled < threshold:
            result = const / np.cosh(z_scaled) ** 2
        else:
            result = const * 4.0 * np.exp(-2.0 * z_scaled)
        return np.clip(result, 1e-300, np.inf)

    # Handle array case (for fitting)
    result = np.empty_like(z_scaled)
    small_mask = z_scaled < threshold

    # Small z: exact expression
    result[small_mask] = const / np.cosh(z_scaled[small_mask]) ** 2

    # Large z: asymptotic approximation
    result[~small_mask] = const * 4.0 * np.exp(-2.0 * z_scaled[~small_mask])

    # Ensure positive finite values
    result = np.clip(result, 1e-300, np.inf)
    return result


def double_exp_model(R, z, sig1, sig2, Rd1, Rd2, z0):
    """Full 3D disk model."""
    rho_R = double_exponential_disk(R, sig1, Rd1, sig2, Rd2)
    sech2 = sech_squared(z, 1.0, z0)
    model = rho_R * sech2

    # Handle both scalar and array inputs
    if np.ndim(model) == 0:  # Scalar case
        if not np.isfinite(model):
            return 1e-300
        return model
    else:  # Array case
        # Replace any NaNs or infs from numerical overflow/underflow
        model[~np.isfinite(model)] = 1e-300
        return model


def exp_cutoff(r, rho0, r_core, r_cutoff, alpha):
    return rho0 * (r_core + r) ** -alpha * np.exp(-r / r_cutoff)


def only_core(r, rho0, rs, r_core, alpha):
    return rho0 * (r_core + r / rs) ** -alpha


def sech_squared_evolving_z0(R, z, const, A, B):
    R = np.asarray(R, dtype=float)
    z0 = A * R + B
    return sech_squared(z, const, z0)


def double_disk_flaring(R, z, rho0_in, rho0_out, Rd_in, Rd_out, A, B):
    rho_R = double_exponential_disk(R, rho0_in, Rd_in, rho0_out, Rd_out)
    rho_z = sech_squared_evolving_z0(R, z, 1.0, A, B)
    model = rho_R * rho_z

    # Handle both scalar and array inputs
    if np.ndim(model) == 0:  # Scalar case
        if not np.isfinite(model):
            return 1e-300
    else:  # Array case
        model[~np.isfinite(model)] = 1e-300

    return model


def single_disk_flaring(R, z, rho0, Rd, A, B):
    rho_R = rho0 * np.exp(-R / Rd)
    rho_z = sech_squared_evolving_z0(R, z, 1.0, A, B)
    model = rho_R * rho_z

    # Handle both scalar and array inputs
    if np.ndim(model) == 0:  # Scalar case
        if not np.isfinite(model):
            return 1e-300
    else:  # Array case
        model[~np.isfinite(model)] = 1e-300

    return model


def exp_model(r, rho0, rs):
    return rho0 * np.exp(-r / rs)
