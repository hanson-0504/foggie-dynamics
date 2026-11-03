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


def linear_params(params: Dict[str, Any]) -> Dict[str, Any]:
    """
    Copy of params with the linear value of each log10_ parameter added.

    Naming convention: parameter names have no underscores (rho0, rs, rc,
    Rd1, z0, rcore, ...). A parameter sampled in log10 space is named
    log10_<name>, and its linear value is <name>:
    log10_rs -> rs = 10**log10_rs, log10_rs_err -> rs_err.
    """
    out = dict(params)
    for name, value in params.items():
        if name.startswith("log10_"):
            out[name[len("log10_"):]] = 10 ** np.asarray(value)
    return out


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
        """
        Register a new model.

        param_names are the sampled parameter names, in the fitter's order
        (= MCMC sample columns, before log_sigma, and the config bounds/p0 keys).
        evaluator, mass_integrator and profile_eq use the linear names; they
        are wrapped with linear_params, so callers can pass sampled values.
        """
        cls._models[name] = {
            "geometry": geometry,
            "fitter": fitter,
            "param_names": param_names,
            "evaluator": lambda coords, s: evaluator(coords, linear_params(s)),
            "mass_integrator": lambda *args: mass_integrator(*args[:-1], linear_params(args[-1])),
            "profile_eq": lambda s: profile_eq(linear_params(s)),
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
            z_min, z_max = data_range[1]

            args = (R, z, rho, R_range, z_max, logger)
            kwargs = {
                "z_min": z_min,
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


# Register all models
#
# param_names: sampled names (fitter order). Evaluators, mass integrators and
# profile equations receive linear_params(...) and use the linear names.
ModelRegistry.register(
    "Hernquist",
    geometry="spherical",
    fitter="fit_bulge_emcee",
    param_names=["log10_rho0", "log10_rs"],
    evaluator=lambda coords, s: np.asarray(hernquist_density(coords[0], s["rho0"], s["rs"])),
    mass_integrator=lambda r, s: integrate_spherical_mass(
        hernquist_density, r, s["rho0"], s["rs"]
    )[0],
    profile_eq=lambda p: (
        "rho(r) = rho0 / [(r/rs) * (1 + r/rs)^3]\n"
        f"  rho0 = {p['rho0']:.3e} Msun/kpc^3\n"
        f"  rs   = {p['rs']:.3f} kpc"
    ),
)
ModelRegistry.register(
    "NFW",
    geometry="spherical",
    fitter="fit_halo_emcee",
    param_names=["log10_rho0", "log10_rs"],
    evaluator=lambda coords, s: np.asarray(nfw_density(coords[0], s["rho0"], s["rs"])),
    mass_integrator=lambda r, s: integrate_spherical_mass(
        nfw_density, r, s["rho0"], s["rs"]
    )[0],
    profile_eq=lambda p: (
        "rho(r) = rho0 / [(r/rs) * (1 + r/rs)^2]\n"
        f"  rho0 = {p['rho0']:.3e} Msun/kpc^3\n"
        f"  rs   = {p['rs']:.3f} kpc"
    ),
)
ModelRegistry.register(
    "Two-Power",
    geometry="spherical",
    fitter="fit_two_power_emcee",
    param_names=["log10_rho0", "log10_rs", "alpha", "beta"],
    evaluator=lambda coords, s: np.asarray(
        two_power_model(coords[0], s["rho0"], s["rs"], s["alpha"], s["beta"])
    ),
    mass_integrator=lambda r, s: integrate_spherical_mass(
        two_power_model, r, s["rho0"], s["rs"], s["alpha"], s["beta"]
    )[0],
    profile_eq=lambda p: (
        "rho(r) = rho0 / [(r/rs)^alpha * (1 + r/rs)^(beta-alpha)]\n"
        f"  rho0  = {p['rho0']:.3e} Msun/kpc^3\n"
        f"  rs    = {p['rs']:.3f} kpc\n"
        f"  alpha = {p['alpha']:.3f}\n"
        f"  beta  = {p['beta']:.3f}"
    ),
)
ModelRegistry.register(
    "Single-Exponential",
    geometry="cylindrical",
    fitter="fit_disk_emcee",
    param_names=["log10_rho01", "Rd1", "z0"],
    evaluator=lambda coords, s: np.asarray(
        disk_density(coords[0], coords[1], s["rho01"], s["Rd1"], s["z0"])
    ),
    mass_integrator=lambda R, z, s: integrate_disk_mass(R, z, s["rho01"], s["Rd1"], s["z0"])[0],
    profile_eq=lambda p: (
        "rho(R, z) = rho01 * exp(-R/Rd1) * sech^2(z/z0)\n"
        f"  rho01 = {p['rho01']:.3e} Msun/kpc^3\n"
        f"  Rd1   = {p['Rd1']:.3f} kpc\n"
        f"  z0    = {p['z0']:.3f} kpc"
    ),
)
ModelRegistry.register(
    "Double-Exponential",
    geometry="cylindrical",
    fitter="fit_double_exponential_disk_emcee",
    param_names=["log10_rho01", "log10_rho02", "Rd1", "Rd2", "log10_z0"],
    evaluator=lambda coords, s: cast(
        np.ndarray,
        np.asarray(
            double_exp_model(
                coords[0], coords[1], s["rho01"], s["rho02"], s["Rd1"], s["Rd2"], s["z0"]
            )
        ),
    ),
    mass_integrator=lambda R, z, s: (
        integrate_disk_mass(R, z, s["rho01"], s["Rd1"], s["z0"])[0]
        + integrate_disk_mass(R, z, s["rho02"], s["Rd2"], s["z0"])[0]
    ),
    profile_eq=lambda p: (
        "rho(R, z) = [rho01 * exp(-R/Rd1) + rho02 * exp(-R/Rd2)] * sech^2(z/z0)\n"
        f"  rho01 = {p['rho01']:.3e} Msun/kpc^3\n"
        f"  Rd1   = {p['Rd1']:.3f} kpc\n"
        f"  rho02 = {p['rho02']:.3e} Msun/kpc^3\n"
        f"  Rd2   = {p['Rd2']:.3f} kpc\n"
        f"  z0    = {p['z0']:.3f} kpc"
    ),
)
ModelRegistry.register(
    "Exponential",
    geometry="spherical",
    fitter="fit_exp_emcee",
    param_names=["log10_rho0", "log10_rs"],
    evaluator=lambda coords, s: np.asarray(exp_model(coords[0], s["rho0"], s["rs"])),
    mass_integrator=lambda r, s: integrate_spherical_mass(exp_model, r, s["rho0"], s["rs"])[0],
    profile_eq=lambda p: (
        "rho(r) = rho0 * exp(-r/rs)\n"
        f"  rho0 = {p['rho0']:.3e} Msun/kpc^3\n"
        f"  rs   = {p['rs']:.3f} kpc"
    ),
)
ModelRegistry.register(
    "Exp-Cutoff",
    geometry="spherical",
    fitter="fit_exp_cutoff_emcee",
    param_names=["log10_rho0", "rcore", "rcutoff", "alpha"],
    evaluator=lambda coords, s: np.asarray(
        exp_cutoff(coords[0], s["rho0"], s["rcore"], s["rcutoff"], s["alpha"])
    ),
    mass_integrator=lambda r, s: integrate_spherical_mass(
        exp_cutoff, r, s["rho0"], s["rcore"], s["rcutoff"], s["alpha"]
    )[0],
    profile_eq=lambda p: (
        "rho(r) = rho0 * exp(-r/rcutoff) / (rcore + r)^alpha\n"
        f"  rho0    = {p['rho0']:.3e} Msun/kpc^3\n"
        f"  rcore   = {p['rcore']:.3f} kpc\n"
        f"  rcutoff = {p['rcutoff']:.3f} kpc\n"
        f"  alpha   = {p['alpha']:.3f}"
    ),
)
ModelRegistry.register(
    "Single-Disk-Flaring",
    geometry="cylindrical",
    fitter="fit_single_disk_flaring_emcee",
    param_names=["log10_rho01", "Rd1", "A", "B"],
    evaluator=lambda coords, s: cast(
        np.ndarray,
        np.asarray(
            single_disk_flaring(coords[0], coords[1], s["rho01"], s["Rd1"], s["A"], s["B"])
        ),
    ),
    mass_integrator=lambda R, z, s: integrate_flaring_single_disk_mass(
        R, z, s["rho01"], s["Rd1"], s["A"], s["B"]
    )[0],
    profile_eq=lambda p: (
        "rho(R, z) = rho01 * exp(-R/Rd1) * sech^2(z/[A*R + B])\n"
        f"  rho01 = {p['rho01']:.3e} Msun/kpc^3\n"
        f"  Rd1   = {p['Rd1']:.3f} kpc\n"
        f"  A     = {p['A']:.3f}\n"
        f"  B     = {p['B']:.3f} kpc"
    ),
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
                coords[0], coords[1], s["rho01"], s["rho02"], s["Rd1"], s["Rd2"], s["A"], s["B"]
            )
        ),
    ),
    mass_integrator=lambda R, z, s: integrate_flaring_double_disk_mass(
        R, z, s["rho01"], s["rho02"], s["Rd1"], s["Rd2"], s["A"], s["B"]
    )[0],
    profile_eq=lambda p: (
        "rho(R, z) = [rho01 * exp(-R/Rd1) + rho02 * exp(-R/Rd2)] * sech^2(z/[A*R + B])\n"
        f"  rho01 = {p['rho01']:.3e} Msun/kpc^3\n"
        f"  Rd1   = {p['Rd1']:.3f} kpc\n"
        f"  rho02 = {p['rho02']:.3e} Msun/kpc^3\n"
        f"  Rd2   = {p['Rd2']:.3f} kpc\n"
        f"  A     = {p['A']:.3f}\n"
        f"  B     = {p['B']:.3f} kpc"
    ),
)
ModelRegistry.register(
    "Cored-bulge",
    geometry="spherical",
    fitter="fit_cored_bulge_emcee",
    param_names=["log10_rho0", "log10_rs", "log10_rc", "alpha", "beta"],
    evaluator=lambda coords, s: np.asarray(
        cored_power_model(coords[0], s["rho0"], s["rs"], s["rc"], s["alpha"], s["beta"])
    ),
    mass_integrator=lambda r, s: integrate_spherical_mass(
        cored_power_model, r, s["rho0"], s["rs"], s["rc"], s["alpha"], s["beta"]
    )[0],
    profile_eq=lambda p: (
        "rho(r) = rho0 / [(rc + r/rs)^alpha * (1 + r/rs)^(beta-alpha)]\n"
        f"  rho0  = {p['rho0']:.3e} Msun/kpc^3\n"
        f"  rs    = {p['rs']:.3f} kpc\n"
        f"  rc    = {p['rc']:.3f} (dimensionless)\n"
        f"  alpha = {p['alpha']:.3f}\n"
        f"  beta  = {p['beta']:.3f}"
    ),
)
ModelRegistry.register(
    "Three-Power",
    geometry="spherical",
    fitter="fit_three_power_emcee",
    param_names=["log10_rho0", "log10_rs", "alpha", "beta", "gamma"],
    evaluator=lambda coords, s: np.asarray(
        three_power_model(coords[0], s["rho0"], s["rs"], s["alpha"], s["beta"], s["gamma"])
    ),
    mass_integrator=lambda r, s: integrate_spherical_mass(
        three_power_model, r, s["rho0"], s["rs"], s["alpha"], s["beta"], s["gamma"]
    )[0],
    profile_eq=lambda p: (
        "rho(r) = rho0 / [(r/rs)^alpha * (1 + (r/rs)^gamma)^((beta-alpha)/gamma)]\n"
        f"  rho0  = {p['rho0']:.3e} Msun/kpc^3\n"
        f"  rs    = {p['rs']:.3f} kpc\n"
        f"  alpha = {p['alpha']:.3f}\n"
        f"  beta  = {p['beta']:.3f}\n"
        f"  gamma = {p['gamma']:.3f}"
    ),
)
ModelRegistry.register(
    "Core-Only",
    geometry="spherical",
    fitter="fit_core_emcee",
    param_names=["log10_rho0", "log10_rs", "log10_rc", "alpha"],
    evaluator=lambda coords, s: np.asarray(
        only_core(coords[0], s["rho0"], s["rs"], s["rc"], s["alpha"])
    ),
    mass_integrator=lambda r, s: integrate_spherical_mass(
        only_core, r, s["rho0"], s["rs"], s["rc"], s["alpha"]
    )[0],
    profile_eq=lambda p: (
        "rho(r) = rho0 / (rc + r/rs)^alpha\n"
        f"  rho0  = {p['rho0']:.3e} Msun/kpc^3\n"
        f"  rs    = {p['rs']:.3f} kpc\n"
        f"  rc    = {p['rc']:.3f} (dimensionless)\n"
        f"  alpha = {p['alpha']:.3f}"
    ),
)


# ---- density models -----
def hernquist_density(r, rho0, rs):
    """
    Spheroid models for bulge stellar components of the galaxy

    Parameters:
    -----------
    r     : spherical radius [kpc]
    rho0  : central density [Msun/kpc^3]
    rs    : scale radius [kpc]
    """
    return two_power_model(r, rho0, rs, 1, 4)


def nfw_density(r, rho0, rs):
    """
    NFW profile for Dark Matter halo

    Parameters:
    ----------
    r     : spherical radius [kpc]
    rho0  : central density [Msun/kpc^3]
    rs    : scale radius [kpc]
    """
    return two_power_model(r, rho0, rs, 1, 3)


def disk_density(R, z, rho0, Rd, z0):
    """
    Exponential disk density with sech^2 vertical profile.
    Numerically stable version.

    Parameters:
    -----------
    R : float or array
        Cylindrical radius
    z : float or array
        Height above disk plane
    rho0 : float
        Central density
    Rd : float
        Radial scale length
    z0 : float
        Vertical scale height

    Returns:
    --------
    rho : float or array
        Density at (R, z)
    """
    z_scaled = np.abs(z) / z0

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
        return rho0 * np.exp(-R / Rd) * result

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

    return rho0 * np.exp(-R / Rd) * result


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
        rc (float): Core term (dimensionless; adds to r/rs)
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


def double_exponential_disk(R, rho01, Rd1, rho02, Rd2):
    """Double exponential radial disk profile (sum of two exponentials)."""
    R = np.asarray(R, dtype=float)
    # Clamp tiny scale lengths to avoid overflow
    Rd1 = np.maximum(Rd1, 1e-3)
    Rd2 = np.maximum(Rd2, 1e-3)
    rho = rho01 * np.exp(-R / Rd1) + rho02 * np.exp(-R / Rd2)
    # Ensure strictly positive
    rho = np.clip(rho, 1e-300, np.inf)
    return rho


def sech_squared(z, const, z0):
    """Numerically stable const * sech^2(z/z0) profile."""
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


def double_exp_model(R, z, rho01, rho02, Rd1, Rd2, z0):
    """Full 3D disk model."""
    rho_R = double_exponential_disk(R, rho01, Rd1, rho02, Rd2)
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


def exp_cutoff(r, rho0, rcore, rcutoff, alpha):
    return rho0 * (rcore + r) ** -alpha * np.exp(-r / rcutoff)


def only_core(r, rho0, rs, rc, alpha):
    return rho0 * (rc + r / rs) ** -alpha


def sech_squared_evolving_z0(R, z, const, A, B):
    R = np.asarray(R, dtype=float)
    z0 = A * R + B
    return sech_squared(z, const, z0)


def double_disk_flaring(R, z, rho01, rho02, Rd1, Rd2, A, B):
    rho_R = double_exponential_disk(R, rho01, Rd1, rho02, Rd2)
    rho_z = sech_squared_evolving_z0(R, z, 1.0, A, B)
    model = rho_R * rho_z

    # Handle both scalar and array inputs
    if np.ndim(model) == 0:  # Scalar case
        if not np.isfinite(model):
            return 1e-300
    else:  # Array case
        model[~np.isfinite(model)] = 1e-300

    return model


def single_disk_flaring(R, z, rho01, Rd1, A, B):
    rho_R = rho01 * np.exp(-R / Rd1)
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
