from logging import getLogger

import numpy as np
import yt
from scipy.integrate import quad
import re, h5py

logger = getLogger(__name__)

# ============================================
#  DENSITY PROFILE CALCULATION
# ============================================


def check_extended_mass(ds, center, sim_type):
    """Check if significant mass exists beyond 50 kpc"""

    if center.units == "code_length":
        center = center.to("kpc")

    # Compare mass in different radii
    radii = [25, 50, 100, 200]

    logger.info("=== ENCLOSED MASS CHECK ===")
    for R in radii:
        logger.info(f"R = {R} kpc:")
        sphere = ds.sphere(center, (R, "kpc"))

        M_dm = sphere[("dm", "particle_mass")].sum().to("Msun")
        M_total = M_dm
        logger.info(f"  DM:   {M_dm:.2e}")
        if sim_type != "dmo":
            M_gas = sphere[("gas", "cell_mass")].sum().to("Msun")
            logger.info(f"  Gas:  {M_gas:.2e}")
            M_total += M_gas
        if sim_type == "full":
            M_star = sphere[("star", "particle_mass")].sum().to("Msun")
            logger.info(f"  Star: {M_star:.2e}")
            M_total += M_star

        logger.info(f"  Total: {M_total:.2e}")


def find_virial_radius(
    ds,
    center,
    tol: float = 1e-2,
    max_iter: int = 100,
    min_rad: float = 5.0,
    max_rad: float = 500.0,
    sim_type: str = "full",
) -> float:
    """Find R200 - virial radius"""
    if center.units == "code_length":
        center = center.to("kpc")

    h = ds.hubble_constant
    H0 = h * 100.0 * yt.units.km / yt.units.s / yt.units.Mpc
    z = ds.current_redshift
    Om = ds.omega_matter
    Ol = ds.omega_lambda
    Hz = H0 * np.sqrt(Om * (1 + z)**3 + Ol)

    G = yt.units.gravitational_constant
    rho_crit = 3 * Hz**2 / (8 * np.pi * G)
    rho_crit = rho_crit.to("Msun/kpc**3")

    target_density = 200 * rho_crit
    logger.info(f"z = {z:.3f}, rho_crit(z): {rho_crit:.2e}, 200*rho_crit: {target_density:.2e}")

    R_min = min_rad  # kpc
    R_max = max_rad  # kpc

    # Ensure R_min is above the minimum sphere size that yt will accept.
    # Use ds.index.get_smallest_dx() to get yt's actual enforced minimum rather than
    # computing from MaximumRefinementLevel, which gives the wrong value for L5 DMO sims.
    try:
        min_code_length = float(ds.index.get_smallest_dx().to("code_length").v)
    except Exception:
        min_code_length = ds.domain_width[0].to("code_length").v / (2 ** ds.parameters.get("MaximumRefinementLevel", 14))
    R_min_code = ds.quan(R_min, "kpc").to("code_length").v
    if R_min_code < min_code_length * 2:  # 2x safety factor
        R_min = float(ds.quan(min_code_length * 2, "code_length").to("kpc").v)
        logger.info(f"R_min increased to {R_min:.4f} kpc (above yt minimum sphere size)")

    # Check density at R_min to ensure R200 exists in search range
    sphere_min = ds.sphere(center, (R_min, "kpc"))
    
    M_min = sphere_min[("dm", "particle_mass")].sum()
    if sim_type != "dmo":
        M_min += sphere_min[("gas", "cell_mass")].sum()
    if sim_type == "full":
        M_min += sphere_min[("star", "particle_mass")].sum()

    volume_min = (4.0 / 3.0) * np.pi * (R_min * yt.units.kpc) ** 3
    density_min = (M_min / volume_min).to("Msun/kpc**3")

    if density_min < target_density:
        logger.warning(f"Density at R_min={R_min} kpc ({density_min:.2e}) is below target ({target_density:.2e})")
        logger.warning(f"This halo may not have a well-defined R200 virial radius")
        logger.warning(f"Returning R_min={R_min} kpc as lower bound estimate")
        return R_min

    # Check density at R_max to set appropriate upper bound
    sphere = ds.sphere(center, (R_max, "kpc"))
    M_tot = sphere[("dm", "particle_mass")].sum()
    if sim_type != "dmo":
        M_tot += sphere[("gas", "cell_mass")].sum()
    if sim_type == "full":
        M_tot += sphere[("star", "particle_mass")].sum()

    volume = (4.0 / 3.0) * np.pi * (R_max * yt.units.kpc) ** 3
    density_max = (M_tot / volume).to("Msun/kpc**3")

    if density_max > target_density:
        logger.warning(f"Density at R={R_max} kpc still above target")
        logger.info(f"Increasing R_max to find virial radius")
        R_max = 1000.0

    # Initialize variables for the loop
    density = None
    ratio = None

    for i in range(max_iter):
        R = (R_min + R_max) / 2.0

        sphere = ds.sphere(center, (R, "kpc"))
        M_tot = sphere[("dm", "particle_mass")].sum()
        if sim_type != "dmo":
            M_tot += sphere[("gas", "cell_mass")].sum()
        if sim_type == "full":
            M_tot += sphere[("star", "particle_mass")].sum()

        volume = (4.0 / 3.0) * np.pi * (R * yt.units.kpc) ** 3
        density = (M_tot / volume).to("Msun/kpc**3")
        logger.debug(
            f"Iter {i}: R={R:.3f} kpc, density={density:.2e}, M_tot={M_tot.to('Msun'):.2e}"
        )

        ratio = (density / target_density).value

        if abs(ratio - 1.0) < tol:
            logger.info(f"converged after {i + 1} iterations")
            logger.info(f"R200 ~ {R:.1f} kpc")
            logger.info(f"M200 ~ {M_tot.to('Msun'):.3e}")
            logger.info(f"Density at R200: {density:.3e} (target: {target_density:.3e})")
            return R

        # Adjust bounds based on density
        if ratio > 1.0:  # density too high, R too small, increase R_min
            R_min = R
        else:  # density too low, R too large, decrease R_max
            R_max = R

        # Check if bounds have collapsed without meeting density criterion
        if abs(R_max - R_min) < 0.01:
            if abs(ratio - 1.0) < tol:
                logger.info(f"Converged within tolerance despite bound collapse")
            else:
                logger.warning(f"Bisection bounds collapsed without meeting density criterion")
                logger.warning(f"Final density: {density:.2e}, Target: {target_density:.2e}, Ratio: {ratio:.3f}")
                logger.warning(f"This halo likely does not have a well-defined R200 at this redshift")
                logger.info(f"Returning R={R:.1f} kpc as best estimate")
            return R
        # Otherwise, continue bisection loop

    # If we exit the loop, check if density criterion was met
    logger.warning(f"Did not converge after {max_iter} iterations")
    final_ratio = (density / target_density).value
    logger.warning(f"Final density ratio: {final_ratio:.3f} (target is 1.0)")
    if abs(final_ratio - 1.0) > 0.5:  # More than 50% off
        logger.warning(f"Density criterion poorly satisfied - R200 may be unreliable")
    logger.info(f"R200 ~ {R:.1f} kpc (best estimate)")
    logger.info(f"M200 ~ {M_tot.to('Msun'):.2e}")
    logger.info(f"Density at R200: {density:.2e} (target: {target_density:.2e})")
    return R


def interpolate_r200_from_metadata(halo_data_dir, target_redshift):
    """Interpolate R200 from full-physics metadata at nearby redshifts."""
    meta_files = sorted(halo_data_dir.glob("z_*/metadata.h5"))
    if not meta_files:
        return None
    
    redshifts, r200_values = [], []
    for mf in meta_files:
        m = re.search(r"z_(\d+\.\d+)", mf.parent.name)
        if not m:
            continue
        with h5py.File(mf, 'r') as f:
            if "R200_kpc" in f.attrs:
                redshifts.append(float(m.group(1)))
                r200_values.append(f.attrs['R200_kpc'])
    if not redshifts:
        return None

    R200 = float(np.interp(target_redshift, redshifts, r200_values))

    # Log bracketing redshifts used for interpolation
    z_arr = np.array(redshifts)
    below = z_arr[z_arr <= target_redshift]
    above = z_arr[z_arr >= target_redshift]
    z_lo = below[-1] if len(below) else None
    z_hi = above[0] if len(above) else None
    r200_dict = dict(zip(redshifts, r200_values))
    if z_lo is not None and z_hi is not None and z_lo != z_hi:
        logger.info(
            f"Interpolating R200 at z={target_redshift:.3f}: "
            f"z={z_lo:.3f} ({r200_dict[z_lo]:.1f} kpc) <-> "
            f"z={z_hi:.3f} ({r200_dict[z_hi]:.1f} kpc) => {R200:.1f} kpc"
        )
    elif z_lo is not None:
        logger.info(
            f"R200 at z={target_redshift:.3f}: nearest z={z_lo:.3f} "
            f"({r200_dict[z_lo]:.1f} kpc) => {R200:.1f} kpc (extrapolated)"
        )
    else:
        logger.info(
            f"R200 at z={target_redshift:.3f}: nearest z={z_hi:.3f} "
            f"({r200_dict[z_hi]:.1f} kpc) => {R200:.1f} kpc (extrapolated)"
        )

    return R200


def iterative_center(
    pos,
    mass,
    center_init=None,
    R_init=None,
    shrink_factor=0.5,
    min_radius=0.1,
    tol=1e-3,
    verbose=True,
    min_iterations=15,
):
    """
    Iteratively find galaxy center by shrinking-sphere mass centroid method.

    Parameters
    ----------
    pos : (N,3) array
        Particle or cell positions [kpc].
    mass : (N,) array
        Particle or cell masses [Msun].
    center_init : (3,) array or None
        Starting center guess. If None, uses mass-weighted mean of all positions.
    R_init : float or None
        Initial sphere radius [kpc]. If None, uses max distance from center_init.
    shrink_factor : float
        Factor by which to shrink radius each iteration (default 0.5).
    min_radius : float
        Stop when radius < min_radius [kpc].
    tol : float
        Stop when center shift < tol * radius.
    verbose : bool
        Print iteration diagnostics.
    min_iterations : int
        Minimum number of iterations to perform before checking convergence criteria.

    Returns
    -------
    center : ndarray (3,)
        Estimated galaxy center position [kpc].
    history : list of dict
        Iteration history (radii, shifts, centers).
    """

    pos = np.asarray(pos)
    mass = np.asarray(mass)
    if center_init is None:
        center_init = np.average(pos, weights=mass, axis=0)
    center = center_init.copy()

    if R_init is None:
        R_init = np.max(np.linalg.norm(pos - center, axis=1))

    R = R_init
    history = []

    for i in range(100):
        # distances from current center
        r = np.linalg.norm(pos - center, axis=1)
        # mask particles within current radius
        mask = r < R
        if mask.sum() == 0:
            if verbose:
                logger.info(
                    f"No particles within radius {R:.3f} kpc - stopping.")
            break

        # Compute new mass-weighted CoM
        new_center = np.average(pos[mask], weights=mass[mask], axis=0)
        shift = np.linalg.norm(new_center - center)
        history.append(dict(iter=i, radius=R, shift=shift, n=mask.sum()))

        if verbose:
            logger.info(
                f"Iter {i:2d}: R = {R:7.3f} kpc, shift = {shift:8.5f} kpc, N={mask.sum()}"
            )

        center = new_center
        R *= shrink_factor

        if R < min_radius:
            break

        if i >= min_iterations and shift < tol * R:
            break

    return center, history


def integrate_spherical_mass(density_func, r_max, *params):
    """
    Integrate mass for a spherically symmetric density profile

    M(<r_max) = ∫₀^r_max 4π r² ρ(r) dr

    Parameters:
    -----------
    density_func : callable
        Density function ρ(r, *params)
    r_max : float
        Maximum radius to integrate to [kpc]
    *params : tuple
        Parameters to pass to density_func

    Returns:
    --------
    M : float
        Enclosed mass [Msun]
    error : float
        Integration error estimate
    """

    def integrand(r):
        return 4 * np.pi * r**2 * density_func(r, *params)

    # Start from small non-zero value to avoid singularities at r=0
    # For profiles like NFW with 1/r behavior, this prevents integration warnings
    r_min = 1e-3  # kpc - still negligible compared to typical scale radii

    # Use relaxed tolerances appropriate for astrophysical calculations
    # Absolute tolerance: ~1 Msun is negligible compared to typical galaxy masses
    # Relative tolerance: 0.1% accuracy is sufficient for our purposes

    # For very large integration ranges, split into inner and outer regions
    # to avoid roundoff errors from vastly different scales
    if r_max / r_min > 1e5:  # Large dynamic range
        r_mid = np.sqrt(r_min * r_max)  # Geometric mean
        mass1, err1 = quad(integrand, r_min, r_mid, epsabs=1.0, epsrel=1e-3, limit=150)
        mass2, err2 = quad(integrand, r_mid, r_max, epsabs=1.0, epsrel=1e-3, limit=150)
        mass = mass1 + mass2
        error = np.sqrt(err1**2 + err2**2)
    else:
        mass, error = quad(integrand, r_min, r_max, epsabs=1.0, epsrel=1e-3, limit=200)

    return mass, error


def integrate_disk_mass(R_max, z_max, rho_0, R_d, z_0):
    """
    Integrate mass for an exponential disk

    M = ∫∫ 2π R ρ(R,z) dR dz

    Parameters:
    -----------
    R_max : float
        Maximum cylindrical radius [kpc]
    z_max : float
        Maximum height (integrates from -z_max to +z_max) [kpc]
    rho_0, R_d, z_0 : float
        Disk parameters

    Returns:
    --------
    M : float
        Enclosed mass [Msun]
    error : float
        Integration error estimate
    """
    from scipy.integrate import dblquad

    from .models import disk_density

    def integrand(z, R):
        return 2 * np.pi * R * disk_density(R, z, rho_0, R_d, z_0)

    mass, error = dblquad(integrand, 0, R_max, -z_max, z_max, epsabs=1e-3, epsrel=1e-3)
    return mass, error


def integrate_flaring_double_disk_mass(R_max, z_max, rho_01, rho_02, R_d1, R_d2, A, B):
    """Integrate flaring disk where z_0(R) = A*R + B"""
    from scipy.integrate import dblquad

    from .models import double_disk_flaring

    def integrand(z, R):
        rho = double_disk_flaring(R, z, rho_01, rho_02, R_d1, R_d2, A, B)
        return 2 * np.pi * R * rho

    mass, error = dblquad(integrand, 0, R_max, -z_max, z_max)
    return mass, error


def integrate_flaring_single_disk_mass(R_max, z_max, rho0, Rd, A, B):
    """Integrate flaring disk where z_0(R) = A*R + B"""
    from scipy.integrate import dblquad

    from .models import single_disk_flaring

    def integrand(z, R):
        rho = single_disk_flaring(R, z, rho0, Rd, A, B)
        return 2 * np.pi * R * rho

    mass, error = dblquad(integrand, 0, R_max, -z_max, z_max)
    return mass, error
