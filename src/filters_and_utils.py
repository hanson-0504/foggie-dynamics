from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Any, Dict, Literal, Tuple, Union

import numpy as np
import yt

if TYPE_CHECKING:
    from yt.data_objects.profiles import ProfileND
    from yt.data_objects.selection_objects.spheroids import YTSphere
    from yt.units import YTQuantity

logger = logging.getLogger(__name__)


def _star(pfilter, data):
    return data["all", "creation_time"] > 0


yt.add_particle_filter(
    "star", function=_star, filtered_type="all", requires=["creation_time"]
)


def _dm(pfilter, data):
    return data["all", "creation_time"] <= 0


yt.add_particle_filter(
    "dm", function=_dm, filtered_type="all", requires=["creation_time"]
)


def filter_highres_particles(
    masses: np.ndarray,
    tolerance: float = 0.1,
) -> Tuple[np.ndarray, float]:
    """
    Identify high-resolution particles in a zoom-in simulation.

    In zoom-in cosmological simulations, the box contains particles at
    multiple mass resolution levels. The high-resolution particles
    (lowest mass) populate the zoom region containing the target halo,
    while heavier boundary particles pad the rest of the box.

    Parameters
    ----------
    masses : (N,) array
        Particle masses (any consistent unit).
    tolerance : float, optional
        Fractional tolerance for mass matching (default 0.1 = 10%).

    Returns
    -------
    mask : (N,) bool array
        True for high-resolution particles.
    highres_mass : float
        The identified high-resolution particle mass.
    """
    # Round to 3 significant figures to group similar masses
    order = np.floor(np.log10(np.abs(masses[masses > 0]).min()))
    rounded = np.round(masses, -int(order - 2))

    unique_masses, counts = np.unique(rounded, return_counts=True)
    # Use the minimum mass (unique_masses[0], since np.unique sorts ascending).
    # In zoom simulations the high-res particles always have the lowest mass;
    # selecting by count instead fails at early snapshots when boundary
    # particles outnumber zoom-region particles.
    highres_mass = unique_masses[0]

    mask = np.abs(masses - highres_mass) / highres_mass < tolerance

    return mask, float(highres_mass)


def rotation_matrix_from_vectors(vec1, vec2):
    """Rotation matrix to align vec1 with vec2"""
    a, b = (vec1 / np.linalg.norm(vec1)), (vec2 / np.linalg.norm(vec2))
    v = np.cross(a, b)
    c = np.dot(a, b)
    s = np.linalg.norm(v)

    # Handle case when vectors are already aligned (or anti-aligned)
    if s < 1e-10:  # vectors are parallel
        if c > 0:  # same direction
            return np.eye(3)
        else:  # opposite direction
            # Return 180° rotation around any perpendicular axis
            return -np.eye(3)

    kmat = np.array([[0, -v[2], v[1]], [v[2], 0, -v[0]], [-v[1], v[0], 0]])
    return np.eye(3) + kmat + kmat @ kmat * ((1 - c) / (s**2))


def add_rotated_fields(ds, Rmat, center, comp="star"):
    """
    Add cylindrical coordinate fields (R_disk, z_disk) aligned with galaxy rotation.

    Parameters:
    -----------
    ds : yt dataset
    Rmat : 3x3 array
        Rotation matrix to align z-axis with angular momentum
    center : yt_array (3,)
        Galaxy center coordinates in kpc
    comp : str
        Particle component to which fields are added ('dm', 'star', 'gas', 'all')
    """
    # Convert center to kpc and extract values for use in field definitions
    if hasattr(center, "units"):
        center_kpc = center.to("kpc")
    else:
        center_kpc = ds.arr(center, "kpc")

    # Extract as plain values to avoid units issues inside field definitions
    cx, cy, cz = center_kpc[0].d, center_kpc[1].d, center_kpc[2].d

    if comp == "all":
        for component in ["dm", "star"]:
            add_rotated_fields(ds, Rmat, center, comp=component)
        add_rotated_fields(ds, Rmat, center, comp="gas")
        return

    if comp in ["dm", "star"]:

        def _R_disk(field, data):
            pos = np.column_stack(
                [
                    data[comp, "particle_position_x"].to("kpc").d - cx,
                    data[comp, "particle_position_y"].to("kpc").d - cy,
                    data[comp, "particle_position_z"].to("kpc").d - cz,
                ]
            )
            pos_rot = pos @ Rmat.T
            return ds.arr(np.sqrt(pos_rot[:, 0] ** 2 + pos_rot[:, 1] ** 2), "kpc")

        def _z_disk(field, data):
            pos = np.column_stack(
                [
                    data[comp, "particle_position_x"].to("kpc").d - cx,
                    data[comp, "particle_position_y"].to("kpc").d - cy,
                    data[comp, "particle_position_z"].to("kpc").d - cz,
                ]
            )
            pos_rot = pos @ Rmat.T
            return ds.arr(pos_rot[:, 2], "kpc")

        def _abs_z_disk(field, data):
            """Calculate absolute value of z_disk coordinate."""
            return np.abs(data[(comp, "z_disk")])

        ds.add_field(
            (comp, "z_disk"), function=_z_disk, units="kpc", sampling_type="particle"
        )
        ds.add_field(
            (comp, "R_disk"), function=_R_disk, units="kpc", sampling_type="particle"
        )
        ds.add_field(
            (comp, "abs_z_disk"),
            function=_abs_z_disk,
            sampling_type="particle",
            units="kpc",
        )

    elif comp == "gas":

        def _gas_R_disk(field, data):
            """Cylindrical radius in rotated frame for gas."""
            x = data[("gas", "x")].to("kpc").value - center[0].to("kpc").value
            y = data[("gas", "y")].to("kpc").value - center[1].to("kpc").value
            z = data[("gas", "z")].to("kpc").value - center[2].to("kpc").value

            original_shape = x.shape

            pos = np.column_stack([x.flatten(), y.flatten(), z.flatten()])
            pos_rot = pos @ Rmat.T

            R = np.sqrt(pos_rot[:, 0] ** 2 + pos_rot[:, 1] ** 2)
            return data.ds.arr(R.reshape(original_shape), "kpc")

        def _gas_z_disk(field, data):
            """Height above disk plane in rotated frame for gas."""
            x = data[("gas", "x")].to("kpc").value - center[0].to("kpc").value
            y = data[("gas", "y")].to("kpc").value - center[1].to("kpc").value
            z = data[("gas", "z")].to("kpc").value - center[2].to("kpc").value

            original_shape = x.shape

            pos = np.column_stack([x.flatten(), y.flatten(), z.flatten()])
            pos_rot = pos @ Rmat.T
            z_rot = pos_rot[:, 2]

            return data.ds.arr(z_rot.reshape(original_shape), "kpc")

        def _abs_z_disk(field, data):
            """Calculate absolute value of z_disk coordinate."""
            return np.abs(data[("gas", "z_disk")])

        ds.add_field(
            ("gas", "R_disk"), function=_gas_R_disk, units="kpc", sampling_type="cell"
        )
        ds.add_field(
            ("gas", "z_disk"), function=_gas_z_disk, units="kpc", sampling_type="cell"
        )
        ds.add_field(
            ("gas", "abs_z_disk"),
            function=_abs_z_disk,
            units="kpc",
            sampling_type="cell",
        )


def create_profiles(
    sphere: "YTSphere",
    min_radius: "YTQuantity",
    radius: "YTQuantity",
    comp: str,
) -> ProfileND | None:
    """
    Wrapper for yt.create_profile to create 3D density profiles in (r, R, |z|).
    Generates mass profiles by binning mass by:
    - r : spherical radius from galaxy center
    - R : cylindrical radius in disk plane
    - |z| : absolute height above/below disk plane

    All axes use log spacing.

    Args:
        sphere (yt.data_objects.sphere.YTSphere):
            yt sphere data object containing the galaxy region to profile.
        min_radius (yt.units.yt_array.YTQuantity):
            Minimum radius for profile bins (typically 0.1 kpc).
        radius (yt.units.yt_array.YTQuantity):
            Maximum radius for profile bins (typically R200 or virial radius).
        comp (str):
            Component to profile: 'dm' (dark matter), 'star', 'gas'

    Raises:
        ValueError: If comp is not one of ['dm', 'star', 'gas'].

    Returns:
        Union ProfileND or None
            - If comp is specific: ProfileND object
            - Returns None if profile creation fails

    Notes:
        - Gas profiles use 'cell_mass' and 'radius' fields
        - Particle profiles (dm, star) use 'particle_mass' and 'particle_radius' fields
        - All profiles are 64x64x64 bins in log-space (~262k bins total)
        - Profiles can be saved with: profile.save_as_dataset('filename.h5')
        - For statistics, use calculate_profile_statistics() on the returned profile
    """
    if comp == "gas":
        x_field = ("index", "radius")
        y_field = ("gas", "R_disk")
        z_field = ("gas", "abs_z_disk")
        mass_field = ("gas", "cell_mass")
    elif comp in ["dm", "star"]:
        x_field = (comp, "particle_radius")
        y_field = (comp, "R_disk")
        z_field = (comp, "abs_z_disk")
        mass_field = (comp, "particle_mass")
    else:
        raise ValueError(f"Unknown component: {comp}")

    try:
        logger.info(f"--- Creating {comp} profile ---")
        profile: ProfileND = yt.create_profile(
            sphere,
            [x_field, y_field, z_field],
            [mass_field],
            weight_field=None,
            n_bins=[64, 64, 64],
            extrema={
                x_field: (min_radius.to("kpc"), radius.to("kpc")),
                y_field: (min_radius.to("kpc"), radius.to("kpc")),
                z_field: (min_radius.to("kpc"), radius.to("kpc")),
            },
            logs={x_field: True, y_field: True, z_field: True},
        )
        logger.info(f"{comp} profile created successfully")
        logger.debug(f"  r range: {profile.x.min()} to {profile.x.max()}")
        logger.debug(f"  R range: {profile.y.min()} to {profile.y.max()}")
        logger.debug(f" |z| range: {profile.z.min()} to {profile.z.max()}")

    except Exception as e:
        logger.error(f"Failed to create profile: {e}", exc_info=True)
        return None

    return profile


def extract_density(
    ds,
    geometry: str = "spherical",
    component: str = "star",
    clean: bool = False,
    include_stats: bool = False,
) -> Dict[str, Any]:
    """
    Extract density profiles from FOGGIE 3D profile datasets.

    Parameters:
    -----------
    ds : yt dataset
        YTProfileDataset with 3D profile data (r, R, z)
    geometry : str
        'spherical' for 1D radial profile, 'cylindrical' for 2D (R, z) profile
    component : str
        'star', 'dm', or 'gas'
    clean : bool
        If True, remove invalid values (zeros, NaN, inf) from arrays
    include_stats : bool
        If True, also extract and return profile statistics (mean, std, variance, sem)

    Returns:
    --------
    dict:
        if spherical returns rho_1d, r_1d, r_bins, counts (Optional)
        else returns rho_2d, R, z, R_bins, z_bins
        if include_stats=True, also includes 'stats' key with statistics dict
    """

    # Determine mass field based on component
    if component in ["star", "dm"]:
        mass_field = "particle_mass"
        mass_3d = ds.data[mass_field].to("Msun").value
        # Count bins with particles in each radial shell
        counts = np.sum((mass_3d > 0) & np.isfinite(mass_3d), axis=(1, 2))
    elif component == "gas":
        mass_field = "cell_mass"
        mass_3d = ds.data[mass_field].to("Msun").value
        # Gas is measured in each cell - not important
        counts = None
    else:
        raise ValueError(f"No component: {component}")

    # Extract statistics if requested
    stats_dict = None
    if include_stats:
        try:
            # Extract from profile dataset
            # Note: yt profile objects require try/except for field access
            stats_dict = {}

            # Mean (mass field)
            try:
                mean = ds.data[mass_field]
                stats_dict["mean"] = (
                    mean.to("Msun").value if hasattr(mean, "to") else mean
                )
            except (KeyError, AttributeError):
                pass

            # Standard deviation
            try:
                std = ds.data["standard_deviation"]
                stats_dict["std"] = std.to("Msun").value if hasattr(std, "to") else std
            except (KeyError, AttributeError):
                pass

            # Fractional variance
            try:
                variance = ds.data["variance"]
                stats_dict["frac"] = variance  # Already dimensionless
            except (KeyError, AttributeError):
                pass

            # Bin count (number of particles/cells per bin)
            try:
                bin_count = ds.data["bin_count"]
                stats_dict["N"] = bin_count
            except (KeyError, AttributeError):
                pass

            # Calculate standard error of the mean
            if "std" in stats_dict and "N" in stats_dict:
                sem = np.zeros_like(stats_dict["std"])
                mask = stats_dict["N"] > 0
                sem[mask] = stats_dict["std"][mask] / np.sqrt(stats_dict["N"][mask])
                stats_dict["sem"] = sem

            # If we didn't get any statistics, set to None
            if not stats_dict:
                stats_dict = None
        except Exception as e:
            logger.warning(f"Could not extract statistics: {e}")
            stats_dict = None

    if geometry == "spherical":
        r_bins = ds.data["x_bins"].to("kpc").value
        r_centers = 0.5 * (r_bins[1:] + r_bins[:-1])
        mass = np.sum(mass_3d, axis=(1, 2))  # Sum over R and z axes
        volume = (4.0 / 3.0) * np.pi * (r_bins[1:] ** 3 - r_bins[:-1] ** 3)
        rho = mass / volume

        if clean:
            mask = (rho > 0) & np.isfinite(rho) & np.isfinite(r_centers)
            rho = rho[mask]
            r_centers = r_centers[mask]
            if counts is not None:
                counts = counts[mask]
            # Note: r_bins cannot be easily filtered, so we return the full array

        result = {"rho_1d": rho, "r_1d": r_centers, "r_bins": r_bins, "counts": counts}
        if stats_dict is not None:
            result["stats"] = stats_dict
        return result

    elif geometry == "cylindrical":
        R_bins = ds.data["y_bins"].to("kpc").value
        z_bins = ds.data["z_bins"].to("kpc").value

        R_centers = 0.5 * (R_bins[1:] + R_bins[:-1])
        z_centers = 0.5 * (z_bins[1:] + z_bins[:-1])

        dR = np.diff(R_bins)
        dz = np.diff(z_bins)

        mass = np.sum(mass_3d, axis=0)  # Sum over r axis to get (R, z)
        volume = 2 * np.pi * R_centers[:, None] * (dR[:, None] * dz[None, :])

        rho = mass / volume

        result = {
            "rho_2d": rho,
            "R": R_centers,
            "z": z_centers,
            "R_bins": R_bins,
            "z_bins": z_bins,
        }
        if stats_dict is not None:
            result["stats"] = stats_dict
        return result

    else:
        raise ValueError(f"Mode must be 'spherical' or 'cylindrical', got: {geometry}")


def profile_eq_str(model_name: str, params: dict) -> str:
    """
    Return formatted string with functional form and fitted parameters.

    Parameters
    ----------
    model_name : str
        Name of the density profile model
    params : dict
        Dictionary of fitted parameters

    Returns
    -------
    str
        Multi-line string with equation and parameter values
    """
    from .models import ModelRegistry

    try:
        profile_eq_func = ModelRegistry.get_profile_eq(model_name)
        return profile_eq_func(params)
    except KeyError:
        return f"Unknown model: {model_name}"


def apply_data_range_mask(coords, data, data_range):
    # create mask based on data_range
    # coords = (r,) or (R, z)
    # data = density values of rho(r) or rho(R, z)
    # data_range = {0: (min, max), 1: (min, max)}
    mask = np.ones(len(data), dtype=bool)
    mask &= np.isfinite(data) & (data > 0)
    if data_range is not None:
        for coord_key, (coord_min, coord_max) in data_range.items():
            # Support both integer indices and string names
            if isinstance(coord_key, int):
                coord_idx = coord_key
            else:
                # Could add name mapping here if needed
                raise ValueError(
                    "String coordinate names not yet supported. Use integer indices."
                )

            mask &= (coords[coord_idx] >= coord_min) & (coords[coord_idx] < coord_max)
    return mask


def add_gaussian_weight(coords, fitting_range=None, sigma_frac=0.2, mean=None):
    """
    Generate Gaussian weights centered on the middle of the fitting range.

    This gives more weight to points near the center of your fit region,
    useful when you want the fit to prioritize certain radii/regions.

    Parameters
    ----------
    coords : array
        Coordinate values (e.g., r, R, z)
    fitting_range : tuple, optional
        (min, max) of the fitting region. If None, uses full range of coords.
    sigma_frac : float, optional
        Width of Gaussian as fraction of fitting range. Default: 0.2
        (i.e., sigma = 0.2 * (max - min))
    mean : float, optional
        coordinate of centre of Gaussian. Default is middle of fitting range.

    Returns
    -------
    weights : array
        Gaussian weights (not normalized, max value = 1.0 at center)

    Examples
    --------
    # Give more weight to r ~ 1-2 kpc region in bulge fit
    weights = add_gaussian_weight(r_1d, fitting_range=(0.3, 5), sigma_frac=0.15)

    # Use in fitting (modify log-likelihood to include weights)
    # log_likelihood += log(weights)  # or multiply chi2 by weights
    """
    coords = np.asarray(coords)

    if fitting_range is None:
        r_min, r_max = np.min(coords), np.max(coords)
    else:
        r_min, r_max = fitting_range

    # Center of the range
    if mean is None:
        mean = 0.5 * (r_min + r_max)

    # Width of Gaussian
    sigma = sigma_frac * (r_max - r_min)

    # Gaussian weights (unnormalized, peak = 1.0)
    weights = np.exp(-0.5 * ((coords - mean) / sigma) ** 2)

    return weights


def add_inverse_radius_weight(coords, fitting_range=None, power=1.0, r_scale=None):
    """
    Generate inverse radius weights: w(r) ∝ 1/r^power

    This gives more weight to inner regions and less to outer regions,
    useful when outer data is noisy or less reliable.

    Parameters
    ----------
    coords : array
        Coordinate values (e.g., r, R)
    fitting_range : tuple, optional
        (min, max) of the fitting region. Used to set normalization.
        If None, uses full range of coords.
    power : float, optional
        Power law exponent. Default: 1.0 (inverse radius)
        - power = 1.0: w ∝ 1/r
        - power = 0.5: w ∝ 1/√r (gentler)
        - power = 2.0: w ∝ 1/r² (stronger weighting to center)
    r_scale : float, optional
        Characteristic scale for weighting. If None, uses r_min from fitting_range.
        Weights are normalized so that w(r_scale) = 1.0

    Returns
    -------
    weights : array
        Inverse radius weights (normalized so w(r_scale) = 1.0)

    Examples
    --------
    # Standard 1/r weighting for halo fit
    weights = add_inverse_radius_weight(r_1d, fitting_range=(30, 250))

    # Gentler 1/√r weighting
    weights = add_inverse_radius_weight(r_1d, fitting_range=(30, 250), power=0.5)

    # Normalize at 50 kpc instead of r_min
    weights = add_inverse_radius_weight(r_1d, fitting_range=(30, 250), r_scale=50)
    """
    coords = np.asarray(coords)

    if fitting_range is None:
        r_min, r_max = np.min(coords), np.max(coords)
    else:
        r_min, r_max = fitting_range

    # Set normalization scale
    if r_scale is None:
        r_scale = r_min

    # Compute weights: w(r) = (r_scale/r)^power
    # This ensures w(r_scale) = 1.0
    weights = (r_scale / coords) ** power

    return weights


def scale_range_with_redshift(
    range_z0: Tuple[float, float],
    redshift: float,
    min_scale: float = 0.1,
    max_scale: float = 1.0,
    scaling_mode: str = "linear",
) -> Tuple[float, float]:
    """
    Scale a radial range based on redshift to account for:
    1. Better physical resolution at high-z (comoving resolution)
    2. Smaller galaxy sizes at high-z

    Parameters
    ----------
    range_z0 : tuple of float
        Reference range at z=0 in kpc: (r_min, r_max)
    redshift : float
        Redshift of the snapshot
    min_scale : float, optional
        Minimum scaling factor (default: 0.1 for z→∞)
    max_scale : float, optional
        Maximum scaling factor (default: 1.0 for z=0)
    scaling_mode : str, optional
        Scaling prescription:
        - "linear": scale_factor = 1/(1+z) = a
        - "sqrt": scale_factor = 1/sqrt(1+z)
        - "log": scale_factor = 1/(1+log10(2+z))
        - "none": no scaling (return range_z0)

    Returns
    -------
    range_scaled : tuple of float
        Scaled range at given redshift in kpc: (r_min_scaled, r_max_scaled)

    Notes
    -----
    The "linear" mode uses the cosmological scale factor a = 1/(1+z), which
    naturally accounts for both resolution and galaxy size evolution.

    For high-z snapshots (z>5), this results in ~6x smaller ranges, allowing
    better sampling of compact galaxy structures.

    Examples
    --------
    # Scale bulge range from z=0 to z=5
    >>> scale_range_with_redshift((0.5, 3.0), redshift=5.0, scaling_mode="linear")
    (0.083, 0.5)  # Factor of 6 smaller

    # Scale halo range with sqrt scaling (gentler)
    >>> scale_range_with_redshift((30, 250), redshift=5.0, scaling_mode="sqrt")
    (12.25, 102.06)  # Factor of 2.45 smaller

    # No scaling
    >>> scale_range_with_redshift((0.5, 3.0), redshift=5.0, scaling_mode="none")
    (0.5, 3.0)
    """
    r_min_z0, r_max_z0 = range_z0

    if scaling_mode == "none":
        return (r_min_z0, r_max_z0)

    # Compute scaling factor based on mode
    if scaling_mode == "linear":
        # Standard scale factor: a = 1/(1+z)
        scale_factor = 1.0 / (1.0 + redshift)
    elif scaling_mode == "sqrt":
        # Gentler scaling: sqrt(a)
        scale_factor = 1.0 / np.sqrt(1.0 + redshift)
    elif scaling_mode == "log":
        # Very gentle scaling for testing
        scale_factor = 1.0 / (1.0 + np.log10(2.0 + redshift))
    else:
        logger.warning(f"Unknown scaling_mode '{scaling_mode}', using 'linear'")
        scale_factor = 1.0 / (1.0 + redshift)

    # Clamp to min/max bounds
    scale_factor = np.clip(scale_factor, min_scale, max_scale)

    # Apply scaling
    r_min_scaled = r_min_z0 * scale_factor
    r_max_scaled = r_max_z0 * scale_factor

    return (r_min_scaled, r_max_scaled)


def convert_r200_range_to_kpc(
    range_r200: Tuple[float, float],
    R200_kpc: float,
    min_radius_kpc: float = 0.1,
) -> Tuple[float, float]:
    """
    Convert a range specified in R200 fractions to physical kpc.

    Parameters
    ----------
    range_r200 : tuple of float
        Range as fractions of R200: (r_min/R200, r_max/R200)
        Example: (0.002, 0.015) means 0.2% to 1.5% of R200
    R200_kpc : float
        Virial radius R200 in kpc
    min_radius_kpc : float, optional
        Minimum allowed radius in kpc to avoid numerical issues.
        Default: 0.1 kpc

    Returns
    -------
    range_kpc : tuple of float
        Range in physical kpc: (r_min_kpc, r_max_kpc)

    Examples
    --------
    # Convert bulge range (0.2% - 1.5% of R200) for R200 = 200 kpc
    >>> convert_r200_range_to_kpc((0.002, 0.015), R200_kpc=200.0)
    (0.4, 3.0)  # 0.002*200=0.4, 0.015*200=3.0

    # With larger R200
    >>> convert_r200_range_to_kpc((0.002, 0.015), R200_kpc=300.0)
    (0.6, 4.5)

    # For high-z where R200 is smaller
    >>> convert_r200_range_to_kpc((0.002, 0.015), R200_kpc=50.0)
    (0.1, 0.75)  # r_min clipped to min_radius_kpc
    """
    r_min_r200, r_max_r200 = range_r200

    # Convert to physical kpc
    r_min_kpc = r_min_r200 * R200_kpc
    r_max_kpc = r_max_r200 * R200_kpc

    # Apply minimum radius floor
    r_min_kpc = max(r_min_kpc, min_radius_kpc)

    # Ensure valid range
    if r_max_kpc <= r_min_kpc:
        logger.warning(
            f"R200 range conversion resulted in invalid range: "
            f"({r_min_kpc:.3f}, {r_max_kpc:.3f}) kpc. "
            f"Adjusting r_max to 2*r_min."
        )
        r_max_kpc = 2.0 * r_min_kpc

    return (r_min_kpc, r_max_kpc)
