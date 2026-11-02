"""
Steps of the per-snapshot profile pipeline, split out of
workflow.generate_snapshot_profiles().

Each function does one stage, in this order:

    1. available_components       which components (star/gas/dm) to process
    2. load_dataset               yt.load + redshift, time, DD number
    3. register_particle_filters  add "star" and (high-res) "dm" particle types
    4. find_center                initial guess + shrinking-sphere refinement
                                  (find_centers: several methods, data read once)
    5. compute_r200               from the dataset, else full-physics metadata, else 300 kpc
    6. make_r200_projections      projection plot with an R200 circle per component
    7. shape_analysis             stellar shape tensors at fractions of R200 (full only)
    8. angular_momentum           per-component angular momentum unit vectors
    9. disk_frame                 rotation matrices + R_disk / z_disk / abs_z_disk fields
   10. build_profiles             3D (r, R, |z|) mass profiles per component
   11. enclosed_masses            enclosed mass per component at given radii

Steps 7-11 take a yt sphere of radius R200 made once by the caller, so each
field is read from disk once and shared. disk_frame must run before
build_profiles, which bins on the rotated fields.

Memory logging (log_memory_usage), saving and the DensityProfile conversion
are done by the caller in workflow.py, because importing workflow here would
be circular.
"""

import gc
import re
from pathlib import Path
import datetime

import numpy as np
import yt  # type: ignore

from .density_profiles import (
    find_virial_radius,
    interpolate_r200_from_metadata,
    iterative_center
)
from .io import load_hdf5
from .filters_and_utils import (
    filter_highres_particles,
    rotation_matrix_from_vectors,
    add_rotated_fields,
    create_profiles
)
from .analysis import get_shape_tensor


# ---------------------------------------------------------------------------
# Setup: components, loading, particle filters
# ---------------------------------------------------------------------------


def available_components(sim_type, component):
    """
    List the components to process.

    Args:
        sim_type: 'full' (stars + gas + dm), 'gas_only' (gas + dm) or 'dmo' (dm only).
        component: 'all' for every component the simulation has,
            or a single name ('star', 'gas', 'dm').

    Returns:
        List of component names, e.g. ['star', 'gas', 'dm'].
    """
    has_gas = sim_type != "dmo"
    has_stars = sim_type == "full"

    if component == "all":
        avail_components = (
            ['star', 'gas', 'dm'] if has_stars
            else ['gas', 'dm'] if has_gas
            else ['dm']
        )
    else:
        avail_components = [component]

    return avail_components


def load_dataset(snap_path, logger, redshift, halo_id):
    """
    Load a snapshot and read its basic properties.

    Args:
        snap_path: Path to the snapshot, e.g. '.../DD2427/DD2427'.
        logger: Logger instance.
        redshift: Redshift to use. If None, uses ds.current_redshift.
            Passing the value from the snapshot table keeps directory
            names (z_X.XXX) consistent between runs.
        halo_id: Halo identifier, only used for logging.

    Returns:
        (ds, snapshot_info) where snapshot_info is a dict with
        'redshift', 'time_Gyr' and 'dd_num' (None if the path has no DD/RD number).
    """
    logger.info(f"Loading dataset: {halo_id}")
    ds = yt.load(snap_path)

    if redshift is None:
        redshift = float(ds.current_redshift)
        logger.debug(f"Using redshift from dataset: {redshift}")
    else:
        logger.debug(f"Using provided redshift: {redshift} (dataset reports {ds.current_redshift})")
    time_Gyr = float(ds.current_time.to("Gyr").value)

    # Output number from the path, e.g. '.../DD2427/DD2427' -> 2427
    snap_match = re.search(r"(DD|RD)(\d+)", str(snap_path))
    dd_num = int(snap_match.group(2)) if snap_match else None

    snapshot_info = {
        "redshift": redshift,
        "time_Gyr": time_Gyr,
        "dd_num": dd_num,
    }
    return ds, snapshot_info


def register_particle_filters(ds, sim_type, logger):
    """
    Add the "star" and "dm" particle types to the dataset.

    - full: the creation_time-based "star"/"dm" filters from filters_and_utils
      (stars have creation_time > 0, DM has creation_time <= 0)
    - gas_only / dmo: no star particles, so creation_time doesn't exist;
      register a mass-based "dm" filter instead (see _register_highres_dm_filter)
    """
    if sim_type == "full":
        ds.add_particle_filter("star")
        ds.add_particle_filter("dm")
    else:
        _register_highres_dm_filter(ds, logger, label="DMO" if sim_type == "dmo" else "gas_only")


def _register_highres_dm_filter(ds, logger, label):
    """
    Register a "dm" particle filter that keeps only high-resolution particles.

    Zoom simulations surround the high-res region with heavier low-res
    "boundary" particles, which would bias centring and masses. Particles
    within ±10% of the high-res (lowest) particle mass are kept.

    Note: yt.add_particle_filter is global, so this replaces the
    creation_time "dm" filter for any dataset loaded later in the same process.

    Args:
        ds: yt dataset (gains the "dm" particle type).
        logger: Logger instance.
        label: Prefix for the log message, e.g. "DMO" or "gas_only".
    """
    # Find the high-res particle mass from all particles in the box
    ad = ds.all_data()
    all_masses = ad["all", "particle_mass"].to("Msun").value
    _, hr_mass = filter_highres_particles(all_masses)
    del ad, all_masses
    gc.collect()

    logger.info(f"{label}: high-res particle mass = {hr_mass:.2e} Msun")
    hr_mass_min = hr_mass * 0.9
    hr_mass_max = hr_mass * 1.1

    # Closure: yt calls this later, and it keeps hr_mass_min/max
    def _dm_highres(pfilter, data):
        mass = data["all", "particle_mass"].to("Msun").value
        return (mass >= hr_mass_min) & (mass <= hr_mass_max)

    yt.add_particle_filter("dm", function=_dm_highres, filtered_type="all")
    ds.add_particle_filter("dm")


# ---------------------------------------------------------------------------
# Centre finding
# ---------------------------------------------------------------------------


def _highres_particles(ds):
    """
    Positions and masses of high-res particles only.

    Returns:
        (pos_kpc [N, 3], mass_Msun [N], hr_mass_Msun)
    """
    ad = ds.all_data()
    masses = ad["all", "particle_mass"].to("Msun").value
    hr_mask, hr_mass = filter_highres_particles(masses)
    pos = np.column_stack([
        ad["all", f"particle_position_{ax}"].to("kpc").value[hr_mask] for ax in "xyz"
    ])
    return pos, masses[hr_mask], hr_mass


def _shrinking_sphere_tracer(ds, sim_type, logger):
    """
    Particles used to refine the centre, and the shrinking-sphere settings.

    - full: star particles, start at R_init = 50 kpc, stop at 0.1 kpc
    - gas_only / dmo: high-res DM particles, start at 200 kpc, stop at 1 kpc
      (DM is more extended than stars, so a larger, coarser sphere is used)

    Returns:
        (pos_kpc [N, 3], mass_Msun [N], R_init_kpc, min_radius_kpc, tracer_name)
    """
    if sim_type == "full":
        # TODO (P1): this reads every star in the box; a ds.sphere around the
        # initial guess of radius R_init would be much faster
        ad = ds.all_data()
        pos = np.column_stack([ad["star", f"particle_position_{ax}"].to("kpc").value for ax in "xyz"])
        mass = ad["star", "particle_mass"].to("Msun").value
        return pos, mass, 50.0, 0.1, "star particles"

    pos, mass, hr_mass = _highres_particles(ds)
    logger.info(f"Shrinking sphere: using {len(mass):,} high-res DM particles (mass ~ {hr_mass:.2e} Msun)")
    return pos, mass, 200.0, 1.0, "high-res DM particles"


def initial_center_guess(ds, sim_type, logger):
    """
    Rough galaxy centre, used to seed the shrinking sphere.

    - full / gas_only: location of the maximum dark_matter_density
    - dmo: centre of the densest bin of a 128^3 histogram of high-res
      particle positions (DMO runs have no gas fields to call find_max on)

    Returns:
        Centre as a yt array.
    """
    if sim_type != "dmo":
        _, center = ds.find_max(("gas", "dark_matter_density"))
        logger.info(f"Galaxy center (find_max method): {center}")
        return center

    pos, _, hr_mass = _highres_particles(ds)
    logger.info(f"DMO histogram center: using {len(pos):,} high-res particles (mass ~ {hr_mass:.2e} Msun)")

    # Unweighted (particle count) histogram; centre = midpoint of the peak bin
    H, edges = np.histogramdd(pos, bins=128)
    idx = np.unravel_index(H.argmax(), H.shape)
    center = ds.arr([(edges[d][idx[d]] + edges[d][idx[d] + 1]) / 2 for d in range(3)], "kpc")
    logger.info(f"Galaxy center (histogram method): {center}")
    return center


def _refine(ds, pos, mass, initial, R_init, min_radius, tracer, logger):
    """
    Shrinking-sphere refinement of the centre.

    Args:
        pos, mass, R_init, min_radius, tracer: from _shrinking_sphere_tracer.
        initial: initial guess (yt array, kpc), or None for the global run.

    If the shrinking sphere fails, returns the initial guess
    (or raises if there is none).

    Returns:
        Centre as a yt array in kpc.
    """
    logger.info(f"Shrinking sphere on {tracer} ({'global' if initial is None else 'seeded'})")
    # Seeded: start at the initial guess with radius R_init.
    # Global: center_init=None, R_init=None -> iterative_center starts from the whole box.
    try:
        refined, _ = iterative_center(
            pos, mass,
            center_init=None if initial is None else initial.to("kpc").value,
            R_init=None if initial is None else R_init,
            min_radius=min_radius,
        )
    except Exception as e:
        if initial is None:
            raise RuntimeError(f"Global shrinking sphere failed with no fallback: {e}") from e
        logger.warning(f"Shrinking sphere failed ({e}); using initial guess")
        return initial.to("kpc")

    refined = ds.arr(refined, "kpc")
    logger.info(f"Galaxy center (iterative method): {refined}")
    if initial is not None:
        _log_center_offset(ds, initial, refined, logger)
    return refined


CENTER_METHODS = ("initial_only", "shrinking_sphere", "shrinking_sphere_global", "average")


def find_centers(ds, sim_type, logger, methods=("initial_only", "shrinking_sphere", "average")):
    """
    Compute several centre estimates, sharing the expensive reads.

    The initial guess (find_max / DMO histogram) and the tracer particles are
    read once, and one seeded shrinking sphere serves both "shrinking_sphere"
    and "average". See find_center for what each method means.

    Returns:
        {method: centre as a yt array in kpc}
    """
    methods = list(methods)
    unknown = set(methods) - set(CENTER_METHODS)
    if unknown:
        raise ValueError(f"Unknown centre method(s) {sorted(unknown)}; choose from {CENTER_METHODS}")
    need_initial = any(m != "shrinking_sphere_global" for m in methods)
    initial = initial_center_guess(ds, sim_type, logger).to("kpc") if need_initial else None

    results = {}
    if "initial_only" in methods:
        results["initial_only"] = initial

    # Every other method needs the tracer particles
    if set(methods) == {"initial_only"}:
        return results
    pos, mass, R_init, min_radius, tracer = _shrinking_sphere_tracer(ds, sim_type, logger)

    if {"shrinking_sphere", "average"} & set(methods):
        refined = _refine(ds, pos, mass, initial, R_init, min_radius, tracer, logger)  # falls back to initial
        if "shrinking_sphere" in methods:
            results["shrinking_sphere"] = refined
        # TODO (C2): "average" mixes two different estimators (DM density peak
        # and stellar shrinking sphere); decide which one to use
        if "average" in methods:
            results["average"] = (initial + refined) / 2

    if "shrinking_sphere_global" in methods:
        results["shrinking_sphere_global"] = _refine(ds, pos, mass, None, None, min_radius, tracer, logger)

    return results


def find_center(ds, sim_type, method, logger):
    """
    Find the galaxy centre.

    Args:
        ds: yt dataset with particle filters registered.
        sim_type: 'full', 'gas_only' or 'dmo' (selects the tracer particles).
        method:
            - "initial_only": initial_center_guess only
            - "shrinking_sphere": refine the initial guess with a shrinking sphere
            - "shrinking_sphere_global": shrinking sphere from the whole box (no guess)
            - "average": mean of the initial guess and the refined centre
        logger: Logger instance.

    If the shrinking sphere fails, falls back to the initial guess
    (or raises for "shrinking_sphere_global", which has no guess).
    To compare several methods, use find_centers (reads the data once).

    Returns:
        Centre as a yt array in kpc.
    """
    return find_centers(ds, sim_type, logger, [method])[method]


def _log_center_offset(ds, initial, refined, logger):
    """Log how far the shrinking-sphere centre is from the initial guess. Logging only."""
    diff = (initial - refined).to("kpc").value
    abs_diff = float(np.linalg.norm(diff))

    logger.info(
        f"Comparison: change in method values (initial - iterative) = {diff} kpc"
    )
    logger.info(f"Absolute difference = {abs_diff:.3f} kpc")

    # Compare the offset to the simulation resolution
    min_resolution_kpc = float(ds.index.get_smallest_dx().to("kpc").v)
    logger.info(f"Simulation resolution (finest cell size) = {min_resolution_kpc:.3f} kpc")
    logger.info(f"Offset relative to finest cell = {abs_diff / min_resolution_kpc:.2f} cells")

    if abs_diff < min_resolution_kpc:
        logger.info("Center difference is below resolution limit - methods agree within numerical precision")
    elif abs_diff < 3 * min_resolution_kpc:
        logger.warning(f"Center difference ({abs_diff:.3f} kpc) is comparable to resolution ({min_resolution_kpc:.3f} kpc)")
    else:
        logger.info(f"Center difference ({abs_diff:.3f} kpc) is {abs_diff / min_resolution_kpc:.1f}x resolution")


# ---------------------------------------------------------------------------
# R200 calculations
# ---------------------------------------------------------------------------


def compute_r200(ds, center, halo_id, redshift, sim_type, paths, logger):
    """R200 (kpc): from the dataset, else full-physics metadata, else 300 kpc."""
    # 1. Direct from this dataset
    try:
        R200 = find_virial_radius(
            ds, center, tol=1e-3, max_iter=500,
            min_rad=0.5, max_rad=300.0, sim_type=sim_type
        )
        logger.info(f"Calculated R200 = {R200:.2f} kpc from dataset.")
        return R200
    except Exception as e:
        logger.warning(f"Direct R200 calculation failed: {e}", exc_info=True)

    # 2. From the matching full-physics run (non-full sims only)
    if sim_type != 'full':
        full_meta_path = paths.snapshot_data(halo_id, redshift) / "metadata.h5"
        if full_meta_path.exists():
            R200 = load_hdf5(full_meta_path).get("R200_kpc")
            source = "full-physics metadata"
        else:
            R200 = interpolate_r200_from_metadata(paths.data / halo_id, redshift)
            source = "interpolated full-physics metadata"
        if R200 is not None:
            logger.info(f"R200 = {R200:.2f} kpc from {source}")
            return R200

    # 3. Last resort
    # TODO (C3): the run continues as normal with this value; consider raising
    # or storing an R200_is_fallback flag in metadata
    logger.error("Failed to determine R200 from dataset and full-physics metadata. Using R200 = 300 kpc as fallback.")
    return 300.0


def make_r200_projections(
    ds, center, r200, redshift, avail_components,
    snap_path, halo_id, paths, logger,
    experiment=None, config=None
):
    """
    Save a projection plot with an R200 circle for each component.

    Written to <snapshot plots dir>/projections/{snap}_{comp}_projection_r200.png.
    A failed plot is logged and skipped; it does not stop the run.
    """
    # Lazy import: plotting imports workflow, which imports this module
    from .plotting import plot_comp_projection_with_r200
    logger.info("Creating projection plot with R200 overlay...")
    
    # Determine projection output directory
    proj_dir = paths.snapshot_plots(
        halo_id, redshift, experiment=experiment
    ) / "projections"
    proj_dir.mkdir(parents=True, exist_ok=True)
    
    # Extract snapshot name
    snap_name = Path(snap_path).name

    for comp in avail_components:
        proj_path = proj_dir / f"{snap_name}_{comp}_projection_r200.png"
        try:
            plot_comp_projection_with_r200(
                ds=ds,
                center=center,
                r200_kpc=r200,
                redshift=redshift,
                output_path=proj_path,
                component=comp,
                config=config
            )
            logger.info(f"Saved projection to {proj_path}")
        except Exception as e:
            logger.error(f"Failed to create projection plot: {e}", exc_info=True)


# ---------------------------------------------------------------------------
# Shape Tensor analysis
# ---------------------------------------------------------------------------


def _shape_tensor_at_radius(positions, weights, r, logger):
    """
    Mass-weighted shape tensor of the given stars; logs axes and a rough morphology.

    Args:
        positions: (3, N) star positions relative to the centre (kpc), already cut to <= r.
        weights: (N,) star masses (Msun).
        r: Radius (kpc), for logging only.
        logger: Logger instance.

    Returns:
        3x3 shape tensor.
    """
    logger.info(f"Radius = {r} kpc ({weights.size:,} stars)")
    
    shape_tensor, (a, b, c) = get_shape_tensor(weights, positions)
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
        
    return shape_tensor
    
    
def shape_analysis(sphere, center, r200, logger, fractions=(0.01, 0.02, 0.05, 0.1, 0.2, 0.3, 0.5)):
    """
    Stellar shape tensors at several fractions of R200 (needs star particles).

    Stars are read from the sphere once and cut to each radius in numpy.
    Radii that fail are logged and skipped.

    Args:
        sphere: yt sphere around center enclosing max(fractions) * r200
            (e.g. the R200 sphere).
        center: Galaxy centre (yt array).
        r200: R200 (kpc).
        logger: Logger instance.
        fractions: Radii to test, as fractions of R200.

    Returns:
        List of (radius_kpc, 3x3 shape tensor).
    """
    logger.info("Shape Analysis at different radii")
    c = center.to("kpc").value
    positions = np.vstack([
        sphere["star", f"particle_position_{ax}"].to("kpc").value - c[i]
        for i, ax in enumerate("xyz")
    ])
    weights = sphere["star", "particle_mass"].to("Msun").value
    r_star = np.sqrt((positions**2).sum(axis=0))

    shape_tensors = []
    for r in [f * r200 for f in fractions]:
        inside = r_star <= r
        try:
            shape_tensors.append((r, _shape_tensor_at_radius(positions[:, inside], weights[inside], r, logger)))
        except Exception as e:
            logger.warning(f"Could not compute shape tensor at r={r} kpc: {e}", exc_info=True)

    return shape_tensors


def angular_momentum(sphere, avail_components, logger):
    """
    Angular momentum direction of each component within the sphere.

    yt's angular_momentum_vector gives the mass-weighted mean specific angular
    momentum about the sphere centre. With more than one component, "total"
    combines them weighted by each component's mass in the sphere.
    Logs each unit vector, and the star-gas misalignment angle when both exist.

    Args:
        sphere: yt sphere centred on the galaxy (e.g. the R200 sphere).
        avail_components: Components to compute, e.g. ['star', 'gas', 'dm'].
        logger: Logger instance.

    Returns:
        {comp: unit vector (3,)}, plus "total" if more than one component.
    """
    logger.info("Calculating Angular Momentum...")
    q = sphere.quantities
    l_vecs, masses = {}, {}
    for comp in avail_components:
        if comp == "gas":
            l = q.angular_momentum_vector(use_gas=True, use_particles=False)
            m = q.total_quantity(("gas", "cell_mass"))
        else:
            l = q.angular_momentum_vector(use_gas=False, use_particles=True, particle_type=comp)
            m = q.total_quantity((comp, "particle_mass"))
        l_vecs[comp] = l.to("cm**2/s").d
        masses[comp] = float(m.to("Msun"))
    if len(l_vecs) > 1:
        l_vecs["total"] = sum(masses[c] * l_vecs[c] for c in masses) / sum(masses.values())
    l_hats = {comp: _unit_vector(l) for comp, l in l_vecs.items()}
    for comp, l_hat in l_hats.items():
        logger.info(f"{comp} angular momentum unit vector = {l_hat}")
    if "star" in l_hats and "gas" in l_hats:
        logger.info(f"Star-gas misalignment = {misalignment_deg(l_hats['star'], l_hats['gas']):.1f} deg")
    return l_hats


def _unit_vector(vec):
    """vec divided by its length."""
    return vec / np.linalg.norm(vec)


def misalignment_deg(a, b):
    """Angle between vectors a and b in degrees (0-180); they need not be unit vectors."""
    a, b = np.asarray(a), np.asarray(b)
    cos = np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))
    return np.degrees(np.arccos(np.clip(cos, -1, 1)))


def disk_frame(ds, center, l_hats, avail_components, logger):
    """
    Rotate each component into its own disk frame and add the rotated fields.

    The rotation takes the z-axis onto that component's angular momentum
    direction. Adds (comp, "R_disk"), (comp, "z_disk") and (comp, "abs_z_disk")
    to ds, which build_profiles bins on, so this must run first.

    Note: the original workflow used one shared frame (from gas + all particles)
    for every component; here each component gets its own.

    Args:
        ds: yt dataset (gains the rotated fields).
        center: Galaxy centre (yt array).
        l_hats: Output of angular_momentum.
        avail_components: Components to rotate.
        logger: Logger instance.

    Returns:
        {comp: 3x3 rotation matrix}
    """
    Rmats = {}
    for comp in avail_components:
        Rmats[comp] = rotation_matrix_from_vectors(np.array([0, 0, 1]), l_hats[comp])
        _log_rotation_matrix(Rmats[comp], logger)
        add_rotated_fields(ds, Rmats[comp], center.to("kpc"), comp=comp)
    return Rmats


def _log_rotation_matrix(Rmat, logger):
    """Debug-log Rmat and checks that it is a proper rotation (finite, det ~1, orthogonal)."""
    logger.debug(f"Rmat:\n{Rmat}")
    logger.debug(f"Rmat contains NaN: {np.any(np.isnan(Rmat))}")
    logger.debug(f"Rmat contains Inf: {np.any(np.isinf(Rmat))}")
    logger.debug(f"Rmat determinant: {np.linalg.det(Rmat):.6f} (should be ~1.0)")
    logger.debug(f"Rmat is orthogonal: {np.allclose(Rmat @ Rmat.T, np.eye(3))}")


# ---------------------------------------------------------------------------
# Profiles and Masses
# ---------------------------------------------------------------------------

def build_profiles(ds, sphere, r200, components, logger):
    """
    3D mass profiles in (r, R_disk, |z_disk|) for each component.

    Bins run from max(0.1% of R200, 0.01 kpc) to R200, 64 log bins per axis
    (see create_profiles). Needs disk_frame to have added the rotated fields.

    Args:
        ds: yt dataset.
        sphere: yt sphere of radius R200 around the galaxy centre.
        r200: R200 (kpc).
        components: Components to profile, e.g. ['star', 'gas', 'dm'].
        logger: Logger instance.

    Raises:
        RuntimeError: If no profile could be created. Partial failures are
            logged as a warning and left out.

    Returns:
        {comp: yt ProfileND} for the components that succeeded.
    """
    logger.info("Getting density profiles...")
    min_rad_kpc = max(0.001 * r200, 0.01) # high redshift floor
    min_radius = ds.quan(min_rad_kpc, "kpc")
    logger.info(f"min_radius = {min_rad_kpc:.3f} kpc (max of 0.1% R200, 0.01 kpc floor)")

    radius = ds.quan(r200, "kpc")
    profiles = {
        comp: create_profiles(sphere, min_radius, radius, comp=comp)
        for comp in components
    }
    
    valid_profiles = {
        comp: prof for comp, prof in profiles.items()
        if prof is not None
    }
    
    if not valid_profiles:
        raise RuntimeError(f"No profiles were created (tried {list(components)})")
    failed = set(components) - set(valid_profiles)
    if failed:
        logger.warning(f"Profile creation failed for: {sorted(failed)}")
    
    logger.info(
        f"Successfully created {len(valid_profiles)} profile(s): {list(valid_profiles.keys())}"
    )
    return valid_profiles


def enclosed_masses(sphere, radii, avail_components, logger):
    """
    Mass of each component enclosed within each radius.

    Each component is read from the sphere once; masses come from a cumulative
    sum over particles/cells sorted by radius (r <= R counts as inside, as in
    a yt sphere).

    Args:
        sphere: yt sphere around the galaxy centre enclosing max(radii).
        radii: Radii (kpc).
        avail_components: Components to sum, e.g. ['star', 'gas', 'dm'].
        logger: Logger instance.

    Returns:
        {"radii": radii, "M_<comp>": enclosed mass (Msun) per radius,
         "M_total": sum over components, if more than one}
    """
    logger.info("Calculating enclosed masses...")
    
    radii = np.asarray(radii)

    fields = {
        "star": (("star", "particle_radius"), ("star", "particle_mass")),
        "dm": (("dm", "particle_radius"), ("dm", "particle_mass")),
        "gas": (("index", "radius"), ("gas", "cell_mass")),
    }

    # initialise masses dict
    masses = {"radii": radii}
    for comp in avail_components:
        r_field, m_field = fields[comp]
        r = sphere[r_field].to("kpc").value
        m = sphere[m_field].to("Msun").value
        order = np.argsort(r)
        cum_m = np.concatenate([[0.0], np.cumsum(m[order])])
        masses[f"M_{comp}"] = cum_m[np.searchsorted(r[order], radii, side="right")]

    if len(avail_components) > 1:
        masses["M_total"] = np.sum([masses[f"M_{c}"] for c in avail_components], axis=0)

    return masses


# ---------------------------------------------------------------------------
# Saving and formatting helpers
# ---------------------------------------------------------------------------

def snapshot_data_dir(paths, halo_id, redshift, sim_type, experiment=None, output_dir=None):
    """Output folder: output_dir if given, else the snapshot data dir
    (experiment defaults to sim_type for non-full runs)."""
    if output_dir is not None:
        data_dir = Path(output_dir)
    else:
        exp = experiment or (sim_type if sim_type != "full" else None)
        data_dir = paths.snapshot_data(halo_id, redshift, experiment=exp)
    data_dir.mkdir(parents=True, exist_ok=True)
    return data_dir


def write_profiles(profiles, data_dir, logger):
    """Write each yt profile to <data_dir>/3d_profile_{comp}.h5."""
    for comp, profile in profiles.items():
        filename = data_dir / f"3d_profile_{comp}.h5"
        profile.save_as_dataset(str(filename))
        logger.info(f"Saved {comp} profile to {filename}")


def mass_radii(r200, config, logger):
    """Radii (kpc) for enclosed_masses: R200 fractions if range_scaling.units == 'r200', else config radii_list (<= R200)."""
    config = config or {}
    if config.get("range_scaling", {}).get("units", "kpc") == "r200":
        fractions = np.array([0.01, 0.02, 0.05, 0.1, 0.15, 0.25, 0.5, 0.75, 1.0])
        logger.info(f"Using R200-scaled mass radii: {fractions} x R200={r200:.1f} kpc")
        return fractions * r200
    radii = np.array(config.get("constants", {}).get("radii_list", [5, 10, 15, 20, 30, 50, 75, 100, 150, 200]), dtype=float)
    return radii[radii <= r200]


def build_metadata(
    halo_id, snapshot_info, center, r200, Rmats, l_hats,
    snap_path, masses, shape_tensors):
    """
    Metadata dict for one snapshot (saved as metadata.h5).

    rotation_matrix / angular_momentum_unit_vector hold the stellar frame
    (or the first component's if there are no stars), named in rotation_frame;
    every component's frame is under rotation_matrices / angular_momentum_unit_vectors.
    """
    main_comp = "star" if "star" in Rmats else next(iter(Rmats))

    metadata = {
        "halo_id": halo_id,
        **snapshot_info,  # redshift, time_Gyr, dd_num
        "center_kpc": center.to("kpc").value,
        "R200_kpc": r200,
        "rotation_matrix": Rmats[main_comp],
        "angular_momentum_unit_vector": l_hats[main_comp],
        "rotation_frame": main_comp,
        "rotation_matrices": Rmats,                # {comp: 3x3}
        "angular_momentum_unit_vectors": l_hats,   # {comp: (3,)}, incl. "total"
        "sim_path": str(snap_path),
        "analysis_timestamp": datetime.datetime.now().isoformat(),
        "enclosed_masses": masses,
    }
    if shape_tensors:
        metadata["shape_tensors"] = {
            "radii": np.array([r for r, _ in shape_tensors]),
            "tensors": np.array([t for _, t in shape_tensors]),
        }
    return metadata
