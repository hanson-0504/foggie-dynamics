# ============================================
# Analysis code to calculate the density profiles (bulge/disk/halo) of
# FOGGIE simulated galaxies
# ============================================

import os
import sys
import time
from pathlib import Path

import numpy as np
import yt

from . import density_profiles as dp
from . import filters_and_utils as fu
from .analysis import get_shape_tensor
from .io import save_hdf5

# Add project root to Python path so we can import from src
script_dir = Path(__file__).resolve().parent
project_root = script_dir.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))


def main():
    if len(sys.argv) != 2:
        print("Usage: python density_profiles.py <path_to_simulation>")
        sys.exit(1)

    t0 = time.time()

    sim_path = str(sys.argv[1])

    ds = yt.load(sim_path)  # type: ignore
    halo_id = sim_path.split("/")[-4]

    print(f"Loaded dataset: {halo_id}")

    ds.add_particle_filter("star")
    ds.add_particle_filter("dm")

    _, center_find_max = ds.find_max(("gas", "dark_matter_density"))
    print(f"Galaxy center (find_max method): {center_find_max}")

    ad = ds.all_data()
    pos = np.column_stack(
        [ad["star", "particle_position_x"].to("kpc").value,
         ad["star", "particle_position_y"].to("kpc").value,
         ad["star", "particle_position_z"].to("kpc").value,]
    )
    mass = ad["star", "particle_mass"].to("Msun").value
    center_iter, _ = dp.iterative_center(
        pos,
        mass,
        center_init=np.asarray(center_find_max.to("kpc").value),
        R_init=50.0,
        min_radius=0.1,
    )
    center_iter = ds.arr(center_iter, "kpc")
    delta = (center_find_max - center_iter).to("kpc")
    abs_diff = np.linalg.norm(delta)
    rel_diff = abs_diff / np.linalg.norm(center_find_max)
    print(f"Galaxy center (iterative method): {center_iter}")
    print(f"Comparison: change in method values (find_max - iterative) = {delta}")
    print(f"Absolute differecnce = {abs_diff:.3f}")
    print(f"Relative difference (find_max - iterative / find_max) = {rel_diff:.3%}")

    center = center_find_max.to("kpc")
    print(f"Galaxy center (Mpc): {center.to('Mpc')}")

    t1 = time.time()
    print(f"Time to get galactic center coordinates = {(t1 - t0) / 60:.2f} min")

    print("Calculating Virial Radius...")
    R200 = dp.find_virial_radius(ds, center.to("kpc"))
    t2 = time.time()
    print(f"Time to calculate virial radius = {(t2 - t1) / 60:.2f} min")

    if R200 is None:
        print("Warning: R200 > 1000 kpc, using 300 kpc for analysis")
        R200 = 300.0
    else:
        print(f"Using R200 = {R200:.1f} kpc for analysis")

    radius = ds.quan(R200, "kpc")
    sphere = ds.sphere(center.to("kpc"), radius)

    # Analyze shape at different radii
    radii_to_test = [10, 20, 30, 50]  # kpc

    print("=== Shape analysis at different radii ===")
    for r in radii_to_test:
        test_sphere = ds.sphere(center.to("kpc"), ds.quan(r, "kpc"))

        # Get positions relative to center
        star_positions = np.array(
            [
                (test_sphere["star", "particle_position_x"] - center[0])
                .to("kpc")
                .value,
                (test_sphere["star", "particle_position_y"] - center[1])
                .to("kpc")
                .value,
                (test_sphere["star", "particle_position_z"] - center[2])
                .to("kpc")
                .value,
            ]
        )
        star_weights = test_sphere["star", "particle_mass"].to("Msun").value

        print(f"\nRadius = {r} kpc:")
        shape_tensor, (a, b, c) = get_shape_tensor(star_weights, star_positions)

        # Interpret results
        axis_ratio_ba = b / a
        axis_ratio_ca = c / a
        print(f"\nShape analysis:")
        print(f"  Major axis (a): {a:.2f}")
        print(f"  Intermediate axis (b): {b:.2f}")
        print(f"  Minor axis (c): {c:.2f}")

        if axis_ratio_ca < 0.3 and axis_ratio_ba > 0.7:
            print("Likely disk-dominated (flattened, circular in-plane)")
        elif axis_ratio_ca < 0.5:
            print("Possibly disk + bulge system")
        else:
            print("Likely spheroidal/elliptical (not disk-dominated)")

    print("Calculating Angular Momentum...")
    # Calculate Angular Momentum and add cylinrical coordinate fields
    l = sphere.quantities.angular_momentum_vector(use_gas=True, use_particles=True)
    l_hat = l / np.linalg.norm(l)
    print(f"Angular Momentum unit vector = {l_hat}")

    Rmat = fu.rotation_matrix_from_vectors(np.array([0, 0, 1]), l_hat.d)

    # DEBUG: Check rotation matrix
    print(f"\n=== DEBUG: Rotation matrix ===")
    print(f"Rmat:\n{Rmat}")
    print(f"Rmat contains NaN: {np.any(np.isnan(Rmat))}")
    print(f"Rmat contains Inf: {np.any(np.isinf(Rmat))}")
    print(f"Rmat determinant: {np.linalg.det(Rmat):.6f} (should be ~1.0)")
    print(f"Rmat is orthogonal: {np.allclose(Rmat @ Rmat.T, np.eye(3))}")

    fu.add_rotated_fields(ds, Rmat, center.to("kpc"), comp="all")

    print("\n=== DEBUG: Testing derived fields ===")
    try:
        # Test DM fields
        R_disk_dm = sphere[("dm", "R_disk")]
        print(f"DM R_disk range: {R_disk_dm.min():.3e} to {R_disk_dm.max():.3e}")

        # Test star fields
        R_disk_star = sphere[("star", "R_disk")]
        print(f"Star R_disk range: {R_disk_star.min():.3e} to {R_disk_star.max():.3e}")

        # Test gas fields
        R_disk_gas = sphere[("gas", "R_disk")]
        print(f"Gas R_disk range: {R_disk_gas.min():.3e} to {R_disk_gas.max():.3e}")
    except Exception as e:
        print(f"ERROR: {e}")
        import traceback

        traceback.print_exc()

    print("\nGetting density profiles...")
    min_radius = ds.quan(0.1, "kpc")

    profiles = fu.create_profiles(sphere, min_radius, radius, comp="all")

    if not profiles:
        print("ERROR: Profile creation failed completely")
        sys.exit(1)

    # Filter out None values
    valid_profiles = {comp: prof for comp, prof in profiles.items() if prof is not None}

    if not valid_profiles:
        print("ERROR: No profiles were successfully created")
        sys.exit(1)

    print(
        f"\nSuccessfully created {len(valid_profiles)} profile(s): {list(valid_profiles.keys())}"
    )

    # Later when saving:
    data_dir = f"datasets/{halo_id}"
    os.makedirs(data_dir, exist_ok=True)

    for comp, profile in valid_profiles.items():
        filename = f"{data_dir}/3d_profile_{comp}.h5"
        profile.save_as_dataset(filename)
        print(f"Saved {comp} profile to {filename}")

    t3 = time.time()
    print(f"Time to get density profiles = {(t3 - t2) / 60:.2f} min")

    # Calculate actual simulation masses at various radii for comparison
    print("\nCalculating actual simulation masses...")
    radii_for_mass = np.array([5, 10, 15, 20, 30, 50, 75, 100, 150, 200])
    radii_for_mass = radii_for_mass[radii_for_mass <= R200]  # Only up to R200

    masses = {
        "radii": radii_for_mass,
        "M_star": [],
        "M_gas": [],
        "M_dm": [],
        "M_total": [],
    }

    for r in radii_for_mass:
        test_sphere = ds.sphere(center.to("kpc"), ds.quan(r, "kpc"))

        M_star = test_sphere[("star", "particle_mass")].sum().to("Msun").value
        M_gas = test_sphere[("gas", "cell_mass")].sum().to("Msun").value
        M_dm = test_sphere[("dm", "particle_mass")].sum().to("Msun").value
        M_total = M_star + M_gas + M_dm

        masses["M_star"].append(M_star)
        masses["M_gas"].append(M_gas)
        masses["M_dm"].append(M_dm)
        masses["M_total"].append(M_total)

        print(
            f"  R = {r:5.1f} kpc: M_star = {M_star:.2e}, M_gas = {M_gas:.2e}, M_dm = {M_dm:.2e}, M_total = {M_total:.2e}"
        )

    # Convert lists to arrays
    for key in ["M_star", "M_gas", "M_dm", "M_total"]:
        masses[key] = np.array(masses[key])

    # Save Any Extra data
    extra_data = {
        "halo_id": halo_id,
        "center_kpc": center.to("kpc").value,
        "R200_kpc": R200,
        "rotation_matrix": Rmat,
        "angular_momentum_unit_vector": l_hat.d,
        "sim_path": sim_path,
        "analysis_timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "enclosed_masses": masses,  # Add mass data
    }
    save_hdf5(f"{data_dir}/metadata", extra_data)

    t4 = time.time()
    print(f"\nTime to run script = {(t4 - t0) / 60:.2f} min")
    print("Done!")


if __name__ == "__main__":
    main()
