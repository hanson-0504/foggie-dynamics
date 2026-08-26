import numpy as np
import yt

from src.density_profiles import find_virial_radius


def test_find_virial_radius():
    """Test with a synthetic NFW halo using yt stream particles."""

    np.random.seed(42)  # For reproducibility

    # Create a particle dataset with NFW profile
    n_particles = 10000
    bbox = np.array([[-250, 250], [-250, 250], [-250, 250]])

    # Generate NFW halo centered at origin
    # rho(r) = rho_0 / ((r/r_s) * (1 + r/r_s)^2)
    r_s = 50.0  # scale radius in kpc
    M200 = 1e12  # solar masses

    def sample_nfw_particles(n, r_s, r_max=250):
        """Sample particle positions from NFW profile using rejection sampling."""
        positions = []
        while len(positions) < n:
            # Sample uniformly in a sphere
            r = np.random.uniform(0, r_max, size=n * 2)
            theta = np.random.uniform(0, np.pi, size=2 * n)
            phi = np.random.uniform(0, 2 * np.pi, size=2 * n)

            # NFW probability (normalized)
            c = r / r_s
            prob = 1.0 / (c * (1 + c)**2)
            prob /= prob.max()

            # Rejection sampling
            accept = np.random.uniform(0, 1, size=n * 2) < prob
            r_accepted = r[accept]
            theta_accepted = theta[accept]
            phi_accepted = phi[accept]

            # Convert to Cartesian
            x = r_accepted * np.sin(theta_accepted) * np.cos(phi_accepted)
            y = r_accepted * np.sin(theta_accepted) * np.sin(phi_accepted)
            z = r_accepted * np.cos(theta_accepted)

            positions.extend(list(zip(x, y, z)))

        return np.array(positions[:n])

    # Generate DM particle positions and masses
    dm_positions = sample_nfw_particles(n_particles, r_s)
    dm_masses = np.ones(n_particles) * (M200 / n_particles)

    # Create empty gas and star arrays (function expects all three components)
    empty_pos = np.zeros((1, 3))
    empty_mass = np.array([0.0])

    # Create stream particle dataset with dm, gas, and star particle types
    data = {
        ('dm', 'particle_position_x'): (dm_positions[:, 0], 'kpc'),
        ('dm', 'particle_position_y'): (dm_positions[:, 1], 'kpc'),
        ('dm', 'particle_position_z'): (dm_positions[:, 2], 'kpc'),
        ('dm', 'particle_mass'): (dm_masses, 'Msun'),
        ('gas', 'particle_position_x'): (empty_pos[:, 0], 'kpc'),
        ('gas', 'particle_position_y'): (empty_pos[:, 1], 'kpc'),
        ('gas', 'particle_position_z'): (empty_pos[:, 2], 'kpc'),
        ('gas', 'particle_mass'): (empty_mass, 'Msun'),
        ('star', 'particle_position_x'): (empty_pos[:, 0], 'kpc'),
        ('star', 'particle_position_y'): (empty_pos[:, 1], 'kpc'),
        ('star', 'particle_position_z'): (empty_pos[:, 2], 'kpc'),
        ('star', 'particle_mass'): (empty_mass, 'Msun'),
    }

    # Use yt.load_particles (stream dataset)
    from yt.loaders import load_particles
    ds = load_particles(data, length_unit='kpc', mass_unit='Msun', bbox=bbox)

    # Add derived field for gas cell_mass (maps to particle_mass)
    def _cell_mass(field, data):
        return data[('gas', 'particle_mass')]

    ds.add_field(
        ('gas', 'cell_mass'),
        function=_cell_mass,
        sampling_type='particle',
        units='Msun'
    )

    # Set cosmological parameters (these are attributes on the dataset)
    ds.current_redshift = 0.0
    ds.hubble_constant = 0.7
    ds.omega_matter = 0.3
    ds.omega_lambda = 0.7

    # Test virial radius finding
    center = ds.arr([0, 0, 0], 'kpc')
    R200 = find_virial_radius(ds, center, tol=0.05, max_iter=50)

    # For NFW, R200 should be in a reasonable range for M200 ~ 1e12
    assert 50 < R200 < 300, f"R200={R200:.1f} kpc outside expected range [50, 300]"

    # Verify we got a positive result
    assert R200 > 0, "R200 must be positive"

    print(f"✓ Test passed: R200 = {R200:.1f} kpc")
