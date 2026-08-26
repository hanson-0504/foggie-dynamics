import logging

import numpy as np
import pytest

from src.workflow import DensityProfile, fit_stellar_components


@pytest.fixture
def mock_density_profile():
    """Create mock density profile data."""
    r = np.logspace(-1, 2, 50)
    rho = 1e9 * (1.0 / r) * (1.0 / (1 + r))**2  # NFW-like

    # Create 2D arrays (minimal for testing)
    R = np.linspace(0, 10, 20)
    z = np.linspace(-5, 5, 20)
    R_2d, z_2d = np.meshgrid(R, z)
    rho_2d = np.zeros_like(R_2d)

    return DensityProfile(
        r_1d=r,
        rho_1d=rho,
        r_bins=r,
        R=R_2d.flatten(),
        z=z_2d.flatten(),
        rho_2d=rho_2d.flatten(),
        R_bins=R,
        z_bins=z
    )


@pytest.fixture
def mock_config():
    """Create minimal config for testing."""
    return {
        'star': {
            'bulge': {
                'enabled': True,
                'r_range': [0.5, 50],
                'models': ['NFW'],
                'bounds': {
                    'log10_rho0': [5.0, 12.0],
                    'log10_rs': [-1.0, 2.0],
                    'log_sigma': [-5.0, 0.0]  # Required for scatter parameter
                },
                'p0': {
                    'log10_rho0': 8.0,
                    'log10_rs': 0.5,
                    'log_sigma': -2.0
                },
                'fitting': {
                    'nsteps': 2000,  # Enough for burn-in + sampling
                    'nwalkers': 24,
                    'nburn': 200,  # Reduced burn-in for faster testing
                    'use_scatter': True
                }
            },
            'disk': {'enabled': False},
            'halo': {'enabled': False}
        },
        'constants': {
            'fixed_uncertainty': 0.1
        },
        'plotting': {
            'default': {'dpi': 150},
            'profile_1d': {'figsize': [10, 8]},
            'corner': {'dpi': 200}
        }
    }


def test_fit_stellar_bulge_workflow(mock_density_profile, mock_config):
    """Test full stellar fitting workflow."""
    logger = logging.getLogger('test')

    results = fit_stellar_components(
        profile=mock_density_profile,
        config=mock_config,
        logger=logger,
        verbose=False,
        save_data=False,  # Don't write files in tests
        halo_id='test_halo',
        n_jobs=1
    )

    assert 'bulge' in results
    assert results['bulge'] is not None
    assert results['bulge'].best_model is not None
