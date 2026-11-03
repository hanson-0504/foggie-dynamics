"""
pytest configuration and shared fixtures for FOGGIE Dynamics tests.

This file is automatically loaded by pytest and makes the src package
importable without manual sys.path manipulation in individual test files.
"""

import sys
from pathlib import Path

import numpy as np
import pytest

# Add project root to Python path so tests can import from src/
PROJECT_ROOT = Path(__file__).parent.parent.resolve()
sys.path.insert(0, str(PROJECT_ROOT))


# Shared fixtures for test data
@pytest.fixture
def mock_nfw_profile():
    """Generate mock NFW density profile for testing."""
    r = np.logspace(-1, 2, 50)
    rho0 = 1e9  # Msun/kpc^3
    rs = 10.0  # kpc
    rho = rho0 / ((r / rs) * (1 + r / rs) ** 2)
    return {"r": r, "rho": rho, "rho0": rho0, "rs": rs}


@pytest.fixture
def mock_hernquist_profile():
    """Generate mock Hernquist density profile for testing."""
    r = np.logspace(-1, 2, 50)
    rho0 = 1e9  # Msun/kpc^3
    rs = 5.0  # kpc
    rho = rho0 / ((r / rs) * (1 + r / rs) ** 3)
    return {"r": r, "rho": rho, "rho0": rho0, "rs": rs}


@pytest.fixture
def mock_exponential_disk():
    """Generate mock exponential disk profile for testing."""
    R = np.logspace(-1, 2, 50)
    Sigma0 = 1e8  # Msun/kpc^2
    Rd1 = 3.0  # kpc
    Sigma = Sigma0 * np.exp(-R / Rd1)
    return {"R": R, "Sigma": Sigma, "Sigma0": Sigma0, "Rd1": Rd1}
