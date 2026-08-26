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
    rho_s = 1e9  # Msun/kpc^3
    r_s = 10.0  # kpc
    rho = rho_s / ((r / r_s) * (1 + r / r_s) ** 2)
    return {"r": r, "rho": rho, "rho_s": rho_s, "r_s": r_s}


@pytest.fixture
def mock_hernquist_profile():
    """Generate mock Hernquist density profile for testing."""
    r = np.logspace(-1, 2, 50)
    rho_0 = 1e9  # Msun/kpc^3
    a = 5.0  # kpc
    rho = rho_0 / ((r / a) * (1 + r / a) ** 3)
    return {"r": r, "rho": rho, "rho_0": rho_0, "a": a}


@pytest.fixture
def mock_exponential_disk():
    """Generate mock exponential disk profile for testing."""
    R = np.logspace(-1, 2, 50)
    Sigma_0 = 1e8  # Msun/kpc^2
    R_d = 3.0  # kpc
    Sigma = Sigma_0 * np.exp(-R / R_d)
    return {"R": R, "Sigma": Sigma, "Sigma_0": Sigma_0, "R_d": R_d}
