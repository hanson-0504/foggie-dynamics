"""Unit tests for density model functions in src/models.py"""

import numpy as np


def test_hernquist_at_scale_radius():
    """At r = rs, Hernquist density = rho0 / 8"""
    from src.models import hernquist_density
    rho = hernquist_density(1.0, 1e9, 1.0)
    assert np.isclose(rho, 1e9 / 8)


def test_nfw_at_scale_radius():
    from src.models import nfw_density
    rho = nfw_density(1.0, 1e9, 1.0)
    expected = 1e9 / 4
    assert np.isclose(rho, expected)


def test_model_registry_completeness():
    """All models should have required components."""
    from src.models import ModelRegistry
    for model in ModelRegistry.list_models():
        assert ModelRegistry.get_fitter(model) is not None
        assert ModelRegistry.get_evaluator(model) is not None
        assert ModelRegistry.get_mass_integrator(model) is not None

# TODO: test all model mass integrators against known values
