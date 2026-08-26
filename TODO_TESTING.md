# Testing Infrastructure TODO

## Critical Priority: Add Testing

Currently **0% test coverage** for ~6,900 lines of code. This is the highest priority improvement.

## Quick Start

```bash
cd foggie-dynamics
pip install pytest pytest-cov
mkdir -p tests/{unit,integration,fixtures}
touch tests/__init__.py
```

## Recommended Test Structure

```
tests/
├── __init__.py
├── conftest.py           # Shared pytest fixtures
├── unit/
│   ├── __init__.py
│   ├── test_models.py    # Test density functions
│   ├── test_fitting.py   # Test MCMC fitting (with mocks)
│   ├── test_analysis.py  # Test model comparison, BIC/AIC
│   ├── test_config.py    # Test path management
│   └── test_density_profiles.py  # Test mass integration
├── integration/
│   ├── __init__.py
│   ├── test_workflow.py       # Test full fitting workflows
│   └── test_model_registry.py # Test registry pattern
└── fixtures/
    ├── test_data.h5           # Small test dataset
    └── test_config.yaml       # Test configuration
```

## Example Test Cases

### Unit Tests (test_models.py)
```python
import pytest
import numpy as np
from src.models import hernquist_density, ModelRegistry

def test_hernquist_at_scale_radius():
    """At r=rs, Hernquist density = rho0/4."""
    rho = hernquist_density(1.0, rho0=1e9, rs=1.0)
    assert np.isclose(rho, 1e9 / 4.0)

def test_hernquist_always_positive():
    """Density should always be positive."""
    r = np.logspace(-2, 2, 100)
    rho = hernquist_density(r, 1e9, 1.0)
    assert np.all(rho > 0)

def test_model_registry_completeness():
    """All models should have required components."""
    for model in ModelRegistry.list_models():
        assert ModelRegistry.get_fitter(model) is not None
        assert ModelRegistry.get_evaluator(model) is not None
        assert ModelRegistry.get_mass_integrator(model) is not None
```

### Integration Tests (test_workflow.py)
```python
import pytest
from unittest.mock import Mock, patch
from src.workflow import fit_stellar_components

@pytest.fixture
def mock_density_profile():
    """Create mock density profile data."""
    r = np.logspace(-1, 2, 50)
    rho = 1e9 * (1.0 / r) * (1.0 / (1 + r))**2  # NFW-like
    return {'r_1d': r, 'rho_1d': rho}

def test_fit_stellar_bulge_workflow(mock_density_profile, tmp_path):
    """Test full stellar fitting workflow."""
    with patch('src.workflow.plotting') as mock_plot:
        results = fit_stellar_components(
            r_1d=mock_density_profile['r_1d'],
            rho_1d=mock_density_profile['rho_1d'],
            # ... other params
            save_data=False  # Don't write files in tests
        )
        
        assert 'bulge' in results
        assert results['bulge']['best_model'] is not None
        assert 'samples' in results['bulge']
```

## Running Tests

```bash
# Run all tests
pytest

# Run with coverage report
pytest --cov=src --cov-report=html

# Run specific test file
pytest tests/unit/test_models.py

# Run specific test
pytest tests/unit/test_models.py::test_hernquist_at_scale_radius

# Run with verbose output
pytest -v
```

## Next Steps

1. Start with unit tests for `models.py` (easiest, most impactful)
2. Add fixtures for common test data
3. Mock file I/O in integration tests
4. Aim for >80% coverage of critical paths
5. Add CI/CD with GitHub Actions to run tests automatically

## Benefits

- Catch bugs before they affect research
- Confidence when refactoring
- Documentation of expected behavior
- Easier for collaborators to contribute
- Research reproducibility
