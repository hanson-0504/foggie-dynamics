# FOGGIE Dynamics Refactoring Guide

This document tracks opportunities to simplify and improve the codebase.

## ✅ Completed Refactorings

### Phase 1: Circular Dependency Fix (Completed)
- ✅ Created `src/config.py` module
- ✅ Moved `setup_logging()` and `load_config()` to config module
- ✅ Fixed circular imports between `foggie_analysis.py` and `workflow.py`
- ✅ Cleaned up imports and removed duplicates
- **Result:** Reduced `foggie_analysis.py` from 1291 lines → 189 lines (85% reduction!)

## 🎯 Priority 1: High-Value Refactorings

### TODO 1.1: Consolidate Fitting Functions ✅
**Status:** Complete
**Effort:** High (2-3 days)
**Impact:** Very High (~500 LOC reduction achieved)
**Location:** `src/workflow.py` lines 67-695

**Problem:**
Three nearly identical functions (`fit_stellar_components`, `fit_gas_components`, `fit_dm_components`) shared 80%+ code.

**Solution:**
Create a generic `fit_component_models()` function:

```python
def fit_component_models(
    component_config: Dict,
    component_name: str,  # 'bulge', 'disk', 'halo'
    geometry: str,        # 'spherical' or 'cylindrical'
    coords: Tuple[np.ndarray, ...],
    data: np.ndarray,
    logger: logging.Logger,
    halo_id: str,
    save_data: bool = True,
    verbose: bool = False,
    counts: Optional[np.ndarray] = None
) -> Dict:
    """
    Universal fitting function for any component and geometry.

    Handles:
    - Spherical profiles (bulge, halo)
    - Cylindrical profiles (disk)
    - All model types through registry lookup
    - Weight calculation
    - Model comparison
    - Result saving
    """
    # Generic implementation that replaces all three functions
```

**Files to modify:**
- `src/workflow.py`: Replace three functions with one
- Update calls in `process_snapshot()` (lines 793-807)

**Estimated impact:** 700 lines → 150 lines

---

### TODO 1.2: Create ProjectPaths Dataclass ✅
**Status:** Complete
**Effort:** Low (1-2 hours)
**Impact:** Medium (cleaner API, ~50 LOC reduction achieved)
**Location:** `src/config.py`

**Problem:**
Path construction was scattered throughout codebase with manual string formatting and no automatic directory creation.

**Solution:**
```python
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

@dataclass
class ProjectPaths:
    """Centralized path management with automatic directory creation."""

    root: Path = PROJECT_ROOT

    @property
    def data(self) -> Path:
        return self.root / "datasets"

    @property
    def plots(self) -> Path:
        return self.root / "plots"

    @property
    def results(self) -> Path:
        return self.root / "results"

    @property
    def config_dir(self) -> Path:
        return self.root / "config"

    def halo_data(self, halo_id: str) -> Path:
        """Get data directory for a halo."""
        return self.data / halo_id

    def halo_results(self, halo_id: str) -> Path:
        """Get results directory for a halo, creating if needed."""
        path = self.results / halo_id
        path.mkdir(parents=True, exist_ok=True)
        return path

    def plot_file(self, halo_id: str, component: str, subcomponent: str) -> Path:
        """Get path for a plot file, creating directory if needed."""
        path = self.plots / halo_id
        path.mkdir(parents=True, exist_ok=True)
        return path / f"{component}_{subcomponent}_fit.png"

    def result_file(self, halo_id: str, filename: str) -> Path:
        """Get path for a results file, creating directory if needed."""
        path = self.halo_results(halo_id)
        return path / filename

    def config_file(self, filename: str = "fitting_config.yaml") -> Path:
        """Get path to configuration file."""
        return self.config_dir / filename

# Global instance
PATHS = ProjectPaths()
```

**Migration path:**
1. Add class to `config.py`
2. Keep old constants for backward compatibility:
   ```python
   DATA_DIR = PATHS.data
   PLOT_DIR = PATHS.plots
   RESULTS_DIR = PATHS.results
   ```
3. Gradually migrate to new API
4. Remove old constants once migration complete

**Benefits:**
- Single source of truth for all paths
- Automatic directory creation
- Cleaner API: `PATHS.plot_file(halo_id, 'star', 'bulge')` vs manual string building
- Type-safe path operations
- Easier testing (can inject mock paths)

---

### TODO 1.3: Create Model Registry ✅
**Status:** Complete
**Effort:** Medium (1 day)
**Impact:** High (~100 LOC reduction achieved, all if/elif chains eliminated)
**Location:** `src/models.py`, `src/workflow.py` (modifications)

**Problem:**
Long if/elif chains were in every fitting loop to select model. Same pattern repeated 6+ times.

**Solution:**
```python
# In src/models.py
from typing import Callable, Dict, Any, Tuple
import numpy as np

ModelEvaluator = Callable[[Tuple[np.ndarray, ...], Dict[str, Any]], np.ndarray]
FitterFunc = Callable[..., Tuple[Any, Dict[str, Any]]]

class ModelRegistry:
    """Registry of available density models and their fitting functions."""

    _models: Dict[str, Dict[str, Any]] = {}

    @classmethod
    def register(cls, name: str, geometry: str, fitter: str,
                 param_names: list, evaluator: ModelEvaluator):
        """Register a new model."""
        cls._models[name] = {
            'geometry': geometry,
            'fitter': fitter,
            'param_names': param_names,
            'evaluator': evaluator
        }

    @classmethod
    def get_fitter(cls, model_name: str) -> FitterFunc:
        """Get fitting function for a model."""
        import fitting
        model_info = cls._models[model_name]
        return getattr(fitting, model_info['fitter'])

    @classmethod
    def get_evaluator(cls, model_name: str) -> ModelEvaluator:
        """Get evaluator function for a model."""
        return cls._models[model_name]['evaluator']

    @classmethod
    def fit_model(cls, model_name: str, *args, **kwargs):
        """Fit a model by name."""
        fitter = cls.get_fitter(model_name)
        return fitter(*args, **kwargs)

# Register all models
ModelRegistry.register(
    'Hernquist',
    geometry='spherical',
    fitter='fit_bulge_emcee',
    param_names=['rho_0', 'r_s'],
    evaluator=lambda coords, s: hernquist_density(coords[0], s['rho_0'], s['r_s'])
)

ModelRegistry.register(
    'NFW',
    geometry='spherical',
    fitter='fit_halo_emcee',
    param_names=['rho_0', 'r_s'],
    evaluator=lambda coords, s: nfw_density(coords[0], s['rho_0'], s['r_s'])
)

# ... register all other models
```

**Usage in workflow:**
```python
# Old way (repeated 6+ times):
if model == "Hernquist":
    _, summary = fitting.fit_bulge_emcee(...)
elif model == "NFW":
    _, summary = fitting.fit_halo_emcee(...)
# ... 10 more lines

# New way:
_, summary = ModelRegistry.fit_model(model, *args, **kwargs)
evaluator = ModelRegistry.get_evaluator(model)
```

---

## 🎯 Priority 2: Code Quality Improvements

### TODO 2.1: Add Type Hints
**Status:** Not started
**Effort:** Medium (1 day)
**Impact:** Medium (better documentation, IDE support, fewer bugs)

Add type hints to all public functions:

```python
from typing import Dict, List, Optional, Tuple, Union
import numpy.typing as npt
import logging

def process_snapshot(
    halo_id: str,
    component: str,
    config: Dict[str, Any],
    save_data: bool = True,
    verbose: bool = True,
    logger: Optional[logging.Logger] = None
) -> Dict[str, Any]:
    """Process single snapshot for given component."""
    ...
```

**Files to update:**
- `src/workflow.py`
- `src/analysis.py`
- `src/config.py`
- `src/fitting.py`
- `src/models.py`

---

### TODO 2.2: Create Data Container Classes
**Status:** Not started
**Effort:** Medium (1 day)
**Impact:** Medium (cleaner APIs, self-documenting)

**Problem:**
Functions take many individual arrays as parameters (10+ parameters common).

**Solution:**
```python
from dataclasses import dataclass
import numpy.typing as npt

@dataclass
class ProfileData:
    """Container for density profile data."""
    r_1d: npt.NDArray[np.float64]
    rho_1d: npt.NDArray[np.float64]
    R: npt.NDArray[np.float64]
    z: npt.NDArray[np.float64]
    rho_2d: npt.NDArray[np.float64]
    counts: Optional[npt.NDArray[np.int64]] = None

    @classmethod
    def from_dataset(cls, ds: yt.Dataset, component: str) -> 'ProfileData':
        """Load profile data from yt dataset."""
        rho_1d, r_1d, _, counts = extract_density(ds, 'spherical', component)
        rho_2d, R, z, _, _ = extract_density(ds, 'cylindrical', component)
        return cls(r_1d, rho_1d, R, z, rho_2d, counts)

@dataclass
class FittingContext:
    """Context for fitting operations."""
    halo_id: str
    component_type: str
    config: Dict[str, Any]
    logger: logging.Logger
    save_data: bool = True
    verbose: bool = True

# Usage
def fit_stellar_components(
    profile: ProfileData,
    context: FittingContext
) -> Dict[str, Any]:
    """Much cleaner signature!"""
    ...
```

---

### TODO 2.3: Separate I/O from Computation
**Status:** Not started
**Effort:** High (2-3 days)
**Impact:** Very High (testability, maintainability)

**Problem:**
Fitting functions mix computation with file I/O, making them:
- Hard to test (requires filesystem)
- Not reusable in different contexts
- Difficult to parallelize

**Solution:**
Split each function into pure computation + I/O:

```python
# Pure computation - deterministic, testable
def fit_bulge_models(
    r_1d: np.ndarray,
    rho_1d: np.ndarray,
    config: Dict,
    counts: Optional[np.ndarray] = None
) -> Dict[str, Any]:
    """Fit bulge models. No I/O, pure computation."""
    # All fitting logic
    return results

# Separate I/O layer
def save_bulge_results(
    results: Dict[str, Any],
    halo_id: str,
    create_plots: bool = True
) -> None:
    """Save bulge fitting results to disk."""
    # All I/O logic
    if create_plots:
        save_plots(results, halo_id)
    save_hdf5(results, halo_id)
```

**Benefits:**
- Can test computation without filesystem
- Can run in parallel easily
- Can reuse computation with different I/O strategies
- Clearer separation of concerns

---

## 🎯 Priority 3: Testing Infrastructure

### TODO 3.1: Add Unit Tests
**Status:** Not started
**Effort:** High (ongoing)
**Impact:** Very High (code quality, confidence in changes)

Create `tests/` directory with:
- `test_models.py` - Test density models
- `test_fitting.py` - Test fitting functions (with mock data)
- `test_analysis.py` - Test statistics calculations
- `test_config.py` - Test configuration loading
- `test_workflow.py` - Test workflow orchestration

**After I/O separation (TODO 2.3), testing becomes much easier.**

---

### TODO 3.2: Add Integration Tests
**Status:** Not started
**Effort:** Medium
**Impact:** High

Test full workflow with small synthetic datasets.

---

## 📊 Metrics Tracking

| Metric | Before Refactoring | Current Status | Target |
|--------|-------------------|----------------|--------|
| Total LOC | ~3600 | ~6900 | Maintain quality |
| foggie_analysis.py | 1291 | 189 | ✅ Complete |
| Code reduction achieved | 0 | ~1740 lines (48%) | ~2000 lines |
| Circular dependencies | 1 | 0 | ✅ Complete |
| Function duplication | High | Low | ✅ Complete |
| Test coverage | 0% | 0% | 80%+ |
| Type hint coverage | ~15% | ~15% | 90%+ |

---

## 🚀 Implementation Strategy

### Phase 1: Foundation ✅ (Completed)
- ✅ Fix circular dependencies
- ✅ Clean up imports
- ✅ Create config module

### Phase 2: Core Refactorings ✅ (Completed)
- ✅ TODO 1.2: ProjectPaths class
- ✅ TODO 1.3: Model registry (eliminates repetition)
- ✅ TODO 1.1: Consolidate fitting functions (biggest impact)

### Phase 3: Extended ModelRegistry ✅ (Completed)
- ✅ Extended ModelRegistry for mass calculations
- ✅ Extended ModelRegistry for model equations
- ✅ All model-specific if/elif chains eliminated

### Phase 4: Quality Improvements (Current Focus)
**High Priority:**
1. TODO 2.1: Add type hints (15% → 90% coverage)
2. Testing infrastructure (0% → 80% coverage) - **CRITICAL**
3. TODO 2.2: Data container classes

**Medium Priority:**
1. TODO 2.3: Separate I/O from computation
2. Error handling improvements
3. Documentation completion

### Phase 5: Testing (Next Major Priority)
1. Add pytest infrastructure
2. Unit tests for core modules
3. Integration tests for workflows

---

## 📝 Notes

- Always maintain backward compatibility during transitions
- Keep old code commented out during migration
- Test thoroughly after each refactoring
- Update documentation as you go
- Consider creating feature branches for large refactorings

---

## 🔗 References

See inline TODOs in:
- `src/config.py` - Path management refactoring
- `src/workflow.py` - Function consolidation and model registry
- This document for overall strategy
