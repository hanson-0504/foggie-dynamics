# FOGGIE Dynamics TODO Checklist

Quick reference for refactoring tasks. See `REFACTORING.md` for detailed plans.

## ✅ Completed

### Phase 1: Foundation

- [x] Fix circular dependency (config.py created)
- [x] Clean up imports
- [x] Reduce foggie_analysis.py from 1291→189 lines

### Phase 2: Core Refactorings (COMPLETE! 🎉)

- [x] **TODO 1.2**: Create ProjectPaths dataclass in `src/config.py`
  - Clean API for path management
  - Automatic directory creation

- [x] **TODO 1.3**: Create ModelRegistry in `src/models.py`
  - Eliminate all if/elif chains for model selection
  - ~100 LOC reduction

- [x] **TODO 1.1**: Consolidate three fitting functions ⭐ **JUST COMPLETED!**
  - Created `fit_component_models()` - universal fitting function
  - Refactored `fit_stellar_components()` - 350→75 lines (79% reduction)
  - Refactored `fit_gas_components()` - 200→52 lines (74% reduction)
  - Refactored `fit_dm_components()` - 100→14 lines (86% reduction)
  - **Total: ~500 LOC reduction achieved!**

### Phase 3: Extended ModelRegistry Features (COMPLETE! 🎉)

- [x] **TODO 1.4**: Extend ModelRegistry to handle mass calculations ⭐ **JUST COMPLETED!**
  - Refactored `calculate_model_masses()` to use ModelRegistry
  - Eliminated all model-specific if/elif chains (~80 lines → ~40 lines)
  - Added `mass_integrator` to all model registrations
  - **Impact**: 50% reduction + complete elimination of hardcoded model logic

- [x] **TODO 1.5**: Extend ModelRegistry to handle model equations ⭐ **ALREADY COMPLETE!**
  - All models registered with `profile_eq` parameter
  - Used by `profile_eq_str()` in `filters_and_utils.py`
  - Eliminates need for model-specific formatting logic

### Phase 4: I/O Consolidation & Data Structures (COMPLETE! 🎉)

- [x] **TODO 2.2**: Create DensityProfile dataclass ⭐ **COMPLETED!**
  - Created `DensityProfile` dataclass in `src/workflow.py:99-110`
  - Fields: `r_1d`, `rho_1d`, `r_bins`, `R`, `z`, `rho_2d`, `R_bins`, `z_bins`, `counts`
  - Used throughout `workflow.py` and `plotting.py`
  - **Impact**: Structured data handling, better type safety

- [x] **TODO 2.2**: Create FittingResult dataclass ⭐ **COMPLETED!**
  - Created `FittingResult` dataclass in `src/workflow.py:126-132`
  - Fields: `best_model`, `samples`, `summary`, `model_comparison`
  - Properly exported in `__init__.py`
  - **Impact**: Clean return values for fitting functions, better type safety

- [x] **TODO 2.3**: Consolidate I/O operations ⭐ **COMPLETED!**
  - Created `src/io.py` with centralized I/O utilities
  - Eliminated ~8 duplicated `mkdir()` calls → `ensure_directory()`
  - Consolidated ~55 lines of HDF5 logic → `save_hdf5()` / `load_hdf5()`
  - Unified 7 `savefig()` calls → `save_figure()`
  - Automatic directory creation for all file operations
  - Consistent logging for all I/O operations
  - **Files updated**: `config.py`, `analysis.py`, `workflow.py`, `filters_and_utils.py`, `plotting.py`
  - **Impact**: ~70 LOC reduction + improved maintainability

- [x] **TODO**: Restructure generate_snapshot_profiles() ⭐ **COMPLETED!**
  - Updated to return `Dict[str, DensityProfile]` instead of raw yt profiles
  - Created `plot_galaxy_density_profile()` for unified component visualization
  - Better separation of concerns (data loading vs plotting)
  - **Impact**: Cleaner interface, more testable code

- [x] **TODO: Modify process_snapshot to support yt dataset files and DensityProfile objects** ⭐ **COMPLETED!**
  - **Location**: [src/workflow.py:936-940](src/workflow.py#L936-L940)
  - Updated `process_snapshot()` to accept either:
    - A yt dataset file path (load and generate profiles)
    - An existing DensityProfile object (skip profile generation)
  - **Benefit**: More flexible workflow, can reuse profiles across analyses
  - **Impact**: Improved code reusability and flexibility

## 🎯 High Priority (Do Next)

### 🔴 CRITICAL: Testing Infrastructure ⚠️

**Status**: Not started | **Impact**: Critical | **Effort**: Medium (2-3 days)

Currently **0% test coverage** for ~6,900 lines of code - highest priority improvement!

- [ ] **TODO: Add Testing Infrastructure** (see [TODO_TESTING.md](TODO_TESTING.md))
  - Install pytest and pytest-cov
  - Create `tests/` directory structure
  - Write unit tests for `src/models.py` (density functions)
  - Write unit tests for `src/fitting.py` (MCMC functions)
  - Add integration tests for workflow
  - Set up GitHub Actions CI/CD
  - **Quick start**: `pip install pytest pytest-cov && mkdir -p tests/{unit,integration,fixtures}`
  - **Reference**: [src/workflow.py:41](src/workflow.py#L41)

### 🔴 Error Handling Fixes

- [ ] **TODO: Fix bare Exception catches**
  - **Location**: [src/fitting.py:184-191](src/fitting.py#L184-L191)
  - **Issue**: Catches KeyboardInterrupt and SystemExit
  - **Fix**: Replace with `except (ValueError, TypeError, ArithmeticError)`
  - **Also fix**: [src/workflow.py:182-183](src/workflow.py#L182-L183)

- [ ] **TODO: Create custom exception hierarchy** (see below in Medium Priority)
  - Create `src/exceptions.py` with FoggeDynamicsError, ModelFitError, DataError
  - **Reference**: [src/workflow.py:56](src/workflow.py#L56)

### 🟡 Type Hints

- [ ] **TODO 2.1**: Add type hints to public functions
  - **Current coverage**: ~15% | **Target**: 90%
  - Better IDE support and documentation
  - **Priority files**: `src/workflow.py`, `src/analysis.py`, `src/plotting.py`
  - Install mypy: `pip install mypy`
  - **Reference**: [src/fitting.py:8](src/fitting.py#L8), [src/workflow.py:36](src/workflow.py#L36)

### Recent Bug Fixes (Completed 2025-11-23)

- [x] Fixed circular import (density_profiles.py → models.py)
- [x] Fixed HDF5 group access error (added Dataset type checking)
- [x] Fixed module import error (relative imports)
- [x] Added Two-Power model registration
- [x] Fixed parameter name mismatches (Exp-Cutoff, Disk-Flaring, Two-Power)
- [x] Added component type to plot filenames (`{component_type}_{component_name}_fit.png`)

### Documentation

- [x] **Updated CLAUDE.md** with comprehensive project documentation:
  - ModelRegistry pattern explanation
  - Fitting workflow documentation
  - Adding new models guide
  - Log analysis workflow
  - Common error patterns and fixes

- [ ] **TODO: Add docstrings to density models**
  - **Location**: [src/models.py:480-636](src/models.py#L480-L636)
  - Functions like `two_power_model`, `cored_power_model`, `disk_flaring` need docstrings
  - Include parameter descriptions and units

## 📋 Medium Priority (Later)

### Configuration & Data Structures

- [ ] **TODO: Extract magic numbers to constants**
  - **Status**: Not started | **Impact**: Medium | **Effort**: Low (2-3 hours)
  - **Create**: `src/constants.py`
  - **Examples**:
    - [src/fitting.py:194](src/fitting.py#L194): `0.05` (systematic uncertainty floor)
    - [src/workflow.py:374](src/workflow.py#L374): `z_max=5` (vertical integration limit)
    - Default MCMC parameters: nwalkers=64, nsteps=4000, nburn=1000
  - **Reference**: [src/workflow.py:55](src/workflow.py#L55)

  ```python
  # src/constants.py
  SYSTEMATIC_UNCERTAINTY_FLOOR = 0.05  # 5% systematic floor (dex)
  DEFAULT_Z_MAX_KPC = 5.0  # Default vertical integration limit
  DEFAULT_NWALKERS = 64
  DEFAULT_NSTEPS = 4000
  ```

- [ ] **TODO: Implement Pydantic config validation**
  - **Status**: Not started | **Impact**: Medium | **Effort**: Medium (half day)
  - **Location**: [src/config.py:69-72](src/config.py#L69-L72)
  - Install: `pip install pydantic`
  - Create `ComponentConfig`, `StarConfig`, `GasConfig`, `DMConfig`, `FittingConfig` classes
  - Add field validators for ranges, models
  - **Benefit**: Catch config errors before running expensive MCMC

  ```python
  from pydantic import BaseModel, field_validator

  class ComponentConfig(BaseModel):
      models: List[str]
      r_range: Tuple[float, float]
      bounds: Dict[str, Tuple[float, float]]

      @field_validator('r_range')
      @classmethod
      def validate_range(cls, v):
          if v[0] >= v[1]:
              raise ValueError("r_min must be < r_max")
          return v
  ```

- [x] ~~**TODO 2.2**: Create data container classes~~ ✅ **COMPLETED!**
  - ✅ `DensityProfile` dataclass created in Phase 4
  - ✅ `FittingResult` dataclass created (see Phase 4)

### Architecture Improvements

- [x] ~~**TODO 2.3**: Separate I/O from computation~~ ✅ **PARTIALLY COMPLETED!**
  - ✅ Created `src/io.py` with generic I/O utilities (see Phase 4)
  - ✅ `generate_snapshot_profiles()` returns `DensityProfile` objects
  - ⚠️ Still TODO: Make `process_snapshot()` fully pure (remove side effects)
  - ⚠️ Still TODO: Consider extracting I/O from `save_results()` and `save_evolution_data()`

  ```python
  # ✅ Already achieved:
  # - ensure_directory(), save_hdf5(), load_hdf5(), save_figure()
  # - generate_snapshot_profiles() returns data structures

  # ⚠️ Future improvement:
  # Make process_snapshot() pure by extracting I/O to caller
  ```

- [ ] **TODO: Create custom exception hierarchy**
  - **Status**: Not started | **Impact**: Medium | **Effort**: Low (1-2 hours)
  - **Create**: `src/exceptions.py`
  - Define FoggeDynamicsError, ModelFitError, DataError, ConvergenceError
  - Update error handling to use custom exceptions

  ```python
  # src/exceptions.py
  class FoggeDynamicsError(Exception):
      """Base exception for FOGGIE dynamics."""
      pass

  class ModelFitError(FoggeDynamicsError):
      """Raised when model fitting fails."""
      pass

  class DataError(FoggeDynamicsError):
      """Raised when data is invalid or missing."""
      pass
  ```

### Code Quality Improvements

- [ ] **TODO 2.4**: Consolidate repeated logging patterns
  - **Issue**: Similar "=" * 70 separator patterns repeated ~15 times
  - **Solution**: Create logging utility functions

    ```python
    def log_section(logger, title):
        logger.info("=" * 70)
        logger.info(title)
        logger.info("=" * 70)
    ```

- [ ] **TODO 2.5**: Create HDF5 I/O helper class
  - **Issue**: 21+ create_dataset/create_group calls scattered across files
  - **Solution**: Encapsulate HDF5 operations

    ```python
    class HDF5Writer:
        def save_model_results(self, file_path, summaries, best_model, table):
            ...
        def save_evolution_data(self, file_path, evolution_data):
            ...
    ```

### Performance & Cluster Preparation

- [x] **TODO: Add memory profiling for cluster job sizing** ⭐ **PARTIALLY COMPLETED!**
  - **Status**: Infrastructure complete, analysis pending | **Impact**: High | **Effort**: Low (1-2 hours remaining)
  - **Purpose**: Determine memory requirements for cluster job scripts
  - **Completed**:
    1. ✅ Added `psutil` process monitoring to track memory usage
    2. ✅ Added `tracemalloc` for detailed memory profiling
    3. ✅ Created `log_memory_usage()` function in `src/workflow.py`
  - **Remaining**:
    1. ⚠️ Run comprehensive analysis on single halo to establish baseline
    2. ⚠️ Document peak memory usage for different components (star/gas/dm)
    3. ⚠️ Create cluster job script template with appropriate memory requests

  **Implementation steps:**

  ```python
  # Add to code/foggie_analysis.py or src/workflow.py
  import psutil
  import tracemalloc
  import os

  def log_memory_usage(logger, stage: str):
      """Log current and peak memory usage."""
      process = psutil.Process(os.getpid())
      mem_gb = process.memory_info().rss / 1024**3
      logger.info(f"Memory at {stage}: {mem_gb:.2f} GB")

      if tracemalloc.is_tracing():
          current, peak = tracemalloc.get_traced_memory()
          logger.info(f"Peak traced memory: {peak / 1024**3:.2f} GB")

  # Use at key stages:
  # - After loading dataset
  # - After all_data()
  # - After each component fitting
  # - After plot generation
  ```

  **Tools to install:**
  - `pip install psutil memory_profiler`
  - For detailed profiling: `mprof run python code/foggie_analysis.py --halo 8508 --component star`

  **Cluster considerations:**
  - yt datasets with `all_data()` can be memory-intensive
  - MCMC chains: memory = n_walkers × n_steps × n_params × 8 bytes
  - Corner plots require keeping full chain in memory
  - Recommend 1.5-2x safety margin for job script memory requests

- [ ] **TODO: Optimize memory usage for cluster runs**
  - **Status**: Not started | **Impact**: Medium | **Effort**: Medium (1 day)
  - **Goal**: Reduce memory footprint to allow more parallel jobs
  - **Strategies**:
    - Replace `ds.all_data()` with targeted selections (`ds.sphere()`, `ds.disk()`)
    - Stream MCMC samples to disk instead of holding in memory
    - Generate corner plots on-demand rather than for all models
    - Process halos sequentially rather than loading multiple datasets
    - Use yt's derived fields lazily

- [ ] **TODO: Create cluster job script templates**
  - **Status**: Not started | **Impact**: High | **Effort**: Low (1-2 hours)
  - **Create**: `cluster/job_templates/`
  - **Templates needed**:
    - Single halo analysis (test run)
    - Array job for multiple halos
    - High-memory job for large datasets

  ```bash
  # cluster/job_templates/single_halo.sh
  #!/bin/bash
  #SBATCH --job-name=foggie_{halo_id}
  #SBATCH --mem=32G              # Based on profiling results
  #SBATCH --time=04:00:00        # Adjust after timing tests
  #SBATCH --cpus-per-task=8      # For MCMC parallelization
  #SBATCH --output=logs/%x_%j.out

  module load anaconda
  conda activate foggie

  python code/foggie_analysis.py --halo ${HALO_ID} --component ${COMPONENT}
  ```

- [x] **TODO: Add multiprocessing support for model fitting** ⭐ **COMPLETED!**
  - **Status**: Complete | **Impact**: High | **Effort**: N/A
  - **Location**: [src/workflow.py](src/workflow.py) (fit_models_parallel, _fit_single_model_worker)
  - **Implemented**: Parallel model fitting within components using `ProcessPoolExecutor`
  - **Features**:
    - Configurable parallelism via `n_jobs` parameter (1=serial, -1=all cores)
    - Automatic fallback to serial on errors
    - Memory logging per model fit
    - Proper worker function (_fit_single_model_worker) for pickling
  - **Impact**: Significant speedup when fitting multiple models for same component

- [ ] **TODO: Add multiprocessing support for multiple halos** ⭐ **RECENTLY ADDED MULTI-HALO SUPPORT**
  - **Status**: Sequential implementation complete | **Impact**: High | **Effort**: Medium (half day)
  - **Location**: [code/foggie_analysis.py:99](code/foggie_analysis.py#L99)
  - **Current**: Script now accepts multiple halos via `--halo 8508 5036 5016` but processes sequentially
  - **Goal**: Add parallel processing across halos using `multiprocessing` or `concurrent.futures`
  - **Note**: Parallel model fitting within components is already implemented (see above)
  - **Considerations**:
    - Each halo analysis is independent (embarrassingly parallel)
    - Memory usage per process (use memory profiling results above)
    - Number of workers should be configurable (default: `min(n_halos, cpu_count-1)`)
    - Handle logging from multiple processes (use `QueueHandler` or separate log files)
    - Cluster environments may prefer sequential (submit array jobs instead)

  **Implementation approach:**

  ```python
  from concurrent.futures import ProcessPoolExecutor
  from multiprocessing import cpu_count

  def analyze_single_halo(halo_id, component, config, save_data, show_plots):
      """Wrapper function for multiprocessing."""
      logger = setup_logging(halo_id, component)
      # ... existing processing logic
      return halo_id, results

  # In main():
  parser.add_argument('--parallel', action='store_true',
                      help='Process halos in parallel using multiprocessing')
  parser.add_argument('--n-jobs', type=int, default=None,
                      help='Number of parallel jobs (default: n_cpus - 1)')

  if args.parallel and len(halo_ids) > 1:
      n_jobs = args.n_jobs or max(1, cpu_count() - 1)
      with ProcessPoolExecutor(max_workers=n_jobs) as executor:
          futures = [executor.submit(analyze_single_halo, halo_id, ...)
                     for halo_id in halo_ids]
          for future in futures:
              halo_id, results = future.result()
  else:
      # Existing sequential loop
      for halo_id in halo_ids:
          ...
  ```

  **Cluster alternative (recommended for HPC):**
  - Use SLURM array jobs instead: `#SBATCH --array=0-9` (one halo per job)
  - Benefits: better resource isolation, easier debugging, cluster scheduler optimization

- [ ] **TODO: Add caching for repeated ModelRegistry lookups**
  - **Status**: Not started | **Impact**: Low | **Effort**: Low (1 hour)
  - **Location**: [src/workflow.py:368-372](src/workflow.py#L368-L372)
  - **Issue**: Same model looked up multiple times
  - **Solution**: Use `@lru_cache` or store in results dict
  - **Reference**: [src/workflow.py:368-369](src/workflow.py#L368-L369)

  ```python
  from functools import lru_cache

  @lru_cache(maxsize=128)
  def get_model_components(model_name: str) -> Dict:
      return {
          'evaluator': ModelRegistry.get_evaluator(model_name),
          'mass_integrator': ModelRegistry.get_mass_integrator(model_name),
      }
  ```

- [ ] **TODO: Add debug logging for MCMC diagnostics**
  - Log acceptance rates at intervals
  - Log convergence diagnostics
  - Add progress bar for long runs

## 🔵 Low Priority (Quick Wins)

- [ ] **TODO: Remove wildcard import**
  - **Location**: [src/fitting.py:3-5](src/fitting.py#L3-L5)
  - **Issue**: `from .models import *` pollutes namespace
  - **Fix**: Replace with explicit imports
  - **Reference**: [src/fitting.py:3-4](src/fitting.py#L3-L4)

  ```python
  # Before
  from .models import *

  # After
  from .models import hernquist_density, NFW_density, ModelRegistry
  ```

- [ ] **TODO: Fix circular import architecture**
  - **Location**: [src/models.py:42-46](src/models.py#L42-L46)
  - **Issue**: models.py imports fitting.py dynamically
  - **Solution**: Create `src/registry.py`, move ModelRegistry there
  - **Reference**: [src/models.py:42-43](src/models.py#L42-L43)

- [ ] **TODO: Organize imports (PEP 8)**
  - Install isort: `pip install isort`
  - Run: `isort src/`
  - Order: stdlib, third-party, local

- [ ] **TODO: Add pre-commit hooks**
  - Install: `pip install pre-commit`
  - Create `.pre-commit-config.yaml` with black, isort, mypy, flake8
  - Run: `pre-commit install`

  ```yaml
  repos:
    - repo: https://github.com/psf/black
      rev: 24.1.1
      hooks:
        - id: black
    - repo: https://github.com/PyCQA/isort
      rev: 5.13.2
      hooks:
        - id: isort
  ```

---

---

## 📊 Progress Summary

### Completed Work

| Phase | Status | LOC Reduction |
|-------|--------|---------------|
| Phase 1: Foundation | ✅ Complete | ~1100 lines |
| Phase 2: Core Refactoring | ✅ Complete | ~600 lines |
| Phase 3: Extended ModelRegistry | ✅ Complete | ~40 lines |
| Phase 4: I/O Consolidation & Data Structures | ✅ Complete | ~70 lines |
| Performance Features (Memory, Parallel) | ✅ Complete | +346 lines (added) |
| Bug Fixes & Error Handling | ✅ Complete | N/A |
| Documentation Updates | ✅ Complete | N/A |
| **Net Code Reduction** | **✅** | **~1464 lines (45% reduction)** |

### Pending Improvements

**Code Metrics** (as of 2025-12-05):

- Total Lines: ~7,304 (src/) + ~8,713 (all Python code)
- Test Coverage: ~0% (55 lines in tests/, mostly empty) → **Target: 80%** ⚠️ **CRITICAL**
- Type Hint Coverage: ~3% (1/38 core functions) → **Target: 90%**
- Documented Functions: ~60% (173 docstrings) → **Target: 100%**
- Error Handling: 19 bare `except Exception` catches → **Target: 0** ⚠️ **CRITICAL**

**Priority Counts**:

- 🔴 Critical: 2 items (Testing Infrastructure, Error Handling/Custom Exceptions)
- 🟡 High: 2 items (Type Hints, Documentation)
- 📋 Medium: 7 items (Config, Architecture, remaining Performance)
- 🔵 Low: 4 items (Quick wins)

**Total Pending**: 15 improvements remaining (2 completed since last update)

### Recent Accomplishments (2025-11-23)

**Bug Fixes:**

- ✅ Resolved circular import error
- ✅ Fixed HDF5 type errors
- ✅ Added missing Two-Power model
- ✅ Fixed 3 parameter name mismatches across models
- ✅ Fixed return value mismatch in compare_models
- ✅ **Fixed scalar assignment errors in 6 density functions** ⭐ **MAJOR FIX!**
  - Fixed: `sech_squared`, `disk_density`, `two_power_model`, `cored_power_model`, `three_power_model`, `double_exp_model`
  - Issue: `'numpy.float64' object does not support item assignment` during mass integration
  - Solution: Added scalar/array handling with `np.ndim()` checks
  - Impact: Enables mass calculations for Disk-Flaring and all cylindrical models

**Feature Additions:**

- ✅ Added component type to plot filenames (star/gas/dm prefixes)
- ✅ Completed ModelRegistry pattern for mass calculations

**Documentation:**

- ✅ Comprehensive CLAUDE.md update with architecture, workflows, and debugging guide
- ✅ Updated TODO.md with current status

**Code Quality:**

- ✅ Complete elimination of model-specific if/elif chains
- ✅ Consistent parameter naming across all models
- ✅ Clean separation of concerns via ModelRegistry

**Next recommended actions** (updated 2025-12-05):

1. **CRITICAL**: Add testing infrastructure (see [TODO_TESTING.md](TODO_TESTING.md))
   - Test directory structure exists but ~99% empty (only 55 lines total)
   - Start with unit tests for density functions in models.py
   - Target: 30% coverage initially, then 80%
2. **CRITICAL**: Fix error handling & create custom exceptions
   - Create [src/exceptions.py](src/exceptions.py) with custom exception hierarchy
   - Fix 19 bare `except Exception` catches across 5 files
   - Priority locations: [src/fitting.py:184](src/fitting.py#L184), [src/workflow.py](src/workflow.py)
3. **HIGH**: Add type hints ([src/workflow.py](src/workflow.py), [src/analysis.py](src/analysis.py), [src/fitting.py](src/fitting.py))
   - Current: ~3% coverage (1/38 functions)
   - Target: 90% coverage

---

## 📚 Additional Resources

- **[TODO_TESTING.md](TODO_TESTING.md)** - Comprehensive testing guide with examples
- **[CLAUDE.md](CLAUDE.md)** - Project instructions and conventions
- **Code TODOs**: Run `grep -r "TODO:" src/` to see all inline comments
- **Comprehensive Analysis**: See initial code review for detailed rationale

---

**Last Updated**: 2026-02-04
**Current Focus**: Testing infrastructure (0% → 80% coverage) ⚠️ **CRITICAL PRIORITY**

**Recent Changes**:

**Feb 4, 2026:**

- ✅ **Multi-simulation-type support** ⭐
  - Added `--sim_type` (`full`, `dmo`, `gas_only`) and `--sim_subdir` CLI flags
  - DMO: histogram-based center finding, all-inclusive dm filter re-registration (no `creation_time` dependency)
  - Gas-only: standard dm filter, skips star-dependent analyses
  - `find_virial_radius`, `generate_snapshot_profiles`, `process_time_series`, `process_snapshot` all sim_type-aware
  - Non-full runs save with sim_type suffix (`3d_profile_dm_dmo.h5`, `metadata_gas_only.h5`)
  - Updated CLAUDE.md, README.md, TODO.md

**Dec 5, 2025:**

- ✅ **Parallel model fitting implemented** ⭐

  - Created `fit_models_parallel()` with configurable parallelism (n_jobs parameter)
  - Added `_fit_single_model_worker()` for ProcessPoolExecutor compatibility
  - Automatic fallback to serial execution on errors
  - Memory logging per model fit
  - **Impact**: Significant performance improvement for multi-model component fitting
- ✅ **Memory tracking infrastructure** ⭐
  - Implemented `log_memory_usage()` using psutil and tracemalloc
  - Foundation for cluster job sizing complete
  - Remaining: Run comprehensive analysis to document memory requirements
- ✅ **FittingResult dataclass** ⭐
  - Created in [src/workflow.py:126-132](src/workflow.py#L126-L132)
  - Clean return values with best_model, samples, summary, model_comparison
  - Properly exported in __init__.py
- ⚠️ **Code quality assessment** conducted
  - Testing: Directory structure exists but ~99% empty (critical gap)
  - Type hints: Only ~3% coverage (1/38 functions)
  - Error handling: 19 bare Exception catches identified
  - Custom exceptions: Not yet implemented

**Dec 4, 2025:**

- **Phase 4 Complete**: I/O consolidation and data structures ⭐
  - Created `src/io.py` with centralized I/O utilities
  - Implemented `DensityProfile` dataclass for structured density data
  - Updated `generate_snapshot_profiles()` to return `DensityProfile` objects
  - Created `plot_galaxy_density_profile()` for unified component visualization
  - Eliminated ~70 lines of duplicated code across 5 files
- **Updated process_snapshot()** ⭐
  - Now accepts both yt dataset file paths and DensityProfile objects
  - Enables more flexible workflow and profile reuse across analyses
- Added multi-halo support to foggie_analysis.py (sequential processing)
- Updated spherical averaging for disk models in plotting.py
- Updated documentation: workflow.py, io.py, TODO.md with I/O consolidation details

**Last Review**: Comprehensive codebase assessment completed (2025-12-05)
