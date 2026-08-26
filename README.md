# FOGGIE Dynamics

MPhys 25/26 project analyzing high-resolution cosmological simulations from the FOGGIE project.

## Quick Start

```bash
# Setup environment
conda create --name foggie --file requirements.txt
conda activate foggie

# Run analysis
cd foggie-dynamics
python code/foggie_analysis.py --halo 8508 --component star

# Start Jupyter for notebooks
jupyter lab
```

## Project Structure

- `code/` - Analysis scripts and main entry point (`foggie_analysis.py`)
- `src/` - Core Python modules implementing ModelRegistry pattern
- `config/` - YAML configuration files for model fitting
- `results/` - Analysis outputs and logs (organized by halo ID)
- `plots/` - Generated figures
- `notes/` - Project documentation and notebooks

## Key Features

- **ModelRegistry Pattern** - Clean, extensible model fitting framework
- **MCMC Fitting** - Bayesian parameter estimation using `emcee`
- **Model Comparison** - Automatic BIC/AIC-based model selection
- **Component Analysis** - Separate fitting for stellar (bulge/disk/halo), gas, and dark matter

## Documentation

- **[TODO.md](TODO.md)** - Comprehensive task tracking and progress
- **[REFACTORING.md](REFACTORING.md)** - Detailed refactoring plans and implementation strategy
- **[REFACTORING_QUICKSTART.md](REFACTORING_QUICKSTART.md)** - Quick reference for completed refactorings
- **[TODO_TESTING.md](TODO_TESTING.md)** - Testing infrastructure guide
- **[CLAUDE.md](../CLAUDE.md)** - Project instructions for Claude Code AI assistant

## Recent Updates

### Multi-Simulation-Type Support (2026-02-04) ✅

- ✅ Added `--sim_type` flag: `full` (default), `dmo`, `gas_only`
- ✅ Added `--sim_subdir` flag for non-standard data directory layouts
- ✅ DMO runs use histogram-based center finding (no gas grid dependency)
- ✅ Particle filters, virial radius, angular momentum, and enclosed masses all sim_type-aware
- ✅ Non-full runs save with sim_type-suffixed filenames to avoid overwriting full-run outputs

### Major Refactoring Complete (Phases 1-3) ✅ (2025-11-30)

- ✅ Complete ModelRegistry implementation with mass integrators and equations
- ✅ Consolidated fitting functions (~500 LOC reduction)
- ✅ ProjectPaths dataclass for centralized path management
- ✅ Fixed all circular imports and parameter mismatches
- ✅ Added component type to plot filenames
- ✅ Comprehensive documentation updates (CLAUDE.md, TODO.md)
- ✅ Total code reduction: ~1740 lines (48%)

### Current Focus

- 🔴 **Testing infrastructure** - Currently 0% coverage, target 80%
- 🟡 **Type hints** - Currently ~15% coverage, target 90%
- 🟡 **Error handling** - Replacing bare exceptions with specific types

## Analysis Workflow

1. Load FOGGIE simulation data using `yt`
2. Extract density profiles for stellar/gas/dark matter
3. Fit multiple density models using MCMC
4. Compare models and select best fit
5. Calculate masses and derived statistics
6. Generate diagnostic plots
