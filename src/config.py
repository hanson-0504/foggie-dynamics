"""
Configuration and logging utilities for FOGGIE analysis.
"""

import logging
import os
import re
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Optional

import yaml

from .io import ensure_directory

# Define project paths
FOGGIE_CODE_ROOT = Path(__file__).resolve().parents[1]

FOGGIE_DATA_ROOT = Path(
    os.environ.get("FOGGIE_DATA", FOGGIE_CODE_ROOT)
)


@dataclass(frozen=True)
class ProjectPaths:
    """Centralized path management with automatic directory creation."""
    code_root: Path
    data_root: Path

    # ---------- factory ----------

    @classmethod
    def create(
        cls,
        code_root: Optional[Path] = None,
        data_root: Optional[Path] = None,
        env_var: str = "FOGGIE_DATA",
    ) -> "ProjectPaths":
        code_root = (
            code_root.resolve()
            if code_root is not None
            else cls._resolve_code_root()
        )

        data_root = (
            data_root.resolve()
            if data_root is not None
            else cls._resolve_data_root(code_root, env_var)
        )

        return cls(code_root=code_root, data_root=data_root)

    # ---------- resolution helpers ----------

    @staticmethod
    def _resolve_code_root() -> Path:
        """Find project root by walking upwards."""
        start = Path.cwd()
        for parent in (start, *start.parents):
            if (parent / "pyproject.toml").exists():
                return parent
        raise RuntimeError("Could not determine project code root")

    @staticmethod
    def _resolve_data_root(code_root: Path, env_var: str) -> Path:
        """Resolve data root from environment or fallback."""
        if env_var in os.environ:
            return Path(os.environ[env_var]).expanduser().resolve()
        return code_root

    # ---------- directory properties -----------

    @property
    def data(self) -> Path:
        return self.data_root / "data"

    @property
    def results(self) -> Path:
        return self.data_root / "results"

    @property
    def plots(self) -> Path:
        return self.code_root / "plots"

    @property
    def logs(self) -> Path:
        return self.code_root / "logs"

    @property
    def config(self) -> Path:
        return self.code_root / "config"

    # ---------- halo / snapshot helpers ----------

    def halo_data(self, halo_id: str) -> Path:
        return self.data / halo_id

    def halo_results(self, halo_id: str, experiment: Optional[str] = None) -> Path:
        base = self.results / experiment if experiment else self.results
        path = base / halo_id
        ensure_directory(path)
        return path

    def result_file(
        self, halo_id: str, filename: str, experiment: Optional[str] = None
    ) -> Path:
        """Get path for a results file"""
        path = self.halo_results(halo_id, experiment=experiment)
        return path / filename

    def plot_file(
        self, halo_id: str, component: str, name: str, experiment: Optional[str] = None
    ) -> Path:
        base = self.plots / experiment if experiment else self.plots
        path = base / halo_id
        ensure_directory(path)
        return path / f"{component}_{name}_fit.png"

    def config_file(self, filename: str = "fitting_config.yaml") -> Path:
        """Get Path to configuration file."""
        return self.config / filename

    @staticmethod
    def format_snapshot_id(redshift: float) -> str:
        """Format redshift as z_X.XXX with 3 decimal."""
        return f"z_{redshift:.3f}"

    def snapshot_data(
        self,
        halo_id: str,
        redshift: float | str,
        dd_num: Optional[int] = None,
        experiment: Optional[str] = None,
    ) -> Path:
        """
        Get data directory for specific snapshot with backward compatibility.

        Args:
            halo_id: Halo identifier (e.g., 'halo_008508')
            redshift: Redshift value (float or string like '6.000' or 'z_6.000')
            dd_num: Optional DD number for backward compatibility with old paths
            experiment: Optional experiment/variant name (e.g., 'dmo', 'gas_only', 'r200_scaling').
                Use this to organize different simulation types or experimental configurations.

        Returns:
            Path to snapshot data directory. Structure:
            - Default (no experiment): data/halo_XXXXXX/z_X.XXX/
            - With experiment: data/experiment/halo_XXXXXX/z_X.XXX/

        Examples:
            - Full simulation: snapshot_data("halo_005036", 0.8)
              → data/halo_005036/z_0.800/
            - DMO simulation: snapshot_data("halo_005036", 0.8, experiment="dmo")
              → data/dmo/halo_005036/z_0.800/
            - R200 experiment: snapshot_data("halo_005036", 0.8, experiment="r200_scaling")
              → data/r200_scaling/halo_005036/z_0.800/
        """
        # Accept redshift as float or string (e.g., '6.000' or 'z_6.000')
        try:
            z_val = float(redshift)
        except Exception:
            m = re.search(r"z_(-?\d+(?:\.\d+)?)", str(redshift))
            if m:
                z_val = float(m.group(1))
            else:
                raise ValueError(f"Could not parse redshift: {redshift}")

        # Build base path with optional experiment
        base = (self.data / experiment) if experiment else self.data
        exact = base / halo_id / self.format_snapshot_id(z_val)

        if exact.exists():
            return exact

        # Fallback to old DD format if provided (only for backward compatibility)
        if dd_num is not None and not experiment:
            old_path = self.data / f"{halo_id}_DD{dd_num:04d}"
            if old_path.exists():
                return old_path

        # If exact z_ folder not found, scan for available z_* directories and
        # return the closest numeric match (useful if rounding differs).
        # Only match if difference is < 0.01 to avoid matching different redshifts.
        pattern = re.compile(r"z_(-?\d+(?:\.\d+)?)")
        halo_dir = base / halo_id
        if not halo_dir.exists():
            # return the expected path (caller can decide to create)
            return exact

        candidates = []
        for p in halo_dir.iterdir():
            if p.is_dir():
                m = pattern.search(p.name)
                if m:
                    try:
                        zcand = float(m.group(1))
                        diff = abs(zcand - z_val)
                        # Only consider as candidate if difference is very small (rounding errors)
                        if diff < 0.01:
                            candidates.append((diff, p))
                    except Exception:
                        continue

        if candidates:
            candidates.sort(key=lambda x: x[0])
            return candidates[0][1]

        # No candidates found — return the expected (non-existent) path
        return exact

    def snapshot_results(
        self,
        halo_id: str,
        redshift: float,
        experiment: Optional[str] = None,
    ) -> Path:
        """Get results directory for specific snapshot."""
        base = (self.results / experiment) if experiment else self.results
        path = base / halo_id / self.format_snapshot_id(redshift)
        ensure_directory(path)
        return path

    def snapshot_plots(
        self, halo_id: str, redshift: float, experiment: Optional[str] = None
    ) -> Path:
        """Get plots directory for specific snapshot."""
        base = (self.plots / experiment) if experiment else self.plots
        path = base / halo_id / self.format_snapshot_id(redshift)
        ensure_directory(path)
        return path


PATHS = ProjectPaths.create(
    code_root=FOGGIE_CODE_ROOT,
    data_root=FOGGIE_DATA_ROOT
)
DATA_DIR = PATHS.data
RESULTS_DIR = PATHS.results
PLOT_DIR = PATHS.plots
LOG_DIR = PATHS.logs

# Benefits:
# - Automatic directory creation
# - Cleaner API: PATHS.halo_results(halo_id) vs manual string formatting
# - Single source of truth for all paths

# TODO: Implement Pydantic validation classes for config
# Create ComponentConfig class (reusable for bulge, disk, halo, cgm, etc.)
# Then create StarConfig, GasConfig, DMConfig, FittingConfig
# See improvement recommendations for structure - use Option 2 (moderate validation)


def setup_logging(component, halo_id=None):
    """
    Setup logging configuration for analysis.

    Args:
        component: Component being analyzed
        halo_id: Optional halo identifier for per-halo logging

    Returns:
        logging.Logger: Configured logger instance
    """
    ensure_directory(LOG_DIR)  # Ensure parent directory exists

    # Create log filename with optional halo_id
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    if halo_id:
        log_file = LOG_DIR / f"{halo_id}_{component}_analysis_{timestamp}.log"
    else:
        log_file = LOG_DIR / f"{component}_analysis_{timestamp}.log"

    # Remove existing handlers to allow reconfiguration
    root_logger = logging.getLogger()
    for handler in root_logger.handlers[:]:
        handler.close()
        root_logger.removeHandler(handler)

    # Configure with new handlers
    logging.basicConfig(
        level=logging.DEBUG,
        format="%(asctime)s - %(filename)s:%(lineno)d - %(levelname)s - %(message)s",
        handlers=[logging.FileHandler(log_file), logging.StreamHandler()],
        force=True,  # Force reconfiguration
    )

    # Suppress verbose matplotlib debug output
    logging.getLogger('matplotlib').setLevel(logging.WARNING)
    logging.getLogger('matplotlib.font_manager').setLevel(logging.WARNING)
    logging.getLogger('matplotlib.ticker').setLevel(logging.WARNING)

    return logging.getLogger(__name__)


def load_config(config_file=None):
    """
    Load fitting configuration from YAML file.

    Args:
        config_file: Path to config file (default: config/fitting_config.yaml)

    Returns:
        dict: Configuration dictionary
    """
    if config_file is None:
        config_file = PATHS.config / "fitting_config.yaml"

    with open(config_file, "r") as f:
        config = yaml.safe_load(f)

    return config
