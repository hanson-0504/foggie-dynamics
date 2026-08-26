import os
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from src import filters_and_utils as fu


def test_rotation_matrix_from_vectors_identity():
    v = np.array([0.0, 0.0, 1.0])
    R = fu.rotation_matrix_from_vectors(v, v)
    assert np.allclose(R, np.eye(3))


def test_rotation_matrix_from_vectors_opposite():
    v1 = np.array([0.0, 0.0, 1.0])
    v2 = np.array([0.0, 0.0, -1.0])
    R = fu.rotation_matrix_from_vectors(v1, v2)
    # should be a 180-degree rotation: R @ v1 == v2 (within sign)
    out = R @ v1
    assert np.allclose(out, v2, atol=1e-8) or np.allclose(out, -v2, atol=1e-8)


def test_add_gaussian_weight_center_and_width():
    coords = np.linspace(0, 10, 11)
    weights = fu.add_gaussian_weight(coords, fitting_range=(0, 10), sigma_frac=0.2)
    assert weights.max() == pytest.approx(1.0)
    # symmetric around center
    assert np.isclose(weights[0], weights[-1])


def test_add_inverse_radius_weight_basic():
    coords = np.array([1.0, 2.0, 4.0])
    w = fu.add_inverse_radius_weight(coords, fitting_range=(1.0, 4.0), power=1.0, r_scale=1.0)
    assert np.allclose(w, np.array([1.0, 0.5, 0.25]))


def test_apply_data_range_mask_numeric_index():
    coords = (np.array([0, 1, 2, 3, 4]),)
    data = np.array([1, 2, 3, 4, 5])
    mask = fu.apply_data_range_mask(coords, data, {0: (1, 4)})
    # should include coords >=1 and <4 -> indices 1,2,3
    assert np.array_equal(np.where(mask)[0], np.array([1, 2, 3]))


@pytest.mark.parametrize("comp_obj_type", ["dict", "fittingresult"])
def test_save_evolution_data_handles_both_types(tmp_path, monkeypatch, comp_obj_type):
    """
    Test that save_evolution_data accepts both dict-like subcomponent results
    and `FittingResult` objects (dataclass) when saving snapshots.
    """
    import h5py

    from src import analysis

    # Build minimal evolution_data structure
    evolution_data = {
        "halo_id": "halo_test",
        "snapshots": [0],
        "redshifts": [0.0],
        "times_Gyr": [0.0],
        "components": {
            "star": {
                "fitted_params": [],
                "best_models": [],
                "masses": [],
            }
        },
    }

    # Create a minimal best_summary
    best_summary = {"rho_0": 1.23, "r_s": 2.0}

    if comp_obj_type == "dict":
        subcomp = {"best_model": "testmodel", "best_summary": best_summary}
    else:
        # Construct a minimal FittingResult-like object
        from src.workflow import FittingResult

        fr = FittingResult(best_model="testmodel", samples=None, summary=best_summary, model_comparison={})
        # Wrap into the expected per-snapshot dict structure
        subcomp = fr

    # Place one snapshot with bulge subcomponent
    per_snapshot = {"bulge": subcomp}
    evolution_data["components"]["star"]["fitted_params"].append(per_snapshot)

    # Monkeypatch RESULTS_DIR to tmp_path and ensure output directory is created
    monkeypatch.setattr(analysis, "RESULTS_DIR", tmp_path)
    # ensure_directory should create the parent directory for the HDF5 file
    monkeypatch.setattr(
        analysis,
        "ensure_directory",
        lambda p: Path(p).mkdir(parents=True, exist_ok=True),
    )

    # Call save_evolution_data; it should create an h5 file without error
    analysis.save_evolution_data(evolution_data, "halo_test", logger=analysis.logger)

    # Check file exists
    out_file = tmp_path / "halo_test" / "evolution_time_series.h5"
    assert out_file.exists(), f"Expected {out_file} to exist"

    # Open and verify stored attributes
    with h5py.File(out_file, "r") as f:
        assert f.attrs["halo_id"] == "halo_test"
        # navigate to snapshot group and bulge
        grp = f["star"]["snapshot_0000"]["bulge"]
        assert grp.attrs["best_model"] == "testmodel"
        # rho_0 saved as attr
        assert float(grp.attrs["rho_0"]) == pytest.approx(1.23)
        assert float(grp.attrs["r_s"]) == pytest.approx(2.0)
