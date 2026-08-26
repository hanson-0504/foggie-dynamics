import numpy as np

def test_compare_masses():
    from src.analysis import compare_masses
    # create mock data
    radii = np.array([1, 2, 3, 4, 5])
    model_masses = {'total': np.array([10, 20, 30, 40, 50])}
    sim_masses = {'total': np.array([5, 20, 35, 40, 60])}
    # expected output
    