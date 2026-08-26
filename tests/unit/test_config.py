from src.config import ProjectPaths

PATHS = ProjectPaths()

# Test formatting
assert PATHS.format_snapshot_id(0.0) == "z_0.000"
assert PATHS.format_snapshot_id(2.5) == "z_2.500"
assert PATHS.format_snapshot_id(10.123456) == "z_10.123"

# Test path construction
path = PATHS.snapshot_data("halo_008508", 2.5)
assert "halo_008508/z_2.500" in str(path)
