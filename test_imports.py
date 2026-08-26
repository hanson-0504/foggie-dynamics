#!/usr/bin/env python
"""
Test script to verify circular import is resolved.
Run this with: conda activate mphys && python test_imports.py
"""

print("Testing imports...")

try:
    from src import analysis

    print("✓ src.analysis imported successfully")
except ImportError as e:
    print(f"✗ Failed to import src.analysis: {e}")
    exit(1)

try:
    from src import fitting

    print("✓ src.fitting imported successfully")
except ImportError as e:
    print(f"✗ Failed to import src.fitting: {e}")
    exit(1)

try:
    from src import workflow

    print("✓ src.workflow imported successfully")
except ImportError as e:
    print(f"✗ Failed to import src.workflow: {e}")
    exit(1)

try:
    from src import plotting

    print("✓ src.plotting imported successfully")
except ImportError as e:
    print(f"✗ Failed to import src.plotting: {e}")
    exit(1)

print("\n✓ All imports successful - circular import is resolved!")
print("\nYou can now run:")
print("  python code/foggie_analysis.py --halo 8508 --component dm --n_jobs 2")
