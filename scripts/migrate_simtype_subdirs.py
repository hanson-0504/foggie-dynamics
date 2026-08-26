#!/usr/bin/env python
"""
Migrate sim_type suffixed files to experiment-based directory structure.

Old structure (what you have):
    data/halo_005036/z_0.800/
    ├── 3d_profile_dm_dmo.h5      ← has suffix
    ├── metadata_dmo.h5           ← has suffix
    ├── 3d_profile_dm.h5          ← full sim
    ├── metadata.h5               ← full sim

New structure (what you want):
    data/halo_005036/z_0.800/     ← full sim stays here
    ├── 3d_profile_dm.h5
    ├── metadata.h5

    data/dmo/halo_005036/z_0.800/ ← dmo files move here
    ├── 3d_profile_dm.h5          ← suffix removed
    └── metadata.h5               ← suffix removed

Usage:
    # Dry run (preview changes)
    python scripts/migrate_simtype_subdirs.py --dry-run

    # Actually migrate
    python scripts/migrate_simtype_subdirs.py

    # Specific halo only
    python scripts/migrate_simtype_subdirs.py --halo 5036

    # Delete old files after moving
    python scripts/migrate_simtype_subdirs.py --delete-old
"""

import argparse
import shutil
from pathlib import Path
from typing import Dict, List, Tuple


def detect_sim_type_from_suffix(filename: str) -> str:
    """Detect simulation type from filename suffix."""
    if filename.endswith("_dmo.h5"):
        return "dmo"
    elif filename.endswith("_gas_only.h5"):
        return "gas_only"
    else:
        return "full"


def remove_suffix(filename: str, sim_type: str) -> str:
    """Remove sim_type suffix from filename."""
    if sim_type == "full":
        return filename

    suffix = f"_{sim_type}"
    if filename.endswith(f"{suffix}.h5"):
        return filename[:-len(suffix) - 3] + ".h5"
    return filename


def scan_snapshot_dir(snapshot_dir: Path) -> List[Tuple[Path, str]]:
    """
    Scan snapshot directory for files with sim_type suffixes.

    Returns:
        List of (file_path, sim_type) tuples for non-full simulations
    """
    files_to_migrate = []

    for h5_file in snapshot_dir.glob("*.h5"):
        sim_type = detect_sim_type_from_suffix(h5_file.name)

        # Only migrate non-full files
        if sim_type != "full":
            files_to_migrate.append((h5_file, sim_type))

    return files_to_migrate


def migrate_file(
    old_path: Path,
    new_path: Path,
    dry_run: bool = False,
    delete_old: bool = False,
) -> bool:
    """Move/copy file to new location."""
    if dry_run:
        print(f"  [DRY RUN] Would move:")
        print(f"    FROM: {old_path}")
        print(f"    TO:   {new_path}")
        return True

    try:
        # Create parent directory
        new_path.parent.mkdir(parents=True, exist_ok=True)

        # Check if destination exists
        if new_path.exists():
            print(f"  ⚠ SKIP: {new_path} already exists")
            return False

        # Move or copy file
        if delete_old:
            shutil.move(str(old_path), str(new_path))
            print(f"  ✓ Moved: {old_path.name} -> {new_path}")
        else:
            shutil.copy2(old_path, new_path)
            print(f"  ✓ Copied: {old_path.name} -> {new_path}")

        return True
    except Exception as e:
        print(f"  ✗ ERROR: {e}")
        return False


def process_halo(
    halo_dir: Path,
    data_root: Path,
    dry_run: bool = False,
    delete_old: bool = False,
) -> int:
    """Process all snapshot directories in a halo directory."""
    halo_id = halo_dir.name
    print(f"\n{'='*70}")
    print(f"Processing {halo_id}")
    print(f"{'='*70}")

    total_migrated = 0

    # Find all z_X.XXX directories
    snapshot_dirs = sorted(halo_dir.glob("z_*"))

    if not snapshot_dirs:
        print(f"  No snapshot directories found (z_*)")
        return 0

    for snap_dir in snapshot_dirs:
        if not snap_dir.is_dir():
            continue

        print(f"\n  Snapshot: {snap_dir.name}")

        # Scan for files with sim_type suffixes
        files_to_migrate = scan_snapshot_dir(snap_dir)

        if not files_to_migrate:
            print(f"    No sim_type suffixed files found")
            continue

        # Group by sim_type
        by_simtype: Dict[str, List[Path]] = {}
        for file_path, sim_type in files_to_migrate:
            if sim_type not in by_simtype:
                by_simtype[sim_type] = []
            by_simtype[sim_type].append(file_path)

        # Migrate each sim_type group
        for sim_type, file_list in by_simtype.items():
            print(f"    {sim_type}: {len(file_list)} files")

            for old_path in file_list:
                # New location: data/sim_type/halo_XXXXXX/z_X.XXX/filename_without_suffix
                new_filename = remove_suffix(old_path.name, sim_type)

                # Build new path with experiment (sim_type) before halo_id
                new_path = data_root / sim_type / halo_id / snap_dir.name / new_filename

                if migrate_file(old_path, new_path, dry_run, delete_old):
                    total_migrated += 1

    return total_migrated


def main():
    parser = argparse.ArgumentParser(
        description="Migrate sim_type suffixed files to experiment-based structure"
    )
    parser.add_argument(
        "--data-root",
        type=Path,
        default=Path("/cephfs/shanson/FOGGIE/data"),
        help="Root data directory",
    )
    parser.add_argument(
        "--halo",
        help="Only process specific halo (e.g., 5036)",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Preview changes without modifying files",
    )
    parser.add_argument(
        "--delete-old",
        action="store_true",
        help="Delete old files after moving (use move instead of copy)",
    )

    args = parser.parse_args()

    if not args.data_root.exists():
        print(f"ERROR: Data root not found: {args.data_root}")
        return 1

    if args.dry_run:
        print("=" * 70)
        print("DRY RUN MODE - No files will be modified")
        print("=" * 70)

    if args.delete_old and not args.dry_run:
        print("\n⚠️  WARNING: --delete-old is enabled!")
        print("   Old files will be MOVED (deleted from original location).")
        response = input("   Continue? [y/N]: ")
        if response.lower() != "y":
            print("Aborted.")
            return 0

    # Find halos to process
    if args.halo:
        halo_dirs = [args.data_root / f"halo_{args.halo.zfill(6)}"]
        if not halo_dirs[0].exists():
            print(f"ERROR: Halo directory not found: {halo_dirs[0]}")
            return 1
    else:
        halo_dirs = sorted(args.data_root.glob("halo_*"))

    if not halo_dirs:
        print(f"No halo directories found in {args.data_root}")
        return 0

    print(f"\nFound {len(halo_dirs)} halo(s) to process")

    if not args.dry_run and len(halo_dirs) > 1:
        response = input(f"Process {len(halo_dirs)} halos? [y/N]: ")
        if response.lower() != "y":
            print("Aborted.")
            return 0

    # Process each halo
    total_migrated = 0
    for halo_dir in halo_dirs:
        count = process_halo(halo_dir, args.data_root, args.dry_run, args.delete_old)
        total_migrated += count

    # Summary
    print("\n" + "=" * 70)
    if args.dry_run:
        print(f"DRY RUN COMPLETE: Would migrate {total_migrated} files")
        print("\nRun without --dry-run to actually migrate files")
    else:
        print(f"MIGRATION COMPLETE: {total_migrated} files migrated")
        if not args.delete_old:
            print("\nOld files kept (use --delete-old to remove them)")

        print("\n📁 New structure:")
        print("  data/halo_XXXXXX/z_X.XXX/        ← full simulations")
        print("  data/dmo/halo_XXXXXX/z_X.XXX/    ← dmo simulations")
        print("  data/gas_only/halo_XXXXXX/...    ← gas_only simulations")

    print("=" * 70)

    return 0


if __name__ == "__main__":
    exit(main())
