#!/usr/bin/env python3
"""Check R200 values stored in full-physics metadata across all redshifts."""

import h5py
from pathlib import Path

# --- 1. Check full-physics metadata ---
data_dir = Path("/ceph/cephfs/shanson/FOGGIE/data/halo_005036")
z_dirs = sorted(data_dir.glob("z_*/metadata.h5"))

print(f"=== Full-physics metadata: {data_dir} ===")
print(f"Found {len(z_dirs)} metadata files\n")

for meta_path in z_dirs:
    z_str = meta_path.parent.name
    with h5py.File(meta_path, "r") as f:
        keys = list(f.keys())
        attrs = dict(f.attrs)
    print(f"  {z_str}: keys={keys}, attrs={list(attrs.keys())}")

# --- 2. Check DMO results metadata ---
print("\n=== DMO results metadata ===")
dmo_dirs = [
    Path("/ceph/cephfs/shanson/FOGGIE/data/dm_only/halo_005036"),
    Path("/ceph/cephfs/shanson/FOGGIE/data/dmo/halo_005036"),
]
for dmo_dir in dmo_dirs:
    if dmo_dir.exists():
        dmo_metas = sorted(dmo_dir.glob("z_*/metadata*.h5"))
        print(f"\n  {dmo_dir}: {len(dmo_metas)} metadata files")
        for mp in dmo_metas[:5]:
            with h5py.File(mp, "r") as f:
                keys = list(f.keys())
            print(f"    {mp.parent.name}/{mp.name}: keys={keys}")
    else:
        print(f"  {dmo_dir}: does not exist")

# --- 3. Check top-level metadata ---
print("\n=== Top-level files ===")
for p in data_dir.glob("*.h5"):
    with h5py.File(p, "r") as f:
        keys = list(f.keys())
    print(f"  {p.name}: keys={keys[:10]}{'...' if len(keys) > 10 else ''}")
