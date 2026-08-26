#!/usr/bin/env python
"""
Diagnostic script to understand DMO density profile spikes
"""
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from src import ProjectPaths, DensityProfile, load_hdf5

PATHS = ProjectPaths.create(
    code_root=Path("/home/shanson/foggie-dynamics"),
    data_root=Path("/cephfs/shanson/FOGGIE")
)

# Load DMO data
halo = "halo_005036"
redshift = 0.0
data_dir = PATHS.snapshot_data(halo, redshift, experiment="gas_sim")

# Load the profile
data = DensityProfile.from_hdf5(data_dir / "3d_profile_dm.h5", component="dm")
metadata = load_hdf5(data_dir / "metadata.h5")

print(f"Halo: {halo}, z={redshift}")
print(f"R200 = {metadata['R200_kpc']:.2f} kpc")
print(f"Number of radial bins: {len(data.r_1d)}")
print(f"Radial range: {data.r_1d.min():.3f} - {data.r_1d.max():.2f} kpc")

# Check for particle counts if available
if hasattr(data, 'counts') and data.counts is not None:
    print(f"\nParticle counts per radial bin:")
    print(f"  Min: {data.counts.min()}")
    print(f"  Max: {data.counts.max()}")
    print(f"  Mean: {data.counts.mean():.1f}")
    print(f"  Median: {np.median(data.counts):.1f}")

    # Find bins with very few particles
    low_count_mask = data.counts < 10
    print(f"\nBins with <10 particles: {low_count_mask.sum()} / {len(data.counts)}")

    if low_count_mask.any():
        print("\nFirst 10 low-count bins:")
        low_indices = np.where(low_count_mask)[0][:10]
        for idx in low_indices:
            print(f"  r={data.r_1d[idx]:.3f} kpc: {data.counts[idx]} particles, "
                  f"rho={data.rho_1d[idx]:.2e} Msun/kpc^3")
else:
    print("\nNo particle count information available in profile")

# Look for density spikes (>10x jumps)
log_rho = np.log10(data.rho_1d)
gradients = np.diff(log_rho)
spike_mask = np.abs(gradients) > 1.0  # >10x jump in adjacent bins

if spike_mask.any():
    print(f"\nFound {spike_mask.sum()} density spikes (>10x jumps)")
    spike_indices = np.where(spike_mask)[0][:10]  # First 10
    print("First 10 spike locations:")
    for idx in spike_indices:
        print(f"  r={data.r_1d[idx]:.3f} -> {data.r_1d[idx+1]:.3f} kpc: "
              f"rho={data.rho_1d[idx]:.2e} -> {data.rho_1d[idx+1]:.2e}")

# Create diagnostic plots
fig, axes = plt.subplots(2, 1, figsize=(10, 10))

# Plot 1: Density profile
ax = axes[0]
ax.loglog(data.r_1d, data.rho_1d, 'o-', markersize=3, linewidth=0.5)
ax.axvline(metadata['R200_kpc'], linestyle='--', color='black',
           label=f'R200 = {metadata["R200_kpc"]:.1f} kpc')
ax.set_xlabel('r [kpc]')
ax.set_ylabel(r'$\rho$ [M$_\odot$ kpc$^{-3}$]')
ax.set_title(f'{halo} DMO Run - Density Profile')
ax.legend()
ax.grid(True, alpha=0.3)

# Plot 2: Particle counts per bin (if available)
ax = axes[1]
if hasattr(data, 'counts') and data.counts is not None:
    ax.semilogx(data.r_1d, data.counts, 'o-', markersize=3, linewidth=0.5)
    ax.axhline(10, linestyle='--', color='red', alpha=0.5, label='N=10 threshold')
    ax.axvline(metadata['R200_kpc'], linestyle='--', color='black')
    ax.set_xlabel('r [kpc]')
    ax.set_ylabel('Particle count per radial bin')
    ax.set_title('Particles per Radial Shell')
    ax.legend()
    ax.grid(True, alpha=0.3)
else:
    ax.text(0.5, 0.5, 'No particle count data available',
            ha='center', va='center', transform=ax.transAxes)

plt.tight_layout()
plt.savefig('dmo_profile_diagnostic.png', dpi=150)
print("\nSaved diagnostic plot to: dmo_profile_diagnostic.png")
plt.show()
