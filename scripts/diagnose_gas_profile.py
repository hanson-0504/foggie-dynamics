#!/usr/bin/env python
"""
Diagnostic script for gas halo density profiles from gas_sim runs.
Checks 1D radial and 2D cylindrical profiles, density structure, and anomalies.
"""
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from src import ProjectPaths, DensityProfile, load_hdf5

PATHS = ProjectPaths.create(
    code_root=Path("/home/shanson/foggie-dynamics"),
    data_root=Path("/cephfs/shanson/FOGGIE")
)

# Load gas_sim data
halo = "halo_005036"
redshift = 0.0
data_dir = PATHS.snapshot_data(halo, redshift, experiment="gas_sim")

# Load the profile and metadata
data = DensityProfile.from_hdf5(data_dir / "3d_profile_gas.h5", component="gas")
metadata = load_hdf5(data_dir / "metadata.h5")

R200 = metadata['R200_kpc']
center = metadata['center_kpc']

print(f"Halo: {halo}, z={redshift}")
print(f"R200 = {R200:.2f} kpc")
print(f"Center (kpc): [{center[0]:.3f}, {center[1]:.3f}, {center[2]:.3f}]")

# ── Center check ──────────────────────────────────────────────────────────────
# Compare center to the box midpoint to flag obviously wrong centers.
# For FOGGIE sims the box is typically ~1-2 Mpc; a well-centered halo
# should be near the box midpoint (within ~100 kpc or so).
# Load sim_path to get box size if available.
sim_path = metadata.get('sim_path', None)
if sim_path:
    print(f"Sim path: {sim_path}")
print(f"\n*** Verify this center! Wrong-galaxy centering produces")
print(f"    oscillating inner profile + scattered 2D density blobs. ***")

# ── 1D radial profile summary ─────────────────────────────────────────────────
print(f"\n=== 1D Radial Profile ===")
print(f"Number of radial bins: {len(data.r_1d)}")
print(f"Radial range: {data.r_1d.min():.3f} - {data.r_1d.max():.2f} kpc")
print(f"Density range: {data.rho_1d.min():.2e} - {data.rho_1d.max():.2e} Msun/kpc^3")

# Check for NaN/zero bins
nan_mask = ~np.isfinite(data.rho_1d) | (data.rho_1d <= 0)
print(f"Bad bins (NaN/zero/negative): {nan_mask.sum()} / {len(data.rho_1d)}")

# Look for density spikes (>10x jumps between adjacent bins)
valid = data.rho_1d > 0
if valid.sum() > 1:
    r_valid = data.r_1d[valid]
    rho_valid = data.rho_1d[valid]
    log_rho = np.log10(rho_valid)
    gradients = np.diff(log_rho)
    spike_mask = np.abs(gradients) > 1.0
    if spike_mask.any():
        print(f"\nDensity spikes (>10x jumps): {spike_mask.sum()}")
        spike_indices = np.where(spike_mask)[0][:10]
        print("First 10 spike locations:")
        for idx in spike_indices:
            print(f"  r={r_valid[idx]:.3f} -> {r_valid[idx+1]:.3f} kpc: "
                  f"rho={rho_valid[idx]:.2e} -> {rho_valid[idx+1]:.2e}")
    else:
        print("\nNo density spikes detected.")

# Disk/halo structure: rough disk mass fraction within R<0.2*R200, |z|<5 kpc
if data.rho_2d is not None and data.R is not None and data.z is not None:
    print(f"\n=== 2D Cylindrical Profile ===")
    print(f"R range: {data.R.min():.3f} - {data.R.max():.2f} kpc")
    print(f"z range: {data.z.min():.3f} - {data.z.max():.2f} kpc")
    print(f"rho_2d shape: {data.rho_2d.shape}")
    valid_2d = data.rho_2d[np.isfinite(data.rho_2d) & (data.rho_2d > 0)]
    print(f"2D density range: {valid_2d.min():.2e} - {valid_2d.max():.2e} Msun/kpc^3")

    # Midplane density profile (z ~ 0 row)
    z_mid_idx = np.argmin(np.abs(data.z))
    rho_midplane = data.rho_2d[:, z_mid_idx]  # R varies along axis 0
    print(f"\nMidplane (z~{data.z[z_mid_idx]:.2f} kpc) peak density: "
          f"{np.nanmax(rho_midplane):.2e} Msun/kpc^3 at R={data.R[np.nanargmax(rho_midplane)]:.2f} kpc")

# ── Plots ─────────────────────────────────────────────────────────────────────
fig, axes = plt.subplots(2, 2, figsize=(14, 10))
fig.suptitle(f'{halo} Gas Sim - Gas Density Profile Diagnostics (z={redshift})', fontsize=13)

# Plot 1: 1D radial density profile
ax = axes[0, 0]
mask_plot = data.rho_1d > 0
ax.loglog(data.r_1d[mask_plot], data.rho_1d[mask_plot], 'o-', markersize=3, linewidth=0.8,
          color='steelblue')
ax.axvline(R200, linestyle='--', color='black', label=f'R200 = {R200:.1f} kpc')
ax.set_xlabel('r [kpc]')
ax.set_ylabel(r'$\rho_\mathrm{gas}$ [M$_\odot$ kpc$^{-3}$]')
ax.set_title('1D Radial Gas Density Profile')
ax.legend()
ax.grid(True, alpha=0.3)

# Plot 2: Log-gradient (to spot spikes)
ax = axes[0, 1]
if valid.sum() > 1:
    r_mids = 0.5 * (r_valid[:-1] + r_valid[1:])
    ax.semilogx(r_mids, gradients, 'o-', markersize=3, linewidth=0.8, color='darkorange')
    ax.axhline(1.0, linestyle='--', color='red', alpha=0.6, label='+10x threshold')
    ax.axhline(-1.0, linestyle='--', color='red', alpha=0.6, label='-10x threshold')
    ax.axvline(R200, linestyle='--', color='black', alpha=0.5)
    ax.set_xlabel('r [kpc]')
    ax.set_ylabel(r'$\Delta \log_{10}\rho / \Delta \log_{10}r$  (step)')
    ax.set_title('Log-density Gradient (spike detector)')
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)

# Plot 3: 2D cylindrical density (R, z) heatmap
# Note: z represents |z| (absolute height), so only positive values exist.
# Clip to R200/2 on each axis to focus on the galaxy region.
ax = axes[1, 0]
if data.rho_2d is not None:
    z_max_plot = min(R200 * 0.5, data.z.max())
    R_max_plot = min(R200 * 1.2, data.R.max())
    R_mask = data.R <= R_max_plot
    z_mask = data.z <= z_max_plot
    rho_clip = data.rho_2d[np.ix_(R_mask, z_mask)]
    rho_2d_plot = np.log10(np.where(rho_clip > 0, rho_clip, np.nan))
    im = ax.pcolormesh(data.R[R_mask], data.z[z_mask], rho_2d_plot.T, shading='auto', cmap='inferno')
    plt.colorbar(im, ax=ax, label=r'$\log_{10}\rho$ [M$_\odot$ kpc$^{-3}$]')
    ax.set(
        xlabel = "R [kpc]",
        ylabel = "|z| [kpc]",
        title = "2D Gas Density (R, |z|)",
        xlim = (data.R[R_mask].min(), R_max_plot),
        ylim = (data.z[z_mask].min(), z_max_plot),
        xscale = "log",
        yscale = "log",
    )
    ax.grid(True, alpha=0.2, color='white')

# Plot 4: Midplane profile (R) and vertical profile (|z|) on separate x-axes
# Both share y (density) axis. z is |z| - the first bin is the closest to midplane.
ax = axes[1, 1]
if data.rho_2d is not None:
    # Midplane profile: density vs R at smallest |z| bin
    z_min_idx = 0  # z is |z|, so index 0 is closest to midplane
    rho_midplane = data.rho_2d[:, z_min_idx]
    good = rho_midplane > 0
    ax.semilogy(data.R[good], rho_midplane[good], 'o-', markersize=3, linewidth=0.8,
                label=f'Midplane (|z|={data.z[z_min_idx]:.2f} kpc)', color='steelblue')
    ax.axvline(R200, linestyle='--', color='black', alpha=0.5, label=f'R200={R200:.0f} kpc')
    ax.set_xlabel('R [kpc]')
    ax.set_ylabel(r'$\rho$ [M$_\odot$ kpc$^{-3}$]')
    ax.set_title('Midplane (blue, vs R) & Vertical (orange, vs |z|)')

    # Vertical profile at R closest to disk scale radius guess (~2 kpc)
    r_disk = 2.0
    R_idx = np.argmin(np.abs(data.R - r_disk))
    rho_vertical = data.rho_2d[R_idx, :]
    good_v = rho_vertical > 0
    ax2 = ax.twiny()  # second x-axis for |z|
    ax2.semilogy(data.z[good_v], rho_vertical[good_v], 's--', markersize=3, linewidth=0.8,
                 label=f'Vertical at R={data.R[R_idx]:.1f} kpc', color='darkorange', alpha=0.8)
    ax2.set_xlabel('|z| [kpc]', color='darkorange')
    ax2.tick_params(axis='x', labelcolor='darkorange')
    ax2.set_xlim(0, min(R200 * 0.5, data.z.max()))
    lines1, labels1 = ax.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax.legend(lines1 + lines2, labels1 + labels2, fontsize=8)
    ax.grid(True, alpha=0.3)

plt.tight_layout()
out_path = 'gas_profile_diagnostic.png'
plt.savefig(out_path, dpi=150)
print(f"\nSaved diagnostic plot to: {out_path}")
plt.show()
