#!/usr/bin/env python3
"""
Visualize redshift-dependent range scaling for fitting ranges.

Shows how fitting ranges adapt to galaxy size and resolution at different redshifts.
"""

import matplotlib.pyplot as plt
import numpy as np


def scale_range_with_redshift(range_z0, redshift, min_scale=0.1, scaling_mode="linear"):
    """Compute scaled range at given redshift."""
    r_min_z0, r_max_z0 = range_z0

    if scaling_mode == "none":
        return (r_min_z0, r_max_z0)
    elif scaling_mode == "linear":
        scale_factor = 1.0 / (1.0 + redshift)
    elif scaling_mode == "sqrt":
        scale_factor = 1.0 / np.sqrt(1.0 + redshift)
    elif scaling_mode == "log":
        scale_factor = 1.0 / (1.0 + np.log10(2.0 + redshift))
    else:
        scale_factor = 1.0 / (1.0 + redshift)

    scale_factor = np.clip(scale_factor, min_scale, 1.0)
    r_min = r_min_z0 * scale_factor
    r_max = r_max_z0 * scale_factor

    return (r_min, r_max)


def main():
    # Redshift range
    redshifts = np.linspace(0, 6, 100)

    # Define z=0 ranges for different components
    ranges_z0 = {
        "Star Bulge": (0.5, 3.0),
        "Star Disk (R)": (3.0, 30.0),
        "Star Halo": (30.0, 250.0),
        "Gas Disk (R)": (1.0, 30.0),
        "DM Halo": (0.3, 250.0),
    }

    # Create figure with subplots
    fig, axes = plt.subplots(2, 3, figsize=(18, 10))
    axes = axes.flatten()

    colors = {'linear': 'blue', 'sqrt': 'green', 'log': 'orange', 'none': 'gray'}

    # Plot each component
    for idx, (component, range_z0) in enumerate(ranges_z0.items()):
        ax = axes[idx]

        for mode in ['linear', 'sqrt', 'none']:
            r_mins = []
            r_maxs = []

            for z in redshifts:
                r_min, r_max = scale_range_with_redshift(
                    range_z0, z, min_scale=0.1, scaling_mode=mode
                )
                r_mins.append(r_min)
                r_maxs.append(r_max)

            # Plot range as shaded region
            ax.fill_between(redshifts, r_mins, r_maxs, alpha=0.3, color=colors[mode])
            # Plot boundaries
            ax.plot(redshifts, r_mins, '--', color=colors[mode], alpha=0.7)
            ax.plot(redshifts, r_maxs, '-', color=colors[mode],
                    label=f'{mode}', linewidth=2)

        ax.set_xlabel('Redshift', fontsize=12)
        ax.set_ylabel('Range [kpc]', fontsize=12)
        ax.set_title(f'{component}\n(z=0: [{range_z0[0]}, {range_z0[1]}] kpc)',
                     fontsize=13, fontweight='bold')
        ax.set_yscale('log')
        ax.grid(True, alpha=0.3)
        ax.legend(fontsize=10)

        # Add specific redshift markers
        for z_mark in [0, 1, 2, 5]:
            ax.axvline(z_mark, color='red', alpha=0.2, linestyle=':')

    # Remove extra subplot
    fig.delaxes(axes[-1])

    # Add text annotation in the empty space
    fig.text(0.70, 0.25,
             'Scaling Modes:\n'
             '• linear: scale = 1/(1+z)\n'
             '• sqrt: scale = 1/√(1+z)\n'
             '• none: no scaling\n\n'
             'Red dashed lines:\n'
             'z=0, 1, 2, 5',
             fontsize=12,
             bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5),
             verticalalignment='center')

    plt.suptitle('Redshift-Dependent Fitting Range Scaling',
                 fontsize=16, fontweight='bold', y=0.98)
    plt.tight_layout(rect=[0, 0, 1, 0.96])

    # Save figure
    output_path = 'foggie-dynamics/plots/range_scaling_visualization.png'
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"Saved figure to {output_path}")

    # Print some example values
    print("\n" + "=" * 70)
    print("EXAMPLE: Stellar Bulge Range Scaling")
    print("=" * 70)
    print(f"{'Redshift':<12} {'Scale Factor':<15} {'Range [kpc]':<30}")
    print("-" * 70)

    for z in [0, 1, 2, 3, 5]:
        scale = 1.0 / (1.0 + z)
        r_min, r_max = scale_range_with_redshift(ranges_z0["Star Bulge"], z,
                                                 min_scale=0.1, scaling_mode="linear")
        print(f"z = {z:<7.1f}  a = {scale:<12.3f}  [{r_min:.3f}, {r_max:.3f}]")

    print("\n" + "=" * 70)
    print("IMPACT: Why This Matters at High-z")
    print("=" * 70)
    print("\nAt z=5 (cosmic time = 1 Gyr):")
    print("  • Galaxies are 6× more compact")
    print("  • Physical resolution is 6× better (for fixed comoving resolution)")
    print("  • Fixed z=0 ranges would miss the galaxy entirely!")
    print("\nWith scaling:")
    print("  • Bulge range: 0.5-3 kpc → 0.08-0.5 kpc")
    print("  • Disk range: 3-30 kpc → 0.5-5 kpc")
    print("  • Now properly samples high-z galaxy structure ✓")

    plt.show()


if __name__ == "__main__":
    main()
