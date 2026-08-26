# Redshift-Dependent Range Scaling

## Overview

As of January 2026, the fitting pipeline now supports **redshift-dependent range scaling** to account for:

1. **Better physical resolution at high-z**: Comoving resolution → smaller physical scales
2. **Smaller galaxy sizes at high-z**: Galaxies grow with cosmic time
3. **Virial radius evolution**: R_vir ∝ 1/(1+z) in standard cosmologies

## How It Works

All fitting ranges in `config/fitting_config.yaml` are specified at **z=0** (reference ranges). During fitting, these ranges are automatically scaled based on redshift using:

```
r(z) = r_z0 × scale_factor
```

Where `scale_factor` depends on the chosen scaling mode.

## Scaling Modes

### 1. Linear Scaling (Recommended)
```yaml
range_scaling:
  mode: "linear"
```
- Uses cosmological scale factor: **a = 1/(1+z)**
- Most physically motivated
- At z=5: scale_factor = 0.167 (6× smaller)
- At z=2: scale_factor = 0.333 (3× smaller)
- At z=0: scale_factor = 1.0 (no scaling)

### 2. Square-Root Scaling
```yaml
range_scaling:
  mode: "sqrt"
```
- Uses: **scale_factor = 1/√(1+z)**
- Gentler scaling than linear
- At z=5: scale_factor = 0.408 (2.45× smaller)

### 3. Logarithmic Scaling
```yaml
range_scaling:
  mode: "log"
```
- Uses: **scale_factor = 1/(1+log₁₀(2+z))**
- Very gentle scaling
- For testing/debugging

### 4. No Scaling
```yaml
range_scaling:
  mode: "none"
```
- Disables scaling (uses r_z0 at all redshifts)
- Backward compatibility default

## Configuration

### Global Settings

In `config/fitting_config.yaml`:

```yaml
range_scaling:
  enabled: true      # Global toggle
  mode: "linear"     # Default mode for all components
  min_scale: 0.1     # Don't scale below 10% of z=0 size
  max_scale: 1.0     # Don't scale above z=0 size
```

### Per-Component Override

Each component can override global settings:

```yaml
star:
  bulge:
    r_range: [0.5, 3]  # Range at z=0
    range_scaling:
      mode: "linear"   # Override global mode
      min_scale: 0.1
      max_scale: 1.0
```

## Examples

### Example 1: Stellar Bulge at Different Redshifts

Config at z=0:
```yaml
star:
  bulge:
    r_range: [0.5, 3.0]  # kpc
    range_scaling:
      mode: "linear"
```

Effective ranges:
- **z=0**: [0.5, 3.0] kpc (no scaling)
- **z=1**: [0.25, 1.5] kpc (scale = 0.5)
- **z=2**: [0.167, 1.0] kpc (scale = 0.333)
- **z=5**: [0.083, 0.5] kpc (scale = 0.167)

### Example 2: Stellar Disk at Different Redshifts

Config at z=0:
```yaml
star:
  disk:
    R_range: [3.0, 30.0]  # kpc (radial)
    z_max: 7.0            # kpc (vertical)
    range_scaling:
      mode: "linear"
      min_scale: 0.15     # Don't go below 15%
```

Effective ranges:
- **z=0**: R=[3.0, 30.0], z_max=7.0 kpc
- **z=2**: R=[1.0, 10.0], z_max=2.33 kpc
- **z=5**: R=[0.5, 5.0], z_max=1.17 kpc (but clamped by min_scale=0.15)

### Example 3: Dark Matter Halo

Config at z=0:
```yaml
dm:
  halo:
    r_range: [0.3, 250.0]  # kpc
    range_scaling:
      mode: "linear"
```

Effective ranges:
- **z=0**: [0.3, 250.0] kpc
- **z=2**: [0.1, 83.3] kpc
- **z=5**: [0.05, 41.7] kpc

## Why This Matters

### Problem: Fixed Ranges at High-z

At z=5 with fixed z=0 ranges:
- Stellar disk fitting: R_range = [3, 30] kpc
- But galaxy disk scale length ~ 0.5-2 kpc at z=5!
- Result: **"Not enough valid points in fitting region: 0"**

### Solution: Scaled Ranges

At z=5 with scaled ranges:
- Stellar disk fitting: R_range = [0.5, 5] kpc (scaled by 1/6)
- Now captures the compact high-z disk structure
- Result: **Successful fit with proper sampling**

## Implementation Details

### Code Location

- **Scaling function**: `src/filters_and_utils.py::scale_range_with_redshift()`
- **Integration**: `src/workflow.py::fit_component_models()`
- **Config**: `config/fitting_config.yaml`

### Workflow

1. Range specified in config at z=0
2. Metadata contains redshift from snapshot
3. `fit_component_models()` extracts range_scaling settings
4. Calls `scale_range_with_redshift()` to compute effective range
5. Logs scaling: `"Scaled r_range for z=5.000: [0.5, 3.0] → [0.083, 0.5] kpc"`
6. Fitting proceeds with scaled range

## Best Practices

1. **Use linear scaling** unless you have specific reasons not to
2. **Set min_scale** to prevent unreasonably small ranges (recommend 0.1-0.15)
3. **Define ranges at z=0** for consistency and physical intuition
4. **Check logs** to verify scaling is working as expected
5. **Test with a range of redshifts** to ensure convergence across cosmic time

## Disabling Scaling

To disable scaling (e.g., for testing or backward compatibility):

### Option 1: Global disable
```yaml
range_scaling:
  enabled: false  # or set mode: "none"
```

### Option 2: Per-component disable
```yaml
star:
  bulge:
    range_scaling:
      mode: "none"
```

## Future Improvements

Potential enhancements:
- [ ] Redshift-dependent initial guesses (p0)
- [ ] Redshift-dependent bounds
- [ ] Comoving coordinate option
- [ ] Automatic R200-relative ranges

## References

- Scale factor: a = 1/(1+z) from FLRW cosmology
- Virial scaling: Mo, van den Bosch, White (2010), Galaxy Formation and Evolution
- Resolution scaling: Bryan & Norman (1998), ApJ 495, 80
