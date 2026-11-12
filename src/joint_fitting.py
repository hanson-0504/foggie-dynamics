"""Joint (simultaneous) fitting of bulge + disk + halo to the 2D (R, z)
profile."""

from dataclasses import dataclass, field
import numpy as np
from .models import ModelRegistry
from .fitting import fit_general_emcee
from .analysis import save_results

@dataclass
class JointModel:
    models: dict
    flatten: tuple = ()
    names: dict = field(init=False)
    slices: dict = field(init=False)
    
    def __post_init__(self):
        self.names, self.slices, i = {}, {}, 0
        for sub, m in self.models.items():
            names = list(ModelRegistry.get_param_names(m))
            if sub in self.flatten:
                if ModelRegistry.get_geometry(m) != "spherical":
                    raise ValueError(f"Can only flatten spherical models; {sub} is {m}")
                names.append("q")
            self.names[sub] = names
            self.slices[sub] = slice(i, i + len(names))
            i += len(names)
        self._eval = {s: ModelRegistry.get_evaluator(m)
                      for s, m in self.models.items()}
        self._geom = {s: ModelRegistry.get_geometry(m)
                      for s, m in self.models.items()}

    @property
    def param_names(self):
        return [f"{s}.{p}" for s in self.models for p in self.names[s]]
    
    def components(self, R, z, theta):
        out = {}
        for s in self.models:
            params = dict(zip(self.names[s], theta[self.slices[s]]))
            if self._geom[s] == "spherical":
                q = params.pop("q", 1.0) # evaluator never sees q
                coords = (np.sqrt(R**2 + (z / q)**2),)
            else:
                coords = (R, z)
            out[s] = self._eval[s](coords, params)
        return out
    
    def __call__(self, R, z, *theta):
        return sum(self.components(R, z, np.asarray(theta)).values())
    
    def split_summary(self, summary):
        """Joint summary -> {sub: per-component summary} in the usual format."""
        shared = {k: summary[k] for k in ("sigma", "acceptance_fraction")
                  if k in summary}
        out = {}
        for s in self.models:
            sub = dict(shared, joint=True, samples=summary["samples"][:, self.slices[s]])
            for p in self.names[s]:
                sub[p] = summary[f"{s}.{p}"]
                sub[f"{p}_err"] = summary[f"{s}.{p}_err"]
            out[s] = sub
        return out

def joint_bounds_p0(jm, comp_config, init_results):
    bounds, p0 = {}, {}
    for s in jm.models:
        cfg_b = comp_config[s]["bounds"]
        init = init_results[s].summary or {}
        for p in jm.names[s]:
            bounds[f"{s}.{p}"] = tuple(cfg_b[p])
            p0[f"{s}.{p}"] = init.get(p, comp_config[s]["p0"].get(p, np.mean(cfg_b[p])))
    bounds["log_sigma"] = (-6, 2)
    return bounds, p0

def make_joint_prior(jm, jcfg):
    """
    Log-prior for the joint fit.
    - ordering: bulge scale radius < halo scale radius (stops bulge/halo swapping)
    - optional Gaussian priors from config: joint.priors: {"bulge.log10_rs": [mu, sigma]}
    """
    gauss = {k: tuple(v) for k, v in (jcfg.get("priors") or {}).items()}
    order = [(a, b) for a, b in jcfg.get("ordering", [("bulge.log10_rs", "halo.log10_rs")])
            if a in jm.param_names and b in jm.param_names]
    
    def log_prior(p):
        for a, b in order:
            if p[a] >= p[b]:
                return -np.inf
        lp = 0.0
        for k, (mu, sig) in gauss.items():
            if k in p:
                lp -= 0.5 * ((p[k] - mu) / sig) **2
        return lp

    return log_prior


def fit_joint(component, profile, config, init_results, logger, metadata,
              uncertainties=None, **fit_kwargs):
    jcfg = config[component].get("joint", {})
    models = jcfg.get("models") or {s: r.best_model for s, r in init_results.items()
                                    if r.best_model}
    jm = JointModel(models, flatten=tuple(jcfg.get("flatten") or ()))
    bounds, p0 = joint_bounds_p0(jm, config[component], init_results)

    R, z = (g.ravel() for g in np.meshgrid(profile.R, profile.z, indexing="ij"))
    R200 = (metadata or {}).get("R200_kpc")
    r_max = jcfg.get("r_max") or R200 or np.hypot(R.max(), z.max())
    if R200 is None and "r_max" not in jcfg:
        logger.warning("No R200_kpc in metadata; joint fit uses full 2D grid")
    
    rho = profile.rho_2d.ravel().astype(float)
    rho = np.where(np.hypot(R, z) <= r_max, rho, np.nan)   # spherical cut at r_max
    logger.info(f"Joint fit: r <= {r_max:.1f} kpc, grid R<={R.max():.1f}, z<={z.max():.1f}")

    samples, summary = fit_general_emcee(
        coords=(R, z), data=rho,
        model_func=jm, param_names=jm.param_names,
        bounds=bounds, p0=p0, logger=logger,
        soft_priors=make_joint_prior(jm, jcfg),
        uncertainties=uncertainties,
        nwalkers=jcfg.get("nwalkers", 128), nsteps=jcfg.get("nsteps", 30000),
        **fit_kwargs,
    )
    if summary is None:
        return None

    summary["param_names"] = np.array(jm.param_names + ["log_sigma"], dtype="S")
    return jm, jm.split_summary(summary), summary
