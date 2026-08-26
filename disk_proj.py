import sys

import numpy as np
import yt
from yt.utilities.math_utils import ortho_find


def _star(pfilter, data):
    return data["all", "creation_time"] > 0


yt.add_particle_filter(
    "star", function=_star, filtered_type="all", requires=["creation_time"]
)


def _dm(pfilter, data):
    return data["all", "creation_time"] <= 0


yt.add_particle_filter(
    "dm", function=_dm, filtered_type="all", requires=["creation_time"]
)

if __name__ == "__main__":
    input_path = str(sys.argv[1])
    # dir_split1 =  "/cephfs/brs/nobackup/FOGGIE/"
    # dir_split2 =  "/nref11c_nref9f/DD2520/DD2520" # halo_008508 has DD2427
    halo_id = input_path.split("/")[-4]

    ds = yt.load(input_path)  # type: ignore
    ds.add_particle_filter("star")
    ds.add_particle_filter("dm")
    v, c = ds.find_max(("gas", "dark_matter_density"))
    sp = ds.sphere(c, (100, "kpc"))
    l = sp.quantities.angular_momentum_vector(use_gas=True, use_particles=True)
    lu = l / np.linalg.norm(l)
    print(f"Angular momentum vector: {lu}.")

    z = ds.current_redshift
    vecs = ortho_find(lu)
    width = (float(2 * sp.radius.to("kpc") / (1 + z)), "kpc")

    for i, vec in enumerate(vecs):
        p = yt.ProjectionPlot(
            ds,
            vec,
            ("gas", "density"),
            weight_field=("gas", "density"),
            # data_source=sp,
            center=sp.center,
            width=width,
            depth=width,
        )
        p.set_axes_unit("kpc")
        p.set_cmap(("gas", "density"), "turbo")
        p.save(f"plots/{halo_id}/{str(ds)}_gas_{i}.png")

        p = yt.ParticleProjectionPlot(
            ds,
            vec,
            ("star", "particle_mass"),
            # data_source=sp
            center=sp.center,
            width=width,
            depth=width,
        )
        p.set_unit(("star", "particle_mass"), "Msun")
        p.set_cmap(("star", "particle_mass"), "turbo")
        p.set_axes_unit("kpc")
        p.save(f"plots/{halo_id}/{str(ds)}_stars_{i}.png")

        p = yt.ParticleProjectionPlot(
            ds,
            vec,
            ("dm", "particle_mass"),
            # data_source=sp
            center=sp.center,
            width=width,
            depth=width,
        )
        p.set_unit(("dm", "particle_mass"), "Msun")
        p.set_cmap(("dm", "particle_mass"), "turbo")
        p.set_axes_unit("kpc")
        p.save(f"plots/{halo_id}/{str(ds)}_dm_{i}.png")

    fields = ["particle_position", "particle_velocity", "particle_mass"]
    data = {
        (ftype, field): sp[ftype, field] for field in fields for ftype in ["dm", "star"]
    }
    ftypes = {(ftype, field): ftype for field in fields for ftype in ["dm", "star"]}
    #yt.save_as_dataset(
    #    ds,
    #    data=data,
    #    field_types=ftypes,
    #    filename=f"datasets/{halo_id}/{str(ds)}_particles.h5",
    #)
