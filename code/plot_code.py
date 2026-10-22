import argparse
from pathlib import Path

import matplotlib
import matplotlib.pyplot as plt
import yt  # type: ignore

from src import (
    DensityProfile,
    FittingResult,
    ModelRegistry,
    ProjectPaths,
    load_config,
    load_hdf5,
    plot_2d_model_comparison,
    plot_combined_components,
    plot_component_mass_evolution,
    plot_corner,
    plot_data_mass_evolution,
    plot_fitted_mass_evolution,
    plot_halo_shape_evolution,
    plot_profiles,
    plot_reddish_2022_figure,
    plot_results,
    plot_shape_analysis,
)

matplotlib.use("Agg")


def main():
    yt.set_log_level("warning")  # type: ignore
    PATHS = ProjectPaths.create(
        code_root=Path(__file__).resolve().parents[1],
        data_root=Path("/cephfs/shanson/FOGGIE"),
    )

    parser = argparse.ArgumentParser(description='Generate plots for FOGGIE analysis')
    parser.add_argument('--halos', nargs='+', default=["halo_002392", "halo_002878", "halo_004123", "halo_005016", "halo_005036", "halo_008508"])
    parser.add_argument('--names', nargs='+', default=['Hurricane', 'Cyclone', 'Blizzard', 'Squall', 'Maelstrom', 'Tempest'])
    parser.add_argument('--components', nargs='+', default=["star", "gas", "dm"])
    parser.add_argument('--regions', nargs='+', default=['bulge', 'disk', 'halo'])
    parser.add_argument('--experiment', default='r200_scaling', type=str)
    parser.add_argument('--config', default="fitting_config_r200.yaml")
    args = parser.parse_args()

    halos = args.halos
    names = args.names
    components = args.components
    regions = args.regions
    experiment = args.experiment
    config = load_config(PATHS.config_file(args.config))

    for halo, name in zip(halos, names):
        # get evolution data
        halo_ev_path = (PATHS.results / experiment / halo / "evolution_time_series.h5"
                        if experiment else PATHS.results / halo / "evolution_time_series.h5")
        halo_evolution_data = load_hdf5(halo_ev_path)
        redshifts = halo_evolution_data['time']['redshifts']
        print(f"Redsihfts for {halo}/{name}: {redshifts}")

        for redshift in redshifts:
            print(f"Processing {halo}/{name} at z={redshift:.2f}")
            data_path = PATHS.snapshot_data(halo, redshift, experiment=experiment)
            results_path = PATHS.snapshot_results(halo, redshift, experiment=experiment)
            plot_path = PATHS.snapshot_plots(halo, redshift, experiment=experiment)
            try:
                plot_profiles(halo, name, redshift, PATHS, config, experiment=experiment)
                plt.close()
            except Exception as e:
                print(f"error plotting profile data: {e}")
            # plot shape analysis if they exist in metadata
            try:
                metadata = load_hdf5(data_path / "metadata.h5")
                if "shape_tensors" in metadata:
                    plot_shape_analysis(halo, name, redshift, config, PATHS, True, experiment, False, metadata)
            except Exception as e:
                print(f"error plotting shape analysis: {e}")
            # plot results and corner plots for best models
            for comp in components:
                profile_file = Path(f"{data_path}/3d_profile_{comp}.h5")
                if not profile_file.exists():
                    print(f"Profile file not found for {comp} at z={redshift:.3f}, skipping")
                    continue
                comp_profile = DensityProfile.from_hdf5(str(profile_file), comp, False)

                for region in regions:
                    print(f"Processing {region} in {comp}...")
                    if not (results_path / f"{comp}_{region}_results.h5").exists():
                        print(f"{region} not in {comp}")
                        continue

                    comp_reg_results = load_hdf5(f"{results_path}/{comp}_{region}_results.h5")

                    models = [key for key in comp_reg_results.keys() if key != 'best_model']

                    evaluators = {model: ModelRegistry.get_evaluator(model) for model in models}

                    if region == "bulge":
                        data_range = {0: (0.8, 5)}
                    elif region == "halo":
                        if comp != "dm":
                            data_range = {0: (15, max(comp_profile.r_1d))}
                        else:
                            data_range = {0: (min(comp_profile.r_1d), max(comp_profile.r_1d))}
                    else:
                        data_range = None

                    best_model = str(comp_reg_results['best_model'])

                    # Skip if no best model exists
                    if not best_model:
                        print(f"No best model for {comp}/{region}. Skipping.")
                        continue

                    if region == "bulge" or region == "halo":
                        try:
                            plot_results(
                                coords=(comp_profile.r_1d,),
                                data=comp_profile.rho_1d,
                                model_summaries=comp_reg_results,
                                data_range=data_range,
                                coord_label="Radius [kpc]",
                                best_model_name=best_model,
                                component_name=region,
                                save_path=f'{plot_path}/{comp}_{region}_fit.png',
                                model_evaluators=evaluators,
                                config=config
                            )
                            plt.close()
                        except Exception as e:
                            print(f"error plotting fitting model results: {e}")
                    else:
                        try:
                            plot_2d_model_comparison(
                                R_bins=comp_profile.R_bins,
                                z_bins=comp_profile.z_bins,
                                rho_data=comp_profile.rho_2d,
                                model_evaluator=evaluators[best_model],
                                model_summary=comp_reg_results[best_model],
                                model_name=str(best_model),
                                config=config,
                                savefig=f'{plot_path}/{comp}_{region}_fit.png',
                                figsize=(20, 6)
                            )
                            plt.close()
                        except Exception as e:
                            print(f"error plotting 2d model comparison plot: {e}")
                    try:
                        plot_corner(
                            model_name=best_model,
                            summary=comp_reg_results[best_model],
                            save_path=f"{plot_path}/corner_plots/{comp}_{region}_{best_model}.png"
                        )
                        plt.close()
                    except Exception as e:
                        print(f"error plotting corner plots: {e}")
                    del comp_reg_results

                # plot combined models for each component
                comp_results = {
                    region: FittingResult.from_hdf5(results_path / f"{comp}_{region}_results.h5")
                    for region in regions
                    if (results_path / f"{comp}_{region}_results.h5").exists()
                }

                try:
                    plot_combined_components(
                        results=comp_results,
                        profile=comp_profile,
                        component_type=comp,
                        r_range=(0.8, max(comp_profile.r_1d)),
                        save_path=f'{plot_path}/total_{comp}_fit.png',
                        config=config
                    )
                    plt.close()
                except Exception as e:
                    print(f"Error fitting combined results as: {e}")
                del comp_results

        # Plot mass evolution
        for comp in ['star', 'gas', 'dm', 'baryon', 'total']:
            try:
                plot_fitted_mass_evolution(
                    halo, name, comp, config, PATHS, True, experiment=experiment
                )
            except Exception as e:
                print(f"Error plotting mass evolution as: {e}")
            try:
                plot_data_mass_evolution(
                    halo, name, comp, config, PATHS, True, experiment=experiment
                )
            except Exception as e:
                print(f"Error plotting mass evolution as: {e}")

        # Plot component mass evolution
        try:
            plot_component_mass_evolution(
                halo, name, config, PATHS, True, experiment=experiment
            )
            plot_reddish_2022_figure(
                halo, name, config, PATHS, save_fig=True, experiment=experiment,
                background_data_file="/home/shanson/foggie-dynamics/datasets/Dump_Growth_Rate_Rotation_eps0_1_test_z1dot3 1.hf5"
            )
        except Exception as e:
            print(f"Error plotting mass evolution as: {e}")

        # Plot halo shape parameter evolution
        try:
            plot_halo_shape_evolution(
                halo=halo,
                name=name,
                config=config,
                PATHS=PATHS,
                save_fig=True,
                experiment=experiment
            )
        except Exception as e:
            print(f"Error plotting halo shape evolution: {e}")

    print("All plots completed!")


if __name__ == "__main__":
    main()
