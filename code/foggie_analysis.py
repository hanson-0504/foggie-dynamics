# Comparison Script for FOGGIE Dynamics Analysis
# Version 2 - run on server
# collect all snapshots for a given halo
# collect density profiles
# complete fitting routine
# calculate masses
# store results with redshift metadata
# analyse and plot

import argparse
import sys
from pathlib import Path
from time import time

from src import ProjectPaths, workflow
from src.analysis import generate_component_summary
from src.config import load_config, setup_logging
from src.io import load_hdf5

# Initialize project paths
PATHS = ProjectPaths.create()


def main():
    t0 = time()
    parser = argparse.ArgumentParser(
        description="FOGGIE Dynamics Analysis: Fit density profiles to simulation data",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Single halo
  python foggie_analysis.py --halo 8508 --component star --show_plots --save_data
  python foggie_analysis.py --halo 8508 --component dm --save_data
  python foggie_analysis.py --halo 8508 --component all --save_data

  # Multiple halos
  python foggie_analysis.py --halo 8508 5036 5016 --component all --save_data
  python foggie_analysis.py --halo 8508 5036 --component star --save_data
        """,
    )

    parser.add_argument(
        "--halo",
        type=str,
        required=True,
        nargs="+",
        help="Halo ID(s) (dataset directory name). Can specify multiple halos.",
    )
    parser.add_argument(
        "--component",
        type=str,
        required=True,
        default="all",
        choices=["star", "dm", "gas", "all"],
        help="Component to analyze: star, dm, gas, or all",
    )
    parser.add_argument(
        "--save_data",
        action="store_true",
        help="Save fitting results to HDF5 files"
    )
    parser.add_argument(
        "--config",
        type=str,
        help="Path to config YAML (default: config/fitting_config.yaml)",
    )
    parser.add_argument(
        "--nwalkers",
        type=int,
        default=None,
        help="Override number of MCMC walkers"
    )
    parser.add_argument(
        "--nsteps",
        type=int,
        default=None,
        help="Override number of MCMC steps"
    )
    parser.add_argument(
        "--mode",
        choices=["single", "time-series"],
        default="single",
        help="Analysis mode: single snapshot or time-series",
    )
    parser.add_argument(
        "--raw_data_root",
        type=Path,
        default=Path("/cephfs/brs/FOGGIE"),
        help="Root directory of raw FOGGIE simulation outputs (read-only)",
    )
    parser.add_argument(
        "--snapshots",
        type=int,
        nargs="+",
        help="Specific DD/RD numbers to process (optional)",
    )
    parser.add_argument(
        "--n_jobs",
        type=int,
        default=1,
        help="Number of parallel jobs. -1 = all cores, 1 = serial, N = use N cores",
    )
    parser.add_argument(
        "--snapshot_type",
        type=str,
        choices=["DD", "RD", "BOTH"],
        default="DD",
        help="Type of snapshots to process in time-series mode (DD, RD, or BOTH) (default: DD)",
    )
    parser.add_argument(
        "--generate_profiles",
        action="store_true",
        help="Regenerate density profiles even if they already exist",
    )
    parser.add_argument(
        "--experiment",
        type=str,
        default=None,
        help="Experiment name for organizing results (creates subdirectory in results/)",
    )
    parser.add_argument(
        "--sim_type",
        type=str,
        default="full",
        choices=["full", "dmo", "gas_only"],
        help="Specify simulation type of data"
    )
    parser.add_argument(
        "--sim_subdir",
        type=str,
        default=None,
        help="Subdirectory within halo folder for raw data (default: nref11c_nref9f)",
    )
    parser.add_argument(
        "--make_projections",
        action="store_true",
        help="Create Projection plots with R200 overlay for each snapshot"
    )
    parser.add_argument(
        "--center_find_method",
        type=str,
        default="average",
        choices=["initial_only", "shrinking_sphere", "shrinking_sphere_global", "average"],
        help="Center-finding: initial_only (find_max/histogram), "
             "shrinking_sphere (refine from initial guess), "
             "shrinking_sphere_global (from whole box, no initial guess), "
             "average (mean of both, default)",
    )
    parser.add_argument(
        "--start_snapshot",
        type=int,
        default=None,
        help="Starting snapshot number for time-series analysis (optional, overrides --snapshots)"
    )

    args = parser.parse_args()

    # load config first
    try:
        config = load_config(args.config)
    except FileNotFoundError:
        print("Config file not found. Please create config/fitting_config.yaml")
        sys.exit(1)

    # Format halo_ids: if just a number is provided, prepend 'halo_00'
    halo_ids = [h if h.startswith("halo_00") else f"halo_00{h}" for h in args.halo]
    component = args.component
    save_data = bool(args.save_data)
    n_jobs = int(args.n_jobs)
    snapshot_type = str(args.snapshot_type)
    sim_type = str(args.sim_type)

    # Handle 'all' components case
    if sim_type == "full":
        components_to_run = ["star", "gas", "dm"] if component == "all" else [component]
    elif sim_type == "gas_only":
        components_to_run = ["gas", "dm"] if component == "all" else [component]
    elif sim_type == "dmo":
        components_to_run = ["dm"]
        if component != "all" and component != "dm":
            print(f"Warning: --component {component} ignored for dmo sim_type")
    else:
        raise ValueError(f"Unknown sim_type: {sim_type}")

    # Get experiment name
    experiment = args.experiment

    # Print initial summary to console
    print(f"Processing {len(halo_ids)} halo(s): {', '.join(halo_ids)}")
    print(f"Components: {', '.join(components_to_run)}")
    if experiment:
        print(f"Experiment: {experiment}")
    print("=" * 80)

    # Apply CLI overrides to config if provided
    if args.nwalkers is not None or args.nsteps is not None:
        for comp_name in components_to_run:
            component_config = config.get(comp_name)
            if component_config:
                for subcomponent in component_config.values():
                    if isinstance(subcomponent, dict) and "fitting" in subcomponent:
                        if args.nwalkers is not None:
                            subcomponent["fitting"]["nwalkers"] = args.nwalkers
                        if args.nsteps is not None:
                            subcomponent["fitting"]["nsteps"] = args.nsteps

    # Loop over all halos
    for halo_id in halo_ids:
        # Choose analysis mode
        if args.mode == "single":
            # Process single snapshot for each component
            for current_component in components_to_run:
                # Setup separate logger for each halo+component combination
                logger = setup_logging(current_component, halo_id)

                logger.info("=" * 80)
                logger.info(f"Starting FOGGIE Dynamics Analysis for {halo_id}")
                logger.info(f"PROCESSING COMPONENT: {current_component.upper()}")
                logger.info("=" * 80)

                try:
                    result = workflow.process_snapshot(
                        halo_id=halo_id,
                        component=current_component,
                        config=config,
                        save_data=save_data,
                        verbose=True,
                        logger=logger,
                        n_jobs=n_jobs,
                        experiment=experiment,
                        sim_type=sim_type,
                    )

                    # Generate summary with statistics
                    if result and "results" in result:
                        suffix = f"_{sim_type}" if sim_type != "full" else ""
                        metadata = load_hdf5(
                            str(PATHS.data / halo_id / f"metadata{suffix}.h5")
                        )
                        generate_component_summary(
                            result["results"],
                            component_type=current_component,
                            model_masses=result.get("model_masses"),
                            metadata_dict=metadata,
                        )

                except Exception as e:
                    logger.error(
                        f"Error processing {current_component}: {e}",
                        exc_info=True
                    )
                    continue

        elif args.mode == "time-series":
            # Process time-series - create one log per halo (all components together)
            logger = setup_logging("timeseries", halo_id)

            # Process time-series for all components
            if args.raw_data_root is None:
                logger.error("--raw_data_root is required for time-series mode")
                sys.exit(1)
            logger.debug(f"Raw simulation root: {args.raw_data_root}")
            logger.debug(f"Processed output root: {PATHS.data_root}")

            logger.info("=" * 80)
            logger.info(f"Starting Time-Series Analysis for {halo_id}")
            logger.info(f"Components: {', '.join(components_to_run)}")
            logger.info("=" * 80)

            try:
                evolution_data = workflow.process_time_series(
                    raw_data_root=args.raw_data_root,
                    halo_id=halo_id,
                    components=components_to_run,
                    snapshot_type=snapshot_type,
                    snapshot_list=args.snapshots,
                    config=config,
                    save_data=save_data,
                    n_jobs=n_jobs,
                    logger=logger,
                    generate_profile=bool(args.generate_profiles),
                    experiment=experiment,
                    sim_type=sim_type,
                    sim_subdir=args.sim_subdir,
                    make_projections=args.make_projections,
                    center_find_method=args.center_find_method,
                    start_snapshot=args.start_snapshot,
                )

                logger.info("Time-series analysis complete!")
                logger.info(f"Processed {len(evolution_data['snapshots'])} snapshots")
                logger.info(f"Completed analysis for {halo_id}\n")

            except Exception as e:
                logger.error(f"Error in time-series analysis: {e}", exc_info=True)
                # Continue to next halo instead of exiting
                continue

    # Print final summary to console
    print("=" * 80)
    print(f"All halos processed. Total analysis time: {time() - t0:.1f} seconds.")
    print("=" * 80)


if __name__ == "__main__":
    main()
