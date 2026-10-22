#!/bin/bash
#SBATCH --job-name=foggie_gas_sim
#SBATCH --output=job_output/gas_sim/job_%j.out
#SBATCH --error=job_output/gas_sim/job_%j.err
#SBATCH --time=60:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=100G
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=s2153833@ed.ac.uk

# ---- Fail fast on errors ----
set -euo pipefail
trap 'echo "ERROR: Job failed for HALO=$HALO at line $LINENO" >&2' ERR

# ---- Environment ----
source /home/brs/tmox-2024/tmox-2024_conda
tmox-2024_activate
conda activate yt

# ---- Project paths ----
export FOGGIE_DATA=/cephfs/shanson/FOGGIE

# ---- Limit threading for libraries ----
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1

# ---- Working directory (code lives in $HOME) ----
cd /home/shanson/foggie-dynamics || exit 1

HALO=5036

# ---- Clean up stale evolution file before rerun ----
RESULTS_DIR="/cephfs/shanson/FOGGIE/results/gas_sim/halo_005036"
#rm -f "${RESULTS_DIR}/evolution_time_series.h5"

# ---- Debug info ----
echo "Job ID: $SLURM_JOB_ID"
echo "Array Task ID: ${SLURM_ARRAY_TASK_ID:-<none>}"
echo "Halo: $HALO"
echo "Running on host: $(hostname)"

# ---- Run analysis ----
python code/foggie_analysis.py \
    --halo "$HALO" \
    --component all \
    --save_data \
    --n_jobs "$SLURM_CPUS_PER_TASK" \
    --mode time-series \
    --snapshot_type DD \
    --raw_data_root /cephfs/brs/bw_mw/mean/ \
    --sim_subdir gas+dm-L4_redo \
    --sim_type gas_only \
    --config config/fitting_config_r200.yaml \
    --experiment gas_sim \
    --generate_profiles \
    --make_projections \
    --nwalkers 200 \
    --nsteps 100000 \
    --center_find_method shrinking_sphere_global \
    --start_snapshot 28

echo "SUCCESS: All components completed for $HALO"
