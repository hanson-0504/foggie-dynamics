#!/bin/bash
#SBATCH --job-name=foggie_r200_exp
#SBATCH --output=job_output/r200_exp/job_%A_%a.out
#SBATCH --error=job_output/r200_exp/job_%A_%a.err
#SBATCH --time=168:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=7
#SBATCH --mem=160G
#SBATCH --array=1-6
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

# ---- Halo list ----
HALOS=(2392 2878 4123 5016 5036 8508)
HALO=${HALOS[$SLURM_ARRAY_TASK_ID-1]}

# ---- Clean up stale evolution file before rerun ----
RESULTS_DIR="/cephfs/shanson/FOGGIE/results/r200_scaling/halo_$(printf '%06d' $HALO)"
rm -f "${RESULTS_DIR}/evolution_time_series.h5"

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
    --snapshot_type RD \
    --raw_data_root /cephfs/brs/FOGGIE \
    --generate_profiles \
    --nwalkers 200 \
    --nsteps 100000 \
    --config config/fitting_config_r200.yaml \
    --experiment r200_scaling \
    --make_projections

echo "SUCCESS: All components completed for $HALO"
