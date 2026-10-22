#!/bin/bash
#SBATCH --job-name=Tempest_analysis
#SBATCH --output=job_output/job_%j.out
#SBATCH --error=job_output/job_%j.err
#SBATCH --time=120:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=7
#SBATCH --mem=100G
##SBATCH --array=1-6
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

HALO=8508

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
    #--snapshots 0012 0013 0014 0015 0016 0017 \
    #0018 0019 0020 0021 0022 0023 0024 0025 0026 \
    #0027 0028 0029 0030 0031 0032 0033 0034 0037 \
    #0038 0039 0040 0041 # skipping 0035/0036/0042


echo "SUCCESS: All components completed for $HALO"
