#!/bin/bash
#SBATCH --job-name=foggie_test
#SBATCH --output=job_output/test_job_%j.out
#SBATCH --error=job_output/test_job_%j.err
#SBATCH --time=05:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=7
#SBATCH --mem=64G
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
export FOGGIE_PROJECT_ROOT=/home/shanson/foggie-dynamics
# ---- Limit threading for libraries ----
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1


# ---- Working directory (code lives in $HOME) ----
cd /home/shanson/foggie-dynamics || exit 1

# ---- Test with single halo ----
HALO=2392

# ---- Debug info ----
echo "Test Job ID: $SLURM_JOB_ID"
echo "Halo: $HALO"
echo "FOGGIE_PROJECT_ROOT: $FOGGIE_PROJECT_ROOT"
echo "Running on host: $(hostname)"
echo "Test run with reduced resources"

# ---- Run analysis (test with single component first) ----
python code/foggie_analysis.py \
    --halo "$HALO" \
    --component star \
    --n_jobs "$SLURM_CPUS_PER_TASK" \
    --mode time-series \
    --snapshot_type RD \
    --raw_data_root /cephfs/brs/FOGGIE \
    --snapshots 0042 \
    --nwalkers 200 \
    --nsteps 50000

echo "SUCCESS: Test completed for halo $HALO"
