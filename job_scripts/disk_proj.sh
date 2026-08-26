#!/bin/bash

# SLURM directives (start with #SBATCH)
#SBATCH --job-name=get_disk_projections
#SBATCH --output=job_output/job_%A_%a.out
#SBATCH --error=job_output/job_%A_%a.err
#SBATCH --time=03:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --array=1-5  # Adjust based on number of directories
#SBATCH --mail-type=ALL
#SBATCH --mail-user=s2153833@ed.ac.uk

# Setup environment
source /home/brs/tmox-2024/tmox-2024_conda
tmox-2024_activate
conda activate yt

# Working directory
cd /home/shanson/foggie-dynamics

DIRS=(
    "/cephfs/brs/FOGGIE/halo_002392/nref11c_nref9f/DD2520/DD2520"
    "/cephfs/brs/FOGGIE/halo_002878/nref11c_nref9f/DD2520/DD2520"
    "/cephfs/brs/FOGGIE/halo_004123/nref11c_nref9f/DD2520/DD2520"
    "/cephfs/brs/FOGGIE/halo_005016/nref11c_nref9f/DD2520/DD2520"
    "/cephfs/brs/FOGGIE/halo_005036/nref11c_nref9f/DD2520/DD2520"
    "/cephfs/brs/FOGGIE/halo_008508/nref11c_nref9f/DD2427/DD2427"
)

DIR=${DIRS[$SLURM_ARRAY_TASK_ID-1]}

# Add some debug info
echo "Job ID: $SLURM_JOB_ID"
echo "Array Task ID: $SLURM_ARRAY_TASK_ID"

echo "Processing directory: $DIR"
python src/disk_proj.py "$DIR"
