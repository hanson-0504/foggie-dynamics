#!/bin/bash

# SLURM directives (start with #SBATCH)
#SBATCH --job-name=plot_foggie_data
#SBATCH --output=job_output/plot_job_%j.out
#SBATCH --error=job_output/plot_job_%j.err
#SBATCH --time=05:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=10G
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=s2153833@ed.ac.uk

# Setup environment
source /home/brs/tmox-2024/tmox-2024_conda
tmox-2024_activate
conda activate yt

# Working directory
cd /home/shanson/foggie-dynamics

# Add some debug info
echo "Job ID: $SLURM_JOB_ID"

python code/plot_code.py \
    --experiment r200_scaling \
    --config fitting_config_r200.yaml
    --halos halo_008508 \
    --names Tempest
    #--components dm \
    #--regions halo
