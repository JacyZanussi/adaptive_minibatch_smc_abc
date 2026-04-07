#!/bin/bash
#SBATCH -p free
#SBATCH --constraint=nvme
#SBATCH --job-name=hps
#SBATCH --error=slurm_logs/hpc_outputs/error_hps_%a.txt
#SBATCH --output=slurm_logs/hpc_outputs/out_hps_%a.txt
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --time=16:00:00
#SBATCH --array=0-7

satid=$SLURM_ARRAY_TASK_ID
echo "Hyperparameter Sweep"
echo "Right now, 0-3 correspond to transcriptional dynamics, while 4-7 correspond to lotka volterra"
echo "Date: $(date)"
echo "Host: $(hostname)"
echo "Job ID: $SLURM_JOB_ID"
echo "Array Task ID: $satid"
echo "Running in: $(pwd)"
echo "---------------------------------------"


# Set directory and prompt metadata to refresh
BASE_DIR="/dfs6b/pub/jzanussi/"
ls -d "$BASE_DIR" > /dev/null 2>&1
sleep 15

cd "$BASE_DIR" || { echo "Directory $BASE_DIR not found"; exit 1; }

# Activate environment
source "${BASE_DIR}/.venv/bin/activate" 

python -u experiment_physical_parameters.py "$satid"
