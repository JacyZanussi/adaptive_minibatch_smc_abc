#!/bin/bash
#SBATCH -p free
#SBATCH --constraint=avx512,nvme
#SBATCH --job-name=pps
#SBATCH --error=slurm_logs/error_pps_%a.txt
#SBATCH --output=slurm_logs/out_pps_%a.txt
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --time=16:00:00
#SBATCH --array=6-11

satid=$SLURM_ARRAY_TASK_ID
echo "Physical Parameter Sweep"
echo "Right now, 0-5 correspond to transcriptional dynamics, while 6-11 correspond to lotka volterra"
echo "Date: $(date)"
echo "Host: $(hostname)"
echo "Job ID: $SLURM_JOB_ID"
echo "Array Task ID: $satid"
echo "Running in: $(pwd)"
echo "---------------------------------------"


# 1. Setup Local NVMe Scratch
LOCAL_SCRATCH="/tmp/$USER_$SLURM_JOB_ID"
mkdir -p "$LOCAL_SCRATCH"

# 2. Stage-In: Copy everything to the node's local NVMe
# This makes those 50-60 data loads nearly instantaneous
cp -r /dfs6b/pub/jzanussi/adaptive_minibatch_smc_abc/* "$LOCAL_SCRATCH/"
cd "$LOCAL_SCRATCH"

# 3. Environment & Run
module load python/3.14.3
source "/dfs6b/pub/jzanussi/adaptive_minibatch_smc_abc/.venv/bin/activate"

# Ensure your script writes its pickled results to the current directory (LOCAL_SCRATCH)
export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK
python -u experiment_physical_parameters.py "$SLURM_ARRAY_TASK_ID"

# Copy only the files from the local NVMe to your persistent storage
cp -v "$LOCAL_SCRATCH"/results/* /dfs6b/pub/jzanussi/adaptive_minibatch_smc_abc/results/

# Cleanup
rm -rf "$LOCAL_SCRATCH"
