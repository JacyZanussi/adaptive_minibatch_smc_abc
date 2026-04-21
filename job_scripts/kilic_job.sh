#!/bin/bash
#SBATCH -p free
#SBATCH --constraint=avx512
#SBATCH --job-name=kilic
#SBATCH --error=slurm_logs/error_kilic_%a.txt
#SBATCH --output=slurm_logs/out_kilic_%a.txt
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem-per-cpu=4G  # Ensure you have enough RAM
#SBATCH --time=3:00:00
#SBATCH --array=0-1

# 1. Setup Local Scratch (This is the "consistency" magic)
# UCI HPC3 clears this automatically after the job finishes
LOCAL_SCRATCH="/tmp/$USER/$SLURM_JOB_ID"
mkdir -p "$LOCAL_SCRATCH"

# 2. Environment Setup
module load python/3.10.2
source "/dfs6b/pub/jzanussi/adaptive_minibatch_smc_abc/.venv/bin/activate"

# Tell Joblib and Multiprocessing where to work
export JOBLIB_TEMP_FOLDER="$LOCAL_SCRATCH"
export JOBLIB_START_METHOD="spawn"
export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK

# 3. Execution
# We stay in the DIR so Python can find your modules,
# but Joblib will use LOCAL_SCRATCH for its heavy I/O.
DIR='/dfs6b/pub/jzanussi/adaptive_minibatch_smc_abc'
cd "$DIR"

python -u application_kilic.py "$SLURM_ARRAY_TASK_ID"
