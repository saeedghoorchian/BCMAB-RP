#!/bin/bash

#SBATCH --array=0-7   # maps 0 to 7 to SLURM_ARRAY_TASK_ID below
#SBATCH --partition=cpu-long

#SBATCH --cpus-per-task=16
#SBATCH --nodes=1
# time in minutes
#SBATCH --time=2440

#SBATCH --output=log/10_run_dlints${SLURM_ARRAY_TASK_ID}.log
#SBATCH --error=log/10_run_dlints${SLURM_ARRAY_TASK_ID}.log

#SBATCH --mail-type=END
#SBATCH --mail-user=elephunker1@gmail.com

# For debug:
scontrol show job $SLURM_JOB_ID


source /home/maghsudi/ekortukov80/.bashrc
conda activate bandit_env
python 4_tune_parameters.py --data movielens --trials 30000 --dimension 15 --num-rep 3 --config config/dlints${SLURM_ARRAY_TASK_ID}.json --tune --non-stationarity
