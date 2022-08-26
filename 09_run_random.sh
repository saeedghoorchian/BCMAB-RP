#!/bin/bash

#SBATCH --partition=cpu-long

#SBATCH --cpus-per-task=1
#SBATCH --nodes=1
# time in minutes
#SBATCH --time=2440

#SBATCH --output=log/09_run_random.log
#SBATCH --error=log/09_run_random.log

#SBATCH --mail-type=END
#SBATCH --mail-user=elephunker1@gmail.com

# For debug:
scontrol show job $SLURM_JOB_ID


source /home/maghsudi/ekortukov80/.bashrc
conda activate bandit_env
 python 4_tune_parameters.py --data movielens --trials 100000 --dimension 15 --num-rep 5 --config config/random.json
