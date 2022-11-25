#!/bin/bash

#SBATCH --partition=cpu-long

#SBATCH --cpus-per-task=16
#SBATCH --nodes=1
# time in minutes
#SBATCH --time=2440

#SBATCH --output=log/082_50.log
#SBATCH --error=log/082_50.err

#SBATCH --mail-type=END
#SBATCH --mail-user=elephunker1@gmail.com

# For debug:
scontrol show job $SLURM_JOB_ID


source /home/maghsudi/ekortukov80/.bashrc
conda activate bandit_env
python 4_tune_parameters.py --data movielens --trials 30000 --dimension 60 --num-rep 3 --config config/cbrap.json --tune
