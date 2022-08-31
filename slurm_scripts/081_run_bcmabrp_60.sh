#!/bin/bash

#SBATCH --partition=cpu-long

#SBATCH --cpus-per-task=32
#SBATCH --nodes=1
# time in minutes
#SBATCH --time=2440

#SBATCH --output=log/081_60_$1.log
#SBATCH --error=log/081_60_$1.err

#SBATCH --mail-type=END
#SBATCH --mail-user=elephunker1@gmail.com

# For debug:
scontrol show job $SLURM_JOB_ID


source /home/maghsudi/ekortukov80/.bashrc
conda activate bandit_env
 python ../4_tune_parameters.py --data movielens --trials 100000 --dimension 24 --num-rep 5 --config config/bcmabrp$1.json
