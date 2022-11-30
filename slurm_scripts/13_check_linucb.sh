#!/bin/bash

#SBATCH --partition=cpu-short

#SBATCH --cpus-per-task=2
#SBATCH --mem=10G
#SBATCH --nodes=1
# time in minutes
#SBATCH --time=300

#SBATCH --output=log/13_check_linucb.log
#SBATCH --error=log/13_check_linucb.err

#SBATCH --mail-type=END
#SBATCH --mail-user=elephunker1@gmail.com

# For debug:
scontrol show job $SLURM_JOB_ID


source /home/maghsudi/ekortukov80/.bashrc
conda activate bandit_env
 python 4_tune_parameters.py --data movielens --trials 100000 --dimension 15 --num-rep 1 \
 --config config/linucb.json  \
 --intervals  "[1, 5000, 10000, 20000, 35000, 50000, 65000, 80000, 100000]"
