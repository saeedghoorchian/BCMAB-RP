#!/bin/bash
#SBATCH --partition=cpu-long
#SBATCH --cpus-per-task=1
#SBATCH --nodes=1
#SBATCH --tasks-per-node=1
# time in minutes
#SBATCH --time=300

#SBATCH --output=log/1_preprocess_movielens.log
#SBATCH --error=log/1_preprocess_movielens.err

#SBATCH --mail-type=END
#SBATCH --mail-user=elephunker1@gmail.com

# For debug:
scontrol show job $SLURM_JOB_ID


source /home/maghsudi/ekortukov80/.bashrc
conda activate bandit_env
python 1_preprocess_movielens.py
