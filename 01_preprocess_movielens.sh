#!/bin/bash
#SBATCH --cpus-per-task=1
#SBATCH --nodes=1
# time in minutes
#SBATCH --time=5

#SBATCH --output=log/1_preprocess_movielens.log
#SBATCH --error=log/1_preprocess_movielens.err

#SBATCH --mail-type=END
#SBATCH --mail-user=elephunker1@gmail.com

# For debug:
scontrol show job $SLURM_JOB_ID


source /home/maghsudi/ekortukov80/.bashrc
conda activate bandit_env
python 1_preprocess_movielens.py 
