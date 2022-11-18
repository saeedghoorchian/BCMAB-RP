#!/bin/bash

#SBATCH --partition=cpu-short

#SBATCH --cpus-per-task=1
#SBATCH --nodes=1
#SBATCH --tasks-per-node=1
# time in minutes
#SBATCH --time=5

#SBATCH --output=log/2_preprocess_jester.log
#SBATCH --error=log/2_preprocess_jester.err

#SBATCH --mail-type=END
#SBATCH --mail-user=elephunker1@gmail.com

# For debug:
scontrol show job $SLURM_JOB_ID


source /home/maghsudi/ekortukov80/.bashrc
conda activate bandit_env
yes Y | python 2_preprocess_jester.py
