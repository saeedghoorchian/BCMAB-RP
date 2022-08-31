#!/bin/bash
#SBATCH --cpus-per-task=1
#SBATCH --nodes=1
#SBATCH --tasks-per-node=1
# time in minutes
#SBATCH --time=5

#SBATCH --output=log/0_preprocess_amazon.log
#SBATCH --error=log/1_preprocess_amazon.err

#SBATCH --mail-type=END
#SBATCH --mail-user=elephunker1@gmail.com

# For debug:
scontrol show job $SLURM_JOB_ID


source /home/maghsudi/ekortukov80/.bashrc
conda activate bandit_env
python ../5_preprocess_amazon.py
