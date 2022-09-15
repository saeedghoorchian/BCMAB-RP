#!/bin/bash

#SBATCH --partition=cpu-long

#SBATCH --cpus-per-task=32
#SBATCH --nodes=1
# time in minutes
#SBATCH --time=2440
#SBATCH --mem=50G
#SBATCH --output=log/081_15.log
#SBATCH --error=log/081_15.err

#SBATCH --mail-type=END
#SBATCH --mail-user=elephunker1@gmail.com

# For debug:
scontrol show job $SLURM_JOB_ID


source /home/maghsudi/ekortukov80/.bashrc
conda activate bandit_env
 python ../4_tune_parameters.py --data amazon --trials 50000 --dimension 15 --num-rep 1 --config config/bcmabrp4.json
