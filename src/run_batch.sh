#!/bin/bash
#SBATCH --time=6:00:00
#SBATCH --gres=gpu:v100:1
#SBATCH --mem=48G

module load anaconda3
srun python main.py
