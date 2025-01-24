#!/bin/bash
#SBATCH -p par-single
#SBATCH --job-name=wandb_sweep_ts809
#SBATCH --output=wandb/slurm_logs/sweep_%j.out
#SBATCH --error=wandb/slurm_logs/sweep_%j.err
#SBATCH --time=20:00
#SBATCH --ntasks=4
#SBATCH --mem=48G

# module load python/3.8  # Or your preferred environment setup
# source activate your_env  # Activate your virtualenv/conda env
conda activate /home/users/ts809/miniconda3/envs/project_CF

# wandb agent <entity>/<project>/<sweep_id>
wandb agent --count 1 ts809_lab/logreg-hyperparameter-tuning/txshuqe7