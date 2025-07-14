#!/bin/bash
#SBATCH --job-name=gemma3-4b-zero-shot-toto
#SBATCH --output=gemma3-4b-zero-shot-toto-%j.out
#SBATCH --error=gemma3-4b-zero-shot-toto-%j.err
#SBATCH --time=08:00:00
#SBATCH --account=plgopenglv-gpu-a100
#SBATCH --partition=plgrid-gpu-a100
#SBATCH --cpus-per-task=4
#SBATCH --mem=48G
#SBATCH --gres=gpu:1

# Set environment variables
export HF_HOME=/net/pr2/projects/plgrid/plggrobovlm/
export UV_CACHE_DIR=/net/pr2/projects/plgrid/plggrobovlm/
export CONDA_PKGS_DIRS=/net/pr2/projects/plgrid/plggrobovlm/conda_pkgs
export PIP_CACHE_DIR=/net/pr2/projects/plgrid/plggrobovlm/pip_cache


# Activate virtual environment assuming the job is submitted from the root of the opengvl directory.
module load Miniconda3
eval "$(conda shell.bash hook)"
conda activate gvl_cuda

# Run the command

# lerobot/fmb, lerobot/utaustin_mutex, lerobot/toto

python src/main.py --name lerobot/toto --max_frames 20 --model gemma --num_eval_steps 200 --num_context_episodes 0
