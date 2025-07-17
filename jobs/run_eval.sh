#!/bin/bash
#SBATCH --job-name=gemma3-12b-zero-shot-mutex
#SBATCH --output=slurm_outputs/gemma3-12b-zero-shot-mutex-%j.out
#SBATCH --time=08:00:00
#SBATCH --account=plgopenglv-gpu-a100
#SBATCH --partition=plgrid-gpu-a100
#SBATCH --cpus-per-task=4
#SBATCH --mem=80GB
#SBATCH --gres=gpu:3

# Set environment variables
export HF_HOME=/net/pr2/projects/plgrid/plggrobovlm/
export UV_CACHE_DIR=/net/pr2/projects/plgrid/plggrobovlm/
export CONDA_PKGS_DIRS=/net/pr2/projects/plgrid/plggrobovlm/conda_pkgs
export PIP_CACHE_DIR=/net/pr2/projects/plgrid/plggrobovlm/pip_cache


# Activate virtual environment assuming the job is submitted from the root of the opengvl directory.
module load Miniconda3
eval "$(conda shell.bash hook)"
conda activate open_gvl2

python src/main.py --config-path configs/gemma3-12b-0shot-mutex.json
