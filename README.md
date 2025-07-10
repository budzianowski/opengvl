# opengvl
Welcome to the Generative Value Learning (GVL) evaluation leaderboard!
We provide a minimal implementation of [Generative Value Learning](https://www.arxiv.org/abs/2411.
04549). You can submit your model to compare with the current leaderboard.
We held hiddent test set that we evaluate internally to prevent from contamination. Please reach out to opengvl@gmail.com for submission.

## Setup

```bash
export HF_HOME=/net/pr2/projects/plgrid/plggrobovlm/
export UV_CACHE_DIR=/net/pr2/projects/plgrid/plggrobovlm/
export CONDA_PKGS_DIRS=/net/pr2/projects/plgrid/plggrobovlm/conda_pkgs

git clone git@github.com:budzianowski/opengvl.git
cd opengvl
uv venv
source .venv/bin/activate
uv sync
cd .. && git clone https://github.com/huggingface/lerobot.git && cd lerobot && uv pip install -e . && cd ../opengvl
```

## HPC setup

```bash
export HF_HOME=/net/pr2/projects/plgrid/plggrobovlm/
export UV_CACHE_DIR=/net/pr2/projects/plgrid/plggrobovlm/
export CONDA_PKGS_DIRS=/net/pr2/projects/plgrid/plggrobovlm/conda_pkgs
export PIP_CACHE_DIR=/net/pr2/projects/plgrid/plggrobovlm/pip_cache

module load Miniconda3
eval "$(conda shell.bash hook)"

# Temporary
conda config --add envs_dirs /net/pr2/projects/plgrid/plggrobovlm/conda/envs
conda config --add pkgs_dirs /net/pr2/projects/plgrid/plggrobovlm/conda/pkgs

# # Create the directory structure
# mkdir -p /net/pr2/projects/plgrid/plggrobovlm/conda/envs
# mkdir -p /net/pr2/projects/plgrid/plggrobovlm/conda/pkgs

# # Create new environment in the desired location
# conda create -p /net/pr2/projects/plgrid/plggrobovlm/conda/envs/gvl_cuda python=3.11 -y

# conda create -n gvl_cuda python=3.11 -y
conda activate gvl_cuda
# conda install -c conda-forge ffmpeg -y
# pip3 install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu128

git clone git@github.com:budzianowski/opengvl.git
cd opengvl
# pip install -r requirements.txt
# cd .. && git clone https://github.com/huggingface/lerobot.git && cd lerobot && pip install -e . && cd ../opengvl
```

# Run open source version

```bash
python src/main.py --name lerobot/fmb --max_frames 1 --model internvl
```
## HPC

```bash
sbatch experiments/eval.job
```

# Current leaderboard

| Model | OXE | FMB | Aloha | Human Video |
|-------|-----|-----|-----|-----|
| GPT-4o | 0.76 | 0.76 | 0.76 | 0.76 |
| InternVL | 0.75 | 0.75 | 0.75 | 0.75 |
| Gemini | 0.43 | 0.43 | 0.43 | 0.43 |
| Gemma | 0.54 | 0.54 | 0.54 | 0.54 |


# TODO:
1. Fix logic
2. Add reward awr for the algorithm for the lerobot as input
3. Adding compatibility to: https://github.com/open-compass/VLMEvalKit

# KNOWN ISSUES:
1. On Mac, you need to set the DYLD_FALLBACK_LIBRARY_PATH due to torchcodec issues.
```bash
export DYLD_FALLBACK_LIBRARY_PATH=/opt/homebrew/lib
```

# Acknowledgements
- [Generative Value Learning](https://www.arxiv.org/abs/2411.04549)