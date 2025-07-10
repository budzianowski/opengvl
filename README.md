# opengvl
Welcome to the Generative Value Learning (GVL) evaluation leaderboard!
We provide a minimal implementation of [Generative Value Learning](https://www.arxiv.org/abs/2411.
04549). You can submit your model to compare with the current leaderboard.
We held hiddent test set that we evaluate internally to prevent from contamination. Please reach out to opengvl@gmail.com for submission.

## Setup

```bash
git clone git@github.com:budzianowski/opengvl.git
cd opengvl
uv venv
source .venv/bin/activate
export HF_HOME=/net/pr2/projects/plgrid/plggrobovlm/
export UV_CACHE_DIR=/net/pr2/projects/plgrid/plggrobovlm/
export CONDA_PKGS_DIRS=/net/pr2/projects/plgrid/plggrobovlm/conda_pkgs
uv sync
cd ..
git clone https://github.com/huggingface/lerobot.git
cd lerobot
uv pip install -e .
cd ../opengvl
```

## HPC setup

```bash
conda create -n gvl python=3.11 -y
conda activate gvl
conda install -c conda-forge scipy tensorflow pytorch torchvision ffmpeg -y
pip install -r requirements.txt
git clone git@github.com:budzianowski/opengvl.git
cd opengvl
uv venv
source .venv/bin/activate
export HF_HOME=/net/pr2/projects/plgrid/plggrobovlm/
export UV_CACHE_DIR=/net/pr2/projects/plgrid/plggrobovlm/

uv sync
cd ..
git clone https://github.com/huggingface/lerobot.git
cd lerobot
uv pip install -e .
cd ../opengvl
```

# Run open source version

```bash
uv run src/main.py --name fmb:0.0.1 --max_frames 30 --model gpt4o
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