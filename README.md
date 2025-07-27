# OpenGVL

Welcome to the Generative Value Learning (GVL) evaluation leaderboard! We provide a minimal implementation of [Generative Value Learning](https://www.arxiv.org/abs/2411.04549). You can submit your model to compare with the current leaderboard.

We hold a hidden test set that we evaluate internally to prevent contamination. Please reach out to opengvl@gmail.com for submission.

## Setup

```bash
git clone https://github.com/budzianowski/opengvl.git
cd opengvl
uv venv
source .venv/bin/activate
uv pip install -e .
```

## Usage

The primary entry point to the application is the `gvl` command-line interface.

To run an evaluation or inference, use the `run` command. You can specify the data source, model, and other parameters.

### Evaluation from a dataset

```bash
gvl run --data-source dataset --dataset-name "lerobot/fmb" --model "gpt-4o-mini" --num-eval-steps 10
```

You can also use a configuration file to specify the evaluation parameters.

```bash
gvl run --config-path configs/gemini.json
```

### Inference from a directory of images

```bash
gvl run --data-source image-dir --image-dir /path/to/images --prompt "open the door" --model "gpt-4o-mini"
```

### Inference from a list of image files

```bash
gvl run --data-source image-files --image-files img1.jpg img2.jpg --prompt "pick up the cup" --model "gpt-4o-mini"
```

## Current Leaderboard

| Model    | OXE  | FMB  | Aloha | Human Video |
|----------|------|------|-------|-------------|
| GPT-4o   | 0.76 | 0.76 | 0.76  | 0.76        |
| InternVL | 0.75 | 0.75 | 0.75  | 0.75        |
| Gemini   | 0.43 | 0.43 | 0.43  | 0.43        |
| Gemma    | 0.54 | 0.54 | 0.54  | 0.54        |

## Known Issues

- On Mac, you may need to set the `DYLD_FALLBACK_LIBRARY_PATH` due to `torchcodec` issues:
  ```bash
  export DYLD_FALLBACK_LIBRARY_PATH=/opt/homebrew/lib
  ```

## Acknowledgements

- [Generative Value Learning](https://www.arxiv.org/abs/2411.04549)
