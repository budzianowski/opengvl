# OpenGVL: Open Generative Value Learning

[![PyPI version](https://badge.fury.io/py/gvl.svg)](https://badge.fury.io/py/gvl)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Version](https://img.shields.io/badge/version-v0.1-blue)](https://github.com/your-username/opengvl-codebase_refactor)
[![arXiv](https://img.shields.io/badge/arXiv-2411.04549-b31b1b.svg)](https://arxiv.org/abs/2411.04549)

An open-source implementation of Generative Value Learning for robotics and beyond.

<table>
  <tr>
    <td align="center"><a href="#quick-start">Quick Start</a></td>
    <td align="center"><a href="#getting-started">Getting Started</a></td>
    <td align="center"><a href="#extending-opengvl">Extending OpenGVL</a></td>
    <td align="center"><a href="#evaluation">Evaluation</a></td>
  </tr>
</table>

## About

OpenGVL provides a benchmark to evaluate how well Vision-Language Models (VLMs) understand temporal progress in robotics tasks. It can automatically annotate and curate large-scale robotics datasets by predicting task completion from video frames, making it practical for data quality assessment.

OpenGVL is a flexible, extensible framework with a simple, unified interface for working with different VLMs and robotics datasets. It serves as a solid foundation for researchers and developers to experiment, build, and contribute to Generative Value Learning in robotics.

### Key Features

- Modular design: swap models, datasets, and prompts with ease.
- Extensible: add new models and datasets with minimal code.
- Configuration-driven: powered by Hydra for reproducible experiments.
- Supports multiple VLMs: out-of-the-box clients for Gemini, Gemma, Kimi, and OpenAI's GPT series.

## Quick Start

Run predictions using the `predict.py` script. Configuration is managed via YAML files in `configs`.

After completing setup in Getting Started, run:

```bash
python opengvl/scripts/predict.py \
    model=gemini \
    dataset=berkeleymvp \
    data_loader=huggingface
```

Results are saved in the `outputs` directory, organized by date and time.

## Getting Started

Follow these steps to set up a local development environment.

### Prerequisites

- Python 3.11+
- [uv](https://github.com/astral-sh/uv) (recommended) or pip for package management

### Installation

1. Clone the repository:
   ```bash
   git clone https://github.com/your-username/opengvl-codebase_refactor.git
   cd opengvl-codebase_refactor
   ```

2. Create a virtual environment:
   Using `uv`:
   ```bash
   uv venv
   source .venv/bin/activate
   ```
   Or using `venv`:
   ```bash
   python -m venv .venv
   source .venv/bin/activate
   ```

3. Install dependencies:
   Using `uv`:
   ```bash
   uv pip install -e .
   ```
   Or using `pip`:
   ```bash
   pip install -e .
   ```

4. Set up environment variables:
   Create a `.env` file in the project root. You will need API keys for proprietary models and, in some cases, a Hugging Face token for models or datasets.
   ```bash
   cp .env.example .env
   ```
   Then edit `.env` with your credentials:
   ```
   OPENAI_API_KEY="your-openai-api-key"
   GOOGLE_API_KEY="your-google-api-key"
   HUGGING_FACE_HUB_TOKEN="your-hugging-face-token"
   ```
   - OpenAI API key: for models like GPT-5
   - Google API key: for models like Gemini
   - Hugging Face Hub token: for downloading models (e.g., Gemma, Kimi) and accessing private datasets

## VLM Input, Output, and Few-Shot Learning

OpenGVL benchmarks temporal understanding of VLMs through a few-shot learning pipeline. Here’s how it works:

### VLM Input

Each prediction uses two components:

1. A prompt constructed from a template (e.g., `configs/prompts/concise.yaml`) and a dataset instruction. For example:
   > Task: Pick up the blue block and place it in the red bowl. Estimate task completion (0–100%) per frame. Frames can be shuffled.

2. A set of images:
   - Evaluation episode: a sequence of shuffled frames for which the model estimates completion percentages.
   - Context episodes (optional): one or more complete episodes used for few-shot learning, with ordered frames and ground-truth percentages.

### VLM Output

The model returns a text response with estimated completion percentages for each evaluation frame. The function `extract_percentages` in `opengvl/utils/inference.py` parses the response into a list of integers.

Example response:
> Frame 1: 50%, Frame 2: 100%, Frame 3: 25%

Parsed as: `[50, 100, 25]`.

### Few-Shot Learning Pipeline

The `predict.py` script orchestrates the pipeline:

1. Data loading: loads `FewShotInput` examples containing an evaluation episode and optional context episodes.
2. Prediction: `predict_on_fewshot_input` formats the prompt and sends text and images to the selected VLM client.
3. Parsing and evaluation: the model response is parsed to extract percentages and compared to ground truth using the Value-Order Correlation (VOC) metric.
4. Saving results: predictions, raw outputs, and metrics are stored in a `.jsonl` file per experiment.

## Evaluation

Evaluation uses the Value-Order Correlation (VOC) metric implemented in `opengvl/metrics/voc.py`.

VOC measures how well predicted completion percentages for shuffled frames correlate with the true chronological order, using Spearman’s rank correlation. Higher is better.

To run a simple evaluation, use the scripts in `jobs`:

```bash
bash jobs/run_eval.sh
```

## Running with Apptainer/Singularity

For a reproducible and portable environment, use Apptainer (formerly Singularity).

### Prerequisites

- [Apptainer](https://apptainer.org/docs/user/main/installation.html) installed on your system
- NVIDIA drivers installed on the host for GPU support

### Building the Container

Build the image from the definition file in `apptainer`:

```bash
cd apptainer
sudo apptainer build opengvl.sif opengvl.def
cd ..
```

This creates `opengvl.sif`.

### Running Experiments in the Container

Run any command inside the container using `apptainer run` or `apptainer exec`. The image is configured to use `uv` with all dependencies installed.

To run the quick start prediction with GPU support (`--nv`):

```bash
apptainer run --nv opengvl.sif python opengvl/scripts/predict.py \
    model=gemini \
    dataset=berkeleymvp \
    data_loader=huggingface
```

Pass API keys and environment variables via `--env` flags or by exporting them in your shell:

```bash
export OPENAI_API_KEY="your-key"
export GOOGLE_API_KEY="your-key"
export HUGGING_FACE_HUB_TOKEN="your-token"

apptainer run --nv opengvl.sif ...
```

## Configuration

Configuration is managed with [Hydra](https://hydra.cc/). Files live in `configs`:

- `configs/model/`: model configurations (e.g., `gemini.yaml`, `gemma.yaml`, `openai.yaml`)
- `configs/dataset/`: dataset configurations
- `configs/data_loader/`: data loader configurations (e.g., `huggingface.yaml`, `local.yaml`)
- `configs/prompts/`: prompt styles
- `configs/experiments/`: complete experiment configurations

Override any parameter from the command line. For example, to change Gemini temperature:

```bash
python opengvl/scripts/predict.py model=gemini dataset=berkeleymvp data_loader=huggingface model.temperature=0.5
```

## Extending OpenGVL

### Adding a New Model

1. Create a client in `opengvl/clients/` (e.g., `my_model.py`) inheriting from `opengvl.clients.base.BaseClient`. Implement `predict`.

   ```python
   # opengvl/clients/my_model.py
   from .base import BaseClient

   class MyModelClient(BaseClient):
       def __init__(self, model_name: str, **kwargs):
           super().__init__(model_name, **kwargs)
           # Initialization logic

       def predict(self, messages, **kwargs):
           # Prediction logic
           pass
   ```

2. Add a configuration file in `configs/model/` (e.g., `my_model.yaml`).

   ```yaml
   # configs/model/my_model.yaml
   _target_: opengvl.clients.my_model.MyModelClient
   model_name: "my-model-name"
   # Other parameters
   ```

3. Run an experiment:

   ```bash
   python opengvl/scripts/predict.py model=my_model ...
   ```

### Adding a New Dataset

1. Add a configuration file in `configs/dataset/` (e.g., `my_dataset.yaml`).

   ```yaml
   # configs/dataset/my_dataset.yaml
   name: "my-dataset-name"
   split: "test"
   # Other parameters
   ```

2. Choose a data loader. Use `huggingface` or `local`, or implement your own. If the dataset is on Hugging Face, select the `huggingface` loader and specify the dataset name.

3. Run an experiment:

   ```bash
   python opengvl/scripts/predict.py dataset=my_dataset ...
   ```

## Repository Structure

```
.
├── apptainer/      # Apptainer/Singularity definition for containerization
├── configs/        # Hydra configuration files
│   ├── data_loader/
│   ├── dataset/
│   ├── model/
│   └── prompts/
├── jobs/           # Shell scripts for running experiments
├── notebooks/      # Jupyter notebooks for analysis and inference
├── opengvl/        # Main Python package
│   ├── clients/      # Clients for different VLMs (Gemini, OpenAI, etc.)
│   ├── data_loaders/ # Data loaders for different data sources
│   ├── metrics/      # Evaluation metrics (e.g., VOC)
│   ├── scripts/      # Main scripts (e.g., prediction)
│   └── utils/        # Utility functions
├── tests/          # Test suite
├── .env.example    # Example environment file
├── pyproject.toml  # Project metadata and dependencies
└── README.md
```

## Known Issues & Troubleshooting

### Common Issues

- macOS library path issue: if you encounter library loading problems on macOS, set:
  ```bash
  export DYLD_FALLBACK_LIBRARY_PATH=/opt/homebrew/lib
  ```

- CUDA memory issues: if you run into GPU OOM errors, reduce `batch_size` in the model config (e.g., `configs/model/gemini.yaml`).

- Hugging Face authentication: ensure your token is set in `.env` (`HUGGING_FACE_HUB_TOKEN`) for gated models or private datasets.

## Citation

If you use OpenGVL in your research, please cite our paper:

```bibtex
@article{opengvl2024,
  title={OpenGVL: Benchmarking Visual Temporal Progress for Data Curation},
  author={[Authors]},
  journal={arXiv preprint arXiv:2411.04549},
  year={2024}
}
```

## Acknowledgments

This work builds upon many excellent projects and ideas:

- Foundational research on Generative Value Learning ([arXiv:2411.04549](https://arxiv.org/abs/2411.04549))
- The [LeRobot](https://huggingface.co/lerobot) project for dataset infrastructure
- The [Hydra](https://hydra.cc/) framework for configuration management
- [Hugging Face](https://huggingface.co/) for dataset hosting and model access
- The broader open-source robotics community for invaluable dataset contributions

## License

This project is licensed under the MIT License. See the `LICENSE` file for details.
