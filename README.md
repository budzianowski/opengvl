# openglv

# openxyz Project

[A short, one-sentence description of what the 'openxyz' project does.]

---

## Getting Started

This guide will walk you through setting up the project for local development.

### Prerequisites

Before you begin, ensure you have the following installed on your system:
- **Python**: Version 3.11 or higher.
- **Poetry**: The dependency manager for this project. You can find installation instructions [here](https://python-poetry.org/docs/#installation).

### Installation & Setup

Follow these steps to get your development environment running:

1.  **Clone the repository:**
    ```bash
    git clone [https://github.com/your-repo](https://github.com/your-repo)
    cd openxyz
    ```

2.  **Install project dependencies:**
    This command reads the `pyproject.toml` file and installs all necessary packages (including development tools) into a managed virtual environment.
    ```bash
    poetry install
    ```

3.  **Install the pre-commit hooks:**
    This activates the automatic code formatting checks that run before each commit. This step is crucial for maintaining code quality.
    ```bash
    pre-commit install
    ```

You are now ready to start developing!

## Usage

To run the main part of the project, use Poetry's `run` command, which executes commands within the project's virtual environment.

```bash
poetry run python [your_main_script.py]
```

To activate the virtual environment shell directly, run:
```bash
poetry shell
```

## Code Formatting & Quality

This project uses `black` and `isort` to automatically format code and sort imports. These checks are managed by `pre-commit` and run automatically when you try to commit your changes.

**How it works:**
- When you run `git commit`, the pre-commit hooks will run.
- If your code is not formatted correctly, the tools will automatically fix the files and the commit will be **stopped**.
- All you need to do is `git add` the newly formatted files and run `git commit` again. This time, the commit will succeed.

```bash
# First commit attempt might fail if formatting is needed
git commit -m "My amazing feature"
# pre-commit runs, formats files, and stops the commit...

# Just add the changes and commit again
git add .
git commit -m "My amazing feature"
# This time it will pass!
```

## Download fmb data

```bash
mkdir -p ~/tensorflow_datasets/fmb/0.0.1
gsutil -m cp -r gs://gresearch/robotics/fmb/0.0.1/ ~/tensorflow_datasets/fmb/0.0.1/
```

## Run few shot glv

```bash
python main.py --model gpt4o --max_frames 30
python main.py --model gemini --max_frames 30
python main.py --model janus --max_frames 30
python main.py --model internvl --max_frames 30
```
