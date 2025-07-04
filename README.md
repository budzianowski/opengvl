# opengvl
Minimal implementation of [Generative Value Learning](https://www.arxiv.org/abs/2411.04549).

## Setup

```bash
git clone git@github.com:budzianowski/opengvl.git
cd opengvl
uv sync --dev
```

# Run open source version

```bash
uv run src/main.py --name fmb:0.0.1 --max_frames 30 --model gpt4o
```



# TODO:
1. Fix logic
2. Add reward awr for the algorithm for the lerobot as input
3. 


# Acknowledgements
- [Generative Value Learning](https://www.arxiv.org/abs/2411.04549)