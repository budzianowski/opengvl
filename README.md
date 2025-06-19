# openglv

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
