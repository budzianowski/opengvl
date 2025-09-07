"""Config for full evaluation of gemma-3-27b-it on all datasets"""

from loguru import logger
from src.main import run_eval

DATASET_LIST = [
    "lerobot/nyu_door_opening_surprising_effectiveness",
    "lerobot/fmb",
    "lerobot/utokyo_pr2_opening_fridge",
    "lerobot/utaustin_mutex",
    "lerobot/berkeley_mvp",
    "lerobot/utokyo_xarm_pick_and_place",
    "lerobot/berkeley_autolab_ur5",
    "lerobot/utokyo_xarm_bimanual",
    "lerobot/utokyo_pr2_tabletop_manipulation",
    "lerobot/austin_sirius_dataset",
    "lerobot/toto",
    "lerobot/dlr_edan_shared_control",
    # low scores
    "lerobot/stanford_hydra_dataset",
    "lerobot/cmu_stretch",
    "lerobot/nyu_franka_play_dataset",
]

MODEL_LIST = [
    "gemma-3-27b-it",
]

CONFIG = {
    "model": "gemma-3-27b-it",
    "num_context_episodes": 2,
    "max_frames": 20,
    "num_eval_steps": 50,
    "shuffle": False,
    "camera_index": 0,
    "output_dir": "results",
    "resume": False,
}


if __name__ == "__main__":
    logger.info("Running full evaluation on all datasets")
    for dataset in DATASET_LIST:
        for model in MODEL_LIST:
            config = CONFIG.copy()
            config["dataset"] = dataset
            config["model"] = model
            config["output_dir"] = "results/"
            config["experiment_name"] = f"{model}_{dataset}"
            logger.info(
                f"Running evaluation on model: {config['model']} on dataset: {config['dataset']} with {config['num_context_episodes']} context episodes and {config['max_frames']} max frames on {config['num_eval_steps']} steps."
            )
            run_eval(
                config["dataset"],
                config["model"],
                config["num_context_episodes"],
                config["max_frames"],
                config["num_eval_steps"],
                config["camera_index"],
                config["output_dir"],
                config.get("experiment_name"),
                config.get("resume", False),
                config.get("shuffle", False),
            )
