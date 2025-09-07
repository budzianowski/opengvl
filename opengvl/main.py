"""Example script to produce GVL predictions.

Run:
    python src/main.py --config-path configs/gemma.json
"""

import argparse
import json
import os
import time
from datetime import datetime

import numpy as np
import utils
from loguru import logger
from models import ModelFactory
from result_evaluator import ResultEvaluator

from opengvl.data_loader import DataLoader


def convert_numpy_types(obj):
    """Convert numpy types to native Python types for JSON serialization"""
    if isinstance(obj, np.integer):
        return int(obj)
    elif isinstance(obj, np.floating):
        return float(obj)
    elif isinstance(obj, np.ndarray):
        return obj.tolist()
    elif isinstance(obj, dict):
        return {key: convert_numpy_types(value) for key, value in obj.items()}
    elif isinstance(obj, list):
        return [convert_numpy_types(item) for item in obj]
    return obj


class ResultCollector:
    def __init__(
        self,
        output_dir: str = "results",
        experiment_name: str = None,
        name: str = None,
        model: str = None,
        num_context_episodes: int = None,
        max_frames: int = None,
        num_eval_steps: int = None,
        camera_index: int = None,
    ):
        self.output_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)

        if experiment_name is None:
            self.experiment_name = f"{model}_{name}"
        else:
            self.experiment_name = experiment_name

        logger.info(f"Saving results to {output_dir}/{self.experiment_name}_results.jsonl")
        self.results_file = os.path.join(output_dir, f"{self.experiment_name}_results.jsonl")
        self.summary_file = os.path.join(output_dir, f"{self.experiment_name}_summary.json")
        self.results = []
        self.experiment_config = {}

        self.config = {
            "dataset_name": name,
            "model": model,
            "num_context_episodes": num_context_episodes,
            "max_frames": max_frames,
            "num_eval_steps": num_eval_steps,
            "camera_index": camera_index,
        }
        self.set_config(self.config)

    def set_config(self, config: dict[str, any]):
        """Store experiment configuration"""
        self.experiment_config = config
        self._save_summary()

    def add_result(
        self, step: int, example, model_response: str, voc_score: float, extracted_percentages: list, model_name: str
    ):
        """Add a single evaluation result"""
        result = {
            "step": step,
            "timestamp": datetime.now().isoformat(),
            "model": model_name,
            "eval_episode": {
                "episode_index": example.eval_episode.episode_index,
                "instruction": example.eval_episode.instruction,
                "original_frames_indices": example.eval_episode.original_frames_indices,
                "shuffled_frames_indices": example.eval_episode.shuffled_frames_indices,
                "ground_truth_completion": example.eval_episode.task_completion_predictions,
                "unshuffled_task_completion_predictions": example.eval_episode.unshuffled_task_completion_predictions,
            },
            "context_episodes": [
                {
                    "episode_index": ctx_ep.episode_index,
                    "instruction": ctx_ep.instruction,
                    "original_frames_indices": ctx_ep.original_frames_indices,
                    "shuffled_frames_indices": ctx_ep.shuffled_frames_indices,
                    "ground_truth_completion": ctx_ep.task_completion_predictions,
                    "unshuffled_task_completion_predictions": ctx_ep.unshuffled_task_completion_predictions,
                }
                for ctx_ep in example.context_episodes
            ],
            "model_response": model_response,
            "extracted_percentages": extracted_percentages,
            "ground_truth_percentages": example.eval_episode.task_completion_predictions,
            "voc_score": voc_score,
            "frame_mapping": {
                "shuffled_to_original": {
                    shuffled_idx: example.eval_episode.original_frames_indices[
                        example.eval_episode.shuffled_frames_indices[shuffled_idx]
                    ]
                    for shuffled_idx in range(len(example.eval_episode.shuffled_frames_indices))
                },
                "original_to_completion": {
                    orig_idx: completion
                    for orig_idx, completion in zip(
                        example.eval_episode.original_frames_indices, example.eval_episode.task_completion_predictions
                    )
                },
            },
        }

        self.results.append(result)
        self._save_result(result)
        self._update_summary()

    def _save_result(self, result: dict[str, any]):
        """Append single result to JSONL file"""
        with open(self.results_file, "a") as f:
            serializable_result = convert_numpy_types(result)
            f.write(json.dumps(serializable_result) + "\n")

    def _save_summary(self):
        """Save experiment summary and config"""
        summary = {
            "experiment_name": self.experiment_name,
            "config": self.experiment_config,
            "results_file": self.results_file,
            "total_steps": len(self.results),
            "created_at": datetime.now().isoformat(),
        }

        if self.results:
            voc_scores = [r["voc_score"] for r in self.results if r["voc_score"] is not None]
            if voc_scores:
                summary["statistics"] = {
                    "mean_voc_score": sum(voc_scores) / len(voc_scores),
                    "min_voc_score": min(voc_scores),
                    "max_voc_score": max(voc_scores),
                    "num_valid_scores": len(voc_scores),
                    "num_total_results": len(self.results),
                }

        with open(self.summary_file, "w") as f:
            json.dump(summary, f, indent=2)

    def _update_summary(self):
        """Update summary with latest statistics"""
        self._save_summary()

    def load_existing_results(self):
        """Load existing results from JSONL file if it exists"""
        logger.info(f"Loading existing results from {self.results_file}...")
        if os.path.exists(self.results_file):
            with open(self.results_file) as f:
                for line in f:
                    if line.strip():
                        self.results.append(json.loads(line))
            logger.info(f"Loaded {len(self.results)} existing results from {self.results_file}")
        return len(self.results)


def batch_evaluate_results(results_file: str, output_file: str = None, batch_size: int = 8):
    """
    Run batch evaluation on an existing results file.

    Args:
        results_file: Path to the JSONL results file
        output_file: Path to save updated results (if None, overwrites input file)
        batch_size: Batch size for model inference
    """
    if not os.path.exists(results_file):
        raise FileNotFoundError(f"Results file not found: {results_file}")

    logger.info(f"Starting batch evaluation of {results_file} with batch size {batch_size}")

    # result_evaluator = ResultEvaluator(batch_size=batch_size)
    # updated_file = result_evaluator.batch_evaluate_jsonl(results_file, output_file)

    logger.info(f"Batch evaluation completed. Updated file: {updated_file}")
    return updated_file


def run_eval(
    name: str,
    model: str = "google/gemma-3-4b-it",
    num_context_episodes: int = 2,
    max_frames: int = 10,
    num_eval_steps: int = 5,
    camera_index: int = 0,
    output_dir: str = "results",
    experiment_name: str = None,
    resume: bool = False,
    shuffle: bool = False,
):
    """Main function to run the data loading and prompt generation."""

    collector = ResultCollector(
        output_dir=output_dir,
        experiment_name=experiment_name,
        name=name,
        model=model,
        num_context_episodes=num_context_episodes,
        max_frames=max_frames,
        num_eval_steps=num_eval_steps,
        camera_index=camera_index,
    )

    if not resume and os.path.exists(collector.results_file):
        logger.error(f"Results file already exists: {collector.results_file}")
        return

    start_step = 0
    if resume:
        logger.info("Resuming from existing results...")
        start_step = collector.load_existing_results()
        if start_step >= num_eval_steps:
            return collector.results_file

    result_evaluator = ResultEvaluator()
    loader = DataLoader(
        dataset_name=name,
        num_context_episodes=num_context_episodes,
        num_frames=max_frames,
        camera_index=camera_index,
        shuffle=shuffle,
    )

    logger.info(f"Loading examples for {num_eval_steps} evaluation steps...")
    examples = loader.load_examples(num_eval_steps)

    logger.info("Creating model client...")
    client = ModelFactory.create_client(model)

    for step, example in enumerate(examples):
        logger.info(f"Processing step {step + 1}/{num_eval_steps}...")
        try:

            prompt = utils.get_prompt(example.eval_episode.instruction)
            logger.info("Waiting for model response ...")
            generating_start = datetime.now()
            response = client.generate_response(
                prompt=prompt,
                eval_episode=example.eval_episode,
                context_episodes=example.context_episodes,
            )
            generating_duration = datetime.now() - generating_start
            logger.info(f"Model response received in {generating_duration.total_seconds()} seconds.")

            extracted_percentages = result_evaluator.extract_and_validate(response)["prediction"]
            logger.info(f"Extracted percentages: {extracted_percentages}")
            collector.add_result(
                step=step + 1,
                example=example,
                model_response=response,
                voc_score=None,
                extracted_percentages=extracted_percentages,
                model_name=model,
            )
            time.sleep(15)

        except Exception as e:
            logger.error(f"Error processing step {step + 1}: {e}")
            error_result = {
                "step": step + 1,
                "timestamp": datetime.now().isoformat(),
                "model": model,
                "error": str(e),
                "status": "failed",
            }
            with open(collector.results_file, "a") as f:
                f.write(json.dumps(error_result) + "\n")
            continue

    return collector.results_file


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("--config-path", type=str, help="Path to the config file")
    args = parser.parse_args()
    # print current path
    logger.info(f"Current working directory: {os.getcwd()}")

    try:
        with open(args.config_path) as f:
            config = json.load(f)
    except Exception as e:
        logger.error(f"Failed to load config file {args.config_path}: {e}")
        exit(1)
    logger.info(
        f"Running evaluation on model: {config['model']} on dataset: {config['dataset']} with {config['num_context_episodes']} context episodes and {config['max_frames']} max frames on {config['num_eval_steps']} steps."
    )

    if config.get("batch_eval"):
        logger.info(f"Batch evaluation mode enabled. Using results file: {config['results_file']}")
        if not config.get("results_file"):
            parser.error("--results_file is required when using --batch_eval")
        batch_evaluate_results(config["results_file"], config["output_file"], config["batch_size"])
    else:
        logger.info("Running regular evaluation...")
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
