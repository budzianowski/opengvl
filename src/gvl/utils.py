import json
import os
from datetime import datetime
from pathlib import Path

import numpy as np
from loguru import logger


def get_prompt(instruction: str) -> str:
    """Creates a prompt for the model."""
    return f"Your task is to predict the percentage of completion for a given task: '{instruction}'."


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
        self.results_file = Path(output_dir) / f"{self.experiment_name}_results.jsonl"
        self.summary_file = Path(output_dir) / f"{self.experiment_name}_summary.json"
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
            "results_file": str(self.results_file),
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
        if self.results_file.exists():
            with open(self.results_file, "r") as f:
                for line in f:
                    if line.strip():
                        self.results.append(json.loads(line))
            logger.info(f"Loaded {len(self.results)} existing results from {self.results_file}")
        return len(self.results)
