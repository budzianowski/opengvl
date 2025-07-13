""" Example script to produce GVL predictions.

Run:
    python src/main.py --name lerobot/fmb --max_frames 4 --model internvl
"""

from __future__ import annotations

import argparse
import json
import os
from datetime import datetime
from typing import Dict, Any
import utils
from data_loader import DataLoader
from models import ModelFactory
from voc_score import parse_response, value_order_correlation
from result_evaluator import ResultEvaluator
import numpy as np


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
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            experiment_name = f"gvl_eval_{timestamp}"
        
        self.experiment_name = experiment_name
        self.results_file = os.path.join(output_dir, f"{experiment_name}_results.jsonl")
        self.summary_file = os.path.join(output_dir, f"{experiment_name}_summary.json")
        
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
    
    def set_config(self, config: Dict[str, Any]):
        """Store experiment configuration"""
        self.experiment_config = config
        self._save_summary()
    
    def add_result(self, step: int, example, model_response: str, voc_score: float, 
                   extracted_percentages: list, model_name: str):
        """Add a single evaluation result"""
        result = {
            "step": step,
            "timestamp": datetime.now().isoformat(),
            "model": model_name,
            
            "eval_episode": {
                "episode_indices": example.eval_episode.episode_indices,
                "instruction": example.eval_episode.instruction,
                "original_frames_indices": example.eval_episode.original_frames_indices,
                "shuffled_frames_indices": example.eval_episode.shuffled_frames_indices,
                "ground_truth_completion": example.eval_episode.task_completion_predictions,
            },
            
            "context_episodes": [
                {
                    "episode_indices": ctx_ep.episode_indices,
                    "instruction": ctx_ep.instruction,
                    "original_frames_indices": ctx_ep.original_frames_indices,
                    "shuffled_frames_indices": ctx_ep.shuffled_frames_indices,
                    "ground_truth_completion": ctx_ep.task_completion_predictions,
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
                        example.eval_episode.original_frames_indices,
                        example.eval_episode.task_completion_predictions
                    )
                }
            }
        }
        
        self.results.append(result)
        self._save_result(result)
        self._update_summary()
    
    def _save_result(self, result: Dict[str, Any]):
        """Append single result to JSONL file"""
        with open(self.results_file, 'a') as f:
            serializable_result = convert_numpy_types(result)
            f.write(json.dumps(serializable_result) + '\n')
    
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
        
        with open(self.summary_file, 'w') as f:
            json.dump(summary, f, indent=2)
    
    def _update_summary(self):
        """Update summary with latest statistics"""
        self._save_summary()
    
    def load_existing_results(self):
        """Load existing results from JSONL file if it exists"""
        if os.path.exists(self.results_file):
            with open(self.results_file, 'r') as f:
                for line in f:
                    if line.strip():
                        self.results.append(json.loads(line))
            print(f"Loaded {len(self.results)} existing results from {self.results_file}")
        return len(self.results)


def run_eval(
        name: str,
        model: str = "gpt4o",
        num_context_episodes: int = 2,
        max_frames: int = 10,
        num_eval_steps: int = 5,
        camera_index: int = 0,
        output_dir: str = "results",
        experiment_name: str = None,
        resume: bool = False,
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

    start_step = 0
    if resume:
        start_step = collector.load_existing_results()
        if start_step >= num_eval_steps:
            return collector.results_file
    
    loader = DataLoader(
        dataset_name=name,
        num_context_episodes=num_context_episodes,
        num_frames=max_frames,
        camera_index=camera_index,
    )

    client = ModelFactory.create_client(model)
    result_evaluator = ResultEvaluator()

    for step in range(start_step, num_eval_steps):

        # try:
        example = loader.load_example()

        prompt = utils.get_prompt(example.eval_episode.instruction)
        response = client.generate_response(
            prompt=prompt,
            eval_episode=example.eval_episode,
            context_episodes=example.context_episodes,
        )

        extracted_percentages = result_evaluator.evaluate(response)
        voc_score_extracted = None
        if extracted_percentages and len(extracted_percentages) == len(example.eval_episode.task_completion_predictions):
            voc_score_extracted = value_order_correlation(
                extracted_percentages, 
                example.eval_episode.task_completion_predictions,
            )

        collector.add_result(
            step=step + 1,
            example=example,
            model_response=response,
            voc_score=voc_score_extracted,
            extracted_percentages=extracted_percentages,
            model_name=model
        )
        
    # except Exception as e:
    #     error_result = {
    #         "step": step + 1,
    #         "timestamp": datetime.now().isoformat(),
    #         "model": model,
    #         "error": str(e),
    #         "status": "failed"
    #     }
    #     with open(collector.results_file, 'a') as f:
    #         f.write(json.dumps(error_result) + '\n')
    #     continue

    return collector.results_file


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--name", type=str, default="lerobot/fmb", help="Dataset name")
    parser.add_argument(
        "--max_frames", 
        type=int, default=10, 
        help="Maximum number of frames to select per episode"
    )
    parser.add_argument(
        "--num_eval_steps", 
        type=int, 
        default=5, 
        help="Number of evaluation steps to run"
    )
    parser.add_argument(
        "--num_context_episodes", 
        type=int, 
        default=2, help="Number of context episodes to use"
    )
    parser.add_argument(
        "--model",
        type=str,
        default="gpt4o",
        choices=["gpt4o", "internvl", "gemma", "gemini", "smolvlm", "qwen"],
        help="Model to use for inference",
    )
    parser.add_argument(
        "--camera_index",
        type=int,
        default=0,
        help="Camera index to use",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="results",
        help="Directory to save results",
    )
    parser.add_argument(
        "--experiment_name",
        type=str,
        default=None,
        help="Name for the experiment (default: auto-generated)",
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Resume from existing results file",
    )
    args = parser.parse_args()
    
    run_eval(
        args.name, 
        args.model,
        args.num_context_episodes, 
        args.max_frames, 
        args.num_eval_steps,
        args.camera_index,
        args.output_dir,
        args.experiment_name,
        args.resume,
    )
