""" Example script to produce GVL predictions.

Run:
    python src/main.py --name lerobot/fmb --max_frames 4 --num_context_episodes 2 --model internvl
    python src/main.py --batch_eval --results_file results/experiment_results.jsonl
"""

import argparse
import json
import os
from datetime import datetime

import numpy as np

import utils
from data_loader import DataLoader
from models import ModelFactory
from result_evaluator import ResultEvaluator
import torch


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
    
    def set_config(self, config: dict[str, any]):
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
                "episode_index": example.eval_episode.episode_index,
                "instruction": example.eval_episode.instruction,
                "original_frames_indices": example.eval_episode.original_frames_indices,
                "shuffled_frames_indices": example.eval_episode.shuffled_frames_indices,
                "ground_truth_completion": example.eval_episode.task_completion_predictions,
            },
            
            "context_episodes": [
                {
                    "episode_index": ctx_ep.episode_index,
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
    
    def _save_result(self, result: dict[str, any]):
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
    
    print(f"Starting batch evaluation of {results_file} with batch size {batch_size}")
    
    result_evaluator = ResultEvaluator(batch_size=batch_size)
    updated_file = result_evaluator.batch_evaluate_jsonl(results_file, output_file)
    
    print(f"Batch evaluation completed. Updated file: {updated_file}")
    return updated_file


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
    print(f'Check 2 {torch.cuda.memory_summary()}')
    client = ModelFactory.create_client(model)
    print(f'Check 3 {torch.cuda.memory_summary()}')
    for step in range(start_step, num_eval_steps):
        # try:
        example = loader.load_example()

        prompt = utils.get_prompt(example.eval_episode.instruction)
        print(f'Check 4 {torch.cuda.memory_summary()}')
        response = client.generate_response(
            prompt=prompt,
            eval_episode=example.eval_episode,
            context_episodes=example.context_episodes,
        )
        
        collector.add_result(
            step=step + 1,
            example=example,
            model_response=response,
            voc_score=None,
            extracted_percentages=None,
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
    
    # Batch evaluation mode
    parser.add_argument(
        "--batch_eval",
        action="store_true",
        help="Run batch evaluation on existing results file"
    )
    parser.add_argument(
        "--results_file",
        type=str,
        help="Path to results JSONL file for batch evaluation"
    )
    parser.add_argument(
        "--output_file",
        type=str,
        help="Path to save updated results (if not specified, overwrites input file)"
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=8,
        help="Batch size for model inference during batch evaluation",
    )
    
    # Regular evaluation mode arguments
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
        choices=["gpt4o", "internvl", "gemma", "gemini", "smolvlm", "qwen", "deepseek"],
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
    
    if args.batch_eval:
        if not args.results_file:
            parser.error("--results_file is required when using --batch_eval")
        batch_evaluate_results(args.results_file, args.output_file, args.batch_size)
    else:
        print(f'Check 1 {torch.cuda.memory_summary()}')
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
