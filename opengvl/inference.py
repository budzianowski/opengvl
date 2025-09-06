#!/usr/bin/env python3
"""
Inference script for GVL model predictions.

This script can:
1. Load data from LeRobot datasets or image paths
2. Generate task completion predictions using trained models
3. Calculate VOC scores for evaluation
4. Plot results with task completion graphs

Usage:
    # Using LeRobot dataset
    python src/inference.py --dataset-name lerobot/nyu_door_opening_surprising_effectiveness \
        --episode-index 1 --model gemma-3-27b-it --num-context-episodes 1 --max-frames 15

    # Using image directory
    python src/inference.py --image-dir /path/to/images --prompt "open the door" --model gemini-2.5-flash-lite-preview-06-17

    # Using specific image files
    python src/inference.py --image-files img1.jpg img2.jpg img3.jpg --prompt "pick up the cup" --model gemini-2.5-flash-lite-preview-06-17

    python src/inference.py --dataset-name dopaul/1500_chess_moves \
        --episode-index 53 --model gemini-2.5-flash-lite-preview-06-17 --num-context-episodes 1 --camera-index 0 --max-frames 30 --no-shuffle

    python src/inference.py --dataset-name Mahimana/excavator_toy_v3_dig_dump_v3_51 \
        --episode-index 0 --model gemini-2.5-flash-lite-preview-06-17 --num-context-episodes 1 --max-frames 30 --camera-index 5 --no-shuffle

    python src/inference.py --dataset-name willx0909/pickplace_joint \
        --episode-index 0 --model gemini-2.5-flash-lite-preview-06-17 --num-context-episodes 1 --max-frames 15 --camera-index 0 --no-shuffle --num-eval-pool 10
    
    python src/inference.py --dataset-name Rorschach4153/so101_60_new \
        --episode-index 93 --model gemini-2.5-flash-lite-preview-06-17 --num-context-episodes 1 --max-frames 15 --camera-index 0 --no-shuffle --num-eval-pool 10

    python src/inference.py --dataset-name  cking616/aloha_flod_shirts  \
        --episode-index 0 --model gemini-2.5-flash-lite-preview-06-17 --num-context-episodes 2 --max-frames 15 --camera-index 0 --no-shuffle --num-eval-pool 10

        """

import argparse
import json
import os
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from data_loader import DataLoader, Episode, Example
from loguru import logger
from PIL import Image
from result_evaluator import ResultEvaluator
from voc_score import VOCScorer

from opengvl.utils.modeling import ModelFactory


class InferenceRunner:
    """Main class for running inference and evaluation."""

    def __init__(self, model_name: str, num_context_episodes: int = 2, max_frames: int = 20, episode_index: int = None):
        """Initialize the inference runner.

        Args:
            model_name: Name of the model to use for inference
            num_context_episodes: Number of context episodes for dataset loading
            max_frames: Maximum number of frames to process
        """
        self.model_name = model_name
        self.num_context_episodes = num_context_episodes
        self.max_frames = max_frames
        self.episode_index = episode_index
        self.model_client = ModelFactory.create_client(model_name)
        self.result_evaluator = ResultEvaluator(expected_length=max_frames)
        self.voc_scorer = VOCScorer()

    def load_from_dataset(
        self,
        dataset_name: str,
        episode_index: int,
        camera_index: int = 0,
        seed: int = 42,
        shuffle: bool = False,
        num_eval_pool: int = 10,
    ) -> Example:
        """Load an episode from a LeRobot dataset.

        Args:
            dataset_name: Name of the LeRobot dataset
            episode_index: Index of the episode to load
            camera_index: Index of the camera to use
            seed: Random seed for reproducibility
            shuffle: Whether to shuffle frames (overrides instance default if provided)

        Returns:
            Example containing the episode data
        """
        logger.info(f"Loading episode {episode_index} from dataset {dataset_name} ")

        # Create data loader
        data_loader = DataLoader(
            dataset_name=dataset_name,
            num_context_episodes=self.num_context_episodes,
            num_frames=self.max_frames,
            camera_index=camera_index,
            seed=seed,
            num_eval_pool=10,
            max_episodes=max(100, episode_index + 10),
            shuffle=shuffle,
            episode_index=episode_index,
        )

        return data_loader.load_example(episode_index)

    def load_from_images(self, image_paths: list[str], prompt: str, shuffle: bool = False, seed: int = 42) -> Episode:
        """Load images from file paths and create an episode.

        Args:
            image_paths: List of paths to image files
            prompt: Task instruction/prompt
            shuffle: Whether to shuffle the images
            seed: Random seed for shuffling

        Returns:
            Episode object containing the image data
        """
        logger.info(f"Loading {len(image_paths)} images from file paths")

        # Load and validate images
        frames = []
        valid_paths = []

        for path in image_paths:
            try:
                if not os.path.exists(path):
                    logger.warning(f"Image not found: {path}")
                    continue

                # Load image as PIL and convert to numpy array
                pil_image = Image.open(path).convert("RGB")
                # Resize to standard size (224x224 is common for vision models)
                pil_image = pil_image.resize((224, 224))
                # Convert to numpy array in (C, H, W) format
                image_array = np.array(pil_image).transpose(2, 0, 1)
                frames.append(image_array)
                valid_paths.append(path)

            except Exception as e:
                logger.warning(f"Failed to load image {path}: {e}")
                continue

        if not frames:
            raise ValueError("No valid images could be loaded")

        # Limit to max_frames
        if len(frames) > self.max_frames:
            logger.info(f"Limiting to {self.max_frames} frames from {len(frames)} available")
            frames = frames[: self.max_frames]
            valid_paths = valid_paths[: self.max_frames]

        # Create frame indices
        original_indices = list(range(len(frames)))

        # Shuffle if requested
        if shuffle:
            rng = np.random.default_rng(seed)
            shuffled_indices = rng.permutation(len(frames))
            shuffled_frames = [frames[i] for i in shuffled_indices]
        else:
            shuffled_indices = np.arange(len(frames))
            shuffled_frames = frames

        # Calculate task completion predictions (assume linear progression)
        completion_predictions = [(i / (len(frames) - 1) * 100) if len(frames) > 1 else 100 for i in range(len(frames))]

        # Create episode
        episode = Episode(
            instruction=prompt,
            starting_frame=frames[0],
            episode_index=0,  # Use 0 for custom episodes
            original_frames_indices=original_indices,
            shuffled_frames_indices=shuffled_indices.tolist(),
            task_completion_predictions=completion_predictions,
            unshuffled_task_completion_predictions=completion_predictions,
            frames=shuffled_frames,
        )

        logger.info(f"Created episode with {len(frames)} frames")
        return episode

    def load_from_directory(self, image_dir: str, prompt: str, shuffle: bool = False, seed: int = 42) -> Episode:
        """Load all images from a directory.

        Args:
            image_dir: Path to directory containing images
            prompt: Task instruction/prompt
            shuffle: Whether to shuffle the images
            seed: Random seed for shuffling

        Returns:
            Episode object containing the image data
        """
        if not os.path.exists(image_dir):
            raise ValueError(f"Directory not found: {image_dir}")

        # Find all image files
        image_extensions = {".jpg", ".jpeg", ".png", ".bmp", ".tiff", ".webp"}
        image_paths = []

        for file_path in Path(image_dir).iterdir():
            if file_path.suffix.lower() in image_extensions:
                image_paths.append(str(file_path))

        if not image_paths:
            raise ValueError(f"No image files found in directory: {image_dir}")

        # Sort paths for consistent ordering
        image_paths.sort()

        logger.info(f"Found {len(image_paths)} images in directory")
        return self.load_from_images(image_paths, prompt, shuffle, seed)

    def run_inference(self, episode: Episode, context_episodes: list[Episode] | None = None) -> dict:
        """Run inference on an episode and return results.

        Args:
            episode: Episode to run inference on
            context_episodes: Optional context episodes for few-shot learning

        Returns:
            Dictionary containing predictions, VOC score, and other metrics
        """
        logger.info(f"Running inference on episode: {episode.instruction}")

        # Use empty context if none provided
        if context_episodes is None:
            context_episodes = []

        # Create a simple prompt for the model
        prompt = f"""You are an expert at analyzing robot task completion. 
        Given images of a robot performing the task "{episode.instruction}", 
        estimate the task completion percentage for each frame.
        
        The frames may be presented in random order. For each frame, provide:
        Frame X: Task Completion: Y%
        
        Be precise and consider the progression of the task."""

        try:
            # Generate model response
            model_response = self.model_client.generate_response(
                prompt=prompt, eval_episode=episode, context_episodes=context_episodes
            )

            logger.info("Model response generated successfully")

            # Extract percentages using the result evaluator
            extraction_result = self.result_evaluator.extract_and_validate(model_response)
            extracted_percentages = extraction_result.get("prediction", [])

            logger.info(f"Extracted {len(extracted_percentages)} percentages: {extracted_percentages}")

            # Calculate VOC score if we have valid predictions
            voc_score = 0.0
            if extracted_percentages and len(extracted_percentages) == len(episode.frames):
                # Create a temporary record for VOC calculation
                temp_record = {
                    "extracted_percentages": extracted_percentages,
                    "eval_episode": {"shuffled_frames_indices": episode.shuffled_frames_indices},
                }

                # Calculate VOC score manually
                if episode.shuffled_frames_indices and len(episode.shuffled_frames_indices) == len(
                    extracted_percentages
                ):
                    try:
                        sorted_indices = np.argsort(episode.shuffled_frames_indices)
                        chrono_preds = np.array(extracted_percentages)[sorted_indices].tolist()
                        voc_score = self.voc_scorer._calculate_voc(chrono_preds)
                    except Exception as e:
                        logger.warning(f"Failed to calculate VOC score: {e}")
                        voc_score = 0.0

            return {
                "model_response": model_response,
                "extracted_percentages": extracted_percentages,
                "voc_score": voc_score,
                "ground_truth_percentages": episode.task_completion_predictions,
                "episode_info": {
                    "instruction": episode.instruction,
                    "num_frames": len(episode.frames),
                    "original_indices": episode.original_frames_indices,
                    "shuffled_indices": episode.shuffled_frames_indices,
                },
            }

        except Exception as e:
            logger.error(f"Inference failed: {e}")
            return {
                "error": str(e),
                "model_response": "",
                "extracted_percentages": [],
                "voc_score": 0.0,
                "ground_truth_percentages": episode.task_completion_predictions,
                "episode_info": {
                    "instruction": episode.instruction,
                    "num_frames": len(episode.frames),
                    "original_indices": episode.original_frames_indices,
                    "shuffled_indices": episode.shuffled_frames_indices,
                },
            }

    def plot_results(self, episode: Episode, results: dict, save_path: str | None = None):
        """Plot the inference results with task completion predictions.

        Args:
            episode: Episode that was processed
            results: Results from run_inference
            save_path: Optional path to save the plot
        """
        logger.info("Plotting inference results")

        # Extract data
        extracted_percentages = results.get("extracted_percentages", [])
        ground_truth_percentages = results.get("ground_truth_percentages", [])
        voc_score = results.get("voc_score", 0.0)

        frames = episode.frames
        num_frames = len(frames)
        # Create figure with subplots for frames and graphs
        cols = min(4, num_frames)
        rows = (num_frames + cols - 1) // cols

        fig = plt.figure(figsize=(16, 4 * rows + 8))
        gs = fig.add_gridspec(rows + 2, cols, height_ratios=[1] * rows + [0.8, 0.8], hspace=0.4)

        # Title
        fig.suptitle(
            f"Inference Results: {episode.instruction}\nVOC Score: {voc_score:.3f}", fontsize=16, fontweight="bold"
        )

        # Plot frames
        for i, frame in enumerate(frames):
            row = i // cols
            col = i % cols
            ax = fig.add_subplot(gs[row, col])

            # Convert frame to displayable format
            if isinstance(frame, np.ndarray):
                if frame.dtype == np.uint8:
                    display_frame = frame
                else:
                    display_frame = (frame * 255).astype(np.uint8)
            else:
                display_frame = np.array(frame)

            # Ensure proper shape (H, W, C)
            if len(display_frame.shape) == 3 and display_frame.shape[0] in [1, 3, 4]:
                display_frame = display_frame.transpose(1, 2, 0)

            ax.imshow(display_frame)

            # Add prediction info to title
            pred_text = ""
            if i < len(extracted_percentages):
                pred_text = f"Pred: {extracted_percentages[i]}%"
            if i < len(ground_truth_percentages):
                gt_text = f"GT: {ground_truth_percentages[i]:.1f}%"
                pred_text = f"{pred_text}\n{gt_text}" if pred_text else gt_text

            ax.set_title(f"Frame {i}\n{pred_text}", fontsize=10)
            ax.axis("off")

        # Hide unused subplots
        for i in range(num_frames, rows * cols):
            row = i // cols
            col = i % cols
            fig.add_subplot(gs[row, col]).axis("off")

        # Plot comparison graph
        if extracted_percentages:
            ax_comparison = fig.add_subplot(gs[-2, :])

            frame_positions = list(range(len(extracted_percentages)))

            # Plot predictions
            ax_comparison.plot(
                frame_positions,
                extracted_percentages,
                "o-",
                label="Model Predictions",
                linewidth=2,
                markersize=8,
                color="red",
            )

            # Plot ground truth if available
            if ground_truth_percentages and len(ground_truth_percentages) == len(extracted_percentages):
                ax_comparison.plot(
                    frame_positions,
                    ground_truth_percentages,
                    "s-",
                    label="Ground Truth",
                    linewidth=2,
                    markersize=8,
                    color="blue",
                )

            ax_comparison.set_xlabel("Frame Position (Shuffled Order)", fontweight="bold")
            ax_comparison.set_ylabel("Task Completion (%)", fontweight="bold")
            ax_comparison.set_title("Task Completion Predictions vs Ground Truth", fontweight="bold")
            ax_comparison.set_ylim(0, 110)
            ax_comparison.grid(True, alpha=0.3)
            ax_comparison.legend()

            # Add value labels
            for i, pred in enumerate(extracted_percentages):
                ax_comparison.annotate(
                    f"{pred}%", (i, pred), textcoords="offset points", xytext=(0, 10), ha="center", fontsize=9
                )

        # Plot chronological order (if we can reconstruct it)
        if extracted_percentages and episode.shuffled_frames_indices:
            ax_chrono = fig.add_subplot(gs[-1, :])

            try:
                # Reconstruct chronological order
                sorted_indices = np.argsort(episode.shuffled_frames_indices)
                chrono_predictions = np.array(extracted_percentages)[sorted_indices]
                chrono_positions = sorted(episode.shuffled_frames_indices)

                ax_chrono.plot(chrono_positions, chrono_predictions, "o-", linewidth=2, markersize=8, color="green")

                ax_chrono.set_xlabel("Original Frame Index", fontweight="bold")
                ax_chrono.set_ylabel("Task Completion (%)", fontweight="bold")
                ax_chrono.set_title("Task Completion in Chronological Order", fontweight="bold")
                ax_chrono.set_ylim(0, 110)
                ax_chrono.grid(True, alpha=0.3)

                # Add value labels
                for pos, pred in zip(chrono_positions, chrono_predictions):
                    ax_chrono.annotate(
                        f"{pred}%", (pos, pred), textcoords="offset points", xytext=(0, 10), ha="center", fontsize=9
                    )
            except Exception as e:
                logger.warning(f"Could not plot chronological order: {e}")
                ax_chrono.text(
                    0.5,
                    0.5,
                    "Could not reconstruct chronological order",
                    ha="center",
                    va="center",
                    transform=ax_chrono.transAxes,
                )

        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches="tight")
            logger.info(f"Plot saved to {save_path}")

        plt.show()

    def save_results(self, results: dict, output_path: str):
        """Save results to a JSON file.

        Args:
            results: Results dictionary to save
            output_path: Path to save the results
        """
        logger.info(f"Saving results to {output_path}")

        # Create output directory if needed
        os.makedirs(os.path.dirname(output_path), exist_ok=True)

        # Convert numpy types for JSON serialization
        def convert_numpy(obj):
            if isinstance(obj, np.integer):
                return int(obj)
            elif isinstance(obj, np.floating):
                return float(obj)
            elif isinstance(obj, np.ndarray):
                return obj.tolist()
            elif isinstance(obj, dict):
                return {key: convert_numpy(value) for key, value in obj.items()}
            elif isinstance(obj, list):
                return [convert_numpy(item) for item in obj]
            return obj

        results_clean = convert_numpy(results)

        with open(output_path, "w") as f:
            json.dump(results_clean, f, indent=2)

        logger.info(f"Results saved successfully")


def main():
    """Main function for command-line usage."""
    parser = argparse.ArgumentParser(description="Run GVL inference on episodes or images")

    # Data source arguments (mutually exclusive)
    data_group = parser.add_mutually_exclusive_group(required=True)
    data_group.add_argument(
        "--dataset-name",
        type=str,
        help="LeRobot dataset name (e.g., lerobot/nyu_door_opening_surprising_effectiveness)",
    )
    data_group.add_argument("--image-dir", type=str, help="Directory containing images")
    data_group.add_argument("--image-files", nargs="+", help="List of image file paths")

    # Episode/prompt arguments
    parser.add_argument("--episode-index", type=int, default=0, help="Episode index for dataset loading (default: 0)")
    parser.add_argument("--prompt", type=str, help="Task instruction/prompt (required for image inputs)")

    # Model arguments
    parser.add_argument("--model", type=str, default="gemini-2.5-pro", help="Model name (default: gemini-2.5-pro)")
    parser.add_argument(
        "--num-context-episodes",
        type=int,
        default=2,
        help="Number of context episodes for dataset loading (default: 2)",
    )
    parser.add_argument("--max-frames", type=int, default=30, help="Maximum number of frames to process (default: 10)")

    # Other arguments
    parser.add_argument("--camera-index", type=int, default=0, help="Camera index for dataset loading (default: 0)")
    parser.add_argument("--shuffle", action="store_true", default=True, help="Shuffle images (for image inputs)")
    parser.add_argument(
        "--no-shuffle", dest="shuffle", action="store_false", help="Do not shuffle images (for image inputs)"
    )
    parser.add_argument("--seed", type=int, default=42, help="Random seed (default: 42)")
    parser.add_argument(
        "--output-dir",
        type=str,
        default="inference_results",
        help="Output directory for results (default: inference_results)",
    )
    parser.add_argument("--save-plot", action="store_true", help="Save plot to file")
    parser.add_argument("--save-results", action="store_true", help="Save results to JSON file")
    parser.add_argument("--num-eval-pool", type=int, default=50, help="Number of evaluation episodes to load")
    args = parser.parse_args()

    if (args.image_dir or args.image_files) and not args.prompt:
        parser.error("--prompt is required when using --image-dir or --image-files")

    inference_runner = InferenceRunner(
        model_name=args.model,
        num_context_episodes=args.num_context_episodes,
        max_frames=args.max_frames,
        episode_index=args.episode_index,
    )

    if args.dataset_name:
        logger.info(f"Loading from dataset: {args.dataset_name}")
        example = inference_runner.load_from_dataset(
            dataset_name=args.dataset_name,
            episode_index=args.episode_index,
            camera_index=args.camera_index,
            seed=args.seed,
            shuffle=args.shuffle,
            num_eval_pool=args.num_eval_pool,
        )
        episode = example.eval_episode
        context_episodes = example.context_episodes

    elif args.image_dir:
        logger.info(f"Loading from directory: {args.image_dir}")
        episode = inference_runner.load_from_directory(
            image_dir=args.image_dir, prompt=args.prompt, shuffle=args.shuffle, seed=args.seed
        )
        context_episodes = []

    elif args.image_files:
        logger.info(f"Loading from image files: {args.image_files}")
        episode = inference_runner.load_from_images(
            image_paths=args.image_files, prompt=args.prompt, shuffle=args.shuffle, seed=args.seed
        )
        context_episodes = []

    logger.info("Starting inference...")
    results = inference_runner.run_inference(episode, context_episodes)

    if "error" in results:
        logger.error(f"Inference failed: {results['error']}")
    else:
        logger.info(f"Inference completed successfully!")
        logger.info(f"VOC Score: {results['voc_score']:.3f}")
        logger.info(f"Extracted {len(results['extracted_percentages'])} predictions")

    os.makedirs(args.output_dir, exist_ok=True)

    if args.save_results:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        results_path = os.path.join(args.output_dir, f"inference_results_{timestamp}.json")
        inference_runner.save_results(results, results_path)

    save_plot_path = None
    if args.save_plot:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S") if not args.save_results else timestamp
        save_plot_path = os.path.join(args.output_dir, f"inference_plot_{timestamp}.png")

    inference_runner.plot_results(episode, results, save_plot_path)
    logger.info("Inference completed!")


if __name__ == "__main__":
    from datetime import datetime

    main()
