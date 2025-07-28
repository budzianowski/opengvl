import json
import sys
import time
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Literal, Optional

import tyro
from loguru import logger

from .data_loader import DataLoader, Episode
from .models import ModelFactory
from .result_evaluator import ResultEvaluator
from .utils import ResultCollector, get_prompt


@dataclass
class RunConfig:
    """Configuration for running evaluation or inference."""

    model: str
    """Name of the model to use."""
    data_source: Literal["dataset", "image-dir", "image-files"]
    """Source of the data to run on."""
    dataset_name: Optional[str] = None
    """Name of the LeRobot dataset to use."""
    image_dir: Optional[Path] = None
    """Directory containing images."""
    image_files: Optional[list[Path]] = None
    """List of image file paths."""
    prompt: Optional[str] = None
    """Task instruction/prompt (required for image inputs)."""
    episode_index: int = 0
    """Episode index for dataset loading."""
    num_context_episodes: int = 2
    """Number of context episodes for dataset loading."""
    max_frames: int = 10
    """Maximum number of frames to process."""
    num_eval_steps: int = 5
    """Number of evaluation steps to run (only for dataset)."""
    camera_index: int = 0
    """Camera index for dataset loading."""
    shuffle: bool = True
    """Shuffle images."""
    seed: int = 42
    """Random seed."""
    output_dir: Path = Path("results")
    """Output directory for results."""
    experiment_name: Optional[str] = None
    """Name of the experiment."""
    resume: bool = False
    """Whether to resume from a previous evaluation."""
    save_plot: bool = False
    """Save plot to file."""
    save_results: bool = False
    """Save results to JSON file."""
    config_path: Optional[str] = None
    """Path to a JSON config file. If provided, it will override the other arguments."""


def run(args: RunConfig):
    """Run evaluation or inference on a model."""
    if args.config_path:
        with open(args.config_path, "r") as f:
            config_data = json.load(f)
            args = tyro.from_dict(RunConfig, config_data)

    if args.data_source == "dataset":
        run_evaluation(args)
    else:
        run_inference(args)


def run_evaluation(args: RunConfig):
    """Run evaluation on a model."""
    collector = ResultCollector(
        output_dir=str(args.output_dir),
        experiment_name=args.experiment_name,
        name=args.dataset_name,
        model=args.model,
        num_context_episodes=args.num_context_episodes,
        max_frames=args.max_frames,
        num_eval_steps=args.num_eval_steps,
        camera_index=args.camera_index,
    )

    if not args.resume and collector.results_file.exists():
        logger.error(f"Results file already exists: {collector.results_file}")
        return

    start_step = 0
    if args.resume:
        logger.info("Resuming from existing results...")
        start_step = collector.load_existing_results()
        if start_step >= args.num_eval_steps:
            return collector.results_file

    result_evaluator = ResultEvaluator()
    loader = DataLoader(
        dataset_name=args.dataset_name,
        num_context_episodes=args.num_context_episodes,
        num_frames=args.max_frames,
        camera_index=args.camera_index,
        shuffle=args.shuffle,
    )

    logger.info(f"Loading examples for {args.num_eval_steps} evaluation steps...")
    examples = loader.load_examples(args.num_eval_steps)

    logger.info("Creating model client...")
    client = ModelFactory.create_client(args.model)

    for step, example in enumerate(examples):
        if step < start_step:
            continue
        logger.info(f"Processing step {step + 1}/{args.num_eval_steps}...")
        try:
            prompt = get_prompt(example.eval_episode.instruction)
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
                model_name=args.model,
            )
            time.sleep(5)

        except Exception as e:
            logger.error(f"Error processing step {step + 1}: {e}")
            error_result = {
                "step": step + 1,
                "timestamp": datetime.now().isoformat(),
                "model": args.model,
                "error": str(e),
                "status": "failed",
            }
            with open(collector.results_file, "a") as f:
                f.write(json.dumps(error_result) + "\n")
            continue

    return collector.results_file


def run_inference(args: RunConfig):
    """Run inference on a model."""
    client = ModelFactory.create_client(args.model)
    result_evaluator = ResultEvaluator(expected_length=args.max_frames)

    if args.data_source == "image-dir":
        if not args.image_dir or not args.prompt:
            logger.error("Image directory and prompt must be provided for data source 'image-dir'")
            sys.exit(1)
        logger.info(f"Loading from directory: {args.image_dir}")
        episode = load_from_directory(
            image_dir=str(args.image_dir),
            prompt=args.prompt,
            shuffle=args.shuffle,
            seed=args.seed,
            max_frames=args.max_frames,
        )
        context_episodes = []

    elif args.data_source == "image-files":
        if not args.image_files or not args.prompt:
            logger.error("Image files and prompt must be provided for data source 'image-files'")
            sys.exit(1)
        logger.info(f"Loading from image files: {args.image_files}")
        episode = load_from_images(
            image_paths=[str(p) for p in args.image_files],
            prompt=args.prompt,
            shuffle=args.shuffle,
            seed=args.seed,
            max_frames=args.max_frames,
        )
        context_episodes = []
    else:
        logger.error(f"Invalid data source: {args.data_source}")
        sys.exit(1)

    try:
        logger.info("Starting inference...")
        prompt = get_prompt(episode.instruction)
        response = client.generate_response(
            prompt=prompt,
            eval_episode=episode,
            context_episodes=context_episodes,
        )
        extracted_percentages = result_evaluator.extract_and_validate(response)["prediction"]
        results = {
            "model_response": response,
            "extracted_percentages": extracted_percentages,
            "ground_truth_percentages": episode.task_completion_predictions,
            "episode_info": {
                "instruction": episode.instruction,
                "num_frames": len(episode.frames),
                "original_indices": episode.original_frames_indices,
                "shuffled_indices": episode.shuffled_frames_indices,
            },
        }

        logger.info("Inference completed successfully!")
        logger.info(f"Extracted {len(results['extracted_percentages'])} predictions")

    except Exception as e:
        logger.error(f"Inference failed: {e}")
        sys.exit(1)

    args.output_dir.mkdir(parents=True, exist_ok=True)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    if args.save_results:
        results_path = args.output_dir / f"inference_results_{timestamp}.json"
        save_results(results, str(results_path))

    save_plot_path = None
    if args.save_plot:
        save_plot_path = args.output_dir / f"inference_plot_{timestamp}.png"

    try:
        plot_results(episode, results, str(save_plot_path) if save_plot_path else None)
    except Exception as e:
        logger.error(f"Failed to create plot: {e}")

    logger.info("Inference completed!")


def load_from_images(image_paths: list[str], prompt: str, shuffle: bool, seed: int, max_frames: int) -> Episode:
    """Load images from file paths and create an episode."""
    logger.info(f"Loading {len(image_paths)} images from file paths")
    frames = []
    for path in image_paths:
        try:
            if not os.path.exists(path):
                logger.warning(f"Image not found: {path}")
                continue
            pil_image = Image.open(path).convert("RGB")
            pil_image = pil_image.resize((224, 224))
            image_array = np.array(pil_image).transpose(2, 0, 1)
            frames.append(image_array)
        except Exception as e:
            logger.warning(f"Failed to load image {path}: {e}")
            continue
    if not frames:
        raise ValueError("No valid images could be loaded")
    if len(frames) > max_frames:
        logger.info(f"Limiting to {max_frames} frames from {len(frames)} available")
        frames = frames[:max_frames]
    original_indices = list(range(len(frames)))
    if shuffle:
        rng = np.random.default_rng(seed)
        shuffled_indices = rng.permutation(len(frames))
        shuffled_frames = [frames[i] for i in shuffled_indices]
    else:
        shuffled_indices = np.arange(len(frames))
        shuffled_frames = frames
    completion_predictions = [(i / (len(frames) - 1) * 100) if len(frames) > 1 else 100 for i in range(len(frames))]
    episode = Episode(
        instruction=prompt,
        starting_frame=frames[0],
        episode_index=0,
        original_frames_indices=original_indices,
        shuffled_frames_indices=shuffled_indices.tolist(),
        task_completion_predictions=completion_predictions,
        unshuffled_task_completion_predictions=completion_predictions,
        frames=shuffled_frames,
    )
    logger.info(f"Created episode with {len(frames)} frames")
    return episode


def load_from_directory(image_dir: str, prompt: str, shuffle: bool, seed: int, max_frames: int) -> Episode:
    """Load all images from a directory."""
    if not os.path.exists(image_dir):
        raise ValueError(f"Directory not found: {image_dir}")
    image_extensions = {".jpg", ".jpeg", ".png", ".bmp", ".tiff", ".webp"}
    image_paths = []
    for file_path in Path(image_dir).iterdir():
        if file_path.suffix.lower() in image_extensions:
            image_paths.append(str(file_path))
    if not image_paths:
        raise ValueError(f"No image files found in directory: {image_dir}")
    image_paths.sort()
    logger.info(f"Found {len(image_paths)} images in directory")
    return load_from_images(image_paths, prompt, shuffle, seed, max_frames)


def save_results(results: dict, output_path: str):
    """Save results to a JSON file."""
    logger.info(f"Saving results to {output_path}")
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    from .utils import convert_numpy_types

    results_clean = convert_numpy_types(results)
    with open(output_path, "w") as f:
        json.dump(results_clean, f, indent=2)
    logger.info("Results saved successfully")


def plot_results(episode: Episode, results: dict, save_path: Optional[str] = None):
    """Plot the inference results with task completion predictions."""
    import matplotlib.pyplot as plt

    logger.info("Plotting inference results")
    extracted_percentages = results.get("extracted_percentages", [])
    ground_truth_percentages = results.get("ground_truth_percentages", [])
    voc_score = results.get("voc_score", 0.0)
    frames = episode.frames
    num_frames = len(frames)
    cols = min(4, num_frames)
    rows = (num_frames + cols - 1) // cols
    fig = plt.figure(figsize=(16, 4 * rows + 8))
    gs = fig.add_gridspec(rows + 2, cols, height_ratios=[1] * rows + [0.8, 0.8], hspace=0.4)
    fig.suptitle(
        f"Inference Results: {episode.instruction}\nVOC Score: {voc_score:.3f}",
        fontsize=16,
        fontweight="bold",
    )
    for i, frame in enumerate(frames):
        row = i // cols
        col = i % cols
        ax = fig.add_subplot(gs[row, col])
        if isinstance(frame, np.ndarray):
            if frame.dtype == np.uint8:
                display_frame = frame
            else:
                display_frame = (frame * 255).astype(np.uint8)
        else:
            display_frame = np.array(frame)
        if len(display_frame.shape) == 3 and display_frame.shape[0] in [1, 3, 4]:
            display_frame = display_frame.transpose(1, 2, 0)
        ax.imshow(display_frame)
        pred_text = ""
        if i < len(extracted_percentages):
            pred_text = f"Pred: {extracted_percentages[i]}%"
        if i < len(ground_truth_percentages):
            gt_text = f"GT: {ground_truth_percentages[i]:.1f}%"
            pred_text = f"{pred_text}\n{gt_text}" if pred_text else gt_text
        ax.set_title(f"Frame {i}\n{pred_text}", fontsize=10)
        ax.axis("off")
    for i in range(num_frames, rows * cols):
        row = i // cols
        col = i % cols
        fig.add_subplot(gs[row, col]).axis("off")
    if extracted_percentages:
        ax_comparison = fig.add_subplot(gs[-2, :])
        frame_positions = list(range(len(extracted_percentages)))
        ax_comparison.plot(
            frame_positions,
            extracted_percentages,
            "o-",
            label="Model Predictions",
            linewidth=2,
            markersize=8,
            color="red",
        )
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
        for i, pred in enumerate(extracted_percentages):
            ax_comparison.annotate(
                f"{pred}%", (i, pred), textcoords="offset points", xytext=(0, 10), ha="center", fontsize=9
            )
    if extracted_percentages and episode.shuffled_frames_indices:
        ax_chrono = fig.add_subplot(gs[-1, :])
        try:
            sorted_indices = np.argsort(episode.shuffled_frames_indices)
            chrono_predictions = np.array(extracted_percentages)[sorted_indices]
            chrono_positions = sorted(episode.shuffled_frames_indices)
            ax_chrono.plot(chrono_positions, chrono_predictions, "o-", linewidth=2, markersize=8, color="green")
            ax_chrono.set_xlabel("Original Frame Index", fontweight="bold")
            ax_chrono.set_ylabel("Task Completion (%)", fontweight="bold")
            ax_chrono.set_title("Task Completion in Chronological Order", fontweight="bold")
            ax_chrono.set_ylim(0, 110)
            ax_chrono.grid(True, alpha=0.3)
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


def main():
    tyro.cli(run)


if __name__ == "__main__":
    main()
