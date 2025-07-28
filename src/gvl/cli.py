import json
import sys
import time
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Literal, Optional

import tyro
from loguru import logger

from .data_loader import DataLoader, Episode, Example
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
    from .utils import InferenceRunner

    inference_runner = InferenceRunner(
        model_name=args.model,
        num_context_episodes=args.num_context_episodes,
        max_frames=args.max_frames,
    )

    try:
        if args.data_source == "image-dir":
            if not args.image_dir or not args.prompt:
                logger.error("Image directory and prompt must be provided for data source 'image-dir'")
                sys.exit(1)
            logger.info(f"Loading from directory: {args.image_dir}")
            episode = inference_runner.load_from_directory(
                image_dir=str(args.image_dir),
                prompt=args.prompt,
                shuffle=args.shuffle,
                seed=args.seed,
            )
            context_episodes = []

        elif args.data_source == "image-files":
            if not args.image_files or not args.prompt:
                logger.error("Image files and prompt must be provided for data source 'image-files'")
                sys.exit(1)
            logger.info(f"Loading from image files: {args.image_files}")
            episode = inference_runner.load_from_images(
                image_paths=[str(p) for p in args.image_files],
                prompt=args.prompt,
                shuffle=args.shuffle,
                seed=args.seed,
            )
            context_episodes = []

    except Exception as e:
        logger.error(f"Failed to load data: {e}")
        sys.exit(1)

    try:
        logger.info("Starting inference...")
        results = inference_runner.run_inference(episode, context_episodes)

        if "error" in results:
            logger.error(f"Inference failed: {results['error']}")
        else:
            logger.info("Inference completed successfully!")
            logger.info(f"VOC Score: {results['voc_score']:.3f}")
            logger.info(f"Extracted {len(results['extracted_percentages'])} predictions")

    except Exception as e:
        logger.error(f"Inference failed: {e}")
        sys.exit(1)

    args.output_dir.mkdir(parents=True, exist_ok=True)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    if args.save_results:
        results_path = args.output_dir / f"inference_results_{timestamp}.json"
        inference_runner.save_results(results, str(results_path))

    save_plot_path = None
    if args.save_plot:
        save_plot_path = args.output_dir / f"inference_plot_{timestamp}.png"

    try:
        inference_runner.plot_results(episode, results, str(save_plot_path) if save_plot_path else None)
    except Exception as e:
        logger.error(f"Failed to create plot: {e}")

    logger.info("Inference completed!")


def main():
    tyro.cli(run)


if __name__ == "__main__":
    main()
