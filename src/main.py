""" Example script to produce GVL predictions.

Run:
    python src/main.py --name lerobot/fmb --max_frames 4 --model internvl
"""

from __future__ import annotations

import argparse
import utils
from data_loader import DataLoader
from models import ModelFactory
from voc_score import parse_response, value_order_correlation


def run_eval(
        name: str,
        model: str = "gpt4o",
        num_context_episodes: int = 2,
        max_frames: int = 10,
        num_eval_steps: int = 5,
        camera_index: int = 0,
    ):
    """Main function to run the data loading and prompt generation."""

    loader = DataLoader(
        dataset_name=name,
        num_context_episodes=num_context_episodes,
        num_frames=max_frames,
        camera_index=camera_index,
    )

    client = ModelFactory.create_client(model)

    for _ in range(num_eval_steps):
        example = loader.load_example()

        print(f"\nSending to {model}...")
        
        # try:
        prompt = utils.get_prompt(example.eval_episode.instruction)
        response = client.generate_response(
            prompt=prompt,
            eval_episode=example.eval_episode,
            context_episodes=example.context_episodes,
        )

        print("\n" + "=" * 80)
        print(f"{model.upper()} RESPONSE:")
        print("=" * 80) 
        print(response)
        print("=" * 80)
        print("\nGround Truth Completion Percentages:")
        for i, completion in enumerate(example.eval_episode.task_completion_predictions):
            print(f"Frame {i+1} (step {i}): {completion:.1f}%")

        print(f"VOC: {value_order_correlation(parse_response(response), example.eval_episode.task_completion_predictions)}")

        # except Exception as e:
        #     print(f"Error generating response: {e}")


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
        choices=["gpt4o", "internvl", "gemma", "gemini"],
        help="Model to use for inference",
    )
    parser.add_argument(
        "--camera_index",
        type=int,
        default=0,
        help="Camera index to use",
    )
    args = parser.parse_args()
    run_eval(
        args.name, 
        args.model,
        args.num_context_episodes, 
        args.max_frames, 
        args.num_eval_steps,
        args.camera_index,
    )
