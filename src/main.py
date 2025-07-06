""" Example script to produce GVL predictions.

Script to load OXE (Open X-Embodiment) robotics data and feed random images
to a task completion percentage prediction prompt.

Download dataset:
gsutil -m cp -r gs://gresearch/robotics/fmb ~/tensorflow_datasets/

Run:
python src/main.py --name fmb:0.0.1 --max_frames 4 --model gpt4o
"""

from __future__ import annotations

import argparse

from data_loader import DataLoader
from models import ModelFactory
from voc_score import parse_response, value_order_correlation


def main(
        name: str,
        num_eval_steps: int = 5,
        max_frames: int = 10, 
        num_context_frames: int = 4, 
        model: str = "gpt4o",
        num_context_episodes: int = 2,
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

    for eval_step in range(num_eval_steps):
        example = loader.load_example()

        # Save images to files
        image_paths = loader.save_images_to_files(selected_images)

        # Generate response using selected model
        print(f"\nSending to {model}...")
        try:
            prompt = f"""You are an expert roboticist tasked to predict task completion
            percentages for frames of a robot for the task of {example.instructions[0]}.
            The task completion percentages are between 0 and 100, where 100
            corresponds to full task completion. We provide several examples of
            the robot performing the task at various stages and their
            corresponding task completion percentages. Note that these frames are
            in random order, so please pay attention to the individual frames
            when reasoning about task completion percentage."""
            response = client.generate_response(
                prompt=prompt,
                image_paths=image_paths,
                task_description=task_description,
                example_indices=selected_indices[:num_context_frames],
                total_frames=total_frames,
                num_context_frames=num_context_frames,
            )

            print("\n" + "=" * 80)
            print(f"{model.upper()} RESPONSE:")
            print("=" * 80)
            print(response)
            print("=" * 80)

            print("\nGround Truth Completion Percentages:")
            for i, idx in enumerate(selected_indices[num_context_frames:]):
                completion = idx / total_frames * 100
                print(f"Frame {i+1} (step {idx}): {completion:.1f}%")

            print(f"VOC: {value_order_correlation(parse_response(response), selected_indices[num_context_frames:])}")

        except Exception as e:
            print(f"Error generating response: {e}")

        break

    print(f"\nImages saved in: {loader.image_dir}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--name", type=str, default="fmb:0.0.1", help="Dataset name")
    parser.add_argument("--max_frames", type=int, default=10, help="Maximum number of frames to select")
    parser.add_argument("--num_context_frames", type=int, default=4, help="Number of context frames to use")
    parser.add_argument(
        "--model",
        type=str,
        default="gpt4o",
        choices=["gpt4o", "internvl", "gemma", "gemini"],
        help="Model to use for inference",
    )
    args = parser.parse_args()
    main(args.name, args.max_frames, args.num_context_frames, args.model)
