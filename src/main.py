""" Example script to produce GVL predictions.

Script to load OXE (Open X-Embodiment) robotics data and feed random images
to a task completion percentage prediction prompt.

Download dataset:
gsutil -m cp -r gs://gresearch/robotics/fmb ~/tensorflow_datasets/

Run:
python main.py --name fmb:0.0.1 --max_frames 20 --model gpt4o
"""
from __future__ import annotations

import argparse

from data_loader import OXEDataLoader
from models import ModelFactory


def main(name: str, max_frames: int = 4, model: str = "gpt4o"):
    """Main function to run the OXE data loading and prompt generation."""

    # Initialize the data loader
    print("Initializing OXE data loader...")
    loader = OXEDataLoader(dataset_name=name)

    # Create model client
    print(f"Initializing {model} client...")
    try:
        client = ModelFactory.create_client(model)
    except Exception as e:
        print(f"Error creating {model} client: {e}")
        return

    # Load dataset
    print("Loading dataset...")
    dataset = loader.load_dataset()

    print("Processing episode...")
    for episode in dataset.take(1):
        images, task_description = loader.extract_episode_images(episode)

        if not images:
            print("No images found in episode")
            continue

        print(f"Found {len(images)} images in episode")
        print(f"Task: {task_description}")

        # Select random frames (including initial frame)
        n_frames = min(max_frames, len(images))
        selected_images, selected_indices, total_frames = loader.select_random_frames(images, n_frames)

        print(f"Selected {len(selected_images)} frames at indices: {selected_indices}")

        # Save images to files
        image_paths = loader.save_images_to_files(selected_images)

        # Generate response using selected model
        print(f"\nSending to {model}...")
        try:
            prompt = f"""You are an expert roboticist tasked to predict task completion
            percentages for frames of a robot for the task of {task_description}.
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
                example_indices=selected_indices[:4],
                selected_indices=selected_indices[4:],
                total_frames=total_frames,
            )

            print("\n" + "=" * 80)
            print(f"{model.upper()} RESPONSE:")
            print("=" * 80)
            print(response)
            print("=" * 80)

            # Print ground truth for comparison
            print("\nGround Truth Completion Percentages:")
            for i, idx in enumerate(selected_indices):
                completion = idx / total_frames * 100
                print(f"Frame {i+1} (step {idx}): {completion:.1f}%")

        except Exception as e:
            print(f"Error generating response: {e}")

        break

    print(f"\nImages saved in: {loader.image_dir}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--name", type=str, default="fmb:0.0.1", help="Dataset name")
    parser.add_argument("--max_frames", type=int, default=4, help="Maximum number of frames to select")
    parser.add_argument(
        "--model",
        type=str,
        default="gpt4o",
        choices=["gpt4o", "internvl", "gemma", "gemini"],
        help="Model to use for inference",
    )
    args = parser.parse_args()
    main(args.name, args.max_frames, args.model)
