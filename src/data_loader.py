""" Data loader for the OXE dataset. """
import os
import random
from typing import List, Tuple

import numpy as np
import tensorflow as tf
import tensorflow_datasets as tfds
from PIL import Image


class OXEDataLoader:
    """Loads and processes OXE robotics dataset for task completion analysis."""

    def __init__(self, dataset_name: str = "bridge:0.1.0"):
        """
        Initialize the OXE data loader.

        Args:
            dataset_name: Name of the OXE dataset (e.g., "bridge:0.1.0", "fractal20220817_data")
        """
        self.dataset_name = dataset_name
        self.image_dir = os.path.join(os.getcwd(), "images")
        os.makedirs(self.image_dir, exist_ok=True)

    def load_dataset(self, split: str = "train") -> tf.data.Dataset:
        """Load the OXE dataset."""
        try:
            ds = tfds.load(
                self.dataset_name,
                split=split,
                shuffle_files=False,
            )
            return ds
        except Exception as e:
            print(f"Error loading {self.dataset_name}: {e}")
            raise e

    def extract_episode_images(self, episode) -> List[np.ndarray]:
        """Extract images from a single episode."""
        try:
            images = []
            instr = episode["episode_metadata"]["episode_language_instruction"]
            instr_str = instr.numpy().decode("utf-8")
            for step in episode["steps"]:
                images.append(step["observation"]["image_side_1"])
            return images, instr_str
        except Exception as e:
            print(f"Error extracting images: {e}")
            return []

    def save_images_to_files(self, images: List[np.ndarray], prefix: str = "frame") -> List[str]:
        """Save images to files and return file paths."""
        file_paths = []
        for i, img in enumerate(images):
            # Convert tensor to numpy if needed
            if isinstance(img, tf.Tensor):
                img = img.numpy()

            # Ensure image is in correct format
            if img.dtype != np.uint8:
                img = (img * 255).astype(np.uint8) if img.max() <= 1.0 else img.astype(np.uint8)

            # Save image
            filename = f"{prefix}_{i}.png"
            filepath = os.path.join(self.image_dir, filename)

            pil_img = Image.fromarray(img)
            pil_img.save(filepath)
            file_paths.append(filepath)

        return file_paths

    def select_random_frames(self, images: List[np.ndarray], n_frames: int = 3) -> Tuple[List[np.ndarray], List[int]]:
        """Select random frames from the episode."""
        if len(images) < n_frames:
            # If not enough images, use all available
            selected_images = images
            indices = list(range(len(images)))
        else:
            # Randomly sample frames
            indices = sorted(random.sample(range(len(images)), n_frames))
            selected_images = [images[i] for i in indices]

        return selected_images, indices, len(images)
