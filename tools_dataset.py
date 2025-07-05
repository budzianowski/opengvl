import math
import random
from abc import ABC, abstractmethod
from typing import Any, Dict, List, Tuple

import matplotlib.pyplot as plt
from datasets import Dataset, DatasetDict, load_dataset
from PIL import Image
from tqdm import tqdm


class BaseEpisodeLoader(ABC):
    """
    An abstract base class for loading and processing episode-based datasets.

    Attributes:
        dataset (Dataset | None): The loaded Hugging Face dataset for a specific split.
        episode_index (List[Tuple[int, int, str]] | None): An index of episodes.
            Each tuple contains (start_row, end_row, goal_text).
    """

    def __init__(self):
        """Initializes the base loader."""
        self.dataset: Dataset | None = None
        self.episode_index: List[Tuple[int, int, str]] | None = None

    def _index_episodes(self):
        """
        Scans the dataset and creates an index of all episodes.

        An episode is defined as a contiguous block of rows with the same 'goal'.
        This private method should be called by the `load` implementation.
        """
        if self.dataset is None:
            raise ValueError("Dataset must be loaded before indexing.")

        print("Indexing episodes... (This may take a moment)")
        self.episode_index = []
        if len(self.dataset) == 0:
            return

        current_goal = self.dataset[0]["goal"]
        episode_start_index = 0

        for i in tqdm(range(1, len(self.dataset)), desc="Scanning Episodes"):
            if self.dataset[i]["goal"] != current_goal:
                self.episode_index.append((episode_start_index, i, current_goal))
                current_goal = self.dataset[i]["goal"]
                episode_start_index = i

        self.episode_index.append(
            (episode_start_index, len(self.dataset), current_goal)
        )
        print(f"Indexing complete. Found {len(self.episode_index)} episodes.")

    @abstractmethod
    def load(self, dataset_path: str, **kwargs) -> Dataset:
        """
        Loads the dataset and triggers episode indexing. Must be implemented
        by a subclass.
        """
        pass

    @property
    def num_episodes(self) -> int:
        """Returns the total number of indexed episodes."""
        return len(self.episode_index) if self.episode_index else 0

    def get_episode(self, episode_id: int) -> Dataset:
        """
        Fetches a full episode's data as a sliced Dataset object.

        Args:
            episode_id: The index of the episode (from 0 to num_episodes - 1).

        Returns:
            A `Dataset` object containing only the rows for that episode.
        """
        if not self.episode_index or not (0 <= episode_id < self.num_episodes):
            raise IndexError("Episode ID is out of range.")

        start, end, _ = self.episode_index[episode_id]
        return self.dataset.select(range(start, end))

    def extract_episode_images(self, episode_id: int) -> List[Image.Image]:
        """
        Extracts all 'img' frames from a specific episode.

        Args:
            episode_id: The index of the episode.

        Returns:
            A list of PIL Image objects for the episode.
        """
        episode_data = self.get_episode(episode_id)
        return [frame["img"] for frame in episode_data]

    def select_random_frame_from_random_episode(self) -> Dict[str, Any]:
        """
        Selects a single random frame from a randomly chosen episode.

        Returns:
            A dictionary containing the data for the randomly selected frame.
        """
        if not self.episode_index:
            raise ValueError("No episodes indexed. Load data first.")

        random_episode_id = random.randint(0, self.num_episodes - 1)

        start_row, end_row, _ = self.episode_index[random_episode_id]

        random_frame_index = random.randint(start_row, end_row - 1)

        return self.dataset[random_frame_index]

    def select_random_frames_from_episode(
        self, episode_id: int, num_frames: int
    ) -> Tuple[Dataset, int, int, List[int]]:
        """
        Selects random frames from an episode, always including the first frame,
        and returns the selection along with critical episode metadata and frame indices.

        Args:
            episode_id: The index of the episode to select frames from.
            num_frames: The total number of frames to return.

        Returns:
            A tuple containing:
            - (Dataset): A `Dataset` object with the selected frames.
            - (int): The total number of frames in the original episode.
            - (int): The starting row index of the episode in the full dataset.
            - (List[int]): A list of the absolute row indices of the selected frames.

        Raises:
            IndexError: If the episode_id is out of range.
            ValueError: If num_frames is 0 or larger than the episode length.
        """
        if not self.episode_index or not (0 <= episode_id < self.num_episodes):
            raise IndexError("Episode ID is out of range.")

        if num_frames == 0:
            raise ValueError("num_frames cannot be zero.")

        # Get episode metadata from the index
        start_row, end_row, _ = self.episode_index[episode_id]
        total_episode_frames = end_row - start_row
        episode_start_index = start_row

        if num_frames > total_episode_frames:
            raise ValueError(
                f"Cannot select {num_frames} frames. Episode {episode_id} "
                f"only has {total_episode_frames} frames."
            )

        # The first frame is always included
        first_frame_index = start_row
        final_indices: List[int]

        if num_frames == 1:
            final_indices = [first_frame_index]
            selected_dataset = self.dataset.select(final_indices)
            return (
                selected_dataset,
                total_episode_frames,
                episode_start_index,
                final_indices,
            )

        # Create a pool of other frame indices to sample from
        other_frame_indices = range(start_row + 1, end_row)

        # Sample the remaining frames
        randomly_selected_indices = random.sample(other_frame_indices, k=num_frames - 1)

        # Combine and sort the final indices to maintain chronological order
        final_indices = [first_frame_index] + randomly_selected_indices
        # final_indices.sort()

        selected_dataset = self.dataset.select(final_indices)

        # Return the dataset slice along with the metadata and indices
        return (
            selected_dataset,
            total_episode_frames,
            episode_start_index,
            final_indices,
        )

    def plot_episode_sequence(self, episode_id: int):
        """
        Plots the sequence of 'img' frames for a given episode in a compact grid.

        The grid will have a maximum of 5 columns, allowing for larger images
        and better readability for long episodes.
        """
        if not self.episode_index or not (0 <= episode_id < self.num_episodes):
            raise IndexError("Episode ID is out of range.")

        images = self.extract_episode_images(episode_id)
        goal = self.episode_index[episode_id][2]
        num_images = len(images)

        if num_images == 0:
            print(f"No images found for episode {episode_id}")
            return

        max_cols = 5
        ncols = min(num_images, max_cols)
        nrows = math.ceil(num_images / ncols)

        fig, axes = plt.subplots(
            nrows, ncols, figsize=(ncols * 3.5, nrows * 3.5), squeeze=False
        )

        fig.suptitle(f"Episode {episode_id}: {goal}", fontsize=16)

        axes_flat = axes.flatten()

        for i, img in enumerate(images):
            ax = axes_flat[i]
            ax.imshow(img)
            ax.axis("off")
            ax.set_title(f"Step {i}")

        for i in range(num_images, len(axes_flat)):
            axes_flat[i].axis("off")

        plt.tight_layout()
        plt.subplots_adjust(top=0.92)
        plt.show()


class BricksDatasetLoader(BaseEpisodeLoader):
    """
    A dataset loader specifically for the Google Robotics 'Bricks' dataset.

    This class handles loading the dataset from Hugging Face and leverages
    the base class to index and process its episodic structure.
    """

    def load(self, dataset_path: str, split: str = "train", **kwargs) -> Dataset:
        """
        Loads the Bricks dataset from Hugging Face, selects a split,
        and indexes the episodes.

        Args:
            dataset_path: The path/name of the dataset on Hugging Face.
            split: The dataset split to use (e.g., 'train').
            **kwargs: Additional arguments for `load_dataset`.

        Returns:
            The loaded and indexed Hugging Face Dataset object.
        """
        try:
            print(f"Loading dataset '{dataset_path}'...")
            dataset_dict = load_dataset(dataset_path, **kwargs)

            if not isinstance(dataset_dict, DatasetDict) or split not in dataset_dict:
                raise KeyError(f"Split '{split}' not found in the loaded DatasetDict.")

            self.dataset = dataset_dict[split]
            print(f"Using split '{split}' with {len(self.dataset)} rows.")

            self._index_episodes()

            return self.dataset
        except Exception as e:
            raise IOError(f"Failed to load or process dataset. Error: {e}")


if __name__ == "__main__":
    loader = BricksDatasetLoader()

    try:
        loader.load(
            "gberseth/mini-bridge-mini64pix", split="train", trust_remote_code=True  #
        )
    except Exception as e:
        print(f"Could not load the dataset: {e}")
        print(
            "Please ensure you have an internet connection and necessary permissions."
        )

    if loader.num_episodes > 0:
        print(f"\n--- Loader is ready. Found {loader.num_episodes} episodes. ---\n")

        print("--- Selecting a random frame from a random episode ---")
        random_frame = loader.select_random_frame_from_random_episode()
        print(f"  Random frame's goal: {random_frame['goal']}")
        print(f"  Random frame's action: {random_frame['action']}")

        target_episode_id = loader.num_episodes // 2
        print(f"\n--- Getting all data for episode {target_episode_id} ---")
        episode_data = loader.get_episode(target_episode_id)
        print(episode_data["img"][0])  # Display the first image in the episode
        print(f"  Episode {target_episode_id} has {len(episode_data)} steps (frames).")
        print(f"  Goal: {episode_data[0]['goal']}")

        if loader.num_episodes > 1:
            plot_episode_id = loader.num_episodes - 1
            print(f"\n--- Plotting image sequence for episode {plot_episode_id} ---")
            loader.plot_episode_sequence(episode_id=plot_episode_id)
