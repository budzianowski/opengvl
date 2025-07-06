""" Data loader utilities """
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from lerobot.datasets.lerobot_dataset import LeRobotDataset, LeRobotDatasetMetadata


@dataclass
class Example:
    instructions: list[str]
    episode_indices: list[int]
    original_frames_indices: list[list[int]]

    task_completion_predictions: list[int]
    shuffled_frames_indices: list[list[int]]

    frames: list[list[np.ndarray]]


class DataLoader:
    def __init__(
        self, dataset_name: str, 
        num_context_episodes: int = 2, 
        num_frames: int = 10,
        camera_index: int = 0,
    ):
        """Wrapper around LeRobotDataset to load examples from the dataset.

        Args:
            dataset_name: name of the dataset to load
            num_context_episodes: number of episodes to use as context
            num_context_frames: number of frames to use as context
        """
        self.ds_meta = LeRobotDatasetMetadata(dataset_name)
        self.dataset_name = dataset_name
        self.cache_episode = np.arange(self.ds_meta.total_episodes)
        self.num_context_episodes = num_context_episodes
        self.camera_index = camera_index
        self.num_frames = num_frames
    
    def load_example(self):
        """From available episdoes choose num_context+1 episodes. Remove these indices from cache_episode"""
        episode_indices = np.random.choice(
            self.cache_episode, self.num_context_episodes + 1, replace=False)
        self.cache_episode = np.setdiff1d(self.cache_episode, episode_indices)
        dataset = LeRobotDataset(self.dataset_name, episodes=episode_indices)

        # Then we grab all the image frames from the first camera:
        camera_key = dataset.meta.camera_keys[self.camera_index]

        task_completion_predictions = []
        original_frames_indices = []
        shuffled_frames_indices = []
        all_frames = []
        instructions = []

        for idx in range(len(dataset.episode_data_index["from"])):
            from_idx = dataset.episode_data_index["from"][idx].item()
            to_idx = dataset.episode_data_index["to"][idx].item()
            frames = [dataset[index][camera_key] for index in range(from_idx, to_idx)]

            # instructions.append(dataset.episode_data["instruction"][idx])
            context_frames_indices = np.random.choice(range(len(frames)), self.num_frames, replace=False)
            completion_prediction = (context_frames_indices / len(frames) * 100).astype(int)
            selected_frames = [frames[i] for i in context_frames_indices]
            
            # shuffle chosen frames
            shuffled_indices = np.random.permutation(len(context_frames_indices))
            shuffled_frames = [selected_frames[i] for i in shuffled_indices]

            # Save the original and new indices
            original_frames_indices.append(context_frames_indices.tolist())
            shuffled_frames_indices.append(shuffled_indices.tolist())
            task_completion_predictions.append(completion_prediction.tolist())
            all_frames.append(shuffled_frames)

        breakpoint()
        return Example(
            # TODO meta.tasks
            # instructions=instructions,
            original_episode_indices=episode_indices,
            original_frames_indices=original_frames_indices,
            shuffled_frames_indices=shuffled_frames_indices,
            frames=all_frames,
            task_completion_predictions=task_completion_predictions,
        )

    def plot_example(self, dataset: LeRobotDataset):
        """Plot the example"""
        breakpoint()
        # TODO: plot the example
        pass


if __name__ == "__main__":
    loader = DataLoader(dataset_name="lerobot/fmb", num_context_episodes=2, num_frames=4)
    example = loader.load_example()
    breakpoint()
    loader.plot_example(example)

    # loader.plot_example(example)
