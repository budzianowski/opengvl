"""Data loader utilities"""

from __future__ import annotations

import argparse
from dataclasses import dataclass

import matplotlib.pyplot as plt
import numpy as np
from datasets.utils.logging import disable_progress_bar
from lerobot.datasets.lerobot_dataset import LeRobotDataset, LeRobotDatasetMetadata
from loguru import logger

disable_progress_bar()


@dataclass
class Episode:
    instruction: str
    starting_frame: np.ndarray
    episode_index: int
    original_frames_indices: list[int]
    shuffled_frames_indices: list[int]
    task_completion_predictions: list[int]
    unshuffled_task_completion_predictions: list[int]
    frames: list[np.ndarray]


@dataclass
class Example:
    eval_episode: Episode
    context_episodes: list[Episode]


class DataLoader:
    def __init__(
        self,
        dataset_name: str,
        num_context_episodes: int = 2,
        num_frames: int = 10,
        camera_index: int = 0,
        seed: int = 42,
        max_episodes: int = 100,
        num_eval_pool: int = 50,
        shuffle: bool = False,
    ):
        """Wrapper around LeRobotDataset to load examples from the dataset.

        Args:
            dataset_name: name of the dataset to load
            num_context_episodes: number of episodes to use as context
            num_frames: number of frames from each episode to use
            camera_index: index of the camera to use
            seed: random seed for reproducibility
            max_episodes: maximum number of episodes to consider from the dataset
            shuffle: whether to shuffle the episode or context frames
        """
        self.ds_meta = LeRobotDatasetMetadata(dataset_name)
        self.dataset_name = dataset_name
        self.num_context_episodes = num_context_episodes
        self.camera_index = camera_index
        self.num_frames = num_frames
        self.seed = seed
        self.rng = np.random.default_rng(seed)
        self.max_episodes = min(max_episodes, self.ds_meta.total_episodes)
        self.shuffle = shuffle
        self.num_eval_pool = num_eval_pool

        self._setup_episodes()

    def _setup_episodes(self):
        """Splits episodes into fixed eval and context sets."""
        all_indices = [i for i in range(self.max_episodes)]
        # if self.shuffle:
        #     self.rng.shuffle(all_indices)

        if self.max_episodes <= self.num_eval_pool:
            raise ValueError("Not enough episodes for context and evaluation split.")

        self.eval_episode_indices = all_indices[:self.num_eval_pool]
        self.context_episode_indices_pool = all_indices[self.num_eval_pool:]
        self.next_eval_idx = 0

    def _load_episodes_from_indices(self, episode_indices: list[int]) -> list[Episode]:
        """Loads and processes episodes for the given indices."""

        if not episode_indices:
            return []
        dataset = LeRobotDataset(self.dataset_name, episodes=episode_indices)
        camera_key = dataset.meta.camera_keys[self.camera_index]

        episodes = []
        for i, episode_index in enumerate(episode_indices):
            from_idx = dataset.episode_data_index["from"][i].item()
            to_idx = dataset.episode_data_index["to"][i].item()
            frames = [dataset[index][camera_key] for index in range(from_idx, to_idx)]

            if len(frames) < self.num_frames:
                # Handle cases where an episode has fewer frames than required
                # You might want to log this or handle it differently
                context_frames_indices = np.arange(len(frames))
            else:
                context_frames_indices = self.rng.choice(range(len(frames)), self.num_frames, replace=False)
                context_frames_indices = np.sort(context_frames_indices)

            completion_prediction = (
                (context_frames_indices / (len(frames) - 1) * 100).astype(int)
                if len(frames) > 1
                else np.array([100] * len(context_frames_indices))
            )
            selected_frames = [frames[i] for i in context_frames_indices]

            if not self.shuffle:
                shuffled_indices = np.arange(len(context_frames_indices))
                shuffled_frames = [selected_frames[i] for i in shuffled_indices]
                shuffled_completion_prediction = completion_prediction[shuffled_indices]
            else:
                shuffled_indices = self.rng.permutation(len(context_frames_indices))
                shuffled_frames = [selected_frames[i] for i in shuffled_indices]
                shuffled_completion_prediction = completion_prediction[shuffled_indices]
 
            episode = Episode(
                starting_frame=frames[0],
                instruction=dataset[from_idx]["task"],
                episode_index=episode_index,
                original_frames_indices=context_frames_indices.tolist(),
                shuffled_frames_indices=shuffled_indices.tolist(),
                task_completion_predictions=shuffled_completion_prediction.tolist(),
                unshuffled_task_completion_predictions=completion_prediction.tolist(),
                frames=shuffled_frames
            )
            episodes.append(episode)
        return episodes

    def load_example(self) -> Example | None:
        """
        Loads a single example with one eval episode and a randomly sampled,
        hierarchical set of context episodes.
        """
        if self.next_eval_idx >= len(self.eval_episode_indices):
            print("All evaluation episodes have been used.")
            return None

        eval_episode_index = self.eval_episode_indices[self.next_eval_idx]
        self.next_eval_idx += 1

        eval_episode = self._load_episodes_from_indices(
            episode_indices=[eval_episode_index]
        )[0]

        # Create a dedicated, deterministic RNG for context sampling for this specific eval episode.
        # This ensures that for the same eval episode, the context is sampled identically
        # across different runs or when changing `num_context_episodes`.
        context_rng = np.random.default_rng(self.seed + eval_episode_index)

        # Shuffle the context pool in a deterministic way for this eval episode.
        shuffled_context_pool = context_rng.permutation(self.context_episode_indices_pool)

        # Select the first `num_context_episodes` from the shuffled pool.
        num_context = min(self.num_context_episodes, len(shuffled_context_pool))
        context_indices = shuffled_context_pool[:num_context]

        context_episodes = self._load_episodes_from_indices(
            episode_indices=context_indices.tolist()
        )

        return Example(eval_episode=eval_episode, context_episodes=context_episodes)

    def load_examples(self, n: int) -> list[Example]:
        """Loads n examples."""
        examples = []
        for _ in range(n):
            example = self.load_example()
            if example is None:
                break
            examples.append(example)
        return examples

    def reset(self):
        """Resets the data loader to start from the beginning of eval episodes."""
        self.next_eval_idx = 0

    def plot_single_episode(self, example: Example, episode_idx: int = 0, plot_eval: bool = True):
        """Plot a single episode from the Example structure with task completion graph

        Args:
            example: Example structure containing episodes data
            episode_idx: Index of episode to plot (default: 0, only used for context episodes)
            plot_eval: If True, plot eval episode; if False, plot context episode at episode_idx
        """
        if plot_eval:
            episode = example.eval_episode
        else:
            if episode_idx >= len(example.context_episodes):
                print(
                    f"Episode index {episode_idx} out of range. Available context episodes: {len(example.context_episodes)}"
                )
                return
            episode = example.context_episodes[episode_idx]

        frames = episode.frames
        instruction = episode.instruction
        original_indices = episode.original_frames_indices
        shuffled_indices = episode.shuffled_frames_indices
        completion_preds = episode.task_completion_predictions

        # Create figure with subplots for frames and completion graph
        num_frames = len(frames)
        cols = min(4, num_frames)
        rows = (num_frames + cols - 1) // cols

        # Create figure with extra space for completion graph
        fig = plt.figure(figsize=(16, 4 * rows + 4))

        # Create gridspec for layout
        gs = fig.add_gridspec(rows + 1, cols, height_ratios=[1] * rows + [0.8], hspace=0.3)

        episode_id = episode.episode_index
        fig.suptitle(f"Episode {episode_id}: {instruction}", fontsize=16, fontweight="bold")

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

            display_frame = display_frame.transpose(1, 2, 0)  # Change (3, 256, 256) to (256, 256, 3)
            ax.imshow(display_frame)

            # Get original frame index
            original_frame_idx = original_indices[shuffled_indices[i]]
            completion_pred = completion_preds[shuffled_indices[i]]

            ax.set_title(
                f"Shuffled pos: {i}\nOriginal frame: {original_frame_idx}\nCompletion: {completion_pred}%", fontsize=10
            )
            ax.axis("off")

        # Create task completion graph
        ax_completion = fig.add_subplot(gs[-1, :])

        # Prepare data for completion graph
        frame_positions = list(range(len(frames)))  # Shuffled positions
        original_frame_positions = [original_indices[shuffled_indices[i]] for i in range(len(frames))]
        completion_values = [completion_preds[shuffled_indices[i]] for i in range(len(frames))]

        # Create line plot
        ax_completion.plot(
            frame_positions,
            completion_values,
            marker="o",
            linewidth=3,
            markersize=8,
            color="#2E86AB",
            markerfacecolor="#A23B72",
            markeredgecolor="white",
            markeredgewidth=2,
        )

        # Add value labels on points
        for i, (pos, completion) in enumerate(zip(frame_positions, completion_values)):
            ax_completion.annotate(
                f"{completion}%\n(orig: {original_frame_positions[i]})",
                (pos, completion),
                textcoords="offset points",
                xytext=(0, 15),
                ha="center",
                va="bottom",
                fontsize=9,
                fontweight="bold",
                bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.8),
            )

        ax_completion.set_xlabel("Shuffled Frame Position", fontsize=12, fontweight="bold")
        ax_completion.set_ylabel("Task Completion (%)", fontsize=12, fontweight="bold")
        ax_completion.set_title("Task Completion Progress Over Frames", fontsize=14, fontweight="bold")
        ax_completion.set_ylim(0, 110)
        ax_completion.grid(True, alpha=0.3)
        ax_completion.set_xticks(frame_positions)
        ax_completion.set_xticklabels([f"Frame {i}" for i in frame_positions])

        plt.tight_layout()
        plt.show()

    def plot_whole_episode(self, episode_index: int):
        """Plots all frames of a single episode in order.

        Args:
            episode_index: The index of the episode to plot.
        """
        logger.info(f"Loading all frames for episode {episode_index}...")
        dataset = LeRobotDataset(self.dataset_name, episodes=[episode_index])
        camera_key = dataset.meta.camera_keys[self.camera_index]

        from_idx = dataset.episode_data_index["from"][0].item()
        to_idx = dataset.episode_data_index["to"][0].item()

        frames = [dataset[index][camera_key] for index in range(from_idx, to_idx)]
        instruction = dataset[from_idx]["task"]

        num_frames = len(frames)
        cols = min(5, num_frames)
        rows = (num_frames + cols - 1) // cols

        fig, axes = plt.subplots(rows, cols, figsize=(16, 4 * rows))
        if rows == 1 and cols == 1:
            axes = [axes]
        axes = axes.flatten()

        fig.suptitle(f"Whole Episode {episode_index}: {instruction}", fontsize=16, fontweight="bold")

        for i, frame in enumerate(frames):
            ax = axes[i]
            if isinstance(frame, np.ndarray):
                if frame.dtype == np.uint8:
                    display_frame = frame
                else:
                    display_frame = (frame * 255).astype(np.uint8)
            else:
                display_frame = np.array(frame)

            display_frame = display_frame.transpose(1, 2, 0)
            ax.imshow(display_frame)
            ax.set_title(f"Frame: {i}", fontsize=10)
            ax.axis("off")

        for i in range(num_frames, len(axes)):
            axes[i].axis("off")

        plt.tight_layout()
        plt.show()

    # plot for a given episode and for a given frmes like [121, 115, 76, 289, 290, 74, 282, 142, 296, 104, 93, 269, 236, 160, 50, 222, 149, 3, 197, 127]

    def plot_frames(self, episode_index: int, frame_indices: list[int]):
        """Plots specific frames from a given episode.

        Args:
            episode_index: The index of the episode to plot.
            frame_indices: List of frame indices to plot.
        """
        logger.info(f"Loading frames {frame_indices} for episode {episode_index}...")
        dataset = LeRobotDataset(self.dataset_name, episodes=[episode_index])
        camera_key = dataset.meta.camera_keys[self.camera_index]

        from_idx = dataset.episode_data_index["from"][0].item()
        to_idx = dataset.episode_data_index["to"][0].item()

        frames = [dataset[index][camera_key] for index in range(from_idx, to_idx)]
        instruction = dataset[from_idx]["task"]

        selected_frames = [frames[i] for i in frame_indices if i < len(frames)]

        num_frames = len(selected_frames)
        cols = min(5, num_frames)
        rows = (num_frames + cols - 1) // cols

        fig, axes = plt.subplots(rows, cols, figsize=(16, 4 * rows))
        if rows == 1 and cols == 1:
            axes = [axes]
        axes = axes.flatten()

        fig.suptitle(f"Selected Frames from Episode {episode_index}: {instruction}", fontsize=16, fontweight="bold")

        for i, frame in enumerate(selected_frames):
            ax = axes[i]
            if isinstance(frame, np.ndarray):
                if frame.dtype == np.uint8:
                    display_frame = frame
                else:
                    display_frame = (frame * 255).astype(np.uint8)
            else:
                display_frame = np.array(frame)

            display_frame = display_frame.transpose(1, 2, 0)
            ax.imshow(display_frame)
            ax.set_title(f"Frame: {frame_indices[i]}", fontsize=10)
            ax.axis("off")

        for i in range(num_frames, len(axes)):
            axes[i].axis("off")

        plt.tight_layout()
        plt.savefig(f"episode_{episode_index}_frames.png")
        plt.show()

if __name__ == "__main__":
    dl = DataLoader(
        dataset_name="lerobot/nyu_door_opening_surprising_effectiveness",
        num_context_episodes=2,
        num_frames=10,
        camera_index=0,
        seed=42,
        max_episodes=500,
        shuffle=False,
    )
    example = dl.load_example()
    dl.plot_single_episode(example=example, episode_idx=0, plot_eval=True)
    dl.plot_single_episode(example=example, episode_idx=0, plot_eval=False)
    dl.plot_single_episode(example=example, episode_idx=1, plot_eval=False)
    # dl.plot_whole_episode(episode_index=example.eval_episode.episode_index)
    # dl.plot_frames(episode_index=50, frame_indices=[121, 115, 76, 289, 290, 74, 282, 142, 296, 104, 93, 269, 236, 160, 50, 222, 149, 3, 197, 127])

