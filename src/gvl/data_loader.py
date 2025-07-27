"""Data loader utilities"""

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
        try:
            self.ds_meta = LeRobotDatasetMetadata(dataset_name)
        except Exception as e:
            logger.error(f"Failed to load dataset metadata for {dataset_name}: {e}")
            raise
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
        all_indices = list(range(self.max_episodes))
        if self.max_episodes <= self.num_eval_pool:
            raise ValueError("Not enough episodes for context and evaluation split.")

        self.eval_episode_indices = all_indices[: self.num_eval_pool]
        self.context_episode_indices_pool = all_indices[self.num_eval_pool :]
        self.next_eval_idx = 0

    def _load_episodes_from_indices(self, episode_indices: list[int]) -> list[Episode]:
        """Loads and processes episodes for the given indices."""
        if not episode_indices:
            return []
        try:
            dataset = LeRobotDataset(self.dataset_name, episodes=episode_indices)
            camera_key = dataset.meta.camera_keys[self.camera_index]
        except Exception as e:
            logger.error(f"Failed to load dataset {self.dataset_name} with indices {episode_indices}: {e}")
            return []

        episodes = []
        for i, episode_index in enumerate(episode_indices):
            try:
                from_idx = dataset.episode_data_index["from"][i].item()
                to_idx = dataset.episode_data_index["to"][i].item()
                frames = [dataset[index][camera_key] for index in range(from_idx, to_idx)]

                if not frames:
                    logger.warning(f"Episode {episode_index} has no frames.")
                    continue

                if len(frames) < self.num_frames:
                    context_frames_indices = np.arange(len(frames))
                else:
                    context_frames_indices = np.sort(
                        self.rng.choice(range(len(frames)), self.num_frames, replace=False)
                    )

                completion_prediction = (
                    (context_frames_indices / (len(frames) - 1) * 100).astype(int)
                    if len(frames) > 1
                    else np.array([100] * len(context_frames_indices))
                )
                selected_frames = [frames[i] for i in context_frames_indices]

                if self.shuffle:
                    shuffled_indices = self.rng.permutation(len(context_frames_indices))
                else:
                    shuffled_indices = np.arange(len(context_frames_indices))

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
                    frames=shuffled_frames,
                )
                episodes.append(episode)
            except Exception as e:
                logger.error(f"Failed to process episode {episode_index}: {e}")
                continue
        return episodes

    def load_example(self) -> Example | None:
        """
        Loads a single example with one eval episode and a randomly sampled,
        hierarchical set of context episodes.
        """
        if self.next_eval_idx >= len(self.eval_episode_indices):
            logger.info("All evaluation episodes have been used.")
            return None

        eval_episode_index = self.eval_episode_indices[self.next_eval_idx]
        self.next_eval_idx += 1

        eval_episodes = self._load_episodes_from_indices(episode_indices=[eval_episode_index])
        if not eval_episodes:
            return self.load_example()  # Try the next one

        eval_episode = eval_episodes[0]

        context_rng = np.random.default_rng(self.seed + eval_episode_index)
        shuffled_context_pool = context_rng.permutation(self.context_episode_indices_pool)
        num_context = min(self.num_context_episodes, len(shuffled_context_pool))
        context_indices = shuffled_context_pool[:num_context]

        context_episodes = self._load_episodes_from_indices(episode_indices=context_indices.tolist())

        return Example(eval_episode=eval_episode, context_episodes=context_episodes)

    def load_examples(self, n: int) -> list[Example]:
        """Loads n examples."""
        examples = []
        while len(examples) < n:
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
                logger.error(
                    f"Episode index {episode_idx} out of range. "
                    f"Available context episodes: {len(example.context_episodes)}"
                )
                return
            episode = example.context_episodes[episode_idx]

        frames = episode.frames
        instruction = episode.instruction
        original_indices = episode.original_frames_indices
        shuffled_indices = episode.shuffled_frames_indices
        completion_preds = episode.task_completion_predictions

        num_frames = len(frames)
        cols = min(4, num_frames)
        rows = (num_frames + cols - 1) // cols

        fig = plt.figure(figsize=(16, 4 * rows + 4))
        gs = fig.add_gridspec(rows + 1, cols, height_ratios=[1] * rows + [0.8], hspace=0.3)
        fig.suptitle(f"Episode {episode.episode_index}: {instruction}", fontsize=16, fontweight="bold")

        for i, frame in enumerate(frames):
            row = i // cols
            col = i % cols
            ax = fig.add_subplot(gs[row, col])

            if isinstance(frame, np.ndarray):
                display_frame = frame.transpose(1, 2, 0) if frame.shape[0] == 3 else frame
                display_frame = (display_frame * 255).astype(np.uint8) if display_frame.dtype != np.uint8 else display_frame
            else:
                display_frame = np.array(frame)

            ax.imshow(display_frame)
            original_frame_idx = original_indices[shuffled_indices[i]]
            completion_pred = completion_preds[i]
            ax.set_title(
                f"Shuffled pos: {i}\nOriginal frame: {original_frame_idx}\nCompletion: {completion_pred}%", fontsize=10
            )
            ax.axis("off")

        ax_completion = fig.add_subplot(gs[-1, :])
        frame_positions = list(range(num_frames))
        completion_values = completion_preds
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
        for i, (pos, completion) in enumerate(zip(frame_positions, completion_values)):
            original_frame_pos = original_indices[shuffled_indices[i]]
            ax_completion.annotate(
                f"{completion}%\n(orig: {original_frame_pos})",
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
