""" Data loader utilities """
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from lerobot.datasets.lerobot_dataset import LeRobotDataset, LeRobotDatasetMetadata
import matplotlib.pyplot as plt
import argparse


@dataclass
class Episode:
    instruction: str
    episode_indices: list[int]
    original_frames_indices: list[list[int]]
    shuffled_frames_indices: list[list[int]]
    task_completion_predictions: list[list[int]]
    frames: list[list[np.ndarray]]

@dataclass
class Example:
    instructions: list[str]
    episode_indices: list[int]
    original_frames_indices: list[list[int]]
    shuffled_frames_indices: list[list[int]]
    task_completion_predictions: list[list[int]]
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
        """From available episodes choose num_context + 1 episodes."""
        if len(self.cache_episode) < self.num_context_episodes + 1:
            raise ValueError("Not enough episodes available in cache")

        episode_indices = np.random.choice(
            self.cache_episode, self.num_context_episodes + 1, replace=False)
        self.cache_episode = np.setdiff1d(self.cache_episode, episode_indices)
        dataset = LeRobotDataset(self.dataset_name, episodes=episode_indices)

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

            # all episode indices have the same task
            instructions.append(dataset[from_idx]["task"])

        return Example(
            instructions=instructions,
            episode_indices=episode_indices,
            original_frames_indices=original_frames_indices,
            shuffled_frames_indices=shuffled_frames_indices,
            frames=all_frames,
            task_completion_predictions=task_completion_predictions,
        )

    def plot_single_episode(self, example: Example, episode_idx: int = 0):
        """Plot a single episode from the Example structure with task completion graph
        
        Args:
            example: Example structure containing episodes data
            episode_idx: Index of episode to plot (default: 0)
        """
        if episode_idx >= len(example.frames):
            print(f"Episode index {episode_idx} out of range. Available episodes: {len(example.frames)}")
            return
        
        frames = example.frames[episode_idx]
        instruction = example.instructions[episode_idx]
        original_indices = example.original_frames_indices[episode_idx]
        shuffled_indices = example.shuffled_frames_indices[episode_idx]
        completion_preds = example.task_completion_predictions[episode_idx]
        
        # Create figure with subplots for frames and completion graph
        num_frames = len(frames)
        cols = min(4, num_frames)
        rows = (num_frames + cols - 1) // cols
        
        # Create figure with extra space for completion graph
        fig = plt.figure(figsize=(16, 4 * rows + 4))
        
        # Create gridspec for layout
        gs = fig.add_gridspec(rows + 1, cols, height_ratios=[1] * rows + [0.8], hspace=0.3)
        
        fig.suptitle(f"Episode {example.episode_indices[episode_idx]}: {instruction}", 
                    fontsize=16, fontweight='bold')
        
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
            
            ax.set_title(f"Shuffled pos: {i}\nOriginal frame: {original_frame_idx}\nCompletion: {completion_pred}%", 
                        fontsize=10)
            ax.axis('off')
        
        # Create task completion graph
        ax_completion = fig.add_subplot(gs[-1, :])
        
        # Prepare data for completion graph
        frame_positions = list(range(len(frames)))  # Shuffled positions
        original_frame_positions = [original_indices[shuffled_indices[i]] for i in range(len(frames))]
        completion_values = [completion_preds[shuffled_indices[i]] for i in range(len(frames))]
        
        # Create line plot
        ax_completion.plot(frame_positions, completion_values, 
                                 marker='o', linewidth=3, markersize=8, 
                                 color='#2E86AB', markerfacecolor='#A23B72', 
                                 markeredgecolor='white', markeredgewidth=2)
        
        # Add value labels on points
        for i, (pos, completion) in enumerate(zip(frame_positions, completion_values)):
            ax_completion.annotate(f'{completion}%\n(orig: {original_frame_positions[i]})',
                                 (pos, completion), 
                                 textcoords="offset points", 
                                 xytext=(0,15), 
                                 ha='center', va='bottom', 
                                 fontsize=9, fontweight='bold',
                                 bbox=dict(boxstyle="round,pad=0.3", facecolor='white', alpha=0.8))
        
        ax_completion.set_xlabel('Shuffled Frame Position', fontsize=12, fontweight='bold')
        ax_completion.set_ylabel('Task Completion (%)', fontsize=12, fontweight='bold')
        ax_completion.set_title('Task Completion Progress Over Frames', fontsize=14, fontweight='bold')
        ax_completion.set_ylim(0, 110)
        ax_completion.grid(True, alpha=0.3)
        ax_completion.set_xticks(frame_positions)
        ax_completion.set_xticklabels([f'Frame {i}' for i in frame_positions])
        
        plt.tight_layout()
        plt.show()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset_name", type=str, default="lerobot/fmb")
    parser.add_argument("--num_context_episodes", type=int, default=2)
    parser.add_argument("--num_frames", type=int, default=4)
    parser.add_argument("--camera_index", type=int, default=0)
    args = parser.parse_args()
    
    loader = DataLoader(
        dataset_name=args.dataset_name, 
        num_context_episodes=args.num_context_episodes, 
        num_frames=args.num_frames,
        camera_index=args.camera_index
    )
    

    example = loader.load_example()
    loader.plot_single_episode(example, episode_idx=0)
