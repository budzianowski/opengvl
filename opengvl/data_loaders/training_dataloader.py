import pandas as pd
from lerobot.datasets.lerobot_dataset import LeRobotDataset
from opengvl.utils.data_types import TrainingEpisode, TrainingFewShotInput
from datasets import load_dataset
from lerobot.datasets.push_dataset_to_hub.utils import calculate_episode_data_index


import numpy as np


class TrainingDataLoader:
    def __init__(self, dataset_name: str, n_samples: int, split: str = "train") -> None:
        self.rng = np.random.default_rng(seed=42)
        self.dataset_name = dataset_name
        self.n_samples = n_samples
        self.split = split
        self.training_data = self.load_data()


    def load_data(self):
        data = load_dataset(self.dataset_name, split=self.split).to_pandas()
        assert self.n_samples <= len(data), "n_samples exceeds available data size."
        return data.sample(n=self.n_samples)
    

    def preprocess_data(self) -> list[TrainingEpisode]:
        unique_names_of_datasets = self.training_data['dataset'].unique()

        data = []

        for dataset_name in unique_names_of_datasets:
            dataset_entries = self.training_data[self.training_data['dataset'] == dataset_name]
            print(f"Preprocessing {len(dataset_entries)} entries from dataset: {dataset_name}")
            # TODO: check if episodes are stored correctly
            episode_list = sorted(list(dataset_entries["episode_id"]))
            dataset_lerobot = LeRobotDataset(dataset_name, episodes=episode_list)
            camera_key = dataset_lerobot.meta.camera_keys[0]

            print(f"Dataset {dataset_name} has {len(episode_list)} episodes.")
            
            for iter, episode_id in enumerate(episode_list):
                row = dataset_entries[dataset_entries["episode_id"] == episode_id].iloc[0]
                episode_data_index = calculate_episode_data_index(dataset_lerobot.hf_dataset)
                from_idx = int(episode_data_index["from"][episode_id].item())
                to_idx = int(episode_data_index["to"][episode_id].item())
                all_frames = [dataset_lerobot[idx][camera_key] for idx in range(from_idx, to_idx)]
                frames = [all_frames[i] for i in row["frames"]]
                instruction = dataset_lerobot[iter]["task"]
                starting_frame = all_frames[0]
                episode_idx = row["episode_id"]
                original_frame_indices = row["frames"]
                original_completion_rates = row["task_completion"]
                descriptions = row["annotated_descriptions"]
                shuffled_indices = self.rng.permutation(len(frames))
                shuffled_frames_indices = [original_frame_indices[i] for i in shuffled_indices]
                shuffled_frames = [frames[i] for i in shuffled_indices]
                shuffled_completion_rates = [original_completion_rates[i] for i in shuffled_indices]
                shuffled_descriptions = [descriptions[i] for i in shuffled_indices]


                # print(f"Original indices: {original_frame_indices}")
                # print(f"Shuffled indices: {shuffled_frames_indices}")
                # print(f"Shuffled frames: {shuffled_frames}")
                # print(f"Shuffled completion rates: {shuffled_completion_rates}")
                # print(f"Shuffled descriptions: {shuffled_descriptions}")
                # print(f"Starting frame: {starting_frame}")
                # print(f"Episode ID: {episode_id}")
                # print(f"Instruction: {instruction}")


                training_episode = TrainingEpisode(
                    instruction=instruction,
                    starting_frame=starting_frame,
                    episode_index=episode_idx,
                    original_frames_indices=original_frame_indices,
                    original_frames_task_completion_rates=original_completion_rates,
                    shuffled_frames_indices=shuffled_frames_indices,
                    shuffled_frames=shuffled_frames,
                    shuffled_frames_approx_completion_rates=shuffled_completion_rates,
                    descriptions=descriptions,
                    shuffled_descriptions=shuffled_descriptions,
                )
                data.append(training_episode)
        return data

    def load_fewshot_input(self) -> list[TrainingFewShotInput]:
        training_episodes = self.preprocess_data()
        fewshot_inputs = [
            TrainingFewShotInput(
                eval_episode=ep,
                context_episodes=[],
            )
            for ep in training_episodes
        ]
        return fewshot_inputs
    

if __name__ == "__main__":
    data_loader = TrainingDataLoader(
        dataset_name="OpenGVL/OpenGVL-Descriptions",
        n_samples=10,
        split="train",
    )
    fewshot_inputs = data_loader.load_fewshot_input()
    print(f"Loaded {len(fewshot_inputs)} few-shot training inputs.")
