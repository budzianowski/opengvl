import numpy as np
from datasets.utils.logging import disable_progress_bar
from lerobot.datasets.lerobot_dataset import LeRobotDataset, LeRobotDatasetMetadata
from loguru import logger

from opengvl.data_loaders.base import BaseDataLoader
from opengvl.utils.data_types import Episode
from opengvl.utils.data_types import Example as FewShotInput

disable_progress_bar()


class HuggingFaceDataLoader(BaseDataLoader):
    """Load episodes from LeRobot datasets hosted on Hugging Face.

    Produces a FewShotInput with one eval episode and up to ``num_context_episodes``
    sampled from the remaining pool. Frame count is controlled by ``num_frames``.
    """

    def __init__(
        self,
        *,
        dataset_name: str,
        camera_index: int = 0,
        num_frames: int = 20,
        num_context_episodes: int = 2,
        shuffle: bool = False,
        seed: int = 42,
        max_episodes: int | None = None,
    ) -> None:
        super().__init__(
            num_frames=num_frames,
            num_context_episodes=num_context_episodes,
            shuffle=shuffle,
            seed=seed,
        )
        self.dataset_name = dataset_name
        self.camera_index = int(camera_index)
        self.ds_meta = LeRobotDatasetMetadata(dataset_name)
        self.max_episodes = min(max_episodes or self.ds_meta.total_episodes, self.ds_meta.total_episodes)
        # deterministic episode order
        self._all_episodes_indices = list(range(self.max_episodes))
        self._cursor = 0

    def _load_episode_frames(self, episode_index: int) -> tuple[list, str]:
        ds = LeRobotDataset(self.dataset_name, episodes=[episode_index])
        camera_key = ds.meta.camera_keys[self.camera_index]
        from_idx = int(ds.episode_data_index["from"][0].item())
        to_idx = int(ds.episode_data_index["to"][0].item())
        frames = [ds[i][camera_key] for i in range(from_idx, to_idx)]
        instruction = ds[from_idx]["task"]
        return frames, instruction

    def _build_context(self, exclude_index: int) -> list[Episode]:
        pool = [i for i in self._all_episodes_indices if i != exclude_index]
        if not pool or self.num_context_episodes <= 0:
            return []
        # Deterministic sampling for the given eval episode
        rng = np.random.default_rng(self.seed + exclude_index)
        rng.shuffle(pool)
        chosen = pool[: self.num_context_episodes]
        ctx_eps: list[Episode] = []
        for idx in chosen:
            frames, instruction = self._load_episode_frames(idx)
            ctx_eps.append(self._build_episode(frames=frames, instruction=instruction, episode_index=idx))
        return ctx_eps

    def load_fewshot_input(self, episode_index: int | None = None) -> FewShotInput:
        if episode_index is None:
            if self._cursor >= len(self._all_episodes_indices):
                self._cursor = 0
            episode_index = self._all_episodes_indices[self._cursor]
            self._cursor += 1

        logger.info(f"Loading episode {episode_index} from {self.dataset_name}")
        frames, instruction = self._load_episode_frames(episode_index)
        eval_ep = self._build_episode(frames=frames, instruction=instruction, episode_index=episode_index)
        context = self._build_context(exclude_index=episode_index)
        return FewShotInput(eval_episode=eval_ep, context_episodes=context)

    def reset(self) -> None:
        self._cursor = 0
