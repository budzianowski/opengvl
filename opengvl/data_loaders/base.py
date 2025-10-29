from abc import ABC, abstractmethod
from collections.abc import Iterable, Sequence

import numpy as np
from loguru import logger

from opengvl.utils.aliases import ImageNumpy, ImageT
from opengvl.utils.data_types import Episode
from opengvl.utils.data_types import Example as FewShotInput
from opengvl.utils.images import to_numpy


class BaseDataLoader(ABC):
    """Abstract base for building Episode/Example structures.

    Subclasses should implement ``load_fewshot_input`` and optionally ``reset``.
    This base provides utility methods to transform raw frames into an
    ``Episode`` that satisfies invariants from ``opengvl.utils.data_types``.
    """

    def __init__(
        self,
        *,
        num_frames: int = 10,
        num_context_episodes: int = 0,
        shuffle: bool = False,
        seed: int = 42,
        sampling_method: str = 'random',
    ) -> None:
        self.num_frames = int(num_frames)
        self.num_context_episodes = int(num_context_episodes)
        self.shuffle = bool(shuffle)
        self.seed = int(seed)
        self._rng = np.random.default_rng(self.seed)
        self.sampling_method = sampling_method

    @abstractmethod
    def load_fewshot_input(self, episode_index: int | None = None) -> FewShotInput:
        """Load a single FewShotInput (eval + optional context episodes)."""

    def load_fewshot_inputs(self, n: int) -> list[FewShotInput]:
        """Load ``n`` FewShotInput structures in sequence."""
        return [self.load_fewshot_input() for _ in range(int(n))]

    def reset(self) -> None:
        logger.info(f"Resetting {self.__class__.__name__} data loader with seed {self.seed}")
        self._rng = np.random.default_rng(self.seed)

    # ---------------------------- helpers ---------------------------------
    def _linear_completion(self, length: int) -> list[int]:
        if length <= 0:
            return []
        if length == 1:
            return [100]
        return [round(i / (length - 1) * 100) for i in range(length)]

    def _select_indices(self, total: int, sampling='random') -> list[int]:
        """Select up to ``num_frames`` indices from a sequence of size ``total``.

        Uses even spacing to maintain temporal coverage and determinism.
        """
        if total <= 0:
            return []
        if total <= self.num_frames:
            return list(range(total))
        # Evenly spaced selection over [1, total-1]
        # Exclude first frame (always included)
        # return np.linspace(1, total - 1, self.num_frames, dtype=int).tolist()
        if sampling == 'random':
            frames = self._rng.choice(range(1, total), self.num_frames, replace=False)
        elif sampling == 'uniform':
            frames = np.linspace(1, total - 1, self.num_frames, dtype=int)
        elif sampling == "heavy_left_tail":
            probs = np.array([1 / (i + 1) for i in range(1, total)])
            probs /= probs.sum()
            frames = self._rng.choice(range(1, total), self.num_frames, replace=False, p=probs)
        elif sampling == "heavy_right_tail":
            probs = np.array([1 / (total - i) for i in range(1, total)])
            probs /= probs.sum()
            frames = self._rng.choice(range(1, total), self.num_frames, replace=False, p=probs)
        elif sampling == 'poisson':
            lam = (self.num_frames + 1) / 2
            frames_set = set()
            while len(frames_set) < self.num_frames:
                sample = int(self._rng.poisson(lam))
                if 1 <= sample < total:
                    frames_set.add(sample)
            frames = np.array(list(frames_set))
        else:
            raise ValueError(f"Unknown sampling method: {sampling}")
        frames = np.sort(frames)
        return frames.tolist()

    def _maybe_shuffle(self, indices: Sequence[int], *, rng: np.random.Generator | None = None) -> list[int]:
        indices = list(indices)
        if not self.shuffle:
            return indices
        rng = rng or self._rng
        perm = rng.permutation(len(indices))
        return [indices[i] for i in perm]

    def _ensure_numpy(self, frames: Iterable[ImageT]) -> list[ImageNumpy]:
        np_frames: list[ImageNumpy] = []
        for f in frames:
            np_frames.append(to_numpy(f))
        return np_frames

    def _build_episode(
        self,
        *,
        frames: Sequence[ImageT],
        instruction: str,
        episode_index: int,
        sampling_method: str = 'random',
    ) -> Episode:
        """Construct an Episode from raw frames.

        - Selects up to ``num_frames`` frames (even spacing)
        - Optionally shuffles their presentation order
        - Fills both original and shuffled completion rates
        """
        # # Deterministic per-episode RNG to ensure stable shuffles across runs
        per_ep_rng = np.random.default_rng(self.seed + int(episode_index))

        if len(frames) == 0:
            raise ValueError

        # Convert and choose subset
        frames_np = self._ensure_numpy(frames)
        selected_orig = self._select_indices(len(frames_np), sampling_method)

        # Original timeline metadata (sorted ascending)
        original_indices = list(selected_orig)
        all_frames_length = len(frames_np)
        original_completion = [
            round(i / (all_frames_length - 1) * 100) for i in original_indices
        ]

        # Shuffled presentation order
        shuffled_indices = self._maybe_shuffle(original_indices, rng=per_ep_rng)
        shuffled_frames = [frames_np[i] for i in shuffled_indices]
        shuffled_completion_approx = [
            original_completion[original_indices.index(i)] for i in shuffled_indices
        ]


        starting_frame = frames_np[original_indices[0]]

        return Episode(
            instruction=str(instruction),
            starting_frame=starting_frame,
            episode_index=int(episode_index),
            original_frames_indices=original_indices,
            shuffled_frames_indices=shuffled_indices,
            shuffled_frames_approx_completion_rates=shuffled_completion_approx,
            original_frames_task_completion_rates=original_completion,
            shuffled_frames=shuffled_frames,
        )
