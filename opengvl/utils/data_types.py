from dataclasses import dataclass

from opengvl.utils.aliases import ImageNumpy
from opengvl.utils.errors import (
    OriginalFramesLengthMismatch,
    ShuffledFramesIndicesNotSubset,
    ShuffledFramesLengthMismatch,
)


@dataclass
class Episode:
    """
    Container for a single episode (or a selected subsequence of it) used in
    evaluation/training.

    Attributes
    - instruction: Natural-language description of the task to complete.
    - starting_frame: The first observation of the (sub)episode.
    - episode_index: Index of this episode within the source dataset.
    - original_frames_indices: Sorted indices from the original episode that
        define the selected subsequence.
    - original_frames_task_completion_rates: Per-frame task completion rates for
        the frames referenced by ``original_frames_indices`` (1:1 aligned; i-th
        value corresponds to the i-th index above).
    - shuffled_frames_indices: Indices from the original episode corresponding to
        ``shuffled_frames``, ordered as they are fed to the model (shuffled order).
        Each entry should also exist in ``original_frames_indices``.
    - shuffled_frames: Frames arranged according to ``shuffled_frames_indices``.
    - shuffled_frames_approx_completion_rates: Per-shuffled-frame approximate
        completion rates (1:1 aligned with ``shuffled_frames``).

    Invariants
    - len(original_frames_indices) == len(original_frames_task_completion_rates)
    - len(shuffled_frames_indices) == len(shuffled_frames)
        == len(shuffled_frames_approx_completion_rates)
    - All values in ``shuffled_frames_indices`` refer to frames from the same
        episode namespace as ``original_frames_indices``.
    """

    instruction: str
    starting_frame: ImageNumpy
    episode_index: int
    original_frames_indices: list[int]  # subsequence of original episode indices, sorted
    shuffled_frames_indices: list[int]  # original-episode indices in model input (shuffled) order
    shuffled_frames_approx_completion_rates: list[int]  # aligned 1:1 with shuffled_frames
    original_frames_task_completion_rates: list[int]  # aligned 1:1 with original_frames_indices
    shuffled_frames: list[ImageNumpy]  # frames ordered per shuffled_frames_indices

    def __post_init__(self):
        if len(self.original_frames_indices) != len(self.original_frames_task_completion_rates):
            raise OriginalFramesLengthMismatch(
                len(self.original_frames_indices), len(self.original_frames_task_completion_rates)
            )
        if not (
            len(self.shuffled_frames_indices)
            == len(self.shuffled_frames)
            == len(self.shuffled_frames_approx_completion_rates)
        ):
            raise ShuffledFramesLengthMismatch(
                len(self.shuffled_frames_indices),
                len(self.shuffled_frames),
                len(self.shuffled_frames_approx_completion_rates),
            )
        # Optional: ensure shuffled indices are a subset of original indices
        if not set(self.shuffled_frames_indices).issubset(set(self.original_frames_indices)):
            raise ShuffledFramesIndicesNotSubset()


@dataclass
class InferredEpisode(Episode):
    """
    Extension of Episode that includes model-predicted completion rates for
    the shuffled frames.
    """

    shuffled_frames_predicted_completion_rates: list[int]  # aligned 1:1 with shuffled_frames

    def __post_init__(self):
        super().__post_init__()
        if len(self.shuffled_frames_predicted_completion_rates) != len(self.shuffled_frames):
            raise ShuffledFramesLengthMismatch(
                len(self.shuffled_frames_predicted_completion_rates),
                len(self.shuffled_frames),
                len(self.shuffled_frames_approx_completion_rates),
            )


@dataclass
class Example:
    """
    Container for a single training/evaluation example consisting of one
    evaluation episode and multiple context episodes.
    """
    eval_episode: Episode
    context_episodes: list[Episode]
