import numpy as np
import pytest

from opengvl.utils.data_types import Episode, Example, InferredEpisode, InferredFewShotResult
from opengvl.utils.errors import (
    OriginalFramesLengthMismatch,
    ShuffledFramesIndicesNotSubset,
    ShuffledFramesLengthMismatch,
)


class TestEpisode:
    """Test suite for Episode data structure."""

    def test_valid_episode_creation(self):
        """Test creation of valid episode."""
        dummy_image = np.zeros((64, 64, 3), dtype=np.uint8)

        episode = Episode(
            instruction="Test task",
            starting_frame=dummy_image,
            episode_index=0,
            original_frames_indices=[0, 1, 2, 3],
            original_frames_task_completion_rates=[0, 25, 50, 75],
            shuffled_frames_indices=[1, 3, 0, 2],
            shuffled_frames=[dummy_image] * 4,
            shuffled_frames_approx_completion_rates=[25, 75, 0, 50],
        )

        assert episode.instruction == "Test task"
        assert episode.episode_index == 0
        assert len(episode.shuffled_frames) == 4

    def test_original_frames_length_mismatch(self):
        """Test validation of original frames length mismatch."""
        dummy_image = np.zeros((64, 64, 3), dtype=np.uint8)

        with pytest.raises(OriginalFramesLengthMismatch):
            Episode(
                instruction="Test task",
                starting_frame=dummy_image,
                episode_index=0,
                original_frames_indices=[0, 1, 2],  # 3 items
                original_frames_task_completion_rates=[0, 25],  # 2 items - mismatch!
                shuffled_frames_indices=[1, 0, 2],
                shuffled_frames=[dummy_image] * 3,
                shuffled_frames_approx_completion_rates=[25, 0, 50],
            )

    def test_shuffled_frames_length_mismatch(self):
        """Test validation of shuffled frames length mismatch."""
        dummy_image = np.zeros((64, 64, 3), dtype=np.uint8)

        with pytest.raises(ShuffledFramesLengthMismatch):
            Episode(
                instruction="Test task",
                starting_frame=dummy_image,
                episode_index=0,
                original_frames_indices=[0, 1, 2],
                original_frames_task_completion_rates=[0, 25, 50],
                shuffled_frames_indices=[1, 0, 2],  # 3 items
                shuffled_frames=[dummy_image] * 2,  # 2 items - mismatch!
                shuffled_frames_approx_completion_rates=[25, 0, 50],  # 3 items
            )

    def test_shuffled_frames_indices_not_subset(self):
        """Test validation that shuffled indices are subset of original."""
        dummy_image = np.zeros((64, 64, 3), dtype=np.uint8)

        with pytest.raises(ShuffledFramesIndicesNotSubset):
            Episode(
                instruction="Test task",
                starting_frame=dummy_image,
                episode_index=0,
                original_frames_indices=[0, 1, 2],
                original_frames_task_completion_rates=[0, 25, 50],
                shuffled_frames_indices=[1, 4, 2],  # 4 not in original!
                shuffled_frames=[dummy_image] * 3,
                shuffled_frames_approx_completion_rates=[25, 75, 50],
            )


class TestInferredEpisode:
    """Test suite for InferredEpisode data structure."""

    def test_valid_inferred_episode_creation(self):
        """Test creation of valid inferred episode."""
        dummy_image = np.zeros((64, 64, 3), dtype=np.uint8)

        episode = InferredEpisode(
            instruction="Test task",
            starting_frame=dummy_image,
            episode_index=0,
            original_frames_indices=[0, 1, 2, 3],
            original_frames_task_completion_rates=[0, 25, 50, 75],
            shuffled_frames_indices=[1, 3, 0, 2],
            shuffled_frames=[dummy_image] * 4,
            shuffled_frames_approx_completion_rates=[25, 75, 0, 50],
            shuffled_frames_predicted_completion_rates=[30, 70, 5, 45],
        )

        assert len(episode.shuffled_frames_predicted_completion_rates) == 4

    def test_predicted_rates_length_mismatch(self):
        """Test validation of predicted rates length mismatch."""
        dummy_image = np.zeros((64, 64, 3), dtype=np.uint8)

        with pytest.raises(ShuffledFramesLengthMismatch):
            InferredEpisode(
                instruction="Test task",
                starting_frame=dummy_image,
                episode_index=0,
                original_frames_indices=[0, 1, 2, 3],
                original_frames_task_completion_rates=[0, 25, 50, 75],
                shuffled_frames_indices=[1, 3, 0, 2],
                shuffled_frames=[dummy_image] * 4,
                shuffled_frames_approx_completion_rates=[25, 75, 0, 50],
                shuffled_frames_predicted_completion_rates=[30, 70],  # Too few predictions!
            )

    def test_from_predictions_factory(self):
        """Test factory method to create InferredEpisode from Episode."""
        dummy_image = np.zeros((64, 64, 3), dtype=np.uint8)

        base_episode = Episode(
            instruction="Test task",
            starting_frame=dummy_image,
            episode_index=0,
            original_frames_indices=[0, 1, 2, 3],
            original_frames_task_completion_rates=[0, 25, 50, 75],
            shuffled_frames_indices=[1, 3, 0, 2],
            shuffled_frames=[dummy_image] * 4,
            shuffled_frames_approx_completion_rates=[25, 75, 0, 50],
        )

        predictions = [30, 70, 5, 45]
        inferred = InferredEpisode.from_predictions(base_episode, predictions)

        assert inferred.instruction == base_episode.instruction
        assert inferred.shuffled_frames_predicted_completion_rates == predictions
        assert len(inferred.shuffled_frames) == len(predictions)


class TestExample:
    """Test suite for Example data structure."""

    def test_example_creation_and_repr(self):
        """Test creation and string representation of Example."""
        dummy_image = np.zeros((64, 64, 3), dtype=np.uint8)

        eval_episode = Episode(
            instruction="Eval task",
            starting_frame=dummy_image,
            episode_index=0,
            original_frames_indices=[0, 1, 2],
            original_frames_task_completion_rates=[0, 50, 100],
            shuffled_frames_indices=[1, 0, 2],
            shuffled_frames=[dummy_image] * 3,
            shuffled_frames_approx_completion_rates=[50, 0, 100],
        )

        context_episode = Episode(
            instruction="Context task",
            starting_frame=dummy_image,
            episode_index=1,
            original_frames_indices=[0, 1],
            original_frames_task_completion_rates=[0, 100],
            shuffled_frames_indices=[0, 1],
            shuffled_frames=[dummy_image] * 2,
            shuffled_frames_approx_completion_rates=[0, 100],
        )

        example = Example(
            eval_episode=eval_episode,
            context_episodes=[context_episode],
        )

        repr_str = repr(example)
        assert "eval_episode_index=0" in repr_str
        assert "eval_frames=3" in repr_str
        assert "context_episodes=1" in repr_str
        assert "context_frames_per_episode=[2]" in repr_str
        assert "context_frames_total=2" in repr_str


class TestInferredFewShotResult:
    """Test suite for InferredFewShotResult data structure."""

    def test_inferred_few_shot_result_creation(self):
        """Test creation of InferredFewShotResult."""
        dummy_image = np.zeros((64, 64, 3), dtype=np.uint8)

        eval_episode = InferredEpisode(
            instruction="Test task",
            starting_frame=dummy_image,
            episode_index=0,
            original_frames_indices=[0, 1, 2],
            original_frames_task_completion_rates=[0, 50, 100],
            shuffled_frames_indices=[1, 0, 2],
            shuffled_frames=[dummy_image] * 3,
            shuffled_frames_approx_completion_rates=[50, 0, 100],
            shuffled_frames_predicted_completion_rates=[45, 5, 95],
        )

        context_episode = Episode(
            instruction="Context task",
            starting_frame=dummy_image,
            episode_index=1,
            original_frames_indices=[0, 1],
            original_frames_task_completion_rates=[0, 100],
            shuffled_frames_indices=[0, 1],
            shuffled_frames=[dummy_image] * 2,
            shuffled_frames_approx_completion_rates=[0, 100],
        )

        result = InferredFewShotResult(
            eval_episode=eval_episode,
            context_episodes=[context_episode],
        )

        assert isinstance(result.eval_episode, InferredEpisode)
        assert len(result.context_episodes) == 1
        assert len(result.eval_episode.shuffled_frames_predicted_completion_rates) == 3
