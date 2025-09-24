import numpy as np

from opengvl.metrics.voc import VOCMetric
from opengvl.utils.data_types import InferredEpisode, InferredFewShotResult


def create_test_episode(
    shuffled_indices: list[int],
    predicted_rates: list[int],
    instruction: str = "Test task",
    episode_index: int = 0,
) -> InferredEpisode:
    """Helper to create test episode data."""
    # Create dummy image arrays
    dummy_image = np.zeros((64, 64, 3), dtype=np.uint8)

    # Create episode with proper data structure
    original_indices = sorted(shuffled_indices)
    original_rates = [i * 10 for i in range(len(original_indices))]  # Linear progression
    approx_rates = [original_rates[original_indices.index(idx)] for idx in shuffled_indices]

    return InferredEpisode(
        instruction=instruction,
        starting_frame=dummy_image,
        episode_index=episode_index,
        original_frames_indices=original_indices,
        original_frames_task_completion_rates=original_rates,
        shuffled_frames_indices=shuffled_indices,
        shuffled_frames=[dummy_image] * len(shuffled_indices),
        shuffled_frames_approx_completion_rates=approx_rates,
        shuffled_frames_predicted_completion_rates=predicted_rates,
    )


def create_test_result(eval_episode: InferredEpisode) -> InferredFewShotResult:
    """Helper to create test InferredFewShotResult."""
    return InferredFewShotResult(
        eval_episode=eval_episode,
        context_episodes=[],
    )


class TestVOCMetric:
    """Test suite for VOCMetric class."""

    def test_metric_name(self):
        """Test that metric name is correct."""
        metric = VOCMetric()
        assert metric.name == "voc"

    def test_perfect_positive_correlation(self):
        """Test perfect positive correlation case."""
        # Shuffled indices: [2, 0, 1, 3] -> chronological predictions: [25, 50, 0, 75]
        # Expected chronological order: [0, 1, 2, 3] with indices [0, 25, 50, 75]
        episode = create_test_episode(
            shuffled_indices=[2, 0, 1, 3],
            predicted_rates=[50, 0, 25, 75],  # When reordered chronologically: [0, 25, 50, 75]
        )
        result = create_test_result(episode)

        metric = VOCMetric()
        metric_result = metric.compute(result)

        assert metric_result.name == "voc"
        assert np.isclose(metric_result.value, 1.0, atol=1e-10)

    def test_perfect_negative_correlation(self):
        """Test perfect negative correlation case."""
        episode = create_test_episode(
            shuffled_indices=[0, 1, 2, 3],
            predicted_rates=[75, 50, 25, 0],  # Decreasing when chronologically ordered
        )
        result = create_test_result(episode)

        metric = VOCMetric()
        metric_result = metric.compute(result)

        assert metric_result.name == "voc"
        assert np.isclose(metric_result.value, -1.0, atol=1e-10)

    def test_constant_predictions(self):
        """Test that constant predictions return score of 0."""
        episode = create_test_episode(
            shuffled_indices=[0, 1, 2, 3],
            predicted_rates=[50, 50, 50, 50],  # All same value
        )
        result = create_test_result(episode)

        metric = VOCMetric()
        metric_result = metric.compute(result)

        assert metric_result.name == "voc"
        assert metric_result.value == 0.0
        assert metric_result.details["note"] == "constant predictions"

    def test_insufficient_length(self):
        """Test that single frame episodes return score of 0."""
        episode = create_test_episode(shuffled_indices=[0], predicted_rates=[50])
        result = create_test_result(episode)

        metric = VOCMetric()
        metric_result = metric.compute(result)

        assert metric_result.name == "voc"
        assert metric_result.value == 0.0
        assert metric_result.details["note"] == "insufficient length"

    def test_partial_correlation(self):
        """Test partial correlation case."""
        # Create a case with partial correlation
        episode = create_test_episode(
            shuffled_indices=[0, 1, 2, 3, 4],
            predicted_rates=[0, 25, 100, 50, 75],  # Mixed order when chronologically arranged
        )
        result = create_test_result(episode)

        metric = VOCMetric()
        metric_result = metric.compute(result)

        assert metric_result.name == "voc"
        assert -1.0 <= metric_result.value <= 1.0  # Should be valid correlation

    def test_random_order(self):
        """Test with random ordering."""
        np.random.seed(42)
        shuffled_indices = [0, 1, 2, 3, 4, 5]
        predicted_rates = [30, 60, 10, 90, 40, 70]

        episode = create_test_episode(shuffled_indices=shuffled_indices, predicted_rates=predicted_rates)
        result = create_test_result(episode)

        metric = VOCMetric()
        metric_result = metric.compute(result)

        assert metric_result.name == "voc"
        assert -1.0 <= metric_result.value <= 1.0
        assert not np.isnan(metric_result.value)

    def test_empty_episodes(self):
        """Test handling of empty episodes."""
        episode = create_test_episode(shuffled_indices=[], predicted_rates=[])
        result = create_test_result(episode)

        metric = VOCMetric()
        metric_result = metric.compute(result)

        assert metric_result.name == "voc"
        assert metric_result.value == 0.0
        assert metric_result.details["note"] == "insufficient length"
