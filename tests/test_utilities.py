import numpy as np
import pytest

from opengvl.utils.constants import (
    MAX_TOKENS_TO_GENERATE,
    N_DEBUG_PROMPT_CHARS,
)


class TestConstants:
    """Test suite for utility constants."""

    def test_max_tokens_to_generate_is_positive(self):
        """Test that MAX_TOKENS_TO_GENERATE is a positive integer."""
        assert isinstance(MAX_TOKENS_TO_GENERATE, int)
        assert MAX_TOKENS_TO_GENERATE > 0

    def test_n_debug_prompt_chars_is_positive(self):
        """Test that N_DEBUG_PROMPT_CHARS is a positive integer."""
        assert isinstance(N_DEBUG_PROMPT_CHARS, int)
        assert N_DEBUG_PROMPT_CHARS > 0

    def test_constants_have_reasonable_values(self):
        """Test that constants have reasonable values."""
        # MAX_TOKENS_TO_GENERATE should be reasonable for LLMs
        assert 100 <= MAX_TOKENS_TO_GENERATE <= 100000

        # N_DEBUG_PROMPT_CHARS should be reasonable for debugging
        assert 10 <= N_DEBUG_PROMPT_CHARS <= 10000


class TestAliases:
    """Test suite for type aliases."""

    def test_image_type_aliases(self):
        """Test that image type aliases work correctly."""
        from opengvl.utils.aliases import ImageNumpy, ImageT

        # Create test images
        numpy_image = np.zeros((64, 64, 3), dtype=np.uint8)

        # Type annotations should accept these without errors
        def process_numpy_image(img: ImageNumpy) -> ImageNumpy:
            return img

        def process_generic_image(img: ImageT) -> ImageT:
            return img

        # These should work without type errors
        result1 = process_numpy_image(numpy_image)
        result2 = process_generic_image(numpy_image)

        assert result1 is numpy_image
        assert result2 is numpy_image

    def test_event_type_aliases(self):
        """Test that event type aliases are properly defined."""
        from opengvl.utils.aliases import Event, ImageEvent, TextEvent

        # These should be available for type annotations
        assert Event is not None
        assert TextEvent is not None
        assert ImageEvent is not None


class TestUtilityFunctions:
    """Test miscellaneous utility functions."""

    def test_image_processing_basic(self):
        """Test basic image processing functionality."""
        # Create a test image
        test_image = np.random.randint(0, 256, (64, 64, 3), dtype=np.uint8)

        # Test that image has correct properties
        assert test_image.shape == (64, 64, 3)
        assert test_image.dtype == np.uint8
        assert test_image.min() >= 0
        assert test_image.max() <= 255

    def test_data_structure_creation_patterns(self):
        """Test common patterns for creating data structures."""
        from opengvl.utils.data_types import Episode, Example

        # Test creating minimal valid episode
        dummy_image = np.zeros((32, 32, 3), dtype=np.uint8)

        episode = Episode(
            instruction="Minimal test",
            starting_frame=dummy_image,
            episode_index=0,
            original_frames_indices=[0],
            original_frames_task_completion_rates=[100],
            shuffled_frames_indices=[0],
            shuffled_frames=[dummy_image],
            shuffled_frames_approx_completion_rates=[100],
        )

        example = Example(
            eval_episode=episode,
            context_episodes=[],
        )

        assert len(example.eval_episode.shuffled_frames) == 1
        assert len(example.context_episodes) == 0

    def test_error_handling_patterns(self):
        """Test common error handling patterns."""
        from opengvl.utils.errors import PercentagesCountMismatch

        # Test that errors can be used in try/except blocks
        try:
            raise PercentagesCountMismatch(5, 3)
        except PercentagesCountMismatch as e:
            assert e.expected == 5
            assert e.found == 3
            assert "Expected 5 percentages, found 3" in str(e)

    def test_metric_computation_patterns(self):
        """Test patterns for metric computation."""
        from opengvl.metrics.voc import VOCMetric
        from opengvl.utils.data_types import InferredEpisode, InferredFewShotResult

        dummy_image = np.zeros((32, 32, 3), dtype=np.uint8)

        # Create test data for metric computation
        eval_episode = InferredEpisode(
            instruction="Test",
            starting_frame=dummy_image,
            episode_index=0,
            original_frames_indices=[0, 1, 2],
            original_frames_task_completion_rates=[0, 50, 100],
            shuffled_frames_indices=[0, 1, 2],
            shuffled_frames=[dummy_image] * 3,
            shuffled_frames_approx_completion_rates=[0, 50, 100],
            shuffled_frames_predicted_completion_rates=[5, 45, 95],
        )

        result = InferredFewShotResult(
            eval_episode=eval_episode,
            context_episodes=[],
        )

        metric = VOCMetric()
        metric_result = metric.compute(result)

        assert metric_result.name == "voc"
        assert isinstance(metric_result.value, float)
        assert -1.0 <= metric_result.value <= 1.0


class TestIntegrationPatterns:
    """Test integration patterns across modules."""

    def test_end_to_end_data_flow(self):
        """Test end-to-end data flow through core components."""
        from opengvl.metrics.voc import VOCMetric
        from opengvl.results.prediction import PredictionRecord, aggregate_metrics
        from opengvl.utils.data_types import Episode, Example, InferredEpisode, InferredFewShotResult

        dummy_image = np.zeros((64, 64, 3), dtype=np.uint8)

        # 1. Create original episode
        episode = Episode(
            instruction="Integration test task",
            starting_frame=dummy_image,
            episode_index=0,
            original_frames_indices=[0, 1, 2, 3],
            original_frames_task_completion_rates=[0, 33, 67, 100],
            shuffled_frames_indices=[2, 0, 3, 1],
            shuffled_frames=[dummy_image] * 4,
            shuffled_frames_approx_completion_rates=[67, 0, 100, 33],
        )

        # 2. Create example with context
        example = Example(
            eval_episode=episode,
            context_episodes=[],
        )

        # 3. Simulate model predictions
        predictions = [70, 5, 95, 30]  # Predictions for shuffled frames

        # 4. Create inferred result
        inferred_episode = InferredEpisode.from_predictions(example.eval_episode, predictions)
        inferred_result = InferredFewShotResult(
            eval_episode=inferred_episode,
            context_episodes=example.context_episodes,
        )

        # 5. Compute metrics
        voc_metric = VOCMetric()
        metric_result = voc_metric.compute(inferred_result)

        # 6. Create prediction record
        record = PredictionRecord(
            index=0,
            dataset="integration_test",
            example=inferred_result,
            predicted_percentages=predictions,
            valid_length=True,
            metrics={metric_result.name: metric_result.value},
            raw_response="Model predicted: 70%, 5%, 95%, 30%",
        )

        # 7. Aggregate metrics
        aggregated = aggregate_metrics([record])

        # Verify the complete flow
        assert record.index == 0
        assert record.dataset == "integration_test"
        assert len(record.predicted_percentages) == 4
        assert aggregated.total_examples == 1
        assert aggregated.valid_predictions == 1
        assert "voc" in aggregated.metric_means

    def test_error_propagation_patterns(self):
        """Test how errors propagate through the system."""
        from opengvl.utils.data_types import Episode
        from opengvl.utils.errors import ShuffledFramesLengthMismatch

        dummy_image = np.zeros((32, 32, 3), dtype=np.uint8)

        # Test that validation errors are properly raised
        with pytest.raises(ShuffledFramesLengthMismatch):
            Episode(
                instruction="Error test",
                starting_frame=dummy_image,
                episode_index=0,
                original_frames_indices=[0, 1],
                original_frames_task_completion_rates=[0, 100],
                shuffled_frames_indices=[0, 1],
                shuffled_frames=[dummy_image],  # Wrong length!
                shuffled_frames_approx_completion_rates=[0, 100],
            )

    def test_serialization_roundtrip(self):
        """Test serialization and deserialization patterns."""
        from opengvl.results.prediction import DatasetMetrics, PredictionRecord
        from opengvl.utils.data_types import InferredEpisode, InferredFewShotResult

        dummy_image = np.zeros((32, 32, 3), dtype=np.uint8)

        # Create test data
        eval_episode = InferredEpisode(
            instruction="Serialization test",
            starting_frame=dummy_image,
            episode_index=42,
            original_frames_indices=[0, 1],
            original_frames_task_completion_rates=[0, 100],
            shuffled_frames_indices=[1, 0],
            shuffled_frames=[dummy_image] * 2,
            shuffled_frames_approx_completion_rates=[100, 0],
            shuffled_frames_predicted_completion_rates=[95, 5],
        )

        result = InferredFewShotResult(
            eval_episode=eval_episode,
            context_episodes=[],
        )

        record = PredictionRecord(
            index=42,
            dataset="serialization_test",
            example=result,
            predicted_percentages=[95, 5],
            valid_length=True,
            metrics={"voc": 0.85},
        )

        # Test serialization
        serialized = record.to_dict()

        # Verify key information is preserved
        assert serialized["index"] == 42
        assert serialized["dataset"] == "serialization_test"
        assert serialized["eval_episode"]["episode_index"] == 42
        assert serialized["eval_episode"]["instruction"] == "Serialization test"
        assert serialized["predicted_percentages"] == [95, 5]
        assert serialized["metrics"]["voc"] == 0.85

        # Test DatasetMetrics serialization
        metrics = DatasetMetrics(
            total_examples=1,
            valid_predictions=1,
            length_valid_ratio=1.0,
            metric_means={"voc": 0.85},
        )

        metrics_dict = metrics.to_dict()
        assert metrics_dict["total_examples"] == 1
        assert metrics_dict["metric_means"]["voc"] == 0.85
