import pytest
import numpy as np
from unittest.mock import Mock

from opengvl.results.prediction import PredictionRecord, DatasetMetrics, aggregate_metrics
from opengvl.utils.data_types import InferredFewShotResult, InferredEpisode, Episode


def create_test_prediction_record(
    index: int = 0,
    dataset: str = "test_dataset",
    predicted_percentages: list[int] = None,
    valid_length: bool = True,
    metrics: dict = None,
    raw_response: str = None,
) -> PredictionRecord:
    """Helper to create test prediction records."""
    if predicted_percentages is None:
        predicted_percentages = [25, 50, 75]
    if metrics is None:
        metrics = {"voc": 0.8}
    
    dummy_image = np.zeros((64, 64, 3), dtype=np.uint8)
    
    # Create test episode data
    eval_episode = InferredEpisode(
        instruction="Test task",
        starting_frame=dummy_image,
        episode_index=index,
        original_frames_indices=[0, 1, 2],
        original_frames_task_completion_rates=[0, 50, 100],
        shuffled_frames_indices=[1, 0, 2],
        shuffled_frames=[dummy_image] * 3,
        shuffled_frames_approx_completion_rates=[50, 0, 100],
        shuffled_frames_predicted_completion_rates=predicted_percentages,
    )
    
    context_episode = Episode(
        instruction="Context task",
        starting_frame=dummy_image,
        episode_index=index + 100,
        original_frames_indices=[0, 1],
        original_frames_task_completion_rates=[0, 100],
        shuffled_frames_indices=[0, 1],
        shuffled_frames=[dummy_image] * 2,
        shuffled_frames_approx_completion_rates=[0, 100],
    )
    
    example = InferredFewShotResult(
        eval_episode=eval_episode,
        context_episodes=[context_episode],
    )
    
    return PredictionRecord(
        index=index,
        dataset=dataset,
        example=example,
        predicted_percentages=predicted_percentages,
        valid_length=valid_length,
        metrics=metrics,
        raw_response=raw_response,
    )


class TestPredictionRecord:
    """Test suite for PredictionRecord class."""

    def test_prediction_record_creation(self):
        """Test basic creation of prediction record."""
        record = create_test_prediction_record()
        
        assert record.index == 0
        assert record.dataset == "test_dataset"
        assert record.predicted_percentages == [25, 50, 75]
        assert record.valid_length == True
        assert record.metrics == {"voc": 0.8}
        assert record.raw_response is None

    def test_prediction_record_with_raw_response(self):
        """Test prediction record with raw response."""
        raw_response = "Model response: 25%, 50%, 75%"
        record = create_test_prediction_record(raw_response=raw_response)
        
        assert record.raw_response == raw_response

    def test_to_dict_without_images(self):
        """Test serialization to dict without images."""
        record = create_test_prediction_record(
            index=5,
            dataset="test_data",
            predicted_percentages=[10, 40, 50],
            metrics={"voc": 0.9, "custom": 1.2},
        )
        
        result_dict = record.to_dict(include_images=False)
        
        assert result_dict["index"] == 5
        assert result_dict["dataset"] == "test_data"
        assert result_dict["predicted_percentages"] == [10, 40, 50]
        assert result_dict["valid_length"] == True
        assert result_dict["metrics"] == {"voc": 0.9, "custom": 1.2}
        
        # Check eval episode structure
        eval_ep = result_dict["eval_episode"]
        assert eval_ep["episode_index"] == 5
        assert eval_ep["instruction"] == "Test task"
        assert eval_ep["original_frames_indices"] == [0, 1, 2]
        assert eval_ep["shuffled_frames_indices"] == [1, 0, 2]
        
        # Check context episode info
        assert result_dict["context_episodes_count"] == 1
        assert result_dict["context_episodes_indices"] == [105]  # index + 100
        assert result_dict["context_episodes_frames_per_episode"] == [2]
        
        # Should not include images by default
        assert "_frames_present" not in eval_ep

    def test_to_dict_with_images(self):
        """Test serialization to dict with images flag."""
        record = create_test_prediction_record()
        
        result_dict = record.to_dict(include_images=True)
        
        # Should include images flag
        assert result_dict["eval_episode"]["_frames_present"] == True

    def test_to_dict_with_raw_response(self):
        """Test serialization includes raw response when present."""
        raw_response = "Raw model output"
        record = create_test_prediction_record(raw_response=raw_response)
        
        result_dict = record.to_dict()
        
        assert result_dict["raw_response"] == raw_response

    def test_to_dict_without_raw_response(self):
        """Test serialization excludes raw response when not present."""
        record = create_test_prediction_record(raw_response=None)
        
        result_dict = record.to_dict()
        
        assert "raw_response" not in result_dict

    def test_to_dict_empty_context_episodes(self):
        """Test serialization with no context episodes."""
        dummy_image = np.zeros((32, 32, 3), dtype=np.uint8)
        
        eval_episode = InferredEpisode(
            instruction="Test task",
            starting_frame=dummy_image,
            episode_index=0,
            original_frames_indices=[0, 1],
            original_frames_task_completion_rates=[0, 100],
            shuffled_frames_indices=[0, 1],
            shuffled_frames=[dummy_image] * 2,
            shuffled_frames_approx_completion_rates=[0, 100],
            shuffled_frames_predicted_completion_rates=[10, 90],
        )
        
        example = InferredFewShotResult(
            eval_episode=eval_episode,
            context_episodes=[],  # No context episodes
        )
        
        record = PredictionRecord(
            index=0,
            dataset="test",
            example=example,
            predicted_percentages=[10, 90],
            valid_length=True,
            metrics={"voc": 0.5},
        )
        
        result_dict = record.to_dict()
        
        assert result_dict["context_episodes_count"] == 0
        assert result_dict["context_episodes_indices"] == []
        assert result_dict["context_episodes_frames_per_episode"] == []


class TestDatasetMetrics:
    """Test suite for DatasetMetrics class."""

    def test_dataset_metrics_creation(self):
        """Test basic creation of dataset metrics."""
        metrics = DatasetMetrics(
            total_examples=10,
            valid_predictions=8,
            length_valid_ratio=0.8,
            metric_means={"voc": 0.75, "accuracy": 0.9},
        )
        
        assert metrics.total_examples == 10
        assert metrics.valid_predictions == 8
        assert metrics.length_valid_ratio == 0.8
        assert metrics.metric_means == {"voc": 0.75, "accuracy": 0.9}

    def test_dataset_metrics_default_metric_means(self):
        """Test default empty metric means."""
        metrics = DatasetMetrics(
            total_examples=5,
            valid_predictions=5,
            length_valid_ratio=1.0,
        )
        
        assert metrics.metric_means == {}

    def test_dataset_metrics_to_dict(self):
        """Test serialization to dict."""
        metrics = DatasetMetrics(
            total_examples=10,
            valid_predictions=8,
            length_valid_ratio=0.8,
            metric_means={"voc": 0.75},
        )
        
        result_dict = metrics.to_dict()
        
        expected = {
            "total_examples": 10,
            "valid_predictions": 8,
            "length_valid_ratio": 0.8,
            "metric_means": {"voc": 0.75},
        }
        assert result_dict == expected


class TestAggregateMetrics:
    """Test suite for aggregate_metrics function."""

    def test_aggregate_metrics_empty_records(self):
        """Test aggregation with empty records list."""
        result = aggregate_metrics([])
        
        assert result.total_examples == 0
        assert result.valid_predictions == 0
        assert result.length_valid_ratio is None
        assert result.metric_means == {}

    def test_aggregate_metrics_single_record(self):
        """Test aggregation with single record."""
        record = create_test_prediction_record(
            valid_length=True,
            metrics={"voc": 0.8, "accuracy": 0.9},
        )
        
        result = aggregate_metrics([record])
        
        assert result.total_examples == 1
        assert result.valid_predictions == 1
        assert result.length_valid_ratio == 1.0
        assert result.metric_means == {"voc": 0.8, "accuracy": 0.9}

    def test_aggregate_metrics_multiple_records(self):
        """Test aggregation with multiple records."""
        records = [
            create_test_prediction_record(
                index=0,
                valid_length=True,
                metrics={"voc": 0.8, "accuracy": 0.9},
            ),
            create_test_prediction_record(
                index=1,
                valid_length=True,
                metrics={"voc": 0.6, "accuracy": 0.7},
            ),
            create_test_prediction_record(
                index=2,
                valid_length=False,  # Invalid prediction
                metrics={"voc": 0.9, "accuracy": 0.8},
            ),
        ]
        
        result = aggregate_metrics(records)
        
        assert result.total_examples == 3
        assert result.valid_predictions == 2
        assert result.length_valid_ratio == 2/3
        
        # Check metric means
        expected_voc = (0.8 + 0.6 + 0.9) / 3  # All records have voc
        expected_accuracy = (0.9 + 0.7 + 0.8) / 3  # All records have accuracy
        
        assert abs(result.metric_means["voc"] - expected_voc) < 1e-10
        assert abs(result.metric_means["accuracy"] - expected_accuracy) < 1e-10

    def test_aggregate_metrics_with_none_values(self):
        """Test aggregation with None metric values."""
        records = [
            create_test_prediction_record(
                index=0,
                metrics={"voc": 0.8, "accuracy": None},  # None value
            ),
            create_test_prediction_record(
                index=1,
                metrics={"voc": 0.6, "accuracy": 0.7},
            ),
        ]
        
        result = aggregate_metrics(records)
        
        # voc should be averaged across both records
        assert abs(result.metric_means["voc"] - 0.7) < 1e-10
        
        # accuracy should only include the non-None value
        assert abs(result.metric_means["accuracy"] - 0.7) < 1e-10

    def test_aggregate_metrics_with_non_numeric_values(self):
        """Test aggregation with non-numeric metric values."""
        records = [
            create_test_prediction_record(
                index=0,
                metrics={"voc": 0.8, "status": "success"},  # String value
            ),
            create_test_prediction_record(
                index=1,
                metrics={"voc": 0.6, "status": "error"},  # String value
            ),
        ]
        
        result = aggregate_metrics(records)
        
        # Only numeric values should be included in means
        assert "voc" in result.metric_means
        assert "status" not in result.metric_means
        assert abs(result.metric_means["voc"] - 0.7) < 1e-10

    def test_aggregate_metrics_mixed_metric_keys(self):
        """Test aggregation with different metric keys across records."""
        records = [
            create_test_prediction_record(
                index=0,
                metrics={"voc": 0.8, "metric_a": 1.0},
            ),
            create_test_prediction_record(
                index=1,
                metrics={"voc": 0.6, "metric_b": 2.0},
            ),
            create_test_prediction_record(
                index=2,
                metrics={"metric_a": 3.0, "metric_c": 4.0},
            ),
        ]
        
        result = aggregate_metrics(records)
        
        # Check each metric mean
        assert abs(result.metric_means["voc"] - 0.7) < 1e-10  # (0.8 + 0.6) / 2
        assert abs(result.metric_means["metric_a"] - 2.0) < 1e-10  # (1.0 + 3.0) / 2
        assert abs(result.metric_means["metric_b"] - 2.0) < 1e-10  # 2.0 / 1
        assert abs(result.metric_means["metric_c"] - 4.0) < 1e-10  # 4.0 / 1

    def test_aggregate_metrics_all_invalid_predictions(self):
        """Test aggregation when all predictions are invalid."""
        records = [
            create_test_prediction_record(
                index=0,
                valid_length=False,
                metrics={"voc": 0.8},
            ),
            create_test_prediction_record(
                index=1,
                valid_length=False,
                metrics={"voc": 0.6},
            ),
        ]
        
        result = aggregate_metrics(records)
        
        assert result.total_examples == 2
        assert result.valid_predictions == 0
        assert result.length_valid_ratio == 0.0
        assert result.metric_means == {"voc": 0.7}  # Metrics still computed


class TestPredictionRecordIntegration:
    """Integration tests for prediction record functionality."""

    def test_round_trip_serialization(self):
        """Test that serialization preserves important information."""
        original_record = create_test_prediction_record(
            index=42,
            dataset="integration_test",
            predicted_percentages=[15, 35, 50],
            valid_length=True,
            metrics={"voc": 0.95, "custom_metric": 1.23},
            raw_response="Raw output text",
        )
        
        # Serialize to dict
        serialized = original_record.to_dict(include_images=True)
        
        # Verify key information is preserved
        assert serialized["index"] == 42
        assert serialized["dataset"] == "integration_test"
        assert serialized["predicted_percentages"] == [15, 35, 50]
        assert serialized["valid_length"] == True
        assert serialized["metrics"]["voc"] == 0.95
        assert serialized["metrics"]["custom_metric"] == 1.23
        assert serialized["raw_response"] == "Raw output text"
        assert serialized["eval_episode"]["_frames_present"] == True

    def test_batch_processing_workflow(self):
        """Test typical batch processing workflow."""
        # Create multiple prediction records
        records = []
        for i in range(5):
            record = create_test_prediction_record(
                index=i,
                dataset="batch_test",
                predicted_percentages=[i*10, (i+1)*10, (i+2)*10],
                valid_length=i % 2 == 0,  # Every other record is valid
                metrics={"voc": 0.5 + i * 0.1},
            )
            records.append(record)
        
        # Aggregate metrics
        aggregated = aggregate_metrics(records)
        
        assert aggregated.total_examples == 5
        assert aggregated.valid_predictions == 3  # Records 0, 2, 4
        assert abs(aggregated.length_valid_ratio - 0.6) < 1e-10
        
        # Average VOC should be (0.5 + 0.6 + 0.7 + 0.8 + 0.9) / 5 = 0.7
        assert abs(aggregated.metric_means["voc"] - 0.7) < 1e-10
        
        # Serialize all records
        serialized_records = [record.to_dict() for record in records]
        assert len(serialized_records) == 5
        
        # Verify each serialized record has required fields
        for i, serialized in enumerate(serialized_records):
            assert serialized["index"] == i
            assert serialized["dataset"] == "batch_test"
            assert serialized["valid_length"] == (i % 2 == 0)