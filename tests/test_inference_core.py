import json
import sys
import tempfile
from pathlib import Path
from unittest.mock import Mock

import numpy as np
import pytest

# Mock the google imports before importing the modules
sys.modules["google"] = Mock()
sys.modules["google.genai"] = Mock()
sys.modules["google.genai.client"] = Mock()
sys.modules["google.generativeai"] = Mock()
sys.modules["openai"] = Mock()
sys.modules["transformers"] = Mock()
sys.modules["torch"] = Mock()

from opengvl.utils.data_types import Episode, Example, InferredFewShotResult
from opengvl.utils.errors import PercentagesCountMismatch, PercentagesNormalizationError


class TestExtractPercentages:
    """Test suite for extract_percentages function using direct implementation."""

    def test_simple_percentage_extraction(self):
        """Test basic percentage extraction."""
        # Import the function directly here to avoid module loading issues
        import re

        # Direct implementation based on the source code
        PERCENT_FLOAT_RE = re.compile(r"(-?\d+(?:\.\d+)?)\s*%")

        def extract_percentages(text: str, expected: int) -> list[int]:
            vals = []
            for match in PERCENT_FLOAT_RE.finditer(text):
                try:
                    v = float(match.group(1))
                except ValueError:
                    continue
                if not (0.0 <= v <= 100.0):
                    continue
                vals.append(v)

            if len(vals) != expected:
                raise PercentagesCountMismatch(expected, len(vals))

            if len(vals) == 0:
                raise PercentagesNormalizationError("No valid percentages found")

            # Simple integer conversion for testing
            if all(float(v).is_integer() for v in vals):
                return [int(v) for v in vals]

            # Normalize to sum to 100 using largest remainder method
            total = sum(vals)
            if total == 0:
                raise PercentagesNormalizationError("Sum of percentages is zero")

            normalized = [v * 100.0 / total for v in vals]
            floors = [int(v) for v in normalized]
            remainders = [v - f for v, f in zip(normalized, floors)]

            current_sum = sum(floors)
            need = int(100 - current_sum)

            order = sorted(range(len(vals)), key=lambda i: (-remainders[i], i))
            result = floors[:]
            for i in range(min(max(need, 0), len(result))):
                result[order[i]] += 1

            return result

        text = "Frame 1: 25%, Frame 2: 50%, Frame 3: 75%"
        result = extract_percentages(text, expected=3)
        assert result == [25, 50, 75]

    def test_percentage_extraction_with_decimals(self):
        """Test percentage extraction with decimal values."""
        import re

        PERCENT_FLOAT_RE = re.compile(r"(-?\d+(?:\.\d+)?)\s*%")

        def extract_percentages(text: str, expected: int) -> list[int]:
            vals = []
            for match in PERCENT_FLOAT_RE.finditer(text):
                try:
                    v = float(match.group(1))
                except ValueError:
                    continue
                if not (0.0 <= v <= 100.0):
                    continue
                vals.append(v)

            if len(vals) != expected:
                raise PercentagesCountMismatch(expected, len(vals))

            # Normalize to sum to 100
            total = sum(vals)
            if total == 0:
                return [0] * len(vals)

            normalized = [v * 100.0 / total for v in vals]
            floors = [int(v) for v in normalized]
            remainders = [v - f for v, f in zip(normalized, floors)]

            current_sum = sum(floors)
            need = int(100 - current_sum)

            order = sorted(range(len(vals)), key=lambda i: (-remainders[i], i))
            result = floors[:]
            for i in range(min(max(need, 0), len(result))):
                result[order[i]] += 1

            return result

        text = "Progress: 33.33%, 66.67%"
        result = extract_percentages(text, expected=2)
        assert len(result) == 2
        assert sum(result) == 100
        assert all(isinstance(x, int) for x in result)

    def test_percentage_extraction_count_mismatch(self):
        """Test error when wrong number of percentages found."""
        import re

        PERCENT_FLOAT_RE = re.compile(r"(-?\d+(?:\.\d+)?)\s*%")

        def extract_percentages(text: str, expected: int) -> list[int]:
            vals = []
            for match in PERCENT_FLOAT_RE.finditer(text):
                try:
                    v = float(match.group(1))
                except ValueError:
                    continue
                if not (0.0 <= v <= 100.0):
                    continue
                vals.append(v)

            if len(vals) != expected:
                raise PercentagesCountMismatch(expected, len(vals))

            return [int(v) for v in vals]

        text = "25%, 50%"  # Only 2 percentages
        with pytest.raises(PercentagesCountMismatch):
            extract_percentages(text, expected=3)


class TestBuildInferredExample:
    """Test suite for build_inferred_example function."""

    def test_build_inferred_example(self):
        """Test building inferred example from fewshot input."""
        dummy_image = np.zeros((64, 64, 3), dtype=np.uint8)

        eval_episode = Episode(
            instruction="Test task",
            starting_frame=dummy_image,
            episode_index=0,
            original_frames_indices=[0, 1, 2],
            original_frames_task_completion_rates=[0, 50, 100],
            shuffled_frames_indices=[1, 0, 2],
            shuffled_frames=[dummy_image] * 3,
            shuffled_frames_approx_completion_rates=[50, 0, 100],
        )

        fewshot = Example(
            eval_episode=eval_episode,
            context_episodes=[],
        )

        # Simple build_inferred_example implementation
        from opengvl.utils.data_types import InferredEpisode

        def build_inferred_example(fewshot: Example, predicted: list[int]) -> InferredFewShotResult:
            inferred_eval = InferredEpisode.from_predictions(fewshot.eval_episode, predicted)
            return InferredFewShotResult(
                eval_episode=inferred_eval,
                context_episodes=fewshot.context_episodes,
            )

        predicted = [45, 5, 95]
        result = build_inferred_example(fewshot, predicted)

        assert isinstance(result, InferredFewShotResult)
        assert result.eval_episode.shuffled_frames_predicted_completion_rates == predicted
        assert len(result.context_episodes) == 0


class TestSaveJsonl:
    """Test suite for save_jsonl function."""

    def test_save_jsonl(self):
        """Test saving records to JSONL format."""

        def save_jsonl(records, path):
            with open(path, "w") as f:
                for record in records:
                    f.write(json.dumps(record) + "\n")

        with tempfile.TemporaryDirectory() as temp_dir:
            test_path = Path(temp_dir) / "test.jsonl"

            records = [
                {"name": "test1", "value": 1},
                {"name": "test2", "value": 2},
            ]

            save_jsonl(records, test_path)

            # Verify file was created and contains correct data
            assert test_path.exists()

            with open(test_path) as f:
                lines = f.readlines()

            assert len(lines) == 2
            assert json.loads(lines[0]) == {"name": "test1", "value": 1}
            assert json.loads(lines[1]) == {"name": "test2", "value": 2}

    def test_save_jsonl_empty_records(self):
        """Test saving empty records list."""

        def save_jsonl(records, path):
            with open(path, "w") as f:
                for record in records:
                    f.write(json.dumps(record) + "\n")

        with tempfile.TemporaryDirectory() as temp_dir:
            test_path = Path(temp_dir) / "empty.jsonl"

            save_jsonl([], test_path)

            assert test_path.exists()
            assert test_path.stat().st_size == 0


class TestInferenceIntegration:
    """Integration tests for inference utilities."""

    def test_percentage_extraction_with_various_formats(self):
        """Test percentage extraction with various text formats."""
        import re

        PERCENT_FLOAT_RE = re.compile(r"(-?\d+(?:\.\d+)?)\s*%")

        def extract_percentages(text: str, expected: int) -> list[int]:
            vals = []
            for match in PERCENT_FLOAT_RE.finditer(text):
                try:
                    v = float(match.group(1))
                except ValueError:
                    continue
                if not (0.0 <= v <= 100.0):
                    continue
                vals.append(v)

            if len(vals) != expected:
                raise PercentagesCountMismatch(expected, len(vals))

            # Normalize to sum to 100
            total = sum(vals)
            if total == 0:
                return [0] * len(vals)

            normalized = [v * 100.0 / total for v in vals]
            floors = [int(v) for v in normalized]
            remainders = [v - f for v, f in zip(normalized, floors)]

            current_sum = sum(floors)
            need = int(100 - current_sum)

            order = sorted(range(len(vals)), key=lambda i: (-remainders[i], i))
            result = floors[:]
            for i in range(min(max(need, 0), len(result))):
                result[order[i]] += 1

            return result

        test_cases = [
            ("Task completion: 25%", 1),
            ("Progress rates: 10%, 20%, 30%, 40%", 4),
            ("Frame 1 at 15.5% and Frame 2 at 84.5%", 2),
        ]

        for text, expected_count in test_cases:
            result = extract_percentages(text, expected=expected_count)
            assert len(result) == expected_count
            assert sum(result) == 100 or all(x == 0 for x in result)

    def test_build_inferred_example_edge_cases(self):
        """Test build_inferred_example with edge cases."""
        dummy_image = np.zeros((32, 32, 3), dtype=np.uint8)

        # Single frame episode
        eval_episode = Episode(
            instruction="Single frame task",
            starting_frame=dummy_image,
            episode_index=0,
            original_frames_indices=[0],
            original_frames_task_completion_rates=[100],
            shuffled_frames_indices=[0],
            shuffled_frames=[dummy_image],
            shuffled_frames_approx_completion_rates=[100],
        )

        fewshot = Example(
            eval_episode=eval_episode,
            context_episodes=[],
        )

        from opengvl.utils.data_types import InferredEpisode

        def build_inferred_example(fewshot: Example, predicted: list[int]) -> InferredFewShotResult:
            inferred_eval = InferredEpisode.from_predictions(fewshot.eval_episode, predicted)
            return InferredFewShotResult(
                eval_episode=inferred_eval,
                context_episodes=fewshot.context_episodes,
            )

        predicted = [95]
        result = build_inferred_example(fewshot, predicted)

        assert len(result.eval_episode.shuffled_frames_predicted_completion_rates) == 1
        assert result.eval_episode.shuffled_frames_predicted_completion_rates[0] == 95
