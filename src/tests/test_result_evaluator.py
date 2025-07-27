""" Tests for the result evaluator. """
import json
import os
import tempfile

import numpy as np
import pytest

from gvl.result_evaluator import ResultEvaluator


class TestResultEvaluator:
    @pytest.fixture(scope="class")
    def evaluator(self):
        return ResultEvaluator()

    def test_extract_and_validate(self, evaluator):
        """Test extraction of percentages from list format."""
        response = "```json\n{\"prediction\": [10, 25, 50, 75, 100]}\n```"
        result = evaluator.extract_and_validate(response)
        expected = {"prediction": [10, 25, 50, 75, 100], "length_is_valid": False}
        assert result == expected

    def test_extract_and_validate_no_percentages(self, evaluator):
        """Test extraction when no percentages are present."""
        response = "The robot is making good progress."
        result = evaluator.extract_and_validate(response)
        assert result == {"prediction": [], "length_is_valid": False}

    def test_extract_and_validate_with_text(self, evaluator):
        """Test extraction with text before and after."""
        response = "The results are ```json\n{\"prediction\": [15, 25, 35]}\n``` for the task."
        result = evaluator.extract_and_validate(response)
        expected = {"prediction": [15, 25, 35], "length_is_valid": False}
        assert result == expected
