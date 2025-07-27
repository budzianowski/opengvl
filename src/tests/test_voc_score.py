from __future__ import annotations

import numpy as np
import pytest

from gvl.voc_score import value_order_correlation


class TestValueOrderCorrelation:
    """Test suite for value_order_correlation function."""

    @pytest.mark.parametrize(
        "values,true_values,expected",
        [
            # Perfect positive correlation
            ([0, 25, 50, 75, 100], [0, 25, 50, 75, 100], 1.0),
            ([1, 2, 3, 4, 5], [0, 1, 2, 3, 4], 1.0),
            # Perfect negative correlation
            ([100, 75, 50, 25, 0], [0, 25, 50, 75, 100], -1.0),
            ([5, 4, 3, 2, 1], [0, 1, 2, 3, 4], -1.0),
            # Two element cases
            ([0, 100], [0, 1], 1.0),
            ([100, 0], [0, 1], -1.0),
            ([50, 50], [0, 1], np.nan),
            # Known partial correlations
            ([25, 100, 0], [0, 1, 2], -0.5),
            ([0, 100, 50], [0, 1, 2], 0.5),
        ],
    )
    def test_known_correlations(self, values, true_values, expected):
        """Test cases with known expected correlations."""
        result = value_order_correlation(values, true_values)
        if np.isnan(expected):
            assert np.isnan(result)
        else:
            assert np.isclose(result, expected, atol=1e-10)

    def test_constant_values(self):
        """Test that constant values return NaN (undefined correlation)."""
        constant_cases = [[50, 50, 50, 50, 50], [0, 0, 0], [100, 100], [42] * 10]

        for values in constant_cases:
            true_values = list(range(len(values)))
            result = value_order_correlation(values, true_values)
            assert np.isnan(result), f"Expected NaN for constant values {values}"

    def test_different_true_values(self):
        """Test same values with different true value orderings."""
        values = [10, 20, 30, 40, 50]

        # Ascending true values -> perfect positive correlation
        voc1 = value_order_correlation(values, [0, 1, 2, 3, 4])

        # Descending true values -> perfect negative correlation
        voc2 = value_order_correlation(values, [4, 3, 2, 1, 0])

        # Random true values -> some correlation
        voc3 = value_order_correlation(values, [2, 0, 4, 1, 3])

        assert np.isclose(voc1, 1.0)
        assert np.isclose(voc2, -1.0)
        assert -1.0 <= voc3 <= 1.0

    @pytest.mark.parametrize("length", [1, 2, 3, 5, 10, 100])
    def test_random_sequences(self, length):
        """Test random sequences of various lengths."""
        np.random.seed(42)  # For reproducibility
        values = np.random.randint(0, 101, length).tolist()
        true_values = np.random.randint(0, 101, length).tolist()

        result = value_order_correlation(values, true_values)

        # Result should be between -1 and 1, or NaN for constant sequences
        assert np.isnan(result) or (-1.0 <= result <= 1.0)

    def test_edge_case_single_value(self):
        """Test single value case."""
        result = value_order_correlation([42], [0])
        # Single value should return NaN (no variance)
        assert np.isnan(result)

    def test_extreme_values(self):
        """Test with extreme value ranges."""
        # Very large values
        large_values = [1000000, 2000000, 3000000]
        result1 = value_order_correlation(large_values, [0, 1, 2])
        assert np.isclose(result1, 1.0)

        # Very small values
        small_values = [0.001, 0.002, 0.003]
        result2 = value_order_correlation(small_values, [0, 1, 2])
        assert np.isclose(result2, 1.0)

        # Negative values (though spec says 0-100)
        negative_values = [-10, -5, 0, 5, 10]
        result3 = value_order_correlation(negative_values, [0, 1, 2, 3, 4])
        assert np.isclose(result3, 1.0)

    def test_duplicate_values_mixed(self):
        """Test sequences with some duplicate values."""
        # Some duplicates but not all
        values = [10, 20, 20, 30, 30, 40]
        true_values = [0, 1, 2, 3, 4, 5]
        result = value_order_correlation(values, true_values)

        # Should still compute correlation, not NaN
        assert not np.isnan(result)
        assert -1.0 <= result <= 1.0

    def test_anticorrelated_sequences(self):
        """Test sequences that show negative correlation."""
        test_cases = [
            # When true_values decrease as predicted values increase
            ([1, 2, 3, 4, 5], [4, 3, 2, 1, 0]),  # Perfect negative correlation
            ([0, 10, 20, 30], [3, 2, 1, 0]),  # Perfect negative correlation
        ]

        for values, true_values in test_cases:
            result = value_order_correlation(values, true_values)
            assert np.isclose(result, -1.0, atol=1e-10)  # Should be -1.0 for perfect negative correlation

    def test_complex_patterns(self):
        """Test more complex correlation patterns."""
        # Alternating pattern
        values = [1, 3, 2, 4, 3, 5]
        true_values = [0, 1, 2, 3, 4, 5]
        result1 = value_order_correlation(values, true_values)
        assert -1.0 <= result1 <= 1.0

        # Mountain pattern (up then down)
        values = [1, 2, 3, 4, 3, 2, 1]
        true_values = [0, 1, 2, 3, 4, 5, 6]
        result2 = value_order_correlation(values, true_values)
        assert -1.0 <= result2 <= 1.0

    def test_input_validation_edge_cases(self):
        """Test edge cases for input validation."""
        result = value_order_correlation([], [])
        assert np.isnan(result)

        with pytest.raises(ValueError):
            value_order_correlation([1, 2, 3], [0, 1, 2, 3])

        with pytest.raises(ValueError):
            value_order_correlation([1, 2, 3], None)

    def test_numpy_array_inputs(self):
        """Test that function works with numpy arrays as well as lists."""
        values_list = [10, 20, 30, 40, 50]
        values_array = np.array(values_list)
        true_values_list = [0, 1, 2, 3, 4]
        true_values_array = np.array(true_values_list)

        result_list = value_order_correlation(values_list, true_values_list)
        result_mixed1 = value_order_correlation(values_array, true_values_list)
        result_mixed2 = value_order_correlation(values_list, true_values_array)
        result_array = value_order_correlation(values_array, true_values_array)

        # All should give the same result
        assert np.isclose(result_list, result_mixed1)
        assert np.isclose(result_list, result_mixed2)
        assert np.isclose(result_list, result_array)
        assert np.isclose(result_list, 1.0)
