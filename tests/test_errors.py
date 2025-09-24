import pytest

from opengvl.utils.errors import (
    ImageEncodingError,
    InputTooLongError,
    MaxRetriesExceeded,
    OriginalFramesLengthMismatch,
    PercentagesCountMismatch,
    PercentagesNormalizationError,
    ShuffledFramesIndicesNotSubset,
    ShuffledFramesLengthMismatch,
)


class TestImageEncodingError:
    """Test suite for ImageEncodingError."""

    def test_default_message(self):
        """Test default error message."""
        error = ImageEncodingError()
        assert "Image encoding error." in str(error)

    def test_custom_message(self):
        """Test custom error message."""
        error = ImageEncodingError("Custom error message")
        assert str(error) == "Custom error message"

    def test_kwargs_in_message(self):
        """Test error message with kwargs details."""
        error = ImageEncodingError(width=100, height=200)
        assert "width=100" in str(error)
        assert "height=200" in str(error)
        assert error.details == {"width": 100, "height": 200}

    def test_message_and_kwargs(self):
        """Test custom message overrides kwargs."""
        error = ImageEncodingError("Custom message", width=100)
        assert str(error) == "Custom message"
        assert error.details == {"width": 100}


class TestOriginalFramesLengthMismatch:
    """Test suite for OriginalFramesLengthMismatch."""

    def test_error_message(self):
        """Test error message formatting."""
        error = OriginalFramesLengthMismatch(5, 3)
        expected = "Lengths of original_frames_indices (5) and original_frames_task_completion_rates (3) must match"
        assert str(error) == expected


class TestShuffledFramesLengthMismatch:
    """Test suite for ShuffledFramesLengthMismatch."""

    def test_error_message(self):
        """Test error message formatting."""
        error = ShuffledFramesLengthMismatch(4, 3, 4)
        expected = (
            "shuffled_frames_indices (4), shuffled_frames (3), shuffled_frames_approx_completion_rates (4) must be 1:1"
        )
        assert str(error) == expected


class TestShuffledFramesIndicesNotSubset:
    """Test suite for ShuffledFramesIndicesNotSubset."""

    def test_error_message(self):
        """Test error message."""
        error = ShuffledFramesIndicesNotSubset()
        expected = "All shuffled_frames_indices must be present in original_frames_indices"
        assert str(error) == expected


class TestPercentagesCountMismatch:
    """Test suite for PercentagesCountMismatch."""

    def test_error_message_and_attributes(self):
        """Test error message and attributes."""
        error = PercentagesCountMismatch(5, 3)
        assert str(error) == "Expected 5 percentages, found 3"
        assert error.expected == 5
        assert error.found == 3


class TestPercentagesNormalizationError:
    """Test suite for PercentagesNormalizationError."""

    def test_default_message(self):
        """Test default error message."""
        error = PercentagesNormalizationError()
        assert "Unable to normalize percentages (invalid sum)" in str(error)

    def test_custom_message(self):
        """Test custom error message."""
        error = PercentagesNormalizationError("Custom normalization error")
        assert str(error) == "Custom normalization error"


class TestMaxRetriesExceeded:
    """Test suite for MaxRetriesExceeded."""

    def test_error_message(self):
        """Test error message formatting."""
        error = MaxRetriesExceeded(3)
        assert str(error) == "Max retries exceeded after 3 attempts"


class TestInputTooLongError:
    """Test suite for InputTooLongError."""

    def test_error_message_and_attributes(self):
        """Test error message and attributes."""
        error = InputTooLongError(1000, 500)
        assert str(error) == "Input length too large: 1000 > 500"
        assert error.length == 1000
        assert error.limit == 500


class TestErrorInheritance:
    """Test error inheritance hierarchy."""

    def test_all_errors_inherit_from_exception(self):
        """Test that all custom errors inherit from appropriate base classes."""
        assert issubclass(ImageEncodingError, RuntimeError)
        assert issubclass(OriginalFramesLengthMismatch, Exception)
        assert issubclass(ShuffledFramesLengthMismatch, Exception)
        assert issubclass(ShuffledFramesIndicesNotSubset, Exception)
        assert issubclass(PercentagesCountMismatch, Exception)
        assert issubclass(PercentagesNormalizationError, Exception)
        assert issubclass(MaxRetriesExceeded, Exception)
        assert issubclass(InputTooLongError, Exception)

    def test_errors_can_be_raised_and_caught(self):
        """Test that errors can be raised and caught properly."""
        with pytest.raises(ImageEncodingError):
            raise ImageEncodingError("Test error")

        with pytest.raises(PercentagesCountMismatch):
            raise PercentagesCountMismatch(5, 3)

        with pytest.raises(MaxRetriesExceeded):
            raise MaxRetriesExceeded(3)

        with pytest.raises(InputTooLongError):
            raise InputTooLongError(1000, 500)


class TestErrorUsagePatterns:
    """Test common error usage patterns."""

    def test_error_chaining(self):
        """Test error chaining with raise from."""
        try:
            try:
                raise ValueError("Original error")
            except ValueError as e:
                raise PercentagesNormalizationError("Wrapped error") from e
        except PercentagesNormalizationError as e:
            assert e.__cause__ is not None
            assert isinstance(e.__cause__, ValueError)

    def test_error_context_manager(self):
        """Test using errors in context managers."""

        def might_fail(should_fail=False):
            if should_fail:
                raise PercentagesCountMismatch(5, 3)
            return "success"

        # Should not raise
        result = might_fail(False)
        assert result == "success"

        # Should raise
        with pytest.raises(PercentagesCountMismatch):
            might_fail(True)

    def test_error_message_interpolation(self):
        """Test error message interpolation."""
        expected = 10
        found = 5
        error = PercentagesCountMismatch(expected, found)

        # Test that error contains expected values
        error_str = str(error)
        assert str(expected) in error_str
        assert str(found) in error_str
