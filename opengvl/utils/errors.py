class ImageEncodingError(RuntimeError):
    """Raised when an image cannot be converted or encoded."""

    def __init__(self, message=None, **kwargs):
        if message is None:
            # Compose a default message from kwargs if available
            details = ", ".join(f"{k}={v}" for k, v in kwargs.items())
            message = f"Image encoding error. {details}" if details else "Image encoding error."
        super().__init__(message)
        self.details = kwargs


class OriginalFramesLengthMismatch(Exception):
    def __init__(self, indices_len, rates_len):
        super().__init__(f"Lengths of original_frames_indices ({indices_len}) and original_frames_task_completion_rates ({rates_len}) must match")


class ShuffledFramesLengthMismatch(Exception):
    def __init__(self, indices_len, frames_len, approx_rates_len):
        super().__init__(
            f"shuffled_frames_indices ({indices_len}), shuffled_frames ({frames_len}), shuffled_frames_approx_completion_rates ({approx_rates_len}) must be 1:1"
        )


class ShuffledFramesIndicesNotSubset(Exception):
    def __init__(self):
        super().__init__("All shuffled_frames_indices must be present in original_frames_indices")


class PercentagesCountMismatch(Exception):
    """Raised when the number of extracted percentages doesn't match the expected length."""

    def __init__(self, expected: int, found: int):
        super().__init__(f"Expected {expected} percentages, found {found}")
        self.expected = expected
        self.found = found


class PercentagesNormalizationError(Exception):
    """Raised when percentages cannot be normalized to sum to 100."""

    def __init__(self, message: str | None = None):
        super().__init__(message or "Unable to normalize percentages (invalid sum)")


class MaxRetriesExceeded(Exception):
    """Raised when an operation fails after exhausting retry attempts."""

    def __init__(self, attempts: int):
        super().__init__(f"Max retries exceeded after {attempts} attempts")


class InputTooLongError(Exception):
    """Raised when model input exceeds provider/model limits."""

    def __init__(self, length: int, limit: int):
        super().__init__(f"Input length too large: {length} > {limit}")
        self.length = length
        self.limit = limit
