class ImageEncodingError(RuntimeError):
    """Raised when an image cannot be converted or encoded."""

    def __init__(self, message=None, **kwargs):
        if message is None:
            # Compose a default message from kwargs if available
            details = ", ".join(f"{k}={v}" for k, v in kwargs.items())
            message = f"Image encoding error. {details}" if details else "Image encoding error."
        super().__init__(message)
        self.details = kwargs
