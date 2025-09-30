from dataclasses import dataclass
from typing import Any, Protocol, Union, runtime_checkable

import numpy as np
import numpy.typing as npt
from PIL import Image as PILImage

# ruff: noqa: UP007 (Allow use of typing.Union for Python 3.8 compatibility in test environment)


# Concrete image container aliases (runtime aliases without TypeAlias for wider compatibility)
ImagePIL = PILImage.Image
ImageNumpyU8 = npt.NDArray[np.uint8]
ImageNumpyF32 = npt.NDArray[np.float32]
ImageNumpyF64 = npt.NDArray[np.float64]
ImageNumpy = Union[ImageNumpyU8, ImageNumpyF32, ImageNumpyF64]


# Minimal torch-like tensor protocol (no hard torch dependency here)
@runtime_checkable
class TorchTensorLike(Protocol):
    def detach(self) -> "TorchTensorLike": ...

    def numpy(self) -> npt.NDArray[Any]: ...

    @property
    def is_cuda(self) -> bool: ...

    def cpu(self) -> "TorchTensorLike": ...


ImageTorch = TorchTensorLike

# Polymorphic image type accepted across the codebase
ImageT = Union[ImagePIL, ImageNumpyU8, ImageNumpyF32, ImageNumpyF64, ImageTorch]

# Base64-encoded PNG chars as produced by encode_image
EncodedImage = bytes


@dataclass(frozen=True)
class Event:
    """Marker base for prompt events."""


@dataclass(frozen=True)
class TextEvent(Event):
    text: str


@dataclass(frozen=True)
class ImageEvent(Event):
    image: ImageT


__all__ = [
    "EncodedImage",
    "ImageNumpy",
    "ImageNumpyF32",
    "ImageNumpyF64",
    "ImageNumpyU8",
    "ImagePIL",
    "ImageT",
    "ImageTorch",
    "TorchTensorLike",
]
