from typing import Any, Protocol, TypeAlias, runtime_checkable

import numpy as np
import numpy.typing as npt
from PIL import Image as PILImage

# Concrete image container aliases
ImagePIL: TypeAlias = PILImage.Image
ImageNumpyU8: TypeAlias = npt.NDArray[np.uint8]
ImageNumpyF32: TypeAlias = npt.NDArray[np.float32]
ImageNumpyF64: TypeAlias = npt.NDArray[np.float64]
ImageNumpy: TypeAlias = ImageNumpyU8 | ImageNumpyF32 | ImageNumpyF64


# Minimal torch-like tensor protocol (no hard torch dependency here)
@runtime_checkable
class TorchTensorLike(Protocol):
    def detach(self) -> "TorchTensorLike": ...

    def numpy(self) -> npt.NDArray[Any]: ...

    @property
    def is_cuda(self) -> bool: ...

    def cpu(self) -> "TorchTensorLike": ...


ImageTorch: TypeAlias = TorchTensorLike

# Polymorphic image type accepted across the codebase
ImageT: TypeAlias = ImagePIL | ImageNumpy | ImageTorch

# Base64-encoded PNG chars as produced by encode_image
EncodedImage: TypeAlias = bytes

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
