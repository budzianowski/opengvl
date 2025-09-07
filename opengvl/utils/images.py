# Utils for image processing


import base64
import io

from loguru import logger
import numpy as np
from PIL import Image

from opengvl.utils.aliases import EncodedImage, ImageNumpy, ImagePIL, ImageT, TorchTensorLike
from opengvl.utils.constants import IMG_SIZE
from opengvl.utils.errors import ImageEncodingError


def normalize_numpy(image: ImageNumpy) -> ImageNumpy:
    """Normalize float arrays in [0,1] to uint8.

    Leaves non-float dtypes unchanged.
    """
    if image.dtype in (np.float32, np.float64) and image.max() <= 1.0:
        image = (image * 255).astype(np.uint8)
    else:
        logger.warning(
            f'Image dtype is {image.dtype} with max {image.max()}. ' \
            'Expected float in [0,1] or uint8. Leaving unchanged.'
        )
    return image


def to_numpy(image: ImageT) -> ImageNumpy:
    """Best-effort conversion to numpy array.

    Supports PIL.Image, numpy arrays, and torch-like tensors implementing
    ``detach`` & ``numpy``. Raises ImageEncodingError otherwise.
    """
    if isinstance(image, np.ndarray):
        return image
    if isinstance(image, ImagePIL):
        return np.array(image)
    if isinstance(image, TorchTensorLike):
        # Torch tensor path; guard against CUDA placement.
        if getattr(image, "is_cuda", False):
            image = image.cpu()
        return image.detach().numpy()
    raise ImageEncodingError(image_type=type(image))


def to_pil(image: ImageT) -> ImagePIL:
    """Convert image-like input to a resized PIL image.

    Accepted input types: PIL.Image.Image, numpy.ndarray, torch.Tensor-like.
    Handles (C,H,W) -> (H,W,C) channel-first conversion. Supports grayscale
    and RGB images. Raises ImageEncodingError on unsupported shapes.
    """
    if isinstance(image, ImagePIL):
        # Fast path for already-PIL images (only resizes)
        return image.resize((IMG_SIZE, IMG_SIZE))

    np_img = normalize_numpy(to_numpy(image))

    # Channel-first -> channel-last
    if np_img.ndim == 3 and np_img.shape[0] in (1, 3, 4):
        np_img = np.transpose(np_img, (1, 2, 0))

    if np_img.ndim == 2:  # grayscale
        pil = Image.fromarray(np_img, "L")  # single-channel
    elif np_img.ndim == 3:  # multi-channel
        if np_img.shape[2] == 3:  # RGB
            pil = Image.fromarray(np_img, "RGB")
        elif np_img.shape[2] == 1:  # grayscale with channel dim
            pil = Image.fromarray(np_img.squeeze(axis=2), "L")
        else:  # alpha or >4 channels not supported for now
            raise ImageEncodingError(message=f"Unsupported channel count: {np_img.shape[2]}")
    else:
        raise ImageEncodingError(image_shape=np_img.shape)

    return pil.resize((IMG_SIZE, IMG_SIZE))


def encode_image(image: ImageT) -> EncodedImage:
    """Encode image to base64 PNG string.

    Args:
        image: Image-like object.
    Returns:
        Base64 text of PNG data.
    Raises:
        ImageEncodingError: On conversion failure.
    """
    try:
        pil_image = to_pil(image)
    except Exception as exc:
        raise ImageEncodingError(exc=exc) from exc

    buffer = io.BytesIO()
    pil_image.save(buffer, format="PNG")
    return base64.b64encode(buffer.getvalue()).decode("utf-8")
