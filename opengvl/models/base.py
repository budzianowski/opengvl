"""Core base model client abstraction and image utilities.

This module intentionally keeps a very small surface area and avoids pulling heavy
dependencies (transformers, torch, cloud SDKs) so importing it is cheap for all
downstream modules.
"""

from __future__ import annotations

import base64
import io
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import List, Any

import numpy as np
from PIL import Image

from data_loader import Episode
from utils.constants import IMG_SIZE
from utils.errors import ImageEncodingError



class BaseModelClient(ABC):
    """Abstract base class for all model clients.

    Subclasses must implement ``generate_response``.
    They inherit image conversion & encoding helpers to produce standardized
    244x244 PNG base64 strings for multimodal APIs.
    """
    @abstractmethod
    def generate_response(
        self,
        prompt: str,
        eval_episode: Episode,
        context_episodes: List[Episode],
    ) -> str:  # pragma: no cover - interface only
        """Generate a textual response for a given evaluation episode.

        Args:
            prompt: Base natural language instruction or system prompt.
            eval_episode: Episode whose frames require prediction.
            context_episodes: Few-shot context episodes with known progress.
        Returns:
            The raw model textual output.
        """
        raise NotImplementedError

    @staticmethod
    def _normalize_numpy(image: np.ndarray) -> np.ndarray:
        """Normalize float arrays in [0,1] to uint8.

        Leaves non-float dtypes unchanged.
        """
        if image.dtype in (np.float32, np.float64) and image.max() <= 1.0:
            image = (image * 255).astype(np.uint8)
        return image

    @staticmethod
    def _to_numpy(image: Any) -> np.ndarray:
        """Best-effort conversion to numpy array.

        Supports PIL.Image, numpy arrays, and torch-like tensors implementing
        ``detach`` & ``numpy``. Raises ImageEncodingError otherwise.
        """
        if isinstance(image, np.ndarray):
            return image
        if isinstance(image, Image.Image):
            return np.array(image)
        if hasattr(image, "detach") and hasattr(image, "numpy"):
            # Torch tensor path; guard against CUDA placement.
            if getattr(image, "is_cuda", False):
                image = image.cpu()
            return image.detach().numpy()
        raise ImageEncodingError(f"Unsupported image type: {type(image)}")

    @classmethod
    def to_pil(cls, image: Any) -> Image.Image:
        """Convert image-like input to a resized PIL image.

        Accepted input types: PIL.Image.Image, numpy.ndarray, torch.Tensor-like.
        Handles (C,H,W) -> (H,W,C) channel-first conversion. Supports grayscale
        and RGB images. Raises ImageEncodingError on unsupported shapes.
        """
        if isinstance(image, Image.Image):
            # Fast path for already-PIL images (only resizes)
            return image.resize((IMG_SIZE, IMG_SIZE))

        np_img = cls._normalize_numpy(cls._to_numpy(image))

        # Channel-first -> channel-last
        if np_img.ndim == 3 and np_img.shape[0] in (1, 3, 4):
            np_img = np.transpose(np_img, (1, 2, 0))

        if np_img.ndim == 2:  # grayscale
            pil = Image.fromarray(np_img, "L")
        elif np_img.ndim == 3:
            if np_img.shape[2] == 3:
                pil = Image.fromarray(np_img, "RGB")
            elif np_img.shape[2] == 1:
                pil = Image.fromarray(np_img.squeeze(axis=2), "L")
            else:  # alpha or >4 channels not supported for now
                raise ImageEncodingError(f"Unsupported channel count: {np_img.shape[2]}")
        else:
            raise ImageEncodingError(f"Unsupported image shape: {np_img.shape}")

        return pil.resize((IMG_SIZE, IMG_SIZE))

    @classmethod
    def encode_image(cls, image: Any) -> str:
        """Encode image to base64 PNG string.

        Args:
            image: Image-like object.
        Returns:
            Base64 text of PNG data.
        Raises:
            ImageEncodingError: On conversion failure.
        """
        try:
            pil_image = cls.to_pil(image)
        except Exception as exc:  # noqa: BLE001 - broad to re-wrap clearly
            raise ImageEncodingError(f"Failed to prepare image: {exc}") from exc

        buffer = io.BytesIO()
        pil_image.save(buffer, format="PNG")
        return base64.b64encode(buffer.getvalue()).decode("utf-8")


