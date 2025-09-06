"""Core base model client abstraction and image utilities.

This module intentionally keeps a very small surface area and avoids pulling heavy
dependencies (transformers, torch, cloud SDKs) so importing it is cheap for all
downstream modules.
"""


import base64
import io
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any, List

import numpy as np
from data_loader import Episode
from PIL import Image
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
