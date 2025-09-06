"""Core base model client abstraction and image utilities.

This module intentionally keeps a very small surface area and avoids pulling heavy
dependencies (transformers, torch, cloud SDKs) so importing it is cheap for all
downstream modules.
"""

from __future__ import annotations

from abc import ABC, abstractmethod

from opengvl.data_loader import Episode


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
        context_episodes: list[Episode],
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
