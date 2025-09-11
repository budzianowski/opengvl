"""Core base model client abstraction and image utilities.

This module intentionally keeps a very small surface area and avoids pulling heavy
dependencies (transformers, torch, cloud SDKs) so importing it is cheap for all
downstream modules.
"""

from abc import ABC, abstractmethod

from loguru import logger

from opengvl.data_loader import Episode
from opengvl.utils.rate_limiter import SECS_PER_MIN, RateLimiter


class BaseModelClient(ABC):
    """Abstract base class for all model clients.

    Subclasses must implement ``generate_response``.
    They inherit image conversion & encoding helpers to produce standardized
    244x244 PNG base64 strings for multimodal APIs.
    """

    def __init__(self, *, rpm: float = 0.0) -> None:
        """Initialize the base model client.

        Args:
            rpm: Requests per minute rate limit (0.0 for no limit).
        """
        self.rpm = float(rpm)

    def generate_response(
        self,
        prompt: str,
        eval_episode: Episode,
        context_episodes: list[Episode],
    ) -> str:
        """Generate a textual response for a given evaluation episode.

        This is the main entry point for generating model predictions.
        It wraps the subclass-specific implementation with rate limiting.

        Args:
            prompt: Base natural language instruction or system prompt.
            eval_episode: Episode whose frames require prediction.
            context_episodes: Few-shot context episodes with known progress.
        Returns:
            The raw model textual output.
        """
        if self.rpm > 0.0:
            logger.info(f"Applying rate limit: {self.rpm} requests per minute")
            with RateLimiter(max_calls=self.rpm, period=SECS_PER_MIN):
                logger.info("Lock acquired, generating response...")
                res = self._generate_response_impl(prompt, eval_episode, context_episodes)

        else:
            res = self._generate_response_impl(prompt, eval_episode, context_episodes)
        logger.info(f"Model response length: {len(res)} characters")
        return res

    @abstractmethod
    def _generate_response_impl(
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
