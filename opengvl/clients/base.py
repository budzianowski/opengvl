"""Core base model client abstraction and image utilities.

This module intentionally keeps a very small surface area and avoids pulling heavy
dependencies (transformers, torch, cloud SDKs) so importing it is cheap for all
downstream modules.
"""

from abc import ABC, abstractmethod
from time import sleep

from loguru import logger

from opengvl.utils.data_types import Episode
from opengvl.utils.errors import MaxRetriesExceeded
from opengvl.utils.rate_limiter import SECS_PER_MIN, RateLimiter

MAX_RETRIES = 4  # how many times to retry on rate limit errors


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
        # Persist a limiter instance so the rolling window spans calls.
        self._rate_limiter: RateLimiter | None = (
            RateLimiter(max_calls=self.rpm, period=SECS_PER_MIN) if self.rpm > 0.0 else None
        )

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
        for call_attempt in range(1, MAX_RETRIES + 1):
            logger.debug(f"Model generation attempt {call_attempt}/{MAX_RETRIES}")
            try:
                if self._rate_limiter is not None:
                    logger.info(f"Applying rate limit: {self.rpm} requests per minute")
                    with self._rate_limiter:
                        logger.info("Lock acquired, generating response...")
                        res = self._generate_response_impl(  # pyright: ignore[reportAttributeAccessIssue]
                            prompt, eval_episode, context_episodes
                        )

                else:
                    res = self._generate_response_impl(  # pyright: ignore[reportAttributeAccessIssue]
                        prompt, eval_episode, context_episodes
                    )
                logger.info(f"Model response length: {len(res)} characters")
            except Exception as e:
                logger.warning(f"Model generation attempt {call_attempt + 1} failed: {e}")
                timesleep = 2 ** (call_attempt + 2)
                logger.warning(f"Retrying after {timesleep} seconds...")
                sleep(timesleep)
                continue
            return res

        raise MaxRetriesExceeded(MAX_RETRIES)

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
