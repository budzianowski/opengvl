"""Core base model client abstraction and image utilities.

This module intentionally keeps a very small surface area and avoids pulling heavy
dependencies (transformers, torch, cloud SDKs) so importing it is cheap for all
downstream modules.
"""

from abc import ABC, abstractmethod
from collections.abc import Iterator, Sequence
from time import sleep

from loguru import logger

from opengvl.utils.aliases import Event, ImageEvent, TextEvent
from opengvl.utils.constants import PromptPhraseKey
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
        *,
        prompt_phrases: dict[str, str],
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
        # Validate phrases early to provide a clear error if configuration is incomplete
        prompt_phrases = self._validate_and_normalize_prompt_phrases(prompt_phrases)

        for call_attempt in range(1, MAX_RETRIES + 1):
            logger.debug(f"Model generation attempt {call_attempt}/{MAX_RETRIES}")
            try:
                if self._rate_limiter is not None:
                    logger.info(f"Applying rate limit: {self.rpm} requests per minute")
                    with self._rate_limiter:
                        logger.info("Lock acquired, generating response...")
                        res = self._generate_response_impl(
                            prompt,
                            eval_episode,
                            context_episodes,
                            prompt_phrases=prompt_phrases,
                        )

                else:
                    res = self._generate_response_impl(
                        prompt,
                        eval_episode,
                        context_episodes,
                        prompt_phrases=prompt_phrases,
                    )
                logger.info(f"Model response length: {len(res)} characters")
            except (RuntimeError, ValueError, OSError) as e:
                logger.warning(f"Model generation attempt {call_attempt + 1} failed: {e}")
                timesleep = 2 ** (call_attempt + 2)
                logger.warning(f"Retrying after {timesleep} seconds...")
                sleep(timesleep)
                continue
            return res

        raise MaxRetriesExceeded(MAX_RETRIES)

    def _validate_and_normalize_prompt_phrases(self, phrases: dict[str, str]) -> dict[str, str]:
        """Ensure all required phrase keys are present and are strings.

        Returns a normalized dict that includes only the required keys.
        Raises ValueError if any required key is missing or not a string.
        Logs a debug message for any extra keys.
        """
        required_keys = [
            PromptPhraseKey.INITIAL_SCENE_LABEL,
            PromptPhraseKey.INITIAL_SCENE_COMPLETION,
            PromptPhraseKey.CONTEXT_FRAME_LABEL_TEMPLATE,
            PromptPhraseKey.CONTEXT_FRAME_COMPLETION_TEMPLATE,
            PromptPhraseKey.EVAL_FRAME_LABEL_TEMPLATE,
        ]
        missing: list[str] = []
        normalized: dict[str, str] = {}
        for k in required_keys:
            key = k.value
            if key not in phrases or not isinstance(phrases[key], str):
                missing.append(key)
            else:
                normalized[key] = phrases[key]

        if missing:
            raise ValueError("Missing or invalid prompt phrases for required keys: " + ", ".join(missing))

        # Log extra keys to help users diagnose config typos
        extras = [k for k in phrases if k not in normalized]
        if extras:
            logger.debug(f"Ignoring extra prompt phrase keys: {extras}")
        return normalized

    def _iter_prompt_events(
        self,
        prompt_text: str,
        eval_episode: Episode,
        context_episodes: Sequence[Episode],
        *,
        prompt_phrases: dict[str, str],
    ) -> Iterator[Event]:
        phrases = prompt_phrases
        # Instruction
        yield TextEvent(prompt_text)
        yield TextEvent(phrases[PromptPhraseKey.INITIAL_SCENE_LABEL.value])
        yield ImageEvent(eval_episode.starting_frame)
        yield TextEvent(phrases[PromptPhraseKey.INITIAL_SCENE_COMPLETION.value])

        # Context frames (with known completion)
        counter = 1
        for ctx_episode in context_episodes:
            for task_completion, frame in zip(
                ctx_episode.shuffled_frames_approx_completion_rates, ctx_episode.shuffled_frames
            ):
                yield TextEvent(phrases[PromptPhraseKey.CONTEXT_FRAME_LABEL_TEMPLATE.value].format(i=counter))
                yield ImageEvent(frame)
                yield TextEvent(
                    phrases[PromptPhraseKey.CONTEXT_FRAME_COMPLETION_TEMPLATE.value].format(p=task_completion)
                )
                counter += 1

        # Evaluation frames (no completion values)
        for frame in eval_episode.shuffled_frames:
            yield TextEvent(phrases[PromptPhraseKey.EVAL_FRAME_LABEL_TEMPLATE.value].format(i=counter))
            yield ImageEvent(frame)
            counter += 1

    def _generate_response_impl(
        self,
        prompt: str,
        eval_episode: Episode,
        context_episodes: list[Episode],
        *,
        prompt_phrases: dict[str, str],
    ) -> str:
        """Default implementation builds generic events and delegates to provider hook."""
        events = list(
            self._iter_prompt_events(
                prompt,
                eval_episode,
                context_episodes,
                prompt_phrases=prompt_phrases,
            )
        )
        return self._generate_from_events(events)

    @abstractmethod
    def _generate_from_events(self, events: list[Event]) -> str:  # pragma: no cover - interface only
        """Transform provider-agnostic prompt events into a model response."""
        raise NotImplementedError
