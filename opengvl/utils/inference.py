import json
import math
import re
from collections.abc import Iterable
from pathlib import Path

from loguru import logger
from omegaconf import DictConfig

from opengvl.clients.base import BaseModelClient
from opengvl.results.prediction import PredictionRecord
from opengvl.utils.constants import N_DEBUG_PROMPT_CHARS
from opengvl.utils.data_types import Example as FewShotInput
from opengvl.utils.data_types import InferredEpisode, InferredFewShotResult
from opengvl.utils.errors import (PercentagesCountMismatch,
                                  PercentagesNormalizationError)
from opengvl.utils.hydra import ensure_required_keys
from opengvl.utils.prompts import format_prompt

PERCENT_FLOAT_RE = re.compile(r"(-?\d+(?:\.\d+)?)\s*%")


def extract_percentages(
    text: str,
    expected: int,
) -> list[int]:
    """Extract percentages in order of appearance and return integers.

    - Accepts both integer and floating-point percentages in the input text.
    - If any extracted value has a fractional part, round the list so that
      the final integers sum to 100 using the largest remainder method.
    - For purely integer inputs, values are returned as-is (cast to int).

    Args:
        text: Source text.
        expected: Expected number of percentages. The function
            will validate that exactly this many values are present. If the
            count does not match, a ValueError is raised. Extraction does not
            truncate; all percentages found are considered.
    Returns:
        List of integer percentages within [0, 100].
    """

    vals: list[float] = []
    for match in PERCENT_FLOAT_RE.finditer(text):
        try:
            v = float(match.group(1))
        except ValueError:
            continue
        if not (0.0 <= v <= 100.0):
            continue
        vals.append(v)

    # If no values found, return empty list
    if not vals:
        return []

    # Enforce expected length if provided
    if expected is not None and len(vals) != expected:
        raise PercentagesCountMismatch(expected, len(vals))

    has_fractional = any((v % 1) != 0 for v in vals)
    if not has_fractional:
        # All values are already integers; just cast
        return [int(v) for v in vals]

    total = sum(vals)
    if total <= 0:
        # Degenerate case; cannot normalize meaningfully
        raise PercentagesNormalizationError()

    # Normalize to sum to 100, then distribute remainders
    scale = 100.0 / total
    scaled = [v * scale for v in vals]
    floors = [math.floor(x) for x in scaled]
    remainders = [x - f for x, f in zip(scaled, floors, strict=False)]
    current_sum = sum(floors)
    need = int(100 - current_sum)

    # Indices sorted by largest remainder (stable by original index for ties)
    order = sorted(range(len(vals)), key=lambda i: (-remainders[i], i))
    result = floors[:]
    for i in range(min(max(need, 0), len(result))):
        result[order[i]] += 1

    return list(map(int, result))


def build_inferred_example(
    fewshot: FewShotInput,
    predicted: list[int],
) -> InferredFewShotResult:
    inferred_ep = InferredEpisode.from_predictions(fewshot.eval_episode, predictions=predicted)
    return InferredFewShotResult(eval_episode=inferred_ep, context_episodes=fewshot.context_episodes)


def save_jsonl(records: Iterable[dict], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for rec in records:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")


def validate_prediction_config(config: DictConfig) -> None:
    """Ensure required top-level keys are present for prediction runs.

    This mirrors the previous local _validate_config in the script.
    """
    for key in ("dataset", "data_loader", "model", "prompts", "prediction"):
        ensure_required_keys(config, key)


def load_fewshot_examples(loader, n: int, dataset_name: str) -> list[FewShotInput]:
    """Load N few-shot inputs from a data loader with logging.

    Args:
        loader: Instance of BaseDataLoader.
        n: Number of examples to load.
        dataset_name: Human-friendly dataset identifier for logs.
    Returns:
        List of FewShotInput objects.
    """
    logger.info(f"Generating {n} examples…")
    examples: list[FewShotInput] = []
    for i in range(n):
        logger.info(f"Loading example {i + 1}/{n}")
        ex = loader.load_fewshot_input()
        examples.append(ex)
    logger.success(f"Loaded {len(examples)} few-shot examples from dataset '{dataset_name}'")
    return examples


def predict_on_fewshot_input(
    idx: int,
    total: int,
    ex: FewShotInput,
    client: BaseModelClient,
    prompt_template: str,
    save_raw: bool,
    voc_metric,
    dataset_name: str,
    *,
    prompt_phrases: dict[str, str] | None = None,
):
    """Run model prediction and metric computation on a single few-shot input.

    The logic mirrors the original script function without changes.
    """
    logger.info(f"Processing example {idx + 1}/{total} (episode_index={ex.eval_episode.episode_index}) from {dataset_name}")
    prompt = format_prompt(prompt_template, instruction=ex.eval_episode.instruction)
    logger.debug(f"Prompt (truncated {N_DEBUG_PROMPT_CHARS} chars): {prompt[:N_DEBUG_PROMPT_CHARS]}...")
    try:
        response_text = client.generate_response(
            prompt,
            ex.eval_episode,
            ex.context_episodes,
            prompt_phrases=(prompt_phrases or {}),
        )
    except (RuntimeError, ValueError, OSError) as e:
        logger.error(f"Model generation failed for example {idx}: {e}")
        predicted: list[int] = []
        response_text = f"<error: {e}>"
    logger.debug(f"Response on example {idx}:\n{response_text}")

    expected_len = len(ex.eval_episode.shuffled_frames)
    try:
        predicted = extract_percentages(response_text, expected=expected_len)
        logger.success(f"Extracted {len(predicted)} percentages on example {idx}")
    except (PercentagesCountMismatch, PercentagesNormalizationError) as e:
        logger.error(f"Extraction error on example {idx}: {e}")
        raise

    inferred: InferredFewShotResult = build_inferred_example(ex, predicted)
    metric_res = voc_metric.compute(inferred)
    metrics_payload = {metric_res.name: metric_res.value}

    if metric_res.details:
        for k, v in metric_res.details.items():
            metrics_payload[f"{metric_res.name}_{k}"] = v

    logger.debug(
        f"Metrics example {idx}: {metric_res.name}="
        f"{(metric_res.value if metric_res.value is not None else float('nan')):.4f}"
        f"{(' details=' + str(metric_res.details)) if metric_res.details else ''}"
    )

    record = PredictionRecord(
        index=idx,
        dataset=dataset_name,
        example=inferred,
        predicted_percentages=predicted,
        valid_length=len(predicted) == len(ex.eval_episode.shuffled_frames),
        metrics=metrics_payload,
        raw_response=response_text if save_raw else None,
    )
    logger.info(f"Example {idx}: preds={len(predicted)}/{len(ex.eval_episode.shuffled_frames)} VOC={metric_res.value}")
    return record
