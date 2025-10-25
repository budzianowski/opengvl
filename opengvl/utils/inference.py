import json
import math
import re
from collections.abc import Iterable
from pathlib import Path

from loguru import logger
from omegaconf import DictConfig

from opengvl.clients.base import BaseModelClient
from opengvl.metrics.base import MetricResult
from opengvl.metrics.voc import VOCMetric
from opengvl.results.prediction import PredictionRecord
from opengvl.utils.constants import N_DEBUG_PROMPT_CHARS
from opengvl.utils.data_types import Example as FewShotInput
from opengvl.utils.data_types import InferredEpisode, InferredFewShotResult
from opengvl.utils.errors import PercentagesCountMismatch, PercentagesNormalizationError
from opengvl.utils.hydra import ensure_required_keys
from opengvl.utils.prompts import format_prompt
from opengvl.mapper.base import BaseMapper


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
    voc_metric: VOCMetric,
    dataset_name: str,
    temperature: float,
    mapper: BaseMapper,
    *,
    prompt_phrases: dict[str, str] | None = None,
) -> PredictionRecord:
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
            temperature=temperature,
            prompt_phrases=(prompt_phrases or {}),
        )
    except (RuntimeError, ValueError, OSError) as e:
        logger.error(f"Model generation failed for example {idx}: {e}")
        predicted: list[int] = []
        response_text = f"<error: {e}>"
    logger.debug(f"Response on example {idx}:\n{response_text}")

    expected_len = len(ex.eval_episode.shuffled_frames)
    error_count: dict[str, int] = {
        PercentagesCountMismatch.__name__: 0,
        PercentagesNormalizationError.__name__: 0,
    }

    try:
        predicted = mapper.extract_percentages(response_text)
        logger.success(f"Extracted {len(predicted)} percentages on example {idx}")
    except PercentagesNormalizationError as e:
        logger.error(f"Extraction error on example {idx}: {e}")
        predicted = []
        error_count[PercentagesNormalizationError.__name__] += 1

    if len(predicted) != expected_len:
        logger.error(
            f"Count mismatch on example {idx}: expected {expected_len}, "
            f"got {len(predicted)}"
        )
        error_count[PercentagesCountMismatch.__name__] += 1

    inferred: InferredFewShotResult = build_inferred_example(ex, predicted)

    if sum(error_count.values()) > 0:
        metric_res = MetricResult(name=voc_metric.name, value=0, details={
            "note": f"errors in prediction prevented metric computation {error_count!s}"
        })
    else:
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
        error_count=error_count,
    )
    logger.info(f"Example {idx}: preds={len(predicted)}/{len(ex.eval_episode.shuffled_frames)} VOC={metric_res.value}")
    return record
