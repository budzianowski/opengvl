"""Prediction script producing model inferences + metrics.

Steps:
1. Instantiate data loader & model client via Hydra.
2. Sample N examples (FewShotInput) from loader.
3. For each example:
   - Build prompt from template.
   - Call model client.
   - Extract percentages via regex.
   - Build InferredEpisode / InferredExample structures.
   - Compute metrics (VOC, etc.).
4. Persist JSONL outputs (one line per example) + aggregated metrics summary.
"""

import json
from pathlib import Path

import hydra
from dotenv import load_dotenv
from hydra.utils import instantiate
from loguru import logger
from omegaconf import DictConfig, OmegaConf

from opengvl.clients.base import BaseModelClient
from opengvl.data_loaders.base import BaseDataLoader
from opengvl.metrics.voc import VOCMetric
from opengvl.results.prediction import PredictionRecord, aggregate_metrics
from opengvl.utils import inference as infer_utils
from opengvl.utils.data_types import Example as FewShotInput
from opengvl.utils.hydra import ensure_required_keys
from opengvl.utils.prompts import format_prompt


def _validate_config(config: DictConfig) -> None:
    for key in ("dataset", "data_loader", "model", "prompts", "prediction"):
        ensure_required_keys(config, key)


def _load_examples(loader: BaseDataLoader, n: int, dataset_name: str) -> list[FewShotInput]:
    logger.info(f"Generating {n} examples…")
    examples: list[FewShotInput] = [loader.load_fewshot_input() for _ in range(n)]
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
) -> PredictionRecord:
    logger.info(
        f"Processing example {idx + 1}/{total} (episode_index={ex.eval_episode.episode_index}) from {dataset_name}"
    )
    prompt = format_prompt(prompt_template, instruction=ex.eval_episode.instruction)
    logger.debug(f"Prompt (truncated 200 chars): {prompt[:200]}")
    try:
        response_text = client.generate_response(prompt, ex.eval_episode, ex.context_episodes)
    except (RuntimeError, ValueError, OSError) as e:
        logger.error(f"Model generation failed for example {idx}: {e}")
        predicted: list[int] = []
        response_text = f"<error: {e}>"
    else:
        expected_len = len(ex.eval_episode.shuffled_frames)
        predicted = infer_utils.extract_percentages(response_text, expected=expected_len)
        if not predicted:
            logger.warning(f"No percentages extracted for example {idx}")
        elif len(predicted) != expected_len:
            logger.warning(f"Length mismatch example {idx}: predicted={len(predicted)} expected={expected_len}")
        else:
            logger.success(f"Extracted {len(predicted)} percentages (expected) for example {idx}")
        logger.debug(f"Predictions: {predicted}")
        if save_raw:
            logger.debug(f"Raw response length: {len(response_text)} chars")

    inferred = infer_utils.build_inferred_example(ex, predicted)
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


@hydra.main(version_base=None, config_path="../../configs", config_name="experiments/predict")
def main(config: DictConfig) -> None:
    """Main prediction script entry point."""
    _validate_config(config)
    load_dotenv(override=True)
    logger.info("Environment variables loaded (dotenv)")
    logger.info(f"Configuration:\n{OmegaConf.to_yaml(config)}")

    data_loader: BaseDataLoader = instantiate(config.data_loader)
    client: BaseModelClient = instantiate(config.model)
    prompt_template: str = config.prompts.template
    logger.info(
        f"Instantiated components | dataset={config.dataset.name} loader={data_loader.__class__.__name__} "
        f"model={client.__class__.__name__} prompt_template_chars={len(prompt_template)}"
    )

    num_examples = int(config.prediction.num_examples)
    save_raw = bool(config.prediction.save_raw)
    output_dir = Path(str(config.prediction.output_dir))
    output_dir.mkdir(parents=True, exist_ok=True)
    jsonl_path = output_dir / "predictions.jsonl"

    examples = _load_examples(data_loader, num_examples, config.dataset.name)
    voc_metric = VOCMetric()
    logger.debug(f"Metrics initialized: {voc_metric.name}")

    records = [
        predict_on_fewshot_input(
            idx, num_examples, ex, client, prompt_template, save_raw, voc_metric, config.dataset.name
        )
        for idx, ex in enumerate(examples)
    ]

    logger.info(f"Serializing {len(records)} prediction records to {jsonl_path}")
    jsonl_payload_iter = (r.to_dict(include_images=False) for r in records)
    infer_utils.save_jsonl(jsonl_payload_iter, jsonl_path)
    dataset_metrics = aggregate_metrics(records)
    logger.success(
        f"Aggregate metrics: total={dataset_metrics.total_examples} valid={dataset_metrics.valid_predictions} "
        f"ratio={(dataset_metrics.length_valid_ratio if dataset_metrics.length_valid_ratio is not None else 0.0):.2f} "
        f"voc_mean={dataset_metrics.metric_means.get('voc', float('nan')):.4f}"
    )
    with (output_dir / "summary.json").open("w", encoding="utf-8") as f:
        json.dump(dataset_metrics.to_dict(), f, indent=2)
    logger.info(f"Wrote {len(records)} records to {jsonl_path}")
    logger.info(f"Summary: {dataset_metrics.to_dict()}")


if __name__ == "__main__":  # pragma: no cover
    # pylint: disable=no-value-for-parameter
    main()
