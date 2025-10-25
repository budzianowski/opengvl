"""Prediction script producing model inferences + metrics.

Steps:
1. Instantiate data loader & model client via Hydra.
2. Sample N examples (FewShotInput) from loader.
3. For each example, call the shared prediction helper.
4. Persist JSONL outputs (one line per example) + aggregated metrics summary.
"""

import json
from pathlib import Path

import hydra
from dotenv import load_dotenv
from hydra.utils import instantiate
from loguru import logger
from omegaconf import DictConfig, OmegaConf
from tqdm import tqdm
from datetime import datetime

from opengvl.clients.base import BaseModelClient
from opengvl.data_loaders.base import BaseDataLoader
from opengvl.metrics.voc import VOCMetric
from opengvl.results.prediction import aggregate_metrics
from opengvl.utils import inference as infer_utils
from opengvl.mapper.base import BaseMapper


@hydra.main(version_base=None, config_path="../../configs", config_name="experiments/predict")
def main(config: DictConfig) -> None:
    """Main prediction script entry point."""
    infer_utils.validate_prediction_config(config)
    load_dotenv(override=True)
    logger.info("Environment variables loaded (dotenv)")
    logger.info(f"Configuration:\n{OmegaConf.to_yaml(config)}")

    data_loader: BaseDataLoader = instantiate(config.data_loader)
    client: BaseModelClient = instantiate(config.model)
    mapper: BaseMapper = instantiate(config.mapper)
    prompt_template: str = config.prompts.template
    
    logger.info(
        f"Instantiated components | dataset={config.dataset.name} loader={data_loader.__class__.__name__} "
        f"model={client.__class__.__name__} prompt_template_chars={len(prompt_template)}"
    )

    num_examples = int(config.prediction.num_examples)
    save_raw = bool(config.prediction.save_raw)
    output_dir = Path(str(config.prediction.output_dir))
    output_dir.mkdir(parents=True, exist_ok=True)
    model_name_safe = client.model_name.replace("/", "_")
    starting_time = datetime.now().strftime('%Y%m%d_%H%M%S')
    jsonl_path = output_dir / f"{model_name_safe}_{starting_time}_predictions.jsonl"

    examples = infer_utils.load_fewshot_examples(data_loader, num_examples, config.dataset.name)
    logger.info(f"Loaded {len(examples)} (in-context trajectories (0 or more) + eval trajectory) examples for prediction")
    if len(examples) == 0:
        logger.warning("No examples loaded; exiting")
        return
    voc_metric = VOCMetric()
    logger.debug(f"Metrics initialized: {voc_metric.name}")

    # Load prompt phrasing from dedicated config section (required; fall back to empty)
    prompt_phrases = dict(config.get("prompt_phrases", {})) if hasattr(config, "prompt_phrases") else {}
    logger.debug(f"Prompt phrases: {prompt_phrases}")
    records = [
        infer_utils.predict_on_fewshot_input(
            idx,
            num_examples,
            ex,
            client,
            prompt_template,
            save_raw,
            voc_metric,
            config.dataset.name,
            temperature=float(config.prediction.get("temperature", 1.0)),
            mapper=mapper,
            prompt_phrases=prompt_phrases,
        )
        for idx, ex in tqdm(enumerate(examples), total=num_examples, desc="Predicting")
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
    summary = dict()
    summary['model_name'] = client.model_name
    summary['dataset_name'] = config.dataset.name
    summary['prediction_time'] = starting_time
    summary['temperature'] = float(config.prediction.get("temperature", 1.0))
    summary['num_examples'] = len(records)
    summary['metrics'] = dataset_metrics.to_dict()

    with (output_dir / f"{model_name_safe}_{starting_time}_summary.json").open("w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)
    logger.info(f"Wrote {len(records)} records to {jsonl_path}")
    logger.info(f"Summary: {summary}")


if __name__ == "__main__":  # pragma: no cover
    # pylint: disable=no-value-for-parameter
    main()
