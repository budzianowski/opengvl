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

from opengvl.clients.base import BaseModelClient
from opengvl.data_loaders.base import BaseDataLoader
from opengvl.metrics.voc import VOCMetric
from opengvl.results.prediction import aggregate_metrics
from opengvl.utils import inference as infer_utils


@hydra.main(version_base=None, config_path="../../configs", config_name="experiments/predict")
def main(config: DictConfig) -> None:
    """Main prediction script entry point."""
    infer_utils.validate_prediction_config(config)
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

    examples = infer_utils.load_fewshot_examples(data_loader, num_examples, config.dataset.name)
    logger.info(
        f"Loaded {len(examples)} (in-context trajectories (0 or more) + eval trajectory) examples for prediction"
    )
    for ex in examples:
        print(ex)
    if len(examples) == 0:
        logger.warning("No examples loaded; exiting")
        return
    voc_metric = VOCMetric()
    logger.debug(f"Metrics initialized: {voc_metric.name}")

    records = [
        infer_utils.predict_on_fewshot_input(
            idx, num_examples, ex, client, prompt_template, save_raw, voc_metric, config.dataset.name
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
    with (output_dir / "summary.json").open("w", encoding="utf-8") as f:
        json.dump(dataset_metrics.to_dict(), f, indent=2)
    logger.info(f"Wrote {len(records)} records to {jsonl_path}")
    logger.info(f"Summary: {dataset_metrics.to_dict()}")


if __name__ == "__main__":  # pragma: no cover
    # pylint: disable=no-value-for-parameter
    main()
