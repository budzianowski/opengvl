"""Finetuning script using HF Transformers Trainer.

Steps:
1. Instantiate data loader via Hydra.
2. Sample train/val FewShotInput examples.
3. Build text-supervised datasets from examples.
4. Initialize model/tokenizer and run training with callbacks & W&B.
"""

from pathlib import Path

import hydra
from dotenv import load_dotenv
from hydra.utils import instantiate
from loguru import logger
from omegaconf import DictConfig, OmegaConf

from opengvl.data_loaders.base import BaseDataLoader
from opengvl.utils.training import (
    FinetuneHyperParams,
    FinetunePlan,
    FinetuneTrainer,
    TextSupervisedDataset,
    build_finetune_samples,
    validate_finetuning_config,
)


@hydra.main(version_base=None)
def main(config: DictConfig) -> None:
    validate_finetuning_config(config)
    load_dotenv(override=True)
    logger.info("Environment variables loaded (dotenv)")
    logger.info(f"Configuration:\n{OmegaConf.to_yaml(config)}")

    exit(0)

    # Components
    data_loader: BaseDataLoader = instantiate(config.data_loader)
    logger.info(f"Instantiated loader={data_loader.__class__.__name__} dataset={config.dataset.name}")

    # Prepare output dirs
    output_dir = Path(str(config.finetune.output_dir))
    output_dir.mkdir(parents=True, exist_ok=True)

    # Build datasets
    n_train = int(config.finetune.num_train_examples)
    n_val = int(config.finetune.num_val_examples)
    prompt_template: str = config.prompts.template

    logger.info(f"Sampling train={n_train} val={n_val} examples…")
    train_examples = data_loader.load_fewshot_inputs(n_train)
    val_examples = data_loader.load_fewshot_inputs(n_val) if n_val > 0 else []

    train_samples = build_finetune_samples(train_examples, prompt_template)
    val_samples = build_finetune_samples(val_examples, prompt_template) if val_examples else []

    plan = FinetunePlan(
        model_id=str(config.model.model_id),
        max_length=int(config.finetune.max_seq_len),
        output_dir=str(output_dir),
        wandb_project=str(config.finetune.wandb_project),
        wandb_run_name=str(config.finetune.wandb_run_name),
    )
    trainer = FinetuneTrainer(plan)

    train_ds = TextSupervisedDataset(train_samples, trainer.tokenizer, max_length=int(config.finetune.max_seq_len))
    eval_ds = TextSupervisedDataset(val_samples, trainer.tokenizer, max_length=int(config.finetune.max_seq_len)) if val_samples else None

    hparams = FinetuneHyperParams(
        num_epochs=int(config.finetune.num_epochs),
        batch_size=int(config.finetune.batch_size),
        learning_rate=float(config.finetune.lr),
        weight_decay=float(config.finetune.weight_decay),
        warmup_ratio=float(config.finetune.warmup_ratio),
        gradient_accumulation_steps=int(config.finetune.gradient_accumulation_steps),
        logging_steps=int(config.finetune.logging_steps),
        eval_steps=int(config.finetune.eval_steps),
        save_steps=int(config.finetune.save_steps),
        early_stopping_patience=int(config.finetune.early_stopping_patience),
        gradient_checkpointing=bool(config.finetune.gradient_checkpointing),
        bf16=bool(config.finetune.bf16),
    )

    logger.info("Launching training")
    artifacts = trainer.train(train_ds=train_ds, eval_ds=eval_ds, hparams=hparams)

    logger.success(f"Finetuning completed. Model saved to {artifacts.model_dir}")


if __name__ == "__main__":  # pragma: no cover
    # pylint: disable=no-value-for-parameter
    main()
