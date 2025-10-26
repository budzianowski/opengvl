"""Finetuning script using HF Transformers Trainer.

Steps:
1. Instantiate data loader via Hydra.
2. Sample train/val FewShotInput examples.
3. Build text-supervised datasets from examples.
4. Initialize model/tokenizer and run training with callbacks & W&B.
"""

from __future__ import annotations

import random

# pylint: disable=wrong-import-order,ungrouped-imports
from pathlib import Path

import hydra
import numpy as np
import torch
from dotenv import load_dotenv
from hydra.utils import instantiate
from loguru import logger
from omegaconf import DictConfig, OmegaConf

from opengvl.data_loaders.base import BaseDataLoader
from opengvl.utils.data_types import FewShotInput
from opengvl.utils.training import (
    QwenVLFinetuneTrainer,
    QwenVLSupervisedDataset,
    VLFineTuneHyperParams,
    VLFineTunePlan,
    build_vl_samples,
    validate_finetuning_config,
)


@hydra.main(version_base=None)
def main(config: DictConfig) -> None:
    validate_finetuning_config(config)
    load_dotenv(override=True)
    logger.info("Environment variables loaded (dotenv)")
    logger.info(f"Configuration:\n{OmegaConf.to_yaml(config)}")

    # Set seeds for reproducibility
    seed = int(getattr(config, "seed", 42))
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    # Components
    data_loader: BaseDataLoader = instantiate(config.data_loader)
    logger.info(f"Instantiated loader={data_loader.__class__.__name__} dataset={config.dataset.name}")

    # Prepare output dirs
    output_dir = Path(str(config.finetune.output_dir))
    output_dir.mkdir(parents=True, exist_ok=True)

    # Build datasets
    n_train = int(config.finetune.num_train_trajectories)
    n_val = int(config.finetune.num_val_trajectories)
    prompt_template: str = config.prompts.template

    logger.info(f"Sampling train={n_train} val={n_val} examples...")
    train_examples: list[FewShotInput] = data_loader.load_fewshot_inputs(n_train)
    val_examples: list[FewShotInput] = data_loader.load_fewshot_inputs(n_val) if n_val > 0 else []

    if any((len(ex.context_episodes) > 0) for ex in train_examples):
        raise ValueError("Finetuning with in-context examples is not supported. Please set dataset.num_context_episodes=0 in the config.")

    logger.info(f"Sampled train={len(train_examples)} val={len(val_examples)} trajectories")
    logger.info("Building finetune samples...")

    # VL-only training path
    wandb_project = getattr(config.finetune, "wandb_project", None)
    wandb_run_name = getattr(config.finetune, "wandb_run_name", None)
    model_identifier = str(config.model.get("model_id", config.model.get("model_name")))

    vl_plan = VLFineTunePlan(
        model_id=model_identifier,
        output_dir=str(output_dir),
        wandb_project=str(wandb_project) if wandb_project else None,
        wandb_run_name=str(wandb_run_name) if wandb_run_name else None,
    )
    vl_trainer = QwenVLFinetuneTrainer(vl_plan)
    train_vl_samples = build_vl_samples(train_examples, prompt_template)
    val_vl_samples = build_vl_samples(val_examples, prompt_template) if val_examples else []
    train_ds = QwenVLSupervisedDataset(train_vl_samples, vl_trainer.processor)
    eval_ds = QwenVLSupervisedDataset(val_vl_samples, vl_trainer.processor) if val_vl_samples else None

    vl_hparams = VLFineTuneHyperParams(
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
        lr_scheduler_type=str(getattr(config.finetune, "lr_scheduler_type", "cosine")),
        max_grad_norm=float(getattr(config.finetune, "max_grad_norm", 1.0)),
        save_total_limit=int(getattr(config.finetune, "save_total_limit", 2)),
    )

    logger.info("Launching VL training")
    artifacts = vl_trainer.train(train_ds=train_ds, eval_ds=eval_ds, hparams=vl_hparams)

    logger.success(f"Finetuning completed. Model saved to {artifacts['model_dir']}")


if __name__ == "__main__":  # pragma: no cover
    # pylint: disable=no-value-for-parameter
    main()
