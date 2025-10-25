from __future__ import annotations

from collections.abc import Iterable
from dataclasses import asdict, dataclass

import wandb
from loguru import logger
from omegaconf import DictConfig
from torch.utils.data import Dataset
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    DataCollatorForLanguageModeling,
    EarlyStoppingCallback,
    Trainer,
    TrainingArguments,
)

from opengvl.utils.data_types import Episode, Example
from opengvl.utils.hydra import ensure_required_keys
from opengvl.utils.prompts import format_prompt


def validate_finetuning_config(config: DictConfig) -> None:
    """Ensure required top-level keys are present for finetune runs.

    Required sections:
    - dataset (metadata like name)
    - data_loader (instantiation params)
    - model (contains model_id)
    - finetune (training hyperparameters)
    - prompts (template for input formatting)
    """
    for key in ("dataset", "data_loader", "model", "finetune", "prompts"):
        ensure_required_keys(config, key)


def _true_completion_for_shuffled(episode: Episode) -> list[int]:
    """Map original completion rates to shuffled frame order.

    Given Episode fields:
    - original_frames_indices (sorted) with original_frames_task_completion_rates aligned 1:1
    - shuffled_frames_indices specifying the order presented to the model
    returns per-shuffled-frame true completion percentages.
    """
    idx_to_rate: dict[int, int] = dict(
        zip(
            episode.original_frames_indices,
            episode.original_frames_task_completion_rates,
            strict=False,
        )
    )
    return [int(idx_to_rate[i]) for i in episode.shuffled_frames_indices]


@dataclass
class FinetuneSample:
    prompt: str
    target: str


def build_finetune_samples(examples: Iterable[Example], prompt_template: str) -> list[FinetuneSample]:
    """Construct text-to-text finetune pairs from FewShot inputs.

    Input prompt: formatted using the provided template and the eval instruction.
    Target: comma-separated percentages for frames in shuffled order, e.g. "0%, 10%, 20%, ...".
    """
    samples: list[FinetuneSample] = []
    for ex in examples:
        prompt = format_prompt(prompt_template, instruction=ex.eval_episode.instruction).rstrip()
        truth = _true_completion_for_shuffled(ex.eval_episode)
        target = ", ".join(f"{p}%" for p in truth)
        samples.append(FinetuneSample(prompt=prompt, target=target))
    logger.info(f"Built {len(samples)} finetune samples")
    return samples


class TextSupervisedDataset(Dataset):
    """Tokenized dataset for supervised fine-tuning of causal LMs.

    The loss is computed only on the target portion (after the special answer prefix).
    """

    def __init__(self, samples: list[FinetuneSample], tokenizer, *, max_length: int = 1024, answer_prefix: str = "\nAnswer: ") -> None:
        self.samples = samples
        self.tokenizer = tokenizer
        self.max_length = int(max_length)
        self.answer_prefix = answer_prefix

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> dict:
        s = self.samples[idx]
        # Compose full text: [prompt][answer_prefix][target]
        full = s.prompt + self.answer_prefix + s.target
        tokenized = self.tokenizer(
            full,
            truncation=True,
            max_length=self.max_length,
            padding=False,
            return_tensors=None,
        )
        # Build labels such that only target tokens contribute to loss
        # Identify prefix length in tokens
        with self.tokenizer.as_target_tokenizer():
            prefix_len = len(self.tokenizer(s.prompt + self.answer_prefix, add_special_tokens=False)["input_ids"])  # type: ignore[index]
        input_ids = tokenized["input_ids"]
        labels = [-100] * len(input_ids)
        for i in range(prefix_len, len(labels)):
            labels[i] = input_ids[i]
        tokenized["labels"] = labels
        return tokenized


@dataclass
class FinetuneArtifacts:
    model_dir: str
    trainer: Trainer


@dataclass
class FinetuneHyperParams:
    num_epochs: int
    batch_size: int
    learning_rate: float
    weight_decay: float
    warmup_ratio: float
    gradient_accumulation_steps: int
    logging_steps: int
    eval_steps: int
    save_steps: int
    early_stopping_patience: int
    gradient_checkpointing: bool
    bf16: bool


@dataclass
class FinetunePlan:
    model_id: str
    max_length: int
    output_dir: str
    wandb_project: str
    wandb_run_name: str


class FinetuneTrainer:
    """High-level finetuning orchestrator leveraging HF Transformers Trainer."""

    def __init__(self, plan: FinetunePlan) -> None:
        self.plan = plan
        logger.info(f"Loading tokenizer and model: {plan.model_id}")
        self.tokenizer = AutoTokenizer.from_pretrained(plan.model_id, use_fast=True)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        self.model = AutoModelForCausalLM.from_pretrained(plan.model_id)

    def train(self, train_ds: Dataset, eval_ds: Dataset | None, hparams: FinetuneHyperParams) -> FinetuneArtifacts:
        logger.info("Preparing data collator and training arguments")
        data_collator = DataCollatorForLanguageModeling(tokenizer=self.tokenizer, mlm=False)

        args_kwargs: dict[str, str | int | bool | float | list[str] | None] = {
            "output_dir": self.plan.output_dir,
            "num_train_epochs": hparams.num_epochs,
            "per_device_train_batch_size": hparams.batch_size,
            "per_device_eval_batch_size": hparams.batch_size,
            "save_steps": hparams.save_steps,
            "logging_steps": hparams.logging_steps,
            "learning_rate": hparams.learning_rate,
            "warmup_ratio": hparams.warmup_ratio,
            "weight_decay": hparams.weight_decay,
            "gradient_accumulation_steps": hparams.gradient_accumulation_steps,
            "gradient_checkpointing": hparams.gradient_checkpointing,
            "bf16": hparams.bf16,
            "report_to": ["wandb"],
            "save_total_limit": 2,
            "load_best_model_at_end": bool(eval_ds is not None),
            "metric_for_best_model": "eval_loss",
            "greater_is_better": False,
        }
        if eval_ds is not None:
            args_kwargs["evaluation_strategy"] = "steps"
            args_kwargs["eval_steps"] = hparams.eval_steps

        args = TrainingArguments(**args_kwargs)

        callbacks = []
        if eval_ds is not None and hparams.early_stopping_patience > 0:
            callbacks.append(EarlyStoppingCallback(early_stopping_patience=hparams.early_stopping_patience))

        # Initialize W&B with a concise config derived from dataclasses
        wb_cfg = {**asdict(self.plan), **asdict(hparams)}
        wandb.init(project=self.plan.wandb_project, name=self.plan.wandb_run_name, config=wb_cfg)
        logger.info("Weights & Biases logging is enabled")

        trainer = Trainer(
            model=self.model,
            args=args,
            train_dataset=train_ds,
            eval_dataset=eval_ds,
            data_collator=data_collator,
            callbacks=callbacks,
        )

        logger.info("Starting training…")
        trainer.train()
        logger.success("Training finished")
        logger.info("Saving final model and tokenizer")
        trainer.save_model(self.plan.output_dir)
        self.tokenizer.save_pretrained(self.plan.output_dir)

        return FinetuneArtifacts(model_dir=self.plan.output_dir, trainer=trainer)


