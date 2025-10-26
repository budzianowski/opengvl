from __future__ import annotations

# pylint: disable=wrong-import-order,ungrouped-imports
from dataclasses import asdict, dataclass
from typing import Any

import numpy as np
import torch
import wandb
from loguru import logger
from omegaconf import DictConfig
from PIL import Image
from torch.utils.data import Dataset
from transformers import AutoProcessor, EarlyStoppingCallback, Trainer, TrainingArguments

from opengvl.utils.data_types import Episode, FewShotInput
from opengvl.utils.hydra import ensure_required_keys
from opengvl.utils.prompts import format_prompt


def _true_completion_for_shuffled(episode: Episode) -> list[int]:
    idx_to_rate: dict[int, int] = dict(
        zip(
            episode.original_frames_indices,
            episode.original_frames_task_completion_rates,
            strict=True,
        )
    )
    return [int(idx_to_rate[i]) for i in episode.shuffled_frames_indices]


@dataclass
class VLSample:
    prompt: str
    target: str
    images: list[np.ndarray]  # ImageNumpy alias; using np.ndarray here


def validate_finetuning_config(config: DictConfig) -> None:
    """Ensure required top-level keys are present for prediction runs.

    This mirrors the previous local _validate_config in the script.
    """
    for key in ("dataset", "data_loader", "model",
                "mapper", "prompts", "prompt_phrases",
                "mapping_prompts", "finetune", "seed",
                "shuffle"):
        ensure_required_keys(config, key)


def build_vl_samples(examples: list[FewShotInput], prompt_template: str) -> list[VLSample]:
    samples: list[VLSample] = []
    for ex in examples:
        prompt = format_prompt(prompt_template, instruction=ex.eval_episode.instruction).rstrip()
        truth = _true_completion_for_shuffled(ex.eval_episode)
        target = ", ".join(f"{p}%" for p in truth)
        samples.append(VLSample(prompt=prompt, target=target, images=ex.eval_episode.shuffled_frames))
    logger.info(f"Built {len(samples)} VL finetune samples")
    return samples


def _to_pil(img: np.ndarray) -> Image.Image:
    if isinstance(img, Image.Image):
        return img
    arr = np.clip(img, 0, 255).astype(np.uint8) if img.dtype != np.uint8 else img
    if arr.ndim == 2:
        return Image.fromarray(arr, mode="L")
    return Image.fromarray(arr)


class QwenVLSupervisedDataset(Dataset):
    """Dataset that builds Qwen VL chat-style inputs with images and masks labels before assistant output."""

    def __init__(self, samples: list[VLSample], processor: Any) -> None:
        self.samples = samples
        self.processor = processor

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> dict[str, Any]:
        s = self.samples[idx]
        pil_images = [_to_pil(im) for im in s.images]

        # Build a single-turn conversation: user provides prompt + N frames, assistant provides target string
        user_content: list[dict[str, str | Image.Image]] = [{"type": "text", "text": s.prompt}]
        for i, im in enumerate(pil_images, start=1):
            user_content.append({"type": "text", "text": f"\nFrame {i}:"})
            user_content.append({"type": "image", "image": im})

        user_only = [{"role": "user", "content": user_content}]
        messages = [*user_only, {"role": "assistant", "content": [{"type": "text", "text": s.target}]}]

        # Build text strings via chat template, then process images+text together
        full_text: str = self.processor.apply_chat_template(messages, add_generation_prompt=False, tokenize=False)
        user_text: str = self.processor.apply_chat_template(user_only, add_generation_prompt=False, tokenize=False)

        full_proc = self.processor(text=full_text, images=pil_images, return_tensors=None)
        user_proc = self.processor(text=user_text, images=pil_images, return_tensors=None)

        input_ids = full_proc["input_ids"]
        attention_mask = full_proc.get("attention_mask")
        pixel_values = full_proc.get("pixel_values")
        # pixel_values should be present; processor handles image normalization/resize

        prefix_len = len(user_proc["input_ids"])
        labels = [-100] * len(input_ids)
        for i in range(prefix_len, len(labels)):
            labels[i] = input_ids[i]

        return {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "pixel_values": pixel_values,
            "labels": labels,
        }


class QwenVLDataCollator:
    """Simple collator that pads input_ids/labels and stacks pixel_values."""

    def __init__(self, processor: Any) -> None:
        self.processor = processor
        self.pad_id = processor.tokenizer.pad_token_id or processor.tokenizer.eos_token_id

    def __call__(self, features: list[dict[str, Any]]) -> dict[str, Any]:
        max_len = max(len(f["input_ids"]) for f in features)
        batch_input_ids: list[list[int]] = []
        batch_attention_mask: list[list[int]] = []
        batch_labels: list[list[int]] = []
        pixel_values_list: list[torch.Tensor] = []

        for f in features:
            ids = f["input_ids"]
            attn = f.get("attention_mask") or [1] * len(ids)
            lbls = f["labels"]
            pad_len = max_len - len(ids)
            batch_input_ids.append(ids + [self.pad_id] * pad_len)
            batch_attention_mask.append(attn + [0] * pad_len)
            batch_labels.append(lbls + [-100] * pad_len)
            # pixel_values may be tensor per sample (C,H,W) or list; convert to tensor
            pv = f["pixel_values"]
            if not isinstance(pv, torch.Tensor):
                pv = torch.tensor(np.array(pv))  # type: ignore[arg-type]
            pixel_values_list.append(pv)

        batch = {
            "input_ids": torch.tensor(batch_input_ids, dtype=torch.long),
            "attention_mask": torch.tensor(batch_attention_mask, dtype=torch.long),
            "labels": torch.tensor(batch_labels, dtype=torch.long),
            "pixel_values": torch.stack(pixel_values_list, dim=0),
        }
        return batch


@dataclass
class VLFineTunePlan:
    model_id: str
    output_dir: str
    wandb_project: str | None = None
    wandb_run_name: str | None = None


@dataclass
class VLFineTuneHyperParams:
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
    lr_scheduler_type: str = "cosine"
    max_grad_norm: float = 1.0
    save_total_limit: int = 2


class QwenVLFinetuneTrainer:
    def __init__(self, plan: VLFineTunePlan) -> None:
        self.plan = plan
        logger.info(f"Loading Qwen VL processor and model: {plan.model_id}")
        self.processor = AutoProcessor.from_pretrained(plan.model_id, trust_remote_code=True)
        # Import model class lazily to avoid import if not used
        from transformers import Qwen2_5_VLForConditionalGeneration  # type: ignore[attr-defined]

        self.model = Qwen2_5_VLForConditionalGeneration.from_pretrained(plan.model_id, trust_remote_code=True)

    def train(
        self,
        train_ds: Dataset,
        eval_ds: Dataset | None,
        hparams: VLFineTuneHyperParams,
    ) -> dict[str, Any]:
        logger.info("Preparing data collator and training arguments (VL)")
        data_collator = QwenVLDataCollator(self.processor)

        if hparams.gradient_checkpointing and hasattr(self.model.config, "use_cache"):
            self.model.config.use_cache = False  # type: ignore[attr-defined]

        args = TrainingArguments(
            output_dir=self.plan.output_dir,
            num_train_epochs=hparams.num_epochs,
            per_device_train_batch_size=hparams.batch_size,
            per_device_eval_batch_size=hparams.batch_size,
            save_steps=hparams.save_steps,
            logging_steps=hparams.logging_steps,
            learning_rate=hparams.learning_rate,
            warmup_ratio=hparams.warmup_ratio,
            weight_decay=hparams.weight_decay,
            gradient_accumulation_steps=hparams.gradient_accumulation_steps,
            gradient_checkpointing=hparams.gradient_checkpointing,
            bf16=hparams.bf16,
            save_total_limit=hparams.save_total_limit,
            lr_scheduler_type=hparams.lr_scheduler_type,
            max_grad_norm=hparams.max_grad_norm,
            load_best_model_at_end=bool(eval_ds is not None),
            metric_for_best_model="eval_loss",
            greater_is_better=False,
            eval_strategy="epoch" if eval_ds is not None else "no",
            save_strategy="epoch",
            eval_steps=hparams.eval_steps if eval_ds is not None else None,
            report_to=["wandb"] if (self.plan.wandb_project and self.plan.wandb_run_name) else [],
        )

        callbacks = []
        if eval_ds is not None and hparams.early_stopping_patience > 0:
            callbacks.append(EarlyStoppingCallback(early_stopping_patience=hparams.early_stopping_patience))

        # Initialize W&B
        if self.plan.wandb_project and self.plan.wandb_run_name:
            cfg = {**asdict(self.plan), **asdict(hparams)}
            wandb.init(project=self.plan.wandb_project, name=self.plan.wandb_run_name, config=cfg)
            logger.info("Weights & Biases logging is enabled (VL)")

        trainer = Trainer(
            model=self.model,
            args=args,
            train_dataset=train_ds,
            eval_dataset=eval_ds,
            data_collator=data_collator,
            callbacks=callbacks,
        )

        logger.info("Starting VL training...")
        trainer.train()
        logger.success("VL Training finished")
        logger.info("Saving final model and processor")
        trainer.save_model(self.plan.output_dir)
        self.processor.save_pretrained(self.plan.output_dir)

        return {"model_dir": self.plan.output_dir, "trainer": trainer}
