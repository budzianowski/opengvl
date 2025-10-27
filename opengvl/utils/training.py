from __future__ import annotations

# pylint: disable=wrong-import-order,ungrouped-imports
from dataclasses import asdict, dataclass
from typing import Any, cast

import numpy as np
import torch
from loguru import logger
from omegaconf import DictConfig
from qwen_vl_utils import process_vision_info
from torch.utils.data import Dataset
from transformers import AutoProcessor, EarlyStoppingCallback, Trainer, TrainingArguments

import wandb
from opengvl.utils.aliases import Event, ImageEvent, ImageT, TextEvent
from opengvl.utils.data_types import Episode, FewShotInput
from opengvl.utils.hydra import ensure_required_keys
from opengvl.utils.images import to_pil
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

    def to_events(self) -> list[Event]:
        """Convert this sample to a list of Events for consistent formatting."""
        events: list[Event] = [TextEvent(self.prompt)]
        for img in self.images:
            events.append(ImageEvent(img))
        return events


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


def _events_to_qwen_messages(events: list[Event]) -> list[dict[str, Any]]:
    """Convert Event list to Qwen message format, exactly like QwenClient does.
    
    This function replicates the logic from QwenClient._generate_from_events
    to ensure training data is formatted identically to inference.
    """
    messages = [{"role": "user", "content": []}]
    for ev in events:
        if isinstance(ev, TextEvent):
            if ev.text:  # Skip empty text events
                messages[0]["content"].append({"type": "text", "text": ev.text})
        elif isinstance(ev, ImageEvent):
            messages[0]["content"].append({"type": "image", "image": to_pil(cast(ImageT, ev.image))})
        else:
            logger.warning(f"Unknown event type: {type(ev)}")
    return messages


class QwenVLSupervisedDataset(Dataset):
    """Dataset that builds Qwen VL chat-style inputs with images and masks labels before assistant output."""

    def __init__(self, samples: list[VLSample], processor: Any) -> None:
        self.samples = samples
        self.processor = processor

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> dict[str, Any]:
        s = self.samples[idx]

        # Build events exactly like the base client does, then convert to Qwen messages
        user_events = s.to_events()
        user_messages = _events_to_qwen_messages(user_events)

        # Ensure user and full messages do not share mutable lists
        user_content = list(user_messages[0]["content"])
        user_messages = [{"role": "user", "content": user_content}]
        messages = [
            {"role": "user", "content": list(user_content)},
            {"role": "assistant", "content": [{"type": "text", "text": s.target}]},
        ]

        return {
            "user_messages": user_messages,
            "messages": messages,
        }


class QwenVLDataCollator:
    """Simple collator that pads input_ids/labels and stacks pixel_values."""

    def __init__(self, processor: Any) -> None:
        self.processor = processor
        self.pad_id = processor.tokenizer.pad_token_id or processor.tokenizer.eos_token_id

    def __call__(self, features: list[dict[str, Any]]) -> dict[str, Any]:
        messages_batch = [f["messages"] for f in features]
        user_messages_batch = [f["user_messages"] for f in features]

        full_texts = [
            self.processor.apply_chat_template(msg, add_generation_prompt=False, tokenize=False)
            for msg in messages_batch
        ]
        user_texts = [
            self.processor.apply_chat_template(msg, add_generation_prompt=False, tokenize=False)
            for msg in user_messages_batch
        ]

        image_inputs_batch: list[Any] = []
        video_inputs_batch: list[Any] = []
        for msg in messages_batch:
            image_inputs, video_inputs = process_vision_info(msg)  # type: ignore[arg-type]
            image_inputs_batch.append(image_inputs)
            video_inputs_batch.append(video_inputs if video_inputs else None)

        videos_arg = video_inputs_batch if any(v is not None for v in video_inputs_batch) else None

        full_proc = self.processor(
            text=full_texts,
            images=image_inputs_batch,
            videos=videos_arg,
            padding=True,
            return_tensors="pt",
        )

        user_proc = self.processor(
            text=user_texts,
            images=image_inputs_batch,
            videos=videos_arg,
            padding=True,
            return_tensors="pt",
        )

        input_ids: torch.Tensor = full_proc["input_ids"]
        attention_mask: torch.Tensor = full_proc["attention_mask"]
        labels: torch.Tensor = input_ids.clone()

        pad_token_id = self.pad_id if self.pad_id is not None else self.processor.tokenizer.eos_token_id
        for idx, user_ids in enumerate(user_proc["input_ids"]):
            valid_mask = user_ids != pad_token_id
            prefix_len = int(valid_mask.sum().item())
            if prefix_len > 0:
                labels[idx, :prefix_len] = -100

        labels = labels.masked_fill(attention_mask == 0, -100)

        batch: dict[str, torch.Tensor] = {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "labels": labels,
            "pixel_values": full_proc["pixel_values"],
        }

        optional_keys = (
            "image_grid_thw",
            "pixel_attention_mask",
            "video_grid_thw",
            "video_frame_mask",
        )
        for key in optional_keys:
            if key in full_proc:
                batch[key] = full_proc[key]

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
            remove_unused_columns=False,
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
