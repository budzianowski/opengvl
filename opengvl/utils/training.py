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
from typing import Sequence, Iterator
from opengvl.utils.constants import PromptPhraseKey

import wandb
from opengvl.utils.aliases import Event, ImageEvent, ImageT, TextEvent
from opengvl.utils.data_types import Episode, FewShotInput
from opengvl.utils.hydra import ensure_required_keys
from opengvl.utils.images import to_pil
from opengvl.utils.prompts import format_prompt


def _true_completion_for_shuffled(episode: Episode) -> list[int]:
    """Get the true completion rates in the order of the shuffled frames.
    
    argument:
        episode: Episode object containing original and shuffled frame info.
        
    returns:
        List of true completion rates aligned with shuffled_frames_indices.
        
    example:
        original_frames_indices = [1, 2, 3]
        original_frames_task_completion_rates = [10, 20, 30]
        shuffled_frames_indices = [3, 1, 2]
        returns [30, 10, 20]
    """
    idx_to_rate: dict[int, int] = dict(
        zip(
            episode.original_frames_indices,
            episode.original_frames_task_completion_rates,
            strict=True,
        )
    )
    return [int(idx_to_rate[i]) for i in episode.shuffled_frames_indices]


def iter_prompt_events(
    prompt_text: str,
    eval_episode: Episode,
    context_episodes: Sequence[Episode],
    *,
    prompt_phrases: dict[str, str],
) -> Iterator[Event]:
    phrases = prompt_phrases
    # Instruction
    yield TextEvent(prompt_text)
    yield TextEvent(phrases[PromptPhraseKey.INITIAL_SCENE_LABEL.value])
    yield ImageEvent(eval_episode.starting_frame)
    yield TextEvent(phrases[PromptPhraseKey.INITIAL_SCENE_COMPLETION.value])

    # Context frames (with known completion)
    counter = 1
    for ctx_episode in context_episodes:
        for task_completion, frame in zip(ctx_episode.shuffled_frames_approx_completion_rates, ctx_episode.shuffled_frames, strict=False):
            yield TextEvent(phrases[PromptPhraseKey.CONTEXT_FRAME_LABEL_TEMPLATE.value].format(i=counter))
            yield ImageEvent(frame)
            yield TextEvent(phrases[PromptPhraseKey.CONTEXT_FRAME_COMPLETION_TEMPLATE.value].format(p=task_completion))
            counter += 1

    for instruction_str in phrases[PromptPhraseKey.EVAL_TASK_COMPLETION_INSTRUCTION.value]:
        yield TextEvent(instruction_str.format(instruction=eval_episode.instruction))

    for frame in eval_episode.shuffled_frames:
        yield TextEvent(phrases[PromptPhraseKey.EVAL_FRAME_LABEL_TEMPLATE.value].format(i=counter))
        yield ImageEvent(frame)
        yield TextEvent("")
        counter += 1


@dataclass
class VLSample:
    prompt: str
    target: str
    images: list[np.ndarray]  # ImageNumpy alias; using np.ndarray here

    def to_events(self) -> list[Event]:
        """Convert this sample to a list of Events for consistent formatting.
        
        example:
            >>> v = VLSample(prompt="Describe the images.", target="A cat and a dog.", images=[img1, img2])
            >>> events = v.to_events()
            >>> print(events)
            [TextEvent(text='Describe the images.'), ImageEvent(image=img1), ImageEvent(image=img2)]
        """

        events: list[Event] = [TextEvent(self.prompt)]
        for img in self.images:
            events.append(ImageEvent(img))
        return events
    
@dataclass
class TrainingSample:
    prompt_events: list[Event]
    target: str


def validate_finetuning_config(config: DictConfig) -> None:
    """Ensure required top-level keys are present for prediction runs.

    This mirrors the previous local _validate_config in the script.
    """
    for key in ("dataset", "data_loader", "model",
                "mapper", "prompts", "prompt_phrases",
                "mapping_prompts", "finetune", "seed",
                "shuffle"):
        ensure_required_keys(config, key)


def build_vl_samples(examples: list[FewShotInput], prompt_template: str, prompt_phrases: dict[str, str]) -> list[TrainingSample]:
    """Build VL finetuning samples from FewShotInput examples and a prompt template.
    
    arguments:
        examples: List of FewShotInput examples containing eval episodes.
        prompt_template: Template string for formatting prompts.
        
    returns:
        List of TrainingSample objects ready for finetuning.
        
    example
        >>> examples = [FewShotInput(eval_episode=episode1), FewShotInput(eval_episode=episode2)]
        >>> prompt_template = "Please analyze the following instruction: {instruction}"
        >>> samples = build_vl_samples(examples, prompt_template)
        >>> print(samples)
        [TrainingSample(prompt_events=[...], target='...'), ...]
    """
    training_samples = []
    for ex in examples:
        messages = []
        messages.extend(
            iter_prompt_events(
                prompt_text=format_prompt(prompt_template, instruction=ex.eval_episode.instruction).rstrip(),
                eval_episode=ex.eval_episode,
                context_episodes=ex.context_episodes,
                prompt_phrases=prompt_phrases,
            )
        )
        truth = _true_completion_for_shuffled(ex.eval_episode)
        target = "\n".join(f"Frame {i}: {p}%" for i, p in enumerate(truth))
        training_samples.append(TrainingSample(prompt_events=messages, target=target))

    return training_samples


def _events_to_qwen_messages(events: list[Event]) -> list[dict[str, Any]]:
    """Convert Event list to Qwen message format, exactly like QwenClient does.
    
    This function replicates the logic from QwenClient._generate_from_events
    to ensure training data is formatted identically to inference.

    example:
        >>> events = [TextEvent("Hello"), ImageEvent(image_array), ImageEvent(image_array2)]
        >>> messages = _events_to_qwen_messages(events)
        >>> print(messages)
        [{'role': 'user', 'content': [{'type': 'text', 'text': 'Hello'},
                                     {'type': 'image', 'image': <PIL.Image.Image image mode=RGB size=...>},
                                     {'type': 'image', 'image': <PIL.Image.Image image mode=RGB size=...>}]}]
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

    def __init__(self, samples: list[TrainingSample], processor: Any) -> None:
        self.samples = samples
        self.processor = processor

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> dict[str, Any]:
        """
        flow:
            >>> s = self.samples[idx]
            >>> print(s.prompt)
            "Describe the following images.""
        """
        s = self.samples[idx]

        # Build events exactly like the base client does, then convert to Qwen messages
        user_events = s.prompt_events
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
    """
    Collator that:
      - formats with chat template,
      - packs images/videos through the same AutoProcessor,
      - masks labels to only supervise assistant content (+ "<|im_end|>\\n"),
      - supports an explicit max_length (recommended).
    """

    def __init__(self, processor: Any) -> None:
        self.processor = processor

    def __call__(self, features: list[dict[str, Any]]) -> dict[str, torch.Tensor]:
        messages_batch = [f["messages"] for f in features]

        # 1) Template -> text
        full_texts = [
            self.processor.apply_chat_template(
                m, add_generation_prompt=False, tokenize=False
            )
            for m in messages_batch
        ]

        # 2) Vision payloads (per-sample lists)
        image_inputs_batch: list[list[Any]] = []
        for msg in messages_batch:
            imgs, _ = process_vision_info(msg)  # returns lists
            image_inputs_batch.append(imgs)

        # 3) Tokenize / pack
        proc = self.processor(
            text=full_texts,
            images=image_inputs_batch,
            padding=True,
            return_tensors="pt",
        )

        input_ids: torch.Tensor = proc["input_ids"]
        attention_mask: torch.Tensor = proc["attention_mask"]

        # 4) Build labels via span search
        labels = torch.full_like(input_ids, fill_value=-100)
        tok = self.processor.tokenizer

        for b in range(input_ids.size(0)):
            spans = self.find_assistant_content_sublist_indexes(
                input_ids[b].tolist(), tok
            )
            if not spans:
                # no assistant in this example; leave -100s
                continue

            for s, e in spans:
                # guard against any truncation past e
                e = min(e, input_ids.size(1))
                if s < e:
                    labels[b, s:e] = input_ids[b, s:e]

        # 5) Never train on pads
        labels = labels.masked_fill(attention_mask == 0, -100)

        # 6) Return everything Trainer expects
        batch: dict[str, torch.Tensor] = {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "labels": labels,
        }

        # multimodal extras (present only when applicable)
        for key in (
            "pixel_values",
            "pixel_attention_mask",
            "image_grid_thw",
            "video_grid_thw",
            "video_frame_mask",
        ):
            if key in proc:
                batch[key] = proc[key]

        return batch

    @staticmethod
    def find_assistant_content_sublist_indexes(
        input_ids: list[int], tokenizer: Any
    ) -> list[tuple[int, int]]:
        """
        Returns [(begin, end), ...] slices that cover the ASSISTANT **content**
        (excluding the assistant header) and INCLUDE the trailing "<|im_end|>\\n".
        Robust to tokenizer/template updates.
        """
        start_seq = tokenizer.encode(
            "<|im_start|>assistant\n", add_special_tokens=False
        )
        end_seq = tokenizer.encode("<|im_end|>\n", add_special_tokens=False)

        spans: list[tuple[int, int]] = []
        n = len(input_ids)
        i = 0

        while i <= n - len(start_seq):
            # find the next assistant header
            if input_ids[i : i + len(start_seq)] == start_seq:
                begin = i + len(start_seq)  # start after header

                # find matching end marker
                j = begin
                end = None
                while j <= n - len(end_seq):
                    if input_ids[j : j + len(end_seq)] == end_seq:
                        end = j + len(end_seq)  # include end tag
                        break
                    j += 1

                if end is None:
                    # no closing tag (truncated example) -> supervise to end
                    spans.append((begin, n))
                    break

                spans.append((begin, end))
                i = end
            else:
                i += 1

        return spans


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
    logging_strategy: str
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

        model_kwargs: dict[str, Any] = {"trust_remote_code": True}
        if torch.cuda.is_available():            
            model_kwargs["dtype"] = "auto"
        else:
            logger.warning(
                "CUDA is not available; loading Qwen VL in eager attention mode on CPU. This path is experimental and "
                "considerably slower than GPU execution."
            )
            model_kwargs.update({"attn_implementation": "eager", "dtype": torch.float32})

        model_kwargs["device_map"] = "cuda:0"
        logger.info("Using GPU: cuda:0")

        self.model = Qwen2_5_VLForConditionalGeneration.from_pretrained(plan.model_id, **model_kwargs)

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

        if not torch.cuda.is_available() and hparams.bf16:
            logger.warning("bf16 mixed precision requested but CUDA is unavailable; disabling bf16 for CPU training.")
            hparams.bf16 = False

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
            logging_strategy=hparams.logging_strategy,
            eval_strategy="steps" if eval_ds is not None else "no",
            save_strategy="steps",
            eval_steps=hparams.eval_steps if eval_ds is not None else None,
            report_to="wandb" if (self.plan.wandb_project and self.plan.wandb_run_name) else [],
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
