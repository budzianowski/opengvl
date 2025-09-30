from typing import cast

import torch
from loguru import logger
from transformers import AutoModelForCausalLM, AutoProcessor

from opengvl.clients.base import BaseModelClient
from opengvl.utils.aliases import Event, ImageEvent, ImageT, TextEvent
from opengvl.utils.constants import MAX_TOKENS_TO_GENERATE
from opengvl.utils.errors import InputTooLongError
from opengvl.utils.images import to_pil


class KimiThinkingClient(BaseModelClient):
    """Client for Kimi Thinking VL model."""

    def __init__(self, model_id: str = "moonshotai/Kimi-VL-A3B-Thinking-2506", rpm: float = 0.0):
        super().__init__(rpm=rpm)
        logger.info(f"Loading Kimi Thinking model {model_id} ...")
        self.model = AutoModelForCausalLM.from_pretrained(
            model_id,
            torch_dtype="auto",
            device_map="auto",
            trust_remote_code=True,
        )
        self.processor = AutoProcessor.from_pretrained(model_id, trust_remote_code=True)

    def _generate_from_events(self, events: list[Event]) -> str:
        images = []
        messages = [{"role": "user", "content": []}]
        for ev in events:
            if isinstance(ev, TextEvent):
                messages[0]["content"].append({"type": "text", "text": ev.text})
            elif isinstance(ev, ImageEvent):
                messages[0]["content"].append({"type": "image"})
                images.append(to_pil(cast(ImageT, ev.image)))

        prompt_text = self.processor.apply_chat_template(messages, add_generation_prompt=True)
        inputs = self.processor(
            text=prompt_text,
            images=images,
            return_tensors="pt",
        ).to(
            self.model.device,
            dtype=torch.bfloat16,
        )

        input_len = inputs["input_ids"].shape[-1]
        if input_len > 128_000:
            raise InputTooLongError(input_len, 128_000)
        logger.info(f"Input length: {input_len}")

        generated_ids = self.model.generate(
            **inputs,
            max_new_tokens=min(
                MAX_TOKENS_TO_GENERATE,
                32768,
            ),
            temperature=0.8,
        )
        trimmed = [out_ids[len(in_ids) :] for in_ids, out_ids in zip(inputs.input_ids, generated_ids, strict=False)]
        return self.processor.batch_decode(trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False)[0]
