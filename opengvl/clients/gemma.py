from typing import cast

import torch
from loguru import logger
from transformers import AutoProcessor, Gemma3ForConditionalGeneration

from opengvl.clients.base import BaseModelClient
from opengvl.utils.aliases import Event, ImageEvent, ImageT, TextEvent
from opengvl.utils.constants import MAX_TOKENS_TO_GENERATE
from opengvl.utils.images import to_pil


class GemmaClient(BaseModelClient):
    """Client for Gemma 3 image-text model (conditional generation)."""

    def __init__(self, model_name: str = "google/gemma-3-4b-it", rpm: float = 0.0):
        super().__init__(rpm=rpm)
        logger.info(f"Loading Gemma model {model_name} ...")
        self.model = Gemma3ForConditionalGeneration.from_pretrained(model_name, device_map="auto").eval()
        self.model_name = model_name
        self.processor = AutoProcessor.from_pretrained(model_name)
        logger.info(type(self.processor))

    def _generate_from_events(self, events: list[Event], temperature: float) -> str:
        messages = [{"role": "user", "content": []}]
        for ev in events:
            if isinstance(ev, TextEvent):
                messages[0]["content"].append({"type": "text", "text": ev.text})
            elif isinstance(ev, ImageEvent):
                messages[0]["content"].append({"type": "image", "image": to_pil(cast(ImageT, ev.image))})

        inputs = self.processor.apply_chat_template(
            messages,
            add_generation_prompt=True,
            tokenize=True,
            return_dict=True,
            return_tensors="pt",
        ).to(self.model.device, dtype=torch.bfloat16)

        input_len = inputs["input_ids"].shape[-1]
        if input_len > 32000:
            raise ValueError()
        logger.info(f"Input length: {input_len}")

        with torch.inference_mode():
            if temperature > 0.0:
                output = self.model.generate(**inputs, max_new_tokens=MAX_TOKENS_TO_GENERATE, do_sample=True, temperature=temperature)
            else:
                output = self.model.generate(**inputs, max_new_tokens=MAX_TOKENS_TO_GENERATE, do_sample=False)
            output = output[0][input_len:]

        return self.processor.decode(output, skip_special_tokens=True)
