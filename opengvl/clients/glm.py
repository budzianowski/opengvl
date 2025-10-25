from typing import cast

import torch
from loguru import logger
from transformers import AutoProcessor, Glm4vForConditionalGeneration

from opengvl.clients.base import BaseModelClient
from opengvl.utils.aliases import Event, ImageEvent, ImageT, TextEvent
from opengvl.utils.constants import MAX_TOKENS_TO_GENERATE
from opengvl.utils.images import to_pil


class GLMClient(BaseModelClient):
    """GLM client implementation"""

    def __init__(self, model_name: str = "zai-org/GLM-4.1V-9B-Thinking", rpm: float = 0.0):
        super().__init__(rpm=rpm)
        logger.info(f"Loading GLM model {model_name}...")
        self.model = Glm4vForConditionalGeneration.from_pretrained(
            model_name,
            torch_dtype="auto",
            device_map="auto",
            trust_remote_code=True,
        )
        self.model_name = model_name
        self.processor = AutoProcessor.from_pretrained(model_name, trust_remote_code=True)
        logger.info(type(self.processor))


    def _generate_from_events(self, events: list[Event], temperature: float) -> str:
        messages = [{"role": "user", "content": []}]
        images = []
        for ev in events:
            if isinstance(ev, TextEvent):
                messages[0]["content"].append({"type": "text", "text": ev.text})
            elif isinstance(ev, ImageEvent):
                messages[0]["content"].append({"type": "image"})
                images.append(to_pil(cast(ImageT, ev.image)))


        prompt = self.processor.apply_chat_template(
            messages,
            add_generation_prompt=True
        )
        inputs = self.processor(text=prompt, images=images, return_tensors="pt").to(self.model.device, dtype=torch.bfloat16)
        
        input_len = inputs["input_ids"].shape[-1]

        if input_len > self.max_input_length:
            raise ValueError(f"Input length {input_len} exceeds maximum allowed length of {self.max_input_length} tokens.")
        logger.info(f"Input length: {input_len}")

        generated_ids = self.model.generate(**inputs, max_new_tokens=MAX_TOKENS_TO_GENERATE, temperature=temperature)
        generated_ids_trimmed = [
            out_ids[len(in_ids) :] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
        ]
        response = self.processor.batch_decode(
            generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
        )[0]
        
        return response
