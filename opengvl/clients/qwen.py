from typing import cast
import torch
from loguru import logger
from transformers import AutoProcessor, Qwen2_5_VLForConditionalGeneration

from opengvl.clients.base import BaseModelClient
from opengvl.utils.aliases import Event, ImageEvent, ImageT, TextEvent
from opengvl.utils.constants import MAX_TOKENS_TO_GENERATE
from opengvl.utils.images import to_pil
from qwen_vl_utils import process_vision_info


class QwenClient(BaseModelClient):
    def __init__(self, model_name: str = "Qwen/Qwen2.5-VL-3B-Instruct", rpm: float = 0.0):
        super().__init__(rpm=rpm)
        logger.info(f"Loading Qwen model {model_name}...")
        self.model = Qwen2_5_VLForConditionalGeneration.from_pretrained(model_name, torch_dtype="auto", device_map="auto")
        self.processor = AutoProcessor.from_pretrained(model_name, trust_remote_code=True)
        logger.info(type(self.processor))
        self.model_name = model_name

    def _generate_from_events(self, events: list[Event], temperature: float) -> str:
        messages = [{"role": "user", "content": []}]
        for ev in events:
            if isinstance(ev, TextEvent):
                messages[0]["content"].append({"type": "text", "text": ev.text})
            elif isinstance(ev, ImageEvent):
                messages[0]["content"].append({"type": "image", "image": to_pil(cast(ImageT, ev.image))})

        text = self.processor.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        image_inputs, video_inputs = process_vision_info(messages)

        inputs = self.processor(
            text=[text],
            images=image_inputs,
            videos=video_inputs,
            padding=True,
            return_tensors="pt",
        )
        inputs = inputs.to("cuda")

        input_len = inputs["input_ids"].shape[-1]
        if input_len > self.max_input_length:
            raise ValueError()
        logger.info(f"Input length: {input_len}")

        # Inference: Generation of the output
        generated_ids = self.model.generate(**inputs, max_new_tokens=MAX_TOKENS_TO_GENERATE, temperature=temperature)
        generated_ids_trimmed = [
            out_ids[len(in_ids) :] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
        ]
        
        output_text = self.processor.batch_decode(
            generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
        )

        return output_text[0]
