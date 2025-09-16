"""OpenAI multimodal client implementation."""

import os
from typing import cast

import openai
from loguru import logger

from opengvl.clients.base import BaseModelClient
from opengvl.utils.aliases import Event, ImageEvent, ImageT, TextEvent
from opengvl.utils.constants import MAX_TOKENS_TO_GENERATE
from opengvl.utils.images import encode_image


class OpenAIClient(BaseModelClient):
    """OpenAI client wrapping the Responses API for image+text prompting."""

    def __init__(self, model_id: str = "gpt-4o-mini", detail: str = "high", rpm: float = 0.0):
        super().__init__(rpm=rpm)
        api_key = os.getenv("OPENAI_API_KEY")
        if not api_key:
            raise OSError()
        self.client = openai.OpenAI(api_key=api_key)
        self.model_id = model_id
        self.detail = detail
        logger.info(f"Using OpenAI model {self.model_id}")

    def _generate_from_events(self, events: list[Event]) -> str:
        content = []
        for ev in events:
            if isinstance(ev, TextEvent):
                content.append({"type": "input_text", "text": ev.text})
            elif isinstance(ev, ImageEvent):
                img = cast(ImageT, ev.image)
                b64 = encode_image(img)
                content.append(
                    {
                        "type": "input_image",
                        "image_url": f"data:image/png;base64,{b64}",
                        "detail": self.detail,
                    }
                )

        messages = [{"role": "user", "content": content}]
        response = self.client.responses.create(
            model=self.model_id,
            input=messages,  # type: ignore[arg-type]
            max_output_tokens=MAX_TOKENS_TO_GENERATE,
        )
        return response.output_text
