import os
from typing import cast

from google.genai.client import Client
from google.genai.types import Part
from loguru import logger

from opengvl.clients.base import BaseModelClient
from opengvl.utils.aliases import Event, ImageEvent, ImageT, TextEvent
from opengvl.utils.images import encode_image


class GeminiClient(BaseModelClient):
    """Gemini client calling Google GenAI API for image+text content."""

    def __init__(self, *, rpm: float = 0.0, model_name: str):
        super().__init__(rpm=rpm)
        api_key = os.getenv("GEMINI_API_KEY")
        if not api_key:
            raise OSError()
        self.client = Client(api_key=api_key)
        self.model_name = model_name
        logger.info(f"Using Gemini model {self.model_name}")

    def _generate_from_events(self, events: list[Event], temperature: float) -> str:
        contents: list = []
        for ev in events:
            if isinstance(ev, TextEvent):
                contents.append(ev.text)
            elif isinstance(ev, ImageEvent):
                img = cast(ImageT, ev.image)
                contents.append(Part.from_bytes(data=encode_image(img), mime_type="image/png"))

        logger.debug(f"Contents length: {len(contents)} parts")
        response = self.client.models.generate_content(model=self.model_name, contents=contents, temperature=temperature)
        if response.text is None:
            raise RuntimeError()
        return response.text
