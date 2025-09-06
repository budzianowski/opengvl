from __future__ import annotations

import os
from typing import List

# third-party imports
from data_loader import Episode
from google import genai
from google.genai import types
from loguru import logger

from opengvl.clients.base import BaseModelClient


class GeminiClient(BaseModelClient):
    """Gemini client calling Google GenAI API for image+text content."""

    def __init__(self, model_name: str = "gemini-2.5-flash-lite-preview-06-17"):
        api_key = os.getenv("GEMINI_API_KEY")
        if not api_key:
            raise OSError("GEMINI_API_KEY not set in environment")
        self.client = genai.Client(api_key=api_key)
        self.model_name = model_name
        logger.info(f"Using Gemini model {self.model_name}")

    def generate_response(
        self,
        prompt: str,
        eval_episode: Episode,
        context_episodes: List[Episode],
    ) -> str:
        contents: list = [prompt]
        contents.append("Initial robot scene:")
        contents.append(
            types.Part.from_bytes(
                data=encode_image(eval_episode.starting_frame),
                mime_type="image/png",
            )
        )
        contents.append("In the initial robot scene, the task completion percentage is 0.")

        counter = 1
        for ctx_episode in context_episodes:
            for task_completion, frame in zip(ctx_episode.task_completion_predictions, ctx_episode.frames):
                contents.append(f"Frame {counter}:")
                contents.append(types.Part.from_bytes(data=encode_image(frame), mime_type="image/png"))
                contents.append(f"Task Completion Percentage: {task_completion:.1f}%")
                counter += 1

        contents.append(
            f"Now, for the task of {eval_episode.instruction}, output the task completion percentage for the following frames "
            "that are presented in random order. For each frame, format your response as follow: "
            "Frame {i}: Task Completion Percentages:{}%"
        )
        contents.append("Be rigorous and precise; percentage reflects task completion.")
        contents.append("Remember: frames are in random order.")

        for frame in eval_episode.frames:
            contents.append(f"Frame {counter}:")
            contents.append(types.Part.from_bytes(data=encode_image(frame), mime_type="image/png"))
            contents.append("")
            counter += 1

        response = self.client.models.generate_content(model=self.model_name, contents=contents)
        return response.text
