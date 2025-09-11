import os
from typing import Final, List

from dotenv import load_dotenv
from google.genai.client import Client
from google.genai.types import Part
from loguru import logger

from opengvl.clients.base import BaseModelClient
from opengvl.utils.data_types import Episode
from opengvl.utils.images import encode_image

BARE_MIN_LEN_TO_DISPLAY: Final[int] = len("Frame XX:  ")


class GeminiClient(BaseModelClient):
    """Gemini client calling Google GenAI API for image+text content."""

    def __init__(self, rpm, model_name: str):
        super().__init__(rpm=rpm)
        api_key = os.getenv("GEMINI_API_KEY")
        if not api_key:
            raise OSError("GEMINI_API_KEY not set in environment")
        self.client = Client(api_key=api_key)
        self.model_name = model_name
        logger.info(f"Using Gemini model {self.model_name}")

    def _generate_response_impl(
        self,
        prompt: str,
        eval_episode: Episode,
        context_episodes: List[Episode],
    ) -> str:
        contents: list = [prompt]
        contents.append("Initial robot scene:")
        contents.append(
            Part.from_bytes(
                data=encode_image(eval_episode.starting_frame),
                mime_type="image/png",
            )
        )
        contents.append("In the initial robot scene, the task completion percentage is 0%")

        counter = 1
        for ctx_episode in context_episodes:
            for task_completion, frame in zip(
                ctx_episode.shuffled_frames_approx_completion_rates, ctx_episode.shuffled_frames
            ):
                contents.append(f"Frame {counter}:")
                contents.append(Part.from_bytes(data=encode_image(frame), mime_type="image/png"))
                contents.append(f"Task Completion Percentage: {task_completion:.1f}%")
                counter += 1

        contents.append(
            f"Now, for the task of {eval_episode.instruction}, output the task completion percentage for the following frames "
            "that are presented in random order. For each frame, format your response as follow: "
            "Frame {i}: Task Completion Percentages:{}%"
        )
        contents.append("Be rigorous and precise; percentage reflects task completion.")
        contents.append("Remember: frames are in random order.")

        for frame in eval_episode.shuffled_frames:
            contents.append(f"Frame {counter}:")
            contents.append(Part.from_bytes(data=encode_image(frame), mime_type="image/png"))
            contents.append("")
            counter += 1

        logger.debug(f"Contents length: {len(contents)} parts")
        # # prefixes of them
        # for i, c in enumerate(contents):
        #     if isinstance(c, str) and len(c) > BARE_MIN_LEN_TO_DISPLAY:
        #         logger.debug(f"Contents part {i}: text (truncated 100 chars):\n{c[:100]}")

        response = self.client.models.generate_content(model=self.model_name, contents=contents)
        if response.text is None:
            raise RuntimeError("No text response from Gemini model")
        return response.text
