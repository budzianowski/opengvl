"""OpenAI multimodal client implementation."""

import os

import openai
from loguru import logger

from opengvl.clients.base import BaseModelClient
from opengvl.utils.constants import MAX_TOKENS_TO_GENERATE
from opengvl.utils.data_types import Episode
from opengvl.utils.images import encode_image


class OpenAIClient(BaseModelClient):
    """OpenAI client wrapping the Responses API for image+text prompting."""

    def __init__(self, model_id: str = "gpt-4o-mini", detail: str = "high"):
        api_key = os.getenv("OPENAI_API_KEY")
        if not api_key:
            raise OSError("OPENAI_API_KEY not set in environment")
        self.client = openai.OpenAI(api_key=api_key)
        self.model_id = model_id
        self.detail = detail
        logger.info(f"Using OpenAI model {self.model_id}")

    def _generate_response_impl(
        self,
        prompt: str,
        eval_episode: Episode,
        context_episodes: list[Episode],
    ) -> str:
        content = [{"type": "input_text", "text": prompt}]

        # Initial scene
        content.extend(
            [
                {"type": "input_text", "text": "Initial robot scene:"},
                {
                    "type": "input_image",
                    "image_url": f"data:image/png;base64,{encode_image(eval_episode.starting_frame)}",
                    "detail": self.detail,
                },
                {"type": "input_text", "text": "In the initial robot scene, the task completion percentage is 0."},
            ]
        )

        counter = 1
        # Context episodes
        for ctx_episode in context_episodes:
            for task_completion, frame in zip(
                ctx_episode.shuffled_frames_approx_completion_rates, ctx_episode.shuffled_frames
            ):
                content.extend(
                    [
                        {"type": "input_text", "text": f"Frame {counter}:"},
                        {
                            "type": "input_image",
                            "image_url": f"data:image/png;base64,{encode_image(frame)}",
                            "detail": self.detail,
                        },
                        {"type": "input_text", "text": f"Task Completion Percentage: {task_completion:.1f}%"},
                    ]
                )
                counter += 1

        # Query instruction
        content.append(
            {
                "type": "input_text",
                "text": (
                    f"Now, for the task of {eval_episode.instruction}, output the task completion percentage for the following "
                    "frames that are presented in random order. For each frame, format your response as follow: "
                    "Frame {i}: Task Completion Percentages:{}%"
                ),
            }
        )
        content.append({"type": "input_text", "text": "Be rigorous and precise; percentage reflects task completion."})
        content.append({"type": "input_text", "text": "Remember: frames are in random order."})

        # Evaluation frames
        for frame in eval_episode.shuffled_frames:
            content.extend(
                [
                    {"type": "input_text", "text": f"Frame {counter}:"},
                    {
                        "type": "input_image",
                        "image_url": f"data:image/png;base64,{encode_image(frame)}",
                        "detail": self.detail,
                    },
                ]
            )
            counter += 1

        messages = [{"role": "user", "content": content}]
        response = self.client.responses.create(
            model=self.model_id,
            input=messages,
            max_output_tokens=MAX_TOKENS_TO_GENERATE,
        )
        return response.output_text
