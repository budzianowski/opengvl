from typing import List

# third-party imports
import torch
from dotenv import load_dotenv
from loguru import logger
from transformers import AutoProcessor, Gemma3ForConditionalGeneration

from opengvl.clients.base import BaseModelClient
from opengvl.data_loader import Episode
from opengvl.utils.constants import MAX_TOKENS_TO_GENERATE
from opengvl.utils.images import to_pil


class GemmaClient(BaseModelClient):
    """Client for Gemma 3 image-text model (conditional generation)."""

    def __init__(self, model_id: str = "google/gemma-3-4b-it"):
        logger.info(f"Loading Gemma model {model_id} ...")
        self.model = Gemma3ForConditionalGeneration.from_pretrained(model_id, device_map="auto").eval()
        self.processor = AutoProcessor.from_pretrained(model_id)
        logger.info(type(self.processor))

    def generate_response(
        self,
        prompt: str,
        eval_episode: Episode,
        context_episodes: List[Episode],
    ) -> str:
        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": prompt},
                    {"type": "text", "text": "Initial robot scene:"},
                    {"type": "image", "image": to_pil(eval_episode.starting_frame)},
                    {"type": "text", "text": "In the initial robot scene, the task completion percentage is 0."},
                ],
            }
        ]

        for ctx_idx, ctx_episode in enumerate(context_episodes, 1):
            messages[0]["content"].extend(
                [
                    {"type": "text", "text": f"Example episode {ctx_idx}."},
                    {"type": "text", "text": f"Instruction: {ctx_episode.instruction}"},
                ]
            )
            for i, (task_completion, frame) in enumerate(
                zip(ctx_episode.task_completion_predictions, ctx_episode.frames), 1
            ):
                messages[0]["content"].extend(
                    [
                        {"type": "text", "text": f"Frame {i}:"},
                        {"type": "image", "base64": to_pil(frame)},
                        {"type": "text", "text": f"Task Completion Percentage: {task_completion:.1f}%"},
                    ]
                )

        messages[0]["content"].append(
            {
                "type": "text",
                "text": f"Now, for the task of {eval_episode.instruction}, output the task completion percentage for the following frames. Format: Frame: NUMBER, Task Completion: PERCENTAGE%",
            }
        )

        for i, frame in enumerate(eval_episode.frames, 1):
            messages[0]["content"].extend(
                [
                    {"type": "text", "text": f"Frame {i}:"},
                    {"type": "image", "base64": to_pil(frame)},
                ]
            )

        inputs = self.processor.apply_chat_template(
            messages,
            add_generation_prompt=True,
            tokenize=True,
            return_dict=True,
            return_tensors="pt",
        ).to(self.model.device, dtype=torch.bfloat16)

        input_len = inputs["input_ids"].shape[-1]
        if input_len > 32000:
            raise ValueError(f"Input length {input_len} exceeds maximum allowed length of 32000 tokens.")
        logger.info(f"Input length: {input_len}")

        with torch.inference_mode():
            output = self.model.generate(**inputs, max_new_tokens=MAX_TOKENS_TO_GENERATE, do_sample=False)
            output = output[0][input_len:]

        decoded = self.processor.decode(output, skip_special_tokens=True)
        return decoded


if __name__ == "__main__":
    load_dotenv("./.env", override=True)
    client = GemmaClient()
    print(client)
