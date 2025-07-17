"""Model clients for the different models."""

import base64
import io
import math
import os
import tempfile
from abc import abstractmethod
from typing import List

import numpy as np
import torch
import torchvision.transforms as T
from loguru import logger
from PIL import Image
from torchvision.transforms import InterpolationMode
from transformers import (
    AutoConfig,
    AutoModel,
    AutoModelForCausalLM,
    AutoModelForImageTextToText,
    AutoModelForVision2Seq,
    AutoProcessor,
    AutoTokenizer,
    BitsAndBytesConfig,
    Gemma3ForConditionalGeneration,
)

from data_loader import Episode

# third-party imports
try:
    import openai
except ImportError:
    openai = None


IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)


class BaseModelClient:
    """Base class for all model clients"""

    max_new_tokens = 1000

    @abstractmethod
    def generate_response(
        self,
        prompt: str,
        eval_episode: Episode,
        context_episodes: List[Episode],
    ) -> str:
        """Generate response from the model"""
        pass

    def _to_pil(self, image) -> Image.Image:
        """Convert image to PIL Image."""
        if isinstance(image, Image.Image):
            return image

        # Convert tensor to numpy if needed
        if hasattr(image, "detach"):  # PyTorch tensor
            if image.is_cuda:
                image = image.cpu()
            image = image.detach().numpy()

        # Handle different numpy array formats
        if isinstance(image, np.ndarray):
            # Normalize if values are in [0, 1] range
            if image.dtype == np.float32 or image.dtype == np.float64:
                if image.max() <= 1.0:
                    image = (image * 255).astype(np.uint8)

            # Handle channel dimension - convert (C, H, W) to (H, W, C)
            if len(image.shape) == 3 and image.shape[0] in [1, 3, 4]:
                image = np.transpose(image, (1, 2, 0))

            # Convert to PIL Image
            if len(image.shape) == 3 and image.shape[2] == 3:
                pil_image = Image.fromarray(image, "RGB")
            elif len(image.shape) == 3 and image.shape[2] == 1:
                pil_image = Image.fromarray(image.squeeze(), "L")
            elif len(image.shape) == 2:
                pil_image = Image.fromarray(image, "L")
            else:
                raise ValueError(f"Unsupported image shape: {image.shape}")
        else:
            raise ValueError(f"Unsupported image type: {type(image)}")

        return pil_image

    def encode_image(self, image) -> str:
        """Encode image to base64 string for API calls"""
        pil_image = self._to_pil(image)
        # Convert to base64
        buffer = io.BytesIO()
        pil_image.save(buffer, format="PNG")
        return base64.b64encode(buffer.getvalue()).decode("utf-8")


class GemmaClient(BaseModelClient):
    """Gemma client implementation"""

    def __init__(self, model_id: str = "google/gemma-3-4b-it"):

        logger.info(f"Loading Gemma model {model_id}...")
        self.model = Gemma3ForConditionalGeneration.from_pretrained(model_id, device_map="auto").eval()
        self.processor = AutoProcessor.from_pretrained(model_id)

    def generate_response(
        self,
        prompt: str,
        eval_episode: Episode,
        context_episodes: List[Episode],
    ) -> str:

        # Build messages for Gemma following GPT4o structure exactly
        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": prompt},
                    {"type": "text", "text": "Initial robot scene:"},
                    {
                        "type": "image",
                        "image": self._to_pil(eval_episode.starting_frame),
                    },
                    {
                        "type": "text",
                        "text": "In the initial robot scene, the task completion percentage is 0.",
                    },
                ],
            }
        ]

        # Add example images with completion percentages
        for ctx_episode_idx, context_episode in enumerate(context_episodes):
            messages[0]["content"].extend(
                [
                    {"type": "text", "text": f"Example episode {ctx_episode_idx+1}."},
                    {
                        "type": "text",
                        "text": f"Instruction: {context_episode.instruction}",
                    },
                ]
            )

            for i, (task_completion, frame) in enumerate(
                zip(context_episode.task_completion_predictions, context_episode.frames)
            ):
                messages[0]["content"].extend(
                    [
                        {"type": "text", "text": f"Frame {i+1}: "},
                        {"type": "image", "base64": self._to_pil(frame)},
                        {
                            "type": "text",
                            "text": f"Task Completion Percentage: {task_completion:.1f}%",
                        },
                    ]
                )

        # Add query instruction
        messages[0]["content"].append(
            {
                "type": "text",
                "text": f"Now, for the task of {eval_episode.instruction}, output the task completion percentage for the following frames. Format: Frame: NUMBER, Description: DESCRIPTION, Task Completion: PERCENTAGE%\n",
            }
        )

        # Add query images
        for i, frame in enumerate(eval_episode.frames, 1):
            messages[0]["content"].extend(
                [
                    {"type": "text", "text": f"Frame {i}: "},
                    {"type": "image", "base64": self._to_pil(frame)},
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
            output = self.model.generate(**inputs, max_new_tokens=1024, do_sample=False)
            output = output[0][input_len:]

        decoded_outputs = self.processor.decode(output, skip_special_tokens=True)
        return decoded_outputs


class ModelFactory:
    """Factory class to create model clients"""

    @staticmethod
    def create_client(model_name: str) -> BaseModelClient:
        if "internvl" in model_name.lower():
            return InternVLClient()
        elif "smolvlm" in model_name.lower():
            return SmolVLMClient()
        elif "qwen" in model_name.lower():
            return QwenClient()
        elif "deepseek" in model_name.lower():
            return DeepseekClient()
        elif "gemma" in model_name.lower():
            return GemmaClient()
        elif "gemini" in model_name.lower():
            return GeminiClient()
        elif "gpt4o" in model_name.lower():
            return GPT4oClient()

        else:
            raise ValueError(f"Unknown model: {model_name}")
