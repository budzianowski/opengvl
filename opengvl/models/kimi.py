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
from dotenv import load_dotenv
from data_loader import Episode

# third-party imports
import openai

from google import genai
from google.genai import types

"""Kimi Thinking multimodal client implementation."""
from __future__ import annotations

from typing import List
from loguru import logger
import torch
from transformers import AutoModelForCausalLM, AutoProcessor

from data_loader import Episode
from .base import BaseModelClient


class KimiThinkingClient(BaseModelClient):
    """Client for Kimi Thinking VL model."""

    def __init__(self, model_id: str = "moonshotai/Kimi-VL-A3B-Thinking-2506"):
        logger.info(f"Loading Kimi Thinking model {model_id} ...")
        self.model = AutoModelForCausalLM.from_pretrained(
            model_id,
            torch_dtype="auto",
            device_map="auto",
            trust_remote_code=True,
        )
        self.processor = AutoProcessor.from_pretrained(model_id, trust_remote_code=True)

    def generate_response(
        self,
        prompt: str,
        eval_episode: Episode,
        context_episodes: List[Episode],
    ) -> str:
        images = []
        prompt_parts = [prompt, "Initial robot scene:"]
        images.append(self.to_pil(eval_episode.starting_frame))
        prompt_parts.append("In the initial robot scene, the task completion percentage is 0.")

        counter = 1
        for ctx_episode in context_episodes:
            for task_completion, frame in zip(ctx_episode.task_completion_predictions, ctx_episode.frames):
                prompt_parts.append(f"Frame {counter}:")
                images.append(self.to_pil(frame))
                prompt_parts.append(f"Task Completion Percentage: {task_completion:.1f}%")
                counter += 1

        prompt_parts.append(
            f"Now, for the task of {eval_episode.instruction}, output the task completion percentage for the following frames that are presented in random order. For each frame, format your response as follow: Frame {{i}}: Task Completion Percentages:{{}}%"
        )
        prompt_parts.append("Be rigorous and precise; percentage reflects task completion.")
        prompt_parts.append("Remember: frames are in random order.")

        for frame in eval_episode.frames:
            prompt_parts.append(f"Frame {counter}:")
            images.append(self.to_pil(frame))
            counter += 1

        messages = [{"role": "user", "content": []}]
        image_idx = 0
        for part in prompt_parts:
            messages[0]["content"].append({"type": "text", "text": part})
            # Heuristic: after a 'Frame X:' line or initial scene line, attach image if available.
            if (part.startswith("Frame ") or part == "Initial robot scene:") and image_idx < len(images):
                messages[0]["content"].append({"type": "image"})
                image_idx += 1

        prompt_text = self.processor.apply_chat_template(messages, add_generation_prompt=True)
        inputs = self.processor(text=prompt_text, images=images, return_tensors="pt").to(self.model.device, dtype=torch.bfloat16)

        input_len = inputs["input_ids"].shape[-1]
        if input_len > 128_000:
            raise ValueError(f"Input length {input_len} exceeds maximum allowed length of 128000 tokens.")
        logger.info(f"Input length: {input_len}")

        generated_ids = self.model.generate(**inputs, max_new_tokens=min(self.max_new_tokens, 32768), temperature=0.8)
        trimmed = [out_ids[len(in_ids):] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)]
        response = self.processor.batch_decode(trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False)[0]
        return response

