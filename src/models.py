"""Model clients for the different models."""

from __future__ import annotations

import base64
import os
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import List

import numpy as np
import torch
from transformers import AutoModelForImageTextToText, AutoProcessor, Gemma3ForCausalLM

# third-party imports
try:
    import openai
except ImportError:
    openai = None

try:
    from google import genai
    from google.genai import types
except ImportError:
    genai = None


@dataclass
class Episode:
    frames: list[np.ndarray]


@dataclass
class Example:
    instruction: str
    examples: List[Episode]


class BaseModelClient(ABC):
    """Base class for all model clients"""

    @abstractmethod
    def generate_response(
        self,
        prompt: str,
        image_paths: List[str],
        task_description: str,
        example_indices: List[int],
        selected_indices: List[int],
        total_frames: int,
    ) -> str:
        """Generate response from the model"""
        pass


class GPT4oClient(BaseModelClient):
    """GPT-4o client implementation"""

    def __init__(self):
        if openai is None:
            raise ImportError("OpenAI package not installed")
        self.client = openai.OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

    def encode_image(self, image_path: str) -> str:
        with open(image_path, "rb") as f:
            return base64.b64encode(f.read()).decode("utf-8")

    def generate_response(
        self,
        prompt: str,
        image_paths: List[str],
        task_description: str,
        example_indices: List[int],
        selected_indices: List[int],
        total_frames: int,
    ) -> str:

        messages = [{"role": "user", "content": [{"type": "text", "text": prompt}]}]

        messages[0]["content"].extend(
            [
                {"type": "text", "text": "Initial robot scene:"},
                {
                    "type": "image_url",
                    "image_url": {"url": f"data:image/png;base64,{self.encode_image(image_paths[0])}"},
                },
                {
                    "type": "text",
                    "text": "In the initial robot scene, the task completion percentage is 0.",
                },
            ]
        )
        # Add example images with completion percentages
        for i, (idx, path) in enumerate(zip(example_indices, image_paths[:4])):
            messages[0]["content"].extend(
                [
                    {"type": "text", "text": f"Example {i+1}: "},
                    {
                        "type": "image_url",
                        "image_url": {"url": f"data:image/png;base64,{self.encode_image(path)}"},
                    },
                    {
                        "type": "text",
                        "text": f"Task Completion Percentage: {idx/total_frames*100:.1f}%",
                    },
                ]
            )

        messages[0]["content"].append(
            {
                "type": "text",
                "text": f"Now, for the task of {task_description}, output the task completion percentage for the following frames. Format: Frame: NUMBER, Description: DESCRIPTION, Task Completion: PERCENTAGE%",
            }
        )

        # Add query images
        for i, path in enumerate(image_paths[4:], 1):
            messages[0]["content"].extend(
                [
                    {"type": "text", "text": f"Frame {i}: "},
                    {
                        "type": "image_url",
                        "image_url": {"url": f"data:image/png;base64,{self.encode_image(path)}"},
                    },
                ]
            )

        response = self.client.chat.completions.create(model="gpt-4o", messages=messages)
        return response.choices[0].message.content


class InternVLClient(BaseModelClient):
    """InternVL client implementation"""

    def __init__(self):
        if torch is None:
            raise ImportError("PyTorch and transformers not installed")

        self.device = "mps" if torch.backends.mps.is_available() else "cuda" if torch.cuda.is_available() else "cpu"
        model_checkpoint = "OpenGVLab/InternVL3-1B-hf"
        self.processor = AutoProcessor.from_pretrained(model_checkpoint)
        self.model = AutoModelForImageTextToText.from_pretrained(
            model_checkpoint, device_map=self.device, torch_dtype=torch.bfloat16
        )

    def generate_response(
        self,
        prompt: str,
        image_paths: List[str],
        task_description: str,
        example_indices: List[int],
        selected_indices: List[int],
        total_frames: int,
    ) -> str:

        # Build messages for InternVL following GPT4o structure
        content = [{"type": "text", "text": prompt}]

        # Add initial scene
        content.extend(
            [
                {"type": "text", "text": "Initial robot scene:"},
                {"type": "image", "url": image_paths[0]},
                {
                    "type": "text",
                    "text": "In the initial robot scene, the task completion percentage is 0.",
                },
            ]
        )

        # Add example images with completion percentages
        for i, (idx, path) in enumerate(zip(example_indices, image_paths[:4])):
            content.extend(
                [
                    {"type": "text", "text": f"Example {i+1}:"},
                    {"type": "image", "url": path},
                    {
                        "type": "text",
                        "text": f"Task Completion Percentage: {idx/total_frames*100:.1f}%",
                    },
                ]
            )

        # Add query instruction
        content.append(
            {
                "type": "text",
                "text": f"Now, for the task of {task_description}, output the task completion percentage for the following frames. Format: Frame: NUMBER, Description: DESCRIPTION, Task Completion: PERCENTAGE%",
            }
        )

        # Add query images
        for i, path in enumerate(image_paths[4:], 1):
            content.extend(
                [
                    {"type": "text", "text": f"Frame {i}:"},
                    {"type": "image", "url": path},
                ]
            )

        messages = [{"role": "user", "content": content}]

        inputs = self.processor.apply_chat_template(
            messages,
            padding=True,
            add_generation_prompt=True,
            tokenize=True,
            return_dict=True,
            return_tensors="pt",
        ).to(self.model.device, dtype=torch.bfloat16)

        output = self.model.generate(**inputs, max_new_tokens=400)
        decoded_outputs = self.processor.batch_decode(output, skip_special_tokens=True)
        return decoded_outputs[0]


class GemmaClient(BaseModelClient):
    """Gemma client implementation"""

    def __init__(self):
        if torch is None:
            raise ImportError("PyTorch and transformers not installed")

        model_id = "google/gemma-3-1b-it"
        self.model = Gemma3ForCausalLM.from_pretrained(model_id, device_map="auto").eval()
        self.processor = AutoProcessor.from_pretrained(model_id)

    def generate_response(
        self,
        prompt: str,
        image_paths: List[str],
        task_description: str,
        example_indices: List[int],
        selected_indices: List[int],
        total_frames: int,
    ) -> str:

        # Build messages for Gemma following the provided example structure
        messages = [
            {"role": "system", "content": [{"type": "text", "text": "You are a helpful assistant."}]},
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": prompt},
                ],
            },
        ]

        # Add initial scene
        messages[1]["content"].extend(
            [
                {"type": "text", "text": "Initial robot scene:"},
                {"type": "image", "image": image_paths[0]},
                {"type": "text", "text": "In the initial robot scene, the task completion percentage is 0."},
            ]
        )

        # Add example images with completion percentages
        for i, (idx, path) in enumerate(zip(example_indices, image_paths[:4])):
            messages[1]["content"].extend(
                [
                    {"type": "text", "text": f"Example {i+1}:"},
                    {"type": "image", "image": path},
                    {"type": "text", "text": f"Task Completion Percentage: {idx/total_frames*100:.1f}%"},
                ]
            )

        # Add query instruction
        messages[1]["content"].append(
            {
                "type": "text",
                "text": f"Now, for the task of {task_description}, output the task completion percentage for the following frames. Format: Frame: NUMBER, Description: DESCRIPTION, Task Completion: PERCENTAGE%",
            }
        )

        # Add query images
        for i, path in enumerate(image_paths[4:], 1):
            messages[1]["content"].extend([{"type": "text", "text": f"Frame {i}:"}, {"type": "image", "image": path}])

        inputs = self.processor.apply_chat_template(
            messages, padding=True, add_generation_prompt=True, tokenize=True, return_dict=True, return_tensors="pt"
        ).to(self.model.device)

        output = self.model.generate(**inputs, max_new_tokens=400)
        decoded_outputs = self.processor.batch_decode(output, skip_special_tokens=True)
        return decoded_outputs[0]


class GeminiClient(BaseModelClient):
    """Gemini client implementation"""

    def __init__(self):
        if genai is None:
            raise ImportError("Google GenAI package not installed")
        self.client = genai.Client(api_key=os.getenv("GEMINI_API_KEY"))

    def generate_response(
        self,
        prompt: str,
        image_paths: List[str],
        task_description: str,
        example_indices: List[int],
        selected_indices: List[int],
        total_frames: int,
    ) -> str:

        # Build contents following GPT4o structure
        contents = [prompt]

        # Add initial scene
        contents.append("Initial robot scene:")
        with open(image_paths[0], "rb") as f:
            contents.append(types.Part.from_bytes(data=f.read(), mime_type="image/png"))
        contents.append("In the initial robot scene, the task completion percentage is 0.")

        # Add example images with completion percentages
        for i, (idx, path) in enumerate(zip(example_indices, image_paths[:4])):
            contents.append(f"Example {i+1}:")
            with open(path, "rb") as f:
                contents.append(types.Part.from_bytes(data=f.read(), mime_type="image/png"))
            contents.append(f"Task Completion Percentage: {idx/total_frames*100:.1f}%")

        # Add query instruction
        contents.append(
            f"Now, for the task of {task_description}, output the task completion percentage for the following frames. Format: Frame: NUMBER, Description: DESCRIPTION, Task Completion: PERCENTAGE%"
        )

        # Add query images
        for i, path in enumerate(image_paths[4:], 1):
            contents.append(f"Frame {i}:")
            with open(path, "rb") as f:
                contents.append(types.Part.from_bytes(data=f.read(), mime_type="image/png"))

        response = self.client.models.generate_content(model="gemini-2.5-flash", contents=contents)
        return response.text


class ModelFactory:
    """Factory class to create model clients"""

    @staticmethod
    def create_client(model_name: str) -> BaseModelClient:
        if model_name.lower() == "internvl":
            return InternVLClient()
        elif model_name.lower() == "gemma":
            return GemmaClient()
        elif model_name.lower() == "gemini":
            return GeminiClient()
        elif model_name.lower() == "gpt4o":
            return GPT4oClient()

        else:
            raise ValueError(f"Unknown model: {model_name}")
