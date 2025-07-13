"""Model clients for the different models."""
import base64
import io
import os
from abc import abstractmethod
from typing import List

import numpy as np
import torch
from PIL import Image
from transformers import (AutoModelForImageTextToText, AutoProcessor,
                          AutoTokenizer, Gemma3ForCausalLM)

from data_loader import Episode

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

    def encode_image(self, image) -> str:
        """Encode image to base64 string for API calls"""
        
        # Convert tensor to numpy if needed
        if hasattr(image, 'detach'):  # PyTorch tensor
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
                pil_image = Image.fromarray(image, 'RGB')
            elif len(image.shape) == 3 and image.shape[2] == 1:
                pil_image = Image.fromarray(image.squeeze(), 'L')
            elif len(image.shape) == 2:
                pil_image = Image.fromarray(image, 'L')
            else:
                raise ValueError(f"Unsupported image shape: {image.shape}")
        else:
            # Assume it's already a PIL Image
            pil_image = image
        
        # Convert to base64
        buffer = io.BytesIO()
        pil_image.save(buffer, format='PNG')
        return base64.b64encode(buffer.getvalue()).decode('utf-8')


class GPT4oClient(BaseModelClient):
    """GPT-4o client implementation"""

    def __init__(self):
        if openai is None:
            raise ImportError("OpenAI package not installed")
        self.client = openai.OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

    def generate_response(
        self,
        prompt: str,
        eval_episode: Episode,
        context_episodes: List[Episode],
    ) -> str:

        messages = [{"role": "user", "content": [{"type": "text", "text": prompt}]}]

        messages[0]["content"].extend(
            [
                {"type": "text", "text": "Initial robot scene:"},
                {
                    "type": "image_url",
                    "image_url": {"url": f"data:image/png;base64,{self.encode_image(eval_episode.starting_frame)}"},
                },
                {
                    "type": "text",
                    "text": "In the initial robot scene, the task completion percentage is 0.",
                },
            ]
        )
        # Add example images with completion percentages
        for ctx_episode_idx, context_episode in enumerate(context_episodes):
            messages[0]["content"].extend(
                [
                    {"type": "text", "text": f"Example episode {ctx_episode_idx+1}."},
                    {"type": "text", "text": f"Instruction: {context_episode.instruction}"},
                ]
            )

            for i, (task_completion, frame) in enumerate(zip(context_episode.task_completion_predictions, context_episode.frames)):
                messages[0]["content"].extend(
                    [
                        {"type": "text", "text": f"Frame {i+1}: "},
                        {
                            "type": "image_url",
                            "image_url": {"url": f"data:image/png;base64,{self.encode_image(frame)}"},
                        },
                        {
                            "type": "text",
                            "text": f"Task Completion Percentage: {task_completion:.1f}%",
                        },
                    ]
                )

        messages[0]["content"].append(
            {
                "type": "text",
                "text": f"Now, for the task of {eval_episode.instruction}, output the task completion percentage for the following frames. Format: Frame: NUMBER, Description: DESCRIPTION, Task Completion: PERCENTAGE%",
            }
        )

        # Add query images
        for i, frame in enumerate(eval_episode.frames, 1):
            messages[0]["content"].extend(
                [
                    {"type": "text", "text": f"Frame {i}: "},
                    {
                        "type": "image_url",
                        "image_url": {"url": f"data:image/png;base64,{self.encode_image(frame)}"},
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
        ).eval()

    def generate_response(
        self,
        prompt: str,
        eval_episode: Episode,
        context_episodes: List[Episode],
    ) -> str:
        content = [{"type": "text", "text": prompt}]

        # Add initial scene
        content.extend(
            [
                {"type": "text", "text": "Initial robot scene:"},
                {"type": "image", "base64": self.encode_image(eval_episode.starting_frame)},
                {
                    "type": "text",
                    "text": "In the initial robot scene, the task completion percentage is 0.\n",
                },
            ]
        )

        # Add example images with completion percentages
        for ctx_episode_idx, context_episode in enumerate(context_episodes):
            content.extend(
                [
                    {"type": "text", "text": f"Example episode {ctx_episode_idx+1}.\n"},
                    {"type": "text", "text": f"Instruction: {context_episode.instruction}\n"},
                ]
            )

            for i, (task_completion, frame) in enumerate(zip(context_episode.task_completion_predictions, context_episode.frames)):
                content.extend(
                    [
                        {"type": "text", "text": f"Frame {i+1}: \n"},
                        {"type": "image", "base64": self.encode_image(frame)},
                        {
                            "type": "text",
                            "text": f"Task Completion Percentage: {task_completion:.1f}%",
                        },
                    ]
                )

        # Add query instruction
        content.append(
            {
                "type": "text",
                "text": f"Now, for the task of {eval_episode.instruction}, output the task completion percentage (with the format Format: Frame: NUMBER, Description: DESCRIPTION, Task Completion: PERCENTAGE%) for the following frames:",
            }
        )

        # Add query images
        for i, frame in enumerate(eval_episode.frames, 1):
            content.extend(
                [
                    {"type": "text", "text": f"Frame {i}: \n"},
                    {"type": "image", "base64": self.encode_image(frame)},
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

        output = self.model.generate(**inputs, max_new_tokens=self.max_new_tokens)
        decoded_outputs = self.processor.batch_decode(output, skip_special_tokens=True)
        return decoded_outputs[0]


class GemmaClient(BaseModelClient):
    """Gemma client implementation"""

    def __init__(self):
        if torch is None:
            raise ImportError("PyTorch and transformers not installed")

        model_id = "google/gemma-3-1b-it"
        self.model = Gemma3ForCausalLM.from_pretrained(model_id, device_map="auto").eval()
        self.processor = AutoTokenizer.from_pretrained(model_id)

    def generate_response(
        self,
        prompt: str,
        eval_episode: Episode,
        context_episodes: List[Episode],
    ) -> str:

        # Build messages for Gemma following GPT4o structure exactly
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
                {"type": "image", "base64": self.encode_image(eval_episode.starting_frame)},
                {"type": "text", "text": "In the initial robot scene, the task completion percentage is 0."},
            ]
        )

        # Add example images with completion percentages
        for ctx_episode_idx, context_episode in enumerate(context_episodes):
            messages[1]["content"].extend(
                [
                    {"type": "text", "text": f"Example episode {ctx_episode_idx+1}."},
                    {"type": "text", "text": f"Instruction: {context_episode.instruction}"},
                ]
            )

            for i, (task_completion, frame) in enumerate(zip(context_episode.task_completion_predictions, context_episode.frames)):
                messages[1]["content"].extend(
                    [
                        {"type": "text", "text": f"Frame {i+1}: "},
                        {"type": "image", "base64": self.encode_image(frame)},
                        {"type": "text", "text": f"Task Completion Percentage: {task_completion:.1f}%"},
                    ]
                )

        # Add query instruction
        messages[1]["content"].append(
            {
                "type": "text",
                "text": f"Now, for the task of {eval_episode.instruction}, output the task completion percentage for the following frames. Format: Frame: NUMBER, Description: DESCRIPTION, Task Completion: PERCENTAGE%",
            }
        )

        # Add query images
        for i, frame in enumerate(eval_episode.frames, 1):
            messages[1]["content"].extend(
                [
                    {"type": "text", "text": f"Frame {i}: "},
                    {"type": "image", "base64": self.encode_image(frame)}
                ]
            )

        inputs = self.processor.apply_chat_template(
            messages, 
            padding=True, 
            add_generation_prompt=True, 
            tokenize=True, 
            return_dict=True, 
            return_tensors="pt"
        ).to(self.model.device)

        output = self.model.generate(**inputs, max_new_tokens=self.max_new_tokens)
        decoded_outputs = self.processor.batch_decode(output, skip_special_tokens=True)
        return decoded_outputs[0]


class GeminiClient(BaseModelClient):
    """Gemini client implementation"""

    def __init__(self):
        if genai is None:
            raise ImportError("Google GenAI package not installed")
        self.client = genai.Client(api_key=os.getenv("GEMINI_API_KEY"))
        self.model_name = "gemini-2.5-flash"

    def generate_response(
        self,
        prompt: str,
        eval_episode: Episode,
        context_episodes: List[Episode],
    ) -> str:

        contents = [prompt]

        # Add initial scene
        contents.append("Initial robot scene:")
        contents.append(types.Part.from_bytes(data=self.encode_image(eval_episode.starting_frame), mime_type="image/png"))
        contents.append("In the initial robot scene, the task completion percentage is 0.")

        # Add example images with completion percentages
        for ctx_episode_idx, context_episode in enumerate(context_episodes):
            contents.append(f"Example episode {ctx_episode_idx+1}.")
            contents.append(f"Instruction: {context_episode.instruction}")
            
            for i, (task_completion, frame) in enumerate(zip(context_episode.task_completion_predictions, context_episode.frames)):
                contents.append(f"Frame {i+1}: ")
                contents.append(types.Part.from_bytes(data=self.encode_image(frame), mime_type="image/png"))
                contents.append(f"Task Completion Percentage: {task_completion:.1f}%")

        # Add query instruction
        contents.append(
            f"Now, for the task of {eval_episode.instruction}, output the task completion percentage for the following frames. Format: Frame: NUMBER, Description: DESCRIPTION, Task Completion: PERCENTAGE%"
        )

        # Add query images
        for i, frame in enumerate(eval_episode.frames, 1):
            contents.append(f"Frame {i}: ")
            contents.append(types.Part.from_bytes(data=self.encode_image(frame), mime_type="image/png"))

        response = self.client.models.generate_content(model=self.model_name, contents=contents)
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
