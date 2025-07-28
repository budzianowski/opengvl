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
try:
    import openai
except ImportError:
    openai = None

try:
    from google import genai
    from google.genai import types
except ImportError:
    genai = None

load_dotenv()

class BaseModelClient:
    """Base class for all model clients"""

    max_new_tokens = 1024

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
            print("Image is already a PIL Image")
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

        # TODO: test this
        pil_image = pil_image.resize((244, 244))
        return pil_image

    def encode_image(self, image) -> str:
        """Encode image to base64 string for API calls"""
        pil_image = self._to_pil(image)
        # Convert to base64
        buffer = io.BytesIO()
        pil_image.save(buffer, format="PNG")
        return base64.b64encode(buffer.getvalue()).decode("utf-8")


class OpenAIClient(BaseModelClient):
    """OpenAI client implementation"""

    def __init__(self, model_id: str = "gpt-4o-mini", detail: str = "high"):
        if openai is None:
            raise ImportError("OpenAI library is not installed. Please install it with 'pip install openai'.")

        self.client = openai.OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
        self.model_id = model_id
        self.detail = detail
        logger.info(f"Using OpenAI model {self.model_id}")

    def generate_response(
        self,
        prompt: str,
        eval_episode: Episode,
        context_episodes: List[Episode],
    ) -> str:
        """Generate response using OpenAI API"""

        content = [{"type": "input_text", "text": prompt}]

        # Add initial scene
        content.extend(
            [
                {"type": "input_text", "text": "Initial robot scene:"},
                {
                    "type": "input_image",
                    "image_url": f"data:image/png;base64,{self.encode_image(eval_episode.starting_frame)}",
                    "detail": self.detail,
                },
                {
                    "type": "input_text",
                    "text": "\nIn the initial robot scene, the task completion percentage is 0. \n",
                },
            ]
        )

        counter = 1

        # Add context images with completion percentages
        for ctx_episode_idx, context_episode in enumerate(context_episodes):
            for i, (task_completion, frame) in enumerate(
                zip(context_episode.task_completion_predictions, context_episode.frames)
            ):
                content.extend(
                    [
                        {"type": "input_text", "text": f"Frame {counter}: "},
                        {
                            "type": "input_image",
                            "image_url": f"data:image/png;base64,{self.encode_image(frame)}",
                            "detail": self.detail,
                        },
                        {
                            "type": "input_text",
                            "text": f"Task Completion Percentage: {task_completion:.1f}% \n",
                        },
                    ]
                )
                counter += 1

        # Add query instruction
        content.append(
            {
                "type": "input_text",
                "text": f"Now, for the task of {eval_episode.instruction}" + ", output the task completion percentage for the following frames that are presented in random order. For each frame, format your response as follow: Frame {i}: Task Completion Percentages:{}% \n",
            }
        )
        content.append(
            {
                "type": "input_text",
                "text": f"Be rigorous, precise and remember that the task completion percentage is the percentage of the task that has been completed. \n",
            }
        )
        content.append(
            {
                "type": "input_text",
                "text": f"Remember that the frames are presented in random order. \n",
            }
        )

        # Add eval images
        for i, frame in enumerate(eval_episode.frames):
            content.extend(
                [
                    {"type": "input_text", "text": f"Frame {counter}: "},
                    {
                        "type": "input_image",
                        "image_url": f"data:image/png;base64,{self.encode_image(frame)}",
                        "detail": self.detail,
                    },
                    {"type": "input_text", "text": f"\n"},
                ]
            )
            counter += 1

        messages = [{"role": "user", "content": content}]

        response = self.client.responses.create(
            model=self.model_id,
            input=messages,
            max_output_tokens=self.max_new_tokens,
        )
        return response.output_text


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
            output = self.model.generate(**inputs, max_new_tokens=self.max_new_tokens, do_sample=False)
            output = output[0][input_len:]

        decoded_outputs = self.processor.decode(output, skip_special_tokens=True)
        return decoded_outputs


class KimiThinkingClient(BaseModelClient):
    """Kimi Thinking client implementation"""

    def __init__(self, model_id: str = "moonshotai/Kimi-VL-A3B-Thinking-2506"):

        logger.info(f"Loading Kimi Thinking model {model_id}...")
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

        # Follow Gemini client format - use simple content list instead of structured messages
        images = []
        
        # Start with the prompt text
        prompt_parts = [prompt]
        
        # Add initial scene
        prompt_parts.append("Initial robot scene:")
        images.append(self._to_pil(eval_episode.starting_frame))
        prompt_parts.append("In the initial robot scene, the task completion percentage is 0.")
        
        counter = 1
        
        # Add context episodes with continuous counter like Gemini client
        for ctx_episode_idx, context_episode in enumerate(context_episodes):
            for i, (task_completion, frame) in enumerate(
                zip(context_episode.task_completion_predictions, context_episode.frames)
            ):
                prompt_parts.append(f"Frame {counter}: ")
                images.append(self._to_pil(frame))
                prompt_parts.append(f"Task Completion Percentage: {task_completion:.1f}%")
                counter += 1
        
        # Add query instruction matching Gemini client format
        prompt_parts.append(
            f"Now, for the task of {eval_episode.instruction}, output the task completion percentage for the following frames that are presented in random order. For each frame, format your response as follow: Frame {{i}}: Task Completion Percentages:{{}}%"
        )
        prompt_parts.append(
            "Be rigorous, precise and remember that the task completion percentage is the percentage of the task that has been completed."
        )
        prompt_parts.append(
            "Remember that the frames are presented in random order."
        )
        
        # Add eval images with continuous counter
        for i, frame in enumerate(eval_episode.frames):
            prompt_parts.append(f"Frame {counter}: ")
            images.append(self._to_pil(frame))
            counter += 1
        
        # Convert to messages format but with simpler structure
        messages = [
            {
                "role": "user", 
                "content": []
            }
        ]
        
        # Interleave text and images
        image_idx = 0
        for part in prompt_parts:
            if "Initial robot scene:" in part or "Frame " in part and ":" in part:
                messages[0]["content"].append({"type": "text", "text": part})
                if image_idx < len(images):
                    messages[0]["content"].append({"type": "image"})
                    image_idx += 1
            else:
                messages[0]["content"].append({"type": "text", "text": part})

        prompt = self.processor.apply_chat_template(
            messages,
            add_generation_prompt=True
        )
        inputs = self.processor(text=prompt, images=images, return_tensors="pt").to(self.model.device, dtype=torch.bfloat16)
        
        input_len = inputs["input_ids"].shape[-1]

        if input_len > 128_000:
            raise ValueError(f"Input length {input_len} exceeds maximum allowed length of 128000 tokens.")
        logger.info(f"Input length: {input_len}")

        generated_ids = self.model.generate(**inputs, max_new_tokens=32768, temperature=0.8)
        generated_ids_trimmed = [
            out_ids[len(in_ids) :] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
        ]
        response = self.processor.batch_decode(
            generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
        )[0]

        return response


class GeminiClient(BaseModelClient):
    """Gemini client implementation"""

    def __init__(self, model_name: str = "gemini-2.5-flash-lite-preview-06-17"):
        if genai is None:
            raise ImportError("Google GenAI package not installed")
        self.client = genai.Client(api_key=os.getenv("GEMINI_API_KEY"))
        self.model_name = model_name

    def generate_response(
        self,
        prompt: str,
        eval_episode: Episode,
        context_episodes: List[Episode],
    ) -> str:
        contents = [prompt]

        # Add initial scene
        contents.append("Initial robot scene:")
        contents.append(
            types.Part.from_bytes(
                data=self.encode_image(eval_episode.starting_frame),
                mime_type="image/png",
            )
        )
        contents.append("\nIn the initial robot scene, the task completion percentage is 0. \n")

        counter = 1

        # Add context images with completion percentages
        for ctx_episode_idx, context_episode in enumerate(context_episodes):
            for i, (task_completion, frame) in enumerate(
                zip(context_episode.task_completion_predictions, context_episode.frames)
            ):
                contents.append(f"Frame {counter}: ")
                contents.append(types.Part.from_bytes(data=self.encode_image(frame), mime_type="image/png"))
                contents.append(f"Task Completion Percentage: {task_completion:.1f}% \n")
                # self._to_pil(frame).save(f"images2/example_{ctx_episode_idx+1}__frame_{i+1}_taskcompletion_{task_completion:.1f}.jpg", format="JPEG")
                counter += 1

        # contents.append(
        #     f"Now, for the task of {eval_episode.instruction}" + ", output the task completion percentage for the following frames that are presented in random order. For each frame, format your response as follow: Frame {i}: Frame Description: {}, Task Completion Percentages:{}% \n"
        # )
        contents.append(
            f"Now, for the task of {eval_episode.instruction}" + ", output the task completion percentage for the following frames that are presented in random order. For each frame, format your response as follow: Frame {i}: Task Completion Percentages:{}% \n"
        )
        contents.append(
            f"Be rigorous, precise and remember that the task completion percentage is the percentage of the task that has been completed. \n"
        )
        contents.append(
            f"Remember that the frames are presented in random order. \n"
        )

        # Add eval images
        for i, frame in enumerate(eval_episode.frames):
            contents.append(f"Frame {counter}: ")
            contents.append(types.Part.from_bytes(data=self.encode_image(frame), mime_type="image/png"))
            self._to_pil(frame).save(f"images2/query_frame_{i}_gtcompletion_{eval_episode.task_completion_predictions[i]:.1f}.jpg", format="JPEG")
            contents.append(f"\n")
            counter += 1

        response = self.client.models.generate_content(model=self.model_name, contents=contents)
        return response.text


class ModelFactory:
    """Factory class to create model clients"""

    @staticmethod
    def create_client(model_name: str) -> BaseModelClient:
        # if "gemma" in model_name.lower():
        #     return GemmaClient()
        if "gpt" in model_name.lower():
            return OpenAIClient()
        elif "kimi" in model_name.lower():
            return KimiThinkingClient()
        elif "gemini" in model_name.lower() or "gemma" in model_name.lower():
            return GeminiClient(model_name=model_name)
        else:
            raise ValueError(f"Unknown model: {model_name}")
