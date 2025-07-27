"""Model clients for the different models."""

import base64
import io
import os
from abc import abstractmethod
from typing import List

import numpy as np
import torch
from loguru import logger
from PIL import Image
from transformers import (
    AutoProcessor,
    Gemma3ForConditionalGeneration,
)
from dotenv import load_dotenv
from .data_loader import Episode

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

try:
    from vllm import LLM, SamplingParams
except ImportError:
    LLM = None
    SamplingParams = None


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
            return image.resize((224, 224))

        if hasattr(image, "detach"):  # PyTorch tensor
            image = image.cpu().detach().numpy()

        if isinstance(image, np.ndarray):
            if image.dtype == np.float32 or image.dtype == np.float64:
                image = (image * 255).astype(np.uint8)
            if len(image.shape) == 3 and image.shape[0] in [1, 3, 4]:
                image = np.transpose(image, (1, 2, 0))

            if len(image.shape) == 3 and image.shape[2] == 1:
                pil_image = Image.fromarray(image.squeeze(), "L")
            else:
                pil_image = Image.fromarray(image, "RGB")
            return pil_image.resize((224, 224))
        raise ValueError(f"Unsupported image type: {type(image)}")

    def encode_image(self, image) -> str:
        """Encode image to base64 string for API calls"""
        pil_image = self._to_pil(image)
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
        content = [{"type": "text", "text": prompt}]
        content.extend(
            [
                {"type": "text", "text": "Initial robot scene:"},
                {
                    "type": "image_url",
                    "image_url": {"url": f"data:image/png;base64,{self.encode_image(eval_episode.starting_frame)}", "detail": self.detail},
                },
                {"type": "text", "text": "In the initial robot scene, the task completion percentage is 0."},
            ]
        )

        for ctx_episode_idx, context_episode in enumerate(context_episodes):
            content.extend(
                [
                    {"type": "text", "text": f"Example episode {ctx_episode_idx+1}."},
                    {"type": "text", "text": f"Instruction: {context_episode.instruction}"},
                ]
            )
            for i, (task_completion, frame) in enumerate(
                zip(context_episode.task_completion_predictions, context_episode.frames)
            ):
                content.extend(
                    [
                        {"type": "text", "text": f"Frame {i+1}: "},
                        {
                            "type": "image_url",
                            "image_url": {"url": f"data:image/png;base64,{self.encode_image(frame)}", "detail": self.detail},
                        },
                        {"type": "text", "text": f"Task Completion Percentage: {task_completion:.1f}%"},
                    ]
                )
        content.append(
            {
                "type": "text",
                "text": f"Now, for the task of {eval_episode.instruction}, output the task completion percentage for the following frames. Format: Frame: NUMBER, Description: DESCRIPTION, Task Completion: PERCENTAGE%",
            }
        )
        for i, frame in enumerate(eval_episode.frames, 1):
            content.extend(
                [
                    {"type": "text", "text": f"Frame {i}: "},
                    {
                        "type": "image_url",
                        "image_url": {"url": f"data:image/png;base64,{self.encode_image(frame)}", "detail": self.detail},
                    },
                ]
            )

        messages = [{"role": "user", "content": content}]
        response = self.client.chat.completions.create(
            model=self.model_id, messages=messages, max_tokens=self.max_new_tokens
        )
        return response.choices[0].message.content


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
        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": prompt},
                    {"type": "text", "text": "Initial robot scene:"},
                    {"type": "image", "image": self._to_pil(eval_episode.starting_frame)},
                    {"type": "text", "text": "In the initial robot scene, the task completion percentage is 0."},
                ],
            }
        ]

        for ctx_episode_idx, context_episode in enumerate(context_episodes):
            messages[0]["content"].extend(
                [
                    {"type": "text", "text": f"Example episode {ctx_episode_idx+1}."},
                    {"type": "text", "text": f"Instruction: {context_episode.instruction}"},
                ]
            )
            for i, (task_completion, frame) in enumerate(
                zip(context_episode.task_completion_predictions, context_episode.frames)
            ):
                messages[0]["content"].extend(
                    [
                        {"type": "text", "text": f"Frame {i+1}: "},
                        {"type": "image", "image": self._to_pil(frame)},
                        {"type": "text", "text": f"Task Completion Percentage: {task_completion:.1f}%"},
                    ]
                )

        messages[0]["content"].append(
            {
                "type": "text",
                "text": f"Now, for the task of {eval_episode.instruction}, output the task completion percentage for the following frames. Format: Frame: NUMBER, Description: DESCRIPTION, Task Completion: PERCENTAGE%\n",
            }
        )
        for i, frame in enumerate(eval_episode.frames, 1):
            messages[0]["content"].extend(
                [
                    {"type": "text", "text": f"Frame {i}: "},
                    {"type": "image", "image": self._to_pil(frame)},
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

        with torch.inference_mode():
            output = self.model.generate(**inputs, max_new_tokens=self.max_new_tokens, do_sample=False)
            output = output[0][input_len:]

        return self.processor.decode(output, skip_special_tokens=True)


class KimiThinkingClient(BaseModelClient):
    """Kimi Thinking client implementation"""

    def __init__(self, model_id: str = "moonshotai/Kimi-VL-A3B-Thinking-2506"):
        if LLM is None or SamplingParams is None:
            raise ImportError("vllm is not installed. Please install it with 'pip install vllm'.")
        logger.info(f"Loading Kimi Thinking model {model_id}...")
        self.model = LLM(
            model=model_id, trust_remote_code=True, max_num_seqs=8, max_model_len=131072, limit_mm_per_prompt={"image": 256}
        )
        self.processor = AutoProcessor.from_pretrained(model_id, trust_remote_code=True)
        self.sampling_params = SamplingParams(max_tokens=32768, temperature=0.8)

    def extract_thinking_and_summary(self, text: str, bot: str = "◁think▷", eot: str = "◁/think▷") -> str:
        if bot in text and eot not in text:
            return ""
        if eot in text:
            return text[text.index(bot) + len(bot) : text.index(eot)].strip(), text[text.index(eot) + len(eot) :].strip()
        return "", text

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
                    {"type": "image", "image": self._to_pil(eval_episode.starting_frame)},
                    {"type": "text", "text": "In the initial robot scene, the task completion percentage is 0."},
                ],
            }
        ]
        for ctx_episode_idx, context_episode in enumerate(context_episodes):
            messages[0]["content"].extend(
                [
                    {"type": "text", "text": f"Example episode {ctx_episode_idx+1}."},
                    {"type": "text", "text": f"Instruction: {context_episode.instruction}"},
                ]
            )
            for i, (task_completion, frame) in enumerate(
                zip(context_episode.task_completion_predictions, context_episode.frames)
            ):
                messages[0]["content"].extend(
                    [
                        {"type": "text", "text": f"Frame {i+1}: "},
                        {"type": "image", "image": self._to_pil(frame)},
                        {"type": "text", "text": f"Task Completion Percentage: {task_completion:.1f}%"},
                    ]
                )
        messages[0]["content"].append(
            {
                "type": "text",
                "text": f"Now, for the task of {eval_episode.instruction}, output the task completion percentage for the following frames. Format: Frame: NUMBER, Description: DESCRIPTION, Task Completion: PERCENTAGE%\n",
            }
        )
        for i, frame in enumerate(eval_episode.frames, 1):
            messages[0]["content"].extend(
                [
                    {"type": "text", "text": f"Frame {i}: "},
                    {"type": "image", "image": self._to_pil(frame)},
                ]
            )

        inputs = self.processor.apply_chat_template(
            messages, add_generation_prompt=True, tokenize=False
        )
        images = [self._to_pil(eval_episode.starting_frame)]
        for ctx in context_episodes:
            images.extend([self._to_pil(f) for f in ctx.frames])
        images.extend([self._to_pil(f) for f in eval_episode.frames])

        outputs = self.model.generate(
            [{"prompt": inputs, "multi_modal_data": {"image": images}}],
            sampling_params=self.sampling_params,
        )
        generated_text = outputs[0].outputs[0].text
        _, summary = self.extract_thinking_and_summary(generated_text)
        return summary


class GeminiClient(BaseModelClient):
    """Gemini client implementation"""

    def __init__(self, model_name: str = "gemini-2.5-flash-lite-preview-06-17"):
        if genai is None:
            raise ImportError("Google GenAI package not installed")
        self.client = genai.GenerativeModel(model_name=model_name, api_key=os.getenv("GEMINI_API_KEY"))
        self.model_name = model_name

    def generate_response(
        self,
        prompt: str,
        eval_episode: Episode,
        context_episodes: List[Episode],
    ) -> str:
        contents = [prompt]
        contents.append("Initial robot scene:")
        contents.append(self._to_pil(eval_episode.starting_frame))
        contents.append("\nIn the initial robot scene, the task completion percentage is 0. \n")

        counter = 1
        for ctx_episode_idx, context_episode in enumerate(context_episodes):
            for i, (task_completion, frame) in enumerate(
                zip(context_episode.task_completion_predictions, context_episode.frames)
            ):
                contents.append(f"Frame {counter}: ")
                contents.append(self._to_pil(frame))
                contents.append(f"Task Completion Percentage: {task_completion:.1f}% \n")
                counter += 1
        contents.append(
            f"Now, for the task of {eval_episode.instruction}, output the task completion percentage for the following frames that are presented in random order. For each frame, format your response as follow: Frame {{i}}: Task Completion Percentages:{{}}% \n"
        )
        contents.append(
            "Be rigorous, precise and remember that the task completion percentage is the percentage of the task that has been completed. \n"
        )
        contents.append("Remember that the frames are presented in random order. \n")
        for i, frame in enumerate(eval_episode.frames):
            contents.append(f"Frame {counter}: ")
            contents.append(self._to_pil(frame))
            contents.append("\n")
            counter += 1

        response = self.client.generate_content(contents)
        return response.text


class ModelFactory:
    """Factory class to create model clients"""

    @staticmethod
    def create_client(model_name: str) -> BaseModelClient:
        if "gpt" in model_name.lower():
            return OpenAIClient(model_id=model_name)
        elif "kimi" in model_name.lower():
            return KimiThinkingClient(model_id=model_name)
        elif "gemma" in model_name.lower():
            return GemmaClient(model_id=model_name)
        elif "gemini" in model_name.lower():
            return GeminiClient(model_name=model_name)
        else:
            raise ValueError(f"Unknown model: {model_name}")
