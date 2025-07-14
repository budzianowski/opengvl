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
from transformers import (AutoConfig, AutoModel, AutoModelForCausalLM,
                          AutoModelForImageTextToText, AutoModelForVision2Seq,
                          AutoProcessor, AutoTokenizer, BitsAndBytesConfig,
                          Gemma3ForConditionalGeneration,
                          Qwen2VLForConditionalGeneration)

from data_loader import Episode

# third-party imports
try:
    import openai
except ImportError:
    openai = None

try:
    from deepseek_vl.models import MultiModalityCausalLM, VLChatProcessor
    from deepseek_vl.utils.io import load_pil_images as deepseek_load_pil_images
except ImportError:
    MultiModalityCausalLM = None
    VLChatProcessor = None
    deepseek_load_pil_images = None

try:
    from google import genai
    from google.genai import types
except ImportError:
    genai = None


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
                    "image_url": {
                        "url": f"data:image/png;base64,{self.encode_image(eval_episode.starting_frame)}"
                    },
                },
                {
                    "type": "text",
                    "text": "In the initial robot scene, the task completion percentage is 0.",
                },
            ]
        )

        logger.info(f"Frame type: {type(eval_episode.starting_frame)}")
        logger.info(f"Frame: {eval_episode.starting_frame}")
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
                        {
                            "type": "image_url",
                            "image_url": {
                                "url": f"data:image/png;base64,{self.encode_image(frame)}"
                            },
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
                        "image_url": {
                            "url": f"data:image/png;base64,{self.encode_image(frame)}"
                        },
                    },
                ]
            )
        response = self.client.chat.completions.create(
            model="gpt-4o", messages=messages
        )
        return response.choices[0].message.content


class InternVLClient(BaseModelClient):
    """InternVL client implementation"""

    def __init__(self):
        path = "OpenGVLab/InternVL3-8B"
        device_map = self.split_model(path)

        quantization_config = BitsAndBytesConfig(load_in_4bit=True)
        self.processor = AutoProcessor.from_pretrained(path, trust_remote_code=True)
        self.model = AutoModel.from_pretrained(
            path,
            torch_dtype=torch.bfloat16,
            low_cpu_mem_usage=True,
            use_flash_attn=True,
            trust_remote_code=True,
            device_map=device_map,
        ).eval()

    IMAGENET_MEAN = (0.485, 0.456, 0.406)
    IMAGENET_STD = (0.229, 0.224, 0.225)

    @staticmethod
    def build_transform(input_size):
        MEAN, STD = InternVLClient.IMAGENET_MEAN, InternVLClient.IMAGENET_STD
        transform = T.Compose(
            [
                T.Lambda(lambda img: img.convert("RGB") if img.mode != "RGB" else img),
                T.Resize(
                    (input_size, input_size), interpolation=InterpolationMode.BICUBIC
                ),
                T.ToTensor(),
                T.Normalize(mean=MEAN, std=STD),
            ]
        )
        return transform

    @staticmethod
    def find_closest_aspect_ratio(
        aspect_ratio, target_ratios, width, height, image_size
    ):
        best_ratio_diff = float("inf")
        best_ratio = (1, 1)
        area = width * height
        for ratio in target_ratios:
            target_aspect_ratio = ratio[0] / ratio[1]
            ratio_diff = abs(aspect_ratio - target_aspect_ratio)
            if ratio_diff < best_ratio_diff:
                best_ratio_diff = ratio_diff
                best_ratio = ratio
            elif ratio_diff == best_ratio_diff:
                if area > 0.5 * image_size * image_size * ratio[0] * ratio[1]:
                    best_ratio = ratio
        return best_ratio

    @staticmethod
    def dynamic_preprocess(
        image, min_num=1, max_num=12, image_size=448, use_thumbnail=False
    ):
        orig_width, orig_height = image.size
        aspect_ratio = orig_width / orig_height

        # calculate the existing image aspect ratio
        target_ratios = set(
            (i, j)
            for n in range(min_num, max_num + 1)
            for i in range(1, n + 1)
            for j in range(1, n + 1)
            if i * j <= max_num and i * j >= min_num
        )
        target_ratios = sorted(target_ratios, key=lambda x: x[0] * x[1])

        # find the closest aspect ratio to the target
        target_aspect_ratio = InternVLClient.find_closest_aspect_ratio(
            aspect_ratio, target_ratios, orig_width, orig_height, image_size
        )

        # calculate the target width and height
        target_width = image_size * target_aspect_ratio[0]
        target_height = image_size * target_aspect_ratio[1]
        blocks = target_aspect_ratio[0] * target_aspect_ratio[1]

        # resize the image
        resized_img = image.resize((target_width, target_height))
        processed_images = []
        for i in range(blocks):
            box = (
                (i % (target_width // image_size)) * image_size,
                (i // (target_width // image_size)) * image_size,
                ((i % (target_width // image_size)) + 1) * image_size,
                ((i // (target_width // image_size)) + 1) * image_size,
            )
            # split the image
            split_img = resized_img.crop(box)
            processed_images.append(split_img)
        assert len(processed_images) == blocks
        if use_thumbnail and len(processed_images) != 1:
            thumbnail_img = image.resize((image_size, image_size))
            processed_images.append(thumbnail_img)
        return processed_images

    def load_image(self, image, input_size=448, max_num=12):
        transform = self.build_transform(input_size=input_size)
        images = self.dynamic_preprocess(
            image, image_size=input_size, use_thumbnail=True, max_num=max_num
        )
        pixel_values = [transform(image) for image in images]
        pixel_values = torch.stack(pixel_values)
        return pixel_values

    def split_model(self, model_path):
        device_map = {}
        world_size = torch.cuda.device_count()
        print(f"World size: {world_size}")
        config = AutoConfig.from_pretrained(model_path, trust_remote_code=True)
        num_layers = config.llm_config.num_hidden_layers
        # Since the first GPU will be used for ViT, treat it as half a GPU.
        num_layers_per_gpu = math.ceil(num_layers / (world_size - 0.5))
        num_layers_per_gpu = [num_layers_per_gpu] * world_size
        num_layers_per_gpu[0] = math.ceil(num_layers_per_gpu[0] * 0.5)
        layer_cnt = 0
        for i, num_layer in enumerate(num_layers_per_gpu):
            for j in range(num_layer):
                device_map[f"language_model.model.layers.{layer_cnt}"] = i
                layer_cnt += 1
        device_map["vision_model"] = 0
        device_map["mlp1"] = 0
        device_map["language_model.model.tok_embeddings"] = 0
        device_map["language_model.model.embed_tokens"] = 0
        device_map["language_model.output"] = 0
        device_map["language_model.model.norm"] = 0
        device_map["language_model.model.rotary_emb"] = 0
        device_map["language_model.lm_head"] = 0
        if num_layers > 0:
            device_map[f"language_model.model.layers.{num_layers - 1}"] = 0

        return device_map

    def generate_response(
        self,
        prompt: str,
        eval_episode: Episode,
        context_episodes: List[Episode],
    ) -> str:
        pil_images = []
        text_prompt = prompt

        # Add initial scene
        text_prompt += "\nInitial robot scene:<image>"
        pil_images.append(self._to_pil(eval_episode.starting_frame))
        text_prompt += (
            "\nIn the initial robot scene, the task completion percentage is 0."
        )

        # Add example images with completion percentages
        for ctx_episode_idx, context_episode in enumerate(context_episodes):
            text_prompt += f"\nExample episode {ctx_episode_idx+1}.\n"
            text_prompt += f"Instruction: {context_episode.instruction}\n"

            for i, (task_completion, frame) in enumerate(
                zip(context_episode.task_completion_predictions, context_episode.frames)
            ):
                text_prompt += f"Frame {i+1}: <image>"
                pil_images.append(self._to_pil(frame))
                text_prompt += f" Task Completion Percentage: {task_completion:.1f}%"

        # Add query instruction
        text_prompt += f"\nNow, for the task of {eval_episode.instruction}, output the task completion percentage (with the format Format: Frame: NUMBER, Description: DESCRIPTION, Task Completion: PERCENTAGE%) for the following frames:"

        # Add query images
        for i, frame in enumerate(eval_episode.frames, 1):
            text_prompt += f"\nFrame {i}: <image>"
            pil_images.append(self._to_pil(frame))

        pixel_values_list = []
        for img in pil_images:
            pixel_values = self.load_image(img).to(torch.bfloat16)
            pixel_values_list.append(pixel_values)

        pixel_values = torch.cat(pixel_values_list, dim=0)
        num_patches_list = [pv.size(0) for pv in pixel_values_list]

        generation_config = dict(
            num_beams=1,
            max_new_tokens=1024,
            do_sample=False,
        )

        response, history = self.model.chat(
            self.processor,
            pixel_values,
            text_prompt,
            generation_config,
            num_patches_list=num_patches_list,
            history=None,
            return_history=True,
        )
        return response


class SmolVLMClient(BaseModelClient):
    """SmolVLM client implementation"""

    def __init__(self):
        if torch is None:
            raise ImportError("PyTorch and transformers not installed")

        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        model_checkpoint = "HuggingFaceTB/SmolVLM-Instruct"

        self.processor = AutoProcessor.from_pretrained(model_checkpoint)
        self.model = AutoModelForVision2Seq.from_pretrained(
            model_checkpoint,
            torch_dtype=torch.bfloat16,
            _attn_implementation=(
                "flash_attention_2" if self.device == "cuda" else "eager"
            ),
        ).to(self.device)

    def generate_response(
        self,
        prompt: str,
        eval_episode: Episode,
        context_episodes: List[Episode],
    ) -> str:

        images = []
        content = [{"type": "text", "text": prompt}]

        # Add initial scene
        content.extend(
            [
                {"type": "text", "text": "Initial robot scene:"},
                {"type": "image"},
                {
                    "type": "text",
                    "text": "In the initial robot scene, the task completion percentage is 0.",
                },
            ]
        )
        images.append(eval_episode.starting_frame)

        # Add example images with completion percentages
        for ctx_episode_idx, context_episode in enumerate(context_episodes):
            content.extend(
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
                content.extend(
                    [
                        {"type": "text", "text": f"Frame {i+1}: "},
                        {"type": "image"},
                        {
                            "type": "text",
                            "text": f"Task Completion Percentage: {task_completion:.1f}%",
                        },
                    ]
                )
                images.append(frame)

        # Add query instruction
        content.append(
            {
                "type": "text",
                "text": f"Now, for the task of {eval_episode.instruction}, output the task completion percentage for the following frames. Format: Frame: NUMBER, Description: DESCRIPTION, Task Completion: PERCENTAGE%",
            }
        )

        # Add query images
        for i, frame in enumerate(eval_episode.frames, 1):
            content.extend(
                [
                    {"type": "text", "text": f"Frame {i}: "},
                    {"type": "image"},
                ]
            )
            images.append(frame)

        messages = [{"role": "user", "content": content}]

        pil_images = [self._to_pil(img) for img in images]

        prompt_text = self.processor.apply_chat_template(
            messages, add_generation_prompt=True
        )
        inputs = self.processor(
            text=prompt_text, images=pil_images, return_tensors="pt"
        ).to(self.device)

        generated_ids = self.model.generate(
            **inputs, max_new_tokens=self.max_new_tokens
        )
        generated_texts = self.processor.batch_decode(
            generated_ids,
            skip_special_tokens=True,
        )

        return generated_texts[0]


class DeepseekClient(BaseModelClient):
    """Deepseek client implementation"""

    def __init__(self):
        if torch is None or MultiModalityCausalLM is None or VLChatProcessor is None:
            raise ImportError("PyTorch and deepseek-vl not installed")

        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        model_path = "deepseek-ai/deepseek-vl-1.3b-chat"

        self.vl_chat_processor: VLChatProcessor = VLChatProcessor.from_pretrained(
            model_path
        )
        self.tokenizer = self.vl_chat_processor.tokenizer

        self.model: MultiModalityCausalLM = AutoModelForCausalLM.from_pretrained(
            model_path, trust_remote_code=True
        )
        self.model = self.model.to(torch.bfloat16).to(self.device).eval()

    def generate_response(
        self,
        prompt: str,
        eval_episode: Episode,
        context_episodes: List[Episode],
    ) -> str:

        temp_dir = tempfile.mkdtemp()
        image_paths = []

        try:
            # Build the conversation for Deepseek-VL, saving images to temporary files
            conversation_content = prompt + "\nInitial robot scene: <image_placeholder>"

            # Save initial frame and get path
            initial_frame_path = os.path.join(temp_dir, "initial_frame.png")
            self._to_pil(eval_episode.starting_frame).save(initial_frame_path)
            image_paths.append(initial_frame_path)

            conversation_content += (
                "\nIn the initial robot scene, the task completion percentage is 0."
            )

            # Add context episodes
            for ctx_episode_idx, context_episode in enumerate(context_episodes):
                conversation_content += f"\nExample episode {ctx_episode_idx+1}.\nInstruction: {context_episode.instruction}\n"
                for i, (task_completion, frame) in enumerate(
                    zip(
                        context_episode.task_completion_predictions,
                        context_episode.frames,
                    )
                ):
                    conversation_content += f"Frame {i+1}: <image_placeholder> Task Completion Percentage: {task_completion:.1f}%\n"
                    frame_path = os.path.join(
                        temp_dir, f"ctx_{ctx_episode_idx}_frame_{i}.png"
                    )
                    self._to_pil(frame).save(frame_path)
                    image_paths.append(frame_path)

            # Add query instruction
            conversation_content += f"\nNow, for the task of {eval_episode.instruction}, output the task completion percentage for the following frames. Format: Frame: NUMBER, Description: DESCRIPTION, Task Completion: PERCENTAGE%\n"

            # Add query images
            for i, frame in enumerate(eval_episode.frames, 1):
                conversation_content += f"Frame {i}: <image_placeholder>\n"
                frame_path = os.path.join(temp_dir, f"query_frame_{i}.png")
                self._to_pil(frame).save(frame_path)
                image_paths.append(frame_path)

            conversation = [
                {
                    "role": "User",
                    "content": conversation_content,
                    "images": image_paths,
                },
                {"role": "Assistant", "content": ""},
            ]

            # load images and prepare for inputs
            pil_images = deepseek_load_pil_images(conversation)
            prepare_inputs = self.vl_chat_processor(
                conversations=conversation, images=pil_images, force_batchify=True
            ).to(self.model.device)

            # run image encoder to get the image embeddings
            inputs_embeds = self.model.prepare_inputs_embeds(**prepare_inputs)

            # run the model to get the response
            outputs = self.model.language_model.generate(
                inputs_embeds=inputs_embeds,
                attention_mask=prepare_inputs.attention_mask,
                pad_token_id=self.tokenizer.eos_token_id,
                bos_token_id=self.tokenizer.bos_token_id,
                eos_token_id=self.tokenizer.eos_token_id,
                max_new_tokens=self.max_new_tokens,
                do_sample=False,
                use_cache=True,
            )

            answer = self.tokenizer.decode(
                outputs[0].cpu().tolist(), skip_special_tokens=True
            )
            return answer
        finally:
            # Clean up temporary directory
            for path in image_paths:
                if os.path.exists(path):
                    os.remove(path)
            os.rmdir(temp_dir)


class QwenClient(BaseModelClient):
    """Qwen client implementation"""

    def __init__(self):
        if torch is None:
            raise ImportError("PyTorch and transformers not installed")

        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        model_id = "Qwen/Qwen2-VL-2B-Instruct"

        quantization_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.bfloat16,
            bnb_4bit_use_double_quant=True,
        )

        self.model = Qwen2VLForConditionalGeneration.from_pretrained(
            model_id,
            torch_dtype=torch.bfloat16,
            # quantization_config=quantization_config,
            attn_implementation="flash_attention_2",
            device_map="auto",
        )
        self.processor = AutoProcessor.from_pretrained(model_id)

    def generate_response(
        self,
        prompt: str,
        eval_episode: Episode,
        context_episodes: List[Episode],
    ) -> str:

        pil_images = []
        messages = [
            {
                "role": "system",
                "content": [{"type": "text", "text": "You are a helpful assistant."}],
            }
        ]

        # Initial prompt
        user_content = [{"type": "text", "text": prompt}]

        # Assistant response for initial prompt

        def _process_and_get_image_content(frame):
            try:
                pil_images.append(self._to_pil(frame))
                return {"type": "image"}
            except ValueError:
                print(f"[ERROR] Image not loaded")
                return []

        # Add initial scene
        user_content.append({"type": "text", "text": "Initial robot scene: "})
        user_content.append(_process_and_get_image_content(eval_episode.starting_frame))

        # Add example images with completion percentages
        for ctx_episode_idx, context_episode in enumerate(context_episodes):
            user_content.append(
                {
                    "type": "text",
                    "text": f"Example episode {ctx_episode_idx+1}. Instruction: {context_episode.instruction}\n",
                }
            )

            for i, (task_completion, frame) in enumerate(
                zip(context_episode.task_completion_predictions, context_episode.frames)
            ):
                user_content.extend(
                    [
                        {"type": "text", "text": f"Frame {i+1}: "},
                        _process_and_get_image_content(frame),
                        {
                            "type": "text",
                            "text": f"Task Completion Percentage: {task_completion:.1f}% \n",
                        },
                    ]
                )

        # Add query instruction
        user_content.append(
            {
                "type": "text",
                "text": f"Now, for the task of {eval_episode.instruction}, output the task completion percentage for the following frames. Format: Frame: NUMBER, Description: DESCRIPTION, Task Completion: PERCENTAGE%'n",
            }
        )

        # Add query images
        for i, frame in enumerate(eval_episode.frames, 1):
            user_content.extend(
                [
                    {"type": "text", "text": f"Frame {i}: "},
                    _process_and_get_image_content(frame),
                ]
            )

        messages.append({"role": "user", "content": user_content})

        text = self.processor.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )

        print(f"Text: {text}")
        inputs = self.processor(
            text=[text],
            images=pil_images,
            padding=True,
            return_tensors="pt",
        )
        inputs = inputs.to(self.device)

        generated_ids = self.model.generate(
            **inputs, max_new_tokens=self.max_new_tokens
        )
        generated_ids_trimmed = [
            out_ids[len(in_ids) :]
            for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
        ]
        response = self.processor.batch_decode(
            generated_ids_trimmed, skip_special_tokens=True
        )[0]
        return response


class GemmaClient(BaseModelClient):
    """Gemma client implementation"""

    def __init__(self, model_id: str = "google/gemma-3-4b-it"):

        logger.info(f"Loading Gemma model {model_id}...")
        self.model = Gemma3ForConditionalGeneration.from_pretrained(
            model_id, device_map="auto"
        ).eval()
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
                        {"type": "image", "base64": self.encode_image(frame)},
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
                "text": f"Now, for the task of {eval_episode.instruction}, output the task completion percentage for the following frames. Format: Frame: NUMBER, Description: DESCRIPTION, Task Completion: PERCENTAGE%",
            }
        )

        # Add query images
        for i, frame in enumerate(eval_episode.frames, 1):
            messages[0]["content"].extend(
                [
                    {"type": "text", "text": f"Frame {i}: "},
                    {"type": "image", "base64": self.encode_image(frame)},
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
        print(f"Token length: {input_len}")

        if input_len > 32000:
            raise ValueError(
                f"Input length {input_len} exceeds maximum allowed length of 32000 tokens."
            )

        with torch.inference_mode():
            output = self.model.generate(**inputs, max_new_tokens=1024, do_sample=False)
            output = output[0][input_len:]

        decoded_outputs = self.processor.decode(output, skip_special_tokens=True)
        return decoded_outputs


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
        contents.append(
            types.Part.from_bytes(
                data=self.encode_image(eval_episode.starting_frame),
                mime_type="image/png",
            )
        )
        contents.append(
            "In the initial robot scene, the task completion percentage is 0."
        )

        # Add example images with completion percentages
        for ctx_episode_idx, context_episode in enumerate(context_episodes):
            contents.append(f"Example episode {ctx_episode_idx+1}.")
            contents.append(f"Instruction: {context_episode.instruction}")

            for i, (task_completion, frame) in enumerate(
                zip(context_episode.task_completion_predictions, context_episode.frames)
            ):
                contents.append(f"Frame {i+1}: ")
                contents.append(
                    types.Part.from_bytes(
                        data=self.encode_image(frame), mime_type="image/png"
                    )
                )
                contents.append(f"Task Completion Percentage: {task_completion:.1f}%")

        # Add query instruction
        contents.append(
            f"Now, for the task of {eval_episode.instruction}, output the task completion percentage for the following frames. Format: Frame: NUMBER, Description: DESCRIPTION, Task Completion: PERCENTAGE%"
        )

        # Add query images
        for i, frame in enumerate(eval_episode.frames, 1):
            contents.append(f"Frame {i}: ")
            contents.append(
                types.Part.from_bytes(
                    data=self.encode_image(frame), mime_type="image/png"
                )
            )

        response = self.client.models.generate_content(
            model=self.model_name, contents=contents
        )
        return response.text


class ModelFactory:
    """Factory class to create model clients"""

    @staticmethod
    def create_client(model_name: str) -> BaseModelClient:
        if model_name.lower() == "internvl":
            return InternVLClient()
        elif model_name.lower() == "smolvlm":
            return SmolVLMClient()
        elif model_name.lower() == "qwen":
            return QwenClient()
        elif model_name.lower() == "deepseek":
            return DeepseekClient()
        elif model_name.lower() == "gemma":
            return GemmaClient()
        elif model_name.lower() == "gemini":
            return GeminiClient()
        elif model_name.lower() == "gpt4o":
            return GPT4oClient()

        else:
            raise ValueError(f"Unknown model: {model_name}")
