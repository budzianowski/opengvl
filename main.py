"""
Script to load OXE (Open X-Embodiment) robotics data and feed random images
to a task completion percentage prediction prompt.

Download dataset:
gsutil -m cp -r gs://gresearch/robotics/fmb ~/tensorflow_datasets/

Run:
python main.py --name fmb:0.0.1 --max_frames 20 --model gpt4o
"""

import argparse
import base64
import os
import random
import tempfile
from abc import ABC, abstractmethod
from typing import List, Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
import tensorflow_datasets as tfds
from PIL import Image

# Model-specific imports
try:
    import openai
except ImportError:
    openai = None

try:
    import torch
    from transformers import (
        AutoModelForImageTextToText,
        AutoProcessor,
        JanusForConditionalGeneration,
        JanusProcessor,
    )
except ImportError:
    torch = None

try:
    from google import genai
    from google.genai import types
except ImportError:
    genai = None


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
                {"type": "text", "text": f"Initial robot scene:"},
                {
                    "type": "image_url",
                    "image_url": {
                        "url": f"data:image/png;base64,{self.encode_image(image_paths[0])}"
                    },
                },
                {
                    "type": "text",
                    "text": f"In the initial robot scene, the task completion percentage is 0.",
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
                        "image_url": {
                            "url": f"data:image/png;base64,{self.encode_image(path)}"
                        },
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
                        "image_url": {
                            "url": f"data:image/png;base64,{self.encode_image(path)}"
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
        if torch is None:
            raise ImportError("PyTorch and transformers not installed")

        self.device = (
            "mps"
            if torch.backends.mps.is_available()
            else "cuda" if torch.cuda.is_available() else "cpu"
        )
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


class JanusClient(BaseModelClient):
    """Janus client implementation"""

    def __init__(self):
        if torch is None:
            raise ImportError("PyTorch and transformers not installed")

        model_id = "deepseek-community/Janus-Pro-1B"
        self.processor = JanusProcessor.from_pretrained(model_id)
        self.model = JanusForConditionalGeneration.from_pretrained(
            model_id, torch_dtype=torch.bfloat16, device_map="auto"
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

        # Build content following GPT4o structure
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

        messages = [[{"role": "user", "content": content}]]

        inputs = self.processor.apply_chat_template(
            messages,
            add_generation_prompt=True,
            generation_mode="text",
            tokenize=True,
            padding=True,
            return_dict=True,
            return_tensors="pt",
        ).to(self.model.device, dtype=torch.bfloat16)

        output = self.model.generate(
            **inputs, max_new_tokens=400, generation_mode="text", do_sample=False
        )
        text = self.processor.batch_decode(output, skip_special_tokens=True)
        return text[0]


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
        contents.append(
            "In the initial robot scene, the task completion percentage is 0."
        )

        # Add example images with completion percentages
        for i, (idx, path) in enumerate(zip(example_indices, image_paths[:4])):
            contents.append(f"Example {i+1}:")
            with open(path, "rb") as f:
                contents.append(
                    types.Part.from_bytes(data=f.read(), mime_type="image/png")
                )
            contents.append(f"Task Completion Percentage: {idx/total_frames*100:.1f}%")

        # Add query instruction
        contents.append(
            f"Now, for the task of {task_description}, output the task completion percentage for the following frames. Format: Frame: NUMBER, Description: DESCRIPTION, Task Completion: PERCENTAGE%"
        )

        # Add query images
        for i, path in enumerate(image_paths[4:], 1):
            contents.append(f"Frame {i}:")
            with open(path, "rb") as f:
                contents.append(
                    types.Part.from_bytes(data=f.read(), mime_type="image/png")
                )

        response = self.client.models.generate_content(
            model="gemini-2.5-flash", contents=contents
        )
        return response.text


class ModelFactory:
    """Factory class to create model clients"""

    @staticmethod
    def create_client(model_name: str) -> BaseModelClient:
        if model_name.lower() == "gpt4o":
            return GPT4oClient()
        elif model_name.lower() == "internvl":
            return InternVLClient()
        elif model_name.lower() == "janus":
            return JanusClient()
        elif model_name.lower() == "gemini":
            return GeminiClient()
        else:
            raise ValueError(f"Unknown model: {model_name}")


class OXEDataLoader:
    """Loads and processes OXE robotics dataset for task completion analysis."""

    def __init__(self, dataset_name: str = "bridge:0.1.0"):
        """
        Initialize the OXE data loader.

        Args:
            dataset_name: Name of the OXE dataset (e.g., "bridge:0.1.0", "fractal20220817_data")
        """
        self.dataset_name = dataset_name
        self.image_dir = os.path.join(os.getcwd(), "images")
        os.makedirs(self.image_dir, exist_ok=True)

    def load_dataset(self, split: str = "train") -> tf.data.Dataset:
        """Load the OXE dataset."""
        try:
            ds = tfds.load(
                self.dataset_name,
                split=split,
                shuffle_files=False,
            )
            return ds
        except Exception as e:
            print(f"Error loading {self.dataset_name}: {e}")
            raise e

    def extract_episode_images(self, episode) -> List[np.ndarray]:
        """Extract images from a single episode."""
        try:
            images = []
            instr = episode["episode_metadata"]["episode_language_instruction"]
            instr_str = instr.numpy().decode("utf-8")
            for step in episode["steps"]:
                images.append(step["observation"]["image_side_1"])
            return images, instr_str
        except Exception as e:
            print(f"Error extracting images: {e}")
            return []

    def save_images_to_files(
        self, images: List[np.ndarray], prefix: str = "frame"
    ) -> List[str]:
        """Save images to files and return file paths."""
        file_paths = []
        for i, img in enumerate(images):
            # Convert tensor to numpy if needed
            if isinstance(img, tf.Tensor):
                img = img.numpy()

            # Ensure image is in correct format
            if img.dtype != np.uint8:
                img = (
                    (img * 255).astype(np.uint8)
                    if img.max() <= 1.0
                    else img.astype(np.uint8)
                )

            # Save image
            filename = f"{prefix}_{i}.png"
            filepath = os.path.join(self.image_dir, filename)

            pil_img = Image.fromarray(img)
            pil_img.save(filepath)
            file_paths.append(filepath)

        return file_paths

    def select_random_frames(
        self, images: List[np.ndarray], n_frames: int = 3
    ) -> Tuple[List[np.ndarray], List[int]]:
        """Select random frames from the episode."""
        if len(images) < n_frames:
            # If not enough images, use all available
            selected_images = images
            indices = list(range(len(images)))
        else:
            # Randomly sample frames
            indices = sorted(random.sample(range(len(images)), n_frames))
            selected_images = [images[i] for i in indices]

        return selected_images, indices, len(images)


def main(name: str, max_frames: int = 4, model: str = "gpt4o"):
    """Main function to run the OXE data loading and prompt generation."""

    # Initialize the data loader
    print("Initializing OXE data loader...")
    loader = OXEDataLoader(dataset_name=name)

    # Create model client
    print(f"Initializing {model} client...")
    try:
        client = ModelFactory.create_client(model)
    except Exception as e:
        print(f"Error creating {model} client: {e}")
        return

    # Load dataset
    print("Loading dataset...")
    dataset = loader.load_dataset()

    print("Processing episode...")
    for episode in dataset.take(1):
        images, task_description = loader.extract_episode_images(episode)

        if not images:
            print("No images found in episode")
            continue

        print(f"Found {len(images)} images in episode")
        print(f"Task: {task_description}")

        # Select random frames (including initial frame)
        n_frames = min(max_frames, len(images))
        selected_images, selected_indices, total_frames = loader.select_random_frames(
            images, n_frames
        )

        print(f"Selected {len(selected_images)} frames at indices: {selected_indices}")

        # Save images to files
        image_paths = loader.save_images_to_files(selected_images)

        # Generate response using selected model
        print(f"\nSending to {model}...")
        try:
            prompt = f"""You are an expert roboticist tasked to predict task completion
            percentages for frames of a robot for the task of {task_description}.
            The task completion percentages are between 0 and 100, where 100
            corresponds to full task completion. We provide several examples of
            the robot performing the task at various stages and their
            corresponding task completion percentages. Note that these frames are
            in random order, so please pay attention to the individual frames
            when reasoning about task completion percentage."""
            response = client.generate_response(
                prompt=prompt,
                image_paths=image_paths,
                task_description=task_description,
                example_indices=selected_indices[:4],
                selected_indices=selected_indices[4:],
                total_frames=total_frames,
            )

            print("\n" + "=" * 80)
            print(f"{model.upper()} RESPONSE:")
            print("=" * 80)
            print(response)
            print("=" * 80)

            # Print ground truth for comparison
            print("\nGround Truth Completion Percentages:")
            for i, idx in enumerate(selected_indices):
                completion = idx / total_frames * 100
                print(f"Frame {i+1} (step {idx}): {completion:.1f}%")

        except Exception as e:
            print(f"Error generating response: {e}")

        break

    print(f"\nImages saved in: {loader.image_dir}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--name", type=str, default="fmb:0.0.1", help="Dataset name")
    parser.add_argument(
        "--max_frames", type=int, default=4, help="Maximum number of frames to select"
    )
    parser.add_argument(
        "--model",
        type=str,
        default="gpt4o",
        choices=["gpt4o", "internvl", "janus", "gemini"],
        help="Model to use for inference",
    )
    args = parser.parse_args()
    main(args.name, args.max_frames, args.model)
