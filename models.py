import io
import json
import os
from abc import ABC, abstractmethod
from typing import List

from dotenv import load_dotenv

try:
    from google import genai
    from google.genai import types
except ImportError:
    genai = None

from datasets import Dataset
from PIL import Image

from tools_dataset import BricksDatasetLoader

load_dotenv()


class BaseModelClient(ABC):
    """Base class for all model clients, with a simplified interface."""

    @abstractmethod
    def generate_response(
        self,
        prompt: str,
        selected_frames_dataset: Dataset,
        total_episode_frames: int,
        episode_start_index: int,
    ) -> str:
        """
        Generates a response using a Dataset object directly.

        Args:
            prompt: The base instruction or system prompt for the model.
            selected_frames_dataset: The Dataset object returned by the data loader,
                                     containing the selected frames for the prompt.
            total_episode_frames: The total number of frames in the *original*
                                  episode, required for calculating completion percentage.
            episode_start_index: The starting row index of the original episode,
                                 required to calculate relative step numbers.
        """
        pass


class GeminiClient(BaseModelClient):
    """
    Gemini client that correctly converts PIL Images to byte strings before
    sending them to the API, following official documentation best practices.
    """

    def __init__(
        self, api_key: str | None = None, model_name: str = "gemini-1.5-flash"
    ):
        if genai is None:
            raise ImportError(
                "Google GenAI package not installed. Cannot initialize GeminiClient."
            )

        key = api_key or os.getenv("GEMINI_API_KEY")
        if not key:
            raise ValueError(
                "GEMINI_API_KEY not found in environment or passed to constructor."
            )
        self.model_name = model_name
        self.model = genai.Client(api_key=key)

    def _pil_to_part(self, image: Image.Image) -> types.Part:
        """
        Converts a PIL.Image object to a Gemini API Part by saving it
        to an in-memory byte buffer.
        """
        img_byte_arr = io.BytesIO()
        image.save(img_byte_arr, format="PNG")
        image_bytes = img_byte_arr.getvalue()
        return types.Part.from_bytes(data=image_bytes, mime_type="image/png")

    def generate_response(
        self,
        prompt: str,
        selected_frames_dataset: Dataset,
        total_episode_frames: int,
        episode_start_index: int,
        abolute_indices: List[int],
        incontext: int = 4,
    ) -> str:

        images: List[Image.Image] = [frame["img"] for frame in selected_frames_dataset]
        task_description: str = selected_frames_dataset[0]["goal"]
        absolute_indices: List[int] = abolute_indices
        relative_indices: List[int] = [
            idx - episode_start_index for idx in absolute_indices
        ]

        contents = [prompt]

        contents.append("Initial robot scene:")
        contents.append(self._pil_to_part(images[0]))  # Convert image to bytes
        contents.append(
            "In the initial robot scene, the task completion percentage is 0."
        )

        for i in range(1, incontext + 1):
            if i < len(images):
                step_index = relative_indices[i]
                completion_percentage = (step_index / total_episode_frames) * 100

                contents.append(f"Example {i}:")
                contents.append(self._pil_to_part(images[i]))  # Convert image to bytes
                contents.append(
                    f"Task Completion Percentage: {completion_percentage:.1f}%"
                )

        # Add query instruction
        contents.append(
            f"Now, for the task of '{task_description}', output the task completion percentage "
            "for the following frames. Format: Frame: NUMBER, Description: DESCRIPTION, Task Completion: PERCENTAGE%"
        )

        # Add query images
        if len(images) > incontext + 1:
            for i, image in enumerate(images[incontext + 1 :], 1):
                contents.append(f"Frame {i}:")
                contents.append(self._pil_to_part(image))  # Convert image to bytes

        print("Sending request with converted image bytes to Gemini API...")
        response = self.model.models.generate_content(
            model=self.model_name,
            contents=contents,
        )
        return response.text


if __name__ == "__main__":
    # Example usage
    try:
        client = GeminiClient(
            api_key=os.getenv("GEMINI_API_KEY"),
            model_name="gemini-2.5-flash",
        )
        print("GeminiClient initialized successfully.")
        loader = BricksDatasetLoader()
        loader.load(
            "gberseth/mini-bridge-mini64pix", split="train", trust_remote_code=True
        )

        data_to_collect = {}

        incontext = 5  # Number of in-context examples to provide
        print(f"Collecting data with {incontext} in-context examples...")

        for episode_idx in range(200):

            try:
                episode, total_episode_frames, episode_start_index, absolute_idx = (
                    loader.select_random_frames_from_episode(episode_idx, num_frames=35)
                )
                task_description = episode["goal"][0]
            except ValueError as e:
                print(f"Skipping episode {episode_idx} due to error: {e}")
                continue

            print(f"Processing episode {episode_idx} with task: {task_description}")
            print(f"Absolute indices of selected frames: {absolute_idx}")

            prompt = f"""You are an expert roboticist tasked to predict task completion
            percentages for frames of a robot for the task of {task_description}.
            The task completion percentages are between 0 and 100, where 100
            corresponds to full task completion. We provide several examples of
            the robot performing the task at various stages and their
            corresponding task completion percentages. Note that these frames are
            in random order, so please pay attention to the individual frames
            when reasoning about task completion percentage."""

            response = client.generate_response(
                prompt="Analyze the robot's task completion.",
                selected_frames_dataset=episode,
                total_episode_frames=total_episode_frames,
                episode_start_index=episode_start_index,
                abolute_indices=absolute_idx,
                incontext=incontext,
            )
            print(f"Response for episode {episode_idx}: {response}")

            data_to_collect[episode_idx] = {
                "task_description": task_description,
                "response": response,
                "absolute_indices": absolute_idx,
                "incontext": incontext,
            }

            if episode_idx > 100:
                print(
                    f"Collected data for {episode_idx + 1} episodes. "
                    "Stopping collection to avoid excessive API calls."
                )
                break
        print("Data collection complete. Collected data:")

        output_path = f"collected_data_bridge_incontext_{incontext}.json"
        with open(output_path, "w") as f:
            json.dump(data_to_collect, f, indent=2)
        print(f"Results dumped to {output_path}")
    except ImportError as e:
        print(f"Error: {e}. Please ensure the Google GenAI package is installed.")
