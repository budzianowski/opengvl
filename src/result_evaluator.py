""" Result Evaluator """

from transformers import AutoTokenizer, AutoModelForCausalLM
import json
from dataclasses import dataclass
import torch
import re

ERROR_MESSAGE = "No task completion percentage found!!"

PROMPT = f"""
You are a helpful assistant that extracts the task completion percentage from the response.
You will be given a response, you need to extract the task completion percentage from the response.

Here is and example:
### example start ###
Okay, let's analyze the task completion percentages for the “Pick up the blue object” scenario, given the provided examples.

**Task Completion Percentage for “Pick up the blue object”**

Here's the predicted task completion percentage for each frame, formatted as requested:

Frame 1: 61.0%
Frame 2: 41.0%
Frame 3: 24.0%
Frame 4: 93.0%
Frame 5: 33.0%
Frame 6: 45.0%
Frame 7: 44.0%
Frame 8: 82.0%
Frame 9: 8.0%
Frame 10: 61.0%

The proper response:
[61, 41, 24, 93, 33, 45, 44, 82, 8, 61]
### example end ###

Format the response as a list of task completion percentages for each frame. DO NOT ADD ANYTHING ELSE.
If there is no task completion percentage in the response, you need to return only {ERROR_MESSAGE}.
"""


@dataclass
class Result:
    prompt: str
    response: str
    ground_truth: str
    voc_score: float


class ResultEvaluator:
    def __init__(self, model_name: str = "Qwen/Qwen2.5-1.5B-Instruct", max_new_tokens: int = 300):
        self.model_name = model_name
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForCausalLM.from_pretrained(model_name)
        self.max_new_tokens = max_new_tokens

    def evaluate(self, response: str) -> str:
        """Evaluate the response and return the task completion percentage."""
        messages = [
            [
                {"role": "system", "content": PROMPT},
                {"role": "user", "content": response},
            ],
        ]
        inputs = self.tokenizer.apply_chat_template(
            messages,
            add_generation_prompt=True,
            tokenize=True,
            return_dict=True,
            return_tensors="pt",
        ).to(self.model.device).to(torch.bfloat16)
        input_length = inputs.input_ids.shape[1]
    
        with torch.inference_mode():
            outputs = self.model.generate(**inputs, max_new_tokens=self.max_new_tokens)

        new_tokens = outputs[0][input_length:]
        assistant_response = self.tokenizer.decode(new_tokens, skip_special_tokens=True)
        
        return self._extract_task_completion_percentage(assistant_response)
    
    def _extract_task_completion_percentage(self, response: str) -> list[float]:
        """Extract task completion percentage from the response."""
        list_pattern = r'\[([^\]]+)\]'
        list_match = re.search(list_pattern, response)
        
        if list_match:
            numbers_str = list_match.group(1)
            numbers = re.findall(r'\d+\.?\d*', numbers_str)
            try:
                return [float(num) for num in numbers]
            except ValueError:
                raise ValueError(f"Invalid numbers in the response: {numbers_str}")


if __name__ == "__main__":
    result_evaluator = ResultEvaluator()
    result = result_evaluator.evaluate("""
    Frame 1: 61.0%
    Frame 2: 41.0%
    Frame 3: 24.0%
    Frame 4: 93.0%
    Frame 5: 33.0%
    Frame 6: 45.0%
    Frame 7: 44.0%
    Frame 8: 82.0%
    Frame 9: 8.0%
    Frame 10: 61.0%
    """)
    print(result)