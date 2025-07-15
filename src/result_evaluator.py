""" Result Evaluator """

import json
import logging
import os
import re
import shutil
import tempfile
from dataclasses import dataclass

from google import genai

from voc_score import value_order_correlation

ERROR_MESSAGE = "No task completion percentage found!!"

PROMPT = f"""
You are a helpful assistant that extracts the task completion percentage from the response.
You will be given a response, you need to extract the task completion percentage from the response.

Here is and example:
### example start ###
Okay, let's analyze the task completion percentages for the "Pick up the blue object" scenario, given the provided examples.

**Task Completion Percentage for "Pick up the blue object"**

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
If there is no task completion percentages (do not extract percentages from descriptions of the frames) in the response, you need to return only {ERROR_MESSAGE}.
"""


@dataclass
class Result:
    prompt: str
    response: str
    ground_truth: str
    voc_score: float


class ResultEvaluator:
    def __init__(
            self,
            model_name: str = "gemma-3-27b-it",
        ):
        self.logger = logging.getLogger(__name__)
        self.model_name = model_name
        self.client = genai.Client(api_key=os.getenv("GEMINI_API_KEY"))
        
    def evaluate(self, response: str) -> list[float]:
        """Evaluate the response and return the task completion percentage."""
        contents = [{
            'parts': [{'text': PROMPT}, {'text': response}],
            'role': 'user'
        }]
        result = self.client.models.generate_content(
            model=self.model_name, contents=contents
        )
        return self._extract_task_completion_percentage(result.text)

    def evaluate_batch(self, responses: list[str]) -> list[list[float]]:
        """Evaluate multiple responses in batch and return task completion percentages.
        
        Note: the true batching does not work for free-tier setup
        inline_requests = []
        for response in responses:
            inline_requests.append(
            {
                'contents': [{
                        'parts': [{'text': PROMPT}, {'text': response}],
                        'role': 'user'
                    }]
                }
            )

        inline_batch_job = self.client.batches.create(
            model=self.model_name,
            src=inline_requests,
            config={
                'display_name': "inlined-requests-job-1",
            },
        )
        """
        if not responses:
            return []

        batch_results: list[list[float]] = []
        for response in responses:
            try:
                result = self.evaluate(response)
                batch_results.append(result)
            except Exception as e:
                self.logger.error(f"Error evaluating response: {e}")
                batch_results.append(None)
        
        return batch_results
    
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
        return None

    def batch_evaluate_jsonl(self, jsonl_file: str, output_file: str = None) -> str:
        results: list[dict[str, any]] = []
        with open(jsonl_file, 'r') as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                results.append(json.loads(line))

        for i, result in enumerate(results):
            if 'error' in result or 'status' in result:
                continue
            if result.get('extracted_percentages') is not None and result.get('voc_score') is not None:
                continue
            model_response = result.get('model_response', '')
            if not model_response:
                continue
            extracted_percentages = self.evaluate(model_response)
            result['extracted_percentages'] = extracted_percentages
            voc_score = None
            ground_truth = result.get('ground_truth_percentages')
            if extracted_percentages is not None and ground_truth is not None and len(extracted_percentages) == len(ground_truth):
                try:
                    voc_score = value_order_correlation(extracted_percentages, ground_truth)
                except Exception as e:
                    self.logger.error(f"Error calculating VOC score for result {i}: {e}")
                    voc_score = None
            result['voc_score'] = voc_score

        if output_file is None:
            output_file = jsonl_file

        if output_file == jsonl_file:
            with tempfile.NamedTemporaryFile(mode='w', delete=False, suffix='.jsonl') as tmp_f:
                temp_file = tmp_f.name
                for result in results:
                    tmp_f.write(json.dumps(result) + '\n')
            shutil.move(temp_file, jsonl_file)
        else:
            with open(output_file, 'w') as f:
                for result in results:
                    f.write(json.dumps(result) + '\n')
        return output_file


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
    result = result_evaluator.evaluate_batch(["""
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
    """])
    print(result)