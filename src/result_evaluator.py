""" Result Evaluator """

from transformers import AutoTokenizer, AutoModelForCausalLM
import json
from dataclasses import dataclass
import torch
import re
from voc_score import value_order_correlation
import os
import tempfile
import shutil

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
            model_name: str = "Qwen/Qwen2.5-1.5B-Instruct",
            max_new_tokens: int = 300,
            batch_size: int = 8
        ):
        self.model_name = model_name
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForCausalLM.from_pretrained(model_name)
        self.max_new_tokens = max_new_tokens
        self.batch_size = batch_size
        
        # Set padding token if not set
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

    def evaluate(self, response: str) -> list[float]:
        """Evaluate the response and return the task completion percentage."""
        return self.evaluate_batch([response])[0]

    def evaluate_batch(self, responses: list[str]) -> list[list[float]]:
        """Evaluate multiple responses in batch and return task completion percentages."""
        if not responses:
            return []
            
        # Prepare batch messages
        batch_messages = []
        for response in responses:
            messages = [
                {"role": "system", "content": PROMPT},
                {"role": "user", "content": response},
            ]
            batch_messages.append(messages)
        
        # Apply chat template to all messages
        batch_inputs = self.tokenizer.apply_chat_template(
            batch_messages,
            add_generation_prompt=True,
            tokenize=True,
            return_dict=True,
            return_tensors="pt",
            padding=True,
            truncation=True,
        ).to(self.model.device).to(torch.bfloat16)
        
        input_length = batch_inputs.input_ids.shape[1]
        
        # Generate responses for the batch
        with torch.inference_mode():
            outputs = self.model.generate(
                **batch_inputs, 
                max_new_tokens=self.max_new_tokens,
                pad_token_id=self.tokenizer.pad_token_id,
                do_sample=False,
                temperature=None,
                top_p=None
            )
        
        # Extract new tokens for each response
        batch_results = []
        for i, output in enumerate(outputs):
            new_tokens = output[input_length:]
            assistant_response = self.tokenizer.decode(new_tokens, skip_special_tokens=True)
            extracted_percentages = self._extract_task_completion_percentage(assistant_response)
            batch_results.append(extracted_percentages)
        
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
        """
        Batch evaluate a JSONL file and update it with extracted percentages and VOC scores.
        
        Args:
            jsonl_file: Path to the input JSONL file
            output_file: Path to the output JSONL file (if None, overwrites input file)
            
        Returns:
            Path to the updated JSONL file
        """
        if output_file is None:
            output_file = jsonl_file
            
        results = []
        
        # Read existing results
        with open(jsonl_file, 'r') as f:
            for line in f:
                if line.strip():
                    result = json.loads(line)
                    results.append(result)
        
        print(f"Processing {len(results)} results from {jsonl_file}")
        
        # Filter results that need processing
        results_to_process = []
        results_indices = []
        
        for i, result in enumerate(results):
            if 'error' in result or 'status' in result:
                continue
                
            # Skip if already processed and has valid extracted_percentages and voc_score
            if (result.get('extracted_percentages') is not None and 
                result.get('voc_score') is not None):
                continue
                
            model_response = result.get('model_response', '')
            if not model_response:
                continue
                
            results_to_process.append(result)
            results_indices.append(i)
        
        if not results_to_process:
            print("No results need processing")
            return output_file
            
        print(f"Found {len(results_to_process)} results that need processing")
        
        # Process in batches
        updated_count = 0
        for batch_start in range(0, len(results_to_process), self.batch_size):
            batch_end = min(batch_start + self.batch_size, len(results_to_process))
            batch_results = results_to_process[batch_start:batch_end]
            batch_indices = results_indices[batch_start:batch_end]
            
            # Extract responses for this batch
            batch_responses = [result['model_response'] for result in batch_results]
            
            try:
                # Process batch
                batch_extracted_percentages = self.evaluate_batch(batch_responses)
                
                # Update results
                for j, (result_idx, extracted_percentages) in enumerate(zip(batch_indices, batch_extracted_percentages)):
                    original_result = results[result_idx]
                    original_result['extracted_percentages'] = extracted_percentages
                    
                    # Calculate VOC score if we have valid percentages
                    voc_score = None
                    ground_truth = original_result.get('ground_truth_percentages')
                    
                    if (extracted_percentages is not None and 
                        ground_truth is not None and 
                        len(extracted_percentages) == len(ground_truth)):
                        try:
                            voc_score = value_order_correlation(extracted_percentages, ground_truth)
                        except Exception as e:
                            print(f"Error calculating VOC score for result {result_idx}: {e}")
                            voc_score = None
                    
                    original_result['voc_score'] = voc_score
                    updated_count += 1
                
                print(f"Processed batch {batch_start//self.batch_size + 1}/{(len(results_to_process) + self.batch_size - 1)//self.batch_size} ({batch_end}/{len(results_to_process)} results)")
                
            except Exception as e:
                print(f"Error processing batch {batch_start}-{batch_end}: {e}")
                # Set None values for failed batch
                for result_idx in batch_indices:
                    results[result_idx]['extracted_percentages'] = None
                    results[result_idx]['voc_score'] = None
                continue
        
        # Write updated results
        if output_file == jsonl_file:
            # Use temporary file to avoid corruption
            with tempfile.NamedTemporaryFile(mode='w', delete=False, suffix='.jsonl') as tmp_f:
                temp_file = tmp_f.name
                for result in results:
                    tmp_f.write(json.dumps(result) + '\n')
            
            # Replace original file
            shutil.move(temp_file, jsonl_file)
        else:
            with open(output_file, 'w') as f:
                for result in results:
                    f.write(json.dumps(result) + '\n')
        
        print(f"Updated {updated_count} results in {output_file}")
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