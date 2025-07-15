""" Tests for the result evaluator. """
import json
import os
import tempfile

import numpy as np
import pytest

from result_evaluator import ResultEvaluator


class TestResultEvaluator:
    @pytest.fixture(scope="class")
    def evaluator(self):
        return ResultEvaluator()

    def test_extract_list_format(self, evaluator):
        """Test extraction of percentages from list format."""
        response = "Here are the percentages: [10.5, 25.0, 50.0, 75.5, 100.0]"
        result = evaluator._extract_task_completion_percentage(response)
        expected = [10.5, 25.0, 50.0, 75.5, 100.0]
        assert result == expected

    @pytest.mark.slow
    def test_extract_percentage_symbols(self, evaluator):
        """Test extraction of percentages with % symbols."""
        response = """
        Frame 1: 15.5%
        Frame 2: 30%
        Frame 3: 67.2%
        Frame 4: 100%
        """
        result = evaluator.evaluate(response)
        expected = [15.5, 30.0, 67.2, 100.0]
        assert result == expected

    @pytest.mark.slow
    def test_extract_word_percentages(self, evaluator):
        """Test extraction of percentages with 'percent' word."""
        response = "Task is 25 percent complete, then 50 percent, finally 100 percent done."
        result = evaluator.evaluate(response)
        expected = [25.0, 50.0, 100.0]
        assert result == expected

    @pytest.mark.slow
    def test_extract_approximate_percentages(self, evaluator):
        """Test extraction of approximate percentage expressions."""
        response = """
        approximately 15% complete
        roughly 45% done
        about 80% finished
        """
        result = evaluator.evaluate(response)
        expected = [15.0, 45.0, 80.0]
        assert result == expected

    @pytest.mark.slow
    def test_extract_no_percentages(self, evaluator):
        """Test extraction when no percentages are present."""
        response = """
        The robot is making good progress.
        Task execution is proceeding smoothly.
        All systems are functioning normally.
        """
        result = evaluator.evaluate(response)
        assert result == []

    
    @pytest.mark.slow
    def test_complex_response_with_mixed_terminology(self, evaluator):
        """Test response with complex mixed terminology and varying percentage formats."""
        response = """
        Analyzing the robotic manipulation task "Pick up the red cube and place it in the container":
        
        Looking at this sequence, I observe several distinct phases of execution. The robot begins with 
        initial positioning and sensor calibration. Let me break down my assessment:
        
        **Initial Assessment Phase:**
        The robot's gripper is approaching the workspace. At this stage, I'd estimate we're at roughly 
        8.5% task completion since we're just beginning the approach phase.
        
        **Object Detection and Localization:**
        Frame 2 shows the robot has successfully identified the target object. The visual servoing 
        system appears to be functioning correctly. Progress here would be approximately 23% complete.
        
        **Pre-grasp Positioning:**
        In the third observation, the manipulator is positioning itself for optimal grasping. The 
        end-effector alignment looks good. I'd say we're at about 41.7% completion at this point.
        
        **Contact Initiation:**
        Frame 4 demonstrates initial contact with the object. The force feedback suggests proper 
        engagement. This represents roughly 68% task completion.
        
        **Grasp Execution:**
        The fifth frame shows successful object acquisition. The gripper has closed around the target 
        with what appears to be stable contact. We're at approximately 79.3% completion.
        
        **Transport Phase:**
        Frame 6 captures the robot lifting and beginning transport. The object remains securely grasped.
        This phase represents about 87% of the overall task.
        
        **Final Positioning:**
        The seventh frame shows approach to the target container. Precision positioning is critical here.
        I estimate 94.5% completion at this stage.
        
        **Task Completion:**
        The final frame demonstrates successful placement of the object in the container. The task 
        appears to be fully completed at 100%.
        
        Additional observations: The robot demonstrated good trajectory planning and obstacle avoidance 
        throughout the sequence. The manipulation strategy was efficient and the execution was smooth.
        """
        result = evaluator.evaluate(response)
        
        expected = [8.5, 23.0, 41.7, 68.0, 79.3, 87.0, 94.5, 100.0]
        assert result == expected

    @pytest.mark.slow
    def test_noisy_response_with_irrelevant_information(self, evaluator):
        """Test response with lots of irrelevant information and embedded percentages."""
        response = """
        SYSTEM STATUS: All sensors operational at 98.7% efficiency
        TEMPERATURE: 23.4°C (within normal range)
        BATTERY: 87% remaining
        NETWORK LATENCY: 12ms
        
        Beginning task analysis for manipulation sequence...
        
        FRAME_001: Robot initialization complete. The workspace is properly illuminated at 450 lux.
        Camera calibration shows 0.2% distortion (acceptable). Task completion assessment: 
        approximately 5.5% complete as we begin the approach phase.
        
        FRAME_002: Object detection algorithm running at 30 FPS with 95% confidence on target 
        identification. The red object measures 4.2cm x 4.2cm x 4.2cm. Environmental factors 
        are stable. Progress update: roughly 18.7% task completion.
        
        FRAME_003: Inverse kinematics solver converged in 15 iterations. Joint angles within 
        safety limits (max deviation 2.3°). Force sensors reading 0.1N baseline. Current 
        progress: approximately 35% complete.
        
        FRAME_004: Contact detected! Force reading jumped to 2.8N. Gripper servo motors at 
        67% torque. Safety systems green. Task advancement: about 58.2% completion.
        
        FRAME_005: Object securely grasped. Lift trajectory computed with 5-point spline. 
        Acceleration limited to 0.5m/s². Progress status: roughly 73% complete.
        
        FRAME_006: Transport phase active. Path planning avoiding 3 detected obstacles. 
        Current velocity: 0.15m/s. Task completion: approximately 89.4% done.
        
        FRAME_007: Final positioning engaged. Precision mode activated with 0.1mm tolerance. 
        All systems nominal. Final assessment: 100% task completion achieved.
        
        POST-TASK ANALYSIS: Total execution time 47.3 seconds. Energy consumption: 23.7 Wh.
        Success rate for similar tasks: 94.2% over last 50 trials.
        """
        result = evaluator.evaluate(response)
    
        expected = [5.5, 18.7, 35.0, 58.2, 73.0, 89.4, 100.0]
        assert result == expected

    @pytest.mark.slow 
    def test_full_model_evaluation_integration(self, evaluator):
        """Integration test using the full model evaluation pipeline."""
        response = """
        Task completion analysis:
        Frame 1: 25% complete
        Frame 2: 50% complete  
        Frame 3: 75% complete
        Frame 4: 100% complete
        """
        result = evaluator.evaluate(response)
        expected = [25.0, 50.0, 75.0, 100.0]
        assert result == expected


class TestExtractTaskCompletionPercentage:
    """Test the percentage extraction functionality without model inference."""
    
    @pytest.fixture(scope="class")
    def evaluator(self):
        """Create evaluator for testing extraction only."""
        return ResultEvaluator()
    
    @pytest.mark.parametrize(
        "response,expected",
        [
            # Standard format with brackets
            ("[10, 20, 30, 40, 50]", [10.0, 20.0, 30.0, 40.0, 50.0]),
            ("[10.5, 20.2, 30.8]", [10.5, 20.2, 30.8]),
            # With spaces
            ("[ 10 , 20 , 30 ]", [10.0, 20.0, 30.0]),
            # Mixed format
            ("[10, 20.5, 30]", [10.0, 20.5, 30.0]),
            # Single value
            ("[42]", [42.0]),
            # Empty brackets
            ("[]", []),
            # No brackets
            ("10, 20, 30", None),
            # Multiple bracket sets (should take first)
            ("[10, 20] and [30, 40]", [10.0, 20.0]),
            # Invalid format
            ("no numbers here", None),
            # With text before and after
            ("The results are [15, 25, 35] for the task.", [15.0, 25.0, 35.0]),
            # Decimal numbers
            ("[1.1, 2.2, 3.3, 4.4]", [1.1, 2.2, 3.3, 4.4]),
            # Integer and decimal mixed
            ("[1, 2.5, 3, 4.7, 5]", [1.0, 2.5, 3.0, 4.7, 5.0]),
        ],
    )
    def test_extract_task_completion_percentage(self, evaluator, response, expected):
        """Test percentage extraction from various response formats."""
        result = evaluator._extract_task_completion_percentage(response)
        assert result == expected
    
    def test_extract_malformed_numbers(self, evaluator):
        """Test extraction with malformed numbers."""
        with pytest.raises(ValueError):
            evaluator._extract_task_completion_percentage("[10, 20, abc]")
    
    def test_extract_with_extra_characters(self, evaluator):
        """Test extraction with extra characters in brackets."""
        with pytest.raises(ValueError):
            evaluator._extract_task_completion_percentage("[10, 20, 30%]")


class TestBatchEvaluation:
    """Test batch response evaluation with real model."""
    
    @pytest.fixture(scope="class")
    def evaluator(self):
        """Create evaluator for batch testing."""
        return ResultEvaluator(batch_size=2)
    
    @pytest.fixture
    def sample_responses(self):
        """Sample responses for testing."""
        return [
            "Frame 1: 10%\nFrame 2: 25%\nFrame 3: 50%\nFrame 4: 75%\nFrame 5: 90%",
            "Frame 1: 20%\nFrame 2: 40%\nFrame 3: 60%\nFrame 4: 80%\nFrame 5: 100%",
            "No clear percentages in this response. Just some text about the task.",
            "Frame 1: 15.5%\nFrame 2: 30.2%\nFrame 3: 45.8%",
            "The completion percentages are: [5, 15, 25, 35, 45]"
        ]
    
    def test_evaluate_batch_empty_list(self, evaluator):
        """Test batch evaluation with empty list."""
        result = evaluator.evaluate_batch([])
        assert result == []
    
    @pytest.mark.slow
    def test_evaluate_batch_single_response(self, evaluator, sample_responses):
        """Test batch evaluation with single response."""
        result = evaluator.evaluate_batch([sample_responses[0]])
        assert len(result) == 1
        assert result[0] is not None
        assert isinstance(result[0], list)
        assert len(result[0]) == 5  # Should extract 5 percentages
    
    @pytest.mark.slow
    def test_evaluate_batch_multiple_responses(self, evaluator, sample_responses):
        """Test batch evaluation with multiple responses."""
        result = evaluator.evaluate_batch(sample_responses[:3])
        assert len(result) == 3
        # First two should have extracted percentages
        assert result[0] is not None
        assert result[1] is not None
        # Third might be None or empty depending on model behavior
        assert result[2] is not None or result[2] is None
    
    @pytest.mark.slow
    def test_evaluate_batch_different_sizes(self, evaluator, sample_responses):
        """Test batch evaluation with different batch sizes."""
        # Test with batch size larger than responses
        result = evaluator.evaluate_batch(sample_responses[:2])
        assert len(result) == 2
        
        # Test with batch size smaller than responses
        result = evaluator.evaluate_batch(sample_responses)
        assert len(result) == 5
    
    @pytest.mark.slow
    def test_evaluate_batch_consistency(self, evaluator, sample_responses):
        """Test that batch evaluation gives consistent results."""
        # Evaluate same responses multiple times
        result1 = evaluator.evaluate_batch([sample_responses[0]])
        result2 = evaluator.evaluate_batch([sample_responses[0]])
        
        # Results should be consistent (model is deterministic)
        assert result1 == result2


class TestBatchJSONLEvaluation:
    """Test batch JSONL file evaluation."""
    
    @pytest.fixture(scope="class")
    def evaluator(self):
        """Create evaluator for JSONL testing."""
        return ResultEvaluator(batch_size=2)
    
    @pytest.fixture
    def sample_jsonl_data(self):
        """Sample JSONL data for testing batch evaluation."""
        return [
            {
                "step": 1,
                "timestamp": "2024-01-01T10:00:00",
                "model": "gpt4o",
                "model_response": "Frame 1: 10%\nFrame 2: 25%\nFrame 3: 50%\nFrame 4: 75%\nFrame 5: 90%",
                "ground_truth_percentages": [15, 30, 45, 70, 85],
                "extracted_percentages": None,
                "voc_score": None
            },
            {
                "step": 2,
                "timestamp": "2024-01-01T10:01:00",
                "model": "gpt4o",
                "model_response": "Frame 1: 20%\nFrame 2: 40%\nFrame 3: 60%\nFrame 4: 80%\nFrame 5: 100%",
                "ground_truth_percentages": [25, 35, 55, 75, 95],
                "extracted_percentages": None,
                "voc_score": None
            },
            {
                "step": 3,
                "timestamp": "2024-01-01T10:02:00",
                "model": "gpt4o",
                "error": "Model failed to respond",
                "status": "failed"
            },
            {
                "step": 4,
                "timestamp": "2024-01-01T10:03:00",
                "model": "gpt4o",
                "model_response": "No percentages here, just text about the task",
                "ground_truth_percentages": [10, 20, 30, 40, 50],
                "extracted_percentages": None,
                "voc_score": None
            },
            {
                "step": 5,
                "timestamp": "2024-01-01T10:04:00",
                "model": "gpt4o",
                "model_response": "Frame 1: 5%\nFrame 2: 15%\nFrame 3: 25%",
                "ground_truth_percentages": [10, 20, 30],
                "extracted_percentages": [5.0, 15.0, 25.0],  # Already processed
                "voc_score": 1.0
            },
            {
                "step": 6,
                "timestamp": "2024-01-01T10:05:00",
                "model": "gpt4o",
                "model_response": "The completion is [12, 24, 36, 48, 60]",
                "ground_truth_percentages": [10, 25, 35, 45, 55],
                "extracted_percentages": None,
                "voc_score": None
            }
        ]
    
    @pytest.fixture
    def temp_jsonl_file(self, sample_jsonl_data):
        """Create a temporary JSONL file with sample data."""
        with tempfile.NamedTemporaryFile(mode='w', suffix='.jsonl', delete=False) as f:
            for item in sample_jsonl_data:
                f.write(json.dumps(item) + '\n')
            temp_file = f.name
        
        yield temp_file
        
        # Cleanup
        if os.path.exists(temp_file):
            os.unlink(temp_file)
    
    def test_batch_evaluate_jsonl_file_not_found(self, evaluator):
        """Test batch evaluation with non-existent file."""
        with pytest.raises(FileNotFoundError):
            evaluator.batch_evaluate_jsonl("nonexistent.jsonl")
    
    @pytest.mark.slow
    def test_batch_evaluate_jsonl_basic(self, evaluator, temp_jsonl_file):
        """Test basic batch JSONL evaluation."""
        result_file = evaluator.batch_evaluate_jsonl(temp_jsonl_file)
        
        # Should return the same file path
        assert result_file == temp_jsonl_file
        
        # Read and verify results
        with open(result_file, 'r') as f:
            results = [json.loads(line) for line in f if line.strip()]
        
        # Should have 6 results total
        assert len(results) == 6
        
        # Check that results were processed correctly
        processed_results = [r for r in results if r.get('extracted_percentages') is not None]
        assert len(processed_results) >= 2  # At least 2 should be processed (including pre-processed one)
    
    @pytest.mark.slow
    def test_batch_evaluate_jsonl_skip_processed(self, evaluator, temp_jsonl_file):
        """Test that already processed results are skipped."""
        # First run
        result_file = evaluator.batch_evaluate_jsonl(temp_jsonl_file)
        
        # Read results after first run
        with open(result_file, 'r') as f:
            results_after_first = [json.loads(line) for line in f if line.strip()]
        
        # Count processed results
        processed_count_first = len([r for r in results_after_first if r.get('extracted_percentages') is not None])
        
        # Second run should skip already processed results
        result_file = evaluator.batch_evaluate_jsonl(temp_jsonl_file)
        
        # Read results after second run
        with open(result_file, 'r') as f:
            results_after_second = [json.loads(line) for line in f if line.strip()]
        
        # Should have same number of processed results (no additional processing)
        processed_count_second = len([r for r in results_after_second if r.get('extracted_percentages') is not None])
        assert processed_count_second == processed_count_first
    
    @pytest.mark.slow
    def test_batch_evaluate_jsonl_skip_errors(self, evaluator, temp_jsonl_file):
        """Test that error results are skipped."""
        result_file = evaluator.batch_evaluate_jsonl(temp_jsonl_file)
        
        # Read results
        with open(result_file, 'r') as f:
            results = [json.loads(line) for line in f if line.strip()]
        
        # Error result (step 3) should remain unchanged
        error_result = next(r for r in results if r['step'] == 3)
        assert 'error' in error_result
        assert error_result.get('extracted_percentages') is None
        assert error_result.get('voc_score') is None
    
    @pytest.mark.slow
    def test_batch_evaluate_jsonl_output_file(self, evaluator, temp_jsonl_file):
        """Test batch evaluation with separate output file."""
        with tempfile.NamedTemporaryFile(suffix='.jsonl', delete=False) as output_f:
            output_file = output_f.name
        
        try:
            result_file = evaluator.batch_evaluate_jsonl(temp_jsonl_file, output_file)
            
            assert result_file == output_file
            assert os.path.exists(output_file)
            
            # Verify both files exist and are different
            assert os.path.exists(temp_jsonl_file)
            assert temp_jsonl_file != output_file
            
            # Check that output file has content
            with open(output_file, 'r') as f:
                output_results = [json.loads(line) for line in f if line.strip()]
            
            assert len(output_results) == 6  # Should have all original results
            
        finally:
            if os.path.exists(output_file):
                os.unlink(output_file)
    
    @pytest.mark.slow
    def test_batch_evaluate_jsonl_voc_score_calculation(self, evaluator, temp_jsonl_file):
        """Test VOC score calculation during batch evaluation."""
        result_file = evaluator.batch_evaluate_jsonl(temp_jsonl_file)
        
        # Read results
        with open(result_file, 'r') as f:
            results = [json.loads(line) for line in f if line.strip()]
        
        # Check that VOC scores were calculated for valid extractions
        valid_voc_results = [r for r in results if r.get('voc_score') is not None]
        assert len(valid_voc_results) >= 1  # At least one should have valid VOC score
        
        # Check that VOC scores are in valid range
        for result in valid_voc_results:
            voc_score = result['voc_score']
            assert -1.0 <= voc_score <= 1.0 or np.isnan(voc_score)
    
    def test_batch_evaluate_jsonl_no_results_to_process(self, evaluator):
        """Test batch evaluation when no results need processing."""
        # Create a file with only processed results
        processed_data = [
            {
                "step": 1,
                "model_response": "Some response",
                "ground_truth_percentages": [10, 20, 30],
                "extracted_percentages": [10.0, 20.0, 30.0],
                "voc_score": 1.0
            },
            {
                "step": 2,
                "model_response": "Another response",
                "ground_truth_percentages": [15, 25, 35],
                "extracted_percentages": [15.0, 25.0, 35.0],
                "voc_score": 1.0
            }
        ]
        
        with tempfile.NamedTemporaryFile(mode='w', suffix='.jsonl', delete=False) as f:
            for item in processed_data:
                f.write(json.dumps(item) + '\n')
            temp_file = f.name
        
        try:
            result_file = evaluator.batch_evaluate_jsonl(temp_file)
            
            # Should return the same file
            assert result_file == temp_file
            
            # Results should be unchanged
            with open(result_file, 'r') as f:
                results = [json.loads(line) for line in f if line.strip()]
            
            assert len(results) == 2
            assert all(r.get('extracted_percentages') is not None for r in results)
            assert all(r.get('voc_score') is not None for r in results)
            
        finally:
            if os.path.exists(temp_file):
                os.unlink(temp_file)


class TestVOCScoreIntegration:
    """Test VOC score calculation integration."""
    
    @pytest.fixture(scope="class")
    def evaluator(self):
        """Create evaluator for VOC testing."""
        return ResultEvaluator()
    
    def test_voc_score_perfect_correlation(self, evaluator):
        """Test VOC score calculation with perfect correlation."""
        # Create test data with perfect correlation
        test_data = [
            {
                "step": 1,
                "model_response": "The completion is [10, 20, 30, 40, 50]",
                "ground_truth_percentages": [10, 20, 30, 40, 50],
                "extracted_percentages": None,
                "voc_score": None
            }
        ]
        
        with tempfile.NamedTemporaryFile(mode='w', suffix='.jsonl', delete=False) as f:
            for item in test_data:
                f.write(json.dumps(item) + '\n')
            temp_file = f.name
        
        try:
            result_file = evaluator.batch_evaluate_jsonl(temp_file)
            
            with open(result_file, 'r') as f:
                results = [json.loads(line) for line in f if line.strip()]
            
            result = results[0]
            assert result['extracted_percentages'] == [10.0, 20.0, 30.0, 40.0, 50.0]
            assert result['voc_score'] == 1.0  # Perfect correlation
            
        finally:
            if os.path.exists(temp_file):
                os.unlink(temp_file)
    
    def test_voc_score_negative_correlation(self, evaluator):
        """Test VOC score calculation with negative correlation."""
        # Create test data with negative correlation
        test_data = [
            {
                "step": 1,
                "model_response": "The completion is [50, 40, 30, 20, 10]",
                "ground_truth_percentages": [10, 20, 30, 40, 50],
                "extracted_percentages": None,
                "voc_score": None
            }
        ]
        
        with tempfile.NamedTemporaryFile(mode='w', suffix='.jsonl', delete=False) as f:
            for item in test_data:
                f.write(json.dumps(item) + '\n')
            temp_file = f.name
        
        try:
            result_file = evaluator.batch_evaluate_jsonl(temp_file)
            
            with open(result_file, 'r') as f:
                results = [json.loads(line) for line in f if line.strip()]
            
            result = results[0]
            assert result['extracted_percentages'] == [50.0, 40.0, 30.0, 20.0, 10.0]
            assert result['voc_score'] == -1.0  # Perfect negative correlation
            
        finally:
            if os.path.exists(temp_file):
                os.unlink(temp_file)
    
    def test_voc_score_length_mismatch(self, evaluator):
        """Test VOC score calculation with length mismatch."""
        # Create test data with mismatched lengths
        test_data = [
            {
                "step": 1,
                "model_response": "The completion is [10, 20, 30]",
                "ground_truth_percentages": [10, 20, 30, 40, 50],  # Different length
                "extracted_percentages": None,
                "voc_score": None
            }
        ]
        
        with tempfile.NamedTemporaryFile(mode='w', suffix='.jsonl', delete=False) as f:
            for item in test_data:
                f.write(json.dumps(item) + '\n')
            temp_file = f.name
        
        try:
            result_file = evaluator.batch_evaluate_jsonl(temp_file)
            
            with open(result_file, 'r') as f:
                results = [json.loads(line) for line in f if line.strip()]
            
            result = results[0]
            assert result['extracted_percentages'] == [10.0, 20.0, 30.0]
            assert result['voc_score'] is None  # Should be None due to length mismatch
            
        finally:
            if os.path.exists(temp_file):
                os.unlink(temp_file)


class TestGemmaText(TestResultEvaluator):

    @pytest.mark.slow
    def test_extract_percentages_with_gemma_text_1(self, evaluator):
        response = "Okay, let's analyze the task completion percentages for each frame, considering the overall task of picking up the purple object. Here's a breakdown with estimated percentages:\n\n**Frame Analysis & Task Completion Percentage Prediction**\n\n*   **Frame 1:** 15% - The robot is just starting to approach the purple object. Initial positioning and planning are underway.\n*   **Frame 2:** 30% - The robot is now closer to the purple object, likely initiating the grasping action.\n*   **Frame 3:** 45% - The robot is in the process of grasping the purple object.\n*   **Frame 4:** 60% - The purple object is partially grasped.\n*   **Frame 5:** 75% - The purple object is firmly grasped, and the robot is beginning to lift it.\n*   **Frame 6:** 85% - The purple object is being lifted off the surface.\n*   **Frame 7:** 90% - The purple object is fully lifted and being moved away from the base.\n*   **Frame 8:** 95% - The purple object is being moved towards the target location.\n*   **Frame 9:** 98% - The purple object is almost in the target location.\n*   **Frame 10:** 99% - The purple object is in the target location.\n*   **Frame 11:** 99% - The purple object is in the target location.\n*   **Frame 12:** 98% - The purple object is in the target location.\n*   **Frame 13:** 95% - The purple object is in the target location.\n*   **Frame 14:** 80% - The purple object is in the target location.\n*   **Frame 15:** 65% - The purple object is in the target location.\n*   **Frame 16:** 55% - The purple object is in the target location.\n*   **Frame 17:** 70% - The purple object is in the target location.\n*   **Frame 18:** 85% - The purple object is in the target location.\n*   **Frame 19:** 90% - The purple object is in the target location.\n*   **Frame 20:** 95% - The purple object is in the target location.\n\n**Important Note:** These percentages are estimates based on the visual information provided. A more sophisticated analysis would require access to the robot's internal state (e.g., joint angles, object pose estimates) to provide more accurate predictions."
        result = evaluator.evaluate(response)
        expected = [15.0, 30.0, 45.0, 60.0, 75.0, 85.0, 90.0, 95.0, 98.0, 99.0, 99.0, 98.0, 95.0, 80.0, 65.0, 55.0, 70.0, 85.0, 90.0, 95.0]
        assert result == expected

    @pytest.mark.slow
    def test_extract_percentages_with_gemma_text_2(self, evaluator):
        response = "Okay, let's analyze the task completion percentages for each frame, considering the overall goal of \"Pick up the green object.\"\n\nHere's the breakdown:\n\nFrame: 1, Description: Initial scene with green object present, Task Completion: 0%\nFrame: 2, Description: Slight movement, Task Completion: 41%\nFrame: 3, Description: Robot approaching the green object, Task Completion: 69%\nFrame: 4, Description: Robot grasping the green object, Task Completion: 36%\nFrame: 5, Description: Green object being lifted, Task Completion: 71%\nFrame: 6, Description: Green object is in the air, Task Completion: 4%\nFrame: 7, Description: Green object is placed on the target, Task Completion: 75%\nFrame: 8, Description: Green object is on the target, Task Completion: 80%\nFrame: 9, Description: Green object is on the target, Task Completion: 43%\nFrame: 10, Description: Green object is on the target, Task Completion: 0%\nFrame: 11, Description: Green object is on the target, Task Completion: 31%\nFrame: 12, Description: Green object is on the target, Task Completion: 24%\nFrame: 13, Description: Green object is on the target, Task Completion: 48%\nFrame: 14, Description: Green object is on the target, Task Completion: 72%\nFrame: 15, Description: Green object is on the target, Task Completion: 20%\nFrame: 16, Description: Green object is on the target, Task Completion: 56%\nFrame: 17, Description: Green object is on the target, Task Completion: 91%\nFrame: 18, Description: Green object is on the target, Task Completion: 16%\nFrame: 19, Description: Green object is on the target, Task Completion: 81%\nFrame: 20, Description: Green object is on the target, Task Completion: 7%\n\n**Important Note:** The task completion percentages fluctuate significantly, reflecting the robot's progress and potential errors during the pick-up process."
        result = evaluator.evaluate(response)
        expected = [0.0, 41.0, 69.0, 36.0, 71.0, 4.0, 75.0, 80.0, 43.0, 0.0, 31.0, 24.0, 48.0, 72.0, 20.0, 56.0, 91.0, 16.0, 81.0, 7.0]
        assert result == expected

    @pytest.mark.slow
    def test_extract_percentages_with_gemma_text_3(self, evaluator):
        response = "Okay, let\u2019s analyze the task completion percentages for each frame, considering the overall goal of \u201cPick up the purple object.\u201d Here\u2019s a breakdown with estimated percentages:\n\n*   **Frame 1:** 5% - The robot is just starting to approach the purple object. Initial positioning is crucial.\n*   **Frame 2:** 25% - The robot has likely grasped the purple object, but the grip might not be secure yet.\n*   **Frame 3:** 45% - The object is being held, and the robot is beginning to move it away from the base.\n*   **Frame 4:** 65% - The robot is now moving the purple object away from the base.\n*   **Frame 5:** 75% - The robot is almost done with the movement.\n*   **Frame 6:** 85% - The robot is in the final stages of the movement.\n*   **Frame 7:** 90% - The robot is almost done with the movement.\n*   **Frame 8:** 95% - The robot is almost done with the movement.\n*   **Frame 9:** 98% - The robot is almost done with the movement.\n*   **Frame 10:** 99% - The robot is almost done with the movement.\n*   **Frame 11:** 99% - The robot is almost done with the movement.\n*   **Frame 12:** 99% - The robot is almost done with the movement.\n*   **Frame 13:** 99% - The robot is almost done with the movement.\n*   **Frame 14:** 99% - The robot is almost done with the movement.\n*   **Frame 15:** 93% - The robot is almost done with the movement.\n*   **Frame 16:** 88% - The robot is almost done with the movement.\n*   **Frame 17:** 85% - The robot is almost done with the movement.\n*   **Frame 18:** 80% - The robot is almost done with the movement.\n*   **Frame 19:** 75% - The robot is almost done with the movement.\n*   **Frame 20:** 71% - The robot is almost done with the movement.\n\n**Important Note:** These percentages are estimates based on the visual information provided. A more sophisticated analysis would require knowledge of the robot's internal state (e.g., grip force, trajectory)."
        result = evaluator.evaluate(response)
        expected = [5.0, 25.0, 45.0, 65.0, 75.0, 85.0, 90.0, 95.0, 98.0, 99.0, 99.0, 99.0, 99.0, 99.0, 93.0, 88.0, 85.0, 80.0, 75.0, 71.0]
        assert result == expected

    @pytest.mark.slow
    def test_extract_percentages_with_gemma_text_4(self, evaluator):
        response = "Okay, let's analyze the task completion percentages for each frame, considering the overall task of \"Pick up the brown object.\"\n\nHere's the breakdown:\n\n*   **Frame 1:** 95.0% - The robot is actively approaching the brown object, indicating a high level of task progress.\n*   **Frame 2:** 72.0% - The robot is still in the process of grasping the brown object.\n*   **Frame 3:** 18.0% - The robot has started to move the brown object, but the grasp is not yet secure.\n*   **Frame 4:** 74.0% - The robot is holding the brown object, but not yet fully secured.\n*   **Frame 5:** 49.0% - The robot is holding the brown object, but the grasp is not secure.\n*   **Frame 6:** 62.0% - The robot is holding the brown object, but the grasp is not secure.\n*   **Frame 7:** 81.0% - The robot is holding the brown object, but the grasp is not secure.\n*   **Frame 8:** 31.0% - The robot is holding the brown object, but the grasp is not secure.\n*   **Frame 9:** 40.0% - The robot is holding the brown object, but the grasp is not secure.\n*   **Frame 10:** 32.0% - The robot is holding the brown object, but the grasp is not secure.\n*   **Frame 11:** 33.0% - The robot is holding the brown object, but the grasp is not secure.\n*   **Frame 12:** 65.0% - The robot is holding the brown object, but the grasp is not secure.\n*   **Frame 13:** 86.0% - The robot is holding the brown object, but the grasp is not secure.\n*   **Frame 14:** 43.0% - The robot is holding the brown object, but the grasp is not secure.\n*   **Frame 15:** 85.0% - The robot is holding the brown object, but the grasp is not secure.\n*   **Frame 16:** 77.0% - The robot is holding the brown object, but the grasp is not secure.\n*   **Frame 17:** 61.0% - The robot is holding the brown object, but the grasp is not secure.\n*   **Frame 18:** 30.0% - The robot is holding the brown object, but the grasp is not secure.\n*   **Frame 19:** 49.0% - The robot is holding the brown object, but the grasp is not secure.\n*   **Frame 20:** 81.0% - The robot is holding the brown object, but the grasp is not secure.\n\n**Note:** The task completion percentage is based on the assumption that the robot has successfully picked up the brown object.\n\nLet me know if you'd like me to elaborate on any of these predictions or provide a more detailed analysis!"
        result = evaluator.evaluate(response)
        expected = [95.0, 72.0, 18.0, 74.0, 49.0, 62.0, 81.0, 31.0, 40.0, 32.0, 33.0, 65.0, 86.0, 43.0, 85.0, 77.0, 61.0, 30.0, 49.0, 81.0]
        assert result == expected

    @pytest.mark.slow
    def test_extract_percentages_with_gemma_text_5(self, evaluator):
        response = "Okay, let's analyze the task completion percentages for picking up the green object, frame by frame. Here's the breakdown:\n\nFrame 1: 0% - Initial state, robot hasn't started the task.\nFrame 2: 46% - The robot has started to approach the green object.\nFrame 3: 55% - The robot is in the process of grasping the green object.\nFrame 4: 92% - The robot has successfully grasped the green object.\nFrame 5: 5% - The robot is releasing the green object.\nFrame 6: 61% - The robot is moving the green object away from the base.\nFrame 7: 0% - The green object is no longer in the robot's grasp.\nFrame 8: 30% - The robot is repositioning itself.\nFrame 9: 10% - The robot is approaching the green object again.\nFrame 10: 73% - The robot is in the process of grasping the green object.\nFrame 11: 40% - The robot is holding the green object.\nFrame 12: 62% - The robot is moving the green object away from the base.\nFrame 13: 12% - The robot is repositioning itself.\nFrame 14: 91% - The robot is holding the green object.\nFrame 15: 26% - The robot is approaching the base.\nFrame 16: 84% - The robot is holding the green object.\nFrame 17: 60% - The robot is moving the green object away from the base.\nFrame 18: 81% - The robot is holding the green object.\nFrame 19: 19% - The robot is repositioning itself.\nFrame 20: 57% - The robot is approaching the base.\n\n**Summary of Task Completion Percentages:**\n\nFrame 1: 0%\nFrame 2: 46%\nFrame 3: 55%\nFrame 4: 92%\nFrame 5: 5%\nFrame 6: 61%\nFrame 7: 0%\nFrame 8: 30%\nFrame 9: 10%\nFrame 10: 73%\nFrame 11: 40%\nFrame 12: 62%\nFrame 13: 12%\nFrame 14: 91%\nFrame 15: 26%\nFrame 16: 84%\nFrame 17: 60%\nFrame 18: 81%\nFrame 19: 19%\nFrame 20: 57%"
        result = evaluator.evaluate(response)
        expected = [0.0, 46.0, 55.0, 92.0, 5.0, 61.0, 0.0, 30.0, 10.0, 73.0, 40.0, 62.0, 12.0, 91.0, 26.0, 84.0, 60.0, 81.0, 19.0, 57.0]
        assert result == expected
