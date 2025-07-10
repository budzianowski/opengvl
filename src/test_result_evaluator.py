import pytest
from result_evaluator import ResultEvaluator, ERROR_MESSAGE
import re


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
