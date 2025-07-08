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
    def test_extract_fractional_expressions(self, evaluator):
        """Test extraction of fractional expressions."""
        response = """
        Frame 1: one-tenth complete
        Frame 2: one-quarter done
        Frame 3: one-half finished
        Frame 4: three-quarters complete
        Frame 5: 7 out of 10 parts done
        Frame 6: 9/10 complete
        """
        result = evaluator.evaluate(response)
        expected = [10.0, 25.0, 50.0, 75.0, 70.0, 90.0]
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
        expected = []
        assert result == expected

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
    def test_contradictory_response_with_multiple_assessments(self, evaluator):
        """Test response with contradictory information and multiple assessment attempts."""
        response = """
        Task Analysis Report: "Pour liquid from container A to container B"
        
        Initial Assessment:
        Looking at the first frame, the robot hasn't started the pouring motion yet. I'd estimate 
        this is about 10% complete since positioning is underway.
        
        Wait, let me reconsider that. Actually, looking more carefully at the gripper position 
        and the container alignment, this might be closer to 15% completion.
        
        Frame 2 Analysis:
        The robot has grasped the container. This is significant progress - I'd say we're at 
        30% completion. Although, considering the complexity of pouring tasks, maybe this is 
        more like 25% complete.
        
        Frame 3 Observation:
        The container is being lifted. This is tricky to assess because pouring requires precise 
        control. Could be 45% complete, but pouring tasks are non-linear in their progress 
        measurement. Perhaps 40% is more accurate.
        
        Frame 4 Update:
        Pouring motion has begun! This is the critical phase. I initially thought 60% but 
        the actual liquid transfer hasn't started yet. Maybe 55% is better.
        
        Frame 5 Assessment:
        Liquid is flowing! This is the main task execution. I'd estimate 75% complete, though 
        the flow rate suggests we might be at 80% completion.
        
        Frame 6 Analysis:
        The pouring continues with good control. We're definitely in the 85-90% range. Let me 
        say 87% completion.
        
        Frame 7 Final:
        Task appears complete with successful liquid transfer. 100% completion achieved.
        
        Note: Pouring tasks are particularly challenging to assess due to their continuous 
        nature and the importance of the final precision phase.
        """
        result = evaluator.evaluate(response)
        
        expected_percentages = [10.0, 15.0, 30.0, 25.0, 45.0, 40.0, 60.0, 55.0, 75.0, 80.0, 85.0, 90.0, 87.0, 100.0]
        
        assert result == expected_percentages
        
    @pytest.mark.slow
    def test_response_with_no_clear_percentages(self, evaluator):
        """Test response that discusses progress but provides no clear percentage values."""
        response = """
        Analyzing the robotic task execution for "Sort colored blocks by category":
        
        The robot begins by scanning the workspace. This is clearly the initial phase of the 
        task where environmental understanding is being established. Very preliminary stage.
        
        Next, I observe the robot identifying different colored objects. The visual processing 
        system seems to be working effectively. This represents early progress in the sorting 
        task but we're still in the reconnaissance phase.
        
        The robot then selects its first target object. This demonstrates good decision-making 
        and task prioritization. We're moving beyond the planning phase into execution.
        
        I see the robot successfully grasping a blue block. This is concrete progress - the 
        first actual manipulation action. We're transitioning from preparation to active sorting.
        
        The blue block is being moved toward what appears to be the correct sorting area. 
        This shows good spatial reasoning and task understanding. Solid advancement.
        
        The robot places the blue block and returns for another object. This demonstrates 
        successful completion of the first sorting cycle. Good momentum building.
        
        Multiple objects have now been sorted correctly. The robot is showing consistent 
        performance and good efficiency. We're well into the active sorting phase.
        
        The workspace appears mostly organized with objects in their appropriate categories. 
        The sorting task is nearing completion with just a few items remaining.
        
        Final observation shows all objects properly categorized. The sorting task has been 
        completed successfully with good organization and efficiency.
        """
        result = evaluator.evaluate(response)
        
        assert result == []

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