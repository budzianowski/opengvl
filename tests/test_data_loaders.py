import pytest
import numpy as np
from unittest.mock import Mock, patch

from opengvl.data_loaders.base import BaseDataLoader
from opengvl.utils.data_types import Episode, Example


class TestBaseDataLoader:
    """Test suite for BaseDataLoader abstract class."""

    def test_initialization_defaults(self):
        """Test default initialization parameters."""
        # Create a mock concrete implementation
        class MockDataLoader(BaseDataLoader):
            def load_fewshot_input(self, episode_index=None):
                return Mock(spec=Example)
        
        loader = MockDataLoader()
        assert loader.num_frames == 10
        assert loader.num_context_episodes == 0
        assert loader.shuffle == False
        assert loader.seed == 42
        assert loader._rng is not None

    def test_initialization_custom_params(self):
        """Test initialization with custom parameters."""
        class MockDataLoader(BaseDataLoader):
            def load_fewshot_input(self, episode_index=None):
                return Mock(spec=Example)
        
        loader = MockDataLoader(
            num_frames=5,
            num_context_episodes=2,
            shuffle=True,
            seed=123
        )
        assert loader.num_frames == 5
        assert loader.num_context_episodes == 2
        assert loader.shuffle == True
        assert loader.seed == 123

    def test_load_fewshot_inputs(self):
        """Test loading multiple fewshot inputs."""
        class MockDataLoader(BaseDataLoader):
            def __init__(self, **kwargs):
                super().__init__(**kwargs)
                self.call_count = 0
            
            def load_fewshot_input(self, episode_index=None):
                self.call_count += 1
                return Mock(spec=Example)
        
        loader = MockDataLoader()
        results = loader.load_fewshot_inputs(3)
        
        assert len(results) == 3
        assert loader.call_count == 3

    def test_load_fewshot_inputs_zero(self):
        """Test loading zero fewshot inputs."""
        class MockDataLoader(BaseDataLoader):
            def load_fewshot_input(self, episode_index=None):
                return Mock(spec=Example)
        
        loader = MockDataLoader()
        results = loader.load_fewshot_inputs(0)
        
        assert len(results) == 0

    def test_reset(self):
        """Test reset functionality."""
        class MockDataLoader(BaseDataLoader):
            def load_fewshot_input(self, episode_index=None):
                return Mock(spec=Example)
        
        loader = MockDataLoader(seed=42)
        original_rng = loader._rng
        
        # Reset should create a new RNG with the same seed
        loader.reset()
        
        assert loader._rng is not original_rng
        assert loader.seed == 42

    def test_linear_completion(self):
        """Test _linear_completion helper method."""
        class MockDataLoader(BaseDataLoader):
            def load_fewshot_input(self, episode_index=None):
                return Mock(spec=Example)
        
        loader = MockDataLoader()
        
        # Test various lengths
        assert loader._linear_completion(0) == []
        assert loader._linear_completion(1) == [100]
        assert loader._linear_completion(2) == [0, 100]
        assert loader._linear_completion(3) == [0, 50, 100]
        assert loader._linear_completion(4) == [0, 33, 67, 100]  # Fixed: round(33.33) = 33, round(66.67) = 67
        assert loader._linear_completion(5) == [0, 25, 50, 75, 100]

    def test_linear_completion_edge_cases(self):
        """Test _linear_completion with edge cases."""
        class MockDataLoader(BaseDataLoader):
            def load_fewshot_input(self, episode_index=None):
                return Mock(spec=Example)
        
        loader = MockDataLoader()
        
        # Negative length should return empty
        assert loader._linear_completion(-1) == []
        
        # Large length should still work
        result = loader._linear_completion(10)
        assert len(result) == 10
        assert result[0] == 0
        assert result[-1] == 100
        assert all(isinstance(x, int) for x in result)

    def test_abstract_method_enforcement(self):
        """Test that BaseDataLoader cannot be instantiated directly."""
        with pytest.raises(TypeError):
            BaseDataLoader()

    def test_rng_deterministic_behavior(self):
        """Test that RNG produces deterministic results with same seed."""
        class MockDataLoader(BaseDataLoader):
            def load_fewshot_input(self, episode_index=None):
                return Mock(spec=Example)
        
        loader1 = MockDataLoader(seed=42)
        loader2 = MockDataLoader(seed=42)
        
        # Generate some random numbers
        random1 = [loader1._rng.random() for _ in range(5)]
        random2 = [loader2._rng.random() for _ in range(5)]
        
        assert random1 == random2

    def test_rng_different_seeds(self):
        """Test that different seeds produce different results."""
        class MockDataLoader(BaseDataLoader):
            def load_fewshot_input(self, episode_index=None):
                return Mock(spec=Example)
        
        loader1 = MockDataLoader(seed=42)
        loader2 = MockDataLoader(seed=123)
        
        # Generate some random numbers
        random1 = [loader1._rng.random() for _ in range(5)]
        random2 = [loader2._rng.random() for _ in range(5)]
        
        assert random1 != random2


class TestBaseDataLoaderHelpers:
    """Test helper methods in BaseDataLoader."""

    def test_build_episode_helper(self):
        """Test the pattern for building episodes in concrete loaders."""
        dummy_image = np.zeros((64, 64, 3), dtype=np.uint8)
        
        # Simulate what a concrete loader might do
        instruction = "Test task"
        episode_index = 0
        num_frames = 4
        
        # Create linear completion rates
        original_indices = list(range(num_frames))
        original_rates = [i * 100 // (num_frames - 1) for i in range(num_frames)]
        
        # Shuffle for model input
        rng = np.random.default_rng(42)
        shuffled_indices = original_indices.copy()
        rng.shuffle(shuffled_indices)
        
        # Build episode
        episode = Episode(
            instruction=instruction,
            starting_frame=dummy_image,
            episode_index=episode_index,
            original_frames_indices=original_indices,
            original_frames_task_completion_rates=original_rates,
            shuffled_frames_indices=shuffled_indices,
            shuffled_frames=[dummy_image] * num_frames,
            shuffled_frames_approx_completion_rates=[
                original_rates[original_indices.index(idx)] for idx in shuffled_indices
            ],
        )
        
        # Verify episode is valid
        assert episode.instruction == instruction
        assert episode.episode_index == episode_index
        assert len(episode.shuffled_frames) == num_frames
        assert len(episode.shuffled_frames_indices) == num_frames
        assert len(episode.shuffled_frames_approx_completion_rates) == num_frames

    def test_example_creation_pattern(self):
        """Test the pattern for creating examples with context episodes."""
        dummy_image = np.zeros((32, 32, 3), dtype=np.uint8)
        
        # Create eval episode
        eval_episode = Episode(
            instruction="Evaluation task",
            starting_frame=dummy_image,
            episode_index=0,
            original_frames_indices=[0, 1, 2],
            original_frames_task_completion_rates=[0, 50, 100],
            shuffled_frames_indices=[1, 0, 2],
            shuffled_frames=[dummy_image] * 3,
            shuffled_frames_approx_completion_rates=[50, 0, 100],
        )
        
        # Create context episodes
        context_episodes = []
        for i in range(2):  # 2 context episodes
            context_episode = Episode(
                instruction=f"Context task {i}",
                starting_frame=dummy_image,
                episode_index=i + 1,
                original_frames_indices=[0, 1],
                original_frames_task_completion_rates=[0, 100],
                shuffled_frames_indices=[0, 1],
                shuffled_frames=[dummy_image] * 2,
                shuffled_frames_approx_completion_rates=[0, 100],
            )
            context_episodes.append(context_episode)
        
        # Create example
        example = Example(
            eval_episode=eval_episode,
            context_episodes=context_episodes,
        )
        
        assert len(example.context_episodes) == 2
        assert example.eval_episode.instruction == "Evaluation task"

    def test_frame_shuffling_preserves_data(self):
        """Test that frame shuffling preserves data integrity."""
        dummy_image = np.zeros((64, 64, 3), dtype=np.uint8)
        
        # Original data
        original_indices = [0, 1, 2, 3, 4]
        original_rates = [0, 25, 50, 75, 100]
        
        # Shuffle
        rng = np.random.default_rng(42)
        shuffled_indices = original_indices.copy()
        rng.shuffle(shuffled_indices)
        
        # Create corresponding shuffled data
        shuffled_frames = [dummy_image] * len(shuffled_indices)
        shuffled_rates = [original_rates[original_indices.index(idx)] for idx in shuffled_indices]
        
        # Verify data integrity
        for i, shuffled_idx in enumerate(shuffled_indices):
            original_pos = original_indices.index(shuffled_idx)
            assert shuffled_rates[i] == original_rates[original_pos]
        
        # Verify all original indices are present
        assert set(shuffled_indices) == set(original_indices)
        assert len(shuffled_indices) == len(original_indices)


class TestDataLoaderIntegration:
    """Integration tests for data loader patterns."""

    def test_concrete_loader_implementation_pattern(self):
        """Test pattern for implementing a concrete data loader."""
        class ConcreteDataLoader(BaseDataLoader):
            def __init__(self, dataset_size=10, **kwargs):
                super().__init__(**kwargs)
                self.dataset_size = dataset_size
                self.current_index = 0
            
            def load_fewshot_input(self, episode_index=None):
                if episode_index is None:
                    episode_index = self.current_index
                    self.current_index = (self.current_index + 1) % self.dataset_size
                
                dummy_image = np.zeros((64, 64, 3), dtype=np.uint8)
                
                # Create eval episode
                eval_episode = Episode(
                    instruction=f"Task {episode_index}",
                    starting_frame=dummy_image,
                    episode_index=episode_index,
                    original_frames_indices=list(range(self.num_frames)),
                    original_frames_task_completion_rates=self._linear_completion(self.num_frames),
                    shuffled_frames_indices=self._get_shuffled_indices(),
                    shuffled_frames=[dummy_image] * self.num_frames,
                    shuffled_frames_approx_completion_rates=self._get_shuffled_rates(),
                )
                
                # Create context episodes
                context_episodes = []
                for i in range(self.num_context_episodes):
                    context_ep = Episode(
                        instruction=f"Context {i}",
                        starting_frame=dummy_image,
                        episode_index=episode_index + i + 1000,  # Unique index
                        original_frames_indices=[0, 1],
                        original_frames_task_completion_rates=[0, 100],
                        shuffled_frames_indices=[0, 1],
                        shuffled_frames=[dummy_image, dummy_image],
                        shuffled_frames_approx_completion_rates=[0, 100],
                    )
                    context_episodes.append(context_ep)
                
                return Example(
                    eval_episode=eval_episode,
                    context_episodes=context_episodes,
                )
            
            def _get_shuffled_indices(self):
                indices = list(range(self.num_frames))
                if self.shuffle:
                    self._rng.shuffle(indices)
                return indices
            
            def _get_shuffled_rates(self):
                original_rates = self._linear_completion(self.num_frames)
                shuffled_indices = self._get_shuffled_indices()
                return [original_rates[i] for i in shuffled_indices]
        
        # Test the concrete loader
        loader = ConcreteDataLoader(
            dataset_size=5,
            num_frames=3,
            num_context_episodes=1,
            shuffle=True,
            seed=42
        )
        
        # Load multiple examples
        examples = loader.load_fewshot_inputs(3)
        
        assert len(examples) == 3
        for example in examples:
            assert isinstance(example, Example)
            assert len(example.eval_episode.shuffled_frames) == 3
            assert len(example.context_episodes) == 1

    def test_loader_reset_affects_randomness(self):
        """Test that reset affects random behavior."""
        class RandomDataLoader(BaseDataLoader):
            def load_fewshot_input(self, episode_index=None):
                # Use randomness in the loader
                random_instruction = f"Task {self._rng.integers(0, 1000)}"
                
                dummy_image = np.zeros((32, 32, 3), dtype=np.uint8)
                episode = Episode(
                    instruction=random_instruction,
                    starting_frame=dummy_image,
                    episode_index=0,
                    original_frames_indices=[0],
                    original_frames_task_completion_rates=[100],
                    shuffled_frames_indices=[0],
                    shuffled_frames=[dummy_image],
                    shuffled_frames_approx_completion_rates=[100],
                )
                
                return Example(eval_episode=episode, context_episodes=[])
        
        loader = RandomDataLoader(seed=42)
        
        # Generate some examples
        examples1 = [loader.load_fewshot_input().eval_episode.instruction for _ in range(3)]
        
        # Reset and generate again
        loader.reset()
        examples2 = [loader.load_fewshot_input().eval_episode.instruction for _ in range(3)]
        
        # Should be identical due to reset
        assert examples1 == examples2