import pytest
from gvl.data_loader import DataLoader


@pytest.fixture(scope="module")
def args():
    class Args:
        dataset_name = "lerobot/toto"
        num_context_episodes = 2
        num_frames = 15
        camera_index = 0
        seed = 42

    return Args()


def test_deterministic_loading(args):
    """Tests to ensure the data loading is deterministic and hierarchical."""
    # Test 1: Same seed should produce identical data
    loader1 = DataLoader(
        dataset_name=args.dataset_name,
        num_context_episodes=args.num_context_episodes,
        num_frames=args.num_frames,
        camera_index=args.camera_index,
        seed=args.seed,
    )
    loader2 = DataLoader(
        dataset_name=args.dataset_name,
        num_context_episodes=args.num_context_episodes,
        num_frames=args.num_frames,
        camera_index=args.camera_index,
        seed=args.seed,
    )

    examples1 = loader1.load_examples(3)
    examples2 = loader2.load_examples(3)

    eval_indices1 = [ex.eval_episode.episode_index for ex in examples1]
    eval_indices2 = [ex.eval_episode.episode_index for ex in examples2]

    context_indices1 = [[ep.episode_index for ep in ex.context_episodes] for ex in examples1]
    context_indices2 = [[ep.episode_index for ep in ex.context_episodes] for ex in examples2]

    assert eval_indices1 == eval_indices2, "Test 1 Failed: Eval episodes are not deterministic."
    assert context_indices1 == context_indices2, "Test 1 Failed: Context episodes are not deterministic."


def test_different_seeds(args):
    """Test 2: Different seeds should produce different data"""
    loader1 = DataLoader(
        dataset_name=args.dataset_name,
        num_context_episodes=args.num_context_episodes,
        num_frames=args.num_frames,
        camera_index=args.camera_index,
        seed=args.seed,
    )
    examples1 = loader1.load_examples(2)
    eval_indices1 = [ex.eval_episode.episode_index for ex in examples1]

    loader3 = DataLoader(
        dataset_name=args.dataset_name,
        num_context_episodes=args.num_context_episodes,
        num_frames=args.num_frames,
        camera_index=args.camera_index,
        seed=args.seed + 1,
    )
    examples3 = loader3.load_examples(2)
    eval_indices3 = [ex.eval_episode.episode_index for ex in examples3]

    assert eval_indices1 != eval_indices3, "Test 2 Failed: Different seeds produced the same eval episodes."


def test_hierarchical_context(args):
    """Test 3: Hierarchical context"""
    loader_1_ctx = DataLoader(dataset_name=args.dataset_name, num_context_episodes=1, seed=args.seed)
    loader_2_ctx = DataLoader(dataset_name=args.dataset_name, num_context_episodes=2, seed=args.seed)

    example_1_ctx = loader_1_ctx.load_example()
    example_2_ctx = loader_2_ctx.load_example()

    assert (
        example_1_ctx.eval_episode.episode_index == example_2_ctx.eval_episode.episode_index
    ), "Test 3 Failed: Eval episodes should be the same for hierarchical test."

    context_1 = [ep.episode_index for ep in example_1_ctx.context_episodes]
    context_2 = [ep.episode_index for ep in example_2_ctx.context_episodes]

    assert len(context_1) == 1, "Test 3 Failed: Incorrect number of context episodes for loader 1."
    assert len(context_2) == 2, "Test 3 Failed: Incorrect number of context episodes for loader 2."
    assert context_1[0] == context_2[0], "Test 3 Failed: Context is not hierarchical."
