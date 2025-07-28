



def run_deterministic_tests(args):
    """Runs a series of tests to ensure the data loading is deterministic and hierarchical."""
    print("\n--- Running Deterministic Loading Tests ---")

    # Test 1: Same seed should produce identical data
    print("\n[Test 1] Checking for determinism with the same seed...")
    print("="*100)
    loader1 = DataLoader(
        dataset_name=args.dataset_name, num_context_episodes=args.num_context_episodes,
        num_frames=args.num_frames, camera_index=args.camera_index, seed=args.seed
    )
    print("="*100)
    loader2 = DataLoader(
        dataset_name=args.dataset_name, num_context_episodes=args.num_context_episodes,
        num_frames=args.num_frames, camera_index=args.camera_index, seed=args.seed
    )

    print("g="*100)
    examples1 = loader1.load_examples(3)
    print("g="*100)

    examples2 = loader2.load_examples(3)

    eval_indices1 = [ex.eval_episode.episode_index for ex in examples1]
    eval_indices2 = [ex.eval_episode.episode_index for ex in examples2]

    print(f"Eval indices from loader 1: {eval_indices1}")
    print(f"Eval indices from loader 2: {eval_indices2}")

    context_indices1 = [[ep.episode_index for ep in ex.context_episodes] for ex in examples1]
    context_indices2 = [[ep.episode_index for ep in ex.context_episodes] for ex in examples2]

    print(f"Context indices from loader 1: {context_indices1}")
    print(f"Context indices from loader 2: {context_indices2}")

    assert eval_indices1 == eval_indices2, "Test 1 Failed: Eval episodes are not deterministic."
    assert context_indices1 == context_indices2, "Test 1 Failed: Context episodes are not deterministic."
    print("  [PASS] Two loaders with the same seed produced identical episode sequences.")

    # Test 2: Different seeds should produce different data
    print("\n[Test 2] Checking for different results with different seeds...")
    loader3 = DataLoader(
        dataset_name=args.dataset_name, num_context_episodes=args.num_context_episodes,
        num_frames=args.num_frames, camera_index=args.camera_index, seed=args.seed + 1
    )
    examples3 = loader3.load_examples(2)
    eval_indices3 = [ex.eval_episode.episode_index for ex in examples3]

    assert eval_indices1 != eval_indices3, "Test 2 Failed: Different seeds produced the same eval episodes."
    print("  [PASS] Two loaders with different seeds produced different episode sequences.")

    # Test 3: Hierarchical context
    print("\n[Test 3] Checking for hierarchical context...")
    loader_1_ctx = DataLoader(
        dataset_name=args.dataset_name, num_context_episodes=1, seed=args.seed
    )
    loader_2_ctx = DataLoader(
        dataset_name=args.dataset_name, num_context_episodes=2, seed=args.seed
    )

    example_1_ctx = loader_1_ctx.load_example()
    example_2_ctx = loader_2_ctx.load_example()

    assert example_1_ctx.eval_episode.episode_index == example_2_ctx.eval_episode.episode_index, \
        "Test 3 Failed: Eval episodes should be the same for hierarchical test."

    context_1 = [ep.episode_index for ep in example_1_ctx.context_episodes]
    context_2 = [ep.episode_index for ep in example_2_ctx.context_episodes]

    print(f"Context episodes from loader 1: {context_1}")
    print(f"Context episodes from loader 2: {context_2}")

    assert len(context_1) == 1, "Test 3 Failed: Incorrect number of context episodes for loader 1."
    assert len(context_2) == 2, "Test 3 Failed: Incorrect number of context episodes for loader 2."
    assert context_1[0] == context_2[0], "Test 3 Failed: Context is not hierarchical."
    print("  [PASS] Context is hierarchical and consistent.")

    print("\n--- All Deterministic Tests Passed Successfully! ---")


if __name__ == "__main__":
    # python src/data_loader.py --dataset_name lerobot/fmb --num_context_episodes 2 --num_frames 15 --camera_index 0
    parser = argparse.ArgumentParser()
    # lerobot/fmb, lerobot/utaustin_mutex, lerobot/toto
    parser.add_argument("--dataset_name", type=str, default="lerobot/toto")
    parser.add_argument("--num_context_episodes", type=int, default=2)
    parser.add_argument("--num_frames", type=int, default=15)
    parser.add_argument("--camera_index", type=int, default=0)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    print(f"Testing")
    # Run the deterministic tests
    run_deterministic_tests(args)

    # Demonstrate plotting
    loader = DataLoader(
        dataset_name=args.dataset_name, num_context_episodes=args.num_context_episodes,
        num_frames=args.num_frames, camera_index=args.camera_index, seed=args.seed
    )
    example = loader.load_example()
    if example:
        print("\n--- Plotting Sampled Eval Episode ---")
        loader.plot_single_episode(example, plot_eval=True)

        print(f"\n--- Plotting Whole Eval Episode ({example.eval_episode.episode_index}) ---")
        loader.plot_whole_episode(example.eval_episode.episode_index)
