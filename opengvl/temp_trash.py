parser = argparse.ArgumentParser()
    
    # Batch evaluation mode
    parser.add_argument(
        "--batch_eval",
        action="store_true",
        help="Run batch evaluation on existing results file"
    )
    parser.add_argument(
        "--results_file",
        type=str,
        help="Path to results JSONL file for batch evaluation"
    )
    parser.add_argument(
        "--output_file",
        type=str,
        help="Path to save updated results (if not specified, overwrites input file)"
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=8,
        help="Batch size for model inference during batch evaluation",
    )
    
    # Regular evaluation mode arguments
    parser.add_argument("--name", type=str, default="lerobot/fmb", help="Dataset name")
    parser.add_argument(
        "--max_frames", 
        type=int, default=10, 
        help="Maximum number of frames to select per episode"
    )
    parser.add_argument(
        "--num_eval_steps", 
        type=int, 
        default=5, 
        help="Number of evaluation steps to run"
    )
    parser.add_argument(
        "--num_context_episodes", 
        type=int, 
        default=2, help="Number of context episodes to use"
    )
    parser.add_argument(
        "--model",
        type=str,
        default="google/gemma-3-4b-it",
        help="Model to use for inference",
    )
    parser.add_argument(
        "--camera_index",
        type=int,
        default=0,
        help="Camera index to use",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="results",
        help="Directory to save results",
    )
    parser.add_argument(
        "--experiment_name",
        type=str,
        default=None,
        help="Name for the experiment (default: auto-generated)",
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Resume from existing results file",
    )
    
    args = parser.parse_args()