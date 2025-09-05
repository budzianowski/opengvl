""" VOC score calculator """

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from scipy.stats import spearmanr


class VOCScorer:
    """
    A class to calculate, analyze, and visualize Value-Order Correlation (VOC) scores.

    This class processes a file containing model predictions and shuffled indices
    to compute VOC scores, derives key statistics, and generates a histogram of
    the score distribution.
    """

    def __init__(self):
        self.voc_scores = []
        self.skipped_stats = {"empty_preds": 0, "no_indices": 0, "mismatch_length": 0, "total_processed": 0}

    def _calculate_voc(self, chronologically_ordered_preds: list[float]) -> float:

        predicted_values = np.array(chronologically_ordered_preds)
        T = len(predicted_values)

        # The chronological order is just an increasing sequence.
        chronological_order = np.arange(T)

        # if all predictions are the same, return 0 to avoid division by zero
        if np.all(predicted_values == predicted_values[0]):
            return 0.0

        # The spearmanr function directly computes the rank-order correlation.
        voc_score, _ = spearmanr(predicted_values, chronological_order)

        return voc_score

    def process_file(self, file_path: str | Path):
        """
        Reads a JSONL file, processes each line, and computes VOC scores.

        Args:
            file_path: The path to the input JSONL file.
        """
        file_path = Path(file_path)
        if not file_path.exists():
            raise FileNotFoundError(f"File not found: {file_path}")

        with file_path.open("r") as f:
            for line in f:
                self.skipped_stats["total_processed"] += 1
                rec = json.loads(line)

                preds = rec.get("extracted_percentages", [])
                shuffled_indices = rec.get("eval_episode", {}).get("shuffled_frames_indices")

                print(preds)

                if not preds:
                    self.skipped_stats["empty_preds"] += 1
                    continue
                if not shuffled_indices:
                    self.skipped_stats["no_indices"] += 1
                    continue

                if len(shuffled_indices) != len(preds):
                    self.skipped_stats["mismatch_length"] += 1
                    continue

                try:
                    sorted_indices = np.argsort(shuffled_indices)
                    chrono_preds = np.array(preds)[sorted_indices].tolist()
                except IndexError:
                    self.skipped_stats["mismatch_length"] += 1
                    continue

                # --- Calculate and store VOC score ---
                voc = self._calculate_voc(chrono_preds)
                self.voc_scores.append(voc)

    def get_statistics(self) -> dict:
        """
        Calculates descriptive statistics for the collected VOC scores.

        Returns:
            A dictionary containing key statistics.
        """
        if not self.voc_scores:
            return {"message": "No VOC scores were calculated. Cannot generate stats."}

        scores = np.array(self.voc_scores)
        stats = {
            "count": len(scores),
            "mean": np.mean(scores),
            "median": np.median(scores),
            "std_dev": np.std(scores),
            "min": np.min(scores),
            "max": np.max(scores),
            "skipped_counts": self.skipped_stats,
        }
        return stats

    def plot_histogram(self, title: str = f"Distribution of VOC Scores", save_path: str | None = None):
        """
        Generates and displays a histogram of the VOC scores.

        Args:
            title: The title for the plot.
            save_path: Optional path to save the plot image.
        """
        if not self.voc_scores:
            print("No VOC scores available to plot.")
            return

        plt.style.use("seaborn-v0_8-whitegrid")
        fig, ax = plt.subplots(figsize=(10, 6))

        sns.histplot(self.voc_scores, kde=True, ax=ax, bins=20, color="skyblue", edgecolor="black")

        stats = self.get_statistics()
        mean_score = stats["mean"]
        median_score = stats["median"]

        ax.axvline(mean_score, color="red", linestyle="--", linewidth=2, label=f"Mean: {mean_score:.2f}")
        ax.axvline(median_score, color="green", linestyle="-", linewidth=2, label=f"Median: {median_score:.2f}")

        ax.set_title(title, fontsize=16, weight="bold")
        ax.set_xlabel("VOC Score", fontsize=12)
        ax.set_ylabel("Frequency", fontsize=12)
        ax.legend()

        plt.tight_layout()

        if save_path:
            plt.savefig(save_path)
            print(f"Plot saved to {save_path}")

        plt.show()


if __name__ == "__main__":
    # python src/voc_score.py --file_path /Users/pfb30/working_dir/arm/gvl/results/gvl_eval_20250723_121653_results.jsonl
    # python src/voc_score.py --file_path results/gvl_eval_20250723_121139_results.jsonl
    args = argparse.ArgumentParser()
    args.add_argument("--file_path", type=str, required=True)
    args = args.parse_args()

    scorer = VOCScorer()
    try:
        scorer.process_file(Path(args.file_path))

        statistics = scorer.get_statistics()
        print("--- VOC Score Statistics ---")
        for key, value in statistics.items():
            if isinstance(value, dict):
                print(f"{key}:")
                for sub_key, sub_value in value.items():
                    print(f"  {sub_key}: {sub_value}")
            else:
                print(f"{key}: {value:.4f}" if isinstance(value, float) else f"{key}: {value}")
        print("--------------------------\n")

        scorer.plot_histogram(save_path="voc_distribution.png", title=f"VOC Score Distribution for {args.file_path}")

    except FileNotFoundError as e:
        print(e)
