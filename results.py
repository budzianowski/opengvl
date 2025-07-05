import json

import numpy as np

from tools_for_data_extraction import generate_and_check_lists
from voc_score import value_order_correlation

with open("collected_data_bridge_incontext_5.json", "r") as f:
    sample_json_results = json.load(f)
print(sample_json_results)
# 2. PROCESSING LOGIC
all_voc_scores = []
print("Starting VOC score processing...\n" + "=" * 40)

for i, result in sample_json_results.items():
    print(i)
    print(f"\nProcessing Result #{i}...")

    # Extract data from the JSON-like dictionary
    response_text = result["response"]
    absolute_idx = result["absolute_indices"]

    # We use a slice_start_index of 0 to indicate we want to use the full list
    # of absolute_indices. This is the most common use case.
    slice_start_index = 0

    # Use the function from our last conversation
    percentages, indices = generate_and_check_lists(response_text, absolute_idx, 5)

    # print(f"Extracted Percentages: {percentages}")
    # print(f"Extracted Indices: {indices}")

    # Only proceed if the lists are valid (not empty)
    if percentages and indices:
        # NOTE: The original `generate_and_check_lists` function does not guarantee
        # that the `percentages` correspond to the `indices`. It creates two
        # separate lists. Let's assume for VOC calculation that the Nth percentage
        # corresponds to the Nth index in the `indices` list.

        # Calculate the VOC score for this valid result

        percentages_int = [int(p) for p in percentages]

        combined_data = list(zip(percentages_int, indices))
        # Sort by indices to ensure chronological order
        combined_data.sort(key=lambda x: x[1])

        # Unzip the sorted data
        percentages, indices = zip(*combined_data)

        voc_score = value_order_correlation(percentages)
        all_voc_scores.append(voc_score)
        print(f"Result #{i} -> VOC Score: {voc_score:.4f}")
    else:
        print(f"Result #{i} -> Skipped due to data mismatch.")


# 3. FINAL CALCULATION
if all_voc_scores:
    average_voc_score = np.mean(all_voc_scores)
    print("\n" + "=" * 40)
    print(f"Processing Complete.")
    print(
        f"Average VOC Score across {len(all_voc_scores)} valid results: {average_voc_score:.4f}"
    )
else:
    print(
        "\nProcessing Complete. No valid results found to calculate an average score."
    )
