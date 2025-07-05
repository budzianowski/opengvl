import re
from typing import List, Tuple


def generate_and_check_lists(
    data_string: str, indices_to_check: List[int], slice_start_index: int
) -> Tuple[List[float], List[int]]:
    """
    Performs two independent operations and returns the results only if the list lengths match.

    1.  Parses the data_string to get a list of all "Task Completion" percentages
        in the order they appear.
    2.  Slices the `indices_to_check` list using the `slice_start_index`.
    3.  Compares the lengths of the two generated lists. If they are equal, it
        returns a tuple containing both lists. Otherwise, it returns a tuple
        of two empty lists.

    Args:
        data_string: A string containing the data for one or more frames.
        indices_to_check: The original list of integer frame indexes.
        slice_start_index: An integer specifying the starting index for slicing.

    Returns:
        A tuple containing (list_of_percentages, sliced_list_of_indices) if
        their lengths are equal, otherwise returns ([], []).
    """
    # 1. Create a list with values from Task Completion in the Frame order.
    #    This regex finds all occurrences of the percentage value.
    percentages_from_text = [
        float(p) for p in re.findall(r"Task Completion: ([\d.]+)%", data_string)
    ]

    # 2. Create the second list by slicing `indices_to_check`.
    sliced_indices = indices_to_check[slice_start_index:]

    # 3. Check if the two lists have the same length.
    print(f"--- Checking Lengths ---")
    # print(f"Length of percentages list from text: {len(percentages_from_text)}")
    # print(f"Length of sliced indices list: {len(sliced_indices)}")

    if len(percentages_from_text) == len(sliced_indices):
        print("Lengths match. Returning the lists.")
        return (percentages_from_text, sliced_indices)
    else:
        percentages_from_text = [0.0] + percentages_from_text
        print(f"Length of percentages list from text: {len(percentages_from_text)}")
        print(f"Length of sliced indices list: {len(sliced_indices)}")
        return (percentages_from_text, sliced_indices)
