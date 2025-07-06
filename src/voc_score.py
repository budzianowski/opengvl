""" Value-Order Correlation (VOC) score """

import re

from scipy.stats import spearmanr


def value_order_correlation(values: list[int], true_values: list[int]):
    """
    Computes the Value-Order Correlation (VOC) for a sequence of predicted values.

    Args:
        values (list or np.ndarray): Sequence of predicted values v_1, ..., v_T.
        the values are between 0 and 100
        true_values (list or np.ndarray): Sequence of true values t_1, ..., t_T.
    Returns:
        float: Spearman rank correlation coefficient between the predicted value order
               and the chronological order.
    """
    if len(values) != len(true_values):
        raise ValueError("values and true_values must have the same length")
    
    voc, _ = spearmanr(values, true_values)
    return voc


# TODO: this should be handled by simple LLM
def parse_response(response: str) -> list[int]:
    """
    Parse the response from the model and return the predicted values.

    To predict task completion percentages for each frame, we analyze the stages of the task (picking up and inserting the green object) based on the progression shown in the example images. Here's the assessment:

    - **Frame 1**: The gripper is approaching the green object.
    - **Task Completion**: Approximately 18.9%

    - **Frame 2**: The gripper is closer to the green object.
    - **Task Completion**: Approximately 22.5%

    - **Frame 3**: The gripper is touching the green object, about to grip it.
    - **Task Completion**: Approximately 32.4%

    - **Frame 4**: The green object has been lifted from its initial position.
    - **Task Completion**: Approximately 50%

    - **Frame 5**: The green object is above the target area, ready for insertion.
    - **Task Completion**: Approximately 75%

    - **Frame 6**: The green object is inserted, task likely complete.
    - **Task Completion**: 100%

    These predictions are based on observed stages of picking up and inserting the object, in alignment with given examples.
    """
    # Pattern to match task completion percentages
    completion_pattern = r"(?:Task Completion|Completion).*?(\d+(?:\.\d+)?)%"

    # Find all completion percentages
    matches = re.findall(completion_pattern, response, re.IGNORECASE)

    # Convert to integers (round float values)
    values = [int(float(match)) for match in matches]

    return values
