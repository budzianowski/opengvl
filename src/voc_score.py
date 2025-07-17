"""Value-Order Correlation (VOC) score"""

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
    if true_values is None:
        raise ValueError("true_values cannot be None")

    if len(values) != len(true_values):
        raise ValueError("values and true_values must have the same length")

    voc, _ = spearmanr(values, true_values)
    return voc
