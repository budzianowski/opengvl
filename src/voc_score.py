from __future__ import annotations

import numpy as np
from scipy.stats import spearmanr


def value_order_correlation(values: list[int], time_order: list[int]):
    """
    Computes the Value-Order Correlation (VOC) for a sequence of predicted values.

    Args:
        values (list or np.ndarray): Sequence of predicted values v_1, ..., v_T.
        the values are between 0 and 100
        time_order (list or np.ndarray): Sequence of time indices t_1, ..., t_T.
    Returns:
        float: Spearman rank correlation coefficient between the predicted value order
               and the chronological order.
    """
    time_order_values = [values[i] for i in time_order]
    T = len(values)
    voc, _ = spearmanr(time_order_values, np.arange(T))
    return voc
