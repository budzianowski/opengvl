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


def test_value_order_correlation():
    # Increasing order
    assert np.isclose(value_order_correlation([0, 25, 50, 75, 100], [0, 1, 2, 3, 4]), 1.0)
    # Decreasing order
    assert np.isclose(value_order_correlation([100, 75, 50, 25, 0], [4, 3, 2, 1, 0]), 1.0)

    # Constant values
    assert np.isnan(value_order_correlation([50, 50, 50, 50, 50], [0, 1, 2, 3, 4]))
    # Two values increasing
    assert np.isclose(value_order_correlation([0, 100], [0, 1]), 1.0)
    # Two values decreasing
    assert np.isclose(value_order_correlation([100, 0], [0, 1]), -1.0)
    # Mixed order
    assert np.isclose(value_order_correlation([25, 100, 0], [0, 1, 2]), -0.5)
    # Random order
    v = [75, 0, 100, 25]
    voc = value_order_correlation(v, [0, 1, 2, 3])
    assert -1.0 <= voc <= 1.0


if __name__ == "__main__":
    test_value_order_correlation()
    print("All tests passed.")