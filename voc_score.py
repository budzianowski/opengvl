import numpy as np
from scipy.stats import spearmanr


def value_order_correlation(values: list[int]):
    """
    Computes the Value-Order Correlation (VOC) for a sequence of predicted values.

    Args:
        values (list or np.ndarray): Sequence of predicted values v_1, ..., v_T.
        The values are between 0 and 100.

    Returns:
        float: Spearman rank correlation coefficient between the predicted value order
               and the chronological order. Returns 0 for constant input.
    """
    # It's good practice to convert the input to a numpy array
    values = np.asarray(values)

    # A simple check for constant input is more direct
    if np.all(values == values[0]):
        return 0.0

    T = len(values)
    # Handle cases with less than 2 elements where correlation is not well-defined
    if T < 2:
        return 0.0

    voc, _ = spearmanr(values, np.arange(T))

    # spearmanr can return nan in cases of no variance, so we check for it.
    if np.isnan(voc):
        return 0.0

    return voc


def test_value_order_correlation():
    # Increasing order
    assert np.isclose(
        value_order_correlation([0, 25, 50, 75, 100], [0, 1, 2, 3, 4]), 1.0
    )
    # Decreasing order
    assert np.isclose(
        value_order_correlation([100, 75, 50, 25, 0], [4, 3, 2, 1, 0]), 1.0
    )

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
