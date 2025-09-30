# ruff: noqa: UP007 (Allow use of typing.Union for Python <3.10 compatibility in some environments)
from collections.abc import Sequence
from typing import Union

import numpy as np
from loguru import logger
from scipy.stats import spearmanr
from scipy.stats._stats_py import SignificanceResult

from opengvl.metrics.base import Metric, MetricResult
from opengvl.utils.data_types import InferredFewShotResult


def value_order_correlation(
    values: Union[Sequence[float], np.ndarray],
    true_values: Union[Sequence[float], np.ndarray],
) -> float:
    if values is None or true_values is None:  # type: ignore[truthy-bool]
        raise ValueError("values and true_values must not be None")

    # Convert to numpy arrays (copy not required) with float dtype for safety
    v = np.asarray(values, dtype=float)
    t = np.asarray(true_values, dtype=float)

    if v.shape[0] != t.shape[0]:
        raise ValueError(
            f"values and true_values must have the same length (got {v.shape[0]} and {t.shape[0]})"
        )
    if v.size == 0:
        return float('nan')
    if v.size < 2:
        return float('nan')
    # If either array is constant, Spearman is undefined -> NaN
    if np.allclose(v, v[0]) or np.allclose(t, t[0]):
        return float('nan')

    # spearmanr returns a SignificanceResult (SciPy >=1.11) or a tuple in older versions; we type it broadly.
    corr: SignificanceResult = spearmanr(v, t)  # type: ignore[assignment]
    score = float(corr.statistic if hasattr(corr, "statistic") else corr[0])  # type: ignore[index]
    return score


class VOCMetric(Metric):
    """Value-Order Correlation (VOC) as Spearman correlation between predicted
    completion percentages (reordered into chronological order) and their index.
    """

    @property
    def name(self) -> str:
        return "voc"

    def compute(self, example: InferredFewShotResult) -> MetricResult:
        eval_ep = example.eval_episode
        preds = np.array(eval_ep.shuffled_frames_predicted_completion_rates, dtype=float)
        # reorder predictions into chronological order by sorting shuffled indices
        order = np.argsort(eval_ep.shuffled_frames_indices)
        chrono = preds[order]
        logger.debug(f"VOC compute | preds_shape={preds.shape} order_shape={order.shape}")
        # Use the standalone function so implementation logic is unified.
        corr_value = value_order_correlation(chrono, np.arange(len(chrono)))
        if np.isnan(corr_value):
            # Map undefined correlations to 0.0 for downstream aggregation while
            # preserving a note about the degeneracy.
            note = (
                "insufficient length" if len(chrono) <= 1 else "constant predictions"
                if np.allclose(chrono, chrono[0]) else "undefined correlation"
            )
            return MetricResult(name=self.name, value=0.0, details={"note": note})
        return MetricResult(name=self.name, value=float(corr_value))
