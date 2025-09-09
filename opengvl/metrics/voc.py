"""VOC metric implementation using Spearman rank correlation."""

from dataclasses import dataclass

import numpy as np
from scipy.stats import spearmanr
from scipy.stats._stats_py import SignificanceResult

from opengvl.metrics.base import Metric, MetricResult
from opengvl.utils.data_types import InferredExample


@dataclass
class VOCMetric(Metric):
    """Value-Order Correlation (VOC) as Spearman correlation between predicted
    completion percentages (reordered into chronological order) and their index.
    """

    @property
    def name(self) -> str:
        return "voc"

    def compute(self, example: InferredExample) -> MetricResult:
        eval_ep = example.eval_episode
        preds = np.array(eval_ep.shuffled_frames_predicted_completion_rates, dtype=float)
        # reorder predictions into chronological order by sorting shuffled indices
        order = np.argsort(eval_ep.shuffled_frames_indices)
        chrono = preds[order]
        if len(chrono) <= 1:
            return MetricResult(name=self.name, value=0.0, details={"note": "insufficient length"})
        if np.allclose(chrono, chrono[0]):
            return MetricResult(name=self.name, value=0.0, details={"note": "constant predictions"})
        corr: SignificanceResult = spearmanr(chrono, np.arange(len(chrono)))
        score = float(corr.statistic if hasattr(corr, "statistic") else corr[0])  # type: ignore[index]
        if np.isnan(score):
            score = 0.0
        return MetricResult(name=self.name, value=score)
