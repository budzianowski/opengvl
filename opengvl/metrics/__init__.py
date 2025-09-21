"""Metrics package: provides Metric implementations (e.g., VOC)."""

from .base import Metric, MetricResult  # noqa: F401
from .voc import VOCMetric  # noqa: F401
