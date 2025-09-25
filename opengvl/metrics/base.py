from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any

from opengvl.utils.data_types import InferredFewShotResult


@dataclass
class MetricResult:
    name: str
    value: float | None
    details: dict[str, Any] | None = None


class Metric(ABC):
    """Abstract metric interface."""

    @property
    @abstractmethod
    def name(self) -> str:
        pass

    @abstractmethod
    def compute(self, example: InferredFewShotResult) -> MetricResult:
        pass
