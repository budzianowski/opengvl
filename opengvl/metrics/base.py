from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any

from opengvl.utils.data_types import InferredFewShotResult


@dataclass
class MetricResult:
    name: str
    value: float  # Always a concrete float; metrics must normalize to a numeric score
    details: dict[str, Any] = field(default_factory=dict)


class Metric(ABC):
    """Abstract metric interface."""

    @property
    @abstractmethod
    def name(self) -> str:
        pass

    @abstractmethod
    def compute(self, example: InferredFewShotResult) -> MetricResult:
        pass
