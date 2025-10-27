from abc import ABC, abstractmethod


class BaseMapper(ABC):
    """
    Abstract base class for extracting percentages from models' answers.
    """

    @abstractmethod
    def extract_percentages(self, model_response: str) -> list[float]:
        """Extract percentage values from the model's response.

        Args:
            model_response: The raw textual output from the model.

        Returns:
            A list of extracted percentage values as floats.
        """
        pass
