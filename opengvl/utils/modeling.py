from opengvl.clients.base import BaseModelClient
from opengvl.clients.gemini import GeminiClient
from opengvl.clients.gemma import GemmaClient
from opengvl.clients.kimi import KimiThinkingClient
from opengvl.clients.openai import OpenAIClient


class ModelFactory:
    """Factory class to create model clients"""

    @staticmethod
    def create_client(model_name: str) -> BaseModelClient:
        if "gemma" in model_name.lower():
            return GemmaClient()
        if "gpt" in model_name.lower():
            return OpenAIClient()
        elif "kimi" in model_name.lower():
            return KimiThinkingClient()
        elif "gemini" in model_name.lower() or "gemma" in model_name.lower():
            return GeminiClient(model_name=model_name)
        else:
            raise ValueError(f"Unknown model: {model_name}")
