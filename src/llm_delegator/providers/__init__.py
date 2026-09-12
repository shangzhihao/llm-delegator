"""Model provider adapters."""

from llm_delegator.providers.base import ModelProvider
from llm_delegator.providers.deepseek import DeepSeekProvider

__all__ = ["DeepSeekProvider", "ModelProvider"]
