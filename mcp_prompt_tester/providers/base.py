"""Base classes for LLM providers."""

from abc import ABC, abstractmethod
from typing import Dict, Any, Optional


class ProviderError(Exception):
    """Exception raised for errors in the LLM provider."""
    pass


class ProviderBase(ABC):
    """Base class for LLM providers."""

    @abstractmethod
    async def generate(
        self,
        model: str,
        system_prompt: str,
        user_prompt: str,
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None,
        top_p: Optional[float] = None,
        **kwargs: Any,
    ) -> Dict[str, Any]:
        """
        Generate a response from the LLM.

        Args:
            model: The model name to use
            system_prompt: The system prompt
            user_prompt: The user prompt
            temperature: Controls randomness (0.0 to 1.0)
            max_tokens: Maximum number of tokens to generate
            top_p: Controls diversity via nucleus sampling
            **kwargs: Additional provider-specific parameters

        Returns:
            A dictionary containing the generated response and metadata
        """
        pass

    @classmethod
    @abstractmethod
    def get_default_models(cls) -> Dict[str, str]:
        """
        Get the default/recommended models for this provider.
        
        Returns:
            A dictionary mapping model type to model name
        """
        pass 