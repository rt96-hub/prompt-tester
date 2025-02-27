"""LLM providers for MCP Prompt Tester."""

from .base import ProviderBase, ProviderError
from .openai import OpenAIProvider
from .anthropic import AnthropicProvider

# Registry of available providers
PROVIDERS = {
    "openai": OpenAIProvider,
    "anthropic": AnthropicProvider,
}

__all__ = ["ProviderBase", "ProviderError", "OpenAIProvider", "AnthropicProvider", "PROVIDERS"] 