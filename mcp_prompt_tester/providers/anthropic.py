"""Anthropic provider implementation."""

import time
from typing import Dict, Any, Optional

import anthropic
from anthropic import Anthropic, APIConnectionError, APIError, RateLimitError

from .base import ProviderBase, ProviderError
from ..env import get_api_key


class AnthropicProvider(ProviderBase):
    """Provider implementation for Anthropic API."""

    def __init__(self):
        """Initialize the Anthropic provider with API key from environment."""
        try:
            # Get API key using the utility function
            api_key = get_api_key("anthropic")
            self.client = Anthropic(api_key=api_key)
        except ValueError as e:
            raise ProviderError(str(e))

    async def generate(
        self,
        model: str,
        system_prompt: str,
        user_prompt: str,
        temperature: Optional[float] = 0.7,
        max_tokens: Optional[int] = 1000,
        top_p: Optional[float] = None,
        **kwargs: Any,
    ) -> Dict[str, Any]:
        """Generate a response using Anthropic's API."""
        try:
            # Build the request params
            request_params = {
                "model": model,
                "system": system_prompt,
                "messages": [
                    {"role": "user", "content": user_prompt}
                ],
                "temperature": temperature if temperature is not None else 0.7,
                "max_tokens": max_tokens if max_tokens is not None else 1000,
            }
            
            # Add top_p if provided (Anthropic might not use this parameter)
            if top_p is not None:
                request_params["top_p"] = top_p
            
            # Add any additional kwargs
            for key, value in kwargs.items():
                if value is not None and key not in ["model", "system", "messages", "temperature", "max_tokens", "top_p"]:
                    request_params[key] = value
            
            # Make the API call
            start_time = time.time()
            response = self.client.messages.create(**request_params)
            end_time = time.time()
            response_time = end_time - start_time
            
            # Get the generated text
            content_blocks = [block for block in response.content if block.type == "text"]
            if not content_blocks:
                raise ProviderError("No text content in response")
            
            generated_text = content_blocks[0].text
            
            # Prepare the result
            result = {
                "text": generated_text,
                "model": model,
                "usage": {
                    "input_tokens": response.usage.input_tokens,
                    "output_tokens": response.usage.output_tokens,
                },
                "stop_reason": response.stop_reason,
                "response_time": response_time,
            }
            
            return result
        
        except APIConnectionError as e:
            raise ProviderError(f"Connection error: {str(e)}")
        except RateLimitError as e:
            raise ProviderError(f"Rate limit exceeded: {str(e)}")
        except APIError as e:
            raise ProviderError(f"API error: {str(e)}")
        except Exception as e:
            raise ProviderError(f"Unexpected error: {str(e)}")

    # TODO: We should call the API to get available models
    #  the only issue is that we don't have any information about the models themselves, quality, token limits, etc.
    @classmethod
    def get_default_models(cls) -> Dict[str, str]:
        """Get the default Anthropic models."""
        return {
            "fast": "claude-3-5-haiku-20241022",
            "balanced": "claude-3-5-sonnet-20240620",
            "smart": "claude-3-opus-20240229",
        } 