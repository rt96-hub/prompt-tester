"""OpenAI provider implementation."""

import time
from typing import Dict, Any, Optional

import openai
from openai import OpenAI, APIError, APIConnectionError, RateLimitError

from .base import ProviderBase, ProviderError
from ..env import get_api_key


class OpenAIProvider(ProviderBase):
    """Provider implementation for OpenAI API."""

    def __init__(self):
        """Initialize the OpenAI provider with API key from environment."""
        try:
            # Get API key using the utility function
            api_key = get_api_key("openai")
            self.client = OpenAI(api_key=api_key)
        except ValueError as e:
            raise ProviderError(str(e))

    async def generate(
        self,
        model: str,
        system_prompt: str,
        user_prompt: str,
        temperature: Optional[float] = 0.7,
        max_tokens: Optional[int] = 1000,
        top_p: Optional[float] = 1.0,
        **kwargs: Any,
    ) -> Dict[str, Any]:
        """Generate a response using OpenAI's API."""
        try:
            # Prepare the messages
            messages = [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt}
            ]

            # Build the request params
            request_params = {
                "model": model,
                "messages": messages,
                "temperature": temperature if temperature is not None else 0.7,
                "max_tokens": max_tokens if max_tokens is not None else 1000,
                "top_p": top_p if top_p is not None else 1.0,
            }
            
            # Add any additional kwargs
            for key, value in kwargs.items():
                if value is not None:
                    request_params[key] = value
            
            # Make the API call
            start_time = time.time()
            response = self.client.chat.completions.create(**request_params)
            end_time = time.time()
            response_time = end_time - start_time
            
            # Get the generated text
            generated_text = response.choices[0].message.content
            
            # Prepare the result
            result = {
                "text": generated_text,
                "model": model,
                "usage": {
                    "prompt_tokens": response.usage.prompt_tokens,
                    "completion_tokens": response.usage.completion_tokens,
                    "total_tokens": response.usage.total_tokens,
                },
                "finish_reason": response.choices[0].finish_reason,
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
        """Get the default OpenAI models."""
        return {
            "fast": "gpt-4o-mini",
            "smart": "o1-mini",
            "vision": "gpt-4o",
        } 