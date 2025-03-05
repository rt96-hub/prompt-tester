"""Anthropic provider implementation."""

import time
from typing import Dict, Any, Optional

from langfuse.decorators import observe
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

    @observe(as_type="generation")
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
            
            # Calculate cost if the model is in our default models
            default_models = self.get_default_models()
            model_info = None
            
            # Find the model in our default models
            for model_type, info in default_models.items():
                if info["name"] == model:
                    model_info = info
                    break
            
            # If we have pricing data for this model, calculate and add costs
            if model_info:
                input_cost = (response.usage.input_tokens / 1_000_000) * model_info["input_cost"]
                output_cost = (response.usage.output_tokens / 1_000_000) * model_info["output_cost"]
                total_cost = input_cost + output_cost
                
                result["costs"] = {
                    "input_cost": input_cost,
                    "output_cost": output_cost,
                    "total_cost": total_cost,
                    "currency": "USD",
                    "rates": {
                        "input_rate": model_info["input_cost"],
                        "output_rate": model_info["output_cost"],
                    }
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
    def get_default_models(cls) -> Dict[str, Dict[str, Any]]:
        """Get the default Anthropic models with pricing information."""
        return {
            "fast": {
                "name": "claude-3-5-haiku-20241022",
                "input_cost": 0.8,  # $0.80 per million tokens
                "output_cost": 4.0,  # $4.00 per million tokens
                "description": "Fastest Claude model with strong reasoning capabilities"
            },
            "balanced": {
                "name": "claude-3-5-sonnet-20240620",
                "input_cost": 3.0,  # $3.00 per million tokens
                "output_cost": 15.0,  # $15.00 per million tokens
                "description": "Balanced performance and quality with strong reasoning"
            },
            "smart": {
                "name": "claude-3-opus-20240229",
                "input_cost": 15.0,  # $15.00 per million tokens
                "output_cost": 75.0,  # $75.00 per million tokens
                "description": "Most powerful Claude model with superior reasoning"
            },
        } 