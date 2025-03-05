"""Tool for comparing multiple prompts across different providers and models."""

import json
import asyncio
from mcp import types
from langfuse.decorators import observe

from ..providers import PROVIDERS, ProviderError
from ..env import get_api_key


@observe()
async def test_comparison(arguments: dict) -> types.TextContent:
    """
    Compares multiple prompts side-by-side, allowing different providers, models, and parameters.
    """
    try:
        # 1. Input Validation and Configuration
        comparisons = arguments.get("comparisons")
        if not comparisons or not isinstance(comparisons, list):
            return types.TextContent(
                type="text",
                text=json.dumps({"isError": True, "error": "The 'comparisons' argument must be a non-empty list."})
            )

        if not 1 <= len(comparisons) <= 4:
            return types.TextContent(
                type="text",
                text=json.dumps({"isError": True, "error": "You can compare between 1 and 4 configurations."})
            )

        # 2. Prepare and Execute Comparison Runs (Asynchronously)
        async def run_comparison(config: dict) -> dict:
            """Helper function to run a single comparison."""
            provider_name = config.get("provider")
            model = config.get("model")
            system_prompt = config.get("system_prompt")
            user_prompt = config.get("user_prompt", "")  # Default to empty string if not provided
            temperature = config.get("temperature")
            max_tokens = config.get("max_tokens")
            top_p = config.get("top_p")
            
            # Additional kwargs from any remaining arguments
            kwargs = {k: v for k, v in config.items() 
                    if k not in ["provider", "model", "system_prompt", "user_prompt", 
                               "temperature", "max_tokens", "top_p"]}

            # Check required parameters - allow empty string for user_prompt
            if provider_name is None or model is None or system_prompt is None or user_prompt is None:
                return {"isError": True, "error": "Missing required parameters in a comparison configuration."}

            # Validate provider
            if provider_name not in PROVIDERS:
                return {"isError": True, "error": f"Provider '{provider_name}' not supported."}
                
            # Check if API key is available for this provider, but don't block custom models
            # that might not be in the default list
            api_key = get_api_key(provider_name, raise_error=False)
            if not api_key:
                return {"isError": True, "error": f"API key for provider '{provider_name}' is not available. Please set {provider_name.upper()}_API_KEY in your environment or .env file."}

            try:
                provider_class = PROVIDERS[provider_name]
                provider_instance = provider_class()
                
                # Validate if model exists for this provider
                default_models = provider_class.get_default_models()
                
                # Check if model exists in default models, but don't block if it doesn't
                # This allows testing custom or new models not in the default list
                model_exists = any(model_info["name"] == model for model_info in 
                                 [model_data for model_type, model_data in default_models.items()])
                
                if not model_exists:
                    # Just log a warning, but continue anyway - the model might be valid
                    # but not in our default list
                    print(f"Warning: Model '{model}' not found in default models for provider '{provider_name}'. Attempting to use it anyway.")
                
                result = await provider_instance.generate(
                    model=model,
                    system_prompt=system_prompt,
                    user_prompt=user_prompt,
                    temperature=temperature,
                    max_tokens=max_tokens,
                    top_p=top_p,
                    **kwargs
                )
                
                return {
                    "isError": False,
                    "response": result["text"],
                    "model": result["model"],
                    "provider": provider_name,
                    "usage": result.get("usage", {}),
                    "costs": result.get("costs", {}),
                    "response_time": result.get("response_time", 0),
                    "metadata": {
                        k: v for k, v in result.items()
                        if k not in ["text", "model", "usage", "costs", "response_time"]
                    }
                }
            except ProviderError as e:
                # This will catch errors if the model doesn't exist or other provider-specific errors
                return {"isError": True, "error": f"Provider error: {str(e)}"}
            except Exception as e:
                return {"isError": True, "error": f"Unexpected error: {str(e)}"}

        # Use asyncio.gather to run all comparisons concurrently
        results = await asyncio.gather(*(run_comparison(config) for config in comparisons))

        # 3. Aggregate and Return Results
        return types.TextContent(
            type="text",
            text=json.dumps({
                "isError": False,
                "results": results  # A list of results, one for each comparison
            })
        )

    except Exception as e:
        return types.TextContent(
            type="text",
            text=json.dumps({"isError": True, "error": f"Unexpected error: {str(e)}"})
        ) 