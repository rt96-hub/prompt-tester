"""Tool for listing available providers and their default models."""

import json
from mcp import types

from ..providers import PROVIDERS
from ..env import get_api_key


async def list_providers() -> types.TextContent:
    """List available providers and their default models."""
    result = {}
    
    for provider_name, provider_class in PROVIDERS.items():
        # Skip providers without API keys
        if not get_api_key(provider_name, raise_error=False):
            continue
            
        # Get default models
        default_models = provider_class.get_default_models()
        
        # Format for output
        models_list = [
            {
                "type": model_type,
                "name": model_info["name"],
                "input_cost": model_info["input_cost"],
                "output_cost": model_info["output_cost"],
                "description": model_info.get("description", "")
            }
            for model_type, model_info in default_models.items()
        ]
        
        result[provider_name] = models_list
    
    return types.TextContent(
        type="text",
        text=json.dumps({"providers": result})
    ) 