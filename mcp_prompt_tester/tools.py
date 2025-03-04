"""Tool implementations for MCP Prompt Tester."""

import json
from typing import Dict, Any

from mcp import types

from .providers import PROVIDERS, ProviderError


async def test_prompt(arguments: dict) -> types.TextContent:
    """Test a prompt with a specific provider and model."""
    try:
        # Check required arguments
        required_args = ["provider", "model", "system_prompt", "user_prompt"]
        for arg in required_args:
            if arg not in arguments:
                return types.TextContent(
                    type="text",
                    text=json.dumps({"isError": True, "error": f"Missing required argument '{arg}'"})
                )
        
        # Extract arguments
        provider = arguments["provider"]
        model = arguments["model"]
        system_prompt = arguments["system_prompt"]
        user_prompt = arguments["user_prompt"]
        temperature = arguments.get("temperature")
        max_tokens = arguments.get("max_tokens")
        top_p = arguments.get("top_p")
        
        # Additional kwargs from any remaining arguments
        kwargs = {k: v for k, v in arguments.items() 
                 if k not in ["provider", "model", "system_prompt", "user_prompt", 
                            "temperature", "max_tokens", "top_p"]}
        
        # Validate provider
        if provider not in PROVIDERS:
            return types.TextContent(
                type="text",
                text=json.dumps({
                    "isError": True,
                    "error": f"Provider '{provider}' not supported. Available providers: {', '.join(PROVIDERS.keys())}"
                })
            )
        
        # Create provider instance
        provider_class = PROVIDERS[provider]
        provider_instance = provider_class()
        
        # Generate response
        result = await provider_instance.generate(
            model=model,
            system_prompt=system_prompt,
            user_prompt=user_prompt,
            temperature=temperature,
            max_tokens=max_tokens,
            top_p=top_p,
            **kwargs
        )
        
        return types.TextContent(
            type="text",
            text=json.dumps({
                "isError": False,
                "response": result["text"],
                "model": result["model"],
                "provider": provider,
                "usage": result.get("usage", {}),
                "response_time": result.get("response_time", 0),
                "metadata": {
                    k: v for k, v in result.items() 
                    if k not in ["text", "model", "usage", "response_time"]
                }
            })
        )
    
    except ProviderError as e:
        return types.TextContent(
            type="text",
            text=json.dumps({"isError": True, "error": f"Provider error: {str(e)}"})
        )
    except Exception as e:
        return types.TextContent(
            type="text",
            text=json.dumps({"isError": True, "error": f"Unexpected error: {str(e)}"})
        )


async def list_providers() -> types.TextContent:
    """List available providers and their default models."""
    result = {}
    
    for provider_name, provider_class in PROVIDERS.items():
        # Get default models
        default_models = provider_class.get_default_models()
        
        # Format for output
        models_list = [
            {"type": model_type, "name": model_name}
            for model_type, model_name in default_models.items()
        ]
        
        result[provider_name] = models_list
    
    return types.TextContent(
        type="text",
        text=json.dumps({"providers": result})
    )


def get_tool_definitions() -> list[Dict[str, Any]]:
    """Return the tool definitions for the MCP server."""
    return [
        {
            "name": "test_prompt",
            "description": "Test a prompt with the specified provider and model.",
            "parameters": {
                "type": "object",
                "properties": {
                    "provider": {
                        "type": "string",
                        "description": "The LLM provider to use (e.g., 'openai', 'anthropic')"
                    },
                    "model": {
                        "type": "string",
                        "description": "The model name to use"
                    },
                    "system_prompt": {
                        "type": "string",
                        "description": "The system prompt to use"
                    },
                    "user_prompt": {
                        "type": "string",
                        "description": "The user prompt to test"
                    },
                    "temperature": {
                        "type": "number",
                        "description": "Controls randomness (0.0 to 1.0)"
                    },
                    "max_tokens": {
                        "type": "integer",
                        "description": "Maximum number of tokens to generate"
                    },
                    "top_p": {
                        "type": "number",
                        "description": "Controls diversity via nucleus sampling"
                    }
                },
                "required": ["provider", "model", "system_prompt", "user_prompt"]
            }
        },
        {
            "name": "list_providers",
            "description": "List available providers and their default models.",
            "parameters": {
                "type": "object",
                "properties": {}
            }
        }
    ]

