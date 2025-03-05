"""Tools package for MCP Prompt Tester."""

from typing import Dict, Any, List

from .list_providers import list_providers
from .compare_prompts import compare_prompts

__all__ = ["list_providers", "compare_prompts", "get_tool_definitions"]


def get_tool_definitions() -> list[Dict[str, Any]]:
    """Return the tool definitions for the MCP server."""
    return [
        {
            "name": "list_providers",
            "description": "List available providers and their default models.",
            "parameters": {
                "type": "object",
                "properties": {
                    "random_string": {
                        "type": "string",
                        "description": "Dummy parameter for no-parameter tools"
                    }
                },
                "required": ["random_string"]
            }
        },
        {
            "name": "compare_prompts",
            "description": "Compare multiple prompts side-by-side, varying providers, models, and parameters.",
            "parameters": {
                "type": "object",
                "properties": {
                    "comparisons": {
                        "type": "array",
                        "description": "A list of comparison configurations (1 to 4 configurations).",
                        "items": {
                            "type": "object",
                            "properties": {
                                "provider": {"type": "string", "description": "The LLM provider."},
                                "model": {"type": "string", "description": "The model name."},
                                "system_prompt": {"type": "string", "description": "The system prompt."},
                                "user_prompt": {"type": "string", "description": "The user prompt."},
                                "temperature": {"type": "number", "description": "Temperature."},
                                "max_tokens": {"type": "integer", "description": "Max tokens."},
                                "top_p": {"type": "number", "description": "Top P."}
                            },
                            "required": ["provider", "model", "system_prompt", "user_prompt"]
                        },
                        "minItems": 1,
                        "maxItems": 4
                    }
                },
                "required": ["comparisons"]
            }
        }
    ] 