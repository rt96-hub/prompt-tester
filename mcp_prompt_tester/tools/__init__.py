"""Tools package for MCP Prompt Tester."""

from typing import Dict, Any, List

from .list_providers import list_providers
from .test_comparison import test_comparison
from .test_multiturn_conversation import test_multiturn_conversation

__all__ = ["list_providers", "test_comparison", "test_multiturn_conversation", "get_tool_definitions"]


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
            "name": "test_comparison",
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
        },
        {
            "name": "test_multiturn_conversation",
            "description": "Evaluate the quality of a multi-turn conversation.",
            "parameters": {
                "type": "object",
                "properties": {
                    "mode": {
                        "type": "string",
                        "description": "Operation mode: 'start', 'continue', 'get', 'list', or 'close'",
                        "enum": ["start", "continue", "get", "list", "close"]
                    },
                    "conversation_id": {
                        "type": "string",
                        "description": "Unique ID for the conversation (required for continue, get, close modes)"
                    },
                    "provider": {
                        "type": "string",
                        "description": "The LLM provider (required for start mode only)"
                    },
                    "model": {
                        "type": "string",
                        "description": "The model name (required for start mode only)"
                    },
                    "system_prompt": {
                        "type": "string",
                        "description": "The system prompt (required for start mode only, ignored for continue mode)"
                    },
                    "user_prompt": {
                        "type": "string",
                        "description": "The user message (used in start and continue modes)"
                    },
                    "temperature": {
                        "type": "number",
                        "description": "Temperature parameter for the model (start mode only)"
                    },
                    "max_tokens": {
                        "type": "integer",
                        "description": "Maximum number of tokens to generate (start mode only)"
                    },
                    "top_p": {
                        "type": "number",
                        "description": "Top-p sampling parameter (start mode only)"
                    }
                },
                "required": ["mode"],
            }
        }
    ] 