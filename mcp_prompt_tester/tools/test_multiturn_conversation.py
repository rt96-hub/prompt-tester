"""Tool for managing multi-turn conversations with various LLM providers."""

import json
import uuid
from typing import Dict, Any
from mcp import types
from langfuse.decorators import observe

from ..providers import PROVIDERS, ProviderError
from ..env import get_api_key

# In-memory storage for conversations. Key is UUID, value is conversation data.
conversations: Dict[str, Dict[str, Any]] = {}

@observe()
async def test_multiturn_conversation(arguments: dict) -> types.TextContent:
    """
    Manages multi-turn conversations with LLM providers.

    Modes:
        start: Starts a new conversation.
        continue: Continues an existing conversation.
        get: Retrieves conversation history.
        list: Lists all active conversations.
        close: Closes a conversation.
    """
    try:
        mode = arguments.get("mode")

        if not mode:
            return types.TextContent(
                type="text",
                text=json.dumps({"isError": True, "error": "Missing required parameter 'mode'."})
            )

        if mode == "start":
            # Check required parameters for start mode
            if not arguments.get("provider"):
                return types.TextContent(type="text", text=json.dumps({"isError": True, "error": "Missing required parameter 'provider' for 'start' mode."}))
            if not arguments.get("model"):
                return types.TextContent(type="text", text=json.dumps({"isError": True, "error": "Missing required parameter 'model' for 'start' mode."}))
            if not arguments.get("system_prompt"):
                return types.TextContent(type="text", text=json.dumps({"isError": True, "error": "Missing required parameter 'system_prompt' for 'start' mode."}))
            return await _start_conversation(arguments)
        elif mode == "continue":
            # Check required parameters for continue mode
            if not arguments.get("conversation_id"):
                return types.TextContent(type="text", text=json.dumps({"isError": True, "error": "Missing required parameter 'conversation_id' for 'continue' mode."}))
            return await _continue_conversation(arguments)
        elif mode == "get":
            # Check required parameters for get mode
            if not arguments.get("conversation_id"):
                return types.TextContent(type="text", text=json.dumps({"isError": True, "error": "Missing required parameter 'conversation_id' for 'get' mode."}))
            return await _get_conversation(arguments)
        elif mode == "list":
            return await _list_conversations(arguments)
        elif mode == "close":
            # Check required parameters for close mode
            if not arguments.get("conversation_id"):
                return types.TextContent(type="text", text=json.dumps({"isError": True, "error": "Missing required parameter 'conversation_id' for 'close' mode."}))
            return await _close_conversation(arguments)
        else:
            return types.TextContent(
                type="text",
                text=json.dumps({"isError": True, "error": "Invalid mode. Must be 'start', 'continue', 'get', 'list', or 'close'."})
            )

    except Exception as e:
        return types.TextContent(
            type="text",
            text=json.dumps({"isError": True, "error": f"Unexpected error: {str(e)}"})
        )

async def _start_conversation(arguments: dict) -> types.TextContent:
    """Starts a new conversation."""
    provider_name = arguments.get("provider")
    model = arguments.get("model")
    system_prompt = arguments.get("system_prompt")
    user_prompt = arguments.get("user_prompt", "")
    temperature = arguments.get("temperature")
    max_tokens = arguments.get("max_tokens")
    top_p = arguments.get("top_p")
    kwargs = {k: v for k, v in arguments.items()
              if k not in ["mode", "provider", "model", "system_prompt", "user_prompt",
                           "temperature", "max_tokens", "top_p", "conversation_id"]}

    if any(arg is None for arg in [provider_name, model, system_prompt]):
        return types.TextContent(type="text", text=json.dumps({"isError": True, "error": "Missing required parameters (provider, model, system_prompt)."}))

    if provider_name not in PROVIDERS:
        return types.TextContent(type="text", text=json.dumps({"isError": True, "error": f"Provider '{provider_name}' not supported."}))

    api_key = get_api_key(provider_name, raise_error=False)
    if not api_key:
        return types.TextContent(type="text", text=json.dumps({"isError": True, "error": f"API key for provider '{provider_name}' is not available."}))

    conversation_id = str(uuid.uuid4())
    conversations[conversation_id] = {
        "provider": provider_name,
        "model": model,
        "system_prompt": system_prompt,
        "conversation_history": [],  # Start with empty history
        "hyperparameters": {
            "temperature": temperature,
            "max_tokens": max_tokens,
            "top_p": top_p,
            **kwargs
        },
        "usage": {},
        "costs": {},
        "response_time": 0
    }

    try:
        provider_class = PROVIDERS[provider_name]
        provider_instance = provider_class()
        
        # Add the first user message to the history
        conversations[conversation_id]["conversation_history"].append({"role": "user", "content": user_prompt})
        
        # For the first message, we can use the regular generate method
        result = await provider_instance.generate(
            model=model,
            system_prompt=system_prompt,
            user_prompt=user_prompt,
            temperature=temperature,
            max_tokens=max_tokens,
            top_p=top_p,
            **kwargs
        )

        # Add the assistant's response to the history
        conversations[conversation_id]["conversation_history"].append({"role": "assistant", "content": result["text"]})
        conversations[conversation_id]["usage"] = result.get("usage", {})
        conversations[conversation_id]["costs"] = result.get("costs", {})
        conversations[conversation_id]["response_time"] = result.get("response_time", 0)

        return types.TextContent(
            type="text",
            text=json.dumps({
                "isError": False,
                "conversation_id": conversation_id,
                "response": result["text"],
                "model": result["model"],
                "usage": result.get("usage", {}),
                "costs": result.get("costs", {}),
                "response_time": result.get("response_time", 0)
            })
        )

    except ProviderError as e:
        del conversations[conversation_id]  # Clean up on error
        return types.TextContent(type="text", text=json.dumps({"isError": True, "error": f"Provider error: {str(e)}"}))
    except Exception as e:
        del conversations[conversation_id]
        return types.TextContent(type="text", text=json.dumps({"isError": True, "error": f"Unexpected error during generation: {str(e)}"}))

async def _continue_conversation(arguments: dict) -> types.TextContent:
    """Continues an existing conversation."""
    conversation_id = arguments.get("conversation_id")
    user_prompt = arguments.get("user_prompt", "")

    if not conversation_id:
        return types.TextContent(type="text", text=json.dumps({"isError": True, "error": "Missing required parameter 'conversation_id'."}))

    if conversation_id not in conversations:
        return types.TextContent(type="text", text=json.dumps({"isError": True, "error": f"Conversation with ID '{conversation_id}' not found."}))

    # IMPORTANT: We ignore any system_prompt passed in the continuation request
    # We only use the system_prompt that was set when the conversation was started
    if "system_prompt" in arguments:
        # Log a warning about ignoring system_prompt in continuation
        print(f"Warning: system_prompt parameter ignored in conversation continuation.")

    conversation = conversations[conversation_id]
    conversation["conversation_history"].append({"role": "user", "content": user_prompt})

    try:
        provider_class = PROVIDERS[conversation["provider"]]
        provider_instance = provider_class()

        # Extract only the valid hyperparameters to prevent errors
        hyperparameters = {}
        valid_params = ["temperature", "max_tokens", "top_p"]
        for param in valid_params:
            if param in conversation["hyperparameters"] and conversation["hyperparameters"][param] is not None:
                hyperparameters[param] = conversation["hyperparameters"][param]

        # Use the new generate_with_history method with only valid parameters
        result = await provider_instance.generate_with_history(
            model=conversation["model"],
            system_prompt=conversation["system_prompt"],
            message_history=conversation["conversation_history"],
            **hyperparameters
        )

        conversation["conversation_history"].append({"role": "assistant", "content": result["text"]})
        conversation["usage"] = result.get("usage", {})  # Update usage
        conversation["costs"] = result.get("costs", {})
        conversation["response_time"] = result.get("response_time", 0)

        return types.TextContent(
            type="text",
            text=json.dumps({
                "isError": False,
                "conversation_id": conversation_id,
                "response": result["text"],
                "model": result["model"],
                "usage": result.get("usage", {}),
                "costs": result.get("costs", {}),
                "response_time": result.get("response_time", 0)
            })
        )

    except ProviderError as e:
        return types.TextContent(type="text", text=json.dumps({"isError": True, "error": f"Provider error: {str(e)}"}))
    except Exception as e:
        return types.TextContent(type="text", text=json.dumps({"isError": True, "error": f"Unexpected error during continuation: {str(e)}"}))

async def _get_conversation(arguments: dict) -> types.TextContent:
    """Retrieves conversation history."""
    conversation_id = arguments.get("conversation_id")

    if not conversation_id:
        return types.TextContent(type="text", text=json.dumps({"isError": True, "error": "Missing required parameter 'conversation_id'."}))

    if conversation_id not in conversations:
        return types.TextContent(type="text", text=json.dumps({"isError": True, "error": f"Conversation with ID '{conversation_id}' not found."}))

    conversation = conversations[conversation_id]
    return types.TextContent(
        type="text",
        text=json.dumps({
            "isError": False,
            "conversation_id": conversation_id,
            "history": conversation["conversation_history"],
            "usage": conversation["usage"],
            "costs": conversation["costs"],
            "response_time": conversation["response_time"],
            "provider": conversation["provider"],
            "model": conversation["model"],
            "system_prompt": conversation["system_prompt"],
            "hyperparameters": conversation["hyperparameters"]
        })
    )

async def _list_conversations(arguments: dict) -> types.TextContent:
    """Lists all active conversations."""
    
    # Create a summary of each conversation
    conversation_summaries = {}
    for conv_id, conv_data in conversations.items():
        # Get the first user message and last assistant message if they exist
        first_user_message = ""
        last_assistant_message = ""
        
        for msg in conv_data["conversation_history"]:
            if msg["role"] == "user" and not first_user_message:
                first_user_message = msg["content"]
            if msg["role"] == "assistant":
                last_assistant_message = msg["content"]
        
        # Create a summary
        conversation_summaries[conv_id] = {
            "provider": conv_data["provider"],
            "model": conv_data["model"],
            "first_user_message": first_user_message,
            "latest_assistant_message": last_assistant_message,
            "message_count": len(conv_data["conversation_history"]),
            "system_prompt": conv_data["system_prompt"][:100] + "..." if len(conv_data["system_prompt"]) > 100 else conv_data["system_prompt"]
        }
    
    return types.TextContent(
        type="text",
        text=json.dumps({
            "isError": False,
            "conversation_count": len(conversations),
            "conversations": conversation_summaries
        })
    )

async def _close_conversation(arguments: dict) -> types.TextContent:
    """Closes a conversation."""
    conversation_id = arguments.get("conversation_id")

    if not conversation_id:
        return types.TextContent(type="text", text=json.dumps({"isError": True, "error": "Missing required parameter 'conversation_id'."}))

    if conversation_id not in conversations:
        return types.TextContent(type="text", text=json.dumps({"isError": True, "error": f"Conversation with ID '{conversation_id}' not found."}))

    del conversations[conversation_id]
    return types.TextContent(type="text", text=json.dumps({"isError": False, "message": f"Conversation '{conversation_id}' closed."})) 