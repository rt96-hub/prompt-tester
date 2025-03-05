"""Tool for managing multi-turn conversations with various LLM providers."""

import json
import uuid
import sqlite3
import os
import threading
from typing import Dict, Any
from mcp import types
from langfuse.decorators import observe

from ..providers import PROVIDERS, ProviderError
from ..env import get_api_key

# Database file location
DB_PATH = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data", "conversations.db")

# Thread-local storage for SQLite connections
local = threading.local()

def get_db_connection():
    """Get a thread-local database connection."""
    if not hasattr(local, "conn"):
        # Ensure the directory exists
        os.makedirs(os.path.dirname(DB_PATH), exist_ok=True)
        local.conn = sqlite3.connect(DB_PATH)
        # Enable foreign keys and return dict-like rows
        local.conn.execute("PRAGMA foreign_keys = ON")
        local.conn.row_factory = sqlite3.Row
    return local.conn

def close_db_connection():
    """Close the thread-local database connection."""
    if hasattr(local, "conn"):
        local.conn.close()
        del local.conn

def init_db():
    """Initialize the database with necessary tables."""
    conn = get_db_connection()
    cursor = conn.cursor()
    
    # Create conversations table
    cursor.execute('''
    CREATE TABLE IF NOT EXISTS conversations (
        id TEXT PRIMARY KEY,
        provider TEXT NOT NULL,
        model TEXT NOT NULL,
        system_prompt TEXT NOT NULL,
        hyperparameters TEXT NOT NULL,
        usage TEXT,
        costs TEXT,
        response_time REAL DEFAULT 0
    )
    ''')
    
    # Create conversation_history table
    cursor.execute('''
    CREATE TABLE IF NOT EXISTS conversation_history (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        conversation_id TEXT NOT NULL,
        role TEXT NOT NULL,
        content TEXT NOT NULL,
        timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
        FOREIGN KEY (conversation_id) REFERENCES conversations(id) ON DELETE CASCADE
    )
    ''')
    
    conn.commit()

# Initialize the database on module load
init_db()

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
    finally:
        # Close the database connection at the end of the request
        close_db_connection()

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
    conn = get_db_connection()
    cursor = conn.cursor()
    
    # Store hyperparameters as JSON
    hyperparameters = {
        "temperature": temperature,
        "max_tokens": max_tokens,
        "top_p": top_p,
        **kwargs
    }
    
    try:
        provider_class = PROVIDERS[provider_name]
        provider_instance = provider_class()
        
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
        
        usage_json = json.dumps(result.get("usage", {}))
        costs_json = json.dumps(result.get("costs", {}))
        response_time = result.get("response_time", 0)
        
        # Insert data into conversations table
        cursor.execute(
            "INSERT INTO conversations (id, provider, model, system_prompt, hyperparameters, usage, costs, response_time) VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
            (conversation_id, provider_name, model, system_prompt, json.dumps(hyperparameters), usage_json, costs_json, response_time)
        )
        
        # Add the user message to history
        cursor.execute(
            "INSERT INTO conversation_history (conversation_id, role, content) VALUES (?, ?, ?)",
            (conversation_id, "user", user_prompt)
        )
        
        # Add the assistant's response to history
        cursor.execute(
            "INSERT INTO conversation_history (conversation_id, role, content) VALUES (?, ?, ?)",
            (conversation_id, "assistant", result["text"])
        )
        
        conn.commit()

        return types.TextContent(
            type="text",
            text=json.dumps({
                "isError": False,
                "conversation_id": conversation_id,
                "response": result["text"],
                "model": result["model"],
                "usage": result.get("usage", {}),
                "costs": result.get("costs", {}),
                "response_time": response_time
            })
        )

    except ProviderError as e:
        # Rollback on error
        conn.rollback()
        return types.TextContent(type="text", text=json.dumps({"isError": True, "error": f"Provider error: {str(e)}"}))
    except Exception as e:
        # Rollback on error
        conn.rollback()
        return types.TextContent(type="text", text=json.dumps({"isError": True, "error": f"Unexpected error during generation: {str(e)}"}))

async def _continue_conversation(arguments: dict) -> types.TextContent:
    """Continues an existing conversation."""
    conversation_id = arguments.get("conversation_id")
    user_prompt = arguments.get("user_prompt", "")

    if not conversation_id:
        return types.TextContent(type="text", text=json.dumps({"isError": True, "error": "Missing required parameter 'conversation_id'."}))

    conn = get_db_connection()
    cursor = conn.cursor()
    
    # Check if conversation exists
    cursor.execute("SELECT * FROM conversations WHERE id = ?", (conversation_id,))
    conversation_row = cursor.fetchone()
    
    if not conversation_row:
        return types.TextContent(type="text", text=json.dumps({"isError": True, "error": f"Conversation with ID '{conversation_id}' not found."}))

    # IMPORTANT: We ignore any system_prompt passed in the continuation request
    if "system_prompt" in arguments:
        # Log a warning about ignoring system_prompt in conversation continuation.
        print(f"Warning: system_prompt parameter ignored in conversation continuation.")

    # Get conversation data
    provider_name = conversation_row["provider"]
    model = conversation_row["model"]
    system_prompt = conversation_row["system_prompt"]
    hyperparameters = json.loads(conversation_row["hyperparameters"])
    
    # Get conversation history
    cursor.execute("SELECT role, content FROM conversation_history WHERE conversation_id = ? ORDER BY id", (conversation_id,))
    history_rows = cursor.fetchall()
    
    conversation_history = [{"role": row["role"], "content": row["content"]} for row in history_rows]
    
    # Add new user message to history
    cursor.execute(
        "INSERT INTO conversation_history (conversation_id, role, content) VALUES (?, ?, ?)",
        (conversation_id, "user", user_prompt)
    )
    
    # Update conversation history for provider
    conversation_history.append({"role": "user", "content": user_prompt})

    try:
        provider_class = PROVIDERS[provider_name]
        provider_instance = provider_class()

        # Extract only the valid hyperparameters to prevent errors
        valid_hyperparameters = {}
        valid_params = ["temperature", "max_tokens", "top_p"]
        for param in valid_params:
            if param in hyperparameters and hyperparameters[param] is not None:
                valid_hyperparameters[param] = hyperparameters[param]

        # Use the generate_with_history method with only valid parameters
        result = await provider_instance.generate_with_history(
            model=model,
            system_prompt=system_prompt,
            message_history=conversation_history,
            **valid_hyperparameters
        )

        # Add assistant response to history
        cursor.execute(
            "INSERT INTO conversation_history (conversation_id, role, content) VALUES (?, ?, ?)",
            (conversation_id, "assistant", result["text"])
        )
        
        # Update usage, costs, and response time
        usage_json = json.dumps(result.get("usage", {}))
        costs_json = json.dumps(result.get("costs", {}))
        response_time = result.get("response_time", 0)
        
        cursor.execute(
            "UPDATE conversations SET usage = ?, costs = ?, response_time = ? WHERE id = ?",
            (usage_json, costs_json, response_time, conversation_id)
        )
        
        conn.commit()

        return types.TextContent(
            type="text",
            text=json.dumps({
                "isError": False,
                "conversation_id": conversation_id,
                "response": result["text"],
                "model": result["model"],
                "usage": result.get("usage", {}),
                "costs": result.get("costs", {}),
                "response_time": response_time
            })
        )

    except ProviderError as e:
        conn.rollback()
        return types.TextContent(type="text", text=json.dumps({"isError": True, "error": f"Provider error: {str(e)}"}))
    except Exception as e:
        conn.rollback()
        return types.TextContent(type="text", text=json.dumps({"isError": True, "error": f"Unexpected error during continuation: {str(e)}"}))

async def _get_conversation(arguments: dict) -> types.TextContent:
    """Retrieves conversation history."""
    conversation_id = arguments.get("conversation_id")

    if not conversation_id:
        return types.TextContent(type="text", text=json.dumps({"isError": True, "error": "Missing required parameter 'conversation_id'."}))

    conn = get_db_connection()
    cursor = conn.cursor()
    
    # Get conversation data
    cursor.execute("SELECT * FROM conversations WHERE id = ?", (conversation_id,))
    conversation_row = cursor.fetchone()
    
    if not conversation_row:
        return types.TextContent(type="text", text=json.dumps({"isError": True, "error": f"Conversation with ID '{conversation_id}' not found."}))
    
    # Get conversation history
    cursor.execute("SELECT role, content FROM conversation_history WHERE conversation_id = ? ORDER BY id", (conversation_id,))
    history_rows = cursor.fetchall()
    
    conversation_history = [{"role": row["role"], "content": row["content"]} for row in history_rows]
    
    # Parse JSON fields
    usage = json.loads(conversation_row["usage"] or "{}")
    costs = json.loads(conversation_row["costs"] or "{}")
    hyperparameters = json.loads(conversation_row["hyperparameters"])
    
    return types.TextContent(
        type="text",
        text=json.dumps({
            "isError": False,
            "conversation_id": conversation_id,
            "history": conversation_history,
            "usage": usage,
            "costs": costs,
            "response_time": conversation_row["response_time"],
            "provider": conversation_row["provider"],
            "model": conversation_row["model"],
            "system_prompt": conversation_row["system_prompt"],
            "hyperparameters": hyperparameters
        })
    )

async def _list_conversations(arguments: dict) -> types.TextContent:
    """Lists all active conversations."""
    conn = get_db_connection()
    cursor = conn.cursor()
    
    # Get all conversations
    cursor.execute("SELECT id, provider, model, system_prompt FROM conversations")
    conversation_rows = cursor.fetchall()
    
    conversation_summaries = {}
    
    for row in conversation_rows:
        conversation_id = row["id"]
        
        # Get first user message and last assistant message
        first_user_query = """
            SELECT content FROM conversation_history 
            WHERE conversation_id = ? AND role = 'user' 
            ORDER BY id ASC LIMIT 1
        """
        last_assistant_query = """
            SELECT content FROM conversation_history 
            WHERE conversation_id = ? AND role = 'assistant' 
            ORDER BY id DESC LIMIT 1
        """
        
        cursor.execute(first_user_query, (conversation_id,))
        first_user_row = cursor.fetchone()
        first_user_message = first_user_row["content"] if first_user_row else ""
        
        cursor.execute(last_assistant_query, (conversation_id,))
        last_assistant_row = cursor.fetchone()
        last_assistant_message = last_assistant_row["content"] if last_assistant_row else ""
        
        # Count messages
        cursor.execute("SELECT COUNT(*) as count FROM conversation_history WHERE conversation_id = ?", (conversation_id,))
        message_count = cursor.fetchone()["count"]
        
        # Create summary
        system_prompt = row["system_prompt"]
        conversation_summaries[conversation_id] = {
            "provider": row["provider"],
            "model": row["model"],
            "first_user_message": first_user_message,
            "latest_assistant_message": last_assistant_message,
            "message_count": message_count,
            "system_prompt": system_prompt[:100] + "..." if len(system_prompt) > 100 else system_prompt
        }
    
    return types.TextContent(
        type="text",
        text=json.dumps({
            "isError": False,
            "conversation_count": len(conversation_summaries),
            "conversations": conversation_summaries
        })
    )

async def _close_conversation(arguments: dict) -> types.TextContent:
    """Closes a conversation."""
    conversation_id = arguments.get("conversation_id")

    if not conversation_id:
        return types.TextContent(type="text", text=json.dumps({"isError": True, "error": "Missing required parameter 'conversation_id'."}))

    conn = get_db_connection()
    cursor = conn.cursor()
    
    # Check if conversation exists
    cursor.execute("SELECT 1 FROM conversations WHERE id = ?", (conversation_id,))
    if not cursor.fetchone():
        return types.TextContent(type="text", text=json.dumps({"isError": True, "error": f"Conversation with ID '{conversation_id}' not found."}))
    
    # Delete conversation (will cascade to history)
    cursor.execute("DELETE FROM conversations WHERE id = ?", (conversation_id,))
    conn.commit()
    
    return types.TextContent(type="text", text=json.dumps({"isError": False, "message": f"Conversation '{conversation_id}' closed."})) 