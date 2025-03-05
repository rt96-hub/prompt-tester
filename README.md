# MCP Prompt Tester

A simple MCP server that allows agents to test LLM prompts with different providers.

## Features

- Test prompts with OpenAI and Anthropic models
- Configure system prompts, user prompts, and other parameters
- Get formatted responses or error messages
- Easy environment setup with .env file support

## Installation

```bash
# Install with pip
pip install -e .

# Or with uv
uv install -e .
```

## API Key Setup

The server requires API keys for the providers you want to use. You can set these up in two ways:

### Option 1: Environment Variables

Set the following environment variables:
- `OPENAI_API_KEY` - Your OpenAI API key
- `ANTHROPIC_API_KEY` - Your Anthropic API key

### Option 2: .env File (Recommended)

1. Create a file named `.env` in your project directory or home directory
2. Add your API keys in the following format:

```
OPENAI_API_KEY=your-openai-api-key-here
ANTHROPIC_API_KEY=your-anthropic-api-key-here
```

3. The server will automatically detect and load these keys

For convenience, a sample template is included as `.env.example`.

## Usage

Start the server using stdio (default) or SSE transport:

```bash
# Using stdio transport (default)
prompt-tester

# Using SSE transport on custom port
prompt-tester --transport sse --port 8000
```

### Available Tools

The server exposes the following tools for MCP-empowered agents:

#### 1. list_providers
Retrieves available LLM providers and their default models.

**Parameters:**
- None required

**Example Response:**
```json
{
  "providers": {
    "openai": [
      {
        "type": "gpt-4",
        "name": "gpt-4",
        "input_cost": 0.03,
        "output_cost": 0.06,
        "description": "Most capable GPT-4 model"
      },
      // ... other models ...
    ],
    "anthropic": [
      // ... models ...
    ]
  }
}
```

#### 2. test_comparison
Compares multiple prompts side-by-side, allowing you to test different providers, models, and parameters simultaneously.

**Parameters:**
- `comparisons` (array): A list of 1-4 comparison configurations, each containing:
  - `provider` (string): The LLM provider to use ("openai" or "anthropic")
  - `model` (string): The model name
  - `system_prompt` (string): The system prompt (instructions for the model)
  - `user_prompt` (string): The user's message/prompt
  - `temperature` (number, optional): Controls randomness
  - `max_tokens` (integer, optional): Maximum number of tokens to generate
  - `top_p` (number, optional): Controls diversity via nucleus sampling

**Example Usage:**
```json
{
  "comparisons": [
    {
      "provider": "openai",
      "model": "gpt-4",
      "system_prompt": "You are a helpful assistant.",
      "user_prompt": "Explain quantum computing in simple terms.",
      "temperature": 0.7
    },
    {
      "provider": "anthropic",
      "model": "claude-3-opus-20240229",
      "system_prompt": "You are a helpful assistant.",
      "user_prompt": "Explain quantum computing in simple terms.",
      "temperature": 0.7
    }
  ]
}
```

#### 3. test_multiturn_conversation
Manages multi-turn conversations with LLM providers, allowing you to create and maintain stateful conversations.

**Modes:**
- `start`: Begins a new conversation
- `continue`: Continues an existing conversation
- `get`: Retrieves conversation history
- `list`: Lists all active conversations
- `close`: Closes a conversation

**Parameters:**
- `mode` (string): Operation mode ("start", "continue", "get", "list", or "close")
- `conversation_id` (string): Unique ID for the conversation (required for continue, get, close modes)
- `provider` (string): The LLM provider (required for start mode)
- `model` (string): The model name (required for start mode)
- `system_prompt` (string): The system prompt (required for start mode)
- `user_prompt` (string): The user message (used in start and continue modes)
- `temperature` (number, optional): Temperature parameter for the model
- `max_tokens` (integer, optional): Maximum tokens to generate
- `top_p` (number, optional): Top-p sampling parameter

**Example Usage (Starting a Conversation):**
```json
{
  "mode": "start",
  "provider": "openai",
  "model": "gpt-4",
  "system_prompt": "You are a helpful assistant specializing in physics.",
  "user_prompt": "Can you explain what dark matter is?"
}
```

**Example Usage (Continuing a Conversation):**
```json
{
  "mode": "continue",
  "conversation_id": "conv_12345",
  "user_prompt": "How does that relate to dark energy?"
}
```

## Example Usage for Agents

Using the MCP client, an agent can use the tools like this:

```python
import asyncio
import json
from mcp.client.session import ClientSession
from mcp.client.stdio import StdioServerParameters, stdio_client

async def main():
    async with stdio_client(
        StdioServerParameters(command="prompt-tester")
    ) as (read, write):
        async with ClientSession(read, write) as session:
            await session.initialize()
            
            # 1. List available providers and models
            providers_result = await session.call_tool("list_providers", {})
            print("Available providers and models:", providers_result)
            
            # 2. Run a basic test with a single model and prompt
            comparison_result = await session.call_tool("test_comparison", {
                "comparisons": [
                    {
                        "provider": "openai",
                        "model": "gpt-4",
                        "system_prompt": "You are a helpful assistant.",
                        "user_prompt": "Explain quantum computing in simple terms.",
                        "temperature": 0.7,
                        "max_tokens": 500
                    }
                ]
            })
            print("Single model test result:", comparison_result)
            
            # 3. Compare multiple prompts/models side by side
            comparison_result = await session.call_tool("test_comparison", {
                "comparisons": [
                    {
                        "provider": "openai",
                        "model": "gpt-4",
                        "system_prompt": "You are a helpful assistant.",
                        "user_prompt": "Explain quantum computing in simple terms.",
                        "temperature": 0.7
                    },
                    {
                        "provider": "anthropic",
                        "model": "claude-3-opus-20240229",
                        "system_prompt": "You are a helpful assistant.",
                        "user_prompt": "Explain quantum computing in simple terms.",
                        "temperature": 0.7
                    }
                ]
            })
            print("Comparison result:", comparison_result)
            
            # 4. Start a multi-turn conversation
            conversation_start = await session.call_tool("test_multiturn_conversation", {
                "mode": "start",
                "provider": "openai",
                "model": "gpt-4",
                "system_prompt": "You are a helpful assistant specializing in physics.",
                "user_prompt": "Can you explain what dark matter is?"
            })
            print("Conversation started:", conversation_start)
            
            # Get the conversation ID from the response
            response_data = json.loads(conversation_start.text)
            conversation_id = response_data.get("conversation_id")
            
            # Continue the conversation
            if conversation_id:
                conversation_continue = await session.call_tool("test_multiturn_conversation", {
                    "mode": "continue",
                    "conversation_id": conversation_id,
                    "user_prompt": "How does that relate to dark energy?"
                })
                print("Conversation continued:", conversation_continue)
                
                # Get the conversation history
                conversation_history = await session.call_tool("test_multiturn_conversation", {
                    "mode": "get",
                    "conversation_id": conversation_id
                })
                print("Conversation history:", conversation_history)

asyncio.run(main())
```

## MCP Agent Integration

For MCP-empowered agents, integration is straightforward. When your agent needs to test LLM prompts:

1. **Discovery**: The agent can use `list_providers` to discover available models and their capabilities
2. **Simple Testing**: For quick tests, use the `test_comparison` tool with a single configuration
3. **Comparison**: When the agent needs to evaluate different prompts or models, it can use `test_comparison` with multiple configurations
4. **Stateful Interactions**: For multi-turn conversations, the agent can manage a conversation using the `test_multiturn_conversation` tool

This allows agents to:
- Test prompt variants to find the most effective phrasing
- Compare different models for specific tasks
- Maintain context in multi-turn conversations
- Optimize parameters like temperature and max_tokens
- Track token usage and costs during development

## Configuration

You can set API keys and optional tracing configurations using environment variables:

### Required API Keys
- `OPENAI_API_KEY` - Your OpenAI API key
- `ANTHROPIC_API_KEY` - Your Anthropic API key

### Optional Langfuse Tracing
The server supports Langfuse for tracing and observability of LLM calls. These settings are optional:
- `LANGFUSE_SECRET_KEY` - Your Langfuse secret key
- `LANGFUSE_PUBLIC_KEY` - Your Langfuse public key
- `LANGFUSE_HOST` - URL of your Langfuse instance

If you don't want to use Langfuse tracing, simply leave these settings empty.
