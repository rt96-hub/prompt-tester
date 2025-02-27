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

The server exposes a tool named `test_prompt` that accepts the following arguments:

- `provider` (string): The LLM provider to use ("openai" or "anthropic")
- `model` (string): The model name (e.g., "gpt-4" for OpenAI, "claude-3-opus-20240229" for Anthropic)
- `system_prompt` (string): The system prompt (instructions for the model)
- `user_prompt` (string): The user's message/prompt
- `temperature` (number, optional): Controls randomness (0.0 to 1.0)
- `max_tokens` (integer, optional): Maximum number of tokens to generate
- `top_p` (number, optional): Controls diversity via nucleus sampling

## Example

Using the MCP client, you can use the tool like this:

```python
import asyncio
from mcp.client.session import ClientSession
from mcp.client.stdio import StdioServerParameters, stdio_client

async def main():
    async with stdio_client(
        StdioServerParameters(command="prompt-tester")
    ) as (read, write):
        async with ClientSession(read, write) as session:
            await session.initialize()

            # Call the prompt testing tool
            result = await session.call_tool("test_prompt", {
                "provider": "openai",
                "model": "gpt-4",
                "system_prompt": "You are a helpful assistant.",
                "user_prompt": "Explain quantum computing in simple terms.",
                "temperature": 0.7,
                "max_tokens": 500
            })
            print(result)

asyncio.run(main())
```

## Configuration

You can set API keys using environment variables:
- `OPENAI_API_KEY` - Your OpenAI API key
- `ANTHROPIC_API_KEY` - Your Anthropic API key
