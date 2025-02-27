"""Main MCP server implementation for prompt testing."""

import anyio
import click
import mcp.types as types
import json
from mcp.server.lowlevel import Server, NotificationOptions
from typing import Dict, List, Any, Optional

from .providers import PROVIDERS, ProviderError
from .env import load_env_files


def create_server() -> Server:
    """Create the MCP server instance."""
    # Load environment variables from .env files
    load_env_files()
    
    # Create server
    app = Server("prompt-tester")

    @app.call_tool()
    async def call_tool(
        name: str, arguments: dict
    ) -> list[types.TextContent | types.ImageContent | types.EmbeddedResource]:
        try:
            if name == "test_prompt":
                # Check required arguments
                required_args = ["provider", "model", "system_prompt", "user_prompt"]
                for arg in required_args:
                    if arg not in arguments:
                        return [types.TextContent(
                            type="text",
                            text=json.dumps({"isError": True, "error": f"Missing required argument '{arg}'"})
                        )]
                
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
                    return [types.TextContent(
                        type="text",
                        text=json.dumps({
                            "isError": True,
                            "error": f"Provider '{provider}' not supported. Available providers: {', '.join(PROVIDERS.keys())}"
                        })
                    )]
                
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
                
                return [types.TextContent(
                    type="text",
                    text=json.dumps({
                        "isError": False,
                        "response": result["text"],
                        "model": result["model"],
                        "provider": provider,
                        "usage": result.get("usage", {}),
                        "metadata": {
                            k: v for k, v in result.items() 
                            if k not in ["text", "model", "usage"]
                        }
                    })
                )]
                
            elif name == "list_providers":
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
                
                return [types.TextContent(
                    type="text",
                    text=json.dumps({"providers": result})
                )]
            
            else:
                return [types.TextContent(
                    type="text",
                    text=json.dumps({"isError": True, "error": f"Unknown tool: {name}"})
                )]
                
        except ProviderError as e:
            return [types.TextContent(
                type="text",
                text=json.dumps({"isError": True, "error": f"Provider error: {str(e)}"})
            )]
        except Exception as e:
            return [types.TextContent(
                type="text",
                text=json.dumps({"isError": True, "error": f"Unexpected error: {str(e)}"})
            )]

    @app.list_tools()
    async def list_tools() -> list[types.Tool]:
        return [
            types.Tool(
                name="test_prompt",
                description="Test an LLM prompt with a specific provider and model",
                inputSchema={
                    "type": "object",
                    "required": ["provider", "model", "system_prompt", "user_prompt"],
                    "properties": {
                        "provider": {
                            "type": "string",
                            "description": "The LLM provider to use",
                            "enum": list(PROVIDERS.keys())
                        },
                        "model": {
                            "type": "string",
                            "description": "The model name to use"
                        },
                        "system_prompt": {
                            "type": "string",
                            "description": "The system prompt (instructions for the model)"
                        },
                        "user_prompt": {
                            "type": "string",
                            "description": "The user's message/prompt"
                        },
                        "temperature": {
                            "type": "number",
                            "description": "Controls randomness (0.0 to 1.0)",
                            "minimum": 0,
                            "maximum": 1
                        },
                        "max_tokens": {
                            "type": "integer",
                            "description": "Maximum number of tokens to generate",
                            "minimum": 1
                        },
                        "top_p": {
                            "type": "number",
                            "description": "Controls diversity via nucleus sampling",
                            "minimum": 0,
                            "maximum": 1
                        }
                    }
                }
            ),
            types.Tool(
                name="list_providers",
                description="List available LLM providers and their default models",
                inputSchema={
                    "type": "object",
                    "properties": {}
                }
            )
        ]

    return app


@click.command()
@click.option(
    "--port", 
    default=8000, 
    help="Port to listen on for SSE transport"
)
@click.option(
    "--transport",
    type=click.Choice(["stdio", "sse"]),
    default="stdio",
    help="Transport type (stdio or sse)"
)
def main(port: int, transport: str) -> int:
    """Start the prompt-tester MCP server."""
    app = create_server()
    
    if transport == "sse":
        from mcp.server.sse import SseServerTransport
        from starlette.applications import Starlette
        from starlette.routing import Mount, Route

        sse = SseServerTransport("/messages/")

        async def handle_sse(request):
            async with sse.connect_sse(
                request.scope, request.receive, request._send
            ) as streams:
                await app.run(
                    streams[0], streams[1], app.create_initialization_options()
                )

        starlette_app = Starlette(
            debug=True,
            routes=[
                Route("/sse", endpoint=handle_sse),
                Mount("/messages/", app=sse.handle_post_message),
            ],
        )

        import uvicorn
        uvicorn.run(starlette_app, host="0.0.0.0", port=port)
    else:
        from mcp.server.stdio import stdio_server

        async def arun():
            async with stdio_server() as streams:
                await app.run(
                    streams[0], streams[1], app.create_initialization_options()
                )

        anyio.run(arun)
    
    return 0


if __name__ == "__main__":
    main() 