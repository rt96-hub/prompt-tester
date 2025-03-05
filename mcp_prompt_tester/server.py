"""Main MCP server implementation for prompt testing."""

import anyio
import click
import json
import mcp.types as types
from mcp.server.lowlevel import Server #, NotificationOptions
# from typing import Dict, List, Any, Optional

from langfuse.decorators import langfuse_context

# from .providers import PROVIDERS
from .env import load_env_files, get_langfuse_env_vars, is_langfuse_enabled
from .tools import list_providers, test_comparison, test_multiturn_conversation, get_tool_definitions


def create_server() -> Server:
    """Create the MCP server instance."""
    # Load environment variables from .env files
    load_env_files()

    # Initialize Langfuse
    if is_langfuse_enabled():
        langfuse_vars = get_langfuse_env_vars()
        langfuse_context.configure(
            secret_key=langfuse_vars["LANGFUSE_SECRET_KEY"],
            public_key=langfuse_vars["LANGFUSE_PUBLIC_KEY"],
            host=langfuse_vars["LANGFUSE_HOST"],
            enabled=True
        )
    else:
        langfuse_context.configure(
            enabled=False
        )
    
    # Create server
    app = Server("prompt-tester")

    @app.call_tool()
    async def call_tool(
        name: str, arguments: dict
    ) -> list[types.TextContent | types.ImageContent | types.EmbeddedResource]:
        try:
            if name == "list_providers":
                response = await list_providers()
                return [response]
            elif name == "test_comparison":
                response = await test_comparison(arguments)
                return [response]
            elif name == "test_multiturn_conversation":
                response = await test_multiturn_conversation(arguments)
                return [response]
            else:
                return [types.TextContent(
                    type="text",
                    text=json.dumps({"isError": True, "error": f"Unknown tool: {name}"})
                )]
                
        except Exception as e:
            return [types.TextContent(
                type="text",
                text=json.dumps({"isError": True, "error": f"Unexpected error: {str(e)}"})
            )]

    @app.list_tools()
    async def list_tools() -> list[types.Tool]:
        # Convert the tool definitions to MCP Tool objects
        tool_definitions = get_tool_definitions()
        return [
            types.Tool(
                name=tool["name"],
                description=tool["description"],
                inputSchema=tool["parameters"]
            )
            for tool in tool_definitions
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