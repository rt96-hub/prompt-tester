#!/usr/bin/env python3
"""Example client script for the MCP Prompt Tester."""

import asyncio
import json
import os
from pathlib import Path
from dotenv import load_dotenv
from mcp.client.session import ClientSession
from mcp.client.stdio import StdioServerParameters, stdio_client


# Load environment variables from .env file
def load_environment():
    """Load environment variables from .env files."""
    # Try local directory first
    env_path = Path('.env')
    if env_path.exists():
        load_dotenv(env_path)
    else:
        # Try home directory
        home_env = Path.home() / '.env'
        if home_env.exists():
            load_dotenv(home_env)


# Helper function to parse JSON content from response
def parse_response(result):
    """Parse JSON response from tool result."""
    if not result or not result.content:
        return None
    
    try:
        # The response is a JSON string inside the text field
        json_str = result.content[0].text
        return json.loads(json_str)
    except (json.JSONDecodeError, IndexError, AttributeError) as e:
        print(f"Error parsing response: {str(e)}")
        print(f"Raw response: {result}")
        return None


async def main():
    """Run the example client."""
    # Load environment variables
    load_environment()
    
    # Check if required API keys are available
    required_keys = ["OPENAI_API_KEY", "ANTHROPIC_API_KEY"]
    missing_keys = [key for key in required_keys if not os.environ.get(key)]
    
    if missing_keys:
        print(f"Warning: Missing API keys: {', '.join(missing_keys)}")
        print("Some examples may not work without these keys.")
    
    # Connect to the server (assumes the package is installed)
    # Modify the command if running directly from the source
    async with stdio_client(
        StdioServerParameters(command="prompt-tester")
    ) as (read, write):
        async with ClientSession(read, write) as session:
            # Initialize the connection
            await session.initialize()
            
            # List available tools
            tools = await session.list_tools()
            print("\nAvailable tools:")
            # Handle tools format which may vary between MCP versions
            for tool in tools:
                # Check if the tool is a tuple (as returned by low-level API)
                if isinstance(tool, tuple):
                    # Assuming the format is (name, description, schema)
                    name = tool[0]
                    description = tool[1] if len(tool) > 1 else "No description"
                    print(f"- {name}: {description}")
                else:
                    # Handle as object with attributes (higher-level API)
                    print(f"- {tool.name}: {tool.description}")
            
            # List available providers
            print("\nListing available providers...")
            try:
                provider_result = await session.call_tool("list_providers", {})
                parsed = parse_response(provider_result)
                
                if parsed:
                    print("Available providers and models:")
                    print(json.dumps(parsed, indent=2))
                else:
                    print("No provider information returned or error parsing response")
            except Exception as e:
                print(f"Error listing providers: {str(e)}")
            
            # Test an OpenAI prompt
            print("\nTesting OpenAI prompt...")
            try:
                openai_result = await session.call_tool("test_comparison", {
                    "comparisons": [{
                        "provider": "openai",
                        "model": "gpt-4o-mini",
                        "system_prompt": "You are a helpful assistant that speaks like a pirate.",
                        "user_prompt": "Tell me about the weather today.",
                        "temperature": 0.9,
                        "max_tokens": 150
                    }]
                })
                
                parsed = parse_response(openai_result)
                if parsed:
                    if parsed.get("isError", False):
                        print(f"Error: {parsed.get('error', 'Unknown error')}")
                    else:
                        # Extract the first result since we're only using one comparison
                        result = parsed.get("results", [])[0]
                        
                        print("\nOpenAI Response:")
                        print(f"Model: {result.get('model', 'unknown')}")
                        print(f"Provider: {result.get('provider', 'unknown')}")
                        print(f"Response Time: {result.get('response_time', 0):.3f} seconds")
                        print(f"\nResponse text:\n{result.get('response', '')}")
                        
                        # Print usage info if available
                        if result.get("usage"):
                            print("\nUsage statistics:")
                            usage = result["usage"]
                            print(f"  Prompt Tokens: {usage.get('prompt_tokens', 0)}")
                            print(f"  Completion Tokens: {usage.get('completion_tokens', 0)}")
                            print(f"  Total Tokens: {usage.get('total_tokens', 0)}")

                        # Print costs if available
                        if result.get("costs"):
                            print("\nCosts:")
                            costs = result["costs"]
                            print(f"  Input Cost: {costs.get('input_cost', 0):.6f}")
                            print(f"  Output Cost: {costs.get('output_cost', 0):.6f}")
                            print(f"  Total Cost: {costs.get('total_cost', 0):.6f}")

                else:
                    print("No response from OpenAI or error parsing response")
            except Exception as e:
                print(f"Error with OpenAI request: {str(e)}")
            
            # Test an Anthropic prompt
            print("\nTesting Anthropic prompt...")
            try:
                anthropic_result = await session.call_tool("test_comparison", {
                    "comparisons": [{
                        "provider": "anthropic",
                        "model": "claude-3-7-sonnet-20250219",
                        "system_prompt": "You are a helpful assistant that explains complex topics simply.",
                        "user_prompt": "Explain quantum computing to a 10-year old.",
                        "temperature": 0.7,
                        "max_tokens": 200
                    }]
                })
                
                parsed = parse_response(anthropic_result)
                if parsed:
                    if parsed.get("isError", False):
                        print(f"Error: {parsed.get('error', 'Unknown error')}")
                    else:
                        # Extract the first result since we're only using one comparison
                        result = parsed.get("results", [])[0]
                        
                        print("\nAnthropic Response:")
                        print(f"Model: {result.get('model', 'unknown')}")
                        print(f"Provider: {result.get('provider', 'unknown')}")
                        print(f"Response Time: {result.get('response_time', 0):.3f} seconds")
                        print(f"\nResponse text:\n{result.get('response', '')}")
                        
                        # Print usage info if available
                        if result.get("usage"):
                            print("\nUsage statistics:")
                            usage = result["usage"]
                            print(f"  Input Tokens: {usage.get('input_tokens', 0)}")
                            print(f"  Output Tokens: {usage.get('output_tokens', 0)}")

                        # Print costs if available
                        if result.get("costs"):
                            print("\nCosts:")
                            costs = result["costs"]
                            print(f"  Input Cost: {costs.get('input_cost', 0):.6f}")
                            print(f"  Output Cost: {costs.get('output_cost', 0):.6f}")
                            print(f"  Total Cost: {costs.get('total_cost', 0):.6f}")

                else:
                    print("No response from Anthropic or error parsing response")
            except Exception as e:
                print(f"Error with Anthropic request: {str(e)}")


if __name__ == "__main__":
    asyncio.run(main()) 