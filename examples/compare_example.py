#!/usr/bin/env python3
"""Example client script for the MCP Prompt Tester's side-by-side comparison feature."""

import asyncio
import json
import os
from pathlib import Path
from dotenv import load_dotenv
from mcp.client.session import ClientSession
from mcp.client.stdio import StdioServerParameters, stdio_client
from tabulate import tabulate  # For nice tabular output
from rich.console import Console
from rich.table import Table


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


def display_comparison_results(comparison_results):
    """Display the comparison results in a nicely formatted way."""
    console = Console()
    
    # First, create a table for basic info
    basic_table = Table(title="Model Comparison Overview")
    basic_table.add_column("Provider", style="cyan")
    basic_table.add_column("Model", style="magenta")
    basic_table.add_column("Response Time (s)", style="green")
    basic_table.add_column("Total Cost ($)", style="yellow")
    
    for i, result in enumerate(comparison_results):
        if result.get("isError", False):
            basic_table.add_row(
                f"Error #{i+1}",
                "N/A",
                "N/A",
                "N/A"
            )
            continue
            
        provider = result.get("provider", "unknown")
        model = result.get("model", "unknown")
        response_time = f"{result.get('response_time', 0):.3f}"
        
        costs = result.get("costs", {})
        total_cost = costs.get("total_cost", 0)
        total_cost_str = f"{total_cost:.6f}"
        
        basic_table.add_row(provider, model, response_time, total_cost_str)
    
    console.print(basic_table)
    
    # Print each model's response
    for i, result in enumerate(comparison_results):
        if result.get("isError", False):
            console.print(f"\n[bold red]Error in comparison #{i+1}:[/bold red] {result.get('error', 'Unknown error')}")
            continue
            
        provider = result.get("provider", "unknown")
        model = result.get("model", "unknown")
        
        console.print(f"\n[bold cyan]#{i+1}: {provider} - {model}[/bold cyan]")
        console.print(f"[bold white]Response:[/bold white]")
        console.print(result.get("response", "No response"))
        
        # Usage statistics
        usage = result.get("usage", {})
        if usage:
            console.print("\n[bold blue]Usage:[/bold blue]")
            
            # Handle different token formats from different providers
            if "prompt_tokens" in usage:  # OpenAI format
                console.print(f"  Input Tokens: {usage.get('prompt_tokens', 0)}")
                console.print(f"  Output Tokens: {usage.get('completion_tokens', 0)}")
                console.print(f"  Total Tokens: {usage.get('total_tokens', 0)}")
            else:  # Anthropic format
                console.print(f"  Input Tokens: {usage.get('input_tokens', 0)}")
                console.print(f"  Output Tokens: {usage.get('output_tokens', 0)}")
                if "total_tokens" in usage:
                    console.print(f"  Total Tokens: {usage.get('total_tokens', 0)}")
        
        # Costs
        costs = result.get("costs", {})
        if costs:
            console.print("\n[bold yellow]Costs:[/bold yellow]")
            console.print(f"  Input Cost: ${costs.get('input_cost', 0):.6f}")
            console.print(f"  Output Cost: ${costs.get('output_cost', 0):.6f}")
            console.print(f"  Total Cost: ${costs.get('total_cost', 0):.6f}")
        
        console.print("\n" + "-" * 80)


async def compare_same_prompt_different_models():
    """Compare the same prompt across different models."""
    print("\nComparing the same prompt across different models and providers...")
    
    # Create the comparison configuration
    comparison_config = {
        "comparisons": [
            {
                "provider": "openai",
                "model": "gpt-4o-mini",
                "system_prompt": "You are a helpful assistant.",
                "user_prompt": "Explain what a neural network is in exactly one paragraph.",
                "temperature": 0.7
            },
            {
                "provider": "anthropic",
                "model": "claude-3-5-haiku-20241022",
                "system_prompt": "You are a helpful assistant.",
                "user_prompt": "Explain what a neural network is in exactly one paragraph.",
                "temperature": 0.7
            },
            {
                "provider": "openai",
                "model": "gpt-4o",
                "system_prompt": "You are a helpful assistant.",
                "user_prompt": "Explain what a neural network is in exactly one paragraph.",
                "temperature": 0.7
            }
        ]
    }
    
    return comparison_config


async def compare_different_prompts_same_model():
    """Compare different prompts on the same model."""
    print("\nComparing different prompts on the same model...")
    
    # Create the comparison configuration
    comparison_config = {
        "comparisons": [
            {
                "provider": "openai",
                "model": "gpt-4o-mini",
                "system_prompt": "You are a helpful assistant who speaks formally.",
                "user_prompt": "Explain what a neural network is.",
                "temperature": 0.7
            },
            {
                "provider": "openai",
                "model": "gpt-4o-mini",
                "system_prompt": "You are a teacher explaining to a 10-year-old.",
                "user_prompt": "Explain what a neural network is.",
                "temperature": 0.7
            },
            {
                "provider": "openai",
                "model": "gpt-4o-mini",
                "system_prompt": "You are a friendly assistant who uses simple analogies.",
                "user_prompt": "Explain what a neural network is.",
                "temperature": 0.7
            }
        ]
    }
    
    return comparison_config


async def compare_temperature_variations():
    """Compare the effect of different temperature settings."""
    print("\nComparing the effect of different temperature settings...")
    
    # Create the comparison configuration
    comparison_config = {
        "comparisons": [
            {
                "provider": "anthropic",
                "model": "claude-3-5-haiku-20241022",
                "system_prompt": "You are a helpful assistant.",
                "user_prompt": "Give me three creative uses for an old tire.",
                "temperature": 0.1,
                "max_tokens": 150
            },
            {
                "provider": "anthropic",
                "model": "claude-3-5-haiku-20241022",
                "system_prompt": "You are a helpful assistant.",
                "user_prompt": "Give me three creative uses for an old tire.",
                "temperature": 0.5,
                "max_tokens": 150
            },
            {
                "provider": "anthropic",
                "model": "claude-3-5-haiku-20241022",
                "system_prompt": "You are a helpful assistant.",
                "user_prompt": "Give me three creative uses for an old tire.",
                "temperature": 1.0,
                "max_tokens": 150
            }
        ]
    }
    
    return comparison_config


async def compare_single_prompt():
    """Demonstrate using test_comparison for single prompt testing (replacing the old test_prompt functionality)."""
    print("\nSingle prompt testing (replaces old test_prompt functionality)...")
    
    # Create a single prompt configuration using test_comparison
    comparison_config = {
        "comparisons": [
            {
                "provider": "openai",
                "model": "gpt-4o-mini",
                "system_prompt": "You are a helpful assistant that speaks like a pirate.",
                "user_prompt": "Tell me about the weather today.",
                "temperature": 0.9,
                "max_tokens": 150
            }
        ]
    }
    
    return comparison_config


async def custom_comparison():
    """Run a custom comparison based on user input."""
    print("\nCustom Comparison:")
    
    # Get user input for the comparison
    comparisons = []
    
    for i in range(1, 3):  # Allow 2 comparisons by default
        print(f"\nConfiguration #{i}:")
        
        provider = input("Provider (openai/anthropic): ").strip().lower()
        if provider not in ["openai", "anthropic"]:
            print(f"Invalid provider: {provider}. Using 'openai' as default.")
            provider = "openai"
        
        if provider == "openai":
            default_model = "gpt-4o-mini"
        else:
            default_model = "claude-3-5-haiku-20241022"
            
        model = input(f"Model (default: {default_model}): ").strip()
        if not model:
            model = default_model
            
        system_prompt = input("System prompt (default: 'You are a helpful assistant.'): ").strip()
        if not system_prompt:
            system_prompt = "You are a helpful assistant."
            
        user_prompt = input("User prompt: ").strip()
        if not user_prompt:
            print("User prompt cannot be empty.")
            continue
            
        temp_input = input("Temperature (0.0-1.0, default: 0.7): ").strip()
        try:
            temperature = float(temp_input) if temp_input else 0.7
        except ValueError:
            print("Invalid temperature. Using default 0.7.")
            temperature = 0.7
            
        comparisons.append({
            "provider": provider,
            "model": model,
            "system_prompt": system_prompt,
            "user_prompt": user_prompt,
            "temperature": temperature
        })
        
        if i < 2:  # Only ask to continue if we're not at the maximum
            continue_input = input("\nAdd another configuration? (y/n): ").lower()
            if continue_input != 'y':
                break
    
    if not comparisons:
        print("No valid configurations provided.")
        return None
        
    return {"comparisons": comparisons}


async def main():
    """Run the example client for comparing prompts."""
    # Load environment variables
    load_environment()
    
    # Check if required API keys are available
    required_keys = ["OPENAI_API_KEY", "ANTHROPIC_API_KEY"]
    missing_keys = [key for key in required_keys if not os.environ.get(key)]
    
    if missing_keys:
        print(f"Warning: Missing API keys: {', '.join(missing_keys)}")
        print("Some examples may not work without these keys.")
    
    # Connect to the server
    async with stdio_client(
        StdioServerParameters(command="prompt-tester")
    ) as (read, write):
        async with ClientSession(read, write) as session:
            # Initialize the connection
            await session.initialize()
            
            # List available tools to verify the test_comparison tool exists
            raw_tools = await session.list_tools()
            
            # Print all available tools for debugging
            print("\n=== AVAILABLE TOOLS ===")
            
            # Extract the actual tool objects from the response
            tools = []
            # Check if raw_tools is a dict with a 'tools' key (newer MCP format)
            if hasattr(raw_tools, 'tools'):
                # Handle if tools is an attribute
                tools = raw_tools.tools
            elif isinstance(raw_tools, dict) and 'tools' in raw_tools:
                # Handle if tools is a dictionary key
                tools = raw_tools['tools']
            else:
                # Assume raw_tools itself is the list of tools (older format)
                tools = raw_tools
            
            # Now extract names from the tool objects
            tool_names = []
            
            for tool in tools:
                # Handle tuple format (low-level API)
                if isinstance(tool, tuple):
                    name = tool[0]
                    description = tool[1] if len(tool) > 1 else "No description"
                    print(f"- {name}: {description}")
                    tool_names.append(name)
                # Handle object format (higher-level API) with name attribute
                elif hasattr(tool, 'name'):
                    print(f"- {tool.name}: {tool.description if hasattr(tool, 'description') else 'No description'}")
                    tool_names.append(tool.name)
                # Handle dictionary format
                elif isinstance(tool, dict) and 'name' in tool:
                    print(f"- {tool['name']}: {tool.get('description', 'No description')}")
                    tool_names.append(tool['name'])
                else:
                    print(f"- Unknown tool format: {type(tool)}")
            
            print(f"\nExtracted tool names: {', '.join(tool_names)}")
            print("===========================\n")
                
            if "test_comparison" not in tool_names:
                print("Error: The 'test_comparison' tool is not available. Please check your installation.")
                print("\nPossible issues:")
                print("1. The MCP Prompt Tester server may not be running the latest version with the test_comparison tool")
                print("2. The tools.py implementation may not have been properly updated")
                print("3. The command to start the server might need adjustment")
                print("\nTry manually running:")
                print("  cd mcp_prompt_tester && python -m mcp_prompt_tester --transport=sse --port=8000")
                return
                
            print("Welcome to the Prompt Comparison Tool!")
            print("This tool allows you to compare responses from different models and providers.")
            
            while True:
                print("\nPlease select a comparison type:")
                print("1. Same prompt across different models")
                print("2. Different prompts on the same model")
                print("3. Temperature variations comparison")
                print("4. Single prompt test (replaces old test_prompt)")
                print("5. Custom comparison")
                print("6. Exit")
                
                choice = input("\nEnter your choice (1-6): ").strip()
                
                if choice == "6":
                    print("Exiting...")
                    break
                    
                comparison_config = None
                
                if choice == "1":
                    comparison_config = await compare_same_prompt_different_models()
                elif choice == "2":
                    comparison_config = await compare_different_prompts_same_model()
                elif choice == "3":
                    comparison_config = await compare_temperature_variations()
                elif choice == "4":
                    comparison_config = await compare_single_prompt()
                elif choice == "5":
                    comparison_config = await custom_comparison()
                else:
                    print("Invalid choice. Please try again.")
                    continue
                
                if not comparison_config:
                    continue
                
                # Run the comparison
                try:
                    print("\nRunning comparison...")
                    comparison_result = await session.call_tool("test_comparison", comparison_config)
                    parsed = parse_response(comparison_result)
                    
                    if parsed and not parsed.get("isError", False):
                        print("\nComparison completed successfully!\n")
                        # Display the comparison results
                        display_comparison_results(parsed["results"])
                    else:
                        error = parsed.get("error", "Unknown error") if parsed else "Failed to parse response"
                        print(f"Error running comparison: {error}")
                        
                except Exception as e:
                    print(f"Error running comparison: {str(e)}")
                
                cont = input("\nPress Enter to continue or 'q' to quit: ")
                if cont.lower() == 'q':
                    break


if __name__ == "__main__":
    asyncio.run(main()) 