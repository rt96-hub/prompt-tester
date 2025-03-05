#!/usr/bin/env python
"""
Example demonstrating the multi-turn conversation capabilities.

This example shows how to:
1. Start a new conversation
2. Continue an existing conversation
3. Get conversation details
4. List all active conversations
5. Close a conversation

Run this example with:
    python examples/conversation_example.py
"""

import asyncio
import json
import os
import sys
import dotenv
from tabulate import tabulate
from rich.console import Console
from rich.panel import Panel
from rich.markdown import Markdown
from rich import print as rprint

# Add the parent directory to sys.path to import from mcp_prompt_tester
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from mcp_prompt_tester.tools.test_multiturn_conversation import test_multiturn_conversation
from mcp_prompt_tester.env import load_env_files


def parse_response(result):
    """Parse the JSON response from the tool."""
    try:
        return json.loads(result.text)
    except json.JSONDecodeError:
        rprint(f"[red]Error: Could not decode JSON response:[/red] {result.text}")  # Improved error reporting
        return {"isError": True, "error": "Invalid JSON response from server"}  # Return a consistent error structure
    except Exception as e:
        rprint(f"[red]Error parsing response:[/red] {str(e)}")
        return {"isError": True, "error": f"Error parsing response: {str(e)}"}


async def send_request(mode, **kwargs):
    """Send a request to the test_multiturn_conversation tool."""
    # Combine mode with any other arguments
    arguments = {"mode": mode, **kwargs}
    
    # Call the test_multiturn_conversation tool directly
    result = await test_multiturn_conversation(arguments)
    return parse_response(result)


def display_conversation(conversation_data, title=None):
    """Display a conversation in a formatted way."""
    console = Console()
    
    if title:
        console.print(f"[bold blue]{title}[/bold blue]")
    
    # Display basic conversation info
    if "provider" in conversation_data and "model" in conversation_data:
        console.print(f"Provider: [green]{conversation_data['provider']}[/green]")
        console.print(f"Model: [green]{conversation_data['model']}[/green]")
    
    if "system_prompt" in conversation_data:
        console.print(Panel(
            conversation_data["system_prompt"], 
            title="System Prompt",
            border_style="blue"
        ))
    
    # Display message history
    history = conversation_data.get("history", [])
    
    for i, message in enumerate(history):
        role = message["role"]
        content = message["content"]
        
        if role == "user":
            console.print(Panel(
                content,
                title="User",
                border_style="green"
            ))
        else:
            console.print(Panel(
                Markdown(content),
                title="Assistant",
                border_style="purple"
            ))
    
    # Display usage stats if available
    if "usage" in conversation_data and conversation_data["usage"]:
        tokens = conversation_data["usage"]
        costs = conversation_data.get("costs", {})
        
        usage_table = [
            ["Prompt Tokens", tokens.get("prompt_tokens", "N/A")],
            ["Completion Tokens", tokens.get("completion_tokens", "N/A")],
            ["Total Tokens", tokens.get("total_tokens", "N/A")],
            ["Total Cost (USD)", f"${costs.get('total_cost_usd', 0):.6f}"],
        ]
        
        console.print(tabulate(
            usage_table,
            headers=["Metric", "Value"],
            tablefmt="rounded_grid"
        ))
    
    console.print("\n")


def display_conversation_list(list_data):
    """Display a list of conversations."""
    console = Console()
    console.print("[bold blue]Active Conversations[/bold blue]")
    
    conversation_count = list_data.get("conversation_count", 0)
    console.print(f"Found [green]{conversation_count}[/green] active conversations\n")
    
    conversations = list_data.get("conversations", {})
    
    for conv_id, conv_data in conversations.items():
        console.print(Panel(
            f"[bold]Provider:[/bold] {conv_data.get('provider')}\n"
            f"[bold]Model:[/bold] {conv_data.get('model')}\n"
            f"[bold]First message:[/bold] {conv_data.get('first_user_message')[:50]}...\n"
            f"[bold]Latest response:[/bold] {conv_data.get('latest_assistant_message')[:50]}...\n"
            f"[bold]Messages:[/bold] {conv_data.get('message_count')}",
            title=f"Conversation ID: {conv_id}",
            border_style="blue"
        ))
    
    console.print("\n")


async def simple_conversation_example():
    """Demonstrate a simple two-turn conversation."""
    console = Console()
    console.print("[bold green]===== Simple Conversation Example =====[/bold green]\n")
    
    # Start a conversation
    console.print("[bold]Starting a new conversation...[/bold]")
    start_result = await send_request(
        mode="start",
        provider="openai",
        model="gpt-3.5-turbo",
        system_prompt="You are a helpful assistant that speaks concisely.",
        user_prompt="What are the three laws of robotics?",
        temperature=0.7,
        max_tokens=500
    )
    
    if start_result.get("isError", False):
        console.print(f"[bold red]Error:[/bold red] {start_result.get('error')}")
        return
    
    # Store the conversation ID
    conversation_id = start_result["conversation_id"]
    console.print(f"Conversation started with ID: [bold cyan]{conversation_id}[/bold cyan]\n")
    
    # Display the initial response
    console.print(Panel(
        start_result["response"],
        title="Response to initial question",
        border_style="purple"
    ))
    
    # Continue the conversation
    console.print("[bold]Continuing the conversation...[/bold]")
    continue_result = await send_request(
        mode="continue",
        conversation_id=conversation_id,
        user_prompt="Who formulated these laws and in what work did they first appear?"
    )
    
    if continue_result.get("isError", False):
        console.print(f"[bold red]Error:[/bold red] {continue_result.get('error')}")
        return
    
    # Display the follow-up response
    console.print(Panel(
        continue_result["response"],
        title="Response to follow-up question",
        border_style="purple"
    ))
    
    # Get the full conversation details
    console.print("[bold]Getting full conversation details...[/bold]")
    get_result = await send_request(
        mode="get",
        conversation_id=conversation_id
    )
    
    if get_result.get("isError", False):
        console.print(f"[bold red]Error:[/bold red] {get_result.get('error')}")
        return
    
    # Display the full conversation
    display_conversation(get_result, title="Full Conversation History")
    
    # Close the conversation
    console.print("[bold]Closing the conversation...[/bold]")
    close_result = await send_request(
        mode="close",
        conversation_id=conversation_id
    )
    
    if close_result.get("isError", False):
        console.print(f"[bold red]Error:[/bold red] {close_result.get('error')}")
        return
    
    console.print(f"[green]{close_result.get('message', 'Conversation closed successfully')}[/green]\n")


async def multi_provider_example():
    """Demonstrate conversations with different providers."""
    console = Console()
    console.print("[bold green]===== Multi-Provider Example =====[/bold green]\n")
    
    conversation_ids = []
    
    # Start conversations with different providers
    providers = [
        {"name": "openai", "model": "gpt-3.5-turbo"},
        {"name": "anthropic", "model": "claude-3-haiku-20240307"}
    ]
    
    for provider in providers:
        console.print(f"[bold]Starting a conversation with {provider['name']}...[/bold]")
        
        start_result = await send_request(
            mode="start",
            provider=provider["name"],
            model=provider["model"],
            system_prompt="You are a helpful assistant with extensive knowledge about machine learning.",
            user_prompt="Explain the concept of attention in transformer models briefly.",
            temperature=0.7
        )
        
        if start_result.get("isError", False):
            console.print(f"[bold red]Error with {provider['name']}:[/bold red] {start_result.get('error')}")
            continue
        
        conversation_id = start_result["conversation_id"]
        conversation_ids.append(conversation_id)
        
        console.print(f"{provider['name']} conversation started with ID: [bold cyan]{conversation_id}[/bold cyan]")
        console.print(Panel(
            start_result["response"],
            title=f"{provider['name']} / {provider['model']} Response",
            border_style="purple"
        ))
    
    # List all active conversations
    console.print("[bold]Listing all active conversations...[/bold]")
    list_result = await send_request(mode="list")
    
    if list_result.get("isError", False):
        console.print(f"[bold red]Error:[/bold red] {list_result.get('error')}")
    else:
        display_conversation_list(list_result)
    
    # Clean up by closing all conversations
    for conv_id in conversation_ids:
        close_result = await send_request(
            mode="close",
            conversation_id=conv_id
        )
        
        if close_result.get("isError", False):
            console.print(f"[bold red]Error closing {conv_id}:[/bold red] {close_result.get('error')}")
        else:
            console.print(f"Closed conversation [cyan]{conv_id}[/cyan]")


async def multi_turn_conversation_example():
    """Demonstrate a longer multi-turn conversation with context retention."""
    console = Console()
    console.print("[bold green]===== Multi-Turn Conversation Example =====[/bold green]\n")
    
    # Start a conversation
    console.print("[bold]Starting a new conversation...[/bold]")
    start_result = await send_request(
        mode="start",
        provider="openai",
        model="gpt-3.5-turbo",
        system_prompt="You are a creative writing assistant who helps craft short stories one step at a time.",
        user_prompt="Let's create a short story together. Start by suggesting a main character and setting.",
        temperature=0.9  # Higher temperature for more creativity
    )
    
    if start_result.get("isError", False):
        console.print(f"[bold red]Error:[/bold red] {start_result.get('error')}")
        return
    
    # Store the conversation ID
    conversation_id = start_result["conversation_id"]
    console.print(f"Conversation started with ID: [bold cyan]{conversation_id}[/bold cyan]\n")
    
    # Display the initial response
    console.print(Panel(
        start_result["response"],
        title="Assistant's Initial Suggestion",
        border_style="purple"
    ))
    
    # Continue the conversation for multiple turns
    follow_up_questions = [
        "Great! Now add a conflict or challenge for this character.",
        "How does the character initially react to this challenge?",
        "Now introduce a supporting character who helps our protagonist.",
        "What unexpected twist occurs in the middle of the story?",
        "How does the story resolve?"
    ]
    
    for i, question in enumerate(follow_up_questions, 1):
        console.print(f"[bold]Turn {i+1}: {question}[/bold]")
        
        continue_result = await send_request(
            mode="continue",
            conversation_id=conversation_id,
            user_prompt=question
        )
        
        if continue_result.get("isError", False):
            console.print(f"[bold red]Error:[/bold red] {continue_result.get('error')}")
            break
        
        # Display the response for this turn
        console.print(Panel(
            continue_result["response"],
            title=f"Turn {i+1} Response",
            border_style="purple"
        ))
    
    # Get the full conversation to see the complete story
    console.print("[bold]Getting the complete story...[/bold]")
    get_result = await send_request(
        mode="get",
        conversation_id=conversation_id
    )
    
    if get_result.get("isError", False):
        console.print(f"[bold red]Error:[/bold red] {get_result.get('error')}")
        return
    
    # Display the full conversation as a story
    display_conversation(get_result, title="Complete Story Development")
    
    # Display usage information
    if "usage" in get_result and get_result["usage"]:
        console.print("[bold]Final Usage Statistics[/bold]")
        tokens = get_result["usage"]
        costs = get_result.get("costs", {})
        
        usage_table = [
            ["Total Tokens", tokens.get("total_tokens", "N/A")],
            ["Total Cost (USD)", f"${costs.get('total_cost_usd', 0):.6f}"],
        ]
        
        console.print(tabulate(
            usage_table,
            headers=["Metric", "Value"],
            tablefmt="rounded_grid"
        ))
    
    # Close the conversation
    await send_request(
        mode="close",
        conversation_id=conversation_id
    )


async def error_handling_example():
    """Demonstrate error handling scenarios."""
    console = Console()
    console.print("[bold green]===== Error Handling Example =====[/bold green]\n")
    
    # Try to continue a non-existent conversation
    console.print("[bold]Trying to continue a non-existent conversation...[/bold]")
    invalid_continue = await send_request(
        mode="continue",
        conversation_id="non-existent-uuid",
        user_prompt="This should fail"
    )
    
    console.print(Panel(
        f"Error: {invalid_continue.get('error', 'Unknown error')}",
        title="Continue Non-existent Conversation",
        border_style="red"
    ))
    
    # Try to start a conversation with an invalid provider
    console.print("[bold]Trying to start a conversation with an invalid provider...[/bold]")
    invalid_provider = await send_request(
        mode="start",
        provider="invalid-provider",
        model="some-model",
        system_prompt="This should fail",
        user_prompt="Test"
    )
    
    console.print(Panel(
        f"Error: {invalid_provider.get('error', 'Unknown error')}",
        title="Invalid Provider",
        border_style="red"
    ))
    
    # Try with an invalid mode
    console.print("[bold]Trying to use an invalid mode...[/bold]")
    invalid_mode = await send_request(
        mode="invalid-mode"
    )
    
    console.print(Panel(
        f"Error: {invalid_mode.get('error', 'Unknown error')}",
        title="Invalid Mode",
        border_style="red"
    ))


async def main():
    """Run the conversation examples."""
    # Load environment variables from .env files
    load_env_files()
    
    console = Console()
    console.print("[bold yellow]==================================[/bold yellow]")
    console.print("[bold yellow]    Multi-turn Conversation Examples    [/bold yellow]")
    console.print("[bold yellow]==================================[/bold yellow]\n")
    
    try:
        # Simple conversation example
        await simple_conversation_example()
        
        console.print("\n[bold yellow]----------------------------------[/bold yellow]\n")
        
        # Multi-provider example
        await multi_provider_example()
        
        console.print("\n[bold yellow]----------------------------------[/bold yellow]\n")
        
        # Longer multi-turn conversation
        await multi_turn_conversation_example()
        
        console.print("\n[bold yellow]----------------------------------[/bold yellow]\n")
        
        # Error handling example
        await error_handling_example()
        
    except Exception as e:
        console.print(f"[bold red]Unexpected error:[/bold red] {str(e)}")
    
    console.print("\n[bold green]Examples completed![/bold green]")


if __name__ == "__main__":
    asyncio.run(main()) 