"""Environment management utilities."""

import os
from pathlib import Path
from typing import Dict, Optional

from dotenv import load_dotenv


def load_env_files() -> None:
    """
    Load environment variables from .env files.
    
    Looks for .env files in the following locations (in order of precedence):
    1. Current working directory
    2. User's home directory
    """
    # Try loading from current directory first
    local_env = Path('.env')
    if local_env.exists():
        load_dotenv(local_env)
    
    # Then try home directory
    home_env = Path.home() / '.env'
    if home_env.exists():
        load_dotenv(home_env)


def get_required_env_vars() -> Dict[str, str]:
    """
    Get a dictionary of required environment variables and their values.
    
    Returns:
        Dict mapping environment variable names to their values
    
    Raises:
        ValueError: If any required environment variable is missing
    """
    required_vars = {
        "OPENAI_API_KEY": os.environ.get("OPENAI_API_KEY"),
        "ANTHROPIC_API_KEY": os.environ.get("ANTHROPIC_API_KEY"),
    }
    
    missing_vars = [name for name, value in required_vars.items() if not value]
    if missing_vars:
        raise ValueError(
            f"Missing required environment variables: {', '.join(missing_vars)}.\n"
            f"Please set these in your environment or in a .env file."
        )
    
    return required_vars


def get_api_key(provider: str) -> str:
    """
    Get the API key for a specific provider.
    
    Args:
        provider: The provider name (e.g., 'openai', 'anthropic')
        
    Returns:
        The API key as a string
        
    Raises:
        ValueError: If the API key is not set
    """
    env_var_name = f"{provider.upper()}_API_KEY"
    api_key = os.environ.get(env_var_name)
    
    if not api_key:
        raise ValueError(
            f"Missing API key for {provider}. "
            f"Please set {env_var_name} in your environment or in a .env file."
        )
    
    return api_key 