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
        ValueError: If all API keys are missing (at least one is required)
    """
    api_keys = {
        "OPENAI_API_KEY": os.environ.get("OPENAI_API_KEY"),
        "ANTHROPIC_API_KEY": os.environ.get("ANTHROPIC_API_KEY"),
    }
    
    # Check if at least one API key is provided
    if not any(api_keys.values()):
        raise ValueError(
            "At least one API key is required. Please set either OPENAI_API_KEY or "
            "ANTHROPIC_API_KEY in your environment or in a .env file."
        )
    
    # Return only the keys that are actually present
    return {name: value for name, value in api_keys.items() if value}


def get_api_key(provider: str, raise_error: bool = True) -> Optional[str]:
    """
    Get the API key for a specific provider.
    
    Args:
        provider: The provider name (e.g., 'openai', 'anthropic')
        raise_error: Whether to raise an error if the key is missing (default: True)
        
    Returns:
        The API key as a string, or None if not found and raise_error is False
        
    Raises:
        ValueError: If the API key is not set and raise_error is True
    """
    env_var_name = f"{provider.upper()}_API_KEY"
    api_key = os.environ.get(env_var_name)
    
    if not api_key and raise_error:
        raise ValueError(
            f"Missing API key for {provider}. "
            f"Please set {env_var_name} in your environment or in a .env file."
        )
    
    return api_key 