
from typing import Optional
from ..base import BaseAIProvider
from ..models import AIConfig
from .openai import OpenAIProvider
from .anthropic import AnthropicProvider
from .gemini import GeminiProvider


def create_provider(config: AIConfig) -> BaseAIProvider:
    provider_registry = {
        "openai": OpenAIProvider,
        "anthropic": AnthropicProvider,
        "gemini": GeminiProvider,
    }
    
    provider_class = provider_registry.get(config.model_name)
    
    if not provider_class:
        raise ValueError(
            f"Unknown provider: {config.model_name}. "
            f"Supported providers: {list(provider_registry.keys())}"
        )
    
    if config.model_name == "openai":
        api_key = config.open_api_key
        default_model = "gpt-5-mini-2025-08-07"  # Latest GPT-5 mini snapshot (2025-08-07)
    elif config.model_name == "anthropic":
        api_key = config.anthopic_api_key
        default_model = "claude-haiku-4-5"  # Latest 2025 model
    elif config.model_name == "gemini":
        api_key = config.gemini_api_key
        default_model = "gemini-2.5-flash"  # Latest hybrid reasoning model with 1M context
    else:
        raise ValueError(f"Provider configuration not implemented: {config.model_name}")
    
    return provider_class(api_key=api_key, default_model=default_model)


__all__ = ["create_provider", "OpenAIProvider", "AnthropicProvider", "GeminiProvider"]

