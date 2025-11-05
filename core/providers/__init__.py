
from typing import Optional
from ..base_provider import BaseAIProvider
from ..ai_models import AIConfig
from .openai_provider import OpenAIProvider
from .anthropic_provider import AnthropicProvider


def create_provider(config: AIConfig) -> BaseAIProvider:
    provider_registry = {
        "openai": OpenAIProvider,
        "anthropic": AnthropicProvider,
    }
    
    provider_class = provider_registry.get(config.model_name)
    
    if not provider_class:
        raise ValueError(
            f"Unknown provider: {config.model_name}. "
            f"Supported providers: {list(provider_registry.keys())}"
        )
    
    if config.model_name == "openai":
        api_key = config.open_api_key
        default_model = "gpt-4o-mini"  # Stable and always returns content
    elif config.model_name == "anthropic":
        api_key = config.anthopic_api_key
        default_model = "claude-haiku-4-5"  # Latest 2025 model
    else:
        raise ValueError(f"Provider configuration not implemented: {config.model_name}")
    
    return provider_class(api_key=api_key, default_model=default_model)


__all__ = ["create_provider", "OpenAIProvider", "AnthropicProvider"]

