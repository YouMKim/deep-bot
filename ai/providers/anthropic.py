import time
import asyncio
from typing import Optional
from anthropic import AsyncAnthropic

from ..base import BaseAIProvider
from ..models import AIRequest, AIResponse, TokenUsage, CostDetails


class AnthropicProvider(BaseAIProvider):
    # Anthropic pricing per 1M tokens (updated 2025 with Claude 4.5 models)
    # Source: https://www.anthropic.com/pricing
    PRICING_TABLE = {
        # Claude 4.5 Series (Latest - 2025)
        "claude-sonnet-4-5": {"prompt": 3.00, "completion": 15.00},  # $3/$15 per 1M tokens
        "claude-haiku-4-5": {"prompt": 1.00, "completion": 5.00},  # $1/$5 per 1M tokens
        "claude-opus-4-1": {"prompt": 15.00, "completion": 75.00},  # $15/$75 per 1M tokens
        
        # Claude 3.5 Series
        "claude-3-5-sonnet-20241022": {"prompt": 3.00, "completion": 15.00},  # $3/$15 per 1M tokens
        "claude-3-5-sonnet": {"prompt": 3.00, "completion": 15.00},
        "claude-3-5-haiku": {"prompt": 1.00, "completion": 5.00},  # $1/$5 per 1M tokens
        
        # Claude 3 Series
        "claude-3-opus": {"prompt": 15.00, "completion": 75.00},  # $15/$75 per 1M tokens
        "claude-3-opus-20240229": {"prompt": 15.00, "completion": 75.00},
        "claude-3-sonnet": {"prompt": 3.00, "completion": 15.00},  # $3/$15 per 1M tokens
        "claude-3-sonnet-20240229": {"prompt": 3.00, "completion": 15.00},
        "claude-3-haiku": {"prompt": 0.25, "completion": 1.25},  # $0.25/$1.25 per 1M tokens
        "claude-3-haiku-20240307": {"prompt": 0.25, "completion": 1.25},
    }
    
    def __init__(self, api_key: str, default_model: str = "claude-haiku-4-5"):
        from anthropic import Anthropic
        self.client = AsyncAnthropic(api_key=api_key)
        self.default_model = default_model
    
    async def complete(self, request: AIRequest) -> AIResponse:
        start_time = time.time()
        self.validate_request(request)
        model = request.model or self.default_model
        # Convert prompt to Anthropic message format
        params = {
            "model": model,
            "messages": [{"role": "user", "content": request.prompt}],
        }
        
        if request.max_tokens:
            params["max_tokens"] = request.max_tokens
        
        if request.temperature is not None:
            params["temperature"] = request.temperature

        try:
            response = await self.client.messages.create(**params)
        except Exception as e:
            raise Exception(f"Anthropic API error: {e}")
        
        latency_ms = (time.time() - start_time) * 1000
        
        usage = TokenUsage.from_anthropic(
            input_tokens=response.usage.input_tokens,
            output_tokens=response.usage.output_tokens
        )
        
        cost = self.calculate_cost(model, usage)
        
        return AIResponse(
            content=response.content[0].text,
            model=response.model,
            usage=usage,
            cost=cost,
            latency_ms=latency_ms,
            metadata={
                "stop_reason": response.stop_reason,
                "stop_sequence": response.stop_sequence,
                "model": response.model,
            }
        )
    
    def calculate_cost(self, model: str, usage: TokenUsage) -> CostDetails:
        if model not in self.PRICING_TABLE:
            rates = {"prompt": 0.0, "completion": 0.0}
        else:
            rates = self.PRICING_TABLE[model]
        
        input_cost = (usage.prompt_tokens / 1_000_000) * rates["prompt"]
        output_cost = (usage.completion_tokens / 1_000_000) * rates["completion"]
        
        return CostDetails(
            provider="anthropic",
            model=model,
            input_cost=input_cost,
            output_cost=output_cost,
            total_cost=input_cost + output_cost
        )
    
    def supports_model(self, model: str) -> bool:
        """Check if this provider supports a model."""
        supported_models = [
            "claude-sonnet-4-5",  # Latest Claude 4.5 series (2025)
            "claude-haiku-4-5",   # Latest Claude 4.5 series (2025)
            "claude-opus-4-1",    # Claude 4.1 series (2025)
            "claude-3-5-sonnet",  # Claude 3.5 series
            "claude-3-5-haiku",   # Claude 3.5 series
            "claude-3-opus",      # Claude 3 series
            "claude-3-sonnet",    # Claude 3 series
            "claude-3-haiku",     # Claude 3 series
        ]
        
        # Check if model starts with any supported prefix
        for supported in supported_models:
            if model.startswith(supported):
                return True
        
        return False
    
    def get_default_model(self) -> str:
        """Get the default model."""
        return self.default_model

