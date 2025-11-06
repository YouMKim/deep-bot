import time
import openai
from typing import Optional
from ..base_provider import BaseAIProvider
from ..ai_models import AIRequest, AIResponse, TokenUsage, CostDetails


class OpenAIProvider(BaseAIProvider):

    # OpenAI pricing per 1K tokens (updated January 2025)
    # Source: https://openai.com/api/pricing/
    PRICING_TABLE = {
        # o1 Series (Reasoning models)
        "o1-preview": {"prompt": 0.015, "completion": 0.06},  # $15/$60 per 1M tokens
        "o1-mini": {"prompt": 0.003, "completion": 0.012},  # $3/$12 per 1M tokens

        # GPT-4o Series (Latest)
        "gpt-4o": {"prompt": 0.0025, "completion": 0.01},  # $2.50/$10 per 1M tokens
        "gpt-4o-2024-11-20": {"prompt": 0.0025, "completion": 0.01},
        "gpt-4o-2024-08-06": {"prompt": 0.0025, "completion": 0.01},
        "gpt-4o-2024-05-13": {"prompt": 0.0025, "completion": 0.01},

        # GPT-4o Mini
        "gpt-4o-mini": {"prompt": 0.00015, "completion": 0.0006},  # $0.15/$0.60 per 1M tokens
        "gpt-4o-mini-2024-07-18": {"prompt": 0.00015, "completion": 0.0006},

        # GPT-4 Turbo
        "gpt-4-turbo": {"prompt": 0.01, "completion": 0.03},  # $10/$30 per 1M tokens
        "gpt-4-turbo-2024-04-09": {"prompt": 0.01, "completion": 0.03},

        # GPT-4 (Legacy)
        "gpt-4": {"prompt": 0.03, "completion": 0.06},  # $30/$60 per 1M tokens
        "gpt-4-0613": {"prompt": 0.03, "completion": 0.06},

        # GPT-3.5 Turbo
        "gpt-3.5-turbo": {"prompt": 0.0005, "completion": 0.0015},  # $0.50/$1.50 per 1M tokens
        "gpt-3.5-turbo-0125": {"prompt": 0.0005, "completion": 0.0015},
    }
    
    def __init__(self, api_key: str, default_model: str = "gpt-4o-mini"):
        self.client = openai.OpenAI(api_key=api_key)
        self.default_model = default_model
    
    async def complete(self, request: AIRequest) -> AIResponse:

        start_time = time.time()
        self.validate_request(request)
        model = request.model or self.default_model
        params = {
            "model": model,
            "messages": [{"role": "user", "content": request.prompt}],
        }
        
        # o1 models use max_completion_tokens instead of max_tokens
        if request.max_tokens:
            if model.startswith("o1"):
                params["max_completion_tokens"] = request.max_tokens
            else:
                params["max_tokens"] = request.max_tokens
        
        if request.temperature is not None:
            params["temperature"] = request.temperature
        
        try:
            response = self.client.chat.completions.create(**params)
        except Exception as e:
            raise Exception(f"OpenAI API error: {e}")
        
        latency_ms = (time.time() - start_time) * 1000
        
        usage = TokenUsage.from_openai(response.usage)
        
        cost = self.calculate_cost(model, usage)
        
        # Handle empty content (can happen with o1 models)
        content = response.choices[0].message.content or ""
        
        return AIResponse(
            content=content,
            model=response.model,
            usage=usage,
            cost=cost,
            latency_ms=latency_ms,
            metadata={
                "finish_reason": response.choices[0].finish_reason,
                "request_id": response.id,
            }
        )
    
    def calculate_cost(self, model: str, usage: TokenUsage) -> CostDetails:
        """Calculate cost based on OpenAI pricing."""
        if model not in self.PRICING_TABLE:
            rates = {"prompt": 0.0, "completion": 0.0}
        else:
            rates = self.PRICING_TABLE[model]
        
        input_cost = (usage.prompt_tokens / 1000) * rates["prompt"]
        output_cost = (usage.completion_tokens / 1000) * rates["completion"]
        
        return CostDetails(
            provider="openai",
            model=model,
            input_cost=input_cost,
            output_cost=output_cost,
            total_cost=input_cost + output_cost
        )
    
    def supports_model(self, model: str) -> bool:
        """Check if this provider supports a model."""
        supported_models = [
            "o1-preview",       # o1 reasoning series
            "o1-mini",          # o1 mini reasoning series
            "gpt-4o",           # GPT-4o series
            "gpt-4o-mini",      # GPT-4o mini series
            "gpt-4-turbo",      # GPT-4 turbo series
            "gpt-4",            # GPT-4 series
            "gpt-3.5-turbo",    # GPT-3.5 turbo series
        ]

        for supported in supported_models:
            if model.startswith(supported):
                return True

        return False
    
    def get_default_model(self) -> str:
        """Get the default model."""
        return self.default_model

