import time
import asyncio
import logging
from openai import AsyncOpenAI, RateLimitError, APIError
from typing import Optional
from ..base import BaseAIProvider
from ..models import AIRequest, AIResponse, TokenUsage, CostDetails

logger = logging.getLogger(__name__)


class OpenAIProvider(BaseAIProvider):
    # Source: https://openai.com/api/pricing/
    PRICING_TABLE = {
        # GPT-5 Series (Latest - 2025)
        "gpt-5": {"prompt": 0.00125, "completion": 0.01},  # $1.25/$10 per 1M tokens
        "gpt-5-mini": {"prompt": 0.00025, "completion": 0.002},  # $0.25/$2 per 1M tokens
        "gpt-5-mini-2025-08-07": {"prompt": 0.25, "completion": 2.00},  # $0.25/$2.00 per 1M tokens (snapshot)
        
        # GPT-4.1 Series
        "gpt-4-1": {"prompt": 0.002, "completion": 0.008},  # $2/$8 per 1M tokens
        
        # GPT-4o Series
        "gpt-4o": {"prompt": 0.0025, "completion": 0.01},  # $2.50/$10 per 1M tokens
        "gpt-4o-2024-08-06": {"prompt": 0.0025, "completion": 0.01},
        "gpt-4o-2024-05-13": {"prompt": 0.0025, "completion": 0.01},
        
        # GPT-4o Mini
        "gpt-4o-mini": {"prompt": 0.00015, "completion": 0.0006},  # $0.15/$0.60 per 1M tokens
        "gpt-4o-mini-2024-07-18": {"prompt": 0.00015, "completion": 0.0006},
        
        # GPT-4 Series
        "gpt-4": {"prompt": 0.03, "completion": 0.06},  # $30/$60 per 1M tokens
        "gpt-4-32k": {"prompt": 0.06, "completion": 0.12},  # $60/$120 per 1M tokens
        "gpt-4-turbo": {"prompt": 0.01, "completion": 0.03},  # $10/$30 per 1M tokens
        
        # GPT-3.5 Series
        "gpt-3.5-turbo": {"prompt": 0.0005, "completion": 0.0015},  # $0.50/$1.50 per 1M tokens
    }
    
    def __init__(self, api_key: str, default_model: str = "gpt-5-mini-2025-08-07"):
        if not api_key:
            raise ValueError("OpenAIProvider requires a valid API key")
        self.client = AsyncOpenAI(api_key=api_key)
        self.default_model = default_model
    
    async def complete(self, request: AIRequest, max_retries: int = 3) -> AIResponse:

        start_time = time.time()
        self.validate_request(request)
        model = request.model or self.default_model
        params = {
            "model": model,
            "messages": [{"role": "user", "content": request.prompt}],
        }
        
        # GPT-5 models REQUIRE max_completion_tokens (max_tokens is not supported)
        # CRITICAL BUG: GPT-5 returns empty content when hitting max_completion_tokens limit
        # Workaround: Use a fixed very high limit (16000) so responses almost never hit it
        # The bug only triggers when hitting the exact limit, so we set it extremely high
        # Most responses are 100-1000 tokens, so 16000 is safe and avoids the bug
        is_gpt5 = model.startswith("gpt-5") or "gpt-5" in model
        if request.max_tokens:
            if is_gpt5:
                # GPT-5 bug workaround: Use fixed high limit (16000 tokens) regardless of request
                # This ensures responses complete naturally before hitting the limit
                # Only extremely long responses (>16000 tokens) will hit it (very rare)
                params["max_completion_tokens"] = 16000  # Fixed high limit to avoid bug
            else:
                params["max_tokens"] = request.max_tokens
        
        # GPT-5 models only support temperature=1 (default), so don't pass temperature parameter
        # For other models, use the requested temperature
        if request.temperature is not None and not is_gpt5:
            params["temperature"] = request.temperature
        # For GPT-5, temperature is always 1 (default), so we don't set it
        
        # Retry logic with exponential backoff
        
        for attempt in range(max_retries):
            try:
                response = await self.client.chat.completions.create(**params)
                break  # Success, exit retry loop
            except RateLimitError as e:
                if attempt == max_retries - 1:
                    # Last attempt failed, raise the error
                    logger.error(f"OpenAI rate limit exceeded after {max_retries} attempts")
                    raise
                
                # Calculate wait time: 1s, 2s, 4s (exponential backoff)
                wait_time = 2 ** attempt
                logger.warning(
                    f"OpenAI rate limit exceeded (attempt {attempt + 1}/{max_retries}), "
                    f"retrying in {wait_time}s..."
                )
                await asyncio.sleep(wait_time)
            except APIError as e:
                # For other API errors (like 429 from other sources), also retry
                if e.status_code == 429 and attempt < max_retries - 1:
                    wait_time = 2 ** attempt
                    logger.warning(
                        f"OpenAI API error 429 (attempt {attempt + 1}/{max_retries}), "
                        f"retrying in {wait_time}s..."
                    )
                    await asyncio.sleep(wait_time)
                else:
                    # Non-retryable error or last attempt
                    raise Exception(f"OpenAI API error: {e}")
            except Exception as e:
                # Other exceptions - don't retry
                raise Exception(f"OpenAI API error: {e}")
        
        latency_ms = (time.time() - start_time) * 1000
        
        usage = TokenUsage.from_openai(response.usage)
        
        cost = self.calculate_cost(model, usage)
        
        # Handle empty content (can happen with GPT-5 models)
        if not response.choices or len(response.choices) == 0:
            logger.error(f"[OpenAI] API returned no choices! Model: {model}, Response ID: {response.id}, Usage: {usage}")
            raise Exception(f"OpenAI API returned no choices in response for model {model}")
        
        choice = response.choices[0]
        content = choice.message.content or ""
        
        if not content and usage.completion_tokens > 0:
            logger.error(f"Empty content returned but {usage.completion_tokens} completion tokens used. "
                        f"Model: {model}, Finish reason: {choice.finish_reason}")
            # If we hit the limit and got empty content, this is a known GPT-5 bug
            if choice.finish_reason == 'length' and is_gpt5:
                # Return a fallback message instead of raising an error
                content = (
                    "⚠️ I encountered a technical issue generating a response. "
                    "This appears to be a bug with the GPT-5 model. Please try again or rephrase your question."
                )
                logger.warning("Using fallback message due to GPT-5 empty content bug")
        
        return AIResponse(
            content=content,
            model=response.model,
            usage=usage,
            cost=cost,
            latency_ms=latency_ms,
            metadata={
                "finish_reason": choice.finish_reason,
                "request_id": response.id,
            }
        )
    
    def calculate_cost(self, model: str, usage: TokenUsage) -> CostDetails:
        """Calculate cost based on OpenAI pricing."""
        if model not in self.PRICING_TABLE:
            rates = {"prompt": 0.0, "completion": 0.0}
        else:
            rates = self.PRICING_TABLE[model]
        
        # Pricing table rates are per 1M tokens, so divide by 1,000,000
        input_cost = (usage.prompt_tokens / 1_000_000) * rates["prompt"]
        output_cost = (usage.completion_tokens / 1_000_000) * rates["completion"]
        
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
            "gpt-5",            # Latest GPT-5 series (2025)
            "gpt-5-mini",       # Latest GPT-5 mini series (2025)
            "gpt-4-1",          # GPT-4.1 series
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
    
    async def close(self):
        await self.client.close()

