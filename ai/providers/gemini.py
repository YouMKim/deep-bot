import time
from typing import Optional
import google.generativeai as genai
from ..base import BaseAIProvider
from ..models import AIRequest, AIResponse, TokenUsage, CostDetails


class GeminiProvider(BaseAIProvider):
    # Source: https://ai.google.dev/pricing
    PRICING_TABLE = {
        # Gemini 2.5 Series (Latest - 2025)
        "gemini-2.5-flash": {"prompt": 0.30, "completion": 2.50},  # $0.30/$2.50 per 1M tokens (hybrid reasoning, 1M context)
        "gemini-2.5-flash-lite": {"prompt": 0.10, "completion": 0.40},  # $0.10/$0.40 per 1M tokens (most cost-effective)
        
        # Gemini 2.0 Series
        "gemini-2.0-flash-exp": {"prompt": 0.075, "completion": 0.30},  # $0.075/$0.30 per 1M tokens
        "gemini-2.0-flash-thinking-exp": {"prompt": 0.075, "completion": 0.30},
        
        # Gemini 1.5 Series
        "gemini-1.5-pro": {"prompt": 1.25, "completion": 5.00},  # $1.25/$5.00 per 1M tokens
        "gemini-1.5-pro-latest": {"prompt": 1.25, "completion": 5.00},
        "gemini-1.5-flash": {"prompt": 0.075, "completion": 0.30},  # $0.075/$0.30 per 1M tokens
        "gemini-1.5-flash-latest": {"prompt": 0.075, "completion": 0.30},
        "gemini-1.5-flash-8b": {"prompt": 0.0375, "completion": 0.15},  # $0.0375/$0.15 per 1M tokens
        
        # Gemini 1.0 Series
        "gemini-pro": {"prompt": 0.50, "completion": 1.50},  # $0.50/$1.50 per 1M tokens
        "gemini-pro-vision": {"prompt": 0.50, "completion": 1.50},
    }
    
    def __init__(self, api_key: str, default_model: str = "gemini-2.5-flash"):
        genai.configure(api_key=api_key)
        self.client = genai.GenerativeModel
        self.default_model = default_model
    
    async def complete(self, request: AIRequest) -> AIResponse:
        start_time = time.time()
        self.validate_request(request)
        model = request.model or self.default_model
        
        # Create model instance
        genai_model = self.client(model)
        
        # Prepare generation config
        generation_config = {}
        if request.max_tokens:
            generation_config["max_output_tokens"] = request.max_tokens
        if request.temperature is not None:
            generation_config["temperature"] = max(0.0, min(1.0, request.temperature))
        
        try:
            import asyncio
            if generation_config:
                response = await asyncio.to_thread(
                    genai_model.generate_content,
                    request.prompt,
                    generation_config=generation_config
                )
            else:
                response = await asyncio.to_thread(
                    genai_model.generate_content,
                    request.prompt
                )
        except Exception as e:
            raise Exception(f"Gemini API error: {e}")
        
        latency_ms = (time.time() - start_time) * 1000
        
        # Extract token usage from response
        prompt_tokens = 0
        completion_tokens = 0
        if hasattr(response, 'usage_metadata') and response.usage_metadata:
            prompt_tokens = getattr(response.usage_metadata, 'prompt_token_count', 0)
            completion_tokens = getattr(response.usage_metadata, 'total_token_count', 0) - prompt_tokens
        
        usage = TokenUsage.from_gemini(
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens
        )
        
        cost = self.calculate_cost(model, usage)
        
        # Extract content
        content = ""
        if hasattr(response, 'text') and response.text:
            content = response.text
        elif hasattr(response, 'candidates') and response.candidates:
            # Fallback: try to get text from first candidate
            candidate = response.candidates[0]
            if hasattr(candidate, 'content') and candidate.content:
                if hasattr(candidate.content, 'parts'):
                    content = "".join(part.text for part in candidate.content.parts if hasattr(part, 'text'))
        
        # Extract metadata
        finish_reason = None
        safety_ratings = None
        if hasattr(response, 'candidates') and response.candidates:
            candidate = response.candidates[0]
            finish_reason = getattr(candidate, 'finish_reason', None)
            if hasattr(candidate, 'safety_ratings'):
                safety_ratings = [
                    {
                        "category": getattr(rating, 'category', None),
                        "probability": getattr(rating, 'probability', None)
                    }
                    for rating in candidate.safety_ratings
                ]
        
        return AIResponse(
            content=content,
            model=model,
            usage=usage,
            cost=cost,
            latency_ms=latency_ms,
            metadata={
                "finish_reason": finish_reason,
                "safety_ratings": safety_ratings,
            }
        )
    
    def calculate_cost(self, model: str, usage: TokenUsage) -> CostDetails:
        """Calculate cost based on Gemini pricing."""
        if model not in self.PRICING_TABLE:
            rates = {"prompt": 0.0, "completion": 0.0}
        else:
            rates = self.PRICING_TABLE[model]
        
        input_cost = (usage.prompt_tokens / 1_000_000) * rates["prompt"]
        output_cost = (usage.completion_tokens / 1_000_000) * rates["completion"]
        
        return CostDetails(
            provider="gemini",
            model=model,
            input_cost=input_cost,
            output_cost=output_cost,
            total_cost=input_cost + output_cost
        )
    
    def supports_model(self, model: str) -> bool:
        """Check if this provider supports a model."""
        supported_models = [
            "gemini-2.5-flash",          
            "gemini-2.5-flash-lite",     
            "gemini-2.0-flash-exp",      
            "gemini-2.0-flash-thinking-exp",
            "gemini-1.5-pro",           
            "gemini-1.5-pro-latest",
            "gemini-1.5-flash",
            "gemini-1.5-flash-latest",
            "gemini-1.5-flash-8b",
            "gemini-pro",                
            "gemini-pro-vision",
        ]
        
        for supported in supported_models:
            if model.startswith(supported):
                return True
        
        return False
    
    def get_default_model(self) -> str:
        """Get the default model."""
        return self.default_model

