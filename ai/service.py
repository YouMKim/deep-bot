from .models import AIConfig, AIRequest
from .providers import create_provider

class AIService:
    PROMPT_STYLES = {
        "generic": {
            "system": "You are a helpful assistant that creates natural, flowing summaries of Discord conversations. Write in paragraph form, capturing the overall flow and context of the discussion.",
            "user_template": "Please provide a natural, paragraph-style summary of this Discord conversation:\n\n{text}\n\nSummary:",
        },
        "bullet_points": {
            "system": "You are a helpful assistant that extracts key points from Discord conversations. Present the main topics, decisions, and important information as clear bullet points.",
            "user_template": "Extract the key points from this Discord conversation and present them as bullet points:\n\n{text}\n\nKey Points:",
        },
        "headline": {
            "system": "You are a helpful assistant that creates very brief, headline-style summaries. Capture the essence of the conversation in 1-2 short sentences.",
            "user_template": "Create a brief headline-style summary (1-2 sentences) of this Discord conversation:\n\n{text}\n\nHeadline Summary:",
        },
    }

    def __init__(self, provider_name: str = "openai"):
        """
        Initialize AI service with a specific provider.
        
        Args:
            provider_name: Either "openai", "anthropic", or "gemini"
        """
        config = AIConfig(model_name=provider_name)
        self.provider = create_provider(config)
        self.provider_name = provider_name

    async def summarize_with_style(self, text: str, style: str = "generic") -> dict:
        """
        Generate a summary of text in the specified style.
        
        Args:
            text: The text to summarize
            style: The summary style ("generic", "bullet_points", "headline")
            
        Returns:
            Dictionary with summary results and metadata
        """
        prompt_config = self.PROMPT_STYLES[style]
        full_prompt = self._build_prompt(prompt_config, text)
        
        request = AIRequest(
            prompt=full_prompt,
            max_tokens=self._get_max_tokens_for_style(style)
        )
        response = await self.provider.complete(request)
        
        return {
            "summary": response.content,
            "style": style,
            "tokens_prompt": response.usage.prompt_tokens,
            "tokens_completion": response.usage.completion_tokens,
            "tokens_total": response.usage.total_tokens,
            "model": response.model,
            "cost": response.cost.total_cost,
        }
    
    def _build_prompt(self, prompt_config: dict, text: str) -> str:
        """
        Build a complete prompt from template and text.
        
        Combines system message and user template into a single prompt.
        """
        system_msg = prompt_config["system"]
        user_msg = prompt_config["user_template"].format(text=text)
        
        return f"{system_msg}\n\n{user_msg}"

    def _get_max_tokens_for_style(self, style: str) -> int:
        """Set appropriate max tokens for each style."""
        token_limits = {
            "generic": 300,  
            "bullet_points": 400, 
            "headline": 100,  
        }
        return token_limits.get(style, 300)

    async def compare_all_styles(self, text: str) -> dict:
        results = {}
        for style in self.PROMPT_STYLES.keys():
            results[style] = await self.summarize_with_style(text, style)
        return results
    
    def _clamp_temperature(self, temperature: float) -> float:
        """
        Normalize and clamp temperature to provider-specific limits.
        
        This ensures consistent behavior across providers:
        - Temperature values 0-1 are treated as normalized (0% to 100% creativity)
        - Anthropic (0-1 range): uses value directly
        - Gemini (0-1 range): uses value directly
        - OpenAI (0-2 range): scales value by 2x for equivalent creativity
        
        Examples:
        - 0.7 → Anthropic: 0.7, Gemini: 0.7, OpenAI: 1.4 (both 70% of their range)
        - 0.5 → Anthropic: 0.5, Gemini: 0.5, OpenAI: 1.0 (both 50% of their range)
        - 1.5 → Anthropic: 1.0 (clamped), Gemini: 1.0 (clamped), OpenAI: 1.5 (if >1, assume OpenAI scale)
        
        Args:
            temperature: Requested temperature value (0-1 normalized or 0-2 OpenAI scale)
            
        Returns:
            Normalized and clamped temperature value for the current provider
        """
        if self.provider_name in ["anthropic", "gemini"]:
            # Anthropic and Gemini only support temperature 0-1
            # If value is > 1, assume it's in OpenAI scale, normalize to 0-1
            if temperature > 1.0:
                normalized = temperature / 2.0  
                return max(0.0, min(1.0, normalized))
            else:
                return max(0.0, min(1.0, temperature))
        else:
            # OpenAI supports temperature 0-2
            # If value is <= 1, assume it's normalized, scale to OpenAI range
            if temperature <= 1.0:
                scaled = temperature * 2.0  
                return max(0.0, min(2.0, scaled))
            else:
                return max(0.0, min(2.0, temperature))
    
    async def generate(self, prompt: str, max_tokens: int = 200, temperature: float = 0.7) -> dict:
        """
        Generate a response for any prompt (generic AI completion).
        
        Args:
            prompt: The prompt text to send to the AI
            max_tokens: Maximum tokens in the response (default: 200)
            temperature: Model temperature (default: 0.7)
                       - Values 0-1 are normalized across providers for consistent behavior
                       - Anthropic: 0-1 range (0.7 = 70% creativity)
                       - Gemini: 0-1 range (0.7 = 70% creativity)
                       - OpenAI: 0-2 range (0.7 → normalized to 1.4 = 70% creativity)
                       - Values >1 are assumed to be in OpenAI scale
                       - Automatically normalized and clamped per provider
            
        Returns:
            Dictionary with response results and metadata
        """
        # Normalize and clamp temperature to provider-specific limits
        normalized_temperature = self._clamp_temperature(temperature)
        
        request = AIRequest(
            prompt=prompt,
            max_tokens=max_tokens,
            temperature=normalized_temperature
        )
        
        response = await self.provider.complete(request)
        
        return {
            "content": response.content,
            "tokens_prompt": response.usage.prompt_tokens,
            "tokens_completion": response.usage.completion_tokens,
            "tokens_total": response.usage.total_tokens,
            "model": response.model,
            "cost": response.cost.total_cost,
            "latency_ms": response.latency_ms,
        }

