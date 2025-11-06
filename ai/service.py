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
            provider_name: Either "openai" or "anthropic"
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
        Clamp temperature to provider-specific limits.
        
        Args:
            temperature: Requested temperature value
            
        Returns:
            Clamped temperature value
        """
        if self.provider_name == "anthropic":
            # Anthropic only supports temperature 0-1
            return max(0.0, min(1.0, temperature))
        else:
            # OpenAI supports temperature 0-2
            return max(0.0, min(2.0, temperature))
    
    async def generate(self, prompt: str, max_tokens: int = 200, temperature: float = 0.7) -> dict:
        """
        Generate a response for any prompt (generic AI completion).
        
        Args:
            prompt: The prompt text to send to the AI
            max_tokens: Maximum tokens in the response (default: 200)
            temperature: Model temperature (default: 0.7)
                       - Anthropic: 0-1 range (clamped automatically)
                       - OpenAI: 0-2 range (clamped automatically)
            
        Returns:
            Dictionary with response results and metadata
        """
        # Clamp temperature to provider-specific limits
        clamped_temperature = self._clamp_temperature(temperature)
        
        request = AIRequest(
            prompt=prompt,
            max_tokens=max_tokens,
            temperature=clamped_temperature
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

