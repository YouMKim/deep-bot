from core import AIConfig, AIRequest, create_provider

class AIService:
    # Prompt templates for different summary styles
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

    def __init__(self, provider_name: str = "openai", model: str = None):
        """
        Initialize AI service with a specific provider.

        Args:
            provider_name: Either "openai" or "anthropic"
            model: Specific model to use (optional, uses provider default if not specified)
        """
        config = AIConfig(model_name=provider_name)
        self.provider = create_provider(config)
        self.provider_name = provider_name

        # Override default model if specified
        if model:
            if self.provider.supports_model(model):
                self.provider.default_model = model
            else:
                raise ValueError(f"Model '{model}' not supported by provider '{provider_name}'")

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
    
    async def generate(self, prompt: str, max_tokens: int = 200, temperature: float = 0.7) -> dict:
        """
        Generate a response for any prompt (generic AI completion).

        Args:
            prompt: The prompt text to send to the AI
            max_tokens: Maximum tokens in the response (default: 200)
            temperature: Model temperature 0-2 (default: 0.7)

        Returns:
            Dictionary with response results and metadata
        """
        request = AIRequest(
            prompt=prompt,
            max_tokens=max_tokens,
            temperature=temperature
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

    def get_current_model(self) -> str:
        """Get the currently configured model."""
        return self.provider.get_default_model()

    def get_available_models(self) -> list:
        """Get list of available models for the current provider."""
        return list(self.provider.PRICING_TABLE.keys())

    def switch_model(self, model: str) -> None:
        """
        Switch to a different model within the current provider.

        Args:
            model: The model name to switch to

        Raises:
            ValueError: If the model is not supported by the current provider
        """
        if not self.provider.supports_model(model):
            available = ", ".join(self.get_available_models())
            raise ValueError(
                f"Model '{model}' not supported by provider '{self.provider_name}'. "
                f"Available models: {available}"
            )

        self.provider.default_model = model
