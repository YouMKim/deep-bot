import openai
from config import Config


class AIService:

    def __init__(self):
        self.openai_client = openai.OpenAI(api_key=Config.OPENAI_API_KEY)

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

    async def summarize_with_style(self, text: str, style: str = "generic") -> dict:
        prompt_config = self.PROMPT_STYLES[style]
        response = self.openai_client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {"role": "system", "content": prompt_config["system"]},
                {
                    "role": "user",
                    "content": prompt_config["user_template"].format(text=text),
                },
            ],
            max_tokens=self._get_max_tokens_for_style(style),
        )

        usage = response.usage
        cost = self.calculate_cost(
            response.model, usage.prompt_tokens, usage.completion_tokens
        )
        return {
            "summary": response.choices[0].message.content,
            "style": style,
            "tokens_prompt": usage.prompt_tokens,
            "tokens_completion": usage.completion_tokens,
            "tokens_total": usage.total_tokens,
            "model": response.model,
            "cost": cost,
        }

    def calculate_cost(
        self, model: str, prompt_tokens: int, completion_tokens: int
    ) -> float:
        # cost per 1K tokens
        cost_table = {
            "gpt-4o-mini": {"prompt": 0.00015, "completion": 0.00060},
            "gpt-4o-mini-2024-07-18": {"prompt": 0.00015, "completion": 0.00060},
            "gpt-4o": {"prompt": 0.005, "completion": 0.015},
            # add more models if needed
        }

        if model not in cost_table:
            raise ValueError(f"Unknown model: {model}")

        rates = cost_table[model]
        total = (prompt_tokens / 1000) * rates["prompt"] + (
            completion_tokens / 1000
        ) * rates["completion"]
        return total

    def _get_max_tokens_for_style(self, style: str) -> int:
        """Set appropriate max tokens for each style."""
        token_limits = {
            "generic": 300,  # 2-4 sentences
            "bullet_points": 400,  # 3-7 bullet points
            "headline": 100,  # 1-2 sentences
        }
        return token_limits.get(style, 300)

    async def compare_all_styles(self, text: str) -> dict:
        results = {}
        for style in self.PROMPT_STYLES.keys():
            results[style] = await self.summarize_with_style(text, style)
        return results
