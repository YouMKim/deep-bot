from typing import List, Optional
from ai.service import AIService
import logging

class QueryEnhancementService:
    def __init__(self, ai_service: Optional[AIService] = None):
        self.ai_service = ai_service or AIService()
        self.logger = logging.getLogger(__name__)

    async def generate_multi_queries(self, query: str, num_queries: int = 3) -> List[str]:
        """
        Generate multiple variations of a query for improved retrieval.
        """
        prompt = f"""You are a helpful assistant for an LLM system that generates alternative phrasings of questions to improve search results.

        Given the original question, generate {num_queries} alternative ways to ask the same question.
        Each alternative should capture the same intent but use different words or phrasing.

        Original Question: {query}

        Generate {num_queries} alternative questions (one per line, no numbering):"""

        result = await self.ai_service.generate(
            prompt=prompt,
            max_tokens=200,
            temperature=0.7
        )

        variations = [line.strip() for line in result['content'].strip().split('\n') if line.strip()]
        variations = [v.lstrip('0123456789.- ') for v in variations]  # Remove numbering
        variations = variations[:num_queries]

        all_queries = [query] + variations

        self.logger.info(
            f"Generated {len(variations)} query variations for: {query[:50]}..."
        )
        for i, var in enumerate(variations, 1):
            self.logger.info(f"  Variation {i}: {var}")

        return all_queries

    async def generate_hyde_document(self, query: str) -> str:
        """
        Generate a hypothetical answer to the query (HyDE technique).

        Instead of embedding the query, we:
        1. Generate a hypothetical answer
        2. Embed the answer
        3. Search for documents similar to the answer

        This should work ??  because answers are more similar to documents than queries.

        """
        prompt = f"""You are a Discord conversation participant. Generate a hypothetical answer to this question as if you were responding in a Discord channel.

        Make the answer:
        - Conversational and natural (like Discord chat)
        - Specific and detailed
        - 2-3 sentences

        Question: {query}

        Hypothetical Answer:"""

        result = await self.ai_service.generate(
            prompt=prompt,
            max_tokens=150,
            temperature=0.8
        )

        hyde_doc = result['content'].strip()

        self.logger.info(
            f"Generated HyDE document for query: {query[:50]}...\n"
            f"HyDE: {hyde_doc[:100]}..."
        )

        return hyde_doc