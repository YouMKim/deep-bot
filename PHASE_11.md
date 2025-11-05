# Phase 11: Conversational Chatbot with Memory

[← Back to Implementation Index](./IMPLEMENTATION_INDEX.md)

---

Conversational Chatbot with Multi-Turn Memory

### Learning Objectives
- Understand conversation state management
- Learn multi-turn dialogue handling
- Practice combining RAG with conversation history
- Design natural conversation flow
- Implement conversation memory (short-term and long-term)

### Design Principles
- **Context Awareness**: Remember what was said earlier in conversation
- **Natural Flow**: Respond like a human, not a search engine
- **RAG Integration**: Use chat history knowledge when relevant
- **Graceful Degradation**: Work without RAG if needed

### The Conversation Pipeline

```
User: "What trips have people taken in the past 2 years?"
    ↓
1. Retrieve conversation history (last 5 messages)
2. RAG search for trip-related messages
3. Combine: conversation context + RAG results
4. Generate contextual response
5. Save to conversation memory
    ↓
Bot: "Based on the chat history, I found several trips:
     - Japan trip in March 2023 (discussed by Alice, Bob)
     - Camping in Yosemite, July 2023 (mentioned by Charlie)
     Would you like details about any of these?"
    ↓
User: "Tell me more about the Japan trip"
    ↓
1. Retrieve previous context (knows we're talking about trips)
2. RAG search for "Japan trip March 2023"
3. Generate detailed response
    ↓
Bot: "The Japan trip was organized by Alice in March 2023..."
```

---

## Implementation Steps

### Step 11.1: Create Conversation Memory Service

Create `services/conversation_memory.py`:

```python
"""
Conversation memory for multi-turn dialogues.

Learning: Chatbots need to remember context across messages.
Two types of memory:
1. Short-term: Last N messages in conversation
2. Long-term: Important facts stored in vector DB
"""

from typing import List, Dict, Optional
from datetime import datetime, timedelta
import logging

class ConversationMemory:
    """
    Manage conversation state and history.

    Learning: This is like a human's working memory.
    We remember recent messages and can recall important facts.
    """

    def __init__(self, max_history: int = 10, memory_timeout: int = 3600):
        """
        Args:
            max_history: Maximum messages to keep in memory per conversation
            memory_timeout: Seconds before conversation expires (default 1 hour)
        """
        self.conversations: Dict[str, Dict] = {}
        self.max_history = max_history
        self.memory_timeout = memory_timeout
        self.logger = logging.getLogger(__name__)

    def _get_conversation_key(self, channel_id: str, user_id: str) -> str:
        """Create unique key for conversation"""
        return f"{channel_id}:{user_id}"

    def add_message(
        self,
        channel_id: str,
        user_id: str,
        role: str,  # "user" or "assistant"
        content: str,
        metadata: Dict = None
    ):
        """
        Add message to conversation history.

        Learning: Tracking role (user vs assistant) helps maintain context.
        Metadata can store things like RAG sources used.
        """
        key = self._get_conversation_key(channel_id, user_id)

        if key not in self.conversations:
            self.conversations[key] = {
                "messages": [],
                "created_at": datetime.now(),
                "last_active": datetime.now(),
                "metadata": {}
            }

        conversation = self.conversations[key]

        # Add message
        message = {
            "role": role,
            "content": content,
            "timestamp": datetime.now().isoformat(),
            "metadata": metadata or {}
        }
        conversation["messages"].append(message)
        conversation["last_active"] = datetime.now()

        # Trim if too long
        if len(conversation["messages"]) > self.max_history:
            # Keep system message if present, trim oldest user/assistant messages
            system_messages = [m for m in conversation["messages"] if m["role"] == "system"]
            other_messages = [m for m in conversation["messages"] if m["role"] != "system"]
            conversation["messages"] = system_messages + other_messages[-self.max_history:]

        self.logger.debug(
            f"Added {role} message to conversation {key} "
            f"({len(conversation['messages'])} total messages)"
        )

    def get_history(
        self,
        channel_id: str,
        user_id: str,
        limit: int = None
    ) -> List[Dict]:
        """
        Get conversation history.

        Returns:
            List of messages in chronological order
        """
        key = self._get_conversation_key(channel_id, user_id)

        if key not in self.conversations:
            return []

        # Check if conversation expired
        conversation = self.conversations[key]
        time_since_active = (datetime.now() - conversation["last_active"]).total_seconds()

        if time_since_active > self.memory_timeout:
            self.logger.info(f"Conversation {key} expired, clearing history")
            del self.conversations[key]
            return []

        messages = conversation["messages"]
        if limit:
            messages = messages[-limit:]

        return messages

    def clear_conversation(self, channel_id: str, user_id: str):
        """Clear conversation history"""
        key = self._get_conversation_key(channel_id, user_id)
        if key in self.conversations:
            del self.conversations[key]
            self.logger.info(f"Cleared conversation {key}")

    def get_conversation_summary(self, channel_id: str, user_id: str) -> str:
        """
        Get summary of conversation for context.

        Learning: Instead of sending all history to LLM,
        we can summarize older messages to save tokens.
        """
        history = self.get_history(channel_id, user_id)
        if not history:
            return ""

        # Last 3 messages are kept verbatim
        recent_messages = history[-3:]
        older_messages = history[:-3]

        summary_parts = []

        # Summarize older messages
        if older_messages:
            summary_parts.append("Earlier in conversation:")
            for msg in older_messages:
                role = "User" if msg["role"] == "user" else "Assistant"
                content = msg["content"][:50] + "..." if len(msg["content"]) > 50 else msg["content"]
                summary_parts.append(f"- {role}: {content}")

        # Recent messages in full
        if recent_messages:
            summary_parts.append("\nRecent messages:")
            for msg in recent_messages:
                role = "User" if msg["role"] == "user" else "Assistant"
                summary_parts.append(f"- {role}: {msg['content']}")

        return "\n".join(summary_parts)

    def set_conversation_context(
        self,
        channel_id: str,
        user_id: str,
        context: str
    ):
        """
        Set system context for conversation (like a prompt).

        Example: "You are a helpful assistant that knows about Discord chat history."
        """
        key = self._get_conversation_key(channel_id, user_id)

        if key not in self.conversations:
            self.conversations[key] = {
                "messages": [],
                "created_at": datetime.now(),
                "last_active": datetime.now(),
                "metadata": {}
            }

        # Add system message at the beginning
        self.conversations[key]["messages"].insert(0, {
            "role": "system",
            "content": context,
            "timestamp": datetime.now().isoformat(),
            "metadata": {}
        })
```

### Step 11.2: Create Conversational RAG Service

Create `services/conversational_rag_service.py`:

```python
"""
Conversational RAG: Combines RAG retrieval with conversation memory.

Learning: This is the key to natural chatbot conversations.
We use both RAG (knowledge) and memory (context).
"""

from services.rag_query_service import RAGQueryService
from services.conversation_memory import ConversationMemory
from services.ai_service import AIService
from typing import Dict, Optional
from config import Config
import logging

class ConversationalRAGService:
    """
    RAG + Conversation Memory = Natural Chatbot

    Learning: Multi-turn conversations need:
    1. Short-term memory (what was just said)
    2. Long-term memory (knowledge from RAG)
    3. Ability to combine both seamlessly
    """

    def __init__(
        self,
        rag_service: RAGQueryService,
        conversation_memory: ConversationMemory,
        ai_service: AIService
    ):
        self.rag = rag_service
        self.memory = conversation_memory
        self.ai = ai_service
        self.logger = logging.getLogger(__name__)

    async def chat(
        self,
        user_message: str,
        channel_id: str,
        user_id: str,
        use_rag: bool = True,
        rag_threshold: float = 0.5  # Lower threshold for conversational queries
    ) -> Dict[str, any]:
        """
        Handle conversational message with optional RAG.

        Args:
            user_message: What the user said
            channel_id: Discord channel ID
            user_id: Discord user ID
            use_rag: Whether to use RAG retrieval
            rag_threshold: Similarity threshold for RAG

        Returns:
            {
                "response": "Bot's response",
                "rag_used": True/False,
                "sources": [...] if RAG was used,
                "conversation_length": 5
            }
        """
        # Step 1: Add user message to memory
        self.memory.add_message(
            channel_id=channel_id,
            user_id=user_id,
            role="user",
            content=user_message
        )

        # Step 2: Decide if we need RAG
        # Some questions are conversational, some need knowledge retrieval
        needs_rag = use_rag and self._should_use_rag(user_message)

        rag_context = ""
        rag_sources = []

        if needs_rag:
            # Step 3: Retrieve from RAG
            rag_result = await self.rag.query(
                user_question=user_message,
                min_similarity=rag_threshold
            )

            if rag_result["chunks_used"] > 0:
                # Build RAG context
                rag_context = self._build_rag_context(rag_result)
                rag_sources = rag_result.get("sources", [])
                self.logger.info(
                    f"RAG retrieved {rag_result['chunks_used']} chunks "
                    f"for conversational query"
                )

        # Step 4: Get conversation history
        conversation_summary = self.memory.get_conversation_summary(channel_id, user_id)

        # Step 5: Build prompt combining conversation + RAG
        prompt = self._build_conversational_prompt(
            user_message=user_message,
            conversation_summary=conversation_summary,
            rag_context=rag_context
        )

        # Step 6: Generate response
        response = await self.ai.generate(
            prompt=prompt,
            max_tokens=500,
            temperature=0.7  # Higher for more natural conversation
        )

        # Step 7: Add assistant response to memory
        self.memory.add_message(
            channel_id=channel_id,
            user_id=user_id,
            role="assistant",
            content=response,
            metadata={
                "rag_used": needs_rag and len(rag_sources) > 0,
                "sources_count": len(rag_sources)
            }
        )

        return {
            "response": response.strip(),
            "rag_used": needs_rag and len(rag_sources) > 0,
            "sources": rag_sources,
            "conversation_length": len(self.memory.get_history(channel_id, user_id))
        }

    def _should_use_rag(self, message: str) -> bool:
        """
        Determine if message needs RAG retrieval.

        Learning: Not every message needs knowledge retrieval.
        - "Hello" -> No RAG needed
        - "What trips have people taken?" -> RAG needed
        - "Tell me more about that" -> Use conversation context, maybe RAG

        Simple heuristic: Check for question words or knowledge-seeking phrases.
        """
        knowledge_indicators = [
            "what", "when", "where", "who", "how", "why",
            "tell me", "show me", "find", "list",
            "have people", "did anyone", "has someone"
        ]

        message_lower = message.lower()
        return any(indicator in message_lower for indicator in knowledge_indicators)

    def _build_rag_context(self, rag_result: Dict) -> str:
        """Build context from RAG results"""
        if not rag_result.get("sources"):
            return ""

        context_parts = ["Relevant information from chat history:"]

        for i, source in enumerate(rag_result["sources"][:3], 1):
            context_parts.append(
                f"\n[{i}] {source.get('preview', '')}"
            )

        return "\n".join(context_parts)

    def _build_conversational_prompt(
        self,
        user_message: str,
        conversation_summary: str,
        rag_context: str
    ) -> str:
        """
        Build prompt for conversational response.

        Learning: Good conversational prompts:
        - Include conversation history
        - Include relevant knowledge (RAG)
        - Give personality/tone instructions
        - Encourage natural responses
        """
        prompt_parts = [
            "You are a friendly, helpful chatbot with access to Discord chat history.",
            "You have a conversational, casual tone and can remember the conversation context.",
            ""
        ]

        # Add conversation history if exists
        if conversation_summary:
            prompt_parts.append("Conversation so far:")
            prompt_parts.append(conversation_summary)
            prompt_parts.append("")

        # Add RAG context if available
        if rag_context:
            prompt_parts.append(rag_context)
            prompt_parts.append("")

        # Add current user message
        prompt_parts.append(f"User: {user_message}")
        prompt_parts.append("")
        prompt_parts.append("Assistant (respond naturally and helpfully):")

        return "\n".join(prompt_parts)

    def clear_conversation(self, channel_id: str, user_id: str):
        """Clear conversation history"""
        self.memory.clear_conversation(channel_id, user_id)
```

### Step 11.3: Add Chatbot Commands

Create `cogs/chatbot.py`:

```python
import discord
from discord.ext import commands
from services.conversational_rag_service import ConversationalRAGService
from services.conversation_memory import ConversationMemory
from services.rag_query_service import RAGQueryService
from services.chunked_memory_service import ChunkedMemoryService
from services.vector_store_factory import VectorStoreFactory
from services.embedding_service import EmbeddingServiceFactory
from services.ai_service import AIService
from config import Config
import logging

class Chatbot(commands.Cog):
    """Conversational chatbot with RAG"""

    def __init__(self, bot):
        self.bot = bot
        self.logger = logging.getLogger(__name__)

        # Initialize services
        vector_store = VectorStoreFactory.create()
        embedding_provider = EmbeddingServiceFactory.create()
        chunked_memory = ChunkedMemoryService(vector_store, embedding_provider)
        ai_service = AIService()
        rag_service = RAGQueryService(chunked_memory, ai_service)
        conversation_memory = ConversationMemory()

        self.conversational_rag = ConversationalRAGService(
            rag_service,
            conversation_memory,
            ai_service
        )

    @commands.command(name='chat')
    async def chat(self, ctx, *, message: str):
        """
        Chat with the bot (remembers conversation context).

        Usage:
            !chat What trips have people taken in the past 2 years?
            !chat Tell me more about the Japan trip
            !chat Thanks!

        Learning: This is a natural conversation, not just Q&A.
        """
        try:
            async with ctx.typing():
                result = await self.conversational_rag.chat(
                    user_message=message,
                    channel_id=str(ctx.channel.id),
                    user_id=str(ctx.author.id)
                )

                # Create embed
                embed = discord.Embed(
                    description=result["response"],
                    color=discord.Color.green() if result["rag_used"] else discord.Color.blue()
                )

                # Add footer with metadata
                footer_parts = [f"Conversation length: {result['conversation_length']}"]
                if result["rag_used"]:
                    footer_parts.append(f"📚 Used {len(result['sources'])} sources")

                embed.set_footer(text=" | ".join(footer_parts))

                await ctx.send(embed=embed)

        except Exception as e:
            self.logger.error(f"Error in chat: {e}", exc_info=True)
            await ctx.send(f"❌ Error: {e}")

    @commands.command(name='clear_chat')
    async def clear_chat(self, ctx):
        """Clear conversation history"""
        self.conversational_rag.clear_conversation(
            str(ctx.channel.id),
            str(ctx.author.id)
        )
        await ctx.send("✅ Conversation history cleared!")

    @commands.command(name='chat_history')
    async def chat_history(self, ctx):
        """Show conversation history"""
        history = self.conversational_rag.memory.get_history(
            str(ctx.channel.id),
            str(ctx.author.id)
        )

        if not history:
            await ctx.send("No conversation history.")
            return

        embed = discord.Embed(
            title="💬 Conversation History",
            color=discord.Color.blue()
        )

        for msg in history[-5:]:  # Last 5 messages
            role_emoji = "👤" if msg["role"] == "user" else "🤖"
            content = msg["content"][:100] + "..." if len(msg["content"]) > 100 else msg["content"]
            embed.add_field(
                name=f"{role_emoji} {msg['role'].title()}",
                value=content,
                inline=False
            )

        await ctx.send(embed=embed)

async def setup(bot):
    await bot.add_cog(Chatbot(bot))
```

---

## Usage Examples

### Basic Conversation

```
User: !chat What trips have people taken in the past 2 years?
Bot: Based on the chat history, I found several trips:
     - Japan trip in March 2023 (discussed by Alice, Bob)
     - Camping in Yosemite, July 2023 (Charlie)
     - NYC weekend, Dec 2023 (mentioned by Dave)
     Would you like to know more about any of these?

User: !chat Tell me about the Japan trip
Bot: The Japan trip was organized by Alice in March 2023. They visited
     Tokyo, Kyoto, and Osaka. Bob mentioned the amazing food and temples.
     The trip lasted 2 weeks.

User: !chat Who else was interested in going?
Bot: Based on our conversation and the chat history, Charlie expressed
     interest but couldn't make it due to work commitments.
```

### Without RAG (Pure Conversation)

```
User: !chat Hello!
Bot: Hi there! How can I help you today?

User: !chat How are you?
Bot: I'm doing great, thanks for asking! I'm here to help you with
     questions about the Discord chat history or just chat. What's on
     your mind?
```

---

## Common Pitfalls - Phase 11

1. **Memory leaks**: Conversations stored forever (use timeout)
2. **Token overflow**: Too much history sent to LLM
3. **Context confusion**: Mixing multiple users' conversations
4. **RAG over-use**: Using RAG for simple greetings
5. **No fallback**: What if RAG returns nothing?

## Debugging Tips - Phase 11

- **Log conversation keys**: Verify correct user/channel pairing
- **Monitor memory size**: Check conversation doesn't grow unbounded
- **Test multi-turn**: Verify context is maintained across messages
- **Check RAG triggering**: Log when RAG is/isn't used

## Performance Considerations - Phase 11

- **Memory timeout**: 1 hour is good default (adjust based on usage)
- **History limit**: 10 messages prevents token overflow
- **RAG threshold**: Lower (0.5) for conversational queries
- **Temperature**: 0.7 for natural conversation, 0.3 for factual Q&A

---

## Next Steps

This gives you a **conversational chatbot** that:
- ✅ Remembers context across messages
- ✅ Retrieves knowledge from RAG when needed
- ✅ Responds naturally like a human
- ✅ Handles both knowledge queries and casual chat

Next up: **Phase 12 - User Emulation Mode** 🎭
