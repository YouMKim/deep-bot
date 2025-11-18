# Chatbot Feature Implementation Plan

**Project:** Deep Bot - Conversational RAG Chatbot
**Date:** 2025-11-18
**Status:** Planning Phase

---

## Executive Summary

**Goal:** Implement a conversational chatbot that uses RAG over Discord history to have contextual, multi-turn conversations with users.

**Key Difference from Current System:**
- **Current:** One-shot Q&A (`!ask` → single answer)
- **Proposed:** Multi-turn conversations (`!chat` → ongoing dialogue with memory)

**Benefits:**
- Natural conversation flow
- Follow-up questions without repeating context
- Personalized responses based on Discord history
- More engaging user experience

**Estimated Effort:** 3-4 weeks (60-80 hours)

---

## Table of Contents

1. [Feature Overview](#1-feature-overview)
2. [Architecture Design](#2-architecture-design)
3. [Conversation Management](#3-conversation-management)
4. [RAG Integration](#4-rag-integration)
5. [User Experience Design](#5-user-experience-design)
6. [Implementation Phases](#6-implementation-phases)
7. [Technical Specifications](#7-technical-specifications)
8. [Advanced Features](#8-advanced-features)
9. [Performance Considerations](#9-performance-considerations)
10. [Security & Safety](#10-security--safety)

---

## 1. Feature Overview

### 1.1 Current System (Q&A Mode)

```
User: !ask What database did we choose?
Bot: We decided on PostgreSQL. John mentioned it has better JSON support.
[END]
```

**Limitations:**
- No memory of previous questions
- Cannot ask follow-up questions
- Each query is isolated
- No conversational flow

### 1.2 Proposed System (Chatbot Mode)

```
User: !chat start
Bot: 👋 Hey! I'm ready to chat. I can answer questions about our Discord history. What would you like to know?

User: What database did we choose?
Bot: We decided on PostgreSQL. John mentioned it has better JSON support and Alice confirmed it integrates well with our ORM.

User: Why did we pick it over MySQL?
Bot: [Remembers we're talking about PostgreSQL] Based on the discussion, the main reasons were:
1. Better JSON support for our document storage needs
2. Superior query optimizer for complex joins
3. Alice had experience with it from previous projects

User: When was this decision made?
Bot: [Still remembers context] According to the messages, this was decided on October 15th, 2024, after a 3-day discussion.

User: !chat end
Bot: 👋 Chat session ended. We discussed database selection (PostgreSQL). Feel free to start a new chat anytime!
```

**Advantages:**
- Maintains conversation context
- Understands follow-up questions ("it", "this", "why")
- More natural interaction
- Can clarify and expand on answers

---

## 2. Architecture Design

### 2.1 High-Level Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                        Discord User                              │
└───────────────────────────┬─────────────────────────────────────┘
                            │ !chat <message>
                            ▼
┌─────────────────────────────────────────────────────────────────┐
│                      ChatBot Cog                                 │
│  - Session management                                            │
│  - Command routing (!chat start/end/clear)                       │
│  - User message handling                                         │
└───────────────────────────┬─────────────────────────────────────┘
                            │
                            ▼
┌─────────────────────────────────────────────────────────────────┐
│                  ConversationManager                             │
│  - Track active sessions per user                                │
│  - Maintain conversation history                                 │
│  - Context window management                                     │
│  - Session persistence (optional)                                │
└───────────────────────────┬─────────────────────────────────────┘
                            │
                            ▼
┌─────────────────────────────────────────────────────────────────┐
│               ConversationalRAGPipeline                          │
│  ┌─────────────────────────────────────────────────────────┐   │
│  │ 1. Query Understanding                                   │   │
│  │    - Extract intent from message                         │   │
│  │    - Resolve references (it, that, this)                 │   │
│  │    - Incorporate conversation history                    │   │
│  └─────────────────────────────────────────────────────────┘   │
│  ┌─────────────────────────────────────────────────────────┐   │
│  │ 2. Contextual Retrieval                                  │   │
│  │    - Enhanced query with conversation context            │   │
│  │    - RAG search over Discord history                     │   │
│  │    - Hybrid search (BM25 + vector)                       │   │
│  └─────────────────────────────────────────────────────────┘   │
│  ┌─────────────────────────────────────────────────────────┐   │
│  │ 3. Conversational Response Generation                    │   │
│  │    - Combine: user query + conv history + RAG context    │   │
│  │    - Generate natural response                            │   │
│  │    - Track response in conversation history              │   │
│  └─────────────────────────────────────────────────────────┘   │
└───────────────────────────┬─────────────────────────────────────┘
                            │
                            ▼
┌─────────────────────────────────────────────────────────────────┐
│                     Existing RAG Pipeline                        │
│  - ChunkedMemoryService                                          │
│  - Hybrid search                                                 │
│  - Re-ranking                                                    │
│  - AIService (OpenAI/Anthropic)                                  │
└─────────────────────────────────────────────────────────────────┘
```

### 2.2 Key Components

#### Component 1: ConversationSession

```python
@dataclass
class Message:
    """Single message in conversation."""
    role: str  # 'user' or 'assistant'
    content: str
    timestamp: datetime
    metadata: Dict[str, Any]  # RAG sources, costs, etc.

@dataclass
class ConversationSession:
    """Represents an active chat session."""
    session_id: str
    user_id: int
    channel_id: int
    messages: List[Message]
    created_at: datetime
    last_active: datetime
    context_mode: str  # 'discord_history', 'general', 'hybrid'
    total_cost: float
    metadata: Dict[str, Any]
```

#### Component 2: ConversationManager

```python
class ConversationManager:
    """Manages multiple conversation sessions."""

    def __init__(self, max_sessions: int = 100):
        self.active_sessions: Dict[str, ConversationSession] = {}
        self.max_sessions = max_sessions

    def create_session(self, user_id: int, channel_id: int) -> ConversationSession
    def get_session(self, user_id: int, channel_id: int) -> Optional[ConversationSession]
    def add_message(self, session: ConversationSession, message: Message)
    def end_session(self, user_id: int, channel_id: int) -> ConversationSession
    def cleanup_inactive_sessions(self, max_age_minutes: int = 60)
    def get_conversation_context(self, session: ConversationSession, max_tokens: int = 2000) -> str
```

#### Component 3: ConversationalRAGPipeline

```python
class ConversationalRAGPipeline:
    """RAG pipeline with conversation awareness."""

    def __init__(self, rag_pipeline: RAGPipeline):
        self.rag_pipeline = rag_pipeline
        self.query_enhancer = ConversationalQueryEnhancer()

    async def answer_with_context(
        self,
        query: str,
        conversation_history: List[Message],
        config: RAGConfig
    ) -> ConversationalRAGResult:
        """
        Answer query using both conversation history and Discord RAG.

        Steps:
        1. Enhance query with conversation context
        2. Retrieve from Discord history via RAG
        3. Generate response with both contexts
        4. Return structured result
        """
        pass
```

---

## 3. Conversation Management

### 3.1 Session Lifecycle

```
START → ACTIVE → IDLE → TIMEOUT/END
  │       │        │         │
  │       │        └─────────┴──→ ARCHIVED
  │       │
  │       └─ Messages added continuously
  │
  └─ Initialize with system prompt
```

**States:**
- **START:** User types `!chat start`, session created
- **ACTIVE:** User is actively chatting (< 5 min since last message)
- **IDLE:** No activity for 5-30 minutes
- **TIMEOUT:** Auto-end after 30 minutes of inactivity
- **END:** User types `!chat end` or timeout
- **ARCHIVED:** Optionally saved to database for future reference

### 3.2 Context Window Management

**Challenge:** LLMs have token limits (e.g., GPT-4o-mini: 128k input, but costs increase)

**Strategy: Sliding Window with Summarization**

```python
class ContextWindowManager:
    """Manages conversation context within token limits."""

    MAX_CONTEXT_TOKENS = 4000  # Configurable

    def get_context(self, messages: List[Message]) -> str:
        """
        Get conversation context within token limit.

        Strategy:
        1. Always include last N messages (most recent)
        2. Summarize older messages if needed
        3. Include system prompt

        Layout:
        [System Prompt] + [Summarized Old Context] + [Recent Messages]
        """

        # Reserve tokens
        system_tokens = 200
        recent_tokens = 2000
        summary_tokens = 1800

        # Get recent messages (last 5-10 messages)
        recent_messages = messages[-10:]
        recent_text = self._format_messages(recent_messages)

        # If conversation is long, summarize older messages
        if len(messages) > 10:
            old_messages = messages[:-10]
            summary = await self._summarize_conversation(old_messages)
            context = f"[Previous conversation summary: {summary}]\n\n{recent_text}"
        else:
            context = recent_text

        return context
```

### 3.3 Reference Resolution

**Problem:** Users use pronouns and references:
- "What about **it**?"
- "Tell me more about **that**"
- "Why did **they** choose **this**?"

**Solution: Coreference Resolution**

```python
class ConversationalQueryEnhancer:
    """Enhances queries with conversation context."""

    async def resolve_references(
        self,
        query: str,
        conversation_history: List[Message]
    ) -> str:
        """
        Resolve pronouns and references in query.

        Example:
        History:
          User: "What database did we choose?"
          Bot: "We chose PostgreSQL."
          User: "Why did we pick it?"

        Resolution:
          "Why did we pick it?" → "Why did we pick PostgreSQL?"
        """

        # Use LLM to rewrite query with context
        prompt = f"""Given the conversation history, rewrite the user's question to be standalone (replace pronouns and references with specific nouns).

Conversation History:
{self._format_history(conversation_history[-5:])}

User's Question: {query}

Rewritten Question (standalone, no pronouns):"""

        result = await self.ai_service.generate(prompt, max_tokens=100, temperature=0.3)

        enhanced_query = result['content'].strip()

        return enhanced_query
```

---

## 4. RAG Integration

### 4.1 Two-Source RAG

The chatbot uses **two sources of information**:

1. **Conversation History** (short-term memory)
   - Recent messages in current chat session
   - Helps maintain context and coherence

2. **Discord History** (long-term memory)
   - Retrieved via existing RAG pipeline
   - Provides factual information

```python
async def generate_response(
    self,
    query: str,
    conversation_history: List[Message],
    config: RAGConfig
) -> str:
    """Generate response using both conversation and Discord history."""

    # 1. Enhance query with conversation context
    enhanced_query = await self.query_enhancer.resolve_references(
        query, conversation_history
    )

    # 2. Retrieve from Discord history
    rag_result = await self.rag_pipeline.answer_question(
        enhanced_query, config
    )

    # 3. Build combined context
    conversation_context = self._format_conversation(conversation_history)
    discord_context = self._format_rag_sources(rag_result.sources)

    # 4. Generate response
    prompt = self._build_chatbot_prompt(
        query=query,
        conversation_context=conversation_context,
        discord_context=discord_context
    )

    response = await self.ai_service.generate(prompt, temperature=0.7)

    return response['content']
```

### 4.2 Chatbot System Prompt

```python
CHATBOT_SYSTEM_PROMPT = """You are a helpful Discord chatbot assistant with access to this server's conversation history.

Your capabilities:
- Answer questions about past Discord conversations
- Provide context from message history
- Have natural, flowing conversations
- Ask clarifying questions when needed

Guidelines:
1. Use conversation history to understand context and references
2. Use Discord message history to answer factual questions
3. If you don't know something, say so - don't make up information
4. Be conversational and friendly, not robotic
5. Reference specific messages/users when relevant
6. If a question is ambiguous, ask for clarification

Conversation style:
- Natural and engaging
- Use "we" when referring to the Discord community
- Acknowledge when switching topics
- Provide sources when citing Discord history
"""
```

### 4.3 Adaptive Retrieval

**Smart RAG:** Only retrieve from Discord history when needed.

```python
class AdaptiveRAGStrategy:
    """Decides when to use RAG retrieval."""

    async def should_retrieve(self, query: str, conversation: List[Message]) -> bool:
        """
        Determine if Discord history retrieval is needed.

        Retrieve if:
        - Query asks about past events/conversations
        - Query mentions specific topics/people
        - Query requires factual information

        Don't retrieve if:
        - Query is about current conversation only
        - Query is general chitchat
        - Query is a clarification request
        """

        # Use lightweight classifier
        classifier_prompt = f"""Does this question require looking up past Discord messages?

Question: {query}

Answer YES or NO:"""

        result = await self.ai_service.generate(
            classifier_prompt,
            max_tokens=5,
            temperature=0.0
        )

        return 'yes' in result['content'].lower()
```

---

## 5. User Experience Design

### 5.1 Commands

| Command | Description | Example |
|---------|-------------|---------|
| `!chat start` | Start new chat session | `!chat start` |
| `!chat <message>` | Send message in active session | `!chat What database did we use?` |
| `!chat end` | End current session | `!chat end` |
| `!chat clear` | Clear conversation history but keep session | `!chat clear` |
| `!chat history` | Show conversation summary | `!chat history` |
| `!chat export` | Export conversation as text/JSON | `!chat export` |

### 5.2 Visual Design (Discord Embeds)

**Session Start:**
```
┌─────────────────────────────────────────┐
│ 💬 Chat Session Started                 │
├─────────────────────────────────────────┤
│ I'm ready to chat! I can answer         │
│ questions about our Discord history.    │
│                                          │
│ Tips:                                    │
│ • Ask follow-up questions naturally     │
│ • Use !chat end to close session        │
│ • Use !chat clear to reset context      │
│                                          │
│ Session ID: abc123                       │
└─────────────────────────────────────────┘
```

**Response:**
```
┌─────────────────────────────────────────┐
│ 🤖 Deep Bot                              │
├─────────────────────────────────────────┤
│ We chose PostgreSQL for the database.   │
│ John mentioned it has better JSON       │
│ support, and Alice confirmed it         │
│ integrates well with our ORM.           │
│                                          │
│ This was decided on Oct 15, 2024.       │
├─────────────────────────────────────────┤
│ 📚 Sources: 3 messages                   │
│ 💰 Cost: $0.0023                         │
│ ⏱️ Response time: 1.2s                  │
└─────────────────────────────────────────┘
```

**Session End:**
```
┌─────────────────────────────────────────┐
│ 👋 Chat Session Ended                    │
├─────────────────────────────────────────┤
│ Summary:                                 │
│ • Messages: 8 (4 from you, 4 from me)   │
│ • Topics: Database selection, schemas   │
│ • Duration: 5 minutes                    │
│ • Total cost: $0.0156                    │
│                                          │
│ Thanks for chatting! 😊                  │
└─────────────────────────────────────────┘
```

### 5.3 Typing Indicators

```python
async def respond_to_message(self, ctx, message: str):
    """Respond to chat message with typing indicator."""

    async with ctx.typing():  # Shows "Bot is typing..."
        # Generate response (may take 2-5 seconds)
        response = await self.pipeline.answer_with_context(...)

    await ctx.send(embed=response_embed)
```

### 5.4 Session Limits

To prevent abuse:

```python
# config.py
CHAT_MAX_SESSIONS_PER_USER = 1  # One session at a time
CHAT_MAX_MESSAGE_LENGTH = 500  # Characters per message
CHAT_MAX_MESSAGES_PER_SESSION = 50  # Auto-end after 50 messages
CHAT_SESSION_TIMEOUT = 30  # Minutes
CHAT_RATE_LIMIT = 10  # Messages per minute
```

---

## 6. Implementation Phases

### Phase 1: Basic Chatbot (Week 1) ⭐ MVP

**Goal:** Simple multi-turn conversations without RAG

**Tasks:**
1. Create `ConversationSession` data model
2. Implement `ConversationManager` for session tracking
3. Add basic `!chat start/end` commands
4. Implement conversation history tracking
5. Generate responses using conversation context only

**Deliverables:**
- Users can start/end chat sessions
- Bot remembers conversation within session
- Basic conversational responses

**Code:**
```python
# bot/cogs/chatbot.py
class ChatBot(commands.Cog):
    def __init__(self, bot):
        self.bot = bot
        self.conversation_manager = ConversationManager()

    @commands.command(name='chat')
    async def chat(self, ctx, *, message: str = None):
        if message is None or message.lower() == 'start':
            # Start session
            session = self.conversation_manager.create_session(
                ctx.author.id, ctx.channel.id
            )
            await ctx.send(embed=self._create_start_embed(session))

        elif message.lower() == 'end':
            # End session
            session = self.conversation_manager.end_session(
                ctx.author.id, ctx.channel.id
            )
            await ctx.send(embed=self._create_end_embed(session))

        else:
            # Handle message
            session = self.conversation_manager.get_session(
                ctx.author.id, ctx.channel.id
            )
            if not session:
                await ctx.send("No active chat session. Use `!chat start` first.")
                return

            response = await self._generate_response(message, session)
            await ctx.send(embed=response)
```

**Testing:**
```
User: !chat start
Bot: Chat session started!

User: !chat My name is Alice
Bot: Nice to meet you, Alice!

User: !chat What's my name?
Bot: Your name is Alice!

User: !chat end
Bot: Chat session ended. We discussed your name (Alice).
```

---

### Phase 2: RAG Integration (Week 2) 🔍

**Goal:** Add Discord history retrieval to conversations

**Tasks:**
1. Create `ConversationalRAGPipeline` that wraps existing `RAGPipeline`
2. Implement query enhancement with conversation context
3. Add reference resolution (pronouns → entities)
4. Combine conversation context + Discord RAG context
5. Update prompt engineering for conversational style

**Deliverables:**
- Bot can answer questions using Discord history
- Bot understands follow-up questions with pronouns
- Natural conversation flow with factual grounding

**Code:**
```python
# rag/conversational_pipeline.py
class ConversationalRAGPipeline:
    def __init__(self, rag_pipeline: RAGPipeline, ai_service: AIService):
        self.rag_pipeline = rag_pipeline
        self.ai_service = ai_service

    async def answer_with_context(
        self,
        query: str,
        conversation_history: List[Message],
        config: RAGConfig
    ) -> ConversationalRAGResult:
        # 1. Enhance query with conversation context
        enhanced_query = await self._enhance_query(query, conversation_history)

        # 2. Retrieve from Discord history
        rag_result = await self.rag_pipeline.answer_question(
            enhanced_query, config
        )

        # 3. Generate conversational response
        response = await self._generate_conversational_response(
            query=query,
            conversation_history=conversation_history,
            rag_result=rag_result
        )

        return ConversationalRAGResult(
            response=response,
            sources=rag_result.sources,
            conversation_context=conversation_history,
            enhanced_query=enhanced_query,
            cost=rag_result.cost
        )
```

**Testing:**
```
User: !chat start
Bot: Ready to chat!

User: !chat What database did we choose?
Bot: We chose PostgreSQL. [retrieves from Discord history]

User: !chat Why did we pick it over MySQL?
Bot: [understands "it" = PostgreSQL] The main reasons were...

User: !chat When was this decided?
Bot: [understands "this" = PostgreSQL decision] On October 15, 2024.
```

---

### Phase 3: Advanced Features (Week 3) 🚀

**Goal:** Enhance UX and add intelligent features

**Tasks:**
1. Implement adaptive RAG (only retrieve when needed)
2. Add conversation summarization for long sessions
3. Implement session persistence (save/load from DB)
4. Add conversation export functionality
5. Create `!chat history` command to view session
6. Add intelligent topic detection and transitions

**Deliverables:**
- Efficient RAG usage (not every message)
- Long conversations stay within token limits
- Users can export conversations
- Better handling of topic changes

**Code:**
```python
class AdaptiveConversationalPipeline:
    async def process_message(
        self,
        message: str,
        session: ConversationSession
    ) -> str:
        # 1. Classify intent
        needs_rag = await self._needs_discord_history(message, session)

        if needs_rag:
            # Use full RAG pipeline
            response = await self.rag_pipeline.answer_with_context(...)
        else:
            # Use conversation-only context
            response = await self._generate_simple_response(...)

        return response

    async def _needs_discord_history(
        self,
        message: str,
        session: ConversationSession
    ) -> bool:
        """Determine if Discord RAG retrieval is needed."""

        # Keywords that suggest historical queries
        historical_keywords = [
            'what did', 'when did', 'who said', 'previous',
            'earlier', 'discussed', 'mentioned', 'talked about'
        ]

        message_lower = message.lower()

        # Check for historical keywords
        if any(keyword in message_lower for keyword in historical_keywords):
            return True

        # Check if asking about specific topics (use LLM classifier)
        # ...

        return False
```

---

### Phase 4: Production Ready (Week 4) 💎

**Goal:** Polish, optimize, and deploy

**Tasks:**
1. Add comprehensive error handling
2. Implement session cleanup and garbage collection
3. Add monitoring and logging
4. Create admin commands for session management
5. Add user feedback mechanism (👍/👎 reactions)
6. Performance optimization (caching, batching)
7. Write comprehensive tests
8. Create user documentation

**Deliverables:**
- Production-ready chatbot
- Admin tools for monitoring
- Full test coverage
- User documentation

---

## 7. Technical Specifications

### 7.1 Database Schema (Optional Persistence)

```sql
-- Conversation Sessions
CREATE TABLE conversation_sessions (
    session_id TEXT PRIMARY KEY,
    user_id INTEGER NOT NULL,
    channel_id INTEGER NOT NULL,
    created_at TIMESTAMP NOT NULL,
    ended_at TIMESTAMP,
    message_count INTEGER DEFAULT 0,
    total_cost REAL DEFAULT 0.0,
    metadata JSON
);

-- Conversation Messages
CREATE TABLE conversation_messages (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    session_id TEXT NOT NULL,
    role TEXT NOT NULL,  -- 'user' or 'assistant'
    content TEXT NOT NULL,
    timestamp TIMESTAMP NOT NULL,
    metadata JSON,  -- RAG sources, costs, etc.
    FOREIGN KEY (session_id) REFERENCES conversation_sessions(session_id)
);

-- Indexes
CREATE INDEX idx_sessions_user ON conversation_sessions(user_id);
CREATE INDEX idx_sessions_created ON conversation_sessions(created_at);
CREATE INDEX idx_messages_session ON conversation_messages(session_id);
```

### 7.2 API Design

```python
# rag/models.py

@dataclass
class ConversationalRAGResult:
    """Result from conversational RAG pipeline."""
    response: str
    sources: List[Dict]  # RAG sources from Discord history
    conversation_context: List[Message]  # Recent conversation
    enhanced_query: str  # Query after reference resolution
    cost: float
    tokens_used: int
    retrieval_used: bool  # Whether RAG retrieval was used
    metadata: Dict[str, Any]

@dataclass
class ConversationConfig:
    """Configuration for conversation settings."""
    max_context_tokens: int = 4000
    max_messages_per_session: int = 50
    session_timeout_minutes: int = 30
    enable_adaptive_rag: bool = True
    enable_summarization: bool = True
    temperature: float = 0.7
    rag_config: RAGConfig = field(default_factory=RAGConfig)
```

### 7.3 Error Handling

```python
class ChatBotError(Exception):
    """Base exception for chatbot errors."""
    pass

class SessionNotFoundError(ChatBotError):
    """Raised when session doesn't exist."""
    pass

class SessionLimitExceededError(ChatBotError):
    """Raised when user has too many active sessions."""
    pass

class MessageTooLongError(ChatBotError):
    """Raised when message exceeds length limit."""
    pass

# Usage in cog:
try:
    response = await self.pipeline.answer_with_context(...)
except SessionNotFoundError:
    await ctx.send("❌ No active chat session. Use `!chat start` first.")
except MessageTooLongError as e:
    await ctx.send(f"❌ Message too long (max {e.max_length} characters)")
except Exception as e:
    logger.error(f"Chat error: {e}", exc_info=True)
    await ctx.send("❌ Something went wrong. Please try again.")
```

---

## 8. Advanced Features

### 8.1 Multi-Modal Conversations

Support for images, links, and reactions:

```python
class MultiModalChatBot:
    async def handle_message(self, ctx):
        message_content = ctx.message.content
        attachments = ctx.message.attachments

        # Handle images
        if attachments:
            images = [att for att in attachments if att.content_type.startswith('image/')]
            if images:
                # Use GPT-4 Vision or Claude for image understanding
                response = await self._process_with_image(message_content, images)

        # Handle links
        urls = self._extract_urls(message_content)
        if urls:
            # Fetch and summarize linked content
            response = await self._process_with_urls(message_content, urls)
```

### 8.2 Personality Customization

Allow users to choose chatbot personality:

```python
PERSONALITIES = {
    'professional': {
        'system_prompt': 'You are a professional assistant...',
        'temperature': 0.5,
        'style': 'formal'
    },
    'friendly': {
        'system_prompt': 'You are a friendly and casual assistant...',
        'temperature': 0.8,
        'style': 'casual'
    },
    'concise': {
        'system_prompt': 'You provide brief, concise answers...',
        'temperature': 0.3,
        'style': 'minimal'
    }
}

# Command:
@commands.command(name='chat_personality')
async def set_personality(self, ctx, personality: str):
    if personality not in PERSONALITIES:
        await ctx.send(f"Available: {', '.join(PERSONALITIES.keys())}")
        return

    session = self.get_session(ctx.author.id, ctx.channel.id)
    session.personality = personality
    await ctx.send(f"✅ Personality set to: {personality}")
```

### 8.3 Conversation Branching

Support for exploring alternative conversation paths:

```python
# Commands:
!chat rewind 3     # Go back 3 messages
!chat branch       # Create alternate conversation branch
!chat branches     # List all branches
!chat switch 2     # Switch to branch 2
```

### 8.4 Collaborative Chatting

Multiple users in same channel sharing a session:

```python
@commands.command(name='chat_group')
async def start_group_chat(self, ctx):
    """Start a group chat session for the channel."""
    session = self.conversation_manager.create_group_session(
        channel_id=ctx.channel.id,
        creator_id=ctx.author.id
    )

    await ctx.send(
        f"🎉 Group chat started! Anyone in this channel can participate.\n"
        f"Use `!chat <message>` to chat."
    )
```

### 8.5 Conversation Search

Search through past conversations:

```python
@commands.command(name='chat_search')
async def search_conversations(self, ctx, *, query: str):
    """Search your past conversations."""

    results = await self.conversation_db.search(
        user_id=ctx.author.id,
        query=query,
        limit=5
    )

    embed = discord.Embed(title=f"🔍 Search Results for: {query}")

    for result in results:
        embed.add_field(
            name=f"Session {result.session_id[:8]} - {result.created_at}",
            value=result.matched_text[:200],
            inline=False
        )

    await ctx.send(embed=embed)
```

---

## 9. Performance Considerations

### 9.1 Caching Strategies

```python
class ConversationCache:
    """Cache for conversation-related data."""

    def __init__(self):
        self.query_cache = {}  # Cache enhanced queries
        self.rag_cache = {}    # Cache RAG results
        self.summary_cache = {} # Cache conversation summaries

    def cache_enhanced_query(self, original: str, enhanced: str):
        """Cache query enhancement to avoid repeated LLM calls."""
        key = hashlib.md5(original.encode()).hexdigest()
        self.query_cache[key] = {
            'enhanced': enhanced,
            'timestamp': time.time()
        }

    def get_cached_query(self, original: str, max_age: int = 300) -> Optional[str]:
        """Get cached enhanced query if fresh."""
        key = hashlib.md5(original.encode()).hexdigest()
        if key in self.query_cache:
            cache_entry = self.query_cache[key]
            if time.time() - cache_entry['timestamp'] < max_age:
                return cache_entry['enhanced']
        return None
```

### 9.2 Async Optimization

```python
async def generate_response_optimized(self, query: str, session: ConversationSession):
    """Optimized response generation with parallel processing."""

    # Run these in parallel:
    tasks = [
        self._enhance_query(query, session.messages),  # Task 1
        self._detect_topic(query),                      # Task 2
        self._check_needs_rag(query)                    # Task 3
    ]

    enhanced_query, topic, needs_rag = await asyncio.gather(*tasks)

    if needs_rag:
        # Retrieve from Discord history
        rag_result = await self.rag_pipeline.answer_question(enhanced_query)
    else:
        rag_result = None

    # Generate response
    response = await self._generate(query, session, rag_result)

    return response
```

### 9.3 Rate Limiting

```python
# Separate rate limits for chat commands
@commands.command(name='chat')
@cooldown(rate=10, per=60, type=BucketType.user)  # 10 messages per minute
async def chat(self, ctx, *, message: str):
    """Chat with the bot."""
    pass

# Higher limit for premium users
def get_chat_rate_limit(user_id: int) -> int:
    """Dynamic rate limits based on user tier."""
    if user_id in PREMIUM_USERS:
        return 20  # 20 messages per minute
    elif user_id in TRUSTED_USERS:
        return 15
    else:
        return 10  # Default
```

---

## 10. Security & Safety

### 10.1 Content Filtering

```python
class ContentModerator:
    """Moderate conversation content for safety."""

    FORBIDDEN_TOPICS = [
        'personal information', 'credentials', 'passwords',
        'private messages', 'DMs'
    ]

    async def check_message(self, message: str) -> Tuple[bool, Optional[str]]:
        """
        Check if message is safe.

        Returns:
            (is_safe, reason)
        """

        # Check for forbidden topics
        for topic in self.FORBIDDEN_TOPICS:
            if topic.lower() in message.lower():
                return False, f"Cannot discuss {topic}"

        # Check message length
        if len(message) > 500:
            return False, "Message too long (max 500 characters)"

        # Use OpenAI Moderation API
        moderation = await self._check_openai_moderation(message)
        if moderation['flagged']:
            return False, "Content violates usage policies"

        return True, None
```

### 10.2 Privacy Protection

```python
class PrivacyFilter:
    """Filter out private information from responses."""

    async def sanitize_response(self, response: str) -> str:
        """Remove private information from response."""

        # Redact email addresses
        response = re.sub(
            r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b',
            '[EMAIL]',
            response
        )

        # Redact phone numbers
        response = re.sub(
            r'\b\d{3}[-.]?\d{3}[-.]?\d{4}\b',
            '[PHONE]',
            response
        )

        # Redact API keys (common patterns)
        response = re.sub(
            r'\b[A-Za-z0-9]{32,}\b',  # Long alphanumeric strings
            '[REDACTED]',
            response
        )

        return response
```

### 10.3 User Consent

```python
@commands.command(name='chat_start')
async def chat_start(self, ctx):
    """Start chat with consent notice."""

    consent_embed = discord.Embed(
        title="💬 Start Chat Session?",
        description=(
            "By starting a chat session, you agree that:\n\n"
            "✓ Your messages will be processed by AI\n"
            "✓ Conversation may be temporarily stored\n"
            "✓ Bot has access to server message history\n"
            "✓ Conversations may be used to improve the bot\n\n"
            "React with ✅ to start, ❌ to cancel."
        ),
        color=discord.Color.blue()
    )

    message = await ctx.send(embed=consent_embed)
    await message.add_reaction("✅")
    await message.add_reaction("❌")

    # Wait for reaction
    def check(reaction, user):
        return user == ctx.author and str(reaction.emoji) in ["✅", "❌"]

    try:
        reaction, user = await self.bot.wait_for(
            'reaction_add',
            timeout=30.0,
            check=check
        )

        if str(reaction.emoji) == "✅":
            # Start session
            session = self.conversation_manager.create_session(...)
            await ctx.send("✅ Chat session started!")
        else:
            await ctx.send("❌ Chat cancelled.")

    except asyncio.TimeoutError:
        await ctx.send("⏱️ Consent timeout. Please try again.")
```

---

## 11. Success Metrics

### 11.1 Key Performance Indicators (KPIs)

**Engagement Metrics:**
- Average session length (messages)
- Sessions per user per week
- Session completion rate
- Return user rate

**Quality Metrics:**
- User satisfaction (👍/👎 reactions)
- Average response time
- RAG retrieval accuracy
- Query success rate

**Cost Metrics:**
- Cost per session
- Cost per message
- Token usage efficiency
- RAG retrieval rate

### 11.2 Monitoring Dashboard

```python
class ChatBotMetrics:
    """Track chatbot performance metrics."""

    def __init__(self):
        self.metrics = {
            'sessions_created': 0,
            'sessions_completed': 0,
            'messages_sent': 0,
            'total_cost': 0.0,
            'total_tokens': 0,
            'rag_retrievals': 0,
            'errors': 0,
            'user_satisfaction': []  # List of 👍/👎
        }

    def log_session_start(self, session: ConversationSession):
        self.metrics['sessions_created'] += 1

    def log_session_end(self, session: ConversationSession):
        self.metrics['sessions_completed'] += 1
        self.metrics['total_cost'] += session.total_cost

    def get_summary(self) -> Dict[str, Any]:
        """Get metrics summary."""
        return {
            'total_sessions': self.metrics['sessions_created'],
            'completion_rate': self.metrics['sessions_completed'] / max(1, self.metrics['sessions_created']),
            'avg_cost_per_session': self.metrics['total_cost'] / max(1, self.metrics['sessions_completed']),
            'messages_per_session': self.metrics['messages_sent'] / max(1, self.metrics['sessions_created']),
            'satisfaction_score': sum(self.metrics['user_satisfaction']) / max(1, len(self.metrics['user_satisfaction']))
        }
```

---

## 12. Timeline & Milestones

### Week 1: Foundation
- [ ] Day 1-2: Design architecture, data models
- [ ] Day 3-4: Implement ConversationManager
- [ ] Day 5: Create basic chat commands
- [ ] Day 6-7: Testing & bug fixes
- **Milestone:** Basic chatbot working

### Week 2: RAG Integration
- [ ] Day 8-9: Implement ConversationalRAGPipeline
- [ ] Day 10-11: Add query enhancement & reference resolution
- [ ] Day 12-13: Integrate with existing RAG system
- [ ] Day 14: Testing RAG accuracy
- **Milestone:** Chatbot answers questions from Discord history

### Week 3: Advanced Features
- [ ] Day 15-16: Implement adaptive RAG
- [ ] Day 17-18: Add conversation summarization
- [ ] Day 19-20: Session persistence & export
- [ ] Day 21: Polish UX
- **Milestone:** Production-ready features

### Week 4: Polish & Deploy
- [ ] Day 22-23: Comprehensive testing
- [ ] Day 24-25: Performance optimization
- [ ] Day 26-27: Documentation
- [ ] Day 28: Deploy to production
- **Milestone:** Live chatbot feature

---

## 13. Risks & Mitigation

| Risk | Impact | Probability | Mitigation |
|------|--------|-------------|------------|
| High API costs | HIGH | MEDIUM | Implement aggressive caching, rate limiting, adaptive RAG |
| Poor conversation quality | MEDIUM | MEDIUM | A/B testing, user feedback, continuous prompt tuning |
| Context window limits | MEDIUM | HIGH | Implement summarization, sliding window |
| Session management bugs | LOW | MEDIUM | Comprehensive testing, automatic cleanup |
| User privacy concerns | HIGH | LOW | Clear consent, data minimization, privacy filters |
| Scale issues (many concurrent users) | MEDIUM | LOW | Connection pooling, async optimization, queue system |

---

## 14. Future Enhancements

**Post-Launch (Month 2+):**

1. **Voice Integration**
   - Text-to-speech for responses
   - Speech-to-text for voice channel support

2. **Multi-Language Support**
   - Detect user language
   - Respond in same language
   - Translate Discord history on-the-fly

3. **Custom Training**
   - Fine-tune model on server's conversation style
   - Learn server-specific terminology
   - Personalize to individual users

4. **Integrations**
   - Connect to external APIs (weather, news, etc.)
   - Integration with server bots (music, moderation)
   - Calendar and reminder features

5. **Analytics Dashboard**
   - Web dashboard for conversation analytics
   - Topic trending over time
   - User engagement graphs

---

## Conclusion

This chatbot feature will transform Deep Bot from a Q&A tool into an interactive conversational assistant. The phased implementation allows for iterative development and testing, while the modular architecture ensures maintainability and extensibility.

**Key Success Factors:**
- ✅ Leverage existing RAG infrastructure
- ✅ Focus on UX from day one
- ✅ Implement aggressive cost controls
- ✅ Comprehensive testing at each phase
- ✅ User feedback integration

**Next Steps:**
1. Review this plan and gather feedback
2. Set up development environment
3. Begin Phase 1 implementation
4. Create tracking issues for each milestone

Ready to build the future of Discord bots! 🚀
